"""Standalone RPC transport for physical YAM deployment, wire-compatible with DROID.

This protocol uses pickle and must only be exposed through a trusted local
connection, such as the localhost endpoint of an SSH tunnel.
"""

from __future__ import annotations

import pickle
import socket
import socketserver
import struct
import threading
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

HEADER_STRUCT = struct.Struct("!Q")
PICKLE_PROTOCOL = 4
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
MAX_EPISODE_SEED = (1 << 63) - 1


class _CompatUnpickler(pickle.Unpickler):
    """Unpickle NumPy 2 payloads in NumPy 1.x environments."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core" or module.startswith("numpy._core."):
            compat_module = module.replace("numpy._core", "numpy.core", 1)
            try:
                return super().find_class(compat_module, name)
            except ModuleNotFoundError:
                pass
        return super().find_class(module, name)


def _loads(payload: bytes) -> Any:
    return _CompatUnpickler(BytesIO(payload)).load()


def _recvall(sock: socket.socket, num_bytes: int) -> bytes:
    chunks: List[bytes] = []
    remaining = int(num_bytes)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("Socket closed while receiving RPC payload.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=PICKLE_PROTOCOL)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"RPC payload exceeds the maximum size: bytes={len(data)} maximum={MAX_MESSAGE_BYTES}"
        )
    sock.sendall(HEADER_STRUCT.pack(len(data)) + data)


def recv_message(sock: socket.socket) -> Any:
    header = _recvall(sock, HEADER_STRUCT.size)
    (num_bytes,) = HEADER_STRUCT.unpack(header)
    if num_bytes <= 0 or num_bytes > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Invalid RPC payload size: bytes={num_bytes} allowed=[1,{MAX_MESSAGE_BYTES}]"
        )
    return _loads(_recvall(sock, num_bytes))


def _validate_episode_seed(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"RPC reset episode_seed must be an integer or None, got {value!r}.")
    if not 0 <= value <= MAX_EPISODE_SEED:
        raise ValueError(f"RPC reset episode_seed must be in [0,{MAX_EPISODE_SEED}], got {value}.")
    return value


class PolicyRpcClient:
    def __init__(self, host: str, port: int, *, timeout_sec: float = 600.0) -> None:
        self.host = str(host)
        self.port = int(port)
        self._sock = socket.create_connection((self.host, self.port), timeout=timeout_sec)
        self._sock.settimeout(timeout_sec)

    def request(self, payload: Dict[str, Any]) -> Any:
        send_message(self._sock, payload)
        response = recv_message(self._sock)
        if isinstance(response, dict) and response.get("ok") is False:
            raise RuntimeError(str(response.get("error", "Policy RPC request failed.")))
        if isinstance(response, dict) and response.get("ok") is True and "data" in response:
            return response["data"]
        return response

    def infer(self, observation: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        return self.request(
            {
                "endpoint": "infer",
                "observation": observation,
                "instruction": str(instruction),
            }
        )

    def info(self) -> Dict[str, Any]:
        return self.request({"endpoint": "info"})

    def reset(self, episode_seed: Optional[int] = None) -> Dict[str, Any]:
        request: Dict[str, Any] = {"endpoint": "reset"}
        if episode_seed is not None:
            request["episode_seed"] = episode_seed
        return self.request(request)

    def close(self) -> None:
        self._sock.close()


class _PolicyRpcHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, PolicyRpcServer)
        while True:
            try:
                request = recv_message(self.request)
            except EOFError:
                return
            except Exception as exc:
                send_message(
                    self.request,
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                )
                return
            try:
                response = server.dispatch(request)
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            send_message(self.request, response)


class PolicyRpcServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        *,
        infer_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
        reset_fn: Optional[Callable[[Optional[int]], Optional[Dict[str, Any]]]] = None,
        info_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        validate_image_preprocessing_fn: Optional[
            Callable[[Dict[str, Any]], Dict[str, Any]]
        ] = None,
    ) -> None:
        super().__init__(server_address, _PolicyRpcHandler)
        self._infer_fn = infer_fn
        self._reset_fn = reset_fn
        self._info_fn = info_fn
        self._validate_image_preprocessing_fn = validate_image_preprocessing_fn
        self._policy_lock = threading.Lock()

    def dispatch(self, request: Any) -> Dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError(f"RPC request must be a dict, got {type(request).__name__}.")
        endpoint = str(request.get("endpoint", ""))
        if endpoint == "info":
            data = self._info_fn() if self._info_fn is not None else {}
            return {"ok": True, "data": data}
        if endpoint == "infer":
            observation = request.get("observation")
            if not isinstance(observation, dict):
                raise TypeError("RPC infer request requires dict `observation`.")
            instruction = str(request.get("instruction", ""))
            with self._policy_lock:
                data = self._infer_fn(observation, instruction)
            return {"ok": True, "data": data}
        if endpoint == "validate_image_preprocessing":
            if self._validate_image_preprocessing_fn is None:
                raise ValueError("This server does not support image preprocessing validation.")
            with self._policy_lock:
                data = self._validate_image_preprocessing_fn(request)
            return {"ok": True, "data": data}
        if endpoint == "reset":
            episode_seed = _validate_episode_seed(request.get("episode_seed"))
            reset_data: Optional[Dict[str, Any]] = None
            with self._policy_lock:
                if self._reset_fn is not None:
                    reset_data = self._reset_fn(episode_seed)
            if reset_data is not None and not isinstance(reset_data, dict):
                raise TypeError(
                    "RPC reset callback must return a dict or None, "
                    f"got {type(reset_data).__name__}."
                )
            data = {} if reset_data is None else dict(reset_data)
            data["status"] = "reset"
            return {"ok": True, "data": data}
        raise ValueError(f"Unsupported RPC endpoint: {endpoint!r}")
