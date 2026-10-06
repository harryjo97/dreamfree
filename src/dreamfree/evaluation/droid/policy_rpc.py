"""Small local RPC helper for DROID sim policy inference.

The sim process runs in an IsaacLab-specific environment while the DreamFree policy
server runs in the normal DreamFree environment. This module intentionally uses only
the Python standard library so both interpreters can share the same protocol
without installing OpenPI/websocket dependencies into DreamFree.
"""

from __future__ import annotations

import pickle
import socket
import struct
from io import BytesIO
from typing import Any

HEADER_STRUCT = struct.Struct("!Q")


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
    chunks: list[bytes] = []
    remaining = int(num_bytes)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("Socket closed while receiving RPC payload.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(HEADER_STRUCT.pack(len(data)) + data)


def recv_message(sock: socket.socket) -> Any:
    header = _recvall(sock, HEADER_STRUCT.size)
    (num_bytes,) = HEADER_STRUCT.unpack(header)
    if num_bytes <= 0:
        raise ValueError(f"Invalid RPC payload size: {num_bytes}")
    return _loads(_recvall(sock, num_bytes))


class PolicyRpcClient:
    def __init__(self, host: str, port: int, *, timeout_sec: float = 600.0) -> None:
        self.host = str(host)
        self.port = int(port)
        self._sock = socket.create_connection((self.host, self.port), timeout=timeout_sec)
        self._sock.settimeout(timeout_sec)

    def request(self, payload: dict[str, Any]) -> Any:
        send_message(self._sock, payload)
        response = recv_message(self._sock)
        if isinstance(response, dict) and response.get("ok") is False:
            raise RuntimeError(str(response.get("error", "Policy RPC request failed.")))
        if isinstance(response, dict) and response.get("ok") is True and ("data" in response):
            return response["data"]
        return response

    def infer(self, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        return self.request(
            {"endpoint": "infer", "observation": observation, "instruction": str(instruction)}
        )

    def info(self) -> dict[str, Any]:
        return self.request({"endpoint": "info"})

    def reset(self) -> None:
        self.request({"endpoint": "reset"})

    def close(self) -> None:
        self._sock.close()
