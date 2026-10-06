from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def is_distributed_initialized() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )


def should_broadcast_pretrained_load(enabled: bool) -> bool:
    return bool(enabled) and is_distributed_initialized()


def get_distributed_rank() -> int:
    if not is_distributed_initialized():
        return 0
    return int(torch.distributed.get_rank())


def get_pretrained_broadcast_device() -> str:
    value = (
        os.environ.get(
            "DREAMFREE_PRETRAINED_BROADCAST_DEVICE",
            os.environ.get("LIBERO_PRETRAINED_BROADCAST_DEVICE", "cuda"),
        )
        .strip()
        .lower()
    )
    if value not in {"cpu", "cuda"}:
        raise ValueError(
            f"DREAMFREE_PRETRAINED_BROADCAST_DEVICE must be 'cpu' or 'cuda', got {value!r}."
        )
    return value


def get_pretrained_broadcast_sync_device(fallback_device: str | torch.device) -> torch.device:
    if get_pretrained_broadcast_device() == "cpu":
        return torch.device("cpu")
    sync_device = torch.device(fallback_device)
    if sync_device.type == "cuda" and sync_device.index is None:
        sync_device = torch.device("cuda", torch.cuda.current_device())
    return sync_device


def sync_distributed_load_error(
    error: BaseException | None,
    *,
    device: str | torch.device,
    src: int = 0,
    context: str = "distributed pretrained load",
) -> None:
    if not is_distributed_initialized():
        if error is not None:
            raise error
        return
    rank = get_distributed_rank()
    sync_device = get_pretrained_broadcast_sync_device(device)
    flag = torch.tensor([0 if error is not None else 1], device=sync_device, dtype=torch.int32)
    torch.distributed.broadcast(flag, src=src)
    if int(flag.item()) == 1:
        return
    message: list[Any] = [repr(error) if rank == src else ""]
    torch.distributed.broadcast_object_list(message, src=src)
    raise RuntimeError(f"{context} failed on rank {src}: {message[0]}")


def broadcast_module_state(module: torch.nn.Module, *, src: int = 0) -> None:
    if not is_distributed_initialized():
        return
    for tensor in module.state_dict().values():
        if torch.is_tensor(tensor):
            torch.distributed.broadcast(tensor, src=src)


def load_safetensors_rank0(
    filename: str | os.PathLike[str], *, device: str | torch.device = "cpu", src: int = 0
) -> dict[str, torch.Tensor]:
    """Load a safetensor file once on ``src`` and stream it to all ranks."""
    if not is_distributed_initialized():
        from safetensors.torch import load_file

        return load_file(str(filename), device=str(device))
    world_size = int(torch.distributed.get_world_size())
    if not 0 <= src < world_size:
        raise ValueError(f"src must be in [0, {world_size}), got {src}.")
    if get_pretrained_broadcast_device() != "cpu":
        raise ValueError(
            "Distributed safetensor loading requires DREAMFREE_PRETRAINED_BROADCAST_DEVICE=cpu."
        )
    rank = get_distributed_rank()
    path = Path(filename).expanduser().resolve()
    with ExitStack() as stack:
        source: Any | None = None
        open_error: BaseException | None = None
        if rank == src:
            try:
                source = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
                if source is None:
                    raise RuntimeError(f"safe_open returned no handle for {path}.")
            except BaseException as exc:
                open_error = exc
        sync_distributed_load_error(
            open_error, device="cpu", src=src, context=f"safetensor open for {path}"
        )
        keys: list[str] | None = None
        keys_error: BaseException | None = None
        if rank == src:
            try:
                keys = list(source.keys())
            except BaseException as exc:
                keys_error = exc
        sync_distributed_load_error(
            keys_error, device="cpu", src=src, context=f"safetensor key read for {path}"
        )
        keys_payload: list[Any] = [keys]
        torch.distributed.broadcast_object_list(keys_payload, src=src)
        keys = keys_payload[0]
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise RuntimeError(f"Invalid safetensor key payload broadcast for {path}: {keys!r}")
        output: dict[str, torch.Tensor] = {}
        for key in keys:
            tensor: torch.Tensor | None = None
            tensor_error: BaseException | None = None
            if rank == src:
                try:
                    tensor = source.get_tensor(key).contiguous()
                except BaseException as exc:
                    tensor_error = exc
            sync_distributed_load_error(
                tensor_error, device="cpu", src=src, context=f"safetensor read for {path}:{key}"
            )
            descriptor: list[Any] = [
                (tuple(tensor.shape), str(tensor.dtype).removeprefix("torch."))
                if rank == src and tensor is not None
                else None
            ]
            torch.distributed.broadcast_object_list(descriptor, src=src)
            (shape, dtype_name) = descriptor[0]
            dtype = getattr(torch, str(dtype_name), None)
            if not isinstance(dtype, torch.dtype):
                raise TypeError(
                    f"Unsupported tensor dtype broadcast for {path}:{key}: {dtype_name!r}"
                )
            if tensor is None:
                tensor = torch.empty(tuple(shape), dtype=dtype, device="cpu")
            torch.distributed.broadcast(tensor, src=src)
            output[key] = tensor if torch.device(device).type == "cpu" else tensor.to(device=device)
    return output
