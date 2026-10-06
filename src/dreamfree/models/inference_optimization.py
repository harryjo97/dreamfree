"""Inference-only attention optimizations for BF16 deployment."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from dreamfree.models.lora import has_lora_adapters, merge_lora_adapters
from dreamfree.models.wan22.wan_video_dit import CrossAttention, SelfAttention

FUSED_SDPA_BACKENDS = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
)


@contextmanager
def fused_sdpa_only() -> Iterator[None]:
    """Allow fused CUDA SDPA implementations while excluding the math backend."""
    with sdpa_kernel(list(FUSED_SDPA_BACKENDS), set_priority=False):
        yield


def merge_active_lora_for_inference(module: nn.Module, *, adapter_name: str | None) -> int:
    """Merge an active deployment LoRA and fail if any adapter remains."""
    if not has_lora_adapters(module):
        return 0
    merged = merge_lora_adapters(
        module, adapter_name=adapter_name, allow_single_adapter_from_multi=adapter_name is not None
    )
    if not merged or has_lora_adapters(module):
        raise RuntimeError(
            "Inference optimization requires every active LoRA adapter to be merged before projection fusion."
        )
    return len(merged)


def _fused_linear(linears: tuple[nn.Module | None, ...], *, label: str) -> nn.Linear:
    if not linears or not all(isinstance(layer, nn.Linear) for layer in linears):
        types = [type(layer).__name__ for layer in linears]
        raise TypeError(f"{label} fusion requires plain nn.Linear projections, got {types}.")
    typed = tuple(layer for layer in linears if isinstance(layer, nn.Linear))
    in_features = typed[0].in_features
    if any(layer.in_features != in_features for layer in typed):
        raise ValueError(f"{label} projections must share in_features.")
    has_bias = typed[0].bias is not None
    if any((layer.bias is not None) != has_bias for layer in typed):
        raise ValueError(f"{label} projections must use the same bias contract.")
    reference = typed[0].weight
    fused = nn.Linear(
        in_features,
        sum(layer.out_features for layer in typed),
        bias=has_bias,
        device=reference.device,
        dtype=reference.dtype,
    )
    with torch.no_grad():
        fused.weight.copy_(torch.cat([layer.weight for layer in typed], dim=0))
        if has_bias:
            fused.bias.copy_(torch.cat([layer.bias for layer in typed], dim=0))
    fused.requires_grad_(False)
    fused.eval()
    return fused


def fuse_attention_projections(module: nn.Module) -> dict[str, int]:
    """Fuse QKV/KV projections in all Wan attention modules below ``module``."""
    counts = {"self_qkv": 0, "cross_kv": 0}
    for child in module.modules():
        if isinstance(child, SelfAttention):
            if child.qkv is not None:
                raise RuntimeError("Self-attention QKV projection was already fused.")
            child.qkv = _fused_linear((child.q, child.k, child.v), label="self-attention QKV")
            child.q = None
            child.k = None
            child.v = None
            counts["self_qkv"] += 1
        elif isinstance(child, CrossAttention):
            if child.kv is not None:
                raise RuntimeError("Cross-attention KV projection was already fused.")
            child.kv = _fused_linear((child.k, child.v), label="cross-attention KV")
            child.k = None
            child.v = None
            counts["cross_kv"] += 1
    if counts["self_qkv"] <= 0 or counts["cross_kv"] <= 0:
        raise RuntimeError(f"Attention projection fusion matched incomplete module sets: {counts}.")
    return counts
