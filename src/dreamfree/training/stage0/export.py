from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from safetensors.torch import save_file as save_safetensors

from dreamfree.data.contracts import _normalize_latent_frame0_mode
from dreamfree.models.adaptation import _stage0_sigma_metadata
from dreamfree.models.lora import LoRALinear
from dreamfree.training.runtime import _dtype_from_name, _json_dump
from dreamfree.training.stage0.config import Stage0Config
from dreamfree.training.stage0.constants import logger


def _tensor_export_size(tensor: torch.Tensor, export_dtype: torch.dtype) -> int:
    """Estimate serialized tensor size for export sharding.

    Args:
        tensor: Tensor or parameter being exported.
        export_dtype: Dtype used for floating-point export tensors.

    Returns:
        Estimated byte count after dtype conversion.
    """
    if torch.is_floating_point(tensor):
        elem_size = torch.tensor([], dtype=export_dtype).element_size()
    else:
        elem_size = tensor.element_size()
    return int(tensor.numel()) * int(elem_size)


def _iter_merged_state_specs(module: torch.nn.Module, prefix: str = ""):
    """Yield export specs, replacing LoRA modules with merged base tensors.

    Args:
        module: Module tree to export.
        prefix: State-dict key prefix for recursive traversal.

    Yields:
        Tuples describing either a plain tensor or a LoRA-merged tensor.
    """
    for name, param in module.named_parameters(recurse=False):
        yield ("param", prefix + name, param)
    for name, buffer in module.named_buffers(recurse=False):
        yield ("buffer", prefix + name, buffer)
    for child_name, child in module.named_children():
        child_prefix = prefix + child_name + "."
        if isinstance(child, LoRALinear):
            yield ("lora_weight", child_prefix + "weight", child)
            if child.bias is not None:
                yield ("lora_bias", child_prefix + "bias", child)
            continue
        yield from _iter_merged_state_specs(child, child_prefix)


def _spec_size(spec: tuple[str, str, Any], export_dtype: torch.dtype) -> int:
    """Estimate one export spec's byte size.

    Args:
        spec: Export spec from ``_iter_merged_state_specs``.
        export_dtype: Dtype for floating-point export tensors.

    Returns:
        Estimated byte count for the materialized tensor.
    """
    (kind, _, obj) = spec
    if kind in {"param", "buffer"}:
        return _tensor_export_size(obj, export_dtype)
    if kind == "lora_weight":
        return _tensor_export_size(obj.base_layer.weight, export_dtype)
    if kind == "lora_bias":
        return _tensor_export_size(obj.bias, export_dtype)
    raise ValueError(f"Unknown export spec kind: {kind}")


def _materialize_spec_tensor(spec: tuple[str, str, Any], export_dtype: torch.dtype) -> torch.Tensor:
    """Materialize one export tensor on CPU.

    Args:
        spec: Export spec from ``_iter_merged_state_specs``.
        export_dtype: Dtype for floating-point export tensors.

    Returns:
        Contiguous CPU tensor ready for safetensors serialization.
    """
    (kind, _, obj) = spec
    if kind in {"param", "buffer"}:
        tensor = obj.detach()
    elif kind == "lora_bias":
        tensor = obj.bias.detach()
    elif kind == "lora_weight":
        layer: LoRALinear = obj
        base = layer.base_layer.weight.detach().float()
        lora_a = layer.lora_A.weight.detach().float()
        lora_b = layer.lora_B.weight.detach().float()
        tensor = base + torch.mm(lora_b, lora_a) * float(layer.scaling)
    else:
        raise ValueError(f"Unknown export spec kind: {kind}")
    if torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=export_dtype)
    return tensor.detach().cpu().contiguous()


@torch.no_grad()
def export_merged_dit_safetensors(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    export_dir: str | Path,
    cfg: Stage0Config,
    global_step: int,
) -> None:
    """Export the DiT with LoRA adapters merged into base weights.

    Args:
        accelerator: Active Accelerate wrapper.
        model: Possibly distributed Stage 0 model.
        export_dir: Destination directory for Wan-compatible DiT shards.
        cfg: Stage 0 export configuration.
        global_step: Training step represented by the export.
    """
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    dit = unwrapped.core.dit
    export_dtype = _dtype_from_name(cfg.export_dtype)
    max_shard_bytes = int(float(cfg.export_max_shard_size_gb) * 1024**3)
    specs = list(_iter_merged_state_specs(dit))
    if not specs:
        raise RuntimeError("No DiT tensors found for export.")
    shards: list[list[tuple[str, str, Any]]] = []
    current: list[tuple[str, str, Any]] = []
    current_size = 0
    for spec in specs:
        size = _spec_size(spec, export_dtype)
        if current and current_size + size > max_shard_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append(spec)
        current_size += size
    if current:
        shards.append(current)
    total = len(shards)
    logger.info(
        "Exporting merged DiT checkpoint: export_dir=%s tensors=%d shards=%d dtype=%s",
        export_dir,
        len(specs),
        total,
        cfg.export_dtype,
    )
    existing = sorted(export_dir.glob("diffusion_pytorch_model-*.safetensors"))
    if existing and (not (export_dir / "stage0_config.json").exists()):
        raise RuntimeError(
            f"Refusing to overwrite existing Wan shards in an export directory that was not created by this stage-0 exporter: {export_dir}"
        )
    for path in existing:
        path.unlink()
    for idx, shard_specs in enumerate(shards, start=1):
        shard_name = f"diffusion_pytorch_model-{idx:05d}-of-{total:05d}.safetensors"
        tensors = {spec[1]: _materialize_spec_tensor(spec, export_dtype) for spec in shard_specs}
        save_safetensors(tensors, str(export_dir / shard_name), metadata={"format": "pt"})
        shard_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        logger.info("[export] wrote %s (%.2f GB)", shard_name, shard_bytes / 1024**3)
        del tensors
    _json_dump(
        export_dir / "stage0_config.json",
        {
            "global_step": int(global_step),
            "base_model_id": str(cfg.model_id),
            "video_model_family": str(cfg.video_model_family),
            "dit_role": str(cfg.dit_role),
            "vae_family": "wan2.1" if str(cfg.video_model_family) == "wan22_i2v_a14b" else "wan2.2",
            "i2v_condition_mode": str(cfg.i2v_condition_mode),
            "latent_frame0_mode": _normalize_latent_frame0_mode(cfg.latent_frame0_mode),
            **_stage0_sigma_metadata(cfg),
            "tokenizer_model_id": str(cfg.tokenizer_model_id),
            "lora_config": {
                "rank": int(cfg.lora_rank),
                "alpha": float(cfg.lora_alpha),
                "dropout": float(cfg.lora_dropout),
                "target_modules": str(cfg.lora_target_modules),
            },
            "config": asdict(cfg),
        },
    )
    accelerator.wait_for_everyone()
