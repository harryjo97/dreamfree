from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset

from dreamfree.training.stage0.config import Stage0Config


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError(f"Expected integer or null, got boolean: {value}")
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if text in {"", "null", "None"}:
        return None
    try:
        return int(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Expected integer or null, got: {value}") from err


def _normalize_mixed_precision(mixed_precision: str) -> str:
    value = str(mixed_precision).strip().lower()
    if value not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision={mixed_precision}. Expected one of: no, fp16, bf16."
        )
    return value


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    value = _normalize_mixed_precision(mixed_precision)
    if value == "no":
        return torch.float32
    if value == "fp16":
        return torch.float16
    return torch.bfloat16


def _dtype_from_name(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"fp32", "float32"}:
        return torch.float32
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype name: {name}")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    from dreamfree.config_names import normalize_checkpoint_metadata

    return normalize_checkpoint_metadata(payload)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Save a torch payload through a sibling temporary file before rename.

    Args:
        path: Final checkpoint path.
        payload: Serializable payload passed to ``torch.save``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, path)


@dataclass(frozen=True)
class ResolvedTrainingBudget:
    source: str
    train_samples: int
    effective_global_batch: int
    max_steps: int


def _resolve_training_budget(
    *,
    num_epochs: float | None,
    max_steps: int,
    train_samples: int,
    world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    stage_label: str,
) -> ResolvedTrainingBudget:
    resolved_train_samples = int(train_samples)
    if resolved_train_samples < 0:
        raise ValueError(f"{stage_label} train_samples must be non-negative, got {train_samples}.")
    resolved_world_size = int(world_size)
    if resolved_world_size <= 0:
        raise ValueError(f"{stage_label} world_size must be positive, got {world_size}.")
    effective_global_batch = (
        resolved_world_size * int(per_device_batch_size) * int(gradient_accumulation_steps)
    )
    if effective_global_batch <= 0:
        raise ValueError(
            f"{stage_label} effective global batch must be positive, got world_size={resolved_world_size}, per_device_batch_size={per_device_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}."
        )
    if num_epochs is None:
        return ResolvedTrainingBudget(
            source="max_steps",
            train_samples=resolved_train_samples,
            effective_global_batch=effective_global_batch,
            max_steps=int(max_steps),
        )
    resolved_num_epochs = float(num_epochs)
    if not math.isfinite(resolved_num_epochs) or resolved_num_epochs <= 0.0:
        raise ValueError(
            f"{stage_label} num_epochs must be a finite positive value, got {num_epochs}."
        )
    if resolved_train_samples <= 0:
        raise ValueError(
            f"Cannot derive {stage_label} max_steps from num_epochs with an empty training dataset."
        )
    derived_steps = int(
        math.ceil(resolved_num_epochs * resolved_train_samples / effective_global_batch)
    )
    return ResolvedTrainingBudget(
        source="num_epochs",
        train_samples=resolved_train_samples,
        effective_global_batch=effective_global_batch,
        max_steps=max(derived_steps, 1),
    )


def collect_trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Collect unique trainable parameters for the optimizer.

    Args:
        model: Model containing LoRA adapters.

    Returns:
        Unique parameters with ``requires_grad=True``.
    """
    params = []
    seen = set()
    for param in model.parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        params.append(param)
    if not params:
        raise ValueError("No trainable parameters found for stage-0 training.")
    return params


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Stage0Config):
    """Build the learning-rate schedule for optimizer steps.

    Args:
        optimizer: Optimizer to schedule.
        cfg: Stage 0 optimization configuration.

    Returns:
        PyTorch scheduler with optional linear warmup.
    """
    total_steps = max(int(cfg.max_steps), 1)
    warmup_steps = min(int(total_steps * float(cfg.warmup_ratio)), total_steps - 1)
    scheduler_name = str(cfg.lr_scheduler).strip().lower()
    if scheduler_name == "cosine":
        main = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=float(cfg.learning_rate) * 0.01,
        )
    elif scheduler_name == "constant":
        main = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=max(total_steps - warmup_steps, 1)
        )
    else:
        raise ValueError(
            f"Unsupported lr_scheduler={cfg.lr_scheduler}. Expected cosine or constant."
        )
    if warmup_steps <= 0:
        return main
    warmup = LinearLR(
        optimizer, start_factor=1.0 / max(warmup_steps, 1), end_factor=1.0, total_iters=warmup_steps
    )
    return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_steps])


def _resolve_train_device() -> str:
    """Choose the device string for this process.

    Returns:
        CPU, single-GPU, or local-rank CUDA device string.
    """
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def _format_eta(
    start_time: float, start_step: int, current_step: int, max_steps: int
) -> tuple[str, float]:
    """Estimate remaining training time from completed optimizer steps.

    Args:
        start_time: ``time.perf_counter`` value captured at loop start.
        start_step: Global step at loop start, including resume offset.
        current_step: Current global optimizer step.
        max_steps: Target final optimizer step.

    Returns:
        Pair of formatted ETA string and measured steps per second.
    """
    elapsed = max(time.perf_counter() - start_time, 1e-06)
    done = max(current_step - start_step, 1)
    steps_per_sec = done / elapsed
    remaining = max(max_steps - current_step, 0)
    eta_seconds = int(remaining / max(steps_per_sec, 1e-09))
    (hours, rem) = divmod(eta_seconds, 3600)
    (minutes, seconds) = divmod(rem, 60)
    return (f"{hours:02d}:{minutes:02d}:{seconds:02d}", steps_per_sec)


def _init_wandb(accelerator: Accelerator, cfg: Stage0Config):
    """Initialize wandb only on the main process when enabled.

    Args:
        accelerator: Active Accelerate wrapper.
        cfg: Stage 0 wandb configuration.

    Returns:
        A wandb run object or ``None``.
    """
    if not cfg.wandb_enabled or not accelerator.is_main_process:
        return None
    try:
        import wandb
    except ImportError as err:
        raise ImportError("wandb is enabled but the package is not installed.") from err
    return wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        mode=cfg.wandb_mode,
        dir=cfg.output_dir,
        config=asdict(cfg),
    )


def _wandb_log(run, payload: dict[str, Any], step: int) -> None:
    """Log metrics when a wandb run exists.

    Args:
        run: wandb run object or ``None``.
        payload: Metrics to log.
        step: Global training step.
    """
    if run is not None:
        run.log(payload, step=step)


def assert_dataset_lengths_consistent(
    accelerator: Accelerator, dataset: Dataset, name: str
) -> None:
    """Ensure all distributed ranks expose the same dataset length.

    Args:
        accelerator: Active Accelerate wrapper.
        dataset: Dataset built on this rank.
        name: Human-readable dataset name for error messages.
    """
    local_length = torch.tensor([len(dataset)], device=accelerator.device, dtype=torch.long)
    lengths = accelerator.gather(local_length).reshape(-1)
    if torch.all(lengths == lengths[0]):
        return
    raise RuntimeError(f"{name} length differs across ranks: {lengths.cpu().tolist()}")


def optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    """Return unique parameters from optimizer groups.

    Args:
        optimizer: Optimizer prepared by Accelerate.

    Returns:
        Unique optimizer parameters, preserving first-seen order.
    """
    params = []
    seen = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    return params
