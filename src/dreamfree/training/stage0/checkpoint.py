from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from dreamfree.data.contracts import _normalize_latent_frame0_mode, _stage0_visual_preprocess_id
from dreamfree.data.video import _stage0_14b_frame0_mode_from_state
from dreamfree.models.adaptation import VideoAdaptationModel, _stage0_sigma_metadata
from dreamfree.models.lora import collect_lora_state_dict, load_lora_state_dict
from dreamfree.training.runtime import (
    _atomic_torch_save,
    _json_dump,
    _json_load,
    _parse_optional_int,
)
from dreamfree.training.stage0.config import Stage0Config
from dreamfree.training.stage0.constants import STAGE0_CHECKPOINT_DIR_RE, logger


def _prune_stage0_checkpoints(cfg: Stage0Config) -> list[Path]:
    """Remove older Stage 0 checkpoint directories when retention is configured."""
    max_checkpoints_to_keep = _parse_optional_int(cfg.max_checkpoints_to_keep)
    if max_checkpoints_to_keep is None:
        return []
    if max_checkpoints_to_keep <= 0:
        raise ValueError(
            f"max_checkpoints_to_keep must be null or a positive integer, got {cfg.max_checkpoints_to_keep}."
        )
    checkpoint_root = Path(cfg.output_dir) / "checkpoints"
    if not checkpoint_root.is_dir():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for child in checkpoint_root.iterdir():
        if child.is_symlink() or not child.is_dir():
            continue
        match = STAGE0_CHECKPOINT_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), child))
    checkpoints.sort(key=lambda item: (item[0], item[1].name))
    stale = checkpoints[: max(0, len(checkpoints) - max_checkpoints_to_keep)]
    removed: list[Path] = []
    for _, path in stale:
        shutil.rmtree(path)
        removed.append(path)
        logger.info("[ckpt] removed old Stage 0 checkpoint %s", path)
    return removed


def save_lora_checkpoint(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: Stage0Config,
    global_step: int,
) -> Path | None:
    """Save LoRA weights and trainer state from the main process.

    Args:
        accelerator: Active Accelerate wrapper.
        model: Possibly distributed model wrapper.
        optimizer: Optimizer whose state may be saved.
        scheduler: LR scheduler whose state may be saved.
        cfg: Stage 0 checkpoint configuration.
        global_step: Optimizer step represented by the checkpoint.

    Returns:
        Checkpoint directory on the main process, otherwise ``None``.
    """
    checkpoint_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{global_step:06d}"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        lora_state = collect_lora_state_dict(unwrapped.core.dit)
        save_safetensors(lora_state, str(checkpoint_dir / "lora.safetensors"))
        sigma_metadata = _stage0_sigma_metadata(cfg)
        _json_dump(
            checkpoint_dir / "trainer_state.json",
            {
                "checkpoint_type": "dreamfree_stage0",
                "checkpoint_version": 1,
                "global_step": int(global_step),
                **(
                    {"visual_preprocess_id": _stage0_visual_preprocess_id(cfg)}
                    if _stage0_visual_preprocess_id(cfg) is not None
                    else {}
                ),
                "lora_config": getattr(unwrapped, "lora_config", {}),
                "video_model_family": str(cfg.video_model_family),
                "dit_role": str(cfg.dit_role),
                "vae_family": "wan2.1"
                if str(cfg.video_model_family) == "wan22_i2v_a14b"
                else "wan2.2",
                "i2v_condition_mode": str(cfg.i2v_condition_mode),
                "latent_frame0_mode": _normalize_latent_frame0_mode(cfg.latent_frame0_mode),
                **sigma_metadata,
                "config": asdict(cfg),
            },
        )
        if cfg.save_optimizer_state:
            _atomic_torch_save(
                checkpoint_dir / "optim.pt",
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            )
        logger.info("[ckpt] saved LoRA checkpoint to %s", checkpoint_dir)
        _prune_stage0_checkpoints(cfg)
    accelerator.wait_for_everyone()
    return checkpoint_dir if accelerator.is_main_process else None


def load_lora_checkpoint_if_requested(*, model: VideoAdaptationModel, cfg: Stage0Config) -> int:
    """Load LoRA weights from ``cfg.resume`` when provided.

    Args:
        model: Stage 0 adapter whose DiT contains LoRA modules.
        cfg: Stage 0 configuration with optional resume path.

    Returns:
        Resumed global step, or ``0`` for a fresh run.
    """
    if not cfg.resume:
        return 0
    resume_path = Path(cfg.resume)
    if resume_path.is_dir():
        lora_path = resume_path / "lora.safetensors"
        state_path = resume_path / "trainer_state.json"
    else:
        lora_path = resume_path
        state_path = resume_path.with_name("trainer_state.json")
    if not lora_path.exists():
        raise FileNotFoundError(f"Resume LoRA checkpoint not found: {lora_path}")
    if state_path.exists():
        state = _json_load(state_path)
        if (
            state.get("checkpoint_type") != "dreamfree_stage0"
            or state.get("checkpoint_version") != 1
        ):
            raise ValueError(f"Expected a DreamFree stage 0 checkpoint: {state_path}")
        if state.get("video_model_family") != cfg.video_model_family:
            raise ValueError("Resume checkpoint video model family does not match configuration.")
        global_step = int(state.get("global_step", 0))
        if str(cfg.video_model_family) == "wan22_i2v_a14b":
            state_config = state.get("config")
            state_family = state.get("video_model_family")
            if state_family is None and isinstance(state_config, dict):
                state_family = state_config.get("video_model_family")
            if state_family is not None and str(state_family) != "wan22_i2v_a14b":
                raise ValueError(
                    f"Cannot resume an 14B Stage 0 run from checkpoint metadata with video_model_family={state_family!r}."
                )
            saved_mode = _stage0_14b_frame0_mode_from_state(state)
            expected_mode = _normalize_latent_frame0_mode(cfg.latent_frame0_mode)
            if saved_mode != expected_mode:
                raise ValueError(
                    f"14B Stage 0 resume requires matching latent_frame0_mode: checkpoint={saved_mode!r} current={expected_mode!r}."
                )
            expected_visual_id = _stage0_visual_preprocess_id(cfg)
            if (
                expected_visual_id is not None
                and state.get("visual_preprocess_id") != expected_visual_id
            ):
                raise ValueError(
                    f"Direct-bilinear RoboCasa Stage 0 resume requires visual_preprocess_id={expected_visual_id!r}, got {state.get('visual_preprocess_id')!r} in {state_path}. Start a fresh Stage 0 run from the rebuilt cache."
                )
    else:
        raise FileNotFoundError(f"Resume checkpoint metadata is missing: {state_path}")
    lora_state = load_safetensors(str(lora_path), device="cpu")
    load_lora_state_dict(model.core.dit, lora_state, strict=True)
    logger.info("Loaded stage-0 LoRA checkpoint from %s at global_step=%d", lora_path, global_step)
    return global_step


def load_optimizer_state_if_requested(
    *, optimizer: torch.optim.Optimizer, scheduler, cfg: Stage0Config
) -> None:
    """Load optimizer and scheduler state from a resumed checkpoint.

    Args:
        optimizer: Optimizer to restore.
        scheduler: LR scheduler to restore.
        cfg: Stage 0 configuration with optional resume path.
    """
    if not cfg.resume:
        return
    resume_path = Path(cfg.resume)
    optim_path = (resume_path if resume_path.is_dir() else resume_path.parent) / "optim.pt"
    if not optim_path.exists():
        raise FileNotFoundError(f"Resume optimizer state is missing: {optim_path}")
    payload = torch.load(str(optim_path), map_location="cpu")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    logger.info("Loaded optimizer/scheduler state from %s", optim_path)
