from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from safetensors import safe_open

from dreamfree.data.distillation import (
    _first_stage1_latent_cache_file,
    _is_stage1_droid_dataset,
    _is_stage1_robocasa_success_family_dataset,
    _parse_stage1_dataset_dirs,
    _stage1_robocasa_layout_spec,
)
from dreamfree.data.lerobot.cached_latent_dataset import validate_stage2_latent_cache_file
from dreamfree.data.stage1_cache import (
    _stage1_latent_cache_expected_contract,
    resolve_stage1_latent_cache_dir,
)
from dreamfree.data.stage1_cache import (
    _validate_stage1_latent_cache_config as _shared_validate_stage1_latent_cache_config,
)
from dreamfree.data.video import (
    _normalize_stage0_dataset_format,
    _parse_stage0_camera_keys,
    _validate_droid_dreamzero_latent_cache_file,
)
from dreamfree.models.artifacts import _stage0_lora_checkpoint_identity
from dreamfree.models.distillation import (
    _normalize_teacher_target_token_scope,
    resolve_stage1_student_sigma_config,
)
from dreamfree.models.wan22.future_state import (
    FUTURE_TOKEN_ATTENTION_MODES,
    I2V_CONDITION_FIRST_IMAGE_MASK,
    I2V_CONDITION_NONE,
    TEACHER_FRAME0_CLEAN_FIRST_FRAME,
    TEACHER_FRAME0_NATIVE_NOISED,
    TEXT_CONDITION_SCOPE_FIRST_FRAME,
    TEXT_CONDITION_SCOPES,
    normalize_teacher_frame0_mode,
)
from dreamfree.models.wan22.stage0_loader import (
    STAGE1_STUDENT_INIT_MODES,
    STAGE1_STUDENT_INIT_PRETRAINED_BASE,
)
from dreamfree.training.runtime import _parse_optional_int
from dreamfree.training.stage0.constants import (
    HORIZONTAL_CONCAT_MULTI_CAMERA,
    LIBERO_CAMERA_KEYS,
    ROBOTWIN_CAMERA_KEYS,
)
from dreamfree.training.stage1.checkpoint import (
    validate_stage1_teacher_sigma_bounds_against_stage0_checkpoint,
)
from dreamfree.training.stage1.config import Stage1Config
from dreamfree.training.stage1.constants import (
    DROID_STAGE1_CAMERA_KEYS,
    DROID_STAGE1_CONCAT_MULTI_CAMERA,
    DROID_STAGE1_HEIGHT,
    DROID_STAGE1_WIDTH,
    LIBERO_STAGE1_DATASET_NAMES,
    ROBOCASA_STAGE1_CAMERA_KEYS,
    WAN22_A14B_DIT_ROLES,
    ScalarMetric,
)


def _validate_stage1_robocasa_dataset_config(
    cfg: Stage1Config, *, camera_keys: tuple[str, ...], dataset_dirs: list[str]
) -> None:
    dataset_format = _normalize_stage0_dataset_format(
        getattr(cfg, "dataset_format", "lerobot_root")
    )
    if dataset_format != "lerobot_root":
        raise ValueError("Stage 1 Robocasa training requires dataset_format='lerobot_root'.")
    if tuple(camera_keys) != tuple(ROBOCASA_STAGE1_CAMERA_KEYS):
        raise ValueError(
            f"Stage 1 Robocasa training requires camera_keys={','.join(ROBOCASA_STAGE1_CAMERA_KEYS)!r}."
        )
    layout = _stage1_robocasa_layout_spec(cfg)
    expected_concat = str(layout["camera_layout"])
    expected_size = (int(layout["height"]), int(layout["width"]))
    if str(cfg.camera_layout).strip().lower() != expected_concat:
        raise ValueError(f"Stage 1 Robocasa training requires camera_layout={expected_concat!r}.")
    if (int(cfg.height), int(cfg.width)) != expected_size:
        raise ValueError(
            f"Stage 1 Robocasa training requires height={expected_size[0]},width={expected_size[1]}, got {cfg.height}x{cfg.width}."
        )


def _validate_stage1_droid_dataset_config(
    cfg: Stage1Config, *, camera_keys: tuple[str, ...]
) -> None:
    dataset_format = _normalize_stage0_dataset_format(
        getattr(cfg, "dataset_format", "lerobot_root")
    )
    if dataset_format != "lerobot_root":
        raise ValueError("Stage 1 DROID training requires dataset_format='lerobot_root'.")
    if tuple(camera_keys) != tuple(DROID_STAGE1_CAMERA_KEYS):
        raise ValueError(
            f"Stage 1 DROID training requires camera_keys={','.join(DROID_STAGE1_CAMERA_KEYS)!r}."
        )
    if str(cfg.camera_layout).strip().lower() != DROID_STAGE1_CONCAT_MULTI_CAMERA:
        raise ValueError(
            f"Stage 1 DROID training requires camera_layout={DROID_STAGE1_CONCAT_MULTI_CAMERA!r}."
        )
    if (int(cfg.height), int(cfg.width)) != (DROID_STAGE1_HEIGHT, DROID_STAGE1_WIDTH):
        raise ValueError(
            f"Stage 1 DROID training requires height={DROID_STAGE1_HEIGHT},width={DROID_STAGE1_WIDTH}, got {cfg.height}x{cfg.width}."
        )


def _validate_stage1_robocasa_latent_cache_metadata(path: Path, cfg: Stage1Config) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
    expected = {
        "dataset_kind": "robocasa",
        "concat_layout": str(_stage1_robocasa_layout_spec(cfg)["camera_layout"]),
    }
    mismatches = [
        f"{key}: actual={meta.get(key)!r} expected={value!r}"
        for (key, value) in expected.items()
        if str(meta.get(key)) != str(value)
    ]
    if mismatches:
        raise ValueError(
            f"Stage 1 Robocasa latent cache metadata mismatch in {path}: " + "; ".join(mismatches)
        )


def _validate_stage1_robocasa_latent_cache_config(cfg: Stage1Config) -> None:
    latent_cache_dir = resolve_stage1_latent_cache_dir(cfg)
    cfg.video_latent_cache_dir = str(latent_cache_dir)
    if not latent_cache_dir.exists():
        raise FileNotFoundError(
            f"Stage 1 Robocasa video latent cache directory not found: {latent_cache_dir}. Precompute the cache first or set use_cached_latents=false."
        )
    first_episode = _first_stage1_latent_cache_file(latent_cache_dir)
    validate_stage2_latent_cache_file(
        first_episode,
        expected=_stage1_latent_cache_expected_contract(cfg),
        stage_label="Stage 1 Robocasa",
    )
    _validate_stage1_robocasa_latent_cache_metadata(first_episode, cfg)


def _validate_stage1_droid_latent_cache_config(cfg: Stage1Config) -> None:
    latent_cache_dir = resolve_stage1_latent_cache_dir(cfg)
    cfg.video_latent_cache_dir = str(latent_cache_dir)
    if not latent_cache_dir.exists():
        raise FileNotFoundError(
            f"Stage 1 DROID video latent cache directory not found: {latent_cache_dir}. Precompute the cache first or set use_cached_latents=false."
        )
    first_episode = _first_stage1_latent_cache_file(latent_cache_dir)
    _validate_droid_dreamzero_latent_cache_file(
        first_episode, expected=_stage1_latent_cache_expected_contract(cfg), stage_label="Stage 1"
    )


def _validate_stage1_latent_cache_config(cfg: Stage1Config) -> None:
    if not cfg.use_cached_latents:
        return
    if _is_stage1_robocasa_success_family_dataset(cfg):
        _validate_stage1_robocasa_latent_cache_config(cfg)
        return
    if _is_stage1_droid_dataset(cfg):
        _validate_stage1_droid_latent_cache_config(cfg)
        return
    _shared_validate_stage1_latent_cache_config(cfg)


def validate_stage1_config(cfg: Stage1Config) -> None:
    """Validate the strict Stage 1 v1 experiment contract.

    Args:
        cfg: Stage 1 configuration to validate before any expensive setup.

    Raises:
        ValueError: If a config field requests unsupported or incompatible
            Stage 1 behavior.
    """
    _normalize_stage0_dataset_format(getattr(cfg, "dataset_format", "lerobot_root"))
    camera_keys = _parse_stage0_camera_keys(
        getattr(cfg, "camera_keys", ROBOTWIN_CAMERA_KEYS),
        allow_single=bool(getattr(cfg, "single_view_adaptation_enabled", False)),
    )
    dataset_name = str(getattr(cfg, "dataset_name", "robotwin")).strip().lower()
    dataset_dirs = _parse_stage1_dataset_dirs(getattr(cfg, "dataset_dirs", None))
    if dataset_dirs and str(cfg.dataset_format) != "lerobot_root":
        raise ValueError("Stage 1 dataset_dirs requires dataset_format='lerobot_root'.")
    if int(getattr(cfg, "eval_batches", 1)) <= 0:
        raise ValueError(f"eval_batches must be positive, got {cfg.eval_batches}.")
    if dataset_name in LIBERO_STAGE1_DATASET_NAMES:
        if not dataset_dirs:
            raise ValueError(
                "Stage 1 LIBERO-family training requires dataset_dirs to list LeRobot roots."
            )
        allowed_camera_keys = {tuple(LIBERO_CAMERA_KEYS)}
        if tuple(camera_keys) not in allowed_camera_keys:
            allowed_text = ["[" + ",".join(keys) + "]" for keys in sorted(allowed_camera_keys)]
            raise ValueError(
                f"Stage 1 {dataset_name} requires camera_keys to be one of {allowed_text}."
            )
        if str(cfg.camera_layout).strip().lower() != HORIZONTAL_CONCAT_MULTI_CAMERA:
            raise ValueError("Stage 1 LIBERO-family requires camera_layout='horizontal'.")
        if (int(cfg.height), int(cfg.width)) != (224, 448):
            raise ValueError(
                f"Stage 1 LIBERO-family requires height=224,width=448, got {cfg.height}x{cfg.width}."
            )
    if _is_stage1_robocasa_success_family_dataset(cfg):
        _validate_stage1_robocasa_dataset_config(
            cfg, camera_keys=camera_keys, dataset_dirs=dataset_dirs
        )
    if _is_stage1_droid_dataset(cfg):
        _validate_stage1_droid_dataset_config(cfg, camera_keys=camera_keys)
    if str(cfg.prompt_policy).strip().lower() != "task_text":
        raise ValueError("Stage 1 v1 rejects prompt_policy other than 'task_text'.")
    if not bool(cfg.strict_task_text) or not bool(cfg.require_text_cache):
        raise ValueError("Stage 1 requires strict_task_text=true and require_text_cache=true.")
    if cfg.student_train_mode not in ("lora", "full"):
        raise ValueError("Stage 1 student_train_mode must be 'lora' or 'full'.")
    if bool(cfg.use_shared_teacher_student_dit) != (cfg.student_train_mode == "lora"):
        raise ValueError("Full Stage 1 requires a separate frozen teacher; LoRA shares the DiT.")
    if cfg.trim_frozen_teacher_blocks and cfg.student_train_mode != "full":
        raise ValueError("trim_frozen_teacher_blocks requires full Stage 1 training.")
    if cfg.student_train_mode == "lora" and (
        cfg.projector_learning_rate is not None or cfg.warmup_steps is not None
    ):
        raise ValueError("projector_learning_rate and warmup_steps require full Stage 1 training.")
    if bool(cfg.compile_dit) and bool(cfg.use_gradient_checkpointing):
        raise ValueError("compile_dit=true is not supported with use_gradient_checkpointing=true.")
    student_text_scope = str(cfg.student_text_scope).strip().lower()
    if student_text_scope not in TEXT_CONDITION_SCOPES:
        raise ValueError(
            f"Unsupported student_text_scope={cfg.student_text_scope!r}; expected one of {sorted(TEXT_CONDITION_SCOPES)}."
        )
    student_init_mode = str(cfg.student_init_mode).strip().lower()
    if student_init_mode not in STAGE1_STUDENT_INIT_MODES:
        raise ValueError(
            f"Unsupported student_init_mode={cfg.student_init_mode!r}; expected one of {sorted(STAGE1_STUDENT_INIT_MODES)}."
        )
    if (
        student_init_mode == STAGE1_STUDENT_INIT_PRETRAINED_BASE
        and _stage0_lora_checkpoint_identity(cfg.stage0_lora_checkpoint) is None
    ):
        raise ValueError(
            f"student_init_mode={student_init_mode!r} requires stage0_lora_checkpoint for the teacher adapter."
        )
    family = str(cfg.video_model_family)
    if family not in {"wan22_ti2v_5b", "wan22_i2v_a14b"}:
        raise ValueError(
            f"Unsupported video_model_family={cfg.video_model_family!r}; expected 'wan22_ti2v_5b' or 'wan22_i2v_a14b'."
        )
    teacher_frame0_mode = normalize_teacher_frame0_mode(cfg.teacher_frame0_mode)
    if family == "wan22_ti2v_5b":
        if str(cfg.model_id) != "Wan-AI/Wan2.2-TI2V-5B":
            raise ValueError(
                f"wan22_ti2v_5b requires model_id='Wan-AI/Wan2.2-TI2V-5B', got {cfg.model_id!r}."
            )
        if str(cfg.teacher_dit_role) != "single" or str(cfg.student_dit_role) != "single":
            raise ValueError("wan22_ti2v_5b requires teacher_dit_role=student_dit_role='single'.")
        if (
            int(cfg.video_hidden_dim),
            int(cfg.video_latent_dim),
            int(cfg.video_dit_input_dim),
            int(cfg.video_dit_output_dim),
        ) != (3072, 48, 48, 48):
            raise ValueError(
                "wan22_ti2v_5b requires video_hidden_dim=3072 and 48-channel latent/input/output dims."
            )
        if (
            str(cfg.teacher_i2v_condition_mode) != I2V_CONDITION_NONE
            or str(cfg.student_i2v_condition_mode) != I2V_CONDITION_NONE
        ):
            raise ValueError("wan22_ti2v_5b requires teacher/student i2v_condition_mode='none'.")
        if teacher_frame0_mode != TEACHER_FRAME0_CLEAN_FIRST_FRAME:
            raise ValueError(
                "wan22_ti2v_5b requires teacher_frame0_mode='clean_first_frame' to preserve legacy Stage 1 behavior."
            )
        if str(cfg.student_input_mode) != "flow_noised_video_first_frame_clean":
            raise ValueError(
                "Stage 1 requires flow_noised_video_first_frame_clean student input for wan22_ti2v_5b."
            )
        if _stage0_lora_checkpoint_identity(cfg.stage0_lora_checkpoint) is None:
            raise ValueError("wan22_ti2v_5b requires a Stage 0 LoRA checkpoint.")
    else:
        if str(cfg.model_id) != "Wan-AI/Wan2.2-I2V-A14B":
            raise ValueError(
                f"wan22_i2v_a14b requires model_id='Wan-AI/Wan2.2-I2V-A14B', got {cfg.model_id!r}."
            )
        teacher_role = str(cfg.teacher_dit_role)
        student_role = str(cfg.student_dit_role)
        if teacher_role not in WAN22_A14B_DIT_ROLES:
            raise ValueError(
                f"wan22_i2v_a14b requires teacher_dit_role in {sorted(WAN22_A14B_DIT_ROLES)}, got {cfg.teacher_dit_role!r}."
            )
        if student_role != teacher_role:
            raise ValueError(
                f"wan22_i2v_a14b Stage 1 uses a shared teacher/student DiT and requires student_dit_role to match teacher_dit_role={teacher_role!r}, got {cfg.student_dit_role!r}."
            )
        if (
            int(cfg.video_hidden_dim),
            int(cfg.video_latent_dim),
            int(cfg.video_dit_input_dim),
            int(cfg.video_dit_output_dim),
        ) != (5120, 16, 36, 16):
            raise ValueError(
                "wan22_i2v_a14b requires hidden_dim=5120, latent_dim=16, DiT in_dim=36, out_dim=16."
            )
        if str(cfg.teacher_i2v_condition_mode) != I2V_CONDITION_FIRST_IMAGE_MASK:
            raise ValueError(
                "wan22_i2v_a14b requires teacher_i2v_condition_mode='first_image_mask_condition'."
            )
        if teacher_frame0_mode != TEACHER_FRAME0_NATIVE_NOISED:
            raise ValueError(
                "wan22_i2v_a14b requires teacher_frame0_mode='native_noised' so the teacher matches 14B Stage 0."
            )
        if str(cfg.student_i2v_condition_mode) != I2V_CONDITION_FIRST_IMAGE_MASK:
            raise ValueError(
                "wan22_i2v_a14b requires student_i2v_condition_mode='first_image_mask_condition'."
            )
        if str(cfg.student_input_mode) != "wan22_i2v_clean_first_frame":
            raise ValueError(
                "wan22_i2v_a14b requires student_input_mode='wan22_i2v_clean_first_frame'."
            )
        if str(cfg.stage0_expected_video_model_family) != "wan22_i2v_a14b":
            raise ValueError(
                "wan22_i2v_a14b requires stage0_expected_video_model_family='wan22_i2v_a14b'."
            )
        if str(cfg.stage0_expected_dit_role) != teacher_role:
            raise ValueError(
                f"wan22_i2v_a14b requires stage0_expected_dit_role to match teacher_dit_role={teacher_role!r}, got {cfg.stage0_expected_dit_role!r}."
            )
        if str(cfg.stage0_expected_model_id) != "Wan-AI/Wan2.2-I2V-A14B":
            raise ValueError(
                "wan22_i2v_a14b requires stage0_expected_model_id='Wan-AI/Wan2.2-I2V-A14B'."
            )
        if _stage0_lora_checkpoint_identity(cfg.stage0_lora_checkpoint) is None:
            raise ValueError(f"wan22_i2v_a14b requires a 14B {teacher_role} stage 0 checkpoint.")
    resolve_stage1_student_sigma_config(cfg, require_stage0_metadata=False)
    if str(cfg.teacher_target_type) != "hidden":
        raise ValueError("Stage 1 requires teacher_target_type='hidden'.")
    _normalize_teacher_target_token_scope(cfg.teacher_target_token_scope)
    if str(cfg.teacher_hook_point) != "post_block" or str(cfg.student_hook_point) != "post_block":
        raise ValueError("Stage 1 requires teacher/student hook_point='post_block'.")
    if str(cfg.teacher_video_attention_mask_mode) != "bidirectional":
        raise ValueError("Stage 1 requires teacher_video_attention_mask_mode='bidirectional'.")
    if str(cfg.student_video_attention_mask_mode) != "first_frame_causal":
        raise ValueError("Stage 1 requires student_video_attention_mask_mode='first_frame_causal'.")
    if str(cfg.future_token_attention).strip().lower() not in FUTURE_TOKEN_ATTENTION_MODES:
        raise ValueError(
            f"Unsupported future_token_attention={cfg.future_token_attention!r}; expected one of {sorted(FUTURE_TOKEN_ATTENTION_MODES)}."
        )
    if (
        student_text_scope == TEXT_CONDITION_SCOPE_FIRST_FRAME
        and str(cfg.student_video_attention_mask_mode) != "first_frame_causal"
    ):
        raise ValueError(
            "student_text_scope='first_frame' requires student_video_attention_mask_mode='first_frame_causal'."
        )
    if str(cfg.projector_type) != "first_frame_decoder":
        if str(cfg.projector_type) == "token_mlp":
            raise ValueError(
                "Stage 1 v1 rejects projector_type='token_mlp'; use 'first_frame_decoder'."
            )
        raise ValueError("Stage 1 requires projector_type='first_frame_decoder'.")
    if str(cfg.distillation_loss) != "cosine":
        raise ValueError("DreamFree Stage 1 requires distillation_loss='cosine'.")
    if int(cfg.clip_frames) <= 1 or int(cfg.clip_frames) % 4 != 1:
        raise ValueError(f"clip_frames must satisfy >1 and %4==1, got {cfg.clip_frames}.")
    if int(cfg.raw_frame_stride) <= 0 or int(cfg.sample_stride_frames) <= 0:
        raise ValueError(
            f"raw_frame_stride and sample_stride_frames must be positive, got {cfg.raw_frame_stride} and {cfg.sample_stride_frames}."
        )
    if int(cfg.height) % 16 != 0 or int(cfg.width) % 16 != 0:
        raise ValueError(f"height/width must be multiples of 16, got {cfg.height}x{cfg.width}.")
    if int(cfg.teacher_layer_k) < 0 or int(cfg.student_layer_j) < 0:
        raise ValueError(
            "teacher_layer_k and student_layer_j must be zero-based non-negative indices."
        )
    max_layers = 40 if family == "wan22_i2v_a14b" else 30
    if int(cfg.teacher_layer_k) >= max_layers or int(cfg.student_layer_j) >= max_layers:
        raise ValueError(
            f"{family} has {max_layers} DiT layers; got teacher_layer_k={cfg.teacher_layer_k}, student_layer_j={cfg.student_layer_j}."
        )
    if not 0.0 <= float(cfg.teacher_sigma_min) <= float(cfg.teacher_sigma_max) <= 1.0:
        raise ValueError(
            f"teacher sigma bounds must satisfy 0 <= min <= max <= 1, got {cfg.teacher_sigma_min}, {cfg.teacher_sigma_max}."
        )
    if int(cfg.projector_dim) <= 0:
        raise ValueError(f"projector_dim must be positive, got {cfg.projector_dim}.")
    if int(cfg.projector_num_layers) <= 0:
        raise ValueError(f"projector_num_layers must be positive, got {cfg.projector_num_layers}.")
    if (
        int(cfg.projector_num_heads) <= 0
        or int(cfg.projector_dim) % int(cfg.projector_num_heads) != 0
    ):
        raise ValueError(
            f"projector_num_heads must divide projector_dim, got {cfg.projector_num_heads} and {cfg.projector_dim}."
        )
    if float(cfg.distillation_loss_weight) < 0.0:
        raise ValueError(
            f"distillation_loss_weight must be non-negative, got {cfg.distillation_loss_weight}."
        )
    if float(cfg.stage0_expected_lora_dropout) != 0.0:
        raise ValueError("Stage 1 requires stage0_expected_lora_dropout=0.0 before merge.")
    if int(cfg.tokenizer_max_len) != 128 or int(cfg.text_dim) != 4096:
        raise ValueError(
            "Stage 1 v1 cached text shape requires tokenizer_max_len=128 and text_dim=4096."
        )
    if cfg.num_epochs is not None:
        num_epochs = float(cfg.num_epochs)
        if not math.isfinite(num_epochs) or num_epochs <= 0.0:
            raise ValueError(f"num_epochs must be a finite positive value, got {cfg.num_epochs}.")
    try:
        max_checkpoints_to_keep = _parse_optional_int(getattr(cfg, "max_checkpoints_to_keep", None))
    except argparse.ArgumentTypeError as err:
        raise ValueError(
            f"max_checkpoints_to_keep must be null or a positive integer, got {cfg.max_checkpoints_to_keep}."
        ) from err
    if max_checkpoints_to_keep is not None and max_checkpoints_to_keep <= 0:
        raise ValueError(
            f"max_checkpoints_to_keep must be null or a positive integer, got {cfg.max_checkpoints_to_keep}."
        )
    if int(cfg.max_steps) < 0:
        raise ValueError(f"max_steps must be non-negative, got {cfg.max_steps}.")
    _validate_stage1_latent_cache_config(cfg)


def validate_stage1_launch_config(cfg: Stage1Config) -> None:
    """Validate all Stage 1 launch contracts before distributed setup."""
    validate_stage1_config(cfg)
    if not bool(cfg.prepare_only):
        validate_stage1_teacher_sigma_bounds_against_stage0_checkpoint(cfg)


def _scalar_metric_to_tensor(value: ScalarMetric, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar metric tensor, got shape {tuple(value.shape)}.")
        return value.detach().to(device=device, dtype=torch.float32).reshape(())
    return torch.tensor(float(value), device=device, dtype=torch.float32)


def _gather_scalar_metrics(
    metrics: dict[str, ScalarMetric], *, device: torch.device, gather_fn: Any
) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(metrics)
    values = torch.stack(
        [_scalar_metric_to_tensor(metrics[key], device=device) for key in keys]
    ).reshape(1, -1)
    gathered = gather_fn(values).reshape(-1, len(keys))
    means = gathered.mean(dim=0).detach().cpu().tolist()
    return {key: float(value) for (key, value) in zip(keys, means, strict=True)}


@torch.no_grad()
def evaluate_one_stage1_batch(
    *, accelerator: Accelerator, model: torch.nn.Module, val_iter: Any
) -> dict[str, float] | None:
    """Evaluate one validation batch and gather metrics across processes.

    Args:
        accelerator: Accelerator used to unwrap the model and gather metrics.
        model: Possibly wrapped Stage 1 model.
        val_iter: Iterator over prepared validation batches.

    Returns:
        Mean validation metrics, or `None` if the iterator is exhausted.
    """
    try:
        sample = next(val_iter)
    except StopIteration:
        return None
    unwrapped = accelerator.unwrap_model(model)
    was_training = unwrapped.training
    unwrapped.eval()
    (loss, loss_dict) = model(sample)
    mean_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
    gathered = _gather_scalar_metrics(loss_dict, device=loss.device, gather_fn=accelerator.gather)
    if was_training:
        unwrapped.set_stage1_train_mode()
    return {
        "val_loss": float(mean_loss),
        **{f"val_{key}": float(value) for (key, value) in gathered.items()},
    }


@torch.no_grad()
def evaluate_stage1_batches(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    val_iter: Any,
    val_loader: Any,
    cfg: Stage1Config,
    evaluate_one_batch_fn=evaluate_one_stage1_batch,
) -> tuple[dict[str, float] | None, Any]:
    """Evaluate one or more Stage 1 validation batches and average metrics."""
    max_batches = int(cfg.eval_batches)
    if max_batches <= 0:
        raise ValueError(f"eval_batches must be positive, got {cfg.eval_batches}.")
    metric_sums: dict[str, float] = {}
    completed_batches = 0
    for _ in range(max_batches):
        metrics = evaluate_one_batch_fn(accelerator=accelerator, model=model, val_iter=val_iter)
        if metrics is None:
            if val_loader is None:
                break
            val_iter = iter(val_loader)
            metrics = evaluate_one_batch_fn(accelerator=accelerator, model=model, val_iter=val_iter)
        if metrics is None:
            break
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and (not isinstance(value, bool)):
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        completed_batches += 1
    if completed_batches <= 0:
        return (None, val_iter)
    averaged = {
        key: float(value / completed_batches) for (key, value) in sorted(metric_sums.items())
    }
    averaged["val_batches"] = float(completed_batches)
    return (averaged, val_iter)
