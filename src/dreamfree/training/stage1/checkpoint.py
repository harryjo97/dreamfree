from __future__ import annotations

import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from dreamfree.data.distillation import _stage1_visual_preprocess_id
from dreamfree.models.artifacts import _optional_path_identity, _stage0_lora_checkpoint_identity
from dreamfree.models.distillation import (
    DistillationModel,
    _normalize_teacher_target_token_scope,
    resolve_stage1_student_sigma_config,
)
from dreamfree.models.lora import collect_lora_state_dict, load_lora_state_dict
from dreamfree.models.wan22.future_state import (
    FUTURE_TOKEN_ATTENTION_FULL,
    I2V_CONDITION_NONE,
    TEACHER_FRAME0_CLEAN_FIRST_FRAME,
    TEACHER_STATE_LAYERNORM_EPS,
    TEACHER_STATE_LAYERNORM_TYPE,
    TEACHER_STATE_LOSS_AFFINE,
    TEXT_CONDITION_SCOPE_ALL_FRAMES,
    normalize_teacher_frame0_mode,
)
from dreamfree.models.wan22.stage0_loader import (
    STAGE1_STUDENT_INIT_PRETRAINED_BASE,
    STAGE1_STUDENT_INIT_STAGE0_MERGED,
    STAGE1_STUDENT_LORA_ADAPTER_NAME,
    STAGE1_TEACHER_LORA_ADAPTER_NAME,
    validate_teacher_sigma_bounds_against_stage0_checkpoint,
)
from dreamfree.training.runtime import (
    _atomic_torch_save,
    _json_dump,
    _json_load,
    _parse_optional_int,
)
from dreamfree.training.stage1.config import Stage1Config
from dreamfree.training.stage1.constants import (
    STAGE1_CHECKPOINT_DIR_RE,
    STAGE1_CHECKPOINT_TYPE,
    STAGE1_CHECKPOINT_VERSION,
    STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL,
    TEACHER_TARGET_TOKEN_SCOPE_ALL,
    logger,
)


def validate_stage1_teacher_sigma_bounds_against_stage0_checkpoint(cfg: Stage1Config) -> None:
    validate_teacher_sigma_bounds_against_stage0_checkpoint(cfg, stage_label="Stage 1")


def build_stage1_checkpoint_metadata(
    *, cfg: Stage1Config, model: DistillationModel, global_step: int
) -> dict[str, Any]:
    """Build the JSON-serializable metadata saved with a Stage 1 checkpoint.

    Args:
        cfg: Stage 1 configuration used for the run.
        model: Unwrapped Stage 1 model containing Stage 0 and Stage 1 metadata.
        global_step: Training step represented by the checkpoint.

    Returns:
        Metadata dictionary used for checkpoint validation and resume safety.
    """
    shape_meta = dict(model.last_shape_metadata)
    uses_stage0_teacher = bool(dict(model.stage0_lora_config).get("enabled", True))
    student_init_mode = str(getattr(model, "student_init_mode", cfg.student_init_mode))
    student_text_scope = str(cfg.student_text_scope).strip().lower()
    stage0_merged_into_student_base = bool(getattr(model, "stage0_merged_into_student_base", True))
    student_sigma_config = getattr(model, "student_sigma_config", None)
    if student_sigma_config is None:
        student_sigma_config = resolve_stage1_student_sigma_config(
            cfg, require_stage0_metadata=False
        )
    return {
        "checkpoint_type": STAGE1_CHECKPOINT_TYPE,
        "checkpoint_version": STAGE1_CHECKPOINT_VERSION,
        "global_step": int(global_step),
        **(
            {
                "visual_preprocess_id": _stage1_visual_preprocess_id(cfg),
                "source_stage0_visual_preprocess_id": model.stage0_trainer_state.get(
                    "visual_preprocess_id"
                ),
            }
            if _stage1_visual_preprocess_id(cfg) is not None
            else {}
        ),
        "config": asdict(cfg),
        **{},
        "model_contract": {
            "video_model_family": str(cfg.video_model_family),
            "model_id": str(cfg.model_id),
            "teacher_dit_role": str(cfg.teacher_dit_role),
            "student_dit_role": str(cfg.student_dit_role),
            "video_hidden_dim": int(cfg.video_hidden_dim),
            "video_latent_dim": int(cfg.video_latent_dim),
            "video_dit_input_dim": int(cfg.video_dit_input_dim),
            "video_dit_output_dim": int(cfg.video_dit_output_dim),
        },
        "stage0_adaptation": {
            "enabled": bool(uses_stage0_teacher),
            "debug_base_without_stage0": False,
            "video_model_family": str(cfg.stage0_expected_video_model_family),
            "dit_role": str(cfg.stage0_expected_dit_role),
            "vae_family": "wan2.1" if str(cfg.video_model_family) == "wan22_i2v_a14b" else "wan2.2",
        },
        "stage0_lora_checkpoint": _stage0_lora_checkpoint_identity(cfg.stage0_lora_checkpoint)
        if uses_stage0_teacher
        else None,
        "stage0_eval_reference": _optional_path_identity(cfg.stage0_eval_reference)
        if uses_stage0_teacher
        else None,
        "stage0_expected_global_step": int(cfg.stage0_expected_global_step)
        if uses_stage0_teacher
        else None,
        "stage0_global_step": int(model.stage0_trainer_state["global_step"]),
        "stage0_lora_config": dict(model.stage0_lora_config),
        "student_init_mode": student_init_mode,
        "stage0_merged_into_student_base": stage0_merged_into_student_base,
        "teacher_lora_adapter_name": getattr(model, "teacher_lora_adapter_name", None),
        "student_lora_adapter_name": getattr(model, "student_lora_adapter_name", None),
        "stage1_lora_config": dict(model.stage1_lora_config),
        "projector": {
            "type": str(cfg.projector_type),
            "projector_dim": int(cfg.projector_dim),
            "num_layers": int(cfg.projector_num_layers),
            "num_heads": int(cfg.projector_num_heads),
            "mlp_ratio": float(cfg.projector_mlp_ratio),
        },
        "teacher": {
            "target_type": str(cfg.teacher_target_type),
            "layer_k": int(cfg.teacher_layer_k),
            "hook_point": str(cfg.teacher_hook_point),
            "video_attention_mask_mode": str(cfg.teacher_video_attention_mask_mode),
            "i2v_condition_mode": str(cfg.teacher_i2v_condition_mode),
            "frame0_mode": normalize_teacher_frame0_mode(cfg.teacher_frame0_mode),
            "sigma_policy": str(cfg.teacher_sigma_policy),
            "sigma_min": float(cfg.teacher_sigma_min),
            "sigma_max": float(cfg.teacher_sigma_max),
        },
        "student": {
            "layer_j": int(cfg.student_layer_j),
            "hook_point": str(cfg.student_hook_point),
            "video_attention_mask_mode": str(cfg.student_video_attention_mask_mode),
            "future_token_attention": str(cfg.future_token_attention).strip().lower(),
            "text_scope": student_text_scope,
            "input_mode": str(cfg.student_input_mode),
            "i2v_condition_mode": str(cfg.student_i2v_condition_mode),
            "sigma_policy": str(student_sigma_config.sigma_policy),
            "sigma_min": float(student_sigma_config.sigma_min)
            if student_sigma_config.sigma_min is not None
            else None,
            "sigma_max": float(student_sigma_config.sigma_max)
            if student_sigma_config.sigma_max is not None
            else None,
            "sigma_source": str(student_sigma_config.source),
        },
        "teacher_state": {
            "loss": str(cfg.teacher_state_loss),
            "lambda_teacher_state": float(cfg.lambda_teacher_state),
            "target_token_scope": _normalize_teacher_target_token_scope(
                cfg.teacher_target_token_scope
            ),
            "normalization": {
                "type": TEACHER_STATE_LAYERNORM_TYPE,
                "eps": float(TEACHER_STATE_LAYERNORM_EPS),
                "affine_in_loss": bool(TEACHER_STATE_LOSS_AFFINE),
            },
        },
        "shapes": shape_meta,
    }


def validate_stage1_checkpoint_metadata(cfg: Stage1Config, metadata: dict[str, Any]) -> None:
    """Validate that a checkpoint is compatible with the requested run config.

    Args:
        cfg: Current Stage 1 config.
        metadata: Loaded `trainer_state.json` payload from a checkpoint.

    Raises:
        ValueError: If checkpoint identity, LoRA config,
            teacher/student/projector settings, or recorded tensor shapes do
            not match the current config.
    """
    mismatches: list[str] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        """Record a named mismatch without stopping at the first difference."""
        if actual != expected:
            mismatches.append(f"{name}: actual={actual!r} expected={expected!r}")

    check("checkpoint_type", metadata.get("checkpoint_type"), STAGE1_CHECKPOINT_TYPE)
    check("checkpoint_version", metadata.get("checkpoint_version"), STAGE1_CHECKPOINT_VERSION)
    if metadata.get("episode_subset") is not None:
        check("episode_subset", metadata.get("episode_subset"), None)
    model_contract = metadata.get("model_contract", {})
    metadata_config = metadata.get("config", {})
    expected_visual_id = _stage1_visual_preprocess_id(cfg)
    if expected_visual_id is not None:
        check("visual_preprocess_id", metadata.get("visual_preprocess_id"), expected_visual_id)
        check(
            "source_stage0_visual_preprocess_id",
            metadata.get("source_stage0_visual_preprocess_id"),
            expected_visual_id,
        )
    check(
        "model_contract.video_model_family",
        model_contract.get(
            "video_model_family", metadata_config.get("video_model_family", "wan22_ti2v_5b")
        ),
        str(cfg.video_model_family),
    )
    check(
        "model_contract.model_id",
        model_contract.get("model_id", metadata_config.get("model_id")),
        str(cfg.model_id),
    )
    check(
        "model_contract.teacher_dit_role",
        model_contract.get("teacher_dit_role", metadata_config.get("teacher_dit_role", "single")),
        str(cfg.teacher_dit_role),
    )
    check(
        "model_contract.student_dit_role",
        model_contract.get("student_dit_role", metadata_config.get("student_dit_role", "single")),
        str(cfg.student_dit_role),
    )
    check(
        "model_contract.video_hidden_dim",
        int(model_contract.get("video_hidden_dim", metadata_config.get("video_hidden_dim", -1))),
        int(cfg.video_hidden_dim),
    )
    check(
        "model_contract.video_latent_dim",
        int(model_contract.get("video_latent_dim", metadata_config.get("video_latent_dim", 48))),
        int(cfg.video_latent_dim),
    )
    check(
        "model_contract.video_dit_input_dim",
        int(
            model_contract.get(
                "video_dit_input_dim", metadata_config.get("video_dit_input_dim", 48)
            )
        ),
        int(cfg.video_dit_input_dim),
    )
    check(
        "model_contract.video_dit_output_dim",
        int(
            model_contract.get(
                "video_dit_output_dim", metadata_config.get("video_dit_output_dim", 48)
            )
        ),
        int(cfg.video_dit_output_dim),
    )
    loaded_student_init_mode = str(
        metadata.get(
            "student_init_mode",
            metadata_config.get("student_init_mode", STAGE1_STUDENT_INIT_STAGE0_MERGED),
        )
    )
    check("student_init_mode", loaded_student_init_mode, str(cfg.student_init_mode))
    expected_stage0_merged = loaded_student_init_mode == STAGE1_STUDENT_INIT_STAGE0_MERGED
    check(
        "stage0_merged_into_student_base",
        bool(metadata.get("stage0_merged_into_student_base", True)),
        expected_stage0_merged,
    )
    expected_teacher_adapter = (
        STAGE1_TEACHER_LORA_ADAPTER_NAME
        if loaded_student_init_mode == STAGE1_STUDENT_INIT_PRETRAINED_BASE
        else None
    )
    expected_student_adapter = (
        STAGE1_STUDENT_LORA_ADAPTER_NAME
        if loaded_student_init_mode == STAGE1_STUDENT_INIT_PRETRAINED_BASE
        else None
    )
    check(
        "teacher_lora_adapter_name",
        metadata.get("teacher_lora_adapter_name"),
        expected_teacher_adapter,
    )
    check(
        "student_lora_adapter_name",
        metadata.get("student_lora_adapter_name"),
        expected_student_adapter,
    )
    check(
        "stage0_lora_checkpoint",
        metadata.get("stage0_lora_checkpoint"),
        _stage0_lora_checkpoint_identity(cfg.stage0_lora_checkpoint),
    )
    uses_stage0_teacher = _stage0_lora_checkpoint_identity(cfg.stage0_lora_checkpoint) is not None
    expected_stage0_eval_reference = (
        _optional_path_identity(cfg.stage0_eval_reference) if uses_stage0_teacher else None
    )
    expected_stage0_global_step = (
        int(cfg.stage0_expected_global_step) if uses_stage0_teacher else None
    )
    check(
        "stage0_eval_reference",
        metadata.get("stage0_eval_reference"),
        expected_stage0_eval_reference,
    )
    check(
        "stage0_expected_global_step",
        metadata.get("stage0_expected_global_step"),
        expected_stage0_global_step,
    )
    check("teacher.layer_k", metadata.get("teacher", {}).get("layer_k"), int(cfg.teacher_layer_k))
    check(
        "teacher.hook_point",
        metadata.get("teacher", {}).get("hook_point"),
        str(cfg.teacher_hook_point),
    )
    check(
        "teacher.video_attention_mask_mode",
        metadata.get("teacher", {}).get("video_attention_mask_mode"),
        str(cfg.teacher_video_attention_mask_mode),
    )
    check(
        "teacher.i2v_condition_mode",
        metadata.get("teacher", {}).get("i2v_condition_mode", I2V_CONDITION_NONE),
        str(cfg.teacher_i2v_condition_mode),
    )
    check(
        "teacher.frame0_mode",
        normalize_teacher_frame0_mode(
            metadata.get("teacher", {}).get("frame0_mode", TEACHER_FRAME0_CLEAN_FIRST_FRAME)
        ),
        normalize_teacher_frame0_mode(cfg.teacher_frame0_mode),
    )
    check(
        "teacher.sigma_policy",
        metadata.get("teacher", {}).get("sigma_policy"),
        str(cfg.teacher_sigma_policy),
    )
    check(
        "teacher.sigma_min",
        float(metadata.get("teacher", {}).get("sigma_min", -1)),
        float(cfg.teacher_sigma_min),
    )
    check(
        "teacher.sigma_max",
        float(metadata.get("teacher", {}).get("sigma_max", -1)),
        float(cfg.teacher_sigma_max),
    )
    check("student.layer_j", metadata.get("student", {}).get("layer_j"), int(cfg.student_layer_j))
    check(
        "student.hook_point",
        metadata.get("student", {}).get("hook_point"),
        str(cfg.student_hook_point),
    )
    check(
        "student.video_attention_mask_mode",
        metadata.get("student", {}).get("video_attention_mask_mode"),
        str(cfg.student_video_attention_mask_mode),
    )
    check(
        "student.future_token_attention",
        str(
            metadata.get("student", {}).get(
                "future_token_attention",
                metadata_config.get("future_token_attention", FUTURE_TOKEN_ATTENTION_FULL),
            )
        )
        .strip()
        .lower(),
        str(cfg.future_token_attention).strip().lower(),
    )
    check(
        "student.text_scope",
        metadata.get("student", {}).get(
            "text_scope", metadata_config.get("student_text_scope", TEXT_CONDITION_SCOPE_ALL_FRAMES)
        ),
        str(cfg.student_text_scope).strip().lower(),
    )
    check(
        "student.input_mode",
        metadata.get("student", {}).get("input_mode"),
        str(cfg.student_input_mode),
    )
    check(
        "student.i2v_condition_mode",
        metadata.get("student", {}).get("i2v_condition_mode", I2V_CONDITION_NONE),
        str(cfg.student_i2v_condition_mode),
    )
    expected_student_sigma = resolve_stage1_student_sigma_config(cfg, require_stage0_metadata=False)
    student_metadata = metadata.get("student", {})
    check(
        "student.sigma_policy",
        student_metadata.get("sigma_policy", STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL),
        expected_student_sigma.sigma_policy,
    )
    actual_student_sigma_min = student_metadata.get("sigma_min")
    actual_student_sigma_max = student_metadata.get("sigma_max")
    check(
        "student.sigma_min",
        None if actual_student_sigma_min is None else float(actual_student_sigma_min),
        expected_student_sigma.sigma_min,
    )
    check(
        "student.sigma_max",
        None if actual_student_sigma_max is None else float(actual_student_sigma_max),
        expected_student_sigma.sigma_max,
    )
    if "sigma_source" in student_metadata:
        check(
            "student.sigma_source",
            student_metadata.get("sigma_source"),
            expected_student_sigma.source,
        )
    check("projector.type", metadata.get("projector", {}).get("type"), str(cfg.projector_type))
    check(
        "projector.projector_dim",
        metadata.get("projector", {}).get("projector_dim"),
        int(cfg.projector_dim),
    )
    check(
        "projector.num_layers",
        metadata.get("projector", {}).get("num_layers"),
        int(cfg.projector_num_layers),
    )
    check(
        "projector.num_heads",
        metadata.get("projector", {}).get("num_heads"),
        int(cfg.projector_num_heads),
    )
    check(
        "projector.mlp_ratio",
        float(metadata.get("projector", {}).get("mlp_ratio", -1)),
        float(cfg.projector_mlp_ratio),
    )
    loaded_lora = metadata.get("stage1_lora_config", {})
    check("stage1_lora.rank", int(loaded_lora.get("rank", -1)), int(cfg.stage1_lora_rank))
    check("stage1_lora.alpha", float(loaded_lora.get("alpha", -1)), float(cfg.stage1_lora_alpha))
    check(
        "stage1_lora.dropout", float(loaded_lora.get("dropout", -1)), float(cfg.stage1_lora_dropout)
    )
    check(
        "stage1_lora.target_modules",
        loaded_lora.get("target_modules"),
        str(cfg.stage1_lora_target_modules),
    )
    teacher_state = metadata.get("teacher_state", {})
    check("teacher_state.loss", teacher_state.get("loss"), str(cfg.teacher_state_loss))
    check(
        "teacher_state.lambda_teacher_state",
        float(teacher_state.get("lambda_teacher_state", -1)),
        float(cfg.lambda_teacher_state),
    )
    check(
        "teacher_state.target_token_scope",
        _normalize_teacher_target_token_scope(
            teacher_state.get(
                "target_token_scope",
                metadata_config.get("teacher_target_token_scope", TEACHER_TARGET_TOKEN_SCOPE_ALL),
            )
        ),
        _normalize_teacher_target_token_scope(cfg.teacher_target_token_scope),
    )
    normalization = teacher_state.get("normalization", {})
    check(
        "teacher_state.normalization.type", normalization.get("type"), TEACHER_STATE_LAYERNORM_TYPE
    )
    check(
        "teacher_state.normalization.eps",
        float(normalization.get("eps", -1)),
        float(TEACHER_STATE_LAYERNORM_EPS),
    )
    check(
        "teacher_state.normalization.affine_in_loss",
        bool(normalization.get("affine_in_loss")) if "affine_in_loss" in normalization else None,
        bool(TEACHER_STATE_LOSS_AFFINE),
    )
    global_step = int(metadata.get("global_step", -1))
    if global_step < 0:
        mismatches.append(f"global_step must be non-negative, got {global_step}")
    shapes = metadata.get("shapes", {})
    for key in (
        "latent_shape",
        "teacher_grid_size",
        "tokens_per_frame",
        "teacher_sequence_length",
        "projector_input_tokens",
        "projector_output_tokens",
        "D_student",
        "D_teacher",
        "teacher_state_loss",
    ):
        if key not in shapes:
            mismatches.append(f"shapes.{key} is missing")
    if "teacher_target_token_scope" in shapes:
        check(
            "shapes.teacher_target_token_scope",
            _normalize_teacher_target_token_scope(shapes["teacher_target_token_scope"]),
            _normalize_teacher_target_token_scope(cfg.teacher_target_token_scope),
        )
    if shapes.get("D_student") is not None:
        check("D_student", int(shapes["D_student"]), int(cfg.video_hidden_dim))
    if shapes.get("D_teacher") is not None:
        check("D_teacher", int(shapes["D_teacher"]), int(cfg.video_hidden_dim))
    if shapes.get("teacher_state_loss") is not None:
        check("teacher_state_loss", shapes["teacher_state_loss"], str(cfg.teacher_state_loss))
    if mismatches:
        raise ValueError("Stage 1 checkpoint metadata mismatch: " + "; ".join(mismatches))


def _resolve_resume_dir(cfg: Stage1Config) -> Path | None:
    """Resolve and validate the optional Stage 1 resume directory.

    Args:
        cfg: Stage 1 configuration with optional `resume` path.

    Returns:
        Resolved checkpoint directory, or `None` when resume is disabled.

    Raises:
        ValueError: If `resume` points to a file.
        FileNotFoundError: If the requested directory does not exist.
    """
    if not cfg.resume:
        return None
    resume = Path(cfg.resume).expanduser().resolve()
    if resume.is_file():
        raise ValueError(f"Stage 1 resume must point to a checkpoint directory, got file: {resume}")
    if not resume.is_dir():
        raise FileNotFoundError(f"Stage 1 resume directory not found: {resume}")
    return resume


def _read_resume_metadata(cfg: Stage1Config) -> tuple[Path, dict[str, Any]] | None:
    """Read and validate resume checkpoint metadata before model loading.

    Args:
        cfg: Stage 1 configuration with optional resume path.

    Returns:
        Tuple of `(resume_dir, metadata)` when resuming, otherwise `None`.

    Raises:
        FileNotFoundError: If a required checkpoint file is missing.
        ValueError: If checkpoint metadata is incompatible with `cfg`.
    """
    resume_dir = _resolve_resume_dir(cfg)
    if resume_dir is None:
        return None
    required = ["trainer_state.json", "stage1_lora.safetensors", "projector.safetensors"]
    if bool(cfg.resume_optimizer_state):
        required.append("optim.pt")
    missing = [name for name in required if not (resume_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Stage 1 resume checkpoint is missing files {missing}: {resume_dir}"
        )
    metadata = _json_load(resume_dir / "trainer_state.json")
    validate_stage1_checkpoint_metadata(cfg, metadata)
    step_match = re.fullmatch("step_(\\d+)", resume_dir.name)
    if step_match is not None and int(step_match.group(1)) != int(metadata["global_step"]):
        raise ValueError(
            f"Resume directory step mismatch: dir={resume_dir.name} metadata.global_step={metadata['global_step']}"
        )
    return (resume_dir, metadata)


def _prune_stage1_checkpoints(cfg: Stage1Config) -> list[Path]:
    """Remove older Stage 1 checkpoint directories when retention is configured."""
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
        match = STAGE1_CHECKPOINT_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), child))
    checkpoints.sort(key=lambda item: (item[0], item[1].name))
    stale = checkpoints[: max(0, len(checkpoints) - max_checkpoints_to_keep)]
    removed: list[Path] = []
    for _, path in stale:
        shutil.rmtree(path)
        removed.append(path)
        logger.info("[ckpt] removed old Stage 1 checkpoint %s", path)
    return removed


def save_stage1_checkpoint(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: Stage1Config,
    global_step: int,
) -> Path | None:
    """Save Stage 1 LoRA, projector, metadata, and optional optimizer state.

    Args:
        accelerator: Accelerator coordinating distributed processes.
        model: Possibly wrapped Stage 1 model.
        optimizer: Optimizer whose state may be saved.
        scheduler: Learning-rate scheduler whose state may be saved.
        cfg: Stage 1 configuration.
        global_step: Training step to encode in the checkpoint directory name.

    Returns:
        Checkpoint directory on the main process, otherwise `None`.
    """
    checkpoint_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{int(global_step):06d}"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped: DistillationModel = accelerator.unwrap_model(model)
        save_safetensors(
            collect_lora_state_dict(
                unwrapped.core.dit, adapter_name=unwrapped.student_lora_adapter_name
            ),
            str(checkpoint_dir / "stage1_lora.safetensors"),
        )
        projector_state = {
            name: tensor.detach().cpu().contiguous()
            for (name, tensor) in unwrapped.projector.state_dict().items()
        }
        save_safetensors(projector_state, str(checkpoint_dir / "projector.safetensors"))
        _json_dump(
            checkpoint_dir / "trainer_state.json",
            build_stage1_checkpoint_metadata(cfg=cfg, model=unwrapped, global_step=global_step),
        )
        if cfg.save_optimizer_state:
            _atomic_torch_save(
                checkpoint_dir / "optim.pt",
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            )
        logger.info("[ckpt] saved Stage 1 checkpoint to %s", checkpoint_dir)
        _prune_stage1_checkpoints(cfg)
    accelerator.wait_for_everyone()
    return checkpoint_dir if accelerator.is_main_process else None


def _load_stage1_optimizer_state_if_requested(
    *, optimizer: torch.optim.Optimizer, scheduler: Any, cfg: Stage1Config
) -> None:
    """Restore optimizer and scheduler state when a resume checkpoint provides it.

    Args:
        optimizer: Optimizer created for the current wrapped model.
        scheduler: Learning-rate scheduler created for the current optimizer.
        cfg: Stage 1 configuration with resume and save-state flags.

    Raises:
        FileNotFoundError: If optimizer state is required but absent.
    """
    resume_dir = _resolve_resume_dir(cfg)
    if resume_dir is None:
        return
    if not bool(cfg.resume_optimizer_state):
        logger.info(
            "Skipping Stage 1 optimizer/scheduler resume because resume_optimizer_state=false."
        )
        return
    optim_path = resume_dir / "optim.pt"
    if not optim_path.exists():
        raise FileNotFoundError(f"Missing optimizer state for Stage 1 resume: {optim_path}")
    payload = torch.load(str(optim_path), map_location="cpu")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    logger.info("Loaded Stage 1 optimizer/scheduler state from %s", optim_path)


def _load_stage1_model_weights(wrapper: DistillationModel, resume_dir: Path) -> None:
    """Load Stage 1 trainable model state from a validated checkpoint.

    Args:
        wrapper: Stage 1 wrapper whose DiT LoRA and projector will be updated.
        resume_dir: Checkpoint directory containing safetensors files.
    """
    load_lora_state_dict(
        wrapper.core.dit,
        load_safetensors(str(resume_dir / "stage1_lora.safetensors"), device="cpu"),
        strict=True,
        adapter_name=wrapper.student_lora_adapter_name,
    )
    projector_state = load_safetensors(str(resume_dir / "projector.safetensors"), device="cpu")
    wrapper.projector.load_state_dict(projector_state, strict=True)
