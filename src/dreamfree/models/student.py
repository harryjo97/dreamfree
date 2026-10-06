from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors

from dreamfree.data.distillation import _stage1_visual_preprocess_id
from dreamfree.models.artifacts import _require_checkpoint_dir
from dreamfree.models.lora import load_lora_state_dict, merge_lora_adapters
from dreamfree.models.wan22.future_state import (
    I2V_CONDITION_FIRST_IMAGE_MASK,
    I2V_CONDITION_NONE,
    TEXT_CONDITION_SCOPE_ALL_FRAMES,
    TEXT_CONDITION_SCOPES,
    _lora_enabled,
    temporary_video_attention_mask_mode,
)
from dreamfree.models.wan22.i2v_condition import concat_14b_i2v_condition
from dreamfree.models.wan22.stage0_loader import (
    STAGE1_STUDENT_INIT_STAGE0_MERGED,
    load_stage0_merged_backbone,
)
from dreamfree.training.runtime import _json_load
from dreamfree.training.stage1.constants import (
    STAGE1_CHECKPOINT_TYPE,
    STAGE1_REQUIRED_STUDENT_FILES,
    STAGE1_STUDENT_SOURCE_CHECKPOINT_TYPES,
)


@dataclass(frozen=True)
class Stage1StudentSource:
    checkpoint_dir: Path
    metadata: dict[str, Any]
    global_step: int
    student_layer_idx: int
    student_hook_point: str
    student_video_attention_mask_mode: str
    student_text_scope: str
    student_i2v_condition_mode: str
    expected_student_tokens_per_frame: int
    expected_student_hidden_dim: int
    latent_shape: tuple[int, ...]
    readout_layer_idx: int | None = None


@dataclass
class FrozenStage1Student:
    core: Any
    checkpoint_dir: Path
    metadata: dict[str, Any]
    global_step: int
    source: Stage1StudentSource
    stage0_checkpoint_dir: Path | None
    stage0_trainer_state: dict[str, Any] | None
    stage0_lora_config: dict[str, Any]
    stage1_lora_config: dict[str, Any]
    student_init_mode: str
    stage0_merged_into_student_base: bool
    student_lora_adapter_name: str | None
    student_text_scope: str
    student_i2v_condition_mode: str
    student_layer_idx: int
    student_hook_point: str
    expected_student_tokens_per_frame: int
    expected_student_hidden_dim: int
    latent_shape: tuple[int, ...]
    readout_layer_idx: int | None = None
    student_lora_merged_for_inference: bool = False


def _json_load_dict(path: Path) -> dict[str, Any]:
    payload = _json_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}.")
    return payload


def _require_dict(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict, got {type(value)}.")
    return value


def _stage1_loader_check_equal(
    mismatches: list[str], name: str, actual: Any, expected: Any
) -> None:
    if actual != expected:
        mismatches.append(f"{name}: actual={actual!r} expected={expected!r}")


def _metadata_is_null_like(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def _first_non_null_metadata_value(*values: Any) -> Any:
    for value in values:
        if not _metadata_is_null_like(value):
            return value
    return None


def _stage1_student_view_metadata(metadata):
    if metadata.get("checkpoint_type") != STAGE1_CHECKPOINT_TYPE:
        raise ValueError("Expected a DreamFree stage 1 checkpoint.")
    return metadata


def _stage1_required_student_files_for_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    return STAGE1_REQUIRED_STUDENT_FILES


def read_stage1_student_source_metadata(checkpoint: str | Path | None) -> dict[str, Any]:
    checkpoint_dir = _require_checkpoint_dir(checkpoint, label="Stage 1 source")
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        raise FileNotFoundError(
            f"Stage 1 source checkpoint is missing trainer_state.json: {checkpoint_dir}"
        )
    raw_metadata = _json_load_dict(trainer_state_path)
    missing = [
        name
        for name in _stage1_required_student_files_for_metadata(raw_metadata)
        if not (checkpoint_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Stage 1 source checkpoint is missing files {missing}: {checkpoint_dir}"
        )
    return _stage1_student_view_metadata(raw_metadata)


def stage1_student_source_model_contract(checkpoint: str | Path | None) -> dict[str, Any]:
    metadata = read_stage1_student_source_metadata(checkpoint)
    config = _require_dict(metadata.get("config"), label="Stage 1 metadata.config")
    shapes = _require_dict(metadata.get("shapes"), label="Stage 1 metadata.shapes")
    contract = metadata.get("model_contract")
    if not isinstance(contract, dict):
        contract = {}
    return {
        "checkpoint_type": metadata.get("checkpoint_type"),
        "model_id": _first_non_null_metadata_value(
            config.get("model_id"), contract.get("model_id")
        ),
        "video_model_family": _first_non_null_metadata_value(
            config.get("video_model_family"), contract.get("video_model_family")
        ),
        "tokenizer_model_id": config.get("tokenizer_model_id"),
        "tokenizer_max_len": config.get("tokenizer_max_len"),
        "text_dim": config.get("text_dim"),
        "video_hidden_dim": _first_non_null_metadata_value(
            shapes.get("D_student"),
            config.get("video_hidden_dim"),
            contract.get("video_hidden_dim"),
        ),
        "video_latent_dim": _first_non_null_metadata_value(
            config.get("video_latent_dim"), contract.get("video_latent_dim")
        ),
        "video_dit_input_dim": _first_non_null_metadata_value(
            config.get("video_dit_input_dim"), contract.get("video_dit_input_dim")
        ),
        "video_dit_output_dim": _first_non_null_metadata_value(
            config.get("video_dit_output_dim"), contract.get("video_dit_output_dim")
        ),
    }


def validate_stage1_student_source(cfg: Any) -> Stage1StudentSource:
    checkpoint_dir = _require_checkpoint_dir(
        getattr(cfg, "stage1_checkpoint", None), label="Stage 1 source"
    )
    metadata = read_stage1_student_source_metadata(checkpoint_dir)
    mismatches: list[str] = []
    checkpoint_type = str(metadata.get("checkpoint_type", "")).strip()
    if checkpoint_type not in STAGE1_STUDENT_SOURCE_CHECKPOINT_TYPES:
        mismatches.append(
            f"checkpoint_type: actual={checkpoint_type!r} expected one of {sorted(STAGE1_STUDENT_SOURCE_CHECKPOINT_TYPES)}"
        )
    student = _require_dict(metadata.get("student"), label="Stage 1 metadata.student")
    shapes = _require_dict(metadata.get("shapes"), label="Stage 1 metadata.shapes")
    config = _require_dict(metadata.get("config"), label="Stage 1 metadata.config")
    _require_dict(metadata.get("teacher_state"), label="Stage 1 metadata.teacher_state")
    _require_dict(metadata.get("stage1_lora_config"), label="Stage 1 metadata.stage1_lora_config")
    for key in ("tokens_per_frame", "D_student", "latent_shape"):
        if key not in shapes:
            mismatches.append(f"shapes.{key} is missing")
    layer_idx = int(student.get("layer_j", -1))
    hook_point = str(student.get("hook_point", ""))
    mask_mode = str(student.get("video_attention_mask_mode", ""))
    text_scope = (
        str(
            student.get(
                "text_scope", config.get("student_text_scope", TEXT_CONDITION_SCOPE_ALL_FRAMES)
            )
        )
        .strip()
        .lower()
    )
    i2v_condition_mode = str(student.get("i2v_condition_mode", I2V_CONDITION_NONE))
    if layer_idx < 0:
        mismatches.append(f"student.layer_j must be non-negative, got {layer_idx}.")
    if text_scope not in TEXT_CONDITION_SCOPES:
        mismatches.append(
            f"student.text_scope must be one of {sorted(TEXT_CONDITION_SCOPES)}, got {text_scope!r}."
        )
    requested_layer = getattr(cfg, "student_readout_layer", None)
    readout_layer_idx = layer_idx
    if requested_layer is not None:
        readout_layer_idx = int(requested_layer)
        if readout_layer_idx < 0:
            mismatches.append(
                f"student_readout_layer must be non-negative, got {readout_layer_idx}."
            )
    requested_hook = getattr(cfg, "student_hook_point", None)
    if requested_hook is not None:
        _stage1_loader_check_equal(
            mismatches, "student.hook_point", hook_point, str(requested_hook)
        )
    else:
        requested_hook = hook_point
    _stage1_loader_check_equal(mismatches, "student.hook_point", hook_point, "post_block")
    _stage1_loader_check_equal(
        mismatches, "student.video_attention_mask_mode", mask_mode, "first_frame_causal"
    )
    _stage1_loader_check_equal(
        mismatches,
        "cfg.student_video_attention_mask_mode",
        str(getattr(cfg, "student_video_attention_mask_mode", "first_frame_causal")),
        "first_frame_causal",
    )
    _stage1_loader_check_equal(
        mismatches,
        "cfg.student_train_mode",
        str(getattr(cfg, "student_train_mode", "frozen")),
        "frozen",
    )
    video_family = str(config.get("video_model_family", "wan22_ti2v_5b"))
    _stage1_loader_check_equal(
        mismatches,
        "config.video_model_family",
        video_family,
        str(getattr(cfg, "video_model_family")),
    )
    expected_visual_id = _stage1_visual_preprocess_id(cfg)
    if expected_visual_id is not None:
        _stage1_loader_check_equal(
            mismatches,
            "visual_preprocess_id",
            metadata.get("visual_preprocess_id"),
            expected_visual_id,
        )
        _stage1_loader_check_equal(
            mismatches,
            "source_stage0_visual_preprocess_id",
            metadata.get("source_stage0_visual_preprocess_id"),
            expected_visual_id,
        )
    if "D_student" in shapes:
        _stage1_loader_check_equal(
            mismatches,
            "shapes.D_student",
            int(shapes["D_student"]),
            int(getattr(cfg, "video_hidden_dim")),
        )
    if video_family == "wan22_ti2v_5b":
        _stage1_loader_check_equal(
            mismatches, "student.i2v_condition_mode", i2v_condition_mode, I2V_CONDITION_NONE
        )
    elif video_family == "wan22_i2v_a14b":
        _stage1_loader_check_equal(
            mismatches,
            "student.i2v_condition_mode",
            i2v_condition_mode,
            I2V_CONDITION_FIRST_IMAGE_MASK,
        )
    else:
        mismatches.append(
            f"config.video_model_family: actual={video_family!r} expected='wan22_ti2v_5b' or 'wan22_i2v_a14b'"
        )
    global_step = int(metadata.get("global_step", -1))
    if global_step < 0:
        mismatches.append(f"global_step must be non-negative, got {global_step}.")
    if "tokens_per_frame" in shapes and int(shapes["tokens_per_frame"]) <= 0:
        mismatches.append(
            f"shapes.tokens_per_frame must be positive, got {shapes['tokens_per_frame']!r}."
        )
    if "D_student" in shapes and int(shapes["D_student"]) <= 0:
        mismatches.append(f"shapes.D_student must be positive, got {shapes['D_student']!r}.")
    latent_shape = tuple((int(v) for v in shapes.get("latent_shape", ())))
    if "latent_shape" in shapes and len(latent_shape) != 4:
        mismatches.append(f"shapes.latent_shape must be [C,T,H,W], got {shapes['latent_shape']!r}.")
    if mismatches:
        raise ValueError("Stage 1 student source checkpoint mismatch: " + "; ".join(mismatches))
    return Stage1StudentSource(
        checkpoint_dir=checkpoint_dir,
        metadata=metadata,
        global_step=global_step,
        student_layer_idx=layer_idx,
        student_hook_point=str(requested_hook),
        student_video_attention_mask_mode=mask_mode,
        student_text_scope=text_scope,
        student_i2v_condition_mode=i2v_condition_mode,
        expected_student_tokens_per_frame=int(shapes["tokens_per_frame"]),
        expected_student_hidden_dim=int(shapes["D_student"]),
        latent_shape=latent_shape,
        readout_layer_idx=readout_layer_idx,
    )


def _build_stage1_loader_config(cfg: Any, source: Stage1StudentSource) -> SimpleNamespace:
    metadata = source.metadata
    stage1_config = _require_dict(metadata.get("config"), label="Stage 1 metadata.config")
    stage1_lora_config = _require_dict(
        metadata.get("stage1_lora_config"), label="Stage 1 metadata.stage1_lora_config"
    )
    stage0_lora_config = _require_dict(
        metadata.get("stage0_lora_config"), label="Stage 1 metadata.stage0_lora_config"
    )
    stage0_enabled = bool(stage0_lora_config.get("enabled", True))
    stage1_enabled = bool(stage1_lora_config.get("enabled", True))
    return SimpleNamespace(
        model_base_path=str(getattr(cfg, "model_base_path")),
        stage0_lora_checkpoint=metadata.get("stage0_lora_checkpoint"),
        stage0_eval_reference=metadata.get("stage0_eval_reference"),
        stage0_expected_global_step=int(metadata.get("stage0_expected_global_step") or 0),
        stage0_expected_lora_rank=int(
            stage0_lora_config.get("rank", 0 if not stage0_enabled else -1)
        ),
        stage0_expected_lora_alpha=float(
            stage0_lora_config.get("alpha", 0.0 if not stage0_enabled else -1.0)
        ),
        stage0_expected_lora_dropout=float(stage0_lora_config.get("dropout", 0.0)),
        stage0_expected_lora_target_modules=str(stage0_lora_config.get("target_modules", "")),
        stage0_expected_model_id=str(
            stage1_config.get("stage0_expected_model_id", stage1_config.get("model_id"))
        ),
        stage0_expected_video_model_family=str(
            stage1_config.get(
                "stage0_expected_video_model_family",
                stage1_config.get("video_model_family", "wan22_ti2v_5b"),
            )
        ),
        stage0_expected_dit_role=str(
            stage1_config.get(
                "stage0_expected_dit_role", stage1_config.get("teacher_dit_role", "single")
            )
        ),
        stage0_expected_tokenizer_model_id=str(
            stage1_config.get(
                "stage0_expected_tokenizer_model_id", stage1_config["tokenizer_model_id"]
            )
        ),
        stage0_expected_tokenizer_max_len=int(
            stage1_config.get(
                "stage0_expected_tokenizer_max_len", stage1_config["tokenizer_max_len"]
            )
        ),
        video_model_family=str(stage1_config.get("video_model_family", "wan22_ti2v_5b")),
        allow_base_without_stage0_debug=bool(
            stage1_config.get("allow_base_without_stage0_debug", False)
        ),
        teacher_dit_role=str(stage1_config.get("teacher_dit_role", "single")),
        student_dit_role=str(stage1_config.get("student_dit_role", "single")),
        student_init_mode=str(
            metadata.get(
                "student_init_mode",
                stage1_config.get("student_init_mode", STAGE1_STUDENT_INIT_STAGE0_MERGED),
            )
        ),
        model_id=str(stage1_config["model_id"]),
        tokenizer_model_id=str(stage1_config["tokenizer_model_id"]),
        tokenizer_max_len=int(stage1_config["tokenizer_max_len"]),
        redirect_common_files=bool(stage1_config.get("redirect_common_files", True)),
        pretrained_load_broadcast=bool(getattr(cfg, "pretrained_load_broadcast", True)),
        load_vae=not bool(getattr(cfg, "use_cached_latents", False)),
        train_shift=float(stage1_config.get("train_shift", 5.0)),
        infer_shift=float(stage1_config.get("infer_shift", 5.0)),
        num_train_timesteps=int(stage1_config.get("num_train_timesteps", 1000)),
        use_gradient_checkpointing=bool(getattr(cfg, "use_gradient_checkpointing", False)),
        stage1_lora_rank=int(
            stage1_lora_config["rank"] if stage1_enabled else stage1_lora_config.get("rank", 0)
        ),
        stage1_lora_alpha=float(
            stage1_lora_config["alpha"] if stage1_enabled else stage1_lora_config.get("alpha", 0.0)
        ),
        stage1_lora_dropout=float(
            stage1_lora_config["dropout"]
            if stage1_enabled
            else stage1_lora_config.get("dropout", 0.0)
        ),
        stage1_lora_target_modules=str(
            stage1_lora_config["target_modules"]
            if stage1_enabled
            else stage1_lora_config.get("target_modules", "")
        ),
    )


def _assert_no_trainable_core_parameters(core: Any) -> None:
    offenders = [name for (name, param) in core.named_parameters() if param.requires_grad]
    if offenders:
        preview = offenders[:10]
        suffix = "..." if len(offenders) > 10 else ""
        raise ValueError(f"Frozen Stage 1 student has trainable core parameters: {preview}{suffix}")


def load_frozen_stage1_student(
    cfg: Any,
    *,
    source: Stage1StudentSource,
    device: str,
    dtype: torch.dtype,
    merge_student_lora_for_inference: bool = False,
) -> FrozenStage1Student:
    backbone = load_stage0_merged_backbone(
        _build_stage1_loader_config(cfg, source), device=device, dtype=dtype
    )
    student_lora_adapter_name = getattr(backbone, "student_lora_adapter_name", None)
    load_lora_state_dict(
        backbone.core.dit,
        load_safetensors(str(source.checkpoint_dir / "stage1_lora.safetensors"), device="cpu"),
        strict=True,
        adapter_name=student_lora_adapter_name,
    )
    if merge_student_lora_for_inference:
        merged = merge_lora_adapters(
            backbone.core.dit,
            adapter_name=student_lora_adapter_name,
            allow_single_adapter_from_multi=True,
        )
        if not merged:
            raise RuntimeError("Frozen student LoRA merge matched zero modules.")
    backbone.core.requires_grad_(False)
    backbone.core.eval()
    _assert_no_trainable_core_parameters(backbone.core)
    return FrozenStage1Student(
        core=backbone.core,
        checkpoint_dir=source.checkpoint_dir,
        metadata=source.metadata,
        global_step=source.global_step,
        source=source,
        stage0_checkpoint_dir=getattr(backbone, "checkpoint_dir", None),
        stage0_trainer_state=getattr(backbone, "trainer_state", None),
        stage0_lora_config=dict(getattr(backbone, "stage0_lora_config", {}) or {}),
        stage1_lora_config=dict(getattr(backbone, "stage1_lora_config", {}) or {}),
        student_init_mode=str(
            getattr(backbone, "student_init_mode", STAGE1_STUDENT_INIT_STAGE0_MERGED)
        ),
        stage0_merged_into_student_base=bool(
            getattr(backbone, "stage0_merged_into_student_base", True)
        ),
        student_lora_adapter_name=student_lora_adapter_name,
        student_text_scope=source.student_text_scope,
        student_i2v_condition_mode=source.student_i2v_condition_mode,
        student_layer_idx=source.student_layer_idx,
        student_hook_point=source.student_hook_point,
        expected_student_tokens_per_frame=source.expected_student_tokens_per_frame,
        expected_student_hidden_dim=source.expected_student_hidden_dim,
        latent_shape=source.latent_shape,
        readout_layer_idx=source.readout_layer_idx,
        student_lora_merged_for_inference=bool(merge_student_lora_for_inference),
    )


@torch.no_grad()
def extract_current_only_student_tokens(
    student: FrozenStage1Student,
    *,
    first_frame_latents: torch.Tensor,
    i2v_condition_latents: torch.Tensor | None = None,
    context: torch.Tensor,
    context_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(first_frame_latents, torch.Tensor) or first_frame_latents.ndim != 5:
        raise ValueError(
            f"`first_frame_latents` must be 5D [B,C,1,H,W], got {tuple(getattr(first_frame_latents, 'shape', ()))}."
        )
    if int(first_frame_latents.shape[2]) != 1:
        raise ValueError(
            f"Current-only student extraction requires exactly one latent frame, got {first_frame_latents.shape[2]}."
        )
    timestep = torch.zeros(
        (first_frame_latents.shape[0],),
        device=first_frame_latents.device,
        dtype=first_frame_latents.dtype,
    )
    dit_input = first_frame_latents
    if student.student_i2v_condition_mode == I2V_CONDITION_FIRST_IMAGE_MASK:
        if not isinstance(i2v_condition_latents, torch.Tensor) or i2v_condition_latents.ndim != 5:
            raise ValueError(
                "Current-only 14B student extraction requires `i2v_condition_latents` from official pseudo-video VAE encoding."
            )
        condition_first_frame = i2v_condition_latents[:, :, 0:1]
        if tuple(condition_first_frame.shape) != tuple(first_frame_latents.shape):
            raise ValueError(
                f"Current-only 14B I2V condition first-frame shape must match `first_frame_latents`: condition={tuple(condition_first_frame.shape)} latents={tuple(first_frame_latents.shape)}."
            )
        dit_input = concat_14b_i2v_condition(
            latents=first_frame_latents, i2v_condition_latents=condition_first_frame
        )
    lora_context = (
        nullcontext()
        if bool(getattr(student, "student_lora_merged_for_inference", False))
        or getattr(student, "student_lora_adapter_name", None) is None
        else _lora_enabled(student.core.dit, student.student_lora_adapter_name)
    )
    with lora_context, temporary_video_attention_mask_mode(student.core.dit, "first_frame_causal"):
        active_mode = str(getattr(student.core.dit, "video_attention_mask_mode", ""))
        if active_mode != "first_frame_causal":
            raise RuntimeError(
                f"Current-only student path requires first_frame_causal mask mode, got {active_mode!r}."
            )
        readout_layer_idx = (
            int(student.readout_layer_idx)
            if getattr(student, "readout_layer_idx", None) is not None
            else int(student.student_layer_idx)
        )
        out = student.core.dit.forward_hidden_tokens(
            x=dit_input,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(student.core.dit, "fuse_vae_embedding_in_latents", False)
            ),
            text_condition_scope=student.student_text_scope,
            layer_idx=readout_layer_idx,
            hook_point=student.student_hook_point,
        )
    if not isinstance(out, dict) or "tokens" not in out or "meta" not in out:
        raise ValueError(
            "Stage 1 student forward_hidden_tokens must return a dict with `tokens` and `meta`."
        )
    tokens = out["tokens"].detach()
    meta = out["meta"]
    if int(meta["tokens_per_frame"]) != student.expected_student_tokens_per_frame:
        raise ValueError(
            f"Current-only student tokens_per_frame mismatch: actual={meta['tokens_per_frame']} expected={student.expected_student_tokens_per_frame}."
        )
    if int(meta["sequence_length"]) != student.expected_student_tokens_per_frame:
        raise ValueError(
            f"Current-only student path must return exactly one frame of tokens, got sequence_length={meta['sequence_length']} tokens_per_frame={meta['tokens_per_frame']}."
        )
    expected_shape = (
        int(first_frame_latents.shape[0]),
        int(student.expected_student_tokens_per_frame),
        int(student.expected_student_hidden_dim),
    )
    if tuple(tokens.shape) != expected_shape:
        raise ValueError(
            f"Current-only student token shape mismatch: actual={tuple(tokens.shape)} expected={expected_shape}."
        )
    return (tokens, meta)
