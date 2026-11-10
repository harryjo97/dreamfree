"""Stage 0 LoRA loader for Stage 1 shared teacher/student Wan backbones."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

from dreamfree.models.lora import (
    has_lora_adapters,
    inject_lora_adapters,
    is_lora_parameter_name,
    load_lora_state_dict,
    merge_lora_adapters,
    normalize_lora_config,
    set_lora_requires_grad,
)
from dreamfree.models.wan22.wan22 import Wan22Core

STAGE1_STUDENT_INIT_STAGE0_MERGED = "shared_stage0_merged_dit_with_stage1_lora"
STAGE1_STUDENT_INIT_PRETRAINED_BASE = (
    "shared_pretrained_dit_with_stage0_teacher_lora_and_stage1_student_lora"
)
STAGE1_STUDENT_INIT_MODES = frozenset(
    (STAGE1_STUDENT_INIT_STAGE0_MERGED, STAGE1_STUDENT_INIT_PRETRAINED_BASE)
)
STAGE1_TEACHER_LORA_ADAPTER_NAME = "stage0"
STAGE1_STUDENT_LORA_ADAPTER_NAME = "stage1"


@dataclass
class Stage0MergedBackbone:
    core: Wan22Core
    checkpoint_dir: Path
    trainer_state: dict[str, Any]
    stage0_lora_config: dict[str, Any]
    stage1_lora_config: dict[str, Any]
    merged_module_names: list[str]
    stage0_module_names: list[str]
    stage1_module_names: list[str]
    student_init_mode: str
    stage0_merged_into_student_base: bool
    teacher_lora_adapter_name: str | None
    student_lora_adapter_name: str | None


@dataclass
class Stage0MergedWanCore:
    core: Wan22Core
    checkpoint_dir: Path
    trainer_state: dict[str, Any]
    stage0_lora_config: dict[str, Any]
    merged_module_names: list[str]


@dataclass(frozen=True)
class Stage0TrainSigmaConfig:
    train_sigma_policy: str
    sigma_min: float
    sigma_max: float


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}.")
    from dreamfree.config_names import normalize_checkpoint_metadata

    return normalize_checkpoint_metadata(payload)


def _video_model_family(cfg: Any) -> str:
    return str(getattr(cfg, "video_model_family", "wan22_ti2v_5b"))


def _stage0_lora_checkpoint_enabled(cfg: Any) -> bool:
    value = getattr(cfg, "stage0_lora_checkpoint", None)
    if value is None:
        return False
    return bool(str(value).strip()) and str(value).strip().lower() not in {"none", "null"}


def _allow_base_without_stage0_debug(cfg: Any) -> bool:
    return False


def default_stage1_dit_config(
    *, use_gradient_checkpointing: bool = False, video_model_family: str = "wan22_ti2v_5b"
) -> dict[str, Any]:
    """Return the Wan DiT config used by Stage 0/1-style loaders."""
    family = str(video_model_family)
    if family == "wan22_i2v_a14b":
        return {
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 36,
            "hidden_dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 40,
            "attn_head_dim": 128,
            "num_layers": 40,
            "eps": 1e-06,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "use_gradient_checkpointing": bool(use_gradient_checkpointing),
            "video_attention_mask_mode": "first_frame_causal",
            "action_conditioned": False,
            "action_dim": 1,
            "action_group_causal_mask_mode": "group_diagonal",
        }
    if family != "wan22_ti2v_5b":
        raise ValueError(
            f"Unsupported video_model_family={video_model_family!r}; expected 'wan22_ti2v_5b' or 'wan22_i2v_a14b'."
        )
    return {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1e-06,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": bool(use_gradient_checkpointing),
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": 1,
        "action_group_causal_mask_mode": "group_diagonal",
    }


def resolve_stage0_lora_checkpoint(path: str | Path) -> Path:
    """Resolve a Stage 0 eval reference path to the loadable checkpoint path."""
    checkpoint_dir = Path(path).expanduser().resolve()
    if checkpoint_dir.parent.name == "eval" and checkpoint_dir.name.startswith("step_"):
        checkpoint_dir = checkpoint_dir.parent.parent / "checkpoints" / checkpoint_dir.name
    return checkpoint_dir


def _expected_step_from_dir(path: Path) -> int | None:
    match = re.fullmatch("step_(\\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _record_mismatch(mismatches: list[str], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        mismatches.append(f"{name}: actual={actual!r} expected={expected!r}")


def _cfg_float_with_default(cfg: Any, name: str, default: float) -> float:
    value = getattr(cfg, name, default)
    return float(value)


def _normalize_stage0_train_sigma_policy_from_metadata(
    value: Any, *, metadata_path: Path
) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "uniform":
        return "uniform_range"
    if normalized in {"uniform_range", "wan_shift_full", "wan_shift_range"}:
        return normalized
    raise ValueError(
        f"Unsupported Stage 0 train sigma policy in {metadata_path}: {value!r}. Expected 'uniform_range', 'wan_shift_full', or 'wan_shift_range'."
    )


def stage0_train_sigma_config_from_metadata(
    metadata: dict[str, Any], *, metadata_path: Path
) -> Stage0TrainSigmaConfig:
    config = metadata.get("config", {})
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError(f"Stage 0 checkpoint metadata `config` must be a dict in {metadata_path}.")
    raw_policy = metadata.get("train_sigma_policy")
    if raw_policy is None:
        raw_policy = config.get("train_sigma_policy")
    if raw_policy is None:
        raw_policy = metadata.get("sigma_policy")
    policy = _normalize_stage0_train_sigma_policy_from_metadata(
        raw_policy, metadata_path=metadata_path
    )
    if policy is None:
        raise ValueError(f"Stage 0 checkpoint is missing train_sigma_policy: {metadata_path}")
    if policy == "wan_shift_full":
        return Stage0TrainSigmaConfig(train_sigma_policy=policy, sigma_min=0.0, sigma_max=1.0)
    raw_min = config.get(
        "train_sigma_min", metadata.get("train_sigma_min", metadata.get("sigma_min"))
    )
    raw_max = config.get(
        "train_sigma_max", metadata.get("train_sigma_max", metadata.get("sigma_max"))
    )
    if raw_min is None or raw_max is None:
        raise ValueError(
            f"Stage 0 checkpoint metadata is missing train sigma bounds. Expected config.train_sigma_min/config.train_sigma_max or top-level sigma_min/sigma_max in {metadata_path}."
        )
    sigma_min = float(raw_min)
    sigma_max = float(raw_max)
    if not math.isfinite(sigma_min) or not math.isfinite(sigma_max):
        raise ValueError(
            f"Stage 0 checkpoint train sigma bounds must be finite in {metadata_path}, got train_sigma_min={sigma_min}, train_sigma_max={sigma_max}."
        )
    if not 0.0 <= sigma_min <= sigma_max <= 1.0:
        raise ValueError(
            f"Stage 0 checkpoint train sigma bounds must satisfy 0 <= min <= max <= 1 in {metadata_path}, got train_sigma_min={sigma_min}, train_sigma_max={sigma_max}."
        )
    if policy == "wan_shift_range" and (not 0.0 <= sigma_min < sigma_max < 1.0):
        raise ValueError(
            f"Stage 0 checkpoint train_sigma_policy='wan_shift_range' requires 0 <= train_sigma_min < train_sigma_max < 1 in {metadata_path}, got train_sigma_min={sigma_min}, train_sigma_max={sigma_max}."
        )
    return Stage0TrainSigmaConfig(
        train_sigma_policy=policy, sigma_min=sigma_min, sigma_max=sigma_max
    )


def stage0_train_sigma_config_from_checkpoint(
    path: str | Path, *, stage_label: str
) -> Stage0TrainSigmaConfig:
    """Read Stage 0 train sigma policy and effective bounds from a checkpoint."""
    checkpoint_dir = resolve_stage0_lora_checkpoint(path)
    metadata_path = checkpoint_dir / "trainer_state.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing Stage 0 trainer state for {stage_label} sigma validation: {metadata_path}"
        )
    metadata = _json_load(metadata_path)
    return stage0_train_sigma_config_from_metadata(metadata, metadata_path=metadata_path)


def stage0_effective_train_sigma_bounds_from_checkpoint(
    path: str | Path, *, stage_label: str
) -> tuple[float, float]:
    """Read the effective Stage 0 train sigma range from a checkpoint."""
    sigma_config = stage0_train_sigma_config_from_checkpoint(path, stage_label=stage_label)
    return (sigma_config.sigma_min, sigma_config.sigma_max)


def validate_teacher_sigma_bounds_against_stage0_checkpoint(cfg: Any, *, stage_label: str) -> None:
    """Fail fast if a downstream teacher samples noise outside Stage 0 coverage."""
    if not _stage0_lora_checkpoint_enabled(cfg):
        return
    checkpoint_dir = resolve_stage0_lora_checkpoint(getattr(cfg, "stage0_lora_checkpoint"))
    (train_sigma_min, train_sigma_max) = stage0_effective_train_sigma_bounds_from_checkpoint(
        checkpoint_dir, stage_label=f"{stage_label} teacher"
    )
    teacher_sigma_min = float(getattr(cfg, "teacher_sigma_min"))
    teacher_sigma_max = float(getattr(cfg, "teacher_sigma_max"))
    tolerance = 1e-12
    if (
        teacher_sigma_min < train_sigma_min - tolerance
        or teacher_sigma_max > train_sigma_max + tolerance
    ):
        raise ValueError(
            f"{stage_label} teacher sigma range is outside Stage 0 train sigma range: teacher=[{teacher_sigma_min}, {teacher_sigma_max}], stage0_train=[{train_sigma_min}, {train_sigma_max}], stage0_checkpoint={checkpoint_dir}."
        )


def _validate_stage0_trainer_state(
    cfg: Any, checkpoint_dir: Path, state: dict[str, Any]
) -> dict[str, Any]:
    if state.get("checkpoint_type") != "dreamfree_stage0" or state.get("checkpoint_version") != 1:
        raise ValueError(f"Expected a DreamFree stage 0 checkpoint in {checkpoint_dir}.")
    if "global_step" not in state:
        raise ValueError(f"Stage 0 trainer_state.json is missing `global_step`: {checkpoint_dir}")
    global_step = int(state["global_step"])
    step_from_dir = _expected_step_from_dir(checkpoint_dir)
    mismatches: list[str] = []
    if step_from_dir is not None:
        _record_mismatch(mismatches, "checkpoint_dir.step", step_from_dir, global_step)
    expected_global_step = int(getattr(cfg, "stage0_expected_global_step"))
    _record_mismatch(mismatches, "global_step", global_step, expected_global_step)
    raw_lora_config = state.get("lora_config")
    if not isinstance(raw_lora_config, dict):
        raise ValueError(
            f"Stage 0 trainer_state.json is missing dict `lora_config`: {checkpoint_dir}"
        )
    lora_config = normalize_lora_config(raw_lora_config)
    _record_mismatch(
        mismatches,
        "lora_config.rank",
        int(lora_config["rank"]),
        int(getattr(cfg, "stage0_expected_lora_rank")),
    )
    _record_mismatch(
        mismatches,
        "lora_config.alpha",
        float(lora_config["alpha"]),
        float(getattr(cfg, "stage0_expected_lora_alpha")),
    )
    _record_mismatch(
        mismatches,
        "lora_config.dropout",
        float(lora_config["dropout"]),
        float(getattr(cfg, "stage0_expected_lora_dropout")),
    )
    _record_mismatch(
        mismatches,
        "lora_config.target_modules",
        str(lora_config["target_modules"]),
        str(getattr(cfg, "stage0_expected_lora_target_modules")),
    )
    if float(lora_config["dropout"]) != 0.0:
        mismatches.append(
            f"lora_config.dropout must be 0.0 for Stage 1 merge, got {lora_config['dropout']!r}"
        )
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Stage 0 trainer_state.json is missing dict `config`: {checkpoint_dir}")
    _record_mismatch(
        mismatches,
        "config.model_id",
        config.get("model_id"),
        getattr(cfg, "stage0_expected_model_id"),
    )
    _record_mismatch(
        mismatches,
        "config.tokenizer_model_id",
        config.get("tokenizer_model_id"),
        getattr(cfg, "stage0_expected_tokenizer_model_id"),
    )
    _record_mismatch(
        mismatches,
        "config.tokenizer_max_len",
        int(config.get("tokenizer_max_len")),
        int(getattr(cfg, "stage0_expected_tokenizer_max_len")),
    )
    expected_family = str(
        getattr(
            cfg,
            "stage0_expected_video_model_family",
            getattr(cfg, "video_model_family", "wan22_ti2v_5b"),
        )
    )
    if expected_family == "wan22_i2v_a14b":
        _record_mismatch(
            mismatches,
            "config.video_model_family",
            config.get("video_model_family"),
            expected_family,
        )
        _record_mismatch(
            mismatches,
            "config.dit_role",
            config.get("dit_role"),
            getattr(cfg, "stage0_expected_dit_role", "low_noise"),
        )
        _record_mismatch(
            mismatches,
            "config.i2v_condition_mode",
            config.get("i2v_condition_mode"),
            "first_image_mask_condition",
        )
        vae_family = state.get("vae_family")
        if vae_family is not None:
            _record_mismatch(mismatches, "vae_family", vae_family, "wan2.1")
    if mismatches:
        raise ValueError("Stage 0 checkpoint metadata mismatch: " + "; ".join(mismatches))
    return lora_config


def validate_stage0_eval_reference(cfg: Any) -> None:
    """Validate the optional qualitative eval reference identity."""
    raw_reference = getattr(cfg, "stage0_eval_reference", None)
    if not raw_reference:
        return
    reference = Path(raw_reference).expanduser().resolve()
    metadata_path = reference / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Stage 0 eval reference metadata not found: {metadata_path}")
    metadata = _json_load(metadata_path)
    actual = int(metadata.get("global_step"))
    expected = int(getattr(cfg, "stage0_expected_global_step"))
    if actual != expected:
        raise ValueError(
            f"Stage 0 eval reference global_step mismatch in {metadata_path}: actual={actual} expected={expected}"
        )


def _assert_non_lora_frozen(module: nn.Module) -> None:
    offenders = [
        name
        for (name, param) in module.named_parameters()
        if param.requires_grad and (not is_lora_parameter_name(name))
    ]
    if offenders:
        raise ValueError(f"Non-LoRA parameters are trainable after Stage 1 setup: {offenders[:10]}")


def _stage1_student_init_mode(cfg: Any) -> str:
    mode = str(getattr(cfg, "student_init_mode", STAGE1_STUDENT_INIT_STAGE0_MERGED)).strip().lower()
    if mode not in STAGE1_STUDENT_INIT_MODES:
        raise ValueError(
            f"Unsupported student_init_mode={getattr(cfg, 'student_init_mode', None)!r}; expected one of {sorted(STAGE1_STUDENT_INIT_MODES)}."
        )
    return mode


def _validated_stage0_lora_payload(cfg: Any) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    checkpoint_dir = resolve_stage0_lora_checkpoint(getattr(cfg, "stage0_lora_checkpoint"))
    lora_path = checkpoint_dir / "lora.safetensors"
    state_path = checkpoint_dir / "trainer_state.json"
    if not lora_path.exists():
        raise FileNotFoundError(f"Missing Stage 0 LoRA checkpoint: {lora_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing Stage 0 trainer state: {state_path}")
    trainer_state = _json_load(state_path)
    stage0_lora_config = _validate_stage0_trainer_state(cfg, checkpoint_dir, trainer_state)
    validate_stage0_eval_reference(cfg)
    identity_mismatches: list[str] = []
    _record_mismatch(
        identity_mismatches,
        "cfg.model_id",
        str(getattr(cfg, "model_id")),
        str(getattr(cfg, "stage0_expected_model_id")),
    )
    _record_mismatch(
        identity_mismatches,
        "cfg.tokenizer_model_id",
        str(getattr(cfg, "tokenizer_model_id")),
        str(getattr(cfg, "stage0_expected_tokenizer_model_id")),
    )
    _record_mismatch(
        identity_mismatches,
        "cfg.tokenizer_max_len",
        int(getattr(cfg, "tokenizer_max_len")),
        int(getattr(cfg, "stage0_expected_tokenizer_max_len")),
    )
    if identity_mismatches:
        raise ValueError(
            "Stage 1 config identity mismatch against Stage 0 expectations: "
            + "; ".join(identity_mismatches)
        )
    return (checkpoint_dir, lora_path, trainer_state, stage0_lora_config)


def _state_or_config_value(state: dict[str, Any], key: str) -> Any:
    if key in state:
        return state[key]
    config = state.get("config")
    if isinstance(config, dict):
        return config.get(key)
    return None


def _state_or_config_bool(state: dict[str, Any], key: str) -> bool:
    value = _state_or_config_value(state, key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_stage0_lora_file_or_dir(path: str | Path, *, label: str) -> tuple[Path, Path, Path]:
    checkpoint_path = Path(path).expanduser().resolve()
    if (
        checkpoint_path.is_dir()
        and checkpoint_path.parent.name == "eval"
        and checkpoint_path.name.startswith("step_")
    ):
        checkpoint_path = checkpoint_path.parent.parent / "checkpoints" / checkpoint_path.name
    if checkpoint_path.is_file():
        checkpoint_dir = checkpoint_path.parent
        lora_path = checkpoint_path
        state_path = checkpoint_path.with_name("trainer_state.json")
    else:
        checkpoint_dir = checkpoint_path
        lora_path = checkpoint_dir / "lora.safetensors"
        state_path = checkpoint_dir / "trainer_state.json"
    if not lora_path.exists():
        raise FileNotFoundError(f"Missing {label} LoRA checkpoint: {lora_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing {label} trainer state: {state_path}")
    return (checkpoint_dir, lora_path, state_path)


def _init_lora_checkpoint_value(state: dict[str, Any], *, checkpoint_dir: Path) -> str:
    for key in ("init_lora_checkpoint", "init_lora_resolved_checkpoint"):
        value = _state_or_config_value(state, key)
        if (
            value is not None
            and str(value).strip()
            and (str(value).strip().lower() not in {"none", "null"})
        ):
            return str(value)
    raise ValueError(
        f"Stage 0 checkpoint declares init_lora_merged_into_base=true but does not record init_lora_checkpoint or init_lora_resolved_checkpoint: {checkpoint_dir}"
    )


def _validate_recorded_init_lora_config(
    parent_state: dict[str, Any], source_lora_config: dict[str, Any], *, source_state_path: Path
) -> None:
    recorded = _state_or_config_value(parent_state, "init_lora_config")
    if recorded is None:
        return
    if not isinstance(recorded, dict):
        raise ValueError(
            f"Recorded init_lora_config must be a dict in checkpoint metadata for {source_state_path}."
        )
    expected = normalize_lora_config(recorded)
    actual = normalize_lora_config(source_lora_config)
    mismatches = [
        f"{key}: recorded={expected[key]!r} source={actual[key]!r}"
        for key in ("rank", "alpha", "dropout", "target_modules")
        if expected[key] != actual[key]
    ]
    if mismatches:
        raise ValueError(
            f"Recorded init_lora_config does not match source checkpoint {source_state_path}: "
            + "; ".join(mismatches)
        )


def _merge_lora_checkpoint_into_dit(
    dit: nn.Module,
    *,
    checkpoint_dir: Path,
    lora_path: Path,
    trainer_state: dict[str, Any],
    lora_config: dict[str, Any],
    label: str,
) -> list[str]:
    init_merged_names: list[str] = []
    if _state_or_config_bool(trainer_state, "init_lora_merged_into_base"):
        init_merged_names = _merge_recorded_init_lora_into_dit(
            dit, trainer_state=trainer_state, checkpoint_dir=checkpoint_dir
        )
    injected = inject_lora_adapters(dit, lora_config)
    if not injected:
        raise RuntimeError(
            f"{label} LoRA injection matched zero modules before loading checkpoint."
        )
    load_lora_state_dict(dit, load_safetensors(str(lora_path), device="cpu"), strict=True)
    merged_names = merge_lora_adapters(dit)
    if not merged_names:
        raise RuntimeError(f"{label} LoRA merge merged zero adapters.")
    if has_lora_adapters(dit):
        raise RuntimeError(f"{label} LoRA adapters remain after merge.")
    return [*init_merged_names, *merged_names]


def _merge_recorded_init_lora_into_dit(
    dit: nn.Module, *, trainer_state: dict[str, Any], checkpoint_dir: Path
) -> list[str]:
    init_checkpoint = _init_lora_checkpoint_value(trainer_state, checkpoint_dir=checkpoint_dir)
    (init_dir, init_lora_path, init_state_path) = _resolve_stage0_lora_file_or_dir(
        init_checkpoint, label="recorded init Stage 0"
    )
    if init_dir == checkpoint_dir:
        raise ValueError(f"Stage 0 init_lora_checkpoint points to itself: {checkpoint_dir}")
    init_state = _json_load(init_state_path)
    raw_init_lora_config = init_state.get("lora_config")
    if not isinstance(raw_init_lora_config, dict):
        raise ValueError(
            f"Recorded init Stage 0 trainer state is missing dict `lora_config`: {init_state_path}"
        )
    init_lora_config = normalize_lora_config(raw_init_lora_config)
    if float(init_lora_config["dropout"]) != 0.0:
        raise ValueError(
            f"Recorded init Stage 0 LoRA must have dropout=0.0 for deterministic merging, got {init_lora_config['dropout']!r} in {init_state_path}."
        )
    _validate_recorded_init_lora_config(
        trainer_state, init_lora_config, source_state_path=init_state_path
    )
    return _merge_lora_checkpoint_into_dit(
        dit,
        checkpoint_dir=init_dir,
        lora_path=init_lora_path,
        trainer_state=init_state,
        lora_config=init_lora_config,
        label="Recorded init Stage 0",
    )


def _load_stage1_wan_core(
    cfg: Any, *, device: str, dtype: torch.dtype, family: str, dit_role: str
) -> Wan22Core:
    os.environ.setdefault("DREAMFREE_MODEL_BASE_PATH", str(getattr(cfg, "model_base_path")))
    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=dtype,
        model_id=str(getattr(cfg, "model_id")),
        tokenizer_model_id=str(getattr(cfg, "tokenizer_model_id")),
        tokenizer_max_len=int(getattr(cfg, "tokenizer_max_len")),
        load_text_encoder=False,
        load_vae=bool(getattr(cfg, "load_vae", True)),
        redirect_common_files=bool(getattr(cfg, "redirect_common_files")),
        pretrained_load_broadcast=bool(getattr(cfg, "pretrained_load_broadcast", True)),
        dit_config=default_stage1_dit_config(
            use_gradient_checkpointing=bool(getattr(cfg, "use_gradient_checkpointing")),
            video_model_family=family,
        ),
        video_model_family=family,
        dit_role=dit_role,
        train_shift=_cfg_float_with_default(cfg, "train_shift", 5.0),
        infer_shift=_cfg_float_with_default(cfg, "infer_shift", 5.0),
        num_train_timesteps=int(getattr(cfg, "num_train_timesteps")),
    )


def load_stage0_merged_wan_core(
    cfg: Any, *, device: str, dtype: torch.dtype
) -> Stage0MergedWanCore:
    """Load Wan and merge the Stage 0 LoRA checkpoint into the live base DiT."""
    family = _video_model_family(cfg)
    dit_role = str(getattr(cfg, "teacher_dit_role", getattr(cfg, "student_dit_role", "single")))
    if not _stage0_lora_checkpoint_enabled(cfg):
        if family != "wan22_i2v_a14b":
            raise ValueError(
                "stage0_lora_checkpoint may be disabled only for video_model_family='wan22_i2v_a14b'."
            )
        if not _allow_base_without_stage0_debug(cfg):
            raise ValueError(
                "14B Stage 0 robotics adaptation is required. Set stage0_lora_checkpoint to an 14B Stage 0 checkpoint matching the selected DiT role, or set allow_base_without_stage0_debug=true only for explicit loader/debug smoke tests."
            )
        if dit_role not in {"low_noise", "high_noise"}:
            raise ValueError(
                f"14B base teacher/student loading requires teacher_dit_role='low_noise' or 'high_noise', got {dit_role!r}."
            )
        if str(getattr(cfg, "model_id")) != "Wan-AI/Wan2.2-I2V-A14B":
            raise ValueError(
                f"14B loading requires model_id='Wan-AI/Wan2.2-I2V-A14B', got {getattr(cfg, 'model_id')!r}."
            )
        core = _load_stage1_wan_core(
            cfg, device=device, dtype=dtype, family=family, dit_role=dit_role
        )
        core.requires_grad_(False)
        checkpoint_dir = (
            Path(str(getattr(cfg, "model_base_path")))
            / "Wan-AI"
            / "Wan2.2-I2V-A14B"
            / f"{dit_role}_model"
        )
        return Stage0MergedWanCore(
            core=core,
            checkpoint_dir=checkpoint_dir,
            trainer_state={
                "global_step": 0,
                "lora_config": {"enabled": False},
                "video_model_family": family,
                "dit_role": dit_role,
                "vae_family": "wan2.1",
                "stage0_adaptation": {"enabled": False, "debug_base_without_stage0": True},
                "config": {
                    "model_id": str(getattr(cfg, "model_id")),
                    "tokenizer_model_id": str(getattr(cfg, "tokenizer_model_id")),
                    "tokenizer_max_len": int(getattr(cfg, "tokenizer_max_len")),
                    "video_model_family": family,
                    "dit_role": dit_role,
                    "i2v_condition_mode": "first_image_mask_condition",
                },
            },
            stage0_lora_config={"enabled": False},
            merged_module_names=[],
        )
    (checkpoint_dir, lora_path, trainer_state, stage0_lora_config) = _validated_stage0_lora_payload(
        cfg
    )
    core = _load_stage1_wan_core(cfg, device=device, dtype=dtype, family=family, dit_role=dit_role)
    merged_names = _merge_lora_checkpoint_into_dit(
        core.dit,
        checkpoint_dir=checkpoint_dir,
        lora_path=lora_path,
        trainer_state=trainer_state,
        lora_config=stage0_lora_config,
        label="Stage 0",
    )
    core.requires_grad_(False)
    return Stage0MergedWanCore(
        core=core,
        checkpoint_dir=checkpoint_dir,
        trainer_state=trainer_state,
        stage0_lora_config=stage0_lora_config,
        merged_module_names=merged_names,
    )


def load_stage0_merged_backbone(
    cfg: Any, *, device: str, dtype: torch.dtype
) -> Stage0MergedBackbone:
    """Load the shared Stage 1 Wan backbone and inject the student LoRA."""
    student_init_mode = _stage1_student_init_mode(cfg)
    stage1_lora_config = normalize_lora_config(
        {
            "enabled": True,
            "rank": int(getattr(cfg, "lora_rank")),
            "alpha": float(getattr(cfg, "lora_alpha")),
            "dropout": float(getattr(cfg, "lora_dropout")),
            "target_modules": str(getattr(cfg, "lora_target_modules")),
        }
    )
    if student_init_mode == STAGE1_STUDENT_INIT_STAGE0_MERGED:
        merged = load_stage0_merged_wan_core(cfg, device=device, dtype=dtype)
        core = merged.core
        stage1_module_names = inject_lora_adapters(core.dit, stage1_lora_config)
        if not stage1_module_names:
            raise RuntimeError("Stage 1 LoRA injection matched zero modules.")
        set_lora_requires_grad(core.dit, True)
        _assert_non_lora_frozen(core.dit)
        return Stage0MergedBackbone(
            core=core,
            checkpoint_dir=merged.checkpoint_dir,
            trainer_state=merged.trainer_state,
            stage0_lora_config=merged.stage0_lora_config,
            stage1_lora_config=stage1_lora_config,
            merged_module_names=merged.merged_module_names,
            stage0_module_names=merged.merged_module_names,
            stage1_module_names=stage1_module_names,
            student_init_mode=student_init_mode,
            stage0_merged_into_student_base=True,
            teacher_lora_adapter_name=None,
            student_lora_adapter_name=None,
        )
    if not _stage0_lora_checkpoint_enabled(cfg):
        raise ValueError(
            f"student_init_mode={student_init_mode!r} requires a Stage 0 LoRA checkpoint for the teacher adapter."
        )
    family = _video_model_family(cfg)
    dit_role = str(getattr(cfg, "teacher_dit_role", getattr(cfg, "student_dit_role", "single")))
    (checkpoint_dir, lora_path, trainer_state, stage0_lora_config) = _validated_stage0_lora_payload(
        cfg
    )
    core = _load_stage1_wan_core(cfg, device=device, dtype=dtype, family=family, dit_role=dit_role)
    core.requires_grad_(False)
    init_merged_module_names: list[str] = []
    if _state_or_config_bool(trainer_state, "init_lora_merged_into_base"):
        init_merged_module_names = _merge_recorded_init_lora_into_dit(
            core.dit, trainer_state=trainer_state, checkpoint_dir=checkpoint_dir
        )
    stage0_module_names = inject_lora_adapters(
        core.dit, stage0_lora_config, adapter_name=STAGE1_TEACHER_LORA_ADAPTER_NAME
    )
    if not stage0_module_names:
        raise RuntimeError("Stage 0 teacher LoRA injection matched zero modules.")
    load_lora_state_dict(
        core.dit,
        load_safetensors(str(lora_path), device="cpu"),
        strict=True,
        adapter_name=STAGE1_TEACHER_LORA_ADAPTER_NAME,
    )
    set_lora_requires_grad(core.dit, False, adapter_name=STAGE1_TEACHER_LORA_ADAPTER_NAME)
    stage1_module_names = inject_lora_adapters(
        core.dit, stage1_lora_config, adapter_name=STAGE1_STUDENT_LORA_ADAPTER_NAME
    )
    if not stage1_module_names:
        raise RuntimeError("Stage 1 student LoRA injection matched zero modules.")
    set_lora_requires_grad(core.dit, True, adapter_name=STAGE1_STUDENT_LORA_ADAPTER_NAME)
    _assert_non_lora_frozen(core.dit)
    return Stage0MergedBackbone(
        core=core,
        checkpoint_dir=checkpoint_dir,
        trainer_state=trainer_state,
        stage0_lora_config=stage0_lora_config,
        stage1_lora_config=stage1_lora_config,
        merged_module_names=init_merged_module_names,
        stage0_module_names=stage0_module_names,
        stage1_module_names=stage1_module_names,
        student_init_mode=student_init_mode,
        stage0_merged_into_student_base=False,
        teacher_lora_adapter_name=STAGE1_TEACHER_LORA_ADAPTER_NAME,
        student_lora_adapter_name=STAGE1_STUDENT_LORA_ADAPTER_NAME,
    )
