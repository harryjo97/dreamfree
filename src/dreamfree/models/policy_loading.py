"""Shared checkpoint assembly and text conditioning for inference."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from dreamfree.data.lerobot.processors.robot import DreamFreeProcessor
from dreamfree.data.text_cache import validate_text_cache
from dreamfree.models.lora import (
    has_lora_adapters,
    inject_lora_adapters,
    load_lora_state_dict,
    merge_lora_adapters,
)
from dreamfree.models.wan22.helpers.distributed_load import (
    load_safetensors_rank0 as load_safetensors,
)
from dreamfree.models.wan22.i2v_condition import concat_14b_i2v_condition
from dreamfree.models.wan22.stage0_loader import (
    STAGE1_STUDENT_INIT_PRETRAINED_BASE,
    STAGE1_STUDENT_INIT_STAGE0_MERGED,
    STAGE1_STUDENT_LORA_ADAPTER_NAME,
    STAGE1_TEACHER_LORA_ADAPTER_NAME,
    Stage0MergedWanCore,
    _cfg_float_with_default,
    _validate_stage0_trainer_state,
    default_stage1_dit_config,
    resolve_stage0_lora_checkpoint,
    validate_stage0_eval_reference,
)
from dreamfree.models.wan22.wan22 import Wan22Core

logger = logging.getLogger(__name__)
STAGE1_CHECKPOINT_TYPE = "dreamfree_stage1"
STAGE1_SUPPORTED_CHECKPOINT_TYPES = {STAGE1_CHECKPOINT_TYPE}
WAN22_5B_FAMILY = "wan22_ti2v_5b"
WAN22_A14B_FAMILY = "wan22_i2v_a14b"
WAN22_A14B_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B"
WAN22_5B_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_A14B_I2V_MODE = "first_image_mask_condition"
ROBOTWIN_IMAGE_MOSAIC_WIDTH = 320
ROBOTWIN_IMAGE_MOSAIC_HEIGHT = 384


def _load_deployment_safetensors(
    path: str | Path, *, device: str, distributed_checkpoint_broadcast: bool
) -> dict[str, torch.Tensor]:
    if distributed_checkpoint_broadcast:
        return load_safetensors(str(path), device=device)
    from safetensors.torch import load_file

    return load_file(str(path), device=device)


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}.")
    return payload


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _require_dict(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object, got {type(value).__name__}.")
    return value


def _require_existing_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected {label} to be a file: {path}")
    return path


def _stage1_required_student_files_for_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    str(metadata.get("checkpoint_type", "")).strip()
    return ("trainer_state.json", "stage1_lora.safetensors")


def stage1_required_student_files_for_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    return _stage1_required_student_files_for_metadata(metadata)


def _stage1_student_view_metadata(metadata):
    if metadata.get("checkpoint_type") != STAGE1_CHECKPOINT_TYPE:
        raise ValueError("Expected a DreamFree stage 1 checkpoint.")
    return metadata


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _metadata_video_family(metadata: dict[str, Any], *, label: str) -> str:
    metadata = _stage1_student_view_metadata(metadata)
    config = _require_dict(metadata.get("config"), label=f"{label} metadata.config")
    model_contract = metadata.get("model_contract")
    family = None
    if isinstance(model_contract, dict):
        family = model_contract.get("video_model_family")
    if family is None:
        family = config.get("video_model_family", WAN22_5B_FAMILY)
    family = str(family)
    if family not in {WAN22_5B_FAMILY, WAN22_A14B_FAMILY}:
        raise ValueError(
            f"{label} metadata has unsupported video_model_family={family!r}; expected {WAN22_5B_FAMILY!r} or {WAN22_A14B_FAMILY!r}."
        )
    return family


def _expected_hidden_dim_for_family(family: str) -> int:
    if str(family) == WAN22_A14B_FAMILY:
        return 5120
    if str(family) == WAN22_5B_FAMILY:
        return 3072
    raise ValueError(f"Unsupported video_model_family={family!r}.")


def _expected_model_id_for_family(family: str) -> str:
    if str(family) == WAN22_A14B_FAMILY:
        return WAN22_A14B_MODEL_ID
    if str(family) == WAN22_5B_FAMILY:
        return WAN22_5B_MODEL_ID
    raise ValueError(f"Unsupported video_model_family={family!r}.")


def _text_cache_suffix(model_id: str) -> str:
    if str(model_id) == WAN22_5B_MODEL_ID:
        return "wan22ti2v5b"
    if str(model_id) == WAN22_A14B_MODEL_ID:
        return "wan22i2va14b"
    base = str(model_id).split("/")[-1]
    suffix = re.sub("[^a-z0-9]+", "", base.lower())
    if not suffix:
        raise ValueError(f"Cannot derive text cache suffix from model_id={model_id!r}.")
    return suffix


def _parse_text_embedding_cache_dirs(value: Any) -> list[Path]:
    if _is_none_like(value):
        return []
    if isinstance(value, (list, tuple)):
        raw_parts = [str(part) for part in value]
    else:
        raw_parts = str(value).split(",")
    paths = [Path(part.strip()).expanduser().resolve() for part in raw_parts if part.strip()]
    if not paths:
        raise ValueError("text_embedding_cache_dir must contain at least one path when provided.")
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"text_embedding_cache_dir path(s) not found: {missing}")
    return paths


def _text_cache_candidates(
    cache_dirs: Sequence[Path], *, prompt: str, context_len: int, model_id: str
) -> list[Path]:
    digest = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
    suffix = _text_cache_suffix(model_id)
    return [
        cache_dir / f"{digest}.t5_len{int(context_len)}.{suffix}.pt" for cache_dir in cache_dirs
    ]


def load_cached_text_context_for_action_model(
    *,
    prompt: str,
    cache_dirs: Sequence[Path],
    context_len: int,
    text_dim: int,
    model_id: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not cache_dirs:
        raise ValueError(
            "Cached prompt context lookup requires at least one text_embedding_cache_dir."
        )
    cache_paths = _text_cache_candidates(
        cache_dirs, prompt=prompt, context_len=context_len, model_id=model_id
    )
    cache_path = next((path for path in cache_paths if path.exists()), cache_paths[0])
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing precomputed text context for prompt. Tried: {cache_paths}. prompt={prompt!r}. Run dreamfree.data.prepare_text with the required instructions."
        )
    payload = torch.load(str(cache_path), map_location="cpu")
    validate_text_cache(payload, str(cache_path))
    missing = [key for key in ("context", "mask") if key not in payload]
    if missing:
        raise ValueError(f"Text context cache {cache_path} is missing required key(s): {missing}")
    context = payload["context"]
    mask = payload["mask"].bool()
    if not isinstance(context, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError(
            f"Text context cache {cache_path} must contain tensor `context` and `mask`."
        )
    expected_context_shape = (int(context_len), int(text_dim))
    expected_mask_shape = (int(context_len),)
    if context.ndim != 2 or tuple(context.shape) != expected_context_shape:
        raise ValueError(
            f"Cached `context` in {cache_path} must have shape {expected_context_shape}, got {tuple(context.shape)}."
        )
    if mask.ndim != 1 or tuple(mask.shape) != expected_mask_shape:
        raise ValueError(
            f"Cached `mask` in {cache_path} must have shape {expected_mask_shape}, got {tuple(mask.shape)}."
        )
    context = context.clone()
    context[~mask] = 0.0
    context = context.unsqueeze(0).to(device=device, dtype=dtype)
    context_mask = torch.ones((1, int(context_len)), device=device, dtype=torch.bool)
    return (context, context_mask)


def _expected_dit_role_for_family(family: str) -> str:
    return "low_noise" if str(family) == WAN22_A14B_FAMILY else "single"


def _encode_first_frame_latents_for_deploy(
    *, core: Wan22Core, input_image: torch.Tensor, condition_video_num_frames: int, tiled: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    family = str(getattr(core, "video_model_family", WAN22_5B_FAMILY))
    if family == WAN22_5B_FAMILY:
        first_frame_latents = core._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        ).detach()
        return (first_frame_latents, None)
    if family != WAN22_A14B_FAMILY:
        raise ValueError(f"Unsupported video_model_family={family!r}.")
    if int(condition_video_num_frames) <= 0 or int(condition_video_num_frames) % 4 != 1:
        raise ValueError(
            f"14B deployment I2V conditioning requires condition_video_num_frames % 4 == 1, got {condition_video_num_frames}."
        )
    image = input_image
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or int(image.shape[1]) != 3:
        raise ValueError(
            f"`input_image` must have shape [B,3,H,W] or [3,H,W] for 14B deployment conditioning, got {tuple(image.shape)}."
        )
    image = image.to(device=core.device, dtype=core.torch_dtype)
    pseudo_video = image.new_zeros(
        (
            int(image.shape[0]),
            3,
            int(condition_video_num_frames),
            int(image.shape[2]),
            int(image.shape[3]),
        )
    )
    pseudo_video[:, :, 0:1] = image.unsqueeze(2)
    i2v_condition_latents = core._encode_14b_i2v_condition_latents(
        pseudo_video, tiled=tiled
    ).detach()
    return (i2v_condition_latents[:, :, 0:1], i2v_condition_latents)


def _i2v_model_input_for_core(
    *, core: Wan22Core, latents: torch.Tensor, i2v_condition_latents: torch.Tensor | None = None
) -> torch.Tensor:
    family = str(getattr(core, "video_model_family", WAN22_5B_FAMILY))
    if family == WAN22_5B_FAMILY:
        return latents
    if family != WAN22_A14B_FAMILY:
        raise ValueError(f"Unsupported video_model_family={family!r}.")
    if not isinstance(i2v_condition_latents, torch.Tensor):
        raise ValueError(
            "14B deployment I2V model input requires `i2v_condition_latents` from official pseudo-video VAE encoding."
        )
    return concat_14b_i2v_condition(latents=latents, i2v_condition_latents=i2v_condition_latents)


def _build_stage1_loader_cfg_from_metadata(metadata: dict[str, Any]) -> SimpleNamespace:
    metadata = _stage1_student_view_metadata(metadata)
    config = _require_dict(metadata.get("config"), label="Stage 1 metadata.config")
    stage0_lora_config = _require_dict(
        metadata.get("stage0_lora_config"), label="Stage 1 metadata.stage0_lora_config"
    )
    model_contract = metadata.get("model_contract")
    if not isinstance(model_contract, dict):
        model_contract = {}
    family = str(
        model_contract.get("video_model_family", config.get("video_model_family", WAN22_5B_FAMILY))
    )
    dit_role = str(
        model_contract.get(
            "student_dit_role",
            config.get("student_dit_role", _expected_dit_role_for_family(family)),
        )
    )
    return SimpleNamespace(
        model_base_path=str(
            config.get("model_base_path", os.environ.get("DREAMFREE_MODEL_BASE_PATH", ""))
        ),
        stage0_lora_checkpoint=str(metadata["stage0_lora_checkpoint"]),
        stage0_eval_reference=metadata.get("stage0_eval_reference") or "",
        stage0_expected_global_step=int(
            metadata.get("stage0_expected_global_step", metadata.get("stage0_global_step", -1))
        ),
        stage0_expected_lora_rank=int(stage0_lora_config["rank"]),
        stage0_expected_lora_alpha=float(stage0_lora_config["alpha"]),
        stage0_expected_lora_dropout=float(stage0_lora_config["dropout"]),
        stage0_expected_lora_target_modules=str(stage0_lora_config["target_modules"]),
        stage0_expected_model_id=str(
            config.get("stage0_expected_model_id", config.get("model_id"))
        ),
        stage0_expected_video_model_family=str(
            config.get("stage0_expected_video_model_family", family)
        ),
        stage0_expected_dit_role=str(
            config.get("stage0_expected_dit_role", _expected_dit_role_for_family(family))
        ),
        stage0_expected_tokenizer_model_id=str(
            config.get("stage0_expected_tokenizer_model_id", config.get("tokenizer_model_id"))
        ),
        stage0_expected_tokenizer_max_len=int(
            config.get("stage0_expected_tokenizer_max_len", config.get("tokenizer_max_len", 128))
        ),
        model_id=str(config["model_id"]),
        video_model_family=family,
        student_dit_role=dit_role,
        teacher_dit_role=str(
            model_contract.get(
                "teacher_dit_role",
                config.get("teacher_dit_role", _expected_dit_role_for_family(family)),
            )
        ),
        student_init_mode=str(
            metadata.get(
                "student_init_mode",
                config.get("student_init_mode", STAGE1_STUDENT_INIT_STAGE0_MERGED),
            )
        ),
        tokenizer_model_id=str(config["tokenizer_model_id"]),
        tokenizer_max_len=int(config["tokenizer_max_len"]),
        redirect_common_files=bool(config.get("redirect_common_files", True)),
        train_shift=float(config.get("train_shift", 5.0)),
        infer_shift=float(config.get("infer_shift", 5.0)),
        num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
        use_gradient_checkpointing=bool(config.get("use_gradient_checkpointing", False)),
        pretrained_load_broadcast=bool(config.get("pretrained_load_broadcast", True)),
        allow_base_without_stage0_debug=bool(config.get("allow_base_without_stage0_debug", False)),
    )


def _load_stage0_merged_wan_core_with_text(
    cfg: Any,
    *,
    device: str,
    dtype: torch.dtype,
    load_text_encoder: bool = True,
    load_vae: bool = True,
    distributed_checkpoint_broadcast: bool = True,
) -> Stage0MergedWanCore:
    family = str(getattr(cfg, "video_model_family", WAN22_5B_FAMILY))
    dit_role = str(
        getattr(
            cfg,
            "student_dit_role",
            getattr(cfg, "teacher_dit_role", _expected_dit_role_for_family(family)),
        )
    )
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
    os.environ.setdefault("DREAMFREE_MODEL_BASE_PATH", str(getattr(cfg, "model_base_path")))
    core = Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=dtype,
        model_id=str(getattr(cfg, "model_id")),
        tokenizer_model_id=str(getattr(cfg, "tokenizer_model_id")),
        tokenizer_max_len=int(getattr(cfg, "tokenizer_max_len")),
        load_text_encoder=bool(load_text_encoder),
        load_vae=bool(load_vae),
        redirect_common_files=bool(getattr(cfg, "redirect_common_files")),
        pretrained_load_broadcast=bool(getattr(cfg, "pretrained_load_broadcast", True))
        and bool(distributed_checkpoint_broadcast),
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
    logger.info("Injecting Stage 0 LoRA adapters for deployment.")
    injected_stage0 = inject_lora_adapters(core.dit, stage0_lora_config)
    if not injected_stage0:
        raise RuntimeError("Stage 0 LoRA injection matched zero modules before loading checkpoint.")
    logger.info("Loading Stage 0 LoRA state from %s.", lora_path)
    stage0_lora_state = _load_deployment_safetensors(
        lora_path, device="cpu", distributed_checkpoint_broadcast=distributed_checkpoint_broadcast
    )
    logger.info("Applying %d Stage 0 LoRA state tensors.", len(stage0_lora_state))
    load_lora_state_dict(core.dit, stage0_lora_state, strict=True)
    del stage0_lora_state
    logger.info("Merging %d Stage 0 LoRA modules for deployment.", len(injected_stage0))
    merged_names = merge_lora_adapters(core.dit)
    if not merged_names:
        raise RuntimeError("Stage 0 LoRA merge merged zero adapters.")
    if has_lora_adapters(core.dit):
        raise RuntimeError("Stage 0 LoRA adapters remain after merge.")
    logger.info("Finished merging %d Stage 0 LoRA modules for deployment.", len(merged_names))
    core.requires_grad_(False)
    core.eval()
    return Stage0MergedWanCore(
        core=core,
        checkpoint_dir=checkpoint_dir,
        trainer_state=trainer_state,
        stage0_lora_config=stage0_lora_config,
        merged_module_names=merged_names,
    )


def _load_stage0_teacher_lora_wan_core_with_text(
    cfg: Any,
    *,
    device: str,
    dtype: torch.dtype,
    teacher_adapter_name: str,
    load_text_encoder: bool = True,
    load_vae: bool = True,
    distributed_checkpoint_broadcast: bool = True,
) -> Stage0MergedWanCore:
    family = str(getattr(cfg, "video_model_family", WAN22_5B_FAMILY))
    dit_role = str(
        getattr(
            cfg,
            "student_dit_role",
            getattr(cfg, "teacher_dit_role", _expected_dit_role_for_family(family)),
        )
    )
    checkpoint_value = getattr(cfg, "stage0_lora_checkpoint", None)
    if checkpoint_value is None or not str(checkpoint_value).strip():
        raise ValueError("Pretrained-base Stage 1 student loading requires stage0_lora_checkpoint.")
    checkpoint_dir = resolve_stage0_lora_checkpoint(checkpoint_value)
    lora_path = checkpoint_dir / "lora.safetensors"
    state_path = checkpoint_dir / "trainer_state.json"
    if not lora_path.exists():
        raise FileNotFoundError(f"Missing Stage 0 LoRA checkpoint: {lora_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing Stage 0 trainer state: {state_path}")
    trainer_state = _json_load(state_path)
    stage0_lora_config = _validate_stage0_trainer_state(cfg, checkpoint_dir, trainer_state)
    validate_stage0_eval_reference(cfg)
    os.environ.setdefault("DREAMFREE_MODEL_BASE_PATH", str(getattr(cfg, "model_base_path")))
    core = Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=dtype,
        model_id=str(getattr(cfg, "model_id")),
        tokenizer_model_id=str(getattr(cfg, "tokenizer_model_id")),
        tokenizer_max_len=int(getattr(cfg, "tokenizer_max_len")),
        load_text_encoder=bool(load_text_encoder),
        load_vae=bool(load_vae),
        redirect_common_files=bool(getattr(cfg, "redirect_common_files")),
        pretrained_load_broadcast=bool(getattr(cfg, "pretrained_load_broadcast", True))
        and bool(distributed_checkpoint_broadcast),
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
    injected_stage0 = inject_lora_adapters(
        core.dit, stage0_lora_config, adapter_name=teacher_adapter_name
    )
    if not injected_stage0:
        raise RuntimeError(
            "Stage 0 teacher LoRA injection matched zero modules before loading checkpoint."
        )
    load_lora_state_dict(
        core.dit,
        _load_deployment_safetensors(
            lora_path,
            device="cpu",
            distributed_checkpoint_broadcast=distributed_checkpoint_broadcast,
        ),
        strict=True,
        adapter_name=teacher_adapter_name,
    )
    core.requires_grad_(False)
    core.eval()
    return Stage0MergedWanCore(
        core=core,
        checkpoint_dir=checkpoint_dir,
        trainer_state=trainer_state,
        stage0_lora_config=stage0_lora_config,
        merged_module_names=[],
    )


def _load_stage1_student_core_with_text(
    *,
    stage1_checkpoint_path: Path,
    stage1_lora_state_path: Path | None = None,
    metadata: dict[str, Any],
    device: str,
    dtype: torch.dtype,
    load_text_encoder: bool = True,
    load_vae: bool = True,
    distributed_checkpoint_broadcast: bool = True,
) -> Wan22Core:
    metadata = _stage1_student_view_metadata(metadata)
    cfg = _build_stage1_loader_cfg_from_metadata(metadata)
    student_init_mode = (
        str(getattr(cfg, "student_init_mode", STAGE1_STUDENT_INIT_STAGE0_MERGED)).strip().lower()
    )
    if student_init_mode == STAGE1_STUDENT_INIT_PRETRAINED_BASE:
        bundle = _load_stage0_teacher_lora_wan_core_with_text(
            cfg,
            device=device,
            dtype=dtype,
            teacher_adapter_name=str(
                metadata.get("teacher_lora_adapter_name") or STAGE1_TEACHER_LORA_ADAPTER_NAME
            ),
            load_text_encoder=bool(load_text_encoder),
            load_vae=bool(load_vae),
            distributed_checkpoint_broadcast=distributed_checkpoint_broadcast,
        )
        student_adapter_name = str(
            metadata.get("student_lora_adapter_name") or STAGE1_STUDENT_LORA_ADAPTER_NAME
        )
    elif student_init_mode == STAGE1_STUDENT_INIT_STAGE0_MERGED:
        bundle = _load_stage0_merged_wan_core_with_text(
            cfg,
            device=device,
            dtype=dtype,
            load_text_encoder=bool(load_text_encoder),
            load_vae=bool(load_vae),
            distributed_checkpoint_broadcast=distributed_checkpoint_broadcast,
        )
        student_adapter_name = None
    else:
        raise ValueError(
            f"Unsupported Stage 1 student_init_mode for RobotWin eval: {student_init_mode!r}"
        )
    stage1_lora_config = _require_dict(
        metadata.get("stage1_lora_config"), label="Stage 1 metadata.stage1_lora_config"
    )
    logger.info("Injecting Stage 1 student LoRA adapters for deployment.")
    injected = inject_lora_adapters(
        bundle.core.dit, stage1_lora_config, adapter_name=student_adapter_name
    )
    if not injected:
        raise RuntimeError("Stage 1 LoRA injection matched zero modules for DreamFree eval.")
    if stage1_lora_state_path is None:
        stage1_lora_state_path = stage1_checkpoint_path / "stage1_lora.safetensors"
    _require_existing_file(stage1_lora_state_path, label="student LoRA checkpoint")
    logger.info("Loading Stage 1 student LoRA state from %s.", stage1_lora_state_path)
    stage1_lora_state = _load_deployment_safetensors(
        stage1_lora_state_path,
        device="cpu",
        distributed_checkpoint_broadcast=distributed_checkpoint_broadcast,
    )
    logger.info("Applying %d Stage 1 student LoRA state tensors.", len(stage1_lora_state))
    load_lora_state_dict(
        bundle.core.dit, stage1_lora_state, strict=True, adapter_name=student_adapter_name
    )
    del stage1_lora_state
    logger.info("Finished applying Stage 1 student LoRA state.")
    bundle.core.student_lora_adapter_name = student_adapter_name
    if load_text_encoder and (bundle.core.text_encoder is None or bundle.core.tokenizer is None):
        raise ValueError("DreamFree eval requires a Wan text encoder and tokenizer.")
    bundle.core.requires_grad_(False)
    bundle.core.eval()
    return bundle.core


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def build_robotwin_image_mosaic(observation_data: Dict[str, Any]) -> np.ndarray:
    required_cameras = ("head_camera", "left_camera", "right_camera")
    missing = [camera for camera in required_cameras if camera not in observation_data]
    if missing:
        raise KeyError(f"RobotWin image mosaic is missing required cameras: {missing}.")
    head = _resize_rgb(observation_data["head_camera"]["rgb"], (320, 256))
    left = _resize_rgb(observation_data["left_camera"]["rgb"], (160, 128))
    right = _resize_rgb(observation_data["right_camera"]["rgb"], (160, 128))
    bottom = np.concatenate([left, right], axis=1)
    image = np.concatenate([head, bottom], axis=0)
    expected_shape = (ROBOTWIN_IMAGE_MOSAIC_HEIGHT, ROBOTWIN_IMAGE_MOSAIC_WIDTH, 3)
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise RuntimeError(
            f"RobotWin image mosaic contract mismatch: expected uint8 {expected_shape}, got dtype={image.dtype} shape={image.shape}."
        )
    return np.ascontiguousarray(image)


def _normalize_state_with_processor(
    processor: DreamFreeProcessor, state: np.ndarray
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _prepare_text_context_for_action_model(
    context: torch.Tensor, context_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(context_mask.shape) != tuple(context.shape[:-1]):
        raise ValueError(
            f"Text context mask shape must match context without hidden dim: context={tuple(context.shape)} mask={tuple(context_mask.shape)}."
        )
    context_mask = context_mask.to(device=context.device, dtype=torch.bool)
    context = context.masked_fill(~context_mask.unsqueeze(-1), 0.0)
    return (context, torch.ones_like(context_mask, dtype=torch.bool))


def _denormalize_action_with_processor(
    processor: DreamFreeProcessor, action: torch.Tensor
) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
    return denorm.numpy()


def _build_robotwin_image_tensor(
    observation: Dict[str, Any], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    image = build_robotwin_image_mosaic(observation["observation"])
    image_tensor = (
        torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    )
    image_tensor = image_tensor * (2.0 / 255.0) - 1.0
    return image_tensor


class RobotWinPolicyBase:
    def __init__(
        self,
        *,
        processor: DreamFreeProcessor,
        action_horizon: int,
        replan_steps: int,
        timing_enabled: bool,
    ) -> None:
        self.processor = processor
        self.action_horizon = int(action_horizon)
        self.replan_steps = int(replan_steps)
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {self.action_horizon}.")
        if self.replan_steps <= 0 or self.replan_steps > self.action_horizon:
            raise ValueError(
                f"`replan_steps` must be in [1, action_horizon], got replan_steps={self.replan_steps}, action_horizon={self.action_horizon}."
            )
        self.timing_enabled = bool(timing_enabled)
        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

    @staticmethod
    def build_eval_video_frame(observation_data: Dict[str, Any]) -> np.ndarray:
        return build_robotwin_image_mosaic(observation_data)

    def _enqueue_action_chunk(self, action_chunk: np.ndarray) -> None:
        n_exec = min(self.replan_steps, int(action_chunk.shape[0]))
        for idx in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[idx], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def record_issued_action(self, action: np.ndarray) -> None:
        """Optional policy hook invoked after an action command is issued."""
        return None

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("Observation is required when the action queue is empty.")
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)
        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return
        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        self.record_issued_action(action)
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()
