from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from accelerate import Accelerator
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

from dreamfree.data import yam
from dreamfree.data.action_cache import (
    _validate_cache_payload,
    _validate_stage2_latent_cache_contract,
    validate_droid_stage2_latent_cache_metadata,
)
from dreamfree.data.action_processing import (
    DROID_STAGE2_ACTION_DIM,
    DROID_STAGE2_DATASET_ALIASES,
    DROID_STAGE2_HEIGHT,
    DROID_STAGE2_PROPRIO_DIM,
    DROID_STAGE2_WIDTH,
    HORIZONTAL_CONCAT_MULTI_CAMERA,
    ROBOCASA_STAGE2_448_DATASET_ALIASES,
    ROBOCASA_STAGE2_DATASET_ALIASES,
    ROBOTWIN_CONCAT_MULTI_CAMERA,
    _parse_dataset_dirs,
    _stage2_robocasa_layout_spec,
    resolve_stage2_action_horizon,
)
from dreamfree.data.lerobot.cached_latent_dataset import stage2_text_cache_suffix_for_family
from dreamfree.data.stage2 import DROID_GRID_CONCAT_MULTI_CAMERA
from dreamfree.models.action_config import _parse_optional_int, _parse_optional_int_list
from dreamfree.models.action_history import validate_config as validate_action_history_config
from dreamfree.models.action_training import ActionTrainingModel, _readout_dim
from dreamfree.models.release import read_checkpoint_metadata
from dreamfree.models.student import stage1_student_source_model_contract
from dreamfree.models.student_readout import (
    normalize_readout_input_projection_mlp_ratio,
    normalize_readout_input_projection_type,
)
from dreamfree.training.stage2.config import Stage2Config
from dreamfree.training.stage2.constants import (
    DREAMFREE_STAGE2_SELECTED_DROID_DEFAULT_ACTION_HORIZON,
    LIBERO_DREAMFREE_STAGE2_DATASET_NAMES,
    SUPPORTED_READOUT_TYPES,
)


def _normalize_dreamfree_stage2_dataset_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _is_dreamfree_stage2_robocasa_dataset(value: Any) -> bool:
    name = _normalize_dreamfree_stage2_dataset_name(value)
    return name in ROBOCASA_STAGE2_DATASET_ALIASES or name in ROBOCASA_STAGE2_448_DATASET_ALIASES


def _is_dreamfree_stage2_libero_dataset(value: Any) -> bool:
    return _normalize_dreamfree_stage2_dataset_name(value) in LIBERO_DREAMFREE_STAGE2_DATASET_NAMES


def _is_dreamfree_stage2_droid_dataset(value: Any) -> bool:
    return _normalize_dreamfree_stage2_dataset_name(value) in DROID_STAGE2_DATASET_ALIASES


def _parse_path_list(value: Any, *, label: str) -> list[Path]:
    if value is None:
        raise ValueError(f"{label} must contain at least one path.")
    if isinstance(value, (str, Path)):
        raw_parts = str(value).split(",")
    elif isinstance(value, Sequence):
        raw_parts = [str(part) for part in value]
    else:
        raise TypeError(f"{label} must be a comma-separated string or sequence, got {type(value)}.")
    paths = [Path(part.strip()).expanduser() for part in raw_parts if part.strip()]
    if not paths:
        raise ValueError(f"{label} must contain at least one path.")
    return paths


def _is_empty_dreamfree_stage2_override(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
    )


def _coerce_stage1_contract_value_for_stage2(field_name: str, value: Any) -> Any | None:
    if _is_empty_dreamfree_stage2_override(value):
        return None
    if field_name in {
        "tokenizer_max_len",
        "text_dim",
        "video_hidden_dim",
        "video_latent_dim",
        "video_dit_input_dim",
        "video_dit_output_dim",
    }:
        return int(value)
    return str(value)


def _apply_stage1_checkpoint_model_contract(
    cfg: Stage2Config, *, explicit_fields: set[str] | None = None
) -> None:
    if not cfg.stage1_checkpoint or not str(cfg.stage1_checkpoint).strip():
        return
    contract = stage1_student_source_model_contract(cfg.stage1_checkpoint)
    for field_name in (
        "model_id",
        "video_model_family",
        "tokenizer_model_id",
        "tokenizer_max_len",
        "text_dim",
        "video_hidden_dim",
        "video_latent_dim",
        "video_dit_input_dim",
        "video_dit_output_dim",
    ):
        expected = _coerce_stage1_contract_value_for_stage2(field_name, contract.get(field_name))
        if expected is None:
            continue
        current = getattr(cfg, field_name)
        if current == expected:
            continue
        raise ValueError(
            f"Stage 2 configuration conflicts with its stage 1 source: {field_name}: "
            f"configured={current!r}, checkpoint={expected!r}, source={cfg.stage1_checkpoint!r}."
        )


def _apply_selected_droid_action_horizon_default(
    cfg: Stage2Config, *, explicit_fields: set[str] | None
) -> None:
    if not _is_dreamfree_stage2_droid_dataset(cfg.dataset_name):
        return
    if "action_horizon" in set(explicit_fields or ()):
        return
    cfg.action_horizon = DREAMFREE_STAGE2_SELECTED_DROID_DEFAULT_ACTION_HORIZON


def _stage2_text_cache_suffix(cfg: Any) -> str:
    return stage2_text_cache_suffix_for_family(
        str(getattr(cfg, "video_model_family", "wan22_ti2v_5b"))
    )


def _normalize_readout_budget_config(cfg: Stage2Config) -> None:
    cfg.readout_active_query_budgets = _parse_optional_int_list(cfg.readout_active_query_budgets)
    cfg.readout_eval_num_queries = _parse_optional_int(cfg.readout_eval_num_queries)


def _check_equal(mismatches: list[str], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        mismatches.append(f"{name}: actual={actual!r} expected={expected!r}")


def _metadata_path_identity(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if path.exists():
        return str(path.resolve())
    return text


def _assert_stage2_readout_trainable(wrapper: "ActionTrainingModel") -> None:
    readout_params = list(wrapper.readout.named_parameters())
    if not readout_params:
        raise ValueError("DreamFree Stage 2 readout has no parameters to train.")
    frozen = [name for (name, param) in readout_params if not param.requires_grad]
    if frozen:
        preview = frozen[:10]
        suffix = "..." if len(frozen) > 10 else ""
        raise ValueError(
            f"DreamFree Stage 2 readout must remain trainable; frozen parameters: {preview}{suffix}"
        )


def _validate_dreamfree_stage2_robocasa_latent_cache_metadata(
    path: Path, cfg: Stage2Config, *, expected_norm_stats: str | Path | None = None
) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
    expected = {
        "dataset_kind": "robocasa",
        "concat_layout": str(cfg.camera_layout).strip().lower(),
        "action_dim": str(int(cfg.action_dim)),
        "state_dim": str(int(cfg.proprio_dim)),
    }
    mismatches = [
        f"{key}: actual={meta.get(key)!r} expected={value!r}"
        for (key, value) in expected.items()
        if str(meta.get(key)) != value
    ]
    norm_stats_value = (
        cfg.dataset_stats_path if expected_norm_stats is None else expected_norm_stats
    )
    norm_stats_text = "" if norm_stats_value is None else str(norm_stats_value).strip()
    cached_stats = meta.get("normalizer_stats_path")
    if norm_stats_text not in {"", "null", "None"}:
        if expected_norm_stats is not None and (not cached_stats):
            mismatches.append("normalizer_stats_path: missing")
        elif cached_stats:
            actual_stats = _metadata_path_identity(cached_stats)
            expected_stats = _metadata_path_identity(norm_stats_text)
            if actual_stats != expected_stats:
                mismatches.append(
                    f"normalizer_stats_path: actual={actual_stats!r} expected={expected_stats!r}"
                )
    if mismatches:
        raise ValueError(
            f"DreamFree Stage 2 RoboCasa latent cache metadata mismatch in {path}: "
            + "; ".join(mismatches)
        )


def validate_dreamfree_stage2_config(cfg: Stage2Config) -> None:
    validate_action_history_config(cfg)
    _normalize_readout_budget_config(cfg)
    if int(cfg.eval_batches) <= 0:
        raise ValueError(f"eval_batches must be positive, got {cfg.eval_batches}.")
    max_checkpoints_to_keep = _parse_optional_int(cfg.max_checkpoints_to_keep)
    if max_checkpoints_to_keep is not None and max_checkpoints_to_keep <= 0:
        raise ValueError(
            f"max_checkpoints_to_keep must be null or a positive integer, got {cfg.max_checkpoints_to_keep}."
        )
    cfg.max_checkpoints_to_keep = max_checkpoints_to_keep
    if not cfg.stage1_checkpoint or not str(cfg.stage1_checkpoint).strip():
        raise ValueError(
            "stage1_checkpoint must be set for DreamFree Stage 2 student-native training."
        )
    if bool(cfg.compile_dit) and bool(cfg.use_gradient_checkpointing):
        raise ValueError("compile_dit=true is not supported with use_gradient_checkpointing=true.")
    dataset_name = _normalize_dreamfree_stage2_dataset_name(cfg.dataset_name)
    is_robocasa = _is_dreamfree_stage2_robocasa_dataset(dataset_name)
    is_libero = _is_dreamfree_stage2_libero_dataset(dataset_name)
    is_droid = _is_dreamfree_stage2_droid_dataset(dataset_name)
    dataset_dirs = _parse_dataset_dirs(cfg.dataset_dirs)
    for dataset_dir in dataset_dirs:
        if not Path(dataset_dir).exists():
            raise FileNotFoundError(f"DreamFree Stage 2 dataset directory not found: {dataset_dir}")
    norm_stats_text = "" if cfg.dataset_stats_path is None else str(cfg.dataset_stats_path).strip()
    if not (is_libero or is_droid) or norm_stats_text not in {"", "null", "None"}:
        if not norm_stats_text:
            raise ValueError(
                "DreamFree Stage 2 requires dataset_stats_path for RobotWin/RoboCasa datasets."
            )
        if not Path(norm_stats_text).exists():
            raise FileNotFoundError(
                f"DreamFree Stage 2 required path `dataset_stats_path` not found: {norm_stats_text}"
            )
    text_cache_dirs = _parse_path_list(
        cfg.text_embedding_cache_dir, label="DreamFree Stage 2 text_embedding_cache_dir"
    )
    if bool(cfg.use_cached_latents):
        if not is_libero and len(dataset_dirs) != 1:
            raise ValueError(
                f"Cached DreamFree Stage 2 latents currently support a single dataset_dir; got {len(dataset_dirs)} entries."
            )
        if not cfg.video_latent_cache_dir:
            raise ValueError("use_cached_latents=true requires video_latent_cache_dir to be set.")
        latent_cache_dirs = _parse_path_list(
            cfg.video_latent_cache_dir, label="DreamFree Stage 2 video_latent_cache_dir"
        )
        missing_latent_cache_dirs = [str(path) for path in latent_cache_dirs if not path.exists()]
        if missing_latent_cache_dirs:
            raise FileNotFoundError(
                f"DreamFree Stage 2 latent cache directories not found: {missing_latent_cache_dirs}"
            )
        first_episode = None
        for latent_cache_dir in latent_cache_dirs:
            first_episode = next(latent_cache_dir.glob("episode_*.safetensors"), None)
            if first_episode is None:
                first_episode = next(latent_cache_dir.glob("*/episode_*.safetensors"), None)
            if first_episode is not None:
                break
        if first_episode is None:
            raise FileNotFoundError(
                f"No DreamFree Stage 2 precomputed latent files found under {latent_cache_dirs}."
            )
        _validate_stage2_latent_cache_contract(first_episode, cfg)
        if is_robocasa:
            _validate_dreamfree_stage2_robocasa_latent_cache_metadata(first_episode, cfg)
        if is_droid:
            validate_droid_stage2_latent_cache_metadata(
                first_episode, cfg, stage_label="DreamFree Stage 2"
            )
    concat_mode = str(cfg.camera_layout).strip().lower()
    if dataset_name in yam.DATASET_NAMES:
        yam.validate_config(cfg, stage=2)
        yam.checkpoint_contract(cfg)
        yam.validate_stage1_source(
            read_checkpoint_metadata(Path(cfg.stage1_checkpoint), stage="stage1")
        )
    elif is_libero:
        if concat_mode != HORIZONTAL_CONCAT_MULTI_CAMERA:
            raise ValueError("DreamFree Stage 2 LIBERO-family requires camera_layout='horizontal'.")
        if (int(cfg.height), int(cfg.width), int(cfg.action_dim), int(cfg.proprio_dim)) != (
            224,
            448,
            7,
            8,
        ):
            raise ValueError(
                f"DreamFree Stage 2 LIBERO-family requires height=224,width=448,action_dim=7,proprio_dim=8; got {cfg.height}x{cfg.width}, action_dim={cfg.action_dim}, proprio_dim={cfg.proprio_dim}."
            )
    elif is_robocasa:
        layout = _stage2_robocasa_layout_spec(dataset_name)
        expected_concat = str(layout["camera_layout"])
        if concat_mode != expected_concat:
            raise ValueError(
                f"DreamFree Stage 2 RoboCasa requires camera_layout='{expected_concat}'."
            )
        expected = (
            int(layout["height"]),
            int(layout["width"]),
            int(layout["action_dim"]),
            int(layout["proprio_dim"]),
        )
        actual = (int(cfg.height), int(cfg.width), int(cfg.action_dim), int(cfg.proprio_dim))
        if actual != expected:
            raise ValueError(
                f"DreamFree Stage 2 RoboCasa requires height={expected[0]},width={expected[1]},action_dim={expected[2]},proprio_dim={expected[3]}; got {cfg.height}x{cfg.width}, action_dim={cfg.action_dim}, proprio_dim={cfg.proprio_dim}."
            )
    elif is_droid:
        if concat_mode != DROID_GRID_CONCAT_MULTI_CAMERA:
            raise ValueError(
                f"DreamFree Stage 2 DROID requires camera_layout='{DROID_GRID_CONCAT_MULTI_CAMERA}'."
            )
        expected = (
            DROID_STAGE2_HEIGHT,
            DROID_STAGE2_WIDTH,
            DROID_STAGE2_ACTION_DIM,
            DROID_STAGE2_PROPRIO_DIM,
        )
        actual = (int(cfg.height), int(cfg.width), int(cfg.action_dim), int(cfg.proprio_dim))
        if actual != expected:
            raise ValueError(
                f"DreamFree Stage 2 DROID requires height=352,width=640,action_dim=8,proprio_dim=8; got {cfg.height}x{cfg.width}, action_dim={cfg.action_dim}, proprio_dim={cfg.proprio_dim}."
            )
    elif concat_mode != ROBOTWIN_CONCAT_MULTI_CAMERA:
        raise ValueError(
            f"DreamFree Stage 2 RobotWin requires camera_layout='{ROBOTWIN_CONCAT_MULTI_CAMERA}', got {cfg.camera_layout!r}."
        )
    if int(cfg.tokenizer_max_len) != 128 or int(cfg.text_dim) != 4096:
        raise ValueError(
            "DreamFree Stage 2 cached text contract requires context_len=128 and text_dim=4096."
        )
    if int(cfg.tokenizer_max_len) != 128:
        raise ValueError(
            f"DreamFree Stage 2 requires tokenizer_max_len=128, got {cfg.tokenizer_max_len}."
        )
    family = str(cfg.video_model_family)
    if family == "wan22_ti2v_5b":
        if str(cfg.model_id) != "Wan-AI/Wan2.2-TI2V-5B":
            raise ValueError(
                f"wan22_ti2v_5b requires model_id='Wan-AI/Wan2.2-TI2V-5B', got {cfg.model_id!r}."
            )
        expected = (3072, 48, 48, 48)
    elif family == "wan22_i2v_a14b":
        if str(cfg.model_id) != "Wan-AI/Wan2.2-I2V-A14B":
            raise ValueError(
                f"wan22_i2v_a14b requires model_id='Wan-AI/Wan2.2-I2V-A14B', got {cfg.model_id!r}."
            )
        expected = (5120, 16, 36, 16)
    else:
        raise ValueError(
            f"Unsupported video_model_family={cfg.video_model_family!r}; expected 'wan22_ti2v_5b' or 'wan22_i2v_a14b'."
        )
    actual = (
        int(cfg.video_hidden_dim),
        int(cfg.video_latent_dim),
        int(cfg.video_dit_input_dim),
        int(cfg.video_dit_output_dim),
    )
    if actual != expected:
        raise ValueError(
            f"DreamFree Stage 2 video dims mismatch for {family}: actual={actual} expected=(hidden,latent,in,out)={expected}."
        )
    if int(cfg.height) <= 0 or int(cfg.width) <= 0:
        raise ValueError(f"height/width must be positive, got {cfg.height}x{cfg.width}.")
    if int(cfg.num_frames) <= 1 or int(cfg.num_frames) % 4 != 1:
        raise ValueError(f"num_frames must satisfy >1 and %4==1, got {cfg.num_frames}.")
    if (int(cfg.num_frames) - 1) % int(cfg.action_video_freq_ratio) != 0:
        raise ValueError(
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {cfg.num_frames - 1} and {cfg.action_video_freq_ratio}."
        )
    action_horizon = resolve_stage2_action_horizon(cfg)
    if is_droid and _parse_optional_int(getattr(cfg, "action_horizon", None)) is None:
        cfg.action_horizon = action_horizon
    video_transitions = (int(cfg.num_frames) - 1) // int(cfg.action_video_freq_ratio)
    if action_horizon % video_transitions != 0:
        raise ValueError(
            f"action_horizon must be divisible by sparse video transitions, got action_horizon={action_horizon} and video_transitions={video_transitions}."
        )
    if int(cfg.action_dim) <= 0 or int(cfg.proprio_dim) <= 0:
        raise ValueError("action_dim and proprio_dim must be positive.")
    if cfg.student_readout_layer is not None and int(cfg.student_readout_layer) < 0:
        raise ValueError(
            f"student_readout_layer must be non-negative, got {cfg.student_readout_layer}."
        )
    if float(cfg.learning_rate) <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {cfg.learning_rate}.")
    if float(cfg.weight_decay) < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {cfg.weight_decay}.")
    if float(cfg.warmup_ratio) < 0.0 or float(cfg.warmup_ratio) >= 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {cfg.warmup_ratio}.")
    if float(cfg.max_grad_norm) <= 0.0:
        raise ValueError(f"max_grad_norm must be positive, got {cfg.max_grad_norm}.")
    if int(cfg.per_device_batch_size) <= 0 or int(cfg.gradient_accumulation_steps) <= 0:
        raise ValueError("per_device_batch_size and gradient_accumulation_steps must be positive.")
    if int(cfg.num_workers) < 0:
        raise ValueError(f"num_workers must be >= 0, got {cfg.num_workers}.")
    if int(cfg.dataloader_prefetch_factor) <= 0 and int(cfg.num_workers) > 0:
        raise ValueError("dataloader_prefetch_factor must be positive when num_workers > 0.")
    if cfg.num_epochs is not None:
        num_epochs = float(cfg.num_epochs)
        if not math.isfinite(num_epochs) or num_epochs <= 0.0:
            raise ValueError(f"num_epochs must be a finite positive value, got {cfg.num_epochs}.")
    if int(cfg.max_steps) < 0:
        raise ValueError(f"max_steps must be non-negative, got {cfg.max_steps}.")
    if str(cfg.student_video_attention_mask_mode) != "first_frame_causal":
        raise ValueError(
            "DreamFree Stage 2 requires student_video_attention_mask_mode='first_frame_causal'."
        )
    if str(cfg.student_token_norm) != "fixed_tokenwise_layer_norm":
        raise ValueError(
            "DreamFree Stage 2 requires student_token_norm='fixed_tokenwise_layer_norm'."
        )
    if str(cfg.student_conditioning_mode) != "latent_query_cross_attention":
        raise ValueError(
            "DreamFree Stage 2 requires student_conditioning_mode='latent_query_cross_attention'."
        )
    if str(cfg.student_context_norm) != "layernorm_affine":
        raise ValueError("DreamFree Stage 2 requires student_context_norm='layernorm_affine'.")
    if str(cfg.student_context_concat_order) != "text_then_student_readout":
        raise ValueError(
            "DreamFree Stage 2 requires student_context_concat_order='text_then_student_readout'."
        )
    if str(cfg.student_train_mode) != "frozen":
        raise ValueError("DreamFree Stage 2 strict baseline requires student_train_mode='frozen'.")
    if str(cfg.readout_type) not in SUPPORTED_READOUT_TYPES:
        raise ValueError(
            f"Unsupported readout_type={cfg.readout_type!r}; expected one of {sorted(SUPPORTED_READOUT_TYPES)}."
        )
    readout_dim = _readout_dim(cfg)
    if int(cfg.readout_num_queries) <= 0:
        raise ValueError(f"readout_num_queries must be positive, got {cfg.readout_num_queries}.")
    if cfg.readout_active_query_budgets is not None:
        raise ValueError("readout_active_query_budgets must be unset for the fixed readout.")
    if cfg.readout_eval_num_queries is not None and int(cfg.readout_eval_num_queries) != int(
        cfg.readout_num_queries
    ):
        raise ValueError(
            f"readout_eval_num_queries for the existing fixed readout must be unset or equal to readout_num_queries, got readout_eval_num_queries={cfg.readout_eval_num_queries} readout_num_queries={cfg.readout_num_queries}."
        )
    if int(cfg.readout_num_layers) <= 0:
        raise ValueError(f"readout_num_layers must be positive, got {cfg.readout_num_layers}.")
    if int(cfg.readout_num_heads) <= 0 or readout_dim % int(cfg.readout_num_heads) != 0:
        raise ValueError(
            f"readout_num_heads must divide readout_dim, got heads={cfg.readout_num_heads} readout_dim={readout_dim}."
        )
    if float(cfg.readout_mlp_ratio) <= 0.0:
        raise ValueError(f"readout_mlp_ratio must be positive, got {cfg.readout_mlp_ratio}.")
    if not 0.0 <= float(cfg.readout_dropout) < 1.0:
        raise ValueError(f"readout_dropout must be in [0,1), got {cfg.readout_dropout}.")
    normalize_readout_input_projection_type(cfg.readout_input_projection_type)
    normalize_readout_input_projection_mlp_ratio(cfg.readout_input_projection_mlp_ratio)
    if str(cfg.action_train_mode) != "full_action_dit":
        raise ValueError("DreamFree Stage 2 requires action_train_mode='full_action_dit'.")
    if str(cfg.action_output_parameterization) != "velocity":
        raise ValueError("DreamFree Stage 2 requires action_output_parameterization='velocity'.")
    if str(cfg.action_loss_parameterization) != "flow_matching":
        raise ValueError("DreamFree Stage 2 requires action_loss_parameterization='flow_matching'.")
    if str(cfg.proprio_conditioning_mode) != "query_prefix":
        raise ValueError("DreamFree Stage 2 requires proprio_conditioning_mode='query_prefix'.")
    if int(cfg.proprio_obs_horizon) != 1:
        raise ValueError("DreamFree Stage 2 requires proprio_obs_horizon=1.")
    if float(cfg.proprio_obs_dropout) < 0.0 or float(cfg.proprio_obs_dropout) >= 1.0:
        raise ValueError(f"proprio_obs_dropout must be in [0, 1), got {cfg.proprio_obs_dropout}.")
    if int(cfg.num_train_timesteps) <= 0:
        raise ValueError(f"num_train_timesteps must be positive, got {cfg.num_train_timesteps}.")
    text_cache_suffix = _stage2_text_cache_suffix(cfg)
    for cache_dir in text_cache_dirs:
        first_cache = next(cache_dir.glob(f"*.t5_len128.{text_cache_suffix}.pt"), None)
        if first_cache is None:
            # Missing entries are prepared before dataset construction on rank 0.
            continue
        _validate_cache_payload(
            first_cache, context_len=int(cfg.tokenizer_max_len), text_dim=int(cfg.text_dim)
        )


@torch.no_grad()
def evaluate_dreamfree_stage2_batches(
    *, accelerator: Accelerator, model: torch.nn.Module, val_loader: Any, max_batches: int
) -> dict[str, float] | None:
    max_batches = int(max_batches)
    if max_batches <= 0:
        raise ValueError(f"max_batches must be positive, got {max_batches}.")
    unwrapped = accelerator.unwrap_model(model)
    was_training = bool(unwrapped.training)
    unwrapped.eval()
    losses: list[float] = []
    metric_sums: dict[str, float] = {}
    try:
        for batch_idx, sample in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            (loss, loss_dict) = model(sample)
            mean_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
            losses.append(float(mean_loss))
            for key, value in loss_dict.items():
                tensor = torch.tensor(
                    float(value), device=loss.device, dtype=torch.float32
                ).reshape(1)
                metric_key = f"val_{key}"
                metric_sums[metric_key] = metric_sums.get(metric_key, 0.0) + float(
                    accelerator.gather(tensor).mean().item()
                )
    finally:
        if was_training:
            unwrapped.set_stage2_train_mode()
    num_batches = len(losses)
    if num_batches <= 0:
        return None
    metrics = {"val_loss": float(np.mean(losses)), "val_batches": float(num_batches)}
    metrics.update(
        {key: float(value / num_batches) for (key, value) in sorted(metric_sums.items())}
    )
    return metrics


def _load_dreamfree_stage2_model_weights(wrapper: ActionTrainingModel, resume_dir: Path) -> None:
    wrapper.action_model.load_state_dict(
        load_safetensors(str(resume_dir / "action.safetensors"), device="cpu"), strict=True
    )
    wrapper.readout.load_state_dict(
        load_safetensors(str(resume_dir / "readout.safetensors"), device="cpu"), strict=True
    )
    wrapper.conditioner.load_teacher_state_dict(
        load_safetensors(str(resume_dir / "conditioner.safetensors"), device="cpu"), strict=True
    )
    wrapper.conditioner.load_proprio_state_dict(
        load_safetensors(str(resume_dir / "proprio_encoder.safetensors"), device="cpu"), strict=True
    )
