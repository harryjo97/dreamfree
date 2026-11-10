from __future__ import annotations

import copy
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from safetensors.torch import save_file as save_safetensors

from dreamfree.data.action_processing import _parse_dataset_dirs, resolve_stage2_action_horizon
from dreamfree.models import action_history
from dreamfree.models.action_config import _parse_optional_int, _require_dict
from dreamfree.models.action_training import (
    ActionTrainingModel,
    _frozen_student_readout_layer_idx,
    _readout_dim,
    _readout_eval_num_queries,
    _stage2_readout_layer_idx,
)
from dreamfree.models.student import Stage1StudentSource
from dreamfree.models.student_readout import (
    normalize_readout_input_projection_mlp_ratio,
    normalize_readout_input_projection_type,
    readout_input_projection_metadata,
    readout_input_projection_mlp_ratio_from_metadata,
    readout_input_projection_type_from_metadata,
)
from dreamfree.training.runtime import _atomic_torch_save, _json_dump, _json_load
from dreamfree.training.stage2.config import Stage2Config
from dreamfree.training.stage2.constants import (
    DREAMFREE_STAGE2_CHECKPOINT_DIR_RE,
    DREAMFREE_STAGE2_CHECKPOINT_TYPE,
    DREAMFREE_STAGE2_CHECKPOINT_VERSION,
    DREAMFREE_STAGE2_REQUIRED_RESUME_FILES,
    logger,
)
from dreamfree.training.stage2.validation import (
    _check_equal,
    _is_dreamfree_stage2_droid_dataset,
    _metadata_path_identity,
)


def prepare_release_initialization(cfg: Stage2Config) -> tuple[Path, dict] | None:
    """Initialize a new Stage 2 run from released weights, never optimizer state."""
    from dreamfree.models.release import (
        is_release_checkpoint,
        read_checkpoint_metadata,
        resolve_policy_checkpoint,
    )

    if not cfg.init_checkpoint:
        return None
    if cfg.resume:
        raise ValueError(
            "init_checkpoint and resume are mutually exclusive: release fine-tuning starts a new optimizer and step counter."
        )
    path = resolve_policy_checkpoint(cfg.init_checkpoint)
    if not is_release_checkpoint(path):
        raise ValueError(
            "init_checkpoint requires a DreamFree release root or stage2 directory; use resume for a native training checkpoint."
        )
    metadata = read_checkpoint_metadata(path, stage="stage2")
    cfg.stage1_checkpoint = metadata["source_stage1_checkpoint"]
    if not cfg.dataset_stats_path:
        cfg.dataset_stats_path = metadata["dataset"]["dataset_stats_path"]
    cfg.skip_dit_load_from_pretrain = True
    return path, metadata


def validate_release_initialization(
    cfg: Stage2Config, metadata: dict, *, source: Stage1StudentSource
) -> None:
    # Fine-tuning may change data locations; model and conditioning contracts must match.
    metadata = copy.deepcopy(metadata)
    metadata["dataset"].update(
        dataset_dirs=_parse_dataset_dirs(cfg.dataset_dirs),
        dataset_stats_path=str(cfg.dataset_stats_path),
        text_embedding_cache_dir=str(cfg.text_embedding_cache_dir),
    )
    validate_dreamfree_stage2_checkpoint_metadata(cfg, metadata, source=source)


def build_dreamfree_stage2_checkpoint_metadata(
    *, cfg: Stage2Config, model: ActionTrainingModel, global_step: int
) -> dict[str, Any]:
    dataset_dirs = _parse_dataset_dirs(cfg.dataset_dirs)
    source_meta = model.source_stage1_metadata
    source_student = _require_dict(source_meta.get("student"), label="Stage 1 metadata.student")
    source_shapes = _require_dict(source_meta.get("shapes"), label="Stage 1 metadata.shapes")
    source_teacher_state = _require_dict(
        source_meta.get("teacher_state"), label="Stage 1 metadata.teacher_state"
    )
    shape_meta = dict(model.last_shape_metadata)
    readout_dim = _readout_dim(cfg)
    readout_type = str(cfg.readout_type)
    _readout_eval_num_queries(cfg)
    observed_readout_tokens = shape_meta.get("readout_tokens")
    stage0_enabled = bool(model.student.stage0_lora_config.get("enabled", True))
    readout_metadata = {
        "type": readout_type,
        "input_layer_j": int(_frozen_student_readout_layer_idx(model.student)),
        "num_queries": int(cfg.readout_num_queries),
        "num_layers": int(cfg.readout_num_layers),
        "num_heads": int(cfg.readout_num_heads),
        "mlp_ratio": float(cfg.readout_mlp_ratio),
        "dropout": float(cfg.readout_dropout),
        "input_dim": int(model.student.expected_student_hidden_dim),
        "readout_dim": int(readout_dim),
        "output_dim": int(cfg.action_hidden_dim),
        "observed_tokens": observed_readout_tokens,
    }
    readout_metadata.update(
        readout_input_projection_metadata(
            input_projection_type=cfg.readout_input_projection_type,
            input_projection_mlp_ratio=cfg.readout_input_projection_mlp_ratio,
        )
    )
    metadata = {
        "checkpoint_type": DREAMFREE_STAGE2_CHECKPOINT_TYPE,
        "checkpoint_version": DREAMFREE_STAGE2_CHECKPOINT_VERSION,
        "global_step": int(global_step),
        "config": asdict(cfg),
        "source_stage1_checkpoint": model.source_stage1_checkpoint,
        "source_stage1_global_step": int(model.student.global_step),
        "source_stage1_model_contract": dict(source_meta.get("model_contract", {})),
        "source_stage1_teacher_state": {"loss": str(source_teacher_state["loss"])},
        "source_stage1_student_metadata": {
            "layer_j": int(source_student["layer_j"]),
            "hook_point": str(source_student["hook_point"]),
            "video_attention_mask_mode": str(source_student["video_attention_mask_mode"]),
            "i2v_condition_mode": str(source_student.get("i2v_condition_mode", "none")),
            "tokens_per_frame": int(source_shapes["tokens_per_frame"]),
            "hidden_dim": int(source_shapes["D_student"]),
            "latent_shape": [int(v) for v in source_shapes["latent_shape"]],
        },
        "stage0_lora_checkpoint": None
        if not stage0_enabled
        else str(model.student.stage0_checkpoint_dir),
        "stage0_global_step": None
        if not stage0_enabled or model.student.stage0_trainer_state is None
        else int(model.student.stage0_trainer_state["global_step"]),
        "stage0_lora_config": dict(model.student.stage0_lora_config),
        "student_init_mode": str(model.student.student_init_mode),
        "stage0_merged_into_student_base": bool(model.student.stage0_merged_into_student_base),
        "student_lora_adapter_name": model.student.student_lora_adapter_name,
        "model_contract": {
            "video_model_family": str(cfg.video_model_family),
            "model_id": str(cfg.model_id),
            "video_hidden_dim": int(cfg.video_hidden_dim),
            "video_latent_dim": int(cfg.video_latent_dim),
            "video_dit_input_dim": int(cfg.video_dit_input_dim),
            "video_dit_output_dim": int(cfg.video_dit_output_dim),
        },
        "dataset": {
            "name": str(cfg.dataset_name),
            "dataset_dirs": dataset_dirs,
            "dataset_stats_path": str(cfg.dataset_stats_path),
            "text_embedding_cache_dir": str(cfg.text_embedding_cache_dir),
            "context_len": int(cfg.tokenizer_max_len),
            "num_frames": int(cfg.num_frames),
            "action_video_freq_ratio": int(cfg.action_video_freq_ratio),
            "action_horizon": resolve_stage2_action_horizon(cfg),
            "video_size": [int(cfg.height), int(cfg.width)],
            "action_dim": int(cfg.action_dim),
            "proprio_dim": int(cfg.proprio_dim),
            "camera_layout": str(cfg.camera_layout),
            "use_cached_latents": bool(cfg.use_cached_latents),
            "video_latent_cache_dir": cfg.video_latent_cache_dir,
        },
        "student": {
            "train_mode": "frozen",
            "layer_j": int(model.student.student_layer_idx),
            "hook_point": str(model.student.student_hook_point),
            "video_attention_mask_mode": "first_frame_causal",
            "i2v_condition_mode": str(model.student.student_i2v_condition_mode),
            "token_norm": str(cfg.student_token_norm),
            "tokens_per_frame": int(model.student.expected_student_tokens_per_frame),
            "hidden_dim": int(model.student.expected_student_hidden_dim),
        },
        "readout": readout_metadata,
        "action": {
            "init_mode": str(cfg.action_init_mode),
            "action_dit_pretrained_path": str(cfg.action_dit_pretrained_path),
            "skip_dit_load_from_pretrain": bool(cfg.skip_dit_load_from_pretrain),
            "train_mode": str(cfg.action_train_mode),
            "output_parameterization": "velocity",
            "loss_parameterization": "flow_matching",
            "action_dim": int(cfg.action_dim),
            "proprio_dim": int(cfg.proprio_dim),
            "action_horizon": shape_meta.get("action_horizon"),
            "hidden_dim": int(cfg.action_hidden_dim),
            "proprio_conditioning": {
                "mode": str(cfg.proprio_conditioning_mode),
                "obs_horizon": int(cfg.proprio_obs_horizon),
                "obs_dropout": float(cfg.proprio_obs_dropout),
                "state_timestep_sigma": float(cfg.proprio_state_timestep_sigma),
                "loss_applies_to": "action_suffix_only",
            },
            "scheduler": {
                "family": "continuous_flow_matching",
                "train_shift": float(cfg.action_train_shift),
                "num_train_timesteps": int(cfg.num_train_timesteps),
            },
        },
        "conditioner": {
            "mode": "latent_query_cross_attention",
            "student_readout_input_dim": int(cfg.action_hidden_dim),
            "action_context_dim": int(cfg.action_hidden_dim),
            "projection": "linear",
            "token_policy": "all_readout_tokens",
            "concat_order": str(cfg.student_context_concat_order),
            "context_norm": str(cfg.student_context_norm),
            "text_context_length": int(cfg.tokenizer_max_len),
            "query_prefix_tokens": shape_meta.get("query_prefix_tokens", 1),
            "final_kv_contract": "text_plus_student_readout_only",
        },
        "trainable_modules": {
            "student": False,
            "readout": True,
            "action_dit": True,
            "conditioner": True,
            "stage0_merged_backbone": False,
            "vae": False,
            "text_encoder": False,
        },
        "shapes": {
            "student_tokens_per_frame": shape_meta.get("student_tokens_per_frame"),
            "student_hidden_dim": shape_meta.get("student_hidden_dim"),
            "readout_tokens": observed_readout_tokens,
            "readout_hidden_dim": shape_meta.get("readout_hidden_dim"),
            "student_context_length": shape_meta.get("student_context_length"),
            "query_prefix_tokens": shape_meta.get("query_prefix_tokens"),
        },
    }
    from dreamfree.data import yam

    if cfg.dataset_name in yam.DATASET_NAMES:
        metadata["abc_contract"] = yam.checkpoint_contract(cfg)
    action_history.add_checkpoint_metadata(metadata, cfg)
    return metadata


def validate_dreamfree_stage2_checkpoint_metadata(
    cfg: Stage2Config, metadata: dict[str, Any], *, source: Stage1StudentSource
) -> None:
    action_history.validate_checkpoint(metadata, cfg)
    mismatches: list[str] = []
    checkpoint_type, checkpoint_version = action_history.checkpoint_identity(cfg)
    _check_equal(
        mismatches,
        "checkpoint_type",
        metadata.get("checkpoint_type"),
        checkpoint_type,
    )
    _check_equal(
        mismatches,
        "checkpoint_version",
        metadata.get("checkpoint_version"),
        checkpoint_version,
    )
    from dreamfree.data import yam

    if cfg.dataset_name in yam.DATASET_NAMES:
        _check_equal(
            mismatches, "abc_contract", metadata.get("abc_contract"), yam.checkpoint_contract(cfg)
        )
    metadata_stage1_checkpoint = metadata.get("source_stage1_checkpoint")
    expected_stage1_checkpoint = str(source.checkpoint_dir)
    if _metadata_path_identity(metadata_stage1_checkpoint) != _metadata_path_identity(
        expected_stage1_checkpoint
    ):
        mismatches.append(
            f"source_stage1_checkpoint: actual={metadata_stage1_checkpoint!r} expected={expected_stage1_checkpoint!r}"
        )
    _check_equal(
        mismatches,
        "source_stage1_global_step",
        int(metadata.get("source_stage1_global_step", -1)),
        int(source.global_step),
    )
    source_teacher_state = metadata.get("source_stage1_teacher_state")
    if not isinstance(source_teacher_state, dict):
        mismatches.append("source_stage1_teacher_state must be a dict")
    dataset = _require_dict(
        metadata.get("dataset"), label="DreamFree Stage 2 checkpoint metadata.dataset"
    )
    _check_equal(mismatches, "dataset.name", dataset.get("name"), str(cfg.dataset_name))
    _check_equal(
        mismatches,
        "dataset.dataset_dirs",
        dataset.get("dataset_dirs"),
        _parse_dataset_dirs(cfg.dataset_dirs),
    )
    _check_equal(
        mismatches,
        "dataset.dataset_stats_path",
        dataset.get("dataset_stats_path"),
        str(cfg.dataset_stats_path),
    )
    _check_equal(
        mismatches,
        "dataset.text_embedding_cache_dir",
        dataset.get("text_embedding_cache_dir"),
        str(cfg.text_embedding_cache_dir),
    )
    _check_equal(
        mismatches,
        "dataset.context_len",
        int(dataset.get("context_len", -1)),
        int(cfg.tokenizer_max_len),
    )
    _check_equal(
        mismatches, "dataset.num_frames", int(dataset.get("num_frames", -1)), int(cfg.num_frames)
    )
    _check_equal(
        mismatches,
        "dataset.action_video_freq_ratio",
        int(dataset.get("action_video_freq_ratio", -1)),
        int(cfg.action_video_freq_ratio),
    )
    expected_action_horizon = resolve_stage2_action_horizon(cfg)
    require_action_horizon = _is_dreamfree_stage2_droid_dataset(cfg.dataset_name)
    if dataset.get("action_horizon") is not None or require_action_horizon:
        _check_equal(
            mismatches,
            "dataset.action_horizon",
            int(dataset.get("action_horizon", -1)),
            expected_action_horizon,
        )
    _check_equal(
        mismatches,
        "dataset.video_size",
        dataset.get("video_size"),
        [int(cfg.height), int(cfg.width)],
    )
    _check_equal(
        mismatches, "dataset.action_dim", int(dataset.get("action_dim", -1)), int(cfg.action_dim)
    )
    _check_equal(
        mismatches, "dataset.proprio_dim", int(dataset.get("proprio_dim", -1)), int(cfg.proprio_dim)
    )
    _check_equal(
        mismatches,
        "dataset.camera_layout",
        dataset.get("camera_layout"),
        str(cfg.camera_layout),
    )
    student = _require_dict(
        metadata.get("student"), label="DreamFree Stage 2 checkpoint metadata.student"
    )
    _check_equal(mismatches, "student.train_mode", student.get("train_mode"), "frozen")
    _check_equal(
        mismatches,
        "student.layer_j",
        int(student.get("layer_j", -1)),
        int(source.student_layer_idx),
    )
    _check_equal(
        mismatches, "student.hook_point", student.get("hook_point"), str(source.student_hook_point)
    )
    _check_equal(
        mismatches,
        "student.video_attention_mask_mode",
        student.get("video_attention_mask_mode"),
        "first_frame_causal",
    )
    _check_equal(
        mismatches,
        "student.i2v_condition_mode",
        student.get("i2v_condition_mode"),
        str(source.student_i2v_condition_mode),
    )
    _check_equal(
        mismatches,
        "student.tokens_per_frame",
        int(student.get("tokens_per_frame", -1)),
        int(source.expected_student_tokens_per_frame),
    )
    _check_equal(
        mismatches,
        "student.hidden_dim",
        int(student.get("hidden_dim", -1)),
        int(source.expected_student_hidden_dim),
    )
    readout = _require_dict(
        metadata.get("readout"), label="DreamFree Stage 2 checkpoint metadata.readout"
    )
    metadata_readout_layer = readout.get("input_layer_j", student.get("layer_j", -1))
    _check_equal(mismatches, "readout.type", readout.get("type"), str(cfg.readout_type))
    _check_equal(
        mismatches,
        "readout.input_layer_j",
        int(metadata_readout_layer),
        int(_stage2_readout_layer_idx(cfg, source)),
    )
    _check_equal(
        mismatches,
        "readout.num_queries",
        int(readout.get("num_queries", -1)),
        int(cfg.readout_num_queries),
    )
    _check_equal(
        mismatches,
        "readout.num_layers",
        int(readout.get("num_layers", -1)),
        int(cfg.readout_num_layers),
    )
    _check_equal(
        mismatches,
        "readout.num_heads",
        int(readout.get("num_heads", -1)),
        int(cfg.readout_num_heads),
    )
    _check_equal(
        mismatches,
        "readout.mlp_ratio",
        float(readout.get("mlp_ratio", -1)),
        float(cfg.readout_mlp_ratio),
    )
    _check_equal(
        mismatches, "readout.dropout", float(readout.get("dropout", -1)), float(cfg.readout_dropout)
    )
    _check_equal(
        mismatches,
        "readout.input_dim",
        int(readout.get("input_dim", -1)),
        int(source.expected_student_hidden_dim),
    )
    _check_equal(
        mismatches,
        "readout.readout_dim",
        int(readout.get("readout_dim", -1)),
        int(_readout_dim(cfg)),
    )
    _check_equal(
        mismatches,
        "readout.output_dim",
        int(readout.get("output_dim", -1)),
        int(cfg.action_hidden_dim),
    )
    _check_equal(
        mismatches,
        "readout.input_projection_type",
        readout_input_projection_type_from_metadata(readout),
        normalize_readout_input_projection_type(cfg.readout_input_projection_type),
    )
    _check_equal(
        mismatches,
        "readout.input_projection_mlp_ratio",
        readout_input_projection_mlp_ratio_from_metadata(readout),
        normalize_readout_input_projection_mlp_ratio(cfg.readout_input_projection_mlp_ratio),
    )
    action = _require_dict(
        metadata.get("action"), label="DreamFree Stage 2 checkpoint metadata.action"
    )
    _check_equal(
        mismatches,
        "action.output_parameterization",
        action.get("output_parameterization"),
        "velocity",
    )
    _check_equal(
        mismatches,
        "action.loss_parameterization",
        action.get("loss_parameterization"),
        "flow_matching",
    )
    _check_equal(
        mismatches, "action.action_dim", int(action.get("action_dim", -1)), int(cfg.action_dim)
    )
    _check_equal(
        mismatches, "action.proprio_dim", int(action.get("proprio_dim", -1)), int(cfg.proprio_dim)
    )
    if action.get("action_horizon") is not None or require_action_horizon:
        _check_equal(
            mismatches,
            "action.action_horizon",
            int(action.get("action_horizon", -1)),
            expected_action_horizon,
        )
    _check_equal(
        mismatches,
        "action.hidden_dim",
        int(action.get("hidden_dim", -1)),
        int(cfg.action_hidden_dim),
    )
    scheduler = _require_dict(
        action.get("scheduler"), label="DreamFree Stage 2 checkpoint metadata.action.scheduler"
    )
    _check_equal(
        mismatches, "action.scheduler.family", scheduler.get("family"), "continuous_flow_matching"
    )
    _check_equal(
        mismatches,
        "action.scheduler.train_shift",
        float(scheduler.get("train_shift", -1)),
        float(cfg.action_train_shift),
    )
    _check_equal(
        mismatches,
        "action.scheduler.num_train_timesteps",
        int(scheduler.get("num_train_timesteps", -1)),
        int(cfg.num_train_timesteps),
    )
    proprio_conditioning = _require_dict(
        action.get("proprio_conditioning"),
        label="DreamFree Stage 2 checkpoint metadata.action.proprio_conditioning",
    )
    _check_equal(
        mismatches,
        "action.proprio_conditioning.mode",
        proprio_conditioning.get("mode"),
        str(cfg.proprio_conditioning_mode),
    )
    _check_equal(
        mismatches,
        "action.proprio_conditioning.obs_horizon",
        int(proprio_conditioning.get("obs_horizon", -1)),
        int(cfg.proprio_obs_horizon),
    )
    _check_equal(
        mismatches,
        "action.proprio_conditioning.obs_dropout",
        float(proprio_conditioning.get("obs_dropout", -1)),
        float(cfg.proprio_obs_dropout),
    )
    _check_equal(
        mismatches,
        "action.proprio_conditioning.state_timestep_sigma",
        float(proprio_conditioning.get("state_timestep_sigma", -1)),
        float(cfg.proprio_state_timestep_sigma),
    )
    conditioner = _require_dict(
        metadata.get("conditioner"), label="DreamFree Stage 2 checkpoint metadata.conditioner"
    )
    _check_equal(
        mismatches,
        "conditioner.concat_order",
        conditioner.get("concat_order"),
        str(cfg.student_context_concat_order),
    )
    _check_equal(
        mismatches,
        "conditioner.context_norm",
        conditioner.get("context_norm"),
        str(cfg.student_context_norm),
    )
    _check_equal(
        mismatches,
        "conditioner.final_kv_contract",
        conditioner.get("final_kv_contract"),
        "text_plus_student_readout_only",
    )
    trainable = _require_dict(
        metadata.get("trainable_modules"),
        label="DreamFree Stage 2 checkpoint metadata.trainable_modules",
    )
    _check_equal(mismatches, "trainable_modules.student", bool(trainable.get("student")), False)
    _check_equal(mismatches, "trainable_modules.readout", bool(trainable.get("readout")), True)
    _check_equal(
        mismatches, "trainable_modules.action_dit", bool(trainable.get("action_dit")), True
    )
    _check_equal(
        mismatches, "trainable_modules.conditioner", bool(trainable.get("conditioner")), True
    )
    global_step = int(metadata.get("global_step", -1))
    if global_step < 0:
        mismatches.append(f"global_step must be non-negative, got {global_step}.")
    if mismatches:
        raise ValueError("DreamFree Stage 2 checkpoint metadata mismatch: " + "; ".join(mismatches))


def _resolve_resume_dir(cfg: Stage2Config) -> Path | None:
    if not cfg.resume:
        return None
    resume = Path(cfg.resume).expanduser().resolve()
    if resume.is_file():
        raise ValueError(
            f"DreamFree Stage 2 resume must point to a checkpoint directory, got file: {resume}"
        )
    if not resume.is_dir():
        raise FileNotFoundError(f"DreamFree Stage 2 resume directory not found: {resume}")
    return resume


def _read_resume_metadata(
    cfg: Stage2Config, *, source: Stage1StudentSource
) -> tuple[Path, dict[str, Any]] | None:
    resume_dir = _resolve_resume_dir(cfg)
    if resume_dir is None:
        return None
    required = list(DREAMFREE_STAGE2_REQUIRED_RESUME_FILES)
    if bool(cfg.save_optimizer_state):
        required.append("optim.pt")
    missing = [name for name in required if not (resume_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"DreamFree Stage 2 resume checkpoint is missing files {missing}: {resume_dir}"
        )
    metadata = _json_load(resume_dir / "trainer_state.json")
    metadata_validator = validate_dreamfree_stage2_checkpoint_metadata
    metadata_validator(cfg, metadata, source=source)
    step_match = re.fullmatch("step_(\\d+)", resume_dir.name)
    if step_match is not None and int(step_match.group(1)) != int(metadata["global_step"]):
        raise ValueError(
            f"Resume directory step mismatch: dir={resume_dir.name} metadata.global_step={metadata['global_step']}"
        )
    return (resume_dir, metadata)


def _prune_dreamfree_stage2_checkpoints(
    cfg: Stage2Config, *, stage_label: str = "DreamFree Stage 2"
) -> list[Path]:
    """Remove older numeric Stage 2 checkpoints when retention is configured."""
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
        match = DREAMFREE_STAGE2_CHECKPOINT_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), child))
    checkpoints.sort(key=lambda item: (item[0], item[1].name))
    stale = checkpoints[: max(0, len(checkpoints) - max_checkpoints_to_keep)]
    removed: list[Path] = []
    for _, path in stale:
        shutil.rmtree(path)
        removed.append(path)
        logger.info("[checkpoint] removed old %s checkpoint %s", stage_label, path)
    return removed


def save_dreamfree_stage2_checkpoint(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: Stage2Config,
    global_step: int,
    checkpoint_metadata_builder: Any = None,
    stage_label: str = "DreamFree Stage 2",
) -> Path | None:
    checkpoint_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{int(global_step):06d}"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped: ActionTrainingModel = accelerator.unwrap_model(model)
        save_safetensors(
            {
                name: tensor.detach().cpu().contiguous()
                for (name, tensor) in unwrapped.action_model.state_dict().items()
            },
            str(checkpoint_dir / "action.safetensors"),
        )
        save_safetensors(
            {
                name: tensor.detach().cpu().contiguous()
                for (name, tensor) in unwrapped.readout.state_dict().items()
            },
            str(checkpoint_dir / "readout.safetensors"),
        )
        save_safetensors(
            unwrapped.conditioner.teacher_state_state_dict(),
            str(checkpoint_dir / "conditioner.safetensors"),
        )
        save_safetensors(
            unwrapped.conditioner.proprio_state_dict(),
            str(checkpoint_dir / "proprio_encoder.safetensors"),
        )
        metadata_builder = checkpoint_metadata_builder or build_dreamfree_stage2_checkpoint_metadata
        _json_dump(
            checkpoint_dir / "trainer_state.json",
            metadata_builder(cfg=cfg, model=unwrapped, global_step=global_step),
        )
        if cfg.save_optimizer_state:
            _atomic_torch_save(
                checkpoint_dir / "optim.pt",
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            )
        logger.info("[checkpoint] saved %s checkpoint to %s", stage_label, checkpoint_dir)
        _prune_dreamfree_stage2_checkpoints(cfg, stage_label=stage_label)
    accelerator.wait_for_everyone()
    return checkpoint_dir if accelerator.is_main_process else None


def _load_dreamfree_stage2_optimizer_state_if_requested(
    *, optimizer: torch.optim.Optimizer, scheduler: Any, cfg: Stage2Config
) -> None:
    resume_dir = _resolve_resume_dir(cfg)
    if resume_dir is None:
        return
    optim_path = resume_dir / "optim.pt"
    if not optim_path.exists():
        if cfg.save_optimizer_state:
            raise FileNotFoundError(
                f"Missing optimizer state for DreamFree Stage 2 resume: {optim_path}"
            )
        return
    payload = torch.load(str(optim_path), map_location="cpu")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    logger.info("Loaded DreamFree Stage 2 optimizer/scheduler state from %s", optim_path)
