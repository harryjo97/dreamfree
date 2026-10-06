"""Typed configuration shared by the three training entrypoints."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import types
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf

MODEL_DEFAULTS = {
    "5b": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "video_model_family": "wan22_ti2v_5b",
        "video_hidden_dim": 3072,
        "video_latent_dim": 48,
        "video_dit_input_dim": 48,
        "video_dit_output_dim": 48,
        "dit_role": "single",
        "teacher_dit_role": "single",
        "student_dit_role": "single",
        "i2v_condition_mode": "none",
        "teacher_i2v_condition_mode": "none",
        "student_i2v_condition_mode": "none",
        "student_input_mode": "flow_noised_video_first_frame_clean",
        "teacher_frame0_mode": "clean_first_frame",
    },
    "14b": {
        "model_id": "Wan-AI/Wan2.2-I2V-A14B",
        "video_model_family": "wan22_i2v_a14b",
        "video_hidden_dim": 5120,
        "video_latent_dim": 16,
        "video_dit_input_dim": 36,
        "video_dit_output_dim": 16,
        "dit_role": "low_noise",
        "teacher_dit_role": "low_noise",
        "student_dit_role": "low_noise",
        "i2v_condition_mode": "first_image_mask_condition",
        "teacher_i2v_condition_mode": "first_image_mask_condition",
        "student_i2v_condition_mode": "first_image_mask_condition",
        "student_input_mode": "wan22_i2v_clean_first_frame",
        "teacher_frame0_mode": "native_noised",
        "teacher_sigma_min": 0.7,
        "teacher_sigma_max": 0.7,
        "student_sigma_policy": "wan_shift_full",
    },
}
DATA_DEFAULTS = {
    "robotwin": dict(
        height=384,
        width=320,
        action_dim=14,
        proprio_dim=14,
        concat_multi_camera="robotwin",
        camera_keys="observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist",
    ),
    "libero": dict(
        height=224,
        width=448,
        action_dim=7,
        proprio_dim=8,
        concat_multi_camera="horizontal",
        camera_keys="observation.images.image,observation.images.wrist_image",
    ),
    "robocasa": dict(
        height=352,
        width=352,
        action_dim=7,
        proprio_dim=9,
        concat_multi_camera="robocasa_dreamzero_352",
        camera_keys="observation.images.agentview_left,observation.images.agentview_right,observation.images.wrist",
    ),
    "droid": dict(
        height=352,
        width=640,
        action_dim=8,
        proprio_dim=8,
        action_horizon=32,
        concat_multi_camera="droid_grid",
        camera_keys="observation.images.exterior_image_1_left,observation.images.exterior_image_2_left,observation.images.wrist_image_left",
    ),
}


def config_type(stage: int):
    if stage not in (0, 1, 2):
        raise ValueError(f"Stage must be 0, 1, or 2; got {stage}.")
    module = importlib.import_module(f"dreamfree.training.stage{stage}.config")
    return getattr(module, "Stage2Config" if stage == 2 else f"Stage{stage}Config")


def _check_type(value, expected, key):
    origin = get_origin(expected)
    if origin in (Union, types.UnionType):
        for candidate in get_args(expected):
            try:
                return _check_type(value, candidate, key)
            except TypeError:
                pass
        raise TypeError(f"{key}: expected {expected}, got {value!r}.")
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{key}: expected a list.")
        return [_check_type(item, get_args(expected)[0], key) for item in value]
    if expected is float and type(value) in (int, float):
        return float(value)
    if type(value) is not expected:
        raise TypeError(f"{key}: expected {expected}, got {value!r}.")
    return value


def load_config(stage: int, path: str | Path, overrides=()):
    """Resolve model/dataset defaults, YAML, then explicit key=value overrides."""
    document = OmegaConf.load(path)
    merged = OmegaConf.merge(document, OmegaConf.from_dotlist(list(overrides)))
    supplied = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(supplied, dict):
        raise ValueError("Training configuration must be a mapping.")
    model = supplied.pop("model", None)
    dataset = supplied.pop("dataset", None)
    mode = supplied.pop("data_mode", "live")
    if model not in MODEL_DEFAULTS:
        raise ValueError(f"model must be '5b' or '14b'; got {model!r}.")
    if dataset not in DATA_DEFAULTS:
        raise ValueError(f"dataset must be one of {tuple(DATA_DEFAULTS)}; got {dataset!r}.")
    if mode not in ("live", "cached"):
        raise ValueError("data_mode must be 'live' or 'cached'.")
    cls = config_type(stage)
    hints = get_type_hints(cls)
    unknown = supplied.keys() - hints.keys()
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    defaults = dict(
        model_base_path="./assets/models",
        dataset_name=dataset,
        dataset_root="",
        dataset_dirs="",
        dataset_format="lerobot_root",
        extract_archives=False,
        use_cached_latents=mode == "cached",
        text_embedding_cache_dir="",
        stage0_lora_checkpoint="",
        stage0_eval_reference=None,
        stage0_expected_global_step=None,
        pretrained_norm_stats=None,
        action_dit_pretrained_path="",
        output_dir=f"./outputs/stage{stage}/{dataset}_{model}",
        final_export_dir=f"./outputs/stage{stage}/{dataset}_{model}/merged",
        wandb_project="dreamfree",
        wandb_name=f"stage{stage}_{dataset}_{model}",
    )
    defaults.update(MODEL_DEFAULTS[model])
    defaults.update(DATA_DEFAULTS[dataset])
    defaults.update(
        stage0_expected_model_id=defaults["model_id"],
        stage0_expected_video_model_family=defaults["video_model_family"],
        stage0_expected_dit_role=defaults["dit_role"],
    )
    values = dataclasses.asdict(cls())
    values.update({key: value for (key, value) in defaults.items() if key in hints})
    values.update(supplied)
    if "use_cached_latents" in supplied and supplied["use_cached_latents"] != (mode == "cached"):
        raise ValueError("use_cached_latents conflicts with data_mode.")
    for key, value in list(values.items()):
        if value is None and key in ("pretrained_norm_stats", "stage0_eval_reference"):
            continue
        values[key] = _check_type(value, hints[key], key)
    required = {
        "strict_dataset_errors": True,
        "require_text_cache": True,
        "use_shared_teacher_student_dit": True,
        "teacher_target_token_scope": "all",
        "student_train_mode": "frozen",
        "student_init_mode": "shared_stage0_merged_dit_with_stage1_lora",
        "readout_type": "latent_query_cross_attention",
        "readout_active_query_budgets": None,
        "readout_eval_num_queries": None,
    }
    for key, expected in required.items():
        if key in values and values[key] != expected:
            raise ValueError(f"{key} must be {expected!r} for the standard training workflow.")
    return cls(**values)


def train_main(stage: int, argv=None):
    parser = argparse.ArgumentParser(description=f"Train DreamFree stage {stage}.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(stage, args.config, args.set)
    if args.print_config:
        print(json.dumps(dataclasses.asdict(cfg), indent=2))
        return
    if not cfg.dataset_dirs:
        raise ValueError("dataset_dirs must name the LeRobot dataset root(s).")
    if not cfg.text_embedding_cache_dir:
        raise ValueError("text_embedding_cache_dir is required; run text preparation first.")
    if stage == 1 and (not cfg.stage0_lora_checkpoint):
        raise ValueError("stage0_lora_checkpoint is required for stage 1.")
    if stage == 2 and (not cfg.stage1_checkpoint):
        raise ValueError("stage1_checkpoint is required for stage 2.")
    module = importlib.import_module(f"dreamfree.training.stage{stage}.loop")
    run = getattr(
        module, "run_dreamfree_stage2_training" if stage == 2 else f"run_stage{stage}_training"
    )
    run(cfg)
