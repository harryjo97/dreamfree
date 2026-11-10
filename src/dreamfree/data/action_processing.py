"""Stage 2 dataset geometry, action/state processors, and normalization statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torchvision.transforms as TV

from dreamfree.data import yam
from dreamfree.data.lerobot.lerobot.datasets.compute_stats import aggregate_stats
from dreamfree.data.lerobot.lerobot.lerobot_dataset import (
    LeRobotDatasetMetadata,
)
from dreamfree.data.lerobot.processors.robot import DreamFreeProcessor
from dreamfree.data.lerobot.transforms.action_state_merger import ConcatLeftAlign
from dreamfree.data.lerobot.transforms.image import ToTensor
from dreamfree.data.lerobot.utils.normalizer import load_dataset_stats_from_json
from dreamfree.training.stage0.constants import (
    DROID_GRID_HEIGHT,
    DROID_GRID_WIDTH,
)
from dreamfree.training.stage2.config import Stage2Config

ROBOTWIN_CONCAT_MULTI_CAMERA = "robotwin"
HORIZONTAL_CONCAT_MULTI_CAMERA = "horizontal"
ROBOCASA_DREAMZERO_352_CONCAT_MULTI_CAMERA = "robocasa_dreamzero_352"
ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA = "robocasa_dreamzero_448"
ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA = ROBOCASA_DREAMZERO_352_CONCAT_MULTI_CAMERA
ROBOCASA_STAGE2_DATASET_ALIASES = frozenset({"robocasa"})
ROBOCASA_STAGE2_448_DATASET_ALIASES = frozenset(("robocasa_cosmos_success_448",))
DROID_STAGE2_DATASET_ALIASES = frozenset({"droid"})
ROBOCASA_STAGE2_HEIGHT = 352
ROBOCASA_STAGE2_WIDTH = 352
ROBOCASA_STAGE2_448_HEIGHT = 448
ROBOCASA_STAGE2_448_WIDTH = 448
ROBOCASA_STAGE2_ACTION_DIM = 7
ROBOCASA_STAGE2_PROPRIO_DIM = 9
DROID_STAGE2_HEIGHT = DROID_GRID_HEIGHT
DROID_STAGE2_WIDTH = DROID_GRID_WIDTH
DROID_STAGE2_ACTION_DIM = 8
DROID_STAGE2_PROPRIO_DIM = 8
DROID_STAGE2_ACTION_HORIZON = 24
DROID_DREAMZERO_ACTION_CONTRACT = "droid_dreamzero_rel_joint_abs_gripper_q99_v1"
DROID_DREAMZERO_RELATIVE_STATS_FILE = "relative_stats_dreamzero.json"
DROID_JOINT_POSITION_KEY = "joint_position"
DROID_GRIPPER_POSITION_KEY = "gripper_position"


def _parse_dataset_dirs(value: str) -> list[str]:
    dirs = [part.strip() for part in str(value).split(",") if part.strip()]
    if not dirs:
        raise ValueError("`dataset_dirs` must contain at least one directory.")
    return dirs


def _normalize_stage2_dataset_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _is_stage2_robotwin_dataset_name(value):
    return value == "robotwin"


def _is_stage2_robocasa_dataset_name(value):
    return value == "robocasa"


def _stage2_robocasa_layout_spec(dataset_name: Any) -> dict[str, int | str]:
    name = _normalize_stage2_dataset_name(dataset_name)
    if name in ROBOCASA_STAGE2_448_DATASET_ALIASES:
        return {
            "camera_layout": ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA,
            "height": ROBOCASA_STAGE2_448_HEIGHT,
            "width": ROBOCASA_STAGE2_448_WIDTH,
            "action_dim": ROBOCASA_STAGE2_ACTION_DIM,
            "proprio_dim": ROBOCASA_STAGE2_PROPRIO_DIM,
        }
    return {
        "camera_layout": ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA,
        "height": ROBOCASA_STAGE2_HEIGHT,
        "width": ROBOCASA_STAGE2_WIDTH,
        "action_dim": ROBOCASA_STAGE2_ACTION_DIM,
        "proprio_dim": ROBOCASA_STAGE2_PROPRIO_DIM,
    }


def _is_stage2_droid_dataset_name(value):
    return value == "droid"


def _parse_optional_stage2_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if text in {"", "null", "None", "none"}:
        return None
    return int(text)


def resolve_stage2_action_horizon(cfg: Any) -> int:
    explicit_horizon = _parse_optional_stage2_int(getattr(cfg, "action_horizon", None))
    if explicit_horizon is not None:
        if explicit_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {explicit_horizon}.")
        return int(explicit_horizon)
    if _is_stage2_droid_dataset_name(getattr(cfg, "dataset_name", "")):
        return DROID_STAGE2_ACTION_HORIZON
    return int(getattr(cfg, "num_frames")) - 1


def _is_stage2_libero_dataset_name(value):
    return value == "libero"


def _build_robotwin_shape_meta(cfg: Stage2Config) -> dict[str, Any]:
    return {
        "images": [
            {"key": "cam_high", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]},
            {"key": "cam_left_wrist", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]},
            {"key": "cam_right_wrist", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]},
        ],
        "action": [
            {"key": "default", "raw_shape": int(cfg.action_dim), "shape": int(cfg.action_dim)}
        ],
        "state": [
            {"key": "default", "raw_shape": int(cfg.proprio_dim), "shape": int(cfg.proprio_dim)}
        ],
    }


def _build_libero_shape_meta(cfg: Stage2Config) -> dict[str, Any]:
    return {
        "images": [
            {"key": "image", "raw_shape": [3, 512, 512], "shape": [3, 224, 224]},
            {"key": "wrist_image", "raw_shape": [3, 512, 512], "shape": [3, 224, 224]},
        ],
        "action": [
            {"key": "default", "raw_shape": int(cfg.action_dim), "shape": int(cfg.action_dim)}
        ],
        "state": [
            {"key": "default", "raw_shape": int(cfg.proprio_dim), "shape": int(cfg.proprio_dim)}
        ],
    }


def _build_robocasa_shape_meta(cfg: Stage2Config) -> dict[str, Any]:
    return {
        "images": [
            {"key": "agentview_left", "raw_shape": [3, 224, 224], "shape": [3, 224, 224]},
            {"key": "agentview_right", "raw_shape": [3, 224, 224], "shape": [3, 224, 224]},
            {"key": "wrist", "raw_shape": [3, 224, 224], "shape": [3, 224, 224]},
        ],
        "action": [
            {"key": "default", "raw_shape": int(cfg.action_dim), "shape": int(cfg.action_dim)}
        ],
        "state": [
            {"key": "default", "raw_shape": int(cfg.proprio_dim), "shape": int(cfg.proprio_dim)}
        ],
    }


def _load_droid_modality_config(root: Path) -> dict[str, Any]:
    path = Path(root) / "meta" / "modality.json"
    if not path.exists():
        raise FileNotFoundError(f"DROID dataset requires DreamZero modality metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"DROID modality metadata must be a JSON object: {path}")
    return payload


def _droid_modality_range(root: Path, modality: str, key: str) -> tuple[int, int]:
    payload = _load_droid_modality_config(root)
    modality_payload = payload.get(modality)
    if not isinstance(modality_payload, dict) or key not in modality_payload:
        raise KeyError(
            f"DROID modality metadata missing {modality}.{key} in {Path(root) / 'meta' / 'modality.json'}"
        )
    item = modality_payload[key]
    if not isinstance(item, dict) or "start" not in item or "end" not in item:
        raise ValueError(
            f"DROID modality metadata for {modality}.{key} must define start/end: {item!r}"
        )
    (start, end) = (int(item["start"]), int(item["end"]))
    if start < 0 or end <= start:
        raise ValueError(
            f"Invalid DROID modality range for {modality}.{key}: start={start}, end={end}"
        )
    return (start, end)


def _build_droid_shape_meta(cfg: Stage2Config) -> dict[str, Any]:
    return {
        "images": [
            {"key": "exterior_image_1_left", "raw_shape": [3, 180, 320], "shape": [3, 180, 320]},
            {"key": "exterior_image_2_left", "raw_shape": [3, 180, 320], "shape": [3, 180, 320]},
            {"key": "wrist_image_left", "raw_shape": [3, 180, 320], "shape": [3, 180, 320]},
        ],
        "action": [
            {
                "key": DROID_JOINT_POSITION_KEY,
                "raw_shape": 7,
                "shape": 7,
                "source_key": "action",
                "source_slice": (14, 21),
                "relative_to_state_key": DROID_JOINT_POSITION_KEY,
                "state_source_slice": (7, 14),
            },
            {
                "key": DROID_GRIPPER_POSITION_KEY,
                "raw_shape": 1,
                "shape": 1,
                "source_key": "action",
                "source_slice": (12, 13),
            },
        ],
        "state": [
            {
                "key": DROID_JOINT_POSITION_KEY,
                "raw_shape": 7,
                "shape": 7,
                "source_key": "observation.state",
                "source_slice": (7, 14),
            },
            {
                "key": DROID_GRIPPER_POSITION_KEY,
                "raw_shape": 1,
                "shape": 1,
                "source_key": "observation.state",
                "source_slice": (6, 7),
            },
        ],
    }


def _build_stage2_shape_meta(cfg: Stage2Config) -> dict[str, Any]:
    if cfg.dataset_name in yam.DATASET_NAMES:
        return yam.shape_meta(cfg.dataset_name)
    dataset_name = _normalize_stage2_dataset_name(cfg.dataset_name)
    if _is_stage2_libero_dataset_name(dataset_name):
        return _build_libero_shape_meta(cfg)
    if _is_stage2_robotwin_dataset_name(dataset_name):
        return _build_robotwin_shape_meta(cfg)
    if _is_stage2_robocasa_dataset_name(dataset_name):
        return _build_robocasa_shape_meta(cfg)
    if _is_stage2_droid_dataset_name(dataset_name):
        return _build_droid_shape_meta(cfg)
    raise ValueError(f"Unsupported Stage 2 dataset_name={cfg.dataset_name!r}.")


def _build_robotwin_processor(cfg: Stage2Config) -> DreamFreeProcessor:
    shape_meta = _build_robotwin_shape_meta(cfg)
    resize = TV.Resize(size=[240, 320])
    return DreamFreeProcessor(
        shape_meta=shape_meta,
        num_obs_steps=int(cfg.num_frames),
        num_output_cameras=3,
        action_output_dim=int(cfg.action_dim),
        proprio_output_dim=int(cfg.proprio_dim),
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[ToTensor(), resize],
        val_transforms=[ToTensor(), resize],
        drop_high_level_prob=1.0,
        use_zh_instruction=False,
        tokenizer=None,
        delta_action_dim_mask=None,
    )


def _build_libero_processor(cfg: Stage2Config) -> DreamFreeProcessor:
    shape_meta = _build_libero_shape_meta(cfg)
    resize = TV.Resize(size=[224, 224])
    return DreamFreeProcessor(
        shape_meta=shape_meta,
        num_obs_steps=int(cfg.num_frames),
        num_output_cameras=2,
        action_output_dim=int(cfg.action_dim),
        proprio_output_dim=int(cfg.proprio_dim),
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[ToTensor(), resize],
        val_transforms=[ToTensor(), resize],
        drop_high_level_prob=1.0,
        use_zh_instruction=False,
        tokenizer=None,
        delta_action_dim_mask={"default": [True, True, True, True, True, True, False]},
    )


def _build_robocasa_processor(cfg: Stage2Config) -> DreamFreeProcessor:
    shape_meta = _build_robocasa_shape_meta(cfg)
    resize = TV.Resize(size=[224, 224])
    return DreamFreeProcessor(
        shape_meta=shape_meta,
        num_obs_steps=int(cfg.num_frames),
        num_output_cameras=3,
        action_output_dim=int(cfg.action_dim),
        proprio_output_dim=int(cfg.proprio_dim),
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[ToTensor(), resize],
        val_transforms=[ToTensor(), resize],
        drop_high_level_prob=1.0,
        use_zh_instruction=False,
        tokenizer=None,
        delta_action_dim_mask=None,
    )


def _build_droid_processor(cfg: Stage2Config) -> DreamFreeProcessor:
    shape_meta = _build_droid_shape_meta(cfg)
    return DreamFreeProcessor(
        shape_meta=shape_meta,
        num_obs_steps=int(cfg.num_frames),
        num_output_cameras=3,
        action_output_dim=int(cfg.action_dim),
        proprio_output_dim=int(cfg.proprio_dim),
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="q01/q99",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[ToTensor()],
        val_transforms=[ToTensor()],
        drop_high_level_prob=1.0,
        use_zh_instruction=False,
        tokenizer=None,
        delta_action_dim_mask=None,
    )


def _build_yam_processor(cfg: Stage2Config) -> DreamFreeProcessor:
    shape_meta = yam.shape_meta(cfg.dataset_name)
    return DreamFreeProcessor(
        shape_meta=shape_meta,
        num_obs_steps=int(cfg.num_frames),
        num_output_cameras=3,
        action_output_dim=int(cfg.action_dim),
        proprio_output_dim=int(cfg.proprio_dim),
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="q01/q99",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[ToTensor()],
        val_transforms=[ToTensor()],
        drop_high_level_prob=1.0,
        use_zh_instruction=False,
        tokenizer=None,
        delta_action_dim_mask=None,
    )


def _build_stage2_processor(cfg: Stage2Config) -> DreamFreeProcessor:
    if cfg.dataset_name in yam.DATASET_NAMES:
        return _build_yam_processor(cfg)
    dataset_name = _normalize_stage2_dataset_name(cfg.dataset_name)
    if _is_stage2_libero_dataset_name(dataset_name):
        return _build_libero_processor(cfg)
    if _is_stage2_robotwin_dataset_name(dataset_name):
        return _build_robotwin_processor(cfg)
    if _is_stage2_robocasa_dataset_name(dataset_name):
        return _build_robocasa_processor(cfg)
    if _is_stage2_droid_dataset_name(dataset_name):
        return _build_droid_processor(cfg)
    raise ValueError(f"Unsupported Stage 2 dataset_name={cfg.dataset_name!r}.")


def _load_stage2_dataset_stats(
    *, dataset_stats_path: str | None, dataset_name: str, metas: Sequence[LeRobotDatasetMetadata]
) -> dict[str, Any]:

    def convert_lerobot_stats(
        stats: dict[str, dict[str, Any]], *, action_key: str, state_key: str
    ) -> dict[str, dict[str, dict[str, torch.Tensor]]]:

        def convert_feature_stats(
            source: dict[str, Any], *, feature_key: str
        ) -> dict[str, torch.Tensor]:
            converted: dict[str, torch.Tensor] = {}
            for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
                if stat_name not in source and stat_name in {"min", "max", "mean", "std"}:
                    raise KeyError(
                        f"LeRobot metadata stats for {feature_key!r} are missing {stat_name!r}."
                    )
                if stat_name not in source:
                    continue
                tensor = torch.as_tensor(source[stat_name], dtype=torch.float32)
                converted[f"global_{stat_name}"] = tensor
                converted[f"stepwise_{stat_name}"] = tensor.unsqueeze(0)
            return converted

        def convert_feature(feature_key: str) -> dict[str, torch.Tensor]:
            if feature_key not in stats:
                raise KeyError(
                    f"LeRobot metadata stats are missing required feature {feature_key!r}."
                )
            return convert_feature_stats(stats[feature_key], feature_key=feature_key)

        return {
            "action": {"default": convert_feature(action_key)},
            "state": {"default": convert_feature(state_key)},
        }

    def ensure_feature_count(
        feature_stats: dict[str, Any], *, meta: LeRobotDatasetMetadata, feature_key: str
    ) -> dict[str, Any]:
        normalized = dict(feature_stats)
        if "count" in normalized:
            return normalized
        total_frames = int(getattr(meta, "total_frames", 0))
        if total_frames <= 0:
            raise ValueError(
                f"LeRobot metadata stats for {feature_key!r} in root={meta.root} are missing 'count', and meta/info.json does not provide a positive total_frames value."
            )
        normalized["count"] = np.asarray([total_frames], dtype=np.int64)
        return normalized

    def normalize_libero_stats(meta: LeRobotDatasetMetadata) -> dict[str, dict[str, Any]]:
        stats = meta.stats
        if not stats:
            raise ValueError(
                f"LeRobot metadata stats are empty for LIBERO dataset root={meta.root}."
            )
        if "actions" in stats and "state" in stats:
            return {"action": stats["actions"], "state": stats["state"]}
        if "action" in stats and "observation.state" in stats:
            return {"action": stats["action"], "state": stats["observation.state"]}
        raise KeyError(
            f"LIBERO-family metadata stats for root={meta.root} must contain either ('action', 'observation.state') or ('actions', 'state')."
        )

    def slice_feature_stats(
        source: dict[str, Any], *, feature_key: str, start: int, end: int
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
            if stat_name not in source:
                raise KeyError(
                    f"DROID metadata stats for {feature_key!r} are missing {stat_name!r}."
                )
            values = np.asarray(source[stat_name], dtype=np.float32)
            if values.ndim != 1 or int(values.shape[0]) < end:
                raise ValueError(
                    f"DROID metadata stats for {feature_key!r}.{stat_name} must be 1D with at least {end} values, got shape={values.shape}."
                )
            sliced[stat_name] = values[start:end]
        return sliced

    def normalize_droid_stats(meta: LeRobotDatasetMetadata) -> dict[str, dict[str, Any]]:
        stats = meta.stats
        if not stats:
            raise ValueError(
                f"LeRobot metadata stats are empty for DROID dataset root={meta.root}."
            )
        if "action" not in stats or "observation.state" not in stats:
            raise KeyError(
                f"DROID DreamZero metadata for root={meta.root} must contain ('action', 'observation.state') stats."
            )
        root = Path(meta.root)
        relative_stats_path = root / "meta" / DROID_DREAMZERO_RELATIVE_STATS_FILE
        if not relative_stats_path.exists():
            raise FileNotFoundError(
                f"DROID DreamZero relative action stats not found: {relative_stats_path}"
            )
        with relative_stats_path.open("r", encoding="utf-8") as handle:
            relative_stats = json.load(handle)
        joint_relative_stats = relative_stats.get(DROID_JOINT_POSITION_KEY)
        if not isinstance(joint_relative_stats, dict):
            raise KeyError(
                f"DROID relative stats must contain {DROID_JOINT_POSITION_KEY!r}: {relative_stats_path}"
            )
        (state_joint_start, state_joint_end) = _droid_modality_range(
            root, "state", DROID_JOINT_POSITION_KEY
        )
        (state_gripper_start, state_gripper_end) = _droid_modality_range(
            root, "state", DROID_GRIPPER_POSITION_KEY
        )
        (action_gripper_start, action_gripper_end) = _droid_modality_range(
            root, "action", DROID_GRIPPER_POSITION_KEY
        )
        return {
            "action": {
                DROID_JOINT_POSITION_KEY: joint_relative_stats,
                DROID_GRIPPER_POSITION_KEY: slice_feature_stats(
                    stats["action"],
                    feature_key="action.gripper_position",
                    start=action_gripper_start,
                    end=action_gripper_end,
                ),
            },
            "state": {
                DROID_JOINT_POSITION_KEY: slice_feature_stats(
                    stats["observation.state"],
                    feature_key="state.joint_position",
                    start=state_joint_start,
                    end=state_joint_end,
                ),
                DROID_GRIPPER_POSITION_KEY: slice_feature_stats(
                    stats["observation.state"],
                    feature_key="state.gripper_position",
                    start=state_gripper_start,
                    end=state_gripper_end,
                ),
            },
        }

    def convert_droid_feature_stats(
        source: dict[str, Any], *, feature_key: str
    ) -> dict[str, torch.Tensor]:
        converted: dict[str, torch.Tensor] = {}
        for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
            if stat_name not in source:
                raise KeyError(
                    f"DROID DreamZero stats for {feature_key!r} are missing {stat_name!r}."
                )
            tensor = torch.as_tensor(source[stat_name], dtype=torch.float32)
            converted[f"global_{stat_name}"] = tensor
            converted[f"stepwise_{stat_name}"] = tensor.unsqueeze(0)
        return converted

    def convert_droid_stats(
        stats: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        return {
            "action": {
                DROID_JOINT_POSITION_KEY: convert_droid_feature_stats(
                    stats["action"][DROID_JOINT_POSITION_KEY], feature_key="action.joint_position"
                ),
                DROID_GRIPPER_POSITION_KEY: convert_droid_feature_stats(
                    stats["action"][DROID_GRIPPER_POSITION_KEY],
                    feature_key="action.gripper_position",
                ),
            },
            "state": {
                DROID_JOINT_POSITION_KEY: convert_droid_feature_stats(
                    stats["state"][DROID_JOINT_POSITION_KEY], feature_key="state.joint_position"
                ),
                DROID_GRIPPER_POSITION_KEY: convert_droid_feature_stats(
                    stats["state"][DROID_GRIPPER_POSITION_KEY], feature_key="state.gripper_position"
                ),
            },
        }

    stats_text = "" if dataset_stats_path is None else str(dataset_stats_path).strip()
    if stats_text not in {"", "null", "None"}:
        stats_path = Path(stats_text).expanduser().resolve()
        if not stats_path.exists():
            raise FileNotFoundError(f"Sparse Stage 2 dataset_stats_path not found: {stats_path}")
        return load_dataset_stats_from_json(str(stats_path))
    if _is_stage2_libero_dataset_name(dataset_name):
        unified_stats = [normalize_libero_stats(meta) for meta in metas]
        return convert_lerobot_stats(
            aggregate_stats(unified_stats), action_key="action", state_key="state"
        )
    if _is_stage2_droid_dataset_name(dataset_name):
        if len(metas) != 1:
            raise ValueError(
                f"DROID DreamZero Stage 2 stats require exactly one dataset root, got {len(metas)}."
            )
        return convert_droid_stats(normalize_droid_stats(metas[0]))
    raise ValueError(
        "Sparse Stage 2 dataset requires `dataset_stats_path` for RobotWin/RoboCasa datasets."
    )
