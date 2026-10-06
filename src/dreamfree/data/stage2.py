"""Stage 2 privileged-teacher action pretraining trainer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms as TV
import torchvision.transforms.functional as transforms_F
from omegaconf import OmegaConf
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from dreamfree.data.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from dreamfree.data.lerobot.cached_latent_dataset import (
    DROID_DREAMZERO_OVERLAY_CACHE_KIND,
    DROID_DREAMZERO_REQUIRED_TENSOR_DIMS,
    binarize_droid_gripper_action_history,
    is_droid_dreamzero_runtime_horizon_cache,
    stage2_latent_cache_expected_contract,
    stage2_text_cache_suffix_for_family,
    validate_droid_dreamzero_overlay_cache_file,
    validate_stage2_latent_cache_file,
    validate_stage2_latent_tensor_contract,
)
from dreamfree.data.lerobot.lerobot.datasets.compute_stats import aggregate_stats
from dreamfree.data.lerobot.lerobot.lerobot_dataset import (
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from dreamfree.data.lerobot.processors.robot import DreamFreeProcessor
from dreamfree.data.lerobot.transforms.action_state_merger import ConcatLeftAlign
from dreamfree.data.lerobot.transforms.image import ToTensor
from dreamfree.data.lerobot.utils.normalizer import load_dataset_stats_from_json
from dreamfree.data.robocasa_visual import required_robocasa_visual_preprocess_id
from dreamfree.data.sampling import seed_worker
from dreamfree.data.text_cache import DEFAULT_PROMPT, validate_text_cache
from dreamfree.data.video import resolve_episode_task_candidates
from dreamfree.models.wan22.i2v_condition import WAN22_A14B_I2V_CONDITION_LATENTS_KEY
from dreamfree.training.stage0.constants import (
    DROID_GRID_CONCAT_MULTI_CAMERA,
    DROID_GRID_HEIGHT,
    DROID_GRID_WIDTH,
)
from dreamfree.training.stage2.config import Stage2Config
from dreamfree.utils.logging_config import get_logger

logger = get_logger(__name__)
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
STAGE2_REQUIRED_CACHE_KEYS = ("context", "mask")
DEFAULT_SPLIT_SEED = 42
MAX_GETITEM_ATTEMPT = 5


def _parse_dataset_dirs(value: str) -> list[str]:
    dirs = [part.strip() for part in str(value).split(",") if part.strip()]
    if not dirs:
        raise ValueError("`dataset_dirs` must contain at least one directory.")
    return dirs


def _stage2_metadata_episode_indices(meta: LeRobotDatasetMetadata) -> list[int]:
    if hasattr(meta, "episodes"):
        episode_indices = sorted((int(ep_idx) for ep_idx in meta.episodes))
    else:
        episode_indices = list(range(int(meta.total_episodes)))
    if not episode_indices:
        raise ValueError(
            f"Stage 2 dataset metadata contains no episodes for repo={meta.repo_id} root={getattr(meta, 'root', '<unknown>')}."
        )
    expected_count = int(meta.total_episodes)
    if len(episode_indices) != expected_count:
        raise ValueError(
            f"Stage 2 dataset metadata episode count mismatch for repo={meta.repo_id} root={meta.root}: meta/info.json total_episodes={expected_count}, meta/episodes.jsonl entries={len(episode_indices)}."
        )
    return episode_indices


def _select_stage2_episode_indices_for_split(
    meta: LeRobotDatasetMetadata,
    *,
    split: str,
    val_set_proportion: float,
    test_set_proportion: float,
    seed: int = DEFAULT_SPLIT_SEED,
    allowed_episode_indices: Sequence[int] | None = None,
) -> list[int]:
    split_name = str(split).strip().lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Stage 2 dataset split must be one of train/val/test, got {split!r}.")
    val_prop = float(val_set_proportion)
    test_prop = float(test_set_proportion)
    if val_prop < 0.0 or test_prop < 0.0:
        raise ValueError(
            f"val_set_proportion and test_set_proportion must be non-negative, got val={val_set_proportion} test={test_set_proportion}."
        )
    if val_prop + test_prop >= 1.0:
        raise ValueError(
            f"val_set_proportion + test_set_proportion must be < 1.0, got {val_prop} + {test_prop}."
        )
    known_episode_indices = _stage2_metadata_episode_indices(meta)
    if allowed_episode_indices is None:
        episode_indices = known_episode_indices
    else:
        episode_indices = [int(episode_index) for episode_index in allowed_episode_indices]
        if not episode_indices:
            raise ValueError(f"Stage 2 allowed episode set is empty for repo={meta.repo_id}.")
        if len(episode_indices) != len(set(episode_indices)):
            raise ValueError(
                f"Stage 2 allowed episode set contains duplicates for repo={meta.repo_id}."
            )
        unknown = sorted(set(episode_indices) - set(known_episode_indices))
        if unknown:
            raise ValueError(
                f"Stage 2 allowed episode set contains indices absent from repo={meta.repo_id}: {unknown[:10]}."
            )
    if val_prop < 1e-06 and test_prop < 1e-06:
        return episode_indices
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_indices)
    total_episodes = len(episode_indices)
    val_count = 0 if val_prop < 1e-06 else total_episodes - int(total_episodes * (1.0 - val_prop))
    test_count = (
        0 if test_prop < 1e-06 else total_episodes - int(total_episodes * (1.0 - test_prop))
    )
    train_count = total_episodes - val_count - test_count
    if train_count < 0:
        raise ValueError(
            f"Episode split counts are invalid for repo={meta.repo_id}: total={total_episodes} train={train_count} val={val_count} test={test_count}."
        )
    if split_name == "train":
        return episode_indices[:train_count]
    if split_name == "val":
        return episode_indices[train_count : train_count + val_count]
    return episode_indices[train_count + val_count :]


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
            "concat_multi_camera": ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA,
            "height": ROBOCASA_STAGE2_448_HEIGHT,
            "width": ROBOCASA_STAGE2_448_WIDTH,
            "action_dim": ROBOCASA_STAGE2_ACTION_DIM,
            "proprio_dim": ROBOCASA_STAGE2_PROPRIO_DIM,
        }
    return {
        "concat_multi_camera": ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA,
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


def _validate_cache_payload(path: Path, *, context_len: int, text_dim: int) -> None:
    payload = torch.load(path, map_location="cpu")
    validate_text_cache(payload, path)
    if not isinstance(payload, dict):
        raise ValueError(f"Stage 2 text cache must be a dict, got {type(payload)} in {path}")
    missing = [key for key in STAGE2_REQUIRED_CACHE_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Stage 2 text cache is missing keys {missing}: {path}")
    context = payload["context"]
    mask = payload["mask"]
    if not isinstance(context, torch.Tensor) or tuple(context.shape) != (context_len, text_dim):
        raise ValueError(
            f"Cached `context` must have shape ({context_len}, {text_dim}), got {tuple(getattr(context, 'shape', ()))} in {path}"
        )
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (context_len,):
        raise ValueError(
            f"Cached `mask` must have shape ({context_len},), got {tuple(getattr(mask, 'shape', ()))} in {path}"
        )


def _stage2_latent_cache_expected_contract(cfg: Any) -> dict[str, Any]:
    contract = stage2_latent_cache_expected_contract(
        video_model_family=str(getattr(cfg, "video_model_family", "wan22_ti2v_5b")),
        model_id=getattr(cfg, "model_id", None),
        height=int(getattr(cfg, "height")),
        width=int(getattr(cfg, "width")),
        num_frames=int(getattr(cfg, "num_frames")),
        action_video_freq_ratio=int(getattr(cfg, "action_video_freq_ratio")),
        action_horizon=resolve_stage2_action_horizon(cfg),
        require_action_horizon_metadata=False,
    )
    visual_preprocess_id = required_robocasa_visual_preprocess_id(
        dataset_name=getattr(cfg, "dataset_name", ""),
        concat_multi_camera=getattr(cfg, "concat_multi_camera", ""),
        height=getattr(cfg, "height", -1),
        width=getattr(cfg, "width", -1),
        video_model_family=getattr(cfg, "video_model_family", ""),
    )
    if visual_preprocess_id is not None:
        contract["visual_preprocess_id"] = visual_preprocess_id
        contract["concat_layout"] = str(getattr(cfg, "concat_multi_camera"))
        if contract["video_model_family"] == "wan22_i2v_a14b":
            contract["require_i2v_first_latent_equality"] = True
    return contract


def _validate_stage2_latent_cache_contract(path: Path, cfg: Any) -> None:
    validate_stage2_latent_cache_file(
        path, expected=_stage2_latent_cache_expected_contract(cfg), stage_label="Stage 2"
    )


def validate_droid_stage2_latent_cache_metadata(
    path: Path, cfg: Any, *, stage_label: str = "Stage 2"
) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
        tensor_shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
    if str(meta.get("overlay_cache_kind", "")).strip() == DROID_DREAMZERO_OVERLAY_CACHE_KIND:
        validate_droid_dreamzero_overlay_cache_file(
            path, expected=_stage2_latent_cache_expected_contract(cfg), stage_label=stage_label
        )
    expected = {
        "concat_layout": DROID_GRID_CONCAT_MULTI_CAMERA,
        "action_dim": str(int(getattr(cfg, "action_dim"))),
        "action_contract": DROID_DREAMZERO_ACTION_CONTRACT,
        "norm_mode": "q01/q99",
    }
    mismatches = [
        f"{key}: actual={meta.get(key)!r} expected={value!r}"
        for (key, value) in expected.items()
        if str(meta.get(key)) != value
    ]
    runtime_horizon_cache = is_droid_dreamzero_runtime_horizon_cache(
        metadata=meta, tensor_shapes=tensor_shapes, sample_count=int(meta.get("num_samples", -1))
    )
    expected_action_horizon = str(resolve_stage2_action_horizon(cfg))
    if (
        not runtime_horizon_cache
        and meta.get("action_horizon") is not None
        and (str(meta.get("action_horizon")) != expected_action_horizon)
    ):
        mismatches.append(
            f"action_horizon: actual={meta.get('action_horizon')!r} expected={expected_action_horizon!r}"
        )
    dataset_kind = str(meta.get("dataset_kind", "")).strip().lower()
    if dataset_kind and dataset_kind != "droid":
        mismatches.append(f"dataset_kind: actual={meta.get('dataset_kind')!r} expected='droid'")
    state_dim = meta.get("state_dim", meta.get("proprio_dim"))
    if str(state_dim) != str(int(getattr(cfg, "proprio_dim"))):
        mismatches.append(
            f"state_dim: actual={state_dim!r} expected={int(getattr(cfg, 'proprio_dim'))!r}"
        )
    for tensor_key, tensor_dim in DROID_DREAMZERO_REQUIRED_TENSOR_DIMS.items():
        shape = tensor_shapes.get(tensor_key)
        if shape is None:
            mismatches.append(f"{tensor_key}: missing")
        elif len(shape) != 2 or int(shape[-1]) != tensor_dim:
            mismatches.append(
                f"{tensor_key}: actual_shape={shape!r} expected_last_dim={tensor_dim}"
            )
    if mismatches:
        raise ValueError(
            f"{stage_label} DROID latent cache metadata mismatch in {path}: "
            + "; ".join(mismatches)
        )


def _validate_stage2_input_latents_contract(
    latents: torch.Tensor, cfg: Any, *, stage_label: str = "Stage 2"
) -> None:
    validate_stage2_latent_tensor_contract(
        latents, expected=_stage2_latent_cache_expected_contract(cfg), stage_label=stage_label
    )


def _stage2_text_cache_suffix(cfg: Any) -> str:
    return stage2_text_cache_suffix_for_family(
        str(getattr(cfg, "video_model_family", "wan22_ti2v_5b"))
    )


def _normalize_torch_device(device: Any) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _maybe_cuda_synchronize(device: Any) -> None:
    torch_device = _normalize_torch_device(device)
    if torch_device is None or torch_device.type != "cuda":
        return
    torch.cuda.synchronize(torch_device)


class _TimingRecorder:
    def __init__(self, *, enabled: bool, device: Any):
        self.enabled = bool(enabled)
        self.device = _normalize_torch_device(device)
        self.values: dict[str, float] = {}

    def record(self, key: str):
        if not self.enabled:
            return nullcontext()
        return _TimedSection(self, key)

    def add(self, key: str, value: float) -> None:
        if not self.enabled:
            return
        self.values[key] = self.values.get(key, 0.0) + float(value)


class _TimedSection:
    def __init__(self, recorder: _TimingRecorder, key: str):
        self.recorder = recorder
        self.key = key
        self.start: float | None = None

    def __enter__(self) -> None:
        _maybe_cuda_synchronize(self.recorder.device)
        self.start = time.perf_counter()
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        _maybe_cuda_synchronize(self.recorder.device)
        end = time.perf_counter()
        if self.start is not None:
            self.recorder.add(self.key, end - self.start)
        return False


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


def _build_stage2_processor(cfg: Stage2Config) -> DreamFreeProcessor:
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
    *, pretrained_norm_stats: str | None, dataset_name: str, metas: Sequence[LeRobotDatasetMetadata]
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

    stats_text = "" if pretrained_norm_stats is None else str(pretrained_norm_stats).strip()
    if stats_text not in {"", "null", "None"}:
        stats_path = Path(stats_text).expanduser().resolve()
        if not stats_path.exists():
            raise FileNotFoundError(f"Sparse Stage 2 pretrained_norm_stats not found: {stats_path}")
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
        "Sparse Stage 2 dataset requires `pretrained_norm_stats` for RobotWin/RoboCasa datasets."
    )


def _require_sparse_stage2_video_backend() -> str:
    if importlib.util.find_spec("torchcodec") is None:
        raise RuntimeError(
            "Sparse Stage 2 requires video_backend='torchcodec', but the `torchcodec` package is not available. Install `torchcodec`; falling back to `pyav` is disabled for this trainer."
        )
    return "torchcodec"


class _SparseStage2RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        dataset_dirs: list[str],
        dataset_name: str,
        shape_meta: dict[str, Any],
        num_frames: int,
        action_horizon: int,
        action_history_horizon: int,
        video_size: list[int],
        processor: Any,
        text_embedding_cache_dir: str,
        text_cache_model_suffix: str,
        context_len: int,
        pretrained_norm_stats: str,
        val_set_proportion: float,
        test_set_proportion: float,
        split: str,
        is_training_set: bool,
        global_sample_stride: int,
        action_video_freq_ratio: int,
        skip_padding_as_possible: bool,
        max_padding_retry: int,
        concat_multi_camera: str,
        override_instruction: Optional[str],
        strict_loading: bool,
        strict_errors: bool,
        video_roots: Optional[dict[str, str]],
        split_seed: int = DEFAULT_SPLIT_SEED,
        allow_mixed_fps_for_cached_latents: bool = False,
        metadata_only_for_cached_latents: bool = False,
    ) -> None:
        if processor is None:
            raise ValueError("Sparse Stage 2 dataset requires a processor instance.")
        if OmegaConf.is_config(shape_meta):
            resolved_shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        elif isinstance(shape_meta, dict):
            resolved_shape_meta = shape_meta
        else:
            raise TypeError(f"`shape_meta` must be dict-like or OmegaConf, got {type(shape_meta)}")
        self.dataset_name = str(dataset_name).strip().lower()
        self.split = str(split).strip().lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Stage 2 dataset split must be one of train/val/test, got {split!r}.")
        if bool(is_training_set) != (self.split == "train"):
            raise ValueError(
                f"Stage 2 dataset split/is_training_set mismatch: split={self.split!r} is_training_set={is_training_set}."
            )
        self.shape_meta = resolved_shape_meta
        self.image_meta = [dict(meta) for meta in resolved_shape_meta["images"]]
        self.state_meta = [dict(meta) for meta in resolved_shape_meta["state"]]
        self.action_meta = [dict(meta) for meta in resolved_shape_meta["action"]]
        self.num_frames = int(num_frames)
        self.action_horizon = int(action_horizon)
        self.action_history_horizon = int(action_history_horizon)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                f"num_frames-1 must be divisible by action_video_freq_ratio, got {self.num_frames - 1} and {self.action_video_freq_ratio}."
            )
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}.")
        if self.action_history_horizon < 0:
            raise ValueError(
                f"action_history_horizon must be non-negative, got {self.action_history_horizon}."
            )
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))
        self.video_num_frames = len(self.video_sample_indices)
        if self.video_num_frames % 4 != 1:
            raise ValueError(
                f"Sampled sparse video frames must satisfy Wan temporal constraint `T % 4 == 1`, got {self.video_num_frames}."
            )
        video_transitions = self.video_num_frames - 1
        if video_transitions <= 0:
            raise ValueError("Sampled sparse video must contain at least two frames.")
        if self.action_horizon % video_transitions != 0:
            raise ValueError(
                f"action_horizon must be divisible by sparse video transitions; got action_horizon={self.action_horizon} and video_transitions={video_transitions}."
            )
        self.video_size = [int(video_size[0]), int(video_size[1])]
        self.context_len = int(context_len)
        self.text_embedding_cache_dir = str(text_embedding_cache_dir)
        self.text_embedding_cache_dirs = _parse_path_list(
            text_embedding_cache_dir, label="Stage 2 text_embedding_cache_dir"
        )
        self.text_cache_model_suffix = str(text_cache_model_suffix)
        self.is_training_set = bool(is_training_set)
        self.skip_padding_as_possible = bool(skip_padding_as_possible)
        self.max_padding_retry = int(max_padding_retry)
        self.concat_multi_camera = str(concat_multi_camera)
        self.override_instruction = override_instruction
        self.strict_loading = bool(strict_loading)
        self.strict_errors = bool(strict_errors)
        self.global_sample_stride = int(global_sample_stride)
        self.video_backend = _require_sparse_stage2_video_backend()
        self.metadata_only_for_cached_latents = bool(metadata_only_for_cached_latents)
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            metas.append(LeRobotDatasetMetadata(repo_id=ds_dir, root=ds_root))
        if _is_stage2_droid_dataset_name(self.dataset_name):
            if len(metas) != 1:
                raise ValueError(
                    f"DROID Stage 2 dataset requires exactly one dataset root, got {len(metas)}."
                )
            self._apply_droid_modality_ranges(Path(metas[0].root))
        self.processor = processor.train() if self.is_training_set else processor.eval()
        dataset_stats = _load_stage2_dataset_stats(
            pretrained_norm_stats=pretrained_norm_stats, dataset_name=self.dataset_name, metas=metas
        )
        self.processor.set_normalizer_from_stats(dataset_stats)
        self.image_transforms = (
            self.processor.train_transforms
            if self.is_training_set
            else self.processor.val_transforms
        )
        if self.image_transforms is None:
            raise ValueError("Sparse Stage 2 dataset requires image transforms on the processor.")
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})
        fps_list = [meta.fps for meta in metas]
        if len(set(fps_list)) != 1:
            if not bool(allow_mixed_fps_for_cached_latents):
                raise ValueError(f"All dataset_dirs must have the same fps, got {fps_list}")
            logger.info(
                "Sparse Stage 2 metadata dataset allows mixed fps=%s because full cached latents are required.",
                fps_list,
            )
        self.fps = fps_list[0]
        self.delta_timestamps = self._build_delta_timestamps()
        episodes = self._build_episode_split(
            metas=metas,
            val_set_proportion=float(val_set_proportion),
            test_set_proportion=float(test_set_proportion),
            seed=int(split_seed),
        )
        multi_dataset_kwargs = {
            "dataset_dirs": list(dataset_dirs),
            "episodes": episodes,
            "delta_timestamps": None
            if self.metadata_only_for_cached_latents
            else self.delta_timestamps,
            "video_backend": self.video_backend,
        }
        if video_roots is not None:
            multi_dataset_kwargs["video_roots"] = video_roots
        self.multi_dataset = MultiLeRobotDataset(**multi_dataset_kwargs)
        self.multi_dataset.set_during_training(True)
        self._episode_task_candidates = self._build_episode_task_candidates()

    @staticmethod
    def _scalar_int(value: Any, *, default: int | None = None) -> int | None:
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            value = value.item() if value.numel() == 1 else value.flatten()[0].item()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.item() if value.size == 1 else value.reshape(-1)[0].item()
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _build_episode_task_candidates(self) -> dict[tuple[int, int], tuple[str, ...]]:
        candidates_by_episode: dict[tuple[int, int], tuple[str, ...]] = {}
        for dataset_idx, raw_dataset in enumerate(self.multi_dataset._datasets):
            task_by_index = dict(raw_dataset.meta.tasks)
            episode_rows = getattr(raw_dataset.meta, "episodes", {}) or {}
            selected_episodes = raw_dataset.episodes
            if selected_episodes is None:
                selected_episodes = sorted((int(ep_idx) for ep_idx in episode_rows))
            for ep_idx in selected_episodes:
                ep_idx_int = int(ep_idx)
                row = episode_rows.get(ep_idx_int)
                if row is None:
                    row = episode_rows.get(str(ep_idx_int), {})
                candidates = resolve_episode_task_candidates(
                    row=dict(row or {}),
                    task_by_index=task_by_index,
                    default_task=None,
                    strict=False,
                    stage_label="Stage 2",
                    task_root=getattr(raw_dataset, "root", None),
                )
                if candidates:
                    candidates_by_episode[dataset_idx, ep_idx_int] = candidates
        return candidates_by_episode

    def _sample_instruction(self, lerobot_sample: dict[str, Any]) -> str:
        if self.override_instruction is not None:
            return str(self.override_instruction)
        dataset_idx = self._scalar_int(lerobot_sample.get("dataset_index"), default=0)
        episode_idx = self._scalar_int(lerobot_sample.get("episode_index"), default=None)
        if dataset_idx is not None and episode_idx is not None:
            candidates = self._episode_task_candidates.get((dataset_idx, episode_idx))
            if candidates:
                sample = dict(lerobot_sample)
                sample["task"] = random.choice(candidates)
                return self.processor.augment_instruction(sample)
        return self.processor.augment_instruction(lerobot_sample)

    def _apply_droid_modality_ranges(self, root: Path) -> None:
        state_joint = _droid_modality_range(root, "state", DROID_JOINT_POSITION_KEY)
        state_gripper = _droid_modality_range(root, "state", DROID_GRIPPER_POSITION_KEY)
        action_joint = _droid_modality_range(root, "action", DROID_JOINT_POSITION_KEY)
        action_gripper = _droid_modality_range(root, "action", DROID_GRIPPER_POSITION_KEY)
        state_ranges = {
            DROID_JOINT_POSITION_KEY: state_joint,
            DROID_GRIPPER_POSITION_KEY: state_gripper,
        }
        action_ranges = {
            DROID_JOINT_POSITION_KEY: action_joint,
            DROID_GRIPPER_POSITION_KEY: action_gripper,
        }
        for meta in self.state_meta:
            meta["source_slice"] = state_ranges[meta["key"]]
            meta["source_key"] = "observation.state"
        for meta in self.action_meta:
            meta["source_slice"] = action_ranges[meta["key"]]
            meta["source_key"] = "action"

    def _build_delta_timestamps(self) -> dict[str, list[float]]:
        delta_timestamps: dict[str, list[float]] = {}
        is_droid = _is_stage2_droid_dataset_name(self.dataset_name)
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = (
                f"observation.images.{key}" if key != "default" else "observation.images"
            )
            delta_timestamps[meta["lerobot_key"]] = [
                t * self.action_video_freq_ratio * self.global_sample_stride / self.fps
                for t in range(self.video_num_frames)
            ]
        for meta in self.state_meta:
            key = meta["key"]
            if is_droid:
                meta["lerobot_key"] = str(meta.get("source_key", "observation.state"))
            else:
                meta["lerobot_key"] = (
                    f"observation.state.{key}" if key != "default" else "observation.state"
                )
            delta_timestamps[meta["lerobot_key"]] = [
                t * self.global_sample_stride / self.fps for t in range(self.action_horizon)
            ]
        for meta in self.action_meta:
            key = meta["key"]
            if is_droid:
                meta["lerobot_key"] = str(meta.get("source_key", "action"))
            else:
                meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"
            delta_timestamps[meta["lerobot_key"]] = [
                t * self.global_sample_stride / self.fps
                for t in range(-self.action_history_horizon, self.action_horizon)
            ]
        return delta_timestamps

    def _build_episode_split(
        self,
        *,
        metas: list[LeRobotDatasetMetadata],
        val_set_proportion: float,
        test_set_proportion: float,
        seed: int,
    ) -> dict[str, list[int]]:
        episodes: dict[str, list[int]] = {}
        for meta in metas:
            allowed_episode_indices = None
            selected = _select_stage2_episode_indices_for_split(
                meta,
                split=self.split,
                val_set_proportion=val_set_proportion,
                test_set_proportion=test_set_proportion,
                seed=seed,
                allowed_episode_indices=allowed_episode_indices,
            )
            episodes[meta.repo_id] = selected
        return episodes

    def __len__(self) -> int:
        return self.multi_dataset.num_frames

    def _get_action(
        self,
        meta: dict[str, Any],
        lerobot_sample: dict[str, Any],
        state_by_key: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action = lerobot_sample[meta["lerobot_key"]]
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        source_slice = meta.get("source_slice")
        if source_slice is not None:
            (start, end) = (int(source_slice[0]), int(source_slice[1]))
            if int(action.shape[-1]) < end:
                raise ValueError(
                    f"Action source '{meta['lerobot_key']}' shape {action.shape[-1]} is too small for slice [{start}:{end}]."
                )
            action = action[..., start:end]
        relative_to = meta.get("relative_to_state_key")
        if relative_to is not None:
            state = state_by_key.get(str(relative_to))
            if state is None:
                raise KeyError(
                    f"Action '{meta['key']}' requires missing state key {relative_to!r} for relative DROID action."
                )
            if state.ndim != 2 or int(state.shape[-1]) != int(action.shape[-1]):
                raise ValueError(
                    f"Relative DROID action '{meta['key']}' state shape {tuple(state.shape)} does not match action shape {tuple(action.shape)}."
                )
            action = action - state[0].unsqueeze(0)
        if int(action.shape[-1]) != int(meta["raw_shape"]):
            raise ValueError(
                f"Action '{meta['key']}' shape {action.shape[-1]} mismatch with meta {meta['raw_shape']}."
            )
        return action.float()

    def _get_state(self, meta: dict[str, Any], lerobot_sample: dict[str, Any]) -> torch.Tensor:
        state = lerobot_sample[meta["lerobot_key"]]
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        source_slice = meta.get("source_slice")
        if source_slice is not None:
            (start, end) = (int(source_slice[0]), int(source_slice[1]))
            if int(state.shape[-1]) < end:
                raise ValueError(
                    f"State source '{meta['lerobot_key']}' shape {state.shape[-1]} is too small for slice [{start}:{end}]."
                )
            state = state[..., start:end]
        if int(state.shape[-1]) != int(meta["raw_shape"]):
            raise ValueError(
                f"State '{meta['key']}' shape {state.shape[-1]} mismatch with meta {meta['raw_shape']}."
            )
        return state.float()

    def _get_image(self, meta: dict[str, Any], lerobot_sample: dict[str, Any]) -> torch.Tensor:
        image = lerobot_sample[meta["lerobot_key"]]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(
                f"Image '{meta['key']}' must have 4 dimensions [T,C,H,W], got shape {tuple(image.shape)}."
            )
        return (image * 255).to(torch.uint8)

    def _load_lerobot_sample(self, idx: int) -> tuple[int, dict[str, Any]]:
        sample_idx = idx
        if self.strict_loading:
            return (sample_idx, self.multi_dataset[sample_idx])
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                return (sample_idx, self.multi_dataset[sample_idx])
            except Exception as err:
                attempt += 1
                last_exception = err
                logger.warning(
                    "Error loading sparse Stage 2 sample %s (attempt %s). Retrying with a random index. Error: %s",
                    sample_idx,
                    attempt,
                    err,
                )
                sample_idx = np.random.randint(len(self))
                print(traceback.format_exc())
        raise RuntimeError(
            f"Failed to load a valid sparse Stage 2 sample after {MAX_GETITEM_ATTEMPT} attempts for index {idx}."
        ) from last_exception

    def _build_nested_sample(
        self, sample_idx: int, lerobot_sample: dict[str, Any]
    ) -> dict[str, Any]:
        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)
        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample, sample["state"])
        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)
        sample["action_is_pad"] = torch.as_tensor(
            lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"], dtype=torch.bool
        )
        sample["state_is_pad"] = torch.as_tensor(
            lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"], dtype=torch.bool
        )
        sample["image_is_pad"] = torch.as_tensor(
            lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"], dtype=torch.bool
        )
        return sample

    def _apply_image_transforms(self, nested_sample: dict[str, Any]) -> torch.Tensor:
        processed_images = []
        for meta in self.image_meta:
            key = meta["key"]
            image = nested_sample["images"][key]
            transforms = (
                self.image_transforms[key]
                if isinstance(self.image_transforms, dict)
                else self.image_transforms
            )
            for transform in transforms:
                image = transform(image)
            expected_shape = (self.video_num_frames, *meta["shape"])
            if tuple(image.shape) != expected_shape:
                raise ValueError(
                    f"Expected sparse image shape {expected_shape}, got {tuple(image.shape)} after transforms for key {key}."
                )
            processed_images.append(image)
        return torch.stack(processed_images, dim=0)

    @staticmethod
    def _center_crop_resize_video(
        video: torch.Tensor, size: tuple[int, int], *, crop_scale: float
    ) -> torch.Tensor:
        if video.ndim != 4:
            raise ValueError(f"video must have shape [T,C,H,W], got {tuple(video.shape)}.")
        (_, _, height, width) = video.shape
        crop_h = max(1, min(height, int(round(height * float(crop_scale)))))
        crop_w = max(1, min(width, int(round(width * float(crop_scale)))))
        top = max((height - crop_h) // 2, 0)
        left = max((width - crop_w) // 2, 0)
        cropped = video[..., top : top + crop_h, left : left + crop_w]
        return transforms_F.resize(
            cropped,
            size=list(size),
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _compose_droid_grid(self, video: torch.Tensor) -> torch.Tensor:
        if int(video.shape[0]) != 3:
            raise ValueError(
                f"`concat_multi_camera='{DROID_GRID_CONCAT_MULTI_CAMERA}'` requires exactly 3 cameras, got {int(video.shape[0])}."
            )
        if [int(v) for v in self.video_size] != [DROID_GRID_HEIGHT, DROID_GRID_WIDTH]:
            raise ValueError(
                f"`concat_multi_camera='{DROID_GRID_CONCAT_MULTI_CAMERA}'` requires video_size={[DROID_GRID_HEIGHT, DROID_GRID_WIDTH]}, got {self.video_size}."
            )
        per_camera_h = DROID_GRID_HEIGHT // 2
        per_camera_w = DROID_GRID_WIDTH // 2
        left = self._center_crop_resize_video(
            video[0], (per_camera_h, per_camera_w), crop_scale=0.95
        )
        right = self._center_crop_resize_video(
            video[1], (per_camera_h, per_camera_w), crop_scale=0.95
        )
        wrist = self._center_crop_resize_video(
            video[2], (per_camera_h, per_camera_w), crop_scale=0.95
        )
        wrist_wide = wrist.repeat_interleave(2, dim=-1)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([wrist_wide, bottom], dim=-2)

    def _compose_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(
                f"Sparse Stage 2 `pixel_values` must have shape [num_cameras,T,C,H,W], got {tuple(pixel_values.shape)}."
            )
        (num_cameras, video_t, channels, height, width) = pixel_values.shape
        video = pixel_values.view(num_cameras, video_t, channels, height, width)
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}."
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif self.concat_multi_camera in {
            ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA,
            ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA,
        }:
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='{self.concat_multi_camera}'` requires exactly 3 cameras, got {num_cameras}."
                )
            tile = (
                224
                if self.concat_multi_camera == ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA
                else 176
            )
            full_width = tile * 2
            cam_left = transforms_F.resize(
                video[0],
                size=[tile, tile],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[1],
                size=[tile, tile],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_wrist = transforms_F.resize(
                video[2],
                size=[tile, full_width],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_wrist, bottom], dim=-2)
        elif self.concat_multi_camera == DROID_GRID_CONCAT_MULTI_CAMERA:
            video = self._compose_droid_grid(video)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. Expected one of: horizontal, vertical, robotwin, robocasa_dreamzero_352, robocasa_dreamzero_448, droid_grid."
                )
        else:
            video = video.squeeze(0)
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        return video.permute(1, 0, 2, 3)

    def _normalize_action_state(
        self, nested_sample: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = nested_sample["action"]
        if _is_stage2_droid_dataset_name(self.dataset_name) and self.action_history_horizon > 0:
            if not isinstance(action, dict):
                raise TypeError(
                    f"DROID action-history samples require an action feature mapping, got {type(action).__name__}."
                )
            gripper = action.get(DROID_GRIPPER_POSITION_KEY)
            if gripper is None:
                raise KeyError("DROID action-history samples are missing action.gripper_position.")
            action = dict(action)
            action[DROID_GRIPPER_POSITION_KEY] = binarize_droid_gripper_action_history(
                gripper, action_history_horizon=self.action_history_horizon
            )
        data = {
            "action": action,
            "state": nested_sample["state"],
            "action_is_pad": nested_sample["action_is_pad"],
            "state_is_pad": nested_sample["state_is_pad"],
            "idx": nested_sample["idx"],
        }
        if self.processor.delta_action_dim_mask is not None:
            action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
            if bool(action_is_pad.any().item()):
                for key, dim_mask in self.processor.delta_action_dim_mask.items():
                    cur_action = data["action"][key]
                    cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                    cur_dim_mask = dim_mask.to(device=cur_action.device)
                    pad_delta_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                    cur_action[pad_delta_mask] = 0.0
        data = self.processor.action_state_transform(data)
        data = self.processor.normalizer.forward(data)
        data = self.processor.action_state_merger.forward(data)
        return (data["action"], data["state"], data["action_is_pad"], data["state_is_pad"])

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_paths = [
            cache_dir / f"{hashed}.t5_len{self.context_len}.{self.text_cache_model_suffix}.pt"
            for cache_dir in self.text_embedding_cache_dirs
        ]
        cache_path = next((path for path in cache_paths if path.exists()), cache_paths[0])
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache. Tried: {cache_paths}. Run dreamfree.data.prepare_text first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        validate_text_cache(payload, cache_path)
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        return (context, context_mask)

    def _get(self, idx: int) -> dict[str, Any]:
        sample_idx = idx
        nested_sample = None
        lerobot_sample = None
        for attempt in range(self.max_padding_retry + 1):
            (sample_idx, lerobot_sample) = self._load_lerobot_sample(sample_idx)
            nested_sample = self._build_nested_sample(sample_idx, lerobot_sample)
            if not self.skip_padding_as_possible:
                break
            has_pad = False
            for key in ("action_is_pad", "state_is_pad", "image_is_pad"):
                if bool(nested_sample[key].any().item()):
                    has_pad = True
            if not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = np.random.randint(len(self))
        if nested_sample is None or lerobot_sample is None:
            raise RuntimeError(f"Failed to assemble sparse Stage 2 sample for idx={idx}.")
        pixel_values = self._apply_image_transforms(nested_sample)
        video = self._compose_video(pixel_values)
        (action_with_history, proprio_full, action_is_pad_with_history, proprio_is_pad) = (
            self._normalize_action_state(nested_sample)
        )
        proprio = proprio_full
        image_is_pad = nested_sample["image_is_pad"]
        expected_action_tokens = self.action_history_horizon + self.action_horizon
        if (
            int(action_with_history.shape[0]) != expected_action_tokens
            or int(proprio.shape[0]) != self.action_horizon
        ):
            raise ValueError(
                f"Sparse Stage 2 action/proprio horizon mismatch: action={tuple(action_with_history.shape)} proprio={tuple(proprio.shape)} expected_action_tokens={expected_action_tokens} expected_horizon={self.action_horizon}."
            )
        if tuple(action_is_pad_with_history.shape) != (expected_action_tokens,) or tuple(
            proprio_is_pad.shape
        ) != (self.action_horizon,):
            raise ValueError(
                f"Sparse Stage 2 pad-mask horizon mismatch: action_is_pad={tuple(action_is_pad_with_history.shape)} proprio_is_pad={tuple(proprio_is_pad.shape)} expected_action_tokens={expected_action_tokens}."
            )
        history_end = self.action_history_horizon
        previous_action = action_with_history[:history_end].clone()
        previous_action_is_pad = action_is_pad_with_history[:history_end].clone()
        action = action_with_history[history_end:]
        action_is_pad = action_is_pad_with_history[history_end:]
        if self.action_history_horizon > 0:
            previous_action[previous_action_is_pad] = 0.0
        if video.shape[1] <= 1:
            raise ValueError(
                f"`video` must have at least 2 frames, got shape {tuple(video.shape)}."
            )
        if int(action.shape[0]) % (int(video.shape[1]) - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}."
            )
        task = self._sample_instruction(lerobot_sample)
        task_text = str(task).strip()
        if not task_text:
            raise ValueError(f"Invalid empty task text: {task!r}")
        instruction = DEFAULT_PROMPT.format(task=task_text)
        (context, context_mask) = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        sample = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
        }
        if self.action_history_horizon > 0:
            sample["previous_action"] = previous_action
            sample["previous_action_is_pad"] = previous_action_is_pad
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._get(idx)


def build_stage2_cached_inputs(
    core: Any,
    sample: dict[str, Any],
    tiled: bool = False,
    *,
    timing_recorder: _TimingRecorder | None = None,
) -> dict[str, Any]:
    required_common = ("context", "context_mask", "action", "proprio")
    has_latents = "input_latents" in sample
    has_video = "video" in sample
    required = required_common
    if not has_latents and (not has_video):
        required = required_common + ("video",)
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"Stage 2 training requires keys {required}; missing={missing}")
    action = sample["action"]
    proprio = sample["proprio"]
    previous_action = sample.get("previous_action")
    previous_action_is_pad = sample.get("previous_action_is_pad")
    context = sample["context"]
    context_mask = sample["context_mask"]
    if not isinstance(action, torch.Tensor) or action.ndim != 3:
        raise ValueError(
            f"`sample['action']` must have shape [B,T,A], got {tuple(getattr(action, 'shape', ()))}."
        )
    if not isinstance(proprio, torch.Tensor) or proprio.ndim != 3:
        raise ValueError(
            f"`sample['proprio']` must have shape [B,T,P], got {tuple(getattr(proprio, 'shape', ()))}."
        )
    if (
        not isinstance(context, torch.Tensor)
        or context.ndim != 3
        or tuple(context.shape[1:]) != (128, 4096)
    ):
        raise ValueError(
            f"`sample['context']` must have shape [B,128,4096], got {tuple(getattr(context, 'shape', ()))}."
        )
    if (
        not isinstance(context_mask, torch.Tensor)
        or context_mask.ndim != 2
        or tuple(context_mask.shape[1:]) != (128,)
    ):
        raise ValueError(
            f"`sample['context_mask']` must have shape [B,128], got {tuple(getattr(context_mask, 'shape', ()))}."
        )
    context_mask_bool = context_mask.to(dtype=torch.bool)
    batch_size = int(action.shape[0])
    if int(context.shape[0]) != batch_size or int(context_mask.shape[0]) != batch_size:
        raise ValueError(
            f"Stage 2 cached context batch mismatch: action={batch_size}, context={context.shape[0]}, mask={context_mask.shape[0]}."
        )
    if int(proprio.shape[0]) != batch_size:
        raise ValueError(
            f"Stage 2 batch mismatch: action={batch_size}, proprio={proprio.shape[0]}."
        )
    if (previous_action is None) != (previous_action_is_pad is None):
        raise ValueError(
            "Stage 2 action history requires both `previous_action` and `previous_action_is_pad`, or neither."
        )
    if previous_action is not None:
        if not isinstance(previous_action, torch.Tensor) or previous_action.ndim != 3:
            raise ValueError(
                f"`sample['previous_action']` must have shape [B,H,A], got {tuple(getattr(previous_action, 'shape', ()))}."
            )
        expected_history_shape = (batch_size, int(previous_action.shape[1]))
        if int(previous_action.shape[1]) <= 0 or int(previous_action.shape[2]) != int(
            action.shape[2]
        ):
            raise ValueError(
                f"Stage 2 previous-action shape must use a positive history horizon and match action_dim: previous={tuple(previous_action.shape)} action={tuple(action.shape)}."
            )
        if (
            not isinstance(previous_action_is_pad, torch.Tensor)
            or tuple(previous_action_is_pad.shape) != expected_history_shape
        ):
            raise ValueError(
                f"`sample['previous_action_is_pad']` must have shape {expected_history_shape}, got {tuple(getattr(previous_action_is_pad, 'shape', ()))}."
            )
        history_pad_mask = previous_action_is_pad.to(
            device=previous_action.device, dtype=torch.bool
        )
        padded_history_nonzero = previous_action.ne(0).any(dim=-1) & history_pad_mask
        if bool(padded_history_nonzero.any().item()):
            raise ValueError("Padded previous-action values must be normalized zeros.")
    if int(action.shape[1]) <= 0 or int(proprio.shape[1]) <= 0:
        raise ValueError(
            f"Stage 2 action/proprio horizon must be positive, got action={action.shape[1]} proprio={proprio.shape[1]}."
        )
    if int(action.shape[1]) != int(proprio.shape[1]):
        raise ValueError(
            f"Stage 2 requires proprio to align with action horizon, got action={action.shape[1]} proprio={proprio.shape[1]}."
        )
    video_model_family = str(getattr(core, "video_model_family", "wan22_ti2v_5b"))
    is_14b = video_model_family == "wan22_i2v_a14b"
    action_is_pad = sample.get("action_is_pad")
    if action_is_pad is not None:
        if not isinstance(action_is_pad, torch.Tensor) or tuple(action_is_pad.shape) != (
            batch_size,
            action.shape[1],
        ):
            raise ValueError(
                f"`sample['action_is_pad']` must have shape {(batch_size, action.shape[1])}, got {tuple(getattr(action_is_pad, 'shape', ()))}."
            )
        valid_sum = (~action_is_pad.bool()).sum(dim=1)
        if bool((valid_sum <= 0).any().item()):
            sample_id = sample.get("sample_id")
            prompt = sample.get("prompt")
            raise ValueError(
                f"Stage 2 received an all-padded action sample. sample_id={sample_id!r} prompt={prompt!r}"
            )

    def _resolve_batch_indices(
        key: str, *, raw_indices: Any, subset_batch_size: int
    ) -> torch.Tensor:
        if raw_indices is None:
            if subset_batch_size != batch_size:
                raise ValueError(
                    f"`sample[{key!r}]` is required when modality subset batch size {subset_batch_size} does not match full batch size {batch_size}."
                )
            return torch.arange(batch_size, dtype=torch.long)
        if not isinstance(raw_indices, torch.Tensor) or raw_indices.ndim != 1:
            raise ValueError(
                f"`sample[{key!r}]` must be 1D long tensor of length {subset_batch_size}, got {tuple(getattr(raw_indices, 'shape', ()))}."
            )
        if int(raw_indices.shape[0]) != subset_batch_size:
            raise ValueError(
                f"`sample[{key!r}]` length must match modality batch size {subset_batch_size}, got {raw_indices.shape[0]}."
            )
        return raw_indices.to(dtype=torch.long, device="cpu")

    assigned_rows = torch.zeros(batch_size, dtype=torch.bool)
    latent_batch_indices_cpu: torch.Tensor | None = None
    video_batch_indices_cpu: torch.Tensor | None = None
    input_latents_raw = sample.get("input_latents")
    i2v_condition_raw = sample.get(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)
    video_raw = sample.get("video")
    if has_latents:
        if not isinstance(input_latents_raw, torch.Tensor) or input_latents_raw.ndim != 5:
            raise ValueError(
                f"`sample['input_latents']` must be 5D [B,C,T,H,W]; got {tuple(getattr(input_latents_raw, 'shape', ()))}."
            )
        if is_14b:
            if not isinstance(i2v_condition_raw, torch.Tensor) or i2v_condition_raw.ndim != 5:
                raise ValueError(
                    f"Stage 2 Wan2.2 14B cached-latent training requires `sample[{WAN22_A14B_I2V_CONDITION_LATENTS_KEY!r}]` with shape [B,C,T,H,W]. Regenerate the latent cache with official Wan2.2 I2V pseudo-video conditioning."
                )
            if tuple(i2v_condition_raw.shape) != tuple(input_latents_raw.shape):
                raise ValueError(
                    f"Stage 2 14B cached I2V condition shape must match input latents: condition={tuple(i2v_condition_raw.shape)} latents={tuple(input_latents_raw.shape)}."
                )
        latent_batch_indices_cpu = _resolve_batch_indices(
            "input_latents_batch_indices",
            raw_indices=sample.get("input_latents_batch_indices"),
            subset_batch_size=int(input_latents_raw.shape[0]),
        )
        if bool(
            ((latent_batch_indices_cpu < 0) | (latent_batch_indices_cpu >= batch_size)).any().item()
        ):
            raise ValueError(
                f"`sample['input_latents_batch_indices']` must stay within [0, {batch_size}), got {latent_batch_indices_cpu.tolist()}."
            )
        if bool(assigned_rows[latent_batch_indices_cpu].any().item()):
            raise ValueError(
                f"`sample['input_latents_batch_indices']` contains duplicate rows: {latent_batch_indices_cpu.tolist()}."
            )
        assigned_rows[latent_batch_indices_cpu] = True
    if has_video:
        if (
            not isinstance(video_raw, torch.Tensor)
            or video_raw.ndim != 5
            or int(video_raw.shape[1]) != 3
        ):
            raise ValueError(
                f"`sample['video']` must have shape [B,3,T,H,W], got {tuple(getattr(video_raw, 'shape', ()))}."
            )
        if video_raw.shape[2] % 4 != 1:
            raise ValueError(
                f"Video frame count must satisfy T % 4 == 1, got {video_raw.shape[2]}."
            )
        if video_raw.shape[3] % 16 != 0 or video_raw.shape[4] % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={video_raw.shape[3]}, W={video_raw.shape[4]}."
            )
        video_batch_indices_cpu = _resolve_batch_indices(
            "video_batch_indices",
            raw_indices=sample.get("video_batch_indices"),
            subset_batch_size=int(video_raw.shape[0]),
        )
        if bool(
            ((video_batch_indices_cpu < 0) | (video_batch_indices_cpu >= batch_size)).any().item()
        ):
            raise ValueError(
                f"`sample['video_batch_indices']` must stay within [0, {batch_size}), got {video_batch_indices_cpu.tolist()}."
            )
        if bool(assigned_rows[video_batch_indices_cpu].any().item()):
            raise ValueError(
                f"Stage 2 mixed cached/live batch assigned the same batch rows to both modalities: {video_batch_indices_cpu.tolist()}."
            )
        assigned_rows[video_batch_indices_cpu] = True
    if not bool(assigned_rows.all().item()):
        missing_rows = (~assigned_rows).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Stage 2 mixed cached/live batch left rows without modality assignment: {missing_rows}."
        )
    image_is_pad = sample.get("image_is_pad")
    with (
        timing_recorder.record("diag_input_transfer_seconds")
        if timing_recorder is not None
        else nullcontext()
    ):
        context = context.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        context_mask = context_mask_bool.to(device=core.device, dtype=torch.bool, non_blocking=True)
        dit_context_mask = (
            None
            if context_mask_bool.device.type == "cpu" and bool(context_mask_bool.all().item())
            else context_mask_bool.to(device=core.device, dtype=torch.bool, non_blocking=True)
        )
        action = action.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        proprio = proprio.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        previous_action = (
            None
            if previous_action is None
            else previous_action.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        )
        previous_action_is_pad = (
            None
            if previous_action_is_pad is None
            else previous_action_is_pad.to(device=core.device, dtype=torch.bool, non_blocking=True)
        )
        action_is_pad = (
            None
            if action_is_pad is None
            else action_is_pad.to(device=core.device, dtype=torch.bool, non_blocking=True)
        )
        if image_is_pad is not None:
            if (
                not isinstance(image_is_pad, torch.Tensor)
                or image_is_pad.ndim != 2
                or int(image_is_pad.shape[0]) != batch_size
            ):
                raise ValueError(
                    f"`sample['image_is_pad']` must have shape [B,T], got {tuple(getattr(image_is_pad, 'shape', ()))}."
                )
            image_is_pad = image_is_pad.to(device=core.device, dtype=torch.bool, non_blocking=True)
        input_latents_cached = (
            None
            if not has_latents
            else input_latents_raw.to(
                device=core.device, dtype=core.torch_dtype, non_blocking=True
            ).detach()
        )
        i2v_condition_cached = (
            None
            if not (is_14b and has_latents)
            else i2v_condition_raw.to(
                device=core.device, dtype=core.torch_dtype, non_blocking=True
            ).detach()
        )
        input_video = (
            None
            if not has_video
            else video_raw.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        )
    input_latents_video = None
    i2v_condition_video = None
    if has_video:
        with (
            timing_recorder.record("diag_live_vae_encode_seconds")
            if timing_recorder is not None
            else nullcontext()
        ):
            with torch.no_grad():
                input_latents_video = core._encode_video_latents(input_video, tiled=tiled).detach()
                if is_14b:
                    i2v_condition_video = core._encode_14b_i2v_condition_latents(
                        input_video, tiled=tiled
                    ).detach()
    reference_latents = (
        input_latents_cached if input_latents_cached is not None else input_latents_video
    )
    if reference_latents is None:
        raise AssertionError("Stage 2 batch must provide cached latents, live video, or both.")
    if input_latents_cached is not None and input_latents_video is not None:
        if tuple(input_latents_cached.shape[1:]) != tuple(input_latents_video.shape[1:]):
            raise ValueError(
                f"Stage 2 cached/live latent shapes do not match: cached={tuple(input_latents_cached.shape)} live={tuple(input_latents_video.shape)}."
            )
    reference_i2v_condition = (
        i2v_condition_cached if i2v_condition_cached is not None else i2v_condition_video
    )
    if is_14b:
        if reference_i2v_condition is None:
            raise ValueError(
                f"Stage 2 Wan2.2 14B requires official I2V condition latents. Provide live video or regenerate cached latents with `{WAN22_A14B_I2V_CONDITION_LATENTS_KEY}`."
            )
        if tuple(reference_i2v_condition.shape[1:]) != tuple(reference_latents.shape[1:]):
            raise ValueError(
                f"Stage 2 14B I2V condition shape must match input latents after the batch axis: condition={tuple(reference_i2v_condition.shape)} latents={tuple(reference_latents.shape)}."
            )
        if i2v_condition_cached is not None and input_latents_cached is not None:
            if tuple(i2v_condition_cached.shape) != tuple(input_latents_cached.shape):
                raise ValueError(
                    f"Stage 2 cached 14B I2V condition shape mismatch: condition={tuple(i2v_condition_cached.shape)} latents={tuple(input_latents_cached.shape)}."
                )
        if i2v_condition_video is not None and input_latents_video is not None:
            if tuple(i2v_condition_video.shape) != tuple(input_latents_video.shape):
                raise ValueError(
                    f"Stage 2 live 14B I2V condition shape mismatch: condition={tuple(i2v_condition_video.shape)} latents={tuple(input_latents_video.shape)}."
                )
    latent_t = int(reference_latents.shape[2])
    expected_video_t = (latent_t - 1) * 4 + 1
    if image_is_pad is not None and tuple(image_is_pad.shape) != (batch_size, expected_video_t):
        raise ValueError(
            f"`sample['image_is_pad']` must have shape {(batch_size, expected_video_t)} for latent_t={latent_t}, got {tuple(getattr(image_is_pad, 'shape', ()))}."
        )
    input_latents = torch.empty(
        (batch_size, *reference_latents.shape[1:]), device=core.device, dtype=core.torch_dtype
    )
    if input_latents_cached is not None:
        assert latent_batch_indices_cpu is not None
        input_latents.index_copy_(
            0,
            latent_batch_indices_cpu.to(device=core.device, non_blocking=True),
            input_latents_cached,
        )
    if input_latents_video is not None:
        assert video_batch_indices_cpu is not None
        input_latents.index_copy_(
            0,
            video_batch_indices_cpu.to(device=core.device, non_blocking=True),
            input_latents_video,
        )
    i2v_condition_latents = None
    if is_14b:
        assert reference_i2v_condition is not None
        i2v_condition_latents = torch.empty_like(input_latents)
        if i2v_condition_cached is not None:
            assert latent_batch_indices_cpu is not None
            i2v_condition_latents.index_copy_(
                0,
                latent_batch_indices_cpu.to(device=core.device, non_blocking=True),
                i2v_condition_cached,
            )
        if i2v_condition_video is not None:
            assert video_batch_indices_cpu is not None
            i2v_condition_latents.index_copy_(
                0,
                video_batch_indices_cpu.to(device=core.device, non_blocking=True),
                i2v_condition_video,
            )
    inputs = {
        "context": context,
        "context_mask": context_mask,
        "dit_context_mask": dit_context_mask,
        "input_latents": input_latents,
        "action": action,
        "proprio": proprio,
        "action_is_pad": action_is_pad,
        "image_is_pad": image_is_pad,
    }
    if previous_action is not None:
        inputs["previous_action"] = previous_action
        inputs["previous_action_is_pad"] = previous_action_is_pad
    if i2v_condition_latents is not None:
        inputs[WAN22_A14B_I2V_CONDITION_LATENTS_KEY] = i2v_condition_latents
    for key in ("prompt", "idx"):
        if key in sample:
            inputs[key] = sample[key]
    return inputs


def _build_stage2_live_dataset(
    cfg: Stage2Config,
    *,
    split: str,
    val_set_proportion: float | None = None,
    test_set_proportion: float | None = None,
) -> Dataset:
    is_training = split == "train"
    video_roots = None
    resolved_val_set_proportion = (
        float(cfg.val_set_proportion) if val_set_proportion is None else float(val_set_proportion)
    )
    resolved_test_set_proportion = (
        float(getattr(cfg, "test_set_proportion", 0.0))
        if test_set_proportion is None
        else float(test_set_proportion)
    )
    return _SparseStage2RobotVideoDataset(
        dataset_dirs=_parse_dataset_dirs(cfg.dataset_dirs),
        dataset_name=str(cfg.dataset_name),
        shape_meta=_build_stage2_shape_meta(cfg),
        num_frames=int(cfg.num_frames),
        action_horizon=resolve_stage2_action_horizon(cfg),
        action_history_horizon=int(getattr(cfg, "action_history_horizon", 0)),
        video_size=[int(cfg.height), int(cfg.width)],
        processor=_build_stage2_processor(cfg),
        text_embedding_cache_dir=str(cfg.text_embedding_cache_dir),
        text_cache_model_suffix=_stage2_text_cache_suffix(cfg),
        context_len=int(cfg.context_len),
        pretrained_norm_stats=str(cfg.pretrained_norm_stats),
        val_set_proportion=resolved_val_set_proportion,
        test_set_proportion=resolved_test_set_proportion,
        split=str(split),
        is_training_set=is_training,
        global_sample_stride=int(cfg.global_sample_stride),
        action_video_freq_ratio=int(cfg.action_video_freq_ratio),
        skip_padding_as_possible=False,
        max_padding_retry=3,
        concat_multi_camera=str(cfg.concat_multi_camera),
        override_instruction=None,
        strict_loading=bool(cfg.strict_dataset_errors),
        strict_errors=bool(cfg.strict_dataset_errors),
        video_roots=video_roots,
        allow_mixed_fps_for_cached_latents=False,
        metadata_only_for_cached_latents=False,
    )


def _attach_cache_episode_index(
    dataset: _SparseStage2RobotVideoDataset,
) -> _SparseStage2RobotVideoDataset:
    raw_datasets = getattr(dataset.multi_dataset, "_datasets", None)
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError(
            "Cached Stage 2 action training requires a fallback dataset backed by at least one LeRobot dataset."
        )
    episode_data_indices = []
    end_index = 0
    for raw_dataset in raw_datasets:
        episode_data_index = getattr(raw_dataset, "episode_data_index", None)
        if episode_data_index is None:
            raise ValueError(
                "Sparse Stage 2 fallback dataset is missing episode_data_index for cached latent routing."
            )
        episode_data_indices.append(
            {
                "from": episode_data_index["from"] + end_index,
                "to": episode_data_index["to"] + end_index,
            }
        )
        if len(episode_data_index["to"]) > 0:
            end_index += int(episode_data_index["to"][-1])
    dataset.lerobot_dataset = SimpleNamespace(
        multi_dataset=dataset.multi_dataset,
        episode_data_index={
            "from": torch.cat([item["from"] for item in episode_data_indices]),
            "to": torch.cat([item["to"] for item in episode_data_indices]),
        },
        processor=dataset.processor,
        is_training_set=dataset.is_training_set,
    )
    return dataset


def _build_stage2_dataset(cfg, *, split, val_set_proportion=None, test_set_proportion=None):
    live = _build_stage2_live_dataset(
        cfg,
        split=split,
        val_set_proportion=val_set_proportion,
        test_set_proportion=test_set_proportion,
    )
    if not cfg.use_cached_latents:
        return live
    from dreamfree.data.lerobot.cached_latent_dataset import CachedStage2LatentDataset

    live = _attach_cache_episode_index(live)
    return CachedStage2LatentDataset(
        cache_dir=str(cfg.stage2_latent_cache_dir),
        source_dataset=live,
        expected_cache_contract=_stage2_latent_cache_expected_contract(cfg),
    )


def _stage2_cached_latent_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty Stage 2 batch.")
    for sample in batch:
        if "video" in sample:
            raise ValueError("Stage 2 cached-latent batches must not include `video`.")
        if "input_latents" not in sample:
            raise ValueError(
                "Stage 2 cached-latent batches require `input_latents` for every sample."
            )
    collated: dict[str, Any] = {}
    shared_keys = set().union(*(sample.keys() for sample in batch))
    shared_keys.discard("input_latents")
    for key in sorted(shared_keys):
        values = [sample[key] for sample in batch if key in sample]
        if len(values) != len(batch):
            raise ValueError(
                f"Stage 2 cached-latent collate expected every sample to define `{key}`."
            )
        collated[key] = default_collate(values)
    collated["input_latents"] = default_collate([sample["input_latents"] for sample in batch])
    return collated


def _build_stage2_dataloader(dataset: Dataset, cfg: Stage2Config, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed))
    num_workers = int(cfg.num_workers)
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
    persistent_workers = bool(cfg.dataloader_persistent_workers)
    prefetch_factor = int(cfg.dataloader_prefetch_factor)
    if num_workers == 0:
        if persistent_workers:
            raise ValueError("dataloader_persistent_workers=true requires num_workers > 0.")
        loader_kwargs: dict[str, Any] = {}
    else:
        if prefetch_factor <= 0:
            raise ValueError(
                f"dataloader_prefetch_factor must be > 0 when num_workers > 0, got {prefetch_factor}."
            )
        loader_kwargs = {
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        }
    collate_fn = _stage2_cached_latent_collate if bool(cfg.use_cached_latents) else None
    sampler = None
    return DataLoader(
        dataset,
        batch_size=int(cfg.per_device_batch_size),
        shuffle=bool(shuffle) and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
