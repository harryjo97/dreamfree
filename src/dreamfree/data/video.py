from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as TVF
from safetensors import safe_open
from torch.utils.data import Dataset

from dreamfree.data.contracts import (
    _normalize_latent_frame0_mode,
    _stage0_latent_cache_expected_contract,
)
from dreamfree.data.lerobot.cached_latent_dataset import (
    DROID_DREAMZERO_ACTION_CONTRACT,
    DROID_DREAMZERO_REQUIRED_TENSOR_DIMS,
    validate_stage2_latent_cache_file,
    validate_stage2_latent_cache_header_file,
)
from dreamfree.data.lerobot.lerobot.datasets.video_utils import decode_video_frames
from dreamfree.data.text_cache import DEFAULT_PROMPT
from dreamfree.data.text_cache import load_text_context as _load_stage0_text_cache
from dreamfree.models.adaptation import (
    _normalize_stage0_sigma_policy,
    _stage0_train_sigma_bounds,
    _validate_stage0_train_sigma_bounds_for_policy,
)
from dreamfree.models.wan22.i2v_condition import WAN22_A14B_I2V_CONDITION_LATENTS_KEY
from dreamfree.training.runtime import _json_load, _parse_optional_int
from dreamfree.training.stage0.config import Stage0Config
from dreamfree.training.stage0.constants import (
    DROID_CAMERA_KEYS,
    DROID_DREAMZERO_ACTION_DIM,
    DROID_DREAMZERO_PROPRIO_DIM,
    DROID_GRID_CONCAT_MULTI_CAMERA,
    DROID_GRID_HEIGHT,
    DROID_GRID_WIDTH,
    HORIZONTAL_CONCAT_MULTI_CAMERA,
    ROBOCASA_CAMERA_KEYS,
    ROBOCASA_DREAMZERO_352_CONCAT_MULTI_CAMERA,
    ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA,
    ROBOCASA_DREAMZERO_448_HEIGHT,
    ROBOCASA_DREAMZERO_448_WIDTH,
    ROBOCASA_DREAMZERO_HEIGHT,
    ROBOCASA_DREAMZERO_LAYOUTS,
    ROBOCASA_DREAMZERO_WIDTH,
    ROBOTWIN_CAMERA_KEYS,
    ROBOTWIN_CONCAT_MULTI_CAMERA,
    STAGE0_CACHED_LATENTS_DATASET_FORMAT,
    SUPPORTED_LEROBOT_CODEBASE_VERSIONS,
    VERTICAL_CONCAT_MULTI_CAMERA,
    logger,
)


def _normalize_stage0_dataset_format(value):
    if str(value) != "lerobot_root":
        raise ValueError("dataset_format must be 'lerobot_root'.")
    return "lerobot_root"


def _parse_stage0_text_cache_dirs(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if text in {"", "null", "None", "none"}:
            return []
        return [Path(part.strip()).expanduser() for part in text.split(",") if part.strip()]
    if isinstance(value, Sequence):
        dirs: list[Path] = []
        for part in value:
            text = str(part).strip()
            if text and text not in {"null", "None", "none"}:
                dirs.append(Path(text).expanduser())
        return dirs
    raise TypeError(
        f"text_embedding_cache_dir must be a comma-separated string or sequence, got {type(value)}."
    )


def _parse_stage0_camera_keys(value: Any, *, allow_single: bool = False) -> tuple[str, ...]:
    if isinstance(value, str):
        keys = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        keys = [str(item).strip() for item in value]
    else:
        raise TypeError(
            f"camera_keys must be a comma-separated string or sequence, got {type(value)}"
        )
    keys = [key for key in keys if key]
    allowed_counts = {1, 2, 3} if allow_single else {2, 3}
    if len(keys) not in allowed_counts:
        expected = "1, 2, or 3" if allow_single else "2 or 3"
        raise ValueError(
            f"Stage 0/1 requires exactly {expected} camera keys, got {len(keys)}: {keys}"
        )
    if len(set(keys)) != len(keys):
        raise ValueError(f"Stage 0 camera keys must be unique, got: {keys}")
    return tuple(keys)


def _stage0_dataset_name(cfg: Any) -> str:
    return str(getattr(cfg, "dataset_name", "")).strip().lower()


def _stage0_camera_key_candidates(cfg):
    return (_parse_stage0_camera_keys(cfg.camera_keys),)


def _normalize_stage0_task_candidates_for_source(cfg, task_candidates, *, camera_keys):
    return _dedupe_nonempty_task_texts(task_candidates)


def _stage0_feature_matches_camera_key(cfg, features, camera_keys, key):
    return isinstance(features.get(key), dict) and features[key].get("dtype") == "video"


def _stage0_select_camera_keys_for_features(
    cfg: Any, features: dict[str, Any], *, source: str | Path
) -> tuple[str, ...]:
    for camera_keys in _stage0_camera_key_candidates(cfg):
        if all(
            (
                _stage0_feature_matches_camera_key(cfg, features, camera_keys, key)
                for key in camera_keys
            )
        ):
            return camera_keys
    available_camera_keys = sorted(
        (
            key
            for (key, feature) in features.items()
            if isinstance(feature, dict) and str(feature.get("dtype")) in {"video", "image"}
        )
    )
    expected = ["[" + ", ".join(keys) + "]" for keys in _stage0_camera_key_candidates(cfg)]
    raise ValueError(
        f"{source} is missing required Stage 0 video cameras for dataset_name={_stage0_dataset_name(cfg)!r}. Expected one of {expected}; available video/image features={available_camera_keys}."
    )


@dataclass
class Stage0Episode:
    """Metadata needed to sample one LeRobot episode window.

    Attributes:
        root: Extracted task root containing LeRobot metadata.
        episode_index: Original episode index from the dataset.
        length: Number of frames in the episode.
        fps: Video frame rate used to convert frame indices to timestamps.
        task: Default natural-language task prompt source.
        video_paths: Mapping from camera key to resolved video file path.
        task_candidates: Natural-language prompt candidates for random prompt
            selection. Empty means ``task`` is the only valid prompt.
        video_start_times: Per-camera timestamp offset inside shared video files.
    """

    root: str
    episode_index: int
    length: int
    fps: int
    task: str
    video_paths: dict[str, str]
    task_candidates: tuple[str, ...] = field(default_factory=tuple)
    video_start_times: dict[str, float] = field(default_factory=dict)


def _dedupe_nonempty_task_texts(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    candidates: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        candidates.append(text)
    return tuple(candidates)


def _task_candidates_from_value(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        return _dedupe_nonempty_task_texts((value,))
    if isinstance(value, (list, tuple)):
        candidates: list[str] = []
        for item in value:
            candidates.extend(_task_candidates_from_value(item))
        return _dedupe_nonempty_task_texts(candidates)
    return ()


def _coerce_first_task_index(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.item() if value.size == 1 else value.tolist()[0]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_episode_task_candidates(
    *,
    row: dict[str, Any],
    task_by_index: dict[int, str],
    default_task: str | None = None,
    strict: bool = False,
    stage_label: str = "Stage 0",
    task_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Resolve all natural-language prompt candidates from one LeRobot episode row."""
    candidates = _task_candidates_from_value(row.get("tasks"))
    if candidates:
        return candidates
    indexed_candidates: list[str] = []
    for key in ("task_index", "atomic_task_index", "coarse_task_index"):
        idx = _coerce_first_task_index(row.get(key))
        if idx is not None and idx in task_by_index:
            indexed_candidates.append(task_by_index[idx])
    candidates = _dedupe_nonempty_task_texts(indexed_candidates)
    if candidates:
        return candidates
    if default_task is not None:
        default_candidates = _dedupe_nonempty_task_texts((default_task,))
        if default_candidates:
            return default_candidates
    if strict:
        episode_index = row.get("episode_index", "<missing>")
        root_text = "" if task_root is None else f" task_root={task_root}"
        available_keys = sorted((str(key) for key in row.keys()))
        raise ValueError(
            f"Could not resolve {stage_label} task text candidates: episode_index={episode_index}{root_text} available_keys={available_keys}"
        )
    return ()


def _parse_stage0_latent_cache_dirs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if text in {"", "null", "None", "none"}:
            return []
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, Sequence):
        dirs: list[str] = []
        for part in value:
            text = str(part).strip()
            if text and text not in {"null", "None", "none"}:
                dirs.append(text)
        return dirs
    raise TypeError(
        f"stage0_latent_cache_dir must be a comma-separated string or sequence, got {type(value)}."
    )


def _format_stage0_latent_cache_dirs(cache_dirs: Sequence[Path]) -> str:
    return ",".join((str(Path(path).expanduser()) for path in cache_dirs))


def _first_stage0_latent_cache_file(latent_cache_dir: Path) -> Path | None:
    first_episode = next(iter(sorted(latent_cache_dir.glob("episode_*.safetensors"))), None)
    if first_episode is not None:
        return first_episode
    return next(iter(sorted(latent_cache_dir.glob("*/episode_*.safetensors"))), None)


def _is_droid_cache_meta(meta: dict[str, Any]) -> bool:
    concat_layout = str(meta.get("concat_layout", "")).strip().lower()
    return (
        concat_layout == DROID_GRID_CONCAT_MULTI_CAMERA
        or str(meta.get("action_contract", "")).strip() == DROID_DREAMZERO_ACTION_CONTRACT
    )


def _droid_dreamzero_overlay_latent_source_path(path: Path, meta: dict[str, Any]) -> Path:
    raw_path = str(meta.get("latent_source_path", "")).strip()
    if not raw_path:
        raise ValueError(f"DROID DreamZero overlay cache is missing `latent_source_path`: {path}")
    source_path = Path(raw_path).expanduser()
    if not source_path.is_absolute():
        source_path = path.parent / source_path
    source_path = source_path.resolve()
    if source_path == path.expanduser().resolve():
        raise ValueError(
            f"DROID DreamZero overlay cache cannot point `latent_source_path` to itself: {path}"
        )
    if not source_path.exists():
        raise FileNotFoundError(
            f"DROID DreamZero overlay latent_source_path does not exist: {source_path}"
        )
    return source_path


def _stage0_cached_or_overlay_latent_header(
    path: Path,
) -> tuple[dict[str, Any], tuple[int, ...], Path]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
        latent_shape = tuple((int(v) for v in handle.get_slice("latents").get_shape()))
        return (meta, latent_shape, path)
    source_path = _droid_dreamzero_overlay_latent_source_path(path, meta)
    with safe_open(str(source_path), framework="pt", device="cpu") as source_handle:
        latent_shape = tuple((int(v) for v in source_handle.get_slice("latents").get_shape()))
    return (meta, latent_shape, source_path)


def _parse_stage0_dataset_dirs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        dirs = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        dirs = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise TypeError(
            f"dataset_dirs must be a comma-separated string or sequence, got {type(value)}."
        )
    return dirs


def _stage0_root_cache_name(root: Path) -> str:
    base = re.sub("[^A-Za-z0-9_.-]+", "_", Path(root).name).strip("._-") or "root"
    digest = hashlib.sha1(str(Path(root).expanduser().resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{base}-{digest}"


def _stage0_root_cache_name_map(roots: Sequence[Path | str]) -> dict[Path, str]:
    resolved_roots = [Path(root).expanduser().resolve() for root in roots]
    name_counts: dict[str, int] = {}
    for root in resolved_roots:
        name_counts[root.name] = name_counts.get(root.name, 0) + 1
    return {
        root: root.name if name_counts[root.name] == 1 else _stage0_root_cache_name(root)
        for root in resolved_roots
    }


def _stage0_cache_match_name(value: str) -> str:
    return re.sub("[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _stage0_cache_dir_can_use_flat_episodes(cache_dir: Path, root: Path) -> bool:
    root_name = _stage0_cache_match_name(root.name)
    cache_name = _stage0_cache_match_name(cache_dir.name)
    if not root_name:
        return False
    if cache_name == root_name or cache_name.startswith(f"{root_name}_"):
        return True
    flat_cache_prefixes = {"robocasa_cosmos_success_lerobot": ("robocasa_dreamzero_",)}
    return any((cache_name.startswith(prefix) for prefix in flat_cache_prefixes.get(root_name, ())))


def resolve_stage0_latent_cache_dirs(cfg: Stage0Config) -> list[Path]:
    raw_dirs = _parse_stage0_latent_cache_dirs(getattr(cfg, "stage0_latent_cache_dir", None))
    if not raw_dirs:
        raise ValueError("Stage 0 use_cached_latents=true requires stage0_latent_cache_dir.")
    return [Path(raw).expanduser() for raw in raw_dirs]


def _validate_stage0_latent_cache_config(cfg: Stage0Config) -> None:
    if not bool(getattr(cfg, "use_cached_latents", False)):
        cache_dirs = [
            Path(path).expanduser()
            for path in _parse_stage0_latent_cache_dirs(
                getattr(cfg, "stage0_latent_cache_dir", None)
            )
        ]
        cfg.stage0_latent_cache_dir = (
            _format_stage0_latent_cache_dirs(cache_dirs) if cache_dirs else None
        )
        return
    latent_cache_dirs = resolve_stage0_latent_cache_dirs(cfg)
    cfg.stage0_latent_cache_dir = _format_stage0_latent_cache_dirs(latent_cache_dirs)
    missing_dirs = [str(path) for path in latent_cache_dirs if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Stage 0 video latent cache directories not found: {missing_dirs}. Precompute matching caches first or set use_cached_latents=false."
        )
    first_episodes: list[Path] = []
    empty_dirs: list[str] = []
    for latent_cache_dir in latent_cache_dirs:
        first_episode = _first_stage0_latent_cache_file(latent_cache_dir)
        if first_episode is None:
            empty_dirs.append(str(latent_cache_dir))
        else:
            first_episodes.append(first_episode)
    if empty_dirs:
        raise FileNotFoundError(
            f"No precomputed video latent files found under Stage 0 cache directories: {empty_dirs}. Precompute matching caches first or set use_cached_latents=false."
        )
    for first_episode in first_episodes:
        _validate_stage0_latent_cache_file_for_cfg(first_episode, cfg)


def _stage0_robocasa_layout_spec(cfg: Any) -> dict[str, int | str]:
    concat_layout = str(getattr(cfg, "concat_multi_camera", "")).strip().lower()
    if concat_layout == ROBOCASA_DREAMZERO_352_CONCAT_MULTI_CAMERA:
        return {
            "concat_layout": ROBOCASA_DREAMZERO_352_CONCAT_MULTI_CAMERA,
            "height": ROBOCASA_DREAMZERO_HEIGHT,
            "width": ROBOCASA_DREAMZERO_WIDTH,
            "tile": 176,
        }
    if concat_layout == ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA:
        return {
            "concat_layout": ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA,
            "height": ROBOCASA_DREAMZERO_448_HEIGHT,
            "width": ROBOCASA_DREAMZERO_448_WIDTH,
            "tile": 224,
        }
    raise ValueError(
        f"Unsupported Stage 0 RoboCasa layout {concat_layout!r}; expected one of {sorted(ROBOCASA_DREAMZERO_LAYOUTS)}."
    )


def _is_stage0_robocasa_layout(cfg: Any) -> bool:
    return (
        str(getattr(cfg, "concat_multi_camera", "")).strip().lower() in ROBOCASA_DREAMZERO_LAYOUTS
    )


def _is_stage0_droid_layout(cfg: Any) -> bool:
    return (
        str(getattr(cfg, "concat_multi_camera", "")).strip().lower()
        == DROID_GRID_CONCAT_MULTI_CAMERA
    )


def _validate_stage0_robocasa_layout_config(
    cfg: Stage0Config, *, camera_keys: tuple[str, ...], dataset_format: str
) -> None:
    if dataset_format not in {"lerobot_root", STAGE0_CACHED_LATENTS_DATASET_FORMAT}:
        raise ValueError(
            f"Stage 0 Robocasa layout requires dataset_format='lerobot_root' or '{STAGE0_CACHED_LATENTS_DATASET_FORMAT}'."
        )
    if tuple(camera_keys) != tuple(ROBOCASA_CAMERA_KEYS):
        raise ValueError(
            f"Stage 0 Robocasa layout requires camera_keys={','.join(ROBOCASA_CAMERA_KEYS)!r}."
        )
    spec = _stage0_robocasa_layout_spec(cfg)
    expected_size = (int(spec["height"]), int(spec["width"]))
    if (int(cfg.height), int(cfg.width)) != expected_size:
        raise ValueError(
            f"Stage 0 Robocasa layout requires height={expected_size[0]},width={expected_size[1]}, got {cfg.height}x{cfg.width}."
        )


def _validate_stage0_droid_layout_config(
    cfg: Stage0Config, *, camera_keys: tuple[str, ...], dataset_format: str
) -> None:
    if dataset_format != "lerobot_root":
        raise ValueError("Stage 0 DROID layout requires dataset_format='lerobot_root'.")
    if tuple(camera_keys) != tuple(DROID_CAMERA_KEYS):
        raise ValueError(
            f"Stage 0 DROID layout requires camera_keys={','.join(DROID_CAMERA_KEYS)!r}."
        )
    if (int(cfg.height), int(cfg.width)) != (DROID_GRID_HEIGHT, DROID_GRID_WIDTH):
        raise ValueError(
            f"Stage 0 DROID layout requires height={DROID_GRID_HEIGHT},width={DROID_GRID_WIDTH}, got {cfg.height}x{cfg.width}."
        )


def _validate_stage0_robocasa_latent_cache_metadata(
    path: Path, cfg: Stage0Config, *, allowed_dataset_kinds: Sequence[str] = ("robocasa",)
) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
    expected = {"concat_layout": str(_stage0_robocasa_layout_spec(cfg)["concat_layout"])}
    mismatches = [
        f"{key}: actual={meta.get(key)!r} expected={value!r}"
        for (key, value) in expected.items()
        if str(meta.get(key)) != str(value)
    ]
    dataset_kind = str(meta.get("dataset_kind", "")).strip().lower()
    normalized_allowed_kinds = {
        str(value).strip().lower() for value in allowed_dataset_kinds if str(value).strip()
    }
    if dataset_kind not in normalized_allowed_kinds:
        mismatches.append(
            f"dataset_kind: actual={meta.get('dataset_kind')!r} expected one of={sorted(normalized_allowed_kinds)!r}"
        )
    if mismatches:
        raise ValueError(
            f"Stage 0 Robocasa latent cache metadata mismatch in {path}: " + "; ".join(mismatches)
        )


def _validate_droid_dreamzero_latent_cache_metadata(path: Path, *, stage_label: str) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
        tensor_shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
    expected = {
        "concat_layout": DROID_GRID_CONCAT_MULTI_CAMERA,
        "dataset_kind": "droid",
        "action_dim": DROID_DREAMZERO_ACTION_DIM,
        "action_contract": DROID_DREAMZERO_ACTION_CONTRACT,
        "norm_mode": "q01/q99",
    }
    mismatches = [
        f"{key}: actual={meta.get(key)!r} expected={value!r}"
        for (key, value) in expected.items()
        if str(meta.get(key)) != str(value)
    ]
    state_dim = meta.get("state_dim", meta.get("proprio_dim"))
    if str(state_dim) != str(DROID_DREAMZERO_PROPRIO_DIM):
        mismatches.append(
            f"state_dim/proprio_dim: actual={state_dim!r} expected={DROID_DREAMZERO_PROPRIO_DIM!r}"
        )
    for tensor_key, tensor_dim in DROID_DREAMZERO_REQUIRED_TENSOR_DIMS.items():
        shape = tensor_shapes.get(tensor_key)
        if shape is None:
            mismatches.append(f"{tensor_key}: missing")
        elif len(shape) != 2 or int(shape[-1]) != int(tensor_dim):
            mismatches.append(
                f"{tensor_key}: actual_shape={shape!r} expected_last_dim={tensor_dim}"
            )
    if mismatches:
        raise ValueError(
            f"{stage_label} DROID DreamZero latent cache metadata mismatch in {path}: "
            + "; ".join(mismatches)
        )


def _validate_droid_dreamzero_latent_cache_file(
    path: Path, *, expected: dict[str, Any], stage_label: str
) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        dict(handle.metadata() or {})
    validate_stage2_latent_cache_file(path, expected=expected, stage_label=stage_label)
    _validate_droid_dreamzero_latent_cache_metadata(path, stage_label=stage_label)


def _validate_cached_or_droid_dreamzero_latent_cache_file(
    path: Path, *, expected: dict[str, Any], stage_label: str
) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
    if _is_droid_cache_meta(meta):
        _validate_droid_dreamzero_latent_cache_file(
            path, expected=expected, stage_label=stage_label
        )
        return
    validate_stage2_latent_cache_file(path, expected=expected, stage_label=stage_label)


def _validate_stage0_latent_cache_file_for_cfg(
    path: Path, cfg: Stage0Config, *, allowed_robocasa_dataset_kinds: Sequence[str] = ("robocasa",)
) -> None:
    expected = _stage0_latent_cache_expected_contract(cfg)
    concat_mode = str(getattr(cfg, "concat_multi_camera", "")).strip().lower()
    if concat_mode == DROID_GRID_CONCAT_MULTI_CAMERA:
        _validate_droid_dreamzero_latent_cache_file(path, expected=expected, stage_label="Stage 0")
        return
    validate_stage2_latent_cache_file(path, expected=expected, stage_label="Stage 0")
    if concat_mode in ROBOCASA_DREAMZERO_LAYOUTS:
        _validate_stage0_robocasa_latent_cache_metadata(
            path, cfg, allowed_dataset_kinds=allowed_robocasa_dataset_kinds
        )


def validate_stage0_training_config(cfg: Stage0Config) -> None:
    dataset_format = _normalize_stage0_dataset_format(
        getattr(cfg, "dataset_format", "lerobot_root")
    )
    dataset_dirs = _parse_stage0_dataset_dirs(getattr(cfg, "dataset_dirs", None))
    if dataset_dirs and dataset_format != "lerobot_root":
        raise ValueError("Stage 0 dataset_dirs requires dataset_format='lerobot_root'.")
    if dataset_format == STAGE0_CACHED_LATENTS_DATASET_FORMAT:
        if not bool(getattr(cfg, "use_cached_latents", False)):
            raise ValueError(
                f"Stage 0 dataset_format={STAGE0_CACHED_LATENTS_DATASET_FORMAT!r} requires use_cached_latents=true."
            )
        if int(getattr(cfg, "eval_num_inference_steps", 0)) > 0:
            raise ValueError(
                f"Stage 0 dataset_format={STAGE0_CACHED_LATENTS_DATASET_FORMAT!r} does not have source videos for qualitative rollout eval; set eval_num_inference_steps=0."
            )
    camera_keys = _parse_stage0_camera_keys(getattr(cfg, "camera_keys", ROBOTWIN_CAMERA_KEYS))
    if _is_stage0_robocasa_layout(cfg):
        _validate_stage0_robocasa_layout_config(
            cfg, camera_keys=camera_keys, dataset_format=dataset_format
        )
    if _is_stage0_droid_layout(cfg):
        _validate_stage0_droid_layout_config(
            cfg, camera_keys=camera_keys, dataset_format=dataset_format
        )
    train_sigma_policy = _normalize_stage0_sigma_policy(
        getattr(cfg, "train_sigma_policy", "wan_shift_full")
    )
    (train_sigma_min, train_sigma_max) = _stage0_train_sigma_bounds(cfg)
    _validate_stage0_train_sigma_bounds_for_policy(
        train_sigma_policy, train_sigma_min, train_sigma_max
    )
    latent_frame0_mode = _normalize_latent_frame0_mode(
        getattr(cfg, "latent_frame0_mode", "official")
    )
    family = str(cfg.video_model_family)
    if family not in {"wan22_ti2v_5b", "wan22_i2v_a14b"}:
        raise ValueError(
            f"Unsupported video_model_family={cfg.video_model_family!r}; expected 'wan22_ti2v_5b' or 'wan22_i2v_a14b'."
        )
    if family == "wan22_ti2v_5b":
        if str(cfg.model_id) != "Wan-AI/Wan2.2-TI2V-5B":
            raise ValueError(
                f"wan22_ti2v_5b Stage 0 requires model_id='Wan-AI/Wan2.2-TI2V-5B', got {cfg.model_id!r}."
            )
        if str(cfg.dit_role) != "single":
            raise ValueError(
                f"wan22_ti2v_5b Stage 0 requires dit_role='single', got {cfg.dit_role!r}."
            )
        if str(cfg.i2v_condition_mode) != "none":
            raise ValueError("wan22_ti2v_5b Stage 0 requires i2v_condition_mode='none'.")
        if latent_frame0_mode != "official":
            raise ValueError(
                "latent_frame0_mode only applies to video_model_family='wan22_i2v_a14b'."
            )
    else:
        if str(cfg.model_id) != "Wan-AI/Wan2.2-I2V-A14B":
            raise ValueError(
                f"wan22_i2v_a14b Stage 0 requires model_id='Wan-AI/Wan2.2-I2V-A14B', got {cfg.model_id!r}."
            )
        if str(cfg.dit_role) not in {"low_noise", "high_noise"}:
            raise ValueError(
                f"wan22_i2v_a14b Stage 0 requires dit_role='low_noise' or 'high_noise', got {cfg.dit_role!r}."
            )
        if str(cfg.i2v_condition_mode) != "first_image_mask_condition":
            raise ValueError(
                "wan22_i2v_a14b Stage 0 requires i2v_condition_mode='first_image_mask_condition'."
            )
    if bool(getattr(cfg, "use_text_embedding_cache", True)):
        if not str(getattr(cfg, "text_embedding_cache_dir", "")).strip():
            raise ValueError(
                "use_text_embedding_cache=true requires text_embedding_cache_dir to be set."
            )
        if (
            int(getattr(cfg, "tokenizer_max_len", 128)) != 128
            or int(getattr(cfg, "text_dim", 4096)) != 4096
        ):
            raise ValueError(
                "Stage 0 cached text contract requires tokenizer_max_len=128 and text_dim=4096."
            )
    if bool(getattr(cfg, "compile_dit", False)) and bool(
        getattr(cfg, "use_gradient_checkpointing", False)
    ):
        raise ValueError("compile_dit=true is not supported with use_gradient_checkpointing=true.")
    _validate_stage0_latent_cache_config(cfg)
    if int(cfg.max_steps) < 0:
        raise ValueError(f"max_steps must be non-negative, got {cfg.max_steps}.")
    if int(getattr(cfg, "eval_batches", 1)) <= 0:
        raise ValueError(f"eval_batches must be positive, got {cfg.eval_batches}.")
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
    if bool(getattr(cfg, "export_final", True)) and (
        not str(getattr(cfg, "final_export_dir", "")).strip()
    ):
        raise ValueError("export_final=true requires final_export_dir to be set.")


def _stage0_source_video_root(cfg: Any) -> str:
    """Return the root used to mirror source videos into the re-encoded cache."""
    dataset_format = _normalize_stage0_dataset_format(
        getattr(cfg, "dataset_format", "lerobot_root")
    )
    if dataset_format == "lerobot_root":
        return str(Path(cfg.dataset_root).expanduser().resolve())
    return str(Path(cfg.extract_cache_dir).expanduser().resolve())


class DatasetRoots:
    """Resolve explicitly configured, already-extracted LeRobot dataset roots."""

    def __init__(self, cfg):
        self.cfg = cfg

    def prepare(self):
        raw = self.cfg.dataset_dirs or self.cfg.dataset_root
        if not raw:
            raise ValueError("dataset_dirs must name at least one LeRobot dataset root.")
        roots = [
            Path(part.strip()).expanduser().resolve()
            for part in str(raw).split(",")
            if part.strip()
        ]
        for root in roots:
            if not (root / "meta/info.json").is_file():
                raise FileNotFoundError(f"Missing LeRobot metadata: {root / 'meta/info.json'}")
        if len(set(roots)) != len(roots):
            raise ValueError("dataset_dirs contains duplicate dataset roots.")
        return roots


class VideoDataset(Dataset):
    """Video-only random clip dataset for extracted LeRobot v2.0/v2.1/v3.0 roots."""

    def __init__(self, task_roots: Sequence[Path | str], *, cfg: Stage0Config, split: str):
        """Build an episode-window index for training or validation.

        Args:
            task_roots: Extracted LeRobot task roots containing ``meta/info.json``.
            cfg: Stage 0 dataset and sampling configuration.
            split: Either ``"train"`` for random window sampling or ``"val"`` for
                deterministic window iteration.
        """
        if split not in {"train", "val"}:
            raise ValueError(f"split must be train or val, got {split}")
        self.cfg = cfg
        self.split = split
        self.clip_frames = int(cfg.clip_frames)
        self.raw_frame_stride = int(cfg.raw_frame_stride)
        self.required_span = (self.clip_frames - 1) * self.raw_frame_stride + 1
        self.sample_stride_frames = int(cfg.sample_stride_frames)
        self.height = int(cfg.height)
        self.width = int(cfg.width)
        self.video_backend = cfg.video_backend
        self.use_reencoded_video_cache = False
        self.extract_cache_dir = str(cfg.extract_cache_dir)
        self.source_video_root = _stage0_source_video_root(cfg)
        self.reencoded_video_cache_dir = ""
        self.camera_keys = _parse_stage0_camera_keys(
            getattr(cfg, "camera_keys", ROBOTWIN_CAMERA_KEYS),
            allow_single=bool(getattr(cfg, "single_view_adaptation_enabled", False)),
        )
        self.use_text_embedding_cache = bool(getattr(cfg, "use_text_embedding_cache", True))
        self.text_embedding_cache_dir = str(getattr(cfg, "text_embedding_cache_dir", ""))
        self.text_embedding_cache_dirs = _parse_stage0_text_cache_dirs(
            self.text_embedding_cache_dir
        )
        self.context_len = int(getattr(cfg, "tokenizer_max_len", 128))
        self.text_dim = int(getattr(cfg, "text_dim", 4096))
        self.cache_model_id = str(getattr(cfg, "model_id", "Wan-AI/Wan2.2-TI2V-5B"))
        if self.use_text_embedding_cache:
            if not self.text_embedding_cache_dirs:
                raise ValueError(
                    "Stage 0 use_text_embedding_cache=true requires text_embedding_cache_dir."
                )
            missing_cache_dirs = [
                str(path) for path in self.text_embedding_cache_dirs if not path.exists()
            ]
            if missing_cache_dirs:
                raise FileNotFoundError(
                    f"Stage 0 text embedding cache directories not found: {missing_cache_dirs}"
                )
        self.prompt_policy = str(cfg.prompt_policy).strip().lower()
        if self.prompt_policy not in {"task_text", "generic"}:
            raise ValueError(
                f"Unsupported prompt_policy={cfg.prompt_policy}. Expected task_text or generic."
            )
        if self.clip_frames <= 1 or self.clip_frames % 4 != 1:
            raise ValueError(
                f"clip_frames must satisfy clip_frames > 1 and clip_frames % 4 == 1, got {self.clip_frames}"
            )
        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValueError(
                f"height/width must be multiples of 16, got {self.height}x{self.width}"
            )
        episodes = self._load_all_episodes([Path(root) for root in task_roots])
        episodes = self._split_episodes(episodes, split=split)
        if not episodes:
            raise RuntimeError(f"No usable {split} episodes found for stage-0 dataset.")
        self.episodes = episodes
        self.window_counts = [
            max((episode.length - self.required_span) // self.sample_stride_frames + 1, 0)
            for episode in self.episodes
        ]
        keep = [idx for (idx, count) in enumerate(self.window_counts) if count > 0]
        self.episodes = [self.episodes[idx] for idx in keep]
        self.window_counts = [self.window_counts[idx] for idx in keep]
        if not self.episodes:
            raise RuntimeError(
                f"No {split} episodes are long enough for required_span={self.required_span} frames."
            )
        cumulative = []
        running = 0
        for count in self.window_counts:
            running += int(count)
            cumulative.append(running)
        self.cumulative_windows = cumulative
        self.total_windows = int(running)
        if split == "train":
            self.epoch_size = min(int(cfg.train_epoch_size), self.total_windows)
        else:
            self.epoch_size = min(int(cfg.val_epoch_size), self.total_windows)
        self.epoch_size = max(self.epoch_size, 1)
        logger.info(
            "Built %s dataset: task_roots=%d episodes=%d total_windows=%d epoch_size=%d",
            split,
            len(task_roots),
            len(self.episodes),
            self.total_windows,
            self.epoch_size,
        )

    def _load_all_episodes(self, roots: list[Path]) -> list[Stage0Episode]:
        """Load all usable episodes from extracted task roots.

        Args:
            roots: Extracted LeRobot task roots.

        Returns:
            Episode metadata records for all roots.
        """
        episodes: list[Stage0Episode] = []
        for root in roots:
            episodes.extend(self._load_task_root(root))
        if not episodes:
            raise RuntimeError("No episodes loaded from extracted LeRobot roots.")
        return episodes

    def _camera_keys(self) -> tuple[str, ...]:
        """Return the configured ordered camera keys."""
        return _parse_stage0_camera_keys(
            getattr(self, "camera_keys", ROBOTWIN_CAMERA_KEYS),
            allow_single=bool(getattr(self.cfg, "single_view_adaptation_enabled", False)),
        )

    @staticmethod
    def _normalize_lerobot_version(root: Path, info: dict[str, Any]) -> str:
        """Return the supported LeRobot codebase version for one dataset root."""
        raw_version = info.get("codebase_version")
        if raw_version is None:
            raise ValueError(
                f"Missing codebase_version in {root / 'meta' / 'info.json'}. Supported versions: {sorted(SUPPORTED_LEROBOT_CODEBASE_VERSIONS)}"
            )
        version = str(raw_version).strip().lower()
        if version and version[0].isdigit():
            version = f"v{version}"
        if version not in SUPPORTED_LEROBOT_CODEBASE_VERSIONS:
            raise ValueError(
                f"Unsupported LeRobot codebase_version={raw_version!r} in {root / 'meta' / 'info.json'}. Supported versions: {sorted(SUPPORTED_LEROBOT_CODEBASE_VERSIONS)}"
            )
        return version

    def _load_task_root(self, root: Path) -> list[Stage0Episode]:
        """Load episodes and camera paths from one LeRobot task root.

        Args:
            root: Extracted task root containing ``meta`` metadata.

        Returns:
            Stage 0 episode records with resolved camera video paths.
        """
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing LeRobot info.json: {info_path}")
        info = _json_load(info_path)
        features = info.get("features", {})
        camera_keys = _stage0_select_camera_keys_for_features(self.cfg, features, source=info_path)
        configured_camera_keys = self._camera_keys()
        if tuple(camera_keys) != tuple(configured_camera_keys):
            logger.info(
                "Using root-specific Stage 0 camera keys for %s: configured=%s resolved=%s",
                root,
                ",".join(configured_camera_keys),
                ",".join(camera_keys),
            )
        version = self._normalize_lerobot_version(root, info)
        fps = int(info["fps"])
        chunks_size = int(info.get("chunks_size", 1000))
        video_template = str(info["video_path"])
        if version in {"v2.0", "v2.1"}:
            task_by_index = self._load_tasks_v21(root)
            rows = self._load_episode_rows_v21(root)
        elif version == "v3.0":
            task_by_index = self._load_tasks_v30(root)
            rows = self._load_episode_rows_v30(root, camera_keys=camera_keys)
        else:
            raise AssertionError(f"Unhandled LeRobot codebase_version={version}")
        default_task = next(iter(task_by_index.values()), root.name.replace("_", " "))
        loaded: list[Stage0Episode] = []
        for row in rows:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            task_candidates = self._episode_task_candidates(row, task_by_index, default_task)
            task_candidates = _normalize_stage0_task_candidates_for_source(
                self.cfg, task_candidates, camera_keys=camera_keys
            )
            task = task_candidates[0]
            video_paths = {
                key: str(
                    self._resolve_video_path(
                        root=root,
                        template=video_template,
                        video_key=key,
                        episode_index=episode_index,
                        chunks_size=chunks_size,
                        row=row,
                        version=version,
                    )
                )
                for key in camera_keys
            }
            video_start_times = self._video_start_times(
                row, version=version, camera_keys=camera_keys
            )
            loaded.append(
                Stage0Episode(
                    root=str(root),
                    episode_index=episode_index,
                    length=length,
                    fps=fps,
                    task=task,
                    video_paths=video_paths,
                    task_candidates=task_candidates,
                    video_start_times=video_start_times,
                )
            )
        return loaded

    @staticmethod
    def _load_tasks_v21(root: Path) -> dict[int, str]:
        """Load LeRobot v2.1 task metadata from JSONL."""
        tasks_jsonl = root / "meta" / "tasks.jsonl"
        if not tasks_jsonl.exists():
            raise FileNotFoundError(f"Missing LeRobot v2.1 tasks.jsonl: {tasks_jsonl}")
        mapping = {}
        with tasks_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if "task_index" not in record or "task" not in record:
                    raise KeyError(f"{tasks_jsonl} records must contain task_index and task.")
                mapping[int(record["task_index"])] = str(record["task"])
        if not mapping:
            raise RuntimeError(f"No task records loaded from {tasks_jsonl}")
        return mapping

    @staticmethod
    def _load_tasks_v30(root: Path) -> dict[int, str]:
        """Load LeRobot v3.0 task metadata from parquet."""
        tasks_path = root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing LeRobot v3.0 tasks.parquet: {tasks_path}")
        table = pq.read_table(tasks_path)
        data = table.to_pydict()
        task_column = "task"
        if task_column not in data:
            if "__index_level_0__" in data:
                task_column = "__index_level_0__"
                logger.debug(
                    "Using pandas index column as task text for %s because `task` is missing.",
                    tasks_path,
                )
            else:
                raise KeyError(
                    f"Missing `task` column in {tasks_path}; available columns: {list(data)}"
                )
        if "task_index" in data:
            mapping = {
                int(idx): str(task)
                for (idx, task) in zip(data["task_index"], data[task_column], strict=False)
            }
        else:
            mapping = {idx: str(task) for (idx, task) in enumerate(data[task_column])}
        if not mapping:
            raise RuntimeError(f"No task records loaded from {tasks_path}")
        return mapping

    @staticmethod
    def _load_episode_rows_v21(root: Path) -> list[dict[str, Any]]:
        """Load LeRobot v2.1 per-episode metadata from JSONL."""
        episodes_jsonl = root / "meta" / "episodes.jsonl"
        if not episodes_jsonl.exists():
            raise FileNotFoundError(f"Missing LeRobot v2.1 episodes.jsonl: {episodes_jsonl}")
        rows = []
        with episodes_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if "episode_index" not in record or "length" not in record:
                    raise KeyError(
                        f"{episodes_jsonl} records must contain episode_index and length."
                    )
                rows.append(record)
        if not rows:
            raise RuntimeError(f"No episode records loaded from {episodes_jsonl}")
        return sorted(rows, key=lambda item: int(item["episode_index"]))

    def _load_episode_rows_v30(
        self, root: Path, *, camera_keys: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        """Load LeRobot v3.0 per-episode metadata from parquet shards."""
        rows: list[dict[str, Any]] = []
        episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
        if not episode_files:
            raise FileNotFoundError(
                f"Missing LeRobot v3.0 episode parquet shards under {root / 'meta'}"
            )
        for path in episode_files:
            schema_names = set(pq.read_schema(path).names)
            required_columns = ["episode_index", "length"]
            for key in camera_keys:
                required_columns.extend(
                    [
                        f"videos/{key}/chunk_index",
                        f"videos/{key}/file_index",
                        f"videos/{key}/from_timestamp",
                    ]
                )
            missing = [key for key in required_columns if key not in schema_names]
            if missing:
                raise KeyError(f"{path} is missing required LeRobot v3.0 columns: {missing}")
            optional_columns = ["tasks", "task_index", "atomic_task_index", "coarse_task_index"]
            columns = required_columns + [key for key in optional_columns if key in schema_names]
            table = pq.read_table(path, columns=columns)
            data = table.to_pydict()
            for i in range(table.num_rows):
                rows.append({key: values[i] for (key, values) in data.items()})
        if not rows:
            raise RuntimeError(
                f"No episode records loaded from LeRobot v3.0 shards under {root / 'meta'}"
            )
        return sorted(rows, key=lambda item: int(item["episode_index"]))

    @staticmethod
    def _episode_task(row: dict[str, Any], task_by_index: dict[int, str], default_task: str) -> str:
        """Resolve the natural-language task string for one episode row.

        Args:
            row: Episode metadata row.
            task_by_index: Mapping loaded from task metadata.
            default_task: Fallback task name derived from the task root.

        Returns:
            Task text used to form the training prompt.
        """
        return VideoDataset._episode_task_candidates(row, task_by_index, default_task)[0]

    @staticmethod
    def _episode_task_candidates(
        row: dict[str, Any], task_by_index: dict[int, str], default_task: str
    ) -> tuple[str, ...]:
        return resolve_episode_task_candidates(
            row=row,
            task_by_index=task_by_index,
            default_task=default_task,
            strict=False,
            stage_label="Stage 0",
        )

    @staticmethod
    def _resolve_video_path(
        *,
        root: Path,
        template: str,
        video_key: str,
        episode_index: int,
        chunks_size: int,
        version: str,
        row: dict[str, Any] | None = None,
    ) -> Path:
        """Resolve a camera video path from a LeRobot video template.

        Args:
            root: Extracted task root.
            template: LeRobot ``video_path`` format string.
            video_key: Camera feature key such as ``images.rgb.head``.
            episode_index: Episode index to load.
            chunks_size: Episodes per chunk in the dataset layout.
            version: Normalized LeRobot codebase version.
            row: Optional episode metadata row. Required for LeRobot v3.0.

        Returns:
            Existing video path for the camera/episode pair.
        """
        if version in {"v2.0", "v2.1"}:
            chunk_index = int(episode_index) // int(chunks_size)
            file_index = int(episode_index) % int(chunks_size)
        elif version == "v3.0":
            if row is None:
                raise KeyError(
                    "LeRobot v3.0 video path resolution requires an episode metadata row."
                )
            packed_chunk_key = f"videos/{video_key}/chunk_index"
            packed_file_key = f"videos/{video_key}/file_index"
            missing = [key for key in (packed_chunk_key, packed_file_key) if key not in row]
            if missing:
                raise KeyError(
                    f"LeRobot v3.0 episode {episode_index}, camera {video_key} is missing video index columns: {missing}"
                )
            chunk_index = int(row[packed_chunk_key])
            file_index = int(row[packed_file_key])
        else:
            raise AssertionError(f"Unhandled LeRobot codebase_version={version}")
        try:
            rel = template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=int(file_index),
                episode_index=int(episode_index),
                episode_chunk=chunk_index,
            )
        except KeyError as err:
            raise KeyError(f"Unsupported video_path placeholder in template {template}") from err
        path = root / rel
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Missing video file for episode {episode_index}, camera {video_key}. Tried: {path}"
        )

    def _video_start_times(
        self, row: dict[str, Any], *, version: str, camera_keys: tuple[str, ...]
    ) -> dict[str, float]:
        """Read per-camera timestamp offsets for the selected LeRobot version."""
        if version in {"v2.0", "v2.1"}:
            return {key: 0.0 for key in camera_keys}
        if version == "v3.0":
            return {key: VideoDataset._video_start_time_v30(row, key) for key in camera_keys}
        raise AssertionError(f"Unhandled LeRobot codebase_version={version}")

    @staticmethod
    def _video_start_time_v30(row: dict[str, Any], video_key: str) -> float:
        """Read the episode offset inside a LeRobot v3.0 packed camera video."""
        key = f"videos/{video_key}/from_timestamp"
        if key not in row or row[key] is None:
            raise KeyError(f"LeRobot v3.0 episode row is missing {key}")
        return float(row[key])

    def _split_episodes(self, episodes: list[Stage0Episode], *, split: str) -> list[Stage0Episode]:
        """Create a deterministic episode-level train/val split.

        Args:
            episodes: Full episode list.
            split: Target split name.

        Returns:
            Episodes belonging to the requested split.
        """
        val_fraction = float(self.cfg.val_fraction)
        if val_fraction <= 0.0:
            if split == "val":
                return []
            return episodes
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
        rng = random.Random(int(self.cfg.seed))
        indices = list(range(len(episodes)))
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_fraction)))
        val_set = set(indices[:val_count])
        if split == "val":
            return [episode for (idx, episode) in enumerate(episodes) if idx in val_set]
        return [episode for (idx, episode) in enumerate(episodes) if idx not in val_set]

    def __len__(self) -> int:
        """Return the capped number of samples exposed per epoch.

        Returns:
            Epoch size for the configured split.
        """
        return int(self.epoch_size)

    def _window_from_flat_index(self, flat_index: int) -> tuple[Stage0Episode, int]:
        """Map a flat window index to an episode and start frame.

        Args:
            flat_index: Global sliding-window id.

        Returns:
            Pair of episode metadata and starting raw frame index.
        """
        flat_index = int(flat_index) % self.total_windows
        episode_idx = bisect.bisect_right(self.cumulative_windows, flat_index)
        prev = 0 if episode_idx == 0 else self.cumulative_windows[episode_idx - 1]
        window_idx = flat_index - prev
        start_frame = int(window_idx) * self.sample_stride_frames
        return (self.episodes[episode_idx], start_frame)

    def _sample_window(self, idx: int) -> tuple[Stage0Episode, int]:
        """Select a training or validation window.

        Args:
            idx: Dataloader-provided item index.

        Returns:
            Episode metadata and raw start frame for the clip.
        """
        return self._window_from_flat_index(int(idx))

    def _prompt_for_task(self, task: str) -> str:
        """Format the prompt for a task string.

        Args:
            task: Dataset task text.

        Returns:
            Prompt string consumed by the Wan text encoder.
        """
        if self.prompt_policy == "generic":
            return DEFAULT_PROMPT.format(task="a robot manipulation task")
        task_text = str(task).strip()
        if not task_text:
            raise ValueError("Stage 0 task text must be non-empty.")
        return DEFAULT_PROMPT.format(task=task_text)

    @staticmethod
    def _select_episode_task(episode: Stage0Episode) -> str:
        candidates = tuple(
            (str(task).strip() for task in episode.task_candidates if str(task).strip())
        )
        if candidates:
            return random.choice(candidates)
        return str(episode.task).strip()

    def _decode_camera(
        self, video_key: str, path: str, episode: Stage0Episode, start_frame: int
    ) -> torch.Tensor:
        """Decode one camera stream for a sampled clip.

        Args:
            video_key: Camera feature key such as ``images.rgb.head``.
            path: Video file path for the selected camera.
            episode: Episode metadata containing fps.
            start_frame: Raw frame index for the first sampled frame.

        Returns:
            Float video tensor with shape ``[T, 3, H, W]``.
        """
        timestamps = [
            float(episode.video_start_times.get(video_key, 0.0))
            + (start_frame + frame_idx * self.raw_frame_stride) / float(episode.fps)
            for frame_idx in range(self.clip_frames)
        ]
        tolerance_s = max(1.0 / float(episode.fps) - 0.0001, 0.0001)
        decode_path = path
        frames = decode_video_frames(
            video_path=decode_path,
            timestamps=timestamps,
            tolerance_s=tolerance_s,
            backend=self.video_backend,
        )
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                f"Decoded video frames must have shape [T,3,H,W], got {tuple(frames.shape)}"
            )
        return frames

    @staticmethod
    def _resize_video(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize all frames in a ``[T, C, H, W]`` video tensor.

        Args:
            video: Video tensor to resize.
            size: Output ``(height, width)``.

        Returns:
            Resized video tensor with unchanged ``T`` and ``C`` dimensions.
        """
        return TVF.resize(
            video, size=list(size), interpolation=TVF.InterpolationMode.BILINEAR, antialias=True
        )

    @staticmethod
    def _center_crop_resize_video(
        video: torch.Tensor, size: tuple[int, int], *, crop_scale: float
    ) -> torch.Tensor:
        """Center-crop a video tensor before resizing it to the target size."""
        if not 0.0 < float(crop_scale) <= 1.0:
            raise ValueError(f"crop_scale must be in (0, 1], got {crop_scale}.")
        if video.ndim != 4:
            raise ValueError(
                f"Expected video tensor with shape [T,C,H,W], got {tuple(video.shape)}."
            )
        height = int(video.shape[-2])
        width = int(video.shape[-1])
        crop_h = max(1, min(height, int(round(height * float(crop_scale)))))
        crop_w = max(1, min(width, int(round(width * float(crop_scale)))))
        top = max((height - crop_h) // 2, 0)
        left = max((width - crop_w) // 2, 0)
        cropped = video[..., top : top + crop_h, left : left + crop_w]
        return VideoDataset._resize_video(cropped, size)

    @staticmethod
    def _center_square_resize_video(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Center-crop a video to a square before resizing."""
        if video.ndim != 4:
            raise ValueError(
                f"Expected video tensor with shape [T,C,H,W], got {tuple(video.shape)}."
            )
        height = int(video.shape[-2])
        width = int(video.shape[-1])
        crop_size = min(height, width)
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        cropped = video[..., top : top + crop_size, left : left + crop_size]
        return VideoDataset._resize_video(cropped, size)

    def _compose_droid_grid(
        self, cameras: dict[str, torch.Tensor], camera_keys: tuple[str, ...]
    ) -> torch.Tensor:
        if tuple(camera_keys) != tuple(DROID_CAMERA_KEYS):
            raise ValueError(
                f"DROID grid layout requires camera_keys={','.join(DROID_CAMERA_KEYS)!r}, got {','.join(camera_keys)!r}."
            )
        if (self.height, self.width) != (DROID_GRID_HEIGHT, DROID_GRID_WIDTH):
            raise ValueError(
                f"DROID grid layout requires {DROID_GRID_HEIGHT}x{DROID_GRID_WIDTH}, got {self.height}x{self.width}."
            )
        return self._compose_wrist_top_grid(cameras, DROID_CAMERA_KEYS)

    def _compose_wrist_top_grid(
        self, cameras: dict[str, torch.Tensor], camera_keys: tuple[str, ...]
    ) -> torch.Tensor:
        if len(camera_keys) != 3:
            raise ValueError(
                f"Wrist-top grid layout requires 3 camera keys, got {len(camera_keys)}."
            )
        (left_key, right_key, wrist_key) = camera_keys
        per_camera_h = DROID_GRID_HEIGHT // 2
        per_camera_w = DROID_GRID_WIDTH // 2
        left = self._center_crop_resize_video(
            cameras[left_key], (per_camera_h, per_camera_w), crop_scale=0.95
        )
        right = self._center_crop_resize_video(
            cameras[right_key], (per_camera_h, per_camera_w), crop_scale=0.95
        )
        wrist = self._center_crop_resize_video(
            cameras[wrist_key], (per_camera_h, per_camera_w), crop_scale=0.95
        )
        wrist_wide = wrist.repeat_interleave(2, dim=-1)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([wrist_wide, bottom], dim=-2)

    def _compose_multicam(self, cameras: dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack configured camera streams into one training view.

        Args:
            cameras: Mapping from each required camera key to ``[T, 3, H, W]`` video.

        Returns:
            Composed video tensor with shape ``[T, 3, height, width]``.
        """
        camera_keys = self._camera_keys_for_available_tensors(cameras)
        missing = [key for key in camera_keys if key not in cameras]
        if missing:
            raise KeyError(f"Missing configured Stage 0 camera tensors: {missing}")
        concat_mode = (
            str(getattr(self.cfg, "concat_multi_camera", ROBOTWIN_CONCAT_MULTI_CAMERA))
            .strip()
            .lower()
        )
        if len(camera_keys) == 3 and concat_mode == ROBOTWIN_CONCAT_MULTI_CAMERA:
            (top_key, left_key, right_key) = camera_keys
            head = self._resize_video(cameras[top_key], (256, 320))
            left = self._resize_video(cameras[left_key], (128, 160))
            right = self._resize_video(cameras[right_key], (128, 160))
            bottom = torch.cat([left, right], dim=-1)
            video = torch.cat([head, bottom], dim=-2)
        elif concat_mode in ROBOCASA_DREAMZERO_LAYOUTS:
            if tuple(camera_keys) != tuple(ROBOCASA_CAMERA_KEYS):
                raise ValueError(
                    f"Robocasa DreamZero layout requires camera_keys={','.join(ROBOCASA_CAMERA_KEYS)!r}, got {','.join(camera_keys)!r}."
                )
            spec = _stage0_robocasa_layout_spec(self.cfg)
            expected_size = (int(spec["height"]), int(spec["width"]))
            tile = int(spec["tile"])
            if (self.height, self.width) != expected_size:
                raise ValueError(
                    f"Robocasa DreamZero layout requires {expected_size[0]}x{expected_size[1]}, got {self.height}x{self.width}."
                )
            (left_key, right_key, wrist_key) = ROBOCASA_CAMERA_KEYS
            wrist = self._resize_video(cameras[wrist_key], (tile, self.width))
            left = self._resize_video(cameras[left_key], (tile, tile))
            right = self._resize_video(cameras[right_key], (tile, tile))
            bottom = torch.cat([left, right], dim=-1)
            video = torch.cat([wrist, bottom], dim=-2)
        elif concat_mode == DROID_GRID_CONCAT_MULTI_CAMERA:
            video = self._compose_droid_grid(cameras, camera_keys)
        elif concat_mode == HORIZONTAL_CONCAT_MULTI_CAMERA:
            if self.width % len(camera_keys) != 0:
                raise ValueError(
                    f"Horizontal multi-camera concat requires width divisible by camera count; width={self.width}, cameras={len(camera_keys)}."
                )
            per_camera_width = self.width // len(camera_keys)
            resized = [
                self._resize_video(cameras[key], (self.height, per_camera_width))
                for key in camera_keys
            ]
            video = torch.cat(resized, dim=-1)
        elif concat_mode == VERTICAL_CONCAT_MULTI_CAMERA:
            if self.height % len(camera_keys) != 0:
                raise ValueError(
                    f"Vertical multi-camera concat requires height divisible by camera count; height={self.height}, cameras={len(camera_keys)}."
                )
            per_camera_height = self.height // len(camera_keys)
            resized = [
                self._resize_video(cameras[key], (per_camera_height, self.width))
                for key in camera_keys
            ]
            video = torch.cat(resized, dim=-2)
        else:
            raise ValueError(
                f"Unsupported concat_multi_camera={concat_mode!r} for {len(camera_keys)} configured cameras. Expected robotwin, robocasa_dreamzero_352, droid_grid, or horizontal/vertical."
            )
        if tuple(video.shape[-2:]) != (self.height, self.width):
            video = self._resize_video(video, (self.height, self.width))
        return video

    def _camera_keys_for_available_tensors(
        self, cameras: dict[str, torch.Tensor]
    ) -> tuple[str, ...]:
        """Resolve the ordered camera keys for one loaded episode tensor mapping."""
        for camera_keys in _stage0_camera_key_candidates(self.cfg):
            if all((key in cameras for key in camera_keys)):
                return camera_keys
        available_keys = sorted(cameras)
        expected = ["[" + ", ".join(keys) + "]" for keys in _stage0_camera_key_candidates(self.cfg)]
        raise KeyError(
            f"Missing configured Stage 0 camera tensors. Expected one of {expected}; available camera tensors={available_keys}."
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one normalized video clip and its text prompt.

        Args:
            idx: Dataloader-provided item index.

        Returns:
            Sample dict whose ``video`` has shape ``[3, T, H, W]`` in ``[-1, 1]``.
        """
        (episode, start_frame) = self._sample_window(idx)
        cameras = {
            key: self._decode_camera(key, path, episode, start_frame)
            for (key, path) in episode.video_paths.items()
        }
        video = self._compose_multicam(cameras)
        video = video.mul(2.0).sub(1.0)
        video = video.permute(1, 0, 2, 3).contiguous()
        task = self._select_episode_task(episode)
        prompt = self._prompt_for_task(task)
        data = {
            "video": video,
            "prompt": prompt,
            "task": task,
            "episode_index": torch.tensor(episode.episode_index, dtype=torch.long),
            "start_frame": torch.tensor(start_frame, dtype=torch.long),
        }
        if self.use_text_embedding_cache:
            (context, context_mask) = _load_stage0_text_cache(
                cache_dir=self.text_embedding_cache_dirs,
                prompt=prompt,
                context_len=self.context_len,
                text_dim=self.text_dim,
                model_id=self.cache_model_id,
            )
            data["context"] = context
            data["context_mask"] = context_mask
        return data


class _Stage0EpisodeHandleCache:
    """Small LRU cache of open safetensors handles for cached latent episodes."""

    def __init__(self, maxsize: int = 64):
        self.maxsize = int(maxsize)
        self._store: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path):
        handle = self._store.get(path)
        if handle is None:
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._store[path] = handle
            if len(self._store) > self.maxsize:
                (_, victim) = self._store.popitem(last=False)
                try:
                    victim.close()
                except AttributeError:
                    pass
            return handle
        self._store.move_to_end(path)
        return handle


class CachedStage0LatentDataset(Dataset):
    """Stage 0 dataset wrapper that serves precomputed VAE latents instead of decoded video."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | Sequence[str | Path],
        source_dataset: VideoDataset,
        expected_cache_contract: dict[str, Any],
    ):
        super().__init__()
        self.cache_dirs = [
            Path(path).expanduser().resolve() for path in _parse_stage0_latent_cache_dirs(cache_dir)
        ]
        if not self.cache_dirs:
            raise ValueError(
                "CachedStage0LatentDataset requires at least one latent cache directory."
            )
        self.cache_dir = self.cache_dirs[0]
        self.source_dataset = source_dataset
        self.expected_cache_contract = dict(expected_cache_contract)
        self.episodes = list(source_dataset.episodes)
        self.window_counts = list(source_dataset.window_counts)
        self.total_windows = int(getattr(source_dataset, "total_windows"))
        self.epoch_size = int(len(source_dataset))
        self.context_len = int(source_dataset.context_len)
        self.text_dim = int(source_dataset.text_dim)
        self.cache_model_id = str(source_dataset.cache_model_id)
        self.text_embedding_cache_dir = str(source_dataset.text_embedding_cache_dir)
        self.text_embedding_cache_dirs = list(source_dataset.text_embedding_cache_dirs)
        self.use_text_embedding_cache = bool(source_dataset.use_text_embedding_cache)
        missing_cache_dirs = [
            str(cache_dir) for cache_dir in self.cache_dirs if not cache_dir.exists()
        ]
        if missing_cache_dirs:
            raise FileNotFoundError(
                f"Stage 0 video latent cache directories not found: {missing_cache_dirs}. Precompute matching caches first or set use_cached_latents=false."
            )
        roots = sorted(
            {Path(episode.root).expanduser().resolve() for episode in source_dataset.episodes}
        )
        self._root_cache_names = _stage0_root_cache_name_map(roots)
        self._single_root_cache = len(roots) == 1
        self._validate_episode_cache_files()
        self._handles = _Stage0EpisodeHandleCache(maxsize=64)

    def __len__(self) -> int:
        return self.epoch_size

    def _episode_path_candidates(self, episode: Stage0Episode) -> list[Path]:
        filename = f"episode_{int(episode.episode_index):06d}.safetensors"
        root = Path(episode.root).expanduser().resolve()
        root_name = self._root_cache_names.get(root)
        if root_name is None:
            raise KeyError(f"Stage 0 cached latent root was not registered: {root}")
        root_names = []
        for candidate in (root_name, root.name, _stage0_root_cache_name(root)):
            if candidate not in root_names:
                root_names.append(candidate)
        candidates: list[Path] = []
        for cache_dir in self.cache_dirs:
            flat_path = cache_dir / filename
            if self._single_root_cache or _stage0_cache_dir_can_use_flat_episodes(cache_dir, root):
                candidates.append(flat_path)
            candidates.extend((cache_dir / candidate / filename for candidate in root_names))
        return candidates

    def _episode_path(self, episode: Stage0Episode) -> Path:
        candidates = self._episode_path_candidates(episode)
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def _latent_handle_for_cache_path(self, path: Path, handle: Any) -> tuple[Any, Path]:
        return (handle, path)

    def _validate_episode_cache_files(self) -> None:
        missing: list[str] = []
        too_short: list[str] = []
        first_existing: Path | None = None
        for episode in self.source_dataset.episodes:
            path = self._episode_path(episode)
            if not path.exists():
                root = Path(episode.root).expanduser().resolve()
                root_name = self._root_cache_names.get(root, Path(episode.root).name)
                candidate_dirs = sorted(
                    {str(path.parent) for path in self._episode_path_candidates(episode)}
                )
                missing.append(
                    f"{root_name}/episode_{int(episode.episode_index):06d} candidates={candidate_dirs}"
                )
                continue
            if first_existing is None:
                first_existing = path
            (meta, latent_shape, _) = _stage0_cached_or_overlay_latent_header(path)
            if _is_droid_cache_meta(meta):
                _validate_droid_dreamzero_latent_cache_metadata(path, stage_label="Stage 0")
            elif self.expected_cache_contract.get("visual_preprocess_id") is not None:
                validate_stage2_latent_cache_header_file(
                    path, expected=self.expected_cache_contract, stage_label="Stage 0"
                )
            latent_count = int(latent_shape[0])
            num_samples = int(meta.get("num_samples", latent_count))
            if num_samples != latent_count:
                raise ValueError(
                    f"Stage 0 latent cache metadata/sample mismatch in {path}: num_samples={num_samples} latent_shape[0]={latent_count}."
                )
            if num_samples < int(episode.length):
                too_short.append(
                    f"episode={episode.episode_index} cache_samples={num_samples} episode_length={episode.length}"
                )
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} Stage 0 cached episode files are missing under {self.cache_dirs}. First few: {missing[:5]}. Precompute matching caches first or set use_cached_latents=false."
            )
        if first_existing is None:
            raise FileNotFoundError(
                f"No Stage 0 cached episode files found under {self.cache_dirs}."
            )
        for cache_dir in self.cache_dirs:
            first_for_cache = _first_stage0_latent_cache_file(cache_dir)
            if first_for_cache is None:
                raise FileNotFoundError(f"No Stage 0 cached episode files found under {cache_dir}.")
            _validate_cached_or_droid_dreamzero_latent_cache_file(
                first_for_cache, expected=self.expected_cache_contract, stage_label="Stage 0"
            )
        if too_short:
            raise ValueError(
                f"Stage 0 cached latents must cover full episodes. First mismatches: {too_short[:5]}. Precompute a full-coverage cache or set use_cached_latents=false."
            )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        (episode, start_frame) = self.source_dataset._sample_window(idx)
        path = self._episode_path(episode)
        handle = self._handles.get(path)
        (latent_handle, latent_path) = self._latent_handle_for_cache_path(path, handle)
        latents_slice = latent_handle.get_slice("latents")
        latent_count = int(latents_slice.get_shape()[0])
        if int(start_frame) < 0 or int(start_frame) >= latent_count:
            raise IndexError(
                f"Stage 0 latent cache frame index out of range: episode={episode.episode_index} start_frame={start_frame} latent_count={latent_count} cache={path} latent_source={latent_path}."
            )
        latent = latents_slice[int(start_frame) : int(start_frame) + 1, ...]
        if not isinstance(latent, torch.Tensor):
            raise RuntimeError(f"Unexpected latent slice type {type(latent)} from {latent_path}.")
        latent = latent.squeeze(0).contiguous()
        i2v_condition_latents = None
        if str(self.expected_cache_contract.get("video_model_family")) == "wan22_i2v_a14b":
            try:
                i2v_slice = latent_handle.get_slice(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)
            except Exception as exc:
                raise KeyError(
                    f"14B Stage 0 cached-latent training requires tensor {WAN22_A14B_I2V_CONDITION_LATENTS_KEY!r} in {latent_path}. Regenerate the cache with official Wan2.2 I2V conditioning."
                ) from exc
            i2v_condition = i2v_slice[int(start_frame) : int(start_frame) + 1, ...]
            if not isinstance(i2v_condition, torch.Tensor):
                raise RuntimeError(
                    f"Unexpected {WAN22_A14B_I2V_CONDITION_LATENTS_KEY} slice type {type(i2v_condition)} from {latent_path}."
                )
            i2v_condition_latents = i2v_condition.squeeze(0).contiguous()
        task = self.source_dataset._select_episode_task(episode)
        prompt = self.source_dataset._prompt_for_task(task)
        data = {
            "input_latents": latent,
            "prompt": prompt,
            "task": task,
            "episode_index": torch.tensor(episode.episode_index, dtype=torch.long),
            "start_frame": torch.tensor(start_frame, dtype=torch.long),
            "sample_id": f"{episode.root}|episode={episode.episode_index}|start={start_frame}|frames={self.source_dataset.clip_frames}|stride={self.source_dataset.raw_frame_stride}",
        }
        if i2v_condition_latents is not None:
            data[WAN22_A14B_I2V_CONDITION_LATENTS_KEY] = i2v_condition_latents
        if self.use_text_embedding_cache:
            (context, context_mask) = _load_stage0_text_cache(
                cache_dir=self.text_embedding_cache_dirs,
                prompt=prompt,
                context_len=self.context_len,
                text_dim=self.text_dim,
                model_id=self.cache_model_id,
            )
            data["context"] = context
            data["context_mask"] = context_mask
        return data


def build_stage0_dataset(task_roots, cfg, *, split):
    live = VideoDataset(task_roots, cfg=cfg, split=split)
    if not cfg.use_cached_latents:
        return live
    return CachedStage0LatentDataset(
        cache_dir=resolve_stage0_latent_cache_dirs(cfg),
        source_dataset=live,
        expected_cache_contract=_stage0_latent_cache_expected_contract(cfg),
    )


def _stage0_14b_frame0_mode_from_state(state: dict[str, Any]) -> str:
    """Read 14B frame-0 mode from checkpoint metadata.

    Legacy 14B Stage 0 checkpoints predate the explicit flag and used the
    clean-overwrite behavior.
    """
    value = state.get("latent_frame0_mode")
    if value is None:
        config = state.get("config")
        if isinstance(config, dict):
            value = config.get("latent_frame0_mode")
    if value is None:
        value = "clean_overwrite"
    return _normalize_latent_frame0_mode(value)
