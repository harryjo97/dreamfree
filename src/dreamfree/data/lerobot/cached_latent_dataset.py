"""Full-coverage video latent, action, and state cache reader for stage 2."""

from __future__ import annotations

import functools
import hashlib
import random
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from dreamfree.data import yam
from dreamfree.data.text_cache import DEFAULT_PROMPT, validate_text_cache
from dreamfree.models.wan22.i2v_condition import (
    WAN22_A14B_I2V_CONDITION_LATENTS_KEY,
    WAN22_A14B_I2V_CONDITION_SOURCE_OFFICIAL,
)
from dreamfree.utils.logging_config import get_logger

logger = get_logger(__name__)
STAGE2_REQUIRED_CACHE_KEYS = ("context", "mask")
STAGE2_LATENT_CACHE_VERSION = 3
STAGE2_LATENT_CACHE_FULL_COVERAGE = "full_episode"
STAGE2_LATENT_CACHE_PADDING_POLICY = "clamp_to_episode_end"
DROID_DREAMZERO_ACTION_CONTRACT = "droid_dreamzero_rel_joint_abs_gripper_q99_v1"
DROID_GRIPPER_HISTORY_THRESHOLD = 0.5
DROID_DREAMZERO_OVERLAY_CACHE_KIND = "droid_dreamzero_action_overlay_v1"
DROID_DREAMZERO_REQUIRED_TENSOR_DIMS = {
    "droid_action_joint_position": 7,
    "droid_action_gripper_position": 1,
    "droid_state_joint_position": 7,
    "droid_state_gripper_position": 1,
}
STAGE2_WAN22_5B_ONLY_LATENT_CACHE_ROOT = Path("./assets").resolve()
STAGE2_TEXT_CACHE_SUFFIXES = {"wan22_ti2v_5b": "wan22ti2v5b", "wan22_i2v_a14b": "wan22i2va14b"}
STAGE2_LATENT_CACHE_CONTRACTS = {
    "wan22_ti2v_5b": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "vae_family": "wan2.2",
        "latent_channels": 48,
        "vae_z_dim": 48,
        "vae_temporal_stride": 4,
        "vae_spatial_stride": 16,
        "allow_legacy_metadata": False,
    },
    "wan22_i2v_a14b": {
        "model_id": "Wan-AI/Wan2.2-I2V-A14B",
        "vae_family": "wan2.1",
        "latent_channels": 16,
        "vae_z_dim": 16,
        "vae_temporal_stride": 4,
        "vae_spatial_stride": 8,
        "allow_legacy_metadata": False,
    },
}


def binarize_droid_gripper_action_history(
    gripper_actions: torch.Tensor, *, action_history_horizon: int
) -> torch.Tensor:
    """Binarize only the historical prefix of raw DROID gripper actions."""
    if not isinstance(gripper_actions, torch.Tensor):
        raise TypeError(
            f"DROID gripper actions must be a torch.Tensor, got {type(gripper_actions).__name__}."
        )
    if gripper_actions.ndim != 2 or int(gripper_actions.shape[1]) != 1:
        raise ValueError(
            f"DROID gripper actions must have shape [H+T,1], got {tuple(gripper_actions.shape)}."
        )
    history_horizon = int(action_history_horizon)
    if history_horizon < 0 or history_horizon > int(gripper_actions.shape[0]):
        raise ValueError(
            f"DROID action_history_horizon must be in [0,H+T], got horizon={history_horizon} actions={int(gripper_actions.shape[0])}."
        )
    result = gripper_actions.clone()
    if history_horizon > 0:
        result[:history_horizon] = (result[:history_horizon] > DROID_GRIPPER_HISTORY_THRESHOLD).to(
            dtype=result.dtype
        )
    return result


def _parse_path_list(
    value: str | Path | list[str | Path] | tuple[str | Path, ...], *, label: str
) -> list[Path]:
    if isinstance(value, (str, Path)):
        raw_parts = str(value).split(",")
    else:
        raw_parts = [str(part) for part in value]
    paths = [Path(part.strip()).expanduser().resolve() for part in raw_parts if part.strip()]
    if not paths:
        raise ValueError(f"{label} must contain at least one path.")
    return paths


def _dedupe_nonempty_task_texts(values: list[Any] | tuple[Any, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    candidates: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        candidates.append(text)
    return tuple(candidates)


def _task_candidates_from_episode_row(row: Any) -> tuple[str, ...]:
    if not isinstance(row, dict):
        return ()
    value = row.get("tasks")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        return _dedupe_nonempty_task_texts((value,))
    if not isinstance(value, (list, tuple)):
        return ()
    candidates: list[str] = []
    for item in value:
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, (list, tuple)):
            candidates.extend(_task_candidates_from_episode_row({"tasks": item}))
        else:
            candidates.append(item)
    return _dedupe_nonempty_task_texts(tuple(candidates))


def stage2_text_cache_suffix_for_family(video_model_family: str) -> str:
    family = str(video_model_family)
    try:
        return STAGE2_TEXT_CACHE_SUFFIXES[family]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported video_model_family={video_model_family!r}; expected one of {sorted(STAGE2_TEXT_CACHE_SUFFIXES)}."
        ) from exc


def _read_episode_header(path: Path) -> dict[str, Any]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
        latent_shape = handle.get_slice("latents").get_shape()
        tensor_shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
    return {"metadata": meta, "latent_shape": tuple(latent_shape), "tensor_shapes": tensor_shapes}


def _read_cached_or_overlay_episode_header(path: Path) -> dict[str, Any]:
    return _read_episode_header(path)


def _metadata_int(meta: dict[str, Any], key: str) -> int | None:
    raw = meta.get(key)
    if raw is None:
        return None
    return int(raw)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_stage2_latent_cache_path(
    path: Path, expected: dict[str, Any], *, stage_label: str
) -> None:
    resolved = Path(path).expanduser().resolve()
    if str(expected.get("video_model_family")) == "wan22_i2v_a14b" and _is_relative_to(
        resolved, STAGE2_WAN22_5B_ONLY_LATENT_CACHE_ROOT
    ):
        raise ValueError(
            f"{stage_label} cannot use {STAGE2_WAN22_5B_ONLY_LATENT_CACHE_ROOT} for video_model_family='wan22_i2v_a14b'. That cache was produced for Wan2.2-TI2V-5B with the Wan2.2 VAE; precompute a new Wan2.1-VAE 14B latent cache instead."
        )


def stage2_latent_cache_expected_contract(
    *,
    video_model_family: str,
    model_id: str | None,
    height: int,
    width: int,
    num_frames: int,
    action_video_freq_ratio: int,
    action_horizon: int | None = None,
    require_action_horizon_metadata: bool = False,
) -> dict[str, Any]:
    family = str(video_model_family)
    if family not in STAGE2_LATENT_CACHE_CONTRACTS:
        raise ValueError(
            f"Unsupported video_model_family={video_model_family!r}; expected one of {sorted(STAGE2_LATENT_CACHE_CONTRACTS)}."
        )
    base = dict(STAGE2_LATENT_CACHE_CONTRACTS[family])
    requested_model_id = str(model_id or base["model_id"])
    if requested_model_id != base["model_id"]:
        raise ValueError(
            f"{family} Stage 2 latent cache requires model_id={base['model_id']!r}, got {requested_model_id!r}."
        )
    if (
        int(height) % int(base["vae_spatial_stride"]) != 0
        or int(width) % int(base["vae_spatial_stride"]) != 0
    ):
        raise ValueError(
            f"{family} latent cache requires height/width divisible by VAE spatial stride {base['vae_spatial_stride']}, got {height}x{width}."
        )
    if int(num_frames) <= 1 or int(action_video_freq_ratio) <= 0:
        raise ValueError(
            f"num_frames and action_video_freq_ratio must be positive, got {num_frames} and {action_video_freq_ratio}."
        )
    if (int(num_frames) - 1) % int(action_video_freq_ratio) != 0:
        raise ValueError(
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {int(num_frames) - 1} and {action_video_freq_ratio}."
        )
    if action_horizon is not None and int(action_horizon) <= 0:
        raise ValueError(f"action_horizon must be positive when set, got {action_horizon}.")
    sampled_frames = (int(num_frames) - 1) // int(action_video_freq_ratio) + 1
    latent_t = (sampled_frames - 1) // int(base["vae_temporal_stride"]) + 1
    base.update(
        {
            "video_model_family": family,
            "model_id": requested_model_id,
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "action_video_freq_ratio": int(action_video_freq_ratio),
            "action_horizon": None if action_horizon is None else int(action_horizon),
            "require_action_horizon_metadata": bool(require_action_horizon_metadata),
            "latent_t_per_window": int(latent_t),
            "latent_h": int(height) // int(base["vae_spatial_stride"]),
            "latent_w": int(width) // int(base["vae_spatial_stride"]),
        }
    )
    return base


def _check_metadata_value(
    mismatches: list[str], *, meta: dict[str, Any], key: str, expected: Any, require: bool
) -> None:
    actual = meta.get(key)
    if actual is None:
        if require:
            mismatches.append(f"{key}: missing expected={expected!r}")
        return
    if str(actual) != str(expected):
        mismatches.append(f"{key}: actual={actual!r} expected={expected!r}")


def is_droid_dreamzero_runtime_horizon_cache(
    *, metadata: dict[str, Any], tensor_shapes: dict[str, tuple[int, ...]], sample_count: int
) -> bool:
    """Return whether full-episode raw DROID tensors make horizon a runtime choice."""
    if str(metadata.get("action_contract", "")) != DROID_DREAMZERO_ACTION_CONTRACT:
        return False
    if str(metadata.get("coverage", "")) != STAGE2_LATENT_CACHE_FULL_COVERAGE:
        return False
    if str(metadata.get("padding_policy", "")) != STAGE2_LATENT_CACHE_PADDING_POLICY:
        return False
    for tensor_key, tensor_dim in DROID_DREAMZERO_REQUIRED_TENSOR_DIMS.items():
        shape = tensor_shapes.get(tensor_key)
        if shape is None or len(shape) != 2:
            return False
        if int(shape[0]) != int(sample_count) or int(shape[-1]) != tensor_dim:
            return False
    return True


def validate_stage2_latent_cache_header(
    *, path: Path, header: dict[str, Any], expected: dict[str, Any], stage_label: str = "Stage 2"
) -> None:
    _validate_stage2_latent_cache_path(path, expected, stage_label=stage_label)
    meta = header["metadata"]
    latent_shape = tuple(int(v) for v in header["latent_shape"])
    latent_count = int(latent_shape[0])
    cache_version = meta.get("cache_version")
    coverage = meta.get("coverage")
    padding_policy = meta.get("padding_policy")
    num_samples = int(meta.get("num_samples", latent_count))
    if cache_version is None:
        raise ValueError(
            f"{stage_label} cached-latent training requires regenerated full-coverage cache metadata. Missing `cache_version` in {path}."
        )
    if int(cache_version) not in yam.cache_versions(expected, meta):
        raise ValueError(
            f"{stage_label} cached-latent training requires cache_version={STAGE2_LATENT_CACHE_VERSION}, got {cache_version!r} in {path}."
        )
    if str(coverage) != STAGE2_LATENT_CACHE_FULL_COVERAGE:
        raise ValueError(
            f"{stage_label} cached-latent training requires coverage={STAGE2_LATENT_CACHE_FULL_COVERAGE!r}, got {coverage!r} in {path}."
        )
    if str(padding_policy) != STAGE2_LATENT_CACHE_PADDING_POLICY:
        raise ValueError(
            f"{stage_label} cached-latent training requires padding_policy={STAGE2_LATENT_CACHE_PADDING_POLICY!r}, got {padding_policy!r} in {path}."
        )
    if num_samples != latent_count:
        raise ValueError(
            f"{stage_label} latent cache metadata/sample mismatch in {path}: num_samples={num_samples} latent_shape[0]={latent_count}."
        )
    if len(latent_shape) != 5:
        raise ValueError(
            f"{stage_label} latent cache must store 5D latents [N,C,T,H,W], got {latent_shape} in {path}."
        )
    expected_shape_tail = (
        int(expected["latent_channels"]),
        int(expected["latent_t_per_window"]),
        int(expected["latent_h"]),
        int(expected["latent_w"]),
    )
    if tuple(latent_shape[1:]) != expected_shape_tail:
        raise ValueError(
            f"{stage_label} latent cache shape mismatch for video_model_family={expected['video_model_family']!r} vae_family={expected['vae_family']!r} in {path}: got latent_shape={latent_shape}, expected [N,{expected_shape_tail[0]},{expected_shape_tail[1]},{expected_shape_tail[2]},{expected_shape_tail[3]}]."
        )
    tensor_shapes = dict(header.get("tensor_shapes") or {})
    require_model_metadata = not bool(expected.get("allow_legacy_metadata", False))
    mismatches: list[str] = []
    if str(expected.get("video_model_family")) == "wan22_i2v_a14b":
        condition_shape = tensor_shapes.get(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)
        if condition_shape is None:
            mismatches.append(
                f"{WAN22_A14B_I2V_CONDITION_LATENTS_KEY}: missing expected_shape={latent_shape!r}"
            )
        elif tuple(int(v) for v in condition_shape) != latent_shape:
            mismatches.append(
                f"{WAN22_A14B_I2V_CONDITION_LATENTS_KEY}: actual_shape={condition_shape!r} expected_shape={latent_shape!r}"
            )
        _check_metadata_value(
            mismatches,
            meta=meta,
            key="i2v_condition_source",
            expected=WAN22_A14B_I2V_CONDITION_SOURCE_OFFICIAL,
            require=True,
        )
    for key in (
        "video_model_family",
        "model_id",
        "vae_family",
        "latent_channels",
        "latent_t_per_window",
        "latent_h",
        "latent_w",
        "vae_z_dim",
        "vae_temporal_stride",
        "vae_spatial_stride",
    ):
        _check_metadata_value(
            mismatches, meta=meta, key=key, expected=expected[key], require=require_model_metadata
        )
    runtime_horizon_cache = is_droid_dreamzero_runtime_horizon_cache(
        metadata=meta, tensor_shapes=tensor_shapes, sample_count=latent_count
    )
    require_action_horizon = bool(expected.get("require_action_horizon_metadata", False))
    if expected.get("action_horizon") is not None and (
        require_action_horizon or not runtime_horizon_cache
    ):
        _check_metadata_value(
            mismatches,
            meta=meta,
            key="action_horizon",
            expected=int(expected["action_horizon"]),
            require=require_action_horizon,
        )
    if mismatches:
        raise ValueError(
            f"{stage_label} latent cache model/VAE contract mismatch in {path}: "
            + "; ".join(mismatches)
        )


def validate_stage2_latent_cache_file(
    path: Path, *, expected: dict[str, Any], stage_label: str = "Stage 2"
) -> None:
    validate_stage2_latent_cache_header(
        path=path, header=_read_episode_header(path), expected=expected, stage_label=stage_label
    )
    if bool(expected.get("require_i2v_first_latent_equality", False)):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            latents_first = handle.get_slice("latents")[:, :, 0:1]
            condition_first = handle.get_slice(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)[:, :, 0:1]
        if not torch.equal(latents_first, condition_first):
            max_abs_diff = float(
                (latents_first.float() - condition_first.float()).abs().max().item()
            )
            raise ValueError(
                f"{stage_label} requires exact first-latent equality between `latents` and `{WAN22_A14B_I2V_CONDITION_LATENTS_KEY}` in {path}; max_abs_diff={max_abs_diff:.8g}. Regenerate both tensors atomically."
            )


def validate_stage2_latent_cache_header_file(
    path: Path, *, expected: dict[str, Any], stage_label: str = "Stage 2"
) -> None:
    validate_stage2_latent_cache_header(
        path=path, header=_read_episode_header(path), expected=expected, stage_label=stage_label
    )


def _validate_droid_dreamzero_overlay_header(
    path: Path, *, header: dict[str, Any], expected: dict[str, Any], stage_label: str = "Stage 2"
) -> None:
    dict(header["metadata"])
    raise ValueError(
        f"{stage_label} expected a DROID DreamZero overlay cache file with overlay_cache_kind={DROID_DREAMZERO_OVERLAY_CACHE_KIND!r}: {path}"
    )


def validate_droid_dreamzero_overlay_cache_file(
    path: Path, *, expected: dict[str, Any], stage_label: str = "Stage 2"
) -> None:
    _validate_droid_dreamzero_overlay_header(
        path,
        header=_read_cached_or_overlay_episode_header(path),
        expected=expected,
        stage_label=stage_label,
    )


def validate_stage2_latent_tensor_contract(
    latents: torch.Tensor, *, expected: dict[str, Any], stage_label: str = "Stage 2"
) -> None:
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise ValueError(
            f"{stage_label} input_latents must be 5D [B,C,T,H,W], got {tuple(getattr(latents, 'shape', ()))}."
        )
    expected_tail = (
        int(expected["latent_channels"]),
        int(expected["latent_t_per_window"]),
        int(expected["latent_h"]),
        int(expected["latent_w"]),
    )
    if tuple(int(v) for v in latents.shape[1:]) != expected_tail:
        raise ValueError(
            f"{stage_label} input_latents shape mismatch for video_model_family={expected['video_model_family']!r} vae_family={expected['vae_family']!r}: got {tuple(latents.shape)}, expected [B,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]},{expected_tail[3]}]."
        )


class _EpisodeHandleCache:
    """Per-worker LRU of open ``safe_open`` handles."""

    def __init__(self, maxsize: int = 64):
        self._maxsize = int(maxsize)
        self._store: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path):
        handle = self._store.get(path)
        if handle is None:
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._store[path] = handle
            if len(self._store) > self._maxsize:
                (_, victim) = self._store.popitem(last=False)
                try:
                    victim.__exit__(None, None, None)
                except Exception:
                    pass
        else:
            self._store.move_to_end(path)
        return handle


class CachedStage2LatentDataset(torch.utils.data.Dataset):
    """Serve cached Stage 2 latents while preserving the live sample stream."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        source_dataset: Dataset,
        expected_cache_contract: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.cache_dirs = _parse_path_list(cache_dir, label="Stage 2 latent cache directory")
        self.cache_dir = self.cache_dirs[0]
        missing_cache_dirs = [
            str(cache_dir) for cache_dir in self.cache_dirs if not cache_dir.exists()
        ]
        if missing_cache_dirs:
            raise FileNotFoundError(
                f"Stage 2 latent cache directories not found: {missing_cache_dirs}"
            )
        self.source_dataset = source_dataset
        self.expected_cache_contract = (
            None if expected_cache_contract is None else dict(expected_cache_contract)
        )
        self.text_cache_model_suffix = stage2_text_cache_suffix_for_family(
            str(self.expected_cache_contract.get("video_model_family", "wan22_ti2v_5b"))
            if self.expected_cache_contract is not None
            else "wan22_ti2v_5b"
        )
        self.num_frames = int(self.source_dataset.num_frames)
        self.action_video_freq_ratio = int(self.source_dataset.action_video_freq_ratio)
        fallback_action_horizon = getattr(self.source_dataset, "action_horizon", None)
        self.action_horizon = (
            self.num_frames - 1 if fallback_action_horizon is None else int(fallback_action_horizon)
        )
        self.action_history_horizon = int(getattr(self.source_dataset, "action_history_horizon", 0))
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                f"num_frames-1 must be divisible by action_video_freq_ratio; got {self.num_frames}, {self.action_video_freq_ratio}"
            )
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}.")
        if self.action_history_horizon < 0:
            raise ValueError(
                f"action_history_horizon must be non-negative, got {self.action_history_horizon}."
            )
        self.context_len = int(self.source_dataset.context_len)
        self.text_embedding_cache_dirs = _parse_path_list(
            self.source_dataset.text_embedding_cache_dir,
            label="Stage 2 text embedding cache directory",
        )
        self.text_embedding_cache_dir = self.text_embedding_cache_dirs[0]
        missing_text_cache_dirs = [
            str(cache_dir) for cache_dir in self.text_embedding_cache_dirs if not cache_dir.exists()
        ]
        if missing_text_cache_dirs:
            raise FileNotFoundError(
                f"Stage 2 text embedding cache directories not found: {missing_text_cache_dirs}"
            )
        base_dataset = getattr(self.source_dataset, "lerobot_dataset", None)
        if base_dataset is None:
            raise TypeError("source_dataset must expose `lerobot_dataset`.")
        multi_dataset = getattr(base_dataset, "multi_dataset", None)
        raw_datasets = (
            list(getattr(multi_dataset, "_datasets", [])) if multi_dataset is not None else []
        )
        if not raw_datasets:
            raise ValueError(
                "Cached Stage 2 latents require a fallback dataset backed by at least one LeRobot dataset."
            )
        root_names = [Path(raw_dataset.root).name for raw_dataset in raw_datasets]
        if len(root_names) != len(set(root_names)):
            raise ValueError(
                f"Cached Stage 2 root-aware routing requires unique dataset root names; got roots={[str(raw_dataset.root) for raw_dataset in raw_datasets]}."
            )
        records: list[dict[str, Any]] = []
        dataset_offset = 0
        for raw_dataset in raw_datasets:
            if raw_dataset.episodes is None:
                raise ValueError(
                    "source_dataset must resolve the selected episode list before caching."
                )
            episode_data_index = getattr(raw_dataset, "episode_data_index", None)
            if episode_data_index is None:
                raise ValueError(
                    f"fallback LeRobot dataset is missing episode_data_index: {raw_dataset.root}"
                )
            starts = [int(v) for v in episode_data_index["from"].tolist()]
            ends = [int(v) for v in episode_data_index["to"].tolist()]
            selected_episodes = [int(ep) for ep in raw_dataset.episodes]
            if not len(selected_episodes) == len(starts) == len(ends):
                raise ValueError(
                    f"source_dataset episode metadata is inconsistent for {raw_dataset.root}: episodes={len(selected_episodes)} starts={len(starts)} ends={len(ends)}"
                )
            root_name = Path(raw_dataset.root).name
            tasks = dict(raw_dataset.meta.tasks)
            episode_rows = getattr(raw_dataset.meta, "episodes", {}) or {}
            for ep_idx, start, end in zip(selected_episodes, starts, ends, strict=False):
                episode_row = episode_rows.get(int(ep_idx))
                if episode_row is None:
                    episode_row = episode_rows.get(str(int(ep_idx)), {})
                records.append(
                    {
                        "root_name": root_name,
                        "episode_index": int(ep_idx),
                        "start": int(dataset_offset + start),
                        "end": int(dataset_offset + end),
                        "tasks": tasks,
                        "task_candidates": _task_candidates_from_episode_row(episode_row),
                    }
                )
            dataset_offset += int(raw_dataset.num_frames)
        self._episode_records = records
        self._episode_ends: list[int] = [int(record["end"]) for record in records]
        self._single_root_cache = len(raw_datasets) == 1
        processor = getattr(base_dataset, "processor", None)
        self._processor = processor
        self._instruction_builder = getattr(processor, "augment_instruction", None)
        self._video_sample_indices = tuple(int(i) for i in self.source_dataset.video_sample_indices)
        self._frames_per_window_value = len(self._video_sample_indices)
        if self._frames_per_window_value <= 0:
            raise ValueError(
                "source_dataset.video_sample_indices must contain at least one frame index."
            )
        video_transitions = self._frames_per_window_value - 1
        if video_transitions <= 0:
            raise ValueError(
                "source_dataset.video_sample_indices must contain at least two frame indices."
            )
        if self.action_horizon % video_transitions != 0:
            raise ValueError(
                f"action_horizon must be divisible by sparse video transitions; got action_horizon={self.action_horizon} and video_transitions={video_transitions}."
            )
        cached_samples: dict[tuple[str, int], int] = {}
        missing: list[str] = []
        for record in self._episode_records:
            ep_idx = int(record["episode_index"])
            cache_key = self._cache_key(record)
            path = self._episode_file_path(record)
            if not path.exists():
                missing.append(f"{record['root_name']}/episode_{ep_idx:06d}")
                cached_samples[cache_key] = 0
                continue
            header = _read_cached_or_overlay_episode_header(path)
            meta_fields = header["metadata"]
            ep_len = int(record["end"]) - int(record["start"])
            num_samples = _metadata_int(meta_fields, "num_samples")
            if num_samples is None:
                num_samples = _metadata_int(meta_fields, "num_windows")
            if num_samples is None:
                num_samples = int(header["latent_shape"][0])
            if num_samples < 0:
                raise ValueError(
                    f"Cached episode {ep_idx} reported negative num_samples={num_samples}."
                )
            if int(header["latent_shape"][0]) != num_samples:
                raise ValueError(
                    f"Cached episode {ep_idx} latent count mismatch: metadata={num_samples} latent_shape={header['latent_shape'][0]}."
                )
            if num_samples > ep_len:
                raise ValueError(
                    f"Cached episode {ep_idx} has num_samples={num_samples}, which exceeds live episode length {ep_len}."
                )
            if self.expected_cache_contract is not None:
                validate_stage2_latent_cache_header(
                    path=path,
                    header=header,
                    expected=self.expected_cache_contract,
                    stage_label="CachedStage2LatentDataset",
                )
            else:
                cache_version = _metadata_int(meta_fields, "cache_version")
                coverage = str(meta_fields.get("coverage", "")).strip()
                padding_policy = str(meta_fields.get("padding_policy", "")).strip()
                if cache_version != STAGE2_LATENT_CACHE_VERSION:
                    raise ValueError(
                        f"Cached episode {ep_idx} must use cache_version={STAGE2_LATENT_CACHE_VERSION}, got {meta_fields.get('cache_version')!r}."
                    )
                if coverage != STAGE2_LATENT_CACHE_FULL_COVERAGE:
                    raise ValueError(
                        f"Cached episode {ep_idx} must use coverage={STAGE2_LATENT_CACHE_FULL_COVERAGE!r}, got {meta_fields.get('coverage')!r}."
                    )
                if padding_policy != STAGE2_LATENT_CACHE_PADDING_POLICY:
                    raise ValueError(
                        f"Cached episode {ep_idx} must use padding_policy={STAGE2_LATENT_CACHE_PADDING_POLICY!r}, got {meta_fields.get('padding_policy')!r}."
                    )
            if num_samples != ep_len:
                raise ValueError(
                    f"Cached episode {ep_idx} must cover the full episode length {ep_len}, got num_samples={num_samples}."
                )
            cached_samples[cache_key] = num_samples
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cached episode files are missing under {self.cache_dirs}. First few: {missing[:5]}. Regenerate a complete cache before training."
            )
        if missing:
            logger.warning(
                "CachedStage2LatentDataset: partial cache — %d of %d episodes missing; those samples will fall back to live decode. First missing: %s",
                len(missing),
                len(self._episode_records),
                missing[:5],
            )
        self._cached_samples = cached_samples
        self._num_cached_samples = sum(
            self._cached_samples[self._cache_key(record)] for record in self._episode_records
        )
        if self._num_cached_samples <= 0:
            raise RuntimeError(f"No cached Stage 2 windows found under {self.cache_dirs}.")
        split_name = "train" if getattr(base_dataset, "is_training_set", False) else "val"
        logger.info(
            "CachedStage2LatentDataset: split=%s episodes=%d samples=%d cached=%d tail_fallback=0 cache_dirs=%s",
            split_name,
            len(self._episode_records),
            len(self),
            self._num_cached_samples,
            self.cache_dirs,
        )
        self._handles = _EpisodeHandleCache(maxsize=64)

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._get(idx)

    def _cache_key(self, record: dict[str, Any]) -> tuple[str, int]:
        return (str(record["root_name"]), int(record["episode_index"]))

    def _expected_flat_cache_dataset_kind(self, record: dict[str, Any]) -> str | None:
        root_name = str(record["root_name"]).strip().lower().replace("-", "_")
        if root_name.startswith("libero_"):
            return "libero"
        return None

    def _flat_cache_path_matches_record(self, path: Path, record: dict[str, Any]) -> bool:
        if not path.exists():
            return False
        expected_kind = self._expected_flat_cache_dataset_kind(record)
        if expected_kind is None:
            return False
        header = _read_episode_header(path)
        meta = header["metadata"]
        actual_kind = str(meta.get("dataset_kind", "")).strip().lower().replace("-", "_")
        if actual_kind != expected_kind:
            return False
        cached_episode = meta.get("episode_index")
        if cached_episode is not None and int(cached_episode) != int(record["episode_index"]):
            raise ValueError(
                f"Flat Stage 2 cache candidate {path} has episode_index={cached_episode}, expected {record['episode_index']} for root {record['root_name']}."
            )
        return True

    def _episode_file_path(self, record: dict[str, Any]) -> Path:
        ep_idx = int(record["episode_index"])
        filename = f"episode_{ep_idx:06d}.safetensors"
        candidates: list[Path] = []
        flat_candidates: list[Path] = []
        for cache_dir in self.cache_dirs:
            flat_path = cache_dir / filename
            root_path = cache_dir / str(record["root_name"]) / filename
            if self._single_root_cache:
                candidates.append(flat_path)
            candidates.append(root_path)
            if not self._single_root_cache:
                flat_candidates.append(flat_path)
        for path in candidates:
            if path.exists():
                return path
        for path in flat_candidates:
            if self._flat_cache_path_matches_record(path, record):
                return path
        return candidates[0]

    def _episode_file(self, record: dict[str, Any]) -> Any:
        return self._handles.get(self._episode_file_path(record))

    def _latent_episode_file(self, path: Path, handle: Any, meta: dict[str, Any]) -> Any:
        return handle

    def _resolve_index(self, idx: int) -> tuple[dict[str, Any], int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")
        ep_pos = bisect_right(self._episode_ends, idx)
        if ep_pos >= len(self._episode_records):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")
        record = self._episode_records[ep_pos]
        return (record, idx - int(record["start"]))

    def _droid_dreamzero_action_state_from_cache(
        self,
        handle: Any,
        *,
        ep_idx: int,
        ep_len: int,
        action_indices: torch.Tensor,
        proprio_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required = {
            "droid_action_joint_position": 7,
            "droid_action_gripper_position": 1,
            "droid_state_joint_position": 7,
            "droid_state_gripper_position": 1,
        }
        tensors: dict[str, torch.Tensor] = {}
        for key, dim in required.items():
            try:
                tensor = handle.get_tensor(key).to(torch.float32)
            except Exception as exc:
                raise KeyError(
                    f"DROID DreamZero cached action contract requires tensor {key!r} in episode={ep_idx}."
                ) from exc
            if (
                tensor.ndim != 2
                or int(tensor.shape[0]) != int(ep_len)
                or int(tensor.shape[1]) != int(dim)
            ):
                raise ValueError(
                    f"DROID DreamZero tensor {key!r} must have shape [{ep_len},{dim}] for episode={ep_idx}, got {tuple(tensor.shape)}."
                )
            tensors[key] = tensor
        processor = self._processor
        if processor is None:
            raise ValueError(
                "DROID DreamZero cached action normalization requires the fallback dataset processor."
            )
        max_index = int(ep_len) - 1
        action_idx = action_indices.clamp(min=0, max=max_index)
        proprio_idx = proprio_indices.clamp(min=0, max=max_index)
        state_joint_full = tensors["droid_state_joint_position"][proprio_idx]
        data = {
            "action": {
                "joint_position": tensors["droid_action_joint_position"][action_idx]
                - state_joint_full[0].unsqueeze(0),
                "gripper_position": binarize_droid_gripper_action_history(
                    tensors["droid_action_gripper_position"][action_idx],
                    action_history_horizon=self.action_history_horizon,
                ),
            },
            "state": {
                "joint_position": state_joint_full,
                "gripper_position": tensors["droid_state_gripper_position"][proprio_idx],
            },
        }
        data = processor.normalizer.forward(data)
        data = processor.action_state_merger.forward(data)
        return (data["action"], data["state"])

    def _get(self, idx: int) -> dict[str, Any]:
        (record, frame_idx) = self._resolve_index(idx)
        ep_idx = int(record["episode_index"])
        ep_len = int(record["end"]) - int(record["start"])
        cache_key = self._cache_key(record)
        if frame_idx >= self._cached_samples[cache_key]:
            raise ValueError(f"Cached video does not cover episode={ep_idx}, frame={frame_idx}.")
        path = self._episode_file_path(record)
        handle = self._handles.get(path)
        cache_meta = dict(handle.metadata() or {})
        latent_handle = self._latent_episode_file(path, handle, cache_meta)
        latents_slice = latent_handle.get_slice("latents")
        latent = latents_slice[frame_idx : frame_idx + 1, ...]
        if not isinstance(latent, torch.Tensor):
            raise RuntimeError(f"Unexpected latent slice type {type(latent)}")
        latent = latent.squeeze(0)
        i2v_condition_latents = None
        expected_family = (
            str(self.expected_cache_contract.get("video_model_family"))
            if self.expected_cache_contract is not None
            else str(cache_meta.get("video_model_family", ""))
        )
        if expected_family == "wan22_i2v_a14b":
            try:
                i2v_slice = latent_handle.get_slice(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)
            except Exception as exc:
                raise KeyError(
                    f"14B cached-latent samples require tensor {WAN22_A14B_I2V_CONDITION_LATENTS_KEY!r} in the latent source for {path}. Regenerate the cache with official Wan2.2 I2V conditioning."
                ) from exc
            i2v_condition = i2v_slice[frame_idx : frame_idx + 1, ...]
            if not isinstance(i2v_condition, torch.Tensor):
                raise RuntimeError(
                    f"Unexpected {WAN22_A14B_I2V_CONDITION_LATENTS_KEY} slice type {type(i2v_condition)}"
                )
            i2v_condition_latents = i2v_condition.squeeze(0)
        is_droid_dreamzero = (
            str(cache_meta.get("action_contract", "")) == DROID_DREAMZERO_ACTION_CONTRACT
        )
        action_horizon = self.action_horizon
        task_indices_all = handle.get_tensor("task_indices")
        if int(task_indices_all.shape[0]) != ep_len:
            raise ValueError(
                f"Cached task_indices length mismatch for episode={ep_idx}: expected {ep_len}, got {task_indices_all.shape[0]}."
            )
        if not is_droid_dreamzero:
            action_tensor_all = handle.get_tensor("action")
            proprio_tensor_all = handle.get_tensor("proprio")
            if int(action_tensor_all.shape[0]) != ep_len:
                raise ValueError(
                    f"Cached action length mismatch for episode={ep_idx}: expected {ep_len}, got {action_tensor_all.shape[0]}."
                )
            if int(proprio_tensor_all.shape[0]) != ep_len:
                raise ValueError(
                    f"Cached proprio length mismatch for episode={ep_idx}: expected {ep_len}, got {proprio_tensor_all.shape[0]}."
                )
        max_index = ep_len - 1
        action_indices = torch.arange(frame_idx, frame_idx + action_horizon, dtype=torch.long)
        history_indices = torch.arange(
            frame_idx - self.action_history_horizon, frame_idx, dtype=torch.long
        )
        proprio_indices = torch.arange(frame_idx, frame_idx + action_horizon, dtype=torch.long)
        action_is_pad = action_indices >= ep_len
        previous_action_is_pad = history_indices < 0
        proprio_is_pad = proprio_indices >= ep_len
        image_indices = torch.tensor(self._video_sample_indices, dtype=torch.long) + frame_idx
        image_is_pad = image_indices >= ep_len
        if is_droid_dreamzero:
            (action_with_history, proprio_full) = self._droid_dreamzero_action_state_from_cache(
                handle,
                ep_idx=ep_idx,
                ep_len=ep_len,
                action_indices=torch.cat((history_indices, action_indices), dim=0),
                proprio_indices=proprio_indices,
            )
            previous_action = action_with_history[: self.action_history_horizon]
            action_tensor = action_with_history[self.action_history_horizon :]
        else:
            action_tensor = action_tensor_all[action_indices.clamp(max=max_index)]
            previous_action = action_tensor_all[history_indices.clamp(min=0, max=max_index)]
            proprio_full = proprio_tensor_all[proprio_indices.clamp(max=max_index)]
        if tuple(previous_action.shape[:1]) != (self.action_history_horizon,):
            raise ValueError(
                f"Cached previous-action horizon mismatch for episode={ep_idx} frame={frame_idx}: got {tuple(previous_action.shape)} expected history={self.action_history_horizon}."
            )
        previous_action = previous_action.clone()
        previous_action[previous_action_is_pad] = 0.0
        proprio_tensor = proprio_full
        task_candidates = tuple(
            str(task).strip() for task in record.get("task_candidates", ()) if str(task).strip()
        )
        if task_candidates:
            prompt = self._build_prompt(random.choice(task_candidates))
        else:
            task_idx_scalar = int(task_indices_all[frame_idx])
            tasks = record["tasks"]
            if task_idx_scalar not in tasks:
                raise ValueError(
                    f"Cached task_index {task_idx_scalar} not found in meta.tasks for {record['root_name']}"
                )
            prompt = self._build_prompt(tasks[task_idx_scalar])
        (context, context_mask) = self._load_text_context(prompt)
        context = context.clone()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        sample = {
            "input_latents": latent,
            "action": action_tensor,
            "proprio": proprio_tensor,
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
        }
        if self.action_history_horizon > 0:
            sample["previous_action"] = previous_action
            sample["previous_action_is_pad"] = previous_action_is_pad
        if i2v_condition_latents is not None:
            sample[WAN22_A14B_I2V_CONDITION_LATENTS_KEY] = i2v_condition_latents
        return sample

    def _build_prompt(self, task: str) -> str:
        task_text = str(task).strip()
        if not task_text:
            raise ValueError(f"Invalid empty task text: {task!r}")
        instruction = task_text
        if self._instruction_builder is not None:
            instruction = self._instruction_builder({"task": task_text})
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(f"Invalid instruction built from task {task!r}: {instruction!r}")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError(f"Invalid empty instruction built from task {task!r}")
        return DEFAULT_PROMPT.format(task=instruction)

    @functools.lru_cache(maxsize=256)
    def _load_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_paths = [
            cache_dir / f"{hashed}.t5_len{self.context_len}.{self.text_cache_model_suffix}.pt"
            for cache_dir in self.text_embedding_cache_dirs
        ]
        cache_path = next((path for path in cache_paths if path.exists()), cache_paths[0])
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing text embedding cache. Tried: {cache_paths}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        validate_text_cache(payload, cache_path, context_len=self.context_len)
        missing = [k for k in STAGE2_REQUIRED_CACHE_KEYS if k not in payload]
        if missing:
            raise ValueError(f"Text embedding cache missing keys {missing} at {cache_path}")
        context = payload["context"]
        mask = payload["mask"].bool()
        if context.shape[0] != self.context_len or mask.shape[0] != self.context_len:
            raise ValueError(
                f"Text cache shape mismatch at {cache_path}: context={tuple(context.shape)} mask={tuple(mask.shape)}"
            )
        return (context, mask)
