"""Offline VAE-latent precomputation.

For each episode, decode every frame of all cameras, apply the dataset-specific
multi-camera concat + normalize, build one sparse latent window per episode
frame (sub-sampled by ``action_video_freq_ratio`` with tail clamping), and
encode each window through the frozen Wan VAE. Store per-episode ``.safetensors``
with the full-coverage latents plus per-frame normalized action/proprio and task
indices.

Training code can consume these files without touching the mp4s or the VAE.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torchvision.transforms.functional as transforms_F
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

from dreamfree.data.lerobot.cached_latent_dataset import (
    stage2_latent_cache_expected_contract as latent_cache_expected_contract,
)
from dreamfree.data.lerobot.lerobot.datasets.compute_stats import aggregate_stats
from dreamfree.data.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata
from dreamfree.data.lerobot.utils.normalizer import LinearNormalizer, load_dataset_stats_from_json
from dreamfree.data.robocasa_visual import (
    compose_robocasa_dreamzero_uint8,
    expected_robocasa_cache_raw_camera_size,
    normalize_uint8_video,
    required_robocasa_visual_preprocess_id,
)
from dreamfree.models.wan22.helpers.loader import (
    _load_model_from_state_dict,
    load_wan22_i2v_a14b_vae,
    load_wan22_ti2v_5b_vae,
)
from dreamfree.models.wan22.helpers.state_dict_converters import wan_video_vae_state_dict_converter
from dreamfree.models.wan22.i2v_condition import (
    WAN22_A14B_I2V_CONDITION_LATENTS_KEY,
    WAN22_A14B_I2V_CONDITION_SOURCE_OFFICIAL,
)
from dreamfree.models.wan22.wan_video_vae import WanVideoVAE, WanVideoVAE38
from dreamfree.utils.fs import ensure_dir
from dreamfree.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
ROBOTWIN_CONCAT_MULTI_CAMERA = "robotwin"
HORIZONTAL_CONCAT_MULTI_CAMERA = "horizontal"
DROID_GRID_CONCAT_MULTI_CAMERA = "droid_grid"
ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA = "robocasa_dreamzero_352"
LATENT_CACHE_VERSION = 3
LATENT_CACHE_FULL_COVERAGE = "full_episode"
LATENT_CACHE_PADDING_POLICY = "clamp_to_episode_end"
ROBOTWIN_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
LIBERO_CAMERA_KEYS = ("observation.images.image", "observation.images.wrist_image")
DROID_CAMERA_KEYS = (
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
)
ROBOCASA_CAMERA_KEYS = (
    "observation.images.agentview_left",
    "observation.images.agentview_right",
    "observation.images.wrist",
)
ROBOCASA_ACTION_DIM = 7
ROBOCASA_PROPRIO_DIM = 9
LIBERO_DATASET_NAMES = {"libero"}
ROBOTWIN_DATASET_NAMES = {"robotwin"}
DROID_DATASET_NAMES = {"droid"}
ROBOCASA_DATASET_NAMES = {"robocasa"}
DROID_STAGE2_ACTION_HORIZON = 24
DROID_DREAMZERO_ACTION_CONTRACT = "droid_dreamzero_rel_joint_abs_gripper_q99_v1"
DROID_DREAMZERO_RELATIVE_STATS_FILE = "relative_stats_dreamzero.json"
DROID_JOINT_POSITION_KEY = "joint_position"
DROID_GRIPPER_POSITION_KEY = "gripper_position"
VAE_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")


@dataclass
class PrecomputeConfig:
    model_base_path: str = "./assets/models"
    dataset_name: str = "robotwin"
    dataset_dirs: str = "./assets"
    pretrained_norm_stats: str = "./assets"
    num_frames: int = 33
    action_video_freq_ratio: int = 4
    action_horizon: int | None = None
    height: int = 384
    width: int = 320
    action_dim: int = 14
    proprio_dim: int = 14
    concat_multi_camera: str = ROBOTWIN_CONCAT_MULTI_CAMERA
    model_id: str | None = None
    video_model_family: str | None = None
    vae_checkpoint: str | None = None
    video_latent_dim: int | None = None
    redirect_common_files: bool = True
    pretrained_load_broadcast: bool = True
    mixed_precision: str = "bf16"


@dataclass(frozen=True)
class PrecomputeRuntime:
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool
    device: str


@dataclass
class EpisodePayload:
    episode_index: int
    num_frames: int
    video_processed: torch.Tensor
    action_norm: torch.Tensor
    state_norm: torch.Tensor
    task_indices: torch.Tensor
    concat_layout: str
    norm_mode: str
    dataset_kind: str | None = None
    action_layout: str | None = None
    extra_tensors: dict[str, torch.Tensor] | None = None
    source_metadata: dict[str, str] | None = None


@dataclass
class PrecomputeVAEModel:
    vae: Any
    device: torch.device
    torch_dtype: torch.dtype
    vae_path: str
    stats_src: str
    norm_mode: str
    cache_contract: dict[str, Any]
    compile_enabled: bool = False

    def encode_video_latents(
        self,
        video_tensor: torch.Tensor,
        *,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        return self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )


def _parse_dataset_dirs(value: str) -> list[str]:
    dirs = [part.strip() for part in str(value).split(",") if part.strip()]
    if not dirs:
        raise ValueError("`dataset_dirs` must contain at least one directory.")
    return dirs


def _normalize_mixed_precision(mixed_precision: str) -> str:
    value = str(mixed_precision).strip().lower()
    if value not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision={mixed_precision}. Expected one of: no, fp16, bf16."
        )
    return value


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    value = _normalize_mixed_precision(mixed_precision)
    if value == "no":
        return torch.float32
    if value == "fp16":
        return torch.float16
    return torch.bfloat16


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


def validate_latent_cache_header(
    *,
    path: Path,
    header: dict[str, Any],
    expected: dict[str, Any],
    label: str = "Latent precompute",
) -> None:
    meta = header["metadata"]
    latent_shape = tuple((int(v) for v in header["latent_shape"]))
    latent_count = int(latent_shape[0])
    cache_version = meta.get("cache_version")
    coverage = meta.get("coverage")
    padding_policy = meta.get("padding_policy")
    num_samples = int(meta.get("num_samples", latent_count))
    if cache_version is None:
        raise ValueError(
            f"{label} requires regenerated full-coverage cache metadata. Missing `cache_version` in {path}."
        )
    if int(cache_version) != LATENT_CACHE_VERSION:
        raise ValueError(
            f"{label} requires cache_version={LATENT_CACHE_VERSION}, got {cache_version!r} in {path}."
        )
    if str(coverage) != LATENT_CACHE_FULL_COVERAGE:
        raise ValueError(
            f"{label} requires coverage={LATENT_CACHE_FULL_COVERAGE!r}, got {coverage!r} in {path}."
        )
    if str(padding_policy) != LATENT_CACHE_PADDING_POLICY:
        raise ValueError(
            f"{label} requires padding_policy={LATENT_CACHE_PADDING_POLICY!r}, got {padding_policy!r} in {path}."
        )
    if num_samples != latent_count:
        raise ValueError(
            f"{label} metadata/sample mismatch in {path}: num_samples={num_samples} latent_shape[0]={latent_count}."
        )
    if len(latent_shape) != 5:
        raise ValueError(
            f"{label} must store 5D latents [N,C,T,H,W], got {latent_shape} in {path}."
        )
    expected_shape_tail = (
        int(expected["latent_channels"]),
        int(expected["latent_t_per_window"]),
        int(expected["latent_h"]),
        int(expected["latent_w"]),
    )
    if tuple(latent_shape[1:]) != expected_shape_tail:
        raise ValueError(
            f"{label} shape mismatch for video_model_family={expected['video_model_family']!r} vae_family={expected['vae_family']!r} in {path}: got latent_shape={latent_shape}, expected [N,{expected_shape_tail[0]},{expected_shape_tail[1]},{expected_shape_tail[2]},{expected_shape_tail[3]}]."
        )
    require_model_metadata = not bool(expected.get("allow_legacy_metadata", False))
    mismatches: list[str] = []
    if str(expected.get("video_model_family")) == "wan22_i2v_a14b":
        tensor_shapes = dict(header.get("tensor_shapes") or {})
        condition_shape = tensor_shapes.get(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)
        if condition_shape is None:
            mismatches.append(
                f"{WAN22_A14B_I2V_CONDITION_LATENTS_KEY}: missing expected_shape={latent_shape!r}"
            )
        elif tuple((int(v) for v in condition_shape)) != latent_shape:
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
    if expected.get("visual_preprocess_id") is not None:
        _check_metadata_value(
            mismatches,
            meta=meta,
            key="visual_preprocess_id",
            expected=expected["visual_preprocess_id"],
            require=True,
        )
    if expected.get("action_horizon") is not None:
        _check_metadata_value(
            mismatches,
            meta=meta,
            key="action_horizon",
            expected=int(expected["action_horizon"]),
            require=bool(expected.get("require_action_horizon_metadata", False)),
        )
    if mismatches:
        raise ValueError(f"{label} model/VAE contract mismatch in {path}: " + "; ".join(mismatches))


def validate_14b_i2v_first_latent_consistency(
    latents: torch.Tensor, i2v_condition_latents: torch.Tensor, *, label: str
) -> None:
    """Require the causal first latent to agree for video and first-frame-only inputs."""
    if tuple(i2v_condition_latents.shape) != tuple(latents.shape):
        raise ValueError(
            f"{label} {WAN22_A14B_I2V_CONDITION_LATENTS_KEY} shape must match latents: {tuple(i2v_condition_latents.shape)} vs {tuple(latents.shape)}."
        )
    if latents.dtype != i2v_condition_latents.dtype:
        raise ValueError(
            f"{label} {WAN22_A14B_I2V_CONDITION_LATENTS_KEY} dtype must match latents: {i2v_condition_latents.dtype} vs {latents.dtype}."
        )
    video_first = latents[:, :, 0]
    condition_first = i2v_condition_latents[:, :, 0]
    if torch.equal(video_first, condition_first):
        return
    difference = (video_first.to(torch.float32) - condition_first.to(torch.float32)).abs()
    raise ValueError(
        f"{label} has inconsistent 14B causal first-frame latents: latents[:, :, 0] must be bitwise identical to {WAN22_A14B_I2V_CONDITION_LATENTS_KEY}[:, :, 0], max_abs_diff={float(difference.max().item()):.8g}, mean_abs_diff={float(difference.mean().item()):.8g}. Regenerate both tensors atomically from the same preprocessed video."
    )


def _validate_14b_i2v_first_latent_cache_file(path: Path, *, label: str) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        latents = handle.get_slice("latents")[:, :, 0:1]
        i2v_condition_latents = handle.get_slice(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)[:, :, 0:1]
    validate_14b_i2v_first_latent_consistency(
        latents, i2v_condition_latents, label=f"{label} file={path}"
    )


def validate_latent_cache_file(
    path: Path, *, expected: dict[str, Any], label: str = "Latent precompute"
) -> None:
    header = _read_cache_header(path)
    validate_latent_cache_header(
        path=path,
        header={
            "metadata": header["metadata"],
            "latent_shape": header["latents_shape"],
            "tensor_shapes": header.get("tensor_shapes", {}),
        },
        expected=expected,
        label=label,
    )
    if bool(expected.get("require_i2v_first_latent_equality", False)):
        _validate_14b_i2v_first_latent_cache_file(path, label=label)


def validate_latent_tensor_contract(
    latents: torch.Tensor, *, expected: dict[str, Any], label: str = "Latent precompute"
) -> None:
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise ValueError(
            f"{label} latents must be 5D [B,C,T,H,W], got {tuple(getattr(latents, 'shape', ()))}."
        )
    expected_tail = (
        int(expected["latent_channels"]),
        int(expected["latent_t_per_window"]),
        int(expected["latent_h"]),
        int(expected["latent_w"]),
    )
    if tuple((int(v) for v in latents.shape[1:])) != expected_tail:
        raise ValueError(
            f"{label} shape mismatch for video_model_family={expected['video_model_family']!r} vae_family={expected['vae_family']!r}: got {tuple(latents.shape)}, expected [B,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]},{expected_tail[3]}]."
        )


def _optional_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "null", "None"}:
        return None
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "null", "None"}:
        return None
    return text


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if text in {"", "null", "None", "none"}:
        return None
    return int(text)


def _resolve_action_horizon(cfg: PrecomputeConfig) -> int:
    explicit_horizon = _optional_int(getattr(cfg, "action_horizon", None))
    if explicit_horizon is not None:
        if explicit_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {explicit_horizon}.")
        return int(explicit_horizon)
    if _is_droid_dataset(cfg):
        return DROID_STAGE2_ACTION_HORIZON
    return int(cfg.num_frames) - 1


def _dataset_name(cfg: PrecomputeConfig) -> str:
    return str(getattr(cfg, "dataset_name", "")).strip().lower()


def _is_libero_dataset(cfg: PrecomputeConfig) -> bool:
    return _dataset_name(cfg) in LIBERO_DATASET_NAMES


def _is_droid_dataset(cfg: PrecomputeConfig) -> bool:
    return _dataset_name(cfg) in DROID_DATASET_NAMES


def _is_robocasa_dataset(cfg: PrecomputeConfig) -> bool:
    return _dataset_name(cfg) in ROBOCASA_DATASET_NAMES


def _uses_lerobot_metadata_stats(cfg: PrecomputeConfig) -> bool:
    return _is_libero_dataset(cfg) or _is_droid_dataset(cfg)


def _camera_key_candidates_for_dataset(cfg: PrecomputeConfig) -> tuple[tuple[str, ...], ...]:
    dataset_name = _dataset_name(cfg)
    if dataset_name == "droid":
        return (DROID_CAMERA_KEYS,)
    if _is_robocasa_dataset(cfg):
        return (ROBOCASA_CAMERA_KEYS,)
    if not _is_libero_dataset(cfg):
        return (ROBOTWIN_CAMERA_KEYS,)
    if dataset_name == "libero":
        return (LIBERO_CAMERA_KEYS,)
    return (LIBERO_CAMERA_KEYS,)


def _camera_keys_for_features(
    cfg: PrecomputeConfig, features: dict[str, Any], *, source: str | Path
) -> tuple[str, ...]:
    for camera_keys in _camera_key_candidates_for_dataset(cfg):
        if all(
            (
                isinstance(features.get(key), dict) and str(features[key].get("dtype")) == "video"
                for key in camera_keys
            )
        ):
            return camera_keys
    available_video_keys = sorted(
        (
            key
            for (key, feature) in features.items()
            if isinstance(feature, dict) and str(feature.get("dtype")) == "video"
        )
    )
    expected = ["[" + ", ".join(keys) + "]" for keys in _camera_key_candidates_for_dataset(cfg)]
    raise KeyError(
        f"LeRobot metadata is missing required video camera features for dataset_name={_dataset_name(cfg)!r}. Expected one of {expected}; available video features={available_video_keys}: {source}"
    )


def _cache_root_name(root: Path) -> str:
    base = re.sub("[^A-Za-z0-9_.-]+", "_", Path(root).name).strip("._-") or "root"
    digest = hashlib.sha1(str(Path(root).expanduser().resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{base}-{digest}"


def _cache_root_name_map(roots: Sequence[Path | str]) -> dict[Path, str]:
    resolved_roots = [Path(root).expanduser().resolve() for root in roots]
    name_counts: dict[str, int] = {}
    for root in resolved_roots:
        name_counts[root.name] = name_counts.get(root.name, 0) + 1
    return {
        root: root.name if name_counts[root.name] == 1 else _cache_root_name(root)
        for root in resolved_roots
    }


def _normalizer_mode(cfg: PrecomputeConfig) -> str:
    if _is_droid_dataset(cfg):
        return "q01/q99"
    if _is_libero_dataset(cfg):
        return "min/max"
    return "z-score"


def _normalizer_stats_src(cfg: PrecomputeConfig) -> str:
    stats_path = _optional_path(getattr(cfg, "pretrained_norm_stats", None))
    if stats_path is not None:
        return str(Path(stats_path).expanduser().resolve())
    if _uses_lerobot_metadata_stats(cfg):
        dataset_dirs = [
            Path(path).expanduser().resolve() for path in _parse_dataset_dirs(cfg.dataset_dirs)
        ]
        cache_names = _cache_root_name_map(dataset_dirs)
        dataset_names = [cache_names[path] for path in dataset_dirs]
        if _is_droid_dataset(cfg):
            return (
                "lerobot_metadata:"
                + ",".join(dataset_names)
                + f":{DROID_DREAMZERO_ACTION_CONTRACT}"
            )
        return "lerobot_metadata:" + ",".join(dataset_names)
    return ""


def _infer_video_model_family_from_vae_checkpoint(vae_checkpoint: str) -> str:
    name = Path(vae_checkpoint).name
    if name == "Wan2.1_VAE.pth":
        return "wan22_i2v_a14b"
    if name in {"Wan2.2_VAE.pth", "Wan2.2_VAE.safetensors"}:
        return "wan22_ti2v_5b"
    raise ValueError(
        f"Could not infer video_model_family from VAE checkpoint name {name!r}. Pass --video-model-family explicitly."
    )


def _resolve_video_model_family(cfg: PrecomputeConfig) -> str:
    explicit_family = _optional_text(cfg.video_model_family)
    vae_checkpoint = _optional_path(cfg.vae_checkpoint)
    if explicit_family is not None:
        return explicit_family
    if vae_checkpoint is not None:
        return _infer_video_model_family_from_vae_checkpoint(vae_checkpoint)
    if _is_robocasa_dataset(cfg):
        return "wan22_i2v_a14b"
    return "wan22_ti2v_5b"


def _precompute_cache_contract(cfg: PrecomputeConfig) -> dict[str, Any]:
    contract = latent_cache_expected_contract(
        video_model_family=_resolve_video_model_family(cfg),
        model_id=getattr(cfg, "model_id", None),
        height=int(getattr(cfg, "height")),
        width=int(getattr(cfg, "width")),
        num_frames=int(getattr(cfg, "num_frames")),
        action_video_freq_ratio=int(getattr(cfg, "action_video_freq_ratio")),
        action_horizon=_resolve_action_horizon(cfg),
        require_action_horizon_metadata=False,
    )
    visual_preprocess_id = required_robocasa_visual_preprocess_id(
        dataset_name=getattr(cfg, "dataset_name", ""),
        concat_multi_camera=getattr(cfg, "concat_multi_camera", ""),
        height=getattr(cfg, "height", -1),
        width=getattr(cfg, "width", -1),
        video_model_family=contract["video_model_family"],
    )
    if visual_preprocess_id is not None:
        contract["visual_preprocess_id"] = visual_preprocess_id
        if contract["video_model_family"] == "wan22_i2v_a14b":
            contract["require_i2v_first_latent_equality"] = True
    return contract


def _file_fingerprint(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {resolved}")
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "mtime": int(stat.st_mtime)}


def _precompute_cache_tag(cfg: PrecomputeConfig) -> str:
    """Return a short tag for the VAE/preprocessing contract that defines this cache."""
    hasher = hashlib.sha256()
    contract = _precompute_cache_contract(cfg)
    tag_payload = {
        "tag_version": 3 if contract.get("visual_preprocess_id") is not None else 2,
        "video_model_family": contract["video_model_family"],
        "i2v_condition_source": WAN22_A14B_I2V_CONDITION_SOURCE_OFFICIAL
        if str(contract["video_model_family"]) == "wan22_i2v_a14b"
        else None,
        "model_id": contract["model_id"],
        "vae_family": contract["vae_family"],
        "latent_channels": contract["latent_channels"],
        "latent_t_per_window": contract["latent_t_per_window"],
        "latent_h": contract["latent_h"],
        "latent_w": contract["latent_w"],
        "vae_z_dim": contract["vae_z_dim"],
        "vae_temporal_stride": contract["vae_temporal_stride"],
        "vae_spatial_stride": contract["vae_spatial_stride"],
        "height": contract["height"],
        "width": contract["width"],
        "num_frames": contract["num_frames"],
        "action_video_freq_ratio": contract["action_video_freq_ratio"],
        "concat_multi_camera": str(getattr(cfg, "concat_multi_camera")),
        "action_dim": int(getattr(cfg, "action_dim")),
        "proprio_dim": int(getattr(cfg, "proprio_dim")),
        "pretrained_norm_stats": _normalizer_stats_src(cfg),
        "vae_checkpoint": _file_fingerprint(str(cfg.vae_checkpoint))
        if _optional_path(cfg.vae_checkpoint) is not None
        else None,
        "model_base_path": None
        if _optional_path(cfg.vae_checkpoint) is not None
        else str(Path(cfg.model_base_path).expanduser().resolve()),
        "mixed_precision": _normalize_mixed_precision(cfg.mixed_precision),
    }
    if contract.get("visual_preprocess_id") is not None:
        tag_payload["visual_preprocess_id"] = contract["visual_preprocess_id"]
    hasher.update(json.dumps(tag_payload, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()[:16]


def _decode_all_frames(video_path: Path) -> torch.Tensor:
    """Decode every frame of ``video_path`` via torchcodec. Returns [T, 3, H, W] uint8."""
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
    meta = dec.metadata
    num_frames = getattr(meta, "num_frames", None)
    if num_frames is None or int(num_frames) <= 0:
        duration = float(getattr(meta, "duration_seconds", 0.0))
        fps = float(getattr(meta, "average_fps", 0.0))
        num_frames = int(round(duration * fps)) if duration and fps else 0
    if num_frames <= 0:
        raise RuntimeError(f"Could not determine frame count for {video_path}")
    indices = list(range(int(num_frames)))
    batch = dec.get_frames_at(indices=indices)
    return batch.data.contiguous()


def _decode_frames_played_at(video_path: Path, seconds: torch.Tensor) -> torch.Tensor:
    """Decode frames at absolute playback timestamps. Returns [T, 3, H, W] uint8."""
    from torchcodec.decoders import VideoDecoder

    if seconds.ndim != 1:
        raise ValueError(
            f"Expected 1D timestamp tensor for {video_path}, got {tuple(seconds.shape)}"
        )
    if int(seconds.numel()) <= 0:
        raise ValueError(f"Cannot decode an empty timestamp window from {video_path}")
    dec = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
    batch = dec.get_frames_played_at(seconds=[float(v) for v in seconds.tolist()])
    return batch.data.contiguous()


def _apply_robotwin_multicam(video_raw: torch.Tensor) -> torch.Tensor:
    """Match ``RobotVideoDataset._get`` robotwin branch. In: [3cams, T, 3, H, W] uint8; out: [T, 3, 384, 320] fp32 in [-1, 1]."""
    assert video_raw.dim() == 5 and video_raw.shape[0] == 3, (
        f"Expected [3cams, T, 3, H, W], got {tuple(video_raw.shape)}"
    )
    cam_top = transforms_F.resize(
        video_raw[0],
        size=[256, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = transforms_F.resize(
        video_raw[1],
        size=[128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = transforms_F.resize(
        video_raw[2],
        size=[128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    bottom = torch.cat([cam_left, cam_right], dim=-1)
    video = torch.cat([cam_top, bottom], dim=-2)
    video = video.to(dtype=torch.float32) / 255.0
    video = (video - 0.5) / 0.5
    return video


def _apply_horizontal_multicam(video_raw: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if video_raw.dim() != 5:
        raise ValueError(f"Expected [cams, T, 3, H, W], got {tuple(video_raw.shape)}")
    num_cameras = int(video_raw.shape[0])
    if num_cameras <= 0:
        raise ValueError("Horizontal multi-camera concat requires at least one camera.")
    if int(width) % num_cameras != 0:
        raise ValueError(
            f"Horizontal concat width={width} must be divisible by num_cameras={num_cameras}."
        )
    per_camera_width = int(width) // num_cameras
    resized = [
        transforms_F.resize(
            video_raw[i],
            size=[int(height), per_camera_width],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        for i in range(num_cameras)
    ]
    video = torch.cat(resized, dim=-1)
    video = video.to(dtype=torch.float32) / 255.0
    return (video - 0.5) / 0.5


def _apply_robocasa_dreamzero_multicam(
    video_raw: torch.Tensor, *, height: int, width: int, dataset_name: str = "robocasa"
) -> torch.Tensor:
    """Compose wrist-over-agent views using direct antialiased bilinear resizing."""
    if (int(height), int(width)) != (352, 352):
        raise ValueError(
            f"RoboCasa DreamZero concat requires height=352,width=352, got {height}x{width}."
        )
    expected_raw_size = expected_robocasa_cache_raw_camera_size(dataset_name)
    if (
        required_robocasa_visual_preprocess_id(
            dataset_name=dataset_name,
            concat_multi_camera=ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA,
            height=height,
            width=width,
            video_model_family="wan22_i2v_a14b",
        )
        is None
    ):
        expected_raw_size = tuple((int(v) for v in video_raw.shape[-2:]))
    return normalize_uint8_video(
        compose_robocasa_dreamzero_uint8(
            video_raw, output_size=(int(height), int(width)), expected_raw_size=expected_raw_size
        )
    )


def _dreamzero_center_crop_resize(
    frames: torch.Tensor, *, crop_scale: float, height: int, width: int
) -> torch.Tensor:
    if frames.dim() != 4:
        raise ValueError(f"Expected [T, 3, H, W] frames, got {tuple(frames.shape)}")
    (_, _, raw_h, raw_w) = frames.shape
    crop_h = int(raw_h * float(crop_scale))
    crop_w = int(raw_w * float(crop_scale))
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(
            f"Invalid DreamZero DROID crop from raw shape {raw_h}x{raw_w} and crop_scale={crop_scale}."
        )
    frames = frames.to(dtype=torch.float32) / 255.0
    frames = transforms_F.center_crop(frames, output_size=[crop_h, crop_w])
    return transforms_F.resize(
        frames,
        size=[int(height), int(width)],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )


def _apply_droid_grid_multicam(video_raw: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Match DreamZero DROID video layout.

    In: [3cams, T, 3, H, W] uint8; out: [T, 3, 352, 640] fp32 in [-1, 1].
    """
    if video_raw.dim() != 5 or int(video_raw.shape[0]) != 3:
        raise ValueError(
            f"DROID grid concat requires [3cams, T, 3, H, W], got {tuple(video_raw.shape)}"
        )
    if (int(height), int(width)) != (352, 640):
        raise ValueError(
            f"DROID grid concat matches DreamZero Wan2.1 video preprocessing and requires 352x640, got {height}x{width}."
        )
    per_camera_h = int(height) // 2
    per_camera_w = int(width) // 2
    left_exterior = _dreamzero_center_crop_resize(
        video_raw[0], crop_scale=0.95, height=per_camera_h, width=per_camera_w
    )
    right_exterior = _dreamzero_center_crop_resize(
        video_raw[1], crop_scale=0.95, height=per_camera_h, width=per_camera_w
    )
    wrist = _dreamzero_center_crop_resize(
        video_raw[2], crop_scale=0.95, height=per_camera_h, width=per_camera_w
    )
    wrist_wide = wrist.repeat_interleave(2, dim=-1)
    bottom = torch.cat([left_exterior, right_exterior], dim=-1)
    video = torch.cat([wrist_wide, bottom], dim=-2)
    return (video - 0.5) / 0.5


def _apply_multicam(video_raw: torch.Tensor, cfg: PrecomputeConfig) -> torch.Tensor:
    concat_mode = str(cfg.concat_multi_camera).strip().lower()
    if concat_mode == ROBOTWIN_CONCAT_MULTI_CAMERA:
        return _apply_robotwin_multicam(video_raw)
    if concat_mode == HORIZONTAL_CONCAT_MULTI_CAMERA:
        return _apply_horizontal_multicam(video_raw, height=int(cfg.height), width=int(cfg.width))
    if concat_mode == DROID_GRID_CONCAT_MULTI_CAMERA:
        return _apply_droid_grid_multicam(video_raw, height=int(cfg.height), width=int(cfg.width))
    if concat_mode == ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA:
        return _apply_robocasa_dreamzero_multicam(
            video_raw,
            height=int(cfg.height),
            width=int(cfg.width),
            dataset_name=str(cfg.dataset_name),
        )
    raise ValueError(f"Unsupported concat_multi_camera={cfg.concat_multi_camera!r}.")


def _encode_windows(
    encoder: PrecomputeVAEModel,
    video_processed: torch.Tensor,
    *,
    num_frames: int,
    action_video_freq_ratio: int,
    batch_size: int,
) -> torch.Tensor:
    """Build one sparse latent window per episode frame and VAE-encode it."""
    T_ep = int(video_processed.shape[0])
    if T_ep <= 0:
        raise ValueError("Latent precompute requires episodes with at least one frame.")
    return _encode_window_range(
        encoder,
        video_processed,
        num_frames=num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
        batch_size=batch_size,
        start_index=0,
        end_index=T_ep,
    )


def _encode_14b_i2v_condition_windows(
    encoder: PrecomputeVAEModel,
    video_processed: torch.Tensor,
    *,
    num_frames: int,
    action_video_freq_ratio: int,
    batch_size: int,
) -> torch.Tensor:
    """Build official 14B I2V pseudo-video windows and VAE-encode them."""
    T_ep = int(video_processed.shape[0])
    if T_ep <= 0:
        raise ValueError("14B I2V condition precompute requires episodes with at least one frame.")
    return _encode_window_range(
        encoder,
        video_processed,
        num_frames=num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
        batch_size=batch_size,
        start_index=0,
        end_index=T_ep,
        a14b_i2v_condition=True,
    )


def _encode_window_range(
    encoder: PrecomputeVAEModel,
    video_processed: torch.Tensor,
    *,
    num_frames: int,
    action_video_freq_ratio: int,
    batch_size: int,
    start_index: int,
    end_index: int,
    a14b_i2v_condition: bool = False,
) -> torch.Tensor:
    """Build sparse latent windows for start indices in ``[start_index, end_index)``."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"VAE encode batch_size must be positive, got {batch_size}.")
    T_ep = int(video_processed.shape[0])
    if T_ep <= 0:
        raise ValueError("Latent precompute requires episodes with at least one frame.")
    start_index = int(start_index)
    end_index = int(end_index)
    if start_index < 0 or end_index < start_index or end_index > T_ep:
        raise ValueError(
            f"Invalid encode window range [{start_index}, {end_index}) for episode length {T_ep}."
        )
    subsample = torch.arange(0, num_frames, action_video_freq_ratio, dtype=torch.long)
    assert subsample.numel() > 1
    n_samples = end_index - start_index
    if n_samples == 0:
        probe_idx = subsample.clamp(max=T_ep - 1)
        probe = video_processed[probe_idx].unsqueeze(0).permute(0, 2, 1, 3, 4)
        if a14b_i2v_condition:
            condition_probe = torch.zeros_like(probe)
            condition_probe[:, :, 0:1] = probe[:, :, 0:1]
            probe = condition_probe
        probe = probe.to(device=encoder.device, dtype=encoder.torch_dtype)
        if bool(encoder.compile_enabled) and int(probe.shape[0]) < batch_size:
            pad_count = batch_size - int(probe.shape[0])
            probe = torch.cat([probe, probe[-1:].expand(pad_count, *probe.shape[1:])], dim=0)
        with torch.no_grad():
            probe_latents = encoder.encode_video_latents(probe)
        return probe_latents.new_zeros((0, *probe_latents.shape[1:]), dtype=torch.float16).cpu()
    all_latents: list[torch.Tensor] = []
    max_frame_index = T_ep - 1
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        starts = torch.arange(start_index + batch_start, start_index + batch_end, dtype=torch.long)
        window_idx = (starts.unsqueeze(1) + subsample.unsqueeze(0)).clamp(max=max_frame_index)
        windows = video_processed[window_idx]
        windows = windows.permute(0, 2, 1, 3, 4).contiguous()
        if a14b_i2v_condition:
            condition_windows = torch.zeros_like(windows)
            condition_windows[:, :, 0:1] = windows[:, :, 0:1]
            windows = condition_windows
        windows = windows.to(device=encoder.device, dtype=encoder.torch_dtype, non_blocking=True)
        real_batch_size = int(windows.shape[0])
        if real_batch_size <= 0:
            raise ValueError("VAE encode produced an empty window batch before padding.")
        if bool(encoder.compile_enabled) and real_batch_size < batch_size:
            pad_count = batch_size - real_batch_size
            windows = torch.cat(
                [windows, windows[-1:].expand(pad_count, *windows.shape[1:])], dim=0
            )
        with torch.no_grad():
            latents = encoder.encode_video_latents(windows).detach()
        latents = latents[:real_batch_size]
        all_latents.append(latents.to(torch.float16).cpu())
    return torch.cat(all_latents, dim=0)


def _lerobot_codebase_version(info: dict[str, Any]) -> str:
    return str(info.get("codebase_version", "")).strip().lower()


def _is_lerobot_v30_info(info: dict[str, Any]) -> bool:
    return _lerobot_codebase_version(info).startswith("v3")


def _load_lerobot_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    if not isinstance(info, dict):
        raise ValueError(f"LeRobot metadata must be a JSON object: {info_path}")
    return info


def _load_lerobot_episode_rows(dataset_dir: Path) -> dict[int, dict[str, Any]]:
    episode_dir = dataset_dir / "meta" / "episodes"
    episode_files = sorted(episode_dir.rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"Missing LeRobot v3.0 episode parquet shards under {episode_dir}")
    rows: dict[int, dict[str, Any]] = {}
    for path in episode_files:
        table = pq.read_table(str(path))
        data = table.to_pydict()
        if "episode_index" not in data:
            raise KeyError(f"{path} is missing required column 'episode_index'.")
        for i in range(table.num_rows):
            row = {key: values[i] for (key, values) in data.items()}
            ep_idx = int(row["episode_index"])
            if ep_idx in rows:
                raise ValueError(
                    f"Duplicate episode_index={ep_idx} in LeRobot episode metadata under {episode_dir}"
                )
            rows[ep_idx] = row
    if not rows:
        raise RuntimeError(f"No LeRobot v3.0 episode rows loaded from {episode_dir}")
    return dict(sorted(rows.items()))


def _lerobot_row_int(row: dict[str, Any] | None, key: str, default: int) -> int:
    if row is None or key not in row or row[key] is None:
        return int(default)
    return int(row[key])


def _format_lerobot_path_template(
    template: str,
    *,
    episode_index: int,
    chunks_size: int = 1000,
    video_key: str | None = None,
    episode_row: dict[str, Any] | None = None,
) -> Path:
    default_chunk = int(episode_index) // int(chunks_size)
    default_file = int(episode_index) % int(chunks_size)
    if video_key is None:
        chunk_index = _lerobot_row_int(episode_row, "data/chunk_index", default_chunk)
        file_index = _lerobot_row_int(episode_row, "data/file_index", default_file)
    else:
        prefix = f"videos/{video_key}"
        chunk_index = _lerobot_row_int(episode_row, f"{prefix}/chunk_index", default_chunk)
        file_index = _lerobot_row_int(episode_row, f"{prefix}/file_index", default_file)
    try:
        return Path(
            template.format(
                episode_chunk=chunk_index,
                chunk_index=chunk_index,
                file_index=file_index,
                video_key="" if video_key is None else video_key,
                episode_index=int(episode_index),
            )
        )
    except KeyError as exc:
        raise ValueError(
            f"Unsupported LeRobot path placeholder {exc!s} in template {template!r}."
        ) from exc


@dataclass(frozen=True)
class PrecomputeDatasetMetadata:
    root: Path
    info: dict[str, Any]

    @property
    def features(self) -> dict[str, Any]:
        features = self.info.get("features")
        if not isinstance(features, dict):
            raise ValueError(
                f"LeRobot metadata features must be a mapping: {self.root / 'meta' / 'info.json'}"
            )
        return features

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    def get_data_file_path(self, ep_index: int) -> Path:
        return _format_lerobot_path_template(
            str(self.info["data_path"]),
            episode_index=int(ep_index),
            chunks_size=int(self.info.get("chunks_size", 1000)),
        )

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        return _format_lerobot_path_template(
            str(self.info["video_path"]),
            episode_index=int(ep_index),
            chunks_size=int(self.info.get("chunks_size", 1000)),
            video_key=vid_key,
        )


def _load_precompute_dataset_metadata(dataset_dir: Path) -> PrecomputeDatasetMetadata:
    return PrecomputeDatasetMetadata(root=dataset_dir, info=_load_lerobot_info(dataset_dir))


def _resolve_lerobot_data_path(
    dataset_root: Path,
    meta: PrecomputeDatasetMetadata,
    actual_ep_idx: int,
    episode_row: dict[str, Any] | None,
) -> Path:
    if episode_row is None:
        return dataset_root / meta.get_data_file_path(actual_ep_idx)
    return dataset_root / _format_lerobot_path_template(
        str(meta.info["data_path"]),
        episode_index=actual_ep_idx,
        chunks_size=int(meta.info.get("chunks_size", 1000)),
        episode_row=episode_row,
    )


def _resolve_lerobot_video_path(
    dataset_root: Path,
    meta: PrecomputeDatasetMetadata,
    actual_ep_idx: int,
    camera_key: str,
    episode_row: dict[str, Any] | None,
) -> Path:
    if episode_row is None:
        return dataset_root / meta.get_video_file_path(actual_ep_idx, camera_key)
    return dataset_root / _format_lerobot_path_template(
        str(meta.info["video_path"]),
        episode_index=actual_ep_idx,
        chunks_size=int(meta.info.get("chunks_size", 1000)),
        video_key=camera_key,
        episode_row=episode_row,
    )


@functools.lru_cache(maxsize=4)
def _read_episode_parquet_table_cached(parquet_path: str) -> Any:
    return pq.read_table(
        parquet_path,
        columns=[
            "action",
            "observation.state",
            "task_index",
            "timestamp",
            "episode_index",
            "index",
        ],
    )


def _column_to_tensor(table: Any, column_name: str, dtype: torch.dtype) -> torch.Tensor:
    arr = table[column_name].to_numpy()
    if arr.dtype == object:
        arr = np.stack(arr)
    tensor = torch.from_numpy(arr.copy()).to(dtype=dtype)
    return tensor


def _read_episode_parquet(parquet_path: Path) -> dict[str, torch.Tensor]:
    table = pq.read_table(str(parquet_path))
    return {
        "action": _column_to_tensor(table, "action", torch.float32),
        "state": _column_to_tensor(table, "observation.state", torch.float32),
        "task_indices": _column_to_tensor(table, "task_index", torch.int64).to(torch.int32),
        "timestamps": _column_to_tensor(table, "timestamp", torch.float64),
    }


def _read_episode_parquet_slice(
    parquet_path: Path, *, episode_index: int, episode_row: dict[str, Any]
) -> dict[str, torch.Tensor]:
    table = _read_episode_parquet_table_cached(str(parquet_path))
    if "dataset_from_index" in episode_row and "dataset_to_index" in episode_row:
        first_index = int(table["index"][0].as_py())
        start = int(episode_row["dataset_from_index"]) - first_index
        end = int(episode_row["dataset_to_index"]) - first_index
    else:
        episode_indices = _column_to_tensor(table, "episode_index", torch.int64)
        matches = torch.nonzero(episode_indices == int(episode_index), as_tuple=False).flatten()
        if int(matches.numel()) <= 0:
            raise ValueError(f"No rows for episode_index={episode_index} in {parquet_path}")
        start = int(matches[0].item())
        end = int(matches[-1].item()) + 1
    if start < 0 or end <= start or end > int(table.num_rows):
        raise ValueError(
            f"Invalid row slice [{start}, {end}) for episode_index={episode_index} in {parquet_path} with table rows={table.num_rows}."
        )
    episode_table = table.slice(start, end - start)
    episode_indices = _column_to_tensor(episode_table, "episode_index", torch.int64)
    if not bool(torch.all(episode_indices == int(episode_index)).item()):
        raise ValueError(
            f"Parquet slice for episode_index={episode_index} contains other episodes: {parquet_path}"
        )
    return {
        "action": _column_to_tensor(episode_table, "action", torch.float32),
        "state": _column_to_tensor(episode_table, "observation.state", torch.float32),
        "task_indices": _column_to_tensor(episode_table, "task_index", torch.int64).to(torch.int32),
        "timestamps": _column_to_tensor(episode_table, "timestamp", torch.float64),
    }


def _build_robotwin_shape_meta(cfg: PrecomputeConfig) -> dict[str, Any]:
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


def _build_libero_shape_meta(cfg: PrecomputeConfig) -> dict[str, Any]:
    num_cameras = 2
    if int(cfg.width) % num_cameras != 0:
        raise ValueError(
            f"LIBERO horizontal shape metadata requires width divisible by {num_cameras}, got {cfg.width}."
        )
    per_camera_width = int(cfg.width) // num_cameras
    return {
        "images": [
            {
                "key": "image",
                "raw_shape": [3, 512, 512],
                "shape": [3, int(cfg.height), per_camera_width],
            },
            {
                "key": "wrist_image",
                "raw_shape": [3, 512, 512],
                "shape": [3, int(cfg.height), per_camera_width],
            },
        ],
        "action": [
            {"key": "default", "raw_shape": int(cfg.action_dim), "shape": int(cfg.action_dim)}
        ],
        "state": [
            {"key": "default", "raw_shape": int(cfg.proprio_dim), "shape": int(cfg.proprio_dim)}
        ],
    }


def _build_droid_shape_meta(cfg: PrecomputeConfig) -> dict[str, Any]:
    return {
        "images": [
            {"key": "exterior_image_1_left", "raw_shape": [3, 180, 320], "shape": [3, 176, 320]},
            {"key": "exterior_image_2_left", "raw_shape": [3, 180, 320], "shape": [3, 176, 320]},
            {"key": "wrist_image_left", "raw_shape": [3, 180, 320], "shape": [3, 176, 320]},
        ],
        "action": [
            {"key": DROID_JOINT_POSITION_KEY, "raw_shape": 7, "shape": 7},
            {"key": DROID_GRIPPER_POSITION_KEY, "raw_shape": 1, "shape": 1},
        ],
        "state": [
            {"key": DROID_JOINT_POSITION_KEY, "raw_shape": 7, "shape": 7},
            {"key": DROID_GRIPPER_POSITION_KEY, "raw_shape": 1, "shape": 1},
        ],
    }


def _build_robocasa_shape_meta(cfg: PrecomputeConfig) -> dict[str, Any]:
    return {
        "images": [
            {"key": "agentview_left", "raw_shape": [3, 224, 224], "shape": [3, 176, 176]},
            {"key": "agentview_right", "raw_shape": [3, 224, 224], "shape": [3, 176, 176]},
            {"key": "wrist", "raw_shape": [3, 224, 224], "shape": [3, 176, 176]},
        ],
        "action": [
            {"key": "default", "raw_shape": ROBOCASA_ACTION_DIM, "shape": int(cfg.action_dim)}
        ],
        "state": [
            {"key": "default", "raw_shape": ROBOCASA_PROPRIO_DIM, "shape": int(cfg.proprio_dim)}
        ],
    }


def _build_shape_meta(cfg: PrecomputeConfig) -> dict[str, Any]:
    dataset_name = _dataset_name(cfg)
    if _is_libero_dataset(cfg):
        return _build_libero_shape_meta(cfg)
    if _is_droid_dataset(cfg):
        return _build_droid_shape_meta(cfg)
    if _is_robocasa_dataset(cfg):
        return _build_robocasa_shape_meta(cfg)
    if dataset_name in ROBOTWIN_DATASET_NAMES:
        return _build_robotwin_shape_meta(cfg)
    raise ValueError(f"Unsupported dataset_name={cfg.dataset_name!r}.")


def _normalizer_stats_from_lerobot_stats(
    stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:

    def convert_feature(feature_key: str) -> dict[str, torch.Tensor]:
        if feature_key not in stats:
            raise KeyError(f"LeRobot metadata stats are missing required feature {feature_key!r}.")
        source = stats[feature_key]
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

    return {
        "action": {"default": convert_feature("action")},
        "state": {"default": convert_feature("observation.state")},
    }


def _load_droid_modality_config(root: Path) -> dict[str, Any]:
    path = Path(root) / "meta" / "modality.json"
    if not path.exists():
        raise FileNotFoundError(f"DROID precompute requires DreamZero modality metadata: {path}")
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


def _slice_lerobot_feature_stats(
    source: dict[str, Any], *, feature_key: str, start: int, end: int
) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        if stat_name not in source:
            raise KeyError(f"DROID metadata stats for {feature_key!r} are missing {stat_name!r}.")
        values = np.asarray(source[stat_name], dtype=np.float32)
        if values.ndim != 1 or int(values.shape[0]) < end:
            raise ValueError(
                f"DROID metadata stats for {feature_key!r}.{stat_name} must be 1D with at least {end} values, got shape={values.shape}."
            )
        sliced[stat_name] = values[start:end]
    return sliced


def _convert_feature_stats(source: dict[str, Any], *, feature_key: str) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        if stat_name not in source:
            raise KeyError(f"DROID DreamZero stats for {feature_key!r} are missing {stat_name!r}.")
        tensor = torch.as_tensor(source[stat_name], dtype=torch.float32)
        converted[f"global_{stat_name}"] = tensor
        converted[f"stepwise_{stat_name}"] = tensor.unsqueeze(0)
    return converted


def _load_droid_dreamzero_stats(
    dataset_root: Path,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    meta = LeRobotDatasetMetadata(repo_id=str(dataset_root), root=dataset_root)
    stats = meta.stats
    if "action" not in stats or "observation.state" not in stats:
        raise KeyError(
            f"DROID DreamZero metadata for root={dataset_root} must contain ('action', 'observation.state') stats."
        )
    relative_stats_path = dataset_root / "meta" / DROID_DREAMZERO_RELATIVE_STATS_FILE
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
        dataset_root, "state", DROID_JOINT_POSITION_KEY
    )
    (state_gripper_start, state_gripper_end) = _droid_modality_range(
        dataset_root, "state", DROID_GRIPPER_POSITION_KEY
    )
    (action_gripper_start, action_gripper_end) = _droid_modality_range(
        dataset_root, "action", DROID_GRIPPER_POSITION_KEY
    )
    return {
        "action": {
            DROID_JOINT_POSITION_KEY: _convert_feature_stats(
                joint_relative_stats, feature_key="action.joint_position"
            ),
            DROID_GRIPPER_POSITION_KEY: _convert_feature_stats(
                _slice_lerobot_feature_stats(
                    stats["action"],
                    feature_key="action.gripper_position",
                    start=action_gripper_start,
                    end=action_gripper_end,
                ),
                feature_key="action.gripper_position",
            ),
        },
        "state": {
            DROID_JOINT_POSITION_KEY: _convert_feature_stats(
                _slice_lerobot_feature_stats(
                    stats["observation.state"],
                    feature_key="state.joint_position",
                    start=state_joint_start,
                    end=state_joint_end,
                ),
                feature_key="state.joint_position",
            ),
            DROID_GRIPPER_POSITION_KEY: _convert_feature_stats(
                _slice_lerobot_feature_stats(
                    stats["observation.state"],
                    feature_key="state.gripper_position",
                    start=state_gripper_start,
                    end=state_gripper_end,
                ),
                feature_key="state.gripper_position",
            ),
        },
    }


def _load_metadata_stats(
    dataset_dirs: Sequence[str], *, cfg: PrecomputeConfig
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    if _is_droid_dataset(cfg):
        if len(dataset_dirs) != 1:
            raise ValueError(
                f"DROID precompute currently supports exactly one dataset_dir, got {dataset_dirs}."
            )
        return _load_droid_dreamzero_stats(Path(dataset_dirs[0]).expanduser().resolve())
    stats_list = []
    for ds_dir in dataset_dirs:
        root = Path(ds_dir).expanduser().resolve()
        meta = LeRobotDatasetMetadata(repo_id=str(root), root=root)
        if not meta.stats:
            raise ValueError(f"LeRobot metadata stats are empty for dataset_dir={root}.")
        stats_list.append(meta.stats)
    if len(stats_list) == 1:
        return _normalizer_stats_from_lerobot_stats(stats_list[0])
    return _normalizer_stats_from_lerobot_stats(aggregate_stats(stats_list))


def _build_normalizer(cfg: PrecomputeConfig) -> LinearNormalizer:
    shape_meta = _build_shape_meta(cfg)
    stats_path = _optional_path(getattr(cfg, "pretrained_norm_stats", None))
    resolved_stats_path: Path | None = None
    if stats_path is not None:
        resolved_stats_path = Path(stats_path).expanduser().resolve()
        stats = load_dataset_stats_from_json(str(resolved_stats_path))
    elif _uses_lerobot_metadata_stats(cfg):
        stats = _load_metadata_stats(_parse_dataset_dirs(cfg.dataset_dirs), cfg=cfg)
    else:
        raise ValueError(
            "Precompute requires pretrained_norm_stats unless the dataset uses LeRobot metadata stats."
        )
    return LinearNormalizer(
        shape_meta=shape_meta,
        use_stepwise_action_norm=False,
        default_mode=_normalizer_mode(cfg),
        exception_mode=None,
        stats=stats,
    )


def _normalize_episode(
    normalizer: LinearNormalizer,
    action_raw: torch.Tensor,
    state_raw: torch.Tensor,
    *,
    cfg: PrecomputeConfig,
    dataset_root: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_droid_dataset(cfg):
        (state_joint_start, state_joint_end) = _droid_modality_range(
            dataset_root, "state", DROID_JOINT_POSITION_KEY
        )
        (state_gripper_start, state_gripper_end) = _droid_modality_range(
            dataset_root, "state", DROID_GRIPPER_POSITION_KEY
        )
        (action_joint_start, action_joint_end) = _droid_modality_range(
            dataset_root, "action", DROID_JOINT_POSITION_KEY
        )
        (action_gripper_start, action_gripper_end) = _droid_modality_range(
            dataset_root, "action", DROID_GRIPPER_POSITION_KEY
        )
        state_joint = state_raw[:, state_joint_start:state_joint_end]
        batch = {
            "action": {
                DROID_JOINT_POSITION_KEY: action_raw[:, action_joint_start:action_joint_end]
                - state_joint[0].unsqueeze(0),
                DROID_GRIPPER_POSITION_KEY: action_raw[:, action_gripper_start:action_gripper_end],
            },
            "state": {
                DROID_JOINT_POSITION_KEY: state_joint,
                DROID_GRIPPER_POSITION_KEY: state_raw[:, state_gripper_start:state_gripper_end],
            },
        }
        normalized = normalizer.forward(batch)
        action = torch.cat(
            [
                normalized["action"][DROID_JOINT_POSITION_KEY],
                normalized["action"][DROID_GRIPPER_POSITION_KEY],
            ],
            dim=-1,
        )
        state = torch.cat(
            [
                normalized["state"][DROID_JOINT_POSITION_KEY],
                normalized["state"][DROID_GRIPPER_POSITION_KEY],
            ],
            dim=-1,
        )
        return (action, state)
    batch = {"action": {"default": action_raw}, "state": {"default": state_raw}}
    normalized = normalizer.forward(batch)
    return (normalized["action"]["default"], normalized["state"]["default"])


def _droid_cache_extra_tensors(
    *, dataset_root: Path, action_raw: torch.Tensor, state_raw: torch.Tensor
) -> dict[str, torch.Tensor]:
    (state_joint_start, state_joint_end) = _droid_modality_range(
        dataset_root, "state", DROID_JOINT_POSITION_KEY
    )
    (state_gripper_start, state_gripper_end) = _droid_modality_range(
        dataset_root, "state", DROID_GRIPPER_POSITION_KEY
    )
    (action_joint_start, action_joint_end) = _droid_modality_range(
        dataset_root, "action", DROID_JOINT_POSITION_KEY
    )
    (action_gripper_start, action_gripper_end) = _droid_modality_range(
        dataset_root, "action", DROID_GRIPPER_POSITION_KEY
    )
    return {
        "droid_action_joint_position": action_raw[:, action_joint_start:action_joint_end]
        .to(torch.float32)
        .contiguous(),
        "droid_action_gripper_position": action_raw[:, action_gripper_start:action_gripper_end]
        .to(torch.float32)
        .contiguous(),
        "droid_state_joint_position": state_raw[:, state_joint_start:state_joint_end]
        .to(torch.float32)
        .contiguous(),
        "droid_state_gripper_position": state_raw[:, state_gripper_start:state_gripper_end]
        .to(torch.float32)
        .contiguous(),
    }


def _cache_output_dir(
    cfg: PrecomputeConfig,
    cache_root: Path,
    dataset_dir: Path,
    *,
    exact_cache_dir: bool = False,
    dataset_dirs: Sequence[Path | str] | None = None,
) -> Path:
    root_names = _cache_root_name_map([dataset_dir] if dataset_dirs is None else dataset_dirs)
    root_name = root_names[Path(dataset_dir).expanduser().resolve()]
    if exact_cache_dir:
        return cache_root / root_name
    cache_tag = _precompute_cache_tag(cfg)
    return cache_root / cache_tag / root_name


def _validate_lerobot_video_metadata(dataset_dir: Path, cfg: PrecomputeConfig) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    info = _load_lerobot_info(dataset_dir)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"LeRobot metadata features must be a mapping: {info_path}")
    data_template = info.get("data_path")
    if not isinstance(data_template, str) or not data_template.strip():
        raise ValueError(f"LeRobot metadata is missing data_path: {info_path}")
    chunks_size = int(info.get("chunks_size", 1000))
    if chunks_size <= 0:
        raise ValueError(
            f"LeRobot metadata chunks_size must be positive, got {chunks_size} in {info_path}."
        )
    first_episode_row = (
        _load_lerobot_episode_rows(dataset_dir).get(0) if _is_lerobot_v30_info(info) else None
    )
    if _is_lerobot_v30_info(info) and first_episode_row is None:
        raise KeyError(
            f"LeRobot v3.0 metadata is missing episode_index=0 under {dataset_dir / 'meta' / 'episodes'}"
        )
    first_data_path = dataset_dir / _format_lerobot_path_template(
        data_template, episode_index=0, chunks_size=chunks_size, episode_row=first_episode_row
    )
    if not first_data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory is missing the first expected parquet file: {first_data_path}. Wait for the LeRobot data download to finish before precomputing latents."
        )
    video_template = info.get("video_path")
    if not isinstance(video_template, str) or not video_template.strip():
        raise ValueError(
            f"LeRobot metadata is missing video_path: {info_path}. This precompute job requires downloaded MP4 videos."
        )
    for camera_key in _camera_keys_for_features(cfg, features, source=info_path):
        feature = features.get(camera_key)
        if not isinstance(feature, dict):
            raise KeyError(
                f"LeRobot metadata is missing required video feature {camera_key!r}: {info_path}"
            )
        if str(feature.get("dtype")) != "video":
            raise ValueError(
                f"LeRobot feature {camera_key!r} must have dtype='video' for latent precompute, got {feature.get('dtype')!r} in {info_path}."
            )
        first_video_path = dataset_dir / _format_lerobot_path_template(
            video_template,
            episode_index=0,
            chunks_size=chunks_size,
            video_key=camera_key,
            episode_row=first_episode_row,
        )
        if not first_video_path.exists():
            raise FileNotFoundError(
                f"Dataset directory is missing the first expected video for {camera_key!r}: {first_video_path}. Wait for the LeRobot video download to finish before precomputing latents."
            )


def validate_precompute_config(cfg: PrecomputeConfig) -> None:
    dataset_name = _dataset_name(cfg)
    if dataset_name not in {"robotwin", "libero", "robocasa", "droid"}:
        raise ValueError("dataset_name must be robotwin, libero, robocasa, or droid.")
    dataset_dirs = _parse_dataset_dirs(cfg.dataset_dirs)
    for dataset_dir_text in dataset_dirs:
        dataset_dir = Path(dataset_dir_text).expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        if not (dataset_dir / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"Dataset directory is missing LeRobot metadata: {dataset_dir / 'meta' / 'info.json'}"
            )
        _validate_lerobot_video_metadata(dataset_dir, cfg)
    stats_text = _optional_path(cfg.pretrained_norm_stats)
    if stats_text is not None:
        stats_path = Path(stats_text).expanduser().resolve()
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Precompute required path `pretrained_norm_stats` not found: {stats_path}"
            )
    elif not _uses_lerobot_metadata_stats(cfg):
        raise ValueError(
            "Precompute requires pretrained_norm_stats unless the dataset uses LeRobot metadata stats."
        )
    concat_mode = str(cfg.concat_multi_camera).strip().lower()
    if _is_libero_dataset(cfg):
        if concat_mode != HORIZONTAL_CONCAT_MULTI_CAMERA:
            raise ValueError("LIBERO latent precompute requires concat_multi_camera='horizontal'.")
        if (int(cfg.height), int(cfg.width)) not in {(224, 448), (352, 640)}:
            raise ValueError(
                f"LIBERO latent precompute requires height/width to be 224x448 or 352x640; got {cfg.height}x{cfg.width}."
            )
        if (int(cfg.action_dim), int(cfg.proprio_dim)) != (7, 8):
            raise ValueError(
                f"LIBERO latent precompute requires action_dim=7,proprio_dim=8; got action_dim={cfg.action_dim}, proprio_dim={cfg.proprio_dim}."
            )
    elif _is_droid_dataset(cfg):
        if len(dataset_dirs) != 1:
            raise ValueError(
                f"DROID precompute currently supports exactly one dataset_dir, got {cfg.dataset_dirs}."
            )
        droid_root = Path(dataset_dirs[0]).expanduser().resolve()
        _load_droid_modality_config(droid_root)
        relative_stats_path = droid_root / "meta" / DROID_DREAMZERO_RELATIVE_STATS_FILE
        if not relative_stats_path.exists():
            raise FileNotFoundError(
                f"DROID DreamZero relative action stats not found: {relative_stats_path}"
            )
        if concat_mode != DROID_GRID_CONCAT_MULTI_CAMERA:
            raise ValueError("DROID latent precompute requires concat_multi_camera='droid_grid'.")
        if (int(cfg.height), int(cfg.width), int(cfg.action_dim), int(cfg.proprio_dim)) != (
            352,
            640,
            8,
            8,
        ):
            raise ValueError(
                f"DROID latent precompute requires height=352,width=640,action_dim=8,proprio_dim=8; got {cfg.height}x{cfg.width}, action_dim={cfg.action_dim}, proprio_dim={cfg.proprio_dim}."
            )
    elif _is_robocasa_dataset(cfg):
        if len(dataset_dirs) != 1:
            raise ValueError(
                f"RoboCasa precompute currently supports exactly one dataset_dir, got {cfg.dataset_dirs}."
            )
        if concat_mode != ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA:
            raise ValueError(
                f"RoboCasa latent precompute requires concat_multi_camera={ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA!r}."
            )
        if (int(cfg.height), int(cfg.width), int(cfg.action_dim), int(cfg.proprio_dim)) != (
            352,
            352,
            ROBOCASA_ACTION_DIM,
            ROBOCASA_PROPRIO_DIM,
        ):
            raise ValueError(
                f"RoboCasa latent precompute requires height=352,width=352,action_dim=7,proprio_dim=9; got {cfg.height}x{cfg.width}, action_dim={cfg.action_dim}, proprio_dim={cfg.proprio_dim}."
            )
        dataset_dir = Path(dataset_dirs[0]).expanduser().resolve()
        info = _load_lerobot_info(dataset_dir)
        features = info.get("features")
        if not isinstance(features, dict):
            raise ValueError(
                f"LeRobot metadata features must be a mapping: {dataset_dir / 'meta' / 'info.json'}"
            )
        action_shape = (
            features.get("action", {}).get("shape")
            if isinstance(features.get("action"), dict)
            else None
        )
        state_shape = (
            features.get("observation.state", {}).get("shape")
            if isinstance(features.get("observation.state"), dict)
            else None
        )
        if tuple(action_shape or ()) != (ROBOCASA_ACTION_DIM,) or tuple(state_shape or ()) != (
            ROBOCASA_PROPRIO_DIM,
        ):
            raise ValueError(
                f"RoboCasa metadata must expose action shape [7] and observation.state shape [9] in {dataset_dir / 'meta' / 'info.json'}, got action={action_shape}, state={state_shape}."
            )
    else:
        if len(dataset_dirs) != 1:
            raise ValueError(
                f"RobotWin precompute currently supports exactly one dataset_dir, got {cfg.dataset_dirs}."
            )
        if concat_mode != ROBOTWIN_CONCAT_MULTI_CAMERA:
            raise ValueError(
                f"Latent precompute requires concat_multi_camera='{ROBOTWIN_CONCAT_MULTI_CAMERA}', got {cfg.concat_multi_camera!r}."
            )
    if int(cfg.height) % 16 != 0 or int(cfg.width) % 16 != 0:
        raise ValueError(f"height/width must be multiples of 16, got {cfg.height}x{cfg.width}.")
    if int(cfg.num_frames) <= 1 or int(cfg.num_frames) % 4 != 1:
        raise ValueError(f"num_frames must satisfy >1 and %4==1, got {cfg.num_frames}.")
    if (int(cfg.num_frames) - 1) % int(cfg.action_video_freq_ratio) != 0:
        raise ValueError(
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {cfg.num_frames - 1} and {cfg.action_video_freq_ratio}."
        )
    action_horizon = _resolve_action_horizon(cfg)
    if _is_droid_dataset(cfg) and _optional_int(getattr(cfg, "action_horizon", None)) is None:
        cfg.action_horizon = action_horizon
    video_transitions = (int(cfg.num_frames) - 1) // int(cfg.action_video_freq_ratio)
    if action_horizon % video_transitions != 0:
        raise ValueError(
            f"action_horizon must be divisible by sparse video transitions, got action_horizon={action_horizon} and video_transitions={video_transitions}."
        )
    if int(cfg.action_dim) <= 0 or int(cfg.proprio_dim) <= 0:
        raise ValueError("action_dim and proprio_dim must be positive.")
    vae_checkpoint = _optional_path(cfg.vae_checkpoint)
    if vae_checkpoint is not None and (not Path(vae_checkpoint).expanduser().resolve().exists()):
        raise FileNotFoundError(
            f"VAE checkpoint not found: {Path(vae_checkpoint).expanduser().resolve()}"
        )
    expected_contract = _precompute_cache_contract(cfg)
    configured_latent_dim = getattr(cfg, "video_latent_dim", None)
    if configured_latent_dim is not None and int(configured_latent_dim) != int(
        expected_contract["latent_channels"]
    ):
        raise ValueError(
            f"Precompute video_latent_dim must be {expected_contract['latent_channels']} for {expected_contract['video_model_family']} ({expected_contract['vae_family']} VAE), got {configured_latent_dim}."
        )


def _load_explicit_vae_checkpoint(
    cfg: PrecomputeConfig, *, device: str, dtype: torch.dtype, cache_contract: dict[str, Any]
) -> Any:
    vae_path = Path(str(cfg.vae_checkpoint)).expanduser().resolve()
    family = str(cache_contract["video_model_family"])
    z_dim = int(cache_contract["vae_z_dim"])
    if family == "wan22_i2v_a14b":
        return _load_model_from_state_dict(
            path=str(vae_path),
            model_class=WanVideoVAE,
            model_name="wan21_video_vae_for_i2v_14b",
            torch_dtype=dtype,
            device=device,
            model_kwargs={"z_dim": z_dim},
            state_dict_converter=wan_video_vae_state_dict_converter,
            strict=True,
        )
    if family == "wan22_ti2v_5b":
        return _load_model_from_state_dict(
            path=str(vae_path),
            model_class=WanVideoVAE38,
            model_name="wan22_ti2v_5b_vae",
            torch_dtype=dtype,
            device=device,
            model_kwargs={"z_dim": z_dim},
            state_dict_converter=wan_video_vae_state_dict_converter,
            strict=True,
        )
    raise ValueError(
        f"Unsupported video_model_family={family!r} for explicit VAE checkpoint loading."
    )


def _load_precompute_vae(
    cfg: PrecomputeConfig, *, device: str, dtype: torch.dtype
) -> PrecomputeVAEModel:
    os.environ.setdefault("DREAMFREE_MODEL_BASE_PATH", str(getattr(cfg, "model_base_path")))
    cache_contract = _precompute_cache_contract(cfg)
    explicit_vae_checkpoint = _optional_path(cfg.vae_checkpoint)
    if explicit_vae_checkpoint is not None:
        vae_path = Path(explicit_vae_checkpoint).expanduser().resolve()
        logger.info("Loading explicit VAE checkpoint from %s", vae_path)
        vae = _load_explicit_vae_checkpoint(
            cfg, device=device, dtype=dtype, cache_contract=cache_contract
        )
        vae.requires_grad_(False)
        vae.eval()
        return PrecomputeVAEModel(
            vae=vae,
            device=torch.device(device),
            torch_dtype=dtype,
            vae_path=str(vae_path),
            stats_src=_normalizer_stats_src(cfg),
            norm_mode=_normalizer_mode(cfg),
            cache_contract=cache_contract,
        )
    family = str(cache_contract["video_model_family"])
    if family == "wan22_i2v_a14b":
        loaded = load_wan22_i2v_a14b_vae(
            device=device,
            torch_dtype=dtype,
            model_id=str(cache_contract["model_id"]),
            redirect_common_files=bool(getattr(cfg, "redirect_common_files")),
            pretrained_load_broadcast=bool(getattr(cfg, "pretrained_load_broadcast", True)),
        )
    elif family == "wan22_ti2v_5b":
        loaded = load_wan22_ti2v_5b_vae(
            device=device,
            torch_dtype=dtype,
            model_id=str(cache_contract["model_id"]),
            redirect_common_files=bool(getattr(cfg, "redirect_common_files")),
            pretrained_load_broadcast=bool(getattr(cfg, "pretrained_load_broadcast", True)),
        )
    else:
        raise ValueError(f"Unsupported video_model_family={family!r} for latent precompute.")
    loaded.vae.requires_grad_(False)
    loaded.vae.eval()
    return PrecomputeVAEModel(
        vae=loaded.vae,
        device=torch.device(device),
        torch_dtype=dtype,
        vae_path=str(loaded.vae_path),
        stats_src=_normalizer_stats_src(cfg),
        norm_mode=_normalizer_mode(cfg),
        cache_contract=cache_contract,
    )


def _compile_precompute_vae_encode(encoder: PrecomputeVAEModel, *, mode: str) -> None:
    mode = str(mode).strip()
    if mode not in VAE_COMPILE_MODES:
        raise ValueError(
            f"Unsupported compile_vae_mode={mode!r}; expected one of {VAE_COMPILE_MODES}."
        )
    vae_model = getattr(getattr(encoder, "vae", None), "model", None)
    encode = getattr(vae_model, "encode", None)
    if vae_model is None or not callable(encode):
        raise TypeError(
            f"compile_vae requires encoder.vae.model.encode to be callable; got vae_model={type(vae_model).__name__}, encode={type(encode).__name__}."
        )
    try:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(int(torch._dynamo.config.cache_size_limit), 64)
        torch._dynamo.config.suppress_errors = False
        compiled_encode = torch.compile(encode, dynamic=False, mode=mode)
        setattr(vae_model, "encode", compiled_encode)
        encoder.compile_enabled = True
    except Exception as exc:
        raise RuntimeError(
            f"Failed to enable torch.compile for VAE encode with mode={mode!r}."
        ) from exc
    logger.info(
        "Enabled torch.compile for %s.encode with mode=%s dynamic=False.",
        type(vae_model).__name__,
        mode,
    )


def _prepare_episode_payload(
    *,
    cfg: PrecomputeConfig,
    normalizer: LinearNormalizer | None,
    dataset_root: Path,
    meta: PrecomputeDatasetMetadata,
    actual_ep_idx: int,
    episode_rows: dict[int, dict[str, Any]] | None = None,
) -> EpisodePayload:
    episode_row: dict[str, Any] | None = None
    if _is_lerobot_v30_info(meta.info):
        if episode_rows is None:
            raise ValueError(f"LeRobot v3.0 dataset requires loaded episode rows: {dataset_root}")
        episode_row = episode_rows.get(int(actual_ep_idx))
        if episode_row is None:
            raise KeyError(
                f"LeRobot v3.0 metadata is missing episode_index={actual_ep_idx}: {dataset_root}"
            )
    data_path = _resolve_lerobot_data_path(dataset_root, meta, actual_ep_idx, episode_row)
    parquet = (
        _read_episode_parquet_slice(data_path, episode_index=actual_ep_idx, episode_row=episode_row)
        if episode_row is not None
        else _read_episode_parquet(data_path)
    )
    T_ep_parquet = int(parquet["action"].shape[0])
    if T_ep_parquet <= 0:
        raise ValueError(f"Episode {actual_ep_idx} has no parquet rows in {data_path}")
    frame_counts: list[int] = []
    cam_frames: list[torch.Tensor] = []
    camera_keys = _camera_keys_for_features(
        cfg, meta.features, source=meta.root / "meta" / "info.json"
    )
    for cam_key in camera_keys:
        video_path = _resolve_lerobot_video_path(
            dataset_root, meta, actual_ep_idx, cam_key, episode_row
        )
        if episode_row is None:
            frames = _decode_all_frames(video_path)
        else:
            start_key = f"videos/{cam_key}/from_timestamp"
            if start_key not in episode_row or episode_row[start_key] is None:
                raise KeyError(
                    f"LeRobot v3.0 episode {actual_ep_idx} is missing {start_key!r}: {dataset_root}"
                )
            seconds = parquet["timestamps"].to(torch.float64) + float(episode_row[start_key])
            frames = _decode_frames_played_at(video_path, seconds)
        cam_frames.append(frames)
        frame_counts.append(int(frames.shape[0]))
    T_ep_video = min(frame_counts)
    cam_frames = [f[:T_ep_video] for f in cam_frames]
    video_raw = torch.stack(cam_frames, dim=0)
    video_processed = _apply_multicam(video_raw, cfg)
    T_ep = min(T_ep_video, T_ep_parquet)
    video_processed = video_processed[:T_ep]
    action_raw = parquet["action"][:T_ep]
    state_raw = parquet["state"][:T_ep]
    task_indices = parquet["task_indices"][:T_ep]
    expected_raw_action_dim = int(cfg.action_dim)
    expected_raw_state_dim = int(cfg.proprio_dim)
    if int(action_raw.shape[-1]) != expected_raw_action_dim:
        raise ValueError(
            f"Episode {actual_ep_idx} action dim mismatch for dataset={dataset_root.name}: expected raw dim {expected_raw_action_dim}, got {action_raw.shape[-1]}."
        )
    if int(state_raw.shape[-1]) != expected_raw_state_dim:
        raise ValueError(
            f"Episode {actual_ep_idx} state/proprio dim mismatch for dataset={dataset_root.name}: expected raw dim {expected_raw_state_dim}, got {state_raw.shape[-1]}."
        )
    if normalizer is None:
        raise ValueError(
            f"Dataset {cfg.dataset_name!r} requires a normalizer but none was provided."
        )
    (action_norm, state_norm) = _normalize_episode(
        normalizer, action_raw, state_raw, cfg=cfg, dataset_root=dataset_root
    )
    extra_tensors = (
        _droid_cache_extra_tensors(
            dataset_root=dataset_root, action_raw=action_raw, state_raw=state_raw
        )
        if _is_droid_dataset(cfg)
        else None
    )
    dataset_kind = (
        "droid" if _is_droid_dataset(cfg) else "robocasa" if _is_robocasa_dataset(cfg) else None
    )
    action_layout = None
    return EpisodePayload(
        episode_index=int(actual_ep_idx),
        num_frames=int(T_ep),
        video_processed=video_processed,
        action_norm=action_norm,
        state_norm=state_norm,
        task_indices=task_indices,
        concat_layout=str(cfg.concat_multi_camera).strip().lower(),
        norm_mode=_normalizer_mode(cfg),
        dataset_kind=dataset_kind,
        action_layout=action_layout,
        extra_tensors=extra_tensors,
        source_metadata=None,
    )


def _save_episode_cache(
    *,
    output_path: Path,
    latents: torch.Tensor,
    i2v_condition_latents: torch.Tensor | None,
    payload: EpisodePayload,
    encoder: PrecomputeVAEModel,
    num_frames: int,
    action_video_freq_ratio: int,
) -> dict[str, Any]:
    validate_latent_tensor_contract(
        latents, expected=encoder.cache_contract, label="Latent precompute"
    )
    (n_samples, c_lat, t_lat, h_lat, w_lat) = latents.shape
    full_window_count = max(0, payload.num_frames - num_frames + 1)
    contract = encoder.cache_contract
    is_14b = str(contract["video_model_family"]) == "wan22_i2v_a14b"
    if is_14b:
        if not isinstance(i2v_condition_latents, torch.Tensor):
            raise ValueError(
                f"Wan2.2-I2V-A14B latent caches require official {WAN22_A14B_I2V_CONDITION_LATENTS_KEY}."
            )
        if tuple(i2v_condition_latents.shape) != tuple(latents.shape):
            raise ValueError(
                f"{WAN22_A14B_I2V_CONDITION_LATENTS_KEY} shape must match latents: {tuple(i2v_condition_latents.shape)} vs {tuple(latents.shape)}."
            )
        if bool(contract.get("require_i2v_first_latent_equality", False)):
            validate_14b_i2v_first_latent_consistency(
                latents, i2v_condition_latents, label="Latent precompute save"
            )
    elif i2v_condition_latents is not None:
        raise ValueError(
            f"{WAN22_A14B_I2V_CONDITION_LATENTS_KEY} is only valid for video_model_family='wan22_i2v_a14b'."
        )
    metadata = {
        "cache_version": str(LATENT_CACHE_VERSION),
        "coverage": "full_episode",
        "padding_policy": "clamp_to_episode_end",
        "video_model_family": str(contract["video_model_family"]),
        "model_id": str(contract["model_id"]),
        "vae_family": str(contract["vae_family"]),
        "vae_path": str(encoder.vae_path),
        "vae_z_dim": str(int(contract["vae_z_dim"])),
        "vae_temporal_stride": str(int(contract["vae_temporal_stride"])),
        "vae_spatial_stride": str(int(contract["vae_spatial_stride"])),
        "episode_index": str(payload.episode_index),
        "num_frames": str(payload.num_frames),
        "num_samples": str(int(n_samples)),
        "num_windows": str(int(full_window_count)),
        "num_frames_per_window": str(num_frames),
        "action_video_freq_ratio": str(action_video_freq_ratio),
        "latent_channels": str(int(c_lat)),
        "latent_t_per_window": str(int(t_lat)),
        "latent_h": str(int(h_lat)),
        "latent_w": str(int(w_lat)),
        "video_h": str(int(contract["height"])),
        "video_w": str(int(contract["width"])),
        "concat_layout": payload.concat_layout,
        "action_dim": str(int(payload.action_norm.shape[1])),
        "state_dim": str(int(payload.state_norm.shape[1])),
        "proprio_dim": str(int(payload.state_norm.shape[1])),
        "norm_mode": payload.norm_mode,
        "normalizer_stats_path": str(encoder.stats_src),
    }
    if is_14b:
        metadata["i2v_condition_source"] = WAN22_A14B_I2V_CONDITION_SOURCE_OFFICIAL
    if contract.get("visual_preprocess_id") is not None:
        metadata["visual_preprocess_id"] = str(contract["visual_preprocess_id"])
    if contract.get("action_horizon") is not None:
        metadata["action_horizon"] = str(int(contract["action_horizon"]))
    if payload.dataset_kind is not None:
        metadata["dataset_kind"] = str(payload.dataset_kind)
    if payload.action_layout is not None:
        metadata["action_layout"] = payload.action_layout
    if payload.source_metadata:
        metadata.update({str(key): str(value) for (key, value) in payload.source_metadata.items()})
    if payload.concat_layout == DROID_GRID_CONCAT_MULTI_CAMERA:
        metadata["dataset_kind"] = "droid"
        metadata["action_contract"] = DROID_DREAMZERO_ACTION_CONTRACT
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tensors = {
        "latents": latents,
        "action": payload.action_norm.to(torch.float32).contiguous(),
        "proprio": payload.state_norm.to(torch.float32).contiguous(),
        "task_indices": payload.task_indices.to(torch.int32).contiguous(),
    }
    if i2v_condition_latents is not None:
        tensors[WAN22_A14B_I2V_CONDITION_LATENTS_KEY] = i2v_condition_latents.contiguous()
    if payload.extra_tensors:
        tensors.update(
            {
                key: value.to(torch.float32).contiguous()
                for (key, value) in payload.extra_tensors.items()
            }
        )
    save_safetensors(tensors, str(tmp_path), metadata=metadata)
    tmp_path.replace(output_path)
    summary = {
        "episode_index": int(payload.episode_index),
        "num_frames": int(payload.num_frames),
        "num_samples": int(n_samples),
        "num_windows": int(full_window_count),
        "bytes": int(
            latents.numel() * latents.element_size()
            + (
                0
                if i2v_condition_latents is None
                else i2v_condition_latents.numel() * i2v_condition_latents.element_size()
            )
        ),
    }
    if payload.source_metadata:
        summary.update(payload.source_metadata)
    return summary


def _encode_and_save_episode(
    *,
    encoder: PrecomputeVAEModel,
    payload: EpisodePayload,
    output_path: Path,
    num_frames: int,
    action_video_freq_ratio: int,
    batch_size: int,
) -> dict[str, Any]:
    is_14b = str(encoder.cache_contract["video_model_family"]) == "wan22_i2v_a14b"
    latents = _encode_windows(
        encoder,
        payload.video_processed,
        num_frames=num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
        batch_size=batch_size,
    )
    i2v_condition_latents = (
        _encode_14b_i2v_condition_windows(
            encoder,
            payload.video_processed,
            num_frames=num_frames,
            action_video_freq_ratio=action_video_freq_ratio,
            batch_size=batch_size,
        )
        if is_14b
        else None
    )
    return _save_episode_cache(
        output_path=output_path,
        latents=latents,
        i2v_condition_latents=i2v_condition_latents,
        payload=payload,
        encoder=encoder,
        num_frames=num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
    )


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)


def _resolve_runtime(extra: argparse.Namespace) -> PrecomputeRuntime:
    env_world_size = _env_int("WORLD_SIZE")
    env_rank = _env_int("RANK")
    env_local_rank = _env_int("LOCAL_RANK")
    has_dist_env = any((value is not None for value in (env_world_size, env_rank, env_local_rank)))
    if has_dist_env:
        world_size = int(env_world_size or 1)
        rank = int(env_rank or 0)
        local_rank = int(env_local_rank or 0)
        if (
            bool(getattr(extra, "world_size_explicit", False))
            and int(extra.world_size) != world_size
        ):
            raise ValueError(
                f"Conflicting world_size: env WORLD_SIZE={world_size} but CLI requested {extra.world_size}."
            )
        if bool(getattr(extra, "rank_explicit", False)) and int(extra.rank) != rank:
            raise ValueError(f"Conflicting rank: env RANK={rank} but CLI requested {extra.rank}.")
        if getattr(extra, "local_rank", None) is not None and bool(
            getattr(extra, "local_rank_explicit", False)
        ):
            if int(extra.local_rank) != local_rank:
                raise ValueError(
                    f"Conflicting local_rank: env LOCAL_RANK={local_rank} but CLI requested {extra.local_rank}."
                )
        is_distributed = world_size > 1
    else:
        world_size = int(extra.world_size)
        rank = int(extra.rank)
        local_rank = 0 if getattr(extra, "local_rank", None) is None else int(extra.local_rank)
        is_distributed = False
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank} for world_size {world_size}.")
    if local_rank < 0:
        raise ValueError(f"local_rank must be >= 0, got {local_rank}.")
    device = "cpu"
    if torch.cuda.is_available():
        device_count = int(torch.cuda.device_count())
        if local_rank >= device_count:
            raise ValueError(
                f"local_rank {local_rank} is out of range for visible CUDA device_count={device_count}."
            )
        device = f"cuda:{local_rank}"
    return PrecomputeRuntime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=is_distributed,
        device=device,
    )


def _init_distributed_runtime(runtime: PrecomputeRuntime) -> None:
    if runtime.device.startswith("cuda"):
        torch.cuda.set_device(runtime.local_rank)
    if not runtime.is_distributed:
        return
    backend = "nccl" if runtime.device.startswith("cuda") else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")


def _build_episode_shard(all_indices: Sequence[int], *, world_size: int, rank: int) -> list[int]:
    return [int(ep) for (i, ep) in enumerate(all_indices) if i % int(world_size) == int(rank)]


def _aggregate_counts(
    runtime: PrecomputeRuntime, *, processed: int, skipped: int, total: int
) -> dict[str, int]:
    counts = {"processed": int(processed), "skipped": int(skipped), "total": int(total)}
    if not runtime.is_distributed or not dist.is_initialized():
        return counts
    reduce_device = (
        torch.device(runtime.device) if runtime.device.startswith("cuda") else torch.device("cpu")
    )
    tensor = torch.tensor(
        [counts["processed"], counts["skipped"], counts["total"]],
        device=reduce_device,
        dtype=torch.long,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {
        "processed": int(tensor[0].item()),
        "skipped": int(tensor[1].item()),
        "total": int(tensor[2].item()),
    }


def _shutdown_distributed_runtime(runtime: PrecomputeRuntime) -> None:
    if not runtime.is_distributed or not dist.is_initialized():
        return
    dist.barrier()
    dist.destroy_process_group()


def _run_episode_shard(
    *,
    cfg: PrecomputeConfig,
    shard: Sequence[int],
    encoder: PrecomputeVAEModel,
    normalizer: LinearNormalizer | None,
    dataset_root: Path,
    meta: PrecomputeDatasetMetadata,
    episode_rows: dict[int, dict[str, Any]] | None,
    out_dir: Path,
    manifest_path: Path,
    num_frames: int,
    action_video_freq_ratio: int,
    batch_size: int,
    skip_existing: bool,
    rank: int,
) -> tuple[int, int]:
    processed = 0
    skipped = 0
    start_time = time.perf_counter()
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for ep_idx in shard:
            out_path = out_dir / f"episode_{int(ep_idx):06d}.safetensors"
            if skip_existing and out_path.exists():
                validate_latent_cache_file(
                    out_path,
                    expected=encoder.cache_contract,
                    label="Latent precompute skip_existing",
                )
                skipped += 1
                continue
            payload = _prepare_episode_payload(
                cfg=cfg,
                normalizer=normalizer,
                dataset_root=dataset_root,
                meta=meta,
                actual_ep_idx=ep_idx,
                episode_rows=episode_rows,
            )
            summary = _encode_and_save_episode(
                encoder=encoder,
                payload=payload,
                output_path=out_path,
                num_frames=num_frames,
                action_video_freq_ratio=action_video_freq_ratio,
                batch_size=batch_size,
            )
            manifest.write(json.dumps(summary) + "\n")
            manifest.flush()
            processed += 1
            if processed % 10 == 0 or processed + skipped == len(shard):
                logger.info(
                    "[rank %d] processed=%d skipped=%d remaining=%d elapsed=%.1fs",
                    rank,
                    processed,
                    skipped,
                    len(shard) - processed - skipped,
                    time.perf_counter() - start_time,
                )
    return (processed, skipped)


def run_precompute(cfg: PrecomputeConfig, extra: argparse.Namespace) -> None:
    runtime = _resolve_runtime(extra)
    _init_distributed_runtime(runtime)
    setup_logging(log_level=logging.INFO, is_main_process=runtime.rank == 0)
    validate_precompute_config(cfg)
    if extra.cache_root is None and extra.cache_dir is None:
        raise ValueError("Either --cache-root or --cache-dir is required.")
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_dtype(mixed_precision)
    cache_contract = _precompute_cache_contract(cfg)
    try:
        logger.info(
            "Loading %s VAE for %s on %s (dtype=%s)",
            cache_contract["vae_family"],
            cache_contract["video_model_family"],
            runtime.device,
            model_dtype,
        )
        encoder = _load_precompute_vae(cfg, device=runtime.device, dtype=model_dtype)
        if bool(getattr(extra, "compile_vae", False)):
            _compile_precompute_vae_encode(
                encoder, mode=str(getattr(extra, "compile_vae_mode", "default"))
            )
        normalizer = _build_normalizer(cfg)
        cache_base = Path(extra.cache_dir or extra.cache_root).expanduser().resolve()
        exact_cache_dir = extra.cache_dir is not None
        ensure_dir(cache_base)
        dataset_dirs = [
            Path(path).expanduser().resolve() for path in _parse_dataset_dirs(cfg.dataset_dirs)
        ]
        for dataset_dir in dataset_dirs:
            meta = _load_precompute_dataset_metadata(dataset_dir)
            episode_rows = (
                _load_lerobot_episode_rows(dataset_dir) if _is_lerobot_v30_info(meta.info) else None
            )
            total_episodes = (
                len(episode_rows) if episode_rows is not None else int(meta.total_episodes)
            )
            all_indices = (
                list(episode_rows) if episode_rows is not None else list(range(total_episodes))
            )
            if int(extra.limit_episodes) > 0:
                all_indices = (
                    list(episode_rows) if episode_rows is not None else list(range(total_episodes))
                )
            shard = _build_episode_shard(
                all_indices, world_size=runtime.world_size, rank=runtime.rank
            )
            logger.info(
                "Rank %d/%d (local_rank=%d distributed=%s) will process %d episodes (of %d candidates in dataset=%s)",
                runtime.rank,
                runtime.world_size,
                runtime.local_rank,
                runtime.is_distributed,
                len(shard),
                len(all_indices),
                dataset_dir.name,
            )
            out_dir = _cache_output_dir(
                cfg,
                cache_base,
                dataset_dir,
                exact_cache_dir=exact_cache_dir,
                dataset_dirs=dataset_dirs,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Writing per-episode latents under %s", out_dir)
            manifest_path = out_dir / f"manifest_rank{int(runtime.rank):02d}.jsonl"
            start_time = time.perf_counter()
            (processed, skipped) = _run_episode_shard(
                cfg=cfg,
                shard=shard,
                encoder=encoder,
                normalizer=normalizer,
                dataset_root=dataset_dir,
                meta=meta,
                episode_rows=episode_rows,
                out_dir=out_dir,
                manifest_path=manifest_path,
                num_frames=int(cfg.num_frames),
                action_video_freq_ratio=int(cfg.action_video_freq_ratio),
                batch_size=int(extra.encode_batch_size),
                skip_existing=bool(extra.skip_existing),
                rank=runtime.rank,
            )
            elapsed = time.perf_counter() - start_time
            logger.info(
                "[rank %d] done dataset=%s: processed=%d skipped=%d total=%d in %.1fs",
                runtime.rank,
                dataset_dir.name,
                processed,
                skipped,
                len(shard),
                elapsed,
            )
            global_counts = _aggregate_counts(
                runtime, processed=processed, skipped=skipped, total=len(shard)
            )
            if runtime.rank == 0 and runtime.is_distributed:
                logger.info(
                    "[global] done dataset=%s: processed=%d skipped=%d total=%d",
                    dataset_dir.name,
                    global_counts["processed"],
                    global_counts["skipped"],
                    global_counts["total"],
                )
    finally:
        _shutdown_distributed_runtime(runtime)


def _read_cache_header(path: Path) -> dict[str, Any]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {
            "metadata": dict(handle.metadata() or {}),
            "latents_shape": tuple(handle.get_slice("latents").get_shape()),
            "tensor_shapes": {
                key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()
            },
        }
