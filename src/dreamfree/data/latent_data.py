"""Read episode metadata, compose camera views, and normalize cache targets."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as transforms_F

from dreamfree.data import yam
from dreamfree.data.latent_config import (
    DROID_CAMERA_KEYS,
    DROID_DREAMZERO_RELATIVE_STATS_FILE,
    DROID_GRID_CONCAT_MULTI_CAMERA,
    DROID_GRIPPER_POSITION_KEY,
    DROID_JOINT_POSITION_KEY,
    HORIZONTAL_CONCAT_MULTI_CAMERA,
    LIBERO_CAMERA_KEYS,
    ROBOCASA_ACTION_DIM,
    ROBOCASA_CAMERA_KEYS,
    ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA,
    ROBOCASA_PROPRIO_DIM,
    ROBOTWIN_CAMERA_KEYS,
    ROBOTWIN_CONCAT_MULTI_CAMERA,
    ROBOTWIN_DATASET_NAMES,
    PrecomputeConfig,
    _dataset_name,
    _is_droid_dataset,
    _is_libero_dataset,
    _is_robocasa_dataset,
    _normalizer_mode,
    _optional_path,
    _parse_dataset_dirs,
    _uses_lerobot_metadata_stats,
)
from dreamfree.data.lerobot.lerobot.datasets.compute_stats import aggregate_stats
from dreamfree.data.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata
from dreamfree.data.lerobot.utils.normalizer import LinearNormalizer, load_dataset_stats_from_json
from dreamfree.data.robocasa_visual import (
    compose_robocasa_dreamzero,
    expected_robocasa_cache_raw_camera_size,
)


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


def _camera_key_candidates_for_dataset(cfg: PrecomputeConfig) -> tuple[tuple[str, ...], ...]:
    dataset_name = _dataset_name(cfg)
    if dataset_name in yam.DATASET_NAMES:
        return (yam.CAMERA_KEYS,)
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
    return compose_robocasa_dreamzero(
        video_raw,
        output_size=(int(height), int(width)),
        expected_raw_size=expected_robocasa_cache_raw_camera_size(dataset_name),
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
    if cfg.dataset_name in yam.DATASET_NAMES:
        views = [view.float() / 255 if view.dtype == torch.uint8 else view for view in video_raw]
        return yam.pack_views(views, cfg.dataset_name) * 2 - 1
    concat_mode = str(cfg.camera_layout).strip().lower()
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
    raise ValueError(f"Unsupported camera_layout={cfg.camera_layout!r}.")


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
    if cfg.dataset_name in yam.DATASET_NAMES:
        return yam.shape_meta(cfg.dataset_name)
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
    stats_path = _optional_path(getattr(cfg, "dataset_stats_path", None))
    resolved_stats_path: Path | None = None
    if stats_path is not None:
        resolved_stats_path = Path(stats_path).expanduser().resolve()
        stats = load_dataset_stats_from_json(str(resolved_stats_path))
    elif _uses_lerobot_metadata_stats(cfg):
        stats = _load_metadata_stats(_parse_dataset_dirs(cfg.dataset_dirs), cfg=cfg)
    else:
        raise ValueError(
            "Precompute requires dataset_stats_path unless the dataset uses LeRobot metadata stats."
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
    if _is_droid_dataset(cfg):
        # DROID stores all control representations; normalization selects joints and gripper.
        expected_raw_action_dim = int(meta.features["action"]["shape"][0])
        expected_raw_state_dim = int(meta.features["observation.state"]["shape"][0])
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
    if action_norm.shape[-1] != cfg.action_dim or state_norm.shape[-1] != cfg.proprio_dim:
        raise ValueError(
            f"Episode {actual_ep_idx} normalized action/state dimensions must be "
            f"{cfg.action_dim}/{cfg.proprio_dim}, got "
            f"{action_norm.shape[-1]}/{state_norm.shape[-1]}."
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
        concat_layout=str(cfg.camera_layout).strip().lower(),
        norm_mode=_normalizer_mode(cfg),
        dataset_kind=dataset_kind,
        action_layout=action_layout,
        extra_tensors=extra_tensors,
        source_metadata=yam.episode_metadata(dataset_root, actual_ep_idx, cfg.dataset_name)
        if cfg.dataset_name in yam.DATASET_NAMES
        else None,
    )
