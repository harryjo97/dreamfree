"""RoboCasa visual preprocessing contracts shared by caching and evaluation."""

from __future__ import annotations

from typing import Any

import torch
import torchvision.transforms.functional as transforms_F

ROBOCASA_DREAMZERO_352_DIRECT_BILINEAR_V1 = "robocasa_dreamzero_352_direct_bilinear_v1"
ROBOCASA_DREAMZERO_448_DIRECT_BILINEAR_V1 = "robocasa_dreamzero_448_direct_bilinear_v1"
ROBOCASA_DREAMZERO_352_LAYOUT = "robocasa_dreamzero_352"
ROBOCASA_DREAMZERO_448_LAYOUT = "robocasa_dreamzero_448"
ROBOCASA_DREAMZERO_352_SIZE = (352, 352)
ROBOCASA_DREAMZERO_448_SIZE = (448, 448)
ROBOCASA_RAW_CAMERA_SIZE = (224, 224)
ROBOCASA_DIRECT_352_DATASET_NAMES = frozenset({"robocasa"})
ROBOCASA_DIRECT_448_DATASET_NAMES = frozenset(
    {"robocasa_success_448", "robocasa_cosmos_success_448"}
)


def normalize_robocasa_dataset_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def robocasa_visual_preprocess_id(
    *, dataset_name: Any, concat_multi_camera: Any, height: Any, width: Any
) -> str | None:
    name = normalize_robocasa_dataset_name(dataset_name)
    layout = str(concat_multi_camera).strip().lower()
    size = (int(height), int(width))
    if (
        name in ROBOCASA_DIRECT_352_DATASET_NAMES
        and layout == ROBOCASA_DREAMZERO_352_LAYOUT
        and (size == ROBOCASA_DREAMZERO_352_SIZE)
    ):
        return ROBOCASA_DREAMZERO_352_DIRECT_BILINEAR_V1
    if (
        name in ROBOCASA_DIRECT_448_DATASET_NAMES
        and layout == ROBOCASA_DREAMZERO_448_LAYOUT
        and (size == ROBOCASA_DREAMZERO_448_SIZE)
    ):
        return ROBOCASA_DREAMZERO_448_DIRECT_BILINEAR_V1
    return None


def required_robocasa_visual_preprocess_id(
    *, dataset_name: Any, concat_multi_camera: Any, height: Any, width: Any, video_model_family: Any
) -> str | None:
    family = str(video_model_family).strip().lower()
    if family not in {"wan22_ti2v_5b", "wan22_i2v_a14b"}:
        return None
    return robocasa_visual_preprocess_id(
        dataset_name=dataset_name,
        concat_multi_camera=concat_multi_camera,
        height=height,
        width=width,
    )


def compose_robocasa_dreamzero_uint8(
    video_raw: torch.Tensor, *, output_size: tuple[int, int], expected_raw_size: tuple[int, int]
) -> torch.Tensor:
    """Compose three square views into a direct-resized DreamZero mosaic."""
    if not isinstance(video_raw, torch.Tensor):
        raise TypeError(
            f"RoboCasa direct-bilinear preprocessing requires a torch.Tensor, got {type(video_raw)}."
        )
    if video_raw.dtype != torch.uint8:
        raise TypeError(
            f"RoboCasa direct-bilinear preprocessing requires uint8 source pixels, got {video_raw.dtype}."
        )
    if video_raw.ndim != 5 or tuple(video_raw.shape[:1]) != (3,) or int(video_raw.shape[2]) != 3:
        raise ValueError(
            f"RoboCasa direct-bilinear preprocessing requires [3,T,3,H,W], got {tuple(video_raw.shape)}."
        )
    raw_size = tuple((int(v) for v in video_raw.shape[-2:]))
    if raw_size != tuple((int(v) for v in expected_raw_size)):
        raise ValueError(
            f"RoboCasa direct-bilinear preprocessing received an unexpected raw view size: actual={raw_size} expected={tuple(expected_raw_size)}."
        )
    target = tuple((int(v) for v in output_size))
    if target not in {ROBOCASA_DREAMZERO_352_SIZE, ROBOCASA_DREAMZERO_448_SIZE}:
        raise ValueError(
            f"RoboCasa direct-bilinear preprocessing supports output sizes {ROBOCASA_DREAMZERO_352_SIZE} and {ROBOCASA_DREAMZERO_448_SIZE}, got {target}."
        )
    (tile_h, tile_w) = (target[0] // 2, target[1] // 2)
    resize_kwargs = {"interpolation": transforms_F.InterpolationMode.BILINEAR, "antialias": True}
    left = transforms_F.resize(video_raw[0], size=[tile_h, tile_w], **resize_kwargs)
    right = transforms_F.resize(video_raw[1], size=[tile_h, tile_w], **resize_kwargs)
    wrist = transforms_F.resize(video_raw[2], size=[tile_h, target[1]], **resize_kwargs)
    video = torch.cat([wrist, torch.cat([left, right], dim=-1)], dim=-2)
    if video.dtype != torch.uint8 or tuple(video.shape[-3:]) != (3, *target):
        raise RuntimeError(
            f"RoboCasa direct-bilinear preprocessing produced an invalid tensor: shape={tuple(video.shape)} dtype={video.dtype}."
        )
    return video.contiguous()


def normalize_uint8_video(video: torch.Tensor) -> torch.Tensor:
    if video.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 video before normalization, got {video.dtype}.")
    return (video.to(dtype=torch.float32) / 255.0 * 2.0 - 1.0).contiguous()


def expected_robocasa_cache_raw_camera_size(dataset_name: Any) -> tuple[int, int]:
    return ROBOCASA_RAW_CAMERA_SIZE
