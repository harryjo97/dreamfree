"""RoboCasa visual preprocessing contracts shared by caching and evaluation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

ROBOCASA_DREAMZERO_352_LAYOUT = "robocasa_dreamzero_352"
ROBOCASA_DREAMZERO_448_LAYOUT = "robocasa_dreamzero_448"
ROBOCASA_DREAMZERO_352_SIZE = (352, 352)
ROBOCASA_DREAMZERO_448_SIZE = (448, 448)
ROBOCASA_RAW_CAMERA_SIZE = (224, 224)
ROBOCASA_DIRECT_352_DATASET_NAMES = frozenset({"robocasa"})
ROBOCASA_DIRECT_448_DATASET_NAMES = frozenset(
    {"robocasa_success_448", "robocasa_cosmos_success_448"}
)


def compose_robocasa_dreamzero(
    video_raw: torch.Tensor,
    *,
    output_size: tuple[int, int],
    expected_raw_size: tuple[int, int],
    normalize: bool = True,
) -> torch.Tensor:
    """Resize float pixels into a wrist-over-agent mosaic, normalized to [-1, 1]."""
    if not isinstance(video_raw, torch.Tensor):
        raise TypeError(
            f"RoboCasa direct-bilinear preprocessing requires a torch.Tensor, got {type(video_raw)}."
        )
    if video_raw.dtype != torch.uint8 and not torch.is_floating_point(video_raw):
        raise TypeError(
            f"RoboCasa preprocessing requires uint8 pixels or floating-point pixels in [0, 1], got {video_raw.dtype}."
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
    if video_raw.dtype == torch.uint8:
        video_raw = video_raw.to(torch.float32) / 255.0
    else:
        video_raw = video_raw.to(torch.float32)
        if not torch.isfinite(video_raw).all() or video_raw.min() < 0 or video_raw.max() > 1:
            raise ValueError("RoboCasa floating-point source pixels must be finite and in [0, 1].")
    resize_kwargs = {"mode": "bilinear", "align_corners": False, "antialias": True}
    left = F.interpolate(video_raw[0], size=[tile_h, tile_w], **resize_kwargs)
    right = F.interpolate(video_raw[1], size=[tile_h, tile_w], **resize_kwargs)
    wrist = F.interpolate(video_raw[2], size=[tile_h, target[1]], **resize_kwargs)
    video = torch.cat([wrist, torch.cat([left, right], dim=-1)], dim=-2)
    if video.dtype != torch.float32 or tuple(video.shape[-3:]) != (3, *target):
        raise RuntimeError(
            f"RoboCasa direct-bilinear preprocessing produced an invalid tensor: shape={tuple(video.shape)} dtype={video.dtype}."
        )
    return (video * 2.0 - 1.0 if normalize else video).contiguous()


def expected_robocasa_cache_raw_camera_size(dataset_name: Any) -> tuple[int, int]:
    return ROBOCASA_RAW_CAMERA_SIZE
