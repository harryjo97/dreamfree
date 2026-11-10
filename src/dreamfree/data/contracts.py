from __future__ import annotations

from typing import Any

from dreamfree.data import yam
from dreamfree.data.lerobot.cached_latent_dataset import stage2_latent_cache_expected_contract
from dreamfree.training.stage0.config import Stage0Config
from dreamfree.training.stage0.constants import WAN22_A14B_LATENT_FRAME0_MODES


def _normalize_latent_frame0_mode(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in WAN22_A14B_LATENT_FRAME0_MODES:
        raise ValueError(
            f"Unsupported latent_frame0_mode={value!r}. Expected one of: {sorted(WAN22_A14B_LATENT_FRAME0_MODES)}."
        )
    return normalized


def _stage0_latent_cache_expected_contract(cfg: Stage0Config) -> dict[str, Any]:
    contract = stage2_latent_cache_expected_contract(
        video_model_family=str(getattr(cfg, "video_model_family", "wan22_ti2v_5b")),
        model_id=getattr(cfg, "model_id", None),
        height=int(getattr(cfg, "height")),
        width=int(getattr(cfg, "width")),
        num_frames=_stage0_raw_video_frames(cfg),
        action_video_freq_ratio=int(getattr(cfg, "raw_frame_stride")),
    )
    if str(cfg.camera_layout).startswith("robocasa_dreamzero_"):
        contract["concat_layout"] = str(getattr(cfg, "camera_layout"))
        if contract["video_model_family"] == "wan22_i2v_a14b":
            contract["require_i2v_first_latent_equality"] = True
    if cfg.dataset_name in yam.DATASET_NAMES:
        contract["abc_contract"] = yam.PRESETS[cfg.dataset_name].contract
    return contract


def _stage0_raw_video_frames(cfg: Stage0Config) -> int:
    raw_frame_stride = int(getattr(cfg, "raw_frame_stride"))
    clip_frames = int(getattr(cfg, "clip_frames"))
    if raw_frame_stride <= 0:
        raise ValueError(f"raw_frame_stride must be positive, got {raw_frame_stride}.")
    if clip_frames <= 1:
        raise ValueError(
            f"clip_frames must be greater than 1 for cached latents, got {clip_frames}."
        )
    return (clip_frames - 1) * raw_frame_stride + 1
