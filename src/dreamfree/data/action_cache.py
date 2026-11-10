"""Stage 2 text/video cache contracts and strict payload validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from dreamfree.data import yam
from dreamfree.data.action_processing import (
    DROID_DREAMZERO_ACTION_CONTRACT,
    resolve_stage2_action_horizon,
)
from dreamfree.data.lerobot.cached_latent_dataset import (
    DROID_DREAMZERO_OVERLAY_CACHE_KIND,
    DROID_DREAMZERO_REQUIRED_TENSOR_DIMS,
    is_droid_dreamzero_runtime_horizon_cache,
    stage2_latent_cache_expected_contract,
    stage2_text_cache_suffix_for_family,
    validate_droid_dreamzero_overlay_cache_file,
    validate_stage2_latent_cache_file,
    validate_stage2_latent_tensor_contract,
)
from dreamfree.data.text_cache import validate_text_cache
from dreamfree.training.stage0.constants import (
    DROID_GRID_CONCAT_MULTI_CAMERA,
)

STAGE2_REQUIRED_CACHE_KEYS = ("context", "mask")


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
    if str(cfg.camera_layout).startswith("robocasa_dreamzero_"):
        contract["concat_layout"] = str(getattr(cfg, "camera_layout"))
        if contract["video_model_family"] == "wan22_i2v_a14b":
            contract["require_i2v_first_latent_equality"] = True
    if cfg.dataset_name in yam.DATASET_NAMES:
        contract["abc_contract"] = yam.PRESETS[cfg.dataset_name].contract
    return contract


def _validate_stage2_latent_cache_contract(path: Path, cfg: Any) -> None:
    if cfg.dataset_name in yam.DATASET_NAMES:
        from dreamfree.data.abc_dataset import validate_episode_cache

        validate_episode_cache(
            path,
            Path(cfg.dataset_dirs),
            cfg.dataset_name,
            _stage2_latent_cache_expected_contract(cfg),
        )
        return
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
