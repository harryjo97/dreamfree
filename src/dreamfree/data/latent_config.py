"""Latent-cache configuration, identity, and strict artifact contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open

from dreamfree.data import yam
from dreamfree.data.lerobot.cached_latent_dataset import (
    stage2_latent_cache_expected_contract as latent_cache_expected_contract,
)
from dreamfree.models.wan22.i2v_condition import (
    WAN22_A14B_I2V_CONDITION_LATENTS_KEY,
    WAN22_A14B_I2V_CONDITION_SOURCE_OFFICIAL,
)

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
    dataset_stats_path: str = "./assets"
    num_frames: int = 33
    action_video_freq_ratio: int = 4
    action_horizon: int | None = None
    height: int = 384
    width: int = 320
    action_dim: int = 14
    proprio_dim: int = 14
    camera_layout: str = ROBOTWIN_CONCAT_MULTI_CAMERA
    model_id: str | None = None
    video_model_family: str | None = None
    vae_checkpoint: str | None = None
    video_latent_dim: int | None = None
    redirect_common_files: bool = True
    pretrained_load_broadcast: bool = True
    mixed_precision: str = "bf16"


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
    if int(cache_version) not in yam.cache_versions(expected, meta):
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
    if cfg.dataset_name in yam.DATASET_NAMES:
        return "q01/q99"
    if _is_droid_dataset(cfg):
        return "q01/q99"
    if _is_libero_dataset(cfg):
        return "min/max"
    return "z-score"


def _normalizer_stats_src(cfg: PrecomputeConfig) -> str:
    stats_path = _optional_path(getattr(cfg, "dataset_stats_path", None))
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
    if str(cfg.camera_layout).startswith("robocasa_dreamzero_"):
        if contract["video_model_family"] == "wan22_i2v_a14b":
            contract["require_i2v_first_latent_equality"] = True
    if cfg.dataset_name in yam.DATASET_NAMES:
        contract["abc_contract"] = yam.PRESETS[cfg.dataset_name].contract
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
        "tag_version": 4 if str(cfg.camera_layout).startswith("robocasa_dreamzero_") else 2,
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
        # Keep versioned hash keys stable so existing latent caches remain reusable.
        "concat_multi_camera": str(cfg.camera_layout),
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
    hasher.update(json.dumps(tag_payload, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()[:16]


def _read_cache_header(path: Path) -> dict[str, Any]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {
            "metadata": dict(handle.metadata() or {}),
            "latents_shape": tuple(handle.get_slice("latents").get_shape()),
            "tensor_shapes": {
                key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()
            },
        }
