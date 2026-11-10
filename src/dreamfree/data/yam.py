"""Single-embodiment YAM data, packing, and cache contracts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ABC_NAME = "yam_abc130k"
ABC_EEF_NAME = "yam_abc130k_eef"
EEF_OVERLAY_KIND = "yam_eef_action_overlay_v1"
CAMERA_KEYS = tuple(f"observation.images.{key}" for key in ("top", "left", "right"))
JOINT_NAMES = [
    f"{side}_{joint}.pos"
    for side in ("left", "right")
    for joint in (*[f"joint_{i}" for i in range(6)], "gripper")
]
LAYOUT = "yam_abc130k_top_wrist_384x432"
CONTRACT = {
    "yam_contract": "yam_absolute14_center95_v1",
    "camera_keys": ",".join(CAMERA_KEYS),
    "concat_layout": LAYOUT,
    "crop_policy": "center_0.95_bilinear_antialias",
    "fps": "30",
    "action_layout": "yam_absolute_joint_gripper_14_v1",
}


@dataclass(frozen=True)
class YamPreset:
    raw_height: int
    crop_height: int
    width: int
    layout: str
    env_prefix: str
    directory: str
    run_name: str
    contract: dict[str, str]
    action_dim: int = 14
    proprio_dim: int = 14


PRESETS = {
    ABC_NAME: YamPreset(
        480,
        456,
        384,
        "yam_abc130k_top_wrist_384x432",
        "ABC130K",
        "yam_abc130k_640x480_100h_train_5h_val_gap200",
        "yam_abc130k",
        {
            **CONTRACT,
            "yam_contract": "abc130k_yam_absolute14_center95_causal30_v1",
            "concat_layout": "yam_abc130k_top_wrist_384x432",
            "source_dataset": "XDOF/ABC-130k",
            "source_resolution": "640x480",
            "timestamp_alignment": "causal_30hz_common_overlap_v1",
        },
    ),
}
PRESETS[ABC_EEF_NAME] = YamPreset(
    480,
    456,
    384,
    PRESETS[ABC_NAME].layout,
    "ABC130K_EEF",
    PRESETS[ABC_NAME].directory + "_eef",
    "yam_abc130k_eef",
    {
        **PRESETS[ABC_NAME].contract,
        "yam_contract": "abc130k_yam_eef20_jointstate14_center95_causal30_v1",
        "action_layout": "yam_absolute_eef_xyz_rot6d_gripper_20_v1",
        "eef_frame": "per_arm_base",
        "eef_reference": "yam_tcp_site_at_wrist",
        "rotation_layout": "R00_R10_R20_R01_R11_R21",
    },
    action_dim=20,
)
NAME = ABC_NAME
DATASET_NAMES = frozenset(PRESETS)
LAYOUT_DATASETS = {preset.layout: name for name, preset in PRESETS.items() if name != ABC_EEF_NAME}


def paths(dataset_name: str = NAME) -> dict[str, str]:
    preset = PRESETS[dataset_name]
    base = str(Path(__file__).resolve().parents[3] / "data")
    prefix = preset.env_prefix
    root = os.environ.get(f"{prefix}_DATASET_ROOT", f"{base}/{preset.directory}")
    if dataset_name == ABC_EEF_NAME:
        return {
            "root": root,
            "stats": os.environ.get(f"{prefix}_NORM_STATS", f"{root}/dataset_stats.json"),
            "cache": os.environ.get(f"{prefix}_LATENT_CACHE_DIR", f"{root}/action_cache"),
            "text": os.environ.get(f"{prefix}_TEXT_CACHE_DIR", paths(ABC_NAME)["text"]),
        }
    return {
        "root": root,
        "stats": os.environ.get(f"{prefix}_NORM_STATS", f"{root}/dataset_stats.json"),
        "cache": os.environ.get(
            f"{prefix}_LATENT_CACHE_DIR",
            f"{base}/video_latent_cache_a14b_official_i2v_condition/{preset.run_name}_center95_432x{preset.width}_h33_stride4/{Path(root).name}",
        ),
        "text": os.environ.get(
            f"{prefix}_TEXT_CACHE_DIR", f"{base}/text_embeds_cache/{dataset_name}_a14b"
        ),
    }


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_selection(root: str | Path, dataset_name: str = NAME) -> dict:
    path = Path(root) / "yam_selection.json"
    return _load_selection(str(path.resolve()), path.stat().st_mtime_ns, dataset_name)


@lru_cache(maxsize=8)
def _load_selection(filename: str, mtime_ns: int, dataset_name: str) -> dict:
    path = Path(filename)
    value = json.loads(path.read_text())
    unsigned = {key: item for key, item in value.items() if key != "selection_sha256"}
    if value.get("schema_version") != 1 or value.get("selection_sha256") != fingerprint(unsigned):
        raise ValueError(f"Invalid YAM selection fingerprint/schema: {path}")
    if value.get("contract") != PRESETS[dataset_name].contract or value.get("fps") != 30:
        raise ValueError(f"YAM selection has an incompatible preprocessing contract: {path}")
    return value


def episode_metadata(
    root: str | Path, episode_index: int, dataset_name: str = NAME
) -> dict[str, str]:
    selection = load_selection(root, dataset_name)
    episode = selection["episodes"][str(episode_index)]
    return {
        **PRESETS[dataset_name].contract,
        "video_value_contract": "uint8_to_float01_before_resize_minus1_to1_v1",
        "selection_sha256": selection["selection_sha256"],
        "normalization_sha256": selection["normalization_sha256"],
        "source_task": episode["instruction"],
        # Annotations can differ within a task class, so each episode has its own prompt ID.
        "source_task_index": str(episode_index),
        "source_task_class": str(episode["task_id"]),
        "yam_split": episode["split"],
    }


def validate_stats(root: Path, stats_path: Path, dataset_name: str = NAME) -> None:
    selection = load_selection(root, dataset_name)
    if fingerprint(json.loads(stats_path.read_text())) != selection["normalization_sha256"]:
        raise ValueError(f"YAM statistics differ from the selected training pool: {stats_path}")


def validate_config(cfg, *, stage: int) -> None:
    preset = PRESETS[getattr(cfg, "dataset_name", NAME)]
    if getattr(cfg, "dataset_name", NAME) == ABC_EEF_NAME and stage != 2:
        raise ValueError(
            "yam_abc130k_eef is a Stage 2 action dataset; use yam_abc130k for video training."
        )
    expected = {"height": 432, "width": preset.width, "camera_layout": preset.layout}
    if stage in (0, 1):
        expected.update(
            clip_frames=9,
            raw_frame_stride=4,
            dataset_format="lerobot_root",
            camera_keys=",".join(CAMERA_KEYS),
            prompt_policy="task_text",
        )
    else:
        expected.update(
            num_frames=33,
            action_video_freq_ratio=4,
            action_horizon=32,
            action_dim=preset.action_dim,
            proprio_dim=preset.proprio_dim,
        )
    for key, value in expected.items():
        if getattr(cfg, key, None) != value:
            raise ValueError(f"YAM requires {key}={value!r}, got {getattr(cfg, key, None)!r}")
    if stage == 2 and int(getattr(cfg, "global_sample_stride", 1)) != 1:
        raise ValueError(
            "YAM requires global_sample_stride=1 to preserve the native 30 Hz action rate."
        )
    family = getattr(cfg, "video_model_family", None)
    if family != "wan22_i2v_a14b":
        raise ValueError(f"YAM preset requires video_model_family='wan22_i2v_a14b', got {family!r}")


def validate_cache(path: Path, root: str | Path, dataset_name: str = NAME) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        shapes = {
            key: tuple(handle.get_slice(key).get_shape())
            for key in ("action", "proprio", "task_indices")
        }
    index = int(metadata["episode_index"])
    selection = load_selection(root, dataset_name)
    if str(index) not in selection["episodes"]:
        raise ValueError(f"YAM cache episode {index} is absent from the selected pool: {path}")
    length = int(selection["episodes"][str(index)]["length"])
    preset = PRESETS[dataset_name]
    if shapes != {
        "action": (length, preset.action_dim),
        "proprio": (length, preset.proprio_dim),
        "task_indices": (length,),
    }:
        raise ValueError(
            f"YAM cache action/state shapes differ from the selected episode: {path}: {shapes}"
        )
    expected = episode_metadata(root, index, dataset_name)
    expected.update(
        video_h="432",
        video_w=str(PRESETS[dataset_name].width),
        num_frames_per_window="33",
        action_video_freq_ratio="4",
        action_horizon="32",
        num_frames=str(length),
        num_samples=str(length),
        action_dim=str(preset.action_dim),
        proprio_dim=str(preset.proprio_dim),
        norm_mode="q01/q99",
    )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"YAM cache {path}: {key}={metadata.get(key)!r}, expected {value!r}")
    if dataset_name == ABC_EEF_NAME and metadata.get("overlay_cache_kind") == EEF_OVERLAY_KIND:
        from dreamfree.data.yam_eef import validate_overlay_source

        validate_overlay_source(path, metadata, selection)
    return metadata


def select_cache_paths(
    cache_dirs, root: str | Path, split: str, dataset_name: str = NAME
) -> set[Path]:
    if split not in {"train", "val"}:
        raise ValueError(f"YAM supports the prepared train/val splits, got {split!r}")
    selection = load_selection(root, dataset_name)
    expected = {int(key) for key in selection["episodes"]}
    found: dict[int, Path] = {}
    selected = set()
    for directory in cache_dirs:
        for path in sorted(
            set(Path(directory).glob("episode_*.safetensors"))
            | set(Path(directory).glob("*/episode_*.safetensors"))
        ):
            meta = validate_cache(path, root, dataset_name)
            index = int(meta["episode_index"])
            if index in found:
                raise ValueError(f"Duplicate YAM episode {index}: {found[index]} and {path}")
            found[index] = path
            if meta["yam_split"] == split:
                selected.add(path)
    if set(found) != expected:
        raise ValueError(
            f"Incomplete YAM cache: missing episodes {sorted(expected - set(found))[:10]}, unexpected {sorted(set(found) - expected)[:10]}"
        )
    return selected


def shape_meta(dataset_name: str = NAME) -> dict:
    preset = PRESETS[dataset_name]
    height = preset.raw_height
    return {
        "images": [
            {
                "key": key.rsplit(".", 1)[-1],
                "raw_shape": [3, height, 640],
                "shape": [3, height, 640],
            }
            for key in CAMERA_KEYS
        ],
        "action": [{"key": "default", "raw_shape": preset.action_dim, "shape": preset.action_dim}],
        "state": [{"key": "default", "raw_shape": preset.proprio_dim, "shape": preset.proprio_dim}],
    }


def pack_views(views, dataset_name: str = NAME):
    """Pack three [T,C,H,W] float tensors without changing their value range."""
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    preset = PRESETS[dataset_name]
    if len(views) != 3 or any(
        tuple(view.shape[-3:]) != (3, preset.raw_height, 640) for view in views
    ):
        raise ValueError(
            f"{dataset_name} packing requires three RGB views with shape [T,3,{preset.raw_height},640]."
        )
    if len({int(view.shape[0]) for view in views}) != 1:
        raise ValueError("YAM camera frame counts must match.")
    panels = []
    for view, size in zip(
        views,
        ((288, preset.width), (144, preset.width // 2), (144, preset.width // 2)),
        strict=True,
    ):
        cropped = TF.center_crop(view, [preset.crop_height, 608])
        panels.append(
            TF.resize(cropped, list(size), interpolation=InterpolationMode.BILINEAR, antialias=True)
        )
    return torch.cat((panels[0], torch.cat(panels[1:], dim=-1)), dim=-2)


def validate_stage1_source(metadata: dict) -> None:
    """Require the shared ABC visual backbone for both Stage 2 action spaces."""
    cfg = SimpleNamespace(**metadata["config"])
    if cfg.dataset_name != ABC_NAME:
        raise ValueError("ABC Stage 2 requires an ABC joint-dataset Stage 1 visual backbone.")
    validate_config(cfg, stage=1)


def checkpoint_contract(cfg) -> dict:
    """Freeze the prepared labels, preprocessing and normalization with the model."""
    roots = [part.strip() for part in str(cfg.dataset_dirs).split(",") if part.strip()]
    if len(roots) != 1:
        raise ValueError("ABC requires one prepared dataset root.")
    root = Path(roots[0]).expanduser()
    selection = load_selection(root, cfg.dataset_name)
    validate_stats(root, Path(cfg.dataset_stats_path), cfg.dataset_name)
    result = dict(
        dataset_name=cfg.dataset_name,
        selection_sha256=selection["selection_sha256"],
        normalization_sha256=selection["normalization_sha256"],
        preprocessing=PRESETS[cfg.dataset_name].contract,
    )
    if cfg.dataset_name == ABC_EEF_NAME:
        from dreamfree.data.yam_eef import MODEL_SHA256

        recorded = json.loads((root / "kinematics.json").read_text())
        if (
            fingerprint(recorded) != selection["eef_conversion"]["kinematics_sha256"]
            or recorded.get("model_sha256") != MODEL_SHA256
            or recorded.get("frame") != "per_arm_base"
            or recorded.get("site") != "tcp_site"
        ):
            raise ValueError("ABC EEF labels must use the pinned YAM wrist model in each arm base.")
        result["kinematics_sha256"] = selection["eef_conversion"]["kinematics_sha256"]
    return result


def cache_versions(expected: dict, metadata: dict) -> tuple[int, ...]:
    """Admit AWAM v2 caches only under the complete, explicit ABC visual contract."""
    contract = expected.get("abc_contract")
    if contract is None:
        return (3,)
    for key, value in {
        **contract,
        "video_value_contract": "uint8_to_float01_before_resize_minus1_to1_v1",
    }.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"ABC video cache {key}: expected {value!r}, got {metadata.get(key)!r}."
            )
    return (2, 3)
