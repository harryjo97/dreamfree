"""Load portable Hugging Face releases without their original training paths."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dreamfree.config_names import normalize_checkpoint_metadata

POLICY_WEIGHTS = (
    "action.safetensors",
    "readout.safetensors",
    "conditioner.safetensors",
    "proprio_encoder.safetensors",
)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def resolve_policy_checkpoint(path: str | Path) -> Path:
    """Accept a native checkpoint, a release root, or its stage2 directory."""
    path = Path(os.path.expandvars(str(path))).expanduser().resolve()
    if (path / "stage2/config.json").is_file():
        path = path / "stage2"
    if not path.is_dir():
        raise FileNotFoundError(f"Policy checkpoint directory not found: {path}")
    return path


def is_release_checkpoint(path: Path) -> bool:
    return (path / "config.json").is_file() and not (path / "trainer_state.json").is_file()


def _relative_file(directory: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Release filename must be relative to {directory}: {name!r}")
    path = directory / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing release file: {path}")
    return path


def student_weight_paths(path: Path, release: dict) -> list[Path]:
    """Check the shard index against actual tensor names before loading weights."""
    from safetensors import safe_open

    weights = _relative_file(path, release["weights"])
    if weights.suffix == ".safetensors":
        return [weights]
    if not weights.name.endswith(".safetensors.index.json"):
        raise ValueError(f"Unsupported student weights: {weights}")
    weight_map = _read_json(weights).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Missing or empty weight_map in {weights}.")
    paths = []
    for name in sorted(set(weight_map.values())):
        shard = _relative_file(path, name)
        expected = {key for key, filename in weight_map.items() if filename == name}
        with safe_open(shard, framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
        if actual != expected:
            raise ValueError(
                f"Shard/index mismatch in {shard}: "
                f"missing={sorted(expected - actual)[:10]}, extra={sorted(actual - expected)[:10]}"
            )
        paths.append(shard)
    return paths


def read_checkpoint_metadata(path: Path, *, stage: str) -> dict:
    """Adapt version-1 release metadata to the existing runtime validators.

    The published release keeps shape and lineage metadata in provenance JSON.
    Those files are read locally; no weights or datasets are loaded through their
    recorded training paths.
    """
    if stage not in ("stage1", "stage2"):
        raise ValueError(f"Unsupported checkpoint stage: {stage!r}")
    if not is_release_checkpoint(path):
        return normalize_checkpoint_metadata(_read_json(path / "trainer_state.json"))
    release = _read_json(path / "config.json")
    if (release.get("format"), release.get("format_version")) != (f"dreamfree_release_{stage}", 1):
        raise ValueError(f"Unsupported {stage} release format/version in {path / 'config.json'}.")
    metadata = _read_json(path.parent / "provenance" / f"{stage}_trainer_state.json")
    supported = {
        "stage1": {("fastwam_stage1", 2), ("fastwam_stage1_full", 2)},
        "stage2": {("new_idea_stage2_student_native_vel", 2), ("vam_stage2_action_history", 1)},
    }
    source_type = (metadata.get("checkpoint_type"), metadata.get("checkpoint_version"))
    if source_type not in supported[stage]:
        raise ValueError(f"Unsupported {stage} release provenance: {source_type!r}")
    # Never silently combine a release with different provenance or inference settings.
    for key in (
        "global_step",
        "model_contract",
        "student",
        "action",
        "readout",
        "conditioner",
        "shapes",
        "action_history",
    ):
        if key in release and release[key] != metadata.get(key):
            raise ValueError(f"Release/provenance mismatch for {stage}.{key}.")
    for key, value in release["config"].items():
        if metadata["config"].get(key) != value:
            raise ValueError(f"Release/provenance mismatch for {stage}.config.{key}.")
    metadata["config"] = dict(release["config"])
    metadata = normalize_checkpoint_metadata(metadata)
    metadata["config"]["model_base_path"] = os.environ.get(
        "DREAMFREE_MODEL_BASE_PATH", "./assets/models"
    )
    metadata["release"] = release
    if stage == "stage1":
        if (
            release.get("standalone_student") is not True
            or release.get("requires_stage0") is not False
        ):
            raise ValueError("A Stage 1 release must contain a standalone student without Stage 0.")
        mode = metadata.get("student_train_mode", "lora")
        if release.get("source_train_mode") != mode:
            raise ValueError("Release source_train_mode disagrees with Stage 1 provenance.")
        metadata["checkpoint_type"] = "dreamfree_stage1"
        student_weight_paths(path, release)
    else:
        if release.get("stage1_checkpoint") != "../stage1":
            raise ValueError("Version-1 policy releases require stage1_checkpoint='../stage1'.")
        if tuple(release.get("weights", ())) != POLICY_WEIGHTS:
            raise ValueError(f"Stage 2 release must contain policy weights {POLICY_WEIGHTS}.")
        for name in POLICY_WEIGHTS:
            _relative_file(path, name)
        source_path = path.parent / "stage1"
        source = read_checkpoint_metadata(source_path, stage="stage1")
        source_student = {
            key: source["student"][key]
            for key in ("hook_point", "i2v_condition_mode", "layer_j", "video_attention_mask_mode")
        }
        source_student.update(
            hidden_dim=source["shapes"]["D_student"],
            latent_shape=source["shapes"]["latent_shape"],
            tokens_per_frame=source["shapes"]["tokens_per_frame"],
        )
        for key, expected in (
            ("source_stage1_global_step", source["global_step"]),
            ("source_stage1_student_metadata", source_student),
            ("source_stage1_model_contract", source["model_contract"]),
            ("stage0_lora_checkpoint", source.get("stage0_lora_checkpoint")),
        ):
            if metadata.get(key) != expected:
                raise ValueError(f"Release Stage 1/Stage 2 lineage mismatch: {key}.")
        metadata["checkpoint_type"] = (
            "dreamfree_stage2_action_history"
            if source_type[0] == "vam_stage2_action_history"
            else "dreamfree_stage2"
        )
        metadata["source_stage1_checkpoint"] = str(source_path.resolve())
        stats = _relative_file(path, release["dataset_stats"])
        metadata["dataset"]["dataset_stats_path"] = str(stats.resolve())
        metadata["config"]["dataset_stats_path"] = str(stats.resolve())
        metadata["config"]["stage1_checkpoint"] = str(source_path.resolve())
        if source_type[0] == "vam_stage2_action_history":
            _adapt_action_history(metadata)
    return metadata


def _adapt_action_history(metadata: dict) -> None:
    from types import SimpleNamespace

    from dreamfree.models.action_history import TOKEN_ORDER, contract

    if metadata["config"].get("dataset_name") != "droid":
        raise ValueError("Version-1 action-history releases support DROID only.")
    canonical = contract(SimpleNamespace(**metadata["config"]))
    if canonical is None:
        raise ValueError("DROID action-history release must configure a positive history horizon.")
    original = {
        key: value
        for key, value in canonical.items()
        if key not in ("action_layout", "normalization", "deployment_source")
    }
    original.update(
        droid_gripper="absolute",
        droid_gripper_history="binary_absolute_strict_gt_0.5_before_q99",
    )
    if metadata["action_history"] != original:
        raise ValueError("Unsupported DROID release action-history contract.")
    metadata["action_history"] = canonical
    metadata["training_variant"] = "action_history"
    metadata["shapes"]["query_token_order"] = TOKEN_ORDER


def required_policy_files(metadata: dict) -> tuple[str, ...]:
    return ("config.json" if "release" in metadata else "trainer_state.json", *POLICY_WEIGHTS)


def load_release_student(
    path: Path,
    metadata: dict,
    *,
    device,
    dtype,
    load_text_encoder=True,
    load_vae=True,
    distributed_checkpoint_broadcast=True,
):
    """Load the complete student directly, then attach upstream VAE/text assets."""
    from dreamfree.models.wan22.helpers.loader import (
        _load_dit_with_optional_broadcast,
        _load_wan_video_dit_native_shards,
    )
    from dreamfree.models.wan22.stage0_loader import default_stage1_dit_config
    from dreamfree.models.wan22.wan22 import Wan22Core

    release, cfg = metadata["release"], metadata["config"]
    os.environ.setdefault("DREAMFREE_MODEL_BASE_PATH", cfg["model_base_path"])
    os.environ.setdefault("DREAMFREE_DOWNLOAD_SOURCE", "huggingface")
    family = metadata["model_contract"]["video_model_family"]
    architecture = default_stage1_dit_config(video_model_family=family)
    if json.loads(json.dumps(architecture)) != release["architecture"]:
        raise ValueError("Released student architecture does not match its video model family.")
    paths = student_weight_paths(path, release)
    dit = _load_dit_with_optional_broadcast(
        model_name="dreamfree_release_student",
        model_kwargs=architecture,
        torch_dtype=dtype,
        device=device,
        pretrained_load_broadcast=distributed_checkpoint_broadcast,
        rank0_loader=lambda load_device: _load_wan_video_dit_native_shards(
            paths=paths,
            model_name="dreamfree_release_student",
            model_kwargs=architecture,
            torch_dtype=dtype,
            device=load_device,
            strict=True,
        ),
    )
    core = Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=dtype,
        model_id=cfg["model_id"],
        tokenizer_model_id=cfg["tokenizer_model_id"],
        tokenizer_max_len=cfg["tokenizer_max_len"],
        dit_config=architecture,
        dit_override=dit,
        video_model_family=family,
        dit_role=metadata["model_contract"]["student_dit_role"],
        load_text_encoder=load_text_encoder,
        load_vae=load_vae,
        redirect_common_files=cfg.get("redirect_common_files", True),
        pretrained_load_broadcast=distributed_checkpoint_broadcast,
        train_shift=cfg.get("train_shift", 5.0),
        infer_shift=cfg.get("infer_shift", 5.0),
        num_train_timesteps=cfg.get("num_train_timesteps", 1000),
    )
    core.model_paths["dit"] = str(path / release["weights"])
    core.student_lora_adapter_name = None
    return core.requires_grad_(False).eval()
