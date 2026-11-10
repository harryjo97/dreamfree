"""Shared checkpoint contracts, lineage validation, and readout loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import dreamfree.models.policy_loading as dreamfree_deploy
from dreamfree.evaluation.options import is_none_like
from dreamfree.models.student_readout import (
    StudentLatentQueryReadout,
    readout_input_projection_mlp_ratio_from_metadata,
    readout_input_projection_type_from_metadata,
)
from dreamfree.models.wan22.helpers.distributed_load import (
    load_safetensors_rank0 as load_safetensors,
)

DREAMFREE_STAGE2_CHECKPOINT_TYPE = "dreamfree_stage2"
DREAMFREE_STAGE2_CHECKPOINT_VERSION = 2
DREAMFREE_POLICY_RUNTIME = "dreamfree_stage2_student_native"
READOUT_TYPE_LATENT_QUERY = "latent_query_cross_attention"
SUPPORTED_READOUT_TYPES = {READOUT_TYPE_LATENT_QUERY}
STAGE1_SUPPORTED_CHECKPOINT_TYPES = dreamfree_deploy.STAGE1_SUPPORTED_CHECKPOINT_TYPES
DREAMFREE_STAGE2_REQUIRED_FILES = (
    "trainer_state.json",
    "action.safetensors",
    "readout.safetensors",
    "conditioner.safetensors",
    "proprio_encoder.safetensors",
)


def read_native_checkpoint_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}.")
    from dreamfree.config_names import normalize_checkpoint_metadata

    return normalize_checkpoint_metadata(payload)


def require_dict(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object, got {type(value).__name__}.")
    return value


def check_equal(mismatches: list[str], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        mismatches.append(f"{name}: actual={actual!r} expected={expected!r}")


def readout_type_from_metadata(metadata: dict[str, Any]) -> str:
    readout = require_dict(metadata.get("readout"), label="DreamFree Stage 2 metadata.readout")
    return str(readout.get("type", READOUT_TYPE_LATENT_QUERY))


def condition_video_num_frames(metadata: dict[str, Any]) -> int:
    dataset = require_dict(metadata.get("dataset"), label="DreamFree Stage 2 metadata.dataset")
    num_frames = int(dataset.get("num_frames", -1))
    action_video_freq_ratio = int(dataset.get("action_video_freq_ratio", -1))
    if num_frames <= 1 or action_video_freq_ratio <= 0:
        raise ValueError(
            f"DreamFree Stage 2 metadata.dataset must define positive num_frames and action_video_freq_ratio, got num_frames={num_frames} action_video_freq_ratio={action_video_freq_ratio}."
        )
    if (num_frames - 1) % action_video_freq_ratio != 0:
        raise ValueError(
            f"DreamFree Stage 2 metadata.dataset num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {action_video_freq_ratio}."
        )
    condition_video_num_frames = (num_frames - 1) // action_video_freq_ratio + 1
    if condition_video_num_frames % 4 != 1:
        raise ValueError(
            f"DreamFree Stage 2 deployment condition video frame count must satisfy T % 4 == 1, got {condition_video_num_frames} from num_frames={num_frames} action_video_freq_ratio={action_video_freq_ratio}."
        )
    return condition_video_num_frames


def resolve_checkpoint_path(checkpoint_path: str) -> Path:
    resolved = Path(os.path.expanduser(os.path.expandvars(str(checkpoint_path)))).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {resolved}")
    return resolved


def require_checkpoint_dir(path_value: Any, *, label: str) -> Path:
    if is_none_like(path_value):
        raise ValueError(f"{label} checkpoint path is required.")
    path = resolve_checkpoint_path(str(path_value))
    if path.is_file():
        raise ValueError(f"{label} checkpoint must be a directory, got file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} checkpoint directory not found: {path}")
    return path


def require_existing_files(checkpoint_path: Path, names: tuple[str, ...], *, label: str) -> None:
    missing = [name for name in names if not (checkpoint_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label} files {missing}: {checkpoint_path}")


def require_existing_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected {label} to be a file: {path}")
    return path


def path_identity(path_value: Any) -> Path | None:
    if is_none_like(path_value):
        return None
    return Path(str(path_value)).expanduser().resolve()


def stage2_config_from_metadata(metadata: dict[str, Any]) -> SimpleNamespace:
    """Read saved runtime settings without applying current training defaults.

    Args:
        metadata: Validated native or normalized release Stage 2 metadata.

    Returns:
        Attribute namespace containing the checkpoint's config dictionary.
    """
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("DreamFree Stage 2 checkpoint metadata is missing dict `config`.")
    return SimpleNamespace(**config)


def metadata_video_family(metadata: dict[str, Any], *, label: str) -> str:
    return dreamfree_deploy._metadata_video_family(metadata, label=label)


def expected_hidden_dim_for_family(family: str) -> int:
    return dreamfree_deploy._expected_hidden_dim_for_family(family)


def expected_model_id_for_family(family: str) -> str:
    return dreamfree_deploy._expected_model_id_for_family(family)


def readout_layer_from_metadata(metadata: dict[str, Any]) -> int:
    student = require_dict(metadata.get("student"), label="DreamFree Stage 2 metadata.student")
    readout = require_dict(metadata.get("readout"), label="DreamFree Stage 2 metadata.readout")
    return int(readout.get("input_layer_j", student.get("layer_j", -1)))


def validate_stage1_checkpoint(checkpoint_path: Path, metadata: dict[str, Any]) -> None:
    """Check encoder files and geometry needed by every action-policy adapter.

    Args:
        checkpoint_path: Stage 1 directory containing the encoder artifacts.
        metadata: Native or normalized release Stage 1 metadata.

    Raises:
        ValueError: If the encoder metadata violates the shared model contract.
        FileNotFoundError: If required encoder artifacts are absent.
    """
    require_existing_files(
        checkpoint_path,
        dreamfree_deploy.stage1_required_student_files_for_metadata(metadata),
        label="Stage 1",
    )
    mismatches: list[str] = []
    checkpoint_type = str(metadata.get("checkpoint_type", "")).strip()
    if checkpoint_type not in STAGE1_SUPPORTED_CHECKPOINT_TYPES:
        mismatches.append(
            f"checkpoint_type: actual={checkpoint_type!r} expected one of {sorted(STAGE1_SUPPORTED_CHECKPOINT_TYPES)!r}"
        )
    metadata = dreamfree_deploy._stage1_student_view_metadata(metadata)
    family = metadata_video_family(metadata, label="Stage 1")
    expected_hidden_dim = expected_hidden_dim_for_family(family)
    student = require_dict(metadata.get("student"), label="Stage 1 metadata.student")
    shapes = require_dict(metadata.get("shapes"), label="Stage 1 metadata.shapes")
    config = require_dict(metadata.get("config"), label="Stage 1 metadata.config")
    model_contract = metadata.get("model_contract")
    if not isinstance(model_contract, dict):
        model_contract = {}
    check_equal(
        mismatches,
        "student.video_attention_mask_mode",
        student.get("video_attention_mask_mode"),
        "first_frame_causal",
    )
    check_equal(mismatches, "student.hook_point", student.get("hook_point"), "post_block")
    check_equal(
        mismatches, "config.tokenizer_max_len", int(config.get("tokenizer_max_len", -1)), 128
    )
    check_equal(mismatches, "config.text_dim", int(config.get("text_dim", -1)), 4096)
    check_equal(
        mismatches,
        "config.video_hidden_dim",
        int(config.get("video_hidden_dim", -1)),
        expected_hidden_dim,
    )
    check_equal(
        mismatches, "config.model_id", config.get("model_id"), expected_model_id_for_family(family)
    )
    for key in ("tokens_per_frame", "D_student", "latent_shape"):
        if key not in shapes:
            mismatches.append(f"shapes.{key} is missing")
    if "tokens_per_frame" in shapes and int(shapes["tokens_per_frame"]) <= 0:
        mismatches.append(
            f"shapes.tokens_per_frame must be positive, got {shapes['tokens_per_frame']!r}."
        )
    if "D_student" in shapes:
        check_equal(mismatches, "shapes.D_student", int(shapes["D_student"]), expected_hidden_dim)
    if "latent_shape" in shapes and len(tuple(shapes["latent_shape"])) != 4:
        mismatches.append(f"shapes.latent_shape must be [C,T,H,W], got {shapes['latent_shape']!r}.")
    if family == dreamfree_deploy.WAN22_A14B_FAMILY:
        check_equal(
            mismatches,
            "model_contract.video_latent_dim",
            int(model_contract.get("video_latent_dim", config.get("video_latent_dim", -1))),
            16,
        )
        check_equal(
            mismatches,
            "model_contract.video_dit_input_dim",
            int(model_contract.get("video_dit_input_dim", config.get("video_dit_input_dim", -1))),
            36,
        )
        check_equal(
            mismatches,
            "model_contract.video_dit_output_dim",
            int(model_contract.get("video_dit_output_dim", config.get("video_dit_output_dim", -1))),
            16,
        )
        check_equal(
            mismatches,
            "student.i2v_condition_mode",
            student.get("i2v_condition_mode"),
            dreamfree_deploy.WAN22_A14B_I2V_MODE,
        )
    global_step = int(metadata.get("global_step", -1))
    if global_step < 0:
        mismatches.append(f"global_step must be non-negative, got {global_step}.")
    if mismatches:
        raise ValueError(
            "Stage 1 checkpoint is incompatible with DreamFree eval: " + "; ".join(mismatches)
        )


def validate_stage1_stage2_lineage(
    *,
    stage1_checkpoint_path: Path,
    stage1_metadata: dict[str, Any],
    stage2_metadata: dict[str, Any],
) -> None:
    """Require the action checkpoint to match its recorded frozen encoder.

    Args:
        stage1_checkpoint_path: Resolved source encoder directory.
        stage1_metadata: Native or normalized release encoder metadata.
        stage2_metadata: Action-policy metadata naming this encoder and geometry.

    Raises:
        ValueError: If source identity, training step, token layout, or model
            family differs. Historical Stage 0 paths are checked only for
            native encoder sources; portable releases resolve their own files.
    """
    validate_stage1_checkpoint(stage1_checkpoint_path, stage1_metadata)
    stage1_metadata = dreamfree_deploy._stage1_student_view_metadata(stage1_metadata)
    stage1_student = require_dict(stage1_metadata.get("student"), label="Stage 1 metadata.student")
    stage1_shapes = require_dict(stage1_metadata.get("shapes"), label="Stage 1 metadata.shapes")
    stage2_student = require_dict(
        stage2_metadata.get("student"), label="DreamFree Stage 2 metadata.student"
    )
    source_student = require_dict(
        stage2_metadata.get("source_stage1_student_metadata"),
        label="DreamFree Stage 2 metadata.source_stage1_student_metadata",
    )
    stage1_family = metadata_video_family(stage1_metadata, label="Stage 1")
    stage2_family = metadata_video_family(stage2_metadata, label="DreamFree Stage 2")
    mismatches: list[str] = []
    check_equal(
        mismatches,
        "source_stage1_checkpoint",
        stage1_checkpoint_path.resolve(),
        path_identity(stage2_metadata.get("source_stage1_checkpoint")),
    )
    check_equal(
        mismatches,
        "source_stage1_global_step",
        int(stage1_metadata.get("global_step", -1)),
        int(stage2_metadata.get("source_stage1_global_step", -1)),
    )
    check_equal(
        mismatches,
        "Stage 1 video_model_family vs DreamFree Stage 2 video_model_family",
        stage1_family,
        stage2_family,
    )
    check_equal(
        mismatches,
        "Stage 1 layer_j vs DreamFree student.layer_j",
        int(stage1_student.get("layer_j", -1)),
        int(stage2_student.get("layer_j", -1)),
    )
    check_equal(
        mismatches,
        "Stage 1 hook_point vs DreamFree student.hook_point",
        stage1_student.get("hook_point"),
        stage2_student.get("hook_point"),
    )
    check_equal(
        mismatches,
        "Stage 1 tokens_per_frame vs DreamFree student.tokens_per_frame",
        int(stage1_shapes.get("tokens_per_frame", -1)),
        int(stage2_student.get("tokens_per_frame", -1)),
    )
    check_equal(
        mismatches,
        "Stage 1 D_student vs DreamFree student.hidden_dim",
        int(stage1_shapes.get("D_student", -1)),
        int(stage2_student.get("hidden_dim", -1)),
    )
    check_equal(
        mismatches,
        "Stage 1 i2v_condition_mode vs DreamFree source",
        str(stage1_student.get("i2v_condition_mode", "none")),
        str(source_student.get("i2v_condition_mode", "none")),
    )
    if "release" not in stage1_metadata:
        check_equal(
            mismatches,
            "Stage 0 checkpoint identity",
            path_identity(stage1_metadata.get("stage0_lora_checkpoint")),
            path_identity(stage2_metadata.get("stage0_lora_checkpoint")),
        )
    if mismatches:
        raise ValueError(
            "Stage 1 source checkpoint cannot be composed with DreamFree Stage 2 eval: "
            + "; ".join(mismatches)
        )


def load_readout(
    *, metadata: dict[str, Any], checkpoint_path: Path, device: torch.device, dtype: torch.dtype
) -> torch.nn.Module:
    """Construct and strictly load the checkpoint's latent-query readout.

    Args:
        metadata: Stage 2 metadata containing the readout architecture.
        checkpoint_path: Directory containing readout.safetensors.
        device: Device for the loaded module.
        dtype: Computation/storage dtype for the loaded module.

    Returns:
        Readout in eval mode, mapping (B, N, input_dim) visual tokens to
        (B, Q, output_dim) context tokens and a (B, Q) validity mask.
    """
    readout_meta = require_dict(metadata.get("readout"), label="DreamFree Stage 2 metadata.readout")
    readout_type = str(readout_meta.get("type", READOUT_TYPE_LATENT_QUERY))
    common_kwargs = {
        "student_dim": int(readout_meta["input_dim"]),
        "readout_dim": int(readout_meta["readout_dim"]),
        "output_dim": int(readout_meta["output_dim"]),
        "num_layers": int(readout_meta["num_layers"]),
        "num_heads": int(readout_meta["num_heads"]),
        "mlp_ratio": float(readout_meta["mlp_ratio"]),
        "dropout": float(readout_meta["dropout"]),
        "input_projection_type": readout_input_projection_type_from_metadata(readout_meta),
        "input_projection_mlp_ratio": readout_input_projection_mlp_ratio_from_metadata(
            readout_meta
        ),
    }
    if readout_type == READOUT_TYPE_LATENT_QUERY:
        readout = StudentLatentQueryReadout(
            **common_kwargs, num_queries=int(readout_meta["num_queries"])
        )
    else:
        raise ValueError(f"Unsupported DreamFree Stage 2 readout.type={readout_type!r}.")
    readout = readout.to(device=device, dtype=dtype)
    readout.load_state_dict(
        load_safetensors(str(checkpoint_path / "readout.safetensors"), device="cpu"), strict=True
    )
    readout.eval()
    return readout
