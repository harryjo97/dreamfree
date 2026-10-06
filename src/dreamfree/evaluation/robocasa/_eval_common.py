"""Shared helpers for RoboCasa DreamFree Stage 2 evaluation."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path.cwd()
DREAMFREE_POLICY_RUNTIME = "dreamfree_stage2_student_native"
DREAMFREE_STAGE2_CHECKPOINT_TYPE = "dreamfree_stage2"
SUPPORTED_DREAMFREE_STAGE2_CHECKPOINT_TYPES = {DREAMFREE_STAGE2_CHECKPOINT_TYPE}


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    text = _resolve_optional_string(path_value)
    if text is None:
        return None
    return _resolve_path(text, base=base)


def _load_checkpoint_metadata(ckpt_path: Path) -> dict[str, Any] | None:
    if not ckpt_path.is_dir():
        return None
    trainer_state_path = ckpt_path / "trainer_state.json"
    if not trainer_state_path.exists():
        return None
    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in checkpoint trainer state: {trainer_state_path}")
    return payload


def _is_dreamfree_stage2_checkpoint(metadata: dict[str, Any] | None) -> bool:
    if metadata is None:
        return False
    return (
        str(metadata.get("checkpoint_type", "")).strip()
        in SUPPORTED_DREAMFREE_STAGE2_CHECKPOINT_TYPES
    )


def _policy_runtime_for_checkpoint_metadata(metadata: dict[str, Any] | None) -> str:
    checkpoint_type = None if metadata is None else str(metadata.get("checkpoint_type", "")).strip()
    if checkpoint_type in {DREAMFREE_STAGE2_CHECKPOINT_TYPE}:
        return DREAMFREE_POLICY_RUNTIME
    raise ValueError(
        f"Unsupported DreamFree Stage 2 checkpoint type for RoboCasa eval: {checkpoint_type or '<missing>'!r}; expected one of {sorted(SUPPORTED_DREAMFREE_STAGE2_CHECKPOINT_TYPES)}."
    )


def _source_checkpoint_for_results(metadata: dict[str, Any] | None) -> Any:
    if metadata is None:
        return None
    str(metadata.get("checkpoint_type", "")).strip()
    return metadata.get("source_stage1_checkpoint")


def _require_dreamfree_stage2_checkpoint(metadata: dict[str, Any] | None, *, label: str) -> None:
    if _is_dreamfree_stage2_checkpoint(metadata):
        return
    checkpoint_type = None if metadata is None else metadata.get("checkpoint_type")
    raise ValueError(
        f"{label} requires `ckpt` to be a DreamFree Stage 2 checkpoint ({sorted(SUPPORTED_DREAMFREE_STAGE2_CHECKPOINT_TYPES)!r}), got {checkpoint_type!r}."
    )


def _stats_length(stats: dict[str, Any], *, group: str, key: str, stat_name: str) -> int:
    try:
        value = stats[group][key][stat_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Dataset stats are missing `{group}.{key}.{stat_name}`.") from exc
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise ValueError(
            f"Dataset stats `{group}.{key}.{stat_name}` must be a vector, got {type(value).__name__}."
        )
    return len(value)


def _validate_dataset_stats_compatible(
    dataset_stats_path: Path, *, checkpoint_metadata: dict[str, Any] | None
) -> None:
    if checkpoint_metadata is None:
        return
    dataset = checkpoint_metadata.get("dataset", {})
    if not isinstance(dataset, dict):
        return
    expected_state_dim = dataset.get("proprio_dim")
    expected_action_dim = dataset.get("action_dim")
    if expected_state_dim is None and expected_action_dim is None:
        return
    try:
        stats = json.loads(dataset_stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset stats file is not valid JSON: {dataset_stats_path}") from exc
    if not isinstance(stats, dict):
        raise ValueError(f"Dataset stats file must contain a JSON object: {dataset_stats_path}")
    mismatches: list[str] = []
    if expected_state_dim is not None:
        actual = _stats_length(stats, group="state", key="default", stat_name="global_min")
        expected = int(expected_state_dim)
        if actual != expected:
            mismatches.append(
                f"state.default.global_min has {actual} dims, expected dataset.proprio_dim={expected}"
            )
    if expected_action_dim is not None:
        actual = _stats_length(stats, group="action", key="default", stat_name="global_min")
        expected = int(expected_action_dim)
        if actual != expected:
            mismatches.append(
                f"action.default.global_min has {actual} dims, expected dataset.action_dim={expected}"
            )
    if mismatches:
        raise ValueError(
            "Dataset stats are incompatible with DreamFree Stage 2 checkpoint: "
            + "; ".join(mismatches)
            + f". Path: {dataset_stats_path}"
        )


def _resolve_dataset_stats_path(
    *, explicit_value: Any, ckpt_path: Path, checkpoint_metadata: dict[str, Any] | None
) -> Path:
    explicit = _resolve_optional_path(explicit_value, base=PROJECT_ROOT)
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Explicit EVALUATION.dataset_stats_path not found: {explicit}")
        if not explicit.is_file():
            raise FileNotFoundError(
                f"Explicit EVALUATION.dataset_stats_path is not a file: {explicit}"
            )
        _validate_dataset_stats_compatible(explicit, checkpoint_metadata=checkpoint_metadata)
        return explicit
    if checkpoint_metadata is not None:
        metadata_stats = _resolve_optional_string(
            checkpoint_metadata.get("dataset", {}).get("pretrained_norm_stats")
        )
        if metadata_stats is not None:
            metadata_path = _resolve_path(metadata_stats, base=PROJECT_ROOT)
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"DreamFree Stage 2 checkpoint metadata references missing `dataset.pretrained_norm_stats`: {metadata_path}"
                )
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"DreamFree Stage 2 checkpoint metadata `dataset.pretrained_norm_stats` is not a file: {metadata_path}"
                )
            _validate_dataset_stats_compatible(
                metadata_path, checkpoint_metadata=checkpoint_metadata
            )
            return metadata_path
    candidates = [
        (parent / "dataset_stats.json").resolve() for parent in list(ckpt_path.parents)[:4]
    ]
    for path in candidates:
        if path.exists():
            if not path.is_file():
                raise FileNotFoundError(f"Dataset stats candidate is not a file: {path}")
            _validate_dataset_stats_compatible(path, checkpoint_metadata=checkpoint_metadata)
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Failed to locate dataset_stats.json for RoboCasa DreamFree eval. Tried EVALUATION.dataset_stats_path, valid checkpoint metadata, and checkpoint parent directories: {tried}. Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    if ckpt_path.is_dir() and (ckpt_path / "trainer_state.json").exists():
        if ckpt_path.parent.name != "checkpoints":
            raise ValueError(
                f"Checkpoint directory must follow .../<run_name>/checkpoints/<step_dir>, got: {ckpt_path}"
            )
        run_name = ckpt_path.parent.parent.name
        step_dir = ckpt_path.name
        if run_name == "" or step_dir == "":
            raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
        return f"{run_name}_{step_dir}"
    return ckpt_path.stem


def _resolve_run_ts(run_ts_value: Any) -> str:
    run_ts = _resolve_optional_string(run_ts_value)
    if run_ts is not None:
        return run_ts
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_robocasa_run_output_dir(
    *,
    output_root: Path,
    ckpt_tag: str,
    run_ts: str,
    output_namespace: str | None = "robocasa_dreamfree",
) -> Path:
    if output_namespace is not None:
        namespace_path = Path(output_namespace)
        if namespace_path.is_absolute() or len(namespace_path.parts) != 1:
            raise ValueError(
                f"RoboCasa output_namespace must be a single relative directory name or null, got {output_namespace!r}."
            )
        output_root = output_root / namespace_path
    return output_root / ckpt_tag / run_ts


def _resolve_cuda_visible_devices(*, gpu_id: Any, inherited_env: dict[str, str]) -> str:
    inherited = inherited_env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if inherited:
        return inherited
    return str(gpu_id)
