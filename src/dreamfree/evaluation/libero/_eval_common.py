"""Shared helpers for LIBERO DreamFree Stage 2 evaluation."""

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
LIBERO_ACTION_DATA_SOURCE = "libero"
VALID_SINGLE_ACTION_DATA_SOURCES = {LIBERO_ACTION_DATA_SOURCE}
LIBERO_DATASET_STATS_PATH = PROJECT_ROOT / "data/libero_dataset_stats.json"
_LIBERO_ACTION_DATASET_NAMES = {"libero"}
_DREAMFREE_STAT_NAMES = ("min", "max", "mean", "std", "q01", "q99")


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


def _normalize_libero_dataset_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_")


def _infer_action_data_source_from_dataset_name(dataset_name: Any) -> str:
    normalized = _normalize_libero_dataset_name(dataset_name)
    if normalized in _LIBERO_ACTION_DATASET_NAMES:
        return LIBERO_ACTION_DATA_SOURCE
    raise ValueError(
        f"Unsupported LIBERO eval dataset.name for action normalization: {dataset_name!r}. Expected the LIBERO dataset."
    )


def _infer_action_data_source_from_checkpoint_metadata(
    checkpoint_metadata: dict[str, Any] | None,
) -> str:
    if checkpoint_metadata is None:
        raise ValueError(
            "Cannot infer LIBERO action normalization source without checkpoint metadata."
        )
    dataset = checkpoint_metadata.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError(
            "Cannot infer LIBERO action normalization source: checkpoint metadata.dataset is missing."
        )
    return _infer_action_data_source_from_dataset_name(dataset.get("name"))


def _normalize_action_data_source(value: Any) -> str:
    source = str(value).strip().lower().replace("-", "_")
    if source not in VALID_SINGLE_ACTION_DATA_SOURCES:
        raise ValueError(
            f"Unsupported LIBERO action_data_source={value!r}; expected one of {sorted(VALID_SINGLE_ACTION_DATA_SOURCES)}."
        )
    return source


def _normalize_image_preprocess_source(value: Any) -> str:
    source = str(value).strip().lower().replace("-", "_")
    if source not in VALID_SINGLE_ACTION_DATA_SOURCES:
        raise ValueError(
            f"Unsupported LIBERO image_preprocess_source={value!r}; expected one of {sorted(VALID_SINGLE_ACTION_DATA_SOURCES)} or 'auto'."
        )
    return source


def _require_single_action_data_source(checkpoint_metadata: dict[str, Any] | None) -> str:
    source = _infer_action_data_source_from_checkpoint_metadata(checkpoint_metadata)
    return _normalize_action_data_source(source)


def _default_dataset_stats_path_for_action_source(action_data_source: str) -> Path:
    _normalize_action_data_source(action_data_source)
    return LIBERO_DATASET_STATS_PATH


def _as_float_vector(value: Any, *, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise ValueError(f"Dataset stats `{label}` must be a vector.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Dataset stats `{label}` must be a numeric vector.") from exc


def _convert_lerobot_libero_stats(stats: dict[str, Any]) -> dict[str, Any]:

    def convert_group(source_key: str) -> dict[str, Any]:
        if source_key not in stats or not isinstance(stats[source_key], dict):
            raise ValueError(f"LeRobot stats are missing `{source_key}`.")
        source = stats[source_key]
        converted: dict[str, Any] = {}
        for stat_name in _DREAMFREE_STAT_NAMES:
            vector = _as_float_vector(source.get(stat_name), label=f"{source_key}.{stat_name}")
            converted[f"global_{stat_name}"] = vector
            converted[f"stepwise_{stat_name}"] = [list(vector)]
        return converted

    return {
        "action": {"default": convert_group("actions")},
        "state": {"default": convert_group("state")},
    }


def _normalize_dataset_stats_payload(stats: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(stats, dict):
        raise ValueError("Dataset stats payload must be a JSON object.")
    if isinstance(stats.get("action"), dict) and isinstance(stats.get("state"), dict):
        return stats
    if isinstance(stats.get("actions"), dict) and isinstance(stats.get("state"), dict):
        return _convert_lerobot_libero_stats(stats)
    raise ValueError(
        "Dataset stats must be either DreamFree format with `action`/`state` or LeRobot format with `actions`/`state`."
    )


def _load_dataset_stats_json_for_eval(dataset_stats_path: Path | str) -> dict[str, Any]:
    path = Path(dataset_stats_path)
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset stats file is not valid JSON: {path}") from exc
    return _normalize_dataset_stats_payload(stats)


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


def _stats_float(
    stats: dict[str, Any], *, group: str, key: str, stat_name: str, index: int
) -> float:
    try:
        return float(stats[group][key][stat_name][index])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Dataset stats are missing numeric `{group}.{key}.{stat_name}[{index}]`."
        ) from exc


def _validate_action_stats_semantics(
    stats: dict[str, Any], *, action_data_source: str, dataset_stats_path: Path
) -> None:
    source = _normalize_action_data_source(action_data_source)
    gripper_min = _stats_float(
        stats, group="action", key="default", stat_name="global_min", index=-1
    )
    gripper_max = _stats_float(
        stats, group="action", key="default", stat_name="global_max", index=-1
    )
    tol = 0.0001
    if source == LIBERO_ACTION_DATA_SOURCE:
        (expected_min, expected_max) = (0.0, 1.0)
    else:
        (expected_min, expected_max) = (-1.0, 1.0)
    if abs(gripper_min - expected_min) > tol or abs(gripper_max - expected_max) > tol:
        raise ValueError(
            f"Dataset stats gripper range does not match the checkpoint action data source: source={source} stats_gripper_min={gripper_min} stats_gripper_max={gripper_max} expected={expected_min}/{expected_max}. Path: {dataset_stats_path}"
        )


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
    stats = _load_dataset_stats_json_for_eval(dataset_stats_path)
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
        details = "; ".join(mismatches)
        raise ValueError(
            f"Dataset stats are incompatible with DreamFree Stage 2 checkpoint: {details}. Path: {dataset_stats_path}"
        )
    action_data_source = _require_single_action_data_source(checkpoint_metadata)
    _validate_action_stats_semantics(
        stats, action_data_source=action_data_source, dataset_stats_path=dataset_stats_path
    )


def _load_checkpoint_metadata(ckpt_path: Path) -> dict[str, Any] | None:
    if not ckpt_path.is_dir():
        return None
    from dreamfree.models.release import (
        is_release_checkpoint,
        read_checkpoint_metadata,
        resolve_policy_checkpoint,
    )

    ckpt_path = resolve_policy_checkpoint(ckpt_path)
    if is_release_checkpoint(ckpt_path):
        return read_checkpoint_metadata(ckpt_path, stage="stage2")
    trainer_state_path = ckpt_path / "trainer_state.json"
    if not trainer_state_path.exists():
        return None
    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in checkpoint trainer state: {trainer_state_path}")
    from dreamfree.config_names import normalize_checkpoint_metadata

    return normalize_checkpoint_metadata(payload)


def _is_dreamfree_stage2_checkpoint(metadata: dict[str, Any] | None) -> bool:
    if metadata is None:
        return False
    return (
        str(metadata.get("checkpoint_type", "")).strip()
        in SUPPORTED_DREAMFREE_STAGE2_CHECKPOINT_TYPES
    )


def _policy_runtime_for_checkpoint_metadata(metadata: dict[str, Any] | None) -> str:
    checkpoint_type = None if metadata is None else str(metadata.get("checkpoint_type", "")).strip()
    if checkpoint_type == DREAMFREE_STAGE2_CHECKPOINT_TYPE:
        return DREAMFREE_POLICY_RUNTIME
    raise ValueError(
        f"Unsupported DreamFree Stage 2 checkpoint type for LIBERO eval: {checkpoint_type or '<missing>'!r}; expected one of {sorted(SUPPORTED_DREAMFREE_STAGE2_CHECKPOINT_TYPES)}."
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
        f"{label} requires `checkpoint_path` to be a DreamFree Stage 2 checkpoint ({sorted(SUPPORTED_DREAMFREE_STAGE2_CHECKPOINT_TYPES)!r}), got {checkpoint_type!r}."
    )


def _resolve_dataset_stats_path(
    *, explicit_value: Any, ckpt_path: Path, checkpoint_metadata: dict[str, Any] | None
) -> Path:
    action_data_source = _require_single_action_data_source(checkpoint_metadata)
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
            checkpoint_metadata.get("dataset", {}).get("dataset_stats_path")
        )
        if metadata_stats is not None:
            metadata_path = _resolve_path(metadata_stats, base=PROJECT_ROOT)
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"DreamFree Stage 2 checkpoint metadata references missing `dataset.dataset_stats_path`: {metadata_path}"
                )
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"DreamFree Stage 2 checkpoint metadata `dataset.dataset_stats_path` is not a file: {metadata_path}"
                )
            _validate_dataset_stats_compatible(
                metadata_path, checkpoint_metadata=checkpoint_metadata
            )
            return metadata_path
    default_path = _default_dataset_stats_path_for_action_source(action_data_source)
    if default_path.exists():
        if not default_path.is_file():
            raise FileNotFoundError(
                f"Default LIBERO dataset stats path is not a file: {default_path}"
            )
        _validate_dataset_stats_compatible(default_path, checkpoint_metadata=checkpoint_metadata)
        return default_path
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
        f"Failed to locate dataset_stats.json for LIBERO DreamFree eval. Tried EVALUATION.dataset_stats_path, valid checkpoint metadata, default action-source stats {default_path}, and checkpoint parent directories: {tried}. Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
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
    if isinstance(run_ts_value, (int, float)) and (not isinstance(run_ts_value, bool)):
        raise ValueError(
            'EVALUATION.run_ts was parsed as a numeric value, which can change directory names such as 20260526_030339 into 20260526030339. Pass it as a quoted Hydra string, for example: +EVALUATION.run_ts="20260526_030339".'
        )
    run_ts = _resolve_optional_string(run_ts_value)
    if run_ts is not None:
        return run_ts
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_libero_run_output_dir(*, output_root: Path, ckpt_tag: str, run_ts: str) -> Path:
    output_subdir = _resolve_optional_string(
        os.environ.get("LIBERO_DREAMFREE_OUTPUT_SUBDIR", "libero_dreamfree")
    )
    if output_subdir is None:
        return output_root / ckpt_tag / run_ts
    if Path(output_subdir).name != output_subdir or output_subdir in {".", ".."}:
        raise ValueError(
            f"LIBERO_DREAMFREE_OUTPUT_SUBDIR must be a single directory name, got: {output_subdir!r}"
        )
    return output_root / output_subdir / ckpt_tag / run_ts


def _resolve_cuda_visible_devices(*, gpu_id: Any, inherited_env: dict[str, str]) -> str:
    inherited = inherited_env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if inherited:
        return inherited
    return str(gpu_id)
