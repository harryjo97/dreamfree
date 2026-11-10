"""Optional ABC subtask conditioning, aligned to prepared 30 Hz frame indices."""

from __future__ import annotations

import json
import math
import random
import warnings
from bisect import bisect_right
from pathlib import Path

from dreamfree.data import yam

MANIFEST_NAME = "abc130k_subtasks.json"
PROMPT_FORMAT = "task_with_current_subtask_v1"


def combine_instruction(task: str, subtask: str) -> str:
    return f"{task.strip()}\nCurrent subtask: {subtask.strip()}"


def align_annotations(
    annotations: list[tuple[int, str]], source_start_ns: int, length: int
) -> list[list]:
    """Hold each label until the next timestamp; resolve ties as ABC's exporter does."""
    timestamps: dict[int, str] = {}
    for timestamp, text in annotations:
        text = " ".join(text.split())
        if timestamp in timestamps and timestamps[timestamp] != text:
            # ABC uses a stable timestamp sort and searchsorted(side="right"),
            # so the last message at a tied timestamp supplies the active label.
            warnings.warn(
                f"Conflicting ABC subtask labels at timestamp {timestamp}; "
                f"keeping the last label, matching ABC's exporter: {text!r}",
                RuntimeWarning,
                stacklevel=2,
            )
        timestamps[timestamp] = text
    frames: dict[int, str] = {}
    for timestamp, text in sorted(timestamps.items()):
        # Prepared ticks are start + floor(frame * 1e9 / 30). Use integer math
        # to preserve nanosecond boundaries at epoch-sized timestamps.
        frame = max(0, ((timestamp - source_start_ns) * 30 + 999_999_999) // 1_000_000_000)
        if frame < length:
            frames[frame] = text
    segments = []
    previous = ""
    for frame, text in sorted(frames.items()):
        if text != previous:
            segments.append([frame, text])
            previous = text
    return segments


class SubtaskAnnotations:
    def __init__(self, manifest: dict, selection: dict):
        unsigned = {key: value for key, value in manifest.items() if key != "annotations_sha256"}
        if (
            manifest.get("schema_version") != 1
            or manifest.get("prompt_format") != PROMPT_FORMAT
            or manifest.get("annotations_sha256") != yam.fingerprint(unsigned)
        ):
            raise ValueError("Invalid ABC subtask manifest schema, prompt format, or fingerprint.")
        if manifest.get("selection_sha256") != selection["selection_sha256"]:
            raise ValueError(
                "ABC subtask manifest belongs to a different prepared dataset selection."
            )
        self.sha256 = manifest["annotations_sha256"]
        self.episodes = selection["episodes"]
        records = manifest.get("episodes", {})
        if set(records) != set(self.episodes):
            raise ValueError(
                "ABC subtask manifest must cover every selected episode, including unannotated ones."
            )
        self.segments = {}
        self.starts = {}
        for key, record in records.items():
            length = self.episodes[key]["length"]
            if record.get("length") != length:
                raise ValueError(f"ABC subtask episode {key}: prepared length mismatch.")
            segments = record["segments"]
            previous = -1
            for frame, text in segments:
                if (
                    type(frame) is not int
                    or not previous < frame < length
                    or not isinstance(text, str)
                ):
                    raise ValueError(
                        f"ABC subtask episode {key}: invalid segment {[frame, text]!r}."
                    )
                previous = frame
            self.segments[key] = segments
            self.starts[key] = [frame for frame, _ in segments]

    def task_candidates(self, episode_index: int, task: str) -> tuple[str, ...]:
        key = str(episode_index)
        if task.strip() != self.episodes[key]["instruction"].strip():
            raise ValueError(
                f"ABC subtask episode {key}: cached global instruction differs from selection."
            )
        return tuple(
            dict.fromkeys(
                [task, *[combine_instruction(task, text) for _, text in self.segments[key] if text]]
            )
        )

    def sample_task(self, episode_index: int, task: str, frame: int, probability: float) -> str:
        key = str(episode_index)
        if not 0 <= frame < self.episodes[key]["length"]:
            raise ValueError(
                f"ABC subtask episode {key}: frame {frame} is outside the prepared episode."
            )
        segment = bisect_right(self.starts[key], frame) - 1
        text = self.segments[key][segment][1] if segment >= 0 else ""
        if text and probability > 0 and random.random() < probability:
            return combine_instruction(task, text)
        return task


def load_annotations(root: str | Path, dataset_name: str = yam.ABC_NAME) -> SubtaskAnnotations:
    path = Path(root).expanduser().resolve() / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Subtask annotations enabled but {path} is missing. "
            "Run scripts/data/prepare_abc130k_subtasks.py first."
        )
    selection = yam.load_selection(path.parent, dataset_name)
    # This runs only at setup/checkpoint time. Read the contents each time:
    # shared filesystems can retain the same mtime across consecutive writes.
    return SubtaskAnnotations(json.loads(path.read_text()), selection)


def validate_config(cfg) -> None:
    probability = float(getattr(cfg, "subtask_annotation_probability", 0.2))
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("subtask_annotation_probability must be finite and in [0, 1].")
    if not getattr(cfg, "use_subtask_annotations", False):
        return
    if cfg.dataset_name not in {yam.ABC_NAME, yam.ABC_EEF_NAME}:
        raise ValueError(
            "use_subtask_annotations supports only dataset_name=yam_abc130k or yam_abc130k_eef."
        )
    if not cfg.use_cached_latents:
        raise ValueError("ABC subtask conditioning requires the prepared cached-latent dataset.")
    if getattr(cfg, "dataset_format", "stage0_cached_latents") != "stage0_cached_latents":
        raise ValueError("ABC subtask conditioning requires dataset_format=stage0_cached_latents.")
    if not getattr(cfg, "strict_dataset_errors", True):
        raise ValueError(
            "Subtask conditioning requires strict_dataset_errors=true; missing text caches must fail."
        )


def from_config(cfg, *, split: str) -> SubtaskAnnotations | None:
    validate_config(cfg)
    if not getattr(cfg, "use_subtask_annotations", False):
        return None
    root = getattr(cfg, "dataset_root", None) or cfg.dataset_dirs
    annotations = load_annotations(root, cfg.dataset_name)
    previous = getattr(cfg, "_subtask_annotations_sha256", annotations.sha256)
    if previous != annotations.sha256:
        raise ValueError(
            "ABC subtask manifest changed after this run was configured; use a new run."
        )
    cfg._subtask_annotations_sha256 = annotations.sha256
    # Validation measures the task-only inference interface, with no RNG draw.
    return annotations if split == "train" else None


def conditioning_metadata(cfg) -> dict | None:
    annotations = from_config(cfg, split="train")
    if annotations is None:
        return None
    return {
        "annotations_sha256": annotations.sha256,
        "prompt_format": PROMPT_FORMAT,
        "probability": float(cfg.subtask_annotation_probability),
        "frame_policy": "window_start",
        "validation_policy": "task_only",
    }
