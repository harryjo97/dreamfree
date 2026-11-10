"""Lightweight contracts shared by the DreamFree and ABC Python environments."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ABC_REVISION = "d0832d12651d1b260a652861a14648dc5f3660c7"
DEFAULT_SUITE = Path(__file__).with_name("tasks.json")
PROMPT_MANIFEST = Path(__file__).with_name("task_prompts.json")
FIXED_COUNTS = {
    f"count_{word}_into_opaque_box": count
    for count, word in enumerate(("one", "two", "three", "four"), 1)
}
CAMERAS = ("top", "left", "right")
REQUIRED_FILES = (
    "trainer_state.json",
    "action.safetensors",
    "readout.safetensors",
    "conditioner.safetensors",
    "proprio_encoder.safetensors",
)
PROTOCOL = "dreamfree_yam_standard_v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.stem + ".pending.json")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_checkpoint(path):
    """Resolve a step, run, or checkpoints directory; trainer_state is written last."""
    path = Path(path).expanduser().resolve()
    if (path / "trainer_state.json").is_file() or path.name.startswith("step_"):
        candidates = [path]
    else:
        checkpoint_root = path / "checkpoints" if (path / "checkpoints").is_dir() else path
        candidates = sorted(
            (
                p
                for p in checkpoint_root.glob("step_*")
                if p.is_dir() and re.fullmatch(r"step_\d+", p.name)
            ),
            key=lambda p: int(p.name.removeprefix("step_")),
            reverse=True,
        )
    for candidate in candidates:
        if all(
            (candidate / name).is_file() and (candidate / name).stat().st_size > 0
            for name in REQUIRED_FILES
        ):
            return candidate
    raise FileNotFoundError(
        f"No completed Stage 2 checkpoint at {path}. Pass a run directory, its checkpoints directory, "
        "or an explicit step_* directory containing trainer_state.json and all four model safetensors files. "
        "Wait for training to save a checkpoint."
    )


def load_suite(path=DEFAULT_SUITE, names=None):
    suite = read_json(path)
    if not isinstance(suite, dict) or not suite:
        raise ValueError("Task suite must be a nonempty JSON object keyed by ABC task name.")
    selected = list(suite) if not names else names
    if len(set(selected)) != len(selected):
        raise ValueError("Repeated task names are not allowed.")
    result = {}
    for name in selected:
        if name not in suite:
            raise ValueError(f"Task {name!r} is absent from {path}; available: {list(suite)}")
        entry = suite[name]
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("instruction"), str)
            or not entry["instruction"].strip()
        ):
            raise ValueError(f"Task {name}: a nonempty instruction is required.")
        if not isinstance(entry.get("max_steps"), int) or entry["max_steps"] <= 0:
            raise ValueError(f"Task {name}: max_steps must be a positive integer.")
        result[name] = dict(entry)
    return result


@lru_cache(maxsize=1)
def dynamic_task_prompts():
    manifest = read_json(PROMPT_MANIFEST)
    if manifest["abc_revision"] != ABC_REVISION:
        raise ValueError(
            "YAM prompt manifest does not match the pinned ABC revision; regenerate it with task_prompts.py."
        )
    return manifest["tasks"]


def suite_instructions(suite):
    """Include every randomized/runtime instruction, not only the initial prompts."""
    dynamic = dynamic_task_prompts()
    return sorted(
        {
            prompt
            for name, entry in suite.items()
            for prompt in [entry["instruction"], *dynamic.get(name, [])]
        }
    )


def validate_observation(observation):
    state = np.asarray(observation.get("state"), dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"ABC YAM state must be 14 finite values, got {state.shape}.")
    images = observation.get("images", {})
    for name in CAMERAS:
        image = images.get(name)
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.shape != (3, 480, 640)
        ):
            raise ValueError(f"ABC camera {name!r} must be uint8 CHW (3,480,640).")
        if not observation.get("masks", {}).get(name, False):
            raise ValueError(f"ABC camera {name!r} is unavailable.")
    if not isinstance(observation.get("prompt"), str) or not observation["prompt"].strip():
        raise ValueError("ABC observation must include a nonempty task prompt.")
    return state


def executable_chunk(actions, horizon):
    """Keep absolute joints continuous and clamp only grippers to physical apertures."""
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.shape != (horizon, 14) or not np.isfinite(chunk).all():
        raise ValueError(f"Policy actions must be finite ({horizon},14), got {chunk.shape}.")
    chunk = chunk.copy()
    grip = chunk[:, [6, 13]]
    clipped = int(np.count_nonzero((grip < 0) | (grip > 1)))
    chunk[:, [6, 13]] = np.clip(grip, 0, 1)
    return chunk, clipped
