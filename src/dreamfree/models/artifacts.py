from __future__ import annotations

from pathlib import Path

from dreamfree.models.wan22.stage0_loader import resolve_stage0_lora_checkpoint


def _optional_path_identity(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _stage0_lora_checkpoint_identity(path_value: str | Path | None) -> str | None:
    text = _optional_path_identity(path_value)
    if text is None:
        return None
    return str(resolve_stage0_lora_checkpoint(text))


def _require_checkpoint_dir(path: str | Path | None, *, label: str) -> Path:
    if path is None or not str(path).strip():
        raise ValueError(f"{label} checkpoint path must be a non-empty directory path.")
    checkpoint_dir = Path(path).expanduser().resolve()
    if checkpoint_dir.is_file():
        raise ValueError(
            f"{label} checkpoint must point to a directory, got file: {checkpoint_dir}"
        )
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"{label} checkpoint directory not found: {checkpoint_dir}")
    return checkpoint_dir


def resolve_stage0_expectations(cfg):
    """Read source LoRA settings once instead of duplicating them in run configs."""
    import json

    checkpoint = Path(cfg.stage0_lora_checkpoint).expanduser().resolve()
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state.get("checkpoint_type") != "dreamfree_stage0" or state.get("checkpoint_version") != 1:
        raise ValueError(f"Expected a DreamFree stage 0 checkpoint: {checkpoint}")
    expected = {"global_step": state["global_step"]}
    expected.update({f"lora_{key}": value for (key, value) in state["lora_config"].items()})
    for key in ("global_step", "lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules"):
        field = f"stage0_expected_{key}"
        if getattr(cfg, field) is None:
            setattr(cfg, field, expected[key])
