"""Strict parsing of shared policy inference options."""

from __future__ import annotations

from typing import Any

import dreamfree.models.policy_loading as dreamfree_deploy

ACTION_SAMPLER_EULER = "euler"
ACTION_SAMPLER_UNIPC = "unipc"
SUPPORTED_ACTION_SAMPLERS = {ACTION_SAMPLER_EULER, ACTION_SAMPLER_UNIPC}


def is_none_like(value: Any) -> bool:
    return dreamfree_deploy._is_none_like(value)


def parse_int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            return []
        items = [item.strip() for item in text.split(",")]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []
    return [int(item) for item in items]


def normalize_action_sampler(value: Any) -> str:
    sampler = str(value or ACTION_SAMPLER_EULER).strip().lower()
    if sampler not in SUPPORTED_ACTION_SAMPLERS:
        raise ValueError(
            f"Unsupported action_sampler={value!r}. Expected one of {sorted(SUPPORTED_ACTION_SAMPLERS)}."
        )
    return sampler


def parse_optional_float(value: Any) -> float | None:
    if is_none_like(value):
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"Optional float setting must be positive when set, got {value!r}.")
    return parsed


def parse_nonnegative_int_list(value: Any, *, label: str) -> list[int]:
    parsed = parse_int_list(value)
    bad = [item for item in parsed if item < 0]
    if bad:
        raise ValueError(f"{label} must contain non-negative step indices, got {parsed}.")
    return parsed


def parse_bool(value: Any) -> bool:
    """Parse an explicit policy boolean without Python string truthiness.

    Args:
        value: Boolean or supported true/false string.

    Raises:
        ValueError: If the supplied value is not a supported boolean spelling.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def parse_mode(value: Any, name: str) -> str:
    """Normalize an inference feature's auto/true/false selection.

    Args:
        value: Boolean or one of the literal strings auto, true, false.
        name: Option name included in invalid-value errors.
    """
    if isinstance(value, bool):
        value = str(value).lower()
    if value not in ("auto", "true", "false"):
        raise ValueError(f"{name} must be auto, true, or false, got {value!r}.")
    return value
