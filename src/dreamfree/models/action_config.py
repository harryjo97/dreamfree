from __future__ import annotations

from typing import Any


def _parse_optional_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "null", "None", "none"}:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            return None
        items = [item.strip() for item in text.split(",")]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(f"Expected an integer list or comma-separated string, got {value!r}.")
    parsed: list[int] = []
    for item in items:
        if item in {"", None}:
            raise ValueError(f"Invalid empty entry in integer list {value!r}.")
        parsed.append(int(item))
    return parsed


def _require_dict(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict, got {type(value)}.")
    return value


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if text in {"", "null", "None", "none"}:
        return None
    return int(text)
