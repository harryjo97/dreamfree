"""Lightweight wire contract for execution-owned action history."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

ACTION_HISTORY_POLICY_RUNTIME = "dreamfree_stage2_action_history"
ISSUED_ACTION_HISTORY_KEY = "deployment/issued_action_history"
ISSUED_ACTION_HISTORY_IS_PAD_KEY = "deployment/issued_action_history_is_pad"


def _coerce_issued_command(action: Any, *, action_dim: int) -> np.ndarray:
    command = np.asarray(action, dtype=np.float32).reshape(-1)
    if command.shape != (int(action_dim),) or not np.isfinite(command).all():
        raise ValueError(
            f"Issued command must contain {int(action_dim)} finite values, got {command.shape}."
        )
    return command


class IssuedActionHistoryBuffer:
    """Raw commands owned by the deployment execution boundary."""

    def __init__(self, *, horizon: int, action_dim: int) -> None:
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        if self.horizon <= 0 or self.action_dim <= 0:
            raise ValueError(
                f"Issued-action history requires positive horizon and action_dim, got horizon={self.horizon} action_dim={self.action_dim}."
            )
        self._commands: deque[np.ndarray] = deque(maxlen=self.horizon)

    def record_issued_action(self, action: Any) -> None:
        command = _coerce_issued_command(action, action_dim=self.action_dim)
        self._commands.append(command.copy())

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        history = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        is_pad = np.ones((self.horizon,), dtype=np.bool_)
        if self._commands:
            count = len(self._commands)
            history[-count:] = np.stack(tuple(self._commands), axis=0)
            is_pad[-count:] = False
        return (history, is_pad)

    def add_to_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        (history, is_pad) = self.snapshot()
        request = dict(observation)
        request[ISSUED_ACTION_HISTORY_KEY] = history
        request[ISSUED_ACTION_HISTORY_IS_PAD_KEY] = is_pad
        return request

    def clear(self) -> None:
        self._commands.clear()
