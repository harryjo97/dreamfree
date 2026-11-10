"""Run sim-evals DROID Isaac Sim rollouts against a local DreamFree policy server."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dreamfree.evaluation.action_protocol import (
    IssuedActionHistoryBuffer,
)
from dreamfree.evaluation.droid_real.client import (
    DROID_GRIPPER_NORMALIZED_COMMAND_SCALE,
    canonicalize_droid_gripper_command,
)
from dreamfree.evaluation.droid_sim.policy_rpc import PolicyRpcClient

PROJECT_ROOT = Path.cwd()
RUBIKS_CUBE_HALF_EXTENTS_M = (0.0288, 0.0288, 0.0288)
SCENE_CONFIGS = {
    1: {
        "asset_scene": 1,
        "instruction": "put the cube in the bowl",
        "target": "rubiks_cube",
        "receptacle": "_24_bowl",
        "task_type": "pick_place",
        "pick_lift_threshold_m": 0.01,
        "target_half_extents_m": RUBIKS_CUBE_HALF_EXTENTS_M,
        "place_detector": {
            "horizontal_threshold_m": 0.05,
            "containment_half_extents_xy_m": None,
            "vertical_offset_range_m": (0.0, 0.1),
            "min_support_normal_z": 0.25,
            "min_contact_force_n": None,
            "max_receptacle_translation_m": 0.02,
            "max_receptacle_rotation_rad": 0.35,
        },
    },
    2: {
        "asset_scene": 2,
        "instruction": "pick up the can and put it in the mug",
        "target": "_10_potted_meat_can",
        "receptacle": "_25_mug",
        "task_type": "pick_place",
        "pick_lift_threshold_m": None,
        "place_detector": None,
    },
    3: {
        "asset_scene": 3,
        "instruction": "put the banana in the bin",
        "target": "_11_banana",
        "receptacle": "small_KLT_visual_collision",
        "task_type": "pick_place",
        "pick_lift_threshold_m": 0.01,
        "place_detector": {
            "horizontal_threshold_m": None,
            "containment_half_extents_xy_m": (0.091, 0.128),
            "vertical_offset_range_m": (-0.069, 0.079),
            "min_support_normal_z": 0.25,
            "min_contact_force_n": 0.001,
            "max_receptacle_translation_m": 0.05,
            "max_receptacle_rotation_rad": math.radians(15.0),
        },
    },
    4: {
        "asset_scene": 1,
        "instruction": "pick up the cube",
        "target": "rubiks_cube",
        "receptacle": "_24_bowl",
        "task_type": "pick",
        "pick_lift_threshold_m": 0.01,
        "target_half_extents_m": RUBIKS_CUBE_HALF_EXTENTS_M,
        "place_detector": None,
    },
}
GRIPPER_CONTACT_BODIES = (
    ("left", "left_outer_finger"),
    ("left", "left_inner_finger"),
    ("right", "right_outer_finger"),
    ("right", "right_inner_finger"),
)
DIAGNOSTICS_SCHEMA_VERSION = 1
CONTACT_SCOPE = "target_gripper"
DROID_GRID_HEIGHT = 352
DROID_GRID_WIDTH = 640
DROID_PER_CAMERA_HEIGHT = DROID_GRID_HEIGHT // 2
DROID_PER_CAMERA_WIDTH = DROID_GRID_WIDTH // 2
DROID_CENTER_CROP_SCALE = 0.95


def _configure_scene_cameras(env_cfg: Any, scene_config: dict[str, Any]) -> None:
    calibration = scene_config.get("camera_calibration")
    if calibration is None:
        return
    cameras = calibration.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"Camera calibration must define at least one camera: {calibration!r}.")
    for camera_name, values in cameras.items():
        if not isinstance(values, dict):
            raise ValueError(f"Camera calibration for {camera_name!r} must be a mapping.")
        camera_cfg = getattr(env_cfg.scene, camera_name, None)
        if camera_cfg is None:
            raise ValueError(f"Camera calibration references unknown camera {camera_name!r}.")
        pos = np.asarray(values.get("pos"), dtype=np.float64).reshape(-1)
        rot = np.asarray(values.get("rot"), dtype=np.float64).reshape(-1)
        vertical_fov_deg = float(values.get("vertical_fov_deg"))
        if tuple(pos.shape) != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError(f"Camera {camera_name!r} position must be finite [3], got {pos!r}.")
        if tuple(rot.shape) != (4,) or not np.all(np.isfinite(rot)):
            raise ValueError(f"Camera {camera_name!r} rotation must be finite [4], got {rot!r}.")
        if not np.isclose(np.linalg.norm(rot), 1.0, atol=1e-05):
            raise ValueError(
                f"Camera {camera_name!r} rotation must be a unit quaternion, got {rot!r}."
            )
        if not np.isfinite(vertical_fov_deg) or not 0.0 < vertical_fov_deg < 180.0:
            raise ValueError(
                f"Camera {camera_name!r} vertical FOV must be in (0, 180), got {vertical_fov_deg!r}."
            )
        vertical_aperture = float(camera_cfg.spawn.vertical_aperture)
        if not np.isfinite(vertical_aperture) or vertical_aperture <= 0.0:
            raise ValueError(
                f"Camera {camera_name!r} vertical aperture must be positive and finite, got {vertical_aperture!r}."
            )
        focal_length = vertical_aperture / (2.0 * math.tan(math.radians(vertical_fov_deg) / 2.0))
        camera_cfg.offset.pos = tuple(float(value) for value in pos)
        camera_cfg.offset.rot = tuple(float(value) for value in rot)
        camera_cfg.spawn.focal_length = focal_length


def _atomic_write_json(path: Path, payload: dict[str, Any], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if compact:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    partial_path.write_text(encoded, encoding="utf-8")
    os.replace(partial_path, path)


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _single_value(value: Any, *, name: str) -> Any:
    array = _numpy(value).reshape(-1)
    if int(array.size) != 1:
        raise ValueError(f"{name} must contain one value, got shape={array.shape}.")
    return array[0]


def _validate_contact_buffer_capacity(contact_count: int, capacity: int) -> None:
    if contact_count >= capacity:
        raise RuntimeError(
            f"DROID target contact data reached MAX_CONTACTS_PER_STEP and may be truncated: count={contact_count} capacity={capacity}. Increase MAX_CONTACTS_PER_STEP."
        )


def _summarize_contacts(contact_records: list[dict[str, Any]]) -> dict[str, Any]:
    count_by_finger = {"left": 0, "right": 0}
    force_by_finger = {"left": 0.0, "right": 0.0}
    for contact in contact_records:
        finger = str(contact["finger"])
        if finger not in count_by_finger:
            raise ValueError(f"Unknown contact finger {finger!r}.")
        force_n = float(contact["normal_force_n"])
        count_by_finger[finger] += 1
        force_by_finger[finger] = max(force_by_finger[finger], force_n)
    left_touch = count_by_finger["left"] > 0
    right_touch = count_by_finger["right"] > 0
    return {
        "contact_count": len(contact_records),
        "count_by_finger": count_by_finger,
        "force_by_finger_n": force_by_finger,
        "left_touch": left_touch,
        "right_touch": right_touch,
        "bilateral_touch": left_touch and right_touch,
        "contacts": contact_records,
    }


class ContactMetricAccumulator:
    """Accumulate MolmoSpaces-style target touch metrics across one episode."""

    def __init__(self) -> None:
        self.pickup_obj_touched = False
        self.first_touch_step: int | None = None
        self.first_left_touch_step: int | None = None
        self.first_right_touch_step: int | None = None
        self.first_bilateral_touch_step: int | None = None
        self.touch_step_count = 0
        self.touch_contact_count = 0
        self.max_touch_force = 0.0
        self.touch_fingers: set[str] = set()
        self.target_contact_count_by_category = {"left_finger": 0, "right_finger": 0}
        self.bilateral_touch_step_count = 0
        self.longest_bilateral_touch_streak = 0
        self._bilateral_touch_streak = 0

    def update(self, episode_step: int, contact_records: list[dict[str, Any]]) -> dict[str, Any]:
        contacts = _summarize_contacts(contact_records)
        if contacts["contact_count"] > 0:
            self.pickup_obj_touched = True
            if self.first_touch_step is None:
                self.first_touch_step = int(episode_step)
            self.touch_step_count += 1
            self.touch_contact_count += int(contacts["contact_count"])
            self.max_touch_force = max(
                self.max_touch_force,
                max(float(contact["normal_force_n"]) for contact in contact_records),
            )
        if contacts["left_touch"]:
            self.touch_fingers.add("left")
            if self.first_left_touch_step is None:
                self.first_left_touch_step = int(episode_step)
        if contacts["right_touch"]:
            self.touch_fingers.add("right")
            if self.first_right_touch_step is None:
                self.first_right_touch_step = int(episode_step)
        self.target_contact_count_by_category["left_finger"] += int(
            contacts["count_by_finger"]["left"]
        )
        self.target_contact_count_by_category["right_finger"] += int(
            contacts["count_by_finger"]["right"]
        )
        if contacts["bilateral_touch"]:
            if self.first_bilateral_touch_step is None:
                self.first_bilateral_touch_step = int(episode_step)
            self.bilateral_touch_step_count += 1
            self._bilateral_touch_streak += 1
            self.longest_bilateral_touch_streak = max(
                self.longest_bilateral_touch_streak, self._bilateral_touch_streak
            )
        else:
            self._bilateral_touch_streak = 0
        return contacts

    def as_dict(self) -> dict[str, Any]:
        return {
            "pickup_obj_touched": self.pickup_obj_touched,
            "first_touch_step": self.first_touch_step,
            "first_left_touch_step": self.first_left_touch_step,
            "first_right_touch_step": self.first_right_touch_step,
            "first_bilateral_touch_step": self.first_bilateral_touch_step,
            "touch_step_count": self.touch_step_count,
            "touch_contact_count": self.touch_contact_count,
            "max_touch_force": self.max_touch_force,
            "touch_fingers": sorted(self.touch_fingers),
            "target_contact_count_by_category": dict(self.target_contact_count_by_category),
            "bilateral_touch_step_count": self.bilateral_touch_step_count,
            "longest_bilateral_touch_streak": self.longest_bilateral_touch_streak,
        }


def _box_support_half_height_m(quat_wxyz: Any, half_extents_m: tuple[float, float, float]) -> float:
    """Distance from a box's centre to its lowest point, for the box's current orientation.

    This is the box support function evaluated along world -Z. Expressing world +Z in the
    box frame gives the world-vertical component of each body axis, so the vertical
    half-height is sum_i |z_body[i]| * half_extent[i]. It equals half_extent for an
    axis-aligned box and grows to sqrt(2)/sqrt(3) times that when tipped onto an edge/corner.
    """
    z_axis_in_body = _quaternion_rotate_inverse(quat_wxyz, (0.0, 0.0, 1.0))
    return float(
        sum(
            abs(float(component)) * float(extent)
            for (component, extent) in zip(z_axis_in_body, half_extents_m, strict=True)
        )
    )


class PickLiftMetricAccumulator:
    """Track pick success from target displacement relative to the reset state."""

    def __init__(
        self, threshold_m: float, *, half_extents_m: tuple[float, float, float] | None = None
    ) -> None:
        self.threshold_m = float(threshold_m)
        if not np.isfinite(self.threshold_m) or self.threshold_m <= 0.0:
            raise ValueError(
                f"Pick lift threshold must be a positive finite value, got {threshold_m!r}."
            )
        self.half_extents_m: tuple[float, float, float] | None = None
        if half_extents_m is not None:
            extents = [float(value) for value in half_extents_m]
            if len(extents) != 3:
                raise ValueError(
                    f"Pick lift target half extents must be an (x, y, z) triple, got {half_extents_m!r}."
                )
            if not all(np.isfinite(value) and value > 0.0 for value in extents):
                raise ValueError(
                    f"Pick lift target half extents must be positive and finite, got {half_extents_m!r}."
                )
            self.half_extents_m = (extents[0], extents[1], extents[2])
        self.initial_target_height_m: float | None = None
        self.baseline_target_height_m: float | None = None
        self.baseline_target_bottom_m: float | None = None
        self.max_lift_height_m = 0.0
        self.final_lift_height_m = 0.0
        self.max_support_clearance_m = 0.0
        self.final_support_clearance_m = 0.0
        self.pick_success = False
        self.first_pick_step: int | None = None
        self.pick_hold_step_count = 0
        self.longest_pick_hold_streak = 0
        self._pick_hold_streak = 0
        self.dropped_after_pick = False
        self.first_drop_step: int | None = None

    def _target_bottom_m(self, target_height_m: float, target_quat_wxyz: Any) -> float:
        if self.half_extents_m is None:
            return target_height_m
        if target_quat_wxyz is None:
            raise ValueError(
                "Pick lift metrics need the target orientation to locate its lowest point when box half extents are configured."
            )
        return target_height_m - _box_support_half_height_m(target_quat_wxyz, self.half_extents_m)

    def begin_episode(self, target_height_m: float, target_quat_wxyz: Any = None) -> None:
        target_height_m = float(target_height_m)
        if not np.isfinite(target_height_m):
            raise ValueError(f"Initial target height must be finite, got {target_height_m!r}.")
        self.initial_target_height_m = target_height_m
        self.baseline_target_height_m = target_height_m
        self.baseline_target_bottom_m = self._target_bottom_m(target_height_m, target_quat_wxyz)
        self.max_lift_height_m = 0.0
        self.final_lift_height_m = 0.0
        self.max_support_clearance_m = 0.0
        self.final_support_clearance_m = 0.0
        self.pick_success = False
        self.first_pick_step = None
        self.pick_hold_step_count = 0
        self.longest_pick_hold_streak = 0
        self._pick_hold_streak = 0
        self.dropped_after_pick = False
        self.first_drop_step = None

    def update(
        self,
        episode_step: int,
        target_height_m: float,
        *,
        target_quat_wxyz: Any = None,
        gripper_touch: bool = False,
        receptacle_support: bool = False,
    ) -> bool:
        if (
            self.initial_target_height_m is None
            or self.baseline_target_height_m is None
            or self.baseline_target_bottom_m is None
        ):
            raise RuntimeError(
                "Pick lift metrics were updated before the episode baseline was set."
            )
        target_height_m = float(target_height_m)
        if not np.isfinite(target_height_m):
            raise ValueError(f"Target height must be finite, got {target_height_m!r}.")
        target_bottom_m = self._target_bottom_m(target_height_m, target_quat_wxyz)
        if not self.pick_success:
            self.baseline_target_height_m = min(self.baseline_target_height_m, target_height_m)
            self.baseline_target_bottom_m = min(self.baseline_target_bottom_m, target_bottom_m)
        lift_height_m = target_height_m - self.baseline_target_height_m
        support_clearance_m = target_bottom_m - self.baseline_target_bottom_m
        self.final_lift_height_m = lift_height_m
        self.max_lift_height_m = max(self.max_lift_height_m, lift_height_m)
        self.final_support_clearance_m = support_clearance_m
        self.max_support_clearance_m = max(self.max_support_clearance_m, support_clearance_m)
        gate_height_m = lift_height_m if self.half_extents_m is None else support_clearance_m
        current_pick_success = gate_height_m + 1e-09 >= self.threshold_m
        if current_pick_success and (not self.pick_success):
            self.pick_success = True
            self.first_pick_step = int(episode_step)
        if current_pick_success:
            self.pick_hold_step_count += 1
            self._pick_hold_streak += 1
            self.longest_pick_hold_streak = max(
                self.longest_pick_hold_streak, self._pick_hold_streak
            )
        else:
            if (
                self.pick_success
                and (not self.dropped_after_pick)
                and (not gripper_touch)
                and (not receptacle_support)
            ):
                self.dropped_after_pick = True
                self.first_drop_step = int(episode_step)
            self._pick_hold_streak = 0
        return self.pick_success

    def current_state(self, target_height_m: float, target_quat_wxyz: Any = None) -> dict[str, Any]:
        if (
            self.initial_target_height_m is None
            or self.baseline_target_height_m is None
            or self.baseline_target_bottom_m is None
        ):
            raise RuntimeError("Pick lift state was requested before the episode baseline was set.")
        target_height_m = float(target_height_m)
        if not np.isfinite(target_height_m):
            raise ValueError(f"Target height must be finite, got {target_height_m!r}.")
        target_bottom_m = self._target_bottom_m(target_height_m, target_quat_wxyz)
        lift_height_m = target_height_m - self.baseline_target_height_m
        support_clearance_m = target_bottom_m - self.baseline_target_bottom_m
        gate_height_m = lift_height_m if self.half_extents_m is None else support_clearance_m
        return {
            "target_height_m": target_height_m,
            "target_bottom_height_m": target_bottom_m,
            "lift_height_m": lift_height_m,
            "support_clearance_m": support_clearance_m,
            "clear_of_support_surface": support_clearance_m > 1e-09,
            "current_pick_success": gate_height_m + 1e-09 >= self.threshold_m,
            "pick_success": self.pick_success,
        }

    def as_dict(self) -> dict[str, Any]:
        if (
            self.initial_target_height_m is None
            or self.baseline_target_height_m is None
            or self.baseline_target_bottom_m is None
        ):
            raise RuntimeError(
                "Pick lift metrics were requested before the episode baseline was set."
            )
        return {
            "pick_success": self.pick_success,
            "initial_target_height_m": self.initial_target_height_m,
            "baseline_target_height_m": self.baseline_target_height_m,
            "pick_measured_at_target_bottom": self.half_extents_m is not None,
            "target_half_extents_m": None
            if self.half_extents_m is None
            else list(self.half_extents_m),
            "baseline_target_bottom_m": self.baseline_target_bottom_m,
            "max_support_clearance_m": self.max_support_clearance_m,
            "final_support_clearance_m": self.final_support_clearance_m,
            "max_lift_height": self.max_lift_height_m,
            "max_lift_height_m": self.max_lift_height_m,
            "final_lift_height_m": self.final_lift_height_m,
            "pick_height_threshold": self.threshold_m,
            "pick_height_threshold_m": self.threshold_m,
            "first_pick_step": self.first_pick_step,
            "pick_hold_step_count": self.pick_hold_step_count,
            "longest_pick_hold_streak": self.longest_pick_hold_streak,
            "dropped_after_pick": self.dropped_after_pick,
            "first_drop_step": self.first_drop_step,
        }


def _quaternion_angular_distance_rad(first_wxyz: Any, second_wxyz: Any) -> float:
    first = np.asarray(first_wxyz, dtype=np.float64).reshape(-1)
    second = np.asarray(second_wxyz, dtype=np.float64).reshape(-1)
    if tuple(first.shape) != (4,) or tuple(second.shape) != (4,):
        raise ValueError(
            f"Quaternion angular distance requires two [4] values, got {first.shape} and {second.shape}."
        )
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Quaternion angular distance requires finite values.")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise ValueError("Quaternion angular distance requires non-zero quaternions.")
    cosine = abs(float(np.dot(first / first_norm, second / second_norm)))
    return 2.0 * float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _quaternion_rotate_inverse(quat_wxyz: Any, vector_w: Any) -> np.ndarray:
    """Express a world-frame vector in the body frame described by `quat_wxyz`."""
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    vector = np.asarray(vector_w, dtype=np.float64).reshape(-1)
    if tuple(quat.shape) != (4,) or tuple(vector.shape) != (3,):
        raise ValueError(
            f"Inverse quaternion rotation requires a [4] quaternion and a [3] vector, got {quat.shape} and {vector.shape}."
        )
    if not np.all(np.isfinite(quat)) or not np.all(np.isfinite(vector)):
        raise ValueError("Inverse quaternion rotation requires finite values.")
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Inverse quaternion rotation requires a non-zero quaternion.")
    (w, x, y, z) = quat / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation.T @ vector


class PlaceMetricAccumulator:
    """Track MolmoSpaces-style placement after a successful pick."""

    def __init__(
        self,
        *,
        horizontal_threshold_m: float | None,
        containment_half_extents_xy_m: tuple[float, float] | None,
        vertical_offset_range_m: tuple[float, float] | None,
        min_support_normal_z: float,
        min_contact_force_n: float | None,
        max_receptacle_translation_m: float,
        max_receptacle_rotation_rad: float,
    ) -> None:
        self.horizontal_threshold_m = (
            None if horizontal_threshold_m is None else float(horizontal_threshold_m)
        )
        self.containment_half_extents_xy_m: tuple[float, float] | None = None
        if containment_half_extents_xy_m is not None:
            half_extents = [float(value) for value in containment_half_extents_xy_m]
            if len(half_extents) != 2:
                raise ValueError(
                    f"Place detector containment half extents must be an (x, y) pair, got {containment_half_extents_xy_m!r}."
                )
            if not all(np.isfinite(value) and value > 0.0 for value in half_extents):
                raise ValueError(
                    f"Place detector containment half extents must be positive and finite, got {containment_half_extents_xy_m!r}."
                )
            self.containment_half_extents_xy_m = (half_extents[0], half_extents[1])
        if vertical_offset_range_m is None:
            self.min_vertical_offset_m = None
            self.max_vertical_offset_m = None
        else:
            self.min_vertical_offset_m = float(vertical_offset_range_m[0])
            self.max_vertical_offset_m = float(vertical_offset_range_m[1])
        self.min_support_normal_z = float(min_support_normal_z)
        self.min_contact_force_n = (
            None if min_contact_force_n is None else float(min_contact_force_n)
        )
        self.max_receptacle_translation_m = float(max_receptacle_translation_m)
        self.max_receptacle_rotation_rad = float(max_receptacle_rotation_rad)
        values = tuple(
            value
            for value in (
                self.horizontal_threshold_m,
                self.min_vertical_offset_m,
                self.max_vertical_offset_m,
            )
            if value is not None
        ) + tuple(
            value
            for value in (
                self.min_support_normal_z,
                self.min_contact_force_n,
                self.max_receptacle_translation_m,
                self.max_receptacle_rotation_rad,
            )
            if value is not None
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"Place detector thresholds must be finite, got {values!r}.")
        if self.horizontal_threshold_m is not None and self.horizontal_threshold_m <= 0.0:
            raise ValueError("Place detector horizontal threshold must be positive.")
        if (self.min_vertical_offset_m is None) != (self.max_vertical_offset_m is None):
            raise ValueError("Place detector vertical offset bounds must both be set or disabled.")
        if (
            self.min_vertical_offset_m is not None
            and self.max_vertical_offset_m is not None
            and (self.min_vertical_offset_m > self.max_vertical_offset_m)
        ):
            raise ValueError("Place detector vertical offset range is reversed.")
        if not 0.0 <= self.min_support_normal_z <= 1.0:
            raise ValueError("Place detector min_support_normal_z must be in [0, 1].")
        if self.min_contact_force_n is not None and self.min_contact_force_n <= 0.0:
            raise ValueError("Place detector min_contact_force_n must be positive when enabled.")
        if self.max_receptacle_translation_m <= 0.0 or self.max_receptacle_rotation_rad <= 0.0:
            raise ValueError("Place detector receptacle displacement limits must be positive.")
        self.initial_receptacle_pose_w: np.ndarray | None = None
        self.place_success = False
        self.first_place_step: int | None = None
        self.min_horizontal_distance_m = float("inf")
        self.min_position_error_m = float("inf")
        self.final_position_error_m: float | None = None
        self.containment_ever = False
        self.first_containment_step: int | None = None
        self.containment_step_count = 0
        self.receptacle_support_ever = False
        self.first_receptacle_support_step: int | None = None
        self.receptacle_support_step_count = 0
        self.receptacle_support_with_containment_ever = False
        self.receptacle_support_with_translation_ok_ever = False
        self.receptacle_support_with_rotation_ok_ever = False
        self.max_receptacle_translation_observed_m = 0.0
        self.final_receptacle_translation_m = 0.0
        self.max_receptacle_rotation_observed_rad = 0.0
        self.final_receptacle_rotation_rad = 0.0

    def begin_episode(self, receptacle_pose_w: Any) -> None:
        pose = np.asarray(receptacle_pose_w, dtype=np.float64).reshape(-1)
        if tuple(pose.shape) != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"Initial receptacle pose must be finite [7], got {pose!r}.")
        _quaternion_angular_distance_rad(pose[3:], pose[3:])
        self.initial_receptacle_pose_w = pose.copy()
        self.place_success = False
        self.first_place_step = None
        self.min_horizontal_distance_m = float("inf")
        self.min_position_error_m = float("inf")
        self.final_position_error_m = None
        self.containment_ever = False
        self.first_containment_step = None
        self.containment_step_count = 0
        self.receptacle_support_ever = False
        self.first_receptacle_support_step = None
        self.receptacle_support_step_count = 0
        self.receptacle_support_with_containment_ever = False
        self.receptacle_support_with_translation_ok_ever = False
        self.receptacle_support_with_rotation_ok_ever = False
        self.max_receptacle_translation_observed_m = 0.0
        self.final_receptacle_translation_m = 0.0
        self.max_receptacle_rotation_observed_rad = 0.0
        self.final_receptacle_rotation_rad = 0.0

    def classify_contact(
        self, *, point_w: Any, normal_w: Any, normal_force_n: float, receptacle_pose_w: Any
    ) -> dict[str, Any]:
        """Classify whether a target contact is real support on the receptacle interior."""
        point = np.asarray(point_w, dtype=np.float64).reshape(-1)
        normal = np.asarray(normal_w, dtype=np.float64).reshape(-1)
        receptacle_pose = np.asarray(receptacle_pose_w, dtype=np.float64).reshape(-1)
        force_n = abs(float(normal_force_n))
        if tuple(point.shape) != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"Contact point must be finite [3], got {point!r}.")
        if tuple(normal.shape) != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError(f"Contact normal must be finite [3], got {normal!r}.")
        if tuple(receptacle_pose.shape) != (7,) or not np.all(np.isfinite(receptacle_pose)):
            raise ValueError(
                f"Contact receptacle pose must be finite [7], got {receptacle_pose!r}."
            )
        if not np.isfinite(force_n):
            raise ValueError(f"Contact normal force must be finite, got {normal_force_n!r}.")
        point_local = _quaternion_rotate_inverse(receptacle_pose[3:], point - receptacle_pose[:3])
        normal_ok = abs(float(normal[2])) >= self.min_support_normal_z
        force_ok = self.min_contact_force_n is None or force_n >= self.min_contact_force_n
        interior_xy_ok = self.containment_half_extents_xy_m is None or (
            abs(float(point_local[0])) <= self.containment_half_extents_xy_m[0]
            and abs(float(point_local[1])) <= self.containment_half_extents_xy_m[1]
        )
        interior_z_ok = (
            self.containment_half_extents_xy_m is None
            or self.min_vertical_offset_m is None
            or self.max_vertical_offset_m is None
            or (self.min_vertical_offset_m <= float(point_local[2]) <= self.max_vertical_offset_m)
        )
        interior_ok = interior_xy_ok and interior_z_ok
        return {
            "normal_support_ok": normal_ok,
            "contact_force_ok": force_ok,
            "interior_contact_ok": interior_ok,
            "point_local": point_local,
            "support_contact": normal_ok and force_ok and interior_ok,
        }

    def update(
        self,
        episode_step: int,
        *,
        target_position_w: Any,
        receptacle_pose_w: Any,
        support_contact: bool,
        pick_success: bool,
    ) -> bool:
        state = self.current_state(
            target_position_w=target_position_w,
            receptacle_pose_w=receptacle_pose_w,
            support_contact=support_contact,
            pick_success=pick_success,
        )
        self.min_horizontal_distance_m = min(
            self.min_horizontal_distance_m, float(state["place_horizontal_distance_m"])
        )
        position_error_m = float(state["place_position_error_m"])
        translation_m = float(state["receptacle_translation_m"])
        rotation_rad = float(state["receptacle_rotation_rad"])
        self.min_position_error_m = min(self.min_position_error_m, position_error_m)
        self.final_position_error_m = position_error_m
        self.max_receptacle_translation_observed_m = max(
            self.max_receptacle_translation_observed_m, translation_m
        )
        self.final_receptacle_translation_m = translation_m
        self.max_receptacle_rotation_observed_rad = max(
            self.max_receptacle_rotation_observed_rad, rotation_rad
        )
        self.final_receptacle_rotation_rad = rotation_rad
        if bool(state["place_containment_ok"]):
            self.containment_ever = True
            if self.first_containment_step is None:
                self.first_containment_step = int(episode_step)
            self.containment_step_count += 1
        if bool(state["support_contact"]):
            self.receptacle_support_ever = True
            if self.first_receptacle_support_step is None:
                self.first_receptacle_support_step = int(episode_step)
            self.receptacle_support_step_count += 1
            if bool(state["place_containment_ok"]):
                self.receptacle_support_with_containment_ever = True
            if bool(state["receptacle_translation_ok"]):
                self.receptacle_support_with_translation_ok_ever = True
            if bool(state["receptacle_rotation_ok"]):
                self.receptacle_support_with_rotation_ok_ever = True
        if bool(state["current_place_success"]) and (not self.place_success):
            self.place_success = True
            self.first_place_step = int(episode_step)
        return self.place_success

    def current_state(
        self,
        *,
        target_position_w: Any,
        receptacle_pose_w: Any,
        support_contact: bool,
        pick_success: bool,
    ) -> dict[str, Any]:
        if self.initial_receptacle_pose_w is None:
            raise RuntimeError("Place metrics were requested before the episode baseline was set.")
        target_position = np.asarray(target_position_w, dtype=np.float64).reshape(-1)
        receptacle_pose = np.asarray(receptacle_pose_w, dtype=np.float64).reshape(-1)
        if tuple(target_position.shape) != (3,) or not np.all(np.isfinite(target_position)):
            raise ValueError(f"Target position must be finite [3], got {target_position!r}.")
        if tuple(receptacle_pose.shape) != (7,) or not np.all(np.isfinite(receptacle_pose)):
            raise ValueError(f"Receptacle pose must be finite [7], got {receptacle_pose!r}.")
        horizontal_distance_m = float(np.linalg.norm(target_position[:2] - receptacle_pose[:2]))
        position_error_m = float(np.linalg.norm(target_position - receptacle_pose[:3]))
        vertical_offset_m = float(target_position[2] - receptacle_pose[2])
        local_offset_m = _quaternion_rotate_inverse(
            receptacle_pose[3:], target_position - receptacle_pose[:3]
        )
        translation_m = float(
            np.linalg.norm(receptacle_pose[:3] - self.initial_receptacle_pose_w[:3])
        )
        rotation_rad = _quaternion_angular_distance_rad(
            receptacle_pose[3:], self.initial_receptacle_pose_w[3:]
        )
        horizontal_ok = (
            self.horizontal_threshold_m is None
            or horizontal_distance_m <= self.horizontal_threshold_m
        )
        containment_ok = self.containment_half_extents_xy_m is None or (
            abs(float(local_offset_m[0])) <= self.containment_half_extents_xy_m[0]
            and abs(float(local_offset_m[1])) <= self.containment_half_extents_xy_m[1]
        )
        vertical_ok = (
            self.min_vertical_offset_m is None
            or self.max_vertical_offset_m is None
            or self.min_vertical_offset_m <= vertical_offset_m <= self.max_vertical_offset_m
        )
        translation_ok = translation_m <= self.max_receptacle_translation_m
        rotation_ok = rotation_rad <= self.max_receptacle_rotation_rad
        current_place_success = (
            bool(pick_success)
            and bool(support_contact)
            and horizontal_ok
            and containment_ok
            and vertical_ok
            and translation_ok
            and rotation_ok
        )
        return {
            "current_place_success": current_place_success,
            "support_contact": bool(support_contact),
            "place_position_error_m": position_error_m,
            "place_horizontal_distance_m": horizontal_distance_m,
            "place_vertical_offset_m": vertical_offset_m,
            "place_local_offset_xy_m": [float(local_offset_m[0]), float(local_offset_m[1])],
            "receptacle_translation_m": translation_m,
            "receptacle_rotation_rad": rotation_rad,
            "place_horizontal_ok": horizontal_ok,
            "place_containment_ok": containment_ok,
            "place_vertical_ok": vertical_ok,
            "receptacle_translation_ok": translation_ok,
            "receptacle_rotation_ok": rotation_ok,
            "place_success": self.place_success,
        }

    def as_dict(self, *, pick_success: bool) -> dict[str, Any]:
        if self.initial_receptacle_pose_w is None:
            raise RuntimeError("Place metrics were requested before the episode baseline was set.")
        place_success = bool(self.place_success)
        return {
            "place_success": place_success,
            "failure_stage": None if place_success else "place" if pick_success else "pick",
            "first_place_step": self.first_place_step,
            "place_containment_configured": self.containment_half_extents_xy_m is not None,
            "place_containment_half_extents_xy_m": None
            if self.containment_half_extents_xy_m is None
            else list(self.containment_half_extents_xy_m),
            "place_containment_ever": self.containment_ever,
            "first_place_containment_step": self.first_containment_step,
            "place_containment_step_count": self.containment_step_count,
            "receptacle_support_ever": self.receptacle_support_ever,
            "first_receptacle_support_step": self.first_receptacle_support_step,
            "receptacle_support_step_count": self.receptacle_support_step_count,
            "receptacle_support_with_containment_ever": self.receptacle_support_with_containment_ever,
            "receptacle_support_with_translation_ok_ever": self.receptacle_support_with_translation_ok_ever,
            "receptacle_support_with_rotation_ok_ever": self.receptacle_support_with_rotation_ok_ever,
            "min_place_position_error_m": self.min_position_error_m
            if np.isfinite(self.min_position_error_m)
            else None,
            "final_place_position_error_m": self.final_position_error_m,
            "min_place_horizontal_distance_m": self.min_horizontal_distance_m
            if np.isfinite(self.min_horizontal_distance_m)
            else None,
            "place_horizontal_threshold_m": self.horizontal_threshold_m,
            "place_min_contact_force_n": self.min_contact_force_n,
            "place_vertical_offset_range_m": None
            if self.min_vertical_offset_m is None
            else [self.min_vertical_offset_m, self.max_vertical_offset_m],
            "max_receptacle_translation_observed_m": self.max_receptacle_translation_observed_m,
            "final_receptacle_translation_m": self.final_receptacle_translation_m,
            "receptacle_translation_threshold_m": self.max_receptacle_translation_m,
            "max_receptacle_rotation_observed_rad": self.max_receptacle_rotation_observed_rad,
            "final_receptacle_rotation_rad": self.final_receptacle_rotation_rad,
            "receptacle_rotation_threshold_rad": self.max_receptacle_rotation_rad,
            "max_receptacle_translation_m": self.max_receptacle_translation_m,
            "max_receptacle_rotation_rad": self.max_receptacle_rotation_rad,
        }


def _pick_place_failure_reason(metrics: dict[str, Any]) -> str | None:
    if bool(metrics["place_success"]):
        return None
    if not bool(metrics["pickup_obj_touched"]):
        return "no_contact"
    if not bool(metrics["pick_success"]):
        if metrics["first_bilateral_touch_step"] is None:
            return "no_bilateral_contact"
        return "grasp_no_lift"
    if not bool(metrics["receptacle_support_ever"]):
        if bool(metrics["dropped_after_pick"]):
            return "picked_then_dropped"
        return "no_receptacle_support"
    if bool(metrics.get("place_containment_configured")) and (
        not bool(metrics.get("place_containment_ever"))
    ):
        return "target_outside_receptacle"
    translation_ok = bool(metrics["receptacle_support_with_translation_ok_ever"])
    rotation_ok = bool(metrics["receptacle_support_with_rotation_ok_ever"])
    if not translation_ok and (not rotation_ok):
        return "receptacle_displaced_and_rotated"
    if not translation_ok:
        return "receptacle_displaced"
    if not rotation_ok:
        return "receptacle_rotated"
    return "placement_not_stable"


def _build_task_summary(
    scene_config: dict[str, Any], episode_records: list[dict[str, Any]]
) -> dict[str, Any]:
    task_type = str(scene_config["task_type"])
    task = {
        "type": task_type,
        "asset_scene": int(scene_config["asset_scene"]),
        "success_criterion": None,
    }
    threshold_m = scene_config["pick_lift_threshold_m"]
    if threshold_m is None:
        return task
    pick_successes: list[bool] = []
    place_successes: list[bool] = []
    failure_stages: list[str | None] = []
    failure_reasons: list[str | None] = []
    pickup_touches: list[bool] = []
    bilateral_touches: list[bool] = []
    dropped_after_pick: list[bool] = []
    receptacle_supports: list[bool] = []
    max_lift_heights: list[float] = []
    min_place_position_errors: list[float] = []
    for record in episode_records:
        metrics = record.get("episode_metrics", {})
        pick_success = metrics.get("pick_success")
        if not isinstance(pick_success, bool):
            raise ValueError(
                f"DROID task episode metrics must contain boolean pick_success; episode={record.get('episode')!r} value={pick_success!r}."
            )
        pick_successes.append(pick_success)
        if task_type == "pick_place":
            place_success = metrics.get("place_success")
            failure_stage = metrics.get("failure_stage")
            expected_stage = None if place_success else "place" if pick_success else "pick"
            if not isinstance(place_success, bool) or failure_stage != expected_stage:
                raise ValueError(
                    f"DROID pick/place metrics are missing or inconsistent; episode={record.get('episode')!r} pick_success={pick_success!r} place_success={place_success!r} failure_stage={failure_stage!r}."
                )
            required_boolean_metrics = (
                "pickup_obj_touched",
                "dropped_after_pick",
                "receptacle_support_ever",
                "receptacle_support_with_translation_ok_ever",
                "receptacle_support_with_rotation_ok_ever",
            )
            invalid_booleans = [
                key for key in required_boolean_metrics if not isinstance(metrics.get(key), bool)
            ]
            if invalid_booleans:
                raise ValueError(
                    f"DROID pick/place metrics require boolean diagnostic fields; episode={record.get('episode')!r} invalid={invalid_booleans}."
                )
            try:
                expected_reason = _pick_place_failure_reason(metrics)
            except KeyError as exc:
                raise ValueError(
                    f"DROID pick/place metrics are missing a failure-classification field; episode={record.get('episode')!r} missing={exc.args[0]!r}."
                ) from exc
            failure_reason = metrics.get("failure_reason")
            if failure_reason != expected_reason:
                raise ValueError(
                    f"DROID pick/place failure reason is inconsistent; episode={record.get('episode')!r} actual={failure_reason!r} expected={expected_reason!r}."
                )
            max_lift_height = metrics.get("max_lift_height")
            min_place_position_error = metrics.get("min_place_position_error_m")
            if (
                isinstance(max_lift_height, bool)
                or not isinstance(max_lift_height, (int, float))
                or (not math.isfinite(float(max_lift_height)))
                or isinstance(min_place_position_error, bool)
                or (not isinstance(min_place_position_error, (int, float)))
                or (not math.isfinite(float(min_place_position_error)))
            ):
                raise ValueError(
                    f"DROID pick/place distance diagnostics must be finite; episode={record.get('episode')!r} max_lift_height={max_lift_height!r} min_place_position_error_m={min_place_position_error!r}."
                )
            place_successes.append(place_success)
            failure_stages.append(failure_stage)
            failure_reasons.append(failure_reason)
            pickup_touches.append(bool(metrics["pickup_obj_touched"]))
            bilateral_touches.append(metrics.get("first_bilateral_touch_step") is not None)
            dropped_after_pick.append(bool(metrics["dropped_after_pick"]))
            receptacle_supports.append(bool(metrics["receptacle_support_ever"]))
            max_lift_heights.append(float(max_lift_height))
            min_place_position_errors.append(float(min_place_position_error))
    episode_count = len(pick_successes)
    pick_success_count = sum(pick_successes)
    if task_type == "pick_place":
        place_success_count = sum(place_successes)
        place_config = scene_config.get("place_detector")
        if not isinstance(place_config, dict):
            raise ValueError("Configured DROID pick/place metrics require a place_detector config.")
        failure_reason_counts = {
            reason: failure_reasons.count(reason)
            for reason in sorted({reason for reason in failure_reasons if reason is not None})
        }
        dropped_after_pick_count = sum(dropped_after_pick)
        receptacle_support_count = sum(receptacle_supports)
        receptacle_support_given_pick_count = sum(
            picked and supported
            for (picked, supported) in zip(pick_successes, receptacle_supports, strict=True)
        )
        task.update(
            {
                "success_criterion": {
                    "type": "pick_then_receptacle_containment"
                    if place_config.get("containment_half_extents_xy_m") is not None
                    else "pick_then_receptacle_support",
                    "pick_lift_threshold_m": float(threshold_m),
                    **place_config,
                    "terminates_episode": True,
                },
                "pick_success_count": pick_success_count,
                "pick_success_rate": pick_success_count / episode_count if episode_count else 0.0,
                "place_success_count": place_success_count,
                "place_success_rate": place_success_count / episode_count if episode_count else 0.0,
                "failed_at_pick_count": sum(stage == "pick" for stage in failure_stages),
                "failed_at_place_count": sum(stage == "place" for stage in failure_stages),
                "pickup_obj_touch_count": sum(pickup_touches),
                "pickup_obj_touch_rate": sum(pickup_touches) / episode_count
                if episode_count
                else 0.0,
                "bilateral_touch_episode_count": sum(bilateral_touches),
                "bilateral_touch_episode_rate": sum(bilateral_touches) / episode_count
                if episode_count
                else 0.0,
                "dropped_after_pick_count": dropped_after_pick_count,
                "dropped_after_pick_rate": dropped_after_pick_count / pick_success_count
                if pick_success_count
                else 0.0,
                "receptacle_support_episode_count": receptacle_support_count,
                "receptacle_support_episode_rate": receptacle_support_count / episode_count
                if episode_count
                else 0.0,
                "receptacle_support_given_pick_count": receptacle_support_given_pick_count,
                "receptacle_support_given_pick_rate": receptacle_support_given_pick_count
                / pick_success_count
                if pick_success_count
                else 0.0,
                "avg_max_lift_height_m": sum(max_lift_heights) / episode_count
                if episode_count
                else None,
                "avg_min_place_position_error_m": sum(min_place_position_errors) / episode_count
                if episode_count
                else None,
                "failure_reason_counts": failure_reason_counts,
                "success_count": place_success_count,
                "failure_count": episode_count - place_success_count,
                "success_rate": place_success_count / episode_count if episode_count else 0.0,
            }
        )
        return task
    task.update(
        {
            "success_criterion": {
                "type": "target_lift",
                "threshold_m": float(threshold_m),
                "terminates_episode": True,
            },
            "success_count": pick_success_count,
            "failure_count": episode_count - pick_success_count,
            "success_rate": pick_success_count / episode_count if episode_count else 0.0,
        }
    )
    return task


def _build_episode_diagnostics_payload(
    *,
    episode: int,
    scene: int,
    asset_scene: int,
    task_type: str,
    instruction: str,
    target_body: str,
    receptacle_body: str | None,
    open_loop_horizon: int,
    replan_count: int,
    episode_metrics: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    if not steps:
        raise ValueError(f"Episode {episode} diagnostics contain no states.")
    if steps[0].get("action_from_previous_step") is not None:
        raise ValueError(f"Episode {episode} diagnostic state 0 must not have a preceding action.")
    tracking_errors: list[float] = []
    first_gripper_close_step: int | None = None
    for observation_index, step in enumerate(steps):
        if step.get("observation_index") != observation_index:
            raise ValueError(
                f"Episode {episode} diagnostic observation index mismatch: expected {observation_index}, got {step.get('observation_index')!r}."
            )
        if observation_index > 0 and step.get("action_from_previous_step") is None:
            raise ValueError(
                f"Episode {episode} diagnostic state {observation_index} is missing its preceding action."
            )
        tracking = step.get("tracking")
        if isinstance(tracking, dict):
            tracking_errors.append(float(tracking["arm_l2_error"]))
        action = step.get("action_from_previous_step")
        if (
            first_gripper_close_step is None
            and isinstance(action, list)
            and (len(action) == 8)
            and (float(action[-1]) >= 0.5)
        ):
            first_gripper_close_step = observation_index
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "episode": {
            "episode": int(episode),
            "scene": int(scene),
            "asset_scene": int(asset_scene),
            "task_type": task_type,
            "instruction": instruction,
            "target_body": target_body,
            "receptacle_body": receptacle_body,
            "num_observations": len(steps),
            "num_actions": len(steps) - 1,
        },
        "alignment": {
            "state_zero": "reset state before any action",
            "state_i": "post-action simulator state for action i-1 when i > 0",
        },
        "policy": {"open_loop_horizon": int(open_loop_horizon), "replan_count": int(replan_count)},
        "episode_metrics": episode_metrics,
        "diagnostic_summary": {
            "first_gripper_close_step": first_gripper_close_step,
            "mean_arm_tracking_l2_error": float(np.mean(tracking_errors))
            if tracking_errors
            else None,
            "max_arm_tracking_l2_error": max(tracking_errors) if tracking_errors else None,
            "replan_count": int(replan_count),
        },
        "steps": steps,
    }


class DroidSimDiagnosticsCollector:
    """Read target contacts and simulator state before Isaac Lab performs auto-reset."""

    def __init__(
        self,
        *,
        scene: int,
        instruction: str,
        open_loop_horizon: int,
        max_contacts_per_step: int,
        save_diagnostics: bool,
    ) -> None:
        self.scene_id = int(scene)
        self.instruction = instruction
        self.open_loop_horizon = int(open_loop_horizon)
        self.max_contacts_per_step = int(max_contacts_per_step)
        self.save_diagnostics = bool(save_diagnostics)
        scene_config = SCENE_CONFIGS[self.scene_id]
        self.asset_scene_id = int(scene_config["asset_scene"])
        self.task_type = str(scene_config["task_type"])
        self.target_body = str(scene_config["target"])
        receptacle = scene_config.get("receptacle")
        self.receptacle_body = None if receptacle is None else str(receptacle)
        threshold_m = scene_config["pick_lift_threshold_m"]
        self._pick_lift_metrics = (
            PickLiftMetricAccumulator(
                float(threshold_m), half_extents_m=scene_config.get("target_half_extents_m")
            )
            if threshold_m is not None
            else None
        )
        place_config = scene_config.get("place_detector")
        self._place_metrics = (
            PlaceMetricAccumulator(**place_config) if isinstance(place_config, dict) else None
        )
        if self.task_type == "pick_place" and self._pick_lift_metrics is not None:
            if self._place_metrics is None:
                raise ValueError(
                    f"Scene {self.scene_id} enables pick detection without a place detector."
                )
        if self._place_metrics is not None and self.receptacle_body is None:
            raise ValueError(f"Scene {self.scene_id} enables place detection without a receptacle.")
        self._env: Any | None = None
        self._contact_view: Any | None = None
        self._placement_contact_view: Any | None = None
        self._sensor_bodies: list[tuple[str, str]] = []
        self._arm_joint_ids: list[int] = []
        self._gripper_joint_ids: list[int] = []
        self._episode: int | None = None
        self._steps: list[dict[str, Any]] = []
        self._metrics = ContactMetricAccumulator()
        self._pending_action: np.ndarray | None = None
        self._pending_policy_source: dict[str, Any] | None = None
        self._capture_completed = False

    def initialize(self, env: Any) -> None:
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.sim import SimulationContext

        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if not isinstance(base_env, ManagerBasedRLEnv):
            raise TypeError(f"Expected ManagerBasedRLEnv, got {type(base_env).__name__}.")
        if int(base_env.num_envs) != 1:
            raise ValueError(f"DROID diagnostics require num_envs=1, got {base_env.num_envs}.")
        required_entities = ["robot", self.target_body]
        if self.receptacle_body is not None:
            required_entities.append(self.receptacle_body)
        missing_entities = [name for name in required_entities if name not in base_env.scene.keys()]
        if missing_entities:
            raise KeyError(
                f"Scene {self.scene_id} is missing diagnostics entities {missing_entities}; available={base_env.scene.keys()}."
            )
        robot = base_env.scene["robot"]
        joint_names = list(robot.data.joint_names)
        arm_joint_names = [f"panda_joint{index}" for index in range(1, 8)]
        missing_joints = [
            name for name in [*arm_joint_names, "finger_joint"] if name not in joint_names
        ]
        if missing_joints:
            raise KeyError(
                f"DROID robot is missing diagnostics joints {missing_joints}; available={joint_names}."
            )
        self._arm_joint_ids = [joint_names.index(name) for name in arm_joint_names]
        self._gripper_joint_ids = [joint_names.index("finger_joint")]
        env_regex = str(base_env.scene.env_regex_ns)
        gripper_root = f"{env_regex}/robot/Gripper/Robotiq_2F_85"
        sensor_patterns = [
            f"{gripper_root}/{body_name}".replace(".*", "*")
            for (_, body_name) in GRIPPER_CONTACT_BODIES
        ]
        target_pattern = f"{env_regex}/scene/{self.target_body}".replace(".*", "*")
        simulation_context = SimulationContext.instance()
        if simulation_context is None:
            raise RuntimeError("Isaac SimulationContext is not initialized for DROID diagnostics.")
        physics_view = simulation_context.physics_sim_view
        if physics_view is None:
            raise RuntimeError(
                "Isaac physics simulation view is unavailable for DROID diagnostics."
            )
        self._contact_view = physics_view.create_rigid_contact_view(
            sensor_patterns,
            filter_patterns=[[target_pattern] for _ in sensor_patterns],
            max_contact_data_count=self.max_contacts_per_step,
        )
        if int(self._contact_view.sensor_count) != len(GRIPPER_CONTACT_BODIES):
            raise RuntimeError(
                f"DROID target contact view resolved an unexpected number of gripper bodies: expected={len(GRIPPER_CONTACT_BODIES)} actual={self._contact_view.sensor_count} paths={list(self._contact_view.sensor_paths)}."
            )
        if int(self._contact_view.filter_count) != 1:
            raise RuntimeError(
                f"DROID target contact view must resolve exactly one target filter per finger; actual filter_count={self._contact_view.filter_count}."
            )
        if int(self._contact_view.max_contact_data_count) != self.max_contacts_per_step:
            raise RuntimeError(
                f"DROID target contact view has an unexpected contact-data capacity: expected={self.max_contacts_per_step} actual={self._contact_view.max_contact_data_count}."
            )
        if self._place_metrics is not None:
            assert self.receptacle_body is not None
            receptacle_pattern = f"{env_regex}/scene/{self.receptacle_body}".replace(".*", "*")
            self._placement_contact_view = physics_view.create_rigid_contact_view(
                [target_pattern],
                filter_patterns=[[receptacle_pattern]],
                max_contact_data_count=self.max_contacts_per_step,
            )
            if int(self._placement_contact_view.sensor_count) != 1:
                raise RuntimeError(
                    f"DROID placement contact view must resolve exactly one target body; actual={self._placement_contact_view.sensor_count} paths={list(self._placement_contact_view.sensor_paths)}."
                )
            if int(self._placement_contact_view.filter_count) != 1:
                raise RuntimeError(
                    f"DROID placement contact view must resolve exactly one receptacle filter; actual={self._placement_contact_view.filter_count}."
                )
        finger_by_body = {body_name: finger for (finger, body_name) in GRIPPER_CONTACT_BODIES}
        self._sensor_bodies = []
        for sensor_path in self._contact_view.sensor_paths:
            body_name = str(sensor_path).rsplit("/", 1)[-1]
            finger = finger_by_body.get(body_name)
            if finger is None:
                raise RuntimeError(f"Unexpected DROID contact sensor body: {sensor_path}.")
            self._sensor_bodies.append((finger, body_name))
        self._env = base_env

    def begin_episode(self, episode: int) -> None:
        self._require_initialized()
        if self._episode is not None:
            raise RuntimeError(
                f"Cannot start episode {episode}; episode {self._episode} is still active."
            )
        self._episode = int(episode)
        self._steps = []
        self._metrics = ContactMetricAccumulator()
        self._pending_action = None
        self._pending_policy_source = None
        self._capture_completed = False
        if self._pick_lift_metrics is not None:
            self._pick_lift_metrics.begin_episode(self._target_height_m(), self._target_quat_wxyz())
        if self._place_metrics is not None:
            assert self.receptacle_body is not None
            self._place_metrics.begin_episode(self._rigid_object_pose(self.receptacle_body))
        if self.save_diagnostics:
            initial_contacts = _summarize_contacts(self._read_contact_records())
            initial_placement_contacts = self._read_placement_contacts()
            self._steps.append(
                self._state_snapshot(
                    observation_index=0,
                    episode_step=0,
                    contacts=initial_contacts,
                    action=None,
                    policy_source=None,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    task_state=self._current_task_state(
                        placement_contacts=initial_placement_contacts
                    ),
                    placement_contacts=initial_placement_contacts,
                )
            )

    def prepare_action(self, action: Any, policy_source: dict[str, Any]) -> None:
        if self._episode is None:
            raise RuntimeError("Cannot prepare a diagnostics action without an active episode.")
        if self._pending_action is not None or self._capture_completed:
            raise RuntimeError("Previous DROID diagnostics action has not been finalized.")
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if tuple(action_array.shape) != (8,):
            raise ValueError(
                f"DROID diagnostics action must have shape [8], got {action_array.shape}."
            )
        self._pending_action = action_array.copy()
        self._pending_policy_source = dict(policy_source)

    def capture_post_action(self, env: Any) -> bool:
        if self._episode is None or self._pending_action is None:
            raise RuntimeError(
                "DROID diagnostics termination callback ran without a prepared action."
            )
        if env is not self._env:
            raise RuntimeError(
                "DROID diagnostics callback received a different environment instance."
            )
        if self._capture_completed:
            raise RuntimeError("DROID diagnostics captured the same action more than once.")
        episode_step = int(_single_value(env.episode_length_buf, name="episode_length_buf"))
        contact_records = self._read_contact_records()
        contacts = self._metrics.update(episode_step, contact_records)
        placement_contacts = self._read_placement_contacts()
        pick_success = False
        if self._pick_lift_metrics is not None:
            pick_success = self._pick_lift_metrics.update(
                episode_step,
                self._target_height_m(),
                target_quat_wxyz=self._target_quat_wxyz(),
                gripper_touch=bool(contacts["contact_count"]),
                receptacle_support=bool(placement_contacts["support_contact"]),
            )
        place_success = False
        if self._place_metrics is not None:
            assert self.receptacle_body is not None
            place_success = self._place_metrics.update(
                episode_step,
                target_position_w=self._rigid_object_position(self.target_body),
                receptacle_pose_w=self._rigid_object_pose(self.receptacle_body),
                support_contact=bool(placement_contacts["support_contact"]),
                pick_success=pick_success,
            )
        if self.save_diagnostics:
            self._steps.append(
                self._state_snapshot(
                    observation_index=len(self._steps),
                    episode_step=episode_step,
                    contacts=contacts,
                    action=self._pending_action,
                    policy_source=self._pending_policy_source,
                    reward=None,
                    terminated=None,
                    truncated=None,
                    task_state=self._current_task_state(placement_contacts=placement_contacts),
                    placement_contacts=placement_contacts,
                )
            )
        self._capture_completed = True
        if self.task_type == "pick_place" and self._place_metrics is not None:
            return place_success
        return pick_success

    def finish_action(self, *, reward: Any, terminated: Any, truncated: Any) -> None:
        if not self._capture_completed:
            raise RuntimeError(
                "DROID diagnostics action returned without a pre-reset simulator capture."
            )
        if self.save_diagnostics:
            step = self._steps[-1]
            step["reward"] = float(_single_value(reward, name="reward"))
            step["terminated"] = bool(_single_value(terminated, name="terminated"))
            step["truncated"] = bool(_single_value(truncated, name="truncated"))
        self._pending_action = None
        self._pending_policy_source = None
        self._capture_completed = False

    def finalize_episode(
        self, *, replan_count: int
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if self._episode is None:
            raise RuntimeError("Cannot finalize DROID diagnostics without an active episode.")
        if self._pending_action is not None or self._capture_completed:
            raise RuntimeError("Cannot finalize DROID diagnostics with an unfinished action.")
        episode = self._episode
        metrics = self._metrics.as_dict()
        if self._pick_lift_metrics is not None:
            metrics.update(self._pick_lift_metrics.as_dict())
        if self._place_metrics is not None:
            metrics.update(self._place_metrics.as_dict(pick_success=bool(metrics["pick_success"])))
            metrics["failure_reason"] = _pick_place_failure_reason(metrics)
        payload = None
        if self.save_diagnostics:
            payload = _build_episode_diagnostics_payload(
                episode=episode,
                scene=self.scene_id,
                asset_scene=self.asset_scene_id,
                task_type=self.task_type,
                instruction=self.instruction,
                target_body=self.target_body,
                receptacle_body=self.receptacle_body,
                open_loop_horizon=self.open_loop_horizon,
                replan_count=replan_count,
                episode_metrics=metrics,
                steps=self._steps,
            )
        self._episode = None
        self._steps = []
        return (metrics, payload)

    def _require_initialized(self) -> None:
        if self._env is None or self._contact_view is None:
            raise RuntimeError("DROID diagnostics collector has not been initialized.")

    def _read_contact_records(self) -> list[dict[str, Any]]:
        self._require_initialized()
        assert self._contact_view is not None
        assert self._env is not None
        (
            force_buffer,
            point_buffer,
            normal_buffer,
            separation_buffer,
            count_buffer,
            start_buffer,
        ) = self._contact_view.get_contact_data(float(self._env.physics_dt))
        forces = _numpy(force_buffer).reshape(-1)
        points = _numpy(point_buffer).reshape(-1, 3)
        normals = _numpy(normal_buffer).reshape(-1, 3)
        separations = _numpy(separation_buffer).reshape(-1)
        counts = _numpy(count_buffer).reshape(len(self._sensor_bodies), -1).astype(np.int64)
        starts = _numpy(start_buffer).reshape(len(self._sensor_bodies), -1).astype(np.int64)
        if int(counts.shape[1]) != 1:
            raise RuntimeError(
                f"Expected one target contact filter, got count shape={counts.shape}."
            )
        total_contacts = int(counts.sum())
        _validate_contact_buffer_capacity(total_contacts, self.max_contacts_per_step)
        contacts: list[dict[str, Any]] = []
        for sensor_index, (finger, body_name) in enumerate(self._sensor_bodies):
            start = int(starts[sensor_index, 0])
            count = int(counts[sensor_index, 0])
            if start < 0 or count < 0 or start + count > len(forces):
                raise RuntimeError(
                    f"Invalid PhysX target contact buffer slice: body={body_name} start={start} count={count} capacity={len(forces)}."
                )
            for contact_index in range(start, start + count):
                contacts.append(
                    {
                        "finger": finger,
                        "finger_body": body_name,
                        "normal_force_n": abs(float(forces[contact_index])),
                        "point_w": points[contact_index].astype(np.float64).tolist(),
                        "normal_w": normals[contact_index].astype(np.float64).tolist(),
                        "separation_m": float(separations[contact_index]),
                    }
                )
        return contacts

    def _read_placement_contacts(self) -> dict[str, Any]:
        if self._place_metrics is None:
            return {
                "contact_count": 0,
                "normal_support_contact_count": 0,
                "force_contact_count": 0,
                "interior_contact_count": 0,
                "support_contact_count": 0,
                "support_contact": False,
                "max_normal_force_n": 0.0,
                "contacts": [],
            }
        if self._placement_contact_view is None or self._env is None:
            raise RuntimeError("Configured DROID place detector has no placement contact view.")
        (
            force_buffer,
            point_buffer,
            normal_buffer,
            separation_buffer,
            count_buffer,
            start_buffer,
        ) = self._placement_contact_view.get_contact_data(float(self._env.physics_dt))
        forces = _numpy(force_buffer).reshape(-1)
        points = _numpy(point_buffer).reshape(-1, 3)
        normals = _numpy(normal_buffer).reshape(-1, 3)
        separations = _numpy(separation_buffer).reshape(-1)
        counts = _numpy(count_buffer).reshape(1, -1).astype(np.int64)
        starts = _numpy(start_buffer).reshape(1, -1).astype(np.int64)
        if tuple(counts.shape) != (1, 1) or tuple(starts.shape) != (1, 1):
            raise RuntimeError(
                f"Expected one target/receptacle contact pair, got counts={counts.shape} starts={starts.shape}."
            )
        count = int(counts[0, 0])
        start = int(starts[0, 0])
        _validate_contact_buffer_capacity(count, self.max_contacts_per_step)
        if start < 0 or count < 0 or start + count > len(forces):
            raise RuntimeError(
                f"Invalid PhysX placement contact buffer slice: start={start} count={count} capacity={len(forces)}."
            )
        contacts = []
        normal_support_contact_count = 0
        force_contact_count = 0
        interior_contact_count = 0
        support_contact_count = 0
        assert self.receptacle_body is not None
        receptacle_pose = self._rigid_object_pose(self.receptacle_body)
        for contact_index in range(start, start + count):
            normal = normals[contact_index].astype(np.float64)
            normal_force_n = abs(float(forces[contact_index]))
            classification = self._place_metrics.classify_contact(
                point_w=points[contact_index],
                normal_w=normal,
                normal_force_n=normal_force_n,
                receptacle_pose_w=receptacle_pose,
            )
            normal_support_contact_count += int(classification["normal_support_ok"])
            force_contact_count += int(classification["contact_force_ok"])
            interior_contact_count += int(classification["interior_contact_ok"])
            support_contact_count += int(classification["support_contact"])
            contacts.append(
                {
                    "normal_force_n": normal_force_n,
                    "point_w": points[contact_index].astype(np.float64).tolist(),
                    "point_receptacle_m": classification["point_local"].tolist(),
                    "normal_w": normal.tolist(),
                    "separation_m": float(separations[contact_index]),
                    "normal_support_ok": bool(classification["normal_support_ok"]),
                    "contact_force_ok": bool(classification["contact_force_ok"]),
                    "interior_contact_ok": bool(classification["interior_contact_ok"]),
                    "support_contact": bool(classification["support_contact"]),
                }
            )
        return {
            "contact_count": count,
            "normal_support_contact_count": normal_support_contact_count,
            "force_contact_count": force_contact_count,
            "interior_contact_count": interior_contact_count,
            "support_contact_count": support_contact_count,
            "support_contact": support_contact_count > 0,
            "max_normal_force_n": max(
                (float(contact["normal_force_n"]) for contact in contacts), default=0.0
            ),
            "contacts": contacts,
        }

    def _current_task_state(self, *, placement_contacts: dict[str, Any]) -> dict[str, Any] | None:
        if self._pick_lift_metrics is None:
            return None
        state = self._pick_lift_metrics.current_state(
            self._target_height_m(), self._target_quat_wxyz()
        )
        if self._place_metrics is not None:
            assert self.receptacle_body is not None
            state.update(
                self._place_metrics.current_state(
                    target_position_w=self._rigid_object_position(self.target_body),
                    receptacle_pose_w=self._rigid_object_pose(self.receptacle_body),
                    support_contact=bool(placement_contacts["support_contact"]),
                    pick_success=bool(state["pick_success"]),
                )
            )
        return state

    def _target_quat_wxyz(self) -> np.ndarray | None:
        if self._pick_lift_metrics is None or self._pick_lift_metrics.half_extents_m is None:
            return None
        return self._rigid_object_pose(self.target_body)[3:]

    def _target_height_m(self) -> float:
        assert self._env is not None
        target_height_m = float(_numpy(self._env.scene[self.target_body].data.root_pos_w[0, 2]))
        if not np.isfinite(target_height_m):
            raise RuntimeError(
                f"Scene {self.scene_id} target height is not finite: {target_height_m!r}."
            )
        return target_height_m

    def _rigid_object_position(self, name: str) -> np.ndarray:
        assert self._env is not None
        position = _numpy(self._env.scene[name].data.root_pos_w[0]).astype(np.float64)
        if tuple(position.shape) != (3,) or not np.all(np.isfinite(position)):
            raise RuntimeError(
                f"Scene {self.scene_id} rigid object {name!r} has invalid position {position!r}."
            )
        return position

    def _rigid_object_pose(self, name: str) -> np.ndarray:
        assert self._env is not None
        data = self._env.scene[name].data
        pose = np.concatenate(
            [
                _numpy(data.root_pos_w[0]).astype(np.float64),
                _numpy(data.root_quat_w[0]).astype(np.float64),
            ]
        )
        if tuple(pose.shape) != (7,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(
                f"Scene {self.scene_id} rigid object {name!r} has invalid pose {pose!r}."
            )
        return pose

    def _state_snapshot(
        self,
        *,
        observation_index: int,
        episode_step: int,
        contacts: dict[str, Any],
        action: np.ndarray | None,
        policy_source: dict[str, Any] | None,
        reward: float | None,
        terminated: bool | None,
        truncated: bool | None,
        task_state: dict[str, Any] | None,
        placement_contacts: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_initialized()
        assert self._env is not None
        robot = self._env.scene["robot"]
        arm_position = _numpy(robot.data.joint_pos[0, self._arm_joint_ids]).astype(np.float64)
        arm_velocity = _numpy(robot.data.joint_vel[0, self._arm_joint_ids]).astype(np.float64)
        gripper_position = _numpy(robot.data.joint_pos[0, self._gripper_joint_ids]).astype(
            np.float64
        )
        gripper_velocity = _numpy(robot.data.joint_vel[0, self._gripper_joint_ids]).astype(
            np.float64
        )
        robot_root_pose = np.concatenate(
            [
                _numpy(robot.data.root_pos_w[0]).astype(np.float64),
                _numpy(robot.data.root_quat_w[0]).astype(np.float64),
            ]
        )
        tracking = None
        if action is not None:
            arm_error = np.asarray(action[:7], dtype=np.float64) - arm_position
            tracking = {
                "arm_error": arm_error.tolist(),
                "arm_l2_error": float(np.linalg.norm(arm_error)),
                "arm_linf_error": float(np.max(np.abs(arm_error))),
            }
        snapshot = {
            "observation_index": int(observation_index),
            "episode_step": int(episode_step),
            "sim_time_sec": float(episode_step) * float(self._env.step_dt),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "action_from_previous_step": None
            if action is None
            else action.astype(np.float64).tolist(),
            "policy_action_source": policy_source,
            "tracking": tracking,
            "robot": {
                "arm_joint_position": arm_position.tolist(),
                "arm_joint_velocity": arm_velocity.tolist(),
                "gripper_joint_position": gripper_position.tolist(),
                "gripper_joint_velocity": gripper_velocity.tolist(),
                "root_pose_w": robot_root_pose.tolist(),
            },
            "target": self._rigid_object_state(self.target_body),
            "receptacle": self._rigid_object_state(self.receptacle_body)
            if self.receptacle_body is not None
            else None,
            "contacts": contacts,
            "placement_contacts": placement_contacts,
        }
        if task_state is not None:
            snapshot["task"] = task_state
        return snapshot

    def _rigid_object_state(self, name: str) -> dict[str, Any]:
        assert self._env is not None
        data = self._env.scene[name].data
        pose = self._rigid_object_pose(name)
        return {
            "body_name": name,
            "pose_w": pose.tolist(),
            "linear_velocity_w": _numpy(data.root_lin_vel_w[0]).astype(np.float64).tolist(),
            "angular_velocity_w": _numpy(data.root_ang_vel_w[0]).astype(np.float64).tolist(),
        }


def _capture_diagnostic_snapshot(env: Any, collector: DroidSimDiagnosticsCollector) -> torch.Tensor:
    task_success = collector.capture_post_action(env)
    return torch.full((env.num_envs,), task_success, dtype=torch.bool, device=env.device)


def _resize_rgb(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return np.asarray(pil_image.resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)


def _center_crop_resize_rgb(
    image: np.ndarray, *, height: int, width: int, crop_scale: float = DROID_CENTER_CROP_SCALE
) -> np.ndarray:
    if not 0.0 < float(crop_scale) <= 1.0:
        raise ValueError(f"crop_scale must be in (0, 1], got {crop_scale}.")
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or int(image.shape[-1]) != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}.")
    (in_height, in_width) = (int(image.shape[0]), int(image.shape[1]))
    crop_h = max(1, min(in_height, int(round(in_height * float(crop_scale)))))
    crop_w = max(1, min(in_width, int(round(in_width * float(crop_scale)))))
    top = max((in_height - crop_h) // 2, 0)
    left = max((in_width - crop_w) // 2, 0)
    return _resize_rgb(image[top : top + crop_h, left : left + crop_w], height=height, width=width)


def _build_droid_grid_video_frame(request: dict[str, Any]) -> np.ndarray:
    left = _center_crop_resize_rgb(
        request["observation/exterior_image_1_left"],
        height=DROID_PER_CAMERA_HEIGHT,
        width=DROID_PER_CAMERA_WIDTH,
    )
    right = _center_crop_resize_rgb(
        request["observation/exterior_image_2_left"],
        height=DROID_PER_CAMERA_HEIGHT,
        width=DROID_PER_CAMERA_WIDTH,
    )
    wrist = _center_crop_resize_rgb(
        request["observation/wrist_image_left"],
        height=DROID_PER_CAMERA_HEIGHT,
        width=DROID_PER_CAMERA_WIDTH,
    )
    wrist_wide = np.repeat(wrist, 2, axis=1)
    bottom = np.concatenate([left, right], axis=1)
    frame = np.concatenate([wrist_wide, bottom], axis=0)
    if tuple(frame.shape) != (DROID_GRID_HEIGHT, DROID_GRID_WIDTH, 3):
        raise ValueError(f"DROID grid video frame must be [352,640,3], got {frame.shape}.")
    return frame


def _tensor_to_rgb(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 4 and int(array.shape[0]) == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"{name} must be a 3D RGB image after batch squeeze, got shape={array.shape}."
        )
    if int(array.shape[-1]) != 3:
        raise ValueError(f"{name} must be HWC RGB, got shape={array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _tensor_to_1d(value: Any, *, name: str, dim: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = array.astype(np.float32).reshape(-1)
    if int(array.shape[0]) != int(dim):
        raise ValueError(f"{name} must have {dim} dims, got shape={array.shape}.")
    return array


def extract_sim_observation(obs: dict[str, Any]) -> dict[str, Any]:
    if "policy" not in obs or not isinstance(obs["policy"], dict):
        raise KeyError("DROID sim observation is missing dict key `policy`.")
    policy_obs = obs["policy"]
    return {
        "observation/exterior_image_1_left": _tensor_to_rgb(
            policy_obs["external_cam"], name="external_cam"
        ),
        "observation/exterior_image_2_left": _tensor_to_rgb(
            policy_obs["external_cam_2"], name="external_cam_2"
        ),
        "observation/wrist_image_left": _tensor_to_rgb(policy_obs["wrist_cam"], name="wrist_cam"),
        "observation/joint_position": _tensor_to_1d(
            policy_obs["arm_joint_pos"], name="arm_joint_pos", dim=7
        ),
        "observation/gripper_position": _tensor_to_1d(
            policy_obs["gripper_pos"], name="gripper_pos", dim=1
        ),
    }


class DroidSimClient:
    def __init__(self, *, host: str, port: int, open_loop_horizon: int) -> None:
        self.rpc = PolicyRpcClient(host, port)
        server_info = self.rpc.info()
        if not isinstance(server_info, dict):
            raise ValueError(
                f"Policy server info must be a dict, got {type(server_info).__name__}."
            )
        self.open_loop_horizon = int(open_loop_horizon)
        self.requires_episode_seed = bool(server_info.get("requires_episode_seed_on_reset", False))
        self.episode_index = 0
        if self.open_loop_horizon <= 0:
            raise ValueError(f"open_loop_horizon must be positive, got {self.open_loop_horizon}.")
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk: np.ndarray | None = None
        self.replan_id = -1
        self.action_index = 0
        self.action_history_buffer: IssuedActionHistoryBuffer | None = None
        history_horizon = server_info.get("action_history_horizon", 0)
        if type(history_horizon) is not int or history_horizon < 0:
            raise ValueError("Server action_history_horizon must be a nonnegative integer.")
        expected_type = "dreamfree_stage2_action_history" if history_horizon else "dreamfree_stage2"
        if server_info.get("checkpoint_type") != expected_type:
            raise ValueError("Server checkpoint_type disagrees with action_history_horizon.")
        if history_horizon:
            self.action_history_buffer = IssuedActionHistoryBuffer(
                horizon=history_horizon, action_dim=8
            )

    @property
    def replan_count(self) -> int:
        return self.replan_id + 1

    def reset(self) -> None:
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.replan_id = -1
        self.action_index = 0
        if self.action_history_buffer is not None:
            self.action_history_buffer.clear()
        if self.requires_episode_seed:
            self.rpc.reset(episode_seed=self.episode_index)
            self.episode_index += 1
        else:
            self.rpc.reset()

    def close(self) -> None:
        self.rpc.close()

    def infer(self, obs: dict[str, Any], instruction: str) -> dict[str, Any]:
        request = extract_sim_observation(obs)
        if (
            self.pred_action_chunk is None
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            if self.action_history_buffer is not None:
                request = self.action_history_buffer.add_to_observation(request)
            result = self.rpc.infer(request, instruction)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or int(actions.shape[1]) != 8:
                raise ValueError(
                    f"Policy server returned actions with shape {actions.shape}; expected [N,8]."
                )
            if int(actions.shape[0]) < self.open_loop_horizon:
                raise ValueError(
                    f"Policy server returned {actions.shape[0]} actions, fewer than open_loop_horizon={self.open_loop_horizon}."
                )
            self.pred_action_chunk = actions
            self.actions_from_chunk_completed = 0
            self.replan_id += 1
        chunk_index = self.actions_from_chunk_completed
        action = self.pred_action_chunk[chunk_index].copy()
        self.actions_from_chunk_completed += 1
        action[-1] = canonicalize_droid_gripper_command(
            action[-1], DROID_GRIPPER_NORMALIZED_COMMAND_SCALE
        )
        viz = _build_droid_grid_video_frame(request)
        policy_source = {
            "action_index": self.action_index,
            "observation_step": self.action_index,
            "replan_id": self.replan_id,
            "chunk_index": chunk_index,
        }
        self.action_index += 1
        return {"action": action, "viz": viz, "policy_action_source": policy_source}

    def record_issued_action(self, action: np.ndarray) -> None:
        if self.action_history_buffer is not None:
            issued_action = np.asarray(action, dtype=np.float32).reshape(-1)
            if issued_action.shape != (8,) or not np.isfinite(issued_action).all():
                raise ValueError(
                    f"DROID issued action must contain 8 finite values, got {issued_action.shape}."
                )
            if float(issued_action[7]) not in (0.0, 1.0):
                raise ValueError(
                    f"DROID issued-action history requires a binary gripper command, got {float(issued_action[7])}."
                )
            self.action_history_buffer.record_issued_action(issued_action)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6100)
    parser.add_argument("--open-loop-horizon", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument("--max-contacts-per-step", type=int, default=128)
    parser.add_argument("--progress-fd", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    (args, _) = parser.parse_known_args()
    return args


def main() -> None:
    args = _parse_args()
    if args.scene not in SCENE_CONFIGS:
        raise ValueError(
            f"Scene {args.scene} not supported; expected one of {sorted(SCENE_CONFIGS)}."
        )
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}.")
    if args.max_contacts_per_step <= 0:
        raise ValueError(
            f"--max-contacts-per-step must be positive, got {args.max_contacts_per_step}."
        )
    progress_stream = sys.stderr
    if args.progress_fd is not None:
        if args.progress_fd < 0:
            raise ValueError(f"--progress-fd must be non-negative, got {args.progress_fd}.")
        try:
            progress_stream = os.fdopen(args.progress_fd, "w", buffering=1, closefd=False)
        except OSError as exc:
            raise ValueError(f"--progress-fd {args.progress_fd} is not open.") from exc
    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser(description="DROID sim-evals AppLauncher args.")
    AppLauncher.add_app_launcher_args(app_parser)
    (app_args, _) = app_parser.parse_known_args()
    app_args.enable_cameras = True
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app
    import gymnasium as gym
    import mediapy
    import sim_evals.environments  # noqa: F401 - register simulator environments
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab_tasks.utils import parse_env_cfg

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / "droid_sim"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_config = SCENE_CONFIGS[int(args.scene)]
    instruction = str(scene_config["instruction"])
    print(f"Instruction: {instruction}", flush=True)
    diagnostics = DroidSimDiagnosticsCollector(
        scene=args.scene,
        instruction=instruction,
        open_loop_horizon=args.open_loop_horizon,
        max_contacts_per_step=args.max_contacts_per_step,
        save_diagnostics=args.save_diagnostics,
    )
    env_cfg = parse_env_cfg("DROID", device=app_args.device, num_envs=1, use_fabric=True)
    _configure_scene_cameras(env_cfg, scene_config)
    asset_dir = scene_config.get("asset_dir")
    if asset_dir is not None:
        from sim_evals.environments import droid_environment

        asset_dir = Path(asset_dir)
        expected_asset_path = asset_dir / f"scene{scene_config['asset_scene']}.usd"
        if not expected_asset_path.is_file():
            raise FileNotFoundError(
                f"Configured DROID scene asset not found: {expected_asset_path}"
            )
        droid_environment.DATA_PATH = asset_dir
    env_cfg.set_scene(int(scene_config["asset_scene"]))
    env_cfg.terminations.diagnostic_snapshot = DoneTerm(
        func=_capture_diagnostic_snapshot, params={"collector": diagnostics}
    )
    env = gym.make("DROID", cfg=env_cfg)
    diagnostic_term = env.unwrapped.termination_manager.get_term_cfg("diagnostic_snapshot")
    diagnostic_term.params["collector"] = diagnostics
    configured_collector = diagnostic_term.params.get("collector")
    if configured_collector is not diagnostics:
        raise RuntimeError(
            "DROID diagnostics collector identity changed during environment creation."
        )
    diagnostics.initialize(env)
    (obs, _) = env.reset()
    (obs, _) = env.reset()
    client = DroidSimClient(
        host=args.host, port=args.port, open_loop_horizon=args.open_loop_horizon
    )
    client.reset()
    episode_records: list[dict[str, Any]] = []
    max_steps = int(env.env.max_episode_length)
    try:
        with torch.no_grad():
            for ep in range(int(args.episodes)):
                diagnostics.begin_episode(ep)
                video: list[np.ndarray] = []
                term = False
                trunc = False
                steps = 0
                start_time = time.time()
                for _ in tqdm(
                    range(max_steps),
                    desc=f"Scene {args.scene} Episode {ep + 1}/{args.episodes}",
                    disable=not progress_stream.isatty(),
                    file=progress_stream,
                ):
                    ret = client.infer(obs, instruction)
                    video.append(ret["viz"])
                    diagnostics.prepare_action(ret["action"], ret["policy_action_source"])
                    action = torch.tensor(ret["action"], dtype=torch.float32)[None]
                    (obs, reward, term, trunc, _) = env.step(action)
                    client.record_issued_action(ret["action"])
                    diagnostics.finish_action(reward=reward, terminated=term, truncated=trunc)
                    steps += 1
                    if bool(_single_value(term, name="terminated")) or bool(
                        _single_value(trunc, name="truncated")
                    ):
                        break
                mediapy.write_video(output_dir / f"episode_{ep:03d}.mp4", video, fps=15)
                (episode_metrics, diagnostics_payload) = diagnostics.finalize_episode(
                    replan_count=client.replan_count
                )
                episode_record = {
                    "episode": ep,
                    "steps": steps,
                    "terminated": bool(_single_value(term, name="terminated")),
                    "truncated": bool(_single_value(trunc, name="truncated")),
                    "elapsed_sec": time.time() - start_time,
                    "replan_count": client.replan_count,
                    "episode_metrics": episode_metrics,
                }
                if scene_config["pick_lift_threshold_m"] is not None:
                    success_key = (
                        "place_success"
                        if scene_config["task_type"] == "pick_place"
                        else "pick_success"
                    )
                    success = episode_metrics.get(success_key)
                    if not isinstance(success, bool):
                        raise RuntimeError(
                            f"Configured DROID success metric {success_key!r} is missing."
                        )
                    episode_record["success"] = success
                    print(
                        f"Episode {ep} stage metrics: pick_success={episode_metrics['pick_success']} place_success={episode_metrics.get('place_success')} failure_stage={episode_metrics.get('failure_stage')} failure_reason={episode_metrics.get('failure_reason')}"
                    )
                if diagnostics_payload is not None:
                    diagnostics_path = output_dir / f"episode_{ep:03d}_diagnostics.json"
                    _atomic_write_json(diagnostics_path, diagnostics_payload)
                    episode_record["diagnostics"] = {
                        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                        "file": diagnostics_path.name,
                        "size_bytes": diagnostics_path.stat().st_size,
                    }
                episode_records.append(episode_record)
                client.reset()
                if not episode_record["terminated"] and (not episode_record["truncated"]):
                    (obs, _) = env.reset()
        task_summary = _build_task_summary(scene_config, episode_records)
        summary = {
            "schema_version": 1,
            "scene": int(args.scene),
            "instruction": instruction,
            "task": task_summary,
            "episodes": episode_records,
            "max_steps": max_steps,
            "open_loop_horizon": int(args.open_loop_horizon),
            "camera_calibration": scene_config.get(
                "camera_calibration", {"name": "sim_evals_default"}
            ),
            "diagnostics": {
                "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "contact_scope": CONTACT_SCOPE,
                "contact_measure": "PhysX contact patches at the final physics substep of each control step",
                "target_body": diagnostics.target_body,
                "receptacle_body": diagnostics.receptacle_body,
                "finger_bodies": {
                    "left": [body for (finger, body) in GRIPPER_CONTACT_BODIES if finger == "left"],
                    "right": [
                        body for (finger, body) in GRIPPER_CONTACT_BODIES if finger == "right"
                    ],
                },
                "detailed_traces_enabled": bool(args.save_diagnostics),
                "max_contacts_per_step": int(args.max_contacts_per_step),
                "pick_place_metrics_enabled": diagnostics._pick_lift_metrics is not None,
                "place_contact_measure": "target/receptacle PhysX support contact plus configured receptacle stability and optional containment thresholds"
                if diagnostics._place_metrics is not None
                else None,
                "physics_dt_sec": float(env.unwrapped.physics_dt),
                "control_dt_sec": float(env.unwrapped.step_dt),
            },
            "video_layout": {
                "name": "droid_grid",
                "height": DROID_GRID_HEIGHT,
                "width": DROID_GRID_WIDTH,
                "camera_keys": [
                    "observation/exterior_image_1_left",
                    "observation/exterior_image_2_left",
                    "observation/wrist_image_left",
                ],
                "layout": "wrist_image_left top-wide; exterior_image_1_left and exterior_image_2_left bottom-left/right",
                "center_crop_scale": DROID_CENTER_CROP_SCALE,
            },
        }
        _atomic_write_json(output_dir / "summary.json", summary)
        if "success_rate" in task_summary:
            print(
                "DROID task metrics: "
                + json.dumps(
                    {
                        key: task_summary[key]
                        for key in (
                            "pick_success_count",
                            "pick_success_rate",
                            "place_success_count",
                            "place_success_rate",
                            "failed_at_pick_count",
                            "failed_at_place_count",
                            "dropped_after_pick_count",
                            "receptacle_support_episode_count",
                            "failure_reason_counts",
                            "success_count",
                            "success_rate",
                        )
                        if key in task_summary
                    },
                    sort_keys=True,
                )
            )
    finally:
        client.close()
        env.close()
        simulation_app.close()
