"""Explicit synchronous adapter for pinned I2RT; never start its motor/server threads.

Construction of the numerical controller and source validation is hardware-free.
Only open_pinned_can() opens CAN; only SynchronousYamDriver.enable() enables motors.
"""

import hashlib
import importlib.util
import json
import threading
import time
from pathlib import Path

import numpy as np

if __package__:
    from .osc_control import CONTRACT_PATH, MODEL_FILES, OscConfig, OscFault, WristDynamics, vector
else:
    from osc_control import CONTRACT_PATH, MODEL_FILES, OscConfig, OscFault, WristDynamics, vector


def validate_sources():
    spec = importlib.util.find_spec("i2rt")
    if spec is None or spec.origin is None:
        raise ValueError(
            "OSC requires ABC's pinned I2RT installation; no driver is installed automatically."
        )
    root = Path(spec.origin).parent
    contract = json.loads(CONTRACT_PATH.read_text())
    for relative, digest in contract["source_sha256"].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(
                f"OSC I2RT source/model contract mismatch: {path}; expected {contract['revision']}."
            )
    return root


def base_gravity(quaternion):
    w, x, y, z = vector(quaternion, 4, "base quaternion wxyz")
    if not np.isclose(np.linalg.norm([w, x, y, z]), 1.0, atol=1e-6):
        raise ValueError("Base mounting quaternion must be normalized.")
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rotation.T @ np.array([0.0, 0.0, -9.81])


def prepare_setup(path, profile, training_model, *, require_hardware):
    """Validate configuration, installed sources and both dynamics models without CAN."""
    setup = json.loads(Path(path).read_text())
    if (
        set(setup) != {"schema_version", "controller", "hardware", "arms"}
        or setup["schema_version"] != 1
    ):
        raise ValueError(
            "OSC config requires schema_version=1 and controller, hardware, arms sections only."
        )
    cfg = OscConfig(**setup["controller"])
    hardware = setup["hardware"]
    required = {
        "gains_and_limits_validated",
        "external_estop_verified",
        "motor_watchdog_verified",
        "watchdog_timeout_s",
        "station_model_verified",
    }
    if set(hardware) != required:
        raise ValueError(f"OSC hardware section must contain {sorted(required)}.")
    if require_hardware:
        for key in required - {"watchdog_timeout_s"}:
            if hardware[key] is not True:
                raise ValueError(
                    f"OSC hardware execution requires {key}=true after physical validation; example gains are unvalidated."
                )
        timeout = hardware["watchdog_timeout_s"]
        if (
            not isinstance(timeout, (int, float))
            or not np.isfinite(timeout)
            or not 0 < timeout <= cfg.max_period
        ):
            raise ValueError(
                "Verify a firmware motor watchdog no longer than max_period; I2RT cannot configure or verify it."
            )
    if set(setup["arms"]) != {"left", "right"}:
        raise ValueError("OSC requires left and right arm configurations.")
    channels = [profile.robots[side].follower.channel for side in ("left", "right")]
    if (
        any(not isinstance(channel, str) or not channel for channel in channels)
        or len(set(channels)) != 2
    ):
        raise ValueError("OSC left and right arms require distinct, nonempty CAN channels.")
    root, dynamics = validate_sources(), []
    for side in ("left", "right"):
        arm, robot = setup["arms"][side], profile.robots[side]
        if set(arm) != {
            "gripper_type",
            "gripper_limits",
            "motor_offsets",
            "motor_directions",
            "base_quaternion_wxyz",
            "payload_kg",
            "calibration_verified",
        }:
            raise ValueError(f"OSC {side} arm configuration has missing or unknown fields.")
        kind = arm["gripper_type"]
        if kind not in MODEL_FILES or kind != robot.follower.gripper_type:
            raise ValueError(
                f"OSC {side} gripper must match the actual supported ABC profile gripper."
            )
        endpoints = vector(arm["gripper_limits"], 2, f"{side} calibrated gripper endpoints")
        if abs(np.diff(endpoints)[0]) < 0.1:
            raise ValueError(
                f"OSC {side} requires distinct calibrated endpoints; calibration is never run here."
            )
        offsets = vector(arm["motor_offsets"], 7, f"{side} motor offsets")
        directions = vector(arm["motor_directions"], 7, f"{side} motor directions")
        if np.any(directions != 1) or np.any(
            np.abs(offsets / (2 * np.pi) - np.round(offsets / (2 * np.pi))) > 1e-6
        ):
            raise ValueError(
                "OSC supports pinned YAM +1 motor directions and explicit 2*pi wrap offsets only."
            )
        if (
            np.any(cfg.joint_lower + offsets[:6] < -12.5)
            or np.any(cfg.joint_upper + offsets[:6] > 12.5)
            or np.any(np.abs(endpoints + offsets[6]) > 12.5)
        ):
            raise ValueError(
                f"OSC {side} calibrated coordinates cross the motor wire position range; automatic unwrapping is unsupported."
            )
        if not np.allclose(arm["base_quaternion_wxyz"], robot.root_ori, atol=1e-6, rtol=0):
            raise ValueError(f"OSC {side} mounting must agree with the verified ABC root_ori.")
        if arm["payload_kg"] != 0:
            raise ValueError(
                "OSC pinned lumped-inertia models support no added payload; validate a new model explicitly."
            )
        if require_hardware and arm["calibration_verified"] is not True:
            raise ValueError(
                f"OSC {side} needs previously verified motor offsets and gripper endpoints; no auto-calibration."
            )
        if kind == "linear_3507" and cfg.torque_limit[6] > 5:
            raise ValueError("DM3507 gripper torque exceeds its protocol range.")
        filename = MODEL_FILES[kind]
        dynamics.append(
            WristDynamics(
                (root / "robot_models/yam" / filename).read_text(),
                model_file=filename,
                gravity=base_gravity(arm["base_quaternion_wxyz"]),
                training_model=training_model,
            )
        )
    metadata = dict(
        controller="osc",
        history_contract="osc_accepted_cartesian_targets_v1",
        config=cfg.as_dict(),
        arms=[d.metadata for d in dynamics],
        driver_revision=json.loads(CONTRACT_PATH.read_text())["revision"],
        driver_changes="none; repository-owned synchronous transport adapter",
        motor_pd_gains="all_zero_including_gripper",
        shutdown="motor_off_no_homing",
        target_timeout=cfg.target_timeout,
        hardware_validation=hardware,
        station_calibration=setup["arms"],
        motor_channels={side: profile.robots[side].follower.channel for side in ("left", "right")},
        collision_checking=False,
    )
    return cfg, setup, dynamics, metadata


def bounded_exchange(bus, message, expected_id, timeout, *, clock=time.monotonic):
    """One bounded request/response, with no I2RT retries or automatic fault clearing."""
    deadline = clock() + timeout
    # Discard already queued responses. CAN has no command sequence number: a
    # delayed matching frame arriving later cannot be proven fresh by software.
    for _ in range(64):
        if bus.recv(timeout=0) is None:
            break
        if clock() >= deadline:
            raise OscFault("OSC CAN receive queue did not drain within deadline.")
    else:
        raise OscFault("OSC CAN receive queue overflow.")
    sent_wall = time.time()
    remaining = deadline - clock()
    if remaining <= 0:
        raise OscFault("OSC CAN request deadline elapsed before send.")
    bus.send(message, timeout=remaining)
    while clock() < deadline:
        response = bus.recv(timeout=max(0.0, deadline - clock()))
        if response is not None and response.arbitration_id == expected_id:
            # SocketCAN kernel timestamps are wall-clock receipt times.
            if getattr(response, "timestamp", 0.0) < sent_wall:
                continue
            if clock() > deadline:
                break
            return response
    raise OscFault(f"OSC CAN feedback timeout for arbitration id {expected_id}; no retry.")


def open_pinned_can(channel, timeout):
    """Open a bus only. Does not construct MotorChainRobot or DMChainCanInterface."""
    import can
    from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface
    from i2rt.motor_drivers.utils import ReceiveMode

    class BoundedInterface(DMSingleMotorCanInterface):
        def _send_message_get_response(self, id, motor_id, data, max_retry=5, expected_id=None):
            expected_id = (
                self.receive_mode.get_receive_id(motor_id) if expected_id is None else expected_id
            )
            return bounded_exchange(
                self.bus,
                can.Message(arbitration_id=id, data=data, is_extended_id=False),
                expected_id,
                timeout,
            )

    return BoundedInterface(
        channel=channel, receive_mode=ReceiveMode.p16, use_buffered_reader=False
    )


def bounded_motor_torque(requested, bound, previous, step, protocol_max):
    """Choose a 12-bit torque code inside magnitude/slew bounds before encoding.

    The stock encoder floors floats. An interior value within the selected code
    bin avoids rounding into its neighbor; diagnostics report the decoded torque.
    """
    quantum = 2 * protocol_max / 4095
    lower, upper = -bound, bound
    if previous is not None:
        lower, upper = max(lower, previous - step), min(upper, previous + step)
    first = int(np.ceil((lower + protocol_max) / quantum - 1e-10))
    last = int(np.floor((upper + protocol_max) / quantum + 1e-10))
    if first > last:
        raise OscFault(
            "No representable motor torque satisfies the configured magnitude/slew bounds."
        )
    code = int(np.clip(np.round((requested + protocol_max) / quantum), first, last))
    decoded = code * quantum - protocol_max
    encoded_argument = (code + 0.25) * quantum - protocol_max if code < 4095 else protocol_max
    return encoded_argument, decoded


class SynchronousYamDriver:
    """The calling servo thread exclusively owns enable, torque, feedback and disable.

    All seven onboard kp/kd values are zero. Gripper PD is computed in software,
    in motor radians, and shares explicit torque and torque-rate bounds.
    """

    def __init__(self, interface, arm, config, *, clock=time.monotonic):
        self.interface, self.arm, self.clock = interface, arm, clock
        self.config = config
        self.owner = threading.get_ident()
        self.offsets = vector(arm["motor_offsets"], 7, "motor offsets")
        self.signs = vector(arm["motor_directions"], 7, "motor directions")
        self.endpoints = vector(arm["gripper_limits"], 2, "gripper endpoints")
        self.motor_types = (
            ["DM4340"] * 3
            + ["DM4310"] * 3
            + ["DM3507" if arm["gripper_type"] == "linear_3507" else "DM4310"]
        )
        self.enabled = False
        self.closed = False
        self.last_wire_torque = np.zeros(7)
        self.last_send = None

    def _check_owner(self):
        if threading.get_ident() != self.owner:
            raise OscFault("OSC motor I/O called by a competing thread.")
        if self.closed:
            raise OscFault("OSC motor adapter is closed.")

    def enable(self):
        self._check_owner()
        if self.enabled:
            raise OscFault("OSC motors already enabled.")
        # Do not use upstream motor_on(): it clears faults/retries indefinitely.
        try:
            for index, motor_type in enumerate(self.motor_types, 1):
                message = self.interface._send_message_get_response(index, index, [255] * 7 + [252])
                self.interface.parse_recv_message(message, motor_type)
                self.interface.set_control(index, motor_type, 0.0, 0.0, 0.0, 0.0, 0.0)
            self.enabled = True
            return self.exchange(np.zeros(7))
        except BaseException as exc:
            errors = self.disable()
            if errors:
                raise OscFault(
                    f"Enable failed ({exc}); motor-off delivery failed: {errors}"
                ) from exc
            raise

    def exchange(self, torques):
        self._check_owner()
        if not self.enabled:
            raise OscFault("OSC torque command before explicit enable.")
        torques = vector(torques, 7, "motor torque")
        if np.any(np.abs(torques) > self.config.torque_limit + 1e-10):
            raise OscFault("OSC transport received an over-limit torque.")
        q, dq, stamps = [], [], []
        for index, motor_type in enumerate(self.motor_types):
            started = self.clock()
            elapsed = (
                self.config.max_period
                if self.last_send is None
                else started - self.last_send[index]
            )
            previous = None if self.last_send is None else self.last_wire_torque[index]
            protocol_max = 28 if motor_type == "DM4340" else 5 if motor_type == "DM3507" else 10
            argument, decoded = bounded_motor_torque(
                torques[index],
                self.config.torque_limit[index],
                previous,
                self.config.torque_rate[index] * max(0.0, min(elapsed, self.config.max_period)),
                protocol_max,
            )
            feedback = self.interface.set_control(
                index + 1, motor_type, 0.0, 0.0, 0.0, 0.0, argument * self.signs[index]
            )
            self.last_wire_torque[index] = decoded
            if feedback.id != index + 1 or feedback.error_code != "0x1":
                raise OscFault(f"OSC motor {index + 1} feedback id/error mismatch.")
            q.append((feedback.position - self.offsets[index]) * self.signs[index])
            dq.append(feedback.velocity * self.signs[index])
            stamps.append(started)  # Conservative age includes the full transaction.
        self.last_send = np.asarray(stamps)
        q, dq = vector(q, 7, "motor feedback"), vector(dq, 7, "motor velocity")
        span = self.endpoints[1] - self.endpoints[0]
        q[6], dq[6] = (q[6] - self.endpoints[0]) / span, dq[6] / span
        if not -self.config.gripper_margin <= q[6] <= 1 + self.config.gripper_margin:
            raise OscFault("OSC gripper feedback exceeds calibrated aperture range.")
        return q, dq, np.asarray(stamps)

    def disable(self):
        """Best-effort bounded motor-off for every motor; return every failure.

        Motor-off overrides slew limits. No homing, gravity hold, or stale torque
        republishing follows. CAN failure requires the independent firmware watchdog.
        """
        if self.closed:
            return []
        self._check_owner()
        errors = []
        try:
            for index in range(1, 8):
                try:
                    self.interface.motor_off(index)
                except Exception as exc:
                    errors.append(f"motor {index}: {type(exc).__name__}: {exc}")
        finally:
            self.enabled, self.closed = False, True
            try:
                self.interface.close()
            except Exception as exc:
                errors.append(f"CAN close: {exc}")
        return errors
