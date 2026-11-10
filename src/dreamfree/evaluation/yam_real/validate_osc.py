"""Reproducible, CAN-free MuJoCo tracking and workstation CPU timing validation.

Run with python -m dreamfree.evaluation.yam_real.validate_osc --output <report.json>.
The plant uses pinned rigid-body dynamics without contacts or hardware latency.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import mujoco
import numpy as np

from .osc_control import CONTRACT_PATH, ArmOsc, OscConfig, WristDynamics, encode_pose
from .osc_servo import OscServo

TRAINING_MODEL = (
    Path(os.environ.get("ABC_ROOT", Path(__file__).resolve().parents[4] / "third_party/abc"))
    / "abc_sim/models/assets/i2rt_yam/yam.xml"
)
HOME = np.array([0.1, 0.6, 1.2, -0.5, 0.1, 0.2, 0.7])


def make_dynamics(filename="yam_4310_linear.xml"):
    contract = json.loads(CONTRACT_PATH.read_text())
    return WristDynamics(
        contract["models"]["robot_models/yam/" + filename],
        model_file=filename,
        gravity=[0, 0, -9.81],
        training_model=TRAINING_MODEL,
    )


class SimClock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        return self.value

    def advance(self, dt):
        self.value += dt


class MujocoArm:
    """Test transport: rigid-body torque plant plus a simple gripper motor inertia."""

    def __init__(self, dynamics, clock, period=0.01):
        self.model = dynamics.model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = HOME[:6]
        self.clock, self.period = clock, period
        self.grip_pos, self.grip_vel, self.span = -0.7 * 6.57, 0.0, -6.57
        self.enabled, self.sent = False, []
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        return (
            np.r_[self.data.qpos, self.grip_pos / self.span],
            np.r_[self.data.qvel, self.grip_vel / self.span],
            np.full(7, self.clock()),
        )

    def enable(self):
        self.enabled = True
        return self.state()

    def exchange(self, torque):
        if not self.enabled:
            raise RuntimeError("Simulation driver disabled.")
        self.sent.append(np.asarray(torque).copy())
        steps = round(self.period / self.model.opt.timestep)
        for _ in range(steps):
            self.data.qfrc_applied[:] = torque[:6]
            mujoco.mj_step(self.model, self.data)
            dt = self.model.opt.timestep
            self.grip_vel += dt * (torque[6] - 0.1 * self.grip_vel) / 0.02
            self.grip_pos += dt * self.grip_vel
        return self.state()

    def disable(self):
        self.enabled = False
        return []


def run_tracking(seconds=4.0, *, filename="yam_4310_linear.xml"):
    cfg, clock = OscConfig(), SimClock()
    controllers = [ArmOsc(make_dynamics(filename), cfg, [0, -6.57]) for _ in range(2)]
    drivers = [MujocoArm(make_dynamics(filename), clock) for _ in range(2)]
    servo = OscServo(controllers, drivers, cfg, clock=clock)
    servo.start()
    requested = servo.target.copy()
    # A reachable target generated only for this test. Runtime OSC never solves IK.
    for i, arm in enumerate(controllers):
        q = HOME[:6] + np.array([0.025, -0.02, 0.02, 0.015, -0.01, 0.02]) * (1 if i == 0 else -1)
        pose = arm.dynamics.fk.forward(q)
        requested[i * 10 : (i + 1) * 10] = encode_pose(pose[:3, 3], pose[:3, :3], 0.72)
    rows, compute_ms, sequence = [], [], 0
    next_policy_update = 0.0
    for step in range(round(seconds * cfg.rate_hz)):
        clock.advance(1 / cfg.rate_hz)
        t = step / cfg.rate_hz
        if t >= next_policy_update:
            sequence += 1
            servo.submit(requested, sequence, clock(), clock())
            next_policy_update += 1 / 30
        start = time.perf_counter()
        snapshot = servo.tick()
        compute_ms.append((time.perf_counter() - start) * 1000)
        rows.append(snapshot)
    servo.stop()
    errors = np.array(
        [
            [r["position_error_m"], r["orientation_error_rad"]]
            for row in rows
            for r in row["reports"]
        ]
    )
    settled = errors[-round(cfg.rate_hz) * 2 :]
    torques = np.array([row["reports"][i]["torque"] for row in rows for i in range(2)])
    return dict(
        model=filename,
        simulated_seconds=seconds,
        two_arm_servo_ticks=len(rows),
        target_updates=sequence,
        final_position_error_m=[r["position_error_m"] for r in rows[-1]["reports"]],
        final_orientation_error_rad=[r["orientation_error_rad"] for r in rows[-1]["reports"]],
        peak_position_error_m=float(errors[:, 0].max()),
        peak_orientation_error_rad=float(errors[:, 1].max()),
        settled_rms_position_error_m=float(np.sqrt(np.mean(settled[:, 0] ** 2))),
        settled_rms_orientation_error_rad=float(np.sqrt(np.mean(settled[:, 1] ** 2))),
        maximum_abs_torque_nm=np.max(np.abs(torques), axis=0).tolist(),
        two_arm_compute_with_simulation_ms=dict(
            p50=float(np.median(compute_ms)),
            p99=float(np.percentile(compute_ms, 99)),
            maximum=float(max(compute_ms)),
        ),
        config=cfg.as_dict(),
        hardware_validation=False,
    )


def run_wall_clock(seconds=2.0):
    cfg = OscConfig()
    controllers = [ArmOsc(make_dynamics(), cfg, [0, -6.57]) for _ in range(2)]
    drivers = [MujocoArm(make_dynamics(), time.monotonic) for _ in range(2)]
    servo = OscServo(controllers, drivers, cfg)
    servo.start()
    try:
        # No target updates: models a GPU inference pause while the local loop runs.
        while time.monotonic() - servo.started < seconds:
            started = time.monotonic()
            servo.tick(started)
            time.sleep(max(0.0, started + 1 / cfg.rate_hz - time.monotonic()))
        return servo.timing()
    finally:
        servo.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = dict(
        validation="offline_simulation_only",
        mujoco_version=mujoco.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
        model_provenance=json.loads(CONTRACT_PATH.read_text())["source_sha256"],
        tracking=[run_tracking(filename=name) for name in ("yam_4310_linear.xml", "yam.xml")],
        scheduled_loop_with_simulated_io=run_wall_clock(),
        limitations=[
            "No CAN latency, motor quantization, contacts, payload changes or hardware tuning.",
            "Timing is this server CPU, not the robot workstation.",
            "Same-model plant/controller; does not validate physical mass/friction parameters.",
        ],
    )
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
