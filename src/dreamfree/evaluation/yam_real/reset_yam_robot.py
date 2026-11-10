#!/usr/bin/env python3
"""Move both YAM arms to the ABC profile's home pose; optionally hold until Ctrl-C."""

import argparse
import json
import os
import signal
import sys
import time
from contextlib import ExitStack

import numpy as np

if __package__:
    from .hardware import add_station_arguments, load_profile, station_lock
else:
    from hardware import add_station_arguments, load_profile, station_lock


def read_joints(name, robot):
    if not robot.motor_chain.running or not robot._server_thread.is_alive():
        raise RuntimeError(f"YAM {name}: i2rt motor/control thread stopped.")
    state = np.asarray(robot.get_joint_pos(), dtype=np.float64).copy()
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError(f"YAM {name}: expected seven finite joint/gripper positions.")
    return state


def close_robot(robot):
    """Stop the driver and turn off every motor without ABC's extra move to zero."""
    errors = []
    try:
        robot.close()
    except Exception as exc:
        errors.append(str(exc))
    finally:
        chain = robot.motor_chain
        chain.running = False
        # Match ABC's delay for the motor loop's final iteration before motor-off.
        time.sleep(0.1)
        with chain.command_lock:
            for motor_id, _ in chain.motor_list:
                try:
                    chain.motor_interface.motor_off(motor_id)
                except Exception as exc:
                    errors.append(f"motor {motor_id}: {exc}")
    if errors:
        raise RuntimeError("YAM shutdown failed: " + "; ".join(errors))


def reset_to_home(profile, args, open_robot):
    """Initialize, interpolate both arms together, verify the pose, and release on exit."""
    with ExitStack() as cleanup:
        robots, targets, initial = {}, {}, {}
        for name, config in profile.robots.items():
            robot = open_robot(config.follower)
            cleanup.callback(close_robot, robot)
            robots[name] = robot
            target = np.asarray(config.init_q, dtype=np.float64)
            info = robot.get_robot_info()
            limits = np.asarray(info["joint_limits"])
            if limits.shape != (6, 2) or not np.isfinite(limits).all():
                raise ValueError(f"YAM {name}: i2rt must provide six finite joint-limit pairs.")
            if np.any(target[:6] < limits[:, 0]) or np.any(target[:6] > limits[:, 1]):
                raise ValueError(
                    f"YAM {name}: configured home pose exceeds i2rt joint limits; refusing clipped motion."
                )
            robot.update_kp_kd(np.asarray(info["kp"]) * args.kp_scale, np.asarray(info["kd"]))
            targets[name] = target
        for name, robot in robots.items():
            initial[name] = read_joints(name, robot)
        steps = max(1, int(np.ceil(args.duration * 30)))
        for step in range(steps + 1):
            for name, robot in robots.items():
                read_joints(name, robot)
                alpha = step / steps
                robot.command_joint_pos((1 - alpha) * initial[name] + alpha * targets[name])
            time.sleep(args.duration / steps)
        deadline = time.perf_counter() + args.settle_timeout
        while True:
            errors = {
                name: np.abs(read_joints(name, robot) - targets[name])
                for name, robot in robots.items()
            }
            if all(
                np.max(error[:6]) <= args.joint_tolerance and error[6] <= args.gripper_tolerance
                for error in errors.values()
            ):
                break
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    "Home pose was not reached: "
                    + "; ".join(
                        f"{name}: joint error={error[:6].max():.3f} rad, gripper error={error[6]:.3f}"
                        for name, error in errors.items()
                    )
                )
            time.sleep(1 / 30)
        print("YAM home pose reached and verified.", flush=True)
        if args.hold:
            print(
                "Holding home pose. Ctrl-C releases the motors; stop this tool before a rollout.",
                flush=True,
            )
            while True:
                for name, robot in robots.items():
                    read_joints(name, robot)
                time.sleep(0.1)
        else:
            print(
                "Releasing the motors; the arms will no longer actively hold the home pose.",
                flush=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_station_arguments(parser)
    parser.add_argument(
        "--init-q",
        default=os.environ.get("DEPLOY_INIT_Q"),
        help="JSON list of 14 home values, left arm then right.",
    )
    parser.add_argument(
        "--kp-scale",
        default=os.environ.get("DEPLOY_KP_SCALE", "1.0"),
        help="Positive JSON scalar or seven kp multipliers.",
    )
    parser.add_argument(
        "--duration", type=float, default=2, help="Seconds for the interpolated reset motion."
    )
    parser.add_argument("--settle-timeout", type=float, default=5)
    parser.add_argument(
        "--joint-tolerance", type=float, default=0.1, help="Maximum final joint error in radians."
    )
    parser.add_argument(
        "--gripper-tolerance",
        type=float,
        default=0.1,
        help="Maximum final normalized gripper error.",
    )
    parser.add_argument(
        "--hold", action="store_true", help="Keep the drivers active at home until Ctrl-C."
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip confirmation before calibration and reset."
    )
    args = parser.parse_args(argv)
    values = (args.duration, args.settle_timeout, args.joint_tolerance, args.gripper_tolerance)
    if any(not np.isfinite(value) or value <= 0 for value in values):
        parser.error("Duration, timeout, and tolerances must be finite and positive.")
    args.kp_scale = np.asarray(json.loads(args.kp_scale), dtype=np.float64)
    if (
        args.kp_scale.shape not in ((), (7,))
        or not np.isfinite(args.kp_scale).all()
        or np.any(args.kp_scale <= 0)
    ):
        parser.error("--kp-scale must be a positive scalar or seven finite positive values.")
    profile = load_profile(args)
    for name, config in profile.robots.items():
        print(
            f"{name}: CAN={config.follower.channel}, gripper={config.follower.gripper_type}, home={config.init_q}"
        )
    if args.check_only:
        return 0
    with station_lock(args.ipc_dir):
        if not args.yes:
            if not sys.stdin.isatty():
                raise ValueError("Reset needs an interactive terminal, or explicit --yes.")
            input(
                "ENTER to initialize/calibrate the grippers and move both arms home; Ctrl-C to cancel > "
            )
        from deploy.robot.followers.yam_follower import YAMFollowerNode, _patch_gripper_calibration
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        _patch_gripper_calibration()

        def open_robot(config):
            gripper = GripperType[YAMFollowerNode._GRIPPER_TYPE_MAP[config.gripper_type]]
            return get_yam_robot(
                channel=config.channel, gripper_type=gripper, zero_gravity_mode=False
            )

        def stop(_signum, _frame):
            raise KeyboardInterrupt

        previous_handler = signal.signal(signal.SIGTERM, stop)
        try:
            reset_to_home(profile, args, open_robot)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("YAM reset stopped.")
        raise SystemExit(130)
