"""ABC camera/policy client with joint/IK followers or a continuous OSC servo."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dreamfree.evaluation.action_protocol import IssuedActionHistoryBuffer
from dreamfree.evaluation.yam_real.common import (
    ABC_REVISION,
    EefTargetRejected,
    executable_chunk,
    validate_joint_step,
    validate_observation,
    validate_profile,
    validate_server,
    write_json,
)
from dreamfree.evaluation.yam_real.image_preprocessing import (
    RAW_IMAGE_BYTES,
    ClientImagePreprocessor,
)
from dreamfree.evaluation.yam_real.policy_rpc import PolicyRpcClient
from dreamfree.evaluation.yam_real.video_recording import CameraVideoRecorder, check_video_backend


def connect(args):
    client = PolicyRpcClient(args.remote_host, args.remote_port, timeout_sec=args.rpc_timeout)
    try:
        info = client.info()
        validate_server(
            info,
            expected_checkpoint=args.expected_checkpoint,
            expected_variant=args.expected_variant,
            replan_steps=args.open_loop_horizon,
            action_space=args.action_space,
        )
        if (
            args.expected_action_noise_seed is not None
            and info.get("seed") != args.expected_action_noise_seed
        ):
            raise ValueError(
                "Policy server's action-noise seed differs from --expected-action-noise-seed."
            )
        client.image_preprocessor = None
        image_info = dict(
            mode="server", transport_dtype="uint8", image_payload_bytes=RAW_IMAGE_BYTES
        )
        if args.image_preprocessing == "client":
            client.image_preprocessor = ClientImagePreprocessor(
                client,
                info,
                transport_dtype=args.image_transport_dtype,
                threads=args.image_preprocessing_threads,
            )
            image_info = client.image_preprocessor.metadata
        info["client_image_preprocessing"] = image_info
        return client, info
    except BaseException:
        client.close()
        raise


def reset_episode(env, client, args, episode_index):
    episode_seed = args.episode_seed + episode_index
    reset = client.reset(episode_seed=episode_seed)
    if reset.get("episode_seed") != episode_seed:
        raise ValueError("Server did not acknowledge the requested episode seed.")
    observation, _ = env.reset()
    validate_observation(observation)
    return observation


def execute_episode(
    env,
    client,
    info,
    args,
    episode_index,
    *,
    confirm=None,
    save=None,
    recorder=None,
    eef_controller=None,
    initial_observation=None,
):
    """Record only accepted policy commands; resets and retry holds never enter history."""
    history_horizon = info["action_history_horizon"]
    action_space = info.get("action_space", "joint")
    osc = getattr(args, "eef_controller", "ik") == "osc"
    if (action_space == "eef") != (eef_controller is not None or osc) or (
        osc and eef_controller is not None
    ):
        raise ValueError("EEF execution requires exactly its selected IK or OSC adapter.")
    history = (
        IssuedActionHistoryBuffer(horizon=history_horizon, action_dim=info["action_dim"])
        if history_horizon
        else None
    )
    episode_seed = args.episode_seed + episode_index
    retry_limit = getattr(args, "eef_rejection_retries", 0)
    consecutive_rejections = 0
    result = dict(
        episode=episode_index,
        episode_seed=episode_seed,
        instruction=args.instruction,
        debug=args.debug,
        status="running",
        steps=[],
        replans=[],
        issued_actions=[],
        eef_rejection_retries=retry_limit,
        rejections=[],
        video=dict(enabled=recorder is not None),
    )
    if eef_controller is not None:
        result.update(eef_control=eef_controller.contract, issued_policy_actions=[])
    if osc:
        result.update(
            eef_control=env.osc_contract,
            issued_policy_actions=[],
            issued_target_contract="osc_accepted_cartesian_targets_v1",
        )
    preprocessor = getattr(client, "image_preprocessor", None)
    if preprocessor is not None:
        preprocessor.live_validated = False
        result["image_preprocessing"] = preprocessor.metadata
    try:
        if recorder is not None:
            recorder.start()
        observation = initial_observation
        if observation is None:
            observation = reset_episode(env, client, args, episode_index)
        if recorder is not None:
            recorder.check()
        state = validate_observation(observation)
        previous = state.copy()
        started = time.perf_counter()
        result["started_monotonic_seconds"] = started
        while len(result["steps"]) < args.max_timesteps:
            if recorder is not None:
                recorder.check()
            state = validate_observation(observation)
            request = history.add_to_observation(observation) if history else observation
            image_timing = dict(client_preprocess_seconds=0.0, image_validation_seconds=0.0)
            if preprocessor is not None:
                request = preprocessor.prepare(request)
                image_timing = dict(preprocessor.last_timing)
            rpc_start = time.perf_counter()
            response = client.infer(request, args.instruction)
            rpc_seconds = time.perf_counter() - rpc_start
            actions, clipped = executable_chunk(
                response["actions"], info["action_horizon"], action_space=action_space
            )
            count = min(args.open_loop_horizon, args.max_timesteps - len(result["steps"]))
            if args.print_predicted_actions:
                print(actions[:count], flush=True)
            if not result["steps"] and confirm is not None and eef_controller is None and not osc:
                confirm(state, actions[:count])
            result["replans"].append(
                dict(
                    step=len(result["steps"]),
                    rpc_seconds=rpc_seconds,
                    inference_seconds=response["inference_seconds"],
                    gripper_clipped_values=clipped,
                    predicted_actions=actions.tolist(),
                    observed_state=state.tolist(),
                    **image_timing,
                )
            )
            deadline = time.perf_counter()
            rejected = False
            for chunk_index, policy_action in enumerate(actions[:count]):
                ik_report = None
                osc_report = None
                try:
                    if eef_controller is not None:
                        ik_started = time.perf_counter()
                        command, issued_policy_action, ik_report = eef_controller.command(
                            policy_action, previous=previous, measured=state
                        )
                        ik_seconds = time.perf_counter() - ik_started
                        if not result["steps"] and confirm is not None:
                            confirm(state, command[None, :])
                    else:
                        command = issued_policy_action = policy_action
                    if not osc:
                        try:
                            validate_joint_step(command, previous, args.max_joint_delta)
                        except ValueError as exc:
                            if eef_controller is None:
                                raise
                            raise EefTargetRejected(str(exc)) from exc
                    time.sleep(max(0.0, deadline - time.perf_counter()))
                    if recorder is not None:
                        recorder.check()
                    issued_at = time.perf_counter()
                    if osc:
                        state, issued_policy_action, osc_report = env.step_eef(policy_action)
                        command = None  # OSC has no joint-position target.
                    else:
                        state = np.asarray(env.step_state(command), dtype=np.float32)
                except EefTargetRejected as exc:
                    consecutive_rejections += 1
                    retry = consecutive_rejections <= retry_limit
                    result["rejections"].append(
                        dict(
                            step=len(result["steps"]),
                            replan=len(result["replans"]) - 1,
                            chunk_index=chunk_index,
                            controller="osc" if osc else "ik",
                            error=str(exc),
                            consecutive_rejections=consecutive_rejections,
                            retry_scheduled=retry,
                            policy_action=policy_action.tolist(),
                            observed_state=state.tolist(),
                        )
                    )
                    if not retry:
                        raise EefTargetRejected(
                            f"EEF target rejected; retry limit exhausted "
                            f"({retry_limit} retries at step {len(result['steps'])}): {exc}"
                        ) from exc
                    print(
                        f"EEF target rejected at step={len(result['steps'])}; "
                        f"requesting fresh prediction {consecutive_rejections}/{retry_limit}: {exc}",
                        flush=True,
                    )
                    rejected = True
                    break  # Discard every remaining action from this rejected chunk.
                if state.shape != (14,) or not np.isfinite(state).all():
                    raise ValueError(
                        "ABC follower returned an invalid 14D state after a command; stopping episode."
                    )
                consecutive_rejections = 0
                if not args.debug:
                    if history:
                        history.record_issued_action(issued_policy_action)
                    if not osc:
                        result["issued_actions"].append(command.tolist())
                    if eef_controller is not None or osc:
                        result["issued_policy_actions"].append(issued_policy_action.tolist())
                result["steps"].append(
                    dict(
                        time_seconds=issued_at - started,
                        command=command.tolist() if command is not None else None,
                        observed_state=state.tolist(),
                        step_seconds=time.perf_counter() - issued_at,
                    )
                )
                if eef_controller is not None:
                    result["steps"][-1].update(
                        policy_action=policy_action.tolist(),
                        issued_policy_action=issued_policy_action.tolist(),
                        ik=ik_report,
                        ik_seconds=ik_seconds,
                    )
                if osc:
                    result["steps"][-1].update(
                        policy_action=policy_action.tolist(),
                        issued_policy_action=issued_policy_action.tolist(),
                        osc=osc_report,
                    )
                else:
                    previous = command.copy()
                # Never send catch-up bursts after a slow follower response.
                deadline = issued_at + 1 / 30
            if recorder is not None:
                result["video"] = recorder.metadata()
            if save:
                save(result)
            print(
                f"episode={episode_index} steps={len(result['steps'])} "
                f"preprocess={image_timing['client_preprocess_seconds'] * 1000:.1f}ms "
                f"rpc={rpc_seconds * 1000:.1f}ms model={response['inference_seconds'] * 1000:.1f}ms",
                flush=True,
            )
            if len(result["steps"]) < args.max_timesteps:
                time.sleep(max(0.0, deadline - time.perf_counter()))
                # IK reuses command-triggered feedback; OscEnv fetches fresh servo
                # feedback after reading the cameras, including during inference gaps.
                observation = env.get_obs(state=state)
                if rejected and not osc:
                    # ABC publishes state only after a command. Reissue the last
                    # accepted hold after camera capture to obtain current joints;
                    # this recovery hold is logged separately, never as policy history.
                    observation["state"] = np.asarray(env.step_state(previous), dtype=np.float32)
                    result["rejections"][-1]["recovery_hold_command"] = previous.tolist()
                if rejected:
                    state = validate_observation(observation)
                    result["rejections"][-1]["retry_observed_state"] = state.tolist()
        result["status"] = "complete"
    except BaseException as exc:
        result.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
        except BaseException as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if recorder is not None:
                result["video"] = recorder.metadata()
            if save:
                save(result)
    return result


def prepare_eef_controller(args, profile):
    if args.action_space != "eef" or getattr(args, "eef_controller", "ik") == "osc":
        return None
    if __package__:
        from .eef_control import controller_from_args, validate_driver
    else:
        from dreamfree.evaluation.yam_real.eef_control import controller_from_args, validate_driver
    driver = validate_driver()
    controller = controller_from_args(args)
    home = np.concatenate([profile.robots[side].init_q for side in ("left", "right")])
    arm_home = home.reshape(2, 7)[:, :6]
    limits = np.asarray(driver["joint_limits"])
    if np.any(arm_home < limits[:, 0]) or np.any(arm_home > limits[:, 1]):
        raise ValueError(
            "Configured home joints exceed the pinned driver's limits; refusing EEF hardware startup."
        )
    controller.command(controller.kinematics.encode_command(home), previous=home, measured=home)
    return controller


def prepare_osc(args, profile):
    if getattr(args, "eef_controller", "ik") != "osc":
        return None
    if __package__:
        from .osc_driver import prepare_setup
    else:
        from dreamfree.evaluation.yam_real.osc_driver import prepare_setup

    _, _, _, contract = prepare_setup(
        args.osc_config,
        profile,
        Path(args.abc_root) / "abc_sim/models/assets/i2rt_yam/yam.xml",
        require_hardware=not args.check_only,
    )
    contract["rejection_retries"] = getattr(args, "eef_rejection_retries", 0)
    return contract


def run_rollout(args, run_config):
    from deploy.robot.config import get_i2rt_config
    from deploy.robot.gym.policy_safety import prompt_first_action_safety_check
    from deploy.robot.gym.yam_env import YAMEnv

    profile = get_i2rt_config()
    validate_profile(profile)
    client, info = connect(args)
    env = None
    try:
        eef_controller = prepare_eef_controller(args, profile)
        osc_contract = prepare_osc(args, profile)
        if osc_contract is not None:
            if __package__:
                from .osc_client import OscEnv
            else:
                from dreamfree.evaluation.yam_real.osc_client import OscEnv
            env = OscEnv(profile, args, osc_contract)
        else:
            env = YAMEnv(
                profile,
                prompt=args.instruction,
                execute_actions=not args.debug,
                action_resync_max_retries=0,
            )
        for episode in range(args.episodes):
            if not args.yes and not args.debug:
                input(
                    f"Episode {episode}: ENTER to {'capture current pose' if osc_contract else 'initialize/home the arms'}; Ctrl-C to stop > "
                )
            initial_observation = None
            if episode == 0:
                print(
                    "Checking camera and robot communication before creating the output directory...",
                    flush=True,
                )
                # Reuse this reset for episode zero; do not home or reseed twice.
                initial_observation = reset_episode(env, client, args, episode)
                args.output_dir.mkdir(parents=True, exist_ok=False)
                write_json(args.output_dir / "run_config.json", run_config)
                print(f"Startup checks passed; saving rollout to {args.output_dir}", flush=True)
            output = args.output_dir / f"episode_{episode:03d}.json"
            recorder = (
                CameraVideoRecorder(profile.cameras, args.output_dir, episode)
                if args.record_video
                else None
            )
            execute_episode(
                env,
                client,
                info,
                args,
                episode,
                confirm=prompt_first_action_safety_check
                if not args.yes and not args.debug
                else None,
                save=lambda value: write_json(output, value),
                recorder=recorder,
                eef_controller=eef_controller,
                initial_observation=initial_observation,
            )
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            client.close()


def run_node(*, target, kwargs, ipc_dir, motor_startup_retries=5):
    """Set ABC's IPC root in each fresh spawn process before it opens sockets."""
    import deploy.robot.communication as comms

    comms._IPC_DIR = ipc_dir
    module, _, function = target.partition(":")
    run = getattr(importlib.import_module(module), function or "run")
    if module == "deploy.robot.followers.yam_follower" and motor_startup_retries:
        from dreamfree.evaluation.yam_real.motor_startup import startup_retries

        with startup_retries(motor_startup_retries):
            run(**kwargs)
    else:
        run(**kwargs)


def save_session_status(output_dir, episodes, returncode):
    if not output_dir.is_dir():
        return  # Startup failed before the rollout created its output directory.
    statuses = []
    for episode in range(episodes):
        path = output_dir / f"episode_{episode:03d}.json"
        statuses.append(json.loads(path.read_text())["status"] if path.is_file() else "not_started")
    complete = returncode == 0 and all(status == "complete" for status in statuses)
    write_json(
        output_dir / "session_status.json",
        dict(
            status="complete" if complete else "failed" if returncode else "interrupted",
            returncode=returncode,
            episode_statuses=statuses,
        ),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--abc-root", type=Path, default=Path(os.environ.get("ABC_ROOT", "./third_party/abc"))
    )
    parser.add_argument("--robot-profile", default=os.environ.get("ROBOT_PROFILE", "bbox_config"))
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=6100)
    parser.add_argument("--rpc-timeout", type=float, default=60)
    parser.add_argument("--image-preprocessing", choices=("client", "server"), default="client")
    parser.add_argument(
        "--image-transport-dtype",
        choices=("bf16", "fp32"),
        default=None,
        help="Packed client image transport (default bf16); server mode sends raw uint8.",
    )
    parser.add_argument(
        "--image-preprocessing-threads",
        type=int,
        default=2,
        help="CPU PyTorch threads used for client image preprocessing.",
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--expected-checkpoint", help="Exact absolute path reported by the remote server."
    )
    parser.add_argument("--expected-variant", choices=("standard", "action_history"))
    parser.add_argument(
        "--action-space",
        choices=("joint", "eef"),
        default="joint",
        help="Expected policy output space; EEF execution uses --eef-controller.",
    )
    parser.add_argument(
        "--eef-controller",
        choices=("ik", "osc"),
        default="ik",
        help="EEF execution: bounded IK (default) or continuous Cartesian torque control.",
    )
    parser.add_argument(
        "--eef-rejection-retries",
        type=int,
        default=0,
        help="Fresh policy predictions after target rejection at one step (default 0); hardware faults still abort.",
    )
    parser.add_argument(
        "--motor-startup-retries",
        type=int,
        default=None,
        help="Extra CAN attempts per startup/calibration request (0..10; default 5 for joint/IK hardware, 0 for OSC/debug).",
    )
    parser.add_argument(
        "--osc-config",
        type=Path,
        help="Explicit OSC gains, limits, mounting, calibration and hardware-validation JSON.",
    )
    parser.add_argument("--expected-action-noise-seed", type=int)
    parser.add_argument("--episode-seed", type=int, default=0)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--max-timesteps", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save continuous top/left/right MP4s for each episode (default enabled).",
    )
    parser.add_argument(
        "--ipc-dir",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/data-vol2/jaehyeong/tmp")) / "yam_real_ipc",
    )
    parser.add_argument(
        "--init-q",
        default=os.environ.get("DEPLOY_INIT_Q"),
        help="JSON list of 14 home-position values, forwarded to ABC.",
    )
    parser.add_argument(
        "--kp-scale",
        default=os.environ.get("DEPLOY_KP_SCALE"),
        help="JSON list of 7 positive follower kp multipliers, forwarded to ABC.",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        help="Joint step bound in radians: optional for joint policies, default 0.1 for EEF IK.",
    )
    parser.add_argument(
        "--eef-position-tolerance",
        type=float,
        default=0.005,
        help="Maximum IK wrist error in meters.",
    )
    parser.add_argument(
        "--eef-orientation-tolerance",
        type=float,
        default=0.05,
        help="Maximum IK rotation error in radians.",
    )
    parser.add_argument(
        "--eef-max-tracking-error",
        type=float,
        default=0.35,
        help="Maximum EEF-derived joint target distance from measured joints, in radians.",
    )
    parser.add_argument("--eef-ik-iterations", type=int, default=80)
    parser.add_argument("--print-predicted-actions", action="store_true")
    confirmation = parser.add_mutually_exclusive_group()
    confirmation.add_argument(
        "--yes",
        dest="yes",
        action="store_true",
        default=True,
        help="Automatically proceed through homing and first action (default).",
    )
    confirmation.add_argument(
        "--confirm",
        dest="yes",
        action="store_false",
        help="Prompt before homing and the first action.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Cameras and policy only; no follower processes or commands.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check profile, video encoder, remote metadata, and image preprocessing without starting hardware.",
    )
    args = parser.parse_args(argv)
    if args.motor_startup_retries is None:
        args.motor_startup_retries = 0 if args.eef_controller == "osc" or args.debug else 5
    if not 0 <= args.motor_startup_retries <= 10:
        parser.error("--motor-startup-retries must be in 0..10.")
    if args.motor_startup_retries and (args.eef_controller == "osc" or args.debug):
        parser.error(
            "--motor-startup-retries requires joint/IK hardware followers; it does not apply to OSC or --debug."
        )
    if args.eef_rejection_retries < 0 or (
        args.eef_rejection_retries and args.action_space != "eef"
    ):
        parser.error(
            "--eef-rejection-retries must be nonnegative and requires --action-space eef when enabled."
        )
    if args.eef_controller == "osc":
        if args.action_space != "eef" or args.osc_config is None:
            parser.error("--eef-controller osc requires --action-space eef and --osc-config.")
        if (
            args.debug
            or not args.yes
            or args.max_joint_delta is not None
            or args.init_q is not None
            or args.kp_scale is not None
        ):
            parser.error(
                "OSC uses its JSON bounds and captures the current pose: remove --debug, --confirm, --max-joint-delta, --init-q and --kp-scale."
            )
    elif args.osc_config is not None:
        parser.error("--osc-config requires --eef-controller osc.")
    if args.image_preprocessing == "server" and args.image_transport_dtype is not None:
        parser.error("--image-transport-dtype applies only to --image-preprocessing client.")
    if args.image_preprocessing == "client" and args.image_transport_dtype is None:
        args.image_transport_dtype = "bf16"
    if args.image_preprocessing_threads <= 0:
        parser.error("--image-preprocessing-threads must be positive.")
    if not 1 <= args.open_loop_horizon <= 32 or args.max_timesteps <= 0 or args.episodes <= 0:
        parser.error("Require open-loop horizon 1..32 and positive timesteps/episodes.")
    if not 0 <= args.episode_seed <= 2**63 - args.episodes or not args.instruction.strip():
        parser.error("Require nonempty instruction and episode seeds in [0,2**63).")
    if (
        not np.isfinite(args.rpc_timeout)
        or args.rpc_timeout <= 0
        or not 1 <= args.remote_port <= 65535
    ):
        parser.error("Require positive finite RPC timeout and port 1..65535.")
    if args.max_joint_delta is not None and (
        not np.isfinite(args.max_joint_delta) or args.max_joint_delta <= 0
    ):
        parser.error("--max-joint-delta must be finite and positive.")
    for name in ("eef_position_tolerance", "eef_orientation_tolerance", "eef_max_tracking_error"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.eef_ik_iterations <= 0:
        parser.error("--eef-ik-iterations must be positive.")
    if not args.check_only and args.output_dir is None:
        parser.error("--output-dir is required for rollouts.")
    for name, size in (("init_q", 14), ("kp_scale", 7)):
        if getattr(args, name):
            values = np.asarray(json.loads(getattr(args, name)), dtype=np.float64)
            if (
                values.shape != (size,)
                or not np.isfinite(values).all()
                or (name == "kp_scale" and np.any(values <= 0))
            ):
                parser.error(
                    f"--{name.replace('_', '-')} must contain {size} finite {'positive ' if name == 'kp_scale' else ''}values."
                )
    return args


def main():
    args = parse_args()
    abc_root = args.abc_root.expanduser().resolve()
    args.abc_root = abc_root
    revision = subprocess.check_output(
        ["git", "-C", str(abc_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != ABC_REVISION:
        raise ValueError(f"ABC must be at tested revision {ABC_REVISION}, got {revision}.")
    sys.path.insert(0, str(abc_root))
    os.environ["ROBOT_PROFILE"] = args.robot_profile
    for name in ("init_q", "kp_scale"):
        if getattr(args, name) is not None:
            os.environ["DEPLOY_" + name.upper()] = getattr(args, name)
    from deploy.robot.config import get_i2rt_config

    profile = get_i2rt_config()
    validate_profile(profile)
    if args.record_video:
        check_video_backend()
    client, info = connect(args)
    client.close()
    eef_controller = prepare_eef_controller(args, profile)
    if eef_controller is not None:
        info["client_eef_control"] = eef_controller.contract
    osc_contract = prepare_osc(args, profile)
    if osc_contract is not None:
        info["client_eef_control"] = osc_contract
    if args.motor_startup_retries:
        from dreamfree.evaluation.yam_real.motor_startup import validate_driver_sources

        info["client_motor_startup"] = dict(
            validate_driver_sources(), extra_retries=args.motor_startup_retries
        )
    print(json.dumps(info, indent=2), flush=True)
    if args.check_only:
        return 0
    if not args.yes and not args.debug and not sys.stdin.isatty():
        raise ValueError(
            "--confirm requires an interactive terminal; omit it for automatic execution."
        )
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.ipc_dir = args.ipc_dir.expanduser().resolve()
    # Linux AF_UNIX addresses have a short path limit. Refuse before hardware starts.
    if len(str(args.ipc_dir).encode()) + len("/follower_right_state") >= 104:
        raise ValueError(
            "--ipc-dir is too long for a Unix-domain socket; choose a shorter data-volume path."
        )
    args.ipc_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(args.ipc_dir.parent))
    import fcntl

    from deploy.robot.launch import ProcessSpec, launch
    from deploy.robot.specs import camera_specs, follower_specs

    # A directory lock keeps two launches from rebinding the same station sockets.
    directory_fd = os.open(args.ipc_dir, os.O_RDONLY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_config = dict(
            args={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            server=info,
            abc_revision=revision,
            execution="continuous_osc_with_30hz_target_updates"
            if osc_contract
            else "synchronous_30hz_with_inference_pauses",
            effective_init_q={name: robot.init_q for name, robot in profile.robots.items()},
            effective_kp_scale=os.environ.get("DEPLOY_KP_SCALE", "1.0"),
        )
        specs = camera_specs(profile)
        if osc_contract is not None:
            specs.append(
                ProcessSpec(
                    "yam_osc",
                    "dreamfree.evaluation.yam_real.osc_servo:run",
                    {"args": args},
                    quiet=False,
                )
            )
        elif not args.debug:
            specs += follower_specs(profile, quiet=False)
        specs.append(
            ProcessSpec(
                "yam_rollout",
                "dreamfree.evaluation.yam_real.client:run_rollout",
                {"args": args, "run_config": run_config},
                terminal_input=not args.yes and not args.debug,
            )
        )
        wrapped = [
            ProcessSpec(
                spec.name,
                "dreamfree.evaluation.yam_real.client:run_node",
                dict(
                    target=spec.target,
                    kwargs=spec.kwargs,
                    ipc_dir=str(args.ipc_dir),
                    motor_startup_retries=args.motor_startup_retries,
                ),
                quiet=spec.quiet,
                terminal_input=spec.terminal_input,
            )
            for spec in specs
        ]
        status = launch(wrapped)
        save_session_status(args.output_dir, args.episodes, status)
        return status
    finally:
        os.close(directory_fd)


if __name__ == "__main__":
    raise SystemExit(main())
