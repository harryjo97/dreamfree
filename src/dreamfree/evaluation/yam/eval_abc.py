"""Run scored ABC tasks against a remote DreamFree policy; import no DreamFree model dependencies."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from dreamfree.evaluation.action_protocol import IssuedActionHistoryBuffer
from dreamfree.evaluation.yam.common import (
    ABC_REVISION,
    DEFAULT_SUITE,
    FIXED_COUNTS,
    load_suite,
    read_json,
    validate_observation,
    write_json,
)
from dreamfree.evaluation.yam_real.common import executable_chunk
from dreamfree.evaluation.yam_real.common import validate_server as validate_robot_server
from dreamfree.evaluation.yam_real.policy_rpc import PolicyRpcClient


def validate_server(info):
    validate_robot_server(
        info, action_space=info.get("action_space"), replan_steps=info.get("replan_steps")
    )


def resolve_mujoco_jacobian(spec, mode):
    """Use dense for the chess scene unless the caller explicitly chooses a mode."""
    return "dense" if mode == "default" and spec.env_task == "chess" else mode


def configure_mujoco_jacobian(env, mode):
    """Preserve an explicit Jacobian choice when ABC randomizers replace the model."""
    if mode == "default":
        return
    import mujoco

    value = {
        "auto": mujoco.mjtJacobian.mjJAC_AUTO,
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    }[mode]
    bind_model = env._bind_model

    def bind_with_jacobian(model):
        model.opt.jacobian = value
        bind_model(model)

    # Chess resets can reload XML or cached models before calling mj_forward.
    env._bind_model = bind_with_jacobian
    env.model.opt.jacobian = value
    env._yam_expected_jacobian = value


def configure_fixed_count(env, count, instruction):
    """Keep ABC's randomized objects while fixing both the language and scorer target."""
    randomize = env._task_randomizer.randomize

    def randomize_fixed(*args, **kwargs):
        state = randomize(*args, **kwargs)
        objects = state.metadata["objects"]
        if len(objects) < count:
            raise ValueError(f"Fixed-count task needs {count} objects, got {len(objects)}.")
        state.metadata.update(
            prompt=instruction,
            prompt_type="exact_count",
            target_count=count,
            target_objects=[],
            eligible_objects=[obj["name"] for obj in objects],
            attributes={},
        )
        env.prompt = instruction
        return state

    # reset() configures the evaluator from this metadata after randomization.
    env._task_randomizer.randomize = randomize_fixed


def make_env(abc_sim, task, entry, max_steps, *, mujoco_jacobian="default"):
    spec = abc_sim.get_task_spec(task)
    if not spec.evaluator_name:
        raise ValueError(f"ABC task {task!r} has no success evaluator; choose a scored task.")
    if spec.evaluator_name == "ball_tray_balancing":
        raise ValueError(
            "ball_tray_balancing needs a full-horizon maintenance protocol, not completion-and-stop evaluation."
        )
    if spec.name in FIXED_COUNTS and entry["instruction"] != spec.prompt:
        raise ValueError(
            f"Fixed-count task {spec.name} requires instruction {spec.prompt!r} to match its scoring target."
        )
    env = abc_sim.make_env(
        task=task,
        prompt=entry["instruction"],
        render_cameras=True,
        camera_backend="mujoco",
        camera_height=480,
        camera_width=640,
        max_episode_steps=max_steps,
        terminate_on_success=True,
    )
    # Preserve each task's substep count, but make policy actions exactly 30 Hz.
    env._physics_dt = 1.0 / (30 * env._control_decimation)
    env.model.opt.timestep = env._physics_dt
    configure_mujoco_jacobian(env, resolve_mujoco_jacobian(spec, mujoco_jacobian))
    if spec.name in FIXED_COUNTS:
        configure_fixed_count(env, FIXED_COUNTS[spec.name], entry["instruction"])
    return env


def validate_env(env, observation):
    validate_observation(observation)
    expected_jacobian = getattr(env, "_yam_expected_jacobian", None)
    if expected_jacobian is not None and env.model.opt.jacobian != expected_jacobian:
        raise ValueError(
            f"ABC model lost the configured MuJoCo Jacobian mode: expected {expected_jacobian}, "
            f"got {env.model.opt.jacobian}."
        )
    if not math.isclose(env.model.opt.timestep * env._control_decimation, 1 / 30, abs_tol=1e-10):
        raise ValueError("ABC simulation must advance exactly 1/30 second per policy action.")
    if env.evaluate_task() is None:
        raise ValueError("Enabled ABC task evaluator did not initialize.")


def video_frame(observation):
    images = observation["images"]
    bottom = np.concatenate([images[name][:, ::2, ::2] for name in ("left", "right")], axis=2)
    return np.concatenate([images["top"], bottom], axis=1).transpose(1, 2, 0)


def task_metrics(info):
    if "task_success" not in info or "task_reward" not in info:
        raise ValueError("ABC step omitted required task_success/task_reward metrics.")
    if not isinstance(info["task_success"], (bool, np.bool_)):
        raise ValueError("ABC task_success must be a boolean.")
    reward = float(info["task_reward"])
    if not math.isfinite(reward):
        raise ValueError("ABC evaluator returned a non-finite reward.")
    return bool(info["task_success"]), reward


def run_episode(
    env, client, server_info, *, seed, max_steps, video=None, video_stride=3, eef_controller=None
):
    reset_start = time.perf_counter()
    # Reset the arm pose independently for each scored episode.
    env.forget_arm_state()
    observation, _ = env.reset(seed=seed, randomize=True)
    reset_seconds = time.perf_counter() - reset_start
    validate_env(env, observation)
    acknowledgement = client.reset(episode_seed=seed)
    if acknowledgement.get("episode_seed") != seed:
        raise ValueError("Policy server failed to reset its episode seed.")
    action_space = server_info["action_space"]
    history = (
        IssuedActionHistoryBuffer(
            horizon=server_info["action_history_horizon"], action_dim=server_info["action_dim"]
        )
        if server_info["action_history_horizon"]
        else None
    )
    if (action_space == "eef") != (eef_controller is not None):
        raise ValueError("EEF simulation requires its pinned wrist IK controller.")
    previous = np.asarray(observation["state"]).copy()
    if video is not None:
        video.append_data(video_frame(observation))
    issued, states, replans = [], [], []
    success, max_reward, clipped_total = False, -math.inf, 0
    start = time.perf_counter()
    last_report = start
    timings = {
        "physics_seconds": 0.0,
        "render_seconds": 0.0,
        "video_seconds": 0.0,
        "rpc_seconds": 0.0,
    }
    rendered_observations = 1
    state = observation["state"]
    terminated = truncated = False
    while len(issued) < max_steps and not (terminated or truncated):
        validate_observation(observation)
        instruction = observation["prompt"]
        phase_start = time.perf_counter()
        request = history.add_to_observation(observation) if history else observation
        result = client.infer(request, instruction)
        timings["rpc_seconds"] += time.perf_counter() - phase_start
        actions, _ = executable_chunk(
            result["actions"], server_info["action_horizon"], action_space=action_space
        )
        remaining = min(server_info["replan_steps"], max_steps - len(issued))
        raw = np.asarray(result["actions"])
        replans.append(
            {
                "step": len(issued),
                "instruction": instruction,
                "inference_seconds": float(result["inference_seconds"]),
            }
        )
        for i, action in enumerate(actions[:remaining]):
            gripper_indices = [9, 19] if action_space == "eef" else [6, 13]
            clipped_total += int(
                np.count_nonzero(raw[i, gripper_indices] != action[gripper_indices])
            )
            issued_policy_action = action
            if eef_controller is not None:
                action, issued_policy_action, _ = eef_controller.command(
                    action, previous=previous, measured=state
                )
            states.append(np.asarray(state, dtype=np.float32).tolist())
            phase_start = time.perf_counter()
            _, _, terminated, truncated, info = env.step(action, render_obs=False)
            if history:
                history.record_issued_action(issued_policy_action)
            state = env.get_state()
            timings["physics_seconds"] += time.perf_counter() - phase_start
            # Record only commands actually sent to the simulator, after aperture clipping.
            previous = action.copy()
            issued.append(action.tolist())
            step_success, reward = task_metrics(info)
            success = success or step_success
            max_reward = max(max_reward, reward)
            done = terminated or truncated or len(issued) == max_steps
            prompt_changed = env.prompt != instruction
            save_frame = video is not None and (len(issued) % video_stride == 0 or done)
            replan = not done and (i + 1 == remaining or prompt_changed)
            # Render current images only when consumed; physics still advances on every action.
            if save_frame or replan:
                phase_start = time.perf_counter()
                observation = env.get_obs()
                timings["render_seconds"] += time.perf_counter() - phase_start
                rendered_observations += 1
            if save_frame:
                phase_start = time.perf_counter()
                video.append_data(video_frame(observation))
                timings["video_seconds"] += time.perf_counter() - phase_start
            now = time.perf_counter()
            if now - last_report >= 30:
                print(
                    f"Progress seed={seed}: {len(issued)}/{max_steps} steps, {now - start:.1f}s elapsed, "
                    f"physics={timings['physics_seconds']:.1f}s render={timings['render_seconds']:.1f}s "
                    f"policy_rpc={timings['rpc_seconds']:.1f}s video={timings['video_seconds']:.1f}s",
                    flush=True,
                )
                last_report = now
            if done or prompt_changed:
                break
    return {
        "seed": seed,
        "success": success,
        "steps": len(issued),
        "max_task_reward": max_reward,
        "final_task_reward": reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "wall_seconds": time.perf_counter() - start,
        "sim_seconds": len(issued) / 30,
        "timings": {"reset_seconds": reset_seconds, **timings},
        "rendered_observations": rendered_observations,
        "gripper_values_clipped": clipped_total,
        "replans": replans,
        "issued_actions": issued,
        "observed_states_before_action": states,
    }


def summarize(records, expected_episodes):
    tasks = {}
    for name in dict.fromkeys(item["task"] for item in records):
        episodes = [item for item in records if item["task"] == name]
        tasks[name] = {
            "episodes": len(episodes),
            "successes": sum(item["success"] for item in episodes),
            "success_rate": sum(item["success"] for item in episodes) / len(episodes),
            "mean_max_task_reward": float(np.mean([item["max_task_reward"] for item in episodes])),
        }
    return {
        "status": "complete" if len(records) == expected_episodes else "in_progress",
        "completed_episodes": len(records),
        "expected_episodes": expected_episodes,
        "tasks": tasks,
        "overall_success_rate": sum(item["success"] for item in records) / len(records)
        if records
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--action-space", choices=("joint", "eef"), default="joint")
    parser.add_argument("--eef-max-joint-delta", type=float, default=0.1)
    parser.add_argument("--eef-max-tracking-error", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--mujoco-jacobian",
        choices=("default", "auto", "dense", "sparse"),
        default="default",
        help="Default uses dense for chess and preserves other scene settings. Explicit auto/dense/sparse overrides all selected tasks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-stride", type=int, default=3)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Reset and hold current pose once per task; no policy required.",
    )
    args = parser.parse_args()
    if (
        args.episodes <= 0
        or args.video_stride <= 0
        or (args.max_steps is not None and args.max_steps <= 0)
    ):
        parser.error("episodes, video-stride, and max-steps must be positive")
    suite = load_suite(args.suite, args.tasks)
    import abc_sim

    abc_root = Path(abc_sim.__file__).resolve().parent.parent
    revision = subprocess.check_output(
        ["git", "-C", str(abc_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != ABC_REVISION:
        raise ValueError(
            f"ABC revision {revision} differs from tested interface {ABC_REVISION}; run setup_abc.sh."
        )
    if args.check_env:
        for task, entry in suite.items():
            env = make_env(
                abc_sim,
                task,
                entry,
                args.max_steps or entry["max_steps"],
                mujoco_jacobian=args.mujoco_jacobian,
            )
            try:
                observation, _ = env.reset(seed=args.seed, randomize=True)
                validate_env(env, observation)
                observation, _, _, _, info = env.step(observation["state"].copy())
                validate_env(env, observation)
                task_metrics(info)
                mode = resolve_mujoco_jacobian(abc_sim.get_task_spec(task), args.mujoco_jacobian)
                print(
                    f"PASS {task}: three 640x480 RGB cameras, 14D state, 30 Hz, success evaluator, "
                    f"Jacobian override={mode}",
                    flush=True,
                )
            finally:
                env.close()
        return
    if args.ready_file is None or args.output_dir is None:
        parser.error("--ready-file and --output-dir are required for policy evaluation")
    ready = read_json(args.ready_file)
    client = PolicyRpcClient("127.0.0.1", ready["port"], timeout_sec=1800)
    records = []
    expected = len(suite) * args.episodes
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite previous evaluation: {output}")
    try:
        server_info = client.info()
        validate_server(server_info)
        if server_info["action_space"] != args.action_space:
            raise ValueError("Evaluation --action-space differs from the checkpoint.")
        eef_controller = None
        if args.action_space == "eef":
            from dreamfree.evaluation.yam_real.eef_control import EefController

            eef_controller = EefController(
                abc_root / "abc_sim/models/assets/i2rt_yam/yam.xml",
                max_joint_delta=args.eef_max_joint_delta,
                max_tracking_error=args.eef_max_tracking_error,
            )
        if server_info != ready:
            raise ValueError("Connected policy differs from the launcher's ready-file.")
        write_json(
            output / "eval_config.json",
            {
                "policy": server_info,
                "suite": suite,
                "episodes_per_task": args.episodes,
                "seed": args.seed,
                "max_steps_override": args.max_steps,
                "abc_revision": revision,
                "episode_reset": "init_q_empty_history_and_seeded_policy",
                "eef_controller": eef_controller.contract if eef_controller else None,
                "physics": "cpu_mujoco",
                "renderer": "mujoco",
                "control_hz": 30,
                "render_gpu": int(os.environ["MUJOCO_EGL_DEVICE_ID"])
                if "MUJOCO_EGL_DEVICE_ID" in os.environ
                else None,
                "render_schedule": "policy_and_video_frames",
                "mujoco_version": importlib.metadata.version("mujoco"),
                "video_stride": args.video_stride,
                "mujoco_jacobian": args.mujoco_jacobian,
                "mujoco_jacobian_overrides": {
                    task: resolve_mujoco_jacobian(abc_sim.get_task_spec(task), args.mujoco_jacobian)
                    for task in suite
                },
                "save_video": not args.no_video,
            },
        )
        for task, entry in suite.items():
            directory = output / task
            directory.mkdir()
            max_steps = args.max_steps or entry["max_steps"]
            env = make_env(abc_sim, task, entry, max_steps, mujoco_jacobian=args.mujoco_jacobian)
            try:
                for episode in range(args.episodes):
                    print(
                        f"Starting {task} episode {episode + 1}/{args.episodes}, seed={args.seed + episode}, "
                        f"max_steps={max_steps}",
                        flush=True,
                    )
                    video = None
                    try:
                        if not args.no_video:
                            import imageio.v2 as imageio

                            video = imageio.get_writer(
                                directory / f"episode_{episode:03d}.mp4",
                                fps=30 / args.video_stride,
                                codec="libx264",
                                macro_block_size=1,
                            )
                        result = run_episode(
                            env,
                            client,
                            server_info,
                            seed=args.seed + episode,
                            max_steps=max_steps,
                            video=video,
                            video_stride=args.video_stride,
                            eef_controller=eef_controller,
                        )
                    finally:
                        if video is not None:
                            video.close()
                    result.update(task=task, episode=episode)
                    write_json(directory / f"episode_{episode:03d}.json", result)
                    records.append(
                        {
                            key: value
                            for key, value in result.items()
                            if key
                            not in {"issued_actions", "observed_states_before_action", "replans"}
                        }
                    )
                    write_json(summary_path, summarize(records, expected))
                    print(
                        f"{task} {episode + 1}/{args.episodes}: success={result['success']} "
                        f"steps={result['steps']} clipped_grippers={result['gripper_values_clipped']}",
                        flush=True,
                    )
            finally:
                env.close()
    except BaseException as exc:
        summary = summarize(records, expected)
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(summary_path, summary)
        raise
    finally:
        client.close()
    print(f"Results: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
