"""Evaluate DROID checkpoints in MolmoSpaces using the DreamFree simulation server."""

from __future__ import annotations

import argparse
import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from dreamfree.evaluation.droid_sim.policy_rpc import (
    SIMULATION_ACTION_REPRESENTATION,
    SIMULATION_POLICY_RUNTIME,
    SIMULATION_SERVER_NAME,
    PolicyRpcClient,
)

CAMERA_KEYS = {
    "observation/exterior_image_1_left": (
        "droid_shoulder_light_randomization",
        "exo_camera_1",
        "exterior_image_1_left",
    ),
    "observation/exterior_image_2_left": (
        "exo_camera_2",
        "randomized_zed2_analogue_1",
        "randomized_zed2_analogue_2",
        "exterior_image_2_left",
    ),
    "observation/wrist_image_left": ("wrist_camera_zed_mini", "wrist_camera", "wrist_image_left"),
}
GRIPPER_QPOS_OPEN_SCALE = 0.824033


def rpc_observation(observation, *, gripper_qpos_open_scale=GRIPPER_QPOS_OPEN_SCALE):
    """Map one MolmoSpaces observation to raw DROID cameras and proprioception."""
    if isinstance(observation, (list, tuple)):
        if len(observation) != 1:
            raise ValueError("MolmoSpaces DROID evaluation requires exactly one observation.")
        observation = observation[0]
    if not isinstance(observation, Mapping):
        raise TypeError("MolmoSpaces observation must be a mapping.")
    if not np.isfinite(gripper_qpos_open_scale) or gripper_qpos_open_scale <= 0:
        raise ValueError("gripper_qpos_open_scale must be finite and positive.")
    result = {}
    for destination, candidates in CAMERA_KEYS.items():
        key = next((key for key in candidates if key in observation), None)
        if key is None:
            raise KeyError(f"Missing DROID camera {destination!r}; expected one of {candidates}.")
        frame = np.asarray(observation[key])
        if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0:
            raise ValueError(f"Camera {key!r} must be a nonempty RGB [H,W,3] image.")
        if frame.dtype != np.uint8 and not np.issubdtype(frame.dtype, np.floating):
            raise TypeError(f"Camera {key!r} must contain uint8 or floating-point RGB values.")
        if not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 255:
            raise ValueError(f"Camera {key!r} must contain finite RGB values in [0,1] or [0,255].")
        result[destination] = frame
    qpos = observation.get("qpos")
    if not isinstance(qpos, Mapping) or "arm" not in qpos or "gripper" not in qpos:
        raise KeyError("MolmoSpaces observation requires qpos['arm'] and qpos['gripper'].")
    arm = np.asarray(qpos["arm"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(qpos["gripper"], dtype=np.float32).reshape(-1)
    if (
        arm.size < 7
        or gripper.size < 1
        or not np.isfinite(arm).all()
        or not np.isfinite(gripper).all()
    ):
        raise ValueError(
            "MolmoSpaces qpos requires seven finite arm joints and a finite gripper value."
        )
    result["observation/joint_position"] = arm[:7]
    result["observation/gripper_position"] = np.asarray(
        [np.clip(gripper[0] / gripper_qpos_open_scale, 0.0, 1.0)], dtype=np.float32
    )
    return result


class MolmoSpacesClient:
    """Buffer absolute joint actions from a validated DreamFree simulation server."""

    def __init__(
        self,
        *,
        expected_checkpoint,
        host="127.0.0.1",
        port=6100,
        replan_steps=32,
        timeout_sec=600.0,
        gripper_qpos_open_scale=GRIPPER_QPOS_OPEN_SCALE,
    ):
        checkpoint = Path(expected_checkpoint)
        if not checkpoint.is_absolute():
            raise ValueError(
                "expected_checkpoint must be the absolute checkpoint path on the server."
            )
        if type(replan_steps) is not int or replan_steps <= 0:
            raise ValueError("replan_steps must be a positive integer.")
        if not np.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be finite and positive.")
        if not np.isfinite(gripper_qpos_open_scale) or gripper_qpos_open_scale <= 0:
            raise ValueError("gripper_qpos_open_scale must be finite and positive.")
        self.replan_steps = replan_steps
        self.gripper_qpos_open_scale = gripper_qpos_open_scale
        self.actions = deque()
        self.client = PolicyRpcClient(host, port, timeout_sec=timeout_sec)
        try:
            self.server_info = self.client.info()
            if not isinstance(self.server_info, dict):
                raise TypeError("DROID simulation server info must be a mapping.")
            expected = {
                "server": SIMULATION_SERVER_NAME,
                "policy_runtime": SIMULATION_POLICY_RUNTIME,
                "checkpoint_path": str(checkpoint),
                "action_representation": SIMULATION_ACTION_REPRESENTATION,
                "action_dim": 8,
            }
            for key, value in expected.items():
                if self.server_info.get(key) != value:
                    raise ValueError(
                        f"DROID simulation server {key} mismatch: expected {value!r}, "
                        f"got {self.server_info.get(key)!r}."
                    )
            self.action_horizon = self.server_info.get("action_horizon")
            if type(self.action_horizon) is not int or not 0 < replan_steps <= self.action_horizon:
                raise ValueError(
                    "replan_steps must fit the server's positive integer action_horizon."
                )
            self.reset()
        except Exception:
            self.close()
            raise

    def reset(self):
        self.actions.clear()
        from dreamfree.evaluation.action_protocol import IssuedActionHistoryBuffer

        horizon = self.server_info.get("action_history_horizon", 0)
        if type(horizon) is not int or horizon < 0:
            raise ValueError("Server action_history_horizon must be a nonnegative integer.")
        expected_type = "dreamfree_stage2_action_history" if horizon else "dreamfree_stage2"
        if self.server_info.get("checkpoint_type") != expected_type:
            raise ValueError("Server checkpoint_type disagrees with action_history_horizon.")
        self.history = IssuedActionHistoryBuffer(horizon=horizon, action_dim=8) if horizon else None
        if self.server_info.get("requires_episode_seed_on_reset", False):
            import torch

            self.client.reset(episode_seed=torch.initial_seed())
        else:
            self.client.reset()

    def get_action(self, observation, instruction):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("MolmoSpaces evaluation requires a nonempty task instruction.")
        if not self.actions:
            request = rpc_observation(
                observation, gripper_qpos_open_scale=self.gripper_qpos_open_scale
            )
            if self.history is not None:
                request = self.history.add_to_observation(request)
            reply = self.client.infer(request, instruction)
            if not isinstance(reply, dict) or "actions" not in reply:
                raise ValueError("DROID simulation server infer response must contain actions.")
            actions = np.asarray(reply["actions"], dtype=np.float32)
            if actions.shape != (self.action_horizon, 8) or not np.isfinite(actions).all():
                raise ValueError(
                    f"Expected finite DROID actions [{self.action_horizon},8], got {actions.shape}."
                )
            if not np.isin(actions[:, 7], [0.0, 1.0]).all():
                raise ValueError(
                    "DROID simulation server must return binary gripper values 0 or 1."
                )
            for action in actions[: self.replan_steps]:
                self.actions.append(
                    {"arm": action[:7].copy(), "gripper": action[7:].copy() * 255.0}
                )
        action = self.actions.popleft()
        if self.history is not None:
            self.history.record_issued_action(
                np.concatenate((action["arm"], action["gripper"] / 255.0))
            )
        return action

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None


def validate_results(result, expected_count):
    """Reject incomplete evaluations, including errors swallowed by the simulator."""
    if result.total_count != expected_count or len(result.episode_results) != expected_count:
        raise RuntimeError(
            f"MolmoSpaces evaluation is incomplete: expected {expected_count} episodes, "
            f"evaluated {result.total_count}, found {len(result.episode_results)} result records. "
            "Check the simulator log for sampling, rollout, or inference errors."
        )
    if any(type(episode.success) is not bool for episode in result.episode_results):
        raise RuntimeError("MolmoSpaces results must include a boolean success value per episode.")
    successes = sum(episode.success for episode in result.episode_results)
    if result.success_count != successes:
        raise RuntimeError("MolmoSpaces success count does not match its episode result records.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6100)
    parser.add_argument("--replan-steps", type=int, default=32)
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--gripper-qpos-open-scale", type=float, default=GRIPPER_QPOS_OPEN_SCALE)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-episodes", type=int)
    selection.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--task-horizon-steps", type=int)
    parser.add_argument("--save-videos", action="store_true")
    args = parser.parse_args(argv)
    benchmark = args.benchmark_dir.expanduser().resolve()
    if not (benchmark / "benchmark.json").is_file():
        raise FileNotFoundError(f"MolmoSpaces benchmark.json is missing under {benchmark}.")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"MolmoSpaces output directory must be empty: {output}.")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive.")
    if args.task_horizon_steps is not None and args.task_horizon_steps <= 0:
        raise ValueError("--task-horizon-steps must be positive.")
    if args.episode_indices is not None and (
        min(args.episode_indices) < 0 or len(set(args.episode_indices)) != len(args.episode_indices)
    ):
        raise ValueError("--episode-indices must be unique nonnegative integers.")
    try:
        from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
        from molmo_spaces.evaluation.eval_main import run_evaluation
        from pydantic import create_model

        from dreamfree.evaluation.droid_sim.molmospaces_config import (
            DreamFreeDroidEvalConfig,
            DreamFreeDroidPolicy,
            DreamFreeDroidPolicyConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "MolmoSpaces evaluation requires its simulator environment and JSON benchmark API. "
            "Run this command with the MolmoSpaces interpreter and DreamFree's src on PYTHONPATH."
        ) from exc
    episode_count = len(load_all_episodes(benchmark))
    if not episode_count:
        raise ValueError("MolmoSpaces benchmark contains no episodes.")
    if args.episode_indices is not None and max(args.episode_indices) >= episode_count:
        raise ValueError(f"--episode-indices must be below the benchmark size {episode_count}.")
    expected_count = (
        len(args.episode_indices)
        if args.episode_indices is not None
        else min(args.max_episodes or episode_count, episode_count)
    )
    policy_config = DreamFreeDroidPolicyConfig(
        checkpoint_path=args.expected_checkpoint,
        remote_host=args.host,
        remote_port=args.port,
        replan_steps=args.replan_steps,
        timeout_sec=args.timeout_sec,
        gripper_qpos_open_scale=args.gripper_qpos_open_scale,
        eval_save_videos=args.save_videos,
    )
    config_class = create_model(
        "ConfiguredDreamFreeDroidEval",
        __base__=DreamFreeDroidEvalConfig,
        policy_config=(DreamFreeDroidPolicyConfig, policy_config),
    )
    config = config_class()
    policy = DreamFreeDroidPolicy(config, config.task_type)
    try:
        policy.prepare_model()
        result = run_evaluation(
            eval_config_cls=config_class,
            benchmark_dir=benchmark,
            checkpoint_path=args.expected_checkpoint,
            output_dir=output,
            exact_output_dir=True,
            task_horizon_steps=args.task_horizon_steps,
            max_episodes=args.max_episodes,
            episode_idxs=args.episode_indices,
            num_workers=1,
            use_wandb=False,
            preloaded_policy=policy,
            save_trajectories=False,
            save_diagnostics=False,
            save_videos=args.save_videos,
        )
        validate_results(result, expected_count)
        print(
            json.dumps(
                {
                    "success_count": result.success_count,
                    "total_count": result.total_count,
                    "output_dir": str(output),
                }
            )
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
