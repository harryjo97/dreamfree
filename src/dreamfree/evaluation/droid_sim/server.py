"""Serve optimized DROID simulation inference over localhost RPC."""

import argparse
from concurrent.futures import ThreadPoolExecutor

from dreamfree.evaluation.droid_real.policy_rpc import PolicyRpcServer
from dreamfree.evaluation.droid_sim.deploy_policy_optimized import build_policy_from_args
from dreamfree.evaluation.droid_sim.policy_rpc import (
    SIMULATION_ACTION_REPRESENTATION,
    SIMULATION_POLICY_RUNTIME,
    SIMULATION_SERVER_NAME,
)


def policy_info(policy):
    """Describe the checkpoint and action contract shared by both simulators."""
    if policy.policy_runtime != SIMULATION_POLICY_RUNTIME:
        raise ValueError(f"Unsupported DROID simulation runtime: {policy.policy_runtime!r}.")
    return dict(
        server=SIMULATION_SERVER_NAME,
        policy_runtime=policy.policy_runtime,
        checkpoint_path=str(policy.stage2_checkpoint_path),
        stage1_checkpoint_path=str(policy.stage1_checkpoint_path),
        action_representation=SIMULATION_ACTION_REPRESENTATION,
        action_dim=8,
        action_horizon=policy.action_horizon,
        action_history_horizon=getattr(policy, "action_history_horizon", 0),
        checkpoint_type=policy.stage2_metadata["checkpoint_type"],
        replan_steps=policy.replan_steps,
        num_inference_steps=policy.num_inference_steps,
        action_noise_seed=getattr(policy, "action_noise_seed", None),
        requires_episode_seed_on_reset=getattr(policy, "action_noise_seed", None) is not None,
        optimization_contract=policy.optimization_contract,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", "--ckpt", required=True)
    parser.add_argument("--droid-dataset-root", default=None)
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=6100)
    parser.add_argument("--replan-steps", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--action-noise-seed", type=int, default=42)
    parser.add_argument("--action-cuda-graph", choices=("auto", "true", "false"), default="auto")
    parser.add_argument(
        "--encoder-cuda-graph",
        "--stage1-cuda-graph",
        choices=("auto", "true", "false"),
        default="auto",
    )
    parser.add_argument("--vae-first-frame-only", choices=("auto", "true", "false"), default="auto")
    args = parser.parse_args(argv)
    options = vars(args).copy()
    options.update(mixed_precision="bf16", load_text_encoder=True)
    with ThreadPoolExecutor(max_workers=1) as worker:
        policy = worker.submit(build_policy_from_args, options).result()
        worker.submit(policy.warmup_optimized_inference).result()
        info = policy_info(policy)

        def reset(episode_seed):
            if type(episode_seed) is not int or episode_seed < 0:
                raise ValueError("DROID simulation reset requires a nonnegative episode_seed.")
            policy.reset()
            policy.reset_action_noise_for_episode(episode_seed)

        with PolicyRpcServer(
            ("127.0.0.1", args.port),
            infer_fn=lambda observation, instruction: {
                "actions": worker.submit(
                    policy.predict_action_chunk, observation, instruction
                ).result()
            },
            reset_fn=lambda episode_seed: worker.submit(reset, episode_seed).result(),
            info_fn=lambda: info,
        ) as server:
            server.serve_forever()


if __name__ == "__main__":
    main()
