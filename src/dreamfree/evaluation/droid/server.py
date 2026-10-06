"""Serve optimized DROID simulation inference over localhost RPC."""

import argparse
from concurrent.futures import ThreadPoolExecutor

from dreamfree.evaluation.droid.deploy_policy_optimized import build_policy_from_args
from dreamfree.evaluation.droid_real.policy_rpc import PolicyRpcServer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--droid-dataset-root", required=True)
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=6100)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-cuda-graph", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--stage1-cuda-graph", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--vae-first-frame-only", choices=("auto", "true", "false"), default="auto")
    args = parser.parse_args(argv)
    options = vars(args).copy()
    options["ckpt_setting"] = options.pop("ckpt")
    options.update(mixed_precision="bf16", load_text_encoder=True)
    with ThreadPoolExecutor(max_workers=1) as worker:
        policy = worker.submit(build_policy_from_args, options).result()
        worker.submit(policy.warmup_optimized_inference).result()
        info = dict(
            policy_runtime=policy.policy_runtime,
            action_horizon=policy.action_horizon,
            replan_steps=policy.replan_steps,
            optimization_contract=policy.optimization_contract,
        )
        with PolicyRpcServer(
            ("127.0.0.1", args.port),
            infer_fn=lambda observation, instruction: {
                "actions": worker.submit(
                    policy.predict_action_chunk, observation, instruction
                ).result()
            },
            reset_fn=lambda episode_seed: worker.submit(policy.reset).result(),
            info_fn=lambda: info,
        ) as server:
            server.serve_forever()


if __name__ == "__main__":
    main()
