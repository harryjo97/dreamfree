"""Serve one physical YAM client through the same trusted SSH RPC transport as DROID."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dreamfree.evaluation.yam_real.common import (
    CAMERAS,
    PROTOCOL,
    WARMUP_INSTRUCTION,
    action_contract,
    write_json,
)
from dreamfree.evaluation.yam_real.policy_rpc import PolicyRpcServer


class SingleClientServer(PolicyRpcServer):
    """Keep one episode owner; a second connection cannot reset another rollout."""

    def __init__(self, *args, **kwargs):
        self._client_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self._client_lock.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._client_lock.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._client_lock.release()


def preflight(args):
    from dreamfree.evaluation.yam.policy import checkpoint_context

    checkpoint, metadata, stats = checkpoint_context(args.checkpoint_path)
    actual_space = action_contract(metadata["dataset"]["name"])["action_space"]
    if args.action_space != actual_space:
        raise ValueError(
            f"Requested action space {args.action_space}, checkpoint contains {actual_space}."
        )
    return checkpoint, metadata, stats


def initialize_policy(args):
    import numpy as np

    from dreamfree.evaluation.yam.policy import build_policy

    policy = build_policy(
        args.checkpoint_path,
        replan_steps=args.replan_steps,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        model_base_path=args.model_base_path,
    )
    observation = dict(
        state=np.zeros(14, np.float32),
        prompt=WARMUP_INSTRUCTION,
        images={name: np.zeros((3, 480, 640), np.uint8) for name in CAMERAS},
        masks=dict.fromkeys(CAMERAS, True),
    )
    if policy.action_history_horizon:
        from dreamfree.evaluation.action_protocol import IssuedActionHistoryBuffer

        history = IssuedActionHistoryBuffer(
            horizon=policy.action_history_horizon, action_dim=policy.action_dim
        )
        observation = history.add_to_observation(observation)
    policy.predict_action_chunk(observation, WARMUP_INSTRUCTION)
    policy.optimization_contract.update(enabled=True, warmup_complete=True)
    policy.reset()
    return policy


def validate_image_preprocessing(request, model_dtype, device="cpu"):
    import numpy as np
    import torch

    from dreamfree.evaluation.yam.policy import reference_build_image
    from dreamfree.evaluation.yam_real.common import validate_observation
    from dreamfree.evaluation.yam_real.image_preprocessing import (
        PACKED_IMAGE_KEY,
        PREPROCESSING_KEY,
        decode_image,
        validate_packed_image,
    )

    raw, processed = request.get("raw_observation"), request.get("processed_observation")
    if not isinstance(raw, dict) or not isinstance(processed, dict):
        raise ValueError(
            "Image comparison requires raw_observation and processed_observation dictionaries."
        )
    validate_observation(raw, allow_preprocessed=False)
    validate_observation(processed)
    transport_dtype = validate_packed_image(processed)
    reference = reference_build_image(raw, device="cpu", dtype=torch.float32)
    expected = (
        reference[0].to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        if transport_dtype == "bf16"
        else reference[0].numpy()
    )
    transport_equal = np.array_equal(expected, processed[PACKED_IMAGE_KEY])
    actual = decode_image(processed, device=device, dtype=model_dtype)
    model_reference = reference.to(device=device, dtype=model_dtype)
    exact_equal = torch.equal(model_reference, actual)
    if not transport_equal or not exact_equal:
        error = (model_reference.float() - actual.float()).abs().max().item()
        raise ValueError(
            f"YAM client preprocessing disagrees with the training/server transform: "
            f"transport_exact_equal={transport_equal}, model_input_exact_equal={exact_equal}, "
            f"max_abs_error={error:g}. Check workstation PyTorch/torchvision versions, "
            "or explicitly select --image-preprocessing server."
        )
    return dict(
        format=processed[PREPROCESSING_KEY],
        transport_exact_equal=True,
        exact_equal=True,
        model_input_dtype=str(model_dtype).removeprefix("torch."),
    )


def serve(args, policy, worker):
    from dreamfree.evaluation.yam_real.common import executable_chunk, validate_observation
    from dreamfree.evaluation.yam_real.image_preprocessing import server_contract

    contract = action_contract(policy.stage2_metadata["dataset"]["name"])
    info = dict(
        protocol=PROTOCOL,
        checkpoint=str(args.checkpoint_path),
        **contract,
        checkpoint_type=policy.stage2_metadata["checkpoint_type"],
        variant="action_history" if policy.action_history_horizon else "standard",
        global_step=policy.stage2_metadata["global_step"],
        action_horizon=policy.action_horizon,
        action_history_horizon=getattr(policy, "action_history_horizon", 0),
        replan_steps=args.replan_steps,
        fps=30,
        cameras=list(CAMERAS),
        camera_size_hw=[480, 640],
        model_size_hw=[432, 384],
        dataset_stats=str(policy.dataset_stats_path),
        stats_sha256=hashlib.sha256(Path(policy.dataset_stats_path).read_bytes()).hexdigest(),
        source_stage1_checkpoint=str(policy.stage1_checkpoint_path),
        student_weights_checkpoint=str(policy.stage1_checkpoint_path),
        seed=args.seed,
        num_inference_steps=policy.num_inference_steps,
        action_sampler=policy.action_sampler,
        text_mode=args.text_mode,
        compile_dit=True,
        inference_optimization=getattr(policy, "optimization_contract", {"enabled": False}),
        image_preprocessing=server_contract(policy.model_dtype),
    )

    def timed_infer(observation, instruction):
        validate_observation(observation)
        if instruction != observation["prompt"]:
            raise ValueError("RPC instruction and YAM observation prompt disagree.")
        started = time.perf_counter()
        actions = policy.predict_action_chunk(observation, instruction)
        executable_chunk(actions, policy.action_horizon, action_space=contract["action_space"])
        return dict(actions=actions, inference_seconds=time.perf_counter() - started)

    def reset_policy(seed):
        policy.seed = args.seed if seed is None else seed
        policy.reset()
        return dict(episode_seed=policy.seed)

    with SingleClientServer(
        ("127.0.0.1", args.port),
        infer_fn=lambda obs, instruction: worker.submit(timed_infer, obs, instruction).result(),
        reset_fn=lambda seed: worker.submit(reset_policy, seed).result(),
        info_fn=lambda: dict(info),
        validate_image_preprocessing_fn=lambda request: worker.submit(
            validate_image_preprocessing, request, policy.model_dtype, policy.device
        ).result(),
    ) as server:
        info["port"] = server.server_address[1]
        if args.ready_file:
            write_json(args.ready_file, info)
        logging.info("YAM policy ready: %s", json.dumps(info))
        try:
            server.serve_forever()
        finally:
            if args.ready_file:
                args.ready_file.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint",
        "--eval-ckpt",
        type=Path,
        required=True,
        help="Step directory, run directory, or checkpoints directory (latest completed step).",
    )
    parser.add_argument("--action-space", choices=("joint", "eef"), required=True)
    parser.add_argument("--model-base-path", help="Optional upstream model directory override.")
    parser.add_argument("--port", type=int, default=6100)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=16,
        help="Euler denoising steps; fixed for the server lifetime.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--text-mode",
        choices=("encoder",),
        default="encoder",
        help="YAM real always uses the loaded text encoder; precomputed text caches are disabled.",
    )
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check metadata, source Stage 1 and statistics on CPU.",
    )
    args = parser.parse_args(argv)
    if (
        not 1 <= args.port <= 65535
        or not 1 <= args.replan_steps <= 32
        or args.num_inference_steps <= 0
    ):
        parser.error("Require port 1..65535, replan steps 1..32 and positive inference steps.")
    if not 0 <= args.seed < 2**63:
        parser.error("Require a seed in [0,2**63).")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    args.checkpoint_path, metadata, stats = preflight(args)
    print(
        f"Validated {args.checkpoint_path}\nType: {metadata['checkpoint_type']}\nStatistics: {stats}"
        f"\nInference steps: {args.num_inference_steps}",
        flush=True,
    )
    if args.check_only:
        return
    if args.ready_file:
        if args.ready_file.exists():
            raise FileExistsError(f"Readiness file already exists: {args.ready_file}")
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="yam-inference") as worker:
        policy = worker.submit(initialize_policy, args).result()
        serve(args, policy, worker)


if __name__ == "__main__":
    main()
