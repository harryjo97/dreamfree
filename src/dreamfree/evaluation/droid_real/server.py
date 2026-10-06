"""Serve an DreamFree Stage 2 policy for a physical DROID robot."""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

from dreamfree.evaluation.droid.deploy_policy import build_droid_image_tensor
from dreamfree.evaluation.droid.deploy_policy_optimized import build_policy_from_args
from dreamfree.evaluation.droid_real.image_preprocessing import (
    IMAGE_KEYS,
    PREPROCESSING_FORMAT,
    PREPROCESSING_KEY,
    TRANSPORT_FORMATS,
    preprocess_observation,
    transport_dtype_from_format,
)
from dreamfree.evaluation.droid_real.policy_rpc import PICKLE_PROTOCOL, PolicyRpcServer

NATIVE_POLICY_RUNTIME = "dreamfree_stage2_student_native_optimized"
OPTIMIZED_NATIVE_POLICY_RUNTIME = "dreamfree_stage2_student_native_optimized"
OPTIMIZED_POLICY_RUNTIMES = (OPTIMIZED_NATIVE_POLICY_RUNTIME,)
NATIVE_SERVER_NAME = "dreamfree_droid_stage2_real_optimized"
RPC_PROTOCOL_VERSION = 4
GRIPPER_PREDICTION_REPRESENTATION = "continuous_denormalized"
GRIPPER_BINARIZATION = "client_threshold_gt_0.5"
MAX_ACTION_NOISE_SEED = (1 << 63) - 1
OBSERVATION_KEYS = (
    "observation/exterior_image_1_left",
    "observation/exterior_image_2_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/gripper_position",
)


def image_preprocessing_info(model_dtype: torch.dtype) -> dict[str, Any]:
    return {
        "format": PREPROCESSING_FORMAT,
        "formats": {
            key: value
            for (key, value) in TRANSPORT_FORMATS.items()
            if key != "bf16" or model_dtype == torch.bfloat16
        },
        "model_input_dtype": str(model_dtype).removeprefix("torch."),
        "validation_endpoint": "validate_image_preprocessing",
        "torch_version": str(torch.__version__),
    }


def validate_image_preprocessing(
    request: dict[str, Any], *, model_dtype: torch.dtype = torch.bfloat16
) -> dict[str, Any]:
    """Compare CPU model inputs without calling the policy or consuming RNG."""
    raw = request.get("raw_observation")
    processed = request.get("processed_observation")
    if not isinstance(raw, dict) or not isinstance(processed, dict):
        raise TypeError(
            "Image preprocessing validation requires raw and processed observation dicts."
        )
    if PREPROCESSING_KEY in raw:
        raise ValueError(
            "Image preprocessing validation requires raw images and explicitly marked processed images."
        )
    format_name = processed.get(PREPROCESSING_KEY)
    transport_dtype = transport_dtype_from_format(format_name)
    if transport_dtype == "bf16" and model_dtype != torch.bfloat16:
        raise ValueError("BF16 image transport requires a bfloat16 model input.")
    candidate = build_droid_image_tensor(processed)
    expected = preprocess_observation(raw, transport_dtype)
    transport_equal = all(processed[key].tobytes() == expected[key].tobytes() for key in IMAGE_KEYS)
    reference = build_droid_image_tensor(raw)
    if transport_dtype != "fp32":
        reference = reference.to(model_dtype).float()
        candidate = candidate.to(model_dtype).float()
    difference = float((reference - candidate).abs().max().item())
    equal = reference.numpy().tobytes() == candidate.numpy().tobytes()
    shapes = {key: list(raw[key].shape) for key in IMAGE_KEYS}
    if not transport_equal or (transport_dtype != "uint8" and (not equal)):
        raise ValueError(
            f"Image preprocessing parity failed: max_abs_diff={difference}, raw_shapes={shapes}, client_torch={request.get('client_torch_version')}, server_torch={torch.__version__}. No policy inference was performed."
        )
    return {
        "format": format_name,
        "exact_equal": equal,
        "transport_exact_equal": transport_equal,
        "lossy": transport_dtype == "uint8",
        "model_input_dtype": str(model_dtype).removeprefix("torch."),
        "max_abs_diff": difference,
        "server_torch_version": str(torch.__version__),
    }


def _action_noise_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an integer action-noise seed, got {value!r}."
        ) from exc
    if not 0 <= seed <= MAX_ACTION_NOISE_SEED:
        raise argparse.ArgumentTypeError(
            f"Action-noise seed must be in [0,{MAX_ACTION_NOISE_SEED}], got {seed}."
        )
    return seed


def _parse_args(
    argv: Sequence[str] | None = None, *, description: str | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--ckpt", required=True, help="DreamFree Stage 2 checkpoint directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--droid-dataset-root", default="./assets")
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--action-sampler", default="euler", choices=("euler", "unipc"))
    parser.add_argument("--action-infer-shift", type=float, default=None)
    parser.add_argument("--action-unipc-solver-order", type=int, default=2)
    parser.add_argument("--action-unipc-solver-type", default="bh2", choices=("bh1", "bh2"))
    parser.add_argument("--action-unipc-lower-order-final", default="true")
    parser.add_argument("--action-unipc-disable-corrector", default=None)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--random-action-noise", action="store_true")
    parser.add_argument("--action-noise-seed", type=_action_noise_seed, default=None)
    parser.add_argument("--readout-active-num-queries", type=int, default=None)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--timing-enabled", action="store_true")
    parser.set_defaults(compile_dit=True)
    parser.add_argument(
        "--compile-auxiliary",
        choices=("none",),
        default=None,
        help="Additional compilation for the optimized 14B policy; none selects DiT-only compilation.",
    )
    parser.add_argument("--fuse-qkv", default="true", choices=("true", "false"))
    parser.add_argument(
        "--action-cuda-graph",
        default="auto",
        choices=("auto", "true", "false"),
        help="Replay the validated 16-step Euler action loop with CUDA graphs.",
    )
    parser.add_argument(
        "--vae-first-frame-only",
        default="auto",
        choices=("auto", "true", "false"),
        help="Encode only the consumed first latent frame in the validated 14B visual path.",
    )
    parser.add_argument(
        "--stage1-cuda-graph",
        default="auto",
        choices=("auto", "true", "false"),
        help="Replay validated current-frame Stage 1 feature extraction with a CUDA graph.",
    )
    parser.add_argument("--cudnn-benchmark", default="true", choices=("true", "false"))
    parser.add_argument("--text-embedding-cache-dir", default=None)
    args = parser.parse_args(argv)
    if args.random_action_noise and args.action_noise_seed is None:
        parser.error("--random-action-noise requires --action-noise-seed.")
    if not args.random_action_noise and args.action_noise_seed is not None:
        parser.error("--action-noise-seed requires --random-action-noise.")
    return args


def _require_loopback(host: str) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        if host == "localhost":
            return
        raise ValueError(
            f"Physical DROID pickle RPC must bind to a loopback address and be accessed through SSH; got host={host!r}."
        ) from exc
    if not address.is_loopback:
        raise ValueError(
            f"Physical DROID pickle RPC must bind to a loopback address and be accessed through SSH; got host={host!r}."
        )


def _runtime_provenance(policy: Any, policy_runtime: str) -> dict[str, Any]:
    if policy.policy_runtime != policy_runtime:
        raise RuntimeError("Constructed policy runtime does not match the requested runtime.")
    contract = getattr(policy, "optimization_contract", None)
    if not isinstance(contract, dict):
        raise RuntimeError("Optimized DROID inference must report its optimization contract.")
    required = {
        "attention_policy": "sdpa_fused_no_math",
        "action_cross_kv_cache": True,
        "compile_dit": True,
        "warmup_complete": True,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"Optimized DROID inference requires {key}={expected!r}.")
    return {"optimization_contract": dict(contract)}


def _checkpoint_type(policy: Any) -> str:
    metadata = getattr(policy, "stage2_metadata", None)
    if not isinstance(metadata, dict):
        raise RuntimeError("Physical DROID policy is missing Stage 2 checkpoint metadata.")
    checkpoint_type = str(metadata.get("checkpoint_type", "")).strip()
    if not checkpoint_type:
        raise RuntimeError("Physical DROID Stage 2 checkpoint metadata is missing checkpoint_type.")
    return checkpoint_type


def _reset_policy_for_rollout(
    policy: Any, *, random_action_noise: bool, episode_seed: int | None
) -> dict[str, Any]:
    if random_action_noise and episode_seed is None:
        raise ValueError("Random action noise requires episode_seed on every reset.")
    if not random_action_noise and episode_seed is not None:
        raise ValueError("episode_seed is only valid when random action noise is enabled.")
    policy.reset()
    derived_seed = None
    if random_action_noise:
        policy.reset_action_noise_for_episode(episode_seed)
        derived_seed = getattr(policy, "action_noise_derived_seed", None)
        if (
            isinstance(derived_seed, bool)
            or not isinstance(derived_seed, int)
            or (not 0 <= derived_seed <= MAX_ACTION_NOISE_SEED)
        ):
            raise RuntimeError(
                f"Policy did not report action_noise_derived_seed as an integer in [0,{MAX_ACTION_NOISE_SEED}] after reset."
            )
    return {"episode_seed": episode_seed, "action_noise_derived_seed": derived_seed}


def run_server(
    *,
    policy_builder: Callable[[dict[str, Any]], Any] = build_policy_from_args,
    server_name: str = NATIVE_SERVER_NAME,
    policy_runtime: str = NATIVE_POLICY_RUNTIME,
    evaluation_mode: str = NATIVE_POLICY_RUNTIME,
    argv: Sequence[str] | None = None,
    description: str | None = None,
) -> None:
    args = _parse_args(argv, description=description)
    if (
        args.compile_auxiliary not in (None, "none")
        and policy_runtime != OPTIMIZED_NATIVE_POLICY_RUNTIME
    ):
        raise ValueError("--compile-auxiliary requires the optimized 14B runtime.")
    if args.action_cuda_graph == "true" and (
        policy_runtime != OPTIMIZED_NATIVE_POLICY_RUNTIME
        or args.action_sampler != "euler"
        or args.num_inference_steps != 16
    ):
        raise ValueError("--action-cuda-graph=true requires optimized 14B and 16-step Euler.")
    if (args.vae_first_frame_only == "true" or args.stage1_cuda_graph == "true") and (
        policy_runtime != OPTIMIZED_NATIVE_POLICY_RUNTIME
        or args.action_sampler != "euler"
        or args.num_inference_steps != 16
    ):
        raise ValueError("Visual optimizations require optimized 14B and 16-step Euler.")
    if args.vae_first_frame_only == "true" and args.compile_auxiliary not in (None, "none"):
        raise ValueError(
            "--vae-first-frame-only=true requires --compile-auxiliary=none; compiled prefix failed numerical validation."
        )
    _require_loopback(args.host)
    if args.port <= 0 or args.port > 65535:
        raise ValueError(f"port must be in [1,65535], got {args.port}.")
    if args.replan_steps <= 0:
        raise ValueError(f"replan_steps must be positive, got {args.replan_steps}.")
    if args.num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {args.num_inference_steps}.")
    logging.basicConfig(level=logging.INFO, force=True)
    if args.random_action_noise and args.seed is not None:
        logging.info(
            "Random action noise enabled with --action-noise-seed=%d; --seed=%d is ignored.",
            args.action_noise_seed,
            args.seed,
        )
    with ExitStack() as stack:
        executor = stack.enter_context(
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="dreamfree-policy")
        )
        _serve_policy(
            args,
            policy_builder=policy_builder,
            server_name=server_name,
            policy_runtime=policy_runtime,
            evaluation_mode=evaluation_mode,
            executor=executor,
        )


def _serve_policy(
    args: argparse.Namespace,
    *,
    policy_builder: Callable[[dict[str, Any]], Any],
    server_name: str,
    policy_runtime: str,
    evaluation_mode: str,
    executor: ThreadPoolExecutor | None,
) -> None:

    def call_policy(function: Callable, *args: Any, **kwargs: Any) -> Any:
        if executor is None:
            return function(*args, **kwargs)
        return executor.submit(function, *args, **kwargs).result()

    resolved_ckpt = str(Path(args.ckpt).expanduser().resolve())
    resolved_dataset_root = str(Path(args.droid_dataset_root).expanduser().resolve())
    resolved_dataset_stats_path = (
        None
        if args.dataset_stats_path is None
        else str(Path(args.dataset_stats_path).expanduser().resolve())
    )
    policy_seed = None if args.random_action_noise else args.seed
    policy = call_policy(
        policy_builder,
        {
            "ckpt_setting": resolved_ckpt,
            "device": args.device,
            "mixed_precision": args.mixed_precision,
            "droid_dataset_root": resolved_dataset_root,
            "dataset_stats_path": resolved_dataset_stats_path,
            "replan_steps": args.replan_steps,
            "num_inference_steps": args.num_inference_steps,
            "action_sampler": args.action_sampler,
            "action_infer_shift": args.action_infer_shift,
            "action_unipc_solver_order": args.action_unipc_solver_order,
            "action_unipc_solver_type": args.action_unipc_solver_type,
            "action_unipc_lower_order_final": args.action_unipc_lower_order_final,
            "action_unipc_disable_corrector": args.action_unipc_disable_corrector,
            "rand_device": args.rand_device,
            "seed": policy_seed,
            "action_noise_seed": args.action_noise_seed,
            "readout_active_num_queries": args.readout_active_num_queries,
            "tiled": args.tiled,
            "timing_enabled": args.timing_enabled,
            "compile_dit": args.compile_dit,
            "compile_auxiliary": args.compile_auxiliary,
            "action_cuda_graph": args.action_cuda_graph,
            "vae_first_frame_only": args.vae_first_frame_only,
            "stage1_cuda_graph": args.stage1_cuda_graph,
            "fuse_qkv": args.fuse_qkv,
            "cudnn_benchmark": args.cudnn_benchmark,
            "load_text_encoder": True,
            "text_embedding_cache_dir": args.text_embedding_cache_dir,
        },
    )
    if policy_runtime in OPTIMIZED_POLICY_RUNTIMES:
        warmup = getattr(policy, "warmup_optimized_inference", None)
        if not callable(warmup):
            raise RuntimeError(
                "Optimized physical DROID policy is missing its mandatory warmup method."
            )
        warmup_seconds = float(call_policy(warmup))
        logging.info("Optimized DROID inference warmup completed in %.3f seconds.", warmup_seconds)
    runtime_provenance = _runtime_provenance(policy, policy_runtime)
    checkpoint_type = _checkpoint_type(policy)

    def infer_fn(observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        device = torch.device(policy.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        actions = policy.predict_action_chunk(observation, instruction, binarize_gripper=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "actions": actions,
            "server_timing": {"infer_ms": (time.perf_counter() - start) * 1000.0},
        }

    def reset_fn(episode_seed: int | None) -> dict[str, Any]:
        return _reset_policy_for_rollout(
            policy, random_action_noise=args.random_action_noise, episode_seed=episode_seed
        )

    def runtime_environment() -> dict[str, Any]:
        device = torch.device(policy.device)
        return {
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cpu_intraop_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }

    info = {
        "server": server_name,
        "rpc_protocol_version": RPC_PROTOCOL_VERSION,
        "serialization": f"pickle_protocol_{PICKLE_PROTOCOL}",
        "transport_security": "localhost_only_use_ssh_tunnel",
        "checkpoint_path": resolved_ckpt,
        "checkpoint_type": checkpoint_type,
        "host": args.host,
        "port": int(args.port),
        "device": args.device,
        "mixed_precision": args.mixed_precision,
        "droid_dataset_root": resolved_dataset_root,
        "dataset_stats_path": resolved_dataset_stats_path,
        "observation_keys": list(OBSERVATION_KEYS),
        "image_preprocessing": image_preprocessing_info(policy.model_dtype),
        "action_space": "joint_position",
        "action_representation": "absolute",
        "gripper_action_space": "position",
        "gripper_prediction_representation": GRIPPER_PREDICTION_REPRESENTATION,
        "gripper_binarization": GRIPPER_BINARIZATION,
        "action_dim": 8,
        "control_frequency_hz": 15,
        "action_horizon": int(policy.action_horizon),
        "action_history_horizon": getattr(policy, "action_history_horizon", None),
        "replan_steps": int(policy.replan_steps),
        "num_inference_steps": int(policy.num_inference_steps),
        "action_sampler": policy.action_sampler,
        "action_infer_shift": policy.action_infer_shift,
        "action_train_shift": policy.action_train_shift,
        "action_unipc_solver_order": int(policy.action_unipc_solver_order),
        "action_unipc_solver_type": policy.action_unipc_solver_type,
        "action_unipc_lower_order_final": bool(policy.action_unipc_lower_order_final),
        "action_unipc_disable_corrector": list(policy.action_unipc_disable_corrector),
        "rand_device": args.rand_device,
        "seed": policy_seed,
        "policy_seed": policy_seed,
        "random_action_noise": bool(args.random_action_noise),
        "action_noise_seed": args.action_noise_seed,
        "requires_episode_seed_on_reset": bool(args.random_action_noise),
        "policy_runtime": policy_runtime,
        "evaluation_mode": evaluation_mode,
        "readout_active_num_queries": args.readout_active_num_queries,
        "tiled": bool(args.tiled),
        "timing_enabled": bool(args.timing_enabled),
        "server_timing_enabled": True,
        "server_timing_scope": "policy_call",
        "runtime_environment": call_policy(runtime_environment),
        "compile_dit": bool(policy.compile_dit),
        "compile_dit_block_counts": dict(policy.compile_dit_block_counts),
        "load_text_encoder": True,
        "text_embedding_cache_dir": args.text_embedding_cache_dir,
        **runtime_provenance,
    }
    if executor is not None:
        info["policy_execution"] = {
            "mode": "persistent_worker",
            "worker_native_thread_id": call_policy(threading.get_native_id),
            "initialization_warmup_reset_infer_same_thread": True,
        }
    try:
        json.dumps(info, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("Physical DROID policy info must be JSON serializable.") from exc
    server = PolicyRpcServer(
        (args.host, args.port),
        infer_fn=lambda observation, instruction: call_policy(infer_fn, observation, instruction),
        reset_fn=lambda episode_seed: call_policy(reset_fn, episode_seed),
        info_fn=lambda: info,
        validate_image_preprocessing_fn=lambda request: call_policy(
            validate_image_preprocessing, request, model_dtype=policy.model_dtype
        ),
    )
    logging.info(
        "Serving physical DreamFree DROID policy runtime=%s on %s:%d via pickle protocol %d",
        policy_runtime,
        args.host,
        args.port,
        PICKLE_PROTOCOL,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    run_server()


if __name__ == "__main__":
    main()
