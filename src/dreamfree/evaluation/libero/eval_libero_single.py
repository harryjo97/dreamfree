"""LIBERO single-task evaluation entrypoint for DreamFree Stage 2 checkpoints."""

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import dreamfree.evaluation.libero._eval_common as libero_eval
from dreamfree.evaluation.libero.action_ensembler import ActionEnsembler
from dreamfree.evaluation.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    save_rollout_video,
)
from dreamfree.utils.pytorch_utils import set_global_seed

PROJECT_ROOT = Path.cwd()
VALID_BENCHMARK_VARIANTS = {"libero"}
DEFAULT_EVAL_TEXT_CACHE_DIRS = {"libero": Path("./assets/text/libero")}


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{name} must be a boolean string, got {raw_value!r}.")


def _required_int_env(name: str) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        raise ValueError(f"{name} is required when LIBERO_DISTRIBUTED_PRETRAINED_LOAD=true.")
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc


@contextmanager
def _distributed_pretrained_load_context():
    enabled = _env_bool("LIBERO_DISTRIBUTED_PRETRAINED_LOAD", default=False)
    initialized_here = False
    if enabled:
        if not torch.distributed.is_available():
            raise RuntimeError(
                "torch.distributed is not available, but distributed pretrained load was enabled."
            )
        rank = _required_int_env("RANK")
        world_size = _required_int_env("WORLD_SIZE")
        if world_size <= 1:
            raise ValueError(
                f"LIBERO_DISTRIBUTED_PRETRAINED_LOAD=true requires WORLD_SIZE > 1, got WORLD_SIZE={world_size}."
            )
        for name in ("MASTER_ADDR", "MASTER_PORT"):
            if not os.environ.get(name, "").strip():
                raise ValueError(
                    f"{name} is required when LIBERO_DISTRIBUTED_PRETRAINED_LOAD=true."
                )
        backend = os.environ.get("LIBERO_DISTRIBUTED_BACKEND", "gloo").strip()
        timeout_raw = os.environ.get("LIBERO_DISTRIBUTED_INIT_TIMEOUT_SEC", "1800")
        try:
            timeout_sec = int(timeout_raw)
        except ValueError as exc:
            raise ValueError(
                f"LIBERO_DISTRIBUTED_INIT_TIMEOUT_SEC must be an integer, got {timeout_raw!r}."
            ) from exc
        if timeout_sec <= 0:
            raise ValueError(
                f"LIBERO_DISTRIBUTED_INIT_TIMEOUT_SEC must be positive, got {timeout_sec}."
            )
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend, init_method="env://", timeout=timedelta(seconds=timeout_sec)
            )
            initialized_here = True
        actual_world_size = int(torch.distributed.get_world_size())
        actual_rank = int(torch.distributed.get_rank())
        if actual_world_size != world_size or actual_rank != rank:
            raise RuntimeError(
                f"Distributed process group does not match env: env rank/world={rank}/{world_size}, actual={actual_rank}/{actual_world_size}."
            )
        logging.info(
            "Initialized distributed pretrained load: backend=%s rank=%d world_size=%d broadcast_device=%s",
            backend,
            actual_rank,
            actual_world_size,
            os.environ.get("LIBERO_PRETRAINED_BROADCAST_DEVICE", "cuda"),
        )
    try:
        yield
    finally:
        if initialized_here and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@contextmanager
def _libero_init_state_torch_load_compat():
    """Load trusted LIBERO init-state files across PyTorch 2.6+ defaults."""
    original_load = torch.load

    def load_with_legacy_default(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = load_with_legacy_default
    try:
        yield
    finally:
        torch.load = original_load


def _validate_eval_cfg(cfg: DictConfig) -> None:
    if cfg.checkpoint_path is None:
        raise ValueError("cfg.checkpoint_path must not be None.")
    _resolve_benchmark_variant(cfg)
    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. Use run_libero_manager.py for multi-GPU task parallelism."
        )
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError(
            "LIBERO DreamFree Stage 2 action-only eval does not support visualize_future_video=true."
        )
    if libero_eval._resolve_optional_string(cfg.EVALUATION.get("stage1_checkpoint")) is not None:
        raise ValueError(
            "LIBERO DreamFree eval loads Stage 1 from DreamFree Stage 2 metadata; do not set stage1_checkpoint."
        )


def _resolve_benchmark_variant(cfg: DictConfig) -> str:
    value = str(cfg.EVALUATION.get("benchmark_variant", "libero")).strip().lower().replace("-", "_")
    if value not in VALID_BENCHMARK_VARIANTS:
        raise ValueError(
            f"EVALUATION.benchmark_variant must be one of {sorted(VALID_BENCHMARK_VARIANTS)}, got {value!r}."
        )
    return value


def _parse_config_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{label} must be a boolean value, got {value!r}.")


def _ensure_eval_text_cache_cfg(cfg: DictConfig, *, benchmark_variant: str) -> None:
    model_cfg = cfg.get("model", {})
    load_text_encoder = _parse_config_bool(
        model_cfg.get("load_text_encoder", True), label="model.load_text_encoder"
    )
    if load_text_encoder:
        return
    cache_value = libero_eval._resolve_optional_string(
        cfg.EVALUATION.get("text_embedding_cache_dir", None)
    )
    if cache_value is not None:
        return
    cache_dir = DEFAULT_EVAL_TEXT_CACHE_DIRS[benchmark_variant]
    cfg.EVALUATION.text_embedding_cache_dir = str(cache_dir)
    logging.info(
        "model.load_text_encoder=false and EVALUATION.text_embedding_cache_dir was unset; using %s eval text cache: %s",
        benchmark_variant,
        cache_dir,
    )


def _resolve_save_rollout_videos(cfg: DictConfig) -> bool:
    value = cfg.EVALUATION.get("save_rollout_videos", True)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"EVALUATION.save_rollout_videos must be a boolean, got {value!r}.")


def _build_policy(
    cfg: DictConfig,
    *,
    ckpt_path: Path,
    checkpoint_metadata: dict[str, Any] | None,
    dataset_stats_path: Path,
):
    from dreamfree.evaluation.policy import build_policy

    options = dict(cfg.EVALUATION)
    options.setdefault("seed", cfg.get("seed", 42))
    options.update(
        checkpoint_path=str(ckpt_path),
        dataset_stats_path=str(dataset_stats_path),
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        load_text_encoder=cfg.model.get("load_text_encoder", True),
    )
    return build_policy("libero", options)


def _parse_task_batch_file(path: Path) -> list[tuple[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"EVALUATION.task_batch_file not found: {path}")
    task_specs: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid task batch line {line_idx} in {path}: {raw_line.rstrip()!r}"
                )
            (suite_name, task_id_raw) = parts
            if suite_name == "":
                raise ValueError(f"Empty task suite on line {line_idx} in {path}.")
            try:
                task_id = int(task_id_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid task id on line {line_idx} in {path}: {task_id_raw!r}"
                ) from exc
            if task_id < 0:
                raise ValueError(
                    f"Task id must be non-negative on line {line_idx} in {path}: {task_id}"
                )
            task_specs.append((suite_name, task_id))
    if not task_specs:
        raise ValueError(f"EVALUATION.task_batch_file contains no tasks: {path}")
    return task_specs


def _resolve_task_specs(cfg: DictConfig, *, base: Path) -> list[tuple[str, int]]:
    batch_file_raw = libero_eval._resolve_optional_string(cfg.EVALUATION.get("task_batch_file"))
    if batch_file_raw is None:
        return [(str(cfg.EVALUATION.task_suite_name), int(cfg.EVALUATION.task_id))]
    batch_file = libero_eval._resolve_path(batch_file_raw, base=base)
    return _parse_task_batch_file(batch_file)


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    task_suite_name: str,
    policy,
    cfg: DictConfig,
    episode_idx: int,
    collect_replay: bool,
) -> tuple[bool, list]:
    max_steps = _resolve_max_steps(cfg, task_suite_name=task_suite_name)
    replan_steps = int(policy.replan_steps)
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 30))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        reset_fn()
    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()
    replay_images = []
    pending_actions: list[list[float]] = []
    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            (obs, _, done, _) = env.step(get_libero_dummy_action())
            t += 1
            continue
        if len(pending_actions) == 0:
            action_chunk = policy.predict_action_chunk(obs, task_description)
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [
                    ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)
                ]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            if collect_replay:
                replay_images.append(get_libero_image(obs).copy())
        elif collect_replay:
            replay_images.append(get_libero_image(obs).copy())
        (obs, _, done, _) = env.step(pending_actions.pop(0))
        if done:
            break
        t += 1
    pbar.close()
    return (bool(done), replay_images)


def _emit_trial_progress(
    *,
    task_suite_name: str,
    task_id: int,
    trial_idx: int,
    num_trials: int,
    success: bool,
    successes: int,
    task_start_time: float,
) -> None:
    result = "success" if success else "failure"
    elapsed_sec = time.time() - task_start_time
    print(
        f"[libero-eval-progress] suite={task_suite_name} task={task_id} trial={trial_idx + 1}/{num_trials} result={result} successes={successes}/{trial_idx + 1} elapsed_sec={elapsed_sec:.1f}",
        flush=True,
    )


def run_single_task(
    task_suite_name: str,
    task_id: int,
    task,
    initial_states,
    policy,
    cfg: DictConfig,
    video_dir: Path | None,
    save_rollout_videos: bool,
) -> dict[str, Any]:
    (env, task_description) = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    results: dict[str, Any] = {
        "task_name": getattr(task, "name", ""),
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    num_trials = int(cfg.EVALUATION.num_trials)
    task_start_time = time.time()
    try:
        for trial_idx in range(num_trials):
            (success, replay_images) = run_single_episode(
                env=env,
                initial_state=initial_states[trial_idx],
                task_description=task_description,
                task_suite_name=task_suite_name,
                policy=policy,
                cfg=cfg,
                episode_idx=trial_idx,
                collect_replay=save_rollout_videos,
            )
            if success:
                results["successes"] += 1
                results["success_episodes"].append(trial_idx)
            else:
                results["failure_episodes"].append(trial_idx)
            _emit_trial_progress(
                task_suite_name=task_suite_name,
                task_id=task_id,
                trial_idx=trial_idx,
                num_trials=num_trials,
                success=success,
                successes=results["successes"],
                task_start_time=task_start_time,
            )
            if save_rollout_videos:
                if video_dir is None:
                    raise ValueError("video_dir must be provided when save_rollout_videos=true.")
                save_rollout_video(
                    video_dir,
                    replay_images,
                    f"task{task_id}_trial{trial_idx}",
                    success=success,
                    task_description=task_description,
                )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return results


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _resolve_max_steps(cfg: DictConfig, *, task_suite_name: str) -> int:
    max_steps_override = cfg.EVALUATION.get("max_steps", None)
    if max_steps_override is None:
        return _get_max_steps(str(task_suite_name))
    max_steps = int(max_steps_override)
    if max_steps <= 0:
        raise ValueError(f"EVALUATION.max_steps must be positive, got {max_steps_override!r}.")
    return max_steps


def _build_base_results(
    *,
    cfg: DictConfig,
    task_suite_name: str,
    task_id: int,
    ckpt_path: Path,
    checkpoint_metadata: dict[str, Any] | None,
    dataset_stats_path: Path,
    policy,
    benchmark_variant: str,
) -> dict[str, Any]:
    return {
        "benchmark_variant": benchmark_variant,
        "task_suite": task_suite_name,
        "task_id": int(task_id),
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "checkpoint_path": str(ckpt_path),
        "checkpoint_type": checkpoint_metadata.get("checkpoint_type")
        if checkpoint_metadata
        else None,
        "source_stage1_checkpoint": libero_eval._source_checkpoint_for_results(checkpoint_metadata),
        "dataset_stats_path": str(dataset_stats_path),
        "action_data_source": libero_eval._require_single_action_data_source(checkpoint_metadata),
        "image_preprocess_source": getattr(policy, "image_preprocess_source", None),
        "gripper_postprocess": getattr(policy, "gripper_postprocess", None),
        "policy_runtime": libero_eval._policy_runtime_for_checkpoint_metadata(checkpoint_metadata),
        "compile_dit": bool(getattr(policy, "compile_dit", False)),
        "compile_dit_block_counts": dict(getattr(policy, "compile_dit_block_counts", {})),
        "action_horizon": int(policy.action_horizon),
        "replan_steps": int(policy.replan_steps),
        "num_inference_steps": int(policy.num_inference_steps),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }


def _write_task_results(
    *,
    cfg: DictConfig,
    run_output_dir: Path,
    task_suite_name: str,
    task_id: int,
    results: dict[str, Any],
) -> Path:
    output_dir = Path(cfg.EVALUATION.output_dir) / task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    return output_file


def eval_single_process(cfg: DictConfig):
    _validate_eval_cfg(cfg)
    benchmark_variant = _resolve_benchmark_variant(cfg)
    _ensure_eval_text_cache_cfg(cfg, benchmark_variant=benchmark_variant)
    save_rollout_videos = _resolve_save_rollout_videos(cfg)
    env = os.environ
    env["CUDA_VISIBLE_DEVICES"] = libero_eval._resolve_cuda_visible_devices(
        gpu_id=cfg.gpu_id, inherited_env=env
    )
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    ckpt_path = libero_eval._resolve_path(str(cfg.checkpoint_path), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint_metadata = libero_eval._load_checkpoint_metadata(ckpt_path)
    libero_eval._require_dreamfree_stage2_checkpoint(
        checkpoint_metadata, label="LIBERO DreamFree eval"
    )
    dataset_stats_path = libero_eval._resolve_dataset_stats_path(
        explicit_value=cfg.EVALUATION.get("dataset_stats_path"),
        ckpt_path=ckpt_path,
        checkpoint_metadata=checkpoint_metadata,
    )
    output_root_raw = libero_eval._resolve_optional_string(cfg.EVALUATION.output_dir)
    if output_root_raw is None:
        raise ValueError("EVALUATION.output_dir must not be None.")
    output_root = libero_eval._resolve_path(output_root_raw, base=PROJECT_ROOT)
    run_ts = libero_eval._resolve_run_ts(cfg.EVALUATION.get("run_ts"))
    run_output_dir = libero_eval._build_libero_run_output_dir(
        output_root=output_root, ckpt_tag=libero_eval._resolve_ckpt_tag(ckpt_path), run_ts=run_ts
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cfg.EVALUATION.output_dir = str(run_output_dir)
    task_specs = _resolve_task_specs(cfg, base=PROJECT_ROOT)
    logging.info("Using dataset stats: %s", dataset_stats_path)
    with _distributed_pretrained_load_context():
        policy = _build_policy(
            cfg,
            ckpt_path=ckpt_path,
            checkpoint_metadata=checkpoint_metadata,
            dataset_stats_path=dataset_stats_path,
        )
    logging.info(
        "Policy compile_dit=%s compile_dit_block_counts=%s",
        bool(getattr(policy, "compile_dit", False)),
        dict(getattr(policy, "compile_dit_block_counts", {})),
    )
    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dict = benchmark.get_benchmark_dict()
    all_results: list[dict[str, Any]] = []
    logging.info("Running LIBERO DreamFree evaluation with env_num=1 tasks=%d", len(task_specs))
    for task_suite_name, task_id in task_specs:
        start_time = time.time()
        if task_suite_name not in benchmark_dict:
            raise ValueError(
                f"Unknown task suite {task_suite_name!r}. Available suites: {sorted(benchmark_dict.keys())}"
            )
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        with _libero_init_state_torch_load_compat():
            initial_states = task_suite.get_task_init_states(task_id)
        while len(initial_states) < int(cfg.EVALUATION.num_trials):
            initial_states.extend(
                initial_states[: int(cfg.EVALUATION.num_trials) - len(initial_states)]
            )
        video_dir = None
        if save_rollout_videos:
            video_dir = local_log_dir / task_suite_name / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
        results = _build_base_results(
            cfg=cfg,
            task_suite_name=task_suite_name,
            task_id=task_id,
            ckpt_path=ckpt_path,
            checkpoint_metadata=checkpoint_metadata,
            dataset_stats_path=dataset_stats_path,
            policy=policy,
            benchmark_variant=benchmark_variant,
        )
        task_results = run_single_task(
            task_suite_name=task_suite_name,
            task_id=task_id,
            task=task,
            initial_states=initial_states,
            policy=policy,
            cfg=cfg,
            video_dir=video_dir,
            save_rollout_videos=save_rollout_videos,
        )
        results.update(task_results)
        results["duration"] = time.time() - start_time
        _write_task_results(
            cfg=cfg,
            run_output_dir=run_output_dir,
            task_suite_name=task_suite_name,
            task_id=task_id,
            results=results,
        )
        all_results.append(results)
        print(
            f"Task {task_suite_name}/{task_id} completed: {results['successes']}/{cfg.EVALUATION.num_trials} successes"
        )
        print(f"Time taken: {results['duration']:.2f} seconds")
    task_batch_file = libero_eval._resolve_optional_string(cfg.EVALUATION.get("task_batch_file"))
    if task_batch_file is None:
        config_name = f"eval_config_{task_specs[0][0]}_task{int(task_specs[0][1])}.yaml"
    else:
        config_name = f"eval_config_batch_{Path(task_batch_file).stem}.yaml"
    OmegaConf.save(config=cfg, f=str(run_output_dir / config_name))
    return all_results[0] if len(all_results) == 1 else all_results
