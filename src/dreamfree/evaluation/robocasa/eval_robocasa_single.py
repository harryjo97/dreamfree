"""RoboCasa single-task evaluation entrypoint for DreamFree Stage 2 checkpoints."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import dreamfree.evaluation.robocasa._eval_common as robocasa_eval
from dreamfree.evaluation.robocasa.results_html import write_robocasa_results_html
from dreamfree.evaluation.robocasa.robocasa_utils import (
    INITIAL_STATE_MAX_LINEAR_SPEED,
    INITIAL_STATE_MAX_Z_ERROR,
    INITIAL_STATE_VALIDATION_TASKS,
    create_robocasa_env,
    ensure_robocasa_import,
    get_base_task_name,
    get_task_max_steps,
    list_to_jsonable,
    pad_action_for_env,
    prepare_robocasa_observation,
    record_issued_action_if_enabled,
    reset_and_warm_up_robocasa_env,
    save_rollout_video,
    set_seed_everywhere,
    validate_task_name,
)

PROJECT_ROOT = Path.cwd()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _setup_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream)
    root_logger.addHandler(file_handler)
    return log_path


def _validate_eval_cfg(cfg: DictConfig) -> None:
    if cfg.checkpoint_path is None:
        raise ValueError("`checkpoint_path` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")
    if int(cfg.EVALUATION.get("episode_start_idx", 0)) < 0:
        raise ValueError("`EVALUATION.episode_start_idx` must be non-negative.")
    if robocasa_eval._resolve_optional_string(cfg.EVALUATION.get("stage1_checkpoint")) is not None:
        raise ValueError(
            "RoboCasa DreamFree eval loads Stage 1 from DreamFree Stage 2 metadata; do not set stage1_checkpoint."
        )
    get_base_task_name(cfg.EVALUATION.task_name)


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
    return build_policy("robocasa", options)


def run_single_episode(
    env,
    initial_obs: dict[str, Any],
    task_description: str,
    policy,
    cfg: DictConfig,
    episode_idx: int,
) -> tuple[bool, int, list[dict[str, np.ndarray]], dict[str, Any] | None]:
    replan_steps = int(policy.replan_steps)
    max_steps = get_task_max_steps(
        cfg.EVALUATION.task_name, cfg.EVALUATION.get("max_steps_override", None)
    )
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        reset_fn()
    obs = initial_obs
    pending_actions: deque[np.ndarray] = deque()
    replay_frames: list[dict[str, np.ndarray]] = []
    success = False
    episode_length = 0
    progress = tqdm(total=max_steps, desc=f"Trial {episode_idx}", leave=False)
    try:
        for t in range(max_steps):
            progress.update(1)
            if not pending_actions:
                action_chunk = policy.predict_action_chunk(obs, task_description)
                replay_frames.append(
                    prepare_robocasa_observation(obs, flip_images=bool(cfg.EVALUATION.flip_images))
                )
                for action in action_chunk[:replan_steps]:
                    pending_actions.append(np.asarray(action, dtype=np.float32))
            else:
                replay_frames.append(
                    prepare_robocasa_observation(obs, flip_images=bool(cfg.EVALUATION.flip_images))
                )
            policy_action = pending_actions.popleft()
            env_action = pad_action_for_env(policy_action, env)
            (obs, _reward, done, _info) = env.step(env_action)
            record_issued_action_if_enabled(policy, policy_action)
            episode_length = t + 1
            native_success = bool(env._check_success())
            if native_success:
                success = True
                logging.info("Success detected at trial=%d step=%d.", episode_idx, t)
                break
            if bool(done):
                break
    finally:
        progress.close()
    subgoal_trace = None
    return (success, episode_length, replay_frames, subgoal_trace)


def run_task(policy, cfg: DictConfig, *, output_dir: Path) -> dict[str, Any]:
    eval_cfg = cfg.EVALUATION
    num_trials = int(eval_cfg.num_trials)
    episode_start_idx = int(eval_cfg.get("episode_start_idx", 0))
    if episode_start_idx < 0:
        raise ValueError(
            f"EVALUATION.episode_start_idx must be non-negative, got {episode_start_idx}."
        )
    initial_state_validation_enabled = bool(eval_cfg.get("initial_state_validation", True))
    initial_state_max_reset_attempts = int(eval_cfg.get("initial_state_max_reset_attempts", 5))
    if initial_state_max_reset_attempts < 1:
        raise ValueError(
            f"EVALUATION.initial_state_max_reset_attempts must be at least 1, got {initial_state_max_reset_attempts}."
        )
    num_steps_wait = int(eval_cfg.get("num_steps_wait", 10))
    if num_steps_wait < 0:
        raise ValueError(f"EVALUATION.num_steps_wait must be non-negative, got {num_steps_wait}.")
    video_dir = output_dir / "videos"
    successes: list[bool] = []
    episode_lengths: list[int] = []
    episode_indices: list[int] = []
    success_episodes: list[int] = []
    failure_episodes: list[int] = []
    task_descriptions: list[str] = []
    reset_attempt_counts: list[int] = []
    episode_variants: list[dict[str, Any]] = []
    env_kwargs_record: Optional[dict[str, Any]] = None
    base_task_name = get_base_task_name(eval_cfg.task_name)
    for local_trial_idx in range(num_trials):
        episode_idx = episode_start_idx + local_trial_idx
        episode_indices.append(episode_idx)
        if bool(eval_cfg.get("deterministic", True)) or bool(
            eval_cfg.get("deterministic_reset", False)
        ):
            seed = int(eval_cfg.seed) * episode_idx * 256
        else:
            seed = None
        env = None
        try:
            (env, env_kwargs) = create_robocasa_env(
                eval_cfg, project_root=PROJECT_ROOT, seed=seed, episode_idx=episode_idx
            )
            if env_kwargs_record is None:
                env_kwargs_record = {
                    k: str(v) if k == "controller_configs" else v for (k, v) in env_kwargs.items()
                }
            fixture_style_overrides = env_kwargs.get("fixture_style_overrides", {})
            episode_variants.append(
                {
                    "episode_idx": episode_idx,
                    "layout_and_style_ids": env_kwargs.get("layout_and_style_ids"),
                    "coffee_machine_asset": fixture_style_overrides.get("coffee_machine"),
                }
            )
            if bool(eval_cfg.get("deterministic_reset", False)):
                reset_seed = eval_cfg.get("deterministic_reset_seed", None)
                set_seed_everywhere(int(eval_cfg.seed if reset_seed is None else reset_seed))
            (obs, reset_attempt_count, _validation) = reset_and_warm_up_robocasa_env(
                env,
                task_name=base_task_name,
                num_steps_wait=num_steps_wait,
                max_reset_attempts=initial_state_max_reset_attempts,
                episode_idx=episode_idx,
                seed=seed,
                validation_enabled=initial_state_validation_enabled,
            )
            reset_attempt_counts.append(reset_attempt_count)
            task_description = str(env.get_ep_meta()["lang"])
            task_descriptions.append(task_description)
            logging.info(
                "Trial %d/%d | episode_idx=%d | seed=%s | reset_attempts=%d | instruction=%s",
                local_trial_idx + 1,
                num_trials,
                episode_idx,
                seed,
                reset_attempt_count,
                task_description,
            )
            (success, episode_length, replay_frames, subgoal_episode_trace) = run_single_episode(
                env=env,
                initial_obs=obs,
                task_description=task_description,
                policy=policy,
                cfg=cfg,
                episode_idx=episode_idx,
            )
        finally:
            if env is not None:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()
        successes.append(bool(success))
        episode_lengths.append(int(episode_length))
        if success:
            success_episodes.append(episode_idx)
        else:
            failure_episodes.append(episode_idx)
        if bool(eval_cfg.get("save_videos", True)):
            status = "success" if success else "failure"
            save_rollout_video(
                video_dir / f"trial{episode_idx:03d}_{status}.mp4", replay_frames, fps=20
            )
        logging.info(
            "Trial %d/%d episode_idx=%d result: %s length=%d | running success=%d/%d",
            local_trial_idx + 1,
            num_trials,
            episode_idx,
            "SUCCESS" if success else "FAILURE",
            episode_length,
            int(sum(successes)),
            len(successes),
        )
    success_count = int(sum(successes))
    retried_episodes = [
        episode_idx
        for (episode_idx, attempt_count) in zip(episode_indices, reset_attempt_counts, strict=True)
        if attempt_count > 1
    ]
    results = {
        "task_name": str(eval_cfg.task_name),
        "base_task_name": base_task_name,
        "evaluation_protocol": "standard",
        "num_trials": num_trials,
        "episode_start_idx": episode_start_idx,
        "episode_indices": list_to_jsonable(episode_indices),
        "successes": success_count,
        "success_rate": float(success_count / max(num_trials, 1)),
        "episode_lengths": list_to_jsonable(episode_lengths),
        "avg_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "success_episodes": list_to_jsonable(success_episodes),
        "failure_episodes": list_to_jsonable(failure_episodes),
        "task_descriptions": task_descriptions,
        "reset_attempt_counts": list_to_jsonable(reset_attempt_counts),
        "retried_episodes": list_to_jsonable(retried_episodes),
        "total_reset_attempts": int(sum(reset_attempt_counts)),
        "initial_state_validation": {
            "enabled": initial_state_validation_enabled,
            "registered_tasks": list(INITIAL_STATE_VALIDATION_TASKS),
            "max_reset_attempts": initial_state_max_reset_attempts,
            "max_z_error": INITIAL_STATE_MAX_Z_ERROR,
            "max_linear_speed": INITIAL_STATE_MAX_LINEAR_SPEED,
        },
        "env_kwargs": env_kwargs_record,
        "episode_variants": episode_variants,
    }
    return results


def main(cfg: DictConfig):
    start_time = time.time()
    _validate_eval_cfg(cfg)
    env = os.environ
    env["CUDA_VISIBLE_DEVICES"] = robocasa_eval._resolve_cuda_visible_devices(
        gpu_id=cfg.gpu_id, inherited_env=env
    )
    ckpt_path = robocasa_eval._resolve_path(str(cfg.checkpoint_path), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint_metadata = robocasa_eval._load_checkpoint_metadata(ckpt_path)
    robocasa_eval._require_dreamfree_stage2_checkpoint(
        checkpoint_metadata, label="RoboCasa DreamFree eval"
    )
    dataset_stats_path = robocasa_eval._resolve_dataset_stats_path(
        explicit_value=cfg.EVALUATION.get("dataset_stats_path"),
        ckpt_path=ckpt_path,
        checkpoint_metadata=checkpoint_metadata,
    )
    output_root_raw = robocasa_eval._resolve_optional_string(cfg.EVALUATION.output_dir)
    if output_root_raw is None:
        raise ValueError("EVALUATION.output_dir must not be None.")
    output_root = robocasa_eval._resolve_path(output_root_raw, base=PROJECT_ROOT)
    run_ts = robocasa_eval._resolve_run_ts(cfg.EVALUATION.get("run_ts"))
    run_output_dir = robocasa_eval._build_robocasa_run_output_dir(
        output_root=output_root,
        ckpt_tag=robocasa_eval._resolve_ckpt_tag(ckpt_path),
        run_ts=run_ts,
        output_namespace=robocasa_eval._resolve_optional_string(
            cfg.EVALUATION.get("output_namespace", "robocasa_dreamfree")
        ),
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cfg.EVALUATION.output_dir = str(run_output_dir)
    task_output_dir = run_output_dir / str(cfg.EVALUATION.task_name)
    log_path = _setup_logging(task_output_dir)
    ensure_robocasa_import(cfg.EVALUATION.robocasa_root, project_root=PROJECT_ROOT)
    validate_task_name(str(cfg.EVALUATION.task_name))
    if cfg.EVALUATION.get("seed") is not None:
        set_seed_everywhere(int(cfg.EVALUATION.seed))
    if bool(cfg.EVALUATION.get("deterministic", True)):
        os.environ["DETERMINISTIC"] = "True"
    logging.info("Output dir: %s", task_output_dir)
    logging.info("Log file: %s", log_path)
    logging.info("Checkpoint: %s", ckpt_path)
    logging.info("Dataset stats: %s", dataset_stats_path)
    logging.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    logging.info("Evaluation protocol=%s", "standard")
    policy = _build_policy(
        cfg,
        ckpt_path=ckpt_path,
        checkpoint_metadata=checkpoint_metadata,
        dataset_stats_path=dataset_stats_path,
    )
    from dreamfree.evaluation.robocasa.deploy_policy_optimized import prepare_policy_for_evaluation

    prepare_policy_for_evaluation(policy)
    logging.info(
        "Policy runtime=%s action_horizon=%d replan_steps=%d num_inference_steps=%d compile_dit=%s",
        robocasa_eval._policy_runtime_for_checkpoint_metadata(checkpoint_metadata),
        int(policy.action_horizon),
        int(policy.replan_steps),
        int(policy.num_inference_steps),
        bool(getattr(policy, "compile_dit", False)),
    )
    OmegaConf.save(config=cfg, f=str(task_output_dir / "eval_config.yaml"))
    results = run_task(policy=policy, cfg=cfg, output_dir=task_output_dir)
    results.update(
        {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_type": checkpoint_metadata.get("checkpoint_type")
            if checkpoint_metadata
            else None,
            "source_stage1_checkpoint": robocasa_eval._source_checkpoint_for_results(
                checkpoint_metadata
            ),
            "dataset_stats_path": str(dataset_stats_path),
            "policy_runtime": robocasa_eval._policy_runtime_for_checkpoint_metadata(
                checkpoint_metadata
            ),
            "compile_dit": bool(getattr(policy, "compile_dit", False)),
            "compile_dit_block_counts": dict(getattr(policy, "compile_dit_block_counts", {})),
            "inference_optimization": dict(getattr(policy, "optimization_contract", {})),
            "action_horizon": int(policy.action_horizon),
            "replan_steps": int(policy.replan_steps),
            "num_inference_steps": int(policy.num_inference_steps),
            "seed": None if cfg.EVALUATION.get("seed") is None else int(cfg.EVALUATION.seed),
            "deterministic": bool(cfg.EVALUATION.get("deterministic", True)),
            "layout_and_style_ids": cfg.EVALUATION.get("layout_and_style_ids"),
            "duration": float(time.time() - start_time),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_id": int(cfg.gpu_id),
        }
    )
    if getattr(policy, "action_history_horizon", None) is not None:
        results["action_history_horizon"] = int(policy.action_history_horizon)
    output_file = task_output_dir / "results.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, cls=NumpyEncoder)
    gallery_paths = ()
    if bool(cfg.EVALUATION.get("save_videos", True)):
        gallery_paths = write_robocasa_results_html(task_output_dir, results)
        logging.info(
            "Wrote RoboCasa result galleries: %s", ", ".join(str(path) for path in gallery_paths)
        )
    print(
        f"RoboCasa {results['task_name']} completed: {results['successes']}/{results['num_trials']} successes ({results['success_rate'] * 100:.1f}%)."
    )
    print(f"Results saved to: {output_file}")
    if gallery_paths:
        print("Video galleries saved to: " + ", ".join(str(path) for path in gallery_paths))
    return results
