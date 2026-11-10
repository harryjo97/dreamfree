"""Orchestrate full-episode latent preparation and atomic cache writes.

Configuration/contracts live in latent_config, episode loading/composition in
latent_data, and VAE construction/window encoding in latent_encoding."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import save_file as save_safetensors

from dreamfree.data import yam
from dreamfree.data.latent_config import (
    DROID_DREAMZERO_ACTION_CONTRACT,
    DROID_DREAMZERO_RELATIVE_STATS_FILE,
    DROID_GRID_CONCAT_MULTI_CAMERA,
    HORIZONTAL_CONCAT_MULTI_CAMERA,
    LATENT_CACHE_VERSION,
    ROBOCASA_ACTION_DIM,
    ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA,
    ROBOCASA_PROPRIO_DIM,
    ROBOTWIN_CONCAT_MULTI_CAMERA,
    PrecomputeConfig,
    _cache_root_name_map,
    _dataset_name,
    _is_droid_dataset,
    _is_libero_dataset,
    _is_robocasa_dataset,
    _mixed_precision_to_dtype,
    _normalize_mixed_precision,
    _optional_int,
    _optional_path,
    _parse_dataset_dirs,
    _precompute_cache_contract,
    _precompute_cache_tag,
    _resolve_action_horizon,
    _uses_lerobot_metadata_stats,
    validate_14b_i2v_first_latent_consistency,
    validate_latent_cache_file,
    validate_latent_tensor_contract,
)
from dreamfree.data.latent_config import (
    validate_latent_cache_header as validate_latent_cache_header,
)
from dreamfree.data.latent_data import (
    EpisodePayload,
    PrecomputeDatasetMetadata,
    _build_normalizer,
    _is_lerobot_v30_info,
    _load_droid_modality_config,
    _load_lerobot_episode_rows,
    _load_lerobot_info,
    _load_precompute_dataset_metadata,
    _prepare_episode_payload,
    _validate_lerobot_video_metadata,
)
from dreamfree.data.latent_encoding import (
    PrecomputeVAEModel,
    _compile_precompute_vae_encode,
    _encode_14b_i2v_condition_windows,
    _encode_windows,
    _load_precompute_vae,
)
from dreamfree.data.lerobot.utils.normalizer import LinearNormalizer
from dreamfree.models.wan22.i2v_condition import (
    WAN22_A14B_I2V_CONDITION_LATENTS_KEY,
    WAN22_A14B_I2V_CONDITION_SOURCE_OFFICIAL,
)
from dreamfree.utils.fs import ensure_dir
from dreamfree.utils.logging_config import get_logger, setup_logging

logger = get_logger("dreamfree.data.latent_preparation")


@dataclass(frozen=True)
class PrecomputeRuntime:
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool
    device: str


def _cache_output_dir(
    cfg: PrecomputeConfig,
    cache_root: Path,
    dataset_dir: Path,
    *,
    exact_cache_dir: bool = False,
    dataset_dirs: Sequence[Path | str] | None = None,
) -> Path:
    root_names = _cache_root_name_map([dataset_dir] if dataset_dirs is None else dataset_dirs)
    root_name = root_names[Path(dataset_dir).expanduser().resolve()]
    if exact_cache_dir:
        return cache_root / root_name
    cache_tag = _precompute_cache_tag(cfg)
    return cache_root / cache_tag / root_name


def validate_precompute_config(cfg: PrecomputeConfig) -> None:
    dataset_name = _dataset_name(cfg)
    if dataset_name not in {"robotwin", "libero", "robocasa", "droid"} | yam.DATASET_NAMES:
        raise ValueError("dataset_name must be robotwin, libero, robocasa, or droid.")
    dataset_dirs = _parse_dataset_dirs(cfg.dataset_dirs)
    for dataset_dir_text in dataset_dirs:
        dataset_dir = Path(dataset_dir_text).expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        if not (dataset_dir / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"Dataset directory is missing LeRobot metadata: {dataset_dir / 'meta' / 'info.json'}"
            )
        _validate_lerobot_video_metadata(dataset_dir, cfg)
    stats_text = _optional_path(cfg.dataset_stats_path)
    if stats_text is not None:
        stats_path = Path(stats_text).expanduser().resolve()
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Precompute required path `dataset_stats_path` not found: {stats_path}"
            )
    elif not _uses_lerobot_metadata_stats(cfg):
        raise ValueError(
            "Precompute requires dataset_stats_path unless the dataset uses LeRobot metadata stats."
        )
    concat_mode = str(cfg.camera_layout).strip().lower()
    if cfg.dataset_name in yam.DATASET_NAMES:
        yam.validate_config(cfg, stage=2)
        if len(dataset_dirs) != 1:
            raise ValueError("ABC requires one prepared dataset root.")
        yam.validate_stats(Path(dataset_dirs[0]), Path(cfg.dataset_stats_path), cfg.dataset_name)
    elif _is_libero_dataset(cfg):
        if concat_mode != HORIZONTAL_CONCAT_MULTI_CAMERA:
            raise ValueError("LIBERO latent precompute requires camera_layout='horizontal'.")
        if (int(cfg.height), int(cfg.width)) not in {(224, 448), (352, 640)}:
            raise ValueError(
                f"LIBERO latent precompute requires height/width to be 224x448 or 352x640; got {cfg.height}x{cfg.width}."
            )
        if (int(cfg.action_dim), int(cfg.proprio_dim)) != (7, 8):
            raise ValueError(
                f"LIBERO latent precompute requires action_dim=7,proprio_dim=8; got action_dim={cfg.action_dim}, proprio_dim={cfg.proprio_dim}."
            )
    elif _is_droid_dataset(cfg):
        if len(dataset_dirs) != 1:
            raise ValueError(
                f"DROID precompute currently supports exactly one dataset_dir, got {cfg.dataset_dirs}."
            )
        droid_root = Path(dataset_dirs[0]).expanduser().resolve()
        _load_droid_modality_config(droid_root)
        relative_stats_path = droid_root / "meta" / DROID_DREAMZERO_RELATIVE_STATS_FILE
        if not relative_stats_path.exists():
            raise FileNotFoundError(
                f"DROID DreamZero relative action stats not found: {relative_stats_path}"
            )
        if concat_mode != DROID_GRID_CONCAT_MULTI_CAMERA:
            raise ValueError("DROID latent precompute requires camera_layout='droid_grid'.")
        if (int(cfg.height), int(cfg.width), int(cfg.action_dim), int(cfg.proprio_dim)) != (
            352,
            640,
            8,
            8,
        ):
            raise ValueError(
                f"DROID latent precompute requires height=352,width=640,action_dim=8,proprio_dim=8; got {cfg.height}x{cfg.width}, action_dim={cfg.action_dim}, proprio_dim={cfg.proprio_dim}."
            )
    elif _is_robocasa_dataset(cfg):
        if len(dataset_dirs) != 1:
            raise ValueError(
                f"RoboCasa precompute currently supports exactly one dataset_dir, got {cfg.dataset_dirs}."
            )
        if concat_mode != ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA:
            raise ValueError(
                f"RoboCasa latent precompute requires camera_layout={ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA!r}."
            )
        if (int(cfg.height), int(cfg.width), int(cfg.action_dim), int(cfg.proprio_dim)) != (
            352,
            352,
            ROBOCASA_ACTION_DIM,
            ROBOCASA_PROPRIO_DIM,
        ):
            raise ValueError(
                f"RoboCasa latent precompute requires height=352,width=352,action_dim=7,proprio_dim=9; got {cfg.height}x{cfg.width}, action_dim={cfg.action_dim}, proprio_dim={cfg.proprio_dim}."
            )
        dataset_dir = Path(dataset_dirs[0]).expanduser().resolve()
        info = _load_lerobot_info(dataset_dir)
        features = info.get("features")
        if not isinstance(features, dict):
            raise ValueError(
                f"LeRobot metadata features must be a mapping: {dataset_dir / 'meta' / 'info.json'}"
            )
        action_shape = (
            features.get("action", {}).get("shape")
            if isinstance(features.get("action"), dict)
            else None
        )
        state_shape = (
            features.get("observation.state", {}).get("shape")
            if isinstance(features.get("observation.state"), dict)
            else None
        )
        if tuple(action_shape or ()) != (ROBOCASA_ACTION_DIM,) or tuple(state_shape or ()) != (
            ROBOCASA_PROPRIO_DIM,
        ):
            raise ValueError(
                f"RoboCasa metadata must expose action shape [7] and observation.state shape [9] in {dataset_dir / 'meta' / 'info.json'}, got action={action_shape}, state={state_shape}."
            )
    else:
        if len(dataset_dirs) != 1:
            raise ValueError(
                f"RobotWin precompute currently supports exactly one dataset_dir, got {cfg.dataset_dirs}."
            )
        if concat_mode != ROBOTWIN_CONCAT_MULTI_CAMERA:
            raise ValueError(
                f"Latent precompute requires camera_layout='{ROBOTWIN_CONCAT_MULTI_CAMERA}', got {cfg.camera_layout!r}."
            )
    if int(cfg.height) % 16 != 0 or int(cfg.width) % 16 != 0:
        raise ValueError(f"height/width must be multiples of 16, got {cfg.height}x{cfg.width}.")
    if int(cfg.num_frames) <= 1 or int(cfg.num_frames) % 4 != 1:
        raise ValueError(f"num_frames must satisfy >1 and %4==1, got {cfg.num_frames}.")
    if (int(cfg.num_frames) - 1) % int(cfg.action_video_freq_ratio) != 0:
        raise ValueError(
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {cfg.num_frames - 1} and {cfg.action_video_freq_ratio}."
        )
    action_horizon = _resolve_action_horizon(cfg)
    if _is_droid_dataset(cfg) and _optional_int(getattr(cfg, "action_horizon", None)) is None:
        cfg.action_horizon = action_horizon
    video_transitions = (int(cfg.num_frames) - 1) // int(cfg.action_video_freq_ratio)
    if action_horizon % video_transitions != 0:
        raise ValueError(
            f"action_horizon must be divisible by sparse video transitions, got action_horizon={action_horizon} and video_transitions={video_transitions}."
        )
    if int(cfg.action_dim) <= 0 or int(cfg.proprio_dim) <= 0:
        raise ValueError("action_dim and proprio_dim must be positive.")
    vae_checkpoint = _optional_path(cfg.vae_checkpoint)
    if vae_checkpoint is not None and (not Path(vae_checkpoint).expanduser().resolve().exists()):
        raise FileNotFoundError(
            f"VAE checkpoint not found: {Path(vae_checkpoint).expanduser().resolve()}"
        )
    expected_contract = _precompute_cache_contract(cfg)
    configured_latent_dim = getattr(cfg, "video_latent_dim", None)
    if configured_latent_dim is not None and int(configured_latent_dim) != int(
        expected_contract["latent_channels"]
    ):
        raise ValueError(
            f"Precompute video_latent_dim must be {expected_contract['latent_channels']} for {expected_contract['video_model_family']} ({expected_contract['vae_family']} VAE), got {configured_latent_dim}."
        )


def _save_episode_cache(
    *,
    output_path: Path,
    latents: torch.Tensor,
    i2v_condition_latents: torch.Tensor | None,
    payload: EpisodePayload,
    encoder: PrecomputeVAEModel,
    num_frames: int,
    action_video_freq_ratio: int,
) -> dict[str, Any]:
    validate_latent_tensor_contract(
        latents, expected=encoder.cache_contract, label="Latent precompute"
    )
    (n_samples, c_lat, t_lat, h_lat, w_lat) = latents.shape
    full_window_count = max(0, payload.num_frames - num_frames + 1)
    contract = encoder.cache_contract
    is_14b = str(contract["video_model_family"]) == "wan22_i2v_a14b"
    if is_14b:
        if not isinstance(i2v_condition_latents, torch.Tensor):
            raise ValueError(
                f"Wan2.2-I2V-A14B latent caches require official {WAN22_A14B_I2V_CONDITION_LATENTS_KEY}."
            )
        if tuple(i2v_condition_latents.shape) != tuple(latents.shape):
            raise ValueError(
                f"{WAN22_A14B_I2V_CONDITION_LATENTS_KEY} shape must match latents: {tuple(i2v_condition_latents.shape)} vs {tuple(latents.shape)}."
            )
        if bool(contract.get("require_i2v_first_latent_equality", False)):
            validate_14b_i2v_first_latent_consistency(
                latents, i2v_condition_latents, label="Latent precompute save"
            )
    elif i2v_condition_latents is not None:
        raise ValueError(
            f"{WAN22_A14B_I2V_CONDITION_LATENTS_KEY} is only valid for video_model_family='wan22_i2v_a14b'."
        )
    metadata = {
        "cache_version": str(LATENT_CACHE_VERSION),
        "coverage": "full_episode",
        "padding_policy": "clamp_to_episode_end",
        "video_model_family": str(contract["video_model_family"]),
        "model_id": str(contract["model_id"]),
        "vae_family": str(contract["vae_family"]),
        "vae_path": str(encoder.vae_path),
        "vae_z_dim": str(int(contract["vae_z_dim"])),
        "vae_temporal_stride": str(int(contract["vae_temporal_stride"])),
        "vae_spatial_stride": str(int(contract["vae_spatial_stride"])),
        "episode_index": str(payload.episode_index),
        "num_frames": str(payload.num_frames),
        "num_samples": str(int(n_samples)),
        "num_windows": str(int(full_window_count)),
        "num_frames_per_window": str(num_frames),
        "action_video_freq_ratio": str(action_video_freq_ratio),
        "latent_channels": str(int(c_lat)),
        "latent_t_per_window": str(int(t_lat)),
        "latent_h": str(int(h_lat)),
        "latent_w": str(int(w_lat)),
        "video_h": str(int(contract["height"])),
        "video_w": str(int(contract["width"])),
        "concat_layout": payload.concat_layout,
        "action_dim": str(int(payload.action_norm.shape[1])),
        "state_dim": str(int(payload.state_norm.shape[1])),
        "proprio_dim": str(int(payload.state_norm.shape[1])),
        "norm_mode": payload.norm_mode,
        "normalizer_stats_path": str(encoder.stats_src),
    }
    if is_14b:
        metadata["i2v_condition_source"] = WAN22_A14B_I2V_CONDITION_SOURCE_OFFICIAL
    if contract.get("action_horizon") is not None:
        metadata["action_horizon"] = str(int(contract["action_horizon"]))
    if payload.dataset_kind is not None:
        metadata["dataset_kind"] = str(payload.dataset_kind)
    if payload.action_layout is not None:
        metadata["action_layout"] = payload.action_layout
    if payload.source_metadata:
        metadata.update({str(key): str(value) for (key, value) in payload.source_metadata.items()})
    if payload.concat_layout == DROID_GRID_CONCAT_MULTI_CAMERA:
        metadata["dataset_kind"] = "droid"
        metadata["action_contract"] = DROID_DREAMZERO_ACTION_CONTRACT
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tensors = {
        "latents": latents,
        "action": payload.action_norm.to(torch.float32).contiguous(),
        "proprio": payload.state_norm.to(torch.float32).contiguous(),
        "task_indices": payload.task_indices.to(torch.int32).contiguous(),
    }
    if i2v_condition_latents is not None:
        tensors[WAN22_A14B_I2V_CONDITION_LATENTS_KEY] = i2v_condition_latents.contiguous()
    if payload.extra_tensors:
        tensors.update(
            {
                key: value.to(torch.float32).contiguous()
                for (key, value) in payload.extra_tensors.items()
            }
        )
    save_safetensors(tensors, str(tmp_path), metadata=metadata)
    tmp_path.replace(output_path)
    summary = {
        "episode_index": int(payload.episode_index),
        "num_frames": int(payload.num_frames),
        "num_samples": int(n_samples),
        "num_windows": int(full_window_count),
        "bytes": int(
            latents.numel() * latents.element_size()
            + (
                0
                if i2v_condition_latents is None
                else i2v_condition_latents.numel() * i2v_condition_latents.element_size()
            )
        ),
    }
    if payload.source_metadata:
        summary.update(payload.source_metadata)
    return summary


def _encode_and_save_episode(
    *,
    encoder: PrecomputeVAEModel,
    payload: EpisodePayload,
    output_path: Path,
    num_frames: int,
    action_video_freq_ratio: int,
    batch_size: int,
) -> dict[str, Any]:
    is_14b = str(encoder.cache_contract["video_model_family"]) == "wan22_i2v_a14b"
    latents = _encode_windows(
        encoder,
        payload.video_processed,
        num_frames=num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
        batch_size=batch_size,
    )
    i2v_condition_latents = (
        _encode_14b_i2v_condition_windows(
            encoder,
            payload.video_processed,
            num_frames=num_frames,
            action_video_freq_ratio=action_video_freq_ratio,
            batch_size=batch_size,
        )
        if is_14b
        else None
    )
    return _save_episode_cache(
        output_path=output_path,
        latents=latents,
        i2v_condition_latents=i2v_condition_latents,
        payload=payload,
        encoder=encoder,
        num_frames=num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
    )


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)


def _resolve_runtime(extra: argparse.Namespace) -> PrecomputeRuntime:
    env_world_size = _env_int("WORLD_SIZE")
    env_rank = _env_int("RANK")
    env_local_rank = _env_int("LOCAL_RANK")
    has_dist_env = any((value is not None for value in (env_world_size, env_rank, env_local_rank)))
    if has_dist_env:
        world_size = int(env_world_size or 1)
        rank = int(env_rank or 0)
        local_rank = int(env_local_rank or 0)
        if (
            bool(getattr(extra, "world_size_explicit", False))
            and int(extra.world_size) != world_size
        ):
            raise ValueError(
                f"Conflicting world_size: env WORLD_SIZE={world_size} but CLI requested {extra.world_size}."
            )
        if bool(getattr(extra, "rank_explicit", False)) and int(extra.rank) != rank:
            raise ValueError(f"Conflicting rank: env RANK={rank} but CLI requested {extra.rank}.")
        if getattr(extra, "local_rank", None) is not None and bool(
            getattr(extra, "local_rank_explicit", False)
        ):
            if int(extra.local_rank) != local_rank:
                raise ValueError(
                    f"Conflicting local_rank: env LOCAL_RANK={local_rank} but CLI requested {extra.local_rank}."
                )
        is_distributed = world_size > 1
    else:
        world_size = int(extra.world_size)
        rank = int(extra.rank)
        local_rank = 0 if getattr(extra, "local_rank", None) is None else int(extra.local_rank)
        is_distributed = False
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank} for world_size {world_size}.")
    if local_rank < 0:
        raise ValueError(f"local_rank must be >= 0, got {local_rank}.")
    device = "cpu"
    if torch.cuda.is_available():
        device_count = int(torch.cuda.device_count())
        if local_rank >= device_count:
            raise ValueError(
                f"local_rank {local_rank} is out of range for visible CUDA device_count={device_count}."
            )
        device = f"cuda:{local_rank}"
    return PrecomputeRuntime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=is_distributed,
        device=device,
    )


def _init_distributed_runtime(runtime: PrecomputeRuntime) -> None:
    if runtime.device.startswith("cuda"):
        torch.cuda.set_device(runtime.local_rank)
    if not runtime.is_distributed:
        return
    backend = "nccl" if runtime.device.startswith("cuda") else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")


def _build_episode_shard(all_indices: Sequence[int], *, world_size: int, rank: int) -> list[int]:
    return [int(ep) for (i, ep) in enumerate(all_indices) if i % int(world_size) == int(rank)]


def _aggregate_counts(
    runtime: PrecomputeRuntime, *, processed: int, skipped: int, total: int
) -> dict[str, int]:
    counts = {"processed": int(processed), "skipped": int(skipped), "total": int(total)}
    if not runtime.is_distributed or not dist.is_initialized():
        return counts
    reduce_device = (
        torch.device(runtime.device) if runtime.device.startswith("cuda") else torch.device("cpu")
    )
    tensor = torch.tensor(
        [counts["processed"], counts["skipped"], counts["total"]],
        device=reduce_device,
        dtype=torch.long,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {
        "processed": int(tensor[0].item()),
        "skipped": int(tensor[1].item()),
        "total": int(tensor[2].item()),
    }


def _shutdown_distributed_runtime(runtime: PrecomputeRuntime) -> None:
    if not runtime.is_distributed or not dist.is_initialized():
        return
    dist.barrier()
    dist.destroy_process_group()


def _run_episode_shard(
    *,
    cfg: PrecomputeConfig,
    shard: Sequence[int],
    encoder: PrecomputeVAEModel,
    normalizer: LinearNormalizer | None,
    dataset_root: Path,
    meta: PrecomputeDatasetMetadata,
    episode_rows: dict[int, dict[str, Any]] | None,
    out_dir: Path,
    manifest_path: Path,
    num_frames: int,
    action_video_freq_ratio: int,
    batch_size: int,
    skip_existing: bool,
    rank: int,
) -> tuple[int, int]:
    processed = 0
    skipped = 0
    start_time = time.perf_counter()
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for ep_idx in shard:
            out_path = out_dir / f"episode_{int(ep_idx):06d}.safetensors"
            if skip_existing and out_path.exists():
                validate_latent_cache_file(
                    out_path,
                    expected=encoder.cache_contract,
                    label="Latent precompute skip_existing",
                )
                skipped += 1
                continue
            payload = _prepare_episode_payload(
                cfg=cfg,
                normalizer=normalizer,
                dataset_root=dataset_root,
                meta=meta,
                actual_ep_idx=ep_idx,
                episode_rows=episode_rows,
            )
            summary = _encode_and_save_episode(
                encoder=encoder,
                payload=payload,
                output_path=out_path,
                num_frames=num_frames,
                action_video_freq_ratio=action_video_freq_ratio,
                batch_size=batch_size,
            )
            manifest.write(json.dumps(summary) + "\n")
            manifest.flush()
            processed += 1
            if processed % 10 == 0 or processed + skipped == len(shard):
                logger.info(
                    "[rank %d] processed=%d skipped=%d remaining=%d elapsed=%.1fs",
                    rank,
                    processed,
                    skipped,
                    len(shard) - processed - skipped,
                    time.perf_counter() - start_time,
                )
    return (processed, skipped)


def run_precompute(cfg: PrecomputeConfig, extra: argparse.Namespace) -> None:
    runtime = _resolve_runtime(extra)
    _init_distributed_runtime(runtime)
    setup_logging(log_level=logging.INFO, is_main_process=runtime.rank == 0)
    validate_precompute_config(cfg)
    if extra.cache_root is None and extra.cache_dir is None:
        raise ValueError("Either --cache-root or --cache-dir is required.")
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_dtype(mixed_precision)
    cache_contract = _precompute_cache_contract(cfg)
    try:
        logger.info(
            "Loading %s VAE for %s on %s (dtype=%s)",
            cache_contract["vae_family"],
            cache_contract["video_model_family"],
            runtime.device,
            model_dtype,
        )
        encoder = _load_precompute_vae(cfg, device=runtime.device, dtype=model_dtype)
        if bool(getattr(extra, "compile_vae", False)):
            _compile_precompute_vae_encode(
                encoder, mode=str(getattr(extra, "compile_vae_mode", "default"))
            )
        normalizer = _build_normalizer(cfg)
        cache_base = Path(extra.cache_dir or extra.cache_root).expanduser().resolve()
        exact_cache_dir = extra.cache_dir is not None
        ensure_dir(cache_base)
        dataset_dirs = [
            Path(path).expanduser().resolve() for path in _parse_dataset_dirs(cfg.dataset_dirs)
        ]
        for dataset_dir in dataset_dirs:
            meta = _load_precompute_dataset_metadata(dataset_dir)
            episode_rows = (
                _load_lerobot_episode_rows(dataset_dir) if _is_lerobot_v30_info(meta.info) else None
            )
            total_episodes = (
                len(episode_rows) if episode_rows is not None else int(meta.total_episodes)
            )
            all_indices = (
                list(episode_rows) if episode_rows is not None else list(range(total_episodes))
            )
            if int(extra.limit_episodes) > 0:
                all_indices = (
                    list(episode_rows) if episode_rows is not None else list(range(total_episodes))
                )
            shard = _build_episode_shard(
                all_indices, world_size=runtime.world_size, rank=runtime.rank
            )
            logger.info(
                "Rank %d/%d (local_rank=%d distributed=%s) will process %d episodes (of %d candidates in dataset=%s)",
                runtime.rank,
                runtime.world_size,
                runtime.local_rank,
                runtime.is_distributed,
                len(shard),
                len(all_indices),
                dataset_dir.name,
            )
            out_dir = _cache_output_dir(
                cfg,
                cache_base,
                dataset_dir,
                exact_cache_dir=exact_cache_dir,
                dataset_dirs=dataset_dirs,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Writing per-episode latents under %s", out_dir)
            manifest_path = out_dir / f"manifest_rank{int(runtime.rank):02d}.jsonl"
            start_time = time.perf_counter()
            (processed, skipped) = _run_episode_shard(
                cfg=cfg,
                shard=shard,
                encoder=encoder,
                normalizer=normalizer,
                dataset_root=dataset_dir,
                meta=meta,
                episode_rows=episode_rows,
                out_dir=out_dir,
                manifest_path=manifest_path,
                num_frames=int(cfg.num_frames),
                action_video_freq_ratio=int(cfg.action_video_freq_ratio),
                batch_size=int(extra.encode_batch_size),
                skip_existing=bool(extra.skip_existing),
                rank=runtime.rank,
            )
            elapsed = time.perf_counter() - start_time
            logger.info(
                "[rank %d] done dataset=%s: processed=%d skipped=%d total=%d in %.1fs",
                runtime.rank,
                dataset_dir.name,
                processed,
                skipped,
                len(shard),
                elapsed,
            )
            global_counts = _aggregate_counts(
                runtime, processed=processed, skipped=skipped, total=len(shard)
            )
            if runtime.rank == 0 and runtime.is_distributed:
                logger.info(
                    "[global] done dataset=%s: processed=%d skipped=%d total=%d",
                    dataset_dir.name,
                    global_counts["processed"],
                    global_counts["skipped"],
                    global_counts["total"],
                )
    finally:
        _shutdown_distributed_runtime(runtime)
