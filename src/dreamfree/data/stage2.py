"""Stage 2 live/cached datasets, episode splits, and dataloaders.

Dataset geometry and normalization live in action_processing; cache checks in
action_cache; model-facing batch preparation in action_batch."""

from __future__ import annotations

import hashlib
import importlib.util
import random
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from dreamfree.data import yam
from dreamfree.data.action_batch import build_stage2_cached_inputs as build_stage2_cached_inputs
from dreamfree.data.action_cache import (
    _stage2_latent_cache_expected_contract,
    _stage2_text_cache_suffix,
)
from dreamfree.data.action_cache import (
    validate_droid_stage2_latent_cache_metadata as validate_droid_stage2_latent_cache_metadata,
)
from dreamfree.data.action_processing import (
    DROID_GRIPPER_POSITION_KEY,
    DROID_JOINT_POSITION_KEY,
    ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA,
    ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA,
    _build_stage2_processor,
    _build_stage2_shape_meta,
    _droid_modality_range,
    _is_stage2_droid_dataset_name,
    _load_stage2_dataset_stats,
    _parse_dataset_dirs,
    resolve_stage2_action_horizon,
)
from dreamfree.data.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from dreamfree.data.lerobot.cached_latent_dataset import (
    binarize_droid_gripper_action_history,
)
from dreamfree.data.lerobot.lerobot.lerobot_dataset import (
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from dreamfree.data.robocasa_visual import ROBOCASA_RAW_CAMERA_SIZE, compose_robocasa_dreamzero
from dreamfree.data.sampling import seed_worker
from dreamfree.data.text_cache import DEFAULT_PROMPT, validate_text_cache
from dreamfree.data.video import resolve_episode_task_candidates
from dreamfree.training.stage0.constants import (
    DROID_GRID_CONCAT_MULTI_CAMERA,
    DROID_GRID_HEIGHT,
    DROID_GRID_WIDTH,
)
from dreamfree.training.stage2.config import Stage2Config
from dreamfree.utils.logging_config import get_logger

logger = get_logger("dreamfree.data.stage2")
DEFAULT_SPLIT_SEED = 42
MAX_GETITEM_ATTEMPT = 5


def _stage2_metadata_episode_indices(meta: LeRobotDatasetMetadata) -> list[int]:
    if hasattr(meta, "episodes"):
        episode_indices = sorted((int(ep_idx) for ep_idx in meta.episodes))
    else:
        episode_indices = list(range(int(meta.total_episodes)))
    if not episode_indices:
        raise ValueError(
            f"Stage 2 dataset metadata contains no episodes for repo={meta.repo_id} root={getattr(meta, 'root', '<unknown>')}."
        )
    expected_count = int(meta.total_episodes)
    if len(episode_indices) != expected_count:
        raise ValueError(
            f"Stage 2 dataset metadata episode count mismatch for repo={meta.repo_id} root={meta.root}: meta/info.json total_episodes={expected_count}, meta/episodes.jsonl entries={len(episode_indices)}."
        )
    return episode_indices


def _select_stage2_episode_indices_for_split(
    meta: LeRobotDatasetMetadata,
    *,
    split: str,
    val_fraction: float,
    test_fraction: float,
    seed: int = DEFAULT_SPLIT_SEED,
    allowed_episode_indices: Sequence[int] | None = None,
) -> list[int]:
    split_name = str(split).strip().lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Stage 2 dataset split must be one of train/val/test, got {split!r}.")
    val_prop = float(val_fraction)
    test_prop = float(test_fraction)
    if val_prop < 0.0 or test_prop < 0.0:
        raise ValueError(
            f"val_fraction and test_fraction must be non-negative, got val={val_fraction} test={test_fraction}."
        )
    if val_prop + test_prop >= 1.0:
        raise ValueError(
            f"val_fraction + test_fraction must be < 1.0, got {val_prop} + {test_prop}."
        )
    known_episode_indices = _stage2_metadata_episode_indices(meta)
    if allowed_episode_indices is None:
        episode_indices = known_episode_indices
    else:
        episode_indices = [int(episode_index) for episode_index in allowed_episode_indices]
        if not episode_indices:
            raise ValueError(f"Stage 2 allowed episode set is empty for repo={meta.repo_id}.")
        if len(episode_indices) != len(set(episode_indices)):
            raise ValueError(
                f"Stage 2 allowed episode set contains duplicates for repo={meta.repo_id}."
            )
        unknown = sorted(set(episode_indices) - set(known_episode_indices))
        if unknown:
            raise ValueError(
                f"Stage 2 allowed episode set contains indices absent from repo={meta.repo_id}: {unknown[:10]}."
            )
    if val_prop < 1e-06 and test_prop < 1e-06:
        return episode_indices
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_indices)
    total_episodes = len(episode_indices)
    val_count = 0 if val_prop < 1e-06 else total_episodes - int(total_episodes * (1.0 - val_prop))
    test_count = (
        0 if test_prop < 1e-06 else total_episodes - int(total_episodes * (1.0 - test_prop))
    )
    train_count = total_episodes - val_count - test_count
    if train_count < 0:
        raise ValueError(
            f"Episode split counts are invalid for repo={meta.repo_id}: total={total_episodes} train={train_count} val={val_count} test={test_count}."
        )
    if split_name == "train":
        return episode_indices[:train_count]
    if split_name == "val":
        return episode_indices[train_count : train_count + val_count]
    return episode_indices[train_count + val_count :]


def _parse_path_list(value: Any, *, label: str) -> list[Path]:
    if value is None:
        raise ValueError(f"{label} must contain at least one path.")
    if isinstance(value, (str, Path)):
        raw_parts = str(value).split(",")
    elif isinstance(value, Sequence):
        raw_parts = [str(part) for part in value]
    else:
        raise TypeError(f"{label} must be a comma-separated string or sequence, got {type(value)}.")
    paths = [Path(part.strip()).expanduser() for part in raw_parts if part.strip()]
    if not paths:
        raise ValueError(f"{label} must contain at least one path.")
    return paths


def _require_sparse_stage2_video_backend() -> str:
    if importlib.util.find_spec("torchcodec") is None:
        raise RuntimeError(
            "Sparse Stage 2 requires video_backend='torchcodec', but the `torchcodec` package is not available. Install `torchcodec`; falling back to `pyav` is disabled for this trainer."
        )
    return "torchcodec"


class _SparseStage2RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        dataset_dirs: list[str],
        dataset_name: str,
        shape_meta: dict[str, Any],
        num_frames: int,
        action_horizon: int,
        action_history_horizon: int,
        video_size: list[int],
        processor: Any,
        text_embedding_cache_dir: str,
        text_cache_model_suffix: str,
        context_len: int,
        dataset_stats_path: str,
        val_fraction: float,
        test_fraction: float,
        split: str,
        is_training_set: bool,
        global_sample_stride: int,
        action_video_freq_ratio: int,
        skip_padding_as_possible: bool,
        max_padding_retry: int,
        camera_layout: str,
        override_instruction: Optional[str],
        strict_loading: bool,
        strict_errors: bool,
        video_roots: Optional[dict[str, str]],
        split_seed: int = DEFAULT_SPLIT_SEED,
        allow_mixed_fps_for_cached_latents: bool = False,
        metadata_only_for_cached_latents: bool = False,
    ) -> None:
        if processor is None:
            raise ValueError("Sparse Stage 2 dataset requires a processor instance.")
        if OmegaConf.is_config(shape_meta):
            resolved_shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        elif isinstance(shape_meta, dict):
            resolved_shape_meta = shape_meta
        else:
            raise TypeError(f"`shape_meta` must be dict-like or OmegaConf, got {type(shape_meta)}")
        self.dataset_name = str(dataset_name).strip().lower()
        self.split = str(split).strip().lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Stage 2 dataset split must be one of train/val/test, got {split!r}.")
        if bool(is_training_set) != (self.split == "train"):
            raise ValueError(
                f"Stage 2 dataset split/is_training_set mismatch: split={self.split!r} is_training_set={is_training_set}."
            )
        self.shape_meta = resolved_shape_meta
        self.image_meta = [dict(meta) for meta in resolved_shape_meta["images"]]
        self.state_meta = [dict(meta) for meta in resolved_shape_meta["state"]]
        self.action_meta = [dict(meta) for meta in resolved_shape_meta["action"]]
        self.num_frames = int(num_frames)
        self.action_horizon = int(action_horizon)
        self.action_history_horizon = int(action_history_horizon)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                f"num_frames-1 must be divisible by action_video_freq_ratio, got {self.num_frames - 1} and {self.action_video_freq_ratio}."
            )
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}.")
        if self.action_history_horizon < 0:
            raise ValueError(
                f"action_history_horizon must be non-negative, got {self.action_history_horizon}."
            )
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))
        self.video_num_frames = len(self.video_sample_indices)
        if self.video_num_frames % 4 != 1:
            raise ValueError(
                f"Sampled sparse video frames must satisfy Wan temporal constraint `T % 4 == 1`, got {self.video_num_frames}."
            )
        video_transitions = self.video_num_frames - 1
        if video_transitions <= 0:
            raise ValueError("Sampled sparse video must contain at least two frames.")
        if self.action_horizon % video_transitions != 0:
            raise ValueError(
                f"action_horizon must be divisible by sparse video transitions; got action_horizon={self.action_horizon} and video_transitions={video_transitions}."
            )
        self.video_size = [int(video_size[0]), int(video_size[1])]
        self.context_len = int(context_len)
        self.text_embedding_cache_dir = str(text_embedding_cache_dir)
        self.text_embedding_cache_dirs = _parse_path_list(
            text_embedding_cache_dir, label="Stage 2 text_embedding_cache_dir"
        )
        self.text_cache_model_suffix = str(text_cache_model_suffix)
        self.is_training_set = bool(is_training_set)
        self.skip_padding_as_possible = bool(skip_padding_as_possible)
        self.max_padding_retry = int(max_padding_retry)
        self.camera_layout = str(camera_layout)
        self.override_instruction = override_instruction
        self.strict_loading = bool(strict_loading)
        self.strict_errors = bool(strict_errors)
        self.global_sample_stride = int(global_sample_stride)
        self.video_backend = _require_sparse_stage2_video_backend()
        self.metadata_only_for_cached_latents = bool(metadata_only_for_cached_latents)
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            metas.append(LeRobotDatasetMetadata(repo_id=ds_dir, root=ds_root))
        if _is_stage2_droid_dataset_name(self.dataset_name):
            if len(metas) != 1:
                raise ValueError(
                    f"DROID Stage 2 dataset requires exactly one dataset root, got {len(metas)}."
                )
            self._apply_droid_modality_ranges(Path(metas[0].root))
        self.processor = processor.train() if self.is_training_set else processor.eval()
        dataset_stats = _load_stage2_dataset_stats(
            dataset_stats_path=dataset_stats_path, dataset_name=self.dataset_name, metas=metas
        )
        self.processor.set_normalizer_from_stats(dataset_stats)
        self.image_transforms = (
            self.processor.train_transforms
            if self.is_training_set
            else self.processor.val_transforms
        )
        if self.image_transforms is None:
            raise ValueError("Sparse Stage 2 dataset requires image transforms on the processor.")
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})
        fps_list = [meta.fps for meta in metas]
        if len(set(fps_list)) != 1:
            if not bool(allow_mixed_fps_for_cached_latents):
                raise ValueError(f"All dataset_dirs must have the same fps, got {fps_list}")
            logger.info(
                "Sparse Stage 2 metadata dataset allows mixed fps=%s because full cached latents are required.",
                fps_list,
            )
        self.fps = fps_list[0]
        self.delta_timestamps = self._build_delta_timestamps()
        episodes = self._build_episode_split(
            metas=metas,
            val_fraction=float(val_fraction),
            test_fraction=float(test_fraction),
            seed=int(split_seed),
        )
        multi_dataset_kwargs = {
            "dataset_dirs": list(dataset_dirs),
            "episodes": episodes,
            "delta_timestamps": None
            if self.metadata_only_for_cached_latents
            else self.delta_timestamps,
            "video_backend": self.video_backend,
        }
        if video_roots is not None:
            multi_dataset_kwargs["video_roots"] = video_roots
        self.multi_dataset = MultiLeRobotDataset(**multi_dataset_kwargs)
        self.multi_dataset.set_during_training(True)
        self._episode_task_candidates = self._build_episode_task_candidates()

    @staticmethod
    def _scalar_int(value: Any, *, default: int | None = None) -> int | None:
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            value = value.item() if value.numel() == 1 else value.flatten()[0].item()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.item() if value.size == 1 else value.reshape(-1)[0].item()
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _build_episode_task_candidates(self) -> dict[tuple[int, int], tuple[str, ...]]:
        candidates_by_episode: dict[tuple[int, int], tuple[str, ...]] = {}
        for dataset_idx, raw_dataset in enumerate(self.multi_dataset._datasets):
            task_by_index = dict(raw_dataset.meta.tasks)
            episode_rows = getattr(raw_dataset.meta, "episodes", {}) or {}
            selected_episodes = raw_dataset.episodes
            if selected_episodes is None:
                selected_episodes = sorted((int(ep_idx) for ep_idx in episode_rows))
            for ep_idx in selected_episodes:
                ep_idx_int = int(ep_idx)
                row = episode_rows.get(ep_idx_int)
                if row is None:
                    row = episode_rows.get(str(ep_idx_int), {})
                candidates = resolve_episode_task_candidates(
                    row=dict(row or {}),
                    task_by_index=task_by_index,
                    default_task=None,
                    strict=False,
                    stage_label="Stage 2",
                    task_root=getattr(raw_dataset, "root", None),
                )
                if candidates:
                    candidates_by_episode[dataset_idx, ep_idx_int] = candidates
        return candidates_by_episode

    def _sample_instruction(self, lerobot_sample: dict[str, Any]) -> str:
        if self.override_instruction is not None:
            return str(self.override_instruction)
        dataset_idx = self._scalar_int(lerobot_sample.get("dataset_index"), default=0)
        episode_idx = self._scalar_int(lerobot_sample.get("episode_index"), default=None)
        if dataset_idx is not None and episode_idx is not None:
            candidates = self._episode_task_candidates.get((dataset_idx, episode_idx))
            if candidates:
                sample = dict(lerobot_sample)
                sample["task"] = random.choice(candidates)
                return self.processor.augment_instruction(sample)
        return self.processor.augment_instruction(lerobot_sample)

    def _apply_droid_modality_ranges(self, root: Path) -> None:
        state_joint = _droid_modality_range(root, "state", DROID_JOINT_POSITION_KEY)
        state_gripper = _droid_modality_range(root, "state", DROID_GRIPPER_POSITION_KEY)
        action_joint = _droid_modality_range(root, "action", DROID_JOINT_POSITION_KEY)
        action_gripper = _droid_modality_range(root, "action", DROID_GRIPPER_POSITION_KEY)
        state_ranges = {
            DROID_JOINT_POSITION_KEY: state_joint,
            DROID_GRIPPER_POSITION_KEY: state_gripper,
        }
        action_ranges = {
            DROID_JOINT_POSITION_KEY: action_joint,
            DROID_GRIPPER_POSITION_KEY: action_gripper,
        }
        for meta in self.state_meta:
            meta["source_slice"] = state_ranges[meta["key"]]
            meta["source_key"] = "observation.state"
        for meta in self.action_meta:
            meta["source_slice"] = action_ranges[meta["key"]]
            meta["source_key"] = "action"

    def _build_delta_timestamps(self) -> dict[str, list[float]]:
        delta_timestamps: dict[str, list[float]] = {}
        is_droid = _is_stage2_droid_dataset_name(self.dataset_name)
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = (
                f"observation.images.{key}" if key != "default" else "observation.images"
            )
            delta_timestamps[meta["lerobot_key"]] = [
                t * self.action_video_freq_ratio * self.global_sample_stride / self.fps
                for t in range(self.video_num_frames)
            ]
        for meta in self.state_meta:
            key = meta["key"]
            if is_droid:
                meta["lerobot_key"] = str(meta.get("source_key", "observation.state"))
            else:
                meta["lerobot_key"] = (
                    f"observation.state.{key}" if key != "default" else "observation.state"
                )
            delta_timestamps[meta["lerobot_key"]] = [
                t * self.global_sample_stride / self.fps for t in range(self.action_horizon)
            ]
        for meta in self.action_meta:
            key = meta["key"]
            if is_droid:
                meta["lerobot_key"] = str(meta.get("source_key", "action"))
            else:
                meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"
            delta_timestamps[meta["lerobot_key"]] = [
                t * self.global_sample_stride / self.fps
                for t in range(-self.action_history_horizon, self.action_horizon)
            ]
        return delta_timestamps

    def _build_episode_split(
        self,
        *,
        metas: list[LeRobotDatasetMetadata],
        val_fraction: float,
        test_fraction: float,
        seed: int,
    ) -> dict[str, list[int]]:
        episodes: dict[str, list[int]] = {}
        for meta in metas:
            allowed_episode_indices = None
            selected = _select_stage2_episode_indices_for_split(
                meta,
                split=self.split,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed,
                allowed_episode_indices=allowed_episode_indices,
            )
            episodes[meta.repo_id] = selected
        return episodes

    def __len__(self) -> int:
        return self.multi_dataset.num_frames

    def _get_action(
        self,
        meta: dict[str, Any],
        lerobot_sample: dict[str, Any],
        state_by_key: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action = lerobot_sample[meta["lerobot_key"]]
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        source_slice = meta.get("source_slice")
        if source_slice is not None:
            (start, end) = (int(source_slice[0]), int(source_slice[1]))
            if int(action.shape[-1]) < end:
                raise ValueError(
                    f"Action source '{meta['lerobot_key']}' shape {action.shape[-1]} is too small for slice [{start}:{end}]."
                )
            action = action[..., start:end]
        relative_to = meta.get("relative_to_state_key")
        if relative_to is not None:
            state = state_by_key.get(str(relative_to))
            if state is None:
                raise KeyError(
                    f"Action '{meta['key']}' requires missing state key {relative_to!r} for relative DROID action."
                )
            if state.ndim != 2 or int(state.shape[-1]) != int(action.shape[-1]):
                raise ValueError(
                    f"Relative DROID action '{meta['key']}' state shape {tuple(state.shape)} does not match action shape {tuple(action.shape)}."
                )
            action = action - state[0].unsqueeze(0)
        if int(action.shape[-1]) != int(meta["raw_shape"]):
            raise ValueError(
                f"Action '{meta['key']}' shape {action.shape[-1]} mismatch with meta {meta['raw_shape']}."
            )
        return action.float()

    def _get_state(self, meta: dict[str, Any], lerobot_sample: dict[str, Any]) -> torch.Tensor:
        state = lerobot_sample[meta["lerobot_key"]]
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        source_slice = meta.get("source_slice")
        if source_slice is not None:
            (start, end) = (int(source_slice[0]), int(source_slice[1]))
            if int(state.shape[-1]) < end:
                raise ValueError(
                    f"State source '{meta['lerobot_key']}' shape {state.shape[-1]} is too small for slice [{start}:{end}]."
                )
            state = state[..., start:end]
        if int(state.shape[-1]) != int(meta["raw_shape"]):
            raise ValueError(
                f"State '{meta['key']}' shape {state.shape[-1]} mismatch with meta {meta['raw_shape']}."
            )
        return state.float()

    def _get_image(self, meta: dict[str, Any], lerobot_sample: dict[str, Any]) -> torch.Tensor:
        image = lerobot_sample[meta["lerobot_key"]]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(
                f"Image '{meta['key']}' must have 4 dimensions [T,C,H,W], got shape {tuple(image.shape)}."
            )
        return (image * 255).to(torch.uint8)

    def _load_lerobot_sample(self, idx: int) -> tuple[int, dict[str, Any]]:
        sample_idx = idx
        if self.strict_loading:
            return (sample_idx, self.multi_dataset[sample_idx])
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                return (sample_idx, self.multi_dataset[sample_idx])
            except Exception as err:
                attempt += 1
                last_exception = err
                logger.warning(
                    "Error loading sparse Stage 2 sample %s (attempt %s). Retrying with a random index. Error: %s",
                    sample_idx,
                    attempt,
                    err,
                )
                sample_idx = np.random.randint(len(self))
                print(traceback.format_exc())
        raise RuntimeError(
            f"Failed to load a valid sparse Stage 2 sample after {MAX_GETITEM_ATTEMPT} attempts for index {idx}."
        ) from last_exception

    def _build_nested_sample(
        self, sample_idx: int, lerobot_sample: dict[str, Any]
    ) -> dict[str, Any]:
        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)
        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample, sample["state"])
        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)
        sample["action_is_pad"] = torch.as_tensor(
            lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"], dtype=torch.bool
        )
        sample["state_is_pad"] = torch.as_tensor(
            lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"], dtype=torch.bool
        )
        sample["image_is_pad"] = torch.as_tensor(
            lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"], dtype=torch.bool
        )
        return sample

    def _apply_image_transforms(self, nested_sample: dict[str, Any]) -> torch.Tensor:
        processed_images = []
        for meta in self.image_meta:
            key = meta["key"]
            image = nested_sample["images"][key]
            transforms = (
                self.image_transforms[key]
                if isinstance(self.image_transforms, dict)
                else self.image_transforms
            )
            for transform in transforms:
                image = transform(image)
            expected_shape = (self.video_num_frames, *meta["shape"])
            if tuple(image.shape) != expected_shape:
                raise ValueError(
                    f"Expected sparse image shape {expected_shape}, got {tuple(image.shape)} after transforms for key {key}."
                )
            processed_images.append(image)
        return torch.stack(processed_images, dim=0)

    @staticmethod
    def _center_crop_resize_video(
        video: torch.Tensor, size: tuple[int, int], *, crop_scale: float
    ) -> torch.Tensor:
        if video.ndim != 4:
            raise ValueError(f"video must have shape [T,C,H,W], got {tuple(video.shape)}.")
        (_, _, height, width) = video.shape
        crop_h = max(1, min(height, int(round(height * float(crop_scale)))))
        crop_w = max(1, min(width, int(round(width * float(crop_scale)))))
        top = max((height - crop_h) // 2, 0)
        left = max((width - crop_w) // 2, 0)
        cropped = video[..., top : top + crop_h, left : left + crop_w]
        return transforms_F.resize(
            cropped,
            size=list(size),
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _compose_droid_grid(self, video: torch.Tensor) -> torch.Tensor:
        if int(video.shape[0]) != 3:
            raise ValueError(
                f"`camera_layout='{DROID_GRID_CONCAT_MULTI_CAMERA}'` requires exactly 3 cameras, got {int(video.shape[0])}."
            )
        if [int(v) for v in self.video_size] != [DROID_GRID_HEIGHT, DROID_GRID_WIDTH]:
            raise ValueError(
                f"`camera_layout='{DROID_GRID_CONCAT_MULTI_CAMERA}'` requires video_size={[DROID_GRID_HEIGHT, DROID_GRID_WIDTH]}, got {self.video_size}."
            )
        per_camera_h = DROID_GRID_HEIGHT // 2
        per_camera_w = DROID_GRID_WIDTH // 2
        left = self._center_crop_resize_video(
            video[0], (per_camera_h, per_camera_w), crop_scale=0.95
        )
        right = self._center_crop_resize_video(
            video[1], (per_camera_h, per_camera_w), crop_scale=0.95
        )
        wrist = self._center_crop_resize_video(
            video[2], (per_camera_h, per_camera_w), crop_scale=0.95
        )
        wrist_wide = wrist.repeat_interleave(2, dim=-1)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([wrist_wide, bottom], dim=-2)

    def _compose_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(
                f"Sparse Stage 2 `pixel_values` must have shape [num_cameras,T,C,H,W], got {tuple(pixel_values.shape)}."
            )
        (num_cameras, video_t, channels, height, width) = pixel_values.shape
        video = pixel_values.view(num_cameras, video_t, channels, height, width)
        if self.camera_layout == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`camera_layout='robotwin'` requires exactly 3 cameras, got {num_cameras}."
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif self.camera_layout in {
            ROBOCASA_DREAMZERO_CONCAT_MULTI_CAMERA,
            ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA,
        }:
            if num_cameras != 3:
                raise ValueError(
                    f"`camera_layout='{self.camera_layout}'` requires exactly 3 cameras, got {num_cameras}."
                )
            return compose_robocasa_dreamzero(
                video,
                output_size=tuple(self.video_size),
                expected_raw_size=ROBOCASA_RAW_CAMERA_SIZE,
            ).permute(1, 0, 2, 3)
        elif self.camera_layout == DROID_GRID_CONCAT_MULTI_CAMERA:
            video = self._compose_droid_grid(video)
        elif num_cameras > 1:
            if self.camera_layout == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.camera_layout == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid camera_layout: {self.camera_layout}. Expected one of: horizontal, vertical, robotwin, robocasa_dreamzero_352, robocasa_dreamzero_448, droid_grid."
                )
        else:
            video = video.squeeze(0)
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        return video.permute(1, 0, 2, 3)

    def _normalize_action_state(
        self, nested_sample: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = nested_sample["action"]
        if _is_stage2_droid_dataset_name(self.dataset_name) and self.action_history_horizon > 0:
            if not isinstance(action, dict):
                raise TypeError(
                    f"DROID action-history samples require an action feature mapping, got {type(action).__name__}."
                )
            gripper = action.get(DROID_GRIPPER_POSITION_KEY)
            if gripper is None:
                raise KeyError("DROID action-history samples are missing action.gripper_position.")
            action = dict(action)
            action[DROID_GRIPPER_POSITION_KEY] = binarize_droid_gripper_action_history(
                gripper, action_history_horizon=self.action_history_horizon
            )
        data = {
            "action": action,
            "state": nested_sample["state"],
            "action_is_pad": nested_sample["action_is_pad"],
            "state_is_pad": nested_sample["state_is_pad"],
            "idx": nested_sample["idx"],
        }
        if self.processor.delta_action_dim_mask is not None:
            action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
            if bool(action_is_pad.any().item()):
                for key, dim_mask in self.processor.delta_action_dim_mask.items():
                    cur_action = data["action"][key]
                    cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                    cur_dim_mask = dim_mask.to(device=cur_action.device)
                    pad_delta_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                    cur_action[pad_delta_mask] = 0.0
        data = self.processor.action_state_transform(data)
        data = self.processor.normalizer.forward(data)
        data = self.processor.action_state_merger.forward(data)
        return (data["action"], data["state"], data["action_is_pad"], data["state_is_pad"])

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_paths = [
            cache_dir / f"{hashed}.t5_len{self.context_len}.{self.text_cache_model_suffix}.pt"
            for cache_dir in self.text_embedding_cache_dirs
        ]
        cache_path = next((path for path in cache_paths if path.exists()), cache_paths[0])
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache. Tried: {cache_paths}. Run dreamfree.data.prepare_text first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        validate_text_cache(payload, cache_path)
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        return (context, context_mask)

    def _get(self, idx: int) -> dict[str, Any]:
        sample_idx = idx
        nested_sample = None
        lerobot_sample = None
        for attempt in range(self.max_padding_retry + 1):
            (sample_idx, lerobot_sample) = self._load_lerobot_sample(sample_idx)
            nested_sample = self._build_nested_sample(sample_idx, lerobot_sample)
            if not self.skip_padding_as_possible:
                break
            has_pad = False
            for key in ("action_is_pad", "state_is_pad", "image_is_pad"):
                if bool(nested_sample[key].any().item()):
                    has_pad = True
            if not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = np.random.randint(len(self))
        if nested_sample is None or lerobot_sample is None:
            raise RuntimeError(f"Failed to assemble sparse Stage 2 sample for idx={idx}.")
        pixel_values = self._apply_image_transforms(nested_sample)
        video = self._compose_video(pixel_values)
        (action_with_history, proprio_full, action_is_pad_with_history, proprio_is_pad) = (
            self._normalize_action_state(nested_sample)
        )
        proprio = proprio_full
        image_is_pad = nested_sample["image_is_pad"]
        expected_action_tokens = self.action_history_horizon + self.action_horizon
        if (
            int(action_with_history.shape[0]) != expected_action_tokens
            or int(proprio.shape[0]) != self.action_horizon
        ):
            raise ValueError(
                f"Sparse Stage 2 action/proprio horizon mismatch: action={tuple(action_with_history.shape)} proprio={tuple(proprio.shape)} expected_action_tokens={expected_action_tokens} expected_horizon={self.action_horizon}."
            )
        if tuple(action_is_pad_with_history.shape) != (expected_action_tokens,) or tuple(
            proprio_is_pad.shape
        ) != (self.action_horizon,):
            raise ValueError(
                f"Sparse Stage 2 pad-mask horizon mismatch: action_is_pad={tuple(action_is_pad_with_history.shape)} proprio_is_pad={tuple(proprio_is_pad.shape)} expected_action_tokens={expected_action_tokens}."
            )
        history_end = self.action_history_horizon
        previous_action = action_with_history[:history_end].clone()
        previous_action_is_pad = action_is_pad_with_history[:history_end].clone()
        action = action_with_history[history_end:]
        action_is_pad = action_is_pad_with_history[history_end:]
        if self.action_history_horizon > 0:
            previous_action[previous_action_is_pad] = 0.0
        if video.shape[1] <= 1:
            raise ValueError(
                f"`video` must have at least 2 frames, got shape {tuple(video.shape)}."
            )
        if int(action.shape[0]) % (int(video.shape[1]) - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}."
            )
        task = self._sample_instruction(lerobot_sample)
        task_text = str(task).strip()
        if not task_text:
            raise ValueError(f"Invalid empty task text: {task!r}")
        instruction = DEFAULT_PROMPT.format(task=task_text)
        (context, context_mask) = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        sample = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
        }
        if self.action_history_horizon > 0:
            sample["previous_action"] = previous_action
            sample["previous_action_is_pad"] = previous_action_is_pad
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._get(idx)


def _build_stage2_live_dataset(
    cfg: Stage2Config,
    *,
    split: str,
    val_fraction: float | None = None,
    test_fraction: float | None = None,
) -> Dataset:
    is_training = split == "train"
    video_roots = None
    resolved_val_fraction = float(cfg.val_fraction) if val_fraction is None else float(val_fraction)
    resolved_test_fraction = (
        float(getattr(cfg, "test_fraction", 0.0)) if test_fraction is None else float(test_fraction)
    )
    return _SparseStage2RobotVideoDataset(
        dataset_dirs=_parse_dataset_dirs(cfg.dataset_dirs),
        dataset_name=str(cfg.dataset_name),
        shape_meta=_build_stage2_shape_meta(cfg),
        num_frames=int(cfg.num_frames),
        action_horizon=resolve_stage2_action_horizon(cfg),
        action_history_horizon=int(getattr(cfg, "action_history_horizon", 0)),
        video_size=[int(cfg.height), int(cfg.width)],
        processor=_build_stage2_processor(cfg),
        text_embedding_cache_dir=str(cfg.text_embedding_cache_dir),
        text_cache_model_suffix=_stage2_text_cache_suffix(cfg),
        context_len=int(cfg.tokenizer_max_len),
        dataset_stats_path=str(cfg.dataset_stats_path),
        val_fraction=resolved_val_fraction,
        test_fraction=resolved_test_fraction,
        split=str(split),
        is_training_set=is_training,
        global_sample_stride=int(cfg.global_sample_stride),
        action_video_freq_ratio=int(cfg.action_video_freq_ratio),
        skip_padding_as_possible=False,
        max_padding_retry=3,
        camera_layout=str(cfg.camera_layout),
        override_instruction=None,
        strict_loading=bool(cfg.strict_dataset_errors),
        strict_errors=bool(cfg.strict_dataset_errors),
        video_roots=video_roots,
        allow_mixed_fps_for_cached_latents=False,
        metadata_only_for_cached_latents=False,
    )


def _attach_cache_episode_index(
    dataset: _SparseStage2RobotVideoDataset,
) -> _SparseStage2RobotVideoDataset:
    raw_datasets = getattr(dataset.multi_dataset, "_datasets", None)
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError(
            "Cached Stage 2 action training requires a fallback dataset backed by at least one LeRobot dataset."
        )
    episode_data_indices = []
    end_index = 0
    for raw_dataset in raw_datasets:
        episode_data_index = getattr(raw_dataset, "episode_data_index", None)
        if episode_data_index is None:
            raise ValueError(
                "Sparse Stage 2 fallback dataset is missing episode_data_index for cached latent routing."
            )
        episode_data_indices.append(
            {
                "from": episode_data_index["from"] + end_index,
                "to": episode_data_index["to"] + end_index,
            }
        )
        if len(episode_data_index["to"]) > 0:
            end_index += int(episode_data_index["to"][-1])
    dataset.lerobot_dataset = SimpleNamespace(
        multi_dataset=dataset.multi_dataset,
        episode_data_index={
            "from": torch.cat([item["from"] for item in episode_data_indices]),
            "to": torch.cat([item["to"] for item in episode_data_indices]),
        },
        processor=dataset.processor,
        is_training_set=dataset.is_training_set,
    )
    return dataset


def _build_stage2_dataset(cfg, *, split, val_fraction=None, test_fraction=None):
    if cfg.dataset_name in yam.DATASET_NAMES:
        from dreamfree.data.abc_dataset import ABCCachedDataset, ABCLiveDataset

        dataset_type = ABCCachedDataset if cfg.use_cached_latents else ABCLiveDataset
        return dataset_type(cfg, split=split)
    live = _build_stage2_live_dataset(
        cfg,
        split=split,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    if not cfg.use_cached_latents:
        return live
    from dreamfree.data.lerobot.cached_latent_dataset import CachedStage2LatentDataset

    live = _attach_cache_episode_index(live)
    return CachedStage2LatentDataset(
        cache_dir=str(cfg.video_latent_cache_dir),
        source_dataset=live,
        expected_cache_contract=_stage2_latent_cache_expected_contract(cfg),
    )


def _stage2_cached_latent_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty Stage 2 batch.")
    for sample in batch:
        if "video" in sample:
            raise ValueError("Stage 2 cached-latent batches must not include `video`.")
        if "input_latents" not in sample:
            raise ValueError(
                "Stage 2 cached-latent batches require `input_latents` for every sample."
            )
    collated: dict[str, Any] = {}
    shared_keys = set().union(*(sample.keys() for sample in batch))
    shared_keys.discard("input_latents")
    for key in sorted(shared_keys):
        values = [sample[key] for sample in batch if key in sample]
        if len(values) != len(batch):
            raise ValueError(
                f"Stage 2 cached-latent collate expected every sample to define `{key}`."
            )
        collated[key] = default_collate(values)
    collated["input_latents"] = default_collate([sample["input_latents"] for sample in batch])
    return collated


def _build_stage2_dataloader(dataset: Dataset, cfg: Stage2Config, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed))
    num_workers = int(cfg.num_workers)
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
    persistent_workers = bool(cfg.dataloader_persistent_workers)
    prefetch_factor = int(cfg.dataloader_prefetch_factor)
    if num_workers == 0:
        if persistent_workers:
            raise ValueError("dataloader_persistent_workers=true requires num_workers > 0.")
        loader_kwargs: dict[str, Any] = {}
    else:
        if prefetch_factor <= 0:
            raise ValueError(
                f"dataloader_prefetch_factor must be > 0 when num_workers > 0, got {prefetch_factor}."
            )
        loader_kwargs = {
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        }
    collate_fn = _stage2_cached_latent_collate if bool(cfg.use_cached_latents) else None
    sampler = None
    return DataLoader(
        dataset,
        batch_size=int(cfg.per_device_batch_size),
        shuffle=bool(shuffle) and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
