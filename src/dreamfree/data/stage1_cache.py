"""Video latent cache reader for stage 1."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from dreamfree.data import yam
from dreamfree.data.lerobot.cached_latent_dataset import (
    stage2_latent_cache_expected_contract,
    validate_stage2_latent_cache_header_file,
    validate_stage2_latent_tensor_contract,
)
from dreamfree.data.text_cache import load_text_context as _load_stage1_text_cache
from dreamfree.data.video import (
    Stage0Episode,
    _is_droid_cache_meta,
    _parse_stage0_text_cache_dirs,
    _parse_video_latent_cache_dirs,
    _stage0_cache_dir_can_use_flat_episodes,
    _stage0_cached_or_overlay_latent_header,
    _stage0_root_cache_name,
    _stage0_root_cache_name_map,
    _validate_cached_or_droid_dreamzero_latent_cache_file,
    _validate_droid_dreamzero_latent_cache_metadata,
)
from dreamfree.models.wan22.i2v_condition import WAN22_A14B_I2V_CONDITION_LATENTS_KEY
from dreamfree.training.stage1.config import Stage1Config

if TYPE_CHECKING:
    from dreamfree.data.distillation import DistillationDataset


def _optional_path_identity(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _stage1_raw_video_frames(cfg: Stage1Config) -> int:
    raw_frame_stride = int(getattr(cfg, "raw_frame_stride"))
    if raw_frame_stride <= 0:
        raise ValueError(f"raw_frame_stride must be positive, got {raw_frame_stride}.")
    return (int(getattr(cfg, "clip_frames")) - 1) * raw_frame_stride + 1


def _stage1_latent_cache_expected_contract(cfg: Stage1Config) -> dict[str, Any]:
    contract = stage2_latent_cache_expected_contract(
        video_model_family=str(getattr(cfg, "video_model_family", "wan22_ti2v_5b")),
        model_id=getattr(cfg, "model_id", None),
        height=int(getattr(cfg, "height")),
        width=int(getattr(cfg, "width")),
        num_frames=_stage1_raw_video_frames(cfg),
        action_video_freq_ratio=int(getattr(cfg, "raw_frame_stride")),
    )
    if str(cfg.camera_layout).startswith("robocasa_dreamzero_"):
        contract["concat_layout"] = str(getattr(cfg, "camera_layout"))
        if contract["video_model_family"] == "wan22_i2v_a14b":
            contract["require_i2v_first_latent_equality"] = True
    if cfg.dataset_name in yam.DATASET_NAMES:
        contract["abc_contract"] = yam.PRESETS[cfg.dataset_name].contract
    return contract


def resolve_stage1_latent_cache_dirs(cfg: Stage1Config) -> list[Path]:
    raw = _optional_path_identity(getattr(cfg, "video_latent_cache_dir", None))
    if raw is None:
        raise ValueError("Cached video mode requires video_latent_cache_dir.")
    return [Path(path).expanduser() for path in _parse_video_latent_cache_dirs(raw)]


def resolve_stage1_latent_cache_dir(cfg: Stage1Config) -> Path:
    cache_dirs = resolve_stage1_latent_cache_dirs(cfg)
    if len(cache_dirs) != 1:
        raise ValueError(
            "Stage 1 video_latent_cache_dir contains multiple cache roots; use resolve_stage1_latent_cache_dirs instead."
        )
    return cache_dirs[0]


def _validate_stage1_latent_cache_config(cfg):
    if not cfg.use_cached_latents:
        return
    paths = resolve_stage1_latent_cache_dirs(cfg)
    if not paths:
        raise ValueError("Cached video mode requires video_latent_cache_dir.")
    cfg.video_latent_cache_dir = ",".join(str(path) for path in paths)
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Video latent cache directory does not exist: {path}")


def _validate_stage1_input_latents_contract(latents: torch.Tensor, cfg: Stage1Config) -> None:
    validate_stage2_latent_tensor_contract(
        latents, expected=_stage1_latent_cache_expected_contract(cfg), stage_label="Stage 1"
    )


class _Stage1EpisodeHandleCache:
    """Per-worker LRU cache of open safetensors episode handles."""

    def __init__(self, maxsize: int = 64):
        self._maxsize = int(maxsize)
        self._store: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path) -> Any:
        handle = self._store.get(path)
        if handle is None:
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._store[path] = handle
            if len(self._store) > self._maxsize:
                (_, victim) = self._store.popitem(last=False)
                try:
                    victim.__exit__(None, None, None)
                except Exception:
                    pass
        else:
            self._store.move_to_end(path)
        return handle


class CachedStage1LatentDataset(Dataset):
    """Stage 1 dataset wrapper that serves precomputed video latents instead of decoded video."""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        source_dataset: DistillationDataset,
        expected_cache_contract: dict[str, Any],
    ):
        super().__init__()
        self.cache_dirs = [
            Path(path).expanduser().resolve() for path in _parse_video_latent_cache_dirs(cache_dir)
        ]
        if not self.cache_dirs:
            raise ValueError(
                "CachedStage1LatentDataset requires at least one latent cache directory."
            )
        self.cache_dir = self.cache_dirs[0]
        self.source_dataset = source_dataset
        self.expected_cache_contract = dict(expected_cache_contract)
        self.total_windows = int(getattr(source_dataset, "total_windows"))
        self.epoch_size = int(len(source_dataset))
        self.context_len = int(source_dataset.context_len)
        self.text_dim = int(source_dataset.text_dim)
        self.cache_model_id = str(source_dataset.cache_model_id)
        self.text_embedding_cache_dir = str(source_dataset.text_embedding_cache_dir)
        fallback_text_dirs = getattr(source_dataset, "text_embedding_cache_dirs", None)
        if fallback_text_dirs is None:
            self.text_embedding_cache_dirs = _parse_stage0_text_cache_dirs(
                self.text_embedding_cache_dir
            )
        else:
            self.text_embedding_cache_dirs = list(fallback_text_dirs)
        if not self.text_embedding_cache_dirs:
            raise ValueError(
                "CachedStage1LatentDataset requires at least one text embedding cache directory."
            )
        missing_cache_dirs = [
            str(cache_dir) for cache_dir in self.cache_dirs if not cache_dir.exists()
        ]
        if missing_cache_dirs:
            raise FileNotFoundError(
                f"Stage 1 video latent cache directories not found: {missing_cache_dirs}. Precompute the cache first or set use_cached_latents=false."
            )
        roots = sorted({Path(episode.root).resolve() for episode in source_dataset.episodes})
        self._root_cache_names = _stage0_root_cache_name_map(roots)
        self._single_root_cache = len(roots) == 1
        self._validate_episode_cache_files()
        self._handles = _Stage1EpisodeHandleCache(maxsize=64)

    def __len__(self) -> int:
        return self.epoch_size

    def _episode_path_candidates(self, episode: Stage0Episode) -> list[Path]:
        filename = f"episode_{int(episode.episode_index):06d}.safetensors"
        root = Path(episode.root).resolve()
        root_name = self._root_cache_names.get(root)
        if root_name is None:
            raise KeyError(f"Stage 1 cached latent root was not registered: {root}")
        root_names = []
        for candidate in (root_name, root.name, _stage0_root_cache_name(root)):
            if candidate not in root_names:
                root_names.append(candidate)
        candidates: list[Path] = []
        for cache_dir in self.cache_dirs:
            flat_path = cache_dir / filename
            if self._single_root_cache or _stage0_cache_dir_can_use_flat_episodes(cache_dir, root):
                candidates.append(flat_path)
            candidates.extend(cache_dir / candidate / filename for candidate in root_names)
        return candidates

    def _episode_path(self, episode: Stage0Episode) -> Path:
        candidates = self._episode_path_candidates(episode)
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def _latent_handle_for_cache_path(self, path: Path, handle: Any) -> tuple[Any, Path]:
        return (handle, path)

    def _validate_episode_cache_files(self) -> None:
        missing: list[str] = []
        too_short: list[str] = []
        first_existing: Path | None = None
        for episode in self.source_dataset.episodes:
            path = self._episode_path(episode)
            if not path.exists():
                root = Path(episode.root).resolve()
                root_name = self._root_cache_names.get(root, Path(episode.root).name)
                candidate_dirs = sorted(
                    {str(path.parent) for path in self._episode_path_candidates(episode)}
                )
                missing.append(
                    f"{root_name}/episode_{int(episode.episode_index):06d} candidates={candidate_dirs}"
                )
                continue
            if self.source_dataset.cfg.dataset_name in yam.DATASET_NAMES:
                yam.validate_cache(path, episode.root, self.source_dataset.cfg.dataset_name)
            if first_existing is None:
                first_existing = path
            (meta, latent_shape, _) = _stage0_cached_or_overlay_latent_header(path)
            if _is_droid_cache_meta(meta):
                _validate_droid_dreamzero_latent_cache_metadata(path, stage_label="Stage 1")
            elif str(self.expected_cache_contract.get("concat_layout", "")).startswith(
                "robocasa_dreamzero_"
            ):
                validate_stage2_latent_cache_header_file(
                    path, expected=self.expected_cache_contract, stage_label="Stage 1"
                )
            latent_count = int(latent_shape[0])
            num_samples = int(meta.get("num_samples", latent_count))
            if num_samples != latent_count:
                raise ValueError(
                    f"Stage 1 latent cache metadata/sample mismatch in {path}: num_samples={num_samples} latent_shape[0]={latent_count}."
                )
            if num_samples < int(episode.length):
                too_short.append(
                    f"episode={episode.episode_index} cache_samples={num_samples} episode_length={episode.length}"
                )
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} Stage 1 cached episode files are missing under {self.cache_dirs}. First few: {missing[:5]}. Precompute the cache first or set use_cached_latents=false."
            )
        if first_existing is None:
            raise FileNotFoundError(
                f"No Stage 1 cached episode files found under {self.cache_dir}."
            )
        _validate_cached_or_droid_dreamzero_latent_cache_file(
            first_existing, expected=self.expected_cache_contract, stage_label="Stage 1"
        )
        if too_short:
            raise ValueError(
                f"Stage 1 cached latents must cover full episodes. First mismatches: {too_short[:5]}. Precompute a full-coverage cache or set use_cached_latents=false."
            )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        (episode, start_frame) = self.source_dataset._sample_window(idx)
        path = self._episode_path(episode)
        handle = self._handles.get(path)
        (latent_handle, latent_path) = self._latent_handle_for_cache_path(path, handle)
        latents_slice = latent_handle.get_slice("latents")
        latent_count = int(latents_slice.get_shape()[0])
        if int(start_frame) < 0 or int(start_frame) >= latent_count:
            raise IndexError(
                f"Stage 1 latent cache frame index out of range: episode={episode.episode_index} start_frame={start_frame} latent_count={latent_count} cache={path} latent_source={latent_path}."
            )
        latent = latents_slice[int(start_frame) : int(start_frame) + 1, ...]
        if not isinstance(latent, torch.Tensor):
            raise RuntimeError(f"Unexpected latent slice type {type(latent)} from {latent_path}.")
        latent = latent.squeeze(0).contiguous()
        i2v_condition_latents = None
        if str(self.expected_cache_contract.get("video_model_family")) == "wan22_i2v_a14b":
            try:
                i2v_slice = latent_handle.get_slice(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)
            except Exception as exc:
                raise KeyError(
                    f"14B Stage 1 cached-latent training requires tensor {WAN22_A14B_I2V_CONDITION_LATENTS_KEY!r} in {latent_path}. Regenerate the cache with official Wan2.2 I2V conditioning."
                ) from exc
            i2v_condition = i2v_slice[int(start_frame) : int(start_frame) + 1, ...]
            if not isinstance(i2v_condition, torch.Tensor):
                raise RuntimeError(
                    f"Unexpected {WAN22_A14B_I2V_CONDITION_LATENTS_KEY} slice type {type(i2v_condition)} from {latent_path}."
                )
            i2v_condition_latents = i2v_condition.squeeze(0).contiguous()
        task = self.source_dataset._select_episode_task(episode)
        prompt = self.source_dataset._prompt_for_task(task)
        (context, context_mask) = _load_stage1_text_cache(
            cache_dir=self.text_embedding_cache_dirs,
            prompt=prompt,
            context_len=self.context_len,
            text_dim=self.text_dim,
            model_id=self.cache_model_id,
        )
        sample = {
            "input_latents": latent,
            "context": context,
            "context_mask": context_mask,
            "prompt": prompt,
            "task": task,
            "episode_index": torch.tensor(episode.episode_index, dtype=torch.long),
            "start_frame": torch.tensor(start_frame, dtype=torch.long),
            "sample_id": f"{episode.root}|episode={episode.episode_index}|start={start_frame}|frames={self.source_dataset.clip_frames}|stride={self.source_dataset.raw_frame_stride}",
        }
        if i2v_condition_latents is not None:
            sample[WAN22_A14B_I2V_CONDITION_LATENTS_KEY] = i2v_condition_latents
        return sample
