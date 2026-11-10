from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Sequence

import torch

from dreamfree.data.stage1_cache import (
    CachedStage1LatentDataset,
    _stage1_latent_cache_expected_contract,
    resolve_stage1_latent_cache_dirs,
)
from dreamfree.data.text_cache import DEFAULT_PROMPT
from dreamfree.data.text_cache import load_text_context as _load_stage1_text_cache
from dreamfree.data.video import (
    DatasetRoots,
    Stage0Episode,
    VideoDataset,
    _normalize_stage0_task_candidates_for_source,
    _parse_stage0_text_cache_dirs,
    _stage0_select_camera_keys_for_features,
    resolve_episode_task_candidates,
)
from dreamfree.training.runtime import _json_load
from dreamfree.training.stage1.config import Stage1Config
from dreamfree.training.stage1.constants import (
    ROBOCASA_STAGE1_448_CONCAT_MULTI_CAMERA,
    ROBOCASA_STAGE1_448_DATASET_ALIASES,
    ROBOCASA_STAGE1_448_HEIGHT,
    ROBOCASA_STAGE1_448_WIDTH,
    ROBOCASA_STAGE1_CONCAT_MULTI_CAMERA,
    ROBOCASA_STAGE1_HEIGHT,
    ROBOCASA_STAGE1_WIDTH,
    logger,
)


def _normalize_stage1_dataset_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _is_stage1_robocasa_448_dataset(cfg: Stage1Config) -> bool:
    return (
        _normalize_stage1_dataset_name(getattr(cfg, "dataset_name", ""))
        in ROBOCASA_STAGE1_448_DATASET_ALIASES
    )


def _is_stage1_robocasa_success_family_dataset(cfg):
    return cfg.dataset_name == "robocasa"


def _stage1_robocasa_layout_spec(cfg: Stage1Config) -> dict[str, int | str]:
    if _is_stage1_robocasa_448_dataset(cfg):
        return {
            "camera_layout": ROBOCASA_STAGE1_448_CONCAT_MULTI_CAMERA,
            "height": ROBOCASA_STAGE1_448_HEIGHT,
            "width": ROBOCASA_STAGE1_448_WIDTH,
        }
    return {
        "camera_layout": ROBOCASA_STAGE1_CONCAT_MULTI_CAMERA,
        "height": ROBOCASA_STAGE1_HEIGHT,
        "width": ROBOCASA_STAGE1_WIDTH,
    }


def _is_stage1_droid_dataset(cfg):
    return cfg.dataset_name == "droid"


def _stage1_model_suffix(model_id: str) -> str:
    """Return the filename suffix used for model-specific text caches.

    Args:
        model_id: Hugging Face model id or local model name used to build the
            cached T5 context tensors.

    Returns:
        Lowercase cache suffix that is safe to embed in a filename.

    Raises:
        ValueError: If no non-empty suffix can be derived from `model_id`.
    """
    if str(model_id) == "Wan-AI/Wan2.2-TI2V-5B":
        return "wan22ti2v5b"
    if str(model_id) == "Wan-AI/Wan2.2-I2V-A14B":
        return "wan22i2va14b"
    base = str(model_id).split("/")[-1]
    suffix = re.sub("[^a-z0-9]+", "", base.lower())
    if not suffix:
        raise ValueError(f"Cannot derive text cache suffix from model_id={model_id!r}.")
    return suffix


def stage1_text_cache_path(
    cache_dir: str | Path, prompt: str, context_len: int, model_id: str
) -> Path:
    """Build the deterministic text embedding cache path for one prompt.

    Args:
        cache_dir: Directory containing Stage 1 cached text embeddings.
        prompt: Fully formatted prompt string.
        context_len: Expected token/context length in the cached tensor.
        model_id: Text/model id used to derive the cache filename suffix.

    Returns:
        Path named by a SHA256 prompt digest plus context/model metadata.
    """
    digest = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
    suffix = _stage1_model_suffix(model_id)
    return Path(cache_dir).expanduser() / f"{digest}.t5_len{int(context_len)}.{suffix}.pt"


def _stage1_text_cache_candidates(
    cache_dir: str | Path | Sequence[str | Path], prompt: str, context_len: int, model_id: str
) -> list[Path]:
    return [
        stage1_text_cache_path(path, prompt, context_len, model_id)
        for path in _parse_stage0_text_cache_dirs(cache_dir)
    ]


def format_stage1_prompt(task: str) -> str:
    """Format a non-empty task string with the shared robot prompt template.

    Args:
        task: Task description from LeRobot metadata.

    Returns:
        Prompt text consumed by the cached text encoder outputs.

    Raises:
        ValueError: If `task` is empty after stripping whitespace.
    """
    task_text = str(task).strip()
    if not task_text:
        raise ValueError("Stage 1 task text must be non-empty.")
    return DEFAULT_PROMPT.format(task=task_text)


def resolve_stage1_task_candidates(
    *, task_root: str | Path, row: dict[str, Any], task_by_index: dict[int, str]
) -> tuple[str, ...]:
    """Resolve strict Stage 1 task text candidates from one LeRobot episode row.

    Args:
        task_root: Root directory for the LeRobot task shard.
        row: Episode metadata row loaded from the LeRobot metadata table.
        task_by_index: Mapping from LeRobot task indices to task strings.

    Returns:
        Non-empty task texts used to create Stage 1 prompts.

    Raises:
        ValueError: If neither direct task text nor known task-index fields
            resolve to at least one non-empty task string.
    """
    return resolve_episode_task_candidates(
        row=row,
        task_by_index=task_by_index,
        default_task=None,
        strict=True,
        stage_label="strict Stage 1",
        task_root=task_root,
    )


class DistillationDataset(VideoDataset):
    """Stage 1 LeRobot dataset with strict task text and cached text embeddings."""

    def __init__(self, task_roots: Sequence[Path | str], *, cfg: Stage1Config, split: str):
        """Create a Stage 1 dataset over prepared LeRobot task roots.

        Args:
            task_roots: Prepared LeRobot task roots to load.
            cfg: Stage 1 configuration controlling video and text-cache shapes.
            split: Dataset split name passed through to the Stage 0 sampler.

        Raises:
            ValueError: If Stage 1 strict text-cache requirements are disabled.
            FileNotFoundError: If any required prompt cache is missing.
        """
        if str(cfg.prompt_policy).strip().lower() != "task_text":
            raise ValueError("Stage 1 requires prompt_policy='task_text'.")
        if not bool(cfg.strict_task_text):
            raise ValueError("Stage 1 requires strict_task_text=true.")
        if not bool(cfg.require_text_cache):
            raise ValueError("Stage 1 requires require_text_cache=true.")
        self.text_embedding_cache_dir = str(cfg.text_embedding_cache_dir)
        self.text_embedding_cache_dirs = _parse_stage0_text_cache_dirs(
            self.text_embedding_cache_dir
        )
        self.context_len = int(cfg.tokenizer_max_len)
        self.text_dim = int(cfg.text_dim)
        self.cache_model_id = str(cfg.model_id)
        super().__init__(task_roots, cfg=cfg, split=split)
        self._validate_all_text_caches()

    def _load_task_root(self, root: Path) -> list[Stage0Episode]:
        """Load one LeRobot task root into Stage 0 episode descriptors.

        Args:
            root: Prepared LeRobot task root.

        Returns:
            List of episodes with resolved camera paths and strict task text.

        Raises:
            FileNotFoundError: If required LeRobot metadata is missing.
            ValueError: If required cameras or task text cannot be resolved.
        """
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing LeRobot info.json: {info_path}")
        info = _json_load(info_path)
        features = info.get("features", {})
        camera_keys = _stage0_select_camera_keys_for_features(self.cfg, features, source=info_path)
        configured_camera_keys = self._camera_keys()
        if tuple(camera_keys) != tuple(configured_camera_keys):
            logger.info(
                "Using root-specific Stage 1 camera keys for %s: configured=%s resolved=%s",
                root,
                ",".join(configured_camera_keys),
                ",".join(camera_keys),
            )
        version = self._normalize_lerobot_version(root, info)
        fps = int(info["fps"])
        chunks_size = int(info.get("chunks_size", 1000))
        video_template = str(info["video_path"])
        if version in {"v2.0", "v2.1"}:
            task_by_index = self._load_tasks_v21(root)
            rows = self._load_episode_rows_v21(root)
        elif version == "v3.0":
            task_by_index = self._load_tasks_v30(root)
            rows = self._load_episode_rows_v30(root, camera_keys=camera_keys)
        else:
            raise AssertionError(f"Unhandled LeRobot codebase_version={version}")
        loaded: list[Stage0Episode] = []
        for row in rows:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            task_candidates = resolve_stage1_task_candidates(
                task_root=root, row=row, task_by_index=task_by_index
            )
            task_candidates = _normalize_stage0_task_candidates_for_source(
                self.cfg, task_candidates, camera_keys=camera_keys
            )
            task = task_candidates[0]
            video_paths = {
                key: str(
                    self._resolve_video_path(
                        root=root,
                        template=video_template,
                        video_key=key,
                        episode_index=episode_index,
                        chunks_size=chunks_size,
                        row=row,
                        version=version,
                    )
                )
                for key in camera_keys
            }
            loaded.append(
                Stage0Episode(
                    root=str(root),
                    episode_index=episode_index,
                    length=length,
                    fps=fps,
                    task=task,
                    video_paths=video_paths,
                    task_candidates=task_candidates,
                    video_start_times=self._video_start_times(
                        row, version=version, camera_keys=camera_keys
                    ),
                )
            )
        return loaded

    def _prompt_for_task(self, task: str) -> str:
        """Return the formatted prompt associated with one task string.

        Args:
            task: Strict non-empty task text.

        Returns:
            Prompt string used as the text-cache key.
        """
        return format_stage1_prompt(task)

    def _validate_all_text_caches(self) -> None:
        """Fail fast if any unique dataset prompt lacks a cached embedding file.

        Raises:
            FileNotFoundError: If at least one prompt cache file is missing.
        """
        prompts = sorted(
            {
                self._prompt_for_task(task)
                for episode in self.episodes
                for task in episode.task_candidates or (episode.task,)
            }
        )
        for prompt in prompts:
            cache_paths = _stage1_text_cache_candidates(
                self.text_embedding_cache_dirs, prompt, self.context_len, self.cache_model_id
            )
            if not any(path.exists() for path in cache_paths):
                raise FileNotFoundError(
                    f"Missing Stage 1 text embedding cache. Tried: {cache_paths}"
                )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Sample a video window and attach cached text-conditioning tensors.

        Args:
            idx: Sampler index used by the parent Stage 0 dataset window logic.

        Returns:
            Batch item with `video` shaped `(3, T, H, W)`, `context` shaped
            `(128, 4096)`, all-true `context_mask` shaped `(128,)`, and
            metadata fields.
        """
        (episode, start_frame) = self._sample_window(idx)
        cameras = {
            key: self._decode_camera(key, path, episode, start_frame)
            for (key, path) in episode.video_paths.items()
        }
        video = self._compose_multicam(cameras)
        video = video.mul(2.0).sub(1.0)
        video = video.permute(1, 0, 2, 3).contiguous()
        task = self._select_episode_task(episode)
        prompt = self._prompt_for_task(task)
        (context, context_mask) = _load_stage1_text_cache(
            cache_dir=self.text_embedding_cache_dirs,
            prompt=prompt,
            context_len=self.context_len,
            text_dim=self.text_dim,
            model_id=self.cache_model_id,
        )
        return {
            "video": video,
            "context": context,
            "context_mask": context_mask,
            "prompt": prompt,
            "task": task,
            "episode_index": torch.tensor(episode.episode_index, dtype=torch.long),
            "start_frame": torch.tensor(start_frame, dtype=torch.long),
            "sample_id": f"{episode.root}|episode={episode.episode_index}|start={start_frame}|frames={self.clip_frames}|stride={self.raw_frame_stride}",
        }


def _parse_stage1_dataset_dirs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        dirs = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        dirs = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise TypeError(
            f"dataset_dirs must be a comma-separated string or sequence, got {type(value)}."
        )
    return dirs


def _resolve_stage1_task_roots(cfg):
    return DatasetRoots(cfg).prepare()


def _first_stage1_latent_cache_file(latent_cache_dir: Path) -> Path:
    first_episode = next(latent_cache_dir.glob("episode_*.safetensors"), None)
    if first_episode is None:
        first_episode = next(latent_cache_dir.glob("*/episode_*.safetensors"), None)
    if first_episode is None:
        raise FileNotFoundError(
            f"No precomputed video latent files found under {latent_cache_dir}. Precompute the cache first or set use_cached_latents=false."
        )
    return first_episode


def _build_stage1_dataset(task_roots, cfg, *, split):
    live = DistillationDataset(task_roots, cfg=cfg, split=split)
    if not cfg.use_cached_latents:
        return live
    return CachedStage1LatentDataset(
        cache_dir=resolve_stage1_latent_cache_dirs(cfg),
        source_dataset=live,
        expected_cache_contract=_stage1_latent_cache_expected_contract(cfg),
    )
