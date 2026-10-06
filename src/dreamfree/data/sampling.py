from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from dreamfree.training.stage0.config import Stage0Config
from dreamfree.training.stage0.constants import logger


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs inside a dataloader worker.

    Args:
        worker_id: Worker index assigned by ``DataLoader``.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


class Stage01WindowSubsetSampler(Sampler[int]):
    """Seeded sampler over flat Stage 0/1 window ids."""

    def __init__(self, dataset: Dataset, *, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        total_windows = getattr(dataset, "total_windows", None)
        if total_windows is None:
            raise ValueError("Stage 0/1 shuffled sampling requires dataset.total_windows.")
        self.total_windows = int(total_windows)
        self.epoch_size = int(len(dataset))
        if self.total_windows <= 0:
            raise ValueError(
                f"Stage 0/1 shuffled sampling requires total_windows > 0, got {self.total_windows}."
            )
        if self.epoch_size < 0:
            raise ValueError(
                f"Stage 0/1 shuffled sampling requires non-negative epoch_size, got {self.epoch_size}."
            )
        if self.epoch_size > self.total_windows:
            raise ValueError(
                f"Stage 0/1 shuffled sampling requires len(dataset) <= dataset.total_windows, got len(dataset)={self.epoch_size} total_windows={self.total_windows}."
            )

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        population = range(self.total_windows)
        if self.epoch_size == self.total_windows:
            indices = list(population)
            rng.shuffle(indices)
            return iter(indices)
        return iter(rng.sample(population, self.epoch_size))

    def __len__(self) -> int:
        return self.epoch_size


def _parse_dataset_mix_weights(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "null", "None", "none"}:
            return {}
        parts = [part.strip() for part in text.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        parts = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise TypeError(
            f"dataset_mix_weights must be a comma-separated string or sequence, got {type(value)}."
        )
    weights: dict[str, float] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(
                f"dataset_mix_weights entries must be formatted as source=weight, got {part!r}."
            )
        (source, raw_weight) = [item.strip() for item in part.split("=", 1)]
        if not source:
            raise ValueError(f"dataset_mix_weights contains an empty source in entry {part!r}.")
        if source in weights:
            raise ValueError(f"dataset_mix_weights contains duplicate source {source!r}.")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"dataset_mix_weights for {source!r} must be finite and positive, got {weight}."
            )
        weights[source] = weight
    if not weights:
        raise ValueError(
            "dataset_mix_weights must contain at least one source=weight entry when set."
        )
    return weights


def _unwrap_window_dataset(dataset: Dataset) -> Dataset:
    current: Any = dataset
    seen: set[int] = set()
    while hasattr(current, "source_dataset"):
        current_id = id(current)
        if current_id in seen:
            raise RuntimeError(
                "Cycle detected while unwrapping source_dataset for source-weighted sampling."
            )
        seen.add(current_id)
        current = current.source_dataset
    return current


def _source_window_indices(dataset: Dataset) -> tuple[dict[str, list[int]], list[str]]:
    base = _unwrap_window_dataset(dataset)
    episodes = getattr(base, "episodes", None)
    window_counts = getattr(base, "window_counts", None)
    if episodes is None or window_counts is None:
        raise ValueError(
            "Source-weighted sampling requires dataset.episodes and dataset.window_counts."
        )
    if len(episodes) != len(window_counts):
        raise ValueError(
            f"Source-weighted sampling requires episodes/window_counts length match, got {len(episodes)} and {len(window_counts)}."
        )
    indices_by_source: dict[str, list[int]] = {}
    display_sources: list[str] = []
    flat_start = 0
    for episode, count in zip(episodes, window_counts, strict=True):
        count = int(count)
        if count <= 0:
            continue
        root = Path(episode.root).expanduser().resolve()
        source_keys = (root.name, str(root))
        window_indices = list(range(flat_start, flat_start + count))
        for key in source_keys:
            indices_by_source.setdefault(key, []).extend(window_indices)
        if root.name not in display_sources:
            display_sources.append(root.name)
        flat_start += count
    return (indices_by_source, sorted(display_sources))


def _weighted_epoch_counts(weights: dict[str, float], epoch_size: int) -> dict[str, int]:
    if epoch_size <= 0:
        raise ValueError(
            f"epoch_size must be positive for source-weighted sampling, got {epoch_size}."
        )
    total_weight = sum(weights.values())
    raw_counts = {
        source: weight / total_weight * epoch_size for (source, weight) in weights.items()
    }
    counts = {source: int(math.floor(raw)) for (source, raw) in raw_counts.items()}
    remaining = int(epoch_size) - sum(counts.values())
    ranked = sorted(
        weights,
        key=lambda source: (raw_counts[source] - counts[source], weights[source], source),
        reverse=True,
    )
    for source in ranked[:remaining]:
        counts[source] += 1
    if epoch_size >= len(counts):
        zero_sources = [source for (source, count) in counts.items() if count == 0]
        for source in zero_sources:
            donor = max((key for key in counts if key != source), key=lambda key: counts[key])
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[source] = 1
    return counts


class Stage01SourceWeightedWindowSampler(Sampler[int]):
    """Seeded sampler that enforces source or source-group replay weights."""

    def __init__(self, dataset: Dataset, *, seed: int, mix_weights: dict[str, float]):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.epoch_size = int(len(dataset))
        (source_indices, display_sources) = _source_window_indices(dataset)
        self.mix_weights = dict(mix_weights)
        self.source_indices: dict[str, list[int]] = {}
        missing: list[str] = []
        claimed_indices: dict[int, str] = {}
        overlaps: list[str] = []
        for source_group in self.mix_weights:
            members = [member.strip() for member in source_group.split("+") if member.strip()]
            if not members:
                missing.append(source_group)
                continue
            missing.extend(member for member in members if member not in source_indices)
            if any(member not in source_indices for member in members):
                continue
            group_indices = sorted(
                {index for member in members for index in source_indices[member]}
            )
            for index in group_indices:
                previous_group = claimed_indices.get(index)
                if previous_group is not None and previous_group != source_group:
                    overlaps.append(f"{previous_group!r} and {source_group!r}")
                    break
                claimed_indices[index] = source_group
            self.source_indices[source_group] = group_indices
        if missing:
            raise ValueError(
                f"dataset_mix_weights sources not found in dataset: {sorted(set(missing))}. Available root names: {display_sources}."
            )
        if overlaps:
            raise ValueError(
                "dataset_mix_weights source groups must not overlap; found shared windows between "
                + ", ".join(sorted(set(overlaps)))
                + "."
            )
        self.epoch_counts = _weighted_epoch_counts(self.mix_weights, self.epoch_size)
        logger.info(
            "Dataset source-weighted sampling: epoch_size=%d weights=%s counts=%s",
            self.epoch_size,
            self.mix_weights,
            self.epoch_counts,
        )
        undersized_sources = [
            f"{source}: requested={count} available={len(self.source_indices[source])}"
            for (source, count) in self.epoch_counts.items()
            if "+" not in source and count > len(self.source_indices[source])
        ]
        if undersized_sources:
            raise ValueError(
                f"dataset_mix_weights requested more no-replacement samples than a source provides. Use a grouped source entry for intentional replay. Details: {undersized_sources}"
            )
        self.replayed_groups = tuple(
            source
            for (source, count) in self.epoch_counts.items()
            if count > len(self.source_indices[source])
        )
        if self.replayed_groups:
            logger.info(
                "Dataset mix replays grouped windows with replacement: %s",
                {
                    source: {
                        "requested": self.epoch_counts[source],
                        "available": len(self.source_indices[source]),
                    }
                    for source in self.replayed_groups
                },
            )

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices: list[int] = []
        for source, count in self.epoch_counts.items():
            population = self.source_indices[source]
            if int(count) <= len(population):
                indices.extend(rng.sample(population, int(count)))
            else:
                indices.extend(rng.choices(population, k=int(count)))
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.epoch_size


def build_dataloader(dataset: Dataset, cfg: Stage0Config, *, shuffle: bool = False) -> DataLoader:
    """Create a deterministic dataloader wrapper for the dataset.

    Args:
        dataset: Train or validation dataset.
        cfg: Stage 0 dataloader configuration.
        shuffle: Whether to use a seeded train sampler over flat window ids.

    Returns:
        PyTorch dataloader. Train randomness comes from the dataset sampler itself.
    """
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
    mix_weights = _parse_dataset_mix_weights(getattr(cfg, "dataset_mix_weights", None))
    if mix_weights:
        sampler = Stage01SourceWeightedWindowSampler(
            dataset, seed=int(cfg.seed), mix_weights=mix_weights
        )
    else:
        sampler = Stage01WindowSubsetSampler(dataset, seed=int(cfg.seed)) if bool(shuffle) else None
    return DataLoader(
        dataset,
        batch_size=int(cfg.per_device_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
        **loader_kwargs,
    )
