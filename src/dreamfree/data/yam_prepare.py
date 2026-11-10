"""Deterministic episode selection and training-only normalization for YAM datasets."""

import math
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def choose_episodes(rows: list[dict], *, max_hours: float = 300, seed: int = 42) -> dict:
    if not math.isfinite(max_hours) or not 0 < max_hours <= 300:
        raise ValueError("max_hours must be in (0, 300].")
    rng = random.Random(seed)
    queues = defaultdict(list)
    for row in sorted(rows, key=lambda row: int(row["episode_index"])):
        if int(row["length"]) < 33:
            raise ValueError(f"YAM episode {row['episode_index']} has fewer than 33 frames.")
        queues[str(row["task_id"])].append(row)
    for queue in queues.values():
        rng.shuffle(queue)
    spent = {key: 0 for key in queues}
    budget = int(max_hours * 3600 * 30)
    selected = []
    # Always allocate the next whole episode to the least represented task.
    while queues:
        key = min(queues, key=lambda key: (spent[key], key))
        queue = queues[key]
        row = queue.pop()
        length = int(row["length"])
        if length <= budget:
            selected.append(row)
            spent[key] += length
            budget -= length
        if not queue:
            del queues[key]
    by_task = defaultdict(list)
    for row in selected:
        by_task[str(row["task_id"])].append(row)
    episodes = {}
    for key in sorted(by_task):
        group = sorted(by_task[key], key=lambda row: int(row["episode_index"]))
        rng.shuffle(group)
        nval = min(len(group) - 1, max(1, round(len(group) * 0.01))) if len(group) > 1 else 0
        for i, row in enumerate(group):
            episodes[str(int(row["episode_index"]))] = {
                "length": int(row["length"]),
                "task_id": str(row["task_id"]),
                "instruction": row["instruction"],
                "split": "val" if i < nval else "train",
            }
    if not episodes or not any(row["split"] == "val" for row in episodes.values()):
        raise ValueError(
            "YAM selection must contain train and validation episodes; increase max_hours."
        )
    return episodes


def data_path(source: Path, info: dict, row: dict) -> Path:
    return source / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
    )


def compute_stats(
    source: Path, info: dict, rows: list[dict], episodes: dict, scratch: Path
) -> dict:
    action_dim = int(info["features"]["action"]["shape"][0])
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    total_dim = action_dim + state_dim
    selected = {int(key): value for key, value in episodes.items() if value["split"] == "train"}
    total = sum(row["length"] for row in selected.values())
    by_file = defaultdict(list)
    for row in rows:
        if int(row["episode_index"]) in selected:
            by_file[data_path(source, info, row)].append(row)
    scratch.mkdir(parents=True, exist_ok=True)
    # Keep the full training pool on disk; exact quantiles need only one column in RAM.
    with tempfile.TemporaryDirectory(prefix="yam-stats-", dir=scratch) as work:
        values = np.memmap(
            Path(work) / "values.bin", dtype=np.float32, mode="w+", shape=(total, total_dim)
        )
        offset = 0
        for path, file_rows in sorted(by_file.items()):
            table = pq.read_table(
                path, columns=["episode_index", "index", "action", "observation.state"]
            )
            for row in file_rows:
                index = int(row["episode_index"])
                episode = table.filter(pa.compute.equal(table["episode_index"], index))
                ids = np.asarray(episode["index"])
                expected_ids = np.arange(
                    int(row["dataset_from_index"]), int(row["dataset_to_index"])
                )
                if len(episode) != selected[index]["length"] or not np.array_equal(
                    ids, expected_ids
                ):
                    raise ValueError(f"Invalid YAM parquet boundaries for episode {index}: {path}")
                arrays = []
                for key, dim in (("action", action_dim), ("observation.state", state_dim)):
                    column = episode[key].combine_chunks()
                    if (
                        column.null_count
                        or not pa.compute.all(
                            pa.compute.equal(pa.compute.list_value_length(column), dim)
                        ).as_py()
                    ):
                        raise ValueError(f"Invalid YAM {key} vectors in episode {index}: {path}")
                    flat = pa.compute.list_flatten(column)
                    if flat.null_count:
                        raise ValueError(f"Null YAM {key} values in episode {index}: {path}")
                    arrays.append(np.asarray(flat, dtype=np.float32).reshape(-1, dim))
                array = np.concatenate(arrays, axis=1)
                if array.shape != (len(episode), total_dim) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid YAM action/state values in episode {index}: {path}")
                values[offset : offset + len(episode)] = array
                offset += len(episode)
        if offset != total:
            raise ValueError(f"YAM normalization frame count mismatch: {offset} != {total}")
        stats = {}
        for group, start, size in (("action", 0, action_dim), ("state", action_dim, state_dim)):
            fields = {key: [] for key in ("min", "max", "mean", "std", "q01", "q99")}
            for dimension in range(start, start + size):
                column = np.array(values[:, dimension], dtype=np.float64)
                lo, hi = np.quantile(column, [0.01, 0.99])
                for key, value in zip(
                    fields,
                    (column.min(), column.max(), column.mean(), column.std(), lo, hi),
                    strict=True,
                ):
                    fields[key].append(float(value))
            stats[group] = {
                "default": {
                    **{f"global_{key}": value for key, value in fields.items()},
                    **{f"stepwise_{key}": [value] for key, value in fields.items()},
                }
            }
        del values
    return stats
