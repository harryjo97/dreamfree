#!/usr/bin/env python3
"""Create a separate EEF dataset and small action caches; source files are read-only."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors import safe_open
from safetensors.numpy import save_file

from dreamfree.data import yam
from dreamfree.data.yam_eef import EEF_NAMES, YamWristFK, sha256_file
from dreamfree.data.yam_prepare import compute_stats, data_path


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def vectors(table: pa.Table, key: str, dim: int) -> np.ndarray:
    column = table[key].combine_chunks()
    if (
        column.null_count
        or not pa.compute.all(pa.compute.equal(pa.compute.list_value_length(column), dim)).as_py()
    ):
        raise ValueError(f"Invalid {key}: expected non-null {dim}D vectors.")
    flat = pa.compute.list_flatten(column)
    if flat.null_count:
        raise ValueError(f"Null values in {key}.")
    result = np.asarray(flat, dtype=np.float32).reshape(-1, dim)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite values in {key}.")
    return result


def normalize_action(values: np.ndarray, stats: dict) -> np.ndarray:
    """Match SingleFieldLinearNormalizer's float32 q01/q99 arithmetic and clipping."""
    lo, hi = (np.asarray(stats[f"global_{key}"], dtype=np.float32) for key in ("q01", "q99"))
    span = hi - lo
    constant = span < 1e-4
    span[constant] = 2.0
    scale = np.float32(2.0) / span
    offset = np.float32(-1.0) - scale * lo
    offset[constant] = -lo[constant]
    return np.clip(values * scale + offset, -1, 1).astype(np.float32)


def prepare(
    source: Path,
    output: Path,
    source_cache: Path,
    *,
    scratch: Path,
    workers: int = 4,
    resume: bool = False,
    model_path: Path | None = None,
) -> dict:
    source, output, source_cache = source.resolve(), output.resolve(), source_cache.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(
            "EEF output must be separate from, and not an ancestor of, the source dataset."
        )
    if source_cache == output or source_cache in output.parents or output in source_cache.parents:
        raise ValueError("EEF output and original latent cache must be disjoint.")
    selection = yam.load_selection(source, yam.ABC_NAME)
    yam.validate_stats(source, source / "dataset_stats.json", yam.ABC_NAME)
    info = json.loads((source / "meta/info.json").read_text())
    for key in ("action", "observation.state"):
        if (
            info["features"][key]["shape"] != [14]
            or info["features"][key]["names"] != yam.JOINT_NAMES
        ):
            raise ValueError(f"Unexpected source {key} ordering.")
    fk = YamWristFK(model_path) if model_path is not None else YamWristFK()
    metadata_paths = sorted((source / "meta/episodes").rglob("*.parquet"))
    rows = pa.concat_tables([pq.read_table(path) for path in metadata_paths]).to_pylist()
    if len(rows) != len(selection["episodes"]) or {str(r["episode_index"]) for r in rows} != set(
        selection["episodes"]
    ):
        raise ValueError("Source episode metadata does not match the frozen selection.")
    source_paths = [data_path(source, info, row) for row in rows]
    if len(set(source_paths)) != len(rows):
        raise ValueError("ABC EEF conversion requires one source parquet per episode.")
    immutable_paths = sorted(
        set(
            source_paths
            + list((source / "meta").rglob("*.json"))
            + list((source / "meta").rglob("*.parquet"))
            + list(source.glob("*.json"))
        )
    )
    cache_paths = {
        int(row["episode_index"]): source_cache
        / f"episode_{int(row['episode_index']):06d}.safetensors"
        for row in rows
    }
    for path in cache_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing original video cache: {path}")
    plan = {
        "source_root": str(source),
        "source_cache": str(source_cache),
        "source_selection_sha256": selection["selection_sha256"],
        "kinematics_sha256": yam.fingerprint(fk.manifest),
        "contract": yam.PRESETS[yam.ABC_EEF_NAME].contract,
    }
    if output.exists():
        if not resume or not (output / "eef_preparation.json").is_file():
            raise FileExistsError(
                f"Refusing to overwrite {output}; use --resume only for this conversion."
            )
        if json.loads((output / "eef_preparation.json").read_text()) != plan:
            raise ValueError("EEF --resume requires the identical source, cache, and kinematics.")
        if (output / "yam_selection.json").exists():
            raise FileExistsError(f"EEF dataset is already finalized: {output}")
    else:
        output.mkdir(parents=True)
        write_json(output / "eef_preparation.json", plan)
    print(
        f"Hashing {len(immutable_paths)} source data/metadata files before conversion", flush=True
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = dict(
            zip(map(str, immutable_paths), pool.map(sha256_file, immutable_paths), strict=True)
        )
    baseline_path = output / "source_integrity.json"
    if baseline_path.exists() and json.loads(baseline_path.read_text()) != hashes:
        raise ValueError("Original dataset changed since EEF conversion started.")
    write_json(baseline_path, hashes)
    source_cache_stats = {
        index: (path.stat().st_size, path.stat().st_mtime_ns) for index, path in cache_paths.items()
    }
    write_json(output / "kinematics.json", fk.manifest)
    (output / "meta").mkdir(exist_ok=True)
    (output / "meta/eef_conversion").mkdir(exist_ok=True)
    for path in metadata_paths + [source / "meta/tasks.parquet", source / "task_texts.json"]:
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    video_link = output / "videos"
    if not video_link.exists():
        video_link.symlink_to(source / "videos", target_is_directory=True)
    elif video_link.resolve() != source / "videos":
        raise ValueError(f"Unexpected video reference: {video_link}")
    info["features"]["action.joint"] = dict(info["features"]["action"])
    info["features"]["action"] = {"dtype": "float32", "shape": [20], "names": EEF_NAMES}
    write_json(output / "meta/info.json", info)

    def convert(row):
        index = int(row["episode_index"])
        src = data_path(source, info, row)
        dst = output / src.relative_to(source)
        marker = output / "meta/eef_conversion" / f"episode_{index:06d}.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["source_sha256"] != hashes[str(src)] or saved["output_sha256"] != sha256_file(
                dst
            ):
                raise ValueError(f"EEF resume checksum mismatch: {marker}")
            if saved["source_cache_size_mtime_ns"] != list(source_cache_stats[index]):
                raise ValueError(
                    f"Original cache changed since EEF conversion started: {cache_paths[index]}"
                )
            return saved
        table = pq.read_table(src)
        length = selection["episodes"][str(index)]["length"]
        if len(table) != length or not np.all(np.asarray(table["episode_index"]) == index):
            raise ValueError(f"Source episode identity/length mismatch: {src}")
        if not np.array_equal(
            np.asarray(table["index"]),
            np.arange(row["dataset_from_index"], row["dataset_to_index"]),
        ):
            raise ValueError(f"Source episode frame boundaries mismatch: {src}")
        action = vectors(table, "action", 14)
        vectors(table, "observation.state", 14)
        eef = fk.actions(action)
        table = table.append_column("action.joint", table["action"])
        table = table.set_column(
            table.schema.get_field_index("action"),
            "action",
            pa.FixedSizeListArray.from_arrays(pa.array(eef.ravel()), 20),
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        temporary = dst.with_suffix(".partial.parquet")
        pq.write_table(table, temporary)
        temporary.replace(dst)
        joint_values = np.stack((action[:, :6], action[:, 7:13]), axis=1)
        outside = (joint_values < fk.limits[:, 0]) | (joint_values > fk.limits[:, 1])
        saved = {
            "episode_index": index,
            "length": length,
            "source_sha256": hashes[str(src)],
            "output_sha256": sha256_file(dst),
            "outside_model_limits": outside.sum(axis=0).tolist(),
            "source_cache_size_mtime_ns": list(source_cache_stats[index]),
        }
        write_json(marker, saved)
        return saved

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(convert, rows):
            records.append(record)
            if len(records) % 250 == 0:
                print(f"Converted {len(records)}/{len(rows)} episodes", flush=True)
    print("Computing exact training-only EEF normalization statistics", flush=True)
    stats = compute_stats(output, info, rows, selection["episodes"], scratch)
    original_stats = json.loads((source / "dataset_stats.json").read_text())
    if stats["state"] != original_stats["state"]:
        raise ValueError("Joint proprioception statistics changed during EEF conversion.")
    write_json(output / "dataset_stats.json", stats)
    derived = {
        **selection,
        "contract": plan["contract"],
        "normalization_sha256": yam.fingerprint(stats),
        "eef_conversion": plan,
    }
    derived.pop("selection_sha256", None)
    derived["selection_sha256"] = yam.fingerprint(derived)
    annotations = source / "abc130k_subtasks.json"
    if annotations.exists():
        from dreamfree.data.abc130k_subtasks import SubtaskAnnotations

        content = json.loads(annotations.read_text())
        SubtaskAnnotations(content, selection)
        content["selection_sha256"] = derived["selection_sha256"]
        content.pop("annotations_sha256")
        content["annotations_sha256"] = yam.fingerprint(content)
        SubtaskAnnotations(content, derived)
        write_json(output / annotations.name, content)
    cache_output = output / "action_cache"
    cache_output.mkdir(exist_ok=True)

    def overlay(row):
        index = int(row["episode_index"])
        src = cache_paths[index]
        meta = yam.validate_cache(src, source, yam.ABC_NAME)
        table = pq.read_table(data_path(output, info, row), columns=["action"])
        normalized = normalize_action(vectors(table, "action", 20), stats["action"]["default"])
        with safe_open(str(src), framework="np") as handle:
            proprio = handle.get_tensor("proprio")
            tasks = handle.get_tensor("task_indices")
        if not np.all(tasks == index) or not np.isfinite(proprio).all():
            raise ValueError(f"Invalid source task/proprio tensors: {src}")
        metadata = {
            **meta,
            **plan["contract"],
            "dataset_kind": yam.ABC_EEF_NAME,
            "selection_sha256": derived["selection_sha256"],
            "normalization_sha256": derived["normalization_sha256"],
            "normalizer_stats_path": str(output / "dataset_stats.json"),
            "action_dim": "20",
            "overlay_cache_kind": yam.EEF_OVERLAY_KIND,
            "latent_source_path": str(src),
            "latent_source_size": str(source_cache_stats[index][0]),
            "latent_source_mtime_ns": str(source_cache_stats[index][1]),
            "latent_source_metadata_sha256": yam.fingerprint(meta),
            "source_selection_sha256": selection["selection_sha256"],
            "kinematics_sha256": plan["kinematics_sha256"],
        }
        dst = cache_output / src.name
        temporary = dst.with_suffix(".partial.safetensors")
        save_file(
            {"action": normalized, "proprio": proprio, "task_indices": tasks},
            str(temporary),
            metadata=metadata,
        )
        temporary.replace(dst)

    print("Writing EEF action overlays referencing the existing video latents", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, _ in enumerate(pool.map(overlay, rows), 1):
            if count % 500 == 0:
                print(f"Wrote {count}/{len(rows)} action overlays", flush=True)
    print("Verifying original data/metadata hashes and video cache file identities", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        after = dict(
            zip(map(str, immutable_paths), pool.map(sha256_file, immutable_paths), strict=True)
        )
    if after != hashes:
        raise ValueError("Original dataset changed during conversion; refusing to finalize.")
    for index, path in cache_paths.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != source_cache_stats[index]:
            raise ValueError(f"Original cache changed during conversion: {path}")
    report = {
        "episodes": len(rows),
        "frames": selection["total_frames"],
        "splits": {
            split: {
                "episodes": sum(r["split"] == split for r in selection["episodes"].values()),
                "frames": sum(
                    r["length"] for r in selection["episodes"].values() if r["split"] == split
                ),
            }
            for split in ("train", "val")
        },
        "source_sha256_files_verified": len(hashes),
        "source_cache_identities_verified": len(cache_paths),
        "outside_model_joint_limits_per_arm": np.sum(
            [r["outside_model_limits"] for r in records], axis=0
        ).tolist(),
        "joint_values_clipped": False,
        "action_dim": 20,
        "proprio_dim": 14,
        "selection_sha256": derived["selection_sha256"],
    }
    write_json(output / "eef_conversion_report.json", report)
    # The selection is the completion marker; training rejects unfinished outputs.
    write_json(output / "yam_selection.json", derived)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(yam.paths(yam.ABC_NAME)["root"]))
    parser.add_argument(
        "--output-root", type=Path, default=Path(yam.paths(yam.ABC_EEF_NAME)["root"])
    )
    parser.add_argument("--source-cache", type=Path, default=Path(yam.paths(yam.ABC_NAME)["cache"]))
    parser.add_argument("--scratch-dir", type=Path, default=Path("/data-vol2/jaehyeong/tmp"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Pinned ABC i2rt_yam/yam.xml"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    prepare(
        args.source_root,
        args.output_root,
        args.source_cache,
        scratch=args.scratch_dir,
        workers=args.workers,
        resume=args.resume,
        model_path=args.model_path,
    )
