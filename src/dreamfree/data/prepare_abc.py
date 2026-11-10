#!/usr/bin/env python3
"""Prepare local native 640x480 ABC-130K MCAP data at 30 Hz; makes no HF/network requests."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dreamfree.data import yam
from dreamfree.data.abc130k_selection import load_plan, recording_group
from dreamfree.data.yam_prepare import choose_episodes, compute_stats

FPS = 30
CAMERAS = dict(zip(yam.CAMERA_KEYS, ("/top-camera", "/left-wrist-camera", "/right-wrist-camera")))
STATE_TOPICS = [
    (f"/{side}-{part}-state", dim)
    for side in ("left", "right")
    for part, dim in (("arm", 6), ("ee", 1))
]
ACTION_TOPICS = [(topic.replace("-state", "-action"), dim) for topic, dim in STATE_TOPICS]
PRESET = yam.PRESETS[yam.ABC_NAME]


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resumed_plan(saved: dict, requested: dict, *, completed: bool) -> dict:
    """Allow an explicit higher gap limit for unfinished, otherwise identical preparation."""
    if saved.get("plan_sha256") != yam.fingerprint(
        {k: v for k, v in saved.items() if k != "plan_sha256"}
    ):
        raise ValueError("ABC saved preparation plan fingerprint is invalid.")
    ignored = {"plan_sha256", "max_gap_ms", "resumed_gap_limits_ms"}
    if {k: v for k, v in saved.items() if k not in ignored} != {
        k: v for k, v in requested.items() if k not in ignored
    }:
        raise ValueError(
            "ABC --resume requires the same source files, selection, and conversion settings."
        )
    if requested["max_gap_ms"] == saved["max_gap_ms"]:
        return saved
    if completed or requested["max_gap_ms"] < saved["max_gap_ms"]:
        raise ValueError(
            "ABC --resume permits only increasing --max-gap-ms on an unfinished preparation; "
            "use a separate output directory for other gap-policy changes."
        )
    history = {**saved.get("resumed_gap_limits_ms", {}), saved["plan_sha256"]: saved["max_gap_ms"]}
    result = {**requested, "resumed_gap_limits_ms": history}
    result["plan_sha256"] = yam.fingerprint({k: v for k, v in result.items() if k != "plan_sha256"})
    print(
        f"Explicitly increasing max sample age from {saved['max_gap_ms']:g} to "
        f"{requested['max_gap_ms']:g} ms; completed conversions are reused after validation.",
        flush=True,
    )
    return result


def mcap_reader(stream, *, decode: bool = False):
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:
        raise RuntimeError(
            "ABC-130K preparation requires the abc130k extra: uv sync --extra abc130k"
        ) from exc
    return make_reader(stream, decoder_factories=[DecoderFactory()] if decode else [])


def scan_source(source: Path, selection: dict | None = None) -> tuple[list[dict], dict]:
    train = source / "data/train"
    files = (
        sorted(source / row["source_mcap"] for row in selection["episodes"])
        if selection
        else sorted(train.rglob("episode.mcap"))
    )
    expected = {row["source_episode_id"]: row for row in selection["episodes"]} if selection else {}
    if not files:
        raise FileNotFoundError(
            f"Expected original XDOF/ABC-130k files at {train}/<task>/episode_*/episode.mcap"
        )
    rows, rejected, seen = [], Counter(), set()
    required = set(CAMERAS.values()) | {topic for topic, _ in STATE_TOPICS + ACTION_TOPICS}
    for position, path in enumerate(files):
        with path.open("rb") as stream:
            reader = mcap_reader(stream)
            summary = reader.get_summary()
            if summary is None or summary.statistics is None:
                raise ValueError(f"MCAP summary/statistics missing: {path}")
            records = [
                item.metadata for item in reader.iter_metadata() if item.name == "episode-metadata"
            ]
            if len(records) != 1:
                raise ValueError(f"Expected one ABC release episode-metadata record: {path}")
            md = records[0]
            try:
                dimensions = [
                    (int(md[f"{camera}_camera_width"]), int(md[f"{camera}_camera_height"]))
                    for camera in ("top", "left", "right")
                ]
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Missing/invalid ABC camera dimensions: {path}") from exc
            if dimensions != [(640, 480)] * 3:
                rejected["other_camera_resolution"] += 1
                continue
            topics = {channel.topic for channel in summary.channels.values()}
            if topics & {"/top-left-camera", "/top-right-camera"}:
                rejected["stereo_station"] += 1
                continue
            if required - topics:
                raise ValueError(
                    f"ABC episode is missing required streams {sorted(required - topics)}: {path}"
                )
            instruction = " ".join(str(md.get("task_name", "")).split())
            episode_id = str(md.get("session_id", ""))
            if not instruction or not episode_id or path.parent.name != f"episode_{episode_id}":
                raise ValueError(f"Missing/inconsistent ABC task or session_id metadata: {path}")
            if episode_id in seen:
                raise ValueError(f"Duplicate ABC source episode {episode_id}: {path}")
            seen.add(episode_id)
            stats = summary.statistics
            # This bounds the common-overlap duration before decoding selected episodes.
            span_ns = int(stats.message_end_time) - int(stats.message_start_time)
            length = span_ns * FPS // 1_000_000_000 + 1
            if length < 33:
                rejected["short_episode"] += 1
                continue
            stat = path.stat()
            row = {
                "episode_index": len(rows),
                "length": length,
                "task_id": instruction,
                "instruction": instruction,
                "source_episode_id": episode_id,
                "source_mcap": str(path.relative_to(source)),
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
            }
            if selection:
                original = expected[episode_id]
                group = recording_group(
                    str(md.get("operator_id", "")), int(stats.message_start_time)
                )
                for key in ("length", "instruction", "source_mcap", "source_size"):
                    if row[key] != original[key]:
                        raise ValueError(
                            f"Downloaded ABC {key} differs from frozen selection: {path}"
                        )
                if group != original["group"]:
                    raise ValueError(
                        f"Downloaded ABC operator/day group differs from frozen selection: {path}"
                    )
                row.update(
                    task_id=original["task_id"],
                    source_group=group,
                    category=original["category"],
                    selection_phase=original["selection_phase"],
                    split=original["split"],
                )
            rows.append(row)
        if (position + 1) % 1000 == 0:
            print(
                f"Scanned {position + 1}/{len(files)} MCAP files; eligible={len(rows)}", flush=True
            )
    if not rows:
        raise ValueError("No eligible original ABC episodes with three native 640x480 cameras.")
    if selection and (rejected or len(rows) != len(expected)):
        raise ValueError(
            f"Frozen ABC selection lost episodes during scanning: selected={len(expected)}, eligible={len(rows)}, rejected={dict(rejected)}"
        )
    return rows, {
        "source_files": len(files),
        "eligible_episodes": len(rows),
        "eligible_tasks": len({row["task_id"] for row in rows}),
        "excluded": dict(rejected),
    }


def scan_source_union(sources: list[Path]) -> tuple[list[dict], dict]:
    """Assign unique prepared indices while retaining each episode's raw root."""
    sources = [root.resolve() for root in sources]
    if len(set(sources)) != len(sources):
        raise ValueError("ABC union source roots must be distinct.")
    rows, reports, seen, excluded = [], {}, set(), Counter()
    for root in sources:
        if (root / "abc130k_download_selection.json").exists():
            raise ValueError(
                "ABC union cannot re-split a frozen mixed train/val selection; prepare it separately."
            )
        source_rows, report = scan_source(root)
        reports[str(root)] = report
        excluded.update(report["excluded"])
        for row in source_rows:
            if row["source_episode_id"] in seen:
                raise ValueError(
                    f"Duplicate episode across ABC union roots: {row['source_episode_id']}"
                )
            seen.add(row["source_episode_id"])
            rows.append({**row, "episode_index": len(rows), "source_root": str(root)})
    return rows, {
        "source_files": sum(report["source_files"] for report in reports.values()),
        "eligible_episodes": len(rows),
        "eligible_tasks": len({row["task_id"] for row in rows}),
        "excluded": dict(excluded),
        "sources": reports,
    }


def causal_indices(
    timestamps: np.ndarray, ticks: np.ndarray, *, topic: str, max_gap_ms: float
) -> tuple[np.ndarray, float]:
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            f"ABC {topic}: timestamps must be strictly increasing with at least two samples."
        )
    indices = np.searchsorted(timestamps, ticks, side="right") - 1
    if np.any(indices < 0) or ticks[-1] > timestamps[-1]:
        raise ValueError(f"ABC {topic}: target clock extends outside the source stream.")
    age_ns = ticks - timestamps[indices]
    worst_ms = float(age_ns.max()) / 1_000_000
    if worst_ms > max_gap_ms:
        worst = int(np.argmax(age_ns))
        at_seconds = (ticks[worst] - ticks[0]) / 1e9
        raise ValueError(
            f"ABC {topic}: stale sample age {worst_ms:.3f} ms exceeds --max-gap-ms={max_gap_ms} "
            f"at output frame {worst} ({at_seconds:.3f}s)."
        )
    return indices, worst_ms


def encode_camera(
    raw: Path, codec: str, output: Path, indices: np.ndarray, source_count: int
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    decoded_count = written = 0
    with (
        av.open(str(raw), format="h264" if codec == "h264" else "hevc") as source,
        av.open(str(output), "w") as dest,
    ):
        source.streams.video[0].codec_context.thread_count = 1
        encoder = dest.add_stream("libx264", rate=FPS)
        encoder.width, encoder.height, encoder.pix_fmt = 640, 480, "yuv420p"
        encoder.options = {"crf": "18", "preset": "fast", "bf": "0", "g": "30", "threads": "1"}
        for frame in source.decode(video=0):
            if (frame.width, frame.height) != (640, 480):
                raise ValueError(
                    f"ABC decoded camera must be 640x480, got {frame.width}x{frame.height}: {raw}"
                )
            if source.streams.video[0].codec_context.has_b_frames:
                raise ValueError(
                    f"ABC video has reordered B frames; cannot assume MCAP message/frame alignment: {raw}"
                )
            while written < len(indices) and int(indices[written]) == decoded_count:
                frame.pts, frame.time_base = written, Fraction(1, FPS)
                for packet in encoder.encode(frame):
                    dest.mux(packet)
                written += 1
            decoded_count += 1
        for packet in encoder.encode():
            dest.mux(packet)
    if decoded_count != source_count or written != len(indices):
        raise ValueError(
            f"ABC video/message count mismatch: decoded={decoded_count}, messages={source_count}, "
            f"written={written}, expected={len(indices)}: {raw}"
        )
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        if stream.frames != written or stream.average_rate != FPS:
            raise ValueError(f"ABC encoded video frame count/rate mismatch: {output}")


def convert_episode(job: dict) -> dict:
    row, output = job["row"], Path(job["output"])
    source_path = Path(job["source"]) / row["source_mcap"]
    marker = output / "meta/conversion" / f"episode_{row['episode_index']:06d}.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        allowed_plans = {job["plan_sha256"], *job.get("resumed_gap_limits_ms", {})}
        if saved["plan_sha256"] not in allowed_plans:
            raise ValueError(f"ABC conversion belongs to another preparation plan: {marker}")
        age = saved["episode"].get("worst_sample_age_ms", float("nan"))
        original_limit = job.get("resumed_gap_limits_ms", {}).get(
            saved["plan_sha256"], job["max_gap_ms"]
        )
        if not math.isfinite(age) or not 0 <= age <= min(original_limit, job["max_gap_ms"]):
            raise ValueError(
                f"ABC completed conversion violates its recorded sample-age limit: {marker}"
            )
        if (
            saved["episode"]["episode_index"] != row["episode_index"]
            or saved["episode"]["source_mcap"] != row["source_mcap"]
            or saved["episode"].get("source_root") != row.get("source_root")
        ):
            raise ValueError(f"ABC completed conversion has a different episode identity: {marker}")
        for name, size in saved["video_sizes"].items():
            if not (output / name).is_file() or (output / name).stat().st_size != size:
                raise ValueError(f"ABC completed video is missing or changed: {output / name}")
        if (
            pq.ParquetFile(output / saved["parquet"]).metadata.num_rows
            != saved["episode"]["length"]
        ):
            raise ValueError(f"ABC completed parquet has changed: {saved['parquet']}")
        return saved["episode"]
    stat = source_path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (row["source_size"], row["source_mtime_ns"]):
        raise ValueError(f"ABC source changed after selection: {source_path}")
    topic_dimensions = dict(STATE_TOPICS + ACTION_TOPICS)
    timestamps = {topic: [] for topic in (*CAMERAS.values(), *topic_dimensions)}
    values = {topic: [] for topic in topic_dimensions}
    codecs = {}
    scratch = Path(job["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix="abc130k-video-", dir=scratch) as work,
        ExitStack() as stack,
    ):
        raw_paths = {
            topic: Path(work) / f"camera_{i}.bin" for i, topic in enumerate(CAMERAS.values())
        }
        raw_files = {
            topic: stack.enter_context(path.open("wb")) for topic, path in raw_paths.items()
        }
        with source_path.open("rb") as stream:
            reader = mcap_reader(stream, decode=True)
            for _, channel, message, decoded in reader.iter_decoded_messages(
                topics=list(timestamps)
            ):
                topic = channel.topic
                timestamps[topic].append(int(message.log_time))
                if topic in raw_files:
                    codec = str(decoded.format)
                    if codec not in {"h264", "h265"} or codecs.setdefault(topic, codec) != codec:
                        raise ValueError(
                            f"Unsupported/changing ABC video codec {codec!r}: {source_path}, {topic}"
                        )
                    raw_files[topic].write(decoded.data)
                else:
                    position = np.asarray(decoded.position, dtype=np.float32)
                    if (
                        position.shape != (topic_dimensions[topic],)
                        or not np.isfinite(position).all()
                    ):
                        raise ValueError(
                            f"Invalid ABC joint/gripper positions: {source_path}, {topic}"
                        )
                    values[topic].append(position)
        for raw_file in raw_files.values():
            raw_file.flush()
        for topic, times in timestamps.items():
            if len(times) < 2:
                raise ValueError(f"ABC stream has fewer than two samples: {source_path}, {topic}")
        start = max(times[0] for times in timestamps.values())
        end = min(times[-1] for times in timestamps.values())
        length = (end - start) * FPS // 1_000_000_000 + 1
        if length < 33 or length > row["length"]:
            raise ValueError(
                f"ABC common overlap has invalid length {length}, selected upper bound {row['length']}: {source_path}"
            )
        ticks = start + np.arange(length, dtype=np.int64) * 1_000_000_000 // FPS
        indices, worst_gap = {}, {}
        for topic, times in timestamps.items():
            try:
                indices[topic], worst_gap[topic] = causal_indices(
                    np.asarray(times), ticks, topic=topic, max_gap_ms=job["max_gap_ms"]
                )
            except ValueError as exc:
                raise ValueError(f"{exc} Episode {row['episode_index']}: {source_path}") from exc
        action = np.concatenate(
            [np.asarray(values[topic])[indices[topic]] for topic, _ in ACTION_TOPICS], axis=1
        )
        state = np.concatenate(
            [np.asarray(values[topic])[indices[topic]] for topic, _ in STATE_TOPICS], axis=1
        )
        index = int(row["episode_index"])
        chunk, file_index = divmod(index, 1000)
        episode = {
            "episode_index": index,
            "length": length,
            "tasks": [row["instruction"]],
            "data/chunk_index": chunk,
            "data/file_index": file_index,
            "source_episode_id": row["source_episode_id"],
            "source_mcap": row["source_mcap"],
            "source_start_ns": start,
            "worst_sample_age_ms": max(worst_gap.values()),
        }

        def convert_camera(item):
            key, topic = item
            relative = f"videos/{key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
            encode_camera(
                raw_paths[topic],
                codecs[topic],
                output / relative,
                indices[topic],
                len(timestamps[topic]),
            )
            return key, relative

        video_sizes = {}
        with ThreadPoolExecutor(max_workers=job["camera_workers"]) as pool:
            camera_results = list(pool.map(convert_camera, CAMERAS.items()))
        for key, relative in camera_results:
            video_sizes[relative] = (output / relative).stat().st_size
            episode.update(
                {
                    f"videos/{key}/chunk_index": chunk,
                    f"videos/{key}/file_index": file_index,
                    f"videos/{key}/from_timestamp": 0.0,
                    f"videos/{key}/to_timestamp": length / FPS,
                }
            )
    if "source_root" in row:
        episode["source_root"] = row["source_root"]
    parquet = f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
    (output / parquet).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "action": pa.FixedSizeListArray.from_arrays(pa.array(action.ravel()), 14),
                "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.ravel()), 14),
                "episode_index": np.full(length, index, dtype=np.int64),
                "frame_index": np.arange(length, dtype=np.int64),
                "index": np.arange(length, dtype=np.int64),
                "timestamp": np.arange(length, dtype=np.float64) / FPS,
                "task_index": np.full(length, job["task_index"], dtype=np.int64),
            }
        ),
        output / parquet,
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        marker,
        {
            "plan_sha256": job["plan_sha256"],
            "episode": episode,
            "parquet": parquet,
            "video_sizes": video_sizes,
        },
    )
    return episode


def dataset_info(episodes: list[dict], tasks: list[str]) -> dict:
    return {
        "codebase_version": "v3.0",
        "robot_type": "yam_bimanual",
        "fps": FPS,
        "chunks_size": 1000,
        "total_episodes": len(episodes),
        "total_frames": sum(row["length"] for row in episodes),
        "total_tasks": len(tasks),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            **{
                key: {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": FPS,
                        "video.height": 480,
                        "video.width": 640,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.channels": 3,
                        "has_audio": False,
                    },
                }
                for key in yam.CAMERA_KEYS
            },
            **{
                key: {"dtype": "float32", "shape": [14], "names": yam.JOINT_NAMES}
                for key in ("action", "observation.state")
            },
            **{
                key: {"dtype": "int64", "shape": [1], "names": None}
                for key in ("episode_index", "frame_index", "index", "task_index")
            },
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
        },
    }


def prepare(
    source: Path,
    output: Path,
    *,
    max_hours: float | None = None,
    seed: int = 42,
    scratch: Path = Path("/data-vol2/jaehyeong/tmp"),
    workers: int = 4,
    max_gap_ms: float = 100,
    resume: bool = False,
    selection_manifest: Path | None = None,
    camera_workers: int = 3,
    normalization_source: Path | None = None,
    additional_sources: list[Path] | None = None,
    episode_exclusions: dict[str, str] | None = None,
) -> dict:
    source, output = source.resolve(), output.resolve()
    sources = [source, *[root.resolve() for root in (additional_sources or [])]]
    inherited_stats = None
    normalization_provenance = None
    if normalization_source is not None:
        normalization_source = normalization_source.resolve()
        stats_path = normalization_source / "dataset_stats.json"
        yam.validate_stats(normalization_source, stats_path, yam.ABC_NAME)
        inherited_stats = json.loads(stats_path.read_text())
        normalization_provenance = {
            "dataset_root": str(normalization_source),
            "sha256": yam.fingerprint(inherited_stats),
        }
    selection_path = selection_manifest or source / "abc130k_download_selection.json"
    selection = (
        load_plan(selection_path)
        if selection_manifest is not None or selection_path.exists()
        else None
    )
    if selection is not None and len(sources) > 1:
        raise ValueError(
            "ABC union cannot combine additional sources with a frozen train/val selection."
        )
    if selection is not None and episode_exclusions:
        raise ValueError("Cannot exclude episodes from a frozen ABC train/val selection.")
    if selection:
        receipt = source / "abc130k_download_complete.json"
        if not receipt.exists():
            raise ValueError(
                f"Frozen ABC selection download is incomplete; run the downloader's download command: {receipt}"
            )
        completed = json.loads(receipt.read_text())
        if (
            completed.get("selection_sha256") != selection["selection_sha256"]
            or not completed.get("sha256_verified")
            or completed.get("episodes") != len(selection["episodes"])
        ):
            raise ValueError(
                f"ABC download completion marker does not match the frozen selection: {receipt}"
            )
        budget = selection["train_hours"] + selection["val_hours"]
        if max_hours is not None and max_hours != budget:
            raise ValueError(
                f"Frozen ABC selection already defines {budget} total hours; omit --max-hours."
            )
        max_hours, seed = budget, selection["seed"]
    max_hours, max_gap_ms = float(300 if max_hours is None else max_hours), float(max_gap_ms)
    if not math.isfinite(max_hours) or not 0 < max_hours <= 300:
        raise ValueError("max_hours must be in (0, 300].")
    if workers < 1 or not math.isfinite(max_gap_ms) or max_gap_ms <= 0:
        raise ValueError("workers and max_gap_ms must be positive.")
    if camera_workers not in (1, 2, 3):
        raise ValueError("camera_workers must be 1, 2, or 3.")
    if any(root == output or root in output.parents or output in root.parents for root in sources):
        raise ValueError(
            "ABC source and prepared output must be separate directories, neither nested in the other."
        )
    if output.exists() and not resume:
        raise FileExistsError(
            f"Refusing to overwrite ABC dataset: {output}; use --resume for an interrupted matching preparation."
        )
    rows, report = (
        scan_source_union(sources) if len(sources) > 1 else scan_source(source, selection)
    )
    source_index_sha256 = yam.fingerprint(rows)
    exclusions = episode_exclusions or {}
    missing = set(exclusions) - {row["source_episode_id"] for row in rows}
    if missing or any(
        not isinstance(reason, str) or not reason.strip() for reason in exclusions.values()
    ):
        raise ValueError(
            f"ABC episode exclusions require eligible episode IDs and nonempty reasons; missing={sorted(missing)}"
        )
    excluded = [
        {**row, "reason": exclusions[row["source_episode_id"]]}
        for row in rows
        if row["source_episode_id"] in exclusions
    ]
    for row in excluded:
        print(f"Excluding ABC episode {row['source_episode_id']}: {row['reason']}", flush=True)
    rows = [row for row in rows if row["source_episode_id"] not in exclusions]
    if selection:
        episodes = {
            str(row["episode_index"]): {
                key: row[key]
                for key in (
                    "length",
                    "task_id",
                    "instruction",
                    "split",
                    "source_group",
                    "category",
                    "selection_phase",
                )
            }
            for row in rows
        }
    else:
        episodes = choose_episodes(rows, max_hours=max_hours, seed=seed)
    chosen = [row for row in rows if str(row["episode_index"]) in episodes]
    plan = {
        "schema_version": 1,
        "source_root": str(source),
        "source_index_sha256": source_index_sha256,
        "contract": PRESET.contract,
        "max_hours": max_hours,
        "seed": seed,
        "max_gap_ms": max_gap_ms,
        "selected": chosen,
        "splits": episodes,
    }
    if excluded:
        plan["excluded_episodes"] = excluded
    if len(sources) > 1:
        plan["source_roots"] = [str(root) for root in sources]
    if normalization_provenance is not None:
        plan["normalization_source"] = normalization_provenance
    if selection:
        plan["download_selection_sha256"] = selection["selection_sha256"]
    plan["plan_sha256"] = yam.fingerprint(plan)
    plan_path = output / "abc130k_preparation.json"
    if output.exists():
        if not plan_path.exists():
            raise ValueError(
                f"ABC --resume requires the same source files, selection, and conversion settings: {plan_path}"
            )
        saved_plan = json.loads(plan_path.read_text())
        plan = resumed_plan(saved_plan, plan, completed=(output / "yam_selection.json").exists())
        if plan != saved_plan:
            write_json_atomic(plan_path, plan)
        if (output / "yam_selection.json").exists():
            yam.validate_stats(output, output / "dataset_stats.json", yam.ABC_NAME)
            return yam.load_selection(output, yam.ABC_NAME)
    else:
        output.mkdir(parents=True)
        write_json_atomic(plan_path, plan)
    tasks = sorted({row["instruction"] for row in chosen})
    task_indices = {task: i for i, task in enumerate(tasks)}
    jobs = [
        {
            "row": row,
            "source": row.get("source_root", str(source)),
            "output": str(output),
            "scratch": str(scratch.resolve()),
            "max_gap_ms": max_gap_ms,
            "plan_sha256": plan["plan_sha256"],
            "resumed_gap_limits_ms": plan.get("resumed_gap_limits_ms", {}),
            "camera_workers": camera_workers,
            "task_index": task_indices[row["instruction"]],
        }
        for row in chosen
    ]
    jobs.sort(key=lambda job: (-job["row"]["length"], job["row"]["episode_index"]))
    print(
        f"Selected {len(jobs)} episodes across {len(tasks)} tasks; converting original streams to 30 Hz.",
        flush=True,
    )
    converted = []
    with ExitStack() as stack:
        if workers == 1:
            results = map(convert_episode, jobs)
        else:
            # Pool's context terminates pending workers on failure/interrupt; report results as they finish.
            pool = stack.enter_context(multiprocessing.get_context("spawn").Pool(workers))
            results = pool.imap_unordered(convert_episode, jobs)
        for i, episode in enumerate(results, 1):
            converted.append(episode)
            if i == 1 or i % 25 == 0 or i == len(jobs):
                print(f"Converted {i}/{len(jobs)} episodes", flush=True)
    converted.sort(key=lambda row: row["episode_index"])
    offset = 0
    for row in converted:
        index, length = row["episode_index"], row["length"]
        parquet = (
            output
            / f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
        )
        table = pq.ParquetFile(parquet).read()
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            pa.array(np.arange(offset, offset + length)),
        )
        pq.write_table(table, parquet)
        row.update(dataset_from_index=offset, dataset_to_index=offset + length)
        episodes[str(index)].update(
            length=length,
            source_episode_id=row["source_episode_id"],
            source_mcap=row["source_mcap"],
        )
        if "source_root" in row:
            episodes[str(index)]["source_root"] = row["source_root"]
        offset += length
    if offset > int(max_hours * 3600 * FPS):
        raise ValueError(
            f"ABC converted duration exceeds the requested cap: {offset / FPS / 3600} hours"
        )
    info = dataset_info(converted, tasks)
    stats = (
        inherited_stats
        if inherited_stats is not None
        else compute_stats(output, info, converted, episodes, scratch)
    )
    manifest = {
        "schema_version": 1,
        "source_root": str(source),
        "source_info_sha256": plan["source_index_sha256"],
        "preparation_sha256": plan["plan_sha256"],
        "seed": seed,
        "max_hours": max_hours,
        "max_gap_ms": max_gap_ms,
        "fps": FPS,
        "contract": PRESET.contract,
        "normalization_sha256": yam.fingerprint(stats),
        "episodes": episodes,
        "total_frames": offset,
        "source_scan": report,
    }
    if excluded:
        manifest["excluded_episodes"] = excluded
    if len(sources) > 1:
        manifest["source_roots"] = plan["source_roots"]
    if normalization_provenance is not None:
        manifest["normalization_source"] = normalization_provenance
    if selection:
        manifest["download_selection_sha256"] = selection["selection_sha256"]
        manifest["core_tasks"] = selection["core_tasks"]
        manifest["split_hours"] = {}
        for split in ("train", "val"):
            frames = sum(row["length"] for row in episodes.values() if row["split"] == split)
            if frames > int(selection[f"{split}_hours"] * 3600 * FPS):
                raise ValueError(f"ABC converted {split} duration exceeds its frozen budget.")
            manifest["split_hours"][split] = frames / FPS / 3600
            print(
                f"Prepared {split}: {frames / FPS / 3600:.4f} hours "
                f"(selected cap {selection[f'{split}_hours']} hours; common stream overlap)",
                flush=True,
            )
    manifest["selection_sha256"] = yam.fingerprint(manifest)
    metadata = output / "meta/episodes/chunk-000"
    metadata.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted), metadata / "file-000.parquet")
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(
        output / "meta/tasks.parquet"
    )
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "task_texts.json").write_text(json.dumps(tasks, indent=2) + "\n")
    (output / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    # The final manifest is the completion marker consumed by all training/cache paths.
    (output / "yam_selection.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Original XDOF/ABC-130k snapshot root (contains data/train).",
    )
    parser.add_argument(
        "--additional-source-root",
        type=Path,
        action="append",
        default=[],
        help="Additional native ABC training root to union before selecting/splitting episodes; repeatable.",
    )
    parser.add_argument("--output-root", type=Path, default=Path(yam.paths(yam.ABC_NAME)["root"]))
    parser.add_argument(
        "--max-hours",
        type=float,
        help="Legacy selection cap (default 300); omit with a frozen download selection.",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Frozen downloader selection; automatically detected in --source-root when present.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--camera-workers",
        type=int,
        default=3,
        help="Concurrent camera encoders per episode worker (1–3, default 3); each codec uses one thread.",
    )
    parser.add_argument(
        "--max-gap-ms",
        type=float,
        default=100,
        help="Maximum age of any synchronized sample. Can explicitly increase with --resume "
        "on unfinished output, preserving validated completed episodes.",
    )
    parser.add_argument("--scratch-dir", type=Path, default=Path("/data-vol2/jaehyeong/tmp"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--normalization-source",
        type=Path,
        help="Prepared ABC dataset whose action/proprio statistics must be retained for fine-tuning.",
    )
    args = parser.parse_args()
    result = prepare(
        args.source_root,
        args.output_root,
        max_hours=args.max_hours,
        seed=args.seed,
        workers=args.workers,
        max_gap_ms=args.max_gap_ms,
        scratch=args.scratch_dir,
        resume=args.resume,
        selection_manifest=args.selection_manifest,
        camera_workers=args.camera_workers,
        normalization_source=args.normalization_source,
        additional_sources=args.additional_source_root,
    )
    print(
        f"Prepared {len(result['episodes'])} episodes, {result['total_frames'] / FPS / 3600:.3f} hours at {args.output_root}"
    )


if __name__ == "__main__":
    main()
