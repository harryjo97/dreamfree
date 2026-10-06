"""Root-level provenance manifests for unstamped direct-bilinear RoboCasa caches."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

MANIFEST_FILENAME = "robocasa_visual_preprocess_manifest.json"
WAN22_A14B_MANIFEST_SCHEMA_VERSION = 1
FIVE_B_MANIFEST_SCHEMA_VERSION = 2
BITWISE_EQUALITY_MODE = "bitwise_all_episodes"
PRECOMPUTE_MANIFEST_PROVENANCE_MODE = "precompute_manifest_provenance"
WAN22_5B = "wan22_ti2v_5b"
WAN22_A14B = "wan22_i2v_a14b"
VERIFIED_5B_DIRECT_BILINEAR_SOURCE_COMMIT = "b3e93b988752127e02e4d6e4e60184f909c95823"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def cache_header_payload(
    *, metadata: dict[str, Any], tensor_shapes: dict[str, tuple[int, ...]]
) -> dict[str, Any]:
    return {
        "metadata": {str(key): str(value) for (key, value) in sorted(metadata.items())},
        "tensor_shapes": {
            str(key): [int(v) for v in shape] for (key, shape) in sorted(tensor_shapes.items())
        },
    }


def cache_header_sha256(
    *, metadata: dict[str, Any], tensor_shapes: dict[str, tuple[int, ...]]
) -> str:
    return _sha256_json(cache_header_payload(metadata=metadata, tensor_shapes=tensor_shapes))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_candidates(path: Path) -> Iterable[Path]:
    current = path.parent
    for _ in range(3):
        yield (current / MANIFEST_FILENAME)
        if current.parent == current:
            return
        current = current.parent


@lru_cache(maxsize=16)
def _load_manifest_cached(
    manifest_path: str, file_size: int, file_mtime_ns: int, file_ctime_ns: int
) -> tuple[dict[str, Any], str]:
    del file_size, file_mtime_ns, file_ctime_ns
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    episodes = manifest.get("episodes")
    inventory_digest = _sha256_json(episodes) if isinstance(episodes, dict) else ""
    return (manifest, inventory_digest)


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], str]:
    stat = manifest_path.stat()
    return _load_manifest_cached(
        str(manifest_path), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)
    )


@lru_cache(maxsize=32)
def _file_sha256_cached(path: str, file_size: int, file_mtime_ns: int, file_ctime_ns: int) -> str:
    del file_size, file_mtime_ns, file_ctime_ns
    return _file_sha256(Path(path))


def _validate_precompute_manifests(
    *, root: Path, manifest: dict[str, Any], mismatches: list[str]
) -> None:
    rank_manifests = manifest.get("precompute_manifests")
    if not isinstance(rank_manifests, dict) or not rank_manifests:
        mismatches.append("precompute_manifests must be a non-empty object")
        return
    if manifest.get("precompute_manifest_inventory_sha256") != _sha256_json(rank_manifests):
        mismatches.append(
            "precompute_manifest_inventory_sha256 does not match precompute_manifests"
        )
    for relative, entry in rank_manifests.items():
        if not isinstance(relative, str) or not isinstance(entry, dict):
            mismatches.append(f"invalid precompute manifest entry {relative!r}")
            continue
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            mismatches.append(f"precompute manifest path escapes cache root: {relative!r}")
            continue
        if not path.is_file():
            mismatches.append(f"precompute manifest is missing: {relative!r}")
            continue
        stat = path.stat()
        if entry.get("file_size") != int(stat.st_size):
            mismatches.append(f"precompute_manifests[{relative!r}].file_size is stale")
        if entry.get("file_mtime_ns") != int(stat.st_mtime_ns):
            mismatches.append(f"precompute_manifests[{relative!r}].file_mtime_ns is stale")
        digest = _file_sha256_cached(
            str(path), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)
        )
        if entry.get("sha256") != digest:
            mismatches.append(f"precompute_manifests[{relative!r}].sha256 is stale")
        try:
            entry_count = int(entry.get("entry_count", -1))
        except (TypeError, ValueError):
            entry_count = -1
        if entry_count <= 0:
            mismatches.append(f"precompute_manifests[{relative!r}].entry_count is invalid")


def validate_robocasa_visual_cache_manifest_entry(
    *,
    path: Path,
    metadata: dict[str, Any],
    tensor_shapes: dict[str, tuple[int, ...]],
    expected: dict[str, Any],
) -> None:
    manifest_path = next((item for item in _manifest_candidates(path) if item.is_file()), None)
    if manifest_path is None:
        raise ValueError(
            f"visual_preprocess_id is missing in {path}, and no {MANIFEST_FILENAME} was found in its cache root. Regenerate the cache or run the RoboCasa visual-cache audit."
        )
    try:
        (manifest, inventory_digest) = _load_manifest(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid RoboCasa visual cache manifest: {manifest_path}") from exc
    root = manifest_path.parent
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Cache file {path} is outside manifest root {root}.") from exc
    mismatches: list[str] = []
    family = str(expected["video_model_family"])
    checks = {
        "schema_version": FIVE_B_MANIFEST_SCHEMA_VERSION
        if family == WAN22_5B
        else WAN22_A14B_MANIFEST_SCHEMA_VERSION,
        "visual_preprocess_id": expected["visual_preprocess_id"],
        "video_model_family": family,
        "video_size": [int(expected["height"]), int(expected["width"])],
    }
    if family == WAN22_5B:
        checks["verification_mode"] = PRECOMPUTE_MANIFEST_PROVENANCE_MODE
    else:
        checks["first_latent_equality"] = BITWISE_EQUALITY_MODE
    if expected.get("concat_layout") is not None:
        checks["concat_layout"] = str(expected["concat_layout"])
    for key, expected_value in checks.items():
        if manifest.get(key) != expected_value:
            mismatches.append(f"{key}: actual={manifest.get(key)!r} expected={expected_value!r}")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, dict):
        mismatches.append("episodes must be an object")
        episodes = {}
    try:
        episode_count = int(manifest.get("episode_count", -1))
    except (TypeError, ValueError):
        episode_count = -1
    if episode_count != len(episodes):
        mismatches.append(
            f"episode_count: actual={manifest.get('episode_count')!r} expected={len(episodes)}"
        )
    if manifest.get("episode_inventory_sha256") != inventory_digest:
        mismatches.append("episode_inventory_sha256 does not match episodes")
    if family == WAN22_5B:
        if manifest.get("expected_episode_count") != episode_count:
            mismatches.append("expected_episode_count must match the audited episode_count")
        source_commit = str(manifest.get("source_commit", "")).lower()
        if source_commit != VERIFIED_5B_DIRECT_BILINEAR_SOURCE_COMMIT:
            mismatches.append(
                f"source_commit: actual={source_commit!r} expected={VERIFIED_5B_DIRECT_BILINEAR_SOURCE_COMMIT!r}"
            )
        _validate_precompute_manifests(root=root, manifest=manifest, mismatches=mismatches)
    entry = episodes.get(relative)
    if not isinstance(entry, dict):
        mismatches.append(f"episodes[{relative!r}] is missing")
    else:
        stat = path.stat()
        actual_size = int(stat.st_size)
        try:
            expected_size = int(entry.get("file_size", -1))
        except (TypeError, ValueError):
            expected_size = -1
        if expected_size != actual_size:
            mismatches.append(
                f"episodes[{relative!r}].file_size: actual={actual_size} expected={entry.get('file_size')!r}"
            )
        try:
            expected_mtime_ns = int(entry.get("file_mtime_ns", -1))
        except (TypeError, ValueError):
            expected_mtime_ns = -1
        if expected_mtime_ns != int(stat.st_mtime_ns):
            mismatches.append(f"episodes[{relative!r}].file_mtime_ns is stale")
        actual_header = cache_header_sha256(metadata=metadata, tensor_shapes=tensor_shapes)
        if entry.get("header_sha256") != actual_header:
            mismatches.append(f"episodes[{relative!r}].header_sha256 is stale")
        if family == WAN22_A14B:
            digest = entry.get("first_latent_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                mismatches.append(f"episodes[{relative!r}].first_latent_sha256 is invalid")
    if mismatches:
        raise ValueError(
            f"RoboCasa visual cache manifest mismatch for {path} via {manifest_path}: "
            + "; ".join(mismatches)
        )
