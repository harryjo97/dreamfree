"""ABC EEF conversion and loader regression tests, ported from AWAM."""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from dreamfree.data import yam
from dreamfree.data.prepare_abc_eef import normalize_action, prepare, vectors, write_json
from dreamfree.data.yam_eef import MODEL_SHA256, YamWristFK, sha256_file
from dreamfree.data.yam_prepare import compute_stats

ABC_ROOT = Path(os.environ.get("ABC_ROOT", Path(__file__).resolve().parents[1] / "third_party/abc"))
MODEL_PATH = ABC_ROOT / "abc_sim/models/assets/i2rt_yam/yam.xml"
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_file(),
    reason="Set ABC_ROOT to the pinned ABC checkout for FK/conversion integration tests.",
)


def test_fk_matches_independent_mujoco_including_limits_and_unclipped_inputs():
    reference = json.loads(
        (Path(__file__).parent / "fixtures/yam_wrist_fk_reference.json").read_text()
    )
    assert reference["model_sha256"] == MODEL_SHA256
    fk = YamWristFK(MODEL_PATH)
    poses = fk.poses(np.asarray(reference["joints"]))
    np.testing.assert_allclose(poses, reference["poses"], atol=2e-14)
    np.testing.assert_allclose(
        poses[..., :3, :3] @ poses[..., :3, :3].swapaxes(-1, -2),
        np.broadcast_to(np.eye(3), poses[..., :3, :3].shape),
        atol=2e-14,
    )
    action = np.concatenate(
        (reference["joints"], np.full((20, 1), 0.3), reference["joints"], np.full((20, 1), 0.7)),
        axis=1,
    )
    eef = fk.actions(action)
    np.testing.assert_array_equal(eef[:, 9], np.float32(0.3))
    np.testing.assert_array_equal(eef[:, 19], np.float32(0.7))
    np.testing.assert_allclose(eef[:, 3:6], poses[:, :3, 0], atol=1e-7)
    np.testing.assert_allclose(eef[:, 6:9], poses[:, :3, 1], atol=1e-7)
    with pytest.raises(ValueError, match="finite"):
        fk.actions(np.full((1, 14), np.nan))


@pytest.fixture
def converted(tmp_path):
    source, cache, output = (tmp_path / name for name in ("source", "video-cache", "eef"))
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "videos").mkdir()
    cache.mkdir()
    info = {
        "codebase_version": "v3.0",
        "robot_type": "yam_bimanual",
        "fps": 30,
        "total_episodes": 3,
        "total_frames": 108,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            key: {"dtype": "float32", "shape": [14], "names": yam.JOINT_NAMES}
            for key in ("action", "observation.state")
        },
    }
    rows, episodes = [], {}
    rng = np.random.default_rng(3)
    for i in range(3):
        action = rng.uniform(0.2, 0.9, size=(36, 14)).astype(np.float32)
        if i == 2:
            action[:, :6] += 2.0
        state = action + np.float32(0.02)
        table = pa.table(
            {
                "action": pa.FixedSizeListArray.from_arrays(pa.array(action.ravel()), 14),
                "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.ravel()), 14),
                "index": np.arange(i * 36, (i + 1) * 36),
                "frame_index": np.arange(36),
                "timestamp": np.arange(36) / 30,
                "episode_index": np.full(36, i),
                "task_index": np.zeros(36, dtype=np.int64),
            }
        )
        pq.write_table(table, source / f"data/chunk-000/file-{i:03d}.parquet")
        rows.append(
            {
                "episode_index": i,
                "length": 36,
                "tasks": ["move"],
                "data/chunk_index": 0,
                "data/file_index": i,
                "dataset_from_index": i * 36,
                "dataset_to_index": (i + 1) * 36,
            }
        )
        episodes[str(i)] = {
            "length": 36,
            "split": "train" if i < 2 else "val",
            "instruction": "move",
            "task_id": "0",
        }
    pq.write_table(pa.Table.from_pylist(rows), source / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task": ["move"], "task_index": [0]}), source / "meta/tasks.parquet")
    write_json(source / "meta/info.json", info)
    write_json(source / "task_texts.json", {str(i): "move" for i in range(3)})
    stats = compute_stats(source, info, rows, episodes, tmp_path / "scratch")
    write_json(source / "dataset_stats.json", stats)
    selection = {
        "schema_version": 1,
        "fps": 30,
        "total_frames": 108,
        "episodes": episodes,
        "contract": yam.PRESETS[yam.ABC_NAME].contract,
        "normalization_sha256": yam.fingerprint(stats),
    }
    selection["selection_sha256"] = yam.fingerprint(selection)
    write_json(source / "yam_selection.json", selection)
    annotation = {
        "schema_version": 1,
        "prompt_format": "task_with_current_subtask_v1",
        "selection_sha256": selection["selection_sha256"],
        "episodes": {str(i): {"length": 36, "segments": [[0, "reach"]]} for i in range(3)},
    }
    annotation["annotations_sha256"] = yam.fingerprint(annotation)
    write_json(source / "abc130k_subtasks.json", annotation)
    for i in range(3):
        table = pq.read_table(source / f"data/chunk-000/file-{i:03d}.parquet")
        meta = {
            **yam.episode_metadata(source, i, yam.ABC_NAME),
            "episode_index": str(i),
            "num_frames": "36",
            "num_samples": "36",
            "video_h": "432",
            "video_w": "384",
            "num_frames_per_window": "33",
            "action_video_freq_ratio": "4",
            "action_horizon": "32",
            "action_dim": "14",
            "proprio_dim": "14",
            "state_dim": "14",
            "norm_mode": "q01/q99",
            "cache_version": "2",
            "coverage": "full_episode",
            "padding_policy": "clamp_to_episode_end",
            "video_model_family": "wan22_i2v_a14b",
            "model_id": "Wan-AI/Wan2.2-I2V-A14B",
            "vae_family": "wan2.1",
            "vae_z_dim": "16",
            "vae_temporal_stride": "4",
            "vae_spatial_stride": "8",
            "latent_channels": "16",
            "latent_t_per_window": "3",
            "latent_h": "54",
            "latent_w": "48",
            "i2v_condition_source": "official_pseudo_video_vae",
        }
        latent = np.full((36, 16, 3, 54, 48), i, dtype=np.float16)
        save_file(
            {
                "action": normalize_action(
                    vectors(table, "action", 14), stats["action"]["default"]
                ),
                "proprio": normalize_action(
                    vectors(table, "observation.state", 14), stats["state"]["default"]
                ),
                "task_indices": np.full(36, i, dtype=np.int64),
                "latents": latent,
                "i2v_condition_latents": latent.copy(),
            },
            str(cache / f"episode_{i:06d}.safetensors"),
            metadata=meta,
        )
    before = {
        path: sha256_file(path)
        for root in (source, cache)
        for path in root.rglob("*")
        if path.is_file()
    }
    report = prepare(
        source, output, cache, scratch=tmp_path / "scratch", workers=2, model_path=MODEL_PATH
    )
    assert before == {path: sha256_file(path) for path in before}
    return source, cache, output, report


def test_conversion_preserves_originals_and_uses_train_only_stats(converted):
    source, cache, output, report = converted
    assert (report["episodes"], report["frames"]) == (3, 108)
    assert (output / "videos").resolve() == source / "videos"
    train = []
    for i in range(3):
        old = pq.read_table(source / f"data/chunk-000/file-{i:03d}.parquet")
        new = pq.read_table(output / f"data/chunk-000/file-{i:03d}.parquet")
        assert new["action.joint"] == old["action"]
        for key in old.column_names:
            if key != "action":
                assert new[key] == old[key]
        values = vectors(new, "action", 20)
        if i < 2:
            train.append(values)
        with safe_open(
            str(output / f"action_cache/episode_{i:06d}.safetensors"), framework="np"
        ) as h:
            assert "latents" not in h.keys()
            with safe_open(str(cache / f"episode_{i:06d}.safetensors"), framework="np") as original:
                np.testing.assert_array_equal(
                    h.get_tensor("proprio"), original.get_tensor("proprio")
                )
    stats = json.loads((output / "dataset_stats.json").read_text())
    np.testing.assert_allclose(
        stats["action"]["default"]["global_q99"], np.quantile(np.concatenate(train), 0.99, axis=0)
    )
    yam.validate_stats(output, output / "dataset_stats.json", yam.ABC_EEF_NAME)
    with pytest.raises(ValueError, match="separate"):
        prepare(source, source, cache, scratch=output, workers=1, model_path=MODEL_PATH)
    with pytest.raises(FileExistsError):
        prepare(source, output, cache, scratch=output, workers=1, model_path=MODEL_PATH)
