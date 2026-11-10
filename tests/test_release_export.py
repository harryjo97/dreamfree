"""CPU checks for release serialization; run in the AWAM environment."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_awam_release.py"
SPEC = importlib.util.spec_from_file_location("release_export", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_full_student_conversion_and_sharded_roundtrip(tmp_path):
    source = tmp_path / "source.safetensors"
    original = {f"blocks.{i}.weight": torch.arange(36).reshape(6, 6).float() + i for i in range(3)}
    original["counter"] = torch.tensor(7)
    save_file(original, source)
    state = release.full_student(source)
    assert state["counter"].dtype == torch.int64
    assert state["blocks.0.weight"].dtype == torch.bfloat16
    output = tmp_path / "shards"
    output.mkdir()
    filename = release.save_student(state, output, max_shard_size=100)
    index = json.loads((output / filename).read_text())
    assert set(index["weight_map"]) == set(original)
    restored = {}
    for shard in set(index["weight_map"].values()):
        restored.update(load_file(output / shard))
    for name, tensor in state.items():
        torch.testing.assert_close(restored[name], tensor, rtol=0, atol=0)
    assert load_file(source)["blocks.0.weight"].dtype == torch.float32


def test_full_student_rejects_adapters(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file({"layer.lora_A": torch.ones(2, 2)}, source)
    with pytest.raises(ValueError, match="unexpectedly contains LoRA"):
        release.full_student(source)


def test_portable_config_removes_training_machine_paths():
    config = {
        "model_id": "Wan-AI/Wan2.2-I2V-A14B",
        "stage0_lora_checkpoint": "/machine/stage0",
        "dataset_dirs": "/machine/a,/machine/b",
        "cache_dirs": ["/machine/cache"],
        "student_layer_index": 31,
    }
    assert release.portable_config(config) == {
        "model_id": "Wan-AI/Wan2.2-I2V-A14B",
        "student_layer_index": 31,
    }
    assert config["stage0_lora_checkpoint"] == "/machine/stage0"


def test_full_release_has_no_stage0_or_optimizer_dependency(tmp_path):
    pytest.importorskip("fastwam")
    paths = {stage: tmp_path / "source" / stage for stage in ("stage1", "stage2")}
    for path in paths.values():
        path.mkdir(parents=True)
    save_file({"weight": torch.ones(3, 3)}, paths["stage1"] / "student_dit.safetensors")
    for name in release.POLICY_FILES:
        save_file({"weight": torch.ones(2, 2)}, paths["stage2"] / name)
    (paths["stage2"] / "optim.pt").write_bytes(b"excluded optimizer")
    metadata = {
        "stage1": {
            "global_step": 301265,
            "student": {},
            "model_contract": {},
            "stage0_lora_checkpoint": "/missing/stage0",
            "config": {
                "video_model_family": "wan22_i2v_a14b",
                "stage0_lora_checkpoint": "/missing/stage0",
            },
        },
        "stage2": {
            "global_step": 57031,
            "config": {},
            "action_history": {"horizon": 8},
            **{
                key: {}
                for key in (
                    "model_contract",
                    "student",
                    "action",
                    "readout",
                    "conditioner",
                    "shapes",
                )
            },
        },
    }
    prepared = ("DreamFree-DROID", paths, metadata, True)
    stats = {"action": {"global_min": torch.tensor([-1.0])}}
    output = tmp_path / "release"
    release.export("droid", output, prepared, stats)
    target = output / "DreamFree-DROID"
    manifest = release.read_json(target / "release.json")
    assert not list(target.rglob("optim.pt"))
    assert not (target / "stage0").exists()
    assert release.read_json(target / "stage1/config.json")["requires_stage0"] is False
    policy = release.read_json(target / "stage2/config.json")
    assert policy["stage1_checkpoint"] == "../stage1"
    assert policy["action_history"] == {"horizon": 8}
    assert release.read_json(target / "stage2/dataset_stats.json") == {
        "action": {"global_min": [-1.0]}
    }
    for relative, size in manifest["files"].items():
        assert (target / relative).stat().st_size == size
    for name in release.POLICY_FILES:
        assert (paths["stage2"] / name).read_bytes() == (target / "stage2" / name).read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        release.export("droid", output, prepared, stats)
