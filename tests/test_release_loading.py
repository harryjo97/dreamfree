"""Portable release loading, using metadata from the three published checkpoints."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from dreamfree.evaluation import checkpoints
from dreamfree.models import policy_loading, release

FIXTURES = json.loads((Path(__file__).parent / "fixtures/release_metadata.json").read_text())


@pytest.fixture(autouse=True)
def isolate_model_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DREAMFREE_MODEL_BASE_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("DREAMFREE_DOWNLOAD_SOURCE", "huggingface")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_release(tmp_path, name="LIBERO"):
    root = tmp_path / name
    for filename, value in FIXTURES[name].items():
        write_json(root / filename, value)
    weight_map = {"first": "student-1.safetensors", "second": "student-2.safetensors"}
    for key, filename in weight_map.items():
        save_file({key: torch.ones(2)}, root / "stage1" / filename)
    config = FIXTURES[name]["stage1/config.json"]
    write_json(root / "stage1" / config["weights"], {"weight_map": weight_map})
    for filename in release.POLICY_WEIGHTS:
        save_file({}, root / "stage2" / filename)
    return root


@pytest.mark.parametrize("name", FIXTURES)
def test_relocated_release_metadata_lineage_and_stats(tmp_path, name):
    from dreamfree.data.action_processing import _build_stage2_processor
    from dreamfree.data.lerobot.utils.normalizer import load_dataset_stats_from_json

    root = make_release(tmp_path, name)
    moved = tmp_path / "moved" / root.name
    moved.parent.mkdir()
    root.rename(moved)
    checkpoint = release.resolve_policy_checkpoint(moved)
    assert checkpoint == moved / "stage2"
    assert release.resolve_policy_checkpoint(checkpoint) == checkpoint
    metadata = release.read_checkpoint_metadata(checkpoint, stage="stage2")
    source = release.read_checkpoint_metadata(moved / "stage1", stage="stage1")
    assert metadata["source_stage1_checkpoint"] == str(moved / "stage1")
    assert not Path(source["stage0_lora_checkpoint"]).exists()
    module = "droid_sim.deploy_policy" if name == "DROID" else f"{name.lower()}.policy"
    policy = importlib.import_module("dreamfree.evaluation." + module)
    policy._validate_dreamfree_stage2_for_eval(checkpoint, metadata)
    checkpoints.validate_stage1_stage2_lineage(
        stage1_checkpoint_path=moved / "stage1",
        stage1_metadata=source,
        stage2_metadata=metadata,
    )
    if name == "DROID":
        assert metadata["action_history"]["horizon"] == 8
        assert metadata["action_history"]["droid_joint_anchor"] == "current_planning_state"
    else:
        common = importlib.import_module(f"dreamfree.evaluation.{name.lower()}._eval_common")
        assert common._load_checkpoint_metadata(moved) == metadata
        assert (
            common._resolve_dataset_stats_path(
                explicit_value=None,
                ckpt_path=checkpoint,
                checkpoint_metadata=metadata,
            )
            == checkpoint / "dataset_stats.json"
        )
    stats = load_dataset_stats_from_json(metadata["dataset"]["dataset_stats_path"])
    processor = _build_stage2_processor(checkpoints.stage2_config_from_metadata(metadata))
    processor.set_normalizer_from_stats(stats)


@pytest.mark.parametrize(
    "failure", ["missing_shard", "wrong_index", "missing_provenance", "lineage", "version"]
)
def test_incomplete_or_mismatched_release_fails(tmp_path, failure):
    root = make_release(tmp_path)
    if failure == "missing_shard":
        (root / "stage1/student-2.safetensors").unlink()
    elif failure == "wrong_index":
        save_file({"unexpected": torch.ones(2)}, root / "stage1/student-1.safetensors")
    elif failure == "missing_provenance":
        (root / "provenance/stage1_trainer_state.json").unlink()
    elif failure == "lineage":
        metadata = copy.deepcopy(FIXTURES["LIBERO"]["provenance/stage2_trainer_state.json"])
        metadata["source_stage1_global_step"] += 1
        write_json(root / "provenance/stage2_trainer_state.json", metadata)
    else:
        config = copy.deepcopy(FIXTURES["LIBERO"]["stage2/config.json"])
        config["format_version"] = 999
        write_json(root / "stage2/config.json", config)
    with pytest.raises((ValueError, FileNotFoundError)):
        release.read_checkpoint_metadata(root / "stage2", stage="stage2")


def test_droid_history_semantics_cannot_change(tmp_path):
    root = make_release(tmp_path, "DROID")
    for filename in ("stage2/config.json", "provenance/stage2_trainer_state.json"):
        payload = json.loads((root / filename).read_text())
        payload["action_history"]["droid_joint_anchor"] = "previous_planning_state"
        write_json(root / filename, payload)
    with pytest.raises(ValueError, match="action-history contract"):
        release.read_checkpoint_metadata(root / "stage2", stage="stage2")


def test_robocasa_release_does_not_require_stage0_provenance(tmp_path):
    root = make_release(tmp_path, "RoboCasa")
    (root / "provenance/stage0_trainer_state.json").unlink()
    release.read_checkpoint_metadata(root / "stage2", stage="stage2")


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_standalone_student_loads_shards_without_stage0_or_base_dit(tmp_path, monkeypatch, mode):
    from dreamfree.models.wan22 import stage0_loader
    from dreamfree.models.wan22.helpers.io import ModelConfig
    from dreamfree.models.wan22.wan_video_dit import WanVideoDiT

    architecture = stage0_loader.default_stage1_dit_config()
    architecture.update(
        hidden_dim=16,
        ffn_dim=32,
        freq_dim=8,
        text_dim=16,
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
    )
    original = WanVideoDiT(**architecture)
    state = original.state_dict()
    parts = [dict(list(state.items())[::2]), dict(list(state.items())[1::2])]
    weight_map = {}
    for i, part in enumerate(parts):
        name = f"student-{i}.safetensors"
        save_file(part, tmp_path / name)
        weight_map.update({key: name for key in part})
    write_json(tmp_path / "student.safetensors.index.json", {"weight_map": weight_map})
    metadata = {
        "release": {
            "weights": "student.safetensors.index.json",
            "architecture": architecture,
            "source_train_mode": mode,
        },
        "config": {
            "model_id": "Wan-AI/Wan2.2-TI2V-5B",
            "model_base_path": str(tmp_path),
            "tokenizer_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
            "tokenizer_max_len": 128,
        },
        "model_contract": {"video_model_family": "wan22_ti2v_5b", "student_dit_role": "single"},
    }
    monkeypatch.setattr(stage0_loader, "default_stage1_dit_config", lambda **kw: architecture)

    def forbidden(*args, **kwargs):
        pytest.fail("Standalone student tried to load Stage 0 or download upstream DiT weights.")

    monkeypatch.setattr(policy_loading, "_load_stage0_merged_wan_core_with_text", forbidden)
    monkeypatch.setattr(ModelConfig, "download_if_necessary", forbidden)
    core = policy_loading._load_stage1_student_core_with_text(
        stage1_checkpoint_path=tmp_path,
        metadata=metadata,
        device="cpu",
        dtype=torch.bfloat16,
        load_vae=False,
        load_text_encoder=False,
        distributed_checkpoint_broadcast=False,
    )
    assert not core.training and core.student_lora_adapter_name is None
    assert all(not p.requires_grad for p in core.parameters())
    for name, value in core.dit.state_dict().items():
        torch.testing.assert_close(value, state[name].bfloat16(), rtol=0, atol=0)
    assert all(freq.device.type == "cpu" for freq in core.dit.freqs)


@pytest.mark.parametrize("name", FIXTURES)
def test_policy_assembly_uses_local_source_and_bundled_stats(tmp_path, monkeypatch, name):
    from dreamfree.evaluation import optimized
    from dreamfree.evaluation import policy as assembly
    from dreamfree.evaluation.droid_sim import deploy_policy as droid
    from dreamfree.evaluation.robocasa import deploy_policy_optimized as robocasa

    root = make_release(tmp_path, name)
    core_calls = []

    def core_loader(**kwargs):
        core_calls.append(kwargs)
        return SimpleNamespace(device="cpu", torch_dtype=torch.bfloat16)

    class Conditioner(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def load_teacher_state_dict(self, *args, **kwargs):
            pass

        def load_proprio_state_dict(self, *args, **kwargs):
            pass

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(policy_loading, "_load_stage1_student_core_with_text", core_loader)
    monkeypatch.setattr(checkpoints, "load_readout", lambda **kw: None)
    monkeypatch.setattr(assembly.ActionDiT, "from_pretrained", lambda **kw: torch.nn.Identity())
    monkeypatch.setattr(assembly, "TeacherStateActionConditioner", Conditioner)
    monkeypatch.setattr(droid, "TeacherStateActionConditioner", Conditioner)
    monkeypatch.setattr(optimized, "OptimizedLiberoPolicy", SimpleNamespace)
    monkeypatch.setattr(optimized, "optimize_policy", lambda policy: None)
    monkeypatch.setattr(robocasa, "OptimizedNativeRoboCasaPolicy", SimpleNamespace)
    monkeypatch.setattr(robocasa, "prepare_policy_for_evaluation", lambda policy: None)
    if name == "DROID":
        policy = droid.build_policy_from_args(
            {"checkpoint_path": str(root), "device": "cpu"},
            policy_cls=SimpleNamespace,
        )
    else:
        policy = assembly.build_policy(name.lower(), {"checkpoint_path": str(root)})
    assert core_calls[0]["stage1_checkpoint_path"] == root / "stage1"
    assert policy.dataset_stats_path == root / "stage2/dataset_stats.json"
    assert policy.stage2_checkpoint_path == root / "stage2"


@pytest.mark.parametrize("name", FIXTURES)
def test_release_supports_stage2_training_and_finetuning(tmp_path, monkeypatch, name):
    from dreamfree.config import load_config
    from dreamfree.models import student as student_module
    from dreamfree.training.stage2.checkpoint import (
        build_dreamfree_stage2_checkpoint_metadata,
        prepare_release_initialization,
        validate_release_initialization,
    )

    root = make_release(tmp_path, name)
    variant = "5b" if name == "LIBERO" else "14b"
    cfg = load_config(
        2, Path(__file__).parents[1] / f"configs/stage2_{name.lower()}_{variant}.json"
    )
    cfg.init_checkpoint = str(root)
    cfg.dataset_dirs = str(tmp_path / "data")
    initialized = prepare_release_initialization(cfg)
    source = student_module.validate_stage1_student_source(cfg)
    validate_release_initialization(cfg, initialized[1], source=source)
    assert cfg.skip_dit_load_from_pretrain
    assert cfg.stage1_checkpoint == str(root / "stage1")
    assert cfg.dataset_stats_path == str(root / "stage2/dataset_stats.json")
    calls = []

    def load(path, metadata, **kwargs):
        calls.append(kwargs)
        return torch.nn.Linear(4, 4)

    def forbidden(*args, **kwargs):
        pytest.fail("Released encoder must not load Stage 0 or LoRA adapters.")

    monkeypatch.setattr(release, "load_release_student", load)
    monkeypatch.setattr(student_module, "load_stage0_merged_backbone", forbidden)
    monkeypatch.setattr(student_module, "load_stage0_merged_wan_core", forbidden)
    frozen = student_module.load_frozen_stage1_student(
        cfg, source=source, device="cpu", dtype=torch.bfloat16
    )
    assert not frozen.core.training
    assert all(not p.requires_grad for p in frozen.core.parameters())
    assert calls[0]["load_vae"] and not calls[0]["load_text_encoder"]
    assert frozen.stage0_checkpoint_dir is None
    assert not frozen.stage0_lora_config["enabled"]
    shape = dict(initialized[1]["shapes"], action_horizon=32)
    wrapper = SimpleNamespace(
        student=frozen,
        source_stage1_metadata=source.metadata,
        source_stage1_checkpoint=str(source.checkpoint_dir),
        last_shape_metadata=shape,
    )
    saved = build_dreamfree_stage2_checkpoint_metadata(cfg=cfg, model=wrapper, global_step=1)
    assert saved["stage0_lora_checkpoint"] is None
    checkpoints.validate_stage1_stage2_lineage(
        stage1_checkpoint_path=source.checkpoint_dir,
        stage1_metadata=source.metadata,
        stage2_metadata=saved,
    )
    cfg.resume = "checkpoint"
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_release_initialization(cfg)
    cfg.resume = None
    cfg.readout_num_queries += 1
    with pytest.raises(ValueError, match="readout.num_queries"):
        validate_release_initialization(cfg, initialized[1], source=source)
