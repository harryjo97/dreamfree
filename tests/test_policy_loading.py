"""Exercise policy assembly without downloading model weights or opening a simulator."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from dreamfree.data.latent_preparation import _resolve_runtime
from dreamfree.evaluation import checkpoints
from dreamfree.evaluation.droid_sim import deploy_policy as droid
from dreamfree.evaluation.droid_sim.deploy_policy_optimized import _base_args
from dreamfree.evaluation.robocasa.inference_options import FEATURES, resolve_options
from dreamfree.training.stage2.config import Stage2Config


def test_removed_evaluation_environment_fails_before_checkpoint_loading(monkeypatch):
    from dreamfree.evaluation import policy

    def unexpected_load(*args, **kwargs):
        pytest.fail("Unsupported evaluation tried to load a checkpoint.")

    monkeypatch.setattr(policy, "resolve_policy_checkpoint", unexpected_load)
    with pytest.raises(ValueError, match="Unknown evaluation environment: 'robotwin'"):
        policy.build_policy("robotwin", {"checkpoint_path": "/unused/checkpoint"})


@pytest.mark.parametrize("device, precision", [("cpu", "no"), ("cuda:2", "bf16")])
def test_droid_policy_assembly_preserves_inference_settings(
    monkeypatch, tmp_path, device, precision
):
    metadata = {
        "source_stage1_checkpoint": str(tmp_path / "stage1"),
        "conditioner": {"student_readout_input_dim": 8, "action_context_dim": 8},
        "action": {"action_horizon": 24},
    }
    config = Stage2Config()
    core_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(droid, "_resolve_checkpoint_path", lambda path: tmp_path)
    monkeypatch.setattr(droid, "_json_load", lambda path: {})
    monkeypatch.setattr(droid, "load_droid_dataset_stats", lambda path: {})
    monkeypatch.setattr(droid, "build_droid_processor", lambda **kw: object())
    monkeypatch.setattr(droid, "load_safetensors", lambda *args, **kw: {})
    monkeypatch.setattr(checkpoints, "require_checkpoint_dir", lambda path, **kw: tmp_path)
    monkeypatch.setattr(checkpoints, "require_existing_file", lambda path, **kw: path)
    monkeypatch.setattr(
        checkpoints,
        "validate_stage1_stage2_lineage",
        lambda **kw: None,
    )
    monkeypatch.setattr(checkpoints, "stage2_config_from_metadata", lambda meta: config)
    monkeypatch.setattr(checkpoints, "load_readout", lambda **kw: object())

    def load_core(**kwargs):
        core_calls.append(kwargs)
        return SimpleNamespace(device="cpu", torch_dtype=kwargs["dtype"])

    class Conditioner(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def load_teacher_state_dict(self, *args, **kwargs):
            pass

        def load_proprio_state_dict(self, *args, **kwargs):
            pass

    monkeypatch.setattr(droid, "TeacherStateActionConditioner", Conditioner)
    monkeypatch.setattr(droid.ActionDiT, "from_pretrained", lambda **kw: torch.nn.Identity())
    policy = droid.build_policy_from_args(
        {
            "checkpoint_path": str(tmp_path),
            "device": device,
            "mixed_precision": precision,
            "seed": 7,
            "num_inference_steps": 16,
        },
        stage1_core_loader=load_core,
        checkpoint_inspector=lambda path: {
            "metadata": metadata,
            "checkpoint_type": "dreamfree_stage2",
        },
        checkpoint_validator=lambda path, meta: None,
        policy_cls=SimpleNamespace,
    )
    assert core_calls[0]["device"] == device
    assert core_calls[0]["dtype"] == (torch.float32 if precision == "no" else torch.bfloat16)
    assert policy.noise_device == "cpu"
    assert policy.action_unipc_solver_type == "bh2"
    assert policy.num_inference_steps == 16
    assert policy.seed == 7
    assert policy.action_horizon == policy.replan_steps == 24


def test_optimized_droid_defaults_and_invalid_precision():
    args, fuse, benchmark = _base_args({})
    assert args["compile_dit"] is False
    assert fuse and benchmark
    with pytest.raises(ValueError, match="mixed_precision"):
        _base_args({"mixed_precision": "fp16"})
    with pytest.raises(ValueError, match="compile_dit"):
        _base_args({"compile_dit": False})


@pytest.mark.parametrize("model", ["5b", "14b"])
def test_robocasa_optimization_eligibility_and_promotion(model):
    small = model == "5b"
    metadata = {
        "checkpoint_type": "dreamfree_stage2",
        "dataset": {"video_size": [352, 352], "context_len": 128},
        "student": {
            "tokens_per_frame": 121 if small else 484,
            "hidden_dim": 3072 if small else 5120,
            "hook_point": "post_block",
            "layer_j": 15 if small else 31,
        },
        "model_contract": {"video_model_family": "wan22_ti2v_5b" if small else "wan22_i2v_a14b"},
        "config": {"num_frames": 33, "action_video_freq_ratio": 4},
        "action": {"action_dim": 7, "action_horizon": 32},
        "readout": {"num_queries": 64},
    }
    promoted = frozenset(FEATURES)
    options = resolve_options(
        metadata, {}, promoted_features=promoted, promoted_5b_features=promoted
    )
    assert options["enabled"]
    assert options["features"]["action_cuda_graph"]["enabled"]
    assert options["features"]["encoder_cuda_graph"]["enabled"]
    assert options["features"]["vae_first_frame_only"]["enabled"] is not small
    with pytest.raises(ValueError, match="10 or 16 Euler steps"):
        resolve_options(metadata, {"inference_optimization": "true", "num_inference_steps": 3})


def test_cache_preparation_uses_distributed_environment(monkeypatch):
    for name, value in {"WORLD_SIZE": "4", "RANK": "3", "LOCAL_RANK": "1"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = _resolve_runtime(argparse.Namespace(world_size=1, rank=0, local_rank=None))
    assert (runtime.world_size, runtime.rank, runtime.local_rank) == (4, 3, 1)
    assert runtime.is_distributed
