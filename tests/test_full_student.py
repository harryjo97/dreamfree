"""Full Stage 1 gradients, checkpoint resume, and frozen downstream loading on CPU."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dreamfree.config import load_config
from dreamfree.models import full_student, policy_loading, student
from dreamfree.models.distillation import DistillationModel
from dreamfree.models.wan22.future_state import FirstFrameToTeacherStateProjector
from dreamfree.models.wan22.stage0_loader import Stage0MergedWanCore
from dreamfree.models.wan22.wan22 import Wan22Core
from dreamfree.models.wan22.wan_video_dit import WanVideoDiT
from dreamfree.training.stage1 import checkpoint, distributed

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def full_model(monkeypatch, tmp_path):
    cfg = load_config(1, ROOT / "configs/stage1_droid_14b.json")
    cfg = replace(
        cfg,
        height=32,
        width=32,
        video_hidden_dim=16,
        student_layer_j=1,
        teacher_layer_k=1,
        use_cached_latents=True,
        projector_dim=16,
        projector_num_layers=1,
        projector_num_heads=2,
        compile_dit=False,
        stage0_lora_checkpoint=str(tmp_path / "stage0"),
        stage0_expected_global_step=7,
        output_dir=str(tmp_path / "output"),
        max_steps=10,
        num_epochs=None,
    )

    def merged(*args, **kwargs):
        core = Wan22Core(
            WanVideoDiT(
                hidden_dim=16,
                in_dim=36,
                out_dim=16,
                ffn_dim=32,
                text_dim=4096,
                freq_dim=8,
                eps=1e-6,
                patch_size=(1, 2, 2),
                num_heads=2,
                attn_head_dim=8,
                num_layers=3,
                has_image_input=False,
                seperated_timestep=True,
            ),
            vae=None,
            text_encoder=None,
            tokenizer=None,
            video_model_family="wan22_i2v_a14b",
            dit_role="high_noise",
        )
        return Stage0MergedWanCore(
            core,
            tmp_path / "stage0",
            {"global_step": 7},
            {
                "enabled": True,
                "rank": 128,
                "alpha": 128.0,
                "dropout": 0.0,
                "target_modules": "attention_and_ffn",
            },
            [],
        )

    monkeypatch.setattr(full_student, "load_stage0_merged_wan_core", merged)
    backbone, teacher = full_student.build_full_stage1_backbone(
        cfg, device="cpu", dtype=torch.float32
    )
    projector = FirstFrameToTeacherStateProjector(
        d_student=16,
        d_teacher=16,
        projector_dim=16,
        num_layers=1,
        num_heads=2,
        mlp_ratio=4.0,
        max_frames=3,
    )
    model = DistillationModel(backbone, projector, cfg, teacher_core=teacher)
    sample = {
        "input_latents": torch.randn(1, 16, 3, 4, 4),
        "i2v_condition_latents": torch.randn(1, 16, 3, 4, 4),
        "context": torch.randn(1, 128, 4096),
        "context_mask": torch.ones(1, 128, dtype=torch.bool),
    }
    return cfg, model, sample


def test_full_student_trains_without_changing_teacher_and_resumes(full_model, monkeypatch):
    cfg, model, sample = full_model
    assert len(model.teacher_core.dit.blocks) == 2
    assert len(model.core.dit.blocks) == 3
    teacher_before = copy.deepcopy(model.teacher_core.state_dict())
    student_before = model.core.dit.patch_embedding.weight.detach().clone()
    optimizer = full_student.build_full_stage1_optimizer(model, cfg)
    scheduler = full_student.build_full_stage1_scheduler(optimizer, cfg)
    loss, _ = model(sample)
    loss.backward()
    assert all(p.grad is None for p in model.teacher_core.parameters())
    assert model.core.dit.patch_embedding.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.projector.parameters())
    optimizer.step()
    scheduler.step()
    assert not torch.equal(student_before, model.core.dit.patch_embedding.weight)
    for name, value in model.teacher_core.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)

    accelerator = SimpleNamespace(
        is_main_process=True, wait_for_everyone=lambda: None, unwrap_model=lambda m: m
    )
    saved = checkpoint.save_stage1_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        global_step=1,
    )
    cfg.resume = str(saved)
    _, metadata = checkpoint._read_resume_metadata(cfg)
    assert metadata["student_train_mode"] == "full"
    assert not (saved / "stage1_lora.safetensors").exists()
    assert student.read_stage1_student_source_metadata(saved)["student_train_mode"] == "full"
    expected = copy.deepcopy(model.core.dit.state_dict())
    with torch.no_grad():
        for p in model.core.dit.parameters():
            p.add_(10)
    checkpoint._load_stage1_model_weights(model, saved)
    fresh_optimizer = full_student.build_full_stage1_optimizer(model, cfg)
    fresh_scheduler = full_student.build_full_stage1_scheduler(fresh_optimizer, cfg)
    checkpoint._load_stage1_optimizer_state_if_requested(
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        cfg=cfg,
    )
    assert fresh_scheduler.state_dict() == scheduler.state_dict()
    assert len(fresh_optimizer.state) == len(optimizer.state)
    for key, tensor in model.core.dit.state_dict().items():
        torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)

    stage2 = load_config(2, ROOT / "configs/stage2_droid_14b.json")
    stage2 = replace(
        stage2,
        stage1_checkpoint=str(saved),
        video_hidden_dim=16,
        student_readout_layer=1,
        height=32,
        width=32,
    )
    source = student.validate_stage1_student_source(stage2)
    monkeypatch.setattr(
        student, "load_stage0_merged_wan_core", lambda *a, **k: SimpleNamespace(core=model.core)
    )
    frozen = student.load_frozen_stage1_student(
        stage2,
        source=source,
        device="cpu",
        dtype=torch.float32,
        merge_student_lora=True,
    )
    assert frozen.stage1_lora_config == {"enabled": False}
    assert not any(p.requires_grad for p in frozen.core.parameters())

    # Deployment must consume the full tensors without injecting Stage 1 adapters.
    monkeypatch.setattr(
        policy_loading,
        "_load_stage0_merged_wan_core_with_text",
        lambda *a, **k: SimpleNamespace(core=model.core),
    )
    monkeypatch.setattr(
        policy_loading,
        "inject_lora_adapters",
        lambda *a, **k: pytest.fail("Unexpected LoRA injection"),
    )
    core = policy_loading._load_stage1_student_core_with_text(
        stage1_checkpoint_path=saved,
        metadata=metadata,
        device="cpu",
        dtype=torch.float32,
        load_text_encoder=False,
        load_vae=False,
        distributed_checkpoint_broadcast=False,
    )
    assert not any(p.requires_grad for p in core.parameters())
    for key, tensor in core.dit.state_dict().items():
        torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)

    metadata["student_train_mode"] = "lora"
    (saved / "trainer_state.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="student_train_mode"):
        student.read_stage1_student_source_metadata(saved)


def test_full_scheduler_preserves_projector_lr_and_explicit_warmup(full_model):
    cfg, model, _ = full_model
    cfg.learning_rate, cfg.projector_learning_rate = 1e-4, 2e-4
    cfg.warmup_steps, cfg.lr_scheduler = 2, "cosine"
    optimizer = full_student.build_full_stage1_optimizer(model, cfg)
    scheduler = full_student.build_full_stage1_scheduler(optimizer, cfg)
    assert scheduler.get_last_lr() == [5e-5, 1e-4]
    for _ in range(cfg.max_steps):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([1e-6, 2e-6])


def test_fsdp_checkpoint_extracts_only_complete_student_and_projector(full_model):
    _, model, _ = full_model
    accelerator = SimpleNamespace(is_main_process=True, unwrap_model=lambda m: m)
    weights, projector = distributed.portable_states(accelerator, model)
    assert set(weights) == set(model.core.dit.state_dict())
    assert set(projector) == set(model.projector.state_dict())
    model._student_state_keys += ("missing.weight",)
    with pytest.raises(ValueError, match="incomplete"):
        distributed.portable_states(accelerator, model)
