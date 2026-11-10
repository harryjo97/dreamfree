"""Independent full-weight student initialized from the frozen Stage 0 teacher."""

import copy
import math

import torch

from dreamfree.models.wan22.stage0_loader import (
    Stage0MergedBackbone,
    load_stage0_merged_wan_core,
)
from dreamfree.models.wan22.wan22 import Wan22Core


def build_full_stage1_optimizer(model, cfg):
    projector_lr = (
        cfg.learning_rate if cfg.projector_learning_rate is None else cfg.projector_learning_rate
    )
    if any(not math.isfinite(lr) or lr <= 0 for lr in (cfg.learning_rate, projector_lr)):
        raise ValueError("Student and projector learning rates must be finite and positive.")
    return torch.optim.AdamW(
        [
            {"params": model.core.dit.parameters(), "lr": cfg.learning_rate, "name": "student_dit"},
            {"params": model.projector.parameters(), "lr": projector_lr, "name": "projector"},
        ],
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )


def build_full_stage1_scheduler(optimizer, cfg):
    total = max(cfg.max_steps, 1)
    warmup = (
        min(int(total * cfg.warmup_ratio), total - 1)
        if cfg.warmup_steps is None
        else cfg.warmup_steps
    )
    if type(warmup) is not int or not 0 <= warmup <= total:
        raise ValueError("warmup_steps must be an integer between zero and max_steps.")
    if cfg.lr_scheduler not in ("constant", "cosine"):
        raise ValueError("Full Stage 1 lr_scheduler must be constant or cosine.")

    def multiplier(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        if cfg.lr_scheduler == "constant":
            return 1.0
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def build_full_stage1_backbone(cfg, *, device, dtype):
    cfg.load_vae = not cfg.use_cached_latents
    merged = load_stage0_merged_wan_core(cfg, device=device, dtype=dtype)
    teacher = merged.core
    teacher.requires_grad_(False)
    student_dit = copy.deepcopy(teacher.dit)
    if cfg.trim_frozen_teacher_blocks:
        blocks = teacher.dit.blocks
        if not isinstance(blocks, torch.nn.ModuleList) or not 0 <= cfg.teacher_layer_k < len(
            blocks
        ):
            raise ValueError("teacher_layer_k must address a frozen teacher block before trimming.")
        teacher.dit.blocks = torch.nn.ModuleList(list(blocks[: cfg.teacher_layer_k + 1]))
    teacher.dit.use_gradient_checkpointing = False
    student = Wan22Core(
        dit=student_dit,
        vae=teacher.vae,
        text_encoder=teacher.text_encoder,
        tokenizer=teacher.tokenizer,
        device=teacher.device,
        torch_dtype=teacher.torch_dtype,
        train_shift=cfg.train_shift,
        infer_shift=cfg.infer_shift,
        num_train_timesteps=cfg.num_train_timesteps,
        video_model_family=teacher.video_model_family,
        dit_role=teacher.dit_role,
    )
    student.dit.requires_grad_(True)
    backbone = Stage0MergedBackbone(
        core=student,
        checkpoint_dir=merged.checkpoint_dir,
        trainer_state=merged.trainer_state,
        stage0_lora_config=merged.stage0_lora_config,
        stage1_lora_config={"enabled": False},
        merged_module_names=merged.merged_module_names,
        stage0_module_names=merged.merged_module_names,
        stage1_module_names=[],
        student_init_mode=cfg.student_init_mode,
        stage0_merged_into_student_base=True,
        teacher_lora_adapter_name=None,
        student_lora_adapter_name=None,
    )
    return backbone, teacher
