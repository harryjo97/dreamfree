from __future__ import annotations

from dataclasses import dataclass

from dreamfree.models.wan22.future_state import (
    FUTURE_TOKEN_ATTENTION_FULL,
    I2V_CONDITION_NONE,
    TEACHER_FRAME0_CLEAN_FIRST_FRAME,
    TEXT_CONDITION_SCOPE_ALL_FRAMES,
)
from dreamfree.models.wan22.stage0_loader import STAGE1_STUDENT_INIT_STAGE0_MERGED
from dreamfree.training.stage0.constants import ROBOTWIN_CAMERA_KEYS
from dreamfree.training.stage1.constants import TEACHER_TARGET_TOKEN_SCOPE_ALL


@dataclass(frozen=True)
class Stage1StudentSigmaConfig:
    sigma_policy: str
    sigma_min: float | None
    sigma_max: float | None
    source: str


@dataclass
class Stage1Config:
    """Runtime configuration for Stage 1 LeRobot teacher-state prediction.

    Use dreamfree.config.load_config to apply model/dataset/preset defaults.
    LoRA mode shares the Stage 0 merged DiT; full mode owns a separate frozen
    teacher and trainable student. Both train a projector. Missing text-cache
    entries are prepared before dataset construction. See docs/configuration.md.
    """

    model_base_path: str = "./assets/models"
    output_dir: str = "./assets/models"
    mixed_precision: str = "bf16"
    seed: int = 42
    dataset_name: str = "robotwin"
    dataset_root: str = "./assets"
    dataset_dirs: str | None = None
    dataset_mix_weights: str | None = None
    dataset_format: str = "lerobot_root"
    camera_keys: str = ",".join(ROBOTWIN_CAMERA_KEYS)
    extract_cache_dir: str = "./assets"
    text_embedding_cache_dir: str = "./assets"
    require_text_cache: bool = True
    prompt_policy: str = "task_text"
    strict_task_text: bool = True
    prepare_only: bool = False
    clip_frames: int = 9
    raw_frame_stride: int = 4
    sample_stride_frames: int = 4
    height: int = 384
    width: int = 320
    camera_layout: str = "robotwin"
    train_epoch_size: int = 200000
    val_epoch_size: int = 512
    val_fraction: float = 0.01
    video_backend: str | None = None
    use_cached_latents: bool = False
    video_latent_cache_dir: str | None = None
    stage0_lora_checkpoint: str = "./assets/models"
    stage0_eval_reference: str = "./assets/models"
    stage0_expected_global_step: int | None = None
    stage0_expected_lora_rank: int | None = None
    stage0_expected_lora_alpha: float | None = None
    stage0_expected_lora_dropout: float | None = None
    stage0_expected_lora_target_modules: str | None = None
    stage0_expected_model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    stage0_expected_video_model_family: str = "wan22_ti2v_5b"
    stage0_expected_dit_role: str = "single"
    stage0_expected_tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    stage0_expected_tokenizer_max_len: int = 128
    use_shared_teacher_student_dit: bool = True
    student_train_mode: str = "lora"
    trim_frozen_teacher_blocks: bool = False
    video_model_family: str = "wan22_ti2v_5b"
    teacher_dit_role: str = "single"
    student_dit_role: str = "single"
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    tokenizer_max_len: int = 128
    text_dim: int = 4096
    video_hidden_dim: int = 3072
    video_latent_dim: int = 48
    video_dit_input_dim: int = 48
    video_dit_output_dim: int = 48
    redirect_common_files: bool = True
    pretrained_load_broadcast: bool = True
    teacher_target_type: str = "hidden"
    teacher_target_token_scope: str = TEACHER_TARGET_TOKEN_SCOPE_ALL
    teacher_layer_k: int = 19
    teacher_hook_point: str = "post_block"
    teacher_video_attention_mask_mode: str = "bidirectional"
    teacher_i2v_condition_mode: str = I2V_CONDITION_NONE
    teacher_frame0_mode: str = TEACHER_FRAME0_CLEAN_FIRST_FRAME
    teacher_sigma_policy: str = "uniform"
    teacher_sigma_min: float = 0.6
    teacher_sigma_max: float = 0.8
    student_layer_j: int = 19
    student_hook_point: str = "post_block"
    student_video_attention_mask_mode: str = "first_frame_causal"
    future_token_attention: str = FUTURE_TOKEN_ATTENTION_FULL
    student_text_scope: str = TEXT_CONDITION_SCOPE_ALL_FRAMES
    student_input_mode: str = "flow_noised_video_first_frame_clean"
    student_i2v_condition_mode: str = I2V_CONDITION_NONE
    student_sigma_policy: str | None = None
    student_sigma_min: float | None = None
    student_sigma_max: float | None = None
    student_init_mode: str = STAGE1_STUDENT_INIT_STAGE0_MERGED
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "attention_and_ffn"
    projector_type: str = "first_frame_decoder"
    projector_dim: int = 1024
    projector_num_layers: int = 2
    projector_num_heads: int = 8
    projector_mlp_ratio: float = 4.0
    distillation_loss: str = "cosine"
    distillation_loss_weight: float = 1.0
    train_shift: float = 5.0
    infer_shift: float = 5.0
    num_train_timesteps: int = 1000
    use_gradient_checkpointing: bool = False
    compile_dit: bool = False
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_workers: int = 4
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    max_steps: int = 100000
    num_epochs: float | None = None
    learning_rate: float = 0.0001
    projector_learning_rate: float | None = None
    warmup_steps: int | None = None
    gradient_sync_each_batch: bool = False
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 1
    save_every: int = 1000
    max_checkpoints_to_keep: int | None = None
    save_optimizer_state: bool = True
    resume_optimizer_state: bool = True
    resume: str | None = None
    wandb_enabled: bool = False
    wandb_project: str = "dreamfree"
    wandb_name: str = "stage1_robotwin"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
