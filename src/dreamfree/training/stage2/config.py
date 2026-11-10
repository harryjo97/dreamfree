from __future__ import annotations

from dataclasses import dataclass

from dreamfree.models.student_readout import (
    DEFAULT_READOUT_INPUT_PROJECTION_MLP_RATIO,
    READOUT_INPUT_PROJECTION_LINEAR,
)
from dreamfree.training.stage0.constants import ROBOTWIN_CONCAT_MULTI_CAMERA
from dreamfree.training.stage2.constants import READOUT_TYPE_LATENT_QUERY


@dataclass
class Stage2Config:
    """Stage 2 schema; use load_config for selected model/dataset defaults.

    The training loop resolves release initialization and dataset-based budgets
    after parsing. The visual encoder stays frozen. See docs/configuration.md.
    """

    model_base_path: str = "./assets/models"
    output_dir: str = "./assets/models"
    mixed_precision: str = "bf16"
    seed: int = 42
    save_optimizer_state: bool = True
    resume: str | None = None
    init_checkpoint: str | None = None
    stage1_checkpoint: str | None = None
    dataset_name: str = "robotwin"
    dataset_dirs: str = "./assets"
    dataset_stats_path: str = "./assets"
    text_embedding_cache_dir: str = "./assets"
    num_frames: int = 33
    action_video_freq_ratio: int = 4
    action_horizon: int | None = None
    action_history_horizon: int = 0
    action_history_dropout: float | None = None
    height: int = 384
    width: int = 320
    action_dim: int = 14
    proprio_dim: int = 14
    strict_dataset_errors: bool = True
    val_fraction: float = 0.01
    global_sample_stride: int = 1
    camera_layout: str = ROBOTWIN_CONCAT_MULTI_CAMERA
    use_cached_latents: bool = False
    video_latent_cache_dir: str | None = None
    diagnostic_timing: bool = False
    profile_steps: int = 0
    profile_start_step: int = 5
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    video_model_family: str = "wan22_ti2v_5b"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    tokenizer_max_len: int = 128
    text_dim: int = 4096
    video_hidden_dim: int = 3072
    video_latent_dim: int = 48
    video_dit_input_dim: int = 48
    video_dit_output_dim: int = 48
    redirect_common_files: bool = True
    pretrained_load_broadcast: bool = True
    merge_student_lora: bool = True
    student_readout_layer: int | None = None
    student_hook_point: str | None = None
    student_video_attention_mask_mode: str = "first_frame_causal"
    student_token_norm: str = "fixed_tokenwise_layer_norm"
    readout_type: str = READOUT_TYPE_LATENT_QUERY
    readout_num_queries: int = 64
    readout_active_query_budgets: list[int] | None = None
    readout_eval_num_queries: int | None = None
    readout_num_layers: int = 2
    readout_num_heads: int = 8
    readout_mlp_ratio: float = 4.0
    readout_dropout: float = 0.0
    readout_output_dim: int | None = None
    readout_input_projection_type: str = READOUT_INPUT_PROJECTION_LINEAR
    readout_input_projection_mlp_ratio: float = DEFAULT_READOUT_INPUT_PROJECTION_MLP_RATIO
    student_conditioning_mode: str = "latent_query_cross_attention"
    student_context_norm: str = "layernorm_affine"
    student_context_concat_order: str = "text_then_student_readout"
    student_train_mode: str = "frozen"
    action_hidden_dim: int = 1024
    action_ffn_dim: int = 4096
    action_num_layers: int = 30
    action_num_heads: int = 24
    action_attn_head_dim: int = 128
    action_text_dim: int = 4096
    action_freq_dim: int = 256
    action_eps: float = 1e-06
    action_init_mode: str = "pretrained_action_backbone"
    action_dit_pretrained_path: str = "./assets/models"
    skip_dit_load_from_pretrain: bool = False
    action_train_mode: str = "full_action_dit"
    action_output_parameterization: str = "velocity"
    action_loss_parameterization: str = "flow_matching"
    action_train_shift: float = 5.0
    proprio_conditioning_mode: str = "query_prefix"
    proprio_obs_horizon: int = 1
    proprio_obs_dropout: float = 0.2
    proprio_state_timestep_sigma: float = 0.001
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
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 16
    save_every: int = 1000
    max_checkpoints_to_keep: int | None = None
    wandb_enabled: bool = False
    wandb_project: str = "dreamfree"
    wandb_name: str = "dreamfree_stage2_student_native_vel"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
