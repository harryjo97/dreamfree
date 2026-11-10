from __future__ import annotations

from dataclasses import dataclass

from dreamfree.training.stage0.constants import ROBOTWIN_CAMERA_KEYS, ROBOTWIN_CONCAT_MULTI_CAMERA


@dataclass
class Stage0Config:
    """Configuration for Stage 0 LeRobot video adaptation.

    The defaults target a single-node LoRA adaptation run and keep all paths
    under the shared project storage tree.
    """

    model_base_path: str = "./assets/models"
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    video_model_family: str = "wan22_ti2v_5b"
    dit_role: str = "single"
    i2v_condition_mode: str = "none"
    latent_frame0_mode: str = "official"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    tokenizer_max_len: int = 128
    use_text_embedding_cache: bool = True
    text_embedding_cache_dir: str = "./assets"
    text_dim: int = 4096
    redirect_common_files: bool = True
    pretrained_load_broadcast: bool = True
    dataset_root: str = "./assets"
    dataset_dirs: str | None = None
    dataset_format: str = "lerobot_root"
    dataset_name: str = "robotwin"
    dataset_mix_weights: str | None = None
    camera_keys: str = ",".join(ROBOTWIN_CAMERA_KEYS)
    extract_cache_dir: str = "./assets"
    output_dir: str = "./assets/models"
    final_export_dir: str = "./assets/models"
    prompt_policy: str = "task_text"
    prepare_only: bool = False
    clip_frames: int = 9
    raw_frame_stride: int = 4
    sample_stride_frames: int = 4
    height: int = 384
    width: int = 320
    train_epoch_size: int = 200000
    val_epoch_size: int = 512
    val_fraction: float = 0.01
    video_backend: str | None = None
    camera_layout: str = ROBOTWIN_CONCAT_MULTI_CAMERA
    use_cached_latents: bool = False
    video_latent_cache_dir: str | None = None
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_workers: int = 4
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    max_steps: int = 20000
    num_epochs: float | None = None
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    mixed_precision: str = "bf16"
    seed: int = 42
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "attention_and_ffn"
    train_shift: float = 5.0
    infer_shift: float = 5.0
    num_train_timesteps: int = 1000
    train_sigma_policy: str = "wan_shift_full"
    train_sigma_min: float = 0.4
    train_sigma_max: float = 0.875
    train_sigma_stratified_sampling: bool = True
    use_gradient_checkpointing: bool = False
    compile_dit: bool = False
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 1
    eval_num_inference_steps: int = 10
    eval_seed: int = 42
    eval_fps: int = 8
    eval_text_cfg_scale: float = 1.0
    eval_tiled: bool = False
    save_every: int = 1000
    max_checkpoints_to_keep: int | None = None
    export_every: int = 0
    export_final: bool = True
    export_dtype: str = "bf16"
    export_max_shard_size_gb: float = 4.0
    save_optimizer_state: bool = True
    resume: str | None = None
    wandb_enabled: bool = False
    wandb_project: str = "dreamfree"
    wandb_name: str = "stage0_robotwin"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
