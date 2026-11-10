"""Canonical option names and compatibility with saved checkpoint metadata."""

TRAINING_RENAMES = {
    "stage0_latent_cache_dir": "video_latent_cache_dir",
    "stage2_latent_cache_dir": "video_latent_cache_dir",
    "val_set_proportion": "val_fraction",
    "pretrained_norm_stats": "dataset_stats_path",
    "concat_multi_camera": "camera_layout",
    "merge_student_lora_for_inference": "merge_student_lora",
    "teacher_state_loss": "distillation_loss",
    "lambda_teacher_state": "distillation_loss_weight",
    "stage1_lora_rank": "lora_rank",
    "stage1_lora_alpha": "lora_alpha",
    "stage1_lora_dropout": "lora_dropout",
    "stage1_lora_target_modules": "lora_target_modules",
}

INFERENCE_RENAMES = {
    "ckpt": "checkpoint_path",
    "ckpt_setting": "checkpoint_path",
    "rand_device": "noise_device",
    "stage1_cuda_graph": "encoder_cuda_graph",
}


def rename_fields(values, renames):
    """Translate saved names, rejecting contradictory old and new values."""
    result = dict(values)
    for old, new in renames.items():
        if old not in result:
            continue
        value = result.pop(old)
        if new in result and result[new] != value:
            raise ValueError(f"Conflicting values for {old!r} and {new!r}.")
        result[new] = value
    return result


def normalize_checkpoint_metadata(value):
    """Read old checkpoints without rewriting files or changing release provenance."""
    if isinstance(value, list):
        return [normalize_checkpoint_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = rename_fields(value, TRAINING_RENAMES)
    for key, item in result.items():
        if key == "release":
            continue
        result[key] = normalize_checkpoint_metadata(item)
        if key == "config" and isinstance(result[key], dict):
            result[key] = rename_fields(result[key], {"context_len": "tokenizer_max_len"})
    return result
