"""Load a standard stage 2 checkpoint into an optimized benchmark policy."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from dreamfree.config_names import INFERENCE_RENAMES, rename_fields
from dreamfree.data.action_processing import _build_stage2_processor
from dreamfree.data.lerobot.utils.normalizer import load_dataset_stats_from_json
from dreamfree.evaluation import checkpoints
from dreamfree.evaluation.options import parse_bool, parse_mode
from dreamfree.models import policy_loading
from dreamfree.models.conditioning import TeacherStateActionConditioner
from dreamfree.models.release import read_checkpoint_metadata, resolve_policy_checkpoint
from dreamfree.models.wan22.action_dit import ActionDiT


def validate_options(options):
    """Reject unsupported settings before loading weights.

    Args:
        options: Policy paths and inference options from an evaluator or server.

    Returns:
        Options with legacy field names translated and booleans parsed.
    """
    options = rename_fields(options, INFERENCE_RENAMES)
    for key, default in (
        ("compile_dit", True),
        ("tiled", False),
        ("load_text_encoder", True),
        ("timing_enabled", False),
        ("flip_images", True),
        ("binarize_gripper", True),
    ):
        options[key] = parse_bool(options.get(key, default))
    if not options["compile_dit"]:
        raise ValueError("Optimized inference requires compile_dit=true.")
    if options["tiled"]:
        raise ValueError("Optimized inference requires tiled=false.")
    mode = parse_mode(options.get("inference_optimization", "true"), "inference_optimization")
    if mode == "false":
        raise ValueError("Optimized inference requires inference_optimization=true or auto.")
    for key, expected in (("sigma_shift", None), ("text_cfg_scale", 1.0), ("negative_prompt", "")):
        if options.get(key, expected) != expected:
            raise ValueError(
                f"Optimized inference does not support overriding {key}={options[key]!r}."
            )
    if options.get("image_preprocess_source") not in (None, "auto", "libero"):
        raise ValueError("LIBERO image_preprocess_source must be auto or libero.")
    if "eval_num_inference_steps" in options:
        raise ValueError("Use num_inference_steps instead of eval_num_inference_steps.")
    if "policy_runtime" in options:
        raise ValueError(
            "Remove policy_runtime; DreamFree selects its optimized runtime automatically."
        )
    return options


def build_policy(environment: str, options: dict):
    """Assemble checkpoint components and the selected optimized benchmark adapter.

    Args:
        environment: libero, robocasa, droid, or yam.
        options: Checkpoint/model/statistics paths and inference settings.

    Returns:
        Benchmark policy with its existing compilation and warmup behavior.
        DROID uses its server-compatible builder; other adapters share the
        checkpoint, encoder, readout, action-model, and conditioner assembly.
    """
    if environment not in ("libero", "robocasa", "droid", "yam"):
        raise ValueError(f"Unknown evaluation environment: {environment!r}.")
    if environment == "droid":
        from dreamfree.evaluation.droid_sim.deploy_policy_optimized import build_policy_from_args

        return build_policy_from_args(options)
    options = validate_options(options)
    if environment == "robocasa" and not options["flip_images"]:
        raise ValueError("RoboCasa evaluation requires flip_images=true to match training pixels.")
    checkpoint = resolve_policy_checkpoint(options["checkpoint_path"])
    metadata = read_checkpoint_metadata(checkpoint, stage="stage2")
    allowed_types = {"dreamfree_stage2"}
    if environment == "yam":
        from dreamfree.models.action_history import CHECKPOINT_TYPE

        allowed_types.add(CHECKPOINT_TYPE)
    if metadata.get("checkpoint_type") not in allowed_types:
        raise ValueError("Expected a DreamFree stage 2 checkpoint.")
    module = importlib.import_module(f"dreamfree.evaluation.{environment}.policy")
    module._validate_dreamfree_stage2_for_eval(checkpoint, metadata)
    source_path = Path(metadata["source_stage1_checkpoint"])
    source = read_checkpoint_metadata(source_path, stage="stage1")
    checkpoints.validate_stage1_stage2_lineage(
        stage1_checkpoint_path=source_path, stage1_metadata=source, stage2_metadata=metadata
    )
    cfg = checkpoints.stage2_config_from_metadata(metadata)
    os.environ["DREAMFREE_MODEL_BASE_PATH"] = str(
        options.get("model_base_path")
        or os.environ.get("DREAMFREE_MODEL_BASE_PATH")
        or cfg.model_base_path
    )
    device = str(options.get("device", "cuda"))
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Optimized inference requires CUDA.")
    if options.get("mixed_precision", "bf16") != "bf16":
        raise ValueError("Optimized inference requires BF16.")
    core = policy_loading._load_stage1_student_core_with_text(
        stage1_checkpoint_path=source_path,
        metadata=source,
        device=device,
        dtype=torch.bfloat16,
        load_text_encoder=options["load_text_encoder"],
    )
    readout = checkpoints.load_readout(
        metadata=metadata,
        checkpoint_path=checkpoint,
        device=torch.device(device),
        dtype=torch.bfloat16,
    )
    action_config = dict(
        action_dim=cfg.action_dim,
        hidden_dim=cfg.action_hidden_dim,
        ffn_dim=cfg.action_ffn_dim,
        text_dim=cfg.action_text_dim,
        freq_dim=cfg.action_freq_dim,
        eps=cfg.action_eps,
        num_heads=cfg.action_num_heads,
        attn_head_dim=cfg.action_attn_head_dim,
        num_layers=cfg.action_num_layers,
        use_gradient_checkpointing=False,
    )
    action = ActionDiT.from_pretrained(
        action_dit_config=action_config,
        action_dit_pretrained_path="",
        skip_dit_load_from_pretrain=True,
        device=device,
        torch_dtype=torch.bfloat16,
    ).eval()
    condition_meta = metadata["conditioner"]
    conditioner = (
        TeacherStateActionConditioner(
            teacher_state_input_dim=int(condition_meta["student_readout_input_dim"]),
            action_context_dim=int(condition_meta["action_context_dim"]),
            proprio_dim=int(cfg.proprio_dim),
            proprio_obs_dropout=float(cfg.proprio_obs_dropout),
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    action.load_state_dict(load_file(str(checkpoint / "action.safetensors")), strict=True)
    conditioner.load_teacher_state_dict(
        load_file(str(checkpoint / "conditioner.safetensors")), strict=True
    )
    conditioner.load_proprio_state_dict(
        load_file(str(checkpoint / "proprio_encoder.safetensors")), strict=True
    )
    stats_value = options.get("dataset_stats_path")
    if not stats_value and "release" in metadata:
        stats_value = metadata["dataset"]["dataset_stats_path"]
    if not stats_value:
        raise ValueError("dataset_stats_path is required for a native training checkpoint.")
    stats_path = Path(stats_value).expanduser().resolve()
    processor = _build_stage2_processor(cfg).eval()
    stats = (
        module._load_dataset_stats_for_libero_eval(stats_path)
        if environment == "libero"
        else load_dataset_stats_from_json(str(stats_path))
    )
    processor.set_normalizer_from_stats(stats)
    common = dict(
        processor=processor,
        video_core=core,
        readout=readout,
        action_model=action,
        conditioner=conditioner,
        stage1_metadata=source,
        stage2_metadata=metadata,
        stage1_checkpoint_path=source_path,
        stage2_checkpoint_path=checkpoint,
        dataset_stats_path=stats_path,
        action_horizon=int(metadata["action"]["action_horizon"]),
        replan_steps=int(
            options.get("replan_steps", {"libero": 10, "robocasa": 32}.get(environment, 8))
        ),
        num_inference_steps=int(options.get("num_inference_steps", 10)),
        seed=options.get("seed", 42),
        noise_device=options.get("noise_device", "cpu"),
        tiled=False,
        timing_enabled=options["timing_enabled"],
        text_embedding_cache_dir=options.get("text_embedding_cache_dir"),
        compile_dit=False,
        compile_dit_block_counts={},
    )
    if options.get("action_horizon") not in (None, common["action_horizon"]):
        raise ValueError("Evaluation action_horizon must match the checkpoint.")
    if environment == "robocasa":
        from dreamfree.evaluation.robocasa.deploy_policy_optimized import (
            OptimizedNativeRoboCasaPolicy,
            prepare_policy_for_evaluation,
            resolve_options,
        )

        policy = OptimizedNativeRoboCasaPolicy(
            **common,
            flip_images=options["flip_images"],
            robocasa_image_size=module._robocasa_image_size_from_metadata(
                metadata, label="stage 2"
            ),
        )
        policy.optimization_contract = resolve_options(
            metadata, {**options, "inference_optimization": "true"}
        )
        prepare_policy_for_evaluation(policy)
    else:
        from dreamfree.evaluation.optimized import (
            OptimizedLiberoPolicy,
            optimize_policy,
        )

        if environment == "libero":
            (height, width) = metadata["dataset"]["video_size"]
            policy = OptimizedLiberoPolicy(
                **common,
                input_h=height,
                input_w=width,
                camera_layout="horizontal",
                binarize_gripper=options["binarize_gripper"],
                action_data_source="libero",
                image_preprocess_source="libero",
            )
        else:
            from dreamfree.evaluation.yam.policy import YamPolicy

            policy = YamPolicy(**common)
        optimize_policy(policy)
    return policy
