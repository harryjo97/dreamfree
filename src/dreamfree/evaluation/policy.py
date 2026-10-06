"""Load a standard stage 2 checkpoint into an optimized benchmark policy."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from dreamfree.data.lerobot.utils.normalizer import load_dataset_stats_from_json
from dreamfree.data.stage2 import _build_stage2_processor
from dreamfree.evaluation.robotwin import policy as robotwin
from dreamfree.models import policy_loading
from dreamfree.models.conditioning import TeacherStateActionConditioner
from dreamfree.models.wan22.action_dit import ActionDiT


def build_policy(environment: str, options: dict):
    """Build only the optimized runtime; options contain paths and inference settings."""
    if environment not in ("libero", "robocasa", "robotwin", "droid"):
        raise ValueError(f"Unknown evaluation environment: {environment!r}.")
    if environment == "droid":
        from dreamfree.evaluation.droid.deploy_policy_optimized import build_policy_from_args

        return build_policy_from_args(options)
    checkpoint = Path(options["ckpt_setting"]).expanduser().resolve()
    metadata = json.loads((checkpoint / "trainer_state.json").read_text())
    if metadata.get("checkpoint_type") != "dreamfree_stage2":
        raise ValueError("Expected a DreamFree stage 2 checkpoint.")
    module = importlib.import_module(f"dreamfree.evaluation.{environment}.policy")
    module._validate_dreamfree_stage2_for_eval(checkpoint, metadata)
    source_path = Path(metadata["source_stage1_checkpoint"])
    source = json.loads((source_path / "trainer_state.json").read_text())
    robotwin._validate_dreamfree_source_stage1_matches_stage2(
        stage1_checkpoint_path=source_path, stage1_metadata=source, stage2_metadata=metadata
    )
    if environment == "robocasa":
        module._validate_robocasa_visual_lineage(stage1_metadata=source, stage2_metadata=metadata)
    cfg = robotwin._build_dreamfree_stage2_cfg_from_metadata(metadata)
    os.environ["DREAMFREE_MODEL_BASE_PATH"] = str(
        options.get("model_base_path") or cfg.model_base_path
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
        load_text_encoder=bool(options.get("load_text_encoder", True)),
    )
    readout = robotwin._build_readout_for_eval(
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
    stats_path = Path(options["dataset_stats_path"]).expanduser().resolve()
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
        replan_steps=int(options.get("replan_steps", 8)),
        num_inference_steps=int(options.get("num_inference_steps", 10)),
        seed=options.get("seed", 42),
        rand_device=options.get("rand_device", "cpu"),
        tiled=False,
        timing_enabled=bool(options.get("timing_enabled", False)),
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
            flip_images=True,
            robocasa_image_size=module._robocasa_image_size_from_metadata(
                metadata, label="stage 2"
            ),
            robocasa_visual_preprocess_id=module._robocasa_visual_preprocess_id_from_metadata(
                metadata
            ),
        )
        policy.optimization_contract = resolve_options(
            metadata, {**options, "inference_optimization": "true"}
        )
        prepare_policy_for_evaluation(policy)
    else:
        from dreamfree.evaluation.optimized import (
            OptimizedLiberoPolicy,
            OptimizedRobotWinPolicy,
            optimize_policy,
        )

        if environment == "libero":
            (height, width) = metadata["dataset"]["video_size"]
            policy = OptimizedLiberoPolicy(
                **common,
                input_h=height,
                input_w=width,
                concat_multi_camera="horizontal",
                binarize_gripper=True,
                action_data_source="libero",
                image_preprocess_source="libero",
            )
        else:
            policy = OptimizedRobotWinPolicy(**common)
        optimize_policy(policy)
    return policy
