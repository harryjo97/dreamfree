"""Standard DreamFree inference for ABC absolute joint14 or wrist-pose20 actions."""

from pathlib import Path
from types import SimpleNamespace

import torch

from dreamfree.data import yam
from dreamfree.evaluation import checkpoints
from dreamfree.evaluation.action_protocol import (
    ISSUED_ACTION_HISTORY_IS_PAD_KEY,
    ISSUED_ACTION_HISTORY_KEY,
    extract_supplied_action_history,
)
from dreamfree.evaluation.optimized import OptimizedActionMixin
from dreamfree.evaluation.runtime import PolicyRuntime
from dreamfree.evaluation.yam.common import resolve_checkpoint
from dreamfree.evaluation.yam_real.common import validate_observation
from dreamfree.evaluation.yam_real.image_preprocessing import (
    PACKED_IMAGE_KEY,
    PREPROCESSING_KEY,
    decode_image,
    pack_raw_images,
)
from dreamfree.models import action_history
from dreamfree.models import policy_loading as runtime
from dreamfree.models.release import read_checkpoint_metadata


def _validate_dreamfree_stage2_for_eval(path, metadata):
    checkpoints.require_existing_files(
        path, checkpoints.DREAMFREE_STAGE2_REQUIRED_FILES, label="DreamFree ABC Stage 2"
    )
    action_history.validate_checkpoint(metadata)
    cfg = SimpleNamespace(**metadata["config"])
    if cfg.dataset_name not in yam.DATASET_NAMES:
        raise ValueError(f"ABC requires yam_abc130k or yam_abc130k_eef, got {cfg.dataset_name!r}.")
    yam.validate_config(cfg, stage=2)
    preset = yam.PRESETS[cfg.dataset_name]
    expected = {
        "dataset": dict(
            name=cfg.dataset_name,
            video_size=[432, 384],
            action_dim=preset.action_dim,
            proprio_dim=14,
            action_horizon=32,
            num_frames=33,
            action_video_freq_ratio=4,
            camera_layout=preset.layout,
            context_len=128,
        ),
        "student": dict(
            train_mode="frozen",
            hook_point="post_block",
            video_attention_mask_mode="first_frame_causal",
            token_norm="fixed_tokenwise_layer_norm",
            hidden_dim=5120,
            tokens_per_frame=648,
            i2v_condition_mode="first_image_mask_condition",
        ),
        "action": dict(
            action_dim=preset.action_dim,
            proprio_dim=14,
            action_horizon=32,
            output_parameterization="velocity",
            loss_parameterization="flow_matching",
        ),
        "readout": dict(type="latent_query_cross_attention", input_dim=5120),
        "conditioner": dict(
            mode="latent_query_cross_attention",
            token_policy="all_readout_tokens",
            concat_order="text_then_student_readout",
            context_norm="layernorm_affine",
            final_kv_contract="text_plus_student_readout_only",
        ),
        "trainable_modules": dict(student=False, readout=True, action_dit=True, conditioner=True),
    }
    for section, values in expected.items():
        for key, value in values.items():
            actual = metadata.get(section, {}).get(key)
            if actual != value:
                raise ValueError(
                    f"ABC checkpoint {section}.{key}: expected {value!r}, got {actual!r}."
                )
    if metadata["action"].get("scheduler", {}).get("family") != "continuous_flow_matching":
        raise ValueError("ABC requires the standard continuous-flow action scheduler.")
    contract = yam.checkpoint_contract(cfg)
    if metadata.get("abc_contract") != contract:
        raise ValueError(
            "ABC checkpoint selection, normalization or EEF kinematics differ from its training data."
        )
    if metadata["dataset"].get("dataset_dirs") != [
        part.strip() for part in cfg.dataset_dirs.split(",") if part.strip()
    ]:
        raise ValueError("ABC checkpoint dataset roots disagree with its configuration.")
    if metadata["dataset"].get("dataset_stats_path") != cfg.dataset_stats_path:
        raise ValueError("ABC checkpoint normalization path disagrees with its configuration.")
    source = Path(metadata["source_stage1_checkpoint"])
    source_metadata = read_checkpoint_metadata(source, stage="stage1")
    yam.validate_stage1_source(source_metadata)
    checkpoints.validate_stage1_stage2_lineage(
        stage1_checkpoint_path=source, stage1_metadata=source_metadata, stage2_metadata=metadata
    )


def checkpoint_context(checkpoint):
    checkpoint = resolve_checkpoint(checkpoint)
    metadata = read_checkpoint_metadata(checkpoint, stage="stage2")
    _validate_dreamfree_stage2_for_eval(checkpoint, metadata)
    return (
        checkpoint,
        metadata,
        Path(metadata["dataset"]["dataset_stats_path"]).expanduser().resolve(),
    )


def reference_build_image(observation, *, device="cpu", dtype=torch.float32):
    return pack_raw_images(observation).to(device=device, dtype=dtype)


def build_image(observation, *, device="cpu", dtype=torch.float32):
    validate_observation(observation)
    if PREPROCESSING_KEY in observation or PACKED_IMAGE_KEY in observation:
        return decode_image(observation, device=device, dtype=dtype)
    return reference_build_image(observation, device=device, dtype=dtype)


class YamPolicy(OptimizedActionMixin, PolicyRuntime):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_history_horizon = self.stage2_metadata.get("action_history", {}).get(
            "horizon", 0
        )
        self._active_previous_action = self._active_previous_action_is_pad = None

    def reset(self):
        super().reset()
        self._active_previous_action = self._active_previous_action_is_pad = None
        self._noise_generator = torch.Generator(device=self.noise_device).manual_seed(self.seed)

    def _history_inputs(self, observation):
        if not self.action_history_horizon:
            if (
                ISSUED_ACTION_HISTORY_KEY in observation
                or ISSUED_ACTION_HISTORY_IS_PAD_KEY in observation
            ):
                raise ValueError("Standard Stage 2 received unexpected action history.")
            return None, None
        raw, pad = extract_supplied_action_history(
            observation, horizon=self.action_history_horizon, action_dim=self.action_dim
        )
        # Match training normalization, then zero padded rows in normalized space.
        batch = self.processor.normalizer.forward(
            dict(
                action={"default": torch.from_numpy(raw)},
                state={"default": torch.zeros((self.action_history_horizon, 14))},
            )
        )
        normalized = self.processor.action_state_merger.forward(batch)["action"]
        padding = torch.from_numpy(pad)
        normalized[padding] = 0.0
        return (
            normalized.unsqueeze(0).to(device=self.device, dtype=self.model_dtype),
            padding.unsqueeze(0).to(device=self.device),
        )

    def _prepare_action_query(self, *, proprio_query_emb, action_seq_len):
        if not self.action_history_horizon:
            return proprio_query_emb, None
        prefix, mask, _ = action_history.build_query_prefix(
            self.action_model.action_encoder,
            proprio_query_emb,
            self._active_previous_action,
            self._active_previous_action_is_pad,
            horizon=self.action_history_horizon,
            action_dim=self.action_dim,
            future_tokens=action_seq_len,
        )
        return prefix, mask

    def _sample_randn(self, shape):
        return torch.randn(
            shape, generator=self._noise_generator, device=self.noise_device, dtype=torch.float32
        ).to(device=self.device, dtype=self.model_dtype)

    def _infer_action_chunk(self, observation, instruction):
        if instruction != observation.get("prompt"):
            raise ValueError("RPC instruction and ABC observation prompt disagree.")
        self._active_previous_action, self._active_previous_action_is_pad = self._history_inputs(
            observation
        )
        image = build_image(observation, device=self.device, dtype=self.model_dtype)
        proprio = runtime._normalize_state_with_processor(self.processor, observation["state"]).to(
            device=self.device, dtype=self.model_dtype
        )
        context, mask, student, student_mask, query = self._build_student_readout_conditioning(
            image_tensor=image, instruction=instruction, proprio=proprio
        )
        actions = self._infer_dreamfree_action_tokens(
            context=context,
            context_mask=mask,
            student_context_emb=student,
            student_context_mask=student_mask,
            proprio_query_emb=query,
        )
        return runtime._denormalize_action_with_processor(self.processor, actions)[0]

    @torch.inference_mode()
    def predict_action_chunk(self, observation, instruction):
        from dreamfree.models.inference_optimization import fused_sdpa_only

        try:
            with fused_sdpa_only():
                return self._infer_action_chunk(observation, instruction)
        finally:
            self._active_previous_action = self._active_previous_action_is_pad = None


def build_policy(
    checkpoint, *, replan_steps=8, num_inference_steps=16, seed=0, model_base_path=None
):
    from dreamfree.evaluation.policy import build_policy as build_standard_policy

    checkpoint, metadata, stats = checkpoint_context(checkpoint)
    policy = build_standard_policy(
        "yam",
        dict(
            checkpoint_path=str(checkpoint),
            dataset_stats_path=str(stats),
            replan_steps=replan_steps,
            num_inference_steps=num_inference_steps,
            seed=seed,
            load_text_encoder=True,
            model_base_path=model_base_path,
        ),
    )
    policy.reset()
    return policy
