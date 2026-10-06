# Data and artifacts

## Dataset geometry

The `dataset` selector sets the standard camera order, mosaic, and action/state dimensions. Other training parameters can be overridden explicitly.

| Dataset | Video H × W | Action / state dimensions | Camera order |
| --- | --- | --- | --- |
| RoboTwin | 384 × 320 | 14 / 14 | high, left wrist, right wrist |
| LIBERO | 224 × 448 | 7 / 8 | image, wrist image |
| RoboCasa | 352 × 352 | 7 / 9 | left agent view, right agent view, wrist |
| DROID | 352 × 640 | 8 / 8 | exterior left, exterior right, wrist |

Video adaptation/distillation defaults to nine sampled frames separated by four raw frames. The corresponding action configuration has `num_frames: 33` and `action_video_freq_ratio: 4`. A cached sample contains three temporal latent frames. Standard action chunks contain 32 actions.

Stages 0/1 split by episode and sample valid video windows. Stage 2 splits by episode and includes padded episode tails; padding masks exclude padded action targets from the loss. Cached and live modes preserve the same valid action/state stream. Values at masked tail positions need not match.

RoboCasa uses its recorded visual-preprocessing identifier to check consistency across data preparation and all three checkpoints. DROID preserves the joint-position-delta/gripper convention and uses the recorded modality ranges and quantile normalization statistics.

## Text cache

`dreamfree.data.prepare_text` writes one `.pt` file per formatted prompt, keyed by its SHA-256 digest, context length, and upstream backbone identifier. Each file contains:

- `format: dreamfree_text_v1` and `model_id`.
- `context`: a finite `[128, 4096]` tensor.
- `mask`: a boolean `[128]` tensor marking valid tokenizer positions.

Readers zero padded embedding rows and use the full fixed context length, matching the training attention convention. An explicitly configured cache must contain every requested prompt. In evaluation, leaving the cache unset uses the loaded text encoder.

## Video cache

`dreamfree.data.prepare_latents` writes `episode_XXXXXX.safetensors`, with `cache_version: 3`, `coverage: full_episode`, and `padding_policy: clamp_to_episode_end`. Metadata records the model, VAE, geometry, temporal sampling, action/state dimensions, normalization source, and applicable visual contract.

Each episode file contains full-episode `latents`, normalized `action` and `proprio`, and `task_indices`. `14b` also requires official Wan image-conditioning latents. DROID includes raw joint/gripper tensors so sample-relative action normalization remains exact.

The readers validate coverage and model/shape contracts before serving samples. Cached mode does not fall back to decoding video. `--limit-episodes` is useful for preparing a small inspection run; a partial result cannot be used to train over a larger dataset.

## Checkpoints

A checkpoint is a directory containing `trainer_state.json` and tensor files. Metadata records the configuration and source-checkpoint lineage.

| Stage | Type / version | Main weights |
| --- | --- | --- |
| 0 | `dreamfree_stage0` / 1 | `lora.safetensors` |
| 1 | `dreamfree_stage1` / 2 | `stage1_lora.safetensors`, projector weights |
| 2 | `dreamfree_stage2` / 2 | `action.safetensors`, `readout.safetensors`, `conditioner.safetensors`, `proprio_encoder.safetensors` |

Optimizer/scheduler state is saved when configured. Evaluation reconstructs the student using stage 2 → stage 1 → stage 0 lineage and the upstream weights, so keep those source directories available. Moving artifacts requires updating their recorded paths consistently. These formats intentionally require newly generated DreamFree artifacts; no migration loader is provided.
