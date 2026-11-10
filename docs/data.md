# Data and artifacts

## Source map

| Responsibility | Module |
| --- | --- |
| Stage 2 dataset selection, episode splits, live reader, dataloader | [stage2.py](../src/dreamfree/data/stage2.py) |
| Action/state geometry, processors, normalization statistics | [action_processing.py](../src/dreamfree/data/action_processing.py) |
| Stage 2 cache validation | [action_cache.py](../src/dreamfree/data/action_cache.py) |
| Live/cached tensors at the model boundary | [action_batch.py](../src/dreamfree/data/action_batch.py) |
| Offline preparation orchestration and atomic cache writing | [latent_preparation.py](../src/dreamfree/data/latent_preparation.py) |
| Preparation configuration, cache identity, format validation | [latent_config.py](../src/dreamfree/data/latent_config.py) |
| Episode metadata/parquet reading, camera composition, target normalization | [latent_data.py](../src/dreamfree/data/latent_data.py) |
| VAE loading and window encoding | [latent_encoding.py](../src/dreamfree/data/latent_encoding.py) |

Follow the stage loop to the dataset builder, then to the batch-preparation boundary. Offline preparation has a separate path through episode loading, VAE encoding, and cache writing; its output is consumed by the same training cache contracts.

## Dataset geometry

The `dataset` selector sets the standard camera order, mosaic, and action/state dimensions. Other training parameters can be overridden explicitly. Follow each dataset's guide for preparation, training settings, and evaluation.

| Dataset | Video H × W | Action / state dimensions | Camera order |
| --- | --- | --- | --- |
| [RoboTwin](robotwin.md) | 384 × 320 | 14 / 14 | high, left wrist, right wrist |
| [LIBERO](libero.md) | 224 × 448 | 7 / 8 | image, wrist image |
| [RoboCasa](robocasa.md) | 352 × 352 | 7 / 9 | left agent view, right agent view, wrist |
| [DROID](droid.md) | 352 × 640 | 8 / 8 | exterior left, exterior right, wrist |
| [ABC joint](abc.md#representations) | 432 × 384 | 14 / 14 | top, left wrist, right wrist |
| [ABC EEF](abc.md#representations) | 432 × 384 | 20 / 14 | top, left wrist, right wrist |

Video adaptation/distillation defaults to nine sampled frames separated by four raw frames. The corresponding action configuration has `num_frames: 33` and `action_video_freq_ratio: 4`. A cached sample contains three temporal latent frames. Standard action chunks contain 32 actions.

Stages 0/1 split by episode and sample valid video windows. Stage 2 splits by episode and includes padded episode tails; padding masks exclude padded action targets from the loss. Cached and live modes preserve the same valid action/state stream. Values at masked tail positions need not match.

RoboCasa shares floating-point image resizing across training, latent preparation, and evaluation. DROID preserves the joint-position-delta/gripper convention and uses the recorded modality ranges and quantile normalization statistics.

ABC uses absolute dual-arm joint or wrist-pose targets with joint proprioception, train-pool q01/q99 statistics, and an immutable prepared episode split. Both action settings share joint-dataset Stage 0/1 visual training. See [ABC data preparation and training](abc.md) for camera packing, EEF frames, and commands.

## Text cache

`dreamfree.data.prepare_text` writes one `.pt` file per formatted prompt, keyed by its SHA-256 digest, context length, and upstream backbone identifier. Each file contains:

- `format: dreamfree_text_v1` and `model_id`.
- `context`: a finite `[128, 4096]` tensor.
- `mask`: a boolean `[128]` tensor marking valid tokenizer positions.

Readers zero padded embedding rows and use the full fixed context length, matching the training attention convention. Before constructing datasets, all trainers automatically create missing text-cache directories and prompt entries on rank 0, in both live and cached video modes. Valid entries are reused; invalid existing entries still fail validation. The default destination is `./assets/text/<dataset>_<model>` when the training YAML/JSON leaves it unset. In evaluation, leaving the cache unset uses the loaded text encoder.

## Video cache

`dreamfree.data.prepare_latents` writes `episode_XXXXXX.safetensors`, with `cache_version: 3`, `coverage: full_episode`, and `padding_policy: clamp_to_episode_end`. Metadata records the model, VAE, geometry, temporal sampling, action/state dimensions, normalization source, and applicable visual contract.

Each episode file contains full-episode `latents`, normalized `action` and `proprio`, and `task_indices`. `14b` also requires official Wan image-conditioning latents. DROID includes raw joint/gripper tensors so sample-relative action normalization remains exact.

The readers validate coverage and model/shape contracts before serving samples. Cached mode does not fall back to decoding video. `--limit-episodes` is useful for preparing a small inspection run; a partial result cannot be used to train over a larger dataset.

ABC also accepts AWAM version 2 video caches when the complete ABC visual and normalization contract validates. EEF action overlays reference validated joint video caches, avoiding video-latent duplication. Other datasets still require version 3. AWAM text caches and checkpoints remain incompatible.

## Checkpoints

DreamFree supports native training checkpoints and portable release bundles. They have different file layouts and source dependencies.

### Native training checkpoints

A native checkpoint is a directory containing `trainer_state.json` and tensor files. Metadata records the configuration and source-checkpoint lineage.

| Stage | Type / version | Main weights |
| --- | --- | --- |
| 0 | `dreamfree_stage0` / 1 | `lora.safetensors` |
| 1 | `dreamfree_stage1` / 2 | `stage1_lora.safetensors` (LoRA) or `student_dit.safetensors` (full), projector weights |
| 2 | `dreamfree_stage2` / 2 | `action.safetensors`, `readout.safetensors`, `conditioner.safetensors`, `proprio_encoder.safetensors` |
| 2, DROID/ABC history | `dreamfree_stage2_action_history` / 1 | Same four Stage 2 weight files; explicit history-prefix metadata |

DROID and ABC action-history samples read preceding actions from prepared episode parquet in live mode or existing full-episode caches in cached mode, without changing latent geometry or cache format. Both use the same frozen normalization statistics, and history cannot cross episode boundaries. DROID reanchors historical joint targets to the current planning state and binarizes historical gripper commands. See [history training and deployment](abc.md#action-history-stage-2) for padding, dropout, and command ownership.

Optimizer/scheduler state is saved when configured and is required for optimizer resume. A native Stage 2 checkpoint refers to its recorded Stage 1 source. If that source is also native, the current loader uses its recorded Stage 0 assets and upstream backbone, including when loading a full-weight Stage 1 student. Keep those source directories available. If the source is a standalone release, no Stage 0 is needed.

Native checkpoints record source paths; moving them requires updating the references consistently. Raw AWAM training checkpoints cannot be substituted for DreamFree checkpoints. Older DreamFree field names are translated by the metadata loader; that compatibility does not convert an AWAM training directory.

### Portable release bundles

The published bundles contain:

| Directory | Required contents and purpose |
| --- | --- |
| `stage1/` | `config.json` and standalone encoder weights; sharded weights also include their index |
| `stage2/` | `config.json`, the four policy weight files listed above, and bundled normalization statistics |
| `provenance/` | `stage1_trainer_state.json` and `stage2_trainer_state.json`, used to validate configuration and lineage |

Pass the release root or its `stage2/` directory for evaluation or `init_checkpoint`. Keep all three directories together when moving a release. The loader resolves the bundled encoder and statistics locally; historical training paths in provenance are not dependencies. Upstream VAE/text/tokenizer assets are still needed, but Stage 0 and upstream video DiT weights are not.

Releases support inference, training a new Stage 2 policy from `stage1/`, and Stage 2 fine-tuning through `init_checkpoint`. They contain no optimizer state: fine-tuning starts a new optimizer and step counter. Use `resume` with a native training checkpoint to continue optimizer state. See [training from released checkpoints](../README.md#training-from-released-checkpoints).
