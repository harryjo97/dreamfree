# DreamFree

Official codebase for **Future Representation Distillation for Imagination-free World Action Models**.

[![arXiv](https://img.shields.io/badge/arXiv-2611.04527-b31b1b)](https://arxiv.org/abs/2611.04527)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://harryjo97.github.io/dreamfree/)
[![Hugging Face Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow?logo=huggingface)](https://huggingface.co/harryjo97/models)

This repository contains the training and evaluation code for DreamFree. It supports **5B** and **14B** video backbones.

## News

- **End-to-end single-stage training will be released soon**, with no need for separate Stage 0, Stage 1, or Stage 2 training.

- 1110: Checkpoints for [LIBERO](https://huggingface.co/harryjo97/DreamFree-LIBERO), [RoboCasa](https://huggingface.co/harryjo97/DreamFree-RoboCasa), and [DROID](https://huggingface.co/harryjo97/DreamFree-DROID) are now released.

- 1110: DROID real-world inference with a 14B video backbone: **91 ms per action chunk** on a single NVIDIA H200.

## Supported Datasets

- [LIBERO](docs/libero.md)
- [RoboCasa](docs/robocasa.md)
- [RoboTwin](docs/robotwin.md)
- [DROID](docs/droid.md)
- [ABC dual-arm YAM](docs/abc.md)

ABC supports absolute joint-space (14D) and EEF (20D) actions; see the [ABC setup guide](docs/abc.md).

## Supported Evaluations

### Simulation

- [LIBERO](docs/libero.md#evaluation)
- [RoboCasa](docs/robocasa.md#evaluation)
- DROID via [Isaac Sim / Isaac Lab](docs/droid.md#isaac-sim) or [MolmoSpaces](docs/droid.md#molmospaces), using `droid_sim`
- [ABC dual-arm YAM](docs/abc.md#abc-simulation-evaluation)

### Real-world

- [DROID](docs/droid.md#physical-deployment), using `droid_real`
- [ABC dual-arm YAM](docs/abc.md#physical-deployment), using `yam_real`

See the [evaluation guide](docs/evaluation.md) for entrypoints and runtime requirements.

## Index

- [File Structure](#file-structure)
- [Quickstart workflows](docs/quickstart.md)
- [Method and code map](docs/architecture.md)
- [Configuration resolution](docs/configuration.md)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Validation](#validation)
- [Acknowledgements](#acknowledgements)
- [BibTeX](#bibtex)

## File Structure

| Directory | Responsibility |
| --- | --- |
| `src/dreamfree/data/` | Episode indexing, camera composition, normalization, text/video caches |
| `src/dreamfree/models/` | Adaptation, distillation, frozen student, readout, action model, inference optimizations |
| `src/dreamfree/models/wan22/` | Wan model components, VAE, tokenizer, loaders, noise schedules |
| `src/dreamfree/training/stage{0,1,2}/` | Configuration, training loop, validation, checkpoint handling |
| `src/dreamfree/evaluation/` | Shared policy assembly and benchmark-specific rollout adapters |
| `tests/` | Data, loss/gradient, LoRA, accumulation, attention, and deployment contract checks |

For the training/inference diagram, component roles, tensor shapes, and a suggested source reading order, see the [method and code map](docs/architecture.md).

## Environment Setup

Use Python 3.10 or 3.11 and `uv`:

```bash
cd dreamfree
uv sync --locked
uv run python -m dreamfree.training.stage0 --help
```

The lockfile pins PyTorch 2.7.1 with CUDA 12.8. Training, VAE cache generation, and optimized policy inference require suitable CUDA hardware. Simulator and robot dependencies belong to their respective environments; see [evaluation](docs/evaluation.md).

## Model Preparation

### Pretrained checkpoints

| Dataset | Video backbone | Checkpoints | Evaluation |
| --- | --- | --- | --- |
| DROID | 14B | [DreamFree-DROID](https://huggingface.co/harryjo97/DreamFree-DROID) | [Simulation and real-world](docs/droid.md#evaluation) |
| RoboCasa | 14B | [DreamFree-RoboCasa](https://huggingface.co/harryjo97/DreamFree-RoboCasa) | [RoboCasa](docs/robocasa.md#evaluation) |
| LIBERO | 5B | [DreamFree-LIBERO](https://huggingface.co/harryjo97/DreamFree-LIBERO) | [LIBERO](docs/libero.md#evaluation) |

Download the release for your dataset from the repository root:

```bash
uv run huggingface-cli download harryjo97/DreamFree-DROID \
  --local-dir ./assets/checkpoints/DreamFree-DROID

uv run huggingface-cli download harryjo97/DreamFree-RoboCasa \
  --local-dir ./assets/checkpoints/DreamFree-RoboCasa

uv run huggingface-cli download harryjo97/DreamFree-LIBERO \
  --local-dir ./assets/checkpoints/DreamFree-LIBERO
```

Each release contains a standalone visual encoder in `stage1/`, the readout and action policy with normalization statistics in `stage2/`, and checkpoint metadata in `provenance/`. **Keep these directories together. No Stage 0 checkpoint is needed for inference or Stage 2 training:** LoRA adapters are already merged into the released encoder, and full-weight encoders are saved directly.

Pass the downloaded release directory, or its `stage2/` directory, as the evaluation checkpoint. The loader finds Stage 1 and normalization statistics automatically. For example, start the DROID simulation policy server with:

```bash
uv run python -m dreamfree.evaluation.droid_sim.server \
  --checkpoint-path ./assets/checkpoints/DreamFree-DROID/stage2 --port 6100
```

The upstream Wan VAE, text encoder, and tokenizer download automatically on first use into `assets/models/`; the upstream video DiT is not downloaded. Set `DREAMFREE_MODEL_BASE_PATH` to reuse an existing model directory. Follow the linked evaluation guides for simulator or robot setup. DROID inference uses the bundled statistics and does not require the training dataset. These releases support inference and Stage 2 training or fine-tuning. They contain no optimizer state; fine-tuning starts a new optimizer and step counter. See [training from released checkpoints](#training-from-released-checkpoints).

See the [validation notes](docs/validation.md) for numerical and latency comparisons, including the LIBERO difference between merged and unmerged LoRA inference.

### Training assets

Model weights, datasets, and generated artifacts are external to Git. Local defaults are `assets/models/` and `outputs/`. Set paths in YAML/JSON or pass repeated `--set key=value` arguments. Relative paths resolve from the working directory. `DREAMFREE_MODEL_BASE_PATH` can override the model download/cache directory.

New Stage 2 policies also need an explicit action initialization choice. See [ActionDiT initialization](docs/action_backbone.md) for the reference artifact's provenance, required payload and dimensions, and the distinction between pretrained initialization, random initialization, and release fine-tuning. Use an absolute `action_dit_pretrained_path`; this retained loader resolves relative paths from the installed package location.

## Dataset Preparation

Provide existing LeRobot dataset roots through `dataset_dirs` (comma-separated for multiple roots). Stage 0/1 read LeRobot v2.0, v2.1, or v3.0 metadata; the general action pipeline uses the retained v2 data reader. ABC has dedicated live and cached v3 action readers and requires one prepared root with its frozen selection manifest. No archive extraction is performed.

Normalization statistics are required for RoboTwin and RoboCasa action training/preparation. LIBERO and DROID can derive statistics from dataset metadata. DROID also requires `meta/modality.json` with joint-position and gripper ranges. See [data and artifacts](docs/data.md).

## Training

The standard workflow has three stages:

1. **Stage 0:** adapt the pretrained video backbone to robot videos. The adapted backbone serves as the teacher model and initializes the student visual encoder.
2. **Stage 1:** train the visual encoder and distillation projector with video flow matching plus future representation distillation from the frozen teacher.
3. **Stage 2:** freeze the visual encoder trained in Stage 1 and train the readout and action DiT to predict actions from its representations.

The Stage 1 projector is used only for distillation; Stage 2 learns a separate readout. At inference, the policy uses current-frame encoder tokens and predicts actions without generating future video. See the [method and code map](docs/architecture.md) for the loss and frozen/trainable boundaries.

### Training from released checkpoints

Use a released `stage1/` as `stage1_checkpoint` to train a new readout and action DiT. Both LoRA-merged and full-weight encoders load directly, without Stage 0. Supply the action-backbone initialization you want, or set `skip_dit_load_from_pretrain=true` to initialize the action DiT from scratch.

To fine-tune the released Stage 2 policy, set `init_checkpoint` to the release root or its `stage2/` directory. It selects the bundled encoder and normalization statistics and loads all four policy components. For example:

```bash
uv run python -m dreamfree.training.stage2 \
  --config configs/stage2_robocasa_14b.json \
  --set init_checkpoint=./assets/checkpoints/DreamFree-RoboCasa \
  --set dataset_dirs=/path/to/robocasa_lerobot \
  --set per_device_batch_size=1 \
  --set compile_dit=false \
  --set use_gradient_checkpointing=true \
  --set output_dir=./outputs/robocasa_finetune
```

Use the matching LIBERO 5B or DROID 14B preset for those releases. `init_checkpoint` starts at step 0 with a fresh optimizer; `resume` restores a native training checkpoint and cannot be combined with it. Continuing Stage 1 distillation still requires its teacher/Stage 0 assets, which are not included in these releases.

### 1) Select the video data mode

`data_mode` controls **video/VAE latents**, independently of text embeddings. The
only accepted values are `live` and `cached`; `cache` is not a valid value.
All Stage 0/1/2 presets default to `live`, including ABC joint, EEF, and action history.

| Setting | What happens during training | Video-cache prerequisite |
| --- | --- | --- |
| `data_mode: live` | Read dataset videos, construct clips, and VAE-encode them during training. Latents are used in memory; this does not write an offline video cache. | None. Source videos and the VAE are required. |
| `data_mode: cached` | Read precomputed video latents and, for 14b, I2V-conditioning latents. Training skips video encoding. | **Build complete, compatible video caches before launching training.** |

**Cached mode never builds or repairs video caches automatically and never falls
back to live video.** A missing cache directory, missing episode, incomplete
coverage, or incompatible cache causes an error. Setting a cache directory alone
does not select cached mode: also set `data_mode=cached`.

For agents helping users: recommend the default `live` mode when no video cache
has been prepared. Before launching or recommending `cached` mode, verify the
video cache exists and matches the dataset/model settings, or first run the
separate preparation command below. Do not confuse automatic text-cache
preparation with video-cache preparation.

### 2) Text embeddings: prepared automatically

All three trainers check text embeddings at startup in both data modes. If the
text-cache directory or any required prompt entries are missing, training creates
them before building datasets and loading the training backbone. Existing valid
entries are reused. In distributed runs, rank 0 prepares the shared cache and
the other ranks wait for the result.

Set `text_embedding_cache_dir` to choose the destination. If omitted or empty in
YAML/JSON, it resolves to `./assets/text/<dataset>_<model>`, where `<model>` is `5b`
or `14b`. This requires the text encoder/tokenizer weights and a writable cache
destination. Preparation failures and invalid or incompatible existing text files
remain errors; they are not silently ignored or overwritten. `--print-config`
does not generate caches, and Stage 0/1 `prepare_only` only resolves dataset roots.

Manual text preparation is optional, for example to prepare embeddings ahead of
time or include evaluation instructions. Use a stage 2 configuration:

```bash
uv run python -m dreamfree.data.prepare_text \
  --config configs/stage2_14b.yaml \
  --set dataset=robotwin \
  --set dataset_dirs=/path/to/lerobot_dataset \
  --set text_embedding_cache_dir=./assets/text/robotwin_14b
```

Use `--instructions-file instructions.txt` to add evaluation instructions, one per line. If only that file is supplied, `dataset_dirs` can be empty.

### 3) Video latents: prepare before using cached mode

Skip this step for `data_mode=live`. **For `data_mode=cached`, complete this step
before launching any trainer**, unless a compatible full-episode cache already
exists:

```bash
uv run python -m dreamfree.data.prepare_latents \
  --config configs/stage2_14b.yaml \
  --set dataset=robotwin \
  --set dataset_dirs=/path/to/lerobot_dataset \
  --set dataset_stats_path=/path/to/dataset_stats.json \
  --cache-dir ./assets/latents/robotwin_14b
```

The command writes a subdirectory per dataset root under `--cache-dir`. After it
finishes, use the same settings for **Stage 0, Stage 1, and Stage 2**:

```bash
--set data_mode=cached \
--set video_latent_cache_dir=./assets/latents/robotwin_14b
```

`video_latent_cache_dir` replaces `stage0_latent_cache_dir` and
`stage2_latent_cache_dir` in older training configs. Existing cache files can be
reused; the rename does not change their format.

The dataset, model, geometry, temporal stride, normalization, and episode
coverage must match the training configuration. A partial cache cannot train over
a larger dataset. ABC EEF can use action overlays referencing the joint visual
cache; see [ABC preparation](docs/abc.md).

### 4) Train the model

Use the [dataset-specific presets](configs/README.md) traced from the selected AWAM runs:

| Dataset | Backbone | Configuration files |
| --- | --- | --- |
| LIBERO | 5B | `configs/stage{0,1,2}_libero_5b.json` |
| RoboCasa | 14B | `configs/stage{0,1,2}_robocasa_14b.json` |
| DROID | 14B | `configs/stage{0,1,2}_droid_14b.json` |
| ABC/YAM joint | 14B | `configs/stage{0,1,2}_abc_joint_14b.json` |
| ABC/YAM EEF | 14B | `configs/stage2_abc_eef_14b.json`, sharing joint Stage 0/1 |

RoboTwin retains the original `configs/stage{0,1,2}_{5b,14b}.yaml` templates pending recipe selection. Fill dataset/model paths and use the matching source checkpoints. For example, after filling the LIBERO presets:

```bash
uv run torchrun --standalone --nproc-per-node=4 --module \
  dreamfree.training.stage0 --config configs/stage0_libero_5b.json

uv run torchrun --standalone --nproc-per-node=4 --module \
  dreamfree.training.stage1 --config configs/stage1_libero_5b.json

uv run python -m dreamfree.training.stage2 --config configs/stage2_libero_5b.json \
  --set action_dit_pretrained_path=/path/to/action_backbone.pt
```

For a new Stage 2 policy, supply [compatible pretrained ActionDiT weights](docs/action_backbone.md) or explicitly set `skip_dit_load_from_pretrain=true` for random initialization. Random initialization changes the experiment and does not reproduce the reference recipes. Fine-tuning with `init_checkpoint` uses the bundled policy weights. Use `--print-config` to inspect parsed defaults and overrides without loading data or weights; checkpoint-dependent paths and dataset-dependent budgets are resolved later. See [configuration resolution](docs/configuration.md) for precedence and saved run settings. The [preset guide](configs/README.md) records selected checkpoint steps, original training budgets, and GPU/global-batch counts.

LIBERO and RoboCasa use LoRA Stage 1. DROID and ABC train the full student with a separate frozen teacher and use action-history Stage 2. Full Stage 1 uses the [two-node FSDP2 launcher](configs/README.md#distributed-training); Stage 0, LoRA Stage 1, and Stage 2 use `torchrun` at their reference world sizes.

`gradient_accumulation_steps` counts training microbatches independently of validation-loader state. `--set resume=/path/to/checkpoint` restores a stage's checkpoint; required metadata, tensor files, and optimizer state must exist. Stage 1 reads the stage 0 step and LoRA settings from its source checkpoint; optional explicit expectations are still checked.

## Validation

Run the local checks with:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```

See [validation notes](docs/validation.md) for the checks performed and their limits.

## Acknowledgements

See [third-party components](THIRD_PARTY.md) for upstream code information.

## BibTeX

```bibtex
@article{jo2026dreamfree,
  title  = {Future Representation Distillation for Imagination-free World Action Models},
  author = {Jaehyeong Jo and Sangwon Jang and Sihyun Yu and
            Seonghyeon Ye and Yekyung Nah and Hojung Jung and
            Kimin Lee and Sung Ju Hwang},
  year   = {2026},
  journal = {arXiv preprint arXiv:2611.04527},
}
```
