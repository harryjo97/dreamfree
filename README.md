# DreamFree

Robot-policy training and evaluation with Wan video backbones. The project variants are **5b** and **14b**. `14b` uses the upstream **Wan2.2-I2V-A14B** model; `5b` uses **Wan2.2-TI2V-5B**.

The standard workflow has three stages:

1. **Stage 0:** adapt the video backbone with LoRA and video flow matching.
2. **Stage 1:** train student LoRA and a projector with video flow matching and teacher-state prediction.
3. **Stage 2:** freeze the student and train the latent-query readout, action transformer, and conditioning layers with action flow matching.

Supported datasets: **LIBERO, RoboCasa, RoboTwin, and DROID**. Evaluation includes DROID simulation and physical deployment. The evaluation entrypoints build the optimized BF16 runtime.

## Setup

Use Python 3.10 or 3.11 and `uv`:

```bash
cd dreamfree
uv sync --locked
uv run python -m dreamfree.training.stage0 --help
```

The lockfile pins PyTorch 2.7.1 with CUDA 12.8. Training, VAE cache generation, and optimized policy inference require suitable CUDA hardware. Simulator and robot dependencies belong to their respective environments; see [evaluation](docs/evaluation.md).

Model weights, datasets, and generated artifacts are external to Git. Local defaults are `assets/models/` and `outputs/`. Set paths in YAML or pass repeated `--set key=value` arguments. Relative paths resolve from the working directory. `DREAMFREE_MODEL_BASE_PATH` can override the model download/cache directory.

## Prepare data

Provide existing LeRobot dataset roots through `dataset_dirs` (comma-separated for multiple roots). Stage 0/1 read LeRobot v2.0, v2.1, or v3.0 metadata; the action pipeline uses the retained v2 data reader. No archive extraction is performed.

Text embeddings are cached in both video modes. Prepare them with a stage 2 configuration:

```bash
uv run python -m dreamfree.data.prepare_text \
  --config configs/stage2_14b.yaml \
  --set dataset=robotwin \
  --set dataset_dirs=/path/to/lerobot_dataset \
  --set text_embedding_cache_dir=./assets/text/robotwin_14b
```

Use `--instructions-file instructions.txt` to add evaluation instructions, one per line. If only that file is supplied, `dataset_dirs` can be empty.

`data_mode: live` decodes video and encodes it with the VAE during training. For `data_mode: cached`, prepare complete episode caches first:

```bash
uv run python -m dreamfree.data.prepare_latents \
  --config configs/stage2_14b.yaml \
  --set dataset=robotwin \
  --set dataset_dirs=/path/to/lerobot_dataset \
  --set pretrained_norm_stats=/path/to/dataset_stats.json \
  --cache-dir ./assets/latents/robotwin_14b
```

The command writes a subdirectory per dataset root under `--cache-dir`. Point training at that cache directory. Use `stage0_latent_cache_dir` for stage 0 and `stage2_latent_cache_dir` for stages 1/2. The cache geometry and temporal stride must match the training configuration. Missing episodes and incompatible caches are errors.

Normalization statistics are required for RoboTwin and RoboCasa action training/preparation. LIBERO and DROID can derive statistics from dataset metadata. DROID also requires `meta/modality.json` with joint-position and gripper ranges. See [data and artifacts](docs/data.md).

## Train

Start from one of the six `configs/stage{0,1,2}_{5b,14b}.yaml` files. Select `dataset`, fill `dataset_dirs` and `text_embedding_cache_dir`, and supply the source checkpoint for downstream stages.

```bash
uv run python -m dreamfree.training.stage0 --config configs/stage0_14b.yaml

uv run python -m dreamfree.training.stage1 --config configs/stage1_14b.yaml \
  --set stage0_lora_checkpoint=./outputs/stage0_robotwin_14b/checkpoints/step_020000

uv run python -m dreamfree.training.stage2 --config configs/stage2_14b.yaml \
  --set stage1_checkpoint=./outputs/stage1_robotwin_14b/checkpoints/step_100000 \
  --set action_dit_pretrained_path=/path/to/action_backbone.pt
```

These commands assume you have filled the YAML paths. Stage 2 expects compatible pretrained ActionDiT weights; those weights are not bundled. Use `--print-config` to inspect all resolved settings without loading data or weights.

For distributed training, replace `python -m` with `torchrun --standalone --nproc-per-node=N --module`:

```bash
uv run torchrun --standalone --nproc-per-node=2 --module \
  dreamfree.training.stage1 --config configs/stage1_14b.yaml
```

`gradient_accumulation_steps` counts training microbatches independently of validation-loader state. `--set resume=/path/to/checkpoint` restores a stage's checkpoint; required metadata, tensor files, and optimizer state must exist. Stage 1 reads the stage 0 step and LoRA settings from its source checkpoint; optional explicit expectations are still checked.

## Code map

| Directory | Responsibility |
| --- | --- |
| `src/dreamfree/data/` | Episode indexing, camera composition, normalization, text/video caches |
| `src/dreamfree/models/` | Adaptation, distillation, frozen student, readout, action model, inference optimizations |
| `src/dreamfree/models/wan22/` | Wan model components, VAE, tokenizer, loaders, noise schedules |
| `src/dreamfree/training/stage{0,1,2}/` | Configuration, training loop, validation, checkpoint handling |
| `src/dreamfree/evaluation/` | Shared policy assembly and benchmark-specific rollout adapters |
| `tests/` | Data, loss/gradient, LoRA, accumulation, attention, and deployment contract checks |

Run the local checks with:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```

See [validation notes](docs/validation.md) for the checks performed and their limits, and [third-party components](THIRD_PARTY.md) for upstream code information.
