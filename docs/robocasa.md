# RoboCasa

Run commands from the repository root after completing [environment setup](../README.md#environment-setup).

## Data and training

| Setting | Value |
| --- | --- |
| Dataset selector | `robocasa` |
| Video H × W | 352 × 352 |
| Action / state dimensions | 7 / 9 |
| Camera order | left agent view, right agent view, wrist |

Provide existing LeRobot dataset roots through `dataset_dirs` (comma-separated for multiple roots). See [dataset preparation](../README.md#dataset-preparation) for the supported dataset formats.

Normalization statistics are required for RoboCasa action training/preparation; set `dataset_stats_path` to the matching statistics file.

RoboCasa shares floating-point image resizing across training, latent preparation, and evaluation.

Use `configs/stage{0,1,2}_robocasa_14b.json` with your `dataset_dirs`. The selected Cosmos-success recipe uses the 14B high-noise backbone, rank-128 LoRA in Stages 0/1, teacher/student layer 31, and 64 Stage 2 readout queries at layer 29. See the [preset guide](../configs/README.md) for checkpoint lineage, training budgets, and GPU/global-batch counts, and the [shared training workflow](../README.md#training) for preparation commands.

## Evaluation

Read the [shared runtime requirements](evaluation.md) before running the evaluator.

Use an environment with the matching RoboCasa/robosuite installation and assets. Set `ROBOCASA_ROOT` and `ROBOCASA_CONTROLLER_CONFIG`, or override their YAML fields with paths.

After [downloading DreamFree-RoboCasa](../README.md#pretrained-checkpoints), run:

```bash
uv run python -m dreamfree.evaluation.robocasa \
  --config configs/eval_robocasa.yaml \
  --set checkpoint_path=./assets/checkpoints/DreamFree-RoboCasa/stage2 \
  --set EVALUATION.task_name=TurnOffMicrowave
```

The release supplies normalization statistics. Training, latent preparation, and evaluation convert pixels to float32 before bilinear resizing with antialiasing. The 352 × 352 mosaic directly resizes the wrist to 176 × 352 above two 176 × 176 agent views. Regenerate existing video caches in a fresh directory to use this processing. Evaluation uses 32-step replanning and 10 denoising steps; set `EVALUATION.dataset_stats_path` when using external statistics.
