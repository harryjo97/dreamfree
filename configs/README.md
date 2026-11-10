# Training presets

These dataset-specific presets follow the selected AWAM Stage 2 runs and the
Stage 1 / Stage 0 checkpoints recorded in their configurations. All Stage 2
recipes freeze the visual encoder and train the readout, action DiT, and
conditioner. The 14B recipes use Wan2.2-I2V-A14B's **high-noise** DiT.

| Dataset | Presets (replace `N` with 0, 1, or 2) | Stage 1 | Teacher / student layer | Stage 2 readout layer | History |
| --- | --- | --- | --- | --- | --- |
| LIBERO | `stageN_libero_5b.json` | LoRA, rank 128 | 25 / 25 | 25 | None |
| RoboCasa | `stageN_robocasa_14b.json` | LoRA, rank 128 | 31 / 31 | 29 | None |
| DROID | `stageN_droid_14b.json` | Full student | 31 / 31 | 31 | 8 actions, dropout 0.2 |
| ABC/YAM joint | `stageN_abc_joint_14b.json` | Full student | 31 / 31 | 31 | 8 actions, dropout 0.2 |
| ABC/YAM EEF | `stage2_abc_eef_14b.json`; joint presets for Stages 0/1 | Shared with joint | 31 / 31 | 31 | 8 actions, dropout 0.2 |

Layer indices are zero-based. Every selected Stage 0 uses LoRA rank/alpha 128;
every Stage 2 uses 64 readout queries. Stage 1 distills future tokens only,
restricts future-token attention to itself, and conditions the first frame on
text. Teacher sigma is fixed at 0.7 for LIBERO/RoboCasa and 0.8 for DROID/ABC.
The ABC `*_history_14b.json` files are equivalent aliases of the joint/EEF defaults.

LIBERO has only a 5B dataset preset; RoboCasa, DROID, and ABC have only 14B dataset
presets. The six original `stage{0,1,2}_{5b,14b}.yaml` files remain **RoboTwin
templates**, unchanged pending selection of a RoboTwin run.

## Checkpoint lineage and training budget

| Dataset | Selected Stage 0 step | Selected Stage 1 step | Selected Stage 2 step |
| --- | --- | --- | --- |
| LIBERO 5B | 6,894 | 80,000 | 42,869 |
| RoboCasa 14B | 17,269 | 74,226 | 12,245 |
| DROID 14B | 51,554 | 301,265 | 57,031 |
| ABC joint 14B | 41,574 | 200,000 | 40,000 |
| ABC EEF 14B | 41,574 | 200,000 | 30,000 |

[awam_recipe_sources.json](awam_recipe_sources.json) records the original absolute
checkpoint paths, dataset roots, original model/cache/statistics paths, GPU counts,
and budgets. The presets preserve
the original epoch/step budgets, optimizer settings, and schedules. In particular,
ABC Stage 1 has a 249,441-step budget, and both ABC Stage 2 variants have a
42,074-step budget; the selected checkpoints are intermediate snapshots.
Do not replace these budgets with the selected step just to stop earlier: that
changes warmup or cosine decay. Save and select the listed intermediate step.
Epoch-based budgets are recomputed from the supplied dataset and global batch.

## Paths and input preparation

Fill `dataset_dirs` and `action_dit_pretrained_path`; provide matching
`dataset_stats_path` where required. Match the source data selection: LIBERO
uses the four no-noop suites, RoboCasa uses the Cosmos success dataset, DROID uses
the full LeRobot dataset, and ABC uses the prepared 100 h train / 5 h validation
selection with gap 200. The EEF preset uses that selection's EEF conversion.

Output directories and downstream checkpoint paths are relative DreamFree
paths. They refer to **new DreamFree training outputs** at the selected steps.
AWAM checkpoints are provenance references, not directly loadable DreamFree
artifacts; do not substitute their paths without converting the checkpoint
format and metadata. Pretrained ActionDiT weights are external inputs.

The original runs used cached latents. These portable presets retain DreamFree's
`data_mode=live` default so they work after filling dataset/model paths without
assuming a prepared video cache. To use cached inputs, follow the
[cache preparation guide](../README.md#3-video-latents-prepare-before-using-cached-mode),
then set `data_mode=cached` and `video_latent_cache_dir` for any stage. Text-cache
preparation remains automatic. WandB is disabled and local output/cache paths
replace machine-specific paths; training hyperparameters are retained.

## Shared option names

See [configuration resolution](../docs/configuration.md) for default precedence,
public/internal field mappings, and what `--print-config` does and does not validate.

Use the same names for shared settings in Stages 0, 1, and 2:

| Option | Meaning |
| --- | --- |
| `video_latent_cache_dir` | Precomputed video latents shared by all three stages |
| `text_embedding_cache_dir` | Cached task-text embeddings |
| `camera_layout` | How camera views are arranged into the video input |
| `val_fraction` | Fraction of episodes reserved for validation |
| `tokenizer_max_len` | Text token limit, also used by the text cache |
| `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_target_modules` | LoRA settings for the current training stage (0 or 1) |

Stage 2 and latent preparation use `dataset_stats_path` for normalization statistics.
Stage 1 uses `distillation_loss` and `distillation_loss_weight` for the distillation
objective. Stage 2 uses `merge_student_lora` to merge the frozen encoder's adapters
when loading it, including during training.

Existing checkpoints translate older field names when loaded; published weights
and existing latent/text caches do not need to be exported again. New training
configs must use the names above. Passing an old field gives an error naming its
replacement. Original AWAM metadata in `awam_recipe_sources.json` retains its source names.

Frame and stride settings remain distinct because they measure different quantities:
`clip_frames` counts sampled video frames in Stages 0/1; Stage 2's `num_frames`
counts the window before video subsampling. With stride 4, 9 sampled frames span
33 frames. `sample_stride_frames` advances the window start, while
`global_sample_stride` changes the sampling interval for actions and video.
`raw_frame_stride` measures raw video frames between samples; Stage 2's
`action_video_freq_ratio` measures action samples per video sample.

## Distributed training

The presets retain **per-GPU** batch size and gradient accumulation. Global batch
is `per_device_batch_size × gradient_accumulation_steps × world_size`.

| Dataset | Stage 0: GPUs / global batch | Stage 1: GPUs / global batch | Stage 2: GPUs / global batch |
| --- | --- | --- | --- |
| LIBERO 5B | 4 / 128 | 4 / 128 | 1 / 64 |
| RoboCasa 14B | 4 / 32 | 2 / 32 | 2 / 512 |
| DROID 14B | 8 / 32 | 16 / 64 | 16 / 4,096 |
| ABC joint/EEF 14B | 16 / 64 | 16 / 64 | 16 / 4,096 |

Use `torchrun` for Stage 0, LoRA Stage 1, and Stage 2, with the listed total GPU
count. For full Stage 1, the selected runs use FSDP2 across two eight-GPU nodes
(shard size 8, replication size 2), FP32 trainable shards, BF16 computation, and
a separate frozen BF16 teacher. The supplied launcher sets this topology:

```bash
# Run on both nodes, setting MACHINE_RANK=0 on the first and 1 on the second.
MACHINE_RANK=0 MAIN_PROCESS_IP=10.8.0.103 \
  bash scripts/train_stage1_full.sh configs/stage1_droid_14b.json \
  --set dataset_dirs=/path/to/droid_lerobot
```

For ABC, use `configs/stage1_abc_joint_14b.json`. Both presets synchronize student
shards on each microbatch while accumulating the global optimizer step. The
teacher stops at the target layer and stays replicated. ABC uses an explicit
2,000-step warmup; DROID uses its configured warmup ratio. Source model weights,
datasets, and output directories must be accessible from both nodes. Match the
global batch if adapting the launch topology; the presets do not rescale it.

Full Stage 1 saves `student_dit.safetensors` and projector weights; LoRA Stage 1
saves `stage1_lora.safetensors` and projector weights. Both feed the frozen
Stage 2 loader and evaluation runtime. FSDP2 optimizer/scheduler resume uses
`fsdp2_state/`; model weights remain portable. Resume rejects a different
optimizer checkpoint backend unless `resume_optimizer_state=false` is explicit.
