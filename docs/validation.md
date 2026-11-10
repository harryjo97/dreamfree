# Validation

## RoboTwin evaluation removal (2026-11-10)

RoboTwin's evaluator, callbacks, configuration, and policy-builder route were
removed. Its data preparation and training remain supported. The retained
encoder/readout/action runtime now lives in `evaluation/runtime.py`; its shared
method logic and queue behavior are unchanged.

- Full CPU suite: **337 passed, 26 skipped**. The new admission test rejects
  RoboTwin before checkpoint loading. Remaining evaluator/server help commands,
  Ruff checks, `git diff --check`, and 98 local documentation links passed.
- Actual H200 GPU inference: LIBERO matched the preceding saved output exactly;
  RoboCasa matched the source immediately before this removal exactly. Both
  completed four predictions. RoboCasa differed from the preceding saved output
  by up to 0.01134307, but the same difference was present before this removal.
- DROID completed six history/seed cases and matched the prior controlled output
  exactly with `cudnn_benchmark=false`. Compilation and both CUDA graphs remained
  enabled. Repository inference defaults were not changed.

These were synthetic-input policy checks, not simulator rollouts or ABC/YAM GPU
validation. Artifacts are retained under
`/data-vol2/jaehyeong/tmp/dreamfree-remove-robotwin-eval-20261110/`.

## Post-refactor evaluation check (2026-11-10)

Actual released LIBERO, RoboCasa, and DROID policies loaded and generated finite
actions on `nipa3`, GPU 0 (H200), with BF16 and the configured optimized runtime.
The checks used fixed synthetic camera/state inputs and text embeddings, not
simulator rollouts.

- LIBERO: four predictions matched the saved earlier DreamFree audit exactly.
- RoboCasa: the immediately pre-refactor and current source produced identical
  actions. Both differ from the older saved audit by up to **0.01501477**; that
  older discrepancy remains unexplained.
- DROID: all six empty/partial/full history and episode-seed cases ran. Default
  `cudnn_benchmark=true` was not bit-stable between processes: repeating the same
  current code differed by up to **0.01192880**. In a controlled comparison with
  `cudnn_benchmark=false`, pre-refactor and current outputs matched exactly,
  with compilation and both CUDA graphs still enabled. Defaults were not changed.
- RoboTwin: the audit found a startup failure because shared policy assembly called
  `_validate_dreamfree_stage2_for_eval`, which its adapter did not define.
  This missing validator predated the refactor and was reproduced separately.
  RoboTwin evaluation was subsequently removed; data preparation and training remain supported.

These results do not establish simulator task success, ABC/YAM inference, text
encoding, RPC, or physical deployment. Scripts, reports, logs, and the preserved
pre-refactor source are retained locally under
`/data-vol2/jaehyeong/tmp/dreamfree-eval-postrefactor-20261110/`. All GPU workers
exited after the checks.

## Readability and module refactor (2026-11-10)

- The full CPU suite passed with **336 passed, 26 skipped** in the environment
  used for this refactor. Optional external integrations were skipped; this run
  did not set `ABC_ROOT`.
- Eleven new batch-boundary cases cover live/cached equivalence for both video
  families, mixed-batch row ordering, detached cached latents, required 14B
  conditioning, and invalid padding/history inputs.
- Syntax-tree comparisons against the pre-edit working tree confirmed unchanged
  logic in 112 moved data functions/classes and 23 moved evaluation helpers,
  after accounting for import paths, renamed functions, and documentation.
- Ruff lint/format checks, Python compilation, and `git diff --check` passed.
  All 128 local links/anchors in the edited navigation guides resolved. Seven
  documented training commands parsed successfully with `--print-config`;
  this did not load datasets, checkpoints, or model weights.

These checks cover the source reorganization and batch contracts. They do not
establish full-model GPU equivalence, memory requirements, simulator success,
or execution of the documented two-step GPU quickstarts.

## CPU training checks after configuration cleanup (2026-11-10)

All seven recipe chains passed Stage 0 → Stage 1 → Stage 2 training on CPU:

| Recipe | Stage 0 | Stage 1 | Stage 2 |
| --- | --- | --- | --- |
| LIBERO 5B | Passed | Passed, LoRA | Passed |
| RoboCasa 14B | Passed | Passed, LoRA | Passed |
| DROID 14B | Passed | Passed, full student | Passed, action history |
| RoboTwin 5B | Passed | Passed, LoRA | Passed |
| RoboTwin 14B | Passed | Passed, LoRA | Passed |
| ABC/YAM joint 14B | Passed | Passed, full student | Passed, action history |
| ABC/YAM EEF 14B | Passed, shared joint recipe | Passed, shared joint recipe | Passed, action history |

[The CPU integration test](../tests/test_training_cpu.py) runs the production
training loops, losses, optimizers, schedulers, validation, and checkpoint loaders.
Each stage performs two optimizer steps with gradient accumulation, validates and
saves checkpoints, then resumes for a third optimizer step. Each downstream
stage loads the preceding stage's newly saved checkpoint. Saved optimizer states
are nonempty and finite. Both LoRA and full-student checkpoint paths pass.

These loop tests substitute tiny Wan DiTs, a deterministic VAE fixture, and
synthetic dataset batches. They preserve each recipe's image geometry,
action/proprio dimensions, conditioning, and action-history settings. They use
FP32 and disable compilation, gradient checkpointing, and qualitative video
generation. They do not load the full pretrained backbones or ActionDiT weights.

Additional checks passed:

- The complete CPU suite: **349 passed, 2 skipped**, with ABC dependencies enabled
  through `ABC_ROOT`. No dataset in the training matrix was skipped. Existing
  tests cover live and cached dataset readers, cache contracts, normalization,
  splits, padding, and action history. Two outdated ABC test assertions were
  corrected to expect the selected recipes' eight-action history; no trainer
  code fix was needed.
- Actual CPU TorchCodec decoding of nine frames from every camera in one local
  episode for LIBERO, RoboCasa, DROID, RoboTwin, and ABC/YAM. ABC joint and EEF
  share the same videos. This checks the installed decoder and sampled files,
  not every episode in the datasets.
- Two-process CPU/Gloo gradient accumulation matched the global-batch reference.
  Shared text-cache creation, reuse, and failure propagation also passed with
  two processes.
- Ruff lint/format checks, shell syntax checks, and `git diff --check` passed.

Reproduce the suite from the repository root:

```bash
# ABC_ROOT must point to the ABC checkout installed by scripts/setup_abc.sh.
CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  uv run --no-sync pytest -q
```

No GPU was used for this audit. It does not verify full-model GPU memory use,
BF16 kernels, compilation, gradient checkpointing, or multi-node FSDP2 execution.
Training presets still require real dataset/model paths and compatible source
checkpoints; use newly prepared DreamFree caches when selecting cached mode.

## Hugging Face releases (2026-11-10)

- DROID, RoboCasa, and LIBERO repositories are public and ungated. Their published JSON files match the release copies used for validation.
- All three actual released encoders loaded on CPU in BF16 without Stage 0 or upstream DiT weights: 1,095 tensors each for DROID/RoboCasa and 825 for LIBERO.
- Strict tensor-name and shape checks passed for the encoder, readout, action DiT, conditioner, and proprio encoder: 1,971 tensors each for DROID/RoboCasa and 1,701 for LIBERO. Metadata, source lineage, and bundled normalization also passed.
- Release tests cover relocation, automatic policy/statistics discovery, standalone sharded loading, missing files, incompatible metadata, DROID history semantics, and RoboCasa preprocessing. They also cover Stage 2 initialization from released policy weights, frozen standalone encoders, and the resulting checkpoint metadata. The full suite passed with **291 passed, 24 skipped**; Ruff lint/format and `git diff --check` passed.
- RoboCasa live-video training, Stage 2 training, latent preparation, and evaluation produce identical floating-point mosaics on the tested inputs. Tests cover direct wrist resizing at 352 and 448 pixels and preservation of fractional pixel values.
- Configuration and source validation passed against all three actual released bundles for Stage 2 fine-tuning. Fine-tuning initializes the policy weights with a fresh optimizer; the releases do not contain optimizer state.

### Numerical checks against the selected AWAM checkpoints

All four Stage 2 weight files match their AWAM sources by SHA-256 for each release.
Every released encoder tensor matches the BF16 source: directly converted full
weights for DROID and reconstructed Stage 0/Stage 1 LoRA merges for RoboCasa and
LIBERO. All compared tensors are finite.

| Release | Encoder tensors | Maximum encoder difference | Readout forward | ActionDiT velocity forward |
| --- | --- | --- | --- | --- |
| DROID 14B | 1,095 | 0 | Exact | Exact |
| RoboCasa 14B | 1,095 | 0 | Exact | Exact |
| LIBERO 5B | 825 | 0 | Exact | Exact |

The forward comparisons use the actual released readout and 30-layer ActionDiT
weights with fixed synthetic BF16 inputs at the checkpoint's dimensions. These
CPU checks establish export and component agreement; they do not measure task
success rates.

### Full GPU inference and latency

Checked on `nipa2`, GPU 6 (NVIDIA H200), with PyTorch 2.7.1 / CUDA 12.8,
BF16, and batch size 1. Each comparison uses identical synthetic camera images,
state, and fixed text embeddings with the selected AWAM and released DreamFree
weights. The measurements include image processing, VAE, visual encoder,
readout, and action denoising. They exclude text encoding, RPC, and simulator or
robot execution. Latencies are medians of 20 calls after compilation and warmup.

| Policy / history | Denoising steps | AWAM median (ms) | DreamFree median (ms) | Maximum action difference |
| --- | --- | --- | --- | --- |
| DROID, empty history | 24 | 140.48 | 140.95 | 0 |
| DROID, four valid history actions | 24 | 140.06 | 140.74 | 0 |
| DROID, eight valid history actions | 24 | 168.38 | 168.78 | 0 |
| RoboCasa | 10 | 154.14 | 149.99 | 0 |
| LIBERO, merged LoRA reference | 10 | 145.49 | 132.66 | 0 |

DROID actions match bit for bit for each history case with episode seeds 0 and 1.
Both implementations capture the masked and unmasked action CUDA graphs and the
Stage 1 CUDA graph. The audit corrected DreamFree's action-noise seed namespace,
removal of the all-valid history mask, and warmup coverage of both graph variants.

RoboCasa uses the requested direct floating-point resizing in **both** implementations
for this comparison. This checks model/runtime agreement with the shared pixels;
it does not reproduce the archived evaluation's wrist composition or success rate.

LIBERO's recorded AWAM runtime keeps Stage 1 LoRA separate. That runtime measured
158.31 ms, and its returned actions differ from the merged release by a maximum
of **0.0439453** and a mean absolute difference of **0.0089553** on the tested input.
Merging LoRA in AWAM alone produces this difference; the resulting AWAM actions
match DreamFree exactly. The release therefore matches the merged runtime, but
is **not bit-exact with the historical unmerged runtime**. BF16 merging changes
the rounding path. Reproducing that unmerged path requires retaining the adapted
base and separate Stage 1 adapter weights. The 5B optimized runtime also compiles
its cached attention projections to match normalization rounding in the reference
compiled blocks. Closed-loop evaluation is needed to assess the merge's effect
on task success.

These timings describe this offline comparison; they are not a reproduction of
the paper's 91 ms measurement. No simulator rollout or physical action was run.
All audit GPU workers exited. Reports, output arrays, logs, and reproduction
scripts are retained locally in `outputs/release_validation_20261110/` (ignored
by Git). The scripts require the local AWAM source checkpoints and release copies.

## Selected AWAM recipes (2026-11-10)

The dataset presets were traced through the supplied Stage 2 checkpoints to their
Stage 1 and Stage 0 sources. The [preset guide](../configs/README.md) records the
lineages, selected steps, budgets, and reference global batches.

- `OMP_NUM_THREADS=1 uv run pytest -q`: **264 passed, 23 skipped**. New CPU checks
  cover full-student gradients with an unchanged frozen teacher, full-weight
  checkpoint/optimizer resume, frozen Stage 2 and deployment loading, DROID
  history normalization against the training cache path, and MolmoSpaces history
  ownership/reset over RPC.
- Ruff lint/format, shell syntax, preset loading, local documentation links, and
  FSDP2 launcher argument parsing passed.
- These initial preset checks used mocked CUDA operations for warmup variant
  selection. They did not execute the full-student two-node FSDP2 path, actual
  CUDA graph capture, full-model inference, simulator rollouts, or physical
  deployment. The subsequent release checks are described above; the earlier
  training checks below used a different recipe.

## Earlier baseline checks

Checked on 2026-10-06 with the locked Python 3.11 / PyTorch 2.7.1 environment.

## Passed

- `uv run pytest`: **104 tests passed**. Coverage includes configuration parsing and trainer validation for all three stages, both backbones, and all four datasets; live-data indexing and camera composition; cache writer/reader round trips; matching valid action/state targets in live and cached modes; text preparation; LoRA loading and merging; loss/gradient and attention equivalence; accumulation; policy assembly; RPC; and physical-client input checks.
- `uv run ruff check src tests` and `uv run ruff format --check src tests`.
- All 12 public command entrypoints accept `--help` without simulator installations or model weights.
- Static package import/attribute and explicit keyword-argument checks found no unresolved references.
- A two-process CPU/Gloo training check matched the global-batch SGD reference, including accumulation after a validation loader reaches its end.
- A comparison against the reference implementation matched the teacher-state loss and gradients, flow-matching schedule, and all four camera-composition paths exactly on the tested inputs.

Reproduce the distributed check from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' uv run torchrun --standalone --nproc-per-node=2 tests/smoke_distributed.py
```

## Eight-GPU DROID check

On 2026-10-06, all three `14b` stages passed on `nipa4`, using GPUs 0–7 (eight H200 GPUs). Each stage ran two optimizer steps with live video, saved a checkpoint, then resumed for two more steps with cached latents. All six `torchrun` launches exited successfully, each using eight ranks.

The runs used the full Wan2.2-A14B low-noise backbone, 352×640 video, a 32-step action horizon, BF16, batch size 1 per GPU, and gradient accumulation of 2 (effective batch size 16). Gradient checkpointing was enabled; compilation was disabled. Stage 1 used teacher/student layer 31, and Stage 2 used the standard 30-layer ActionDiT.

The bounded dataset contained 65 real frames from each of eight DROID episodes, with original camera videos and normalization statistics. Six episodes were used for training and two for validation. Text embeddings and complete episode latents were prepared afresh; latent preparation also passed with eight ranks. Cached Stage 2 reported complete train/validation coverage, including episode tails, with no video fallback.

| Stage | Live steps 1–2 | Cached resume steps 3–4 | Final live validation loss | Final cached validation loss |
| --- | --- | --- | --- | --- |
| 0 | Passed | Passed | 0.1506 | 0.1503 |
| 1 | Passed | Passed | 0.9276 | 0.8876 |
| 2 | Passed | Passed | 1.8908 | 2.4730 |

Every run validated after each optimizer step and saved model, optimizer, and scheduler state. The checkpoint audit verified finite weights and logged losses/gradient norms, optimizer and scheduler steps of 2 or 4, and changed parameters after every resume. All GPU workers exited, leaving all eight GPUs idle.

The real-data check led to three fixes: validate DROID's raw 28-dimensional actions and 14-dimensional states before selecting the policy's eight dimensions; gather scalar validation metrics as rank scalars; and explicitly close the distributed process group. The DROID regression fixture now uses the actual raw dimensions and modality ranges.

Configs, logs, the subset preparation script, checkpoints, and the audit are retained locally in `outputs/droid_distributed_14b_20261006/` (ignored by Git). From the repository root, the saved launcher reproduces each run:

```bash
# Requires the local smoke dataset, caches, and pretrained weights.
set -e
for stage in 0 1 2; do
  bash outputs/droid_distributed_14b_20261006/run_stage.sh "$stage" live
done
for stage in 0 1 2; do
  bash outputs/droid_distributed_14b_20261006/run_stage.sh "$stage" cached
done
CUDA_VISIBLE_DEVICES='' uv run python outputs/droid_distributed_14b_20261006/audit_checkpoints.py
```

The resume checks extended the stop limit from 2 to 4 while restoring the saved optimizer and scheduler, including the original cosine horizon. They establish successful continuation, not bitwise equivalence to an uninterrupted run. The losses above are smoke-test observations, not convergence or policy-quality results. Pretrained weight loading took several minutes per launch on shared storage.

## Limits

CPU data unit tests use small local metadata/parquet fixtures, deterministic decoded pixels, and synthetic VAE outputs. Some policy assembly tests mock weight loading; the release audits above use actual weights. GPU inference comparisons use fixed offline inputs, while the earlier training smoke checks use real DROID data. Numerical agreement applies to the tested inputs and runtimes and does not establish benchmark success rates.

Long training runs, 5B GPU training, optimizer steps initialized from the public releases, qualitative video generation, simulator rollouts, and physical robot execution were not run in this release audit. No robot was moved.

DreamFree checkpoints and caches use a new artifact contract. The training examples require paths to newly prepared data and the appropriate source-stage artifacts before they can run.
