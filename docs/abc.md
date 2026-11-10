# ABC dual-arm YAM

ABC support uses DreamFree's standard three-stage workflow with the Wan2.2-I2V-A14B
high-noise backbone. Stage 1 trains the full student against a separate frozen
teacher. Stage 2 freezes that student and trains the readout, action transformer,
and conditioning layers. Both joint and EEF defaults include eight preceding
actions with history dropout 0.2.

See [environment setup](../README.md#environment-setup) and [data and artifacts](data.md) for shared prerequisites and formats. Read the [shared evaluation requirements](evaluation.md) before simulation or physical deployment.

## Representations

| Setting | Dataset selector | Action vector | Proprioception |
| --- | --- | --- | --- |
| Joint | `yam_abc130k` | 14 absolute values: left six joints + gripper, then right six joints + gripper | Same 14 joint/gripper values |
| EEF | `yam_abc130k_eef` | 20 absolute values: left xyz + rotation6d + gripper, then right xyz + rotation6d + gripper | 14 joint/gripper values |

EEF positions use each arm's base frame and the pinned YAM `tcp_site` at the wrist,
not a fingertip frame. Rotation6d stores the first two rotation-matrix columns in
`R00,R10,R20,R01,R11,R21` order. Both action spaces use train-pool q01/q99 normalization.

The three RGB cameras are top, left wrist, and right wrist. Native 480 × 640 frames
are center-cropped to 456 × 608 before antialiased bilinear resizing. The top view
is 288 × 384; the two 144 × 192 wrist views sit beneath it, yielding 432 × 384.
Images are converted from uint8 to float [0,1] **before** resizing, then to [-1,1].
Actions run at 30 Hz; each sample has 32 action targets and 33 raw video frames,
sampled every four frames for the video backbone. Episode-tail targets are padded
and excluded from the action loss.

All stages retain the prepared `yam_selection.json` train/validation split.
Selection, statistics, cache preprocessing, and EEF kinematics are checked rather
than silently replaced. Stage 2 checkpoints record these immutable contracts.

## Prepare or reuse data

Run the following commands from the DreamFree repository. Replace paths with your
prepared artifacts. The joint dataset and video cache may point to existing AWAM
artifacts; these inputs are read-only during training and EEF conversion.

```bash
export DREAMFREE_ROOT="$PWD"
export ABC_ROOT=/path/to/abc
export JOINT_ROOT=/path/to/prepared/yam_abc130k
export JOINT_CACHE=/path/to/video/cache/yam_abc130k
export EEF_ROOT=/path/to/prepared/yam_abc130k_eef
export TEXT_CACHE="$DREAMFREE_ROOT/assets/text/abc_14b"
export MODEL_BASE=/path/to/models
```

`JOINT_CACHE` should name the directory containing `episode_XXXXXX.safetensors`.
New DreamFree video caches use version 3. Existing AWAM version 2 ABC video caches
are accepted only when their complete ABC preprocessing, normalization, coverage,
model geometry, and official I2V-conditioning contracts match. This exception does
not enable arbitrary version 2 caches. AWAM text caches and AWAM model checkpoints
are not compatible with DreamFree's formats.

If preparing from raw ABC-130k MCAPs, install the optional reader dependencies and
provide the downloaded source and its frozen selection manifest:

```bash
uv sync --locked --extra abc130k
uv run python -m dreamfree.data.prepare_abc \
  --source-root /path/to/raw/ABC-130k \
  --selection-manifest /path/to/frozen_selection.json \
  --output-root "$JOINT_ROOT" --max-gap-ms 200 \
  --scratch-dir /data-vol2/jaehyeong/tmp
```

This writes native-resolution LeRobot v3 data, causal 30 Hz alignment, the frozen
episode split, and `dataset_stats.json`. Use the same explicit maximum alignment
gap as the source experiment. Existing completed prepared datasets need no raw
conversion. See `--help` for selection, resume, and normalization-source options.

Trainers automatically prepare missing DreamFree text embeddings for both action
settings at startup. To prepare them ahead of time instead:

```bash
uv run python -m dreamfree.data.prepare_text \
  --config configs/stage2_abc_joint_14b.json \
  --set dataset_dirs="$JOINT_ROOT" \
  --set text_embedding_cache_dir="$TEXT_CACHE" \
  --set model_base_path="$MODEL_BASE"
```

Video caches are optional for live training. The current EEF converter also
requires a joint video cache to create its action overlays. To prepare a cache
for either use, generate one with a GPU. This writes an
episode directory named after the dataset root beneath `--cache-dir`:

```bash
uv run python -m dreamfree.data.prepare_latents \
  --config configs/stage2_abc_joint_14b.json \
  --set dataset_dirs="$JOINT_ROOT" \
  --set dataset_stats_path="$JOINT_ROOT/dataset_stats.json" \
  --set model_base_path="$MODEL_BASE" \
  --cache-dir "$DREAMFREE_ROOT/assets/latents/abc_14b"
export JOINT_CACHE="$DREAMFREE_ROOT/assets/latents/abc_14b/$(basename "$JOINT_ROOT")"
```

Convert absolute joint targets to EEF targets using the pinned model XML:

```bash
uv run python -m dreamfree.data.prepare_abc_eef \
  --source-root "$JOINT_ROOT" --source-cache "$JOINT_CACHE" \
  --output-root "$EEF_ROOT" \
  --model-path "$ABC_ROOT/abc_sim/models/assets/i2rt_yam/yam.xml" \
  --scratch-dir /data-vol2/jaehyeong/tmp
```

The converter preserves the source, computes EEF statistics from training episodes
only, keeps joint-state statistics, and writes `action_cache` overlays referencing
the existing video latents. Videos are symlinked. Keep the source files available
and unchanged; overlays validate their recorded source identity. Existing AWAM
EEF roots and overlays can also be used when these contracts validate.

## Train

All stages default to `data_mode=live`: video is decoded and VAE-encoded during
training. Stage 0 and Stage 1 use the same joint dataset for both action settings.
Fill the paths below; checkpoints are explicit completed step directories.
Compatible pretrained ActionDiT weights are required by the standard Stage 2
initialization and are not bundled.

```bash
uv run python -m dreamfree.training.stage0 \
  --config configs/stage0_abc_joint_14b.json \
  --set dataset_dirs="$JOINT_ROOT" \
  --set text_embedding_cache_dir="$TEXT_CACHE" \
  --set model_base_path="$MODEL_BASE"

export STAGE0=/path/to/dreamfree/stage0/checkpoints/step_041574
MACHINE_RANK=0 MAIN_PROCESS_IP=10.8.0.103 \
  bash scripts/train_stage1_full.sh configs/stage1_abc_joint_14b.json \
  --set dataset_dirs="$JOINT_ROOT" \
  --set text_embedding_cache_dir="$TEXT_CACHE" \
  --set model_base_path="$MODEL_BASE" \
  --set stage0_lora_checkpoint="$STAGE0"

export STAGE1=/path/to/dreamfree/stage1/checkpoints/step_200000
export ACTION_BACKBONE=/path/to/action_backbone.pt
uv run python -m dreamfree.training.stage2 \
  --config configs/stage2_abc_joint_14b.json \
  --set dataset_dirs="$JOINT_ROOT" \
  --set dataset_stats_path="$JOINT_ROOT/dataset_stats.json" \
  --set text_embedding_cache_dir="$TEXT_CACHE" \
  --set model_base_path="$MODEL_BASE" \
  --set stage1_checkpoint="$STAGE1" \
  --set action_dit_pretrained_path="$ACTION_BACKBONE"

uv run python -m dreamfree.training.stage2 \
  --config configs/stage2_abc_eef_14b.json \
  --set dataset_dirs="$EEF_ROOT" \
  --set dataset_stats_path="$EEF_ROOT/dataset_stats.json" \
  --set text_embedding_cache_dir="$TEXT_CACHE" \
  --set model_base_path="$MODEL_BASE" \
  --set stage1_checkpoint="$STAGE1" \
  --set action_dit_pretrained_path="$ACTION_BACKBONE"
```

Use `--print-config` to inspect resolved settings. Stage 0 and Stage 2 use 16
GPUs across two nodes; full Stage 1 uses the FSDP2 launcher on both nodes with
`MACHINE_RANK=0` and `1`. See the [preset guide](../configs/README.md) for launch
topology and global batch sizes. The Stage 0/1 lineage is shared by joint and
EEF. Stage 1 selects step 200,000 from its 249,441-step budget; joint Stage 2
selects step 40,000 and EEF selects step 30,000 from their 42,074-step budgets.
The JSON presets keep joint and EEF output directories separate. All stages
automatically prepare missing text embeddings in `text_embedding_cache_dir`;
5b and alternative camera/temporal layouts are rejected.

To use existing video caches, add `--set data_mode=cached` and
`--set video_latent_cache_dir="$JOINT_CACHE"` for Stage 0, Stage 1, and joint Stage 2.
EEF Stage 2 uses `--set video_latent_cache_dir="$EEF_ROOT/action_cache"`.
Cached mode requires complete compatible caches and never falls back to live video.

### Action-history Stage 2

The default `configs/stage2_abc_joint_14b.json` and
`configs/stage2_abc_eef_14b.json` already enable action history. The corresponding
`*_history_14b.json` files are equivalent aliases. Both selected AWAM runs use:

```json
{
  "action_history_horizon": 8,
  "action_history_dropout": 0.2
}
```

For example, the joint-space command is:

```bash
uv run python -m dreamfree.training.stage2 \
  --config configs/stage2_abc_joint_history_14b.json \
  --set dataset_dirs="$JOINT_ROOT" \
  --set dataset_stats_path="$JOINT_ROOT/dataset_stats.json" \
  --set text_embedding_cache_dir="$TEXT_CACHE" \
  --set model_base_path="$MODEL_BASE" \
  --set stage1_checkpoint="$STAGE1" \
  --set action_dit_pretrained_path="$ACTION_BACKBONE"
```

For EEF, use the EEF history preset, `dataset_dirs="$EEF_ROOT"`, and
`dataset_stats_path="$EEF_ROOT/dataset_stats.json"`. History works with both
live video and explicit cached mode using the same cache settings above. No new
latent or text caches are needed. Both action spaces reuse the Stage 1 ABC visual backbone.

At frame `t`, history contains normalized actions `[t-H,t)` from the same episode,
ordered oldest to newest. Missing early actions are left-padded with normalized
zeros and masked as self-attention keys. The ActionDiT query is one current-proprio
token, `H` clean history tokens encoded by the shared action encoder, then 32 noisy
future-action tokens. History uses the configured proprio-prefix timestep sigma;
only the future suffix receives flow-matching loss. During training, each sample's
entire history is independently dropped with probability 0.2. Validation/inference
uses all available history. Proprio dropout remains a separate setting.

Override both settings explicitly to change this experiment. A positive horizon
requires an explicit dropout probability in `[0,1)`, and the total query length
must fit the 1024-token RoPE cache. For a standard Stage 2 experiment, explicitly
set `action_history_horizon=0` and `action_history_dropout=null`.

History checkpoints use `dreamfree_stage2_action_history` version 1 with the same
four weight files as standard Stage 2. Their metadata records history length,
dropout, normalization, padding, token order, and execution ownership. Resume
rejects changes to that contract or attempts to resume standard checkpoints as
history checkpoints. AWAM history checkpoints are not directly loadable.

The ABC server detects the variant from the checkpoint; evaluation commands do
not require a separate history flag. Both clients send an explicit empty history
on each episode's first request and retain only commands accepted and issued.
Joint history contains aperture-clipped joint targets. EEF IK history contains
the wrist FK encoding of the issued joint targets; OSC history contains accepted
Cartesian targets. Rejected predictions, unexecuted chunk tails, reset motions,
and retry holds are excluded. Debug mode supplies empty history because it issues
no commands. On the robot client, `--expected-variant action_history` verifies the
intended checkpoint variant. Missing or malformed history fails before inference.

## ABC simulation evaluation

ABC uses a separate Python 3.12 environment. The optional
`bash scripts/setup_abc.sh` installer clones/checks out ABC revision
`d0832d12651d1b260a652861a14648dc5f3660c7`, installs its environment, and prepares
simulation assets. It uses `ABC_ROOT` when set. It refuses tracked local changes;
use a dedicated checkout. Reusing an existing matching installation is sufficient.

First inspect the simulator interface using its interpreter:

```bash
PYTHONPATH="$DREAMFREE_ROOT/src" "$ABC_ROOT/.venv/bin/python" \
  -m dreamfree.evaluation.yam --check-env
```

On the GPU host, select a checkpoint and its matching action space. The CPU-only
check validates metadata, statistics, and source lineage without loading the model:

```bash
export CHECKPOINT=/path/to/dreamfree/stage2/checkpoints/step_200000
export ACTION_SPACE=joint  # Set eef for the EEF checkpoint.
export READY_FILE="$DREAMFREE_ROOT/outputs/abc_${ACTION_SPACE}_ready.json"
uv run python -m dreamfree.evaluation.yam.server \
  --checkpoint-path "$CHECKPOINT" --action-space "$ACTION_SPACE" --check-only

uv run python -m dreamfree.evaluation.yam.server \
  --checkpoint-path "$CHECKPOINT" --action-space "$ACTION_SPACE" \
  --port 6100 --replan-steps 8 --ready-file "$READY_FILE"
```

The server builds DreamFree's optimized CUDA BF16 runtime and publishes readiness
after warmup. In another terminal, run the simulator client with the same variables:

```bash
PYTHONPATH="$DREAMFREE_ROOT/src" "$ABC_ROOT/.venv/bin/python" \
  -m dreamfree.evaluation.yam \
  --ready-file "$READY_FILE" --action-space "$ACTION_SPACE" \
  --episodes 20 --output-dir "$DREAMFREE_ROOT/outputs/eval_abc_${ACTION_SPACE}"
```

`--tasks` selects entries from the bundled suite. Rollouts preserve the ABC success
evaluator, 30 Hz timing, seeded episode resets, per-episode records, videos, and
aggregate summaries. EEF evaluation solves bounded IK before passing 14D joint
commands to ABC; rejected targets abort the rollout. Output directories must not
contain an existing evaluation summary. These commands require the actual simulator
assets and a trained checkpoint; unit tests do not establish benchmark performance.

## Physical deployment

Use the same model server, or its alias `dreamfree.evaluation.yam_real.server`.
The robot machine needs its configured ABC/i2rt environment, camera profiles, and
CAN devices. Keep the pickle RPC endpoint on localhost and use an SSH tunnel when
the GPU host is remote, for example `ssh -N -L 6100:127.0.0.1:6100 GPU_HOST`.

Run a profile/server check from the robot environment:

```bash
PYTHONPATH=/path/to/dreamfree/src python -m dreamfree.evaluation.yam_real.client \
  --abc-root /path/to/abc --robot-profile bbox_config \
  --remote-port 6100 --instruction 'pick up the cup' \
  --expected-checkpoint /absolute/path/on/server/to/stage2/checkpoint \
  --action-space joint --check-only
```

Set `--action-space eef` for EEF checkpoints; `--eef-controller ik` is the default.
The client validates the representation and pinned wrist/frame convention against
the server. Replace `--check-only` with `--debug --print-predicted-actions` to check
camera observations and predictions without starting followers. Removing those
flags enables homing and action execution; `--confirm` requests prompts before
homing and the first action. The default proceeds automatically.

Client-side camera packing uses CPU PyTorch and defaults to BF16 transport.
Explicit `--image-preprocessing server` sends raw uint8 frames when preprocessing
should run on the GPU host. Admission compares client/server preprocessing exactly.
Joint limits, rate/tracking limits, finite-value checks, motor-startup validation,
and rejected-target handling remain active in deployment.

For Cartesian torque control, choose `--action-space eef --eef-controller osc
--osc-config /path/to/calibrated_osc.json`. Templates and the i2rt contract are in
`src/dreamfree/evaluation/yam_real/`. The standard template is not marked as
hardware-validated: mounting, calibration, gains, and limits must describe the
actual robot and pass the existing OSC admission checks. Offline controller tests
do not establish physical calibration or hardware performance.

## Local verification

```bash
ABC_ROOT=/path/to/abc uv run --no-sync pytest -q \
  --basetemp=/data-vol2/jaehyeong/tmp/dreamfree-abc-tests
uv run --no-sync ruff check src tests
```

The tests cover frozen episode splits, normalization, joint/EEF shapes, immutable
conversion overlays, independent wrist FK references, bounded IK, camera-packing
parity, RPC admission, mocked simulation rollouts, and motor startup. History tests
also check causal indexing, episode resets, dropout, future-only loss/gradients,
optimized attention masking, and accepted-command ownership. FK/conversion
checks require the pinned model XML via `ABC_ROOT`. OSC tests additionally require
MuJoCo and skip if it is unavailable. Tests do not run training or move hardware.
