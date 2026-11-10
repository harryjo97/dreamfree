# Configuration resolution

Training commands use [load_config](../src/dreamfree/config.py). A `Stage0Config`, `Stage1Config`, or `Stage2Config` constructed directly is a schema/default object, not a fully selected experiment. Use `load_config` when reproducing CLI behavior.

## Precedence

The loader first merges the YAML/JSON document with repeated `--set key=value` arguments. The last override for a key wins. It then reads the resulting `model`, `dataset`, and `data_mode` selectors and constructs the configuration in this order (later entries override earlier ones):

1. Stage dataclass defaults in `training/stageN/config.py`.
2. Shared runtime/path defaults in `config.py`.
3. `MODEL_DEFAULTS[model]`: model family, tensor dimensions, and model-specific behavior.
4. `DATA_DEFAULTS[dataset]`: geometry, camera layout, action/state dimensions, and applicable horizon defaults.
5. Explicit file values plus CLI overrides.

Explicit values must still satisfy model and experiment invariants. For example, changing `video_hidden_dim` does not override the width fixed by `model=14b`; it raises an error. Unknown and renamed fields fail with a concrete error. The 14B generic defaults use the low-noise DiT, while selected 14B JSON recipes explicitly select high-noise roles. Choose the actual dataset preset when reproducing a selected run.

## Public selectors and internal fields

| Public setting | Resulting runtime fields |
| --- | --- |
| `model=5b` | `model_id=Wan-AI/Wan2.2-TI2V-5B`, corresponding model family/dimensions |
| `model=14b` | `model_id=Wan-AI/Wan2.2-I2V-A14B`, corresponding model family/dimensions |
| `dataset=libero` (or another supported selector) | `dataset_name` plus the dataset defaults |
| `data_mode=live` / `cached` | `use_cached_latents=false` / `true` |
| Omitted or empty `text_embedding_cache_dir` | `./assets/text/<dataset>_<model>` |

Use `data_mode` in experiment files. Supplying a contradictory `use_cached_latents` value fails. Selecting cached mode does not create caches. Automatic text preparation and explicit video-latent preparation are separate operations; see [data preparation](../README.md#1-select-the-video-data-mode).

Frame counts and stride settings have different units; see the [preset field guide](../configs/README.md#shared-option-names). In particular, nine video samples at stride four span 33 raw frames.

## Parsed configuration versus run configuration

```bash
uv run python -m dreamfree.training.stage2 \
  --config configs/stage2_libero_5b.json \
  --set per_device_batch_size=1 \
  --print-config
```

`--print-config` prints the parsed/default-expanded configuration and exits. It checks types and basic invariants, but does not open source checkpoints, validate all dataset/cache paths, load model weights, create caches, or launch training. Successful printing is not a complete run preflight.

The training loop performs later resolution:

- Stage 1 reads Stage 0 metadata to resolve source expectations and, where configured, inherited noise settings.
- Stage 2 `init_checkpoint` selects the bundled Stage 1 encoder, fills missing normalization statistics, and loads released policy weights with a fresh optimizer. Its source paths can differ from the parsed preset paths. `resume` and `init_checkpoint` are mutually exclusive.
- Stage 2 checks the encoder's model contract against the configured model. Conflicts fail rather than changing the architecture silently.
- If `num_epochs` is set, dataset size and effective global batch determine `max_steps`. Changing world size or dataset selection can therefore change the training schedule. Set `num_epochs=null` for an explicit step budget.

After dataset/budget resolution, the output directory contains `stage0_config.json`, `stage1_config.json`, or Stage 2 `config.json`. Each saved checkpoint's `trainer_state.json` also records the configuration and the resolved source/model contracts. Use these run artifacts and startup logs to inspect what actually ran; use `--print-config` to inspect file/override resolution before startup.

## Paths and incompatible settings

Run documented commands from the repository root. Dataset, cache, and output paths normally resolve from the working directory. Prefer absolute paths when launching from another directory or sharing a configuration across nodes. The retained ActionDiT initialization loader is an exception: use an absolute `action_dit_pretrained_path`, as explained in [ActionDiT initialization](action_backbone.md).

`compile_dit=true` and `use_gradient_checkpointing=true` cannot be combined. A new Stage 2 policy requires an action-backbone input or explicit random initialization. Missing or incompatible video caches never fall back to live video. [Checkpoint formats](data.md#checkpoints) distinguish resumable native checkpoints from portable releases.
