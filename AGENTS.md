# Repository guidance

## Start here

- Read [README.md](README.md) for supported workflows, [the architecture map](docs/architecture.md) for component ownership and tensor shapes, and [configuration resolution](docs/configuration.md) before changing experiment settings.
- Use `uv run` with the repository's `.venv`. For an already prepared environment, `uv run --no-sync` avoids changing dependencies during checks. Initial setup is `uv sync --locked`.
- Run documented commands from the repository root. Keep code comments and docstrings in English. Prefer small explicit functions and ordinary modules over new frameworks or registries.
- Inspect the working tree before editing. Preserve unrelated staged, unstaged, and untracked work. Keep training equations and optimized inference behavior unchanged during structural refactors.
- Project variants are `5b` (Wan2.2-TI2V-5B) and `14b` (Wan2.2-I2V-A14B). Selected 14B JSON recipes use the high-noise DiT; generic defaults use low-noise. Use the [dataset preset](configs/README.md) rather than inferring a recipe from a model name.

## Entrypoints and ownership

| Task | Start with |
| --- | --- |
| Training CLI, defaults, overrides | `src/dreamfree/config.py`; `training/stage{0,1,2}/__main__.py` |
| Training execution, resume, validation | `src/dreamfree/training/stage{0,1,2}/{loop,checkpoint,validation}.py` |
| Stage 0 adaptation / Stage 1 loss | `src/dreamfree/models/{adaptation,distillation,full_student}.py` |
| Frozen encoder / Stage 2 loss | `src/dreamfree/models/{student,action_training,action_history}.py` |
| Dataset construction / batch boundary | `src/dreamfree/data/{video,stage2,action_batch}.py` |
| Normalization and cache contracts | `src/dreamfree/data/{action_processing,action_cache,latent_config}.py`; [data source map](docs/data.md#source-map) |
| Offline text/video preparation | `python -m dreamfree.data.prepare_text`; `python -m dreamfree.data.prepare_latents` |
| Shared policy assembly / metadata | `src/dreamfree/evaluation/{policy,checkpoints,options}.py`; `src/dreamfree/models/release.py` |
| Benchmark observation/rollout changes | `src/dreamfree/evaluation/<benchmark>/`; [evaluation guide](docs/evaluation.md) |

For a first source read, follow configuration → one stage loop → its model loss → data boundary → checkpoint loading → benchmark adapter. Generic evaluation helpers belong in the shared modules; benchmark-specific geometry and optimization eligibility stay with the adapter. Policy classes share `evaluation/runtime.py`. RoboTwin support covers data preparation and training only.

## Experimental invariants

- Stage 1 combines video flow matching with weighted normalized-cosine distillation. Teacher targets are detached. LoRA mode shares a base with controlled adapter participation; full mode uses a separate frozen teacher. The projector is for Stage 1 training, not action inference.
- Stage 2 keeps the visual encoder frozen and uses exactly one latent frame for current-token extraction. Gradients reach the readout, conditioner/proprio encoder, and action DiT. Preserve per-sample valid-action averaging and scheduler weighting.
- Preserve tensor layouts, camera order, image flipping/resizing, action coordinates, and dataset normalization. RoboCasa uses the shared floating-point resizing path. DROID history is reanchored to the current planning state. ABC uses its frozen selection/normalization and joint/EEF conventions.
- Padding masks use `True` for padding; attention-validity masks use `True` for attended positions. Text-cache readers zero tokenizer padding but retain the trained full-length attention convention.
- Action history contains issued commands and resets at episode boundaries. Do not substitute an unexecuted prediction tail or mix history ownership between policy and client.
- Explicitly enabled features must run or fail clearly. Cached video mode requires complete compatible caches and never silently switches to live video. Text-cache completion is automatic and separate; invalid existing entries remain errors.
- Native checkpoints and releases have different source dependencies. Follow [artifact contracts](docs/data.md#checkpoints); keep release `stage1/`, `stage2/`, and `provenance/` together. `init_checkpoint` starts a new optimizer; `resume` restores a native run.
- Preserve casts, LoRA merge behavior, attention masks, compilation, and CUDA graph semantics in inference refactors. BF16 LoRA merging changes rounding; use the [validation notes](docs/validation.md) when assessing numerical equivalence.

## Verification

Use focused checks for the affected subsystem, then the full CPU suite for changes shared across stages or benchmarks. These tests use fixtures/mocks where full weights, GPU execution, or external integrations are unavailable; passing them does not establish simulator success or GPU latency.

| Change | Relevant tests under `tests/` |
| --- | --- |
| Configuration or recipes | `test_contracts.py`, `test_config_names.py`, `test_selected_recipes.py` |
| Live/cached data or normalization | `test_data_pipeline.py`, `test_batch_preparation.py`, `test_robocasa_resize.py`, `test_abc_live.py`, `test_droid_history.py` |
| Teacher/student gradients or checkpoint resume | `test_contracts.py`, `test_full_student.py`, `test_lora.py`, `test_training_cpu.py`, `test_gradient_accumulation.py` |
| Releases or policy assembly | `test_release_loading.py`, `test_release_export.py`, `test_policy_loading.py`, `test_abc_eval.py` |
| Inference optimization or history/RPC | `test_optimized_attention.py`, `test_abc_history.py`, `test_droid_history.py`, `test_rpc.py`, `test_droid_molmospaces.py` |

```bash
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q
git diff --check
```

On this cluster, place temporary files under `/data-vol2/jaehyeong/tmp`, not `/tmp`; set `TMPDIR` and a unique pytest `--basetemp` there. Check the summary for skipped external-integration tests. Inspect [validation.md](docs/validation.md) for the separate distributed/GPU checks and their limits. Do not alter test expectations to conceal a behavioral change during a structural refactor.

## Upstream and generated artifacts

[THIRD_PARTY.md](THIRD_PARTY.md) identifies the adapted Wan components in `models/wan22/` and LeRobot code in `data/lerobot/lerobot/`. Read their callers and relevant numerical tests before modifying these shared components. Keep their attribution intact.

Weights, datasets, caches, simulator assets, and run outputs are external/generated artifacts. Configuration files and documentation should explain required inputs rather than rely on machine-specific paths. Reference ActionDiT initialization is a separate external artifact; see [its format and provenance](docs/action_backbone.md).
