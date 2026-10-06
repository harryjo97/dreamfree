# Validation

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

## Limits

Data tests use small local metadata/parquet fixtures, deterministic decoded pixels, and synthetic VAE outputs. Policy assembly tests mock weight loading. These tests exercise contracts and control flow; they do not establish full-model numerical equivalence or benchmark success rates.

Full 5b/14b GPU training, CUDA graph capture, compiled end-to-end inference, simulator rollouts, and physical robot execution were not run. They require external weights, prepared data, suitable available GPUs, and the benchmark or robot environments. No robot was moved.

DreamFree checkpoints and caches use a new artifact contract. The training examples require paths to newly prepared data and the appropriate source-stage artifacts before they can run.
