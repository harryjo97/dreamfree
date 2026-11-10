"""Check rank-0 text preparation and shared failures with two CPU/Gloo ranks."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator

from dreamfree.config import load_config
from dreamfree.data import prepare_text


def main():
    accelerator = Accelerator(cpu=True)
    root = Path(sys.argv[1])
    if accelerator.is_main_process:
        (root / "dataset/meta").mkdir(parents=True)
        (root / "dataset/meta/tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "pick up the cup"}) + "\n"
        )
    accelerator.wait_for_everyone()
    cfg = load_config(2, Path(__file__).resolve().parents[1] / "configs/stage2_14b.yaml")
    cfg.dataset_dirs = str(root / "dataset")
    cfg.text_embedding_cache_dir = str(root / "text")
    weights = SimpleNamespace(path="fixture", download_if_necessary=lambda: None)
    prepare_text._resolve_configs = lambda *a, **k: (None, weights, None, weights)

    class Encoder:
        def eval(self):
            return self

        def __call__(self, ids, mask):
            return torch.ones(len(ids), 128, 4096, dtype=torch.bfloat16)

    def load(*args, **kwargs):
        assert accelerator.is_main_process
        assert kwargs["pretrained_load_broadcast"] is False
        (root / "encoder-loaded.json").write_text(json.dumps({"rank": accelerator.process_index}))
        return Encoder()

    class Tokenizer:
        def __init__(self, **kwargs):
            pass

        def __call__(self, prompts, **kwargs):
            mask = torch.arange(128).expand(len(prompts), -1) < 8
            return torch.zeros_like(mask, dtype=torch.long), mask

    prepare_text._load_registered_model = load
    prepare_text.HuggingfaceTokenizer = Tokenizer
    prepare_text.ensure_training_text_cache(cfg, accelerator)
    assert len(list((root / "text").glob("*.pt"))) == 1
    assert json.loads((root / "encoder-loaded.json").read_text()) == {"rank": 0}

    def fail_load(*args, **kwargs):
        raise AssertionError("Existing embeddings must not reload the encoder.")

    prepare_text._load_registered_model = fail_load
    prepare_text.ensure_training_text_cache(cfg, accelerator)
    if accelerator.is_main_process:
        torch.save({"format": "invalid"}, next((root / "text").glob("*.pt")))
    accelerator.wait_for_everyone()
    try:
        prepare_text.ensure_training_text_cache(cfg, accelerator)
    except RuntimeError as exc:
        assert "Automatic text-cache preparation failed" in str(exc)
    else:
        raise AssertionError("Every rank must reject an invalid existing cache.")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Two-rank text-cache creation, reuse, and failure propagation passed.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
