"""Validate the portable text-embedding artifact contract."""

import torch


def validate_text_cache(payload, path, *, context_len=128, text_dim=4096):
    if not isinstance(payload, dict) or payload.get("format") != "dreamfree_text_v1":
        raise ValueError(
            f"Expected a DreamFree text cache in {path}; regenerate it with prepare_text."
        )
    (context, mask) = (payload.get("context"), payload.get("mask"))
    if not isinstance(context, torch.Tensor) or tuple(context.shape) != (context_len, text_dim):
        raise ValueError(f"Text cache {path} requires context shape {(context_len, text_dim)}.")
    if (
        not isinstance(mask, torch.Tensor)
        or tuple(mask.shape) != (context_len,)
        or mask.dtype != torch.bool
    ):
        raise ValueError(f"Text cache {path} requires a boolean mask of length {context_len}.")
    if not torch.isfinite(context).all() or not mask.any():
        raise ValueError(f"Text cache {path} contains nonfinite embeddings or an empty mask.")


DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def load_text_context(*, cache_dir, prompt, context_len, text_dim, model_id):
    """Load a prompt cache, zero padding, and expose the full fixed context length."""
    import hashlib
    from pathlib import Path

    suffixes = {"Wan-AI/Wan2.2-TI2V-5B": "wan22ti2v5b", "Wan-AI/Wan2.2-I2V-A14B": "wan22i2va14b"}
    if model_id not in suffixes:
        raise ValueError(f"Unsupported text cache model: {model_id}")
    directories = str(cache_dir).split(",") if isinstance(cache_dir, (str, Path)) else cache_dir
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    paths = [
        Path(directory).expanduser() / f"{digest}.t5_len{context_len}.{suffixes[model_id]}.pt"
        for directory in directories
    ]
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing text embedding cache. Tried: {paths}. Run dreamfree.data.prepare_text."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_text_cache(payload, path, context_len=context_len, text_dim=text_dim)
    if payload.get("model_id") != model_id:
        raise ValueError(f"Text cache model_id mismatch in {path}: expected {model_id}.")
    (context, mask) = (payload["context"].clone(), payload["mask"])
    context[~mask] = 0
    return (context, torch.ones_like(mask))
