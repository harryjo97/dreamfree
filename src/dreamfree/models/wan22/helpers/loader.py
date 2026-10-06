from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from dreamfree.models.wan22.helpers.distributed_load import (
    broadcast_module_state,
    get_distributed_rank,
    get_pretrained_broadcast_device,
    get_pretrained_broadcast_sync_device,
    should_broadcast_pretrained_load,
    sync_distributed_load_error,
)
from dreamfree.models.wan22.helpers.io import ModelConfig, hash_model_file, load_state_dict
from dreamfree.models.wan22.helpers.state_dict_converters import (
    wan_video_dit_state_dict_converter,
    wan_video_vae_state_dict_converter,
)
from dreamfree.models.wan22.wan_video_dit import WanVideoDiT
from dreamfree.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from dreamfree.models.wan22.wan_video_vae import WanVideoVAE, WanVideoVAE38
from dreamfree.utils.logging_config import get_logger

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"
PRECONSTRUCTED_DIT_SENTINEL = "PRECONSTRUCTED_DIT"
WAN_TEXT_ENCODER_MODEL_NAME = "wan_video_text_encoder"
TEXT_ENCODER_FAST_LOAD_ENV = "DREAMFREE_TEXT_ENCODER_FAST_LOAD"
WAN22_I2V_A14B_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B"
WAN22_A14B_DIT_ROLES = {"low_noise", "high_noise"}
WAN22_A14B_PREPARED_CACHE_ENV = "DREAMFREE_WAN22_A14B_PREPARED_DIR"
WAN22_A14B_PREPARED_CACHE_FORMAT = "dreamfree_wan22_14b_dit_cache_v1"
WAN22_A14B_PREPARED_CACHE_METADATA = "metadata.json"
WAN22_A14B_DIT_CONFIG_CONTRACT = {
    "model_type": "i2v",
    "in_dim": 36,
    "out_dim": 16,
    "dim": 5120,
    "ffn_dim": 13824,
    "num_heads": 40,
    "num_layers": 40,
}


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE | WanVideoVAE38 | None
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str | None
    text_encoder_path: str | None
    tokenizer_path: str | None


WAN22_MODEL_REGISTRY = [
    {
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
    {
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": WanVideoDiT,
    },
    {
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE38,
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")
    validated = dict(dit_config)
    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)
    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. Allowed keys: {sorted(allowed_keys)}"
        )
    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. Please specify all required WanVideoDiT constructor args."
        )
    return validated


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")


def _log_text_encoder_phase(phase: str, start_time: float) -> None:
    logger.info(
        "Wan text encoder load phase `%s` took %.2f seconds.", phase, time.time() - start_time
    )


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
    pretrained_load_broadcast: bool = True,
):
    overall_start = time.time()
    fast_text_encoder = model_name == WAN_TEXT_ENCODER_MODEL_NAME and _env_bool(
        TEXT_ENCODER_FAST_LOAD_ENV
    )
    use_broadcast = should_broadcast_pretrained_load(pretrained_load_broadcast)
    if fast_text_encoder and use_broadcast:
        logger.info(
            "%s=true: loading the text encoder independently on each rank instead of broadcasting it.",
            TEXT_ENCODER_FAST_LOAD_ENV,
        )
        use_broadcast = False
    rank = get_distributed_rank()
    matched_config = _match_registry_by_name(model_name) if use_broadcast and rank != 0 else None
    source_error: BaseException | None = None
    registry_start = time.time()
    if not use_broadcast or rank == 0:
        try:
            matched_config = _match_registry_by_hash(path, model_name)
        except BaseException as exc:
            source_error = exc
    if use_broadcast:
        sync_distributed_load_error(
            source_error, device=device, src=0, context=f"registry validation for {model_name}"
        )
    elif source_error is not None:
        raise source_error
    if matched_config is None:
        raise RuntimeError(f"Internal error: no registry config resolved for {model_name}.")
    if model_name == WAN_TEXT_ENCODER_MODEL_NAME:
        _log_text_encoder_phase("registry_validation", registry_start)
    model_class = matched_config["model_class"]
    model_kwargs = dict(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")
    if fast_text_encoder:
        if state_dict_converter is not None:
            raise ValueError(
                f"{TEXT_ENCODER_FAST_LOAD_ENV}=true does not support converted text encoder checkpoints."
            )
        model = _load_text_encoder_fast_from_safetensors(
            path=path,
            model_class=model_class,
            model_kwargs=model_kwargs,
            torch_dtype=torch_dtype,
            device=device,
        )
        _log_text_encoder_phase("total_fast_load", overall_start)
        return model
    if use_broadcast:
        broadcast_device = get_pretrained_broadcast_device()
        load_device = "cpu" if broadcast_device == "cpu" else device
        construct_start = time.time()
        model = model_class(**model_kwargs).to(device=load_device, dtype=torch_dtype)
        if model_name == WAN_TEXT_ENCODER_MODEL_NAME:
            _log_text_encoder_phase("construct_module", construct_start)
        load_error: BaseException | None = None
        if rank == 0:
            try:
                state_load_start = time.time()
                state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
                if model_name == WAN_TEXT_ENCODER_MODEL_NAME:
                    _log_text_encoder_phase("read_state_dict_cpu", state_load_start)
                if state_dict_converter is not None:
                    state_dict = state_dict_converter(state_dict)
                assign_start = time.time()
                if broadcast_device == "cpu":
                    _load_state_dict_with_assign(model, state_dict, strict=False)
                else:
                    model.load_state_dict(state_dict, strict=False)
                if model_name == WAN_TEXT_ENCODER_MODEL_NAME:
                    _log_text_encoder_phase("assign_state_dict", assign_start)
            except BaseException as exc:
                load_error = exc
        sync_distributed_load_error(
            load_error, device=load_device, src=0, context=f"rank0 load for {model_name}"
        )
        broadcast_module_state(model, src=0)
        if broadcast_device == "cpu":
            transfer_start = time.time()
            model = model.to(device=device, dtype=torch_dtype)
            if model_name == WAN_TEXT_ENCODER_MODEL_NAME:
                _log_text_encoder_phase("transfer_to_device", transfer_start)
        logger.info("Loaded %s via rank0 %s broadcast.", model_name, broadcast_device)
        if model_name == WAN_TEXT_ENCODER_MODEL_NAME:
            _log_text_encoder_phase("total_broadcast_load", overall_start)
        return model
    construct_start = time.time()
    model = model_class(**model_kwargs)
    if model_name == WAN_TEXT_ENCODER_MODEL_NAME:
        _log_text_encoder_phase("construct_module", construct_start)
    state_load_start = time.time()
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if model_name == WAN_TEXT_ENCODER_MODEL_NAME:
        _log_text_encoder_phase("read_state_dict_cpu", state_load_start)
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)
    assign_start = time.time()
    _load_state_dict_with_assign(model, state_dict, strict=False)
    if model_name == WAN_TEXT_ENCODER_MODEL_NAME:
        _log_text_encoder_phase("assign_state_dict", assign_start)
    transfer_start = time.time()
    model = model.to(device=device, dtype=torch_dtype)
    if model_name == WAN_TEXT_ENCODER_MODEL_NAME:
        _log_text_encoder_phase("transfer_to_device", transfer_start)
        _log_text_encoder_phase("total_standard_load", overall_start)
    return model


def _match_registry_by_hash(path, model_name: str) -> dict[str, Any]:
    model_hash = hash_model_file(path)
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            return config
    raise ValueError(
        f"Cannot detect model type for {model_name}. File: {path}. Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
    )


def _match_registry_by_name(model_name: str) -> dict[str, Any]:
    matches = [config for config in WAN22_MODEL_REGISTRY if config["model_name"] == model_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one registry entry for {model_name}, got {len(matches)}."
        )
    return matches[0]


def _load_state_dict_with_assign(model, state_dict, strict: bool):
    if "assign" in inspect.signature(model.load_state_dict).parameters:
        return model.load_state_dict(state_dict, strict=strict, assign=True)
    return model.load_state_dict(state_dict, strict=strict)


def _single_checkpoint_path(path: Any, *, env_name: str) -> str:
    if isinstance(path, (list, tuple)):
        raise ValueError(
            f"{env_name}=true supports only one text encoder checkpoint file, got: {path!r}"
        )
    path_text = os.fspath(path)
    if not path_text.endswith(".safetensors"):
        raise ValueError(
            f"{env_name}=true supports only safetensors text encoder checkpoints, got: {path_text}"
        )
    return path_text


def _target_torch_device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    return target


def _load_text_encoder_fast_from_safetensors(
    *,
    path: Any,
    model_class: type[torch.nn.Module],
    model_kwargs: dict[str, Any],
    torch_dtype: torch.dtype,
    device: str | torch.device,
) -> torch.nn.Module:
    path_text = _single_checkpoint_path(path, env_name=TEXT_ENCODER_FAST_LOAD_ENV)
    if "assign" not in inspect.signature(torch.nn.Module.load_state_dict).parameters:
        raise RuntimeError(
            f"{TEXT_ENCODER_FAST_LOAD_ENV}=true requires PyTorch load_state_dict(assign=True)."
        )
    target_device = _target_torch_device(device)
    logger.info(
        "Loading Wan text encoder with fast safetensors path: path=%s device=%s dtype=%s",
        path_text,
        target_device,
        torch_dtype,
    )
    construct_start = time.time()
    with torch.device("meta"):
        model = model_class(**model_kwargs)
    _log_text_encoder_phase("fast_construct_meta_module", construct_start)
    state_load_start = time.time()
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path_text, framework="pt", device=str(target_device)) as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            if torch_dtype is not None and value.dtype != torch_dtype:
                value = value.to(dtype=torch_dtype)
            state_dict[key] = value
    _log_text_encoder_phase("fast_read_state_dict_direct", state_load_start)
    assign_start = time.time()
    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    _log_text_encoder_phase("fast_assign_state_dict", assign_start)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        logger.warning(
            "Wan text encoder fast-load key mismatch: missing=%s unexpected=%s",
            missing[:20],
            unexpected[:20],
        )
    meta_keys = [
        name
        for (name, value) in model.state_dict().items()
        if isinstance(value, torch.Tensor) and value.device.type == "meta"
    ]
    if meta_keys:
        raise ValueError(
            f"Wan text encoder still has meta tensors after fast load: {meta_keys[:20]}"
        )
    return model


def _torch_dtype_cache_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    raise ValueError(f"Unsupported 14B prepared-cache dtype: {dtype}.")


def _14b_local_dim_contract(dit_config: dict[str, Any]) -> dict[str, int]:
    return {
        "in_dim": int(dit_config.get("in_dim", -1)),
        "out_dim": int(dit_config.get("out_dim", -1)),
        "hidden_dim": int(dit_config.get("hidden_dim", -1)),
        "ffn_dim": int(dit_config.get("ffn_dim", -1)),
        "num_heads": int(dit_config.get("num_heads", -1)),
        "num_layers": int(dit_config.get("num_layers", -1)),
    }


def _14b_expected_local_dim_contract() -> dict[str, int]:
    return {
        "in_dim": 36,
        "out_dim": 16,
        "hidden_dim": 5120,
        "ffn_dim": 13824,
        "num_heads": 40,
        "num_layers": 40,
    }


def _instantiate_wan_video_dit_on_meta(model_kwargs: dict[str, Any]) -> WanVideoDiT:
    from dreamfree.models.wan22.wan_video_dit import precompute_freqs_cis_3d

    with torch.device("meta"):
        model = WanVideoDiT(**model_kwargs)
    model.freqs = precompute_freqs_cis_3d(int(model_kwargs["attn_head_dim"]))
    return model


def _load_wan_video_dit_native_shards(
    *,
    paths: list[Path],
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs: dict[str, Any],
    strict: bool = True,
) -> WanVideoDiT:
    model = _instantiate_wan_video_dit_on_meta(model_kwargs)
    expected_keys = set(model.state_dict().keys())
    loaded_keys: set[str] = set()
    unexpected_keys: set[str] = set()
    for path in paths:
        state_dict = load_state_dict(str(path), torch_dtype=torch_dtype, device="cpu")
        duplicate = sorted(set(state_dict) & loaded_keys)
        if duplicate:
            raise ValueError(
                f"{model_name} prepared shard has duplicate keys in {path}: {duplicate[:20]}"
            )
        loaded_keys.update(state_dict.keys())
        incompatible = _load_state_dict_with_assign(model, state_dict, strict=False)
        unexpected_keys.update(getattr(incompatible, "unexpected_keys", []) or [])
        del state_dict
    missing_keys = sorted(expected_keys - loaded_keys)
    unexpected_keys = sorted(unexpected_keys)
    if strict and (missing_keys or unexpected_keys):
        raise ValueError(
            f"{model_name} prepared checkpoint key mismatch: missing={missing_keys[:20]}, unexpected={unexpected_keys[:20]}"
        )
    meta_keys = [
        name
        for (name, value) in model.state_dict().items()
        if isinstance(value, torch.Tensor) and value.device.type == "meta"
    ]
    if meta_keys:
        raise ValueError(
            f"{model_name} still has meta tensors after prepared load: {meta_keys[:20]}"
        )
    return model.to(device=device, dtype=torch_dtype)


def _validate_prepared_14b_metadata(
    metadata: dict[str, Any],
    *,
    metadata_path: Path,
    dit_role: str,
    expected_dtype: torch.dtype,
    expected_config: dict[str, Any],
    expected_shard_names: list[str],
) -> None:
    if metadata.get("format") != WAN22_A14B_PREPARED_CACHE_FORMAT:
        raise ValueError(
            f"14B prepared cache metadata at {metadata_path} has format={metadata.get('format')!r}; expected {WAN22_A14B_PREPARED_CACHE_FORMAT!r}."
        )
    if metadata.get("model_id") != WAN22_I2V_A14B_MODEL_ID:
        raise ValueError(
            f"14B prepared cache metadata at {metadata_path} has model_id={metadata.get('model_id')!r}; expected {WAN22_I2V_A14B_MODEL_ID!r}."
        )
    if metadata.get("dit_role") != dit_role:
        raise ValueError(
            f"14B prepared cache metadata at {metadata_path} has dit_role={metadata.get('dit_role')!r}; expected {dit_role!r}."
        )
    expected_dtype_name = _torch_dtype_cache_name(expected_dtype)
    if metadata.get("dtype") != expected_dtype_name:
        raise ValueError(
            f"14B prepared cache metadata at {metadata_path} has dtype={metadata.get('dtype')!r}; expected {expected_dtype_name!r} for requested torch_dtype={expected_dtype}."
        )
    contract = metadata.get("a14b_dit_config_contract")
    if contract != WAN22_A14B_DIT_CONFIG_CONTRACT:
        raise ValueError(
            f"14B prepared cache metadata at {metadata_path} has an invalid config contract: {contract!r}."
        )
    local_contract = metadata.get("local_dit_config")
    expected_local_contract = _14b_expected_local_dim_contract()
    if local_contract != expected_local_contract:
        raise ValueError(
            f"14B prepared cache metadata at {metadata_path} has local_dit_config={local_contract!r}; expected {expected_local_contract!r}."
        )
    requested_local_contract = _14b_local_dim_contract(expected_config)
    if requested_local_contract != expected_local_contract:
        raise ValueError(
            f"Requested 14B DiT config does not match prepared-cache contract: requested={requested_local_contract!r}, expected={expected_local_contract!r}."
        )
    prepared_shards = metadata.get("prepared_shards")
    if not isinstance(prepared_shards, list):
        raise ValueError(
            f"14B prepared cache metadata at {metadata_path} is missing list `prepared_shards`."
        )
    metadata_shard_names = sorted(
        str(item.get("name", "")) for item in prepared_shards if isinstance(item, dict)
    )
    if metadata_shard_names != expected_shard_names:
        raise ValueError(
            f"14B prepared cache metadata at {metadata_path} lists shards={metadata_shard_names}; expected {expected_shard_names}."
        )


def _resolve_prepared_14b_dit_cache(
    *, dit_role: str, expected_config: dict[str, Any], torch_dtype: torch.dtype
) -> tuple[list[Path], Path] | None:
    cache_root_text = os.environ.get(WAN22_A14B_PREPARED_CACHE_ENV)
    if cache_root_text is None or not cache_root_text.strip():
        return None
    cache_root = Path(cache_root_text).expanduser()
    role_dir = cache_root / f"{dit_role}_model"
    metadata_path = role_dir / WAN22_A14B_PREPARED_CACHE_METADATA
    if not role_dir.is_dir():
        raise FileNotFoundError(
            f"{WAN22_A14B_PREPARED_CACHE_ENV} is set to {cache_root}, but 14B {dit_role} cache directory does not exist: {role_dir}"
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{WAN22_A14B_PREPARED_CACHE_ENV} is set to {cache_root}, but metadata is missing: {metadata_path}"
        )
    shard_paths = _14b_shard_paths(
        sorted(role_dir.glob("diffusion_pytorch_model*.safetensors")), dit_role=dit_role
    )
    expected_shard_names = [path.name for path in shard_paths]
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"14B prepared cache metadata must be a JSON object: {metadata_path}")
    _validate_prepared_14b_metadata(
        metadata,
        metadata_path=metadata_path,
        dit_role=dit_role,
        expected_dtype=torch_dtype,
        expected_config=expected_config,
        expected_shard_names=expected_shard_names,
    )
    missing_or_empty = [
        str(path) for path in shard_paths if not path.exists() or path.stat().st_size <= 0
    ]
    if missing_or_empty:
        raise FileNotFoundError(
            f"14B prepared cache has missing/empty shard(s): {missing_or_empty}"
        )
    return (shard_paths, role_dir)


def _load_model_from_state_dict(
    *,
    path,
    model_class,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs: dict[str, Any],
    state_dict_converter=None,
    strict: bool = True,
):
    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)
    incompatible = _load_state_dict_with_assign(model, state_dict, strict=strict)
    missing = sorted(getattr(incompatible, "missing_keys", []) or [])
    unexpected = sorted(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise ValueError(
            f"{model_name} checkpoint key mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    return model.to(device=device, dtype=torch_dtype)


def _load_dit_with_optional_broadcast(
    *,
    model_name: str,
    model_kwargs: dict[str, Any],
    torch_dtype: torch.dtype,
    device: str,
    pretrained_load_broadcast: bool,
    rank0_loader,
) -> WanVideoDiT:
    """Load a ``WanVideoDiT`` so only rank 0 reads from disk, then broadcast.

    The 14B DiT (multi-shard, ~28 GB) is otherwise read independently by every
    rank. When distributed broadcast is enabled, rank 0 loads the populated
    module via ``rank0_loader`` and broadcasts it to peers, which construct an
    empty module and never touch disk. When broadcast is disabled (single
    process or ``pretrained_load_broadcast=False``) every caller loads from disk
    exactly as before, so behavior is unchanged outside distributed runs.

    Args:
        model_name: Logging/diagnostic name for the module.
        model_kwargs: ``WanVideoDiT`` constructor kwargs used to materialize the
            empty receiving module on non-loading ranks.
        torch_dtype: Target dtype of the module weights.
        device: Final device for the returned module.
        pretrained_load_broadcast: Whether to use rank0-load + broadcast.
        rank0_loader: Callable ``(load_device: str) -> WanVideoDiT`` that fully
            loads the module onto ``load_device`` (only invoked on rank 0 when
            broadcasting, or on every rank otherwise).

    Returns:
        The loaded ``WanVideoDiT`` on ``device``.
    """
    use_broadcast = should_broadcast_pretrained_load(pretrained_load_broadcast)
    if not use_broadcast:
        return rank0_loader(device)
    rank = get_distributed_rank()
    broadcast_device = get_pretrained_broadcast_device()
    load_device = "cpu" if broadcast_device == "cpu" else device
    model: WanVideoDiT | None = None
    load_error: BaseException | None = None
    if rank == 0:
        try:
            model = rank0_loader(load_device)
        except BaseException as exc:
            load_error = exc
    sync_distributed_load_error(
        load_error, device=load_device, src=0, context=f"rank0 load for {model_name}"
    )
    if model is None:
        model = WanVideoDiT(**model_kwargs).to(device=load_device, dtype=torch_dtype)
    broadcast_module_state(model, src=0)
    if broadcast_device == "cpu":
        model = model.to(device=device, dtype=torch_dtype)
    logger.info("Loaded %s via rank0 %s broadcast.", model_name, broadcast_device)
    return model


def _14b_shard_paths(path: str | list[str], *, dit_role: str) -> list[Path]:
    paths = [Path(item) for item in path] if isinstance(path, list) else [Path(path)]
    expected_names = {
        f"diffusion_pytorch_model-{idx:05d}-of-00006.safetensors" for idx in range(1, 7)
    }
    actual_names = {item.name for item in paths}
    if actual_names != expected_names:
        raise FileNotFoundError(
            f"Wan2.2-I2V-A14B {dit_role} DiT requires exactly six local shards {sorted(expected_names)}, got {sorted(actual_names)}."
        )
    parents = {item.parent.resolve() for item in paths}
    if len(parents) != 1:
        raise ValueError(
            f"14B {dit_role} DiT shards must all live in one directory, got {sorted(map(str, parents))}."
        )
    shard_dir = next(iter(parents))
    if shard_dir.name != f"{dit_role}_model":
        raise ValueError(f"14B {dit_role} DiT resolved wrong directory: {shard_dir}.")
    missing = [str(item) for item in paths if not item.exists()]
    if missing:
        raise FileNotFoundError(f"Missing 14B {dit_role} DiT shard(s): {missing}")
    return sorted(paths)


def _validate_14b_config_json(
    shard_dir: Path, *, dit_role: str, expected_config: dict[str, Any]
) -> None:
    config_path = shard_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing 14B {dit_role} DiT config.json: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    mismatches: list[str] = []
    for key, expected in WAN22_A14B_DIT_CONFIG_CONTRACT.items():
        actual = payload.get(key)
        if actual != expected:
            mismatches.append(f"{key}: actual={actual!r} expected={expected!r}")
    local_dim_checks = {
        "in_dim": int(expected_config.get("in_dim", -1)),
        "out_dim": int(expected_config.get("out_dim", -1)),
        "hidden_dim": int(expected_config.get("hidden_dim", -1)),
        "ffn_dim": int(expected_config.get("ffn_dim", -1)),
        "num_heads": int(expected_config.get("num_heads", -1)),
        "num_layers": int(expected_config.get("num_layers", -1)),
    }
    local_expected = {
        "in_dim": 36,
        "out_dim": 16,
        "hidden_dim": 5120,
        "ffn_dim": 13824,
        "num_heads": 40,
        "num_layers": 40,
    }
    for key, actual in local_dim_checks.items():
        expected = local_expected[key]
        if actual != expected:
            mismatches.append(f"local_dit_config.{key}: actual={actual!r} expected={expected!r}")
    if mismatches:
        raise ValueError(
            f"14B {dit_role} DiT config mismatch in {config_path}: " + "; ".join(mismatches)
        )


def _resolve_dit_config(model_id: str, *, dit_role: str = "single") -> ModelConfig:
    if str(model_id) == WAN22_I2V_A14B_MODEL_ID:
        role = str(dit_role)
        if role not in WAN22_A14B_DIT_ROLES:
            raise ValueError(
                f"Wan2.2-I2V-A14B dit_role must be 'low_noise' or 'high_noise', got {dit_role!r}."
            )
        return ModelConfig(
            model_id=model_id,
            origin_file_pattern=f"{role}_model/diffusion_pytorch_model*.safetensors",
        )
    return ModelConfig(
        model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"
    )


def _resolve_configs(
    model_id: str,
    tokenizer_model_id: str,
    redirect_common_files: bool = True,
    *,
    dit_role: str = "single",
):
    dit_config = _resolve_dit_config(model_id, dit_role=dit_role)
    text_config = ModelConfig(
        model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"
    )
    vae_config = _resolve_vae_config(model_id=model_id, redirect_common_files=redirect_common_files)
    tokenizer_config = ModelConfig(
        model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/"
    )
    if redirect_common_files:
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": (
                "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                "models_t5_umt5-xxl-enc-bf16.safetensors",
            )
        }
        (text_config.model_id, text_config.origin_file_pattern) = redirect_dict[
            text_config.origin_file_pattern
        ]
    return (dit_config, text_config, vae_config, tokenizer_config)


def _resolve_vae_config(model_id: str, redirect_common_files: bool = True) -> ModelConfig:
    if str(model_id) == WAN22_I2V_A14B_MODEL_ID:
        return ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth")
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern="Wan2.2_VAE.pth")
    if redirect_common_files:
        vae_config.model_id = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
        vae_config.origin_file_pattern = "Wan2.2_VAE.safetensors"
    return vae_config


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
    load_vae: bool = True,
    pretrained_load_broadcast: bool = True,
    dit_override: WanVideoDiT | None = None,
):
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()
    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")
    validated_dit_config = _validate_dit_config(dit_config)
    (dit_model_config, text_config, vae_config, tokenizer_config) = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    if load_vae:
        _download_if_necessary(vae_config, pretrained_load_broadcast)
    if load_text_encoder:
        _download_if_necessary(text_config, pretrained_load_broadcast)
        _download_if_necessary(tokenizer_config, pretrained_load_broadcast)
    if skip_dit_load_from_pretrain and dit_override is not None:
        raise ValueError("`skip_dit_load_from_pretrain` and `dit_override` cannot both be set.")
    if dit_override is not None:
        if not isinstance(dit_override, WanVideoDiT):
            raise TypeError(
                f"`dit_override` must be WanVideoDiT, got {type(dit_override).__name__}."
            )
        override_parameter = next(dit_override.parameters(), None)
        if override_parameter is None:
            raise ValueError("`dit_override` must expose at least one parameter.")
        expected_device = torch.device(device)
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())
        if override_parameter.device != expected_device:
            raise ValueError(
                f"`dit_override` parameters must be on {expected_device}, got {override_parameter.device}."
            )
        if override_parameter.dtype != torch_dtype:
            raise TypeError(
                f"`dit_override` parameters must use {torch_dtype}, got {override_parameter.dtype}."
            )
        dit = dit_override
        dit_path = PRECONSTRUCTED_DIT_SENTINEL
        logger.info("Using preconstructed Wan2.2-TI2V-5B DiT override.")
    elif skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); initializing video expert randomly and expecting checkpoint override."
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        _download_if_necessary(dit_model_config, pretrained_load_broadcast)
        dit = _load_registered_model(
            dit_model_config.path,
            "wan_video_dit",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs_override=validated_dit_config,
            pretrained_load_broadcast=pretrained_load_broadcast,
        )
        dit_path = str(dit_model_config.path)
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
            pretrained_load_broadcast=pretrained_load_broadcast,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path, seq_len=int(tokenizer_max_len), clean="whitespace"
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); training must provide cached `context/context_mask`."
        )
    vae: WanVideoVAE38 | None = None
    vae_path: str | None = None
    if load_vae:
        vae = _load_registered_model(
            vae_config.path,
            "wan_video_vae",
            torch_dtype=torch_dtype,
            device=device,
            pretrained_load_broadcast=pretrained_load_broadcast,
        )
        vae_path = str(vae_config.path)
    else:
        logger.info(
            "Skipping pretrained VAE load (`load_vae=False`); training must provide precomputed video latents."
        )
    logger.info("Finished loading Wan2.2-TI2V-5B components in %.2f seconds.", time.time() - start)
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=vae_path,
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )


def load_wan22_i2v_a14b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-I2V-A14B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    dit_role: str = "low_noise",
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
    load_vae: bool = True,
    pretrained_load_broadcast: bool = True,
    dit_override: WanVideoDiT | None = None,
):
    """Load Wan2.2-I2V-A14B components for one explicit noise expert."""
    logger.info("Loading Wan2.2-I2V-A14B %s components...", dit_role)
    start = time.time()
    if str(model_id) != WAN22_I2V_A14B_MODEL_ID:
        raise ValueError(
            f"14B loader requires model_id='Wan-AI/Wan2.2-I2V-A14B', got {model_id!r}."
        )
    if str(dit_role) not in WAN22_A14B_DIT_ROLES:
        raise ValueError(f"14B dit_role must be 'low_noise' or 'high_noise', got {dit_role!r}.")
    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-I2V-A14B loading.")
    validated_dit_config = _validate_dit_config(dit_config)
    if (
        int(validated_dit_config.get("in_dim", -1)) != 36
        or int(validated_dit_config.get("out_dim", -1)) != 16
    ):
        raise ValueError(
            f"Wan2.2-I2V-A14B DiT config must use in_dim=36 and out_dim=16, got in_dim={validated_dit_config.get('in_dim')} out_dim={validated_dit_config.get('out_dim')}."
        )
    if int(validated_dit_config.get("hidden_dim", -1)) != 5120:
        raise ValueError(
            f"Wan2.2-I2V-A14B DiT config must use hidden_dim=5120, got {validated_dit_config.get('hidden_dim')}."
        )
    (dit_model_config, text_config, vae_config, tokenizer_config) = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
        dit_role=str(dit_role),
    )
    prepared_cache: tuple[list[Path], Path] | None = None
    if skip_dit_load_from_pretrain and dit_override is not None:
        raise ValueError("`skip_dit_load_from_pretrain` and `dit_override` cannot both be set.")
    if not skip_dit_load_from_pretrain and dit_override is None:
        prepared_cache = _resolve_prepared_14b_dit_cache(
            dit_role=str(dit_role), expected_config=validated_dit_config, torch_dtype=torch_dtype
        )
    if load_vae:
        _download_if_necessary(vae_config, pretrained_load_broadcast)
    if load_text_encoder:
        _download_if_necessary(text_config, pretrained_load_broadcast)
        _download_if_necessary(tokenizer_config, pretrained_load_broadcast)
    if dit_override is not None:
        if not isinstance(dit_override, WanVideoDiT):
            raise TypeError(
                f"`dit_override` must be WanVideoDiT, got {type(dit_override).__name__}."
            )
        override_parameter = next(dit_override.parameters(), None)
        if override_parameter is None:
            raise ValueError("`dit_override` must expose at least one parameter.")
        expected_device = torch.device(device)
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())
        if override_parameter.device != expected_device:
            raise ValueError(
                f"`dit_override` parameters must be on {expected_device}, got {override_parameter.device}."
            )
        if override_parameter.dtype != torch_dtype:
            raise TypeError(
                f"`dit_override` parameters must use {torch_dtype}, got {override_parameter.dtype}."
            )
        dit = dit_override
        dit_path = PRECONSTRUCTED_DIT_SENTINEL
        logger.info("Using preconstructed Wan2.2-I2V-A14B %s DiT override.", dit_role)
    elif skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained 14B %s DiT load; initializing randomly and expecting checkpoint override.",
            dit_role,
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_name = f"wan22_i2v_a14b_{dit_role}_dit"
        if prepared_cache is not None:
            (shard_paths, prepared_role_dir) = prepared_cache
            logger.info(
                "Loading Wan2.2-I2V-A14B %s DiT from prepared cache: %s",
                dit_role,
                prepared_role_dir,
            )
            dit = _load_dit_with_optional_broadcast(
                model_name=dit_model_name,
                model_kwargs=validated_dit_config,
                torch_dtype=torch_dtype,
                device=device,
                pretrained_load_broadcast=pretrained_load_broadcast,
                rank0_loader=lambda load_device: _load_wan_video_dit_native_shards(
                    paths=shard_paths,
                    model_name=dit_model_name,
                    torch_dtype=torch_dtype,
                    device=load_device,
                    model_kwargs=validated_dit_config,
                    strict=True,
                ),
            )
            dit_path = str(prepared_role_dir)
        else:
            _download_if_necessary(dit_model_config, pretrained_load_broadcast)
            shard_paths = _14b_shard_paths(dit_model_config.path, dit_role=str(dit_role))
            _validate_14b_config_json(
                shard_paths[0].parent, dit_role=str(dit_role), expected_config=validated_dit_config
            )
            dit = _load_dit_with_optional_broadcast(
                model_name=dit_model_name,
                model_kwargs=validated_dit_config,
                torch_dtype=torch_dtype,
                device=device,
                pretrained_load_broadcast=pretrained_load_broadcast,
                rank0_loader=lambda load_device: _load_model_from_state_dict(
                    path=dit_model_config.path,
                    model_class=WanVideoDiT,
                    model_name=dit_model_name,
                    torch_dtype=torch_dtype,
                    device=load_device,
                    model_kwargs=validated_dit_config,
                    state_dict_converter=wan_video_dit_state_dict_converter,
                    strict=True,
                ),
            )
            dit_path = str(shard_paths[0].parent)
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
            pretrained_load_broadcast=pretrained_load_broadcast,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path, seq_len=int(tokenizer_max_len), clean="whitespace"
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); training must provide cached `context/context_mask`."
        )
    vae: WanVideoVAE | None = None
    vae_path: Path | None = None
    if load_vae:
        vae_path = Path(str(vae_config.path))
        if vae_path.name != "Wan2.1_VAE.pth" or not vae_path.exists():
            raise FileNotFoundError(
                f"14B I2V requires local Wan2.1_VAE.pth, got {vae_config.path!r}."
            )
        vae = _load_model_from_state_dict(
            path=str(vae_path),
            model_class=WanVideoVAE,
            model_name="wan21_video_vae_for_i2v_14b",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs={"z_dim": 16},
            state_dict_converter=wan_video_vae_state_dict_converter,
            strict=True,
        )
    else:
        logger.info(
            "Skipping pretrained VAE load (`load_vae=False`); training must provide precomputed video latents."
        )
    logger.info(
        "Finished loading Wan2.2-I2V-A14B %s components in %.2f seconds.",
        dit_role,
        time.time() - start,
    )
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=None if vae_path is None else str(vae_path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )


def _download_if_necessary(config: ModelConfig, pretrained_load_broadcast: bool) -> None:
    use_broadcast = should_broadcast_pretrained_load(pretrained_load_broadcast)
    if not use_broadcast:
        config.download_if_necessary()
        return
    rank = get_distributed_rank()
    error: BaseException | None = None
    if rank == 0:
        try:
            config.download_if_necessary()
        except BaseException as exc:
            error = exc
    sync_distributed_load_error(
        error,
        device=get_pretrained_broadcast_sync_device("cuda" if torch.cuda.is_available() else "cpu"),
        src=0,
        context=f"download/resolve for {config.model_id}",
    )
    resolved_path = config.path if rank == 0 else None
    path_message: list[Any] = [resolved_path]
    torch.distributed.broadcast_object_list(path_message, src=0)
    config.path = path_message[0]
    if config.path is None or (isinstance(config.path, list) and len(config.path) == 0):
        raise RuntimeError(f"Rank0 did not resolve a local path for {config.model_id}.")


def load_wan22_ti2v_5b_vae(
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    redirect_common_files: bool = True,
    pretrained_load_broadcast: bool = True,
) -> Wan22LoadedVAE:
    logger.info("Loading Wan2.2-TI2V-5B VAE only...")
    start = time.time()
    vae_config = _resolve_vae_config(model_id=model_id, redirect_common_files=redirect_common_files)
    _download_if_necessary(vae_config, pretrained_load_broadcast)
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch_dtype,
        device=device,
        pretrained_load_broadcast=pretrained_load_broadcast,
    )
    logger.info("Finished loading Wan2.2-TI2V-5B VAE in %.2f seconds.", time.time() - start)
    return Wan22LoadedVAE(vae=vae, vae_path=str(vae_config.path))


def load_wan22_i2v_a14b_vae(
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-I2V-A14B",
    redirect_common_files: bool = True,
    pretrained_load_broadcast: bool = True,
) -> Wan22LoadedVAE:
    logger.info("Loading Wan2.2-I2V-A14B Wan2.1 VAE only...")
    start = time.time()
    if str(model_id) != WAN22_I2V_A14B_MODEL_ID:
        raise ValueError(
            f"14B VAE loader requires model_id={WAN22_I2V_A14B_MODEL_ID!r}, got {model_id!r}."
        )
    vae_config = _resolve_vae_config(model_id=model_id, redirect_common_files=redirect_common_files)
    _download_if_necessary(vae_config, pretrained_load_broadcast)
    vae_path = Path(str(vae_config.path))
    if vae_path.name != "Wan2.1_VAE.pth" or not vae_path.exists():
        raise FileNotFoundError(f"14B I2V requires local Wan2.1_VAE.pth, got {vae_config.path!r}.")
    vae: WanVideoVAE = _load_model_from_state_dict(
        path=str(vae_path),
        model_class=WanVideoVAE,
        model_name="wan21_video_vae_for_i2v_a14b",
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"z_dim": 16},
        state_dict_converter=wan_video_vae_state_dict_converter,
        strict=True,
    )
    logger.info("Finished loading Wan2.2-I2V-A14B Wan2.1 VAE in %.2f seconds.", time.time() - start)
    return Wan22LoadedVAE(vae=vae, vae_path=str(vae_path))


@dataclass
class Wan22LoadedVAE:
    vae: WanVideoVAE | WanVideoVAE38
    vae_path: str
