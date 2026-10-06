from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Sequence

import torch
import torch.nn as nn

LORA_PARAMETER_SUFFIXES = ("lora_A.weight", "lora_B.weight")
DEFAULT_LORA_ADAPTER_NAME = "default"
LORA_ENABLED_STATE_SEPARATOR = "::"
LORA_TARGET_PRESETS = {
    "cross_attention_only": ("cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o"),
    "attention_only": (
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
    ),
    "attention_and_ffn": (
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "ffn.0",
        "ffn.2",
    ),
}


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` with a trainable low-rank residual adapter."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        adapter_name: str | None = None,
    ):
        """Initialize a LoRA adapter around an existing linear layer.

        Args:
            base_layer: Frozen base projection mapping ``(..., in_features) -> (..., out_features)``.
            rank: Low-rank bottleneck dimension for the adapter branch.
            alpha: LoRA scaling factor applied as ``alpha / rank``.
            dropout: Dropout probability on the adapter input only.
            adapter_name: Optional logical name for the first adapter. When
                omitted, legacy single-adapter naming and state dict keys are used.

        Raises:
            TypeError: If ``base_layer`` is not an ``nn.Linear``.
            ValueError: If any LoRA hyperparameter is invalid.
        """
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"`base_layer` must be nn.Linear, got {type(base_layer)}")
        if rank <= 0:
            raise ValueError(f"`rank` must be > 0, got {rank}")
        if alpha <= 0:
            raise ValueError(f"`alpha` must be > 0, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}")
        self.base_layer = base_layer
        self.primary_adapter_name = _normalize_lora_adapter_name(adapter_name)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)
        self.scaling = self.alpha / float(self.rank)
        self.enabled = True
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else nn.Identity()
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        self.lora_A = nn.Linear(
            self.base_layer.in_features, self.rank, bias=False, device=device, dtype=dtype
        )
        self.lora_B = nn.Linear(
            self.rank, self.base_layer.out_features, bias=False, device=device, dtype=dtype
        )
        self.reset_parameters()
        self._extra_lora_A = nn.ModuleDict()
        self._extra_lora_B = nn.ModuleDict()
        self._extra_dropouts = nn.ModuleDict()
        self._adapter_metadata = {
            self.primary_adapter_name: {
                "rank": self.rank,
                "alpha": self.alpha,
                "dropout": self.dropout_p,
                "scaling": self.scaling,
            }
        }
        self._extra_enabled: dict[str, bool] = {}
        self.base_layer.requires_grad_(False)

    def reset_parameters(self) -> None:
        """Reset LoRA weights following the standard zero-initialized residual pattern."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def add_adapter(self, adapter_name: str, rank: int, alpha: float, dropout: float) -> None:
        """Attach an additional named LoRA adapter to this frozen base layer."""
        name = _normalize_lora_adapter_name(adapter_name)
        if self.has_adapter(name):
            raise ValueError(f"LoRA adapter `{name}` already exists.")
        if rank <= 0:
            raise ValueError(f"`rank` must be > 0, got {rank}")
        if alpha <= 0:
            raise ValueError(f"`alpha` must be > 0, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}")
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        self._extra_lora_A[name] = nn.Linear(
            self.base_layer.in_features, int(rank), bias=False, device=device, dtype=dtype
        )
        self._extra_lora_B[name] = nn.Linear(
            int(rank), self.base_layer.out_features, bias=False, device=device, dtype=dtype
        )
        self._extra_dropouts[name] = (
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        )
        nn.init.kaiming_uniform_(self._extra_lora_A[name].weight, a=math.sqrt(5))
        nn.init.zeros_(self._extra_lora_B[name].weight)
        self._adapter_metadata[name] = {
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "scaling": float(alpha) / float(rank),
        }
        self._extra_enabled[name] = True

    def adapter_names(self) -> list[str]:
        """Return adapter names in deterministic application order."""
        return [self.primary_adapter_name, *self._extra_lora_A.keys()]

    def has_adapter(self, adapter_name: str) -> bool:
        name = _normalize_lora_adapter_name(adapter_name)
        return name == self.primary_adapter_name or name in self._extra_lora_A

    def adapter_count(self) -> int:
        return 1 + len(self._extra_lora_A)

    def is_adapter_enabled(self, adapter_name: str) -> bool:
        name = _normalize_lora_adapter_name(adapter_name)
        if name == self.primary_adapter_name:
            return bool(self.enabled)
        if name not in self._extra_lora_A:
            raise ValueError(f"LoRA adapter `{name}` does not exist.")
        return bool(self._extra_enabled[name])

    def set_adapter_enabled(self, adapter_name: str, enabled: bool) -> None:
        name = _normalize_lora_adapter_name(adapter_name)
        if name == self.primary_adapter_name:
            self.enabled = bool(enabled)
            return
        if name not in self._extra_lora_A:
            raise ValueError(f"LoRA adapter `{name}` does not exist.")
        self._extra_enabled[name] = bool(enabled)

    def adapter_parameter_tensors(
        self, adapter_name: str
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        """Return the LoRA A/B weight parameters for one adapter."""
        name = _normalize_lora_adapter_name(adapter_name)
        if name == self.primary_adapter_name:
            return (self.lora_A.weight, self.lora_B.weight)
        if name not in self._extra_lora_A:
            raise ValueError(f"LoRA adapter `{name}` does not exist.")
        return (self._extra_lora_A[name].weight, self._extra_lora_B[name].weight)

    def _adapter_modules(self, adapter_name: str) -> tuple[nn.Module, nn.Module, nn.Module, float]:
        name = _normalize_lora_adapter_name(adapter_name)
        if name == self.primary_adapter_name:
            return (self.lora_A, self.lora_B, self.dropout, float(self.scaling))
        if name not in self._extra_lora_A:
            raise ValueError(f"LoRA adapter `{name}` does not exist.")
        return (
            self._extra_lora_A[name],
            self._extra_lora_B[name],
            self._extra_dropouts[name],
            float(self._adapter_metadata[name]["scaling"]),
        )

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection plus the scaled LoRA residual.

        Args:
            x: Input tensor with shape ``(..., in_features)``.

        Returns:
            Tensor with shape ``(..., out_features)``.
        """
        base_out = self.base_layer(x)
        out = base_out
        for name in self.adapter_names():
            if not self.is_adapter_enabled(name):
                continue
            (lora_A, lora_B, dropout, scaling) = self._adapter_modules(name)
            out = out + lora_B(lora_A(dropout(x))) * scaling
        return out


def _normalize_lora_adapter_name(adapter_name: str | None) -> str:
    if adapter_name is None:
        return DEFAULT_LORA_ADAPTER_NAME
    name = str(adapter_name).strip().lower()
    if not name:
        raise ValueError("LoRA adapter name must be non-empty.")
    if "." in name or LORA_ENABLED_STATE_SEPARATOR in name:
        raise ValueError(
            f"LoRA adapter name {adapter_name!r} cannot contain '.' or '{LORA_ENABLED_STATE_SEPARATOR}'."
        )
    return name


def normalize_lora_config(lora_config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user LoRA config with defaults and validate the final values.

    Args:
        lora_config: Optional user-provided LoRA config dictionary.

    Returns:
        Normalized config with stable Python scalar types.

    Raises:
        ValueError: If the config is not dict-like or contains invalid values.
    """
    default = {
        "enabled": False,
        "rank": 16,
        "alpha": 16.0,
        "dropout": 0.0,
        "target_modules": "attention_only",
    }
    if lora_config is None:
        lora_config = {}
    if not isinstance(lora_config, dict):
        raise ValueError(f"`lora` config must be dict-like, got {type(lora_config)}")
    normalized = dict(default)
    normalized.update(lora_config)
    normalized["enabled"] = bool(normalized["enabled"])
    normalized["rank"] = int(normalized["rank"])
    normalized["alpha"] = float(normalized["alpha"])
    normalized["dropout"] = float(normalized["dropout"])
    normalized["target_modules"] = str(normalized["target_modules"]).strip().lower()
    if normalized["rank"] <= 0:
        raise ValueError(f"`lora.rank` must be > 0, got {normalized['rank']}")
    if normalized["alpha"] <= 0:
        raise ValueError(f"`lora.alpha` must be > 0, got {normalized['alpha']}")
    if not 0.0 <= normalized["dropout"] < 1.0:
        raise ValueError(f"`lora.dropout` must be in [0, 1), got {normalized['dropout']}")
    if normalized["target_modules"] not in set(LORA_TARGET_PRESETS) | {"all_linear"}:
        raise ValueError(
            f"Unsupported `lora.target_modules`: {normalized['target_modules']}. Expected one of: ['cross_attention_only', 'attention_only', 'attention_and_ffn', 'all_linear']."
        )
    return normalized


def has_lora_adapters(module: nn.Module) -> bool:
    """Return whether any descendant module is already LoRA-wrapped.

    Args:
        module: Module tree to inspect.

    Returns:
        ``True`` if a ``LoRALinear`` appears anywhere under ``module``.
    """
    return any(isinstance(child, LoRALinear) for child in module.modules())


def _lora_enabled_state_key(module_name: str, adapter_name: str) -> str:
    return f"{module_name}{LORA_ENABLED_STATE_SEPARATOR}{adapter_name}"


def _split_lora_enabled_state_key(key: str) -> tuple[str, str | None]:
    if LORA_ENABLED_STATE_SEPARATOR not in key:
        return (key, None)
    (module_name, adapter_name) = key.rsplit(LORA_ENABLED_STATE_SEPARATOR, 1)
    return (module_name, _normalize_lora_adapter_name(adapter_name))


def iter_lora_modules(module: nn.Module):
    """Yield `(name, LoRALinear)` pairs from a module tree."""
    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            yield (name, child)


def is_lora_parameter_name(name: str) -> bool:
    """Return whether a parameter name belongs to a LoRA adapter."""
    return (
        name.endswith(LORA_PARAMETER_SUFFIXES)
        or "._extra_lora_A." in name
        or "._extra_lora_B." in name
        or name.startswith("_extra_lora_A.")
        or name.startswith("_extra_lora_B.")
    )


def set_lora_enabled(
    module: nn.Module, enabled: bool, adapter_name: str | None = None
) -> dict[str, bool]:
    """Set the runtime enabled flag on every LoRA adapter in a module tree.

    Args:
        module: Module tree containing optional LoRA adapters.
        enabled: Desired adapter runtime state.
        adapter_name: Optional adapter name. When omitted, all adapters are toggled.

    Returns:
        Mapping from module/adapter names to the previous enabled state.
    """
    previous: dict[str, bool] = {}
    normalized = _normalize_lora_adapter_name(adapter_name) if adapter_name is not None else None
    matched = False
    for module_name, child in iter_lora_modules(module):
        adapter_names = child.adapter_names() if normalized is None else [normalized]
        for name in adapter_names:
            if not child.has_adapter(name):
                continue
            matched = True
            previous[_lora_enabled_state_key(module_name, name)] = child.is_adapter_enabled(name)
            child.set_adapter_enabled(name, bool(enabled))
    if normalized is not None and (not matched):
        raise ValueError(
            f"No LoRA adapter named `{normalized}` exists in module {type(module).__name__}."
        )
    return previous


def restore_lora_enabled(module: nn.Module, state: dict[str, bool]) -> None:
    """Restore LoRA runtime enabled flags captured by ``set_lora_enabled``.

    Args:
        module: Module tree containing optional LoRA adapters.
        state: Mapping from adapter module name to enabled state.

    Raises:
        ValueError: If any recorded adapter name is missing or is no longer LoRA.
    """
    modules = dict(iter_lora_modules(module))
    for key, enabled in state.items():
        (module_name, adapter_name) = _split_lora_enabled_state_key(key)
        child = modules.get(module_name)
        if not isinstance(child, LoRALinear):
            raise ValueError(
                f"Cannot restore LoRA enabled state for missing module `{module_name}`."
            )
        if adapter_name is None:
            child.enabled = bool(enabled)
        else:
            child.set_adapter_enabled(adapter_name, bool(enabled))


def set_lora_active_adapters(
    module: nn.Module, adapter_names: str | Sequence[str]
) -> dict[str, bool]:
    """Enable only the requested LoRA adapter names and disable all others."""
    if isinstance(adapter_names, str):
        requested = {_normalize_lora_adapter_name(adapter_names)}
    else:
        requested = {_normalize_lora_adapter_name(name) for name in adapter_names}
    if not requested:
        raise ValueError("At least one LoRA adapter name must be requested.")
    available: set[str] = set()
    for _, child in iter_lora_modules(module):
        available.update(child.adapter_names())
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"Requested LoRA adapters are missing from {type(module).__name__}: {missing}"
        )
    previous: dict[str, bool] = {}
    for module_name, child in iter_lora_modules(module):
        for name in child.adapter_names():
            previous[_lora_enabled_state_key(module_name, name)] = child.is_adapter_enabled(name)
            child.set_adapter_enabled(name, name in requested)
    return previous


@contextmanager
def lora_active_adapters(module: nn.Module, adapter_names: str | Sequence[str]):
    """Temporarily enable only selected LoRA adapters and restore prior states."""
    state = set_lora_active_adapters(module, adapter_names)
    try:
        yield
    finally:
        restore_lora_enabled(module, state)


@contextmanager
def lora_disabled(module: nn.Module):
    """Temporarily disable all LoRA adapters and restore their prior states."""
    state = set_lora_enabled(module, False)
    try:
        yield
    finally:
        restore_lora_enabled(module, state)


def _format_lora_param_name(module_name: str, suffix: str) -> str:
    return f"{module_name}.{suffix}" if module_name else suffix


def _single_adapter_name(child: LoRALinear, module_name: str) -> str:
    if child.adapter_count() != 1:
        raise ValueError(
            f"Multiple LoRA adapters are present under `{module_name or '<root>'}`; pass adapter_name to select one."
        )
    return child.adapter_names()[0]


def iter_lora_named_parameters(module: nn.Module, adapter_name: str | None = None):
    """Yield only LoRA parameter tensors from a module tree.

    Args:
        module: Module tree containing optional LoRA adapters.
        adapter_name: Optional adapter name. When provided, yielded keys use
            legacy `lora_A.weight` / `lora_B.weight` suffixes for that adapter.

    Yields:
        ``(name, parameter)`` pairs for ``lora_A.weight`` and ``lora_B.weight`` tensors.
    """
    normalized = _normalize_lora_adapter_name(adapter_name) if adapter_name is not None else None
    matched = False
    for module_name, child in iter_lora_modules(module):
        selected = (
            normalized if normalized is not None else _single_adapter_name(child, module_name)
        )
        if not child.has_adapter(selected):
            continue
        matched = True
        (lora_A_weight, lora_B_weight) = child.adapter_parameter_tensors(selected)
        yield (_format_lora_param_name(module_name, "lora_A.weight"), lora_A_weight)
        yield (_format_lora_param_name(module_name, "lora_B.weight"), lora_B_weight)
    if normalized is not None and (not matched):
        raise ValueError(
            f"No LoRA adapter named `{normalized}` exists in module {type(module).__name__}."
        )


@torch.no_grad()
def merge_lora_linear(
    layer: LoRALinear,
    adapter_name: str | None = None,
    *,
    allow_single_adapter_from_multi: bool = False,
) -> nn.Linear:
    """Merge one LoRA adapter into its base ``nn.Linear`` in place.

    Args:
        layer: LoRA-wrapped linear layer.
        adapter_name: Optional adapter name. Must be omitted for legacy
            single-adapter wrappers unless `allow_single_adapter_from_multi`
            is true.
        allow_single_adapter_from_multi: When true, merge only `adapter_name`
            from a multi-adapter wrapper and discard the other adapters by
            replacing the wrapper with the merged base layer.

    Returns:
        The original base ``nn.Linear`` with its weight updated.

    Raises:
        TypeError: If ``layer`` is not ``LoRALinear``.
        ValueError: If the LoRA delta shape does not match the base weight.
    """
    if not isinstance(layer, LoRALinear):
        raise TypeError(f"`merge_lora_linear` expects LoRALinear, got {type(layer)}")
    if adapter_name is None:
        selected = _single_adapter_name(layer, "")
    else:
        selected = _normalize_lora_adapter_name(adapter_name)
        if layer.adapter_count() != 1 and (not bool(allow_single_adapter_from_multi)):
            raise ValueError(
                "Cannot replace a multi-adapter LoRALinear by merging only one adapter."
            )
    (lora_A_weight, lora_B_weight) = layer.adapter_parameter_tensors(selected)
    (_, _, _, scaling) = layer._adapter_modules(selected)
    base = layer.base_layer
    delta = torch.mm(lora_B_weight.detach().float(), lora_A_weight.detach().float()) * float(
        scaling
    )
    if tuple(delta.shape) != tuple(base.weight.shape):
        raise ValueError(
            f"LoRA merge delta shape mismatch: delta={tuple(delta.shape)} base={tuple(base.weight.shape)}"
        )
    merged = base.weight.detach().float() + delta.to(device=base.weight.device)
    base.weight.data.copy_(merged.to(device=base.weight.device, dtype=base.weight.dtype))
    return base


def merge_lora_adapters(
    module: nn.Module,
    adapter_name: str | None = None,
    *,
    allow_single_adapter_from_multi: bool = False,
) -> list[str]:
    """Replace all descendant ``LoRALinear`` modules with merged base linears.

    Args:
        module: Module tree to mutate in place.
        adapter_name: Optional adapter name for single-adapter compatibility.
        allow_single_adapter_from_multi: When true, allow selecting one named
            adapter from multi-adapter wrappers and replacing the wrapper with
            the selected-adapter-merged base layer.

    Returns:
        Fully qualified names of adapters that were merged.
    """
    merged_names: list[str] = []

    def _merge_children(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                setattr(
                    parent,
                    child_name,
                    merge_lora_linear(
                        child,
                        adapter_name=adapter_name,
                        allow_single_adapter_from_multi=allow_single_adapter_from_multi,
                    ),
                )
                merged_names.append(qualified_name)
                continue
            _merge_children(child, qualified_name)

    _merge_children(module)
    return merged_names


def set_lora_requires_grad(
    module: nn.Module, requires_grad: bool, adapter_name: str | None = None
) -> None:
    """Toggle gradient tracking for all LoRA parameters in a module tree.

    Args:
        module: Module tree containing optional LoRA adapters.
        requires_grad: Desired ``requires_grad`` value for adapter weights.
        adapter_name: Optional adapter name.
    """
    for _, param in iter_lora_named_parameters(module, adapter_name=adapter_name):
        param.requires_grad_(requires_grad)


def collect_lora_state_dict(
    module: nn.Module, adapter_name: str | None = None
) -> dict[str, torch.Tensor]:
    """Export LoRA parameters into a detached CPU state dict.

    Args:
        module: Module tree containing LoRA adapters.
        adapter_name: Optional adapter name. Required when multiple adapters exist.

    Returns:
        Mapping from parameter name to cloned tensor on CPU.
    """
    return {
        name: param.detach().cpu().clone()
        for (name, param) in iter_lora_named_parameters(module, adapter_name=adapter_name)
    }


def load_lora_state_dict(
    module: nn.Module,
    lora_state_dict: dict[str, torch.Tensor],
    strict: bool = True,
    adapter_name: str | None = None,
) -> None:
    """Load LoRA-only weights into an already-injected module tree.

    Args:
        module: Module tree whose adapter parameters will be updated in place.
        lora_state_dict: Mapping from LoRA parameter name to tensor.
        strict: Whether to reject missing or unexpected keys.
        adapter_name: Optional adapter name. Required when multiple adapters exist.

    Raises:
        ValueError: If keys, tensor types, or tensor shapes do not match.
    """
    if not isinstance(lora_state_dict, dict):
        raise ValueError(f"`lora_state_dict` must be dict-like, got {type(lora_state_dict)}")
    expected_params = {
        name: param
        for (name, param) in iter_lora_named_parameters(module, adapter_name=adapter_name)
    }
    expected_keys = set(expected_params.keys())
    provided_keys = set(lora_state_dict.keys())
    missing_keys = sorted(expected_keys - provided_keys)
    unexpected_keys = sorted(provided_keys - expected_keys)
    if strict and (missing_keys or unexpected_keys):
        raise ValueError(
            f"LoRA adapter key mismatch: missing={missing_keys[:10]}{('...' if len(missing_keys) > 10 else '')}, unexpected={unexpected_keys[:10]}{('...' if len(unexpected_keys) > 10 else '')}"
        )
    for key, target in expected_params.items():
        if key not in lora_state_dict:
            continue
        value = lora_state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"`lora_state_dict[{key}]` must be torch.Tensor, got {type(value)}")
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"Shape mismatch for LoRA parameter `{key}`: expected {tuple(target.shape)}, got {tuple(value.shape)}"
            )
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))


def _should_target_module(module_name: str, target_modules: str) -> bool:
    """Return whether a qualified module name matches the configured LoRA preset.

    Args:
        module_name: Fully qualified module name from ``named_modules()``.
        target_modules: Preset name or ``all_linear``.

    Returns:
        ``True`` when the module should be wrapped with LoRA.
    """
    if target_modules == "all_linear":
        return True
    return any(module_name.endswith(suffix) for suffix in LORA_TARGET_PRESETS[target_modules])


def _replace_linear_with_lora(
    root: nn.Module, qualified_name: str, config: dict[str, Any], *, adapter_name: str | None = None
) -> None:
    """Replace one ``nn.Linear`` submodule with a ``LoRALinear`` wrapper.

    Args:
        root: Root module that owns the target submodule.
        qualified_name: Dotted path to the target linear layer.
        config: Normalized LoRA config.
        adapter_name: Optional adapter name.

    Raises:
        ValueError: If the target name is empty or already wrapped.
        TypeError: If the resolved target is not an ``nn.Linear``.
    """
    if not qualified_name:
        raise ValueError("LoRA target name cannot be empty.")
    (parent_name, _, child_name) = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    current = getattr(parent, child_name)
    if isinstance(current, LoRALinear):
        if adapter_name is None:
            raise ValueError(f"Module `{qualified_name}` is already LoRA-wrapped.")
        current.add_adapter(
            adapter_name,
            rank=int(config["rank"]),
            alpha=float(config["alpha"]),
            dropout=float(config["dropout"]),
        )
        return
    if not isinstance(current, nn.Linear):
        raise TypeError(f"LoRA target `{qualified_name}` must be nn.Linear, got {type(current)}")
    setattr(
        parent,
        child_name,
        LoRALinear(
            base_layer=current,
            rank=int(config["rank"]),
            alpha=float(config["alpha"]),
            dropout=float(config["dropout"]),
            adapter_name=adapter_name,
        ),
    )


def _iter_lora_injection_candidates(module: nn.Module, prefix: str = ""):
    for child_name, child in module.named_children():
        qualified_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, LoRALinear):
            yield (qualified_name, child)
            continue
        if isinstance(child, nn.Linear):
            yield (qualified_name, child)
            continue
        yield from _iter_lora_injection_candidates(child, qualified_name)


def inject_lora_adapters(
    module: nn.Module, lora_config: dict[str, Any], adapter_name: str | None = None
) -> list[str]:
    """Inject LoRA wrappers into matching linear layers under a module tree.

    Args:
        module: Module tree to modify in place.
        lora_config: User or normalized LoRA config.
        adapter_name: Optional adapter name. When provided, an additional
            adapter may be added to existing LoRA wrappers.

    Returns:
        Qualified names of the linear layers that were wrapped.

    Raises:
        ValueError: If LoRA is already present or no target layer matches the preset.
    """
    config = normalize_lora_config(lora_config)
    normalized_adapter = (
        _normalize_lora_adapter_name(adapter_name) if adapter_name is not None else None
    )
    if has_lora_adapters(module) and normalized_adapter is None:
        raise ValueError("Refusing to inject LoRA twice into the same module tree.")
    target_names = [
        name
        for (name, child) in _iter_lora_injection_candidates(module)
        if name
        and isinstance(child, (nn.Linear, LoRALinear))
        and _should_target_module(name, config["target_modules"])
    ]
    if not target_names:
        raise ValueError(
            f"No nn.Linear targets matched LoRA preset `{config['target_modules']}` in module {type(module).__name__}."
        )
    for name in target_names:
        _replace_linear_with_lora(module, name, config, adapter_name=normalized_adapter)
    return target_names
