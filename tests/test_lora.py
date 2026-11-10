import pytest
import torch

from dreamfree.models.lora import (
    LoRALinear,
    collect_lora_state_dict,
    has_lora_adapters,
    inject_lora_adapters,
    is_lora_parameter_name,
    load_lora_state_dict,
    lora_active_adapters,
    lora_disabled,
    merge_lora_adapters,
    merge_lora_linear,
    normalize_lora_config,
    set_lora_requires_grad,
)


class _ToyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4, 4)
        self.k = torch.nn.Linear(4, 4)
        self.v = torch.nn.Linear(4, 4)
        self.o = torch.nn.Linear(4, 4)


class _ToyDiTBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ToyAttention()
        self.cross_attn = _ToyAttention()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.GELU(),
            torch.nn.Linear(8, 4),
        )


def test_cross_attention_only_lora_targets_cross_attention_qkvo():
    block = _ToyDiTBlock()
    config = normalize_lora_config(
        {
            "enabled": True,
            "rank": 2,
            "alpha": 2.0,
            "dropout": 0.0,
            "target_modules": "cross_attention_only",
        }
    )

    injected = inject_lora_adapters(block, config)

    assert injected == [
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
    ]
    assert all(
        isinstance(getattr(block.cross_attn, name), LoRALinear) for name in ("q", "k", "v", "o")
    )
    assert all(
        isinstance(getattr(block.self_attn, name), torch.nn.Linear) for name in ("q", "k", "v", "o")
    )
    assert isinstance(block.ffn[0], torch.nn.Linear)
    assert isinstance(block.ffn[2], torch.nn.Linear)


def test_lora_target_validation_lists_cross_attention_only():
    with pytest.raises(ValueError, match="cross_attention_only"):
        normalize_lora_config({"target_modules": "unknown"})


def test_lora_disabled_matches_base_and_restores_after_exception():
    base = torch.nn.Linear(3, 2, bias=True)
    layer = LoRALinear(base, rank=2, alpha=2.0, dropout=0.0)
    with torch.no_grad():
        layer.lora_A.weight.fill_(0.5)
        layer.lora_B.weight.fill_(0.25)
    x = torch.randn(4, 3)

    assert not torch.allclose(layer(x), layer.base_layer(x))
    with pytest.raises(RuntimeError, match="boom"):
        with lora_disabled(layer):
            assert torch.allclose(layer(x), layer.base_layer(x))
            raise RuntimeError("boom")
    assert layer.enabled is True
    assert not torch.allclose(layer(x), layer.base_layer(x))


def test_merge_lora_linear_preserves_output_dtype_device_and_bias():
    base = torch.nn.Linear(3, 2, bias=True).to(dtype=torch.float32)
    layer = LoRALinear(base, rank=2, alpha=2.0, dropout=0.0).eval()
    with torch.no_grad():
        layer.lora_A.weight.normal_()
        layer.lora_B.weight.normal_()
    x = torch.randn(5, 3)
    expected = layer(x)
    bias = layer.bias.detach().clone()

    merged = merge_lora_linear(layer)

    assert isinstance(merged, torch.nn.Linear)
    assert merged.weight.dtype == torch.float32
    assert torch.equal(merged.bias, bias)
    assert torch.allclose(merged(x), expected, atol=1e-5)


def test_merge_lora_adapters_replaces_wrappers_and_allows_new_injection():
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    inject_lora_adapters(
        module,
        {"enabled": True, "rank": 2, "alpha": 2.0, "dropout": 0.0, "target_modules": "all_linear"},
    )
    assert has_lora_adapters(module)
    with pytest.raises(ValueError, match="twice"):
        inject_lora_adapters(
            module,
            {
                "enabled": True,
                "rank": 2,
                "alpha": 2.0,
                "dropout": 0.0,
                "target_modules": "all_linear",
            },
        )

    merged = merge_lora_adapters(module)

    assert merged == ["0", "2"]
    assert not has_lora_adapters(module)
    reinjected = inject_lora_adapters(
        module,
        {"enabled": True, "rank": 2, "alpha": 2.0, "dropout": 0.0, "target_modules": "all_linear"},
    )
    assert reinjected == ["0", "2"]


def test_named_lora_adapters_activate_and_checkpoint_independently():
    config = {
        "enabled": True,
        "rank": 2,
        "alpha": 2.0,
        "dropout": 0.0,
        "target_modules": "all_linear",
    }
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    inject_lora_adapters(module, config, adapter_name="stage0")
    inject_lora_adapters(module, config, adapter_name="stage1")

    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("lora_A.weight"):
                param.fill_(0.1)
            elif name.endswith("lora_B.weight"):
                param.fill_(0.2)
            elif "._extra_lora_A.stage1.weight" in name:
                param.fill_(0.3)
            elif "._extra_lora_B.stage1.weight" in name:
                param.fill_(0.4)

    x = torch.randn(5, 3)
    with lora_active_adapters(module, "stage0"):
        stage0_out = module(x)
    with lora_active_adapters(module, "stage1"):
        stage1_out = module(x)

    assert not torch.allclose(stage0_out, stage1_out)
    with pytest.raises(ValueError, match="Multiple LoRA adapters"):
        collect_lora_state_dict(module)

    stage1_state = collect_lora_state_dict(module, adapter_name="stage1")
    assert sorted(stage1_state) == [
        "0.lora_A.weight",
        "0.lora_B.weight",
        "2.lora_A.weight",
        "2.lora_B.weight",
    ]

    reloaded = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    inject_lora_adapters(reloaded, config, adapter_name="stage0")
    inject_lora_adapters(reloaded, config, adapter_name="stage1")
    with torch.no_grad():
        reloaded[0].base_layer.weight.copy_(module[0].base_layer.weight)
        reloaded[0].base_layer.bias.copy_(module[0].base_layer.bias)
        reloaded[2].base_layer.weight.copy_(module[2].base_layer.weight)
        reloaded[2].base_layer.bias.copy_(module[2].base_layer.bias)
    load_lora_state_dict(reloaded, stage1_state, strict=True, adapter_name="stage1")

    with lora_active_adapters(reloaded, "stage1"):
        torch.testing.assert_close(reloaded(x), stage1_out)

    set_lora_requires_grad(module, False, adapter_name="stage0")
    set_lora_requires_grad(module, True, adapter_name="stage1")
    for name, param in module.named_parameters():
        if not is_lora_parameter_name(name):
            assert not param.requires_grad
        elif "stage1" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad


@pytest.mark.parametrize("merge", [False, True])
def test_frozen_student_loader_applies_requested_lora_merge(monkeypatch, tmp_path, merge):
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from dreamfree.models import student as loader

    core = torch.nn.Module()
    core.dit = torch.nn.Sequential(
        LoRALinear(torch.nn.Linear(3, 2), rank=2, alpha=2.0, dropout=0.0)
    )
    with torch.no_grad():
        core.dit[0].lora_A.weight.fill_(0.5)
        core.dit[0].lora_B.weight.fill_(0.25)
    inputs = torch.randn(4, 3)
    expected = core.dit(inputs).detach()
    save_file(collect_lora_state_dict(core.dit), str(tmp_path / "stage1_lora.safetensors"))
    source = loader.Stage1StudentSource(
        checkpoint_dir=tmp_path,
        metadata={},
        global_step=1,
        student_layer_idx=0,
        student_hook_point="post_block",
        student_video_attention_mask_mode="first_frame_causal",
        student_text_scope="all_frames",
        student_i2v_condition_mode="none",
        expected_student_tokens_per_frame=1,
        expected_student_hidden_dim=2,
        latent_shape=(1, 1, 1, 1),
    )
    monkeypatch.setattr(loader, "_build_stage1_loader_config", lambda cfg, source: cfg)
    monkeypatch.setattr(
        loader, "load_stage0_merged_backbone", lambda *args, **kw: SimpleNamespace(core=core)
    )
    result = loader.load_frozen_stage1_student(
        object(),
        source=source,
        device="cpu",
        dtype=torch.float32,
        merge_student_lora=merge,
    )
    torch.testing.assert_close(result.core.dit(inputs), expected)
    assert result.student_lora_merged is merge
    assert has_lora_adapters(result.core.dit) is not merge
    assert not any(parameter.requires_grad for parameter in result.core.parameters())
