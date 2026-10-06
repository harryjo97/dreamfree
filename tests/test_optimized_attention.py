from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from dreamfree.evaluation.droid.deploy_policy_optimized import OptimizedDroidInferenceMixin
from dreamfree.models.compile import compile_dit_blocks
from dreamfree.models.inference_optimization import fuse_attention_projections
from dreamfree.models.wan22.action_dit import ActionDiT


def _action_model() -> ActionDiT:
    torch.manual_seed(7)
    return ActionDiT(
        hidden_dim=16,
        action_dim=4,
        ffn_dim=32,
        text_dim=12,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
    ).eval()


def _inputs():
    torch.manual_seed(11)
    return {
        "action_tokens": torch.randn(1, 4, 4),
        "timestep": torch.tensor([500.0]),
        "context": torch.randn(1, 5, 12),
        "context_mask": torch.ones(1, 5, dtype=torch.bool),
        "query_prefix_emb": torch.randn(1, 1, 16),
        "extra_context_emb": torch.randn(1, 3, 16),
        "extra_context_mask": torch.ones(1, 3, dtype=torch.bool),
    }


def _uncached(model: ActionDiT, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return model.forward_with_query_prefix_and_extra_context_emb(
        **inputs,
        query_prefix_timestep_sigma=0.0,
    )


def _cached(model: ActionDiT, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    prepared = model.prepare_cross_attention_conditioning(
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        extra_context_emb=inputs["extra_context_emb"],
        extra_context_mask=inputs["extra_context_mask"],
    )
    return model.forward_with_query_prefix_and_prepared_context(
        action_tokens=inputs["action_tokens"],
        timestep=inputs["timestep"],
        prepared_conditioning=prepared,
        query_prefix_emb=inputs["query_prefix_emb"],
        query_prefix_timestep_sigma=0.0,
    )


def test_action_dit_prepared_cross_attention_matches_uncached_forward():
    model = _action_model()
    inputs = _inputs()

    expected = _uncached(model, inputs)
    actual = _cached(model, inputs)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_action_dit_prepared_context_projects_each_layer_once():
    model = _action_model()
    inputs = _inputs()
    calls = [0 for _ in model.blocks]
    for idx, block in enumerate(model.blocks):
        original = block.cross_attn.project_context

        def counted(context, *, _idx=idx, _original=original):
            calls[_idx] += 1
            return _original(context)

        block.cross_attn.project_context = counted

    prepared = model.prepare_cross_attention_conditioning(
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        extra_context_emb=inputs["extra_context_emb"],
        extra_context_mask=inputs["extra_context_mask"],
    )
    for timestep in (100.0, 200.0, 300.0):
        model.forward_with_query_prefix_and_prepared_context(
            action_tokens=inputs["action_tokens"],
            timestep=torch.tensor([timestep]),
            prepared_conditioning=prepared,
            query_prefix_emb=inputs["query_prefix_emb"],
            query_prefix_timestep_sigma=0.0,
        )

    assert calls == [1, 1]


def test_action_dit_prepared_context_rejects_nontrivial_cross_mask():
    model = _action_model()
    inputs = _inputs()
    inputs["context_mask"][0, -1] = False

    with pytest.raises(ValueError, match="all-true text context mask"):
        model.prepare_cross_attention_conditioning(
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            extra_context_emb=inputs["extra_context_emb"],
            extra_context_mask=inputs["extra_context_mask"],
        )


def test_action_dit_prepared_context_rejects_stale_metadata():
    model = _action_model()
    inputs = _inputs()
    prepared = model.prepare_cross_attention_conditioning(
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        extra_context_emb=inputs["extra_context_emb"],
        extra_context_mask=inputs["extra_context_mask"],
    )

    with pytest.raises(ValueError, match="K/V shape mismatch at layer 0"):
        model.forward_with_query_prefix_and_prepared_context(
            action_tokens=inputs["action_tokens"],
            timestep=inputs["timestep"],
            prepared_conditioning=replace(
                prepared,
                context_length=prepared.context_length + 1,
            ),
            query_prefix_emb=inputs["query_prefix_emb"],
            query_prefix_timestep_sigma=0.0,
        )


def test_qkv_fusion_preserves_cached_action_dit_output():
    baseline = _action_model()
    fused = copy.deepcopy(baseline)
    inputs = _inputs()
    expected = _cached(baseline, inputs)

    counts = fuse_attention_projections(fused)
    actual = _cached(fused, inputs)

    assert counts == {"self_qkv": 2, "cross_kv": 2}
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    for block in fused.blocks:
        assert block.self_attn.qkv is not None
        assert block.self_attn.q is None
        assert block.cross_attn.kv is not None
        assert block.cross_attn.k is None


def test_compile_dit_blocks_forwards_no_cudagraph_mode(monkeypatch):
    model = _action_model()
    calls = []

    def fake_compile(forward, **kwargs):
        calls.append(kwargs)
        return forward

    monkeypatch.setattr(torch, "compile", fake_compile)
    count = compile_dit_blocks(
        model,
        label="test ActionDiT",
        mode="max-autotune-no-cudagraphs",
    )

    assert count == 2
    assert calls == [
        {"dynamic": False, "mode": "max-autotune-no-cudagraphs"},
        {"dynamic": False, "mode": "max-autotune-no-cudagraphs"},
    ]


def test_standard_query_prefix_preserves_proprioception():
    policy = OptimizedDroidInferenceMixin()
    proprio = torch.randn(1, 1, 16)
    prefix, mask = policy._prepare_action_query(proprio_query_emb=proprio, action_seq_len=5)
    assert prefix is proprio
    assert mask is None
