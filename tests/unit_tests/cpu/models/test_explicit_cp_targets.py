"""CP target binding consumes live, model-owned paths without inspection."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.common.attention.backend_names import CP_SDPA_BACKEND
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.model import KimiLinearForCausalLM
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.context_parallel import kcp
from llmtuner.parallel.context_parallel.context_parallel import (
    _attention_modules,
    apply_cp,
)
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


def test_cp_binding_preserves_state_and_uses_current_blocks():
    model = Qwen3MoeForCausalLM(model_config())
    keys = tuple(model.state_dict())
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    targets = _attention_modules(model)
    assert len(targets) == 2
    assert targets[0] is model.model.layers[0].self_attn
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())
    model.model.layers = nn.ModuleList([model.model.layers[1]])
    rebound = _attention_modules(model)
    assert len(rebound) == 1
    assert rebound[0] is targets[1]


def test_cp_missing_declaration_does_not_search_existing_attention():
    model = Qwen3MoeForCausalLM(model_config())
    model.decoder_structure = replace(model.decoder_structure, attention=None)
    with pytest.raises(ValueError, match="declared direct attention"):
        _attention_modules(model)


def test_cp_bad_later_slot_is_rejected_without_fallback_or_mutation():
    model = Qwen3MoeForCausalLM(model_config())
    block = model.model.layers[1]
    block.attn = block.self_attn
    del block.self_attn
    keys = tuple(model.state_dict())
    first = model.model.layers[0].self_attn
    with pytest.raises(ValueError, match="layers.1.*self_attn"):
        _attention_modules(model)
    assert tuple(model.state_dict()) == keys
    assert not hasattr(first, "_titan_cp_sdpa_kernel")


def test_cp_rejects_missing_input_semantics_before_mutation(monkeypatch):
    model = Qwen3MoeForCausalLM(model_config())
    before = model.config._attn_implementation
    monkeypatch.setattr(model, "set_cp_mesh", None)
    with pytest.raises(TypeError, match="set_cp_mesh input semantics"):
        apply_cp(
            model, mesh=None, cfg=ParallelConfig(context_parallel_kv_allgather_size=2)
        )
    assert model.config._attn_implementation == before


def test_cp_binding_keeps_aliased_block_slots():
    model = Qwen3MoeForCausalLM(model_config())
    block = model.model.layers[0]
    model.model.layers = nn.ModuleList([block, block])
    targets = _attention_modules(model)
    assert len(targets) == 2
    assert targets[0] is targets[1] is block.self_attn


@pytest.mark.parametrize("shared_attention", [False, True])
def test_kcp_declared_targets_ignore_unrelated_projection_modules(
    monkeypatch, shared_attention
):
    config = KimiLinearConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    model = KimiLinearForCausalLM(config)
    if shared_attention:
        model.model.layers.append(model.model.layers[0])
    auxiliary = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj"):
        auxiliary.add_module(name, nn.Linear(8, 8))
    model.auxiliary = auxiliary
    parameters = tuple(model.parameters())
    state_keys = tuple(model.state_dict())
    calls = []
    group = object()
    monkeypatch.setattr(kcp, "_kcp_runtime", lambda: ())

    def wrap(original, selected_group):
        calls.append((original, selected_group))
        return original

    monkeypatch.setattr(kcp, "kcp_kda_forward", wrap)
    attention = model.model.layers[0].self_attn
    assert kcp.apply_kcp(model, SimpleNamespace(get_group=lambda: group)) == 1
    assert calls == [(type(attention).forward, group)]
    assert tuple(model.state_dict()) == state_keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert "forward" not in auxiliary.__dict__


def test_kcp_rejects_declared_non_kimi_attention_before_runtime(monkeypatch):
    model = Qwen3MoeForCausalLM(model_config())
    before = tuple(layer.self_attn.forward for layer in model.model.layers)
    monkeypatch.setattr(
        kcp, "_kcp_runtime", lambda: pytest.fail("invalid targets reached runtime")
    )
    with pytest.raises(ValueError, match="declared Kimi KDA/MLA"):
        kcp.apply_kcp(model, cp_mesh=None)
    assert tuple(layer.self_attn.forward for layer in model.model.layers) == before


@pytest.mark.parametrize("mask", [torch.ones(1, 4), torch.zeros(1, 1, 4, 4)])
def test_qwen3_cp_rejects_explicit_masks_before_kernel_execution(mask):
    model = Qwen3MoeForCausalLM(model_config())
    model.config._attn_implementation = CP_SDPA_BACKEND
    with pytest.raises(UnsupportedCombinationError, match="unpadded causal input"):
        model(torch.arange(4), attention_mask=mask)


def test_qwen3_cp_marker_requires_an_installed_kernel():
    model = Qwen3MoeForCausalLM(model_config())
    model.config._attn_implementation = CP_SDPA_BACKEND
    with pytest.raises(RuntimeError, match="requires its installed kernel"):
        model(torch.arange(4))
