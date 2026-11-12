"""Qwen3 declarations describe real storage without changing model state."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as ReferenceQwen3MoeForCausalLM,
)

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel.model_replacement import resolve_ep_plan


def model_config(**kwargs):
    values = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[1],
        tie_word_embeddings=True,
    )
    return Qwen3MoeConfig(**(values | kwargs))


def test_exact_plan_preserves_rng_aliases_and_state_keys():
    model = Qwen3MoeForCausalLM(model_config())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    plan = model.tp_config
    assert plan["model.layers.0.mlp.experts.gate_up_proj"] == "packed_colwise"
    assert plan["model.layers.1.mlp.gate_proj"] == "colwise"
    assert not any("*" in name for name in plan)
    assert "model.embed_tokens" not in plan
    assert model.enable_weight_tying
    for path in plan:
        if path.endswith(("experts.gate_up_proj", "experts.down_proj")):
            assert isinstance(model.get_parameter(path), nn.Parameter)
        else:
            assert isinstance(model.get_submodule(path), nn.Module)
    with pytest.raises(TypeError):
        plan["lm_head"] = "rowwise"
    model.validate_parallel(tp_degree=2, ep_degree=1)
    assert tuple(model.state_dict()) == keys
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("degree", [0, -1, True, 1.5])
@pytest.mark.parametrize("axis", ["tp_degree", "ep_degree"])
def test_parallel_contract_rejects_invalid_degrees(axis, degree):
    model = Qwen3MoeForCausalLM(model_config())
    values = {"tp_degree": 1, "ep_degree": 1, axis: degree}
    with pytest.raises(ValueError, match="positive integer"):
        model.validate_parallel(**values)


def test_parallel_contract_checks_actual_packed_storage():
    model = Qwen3MoeForCausalLM(model_config())
    model.model.layers[0].mlp.experts.gate_up_proj = nn.Parameter(
        torch.empty(4, 12, 16)
    )
    with pytest.raises(ValueError, match="gate/up storage"):
        model.validate_parallel(tp_degree=1, ep_degree=1)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("projection", ["q_proj", "k_proj", "v_proj", "o_proj"])
def test_parallel_contract_checks_actual_attention_storage(device, projection):
    with torch.device(device):
        model = Qwen3MoeForCausalLM(model_config())
        linear = getattr(model.layers[0].self_attn, projection)
        linear.weight = nn.Parameter(torch.empty_like(linear.weight)[:, :-1])
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=f"{projection} storage differs"):
        model.validate_parallel(tp_degree=1, ep_degree=2)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


def test_joint_layout_does_not_require_expert_width_divisible_by_tp():
    model = Qwen3MoeForCausalLM(model_config(moe_intermediate_size=13))
    model.validate_parallel(tp_degree=2, ep_degree=2)
    with pytest.raises(ValueError, match="Expert width"):
        model.validate_parallel(tp_degree=2, ep_degree=1)


def test_meta_parallel_contract_uses_shapes_without_materialization():
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(model_config())
    model.validate_parallel(tp_degree=2, ep_degree=2)
    assert model.model.embed_tokens.weight.is_meta


def test_explicit_initialization_preserves_reference_rng_state_and_aliases():
    config = model_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        reference = ReferenceQwen3MoeForCausalLM(deepcopy(config))
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(1729)
        actual = Qwen3MoeForCausalLM(deepcopy(config))
        actual_rng = torch.get_rng_state().clone()
    assert torch.equal(actual_rng, reference_rng)
    expected_state = reference.state_dict()
    actual_state = actual.state_dict()
    assert tuple(actual_state) == tuple(expected_state)
    assert len(actual_state) > 10
    for name, expected in expected_state.items():
        torch.testing.assert_close(actual_state[name], expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual.model.embed_tokens.weight) > 0
    assert actual.lm_head.weight is actual.model.embed_tokens.weight
    assert reference.lm_head.weight is reference.model.embed_tokens.weight


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("projection", ["q_proj", "k_proj", "v_proj", "o_proj"])
@pytest.mark.parametrize("change", ["missing", "shape", "unexpected"])
def test_parallel_contract_checks_attention_bias_geometry(device, projection, change):
    with torch.device(device):
        model = Qwen3MoeForCausalLM(model_config(attention_bias=change != "unexpected"))
        linear = getattr(model.layers[0].self_attn, projection)
        linear.bias = (
            None
            if change == "missing"
            else nn.Parameter(torch.empty(linear.out_features + (change == "shape")))
        )
    rng = torch.get_rng_state().clone()
    parameters = tuple(model.parameters())
    with pytest.raises(ValueError, match=f"{projection} bias differs"):
        model.validate_parallel(tp_degree=1, ep_degree=2)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("bias", [False, True])
def test_parallel_contract_accepts_declared_attention_bias(device, bias):
    with torch.device(device):
        model = Qwen3MoeForCausalLM(model_config(attention_bias=bias))
    model.validate_parallel(tp_degree=1, ep_degree=2)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("target", ["gate_proj", "up_proj", "down_proj", "router"])
def test_parallel_contract_checks_dense_and_router_storage(device, target):
    with torch.device(device):
        model = Qwen3MoeForCausalLM(model_config())
        projection = (
            model.layers[0].mlp.gate
            if target == "router"
            else getattr(model.layers[1].mlp, target)
        )
        projection.weight = nn.Parameter(projection.weight[:, :-1])
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="storage differs from declared geometry"):
        model.validate_parallel(tp_degree=1, ep_degree=2)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("target", ["q_norm", "k_norm", "embedding", "norm", "head"])
def test_parallel_contract_checks_norm_and_vocabulary_storage(device, target):
    with torch.device(device):
        model = Qwen3MoeForCausalLM(model_config())
        module = {
            "q_norm": model.layers[0].self_attn.q_norm,
            "k_norm": model.layers[0].self_attn.k_norm,
            "embedding": model.tok_embeddings,
            "norm": model.norm,
            "head": model.lm_head,
        }[target]
        module.weight = nn.Parameter(module.weight[:-1])
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="storage differs from declared geometry"):
        model.validate_parallel(tp_degree=1, ep_degree=2)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))


@pytest.mark.parametrize("first,last", [(True, False), (False, False), (False, True)])
def test_parallel_contract_validates_only_pipeline_owned_vocabulary(first, last):
    model = Qwen3MoeForCausalLM(model_config(tie_word_embeddings=False))
    model.configure_pipeline_stage(first=first, last=last)
    if not first:
        model.model.embed_tokens = nn.Identity()
    if not last:
        model.model.norm = nn.Identity()
        model.lm_head = nn.Identity()
    model.validate_parallel(tp_degree=1, ep_degree=2)
    plan = resolve_ep_plan(model, degree=2)
    assert plan.targets and all(target.module is not None for target in plan.targets)
    if first or last:
        owned = model.tok_embeddings if first else model.lm_head
        owned.weight = nn.Parameter(owned.weight[:-1])
        with pytest.raises(ValueError, match="storage differs from declared geometry"):
            model.validate_parallel(tp_degree=1, ep_degree=2)
