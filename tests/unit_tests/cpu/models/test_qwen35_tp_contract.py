"""Qwen3.5's model-owned tensor-parallel declarations."""

import pytest
import torch
from torch import nn

from llmtuner.config import ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.configuration import build_model_config
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from llmtuner.parallel.expert_parallel.model_replacement import resolve_ep_plan
from llmtuner.parallel.tensor_parallel.tensor_parallel import resolve_tp_plan
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
@pytest.mark.parametrize("block_id", ["0", "7"])
def test_qwen35_declares_a_resolvable_tp_plan(layer_type: str, block_id: str) -> None:
    config = build_model_config(
        "qwen3_5_moe_text",
        seq_len=32,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "layer_types": [layer_type],
        },
    )
    model = Qwen3_5MoeForCausalLM(config)
    if block_id != "0":
        layer = model.layers._modules.pop("0")
        model.layers.add_module(block_id, layer)

    plan = resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))

    assert plan.targets
    assert model.tp_config["lm_head"] == "replicated_with_grad_allreduce"
    block_targets = [
        target for target in plan.targets if target.path.startswith("model.layers.")
    ]
    assert block_targets
    assert all(
        target.path.startswith(f"model.layers.{block_id}.") for target in block_targets
    )
    assert not any(target.path.startswith("base_model") for target in plan.targets)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("target", ["experts", "router", "shared", "shared_gate"])
def test_qwen35_full_tp_plan_rejects_stale_moe_geometry(device, target):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    with torch.device(device):
        model = Qwen3_5MoeForCausalLM(config)
        block = model.layers[0].mlp
        if target == "experts":
            block.experts.gate_up_proj = nn.Parameter(
                block.experts.gate_up_proj[:, :-4]
            )
        elif target == "router":
            block.gate.weight = nn.Parameter(block.gate.weight[:, :-1])
        elif target == "shared_gate":
            block.shared_expert_gate.weight = nn.Parameter(
                block.shared_expert_gate.weight[:, :-1]
            )
        else:
            block.shared_expert.up_proj.weight = nn.Parameter(
                block.shared_expert.up_proj.weight[:, :-1]
            )
    identities = tuple(id(parameter) for parameter in model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="storage differs from declared geometry"):
        resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    assert identities == tuple(id(parameter) for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("axis", ["tp_degree", "ep_degree"])
@pytest.mark.parametrize("degree", [0, -1, True, 1.5])
def test_qwen35_model_parallel_contract_rejects_invalid_degrees(axis, degree):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(config)
    with pytest.raises(ValueError, match="positive integer"):
        model.validate_parallel(**{"tp_degree": 1, "ep_degree": 1, axis: degree})


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize(
    "target", ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]
)
def test_qwen35_full_tp_plan_rejects_stale_attention_geometry(device, target):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    with torch.device(device):
        model = Qwen3_5MoeForCausalLM(config)
        projection = getattr(model.layers[0].self_attn, target)
        projection.weight = nn.Parameter(projection.weight[:-2])
    identities = tuple(id(parameter) for parameter in model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="storage differs from declared geometry"):
        resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    assert identities == tuple(id(parameter) for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("declared_bias", [False, True])
@pytest.mark.parametrize("target", ["q_proj", "k_proj", "v_proj", "o_proj"])
def test_qwen35_attention_bias_contract_checks_presence(declared_bias, target):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config.attention_bias = declared_bias
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(config)
        model.validate_parallel(tp_degree=2, ep_degree=1)
        projection = getattr(model.layers[0].self_attn, target)
        projection.bias = (
            None
            if declared_bias
            else nn.Parameter(torch.empty(projection.weight.shape[0]))
        )
    with pytest.raises(ValueError, match="bias differs from declared geometry"):
        model.validate_parallel(tp_degree=2, ep_degree=1)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize(
    "target",
    [
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "conv1d",
        "norm",
        "dt_bias",
        "A_log",
    ],
)
def test_qwen35_full_tp_plan_rejects_stale_gdn_geometry(device, target):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config.layer_types = ["linear_attention"] * config.num_hidden_layers
    with torch.device(device):
        model = Qwen3_5MoeForCausalLM(config)
        attention = model.layers[0].linear_attn
        if target in ("dt_bias", "A_log"):
            setattr(attention, target, nn.Parameter(getattr(attention, target)[:-1]))
        else:
            projection = getattr(attention, target)
            projection.weight = nn.Parameter(projection.weight[:-1])
    identities = tuple(id(parameter) for parameter in model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(
        ValueError, match="GDN .* storage differs from declared geometry"
    ):
        resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    assert identities == tuple(id(parameter) for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("target", ["embedding", "norm", "head"])
def test_qwen35_full_tp_plan_rejects_stale_vocab_geometry(device, target):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    with torch.device(device):
        model = Qwen3_5MoeForCausalLM(config)
        module = {
            "embedding": model.tok_embeddings,
            "norm": model.norm,
            "head": model.lm_head,
        }[target]
        module.weight = nn.Parameter(module.weight[:-2])
    identities = tuple(id(parameter) for parameter in model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="storage differs from declared geometry"):
        resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    assert identities == tuple(id(parameter) for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
def test_qwen35_ep_only_rejects_gated_shared_expert_without_side_effects(
    device, layer_type
):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config.layer_types = [layer_type] * config.num_hidden_layers
    with torch.device(device):
        model = Qwen3_5MoeForCausalLM(config)
    identities = tuple(id(parameter) for parameter in model.parameters())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    # A valid native layout is insufficient: generic EP cannot preserve the
    # sigmoid-gated shared branch yet. Do not turn this into additive math.
    model.validate_parallel(tp_degree=1, ep_degree=2)
    with pytest.raises(UnsupportedCombinationError, match="gates its shared expert"):
        resolve_ep_plan(model, degree=2)
    assert identities == tuple(id(parameter) for parameter in model.parameters())
    assert keys == tuple(model.state_dict())
    assert torch.equal(torch.get_rng_state(), rng)
    with torch.device(device):
        model.layers[0].mlp.shared_expert_gate.weight = nn.Parameter(
            model.layers[0].mlp.shared_expert_gate.weight[:, :-1]
        )
    damaged_identities = tuple(id(parameter) for parameter in model.parameters())
    with pytest.raises(ValueError, match="shared gate storage differs"):
        model.validate_parallel(tp_degree=1, ep_degree=2)
    assert damaged_identities == tuple(
        id(parameter) for parameter in model.parameters()
    )
    assert keys == tuple(model.state_dict())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
@pytest.mark.parametrize("norm_name", ["input_layernorm", "post_attention_layernorm"])
def test_qwen35_residual_norm_geometry_rejected_before_tp(
    device, layer_type, norm_name
):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config.layer_types = [layer_type] * config.num_hidden_layers
    with torch.device(device):
        model = Qwen3_5MoeForCausalLM(config)
        norm = getattr(model.layers[0], norm_name)
        norm.weight = nn.Parameter(norm.weight[:-1])
    identities = tuple(id(parameter) for parameter in model.parameters())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=f"{norm_name} storage differs"):
        resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    assert identities == tuple(id(parameter) for parameter in model.parameters())
    assert keys == tuple(model.state_dict())
    assert torch.equal(torch.get_rng_state(), rng)
