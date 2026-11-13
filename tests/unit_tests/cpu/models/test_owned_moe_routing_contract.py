"""Routing declarations belong to standalone model-owned expert blocks."""

import pytest
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM, Qwen3MoeSparseMoeBlock
from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeSparseMoeBlock,
)
from llmtuner.parallel.expert_parallel import block_conversion
from llmtuner.parallel.tensor_parallel.tensor_parallel import resolve_tp_plan
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
def test_standalone_routing_resolves_without_inspection(family, monkeypatch):
    if family == "qwen3":
        block = Qwen3MoeSparseMoeBlock(model_config(norm_topk_prob=False))
    else:
        block = Qwen3_5MoeSparseMoeBlock(
            Qwen3_5MoeTextConfig(
                hidden_size=16,
                num_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                shared_expert_intermediate_size=8,
            )
        )
    parameters = dict(block.named_parameters())
    keys = tuple(block.state_dict())
    rng = torch.get_rng_state().clone()

    def forbidden():
        raise AssertionError("explicit block must not enter legacy inspection")

    routing = block_conversion.resolve_moe_routing(block)
    assert routing.source is block and routing.router is block.gate
    assert routing.top_k == 2 and routing.score_func == "softmax"
    assert routing.route_norm == (family == "qwen35")
    if family == "qwen35":
        assert routing.shared is block.shared_expert
        with pytest.raises(UnsupportedCombinationError, match="shared"):
            block_conversion.validate_moe_conversion(
                block,
                routing,
                tp_enabled=False,
                quantile_balancing=False,
                token_dispatcher="alltoall",
                has_ep_group=False,
            )
    else:
        assert routing.shared is None
    assert keys == tuple(block.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(dict(block.named_parameters())[k] is v for k, v in parameters.items())


@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
def test_tp_declaration_retains_shared_block_paths_without_mutation(family):
    if family == "qwen3":
        model = Qwen3MoeForCausalLM(model_config())
    else:
        config = Qwen3_5MoeTextConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            layer_types=["full_attention"],
        )
        model = Qwen3_5MoeForCausalLM(config)
    layer = model.layers[0]
    model.model.layers = torch.nn.ModuleList()
    model.model.layers.add_module("3", layer)
    model.model.layers.add_module("7", layer)
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    plan = model.tp_config
    first = {
        name.removeprefix("model.layers.3."): value
        for name, value in plan.items()
        if name.startswith("model.layers.3.")
    }
    second = {
        name.removeprefix("model.layers.7."): value
        for name, value in plan.items()
        if name.startswith("model.layers.7.")
    }
    assert first and first == second
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())


def test_shared_qwen3_block_tp_rejects_without_mutating_aliases():
    model = Qwen3MoeForCausalLM(model_config())
    layer = model.layers[0]
    model.model.layers = torch.nn.ModuleList([layer, layer])
    parameters = dict(model.named_parameters(remove_duplicate=False))
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="shared weight"):
        resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    assert model.layers[0] is model.layers[1]
    after = dict(model.named_parameters(remove_duplicate=False))
    assert after.keys() == parameters.keys()
    assert all(after[name] is value for name, value in parameters.items())
    assert tuple(model.state_dict()) == keys
    assert torch.equal(rng, torch.get_rng_state())
