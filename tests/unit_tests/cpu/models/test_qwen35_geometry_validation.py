"""Hybrid model geometry must fail before weight allocation or RNG changes."""

import pytest
import torch
from torch import nn

from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextModel,
)
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


@pytest.mark.parametrize("constructor", [Qwen3_5MoeForCausalLM, Qwen3_5MoeTextModel])
@pytest.mark.parametrize(
    "changes",
    [
        {"layer_types": []},
        {"layer_types": "linear_attention"},
        {"layer_types": ["unknown"]},
        {"num_hidden_layers": True},
        {
            "layer_types": ["full_attention"],
            "num_attention_heads": 3,
            "num_key_value_heads": 2,
        },
        {"layer_types": ["full_attention"], "head_dim": 0},
        {"linear_num_key_heads": 2, "linear_num_value_heads": 3},
        {"linear_num_key_heads": 0},
        {"linear_num_value_heads": True},
        {"linear_key_head_dim": 0},
        {"linear_value_head_dim": -1},
        {"linear_conv_kernel_dim": 0},
        {"vocab_size": 0},
        {"hidden_size": True},
        {"num_experts": 0},
        {"num_experts": True},
        {"num_experts_per_tok": 0},
        {"num_experts_per_tok": 1.5},
        {"num_experts": 2, "num_experts_per_tok": 3},
        {"moe_intermediate_size": -1},
        {"shared_expert_intermediate_size": 0},
    ],
)
def test_invalid_geometry_fails_before_embedding_allocation(
    monkeypatch, constructor, changes
):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config.num_hidden_layers = 1
    config.layer_types = ["linear_attention"]
    for name, value in changes.items():
        # Exercise the model boundary independently of external config validators.
        object.__setattr__(config, name, value)
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid model geometry reached weight allocation")

    monkeypatch.setattr(nn.Embedding, "__init__", forbidden)
    with pytest.raises(ValueError, match="Qwen3.5"):
        constructor(config)
    assert torch.equal(torch.get_rng_state(), rng)
