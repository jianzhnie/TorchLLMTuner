"""Reject unavailable model-owned expert execution policies before allocation."""

import pytest
import torch

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_uninstalled_pre_down_policy_rejects_before_allocation(activation):
    """The math helper is not yet a selectable end-to-end model policy."""
    config = model_config(hidden_act=activation)
    config.experts_implementation = "torchtitan"
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="model-owned expert implementation"):
        Qwen3MoeForCausalLM(config)
    assert torch.equal(torch.get_rng_state(), rng)
