"""Dense Qwen3 layer math and initialization belong to the local model."""

import pytest
import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeMLP as ReferenceMLP,
)

from llmtuner.models.qwen.qwen3.model import Qwen3MoeMLP
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("width", [None, 32])
def test_owned_mlp_preserves_initialization_and_backward(
    dtype, activation, width, monkeypatch
):
    config = model_config(hidden_act=activation)
    torch.manual_seed(107)
    reference = ReferenceMLP(config, intermediate_size=width).to(dtype)
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("local MLP must not use inherited construction or forward")

    monkeypatch.setattr(ReferenceMLP, "__init__", forbidden)
    torch.manual_seed(107)
    actual = Qwen3MoeMLP(config, intermediate_size=width).to(dtype)
    assert torch.equal(rng, torch.get_rng_state())
    assert actual.state_dict().keys() == reference.state_dict().keys()
    for name, value in actual.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name])
    hidden = torch.randn(2, 3, 16, dtype=dtype, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    expected = reference(hidden)
    monkeypatch.setattr(ReferenceMLP, "forward", forbidden)
    output = actual(other)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    probe = torch.randn_like(output)
    output.backward(probe)
    expected.backward(probe)
    torch.testing.assert_close(other.grad, hidden.grad, rtol=0, atol=0)
    for name, parameter in actual.named_parameters():
        target = dict(reference.named_parameters())[name]
        torch.testing.assert_close(parameter.grad, target.grad, rtol=0, atol=0)
    assert actual.gate_proj.weight.grad.count_nonzero() > 0
