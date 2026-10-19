"""Kimi SiTU formula and training integration."""

import pytest
import torch

from llmtuner.models.common.activation import SituGLU
from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
from llmtuner.models.kimi.model import KimiMLP
from llmtuner.models.kimi.moe import KimiExpertMLP


@pytest.mark.parametrize("linear_beta", [None, 0.0, 1.7])
def test_situ_matches_explicit_formula_and_gradients(linear_beta):
    gate = torch.randn(2, 5, requires_grad=True)
    up = torch.randn(2, 5, requires_grad=True)
    actual = SituGLU(2.0, linear_beta)(gate, up)
    bounded_gate = 2 * torch.tanh(gate / 2)
    bounded_up = up if not linear_beta else linear_beta * torch.tanh(up / linear_beta)
    expected = bounded_gate / (1 + (-gate).exp()) * bounded_up
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (gate, up), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (gate, up))
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b)


def test_kimi_dense_and_expert_mlp_use_situ_without_global_hf_registration():
    config = KimiLinearConfig(
        hidden_size=8,
        intermediate_size=12,
        hidden_act="situ",
        activation_situ_beta=2.0,
        activation_situ_linear_beta=1.5,
    )
    for model in (KimiMLP(config), KimiExpertMLP(config, 8, 12)):
        inputs = torch.randn(4, 8, requires_grad=True)
        model(inputs).sum().backward()
        assert torch.isfinite(inputs.grad).all()
        assert all(p.grad is not None for p in model.parameters())
