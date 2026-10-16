"""Independent per-token causal sums, including packed document resets."""

import pytest
import torch
from torch.nn import functional as F

from llmtuner.models.kimi.causal_convolution import KimiCausalConvolution


@pytest.mark.parametrize("boundaries", [None, [0, 1, 4, 6]])
def test_kimi_convolution_matches_per_token_sums_and_gradients(boundaries):
    torch.manual_seed(7)
    model = KimiCausalConvolution(3, 3).double()
    inputs = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
    reference_input = inputs.detach().clone().requires_grad_()
    reference_weight = model.weight.detach().clone().requires_grad_()
    expected = []
    for t in range(6):
        start = max(b for b in (boundaries or [0]) if b <= t)
        terms = [
            reference_input[:, t - lag] * reference_weight[:, 0, 2 - lag]
            for lag in range(min(3, t - start + 1))
        ]
        expected.append(F.silu(torch.stack(terms).sum(0)))
    expected = torch.stack(expected, dim=1)
    actual = model(
        inputs,
        cu_seqlens=None if boundaries is None else torch.tensor(boundaries),
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(
        inputs.grad, reference_input.grad, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        model.weight.grad, reference_weight.grad, atol=1e-12, rtol=1e-12
    )


def test_kimi_convolution_empty_input_keeps_parameter_gradient():
    model = KimiCausalConvolution(3, 3)
    inputs = torch.empty(1, 0, 3, requires_grad=True)
    model(inputs).sum().backward()
    assert inputs.grad.shape == inputs.shape
    torch.testing.assert_close(model.weight.grad, torch.zeros_like(model.weight))
