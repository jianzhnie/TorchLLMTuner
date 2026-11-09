"""Gated Kimi normalization must not round before the output gate."""

from copy import deepcopy

import pytest
import torch

from llmtuner.models.kimi.normalization import KimiGatedRMSNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_kimi_gated_norm_keeps_fp32_until_final_cast(dtype, weight_dtype):
    torch.manual_seed(42)
    model = KimiGatedRMSNorm(16).to(dtype=weight_dtype)
    with torch.no_grad():
        model.weight.copy_(torch.randn_like(model.weight))
    reference = deepcopy(model)
    values = torch.randn(2, 5, 3, 16, dtype=dtype, requires_grad=True)
    gate = torch.randn_like(values, requires_grad=True)
    other = values.detach().clone().requires_grad_()
    other_gate = gate.detach().clone().requires_grad_()
    full = other.float()
    normalized = full * (full.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    expected = (
        normalized * reference.weight.float() * other_gate.float().sigmoid()
    ).to(dtype)
    actual = model(values, gate)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for gradient, ref in (
        (values.grad, other.grad),
        (gate.grad, other_gate.grad),
        (model.weight.grad, reference.weight.grad),
    ):
        torch.testing.assert_close(gradient, ref, atol=0, rtol=0)
