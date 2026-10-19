"""Production ATen RMSNorm contracts against independent HF autograd."""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from llmtuner.models.hf.normalization import forward_aten_rms_norm, rms_norm


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(4,), (0, 4), (3, 4), (2, 3, 4)])
def test_aten_norm_matches_hf_forward_and_gradients(input_dtype, weight_dtype, shape):
    torch.manual_seed(42)
    reference = Qwen3RMSNorm(4, eps=1e-6).to(weight_dtype)
    weight = reference.weight.detach().clone().requires_grad_()
    # Transposing twice through a contiguous copy retains noncontiguous strides.
    inputs = torch.randn(shape, dtype=input_dtype)
    if len(shape) > 1:
        inputs = inputs.transpose(0, -1).contiguous().transpose(0, -1)
    inputs.requires_grad_()
    expected_inputs = inputs.detach().clone().requires_grad_()
    actual = rms_norm(inputs, weight, reference.variance_epsilon)
    expected = reference(expected_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, reference.weight.grad, rtol=0, atol=0)


def test_aten_norm_operator_schema_fake_and_autograd():
    inputs = torch.randn(2, 4, 3).transpose(1, 2).requires_grad_()
    weight = torch.randn(4, requires_grad=True)
    torch.library.opcheck(rms_norm, (inputs, weight, 1e-6))


def test_aten_norm_rejects_float64_input():
    module = Qwen3RMSNorm(4)
    with pytest.raises(TypeError, match="supports float16"):
        forward_aten_rms_norm(module, torch.randn(2, 4, dtype=torch.float64))
