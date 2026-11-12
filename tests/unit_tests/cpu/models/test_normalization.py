"""Explicit native RMSNorm preserves reference math without a forward patch."""

from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.backends.registry import lookup_backend
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from llmtuner.models.qwen.qwen3.normalization import (
    build_rms_norm,
    validate_norm_backend,
)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(4,), (0, 4), (3, 4), (2, 3, 4)])
def test_native_norm_matches_reference_forward_and_gradients(
    input_dtype, weight_dtype, shape
):
    torch.manual_seed(42)
    reference = Qwen3RMSNorm(4, eps=1e-6).to(weight_dtype)
    actual_module = build_rms_norm(4, 1e-6, backend="native").to(weight_dtype)
    actual_module.load_state_dict(reference.state_dict())
    # Transposing twice through a contiguous copy retains noncontiguous strides.
    inputs = torch.randn(shape, dtype=input_dtype)
    if len(shape) > 1:
        inputs = inputs.transpose(0, -1).contiguous().transpose(0, -1)
    inputs.requires_grad_()
    expected_inputs = inputs.detach().clone().requires_grad_()
    actual = actual_module(inputs)
    expected = reference(expected_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_module.weight.grad, reference.weight.grad, rtol=0, atol=0
    )


def test_native_norm_fullgraph_compile_preserves_autograd_and_identity():
    module = build_rms_norm(4, 1e-6, backend="native")
    weight = module.weight
    keys = tuple(module.state_dict())
    inputs = torch.randn(2, 4, 3).transpose(1, 2).requires_grad_()
    reference_inputs = inputs.detach().clone().requires_grad_()
    compiled_graphs = []

    def backend(graph_module, example_inputs):
        compiled_graphs.append(graph_module)
        return lookup_backend("aot_eager")(graph_module, example_inputs)

    compiled = torch.compile(module, backend=backend, fullgraph=True)
    actual = compiled(inputs)
    assert compiled_graphs, "compile must execute the graph backend"
    expected = module(reference_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(actual)
    observed_grads = torch.autograd.grad(actual, (inputs, weight), probe)
    expected_grads = torch.autograd.grad(expected, (reference_inputs, weight), probe)
    for observed, target in zip(observed_grads, expected_grads, strict=True):
        assert observed.count_nonzero() > 0
        torch.testing.assert_close(observed, target, rtol=0, atol=0)
    assert module.weight is weight and tuple(module.state_dict()) == keys


def test_removed_aten_policy_is_explicitly_rejected():
    with pytest.raises(ValueError, match="norm_backend"):
        validate_norm_backend(SimpleNamespace(norm_backend="aten"))
