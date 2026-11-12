"""Explicit Qwen3 norm selection preserves state and declares kernel tolerance."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.models.qwen.qwen3.normalization import Qwen3KernelRMSNorm, build_rms_norm
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(4,), (0, 4), (3, 4), (2, 3, 4)])
def test_torch_kernel_norm_matches_reference(input_dtype, weight_dtype, shape):
    torch.manual_seed(39)
    baseline = Qwen3MoeRMSNorm(4, eps=1e-6).to(weight_dtype)
    with torch.no_grad():
        baseline.weight.normal_(mean=0.7, std=0.3)
    actual = build_rms_norm(4, 1e-6, backend="torch").to(weight_dtype)
    actual.load_state_dict(baseline.state_dict())
    inputs = torch.randn(shape, dtype=input_dtype)
    if inputs.ndim > 1:
        inputs = inputs.transpose(0, -1).contiguous().transpose(0, -1)
    inputs.requires_grad_()
    reference_inputs = inputs.detach().clone().requires_grad_()
    expected = baseline(reference_inputs)
    result = actual(inputs)
    # Fused torch RMSNorm can use a different backward reduction order. State
    # migration is exact; kernel comparison uses these predeclared bounds.
    tolerance = 0.02 if torch.bfloat16 in (input_dtype, weight_dtype) else 0.002
    if input_dtype == weight_dtype == torch.float32:
        tolerance = 1e-5
    assert result.dtype == expected.dtype
    torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
    upstream = torch.randn_like(expected)
    result.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(
        inputs.grad, reference_inputs.grad, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        actual.weight.grad, baseline.weight.grad, rtol=tolerance, atol=tolerance
    )
    if inputs.numel():
        assert result.isfinite().all() and result.count_nonzero() > 0
        assert actual.weight.grad.count_nonzero() > 0


def test_model_torch_norms_preserve_initialization_and_execute():
    config = model_config()
    config._experts_implementation = "eager"
    torch.manual_seed(43)
    reference = Qwen3MoeForCausalLM(deepcopy(config))
    rng = torch.get_rng_state().clone()
    config.norm_backend = "torch"
    torch.manual_seed(43)
    actual = Qwen3MoeForCausalLM(config)
    assert torch.equal(torch.get_rng_state(), rng)
    assert reference.state_dict().keys() == actual.state_dict().keys()
    assert actual.enable_weight_tying == reference.enable_weight_tying
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
    native_norms = [
        module for module in reference.modules() if isinstance(module, Qwen3MoeRMSNorm)
    ]
    assert all(type(module) is Qwen3MoeRMSNorm for module in native_norms)
    norms = [
        module for module in actual.modules() if isinstance(module, Qwen3KernelRMSNorm)
    ]
    assert len(norms) == 4 * config.num_hidden_layers + 1
    calls = []
    handles = [
        module.register_forward_hook(lambda module, args, output: calls.append(module))
        for module in norms
    ]
    tokens = torch.tensor([[1, 3, 7, 2], [5, 4, 2, 9]])
    expected = reference(tokens).logits
    try:
        result = actual(tokens).logits
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == len(norms)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    result.square().sum().backward()
    for name, parameter in actual.named_parameters():
        baseline = dict(reference.named_parameters())[name]
        assert (parameter.grad is None) == (baseline.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, baseline.grad, rtol=1e-5, atol=1e-6
            )


@pytest.mark.parametrize("backend", ["aten", "joint", "invalid", False])
def test_unimplemented_norm_rejected_before_model_allocation(backend, monkeypatch):
    config = model_config()
    config.norm_backend = backend

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("norm policy must be checked before model allocation")

    monkeypatch.setattr(nn.Embedding, "__init__", unexpected_allocation)
    with pytest.raises(ValueError, match="norm_backend"):
        Qwen3MoeForCausalLM(config)


def test_npu_norm_on_cpu_fails_without_reference_fallback():
    config = model_config()
    config.norm_backend = "npu"
    model = Qwen3MoeForCausalLM(config)
    with pytest.raises(RuntimeError, match="requires NPU input tensors"):
        model(torch.tensor([[1, 3, 7]]))
