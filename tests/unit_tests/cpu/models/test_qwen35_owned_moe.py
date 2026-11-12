"""Qwen routed/shared expert math is owned by each explicit model."""

import pytest
import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as reference
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts as ReferenceQwen3MoeExperts,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock as ReferenceQwen3MoeSparseMoeBlock,
)

from llmtuner.models.qwen.qwen3.model import Qwen3MoeSparseMoeBlock
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeSparseMoeBlock
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 16), (1, 1, 16)])
def test_owned_moe_matches_reference_output_gradients_and_update(
    dtype, shape, monkeypatch
):
    config = Qwen3_5MoeTextConfig(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    config._experts_implementation = "eager"
    torch.manual_seed(53)
    baseline = reference.Qwen3_5MoeSparseMoeBlock(config).to(dtype)
    # A standalone block has no enclosing post_init; initialize packed expert
    # storage explicitly so zero/uninitialized experts cannot vacuously pass.
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    actual = Qwen3_5MoeSparseMoeBlock(config).to(dtype)
    actual.load_state_dict(baseline.state_dict())
    identities = {name: id(value) for name, value in actual.named_parameters()}
    hidden = torch.randn(shape, dtype=dtype, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    expected = baseline(hidden)

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit MoE must own its forward")

    monkeypatch.setattr(reference.Qwen3_5MoeSparseMoeBlock, "forward", forbidden)
    observed = actual(other)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    upstream = torch.randn_like(observed)
    observed.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(other.grad, hidden.grad, rtol=0, atol=0)
    for name, value in actual.named_parameters():
        target = dict(baseline.named_parameters())[name]
        assert id(value) == identities[name]
        torch.testing.assert_close(value.grad, target.grad, rtol=0, atol=0)
        with torch.no_grad():
            value.add_(value.grad, alpha=-0.01)
            target.add_(target.grad, alpha=-0.01)
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    for name in ("gate.weight", "shared_expert_gate.weight"):
        gradient = dict(actual.named_parameters())[name].grad
        assert gradient.isfinite().all() and gradient.count_nonzero() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
def test_qwen3_owned_forward_matches_reference(dtype, normalize, monkeypatch):
    config = model_config(norm_topk_prob=normalize)
    config._experts_implementation = "eager"
    torch.manual_seed(67)
    baseline = ReferenceQwen3MoeSparseMoeBlock(config).to(dtype)
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.normal_(std=0.1)

    def forbidden_experts(*args, **kwargs):
        raise AssertionError("Qwen3 must own expert construction and execution")

    monkeypatch.setattr(ReferenceQwen3MoeExperts, "__init__", forbidden_experts)
    actual = Qwen3MoeSparseMoeBlock(config).to(dtype)
    actual.load_state_dict(baseline.state_dict())
    hidden = torch.randn(2, 3, 16, dtype=dtype, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    expected = baseline(hidden)
    monkeypatch.setattr(ReferenceQwen3MoeExperts, "forward", forbidden_experts)

    def forbidden(*args, **kwargs):
        raise AssertionError("Qwen3 must own MoE execution")

    monkeypatch.setattr(ReferenceQwen3MoeSparseMoeBlock, "forward", forbidden)
    output = actual(other)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    probe = torch.randn_like(output)
    output.backward(probe)
    expected.backward(probe)
    torch.testing.assert_close(other.grad, hidden.grad, rtol=0, atol=0)
    for name, parameter in actual.named_parameters():
        reference_parameter = dict(baseline.named_parameters())[name]
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, rtol=0, atol=0
        )
        with torch.no_grad():
            parameter.add_(parameter.grad, alpha=-0.01)
            reference_parameter.add_(reference_parameter.grad, alpha=-0.01)
        torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)
    assert actual.gate.weight.grad.count_nonzero() > 0
