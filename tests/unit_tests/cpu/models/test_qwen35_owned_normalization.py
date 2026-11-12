"""Qwen3.5 zero-centered normalization preserves native rounding and VJP."""

import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNorm as ReferenceNorm,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNormGated as ReferenceGatedNorm,
)

from llmtuner.models.qwen.qwen3_5_moe.gated_delta import Qwen3_5MoeRMSNormGated
from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeRMSNorm,
    initialize_model_weights,
)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("empty", [False, True])
def test_owned_norm_matches_reference_without_forward_delegation(
    monkeypatch, input_dtype, weight_dtype, empty
):
    reference = ReferenceNorm(8).to(weight_dtype)
    actual = Qwen3_5MoeRMSNorm(8).to(weight_dtype)
    with torch.no_grad():
        reference.weight.copy_(torch.linspace(-0.3, 0.4, 8))
        actual.weight.copy_(reference.weight)
    source = torch.randn(3, 0 if empty else 2, 8, dtype=input_dtype).transpose(0, 1)
    source.requires_grad_()
    other = source.detach().clone().requires_grad_()
    expected = reference(other)

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit norm delegated forward to reference")

    monkeypatch.setattr(ReferenceNorm, "forward", forbidden)
    observed = actual(source)
    assert observed.dtype == input_dtype
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    observed.backward(upstream)
    torch.testing.assert_close(source.grad, other.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.weight.grad, reference.weight.grad, rtol=0, atol=0
    )
    if not empty:
        assert source.grad.isfinite().all() and source.grad.count_nonzero() > 0


@pytest.mark.parametrize("loaded", [False, True])
def test_owned_norm_initialization_preserves_loaded_weights_and_rng(loaded):
    norm = Qwen3_5MoeRMSNorm(8)
    with torch.no_grad():
        norm.weight.fill_(0.25)
    norm.weight._is_hf_initialized = loaded
    weight = norm.weight
    rng = torch.get_rng_state().clone()
    assert initialize_model_weights(norm, 0.02)
    assert norm.weight is weight
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(
        norm.weight, torch.full_like(weight, 0.25 if loaded else 0), rtol=0, atol=0
    )


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("empty", [False, True])
def test_owned_gated_norm_matches_reference_cast_order_and_all_gradients(
    monkeypatch, input_dtype, weight_dtype, empty
):
    reference = ReferenceGatedNorm(8).to(weight_dtype)
    actual = Qwen3_5MoeRMSNormGated(8).to(weight_dtype)
    with torch.no_grad():
        reference.weight.copy_(torch.linspace(0.7, 1.4, 8))
        actual.weight.copy_(reference.weight)
    source = torch.randn(3, 0 if empty else 2, 8, dtype=input_dtype).transpose(0, 1)
    source.requires_grad_()
    gate = torch.randn_like(source).requires_grad_()
    other = source.detach().clone().requires_grad_()
    other_gate = gate.detach().clone().requires_grad_()
    expected = reference(other, other_gate)

    def forbidden(*args, **kwargs):
        pytest.fail("Owned gated norm delegated its forward")

    monkeypatch.setattr(ReferenceGatedNorm, "forward", forbidden)
    observed = actual(source, gate)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    observed.backward(upstream)
    for left, right in (
        (source.grad, other.grad),
        (gate.grad, other_gate.grad),
        (actual.weight.grad, reference.weight.grad),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        if not empty:
            assert left.isfinite().all() and left.count_nonzero() > 0


@pytest.mark.parametrize("loaded", [False, True])
def test_owned_gated_norm_initializes_to_one_and_preserves_loaded_weights(loaded):
    norm = Qwen3_5MoeRMSNormGated(8)
    with torch.no_grad():
        norm.weight.fill_(0.25)
    norm.weight._is_hf_initialized = loaded
    weight = norm.weight
    rng = torch.get_rng_state().clone()
    assert initialize_model_weights(norm, 0.02)
    assert norm.weight is weight
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(
        norm.weight, torch.full_like(weight, 0.25 if loaded else 1), rtol=0, atol=0
    )
