"""Kimi MLA projection/layout and HF backend numerical contracts."""

from copy import deepcopy

import pytest
import torch

from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
from llmtuner.models.kimi.latent_attention import KimiLatentAttention


@pytest.mark.parametrize("query_rank", [None, 4])
@pytest.mark.parametrize("output_gate", [False, True])
def test_kimi_latent_attention_matches_explicit_projection_reference(
    query_rank, output_gate
):
    torch.manual_seed(42)
    config = KimiLinearConfig(
        hidden_size=12,
        num_attention_heads=2,
        num_key_value_heads=1,
        q_lora_rank=query_rank,
        kv_lora_rank=4,
        qk_nope_head_dim=3,
        qk_rope_head_dim=1,
        v_head_dim=3,
        mla_use_nope=True,
        mla_use_output_gate=output_gate,
    )
    config._attn_implementation = "eager"
    model = KimiLatentAttention(config, 0)
    reference = deepcopy(model)
    inputs = torch.randn(1, 5, 12, requires_grad=True)
    expected_inputs = inputs.detach().clone().requires_grad_()
    mask = torch.full((1, 1, 5, 5), float("-inf")).triu(1)
    # Independent equations: no production forward or reshape helper is reused.
    x = expected_inputs
    q = (
        (
            reference.q_b_proj(reference.q_a_layernorm(reference.q_a_proj(x)))
            if query_rank is not None
            else reference.q_proj(x)
        )
        .reshape(1, 5, 2, 4)
        .transpose(1, 2)
    )
    compressed = reference.kv_a_proj_with_mqa(x)
    kv = reference.kv_b_proj(reference.kv_a_layernorm(compressed[..., :4]))
    kv = kv.reshape(1, 5, 2, 6).transpose(1, 2)
    k = torch.cat(
        (kv[..., :3], compressed[..., 4:].unsqueeze(1).expand(1, 2, 5, 1)), -1
    )
    probabilities = ((q @ k.transpose(-1, -2)) * 0.5 + mask).softmax(-1)
    expected = (probabilities @ kv[..., 3:]).transpose(1, 2).reshape(1, 5, 6)
    if output_gate:
        expected = expected * reference.g_proj(x).sigmoid()
    expected = reference.o_proj(expected)
    actual = model(inputs, attention_mask=mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(inputs.grad, expected_inputs.grad, atol=1e-6, rtol=1e-6)
    for (name, p), (_, r) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        torch.testing.assert_close(p.grad, r.grad, atol=1e-6, rtol=1e-6, msg=name)
    model.zero_grad()
    model.config._attn_implementation = "sdpa"
    sdpa = model(inputs.detach(), attention_mask=mask)
    torch.testing.assert_close(sdpa, expected.detach(), atol=1e-6, rtol=1e-6)


def test_kimi_eager_boolean_mask_and_empty_rows_match_sdpa():
    from types import SimpleNamespace

    from torch.nn import functional as F

    from llmtuner.models.kimi.latent_attention import eager_attention_forward

    q, k, v = [torch.randn(1, 2, 3, 4, requires_grad=True) for _ in range(3)]
    mask = torch.tensor(
        [[[[False, False, False], [True, True, False], [True, True, True]]]]
    )
    actual, _ = eager_attention_forward(
        SimpleNamespace(training=True), q, k, v, mask, scaling=0.5
    )
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward()
    for tensor in (q, k, v):
        assert torch.isfinite(tensor.grad).all()
