"""Qwen3.5 owns full-attention packing, gating and module construction."""

import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention as ReferenceAttention,
)

from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeForCausalLM,
)
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_owned_attention_matches_reference_init_forward_gradients_and_update(
    monkeypatch, dtype, bias, backend
):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config.attention_bias = bias
    config._attn_implementation = backend
    config.attention_dropout = 0.0
    torch.manual_seed(101)
    expected = ReferenceAttention(config, 0).to(dtype)
    expected_rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("Model-owned attention delegated construction or forward")

    monkeypatch.setattr(ReferenceAttention, "__init__", forbidden)
    torch.manual_seed(101)
    actual = Qwen3_5MoeAttention(config, 0).to(dtype)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
    hidden = torch.randn(3, 2, config.hidden_size, dtype=dtype).transpose(0, 1)
    hidden.requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    rotary = (
        torch.ones(2, 3, config.head_dim, dtype=dtype),
        torch.zeros(2, 3, config.head_dim, dtype=dtype),
    )
    result = expected(reference_hidden, rotary, None)[0]
    monkeypatch.setattr(ReferenceAttention, "forward", forbidden)
    observed = actual(hidden, rotary, None)[0]
    torch.testing.assert_close(observed, result, rtol=0, atol=0)
    upstream = torch.randn_like(result)
    result.backward(upstream)
    observed.backward(upstream)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    assert hidden.grad.isfinite().all() and hidden.grad.count_nonzero() > 0
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(expected.named_parameters())[name].grad,
            rtol=0,
            atol=0,
        )
    for model in (actual, expected):
        torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-4, foreach=False).step()
    for name, parameter in actual.named_parameters():
        assert parameter.isfinite().all()
        torch.testing.assert_close(
            parameter, dict(expected.named_parameters())[name], rtol=0, atol=0
        )
