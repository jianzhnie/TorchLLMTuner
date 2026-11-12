"""Explicit Qwen3.5 owns LM output and loss assembly."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as ReferenceModel,
)

from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("router", [False, True])
def test_owned_lm_outputs_losses_and_gradients_match_reference(
    monkeypatch, dtype, router
):
    config = text_config(ReferenceModel.config_class)
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    config.router_aux_loss_coef = 0.01
    torch.manual_seed(113)
    baseline = ReferenceModel(deepcopy(config)).to(dtype)
    actual = Qwen3_5MoeForCausalLM(deepcopy(config)).to(dtype)
    actual.load_state_dict(baseline.state_dict(), strict=True)
    tokens = torch.tensor([[1, 5, 2, 7], [3, 8, 4, 6]])
    labels = tokens.clone()
    labels[0, -1] = -100
    expected = baseline(
        tokens, labels=labels, output_router_logits=router, use_cache=False
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit LM delegated its forward to the reference")

    monkeypatch.setattr(ReferenceModel, "forward", forbidden)
    observed = actual(tokens, labels=labels, output_router_logits=router)
    for name in ("logits", "loss", "aux_loss"):
        left, right = getattr(observed, name), getattr(expected, name)
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    expected.loss.backward()
    observed.loss.backward()
    nonzero = 0
    for name, parameter in actual.named_parameters():
        other = dict(baseline.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
            nonzero += bool(parameter.grad.count_nonzero())
    assert nonzero > 0


@pytest.mark.parametrize("keep", [0, 2])
def test_owned_flat_logits_support_tuple_output_and_selection(keep):
    config = text_config(ReferenceModel.config_class)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(config)
    tokens = torch.tensor([1, 5, 2, 7])
    expected = model(tokens.unsqueeze(0), logits_to_keep=keep).logits.squeeze(0)
    observed = model(tokens, logits_to_keep=keep, return_dict=False)
    assert isinstance(observed, tuple)
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)


def test_owned_lm_keeps_decoder_arguments_out_of_custom_loss():
    config = text_config(ReferenceModel.config_class)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(config)
    calls = []

    def loss(logits, labels, vocab_size):
        calls.append(vocab_size)
        return logits.float().sum() * 0 + labels.float().sum()

    model.loss_function = loss
    tokens = torch.tensor([[1, 5, 2, 7]])
    result = model(
        tokens,
        labels=tokens,
        attention_mask=torch.ones_like(tokens),
        position_ids=torch.arange(4)[None],
        use_cache=True,
    )
    assert calls == [config.vocab_size]
    assert result.loss.item() == tokens.sum().item()
    assert result.past_key_values is None


@pytest.mark.parametrize("configured_dict", [False, True])
@pytest.mark.parametrize("requested_dict", [None, False, True])
@pytest.mark.parametrize("router", [False, True])
def test_output_format_policy_preserves_captured_outputs_and_gradients(
    configured_dict, requested_dict, router
):
    config = text_config(ReferenceModel.config_class)
    config.return_dict = configured_dict
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(config)
    tokens = torch.tensor([[1, 5, 2, 7], [3, 8, 4, 6]])
    expected = model(
        tokens,
        labels=tokens,
        output_hidden_states=True,
        output_router_logits=router,
        return_dict=True,
    )
    observed = model(
        tokens,
        labels=tokens,
        output_hidden_states=True,
        output_router_logits=router,
        return_dict=requested_dict,
    )
    use_dict = configured_dict if requested_dict is None else requested_dict
    assert isinstance(observed, tuple) == (not use_dict)
    values = observed.to_tuple() if use_dict else observed
    expected_values = expected.to_tuple()
    assert len(values) == len(expected_values)

    def compare(left, right):
        if isinstance(right, tuple):
            assert len(left) == len(right)
            for actual, reference in zip(left, right, strict=True):
                compare(actual, reference)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    compare(values, expected_values)
    values[0].backward()
    assert model.lm_head.weight.grad.isfinite().all()
    assert model.lm_head.weight.grad.count_nonzero() > 0


def test_hidden_only_execution_ignores_outer_tuple_policy():
    config = text_config(ReferenceModel.config_class)
    config.return_dict = False
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(config)
    tokens = torch.tensor([1, 5, 2, 7])
    hidden = model(tokens, skip_lm_head=True)
    expected = model.model(
        tokens[None], use_cache=False, return_dict=True
    ).last_hidden_state
    torch.testing.assert_close(hidden, expected.squeeze(0), rtol=0, atol=0)


@pytest.mark.parametrize("padding", [False, True])
def test_router_auxiliary_loss_uses_token_validity_with_pairwise_attention(padding):
    config = text_config(ReferenceModel.config_class)
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    config.router_aux_loss_coef = 0.01
    model = Qwen3_5MoeForCausalLM(config)
    tokens = torch.tensor([[1, 5, 2, 7], [3, 8, 4, 6]])
    keep = torch.ones_like(tokens, dtype=torch.bool)
    if padding:
        keep[0, -1] = False
    baseline = model(tokens, attention_mask=keep, output_router_logits=True)
    causal = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
    allowed = causal & keep[:, None, None, :]
    pairwise = torch.zeros(2, 1, 4, 4).masked_fill(~allowed, float("-inf"))
    observed = model(
        tokens, attention_mask=pairwise, padding_mask=~keep, output_router_logits=True
    )
    torch.testing.assert_close(observed.logits, baseline.logits, rtol=0, atol=0)
    torch.testing.assert_close(observed.aux_loss, baseline.aux_loss, rtol=0, atol=0)
    observed.aux_loss.backward()
    grad = model.layers[0].mlp.gate.weight.grad
    assert grad.isfinite().all() and grad.count_nonzero() > 0
