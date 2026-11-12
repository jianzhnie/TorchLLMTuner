"""Qwen3.5 preserves independent rows through the flat Trainer interface."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


def test_flat_labels_preserve_per_row_next_token_loss():
    config = text_config(Qwen3_5MoeTextConfig)
    config.layer_types = ["full_attention"]
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(179)
    model = Qwen3_5MoeForCausalLM(config)
    reference = deepcopy(model)
    tokens = torch.tensor([[1, 3, 7, 2], [5, 4, 8, 9]])
    labels = tokens.clone()
    labels[1, 2] = -100
    expected = reference(tokens, labels=labels)
    actual = model(tokens.flatten(), input_batch_shape=(2, 4), labels=labels.flatten())
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    actual.loss.backward()
    expected.loss.backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(reference.named_parameters())[name].grad,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
@pytest.mark.parametrize("hidden_only", [False, True])
@pytest.mark.parametrize("metadata", [False, True])
def test_prepared_rows_match_native_batch_forward_and_gradients(
    layer_type, hidden_only, metadata
):
    config = text_config(Qwen3_5MoeTextConfig)
    config.layer_types = [layer_type]
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(81)
    model = Qwen3_5MoeForCausalLM(config).eval()
    reference = deepcopy(model)
    tokens = torch.tensor([[1, 3, 7, 2], [5, 4, 8, 9]])
    batch = {"input": tokens, "labels": tokens}
    reference_kwargs = {}
    if metadata:
        batch["positions"] = torch.arange(4).repeat(2, 1)
        batch["padding_mask"] = torch.tensor(
            [[False, False, False, True], [False, False, False, False]]
        )
        reference_kwargs = {
            "positions": batch["positions"],
            "padding_mask": batch["padding_mask"],
        }
    inputs, labels, extra = model.prepare_inputs(batch, parallel_dims=None)
    assert inputs.ndim == labels.ndim == 1
    assert extra["input_batch_shape"] == (2, 4)
    actual = model(inputs, skip_lm_head=hidden_only, **extra)
    expected = reference(tokens, skip_lm_head=hidden_only, **reference_kwargs)
    if not hidden_only:
        actual, expected = actual.logits, expected.logits
    expected = expected.reshape(-1, expected.shape[-1])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    nonzero = 0
    for name, parameter in model.named_parameters():
        baseline = dict(reference.named_parameters())[name]
        torch.testing.assert_close(parameter.grad, baseline.grad, atol=0, rtol=0)
        if parameter.grad is not None:
            nonzero += int(bool(parameter.grad.count_nonzero()))
    assert nonzero > 0


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
@pytest.mark.parametrize("hidden_only", [False, True])
def test_flat_padding_mask_follows_explicit_input_batch_shape(layer_type, hidden_only):
    config = text_config(Qwen3_5MoeTextConfig)
    config.layer_types = [layer_type]
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(config).eval()
    tokens = torch.tensor([[1, 3, 7, 2], [5, 4, 8, 9]])
    padding = torch.tensor([[False, False, False, True], [False, False, True, True]])
    expected = model(tokens, padding_mask=padding, skip_lm_head=hidden_only)
    actual = model(
        tokens.flatten(),
        input_batch_shape=(2, 4),
        padding_mask=padding.flatten(),
        skip_lm_head=hidden_only,
    )
    if not hidden_only:
        actual, expected = actual.logits, expected.logits
    torch.testing.assert_close(
        actual, expected.reshape(-1, expected.shape[-1]), rtol=0, atol=0
    )
    actual.sum().backward()
    assert model.tok_embeddings.weight.grad.isfinite().all()
    assert model.tok_embeddings.weight.grad.count_nonzero() > 0


@pytest.mark.parametrize("position_layout", ["flat_rows", "shared_sequence"])
@pytest.mark.parametrize("hidden_only", [False, True])
def test_flat_positions_preserve_explicit_batch_rows(position_layout, hidden_only):
    config = text_config(Qwen3_5MoeTextConfig)
    config.layer_types = ["full_attention"]
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(163)
    actual_model = Qwen3_5MoeForCausalLM(config)
    reference = deepcopy(actual_model)
    tokens = torch.tensor([[1, 3, 7, 2], [5, 4, 8, 9]])
    positions = torch.tensor([[0, 1, 3, 6], [2, 4, 7, 9]])
    if position_layout == "shared_sequence":
        positions = positions[:1].expand(2, -1)
        supplied_positions = positions[0]
    else:
        supplied_positions = positions.flatten()
    expected = reference(tokens, positions=positions, skip_lm_head=hidden_only)
    observed = actual_model(
        tokens.flatten(),
        input_batch_shape=(2, 4),
        positions=supplied_positions,
        skip_lm_head=hidden_only,
    )
    if not hidden_only:
        expected, observed = expected.logits, observed.logits
    expected = expected.reshape(-1, expected.shape[-1])
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    probe = torch.randn_like(expected)
    expected.backward(probe)
    observed.backward(probe)
    for name, parameter in actual_model.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(reference.named_parameters())[name].grad,
            rtol=0,
            atol=0,
        )
