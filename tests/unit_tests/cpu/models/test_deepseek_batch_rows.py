"""DeepSeek V4 preserves independent rows through the flat Trainer interface."""

from copy import deepcopy

import pytest
import torch

from llmtuner.models.deepseek.deepseek_v4.modeling import DeepseekV4ForCausalLM
from tests.unit_tests.cpu.models.test_owned_family_construction import small_config


@pytest.mark.parametrize("hidden_only", [False, True])
@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("flatten_metadata", [False, True])
def test_prepared_rows_match_native_batch_forward_and_gradients(
    hidden_only, metadata, flatten_metadata
):
    config = small_config()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(81)
    model = DeepseekV4ForCausalLM(config).eval()
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
    if metadata and flatten_metadata:
        extra["positions"] = batch["positions"].flatten()
        extra["padding_mask"] = batch["padding_mask"].flatten()
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


def test_flat_labels_preserve_per_row_next_token_loss():
    config = small_config()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(181)
    model = DeepseekV4ForCausalLM(config).eval()
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
