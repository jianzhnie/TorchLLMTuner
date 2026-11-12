"""Kimi preserves independent rows through the flat Trainer interface."""

from copy import deepcopy

import pytest
import torch

from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.model import KimiLinearForCausalLM


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
@pytest.mark.parametrize("hidden_only", [False, True])
@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("flatten_metadata", [False, True])
def test_prepared_rows_match_native_batch_forward_and_gradients(
    layer_type, hidden_only, metadata, flatten_metadata
):
    config = KimiLinearConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        kv_lora_rank=4,
        qk_nope_head_dim=3,
        qk_rope_head_dim=1,
        v_head_dim=3,
        mla_use_nope=True,
        linear_attn_config={
            "kda_layers": [1] if layer_type == "linear_attention" else [],
            "full_attn_layers": [1] if layer_type == "full_attention" else [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(81)
    model = KimiLinearForCausalLM(config).eval()
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
    actual = model(inputs, labels=labels, skip_lm_head=hidden_only, **extra)
    expected = reference(
        tokens, labels=tokens, skip_lm_head=hidden_only, **reference_kwargs
    )
    if not hidden_only:
        torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
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
