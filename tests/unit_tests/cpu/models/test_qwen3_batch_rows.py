"""Flat Trainer input must preserve Qwen3's independent sample boundaries."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.common.attention.masks import build_dense_attention_mask
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("padding", [False, True])
def test_flat_rows_match_batched_forward_and_gradients(backend, padding):
    config = model_config()
    config._attn_implementation = backend
    config._experts_implementation = "eager"
    torch.manual_seed(89)
    model = Qwen3MoeForCausalLM(config).eval()
    tokens = torch.tensor([[1, 3, 7, 9], [2, 5, 8, 4]])
    batch = {"input": tokens, "labels": tokens + 1}
    pad = torch.tensor([[False, False, False, True], [False, False, True, True]])
    if padding:
        batch["padding_mask"] = pad
    inputs, labels, kwargs = model.prepare_inputs(batch, parallel_dims=None)
    assert kwargs["positions"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert torch.isneginf(kwargs["attention_masks"][..., 4:, :4]).all()
    assert torch.equal(labels, (tokens + 1).flatten())
    observed = model(inputs, **kwargs).logits
    expected = model(tokens, padding_mask=pad if padding else None).logits.flatten(0, 1)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    probe = torch.randn_like(observed)
    parameters = tuple(model.parameters())
    observed_grads = torch.autograd.grad(observed, parameters, probe)
    expected_grads = torch.autograd.grad(expected, parameters, probe)
    for actual, reference in zip(observed_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    assert observed_grads[0].count_nonzero() > 0


def test_multirow_parallel_rejects_before_sharding():
    model = Qwen3MoeForCausalLM(model_config())
    topology = SimpleNamespace(get_optional_mesh=lambda axis: object())
    with pytest.raises(UnsupportedCombinationError, match="row-aware"):
        model.prepare_inputs(
            {
                "input": torch.ones(2, 4, dtype=torch.long),
                "labels": torch.ones(2, 4, dtype=torch.long),
            },
            parallel_dims=topology,
        )


def test_packed_rows_preserve_document_and_row_boundaries():
    config = model_config()
    config.attn_mask_type = "block_causal"
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = Qwen3MoeForCausalLM(config).eval()
    tokens = torch.tensor([[1, 3, 7, 9], [2, 5, 8, 4]])
    positions = torch.tensor([[0, 1, 0, 1], [5, 6, 7, 8]])
    inputs, _, extra = model.prepare_inputs(
        {"input": tokens, "labels": tokens, "positions": positions},
        parallel_dims=None,
    )
    mask = extra["attention_masks"]
    assert torch.isneginf(mask[..., 2:4, :2]).all()
    assert torch.isneginf(mask[..., 4:, :4]).all()
    observed = model(inputs, **extra).logits
    expected = []
    for row, row_positions in zip(tokens, positions, strict=True):
        row_mask = build_dense_attention_mask(
            row_positions,
            dtype=model.tok_embeddings.weight.dtype,
            block_causal=True,
        )
        expected.append(
            model(
                row,
                positions=row_positions,
                attention_masks=row_mask,
            ).logits
        )
    torch.testing.assert_close(observed, torch.cat(expected), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("size", [0, 1, 5])
def test_invalid_padding_length_rejects_before_mesh_lookup(size):
    model = Qwen3MoeForCausalLM(model_config())

    def forbidden(axis):
        raise AssertionError("malformed input must fail before sharding lookup")

    topology = SimpleNamespace(get_optional_mesh=forbidden)
    with pytest.raises(ValueError, match="padding_mask has"):
        model.prepare_inputs(
            {
                "input": torch.tensor([1, 2, 3, 4]),
                "labels": torch.tensor([2, 3, 4, 5]),
                "padding_mask": torch.zeros(size, dtype=torch.bool),
            },
            parallel_dims=topology,
        )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_single_row_packed_mask_also_excludes_padding_keys(backend):
    config = model_config()
    config.attn_mask_type = "block_causal"
    config._attn_implementation = backend
    config._experts_implementation = "eager"
    model = Qwen3MoeForCausalLM(config).eval()
    tokens = torch.tensor([1, 3, 7, 9])
    # An interior padding key makes silent padding loss affect a later token.
    padding = torch.tensor([False, True, False, False])
    positions = torch.tensor([0, 1, 2, 3])
    inputs, _, kwargs = model.prepare_inputs(
        {
            "input": tokens,
            "labels": tokens,
            "positions": positions,
            "padding_mask": padding,
        },
        parallel_dims=None,
    )
    mask = kwargs["attention_masks"]
    assert torch.isneginf(mask[..., 1]).all()
    expected_mask = build_dense_attention_mask(
        positions,
        dtype=model.tok_embeddings.weight.dtype,
        block_causal=True,
    ).masked_fill(padding[None, :], float("-inf"))
    observed = model(inputs, **kwargs).logits
    expected = model(tokens, positions=positions, attention_masks=expected_mask).logits
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert observed.isfinite().all()


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("padding", [False, True])
def test_flat_rows_router_aux_loss_uses_token_validity_not_attention_edges(
    backend, padding
):
    config = model_config(output_router_logits=True)
    config._attn_implementation = backend
    config._experts_implementation = "eager"
    torch.manual_seed(97)
    model = Qwen3MoeForCausalLM(config).eval()
    tokens = torch.tensor([[1, 3, 7, 9], [2, 5, 8, 4]])
    batch = {"input": tokens, "labels": tokens + 1}
    pad = torch.tensor([[False, False, False, True], [False, False, True, True]])
    if padding:
        batch["padding_mask"] = pad
    inputs, _, kwargs = model.prepare_inputs(batch, parallel_dims=None)
    assert kwargs["attention_masks"].ndim == 4
    actual = model(inputs, **kwargs)
    expected = model(tokens, padding_mask=pad if padding else None)
    assert actual.aux_loss.isfinite() and actual.aux_loss > 0
    torch.testing.assert_close(actual.aux_loss, expected.aux_loss, rtol=1e-5, atol=1e-6)
    router = model.model.layers[0].mlp.gate.weight
    actual_grad = torch.autograd.grad(actual.aux_loss, router)[0]
    expected_grad = torch.autograd.grad(expected.aux_loss, router)[0]
    assert actual_grad.count_nonzero() > 0
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)
