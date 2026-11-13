"""Explicit input preparation constructs masks from semantic positions only."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmtuner.datasets.contracts import Batch
from llmtuner.models.common import input_preparation
from llmtuner.models.common.input_preparation import prepare_text_inputs


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
@pytest.mark.parametrize("packed", [False, True])
def test_unsharded_preparation_uses_declared_mask_semantics(backend, packed):
    positions = torch.tensor([0, 1, 0, 1])
    inputs = torch.tensor([1, 2, 3, 4])
    labels = inputs + 1
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(
        _attn_implementation=backend,
        attn_mask_type="block_causal" if packed else "causal",
    )

    def forbidden(**kwargs):
        raise AssertionError("generic preparation must not inspect backend adapters")

    model.get_attention_masks = forbidden
    actual_inputs, actual_labels, extra = prepare_text_inputs(
        model,
        {"input": inputs, "labels": labels, "positions": positions},
        parallel_dims=None,
    )
    assert torch.equal(actual_inputs, inputs)
    assert torch.equal(actual_labels, labels)
    assert torch.equal(extra["positions"], positions)
    if packed:
        assert extra["attention_masks"].shape == (1, 1, 4, 4)
        mask = extra["attention_masks"][0, 0]
        # Each reset begins a separate document; query 2 cannot read token 1.
        allowed = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, False, True, False],
                [False, False, True, True],
            ]
        )
        assert torch.equal(torch.isfinite(mask), allowed)
        assert torch.equal(mask[allowed], torch.zeros_like(mask[allowed]))
        assert torch.isneginf(mask[~allowed]).all()
    else:
        assert "attention_masks" not in extra


def test_packed_mask_excludes_padding_keys():
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(attn_mask_type="block_causal")
    _, _, extra = prepare_text_inputs(
        model,
        {
            "input": torch.tensor([1, 2, 3, 0]),
            "labels": torch.tensor([2, 3, 4, -100]),
            "positions": torch.tensor([0, 1, 0, 1]),
            "padding_mask": torch.tensor([False, False, False, True]),
        },
        parallel_dims=None,
    )
    assert torch.isneginf(extra["attention_masks"][0, 0, :, 3]).all()
    assert torch.isfinite(extra["attention_masks"][0, 0, 3, 2])


def test_packed_input_requires_positions_before_parallel_lookup():
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(attn_mask_type="block_causal")
    tokens = torch.tensor([1, 2, 3, 4])

    def forbidden(axis):
        pytest.fail("Missing document positions reached parallel lookup")

    with pytest.raises(ValueError, match="Packed text input requires positions"):
        prepare_text_inputs(
            model,
            {"input": tokens, "labels": tokens},
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.bool])
def test_nonintegral_positions_reject_before_mesh_lookup(dtype):
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(attn_mask_type="block_causal")

    def forbidden(axis):
        raise AssertionError("invalid position dtype reached mesh lookup")

    with pytest.raises(TypeError, match="positions must contain"):
        prepare_text_inputs(
            model,
            {
                "input": torch.tensor([1, 2, 3, 4]),
                "labels": torch.tensor([2, 3, 4, 5]),
                "positions": torch.zeros(4, dtype=dtype),
            },
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
        )


@pytest.mark.parametrize("field", ["labels", "positions", "padding_mask"])
def test_misaligned_devices_reject_before_mesh_lookup(field):
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(attn_mask_type="causal")
    batch = {
        "input": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([2, 3, 4, 5]),
        "positions": torch.arange(4),
        "padding_mask": torch.zeros(4, dtype=torch.bool),
    }
    batch[field] = batch[field].to("meta")

    def forbidden(axis):
        raise AssertionError("mixed-device batch reached mesh lookup")

    with pytest.raises(ValueError, match=f"{field} is on meta"):
        prepare_text_inputs(
            model,
            batch,
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
        )


def test_preserved_rows_rebind_packed_mask_and_metadata(monkeypatch):
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(attn_mask_type="block_causal")
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    positions = torch.tensor([[0, 1, 0, 1], [4, 5, 6, 7]])
    padding = torch.tensor([[False, False, False, False], [False, True, False, False]])
    calls = []
    build_mask = input_preparation.build_dense_attention_mask

    def observed(positions, **kwargs):
        calls.append(positions.numel())
        assert positions.numel() == 4
        return build_mask(positions, **kwargs)

    monkeypatch.setattr(input_preparation, "build_dense_attention_mask", observed)
    inputs, labels, extra = prepare_text_inputs(
        model,
        {
            "input": tokens,
            "labels": tokens,
            "positions": positions,
            "padding_mask": padding,
        },
        parallel_dims=None,
        preserve_batch_rows=True,
    )
    assert inputs.shape == labels.shape == (8,)
    assert extra["input_batch_shape"] == (2, 4)
    assert calls == [4, 4]
    assert torch.equal(extra["positions"], positions)
    assert torch.equal(extra["padding_mask"], padding)
    mask = extra["attention_masks"]
    assert mask.shape == (2, 1, 4, 4)
    expected = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, False, True, False],
                [False, False, True, True],
            ],
            [
                [True, False, False, False],
                [True, False, False, False],
                [True, False, True, False],
                [True, False, True, True],
            ],
        ]
    )
    assert torch.equal(mask[:, 0].isfinite(), expected)


@pytest.mark.parametrize("shape", [(), (2, 2, 2)])
def test_invalid_input_rank_rejects_before_mesh_or_model_access(shape):
    def forbidden(axis):
        raise AssertionError("invalid text layout reached mesh lookup")

    tokens = torch.ones(shape, dtype=torch.long)
    with pytest.raises(ValueError, match="Text input must have shape"):
        prepare_text_inputs(
            nn.Identity(),
            {"input": tokens, "labels": torch.ones_like(tokens)},
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.bool])
def test_invalid_token_dtype_rejects_before_mesh_or_model_access(dtype):
    def forbidden(axis):
        raise AssertionError("invalid tokens reached mesh lookup")

    tokens = torch.ones(4, dtype=dtype)
    with pytest.raises(TypeError, match="token indices"):
        prepare_text_inputs(
            nn.Identity(),
            {"input": tokens, "labels": torch.ones(4, dtype=torch.long)},
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
        )


@pytest.mark.parametrize("packed", [False, True])
def test_tp_preserves_attention_metadata_while_sharding_tokens(monkeypatch, packed):
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(
        attn_mask_type="block_causal" if packed else "causal"
    )
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0])
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    padding = torch.tensor([False] * 7 + [True])
    tp_mesh = object()
    topology = SimpleNamespace(
        get_optional_mesh=lambda axis: tp_mesh if axis == "tp" else None
    )

    def shard(inputs, labels, mesh):
        assert mesh is tp_mesh
        return inputs[4:], labels[4:]

    monkeypatch.setattr(input_preparation, "shard_batch_for_tp", shard)
    inputs, labels, extra = prepare_text_inputs(
        model,
        {
            "input": tokens,
            "labels": tokens + 1,
            "positions": positions,
            "padding_mask": padding,
        },
        parallel_dims=topology,
    )
    assert torch.equal(inputs, tokens[4:])
    assert torch.equal(labels, tokens[4:] + 1)
    assert torch.equal(extra["positions"], positions)
    assert torch.equal(extra["padding_mask"], padding)
    if packed:
        mask = extra["attention_masks"]
        assert mask.shape == (1, 1, 8, 8)
        assert torch.isneginf(mask[..., 7]).all()
        assert torch.isneginf(mask[..., 4, :3]).all()
        assert torch.isfinite(mask[..., 4, 3:5]).all()


@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.bool])
def test_nonintegral_labels_reject_before_mesh_lookup(synthetic, dtype):
    tokens = torch.arange(8).reshape(2, 4)
    labels = torch.ones_like(tokens, dtype=dtype)
    batch = (
        Batch(input_ids=tokens, labels=labels)
        if synthetic
        else {"input": tokens, "labels": labels}
    )

    def forbidden(axis):
        pytest.fail("Invalid label dtype reached parallel mesh lookup")

    with pytest.raises(TypeError, match="labels must contain"):
        prepare_text_inputs(
            nn.Identity(),
            batch,
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
        )


@pytest.mark.parametrize("shape", [(8,), (4, 2), (), (2, 3)])
def test_synthetic_labels_cannot_change_document_boundaries(shape):
    tokens = torch.arange(8).reshape(2, 4)
    labels = torch.ones(shape, dtype=torch.long)
    with pytest.raises(ValueError, match="same shape as input_ids"):
        prepare_text_inputs(
            nn.Identity(), Batch(input_ids=tokens, labels=labels), parallel_dims=None
        )


def test_synthetic_shift_preserves_row_boundaries_and_ignores_each_terminal():
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(attn_mask_type="causal")
    tokens = torch.tensor([[1, 2, 3, 4], [9, 8, 7, 6]])
    original = tokens.clone()
    inputs, labels, _ = prepare_text_inputs(
        model,
        Batch(input_ids=tokens, labels=tokens),
        parallel_dims=None,
        preserve_batch_rows=True,
    )
    assert torch.equal(inputs, tokens.flatten())
    assert labels.tolist() == [2, 3, 4, -100, 8, 7, 6, -100]
    assert torch.equal(tokens, original)
