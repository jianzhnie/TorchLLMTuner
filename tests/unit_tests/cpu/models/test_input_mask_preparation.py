"""Explicit input preparation constructs masks from semantic positions only."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.datasets.contracts import Batch
from llmtuner.models.common import input_preparation
from llmtuner.models.common.input_preparation import prepare_text_inputs


@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
def test_unsharded_preparation_uses_declared_mask_semantics(mask_dtype, packed):
    positions = torch.tensor([0, 1, 0, 1])
    inputs = torch.tensor([1, 2, 3, 4])
    labels = inputs + 1

    actual_inputs, actual_labels, extra = prepare_text_inputs(
        {"input": inputs, "labels": labels, "positions": positions},
        parallel_dims=None,
        packed=packed,
        mask_dtype=mask_dtype,
    )
    assert torch.equal(actual_inputs, inputs)
    assert torch.equal(actual_labels, labels)
    assert torch.equal(extra["positions"], positions)
    if packed:
        assert extra["attention_masks"].dtype == mask_dtype
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
    _, _, extra = prepare_text_inputs(
        {
            "input": torch.tensor([1, 2, 3, 0]),
            "labels": torch.tensor([2, 3, 4, -100]),
            "positions": torch.tensor([0, 1, 0, 1]),
            "padding_mask": torch.tensor([False, False, False, True]),
        },
        parallel_dims=None,
        packed=True,
        mask_dtype=torch.float32,
    )
    assert torch.isneginf(extra["attention_masks"][0, 0, :, 3]).all()
    assert torch.isfinite(extra["attention_masks"][0, 0, 3, 2])


def test_packed_input_requires_positions_before_parallel_lookup():
    tokens = torch.tensor([1, 2, 3, 4])

    def forbidden(axis):
        pytest.fail("Missing document positions reached parallel lookup")

    with pytest.raises(ValueError, match="Packed text input requires positions"):
        prepare_text_inputs(
            {"input": tokens, "labels": tokens},
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
            packed=True,
            mask_dtype=torch.float32,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.bool])
def test_nonintegral_positions_reject_before_mesh_lookup(dtype):

    def forbidden(axis):
        raise AssertionError("invalid position dtype reached mesh lookup")

    with pytest.raises(TypeError, match="positions must contain"):
        prepare_text_inputs(
            {
                "input": torch.tensor([1, 2, 3, 4]),
                "labels": torch.tensor([2, 3, 4, 5]),
                "positions": torch.zeros(4, dtype=dtype),
            },
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
            packed=True,
            mask_dtype=torch.float32,
        )


@pytest.mark.parametrize("field", ["labels", "positions", "padding_mask"])
def test_misaligned_devices_reject_before_mesh_lookup(field):
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
            batch,
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
            packed=False,
            mask_dtype=torch.float32,
        )


def test_preserved_rows_rebind_packed_mask_and_metadata(monkeypatch):
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
        {
            "input": tokens,
            "labels": tokens,
            "positions": positions,
            "padding_mask": padding,
        },
        parallel_dims=None,
        preserve_batch_rows=True,
        packed=True,
        mask_dtype=torch.float32,
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
def test_invalid_input_rank_rejects_before_mesh_lookup(shape):
    def forbidden(axis):
        raise AssertionError("invalid text layout reached mesh lookup")

    tokens = torch.ones(shape, dtype=torch.long)
    with pytest.raises(ValueError, match="Text input must have shape"):
        prepare_text_inputs(
            {"input": tokens, "labels": torch.ones_like(tokens)},
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
            packed=False,
            mask_dtype=torch.float32,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.bool])
def test_invalid_token_dtype_rejects_before_mesh_lookup(dtype):
    def forbidden(axis):
        raise AssertionError("invalid tokens reached mesh lookup")

    tokens = torch.ones(4, dtype=dtype)
    with pytest.raises(TypeError, match="token indices"):
        prepare_text_inputs(
            {"input": tokens, "labels": torch.ones(4, dtype=torch.long)},
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
            packed=False,
            mask_dtype=torch.float32,
        )


@pytest.mark.parametrize("packed", [False, True])
def test_tp_preserves_attention_metadata_while_sharding_tokens(monkeypatch, packed):
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
        {
            "input": tokens,
            "labels": tokens + 1,
            "positions": positions,
            "padding_mask": padding,
        },
        parallel_dims=topology,
        packed=packed,
        mask_dtype=torch.float32,
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
            batch,
            parallel_dims=SimpleNamespace(get_optional_mesh=forbidden),
            packed=False,
            mask_dtype=torch.float32,
        )


@pytest.mark.parametrize("shape", [(8,), (4, 2), (), (2, 3)])
def test_synthetic_labels_cannot_change_document_boundaries(shape):
    tokens = torch.arange(8).reshape(2, 4)
    labels = torch.ones(shape, dtype=torch.long)
    with pytest.raises(ValueError, match="same shape as input_ids"):
        prepare_text_inputs(
            Batch(input_ids=tokens, labels=labels),
            parallel_dims=None,
            packed=False,
            mask_dtype=torch.float32,
        )


def test_synthetic_shift_preserves_row_boundaries_and_ignores_each_terminal():
    tokens = torch.tensor([[1, 2, 3, 4], [9, 8, 7, 6]])
    original = tokens.clone()
    inputs, labels, _ = prepare_text_inputs(
        Batch(input_ids=tokens, labels=tokens),
        parallel_dims=None,
        preserve_batch_rows=True,
        packed=False,
        mask_dtype=torch.float32,
    )
    assert torch.equal(inputs, tokens.flatten())
    assert labels.tolist() == [2, 3, 4, -100, 8, 7, 6, -100]
    assert torch.equal(tokens, original)


@pytest.mark.parametrize("batched_labels", [False, True])
def test_restore_text_rows_preserves_independent_label_order(batched_labels):
    tokens = torch.arange(6)
    labels = torch.arange(10, 16)
    if batched_labels:
        labels = labels.reshape(2, 3)
    restored, targets, was_flat = input_preparation.restore_text_batch_rows(
        tokens, labels, batch_shape=(2, 3)
    )
    assert was_flat
    torch.testing.assert_close(restored, torch.tensor([[0, 1, 2], [3, 4, 5]]))
    torch.testing.assert_close(targets, torch.tensor([[10, 11, 12], [13, 14, 15]]))
    assert restored.data_ptr() == tokens.data_ptr()
    assert targets.data_ptr() == labels.data_ptr()


@pytest.mark.parametrize("shape", [(True, 6), (2, 0), (6,), (2, 4)])
def test_restore_text_rows_rejects_invalid_declared_shape(shape):
    with pytest.raises(ValueError, match="input_batch_shape"):
        input_preparation.restore_text_batch_rows(
            torch.arange(6), None, batch_shape=shape
        )


def test_restore_text_rows_rejects_transposed_label_layout():
    with pytest.raises(ValueError, match="labels must match"):
        input_preparation.restore_text_batch_rows(
            torch.arange(6), torch.arange(6).reshape(3, 2), batch_shape=(2, 3)
        )


@pytest.mark.parametrize("field", ["labels", "positions", "padding_mask"])
@pytest.mark.parametrize("shape", [(3, 2), (1, 2, 3)])
def test_preparation_rejects_wrong_token_axes_before_flattening(field, shape):
    tokens = torch.arange(6).reshape(2, 3)
    batch = {"input": tokens, "labels": tokens.clone()}
    batch[field] = torch.zeros(shape, dtype=torch.long)
    # Malformed token geometry must fail before parallel mesh lookup.
    with pytest.raises(ValueError, match=field):
        prepare_text_inputs(
            batch, parallel_dims=None, packed=False, mask_dtype=torch.float32
        )


def test_preparation_accepts_flat_metadata_for_batched_tokens():
    tokens = torch.arange(6).reshape(2, 3)
    labels = torch.arange(10, 16)
    positions = torch.tensor([0, 1, 2, 0, 1, 2])
    padding = torch.tensor([False, False, True, False, False, False])
    inputs, targets, extra = prepare_text_inputs(
        {
            "input": tokens,
            "labels": labels,
            "positions": positions,
            "padding_mask": padding,
        },
        parallel_dims=None,
        preserve_batch_rows=True,
        packed=False,
        mask_dtype=torch.float32,
    )
    torch.testing.assert_close(inputs, tokens.flatten())
    torch.testing.assert_close(targets, labels)
    torch.testing.assert_close(extra["positions"], positions.reshape(2, 3))
    torch.testing.assert_close(extra["padding_mask"], padding.reshape(2, 3))
