"""Flattened independent rows retain batch attention boundaries."""

import pytest
import torch

from llmtuner.models.hf import input_preparation
from llmtuner.models.hf.attention_backend import ATTENTION_BACKEND
from llmtuner.models.hf.factory import build_model_config
from llmtuner.models.hf.model import HFTransformerModel


def test_causal_batch_rows_remain_independent_after_flattening():
    config = build_model_config(
        "qwen3",
        seq_len=64,
        arch_overrides={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    )
    model = HFTransformerModel(config).eval()
    model.model.config._attn_implementation = "sdpa"
    model.model.config.attn_mask_type = "causal"
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    batch = {"input": tokens, "labels": tokens + 1}
    inputs, _, kwargs = model.preprocess_inputs(batch, parallel_dims=None)
    assert kwargs["positions"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    mask = kwargs["attention_masks"]
    assert torch.isneginf(mask[..., 4:, :4]).all()
    with torch.no_grad():
        actual = model(inputs, **kwargs)
        expected = model.model(input_ids=tokens, use_cache=False).logits.flatten(0, 1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual = model(inputs, **kwargs)
    expected = model.model(input_ids=tokens, use_cache=False).logits.flatten(0, 1)
    probe = torch.linspace(-1, 1, actual.numel()).reshape_as(actual)
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(actual, parameters, probe)
    expected_gradients = torch.autograd.grad(expected, parameters, probe)
    for observed, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, reference, atol=1e-5, rtol=1e-4)


def test_multirow_cp_rejects_before_sharding():
    from types import SimpleNamespace

    from llmtuner.errors import UnsupportedCombinationError

    config = build_model_config(
        "qwen3",
        seq_len=64,
        arch_overrides={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    )
    model = HFTransformerModel(config)
    topology = SimpleNamespace(get_optional_mesh=lambda name: object())
    tokens = torch.ones(2, 4, dtype=torch.long)
    with pytest.raises(UnsupportedCombinationError, match="row-aware mask"):
        model.preprocess_inputs(
            {"input": tokens, "labels": tokens}, parallel_dims=topology
        )


@pytest.mark.parametrize("packed", [False, True])
def test_kv_sdpa_multirow_mask_keeps_full_keys_and_row_positions(monkeypatch, packed):
    from types import SimpleNamespace

    from llmtuner.models.hf.attention_backend import CP_SDPA_BACKEND

    config = build_model_config(
        "qwen3",
        seq_len=64,
        arch_overrides={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    )
    model = HFTransformerModel(config)
    model.model.config._attn_implementation = CP_SDPA_BACKEND
    model.model.config.attn_mask_type = "block_causal" if packed else "causal"
    cp = object()
    topology = SimpleNamespace(
        get_optional_mesh=lambda name: cp if name == "cp" else None
    )

    def shard(inputs, labels, positions, mesh, **kwargs):
        assert mesh is cp
        return inputs[4:], labels[4:], positions[4:]

    mask_builds = []
    build_mask = input_preparation.build_dense_attention_mask

    def counted_mask(*args, **kwargs):
        mask_builds.append(1)
        return build_mask(*args, **kwargs)

    monkeypatch.setattr(input_preparation, "shard_batch_for_cp", shard)
    monkeypatch.setattr(input_preparation, "build_dense_attention_mask", counted_mask)
    tokens = torch.arange(8).reshape(2, 4)
    batch = {"input": tokens, "labels": tokens}
    if packed:
        batch["positions"] = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]])
    inputs, _, kwargs = model.preprocess_inputs(batch, parallel_dims=topology)
    assert len(mask_builds) == 1
    assert inputs.tolist() == [4, 5, 6, 7]
    assert kwargs["positions"].tolist() == [0, 1, 2, 3]
    mask = kwargs["attention_masks"]
    assert mask.dtype == torch.bool and mask.shape == (1, 1, 8, 8)
    assert not mask[..., 4:, :4].any()
    assert mask[0, 0, 4, 4] and not mask[0, 0, 4, 5]
    if packed:
        assert not mask[0, 0, 2, :2].any()
    model.cp_mesh = cp
    assert model._apply_attention(kwargs["positions"], mask)["attention_mask"] is mask


@pytest.mark.parametrize("backend", ["sdpa", ATTENTION_BACKEND])
@pytest.mark.parametrize("position_start", [0, 5])
def test_packed_rows_with_continuation_positions_remain_independent(
    backend, position_start, monkeypatch
):
    config = build_model_config(
        "qwen3",
        seq_len=16,
        arch_overrides={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
        },
    )
    model = HFTransformerModel(config).eval()
    model.model.config._attn_implementation = backend
    model.model.config.attn_mask_type = "block_causal"

    def unexpected_generic_mask(**kwargs):
        pytest.fail("multirow preparation must build only the row-aware mask")

    monkeypatch.setattr(model, "get_attention_masks", unexpected_generic_mask)
    tokens = torch.tensor([[1, 2, 3], [7, 11, 13]])
    positions = torch.arange(position_start, position_start + 6).reshape(2, 3)
    original_positions = positions.clone()
    inputs, _, kwargs = model.preprocess_inputs(
        {"input": tokens, "labels": tokens, "positions": positions},
        parallel_dims=None,
    )
    assert torch.equal(positions, original_positions)
    assert torch.equal(kwargs["positions"], original_positions.flatten())
    mask = kwargs["attention_masks"]
    if backend == ATTENTION_BACKEND:
        batch = head = torch.tensor(0)
        query = torch.arange(3, 6)[:, None]
        key = torch.arange(3)[None, :]
        assert not mask.mask_mod(batch, head, query, key).any()
        assert mask.mask_mod(batch, head, torch.tensor(5), torch.tensor(3))
        assert not mask.mask_mod(batch, head, torch.tensor(3), torch.tensor(5))
        return
    assert torch.isneginf(mask[..., 3:, :3]).all()
    with torch.no_grad():
        packed = model(inputs, **kwargs)[3:]
        independent = model(tokens[1], positions=positions[1])
    assert torch.isfinite(packed).all() and packed.count_nonzero() > 0
    torch.testing.assert_close(packed, independent, rtol=1e-5, atol=1e-7)
