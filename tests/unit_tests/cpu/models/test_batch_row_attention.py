"""Flattened independent rows retain batch attention boundaries."""

import pytest
import torch

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

    def shard(mesh, **kwargs):
        assert mesh is cp
        return (
            kwargs["inputs"][4:],
            kwargs["labels"][4:],
            kwargs["positions"][4:],
            None,
            None,
        )

    monkeypatch.setattr(model, "_shard_for_cp", shard)
    tokens = torch.arange(8).reshape(2, 4)
    batch = {"input": tokens, "labels": tokens}
    if packed:
        batch["positions"] = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]])
    inputs, _, kwargs = model.preprocess_inputs(batch, parallel_dims=topology)
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
