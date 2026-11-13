"""Experimental HF inputs must not import removed generic CP mask kernels."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.huggingface_backend import model as hf_backend
from experiments.huggingface_backend.attention_masks import get_cp_attention_masks
from experiments.huggingface_backend.input_preparation import preprocess_inputs
from llmtuner.datasets.contracts import Batch
from llmtuner.errors import UnsupportedCombinationError


def test_experimental_hf_inputs_work_without_cp():
    model = nn.Linear(4, 4)
    model.model = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa", attn_mask_type="causal")
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    inputs, labels, extra = preprocess_inputs(
        model, Batch(input_ids=tokens, labels=tokens.clone()), parallel_dims=None
    )
    assert torch.equal(inputs, tokens.flatten())
    assert torch.equal(labels, torch.tensor([2, 3, 4, -100]))
    assert extra == {}


def test_experimental_hf_rejects_removed_generic_cp_before_exchange():
    model = SimpleNamespace(_cp_strategy="kv_allgather")
    topology = SimpleNamespace(get_optional_mesh=lambda name: object())
    tokens = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(UnsupportedCombinationError, match="kernels have been removed"):
        preprocess_inputs(
            model, Batch(input_ids=tokens, labels=tokens), parallel_dims=topology
        )


def test_experimental_hf_rejects_removed_cp_mask_contract():
    with pytest.raises(UnsupportedCombinationError, match="kernels have been removed"):
        get_cp_attention_masks(object(), torch.arange(4))


def test_experimental_hf_model_native_forward_backward_without_cp(monkeypatch):
    monkeypatch.setattr(
        hf_backend, "resolve_model_class", lambda config: LlamaForCausalLM
    )
    monkeypatch.setattr(hf_backend, "available_attention_backend", lambda: "sdpa")
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    model = hf_backend.HFTransformerModel(config)
    tokens = torch.tensor([1, 4, 2, 7])
    positions = torch.arange(tokens.numel())
    expected = model.model(
        input_ids=tokens.unsqueeze(0),
        position_ids=positions.unsqueeze(0),
        use_cache=False,
    ).logits.squeeze(0)
    actual = model(tokens, positions=positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (4, 32)
    assert torch.isfinite(actual).all()
    actual.square().mean().backward()
    gradient = model.tok_embeddings.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("norm_backend", "npu"),
        ("qkv_projection", "fused"),
        ("rotary_compute_dtype", "float32"),
        ("linear_attention_backend", "npu_triton"),
        ("gdn_numerical_policy", "float32"),
        ("experts_implementation", "torchtitan"),
        ("experts_implementation", "pre_down"),
    ],
)
def test_experimental_hf_rejects_retired_policies_before_construction(
    monkeypatch, name, value
):
    def unexpected_construction(config):
        pytest.fail("Retired policies must fail before model class resolution")

    monkeypatch.setattr(hf_backend, "resolve_model_class", unexpected_construction)
    config = LlamaConfig()
    setattr(config, name, value)
    with pytest.raises(UnsupportedCombinationError, match="explicit model"):
        hf_backend.HFTransformerModel(config)
