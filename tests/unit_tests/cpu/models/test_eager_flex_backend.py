"""HF eager Flex selection preserves causal and packed attention semantics."""

from tests.caps import require_env

require_env("flex_attention", "spmd_types")

import copy

import pytest
import torch

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.models.hf.attention_backend import (
    EAGER_FLEX_BACKEND,
    allow_npu_flex_attention,
    eager_flex_attention_adapter,
)
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel


@pytest.mark.parametrize("packed", [False, True])
def test_eager_flex_backend_model_forward_and_gradients(packed):
    config = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_backend="eager_flex",
            arch_overrides={"head_dim": 8},
        )
    )
    hf_config = build_model_config_for(config)
    hf_config.attn_mask_type = "block_causal" if packed else "causal"
    model = HFTransformerModel(hf_config)
    reference = copy.deepcopy(model)
    reference.model.config._attn_implementation = "sdpa"
    assert model.model.config._attn_implementation == EAGER_FLEX_BACKEND
    tokens = torch.arange(7)
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3]) if packed else torch.arange(7)
    result = model(tokens, positions=positions)
    expected = reference(tokens, positions=positions)
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    expected.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    for actual_parameter, expected_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, atol=2e-6, rtol=2e-6
        )


@pytest.mark.parametrize("option", ["tp", "cp", "compile"])
def test_eager_flex_rejects_unadapted_configuration(option):
    from llmtuner.config.parallel_config import ParallelConfig
    from llmtuner.config.training_config import TrainingConfig

    with pytest.raises(ConfigError, match="eager_flex"):
        LLMTunerConfig(
            model=ModelConfig(attention_backend="eager_flex"),
            parallel=ParallelConfig(
                tensor_parallel_size=2 if option == "tp" else 1,
                context_parallel_kv_allgather_size=2 if option == "cp" else 1,
            ),
            training=TrainingConfig(compile=option == "compile"),
        )


def test_eager_flex_adapter_rejects_dropout_and_attention_biases():
    query = torch.randn(1, 2, 7, 8)
    with pytest.raises(UnsupportedCombinationError, match="dropout"):
        eager_flex_attention_adapter(None, query, query, query, None, dropout=0.1)
    with pytest.raises(UnsupportedCombinationError, match="biases"):
        eager_flex_attention_adapter(None, query, query, query, torch.ones(1, 1, 7, 7))


def test_npu_flex_device_validation_preserves_other_device_checks(monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("torch.nn.attention.flex_attention")

    def reject(*args):
        raise ValueError("rejected")

    monkeypatch.setattr(module, "_validate_device", reject)
    monkeypatch.delattr(module, "_llmtuner_original_validate_device", raising=False)
    allow_npu_flex_attention()
    npu_tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
    cpu_tensor = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    allow_npu_flex_attention()
    module._validate_device(npu_tensor, npu_tensor, npu_tensor)
    with pytest.raises(ValueError, match="rejected"):
        module._validate_device(cpu_tensor, cpu_tensor, cpu_tensor)


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_eager_flex_ac_replay_preserves_gradients(mode):
    from llmtuner.config.activation_checkpoint_config import SelectiveACConfig
    from llmtuner.memory.activation_checkpoint import wrap_full, wrap_selective

    class Attention(torch.nn.Module):
        def forward(self, query, key, value):
            return eager_flex_attention_adapter(None, query, key, value, None)[0]

    torch.manual_seed(91)
    inputs = [torch.randn(1, heads, 7, 8, requires_grad=True) for heads in (4, 2, 2)]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    wrapped = (
        wrap_full(Attention())
        if mode == "full"
        else wrap_selective(Attention(), SelectiveACConfig())
    )
    actual = wrapped(*inputs)
    expected = Attention()(*reference_inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, inputs, upstream)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, upstream)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)


def test_flex_adapter_rejects_dropout_before_dispatch():
    from llmtuner.models.hf.attention_backend import flex_attention_adapter

    query = torch.randn(1, 2, 7, 8)
    with pytest.raises(UnsupportedCombinationError, match="dropout"):
        flex_attention_adapter(None, query, query, query, None, dropout=0.1)


@pytest.mark.parametrize("extra", [{"softcap": 2.0}, {"s_aux": torch.ones(2)}])
def test_explicit_flex_preserves_score_modifiers(monkeypatch, extra):
    from types import SimpleNamespace

    from torch.nn.attention.flex_attention import create_block_mask

    from llmtuner.models.hf import attention_backend

    query = torch.randn(1, 2, 7, 8)
    mask = create_block_mask(lambda b, h, q, k: q >= k, 1, None, 7, 7, device="cpu")
    module = SimpleNamespace(config=SimpleNamespace(attention_backend="flex"))
    observed = {}

    def fallback(*args, **kwargs):
        observed.update(kwargs)
        return query, None

    monkeypatch.setattr(attention_backend, "flex_attention_forward", fallback)
    attention_backend.flex_attention_adapter(module, query, query, query, mask, **extra)
    assert observed.keys() >= extra.keys()
    for name, value in extra.items():
        assert observed[name] is value
