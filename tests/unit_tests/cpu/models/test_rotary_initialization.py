"""Explicit Qwen3 owns rotary initialization and runtime buffer materialization."""

import pytest
import torch
from transformers import PretrainedConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError
from llmtuner.memory.materialization import materialize_meta_model
from llmtuner.models.configuration import build_model_config_for, unwrap_text_config
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM


def make_config(policy: str = "constructor") -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            rotary_init_device=policy,
            arch_overrides={
                "head_dim": 8,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
            },
        )
    )


@pytest.mark.parametrize("policy", ["constructor", "target"])
def test_rotary_policy_reaches_explicit_text_config(policy):
    config = make_config(policy)
    text = build_model_config_for(config)
    assert config.rotary_init_device == policy
    assert text.rotary_init_device == policy
    composite = PretrainedConfig()
    composite.text_config = text
    composite.rotary_init_device = policy
    assert unwrap_text_config(composite).rotary_init_device == policy


def test_invalid_rotary_policy_is_rejected():
    with pytest.raises(ConfigError, match="rotary_init_device"):
        ModelConfig(rotary_init_device="npu")


def test_target_rotary_rejected_before_model_allocation(monkeypatch):
    config = build_model_config_for(make_config("target"))
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported policy reached allocation")

    monkeypatch.setattr(torch.nn.Embedding, "__init__", forbidden)
    with pytest.raises(ValueError, match="constructor RoPE"):
        Qwen3MoeForCausalLM(config)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize(
    "compute,cache", [(None, None), ("float32", None), ("float32", "bfloat16")]
)
def test_meta_rotary_materialization_preserves_state_and_aliases(compute, cache):
    config = build_model_config_for(make_config())
    config.rotary_compute_dtype = compute
    config.rotary_cache_dtype = cache
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    rotary = model.model.rotary_emb
    model.register_buffer("rotary_alias", rotary.inv_freq, persistent=False)
    parameters = {name: id(value) for name, value in model.named_parameters()}
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    buffers = model.materialization_buffers(torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), rng)
    materialize_meta_model(model, torch.device("cpu"), runtime_buffers=buffers)
    assert model.rotary_alias is rotary.inv_freq
    assert parameters == {name: id(value) for name, value in model.named_parameters()}
    assert keys == tuple(model.state_dict())
    reference = Qwen3MoeRotaryEmbedding(config)
    for name in ("inv_freq", "original_inv_freq"):
        value = getattr(rotary, name)
        assert not value.is_meta and value.dtype == torch.float32
        torch.testing.assert_close(value, getattr(reference, name), atol=0, rtol=0)
    hidden = torch.randn(2, 5, config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(5).expand(2, -1)
    cosine, sine = rotary(hidden, positions)
    if compute == "float32":
        expected = reference(hidden.float(), positions)
        if cache == "bfloat16":
            expected = tuple(value.bfloat16() for value in expected)
    else:
        expected = reference(hidden, positions)
    torch.testing.assert_close((cosine, sine), expected, atol=0, rtol=0)
    assert cosine.numel() > 0 and sine.count_nonzero() > 0
    model.initialize_parameters()
    output = model(torch.tensor([[1, 3, 7, 2]])).logits
    assert output.isfinite().all() and output.count_nonzero() > 0
