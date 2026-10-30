"""Explicit RoPE cache precision contracts."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.models.hf.qkv_projection import apply_configured_rotary


def test_bf16_cache_matches_quantized_reference():
    module = SimpleNamespace(
        config=SimpleNamespace(
            rotary_compute_dtype="float32",
            rotary_cache_dtype="bfloat16",
            rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
        )
    )
    torch.manual_seed(37)
    query = torch.randn(1, 2, 32, 8, dtype=torch.bfloat16)
    positions = torch.arange(32)[None]
    actual, _ = apply_configured_rotary(module, query, query, None, None, positions)
    frequencies = 1.0 / (1000000.0 ** (torch.arange(0, 8, 2).float() / 8))
    angles = positions.float().unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), -1).unsqueeze(1)
    cosine = angles.cos().bfloat16().float()
    sine = angles.sin().bfloat16().float()
    first, second = query.float().chunk(2, -1)
    expected = (
        query.float() * cosine + torch.cat((-second, first), -1) * sine
    ).bfloat16()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    module.config.rotary_cache_dtype = None
    unquantized, _ = apply_configured_rotary(
        module, query, query, None, None, positions
    )
    assert not torch.equal(actual, unquantized)


def test_cache_precision_requires_fp32_application():
    with pytest.raises(ConfigError, match="requires FP32"):
        ModelConfig(rotary_cache_dtype="bfloat16")


def test_npu_norm_rejects_cpu_input():
    from llmtuner.models.hf.model_setup import forward_npu_rms_norm

    module = SimpleNamespace(weight=torch.ones(8), variance_epsilon=1e-6)
    with pytest.raises(RuntimeError, match="requires NPU"):
        forward_npu_rms_norm(module, torch.ones(2, 8))


def test_separate_qkv_fp32_rope_preserves_projection_calls():
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    from llmtuner.models.hf.qkv_projection import apply_fp32_rotary_projection

    config = AutoConfig.for_model(
        "qwen3",
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    config.qkv_projection = "separate"
    config.rotary_compute_dtype = "float32"
    config._attn_implementation = "eager"
    attention = Qwen3Attention(config, layer_idx=0)
    calls = []
    handles = [
        module.register_forward_hook(lambda module, args, output: calls.append(module))
        for module in (attention.q_proj, attention.k_proj, attention.v_proj)
    ]
    wrapper = SimpleNamespace(
        config=config,
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]),
    )
    apply_fp32_rotary_projection(wrapper)
    apply_fp32_rotary_projection(wrapper)
    hidden = torch.randn(1, 4, 16, requires_grad=True)
    positions = torch.arange(4)[None]
    output, _ = attention(
        hidden,
        (None, None),
        None,
        position_ids=positions,
    )
    output.sum().backward()
    assert calls == [attention.q_proj, attention.k_proj, attention.v_proj]
    assert torch.isfinite(hidden.grad).all()
    assert all(module.weight.grad is not None for module in calls)
    for handle in handles:
        handle.remove()


def test_fp32_rotary_rejects_custom_forward_before_modifying_any_layer():
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    from llmtuner.models.hf.qkv_projection import apply_fp32_rotary_projection

    config = AutoConfig.for_model(
        "qwen3",
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    config.qkv_projection = "separate"
    config.rotary_compute_dtype = "float32"
    first, second = (Qwen3Attention(config, layer_idx=i) for i in range(2))
    second.forward = lambda *args, **kwargs: None
    wrapper = SimpleNamespace(
        config=config,
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=first), SimpleNamespace(self_attn=second)]
        ),
    )
    with pytest.raises(UnsupportedCombinationError, match="native attention"):
        apply_fp32_rotary_projection(wrapper)
    assert "forward" not in first.__dict__
