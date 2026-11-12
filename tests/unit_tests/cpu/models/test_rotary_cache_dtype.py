"""Explicit RoPE cache precision contracts."""

import pytest
import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention as ReferenceAttention,
)

from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError
from llmtuner.models.qwen.qwen3.model import Qwen3MoeAttention, Qwen3MoeForCausalLM
from llmtuner.models.qwen.qwen3.normalization import build_rms_norm
from llmtuner.models.qwen.qwen3.rotary import (
    Qwen3FP32RotaryEmbedding,
    apply_fp32_rotary,
)
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


def make_rotary(cache_dtype):
    config = model_config(
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"}
    )
    config.rotary_compute_dtype = "float32"
    config.rotary_cache_dtype = cache_dtype
    return Qwen3FP32RotaryEmbedding(config)


@pytest.mark.parametrize("cache_dtype", ["float32", "bfloat16"])
def test_fp32_rotary_accumulates_branches_before_bf16_gradient_cast(cache_dtype):
    rotary = make_rotary(cache_dtype)
    torch.manual_seed(37)
    query = torch.randn(2, 3, 16, 8, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    positions = torch.arange(16)[None].expand(2, -1)
    actual = apply_fp32_rotary(query, key, *rotary(query, positions))
    frequencies = 1.0 / (1000000.0 ** (torch.arange(0, 8, 2).float() / 8))
    angles = positions.float().unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), -1).unsqueeze(1)
    cosine, sine = angles.cos(), angles.sin()
    if cache_dtype == "bfloat16":
        cosine, sine = cosine.bfloat16().float(), sine.bfloat16().float()

    def reference(value):
        # The reference converter shares one FP32 cast across both branches.
        promoted = value.float()
        first, second = promoted.chunk(2, -1)
        return (promoted * cosine + torch.cat((-second, first), -1) * sine).bfloat16()

    expected = tuple(reference(value) for value in (query, key))
    probes = tuple(torch.randn_like(value) for value in actual)
    for observed, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, target, atol=0, rtol=0)
    actual_gradients = torch.autograd.grad(
        actual, (query, key), probes, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected, (query, key), probes)
    for observed, target in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, target, atol=0, rtol=0)


def test_bf16_cache_matches_quantized_reference():
    rotary = make_rotary("bfloat16")
    torch.manual_seed(37)
    query = torch.randn(1, 2, 32, 8, dtype=torch.bfloat16)
    positions = torch.arange(32)[None]
    actual, _ = apply_fp32_rotary(query, query, *rotary(query, positions))
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
    rotary.config.rotary_cache_dtype = None
    unquantized, _ = apply_fp32_rotary(query, query, *rotary(query, positions))
    assert not torch.equal(actual, unquantized)


def test_cache_precision_requires_fp32_application():
    with pytest.raises(ConfigError, match="requires FP32"):
        ModelConfig(rotary_cache_dtype="bfloat16")


def test_npu_norm_rejects_cpu_input():
    module = build_rms_norm(8, 1e-6, backend="npu")
    with pytest.raises(RuntimeError, match="requires NPU"):
        module(torch.ones(2, 8))


def test_separate_qkv_fp32_rope_preserves_projection_calls():
    config = model_config()
    config.qkv_projection = "separate"
    config.rotary_compute_dtype = "float32"
    config._attn_implementation = "eager"
    attention = Qwen3MoeAttention(config, layer_idx=0)
    rotary = Qwen3FP32RotaryEmbedding(config)
    calls = []
    handles = [
        module.register_forward_hook(lambda module, args, output: calls.append(module))
        for module in (attention.q_proj, attention.k_proj, attention.v_proj)
    ]
    hidden = torch.randn(1, 4, 16, requires_grad=True)
    positions = torch.arange(4)[None]
    try:
        output, _ = attention(hidden, rotary(hidden, positions), None)
        output.sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [attention.q_proj, attention.k_proj, attention.v_proj]
    assert torch.isfinite(hidden.grad).all() and hidden.grad.count_nonzero() > 0
    assert all(module.weight.grad.count_nonzero() > 0 for module in calls)


@pytest.mark.parametrize("projection", ["separate", "fused"])
def test_precision_construction_does_not_patch_external_or_other_instances(projection):
    native_config = model_config()
    native_config._attn_implementation = "eager"
    external = ReferenceAttention(native_config, layer_idx=0)
    external.forward = lambda *args, **kwargs: None
    external_forward = external.forward
    reference_forward = ReferenceAttention.forward
    untouched = Qwen3MoeAttention(native_config, layer_idx=0)
    local_forward = untouched.forward
    config = model_config()
    config.rotary_compute_dtype = "float32"
    config.qkv_projection = projection
    model = Qwen3MoeForCausalLM(config)
    parameters = {name: id(value) for name, value in model.named_parameters()}
    state_keys = tuple(model.state_dict())
    assert ReferenceAttention.forward is reference_forward
    assert external.forward is external_forward
    assert untouched.forward == local_forward
    assert "forward" not in untouched.__dict__
    assert all("forward" not in layer.self_attn.__dict__ for layer in model.layers)
    for _ in range(2):
        result = model(torch.tensor([[1, 3, 7, 2]])).logits
        assert result.isfinite().all() and result.count_nonzero() > 0
    assert parameters == {name: id(value) for name, value in model.named_parameters()}
    assert tuple(model.state_dict()) == state_keys
