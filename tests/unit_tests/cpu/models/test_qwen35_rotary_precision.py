"""Original HF partial/mRoPE semantics with instance-local precision boundaries."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.models.qwen.qwen3.qkv_projection import (
    apply_fp32_rotary_projection,
)
from llmtuner.models.qwen.qwen3_5.rotary import apply_qwen35_fp32_rotary


def make_model(cache):
    config = Qwen3_5MoeTextConfig(
        hidden_size=32,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        num_hidden_layers=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
    )
    config.rotary_compute_dtype = "float32"
    config.rotary_cache_dtype = cache
    config._attn_implementation = "eager"
    model = nn.Module()
    model.config = config
    model.model = nn.Module()
    model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config)
    model.model.attn = Qwen3_5MoeAttention(config, layer_idx=3).bfloat16()
    return model


@pytest.mark.parametrize("cache", [None, "float32", "bfloat16"])
@pytest.mark.parametrize("mrope", [False, True])
def test_partial_rotary_cache_forward_vjp_and_instance_isolation(cache, mrope):
    model = make_model(cache)
    untouched = make_model(cache)
    params = dict(model.named_parameters())
    state = {k: v.clone() for k, v in model.state_dict().items()}
    original = Qwen3_5MoeAttention.forward
    apply_fp32_rotary_projection(model)
    assert Qwen3_5MoeAttention.forward is original
    assert untouched.model.attn.forward.__func__ is original
    assert model.model.attn.forward.__func__.__code__ is original.__code__
    assert original.__globals__["apply_rotary_pos_emb"] is apply_rotary_pos_emb
    assert model.state_dict().keys() == state.keys()
    for key, value in model.named_parameters():
        assert value is params[key] and torch.equal(value, state[key])
    generator = torch.Generator().manual_seed(23)
    q = torch.randn((1, 128, 16, 256), generator=generator).bfloat16().requires_grad_()
    k = torch.randn((1, 128, 2, 256), generator=generator).bfloat16().requires_grad_()
    qr, kr = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    positions = torch.arange(128)[None, :]
    if mrope:
        positions = torch.stack((positions, positions * 2 + 1, positions * 3 + 2))
    cos, sin = model.model.rotary_emb(q, positions)
    expected_cos, expected_sin = untouched.model.rotary_emb(q.float(), positions)
    if cache == "bfloat16":
        expected_cos = expected_cos.bfloat16().float()
        expected_sin = expected_sin.bfloat16().float()
    assert cos.dtype == sin.dtype == torch.float32
    assert cos.shape == (1, 128, 64)
    assert torch.equal(cos, expected_cos) and torch.equal(sin, expected_sin)
    a, b = apply_qwen35_fp32_rotary(q, k, cos, sin, unsqueeze_dim=2)
    ar, br = apply_rotary_pos_emb(
        qr.float(), kr.float(), expected_cos, expected_sin, unsqueeze_dim=2
    )
    ar, br = ar.bfloat16(), br.bfloat16()
    assert torch.equal(a, ar) and torch.equal(b, br)
    uq = torch.randn(q.shape, generator=generator).bfloat16()
    uk = torch.randn(k.shape, generator=generator).bfloat16()
    dq, dk = torch.autograd.grad((a, b), (q, k), (uq, uk))
    dqr, dkr = torch.autograd.grad((ar, br), (qr, kr), (uq, uk))
    assert torch.equal(dq, dqr) and torch.equal(dk, dkr)
    for actual, expected in ((a, q), (b, k), (dq, uq), (dk, uk)):
        assert torch.equal(actual[..., 64:], expected[..., 64:])
        assert bool(actual.isfinite().all()) and bool(actual.count_nonzero())
    # Exercise the ORIGINAL attention bytecode, not just the helper directly.
    x = torch.randn((1, 8, 32), generator=generator).bfloat16().requires_grad_()
    c, s = model.model.rotary_emb(x, torch.arange(8)[None, :])
    output, _ = model.model.attn(x, (c, s), attention_mask=None)
    output.float().square().sum().backward()
    assert x.grad is not None and bool(x.grad.isfinite().all())


def test_float32_cache_config_and_legacy_qwen3_constraint():
    assert (
        ModelConfig(
            rotary_compute_dtype="float32", rotary_cache_dtype="float32"
        ).rotary_cache_dtype
        == "float32"
    )
    with pytest.raises(ConfigError, match="requires FP32"):
        ModelConfig(rotary_cache_dtype="float32")
    model = nn.Module()
    model.config = SimpleNamespace(
        model_type="qwen3", rotary_compute_dtype="float32", rotary_cache_dtype="float32"
    )
    with pytest.raises(UnsupportedCombinationError, match="requires qwen3_5"):
        apply_fp32_rotary_projection(model)


@pytest.mark.parametrize("cast_inputs", [True, False])
def test_real_cpu_fsdp_boundary_preserves_unrounded_rotary_cache(cast_inputs):
    """Real FSDP hook + original HF attention, without accelerator/numeric mocks."""
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (dist.get_world_size(),))
        model = make_model("float32")
        apply_fp32_rotary_projection(model)
        generator = torch.Generator().manual_seed(83)
        x = torch.randn((1, 8, 32), generator=generator).bfloat16()
        cos, sin = model.model.rotary_emb(x, torch.arange(8)[None, :])
        assert cos.dtype == sin.dtype == torch.float32
        assert not torch.equal(cos, cos.bfloat16().float())
        expected, _ = model.model.attn(x, (cos, sin), attention_mask=None)
        expected = expected.detach().clone()

        class DecoderBoundary(nn.Module):
            def __init__(self, attention):
                super().__init__()
                self.attention = attention
                self.observed = None

            def forward(self, hidden, *, position_embeddings):
                self.observed = tuple(t.detach().clone() for t in position_embeddings)
                output, _ = self.attention(
                    hidden, position_embeddings, attention_mask=None
                )
                return output

        boundary = DecoderBoundary(model.model.attn)
        fully_shard(
            boundary,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=cast_inputs,
            ),
        )
        x = x.detach().requires_grad_()
        if cast_inputs:
            with pytest.raises(ValueError, match="requires FP32 cos/sin"):
                boundary(x, position_embeddings=(cos, sin))
            assert all(t.dtype == torch.bfloat16 for t in boundary.observed)
            assert not torch.equal(boundary.observed[0].float(), cos)
        else:
            output = boundary(x, position_embeddings=(cos, sin))
            for actual, original in zip(boundary.observed, (cos, sin), strict=True):
                assert actual.dtype == torch.float32
                assert torch.equal(actual, original)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            output.float().square().sum().backward()
            assert x.grad is not None and bool(x.grad.isfinite().all())
            assert bool(x.grad.count_nonzero())
    finally:
        if created_group:
            dist.destroy_process_group()
