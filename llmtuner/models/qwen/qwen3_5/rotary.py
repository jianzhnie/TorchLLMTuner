"""Instance-local Qwen3.5 partial RoPE and cache precision policy."""

from types import FunctionType, MethodType

import torch

from ...errors import UnsupportedCombinationError

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeAttention,
        Qwen3_5MoeTextRotaryEmbedding,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        apply_rotary_pos_emb as apply_qwen35_rotary,
    )
except ImportError:
    Qwen3_5MoeAttention = Qwen3_5MoeTextRotaryEmbedding = None


def forward_qwen35_fp32_cache(module, x, position_ids):
    """Keep original HF frequency generation and interleaved mRoPE selection."""
    cosine, sine = type(module).forward(module, x.float(), position_ids)
    if getattr(module.config, "rotary_cache_dtype", None) == "bfloat16":
        cosine, sine = cosine.bfloat16().float(), sine.bfloat16().float()
    return cosine, sine


def apply_qwen35_fp32_rotary(q, k, cos, sin, unsqueeze_dim=1):
    """Promote only the original HF partial rotation; preserve pass-through bits."""
    dim = cos.shape[-1]
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("Qwen3.5 FP32 rotation requires FP32 cos/sin")
    qr, kr = apply_qwen35_rotary(
        q[..., :dim].float(), k[..., :dim].float(), cos, sin, unsqueeze_dim
    )
    return (
        torch.cat((qr.to(q.dtype), q[..., dim:]), dim=-1),
        torch.cat((kr.to(k.dtype), k[..., dim:]), dim=-1),
    )


def _apply_qwen35_fp32_rotary(model):
    """Rebind a single helper in instance-local original HF forward globals.

    The original attention bytecode, gated Q projection, cache handling and
    backend dispatch remain HF-owned. No class or shared HF globals are patched.
    """
    rope = model.model.rotary_emb
    modules = [m for m in model.modules() if type(m) is Qwen3_5MoeAttention]
    if type(rope) is not Qwen3_5MoeTextRotaryEmbedding or not modules:
        raise UnsupportedCombinationError(
            "FP32 Qwen3.5 RoPE requires native HF modules"
        )
    if model.config.rope_parameters.get("rope_type", "default") != "default":
        raise UnsupportedCombinationError("FP32 Qwen3.5 RoPE requires default RoPE")
    if "forward" in rope.__dict__ or any("forward" in m.__dict__ for m in modules):
        raise UnsupportedCombinationError("Cannot replace custom Qwen3.5 forwards")
    original = Qwen3_5MoeAttention.forward
    if original.__globals__.get("apply_rotary_pos_emb") is not apply_qwen35_rotary:
        raise UnsupportedCombinationError("Unexpected native Qwen3.5 rotary binding")
    namespace = {
        **original.__globals__,
        "apply_rotary_pos_emb": apply_qwen35_fp32_rotary,
    }
    forward = FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    forward.__kwdefaults__ = original.__kwdefaults__
    forward.__annotations__ = original.__annotations__
    for module in modules:
        module.forward = MethodType(forward, module)
    rope.forward = MethodType(forward_qwen35_fp32_cache, rope)
