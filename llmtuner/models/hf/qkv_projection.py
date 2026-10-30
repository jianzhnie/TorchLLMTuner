"""Instance-local Qwen3 projection and rotary precision overrides."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    rotate_half,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from llmtuner.errors import UnsupportedCombinationError


def _validate_projections(module: nn.Module) -> None:
    projections = (module.q_proj, module.k_proj, module.v_proj)
    for projection in projections:
        if type(projection) is not nn.Linear:
            raise UnsupportedCombinationError(
                "model.qkv_projection='fused' requires plain nn.Linear Q/K/V modules; "
                "custom or sharded projection forwards must be adapted explicitly."
            )
        if (
            projection._forward_pre_hooks
            or projection._forward_hooks
            or projection._backward_pre_hooks
            or projection._backward_hooks
            or "forward" in projection.__dict__
        ):
            raise UnsupportedCombinationError(
                "model.qkv_projection='fused' cannot bypass Q/K/V module hooks or "
                "instance forward overrides; use separate projections."
            )
    has_bias = [projection.bias is not None for projection in projections]
    if any(has_bias) and not all(has_bias):
        raise UnsupportedCombinationError(
            "model.qkv_projection='fused' requires all three Q/K/V biases or none."
        )


def fused_qkv_projection(
    module: nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project once in KV-group order without replacing registered parameters."""
    _validate_projections(module)
    head_dim = module.head_dim
    kv_heads = module.config.num_key_value_heads
    heads_per_kv = module.config.num_attention_heads // kv_heads
    hidden_width = hidden_states.shape[-1]
    projections = (module.q_proj, module.k_proj, module.v_proj)
    weights = [
        projection.weight.reshape(kv_heads, -1, head_dim, hidden_width)
        for projection in projections
    ]
    packed_weight = torch.cat(weights, dim=1).reshape(-1, hidden_width)
    packed_bias = None
    if projections[0].bias is not None:
        packed_bias = torch.cat(
            [
                projection.bias.reshape(kv_heads, -1, head_dim)
                for projection in projections
            ],
            dim=1,
        ).reshape(-1)
    combined = functional.linear(hidden_states, packed_weight, packed_bias)
    grouped = combined.unflatten(-1, (kv_heads, heads_per_kv + 2, head_dim))
    return tuple(
        part.flatten(-3, -2).contiguous()
        for part in grouped.split((heads_per_kv, 1, 1), dim=-2)
    )


def apply_configured_rotary(module, query, key, cosine, sine, position_ids):
    if getattr(module.config, "rotary_compute_dtype", None) != "float32":
        return apply_rotary_pos_emb(query, key, cosine, sine)
    if position_ids is None:
        raise ValueError("FP32 RoPE requires explicit position_ids")
    if module.config.rope_parameters.get("rope_type", "default") != "default":
        raise UnsupportedCombinationError(
            "FP32 RoPE currently requires default RoPE without scaling"
        )
    head_dim = query.shape[-1]
    theta = module.config.rope_parameters["rope_theta"]
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=query.device, dtype=torch.float32)
            / head_dim
        )
    )
    angles = position_ids.float().unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    cosine = angles.cos()
    sine = angles.sin()
    if getattr(module.config, "rotary_cache_dtype", None) == "bfloat16":
        # Preserve the baseline buffer quantization while applying RoPE in FP32.
        cosine = cosine.bfloat16().float()
        sine = sine.bfloat16().float()
    query_dtype = query.dtype
    key_dtype = key.dtype
    if query.device.type == "npu" and (
        getattr(module.config, "rotary_cache_dtype", None) == "bfloat16"
    ):
        import torch_npu

        # Match both the baseline rotary kernel and its backward reduction.
        # An algebraically equivalent eager rotation can match BF16 forward
        # exactly while rounding its input gradient differently.
        cosine = cosine.transpose(1, 2)
        sine = sine.transpose(1, 2)
        query = torch_npu.npu_rotary_mul(
            query.transpose(1, 2).float(), cosine, sine
        ).transpose(1, 2)
        key = torch_npu.npu_rotary_mul(
            key.transpose(1, 2).float(), cosine, sine
        ).transpose(1, 2)
        return query.to(query_dtype), key.to(key_dtype)
    query = (query.float() * cosine) + (rotate_half(query.float()) * sine)
    key = (key.float() * cosine) + (rotate_half(key.float()) * sine)
    return query.to(query_dtype), key.to(key_dtype)


def forward_fused_qkv_attention(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Keep Qwen3 attention semantics with configured QKV and rotary precision."""
    if getattr(module.config, "qkv_projection", "separate") == "fused":
        query, key, value = fused_qkv_projection(module, hidden_states)
    else:
        query, key, value = (
            projection(hidden_states).unflatten(-1, (-1, module.head_dim))
            for projection in (module.q_proj, module.k_proj, module.v_proj)
        )
    query = module.q_norm(query).transpose(1, 2)
    key = module.k_norm(key).transpose(1, 2)
    value = value.transpose(1, 2)
    cosine, sine = position_embeddings
    query, key = apply_configured_rotary(
        module, query, key, cosine, sine, kwargs.get("position_ids")
    )
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, module.layer_idx)
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation, eager_attention_forward
    )
    output, weights = attention_interface(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        sliding_window=module.sliding_window,
        **kwargs,
    )
    output = output.reshape(*hidden_states.shape[:-1], -1).contiguous()
    return module.o_proj(output), weights


def apply_fused_qkv_projection(model: nn.Module) -> None:
    """Install only on validated Qwen3 instances, before parallel transforms."""
    if getattr(model.config, "qkv_projection", "separate") != "fused":
        raise ValueError("model.qkv_projection must be 'fused' for fused installation")
    if model.config.model_type not in ("qwen3", "qwen3_moe"):
        raise UnsupportedCombinationError(
            "model.qkv_projection='fused' requires Qwen3 or Qwen3-MoE."
        )
    attention_type = (
        Qwen3Attention if model.config.model_type == "qwen3" else Qwen3MoeAttention
    )
    attention_modules = [layer.self_attn for layer in model.model.layers]
    for module in attention_modules:
        if type(module) is not attention_type:
            raise UnsupportedCombinationError(
                "model.qkv_projection='fused' requires native Qwen3 attention."
            )
        if "forward" in module.__dict__ and (
            getattr(module.forward, "__func__", None) is not forward_fused_qkv_attention
        ):
            raise UnsupportedCombinationError(
                "model.qkv_projection='fused' cannot replace a custom "
                "attention forward."
            )
        _validate_projections(module)
    for module in attention_modules:
        module.forward = MethodType(forward_fused_qkv_attention, module)
        module._llmtuner_fused_qkv = True


def apply_fp32_rotary_projection(model: nn.Module) -> None:
    """Keep separate Q/K/V autograd branches while selecting FP32 RoPE."""
    if getattr(model.config, "qkv_projection", "separate") != "separate":
        raise ValueError("FP32 rotary installation requires separate Q/K/V")
    if getattr(model.config, "rotary_compute_dtype", None) != "float32":
        raise ValueError(
            "FP32 rotary installation requires rotary_compute_dtype=float32"
        )
    if model.config.model_type not in ("qwen3", "qwen3_moe"):
        raise UnsupportedCombinationError("FP32 RoPE requires Qwen3 or Qwen3-MoE")
    attention_type = (
        Qwen3Attention if model.config.model_type == "qwen3" else Qwen3MoeAttention
    )
    modules = [layer.self_attn for layer in model.model.layers]
    for module in modules:
        if type(module) is not attention_type or (
            "forward" in module.__dict__
            and getattr(module.forward, "__func__", None)
            is not forward_fused_qkv_attention
        ):
            raise UnsupportedCombinationError("FP32 RoPE requires native attention")
    for module in modules:
        module.forward = MethodType(forward_fused_qkv_attention, module)
