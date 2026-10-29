"""Instance-local Qwen3 fused projection with separate parameter ownership."""

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
)

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


def forward_fused_qkv_attention(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Keep Qwen3 attention semantics around a stateless fused projection."""
    query, key, value = fused_qkv_projection(module, hidden_states)
    query = module.q_norm(query).transpose(1, 2)
    key = module.k_norm(key).transpose(1, 2)
    value = value.transpose(1, 2)
    cosine, sine = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cosine, sine)
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
    if model.config.model_type != "qwen3":
        raise UnsupportedCombinationError(
            "model.qkv_projection='fused' currently requires the Qwen3 dense model."
        )
    attention_modules = [layer.self_attn for layer in model.model.layers]
    for module in attention_modules:
        if type(module) is not Qwen3Attention:
            raise UnsupportedCombinationError(
                "model.qkv_projection='fused' requires native Qwen3Attention instances."
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
