# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""Qwen operator adapters shared only by the explicit Qwen model subclasses."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5 import modeling_qwen3_5 as reference
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    causal_conv1d_fn,
    rotate_half,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    causal_conv1d_update as hf_causal_conv1d_update,
)

if TYPE_CHECKING:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


def initialize_packed_experts(
    self: torch.nn.Module, config: "Qwen3MoeConfig | Qwen3_5MoeTextConfig"
) -> None:
    """Register Qwen packed expert storage in the model initialization order."""
    torch.nn.Module.__init__(self)
    self.num_experts = config.num_experts
    self.hidden_dim = config.hidden_size
    self.intermediate_dim = config.moe_intermediate_size
    self.gate_up_proj = torch.nn.Parameter(
        torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
    )
    self.down_proj = torch.nn.Parameter(
        torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
    )
    self.act_fn = ACT2FN[config.hidden_act]


def resolve_attention_interface(implementation: str, *, eager: Callable) -> Callable:
    """Select Qwen's flash adapter or the caller-owned reference fallback."""
    if implementation == "flash_attention_2":
        return flash_attention_forward
    return ALL_ATTENTION_FUNCTIONS.get_interface(implementation, eager)


def eager_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply packed gate/up experts and accumulate scores in expert order."""
    output = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        active_experts = (mask.sum(dim=(-1, -2)) > 0).nonzero()
    for expert in active_experts:
        expert_idx = expert[0]
        slots, tokens = torch.where(mask[expert_idx])
        gate, up = F.linear(hidden_states[tokens], self.gate_up_proj[expert_idx]).chunk(
            2, dim=-1
        )
        values = F.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
        values = values * top_k_weights[tokens, slots, None]
        output.index_add_(0, tokens, values.to(output.dtype))
    return output


def rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch.nn.functional.rms_norm(hidden_states, (weight.numel(),), weight, eps)


def residual_rms_norm_forward(
    self: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Normalize with Qwen's residual weight scale and device precision policy."""
    use_eager = hidden_states.device.type != "npu"
    weight = 1.0 + (self.weight.float() if use_eager else self.weight)
    return rms_norm(hidden_states, weight, self.eps).to(hidden_states.dtype)


def gated_rms_norm_forward(
    self: torch.nn.Module, hidden_states: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Normalize and scale before applying the SiLU gate."""
    normalized = rms_norm(hidden_states, self.weight, self.variance_epsilon)
    return normalized * F.silu(gate)


def rotary_mul(query, cos, sin):
    return query * cos + rotate_half(query) * sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the configured prefix of Q/K and preserve non-rotary channels."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = rotary_mul(q_rot, cos, sin)
    k_embed = rotary_mul(k_rot, cos, sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def gated_attention_inputs(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: Cache | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Qwen3.5 Q/K/V and gate, apply RoPE, and update the KV cache."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states, gate = torch.chunk(
        module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(
        module.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, module.layer_idx
        )

    return query_states, key_states, value_states, gate


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute vision RoPE in FP32 before restoring Q/K dtypes."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = rotary_mul(q, cos, sin)
    k_embed = rotary_mul(k, cos, sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def vision_attention_forward(
    module: torch.nn.Module,
    resolve_attention: Callable[[str], Callable],
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    /,
    **kwargs: Any,
) -> torch.Tensor:
    """Attend within packed vision sequences using the model's backend resolver.

    Each sequence remains bidirectional and independent. Parallel execution
    is dispatched by the owning model before entering this local path.
    """
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        module.qkv(hidden_states)
        .reshape(seq_length, 3, module.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )

    attention_interface = resolve_attention(module.config._attn_implementation)

    if module.config._attn_implementation == "flash_attention_2":
        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)

        # Flash Attention 2: Use cu_seqlens for variable length attention
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output, _ = attention_interface(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=module.scaling,
            dropout=0.0 if not module.training else module.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            total_seq_len=seq_length,
            input_layout="1TND",
            **kwargs,
        )
    else:
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Other implementations: Process each chunk separately
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [
            torch.split(tensor, lengths, dim=2)
            for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            attention_interface(
                module,
                q,
                k,
                v,
                attention_mask=None,
                scaling=module.scaling,
                dropout=0.0 if not module.training else module.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits, strict=True)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = module.proj(attn_output)
    return attn_output


def flash_attention_forward(*args, **kwargs):
    return sdpa_attention_forward(*args, **kwargs)


def causal_conv1d(inputs, weight, bias=None, activation=None, **kwargs):
    if activation not in (None, "silu", "swish"):
        raise ValueError("Qwen causal convolution supports silu/swish or no activation")
    kwargs.pop("head_num", None)
    kwargs.pop("implementation", None)
    if kwargs.get("cu_seqlens") is not None:
        raise ValueError(
            "Packed convolution requires a backend with explicit document boundaries"
        )
    kwargs.pop("cu_seqlens", None)
    if causal_conv1d_fn is not None:
        # HF and causal-conv1d accept [B, channels, T] and [channels, kernel];
        # this family interface uses [B, T, channels] and [kernel, channels].
        output = causal_conv1d_fn(
            inputs.transpose(1, 2),
            weight.transpose(0, 1),
            bias,
            activation=activation,
            **kwargs,
        )
        return output.transpose(1, 2), None
    sequence_length = inputs.shape[1]
    output = torch.nn.functional.conv1d(
        inputs.transpose(1, 2),
        weight.transpose(0, 1).unsqueeze(1),
        bias,
        padding=weight.shape[0] - 1,
        groups=inputs.shape[-1],
    )
    if activation in ("silu", "swish"):
        output = torch.nn.functional.silu(output)
    return output.transpose(1, 2)[:, :sequence_length], None


def causal_conv1d_update(
    inputs, conv_state, weight, bias=None, activation=None, **kwargs
):
    # Validate before invoking a backend that updates conv_state in place.
    if activation not in (None, "silu", "swish"):
        raise ValueError("Qwen causal convolution supports silu/swish or no activation")
    kwargs.pop("implementation", None)
    if kwargs:
        raise ValueError(
            "Unsupported incremental convolution options: " + ", ".join(sorted(kwargs))
        )
    torch_update = getattr(reference, "torch_causal_conv1d_update", None)
    update = (
        hf_causal_conv1d_update if hf_causal_conv1d_update is not None else torch_update
    )
    if update is None:
        raise NotImplementedError(
            "Qwen recurrent convolution requires causal-conv1d; this Transformers "
            "version does not provide the torch update implementation"
        )
    output = update(inputs.transpose(1, 2), conv_state, weight, bias, activation)
    return output.transpose(1, 2), None


def chunk_gdn(*args, implementation="eager", **kwargs):
    if implementation != "eager":
        raise NotImplementedError(
            "Portable Qwen chunk GDN supports implementation='eager' only"
        )
    return torch_chunk_gated_delta_rule(*args, **kwargs)


def fused_recurrent_gdn(*args, implementation="eager", **kwargs):
    if implementation != "eager":
        raise NotImplementedError(
            "Portable Qwen recurrent GDN supports implementation='eager' only"
        )
    return torch_recurrent_gated_delta_rule(*args, **kwargs)


def gated_delta_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    cache_params: Cache | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the shared Qwen3.5 local GDN math after model-owned padding masking.

    Dense and MoE blocks provide the same projection and cache layout. The
    model's forward retains backend dispatch, including the separate CP path.
    """
    batch_size, sequence_length, _ = hidden_states.shape
    has_state = cache_params is not None and cache_params.has_previous_state(
        module.layer_idx, state_idx=0
    )
    decode = (
        has_state
        and sequence_length == 1
        and not cache_params.layers[module.layer_idx].record_past
    )
    recurrent_state = (
        cache_params.layers[module.layer_idx].recurrent_states[0] if has_state else None
    )
    mixed_qkv = module.in_proj_qkv(hidden_states)
    gate = module.in_proj_z(hidden_states).reshape(
        batch_size, sequence_length, -1, module.head_v_dim
    )
    beta = module.in_proj_b(hidden_states).sigmoid()
    decay = -module.A_log.float().exp() * F.softplus(
        module.in_proj_a(hidden_states).float() + module.dt_bias
    )
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(torch.int64)
    if decode:
        mixed_qkv, _ = causal_conv1d_update(
            mixed_qkv,
            cache_params.layers[module.layer_idx].conv_states[0],
            module.conv1d.weight.squeeze(1),
            module.conv1d.bias,
            activation=module.activation,
            implementation=module.causal_conv1d_implementation,
        )
    else:
        if cache_params is not None:
            mixed_qkv = cache_params.update_conv_state(
                mixed_qkv.transpose(1, 2),
                module.layer_idx,
                conv_kernel_size=module.conv_kernel_size,
            ).transpose(1, 2)
        mixed_qkv, _ = causal_conv1d(
            mixed_qkv,
            module.conv1d.weight.squeeze(1).transpose(0, 1),
            module.conv1d.bias,
            activation=module.activation,
            cu_seqlens=cu_seqlens,
            head_num=2 * module.num_k_heads + module.num_v_heads,
            implementation=module.causal_conv1d_implementation,
        )
        mixed_qkv = mixed_qkv[:, -sequence_length:]
    query, key, value = torch.split(
        mixed_qkv, [module.key_dim, module.key_dim, module.value_dim], dim=-1
    )
    query = query.reshape(
        batch_size, sequence_length, module.num_k_heads, module.head_k_dim
    )
    key = key.reshape(
        batch_size, sequence_length, module.num_k_heads, module.head_k_dim
    )
    value = value.reshape(
        batch_size, sequence_length, module.num_v_heads, module.head_v_dim
    )
    if module.num_v_heads != module.num_k_heads:
        query = query.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
        key = key.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
    if decode:
        output, state = fused_recurrent_gdn(
            query,
            key,
            value,
            decay,
            beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            implementation="eager",
        )
    else:
        output, state = chunk_gdn(
            query,
            key,
            value,
            decay,
            beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            implementation=module.gdn_implementation,
        )
    if cache_params is not None:
        cache_params.update_recurrent_state(state, module.layer_idx)
    output = module.norm(
        output.reshape(-1, module.head_v_dim), gate.reshape(-1, module.head_v_dim)
    )
    return module.out_proj(output.reshape(batch_size, sequence_length, -1))


def grouped_matmul(inputs, counts, weights, weights_layout="linear"):
    if weights_layout != "linear":
        raise ValueError("Explicit Qwen grouped GEMM expects linear weight layout")
    return torch._grouped_mm(
        inputs, weights.transpose(-2, -1), offs=counts.cumsum(0, dtype=torch.int32)
    )


def permute(tokens, indices, num_out_tokens=None):
    if num_out_tokens is not None and num_out_tokens != indices.numel():
        raise ValueError(
            "Portable Qwen permutation requires all routed rows; "
            "capacity truncation and padding are not implemented"
        )
    sorted_indices = torch.argsort(indices.reshape(-1), stable=True)
    top_k = indices.shape[-1]
    return tokens[sorted_indices // top_k], sorted_indices


def unpermute(
    permuted_tokens,
    sorted_indices,
    probs=None,
    restore_shape=None,
):
    if probs is None and restore_shape is None:
        raise ValueError("unpermute requires routing probabilities or restore_shape")
    if restore_shape is not None:
        output_shape = restore_shape
    else:
        output_shape = (probs.shape[0], permuted_tokens.shape[-1])
    output = permuted_tokens.new_zeros(output_shape)
    if probs is not None:
        top_k = probs.shape[-1]
    elif output_shape[0] == 0:
        if sorted_indices.numel():
            raise ValueError("Cannot restore nonempty routed rows to zero tokens")
        return output
    else:
        rows, remainder = divmod(sorted_indices.numel(), output_shape[0])
        if remainder or rows == 0:
            raise ValueError("Routed rows must contain the same top-k count per token")
        top_k = rows
    if probs is not None:
        # Rows are expert-sorted; scores must follow the same permutation.
        weights = probs.reshape(-1)[sorted_indices].unsqueeze(1)
        permuted_tokens = permuted_tokens * weights.to(permuted_tokens.dtype)
    token_indices = sorted_indices // top_k
    return output.index_add(0, token_indices, permuted_tokens)


def swiglu(inputs, dim=-1):
    gate, up = inputs.chunk(2, dim=dim)
    return torch.nn.functional.silu(gate) * up
