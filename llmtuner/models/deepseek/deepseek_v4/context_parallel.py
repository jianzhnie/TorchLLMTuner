# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 model mathematics in global context-parallel coordinates."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    eager_attention_forward,
)

from .rotary import apply_rotary_pos_emb


def build_raw_attention_mask(
    full_attention_mask: torch.Tensor | None,
    global_position_ids: torch.Tensor,
    query_indices: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    sliding_window: int | None,
) -> torch.Tensor:
    """Build a causal mask while preserving packed-document boundaries."""
    global_length = global_position_ids.shape[-1]
    query_length = query_indices.numel()
    key_indices = torch.arange(global_length, device=global_position_ids.device)
    distance = query_indices.view(1, -1, 1) - key_indices.view(1, 1, -1)
    allowed = distance >= 0
    if sliding_window is not None:
        allowed = allowed & (distance < sliding_window)

    if full_attention_mask is None:
        key_positions = global_position_ids.expand(batch_size, -1)
        first_dummy_value = key_positions[:, :1] - 1
        position_diff = torch.diff(key_positions, prepend=first_dummy_value, dim=-1)
        document_ids = (position_diff != 1).cumsum(-1)
        query_document_ids = document_ids.index_select(-1, query_indices)
        allowed = allowed & (
            query_document_ids.unsqueeze(-1) == document_ids.unsqueeze(-2)
        ).unsqueeze(1)
    else:
        allowed = allowed & full_attention_mask.to(torch.bool).view(
            batch_size, 1, 1, global_length
        )

    raw_mask = torch.zeros(
        (batch_size, 1, query_length, global_length),
        dtype=dtype,
        device=global_position_ids.device,
    )
    return raw_mask.masked_fill(~allowed, torch.finfo(dtype).min)


def deepseek_v4_cp_attention_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: object,
    query_indices: torch.Tensor,
    cp_state: dict[str, object],
    group: object,
    gather_tensor,
    kwargs: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute DeepSeek V4 attention with CP-prepared compressor inputs."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = position_embeddings[module.rope_layer_type]
    q_residual = module.q_a_norm(module.q_a_proj(hidden_states))
    query = module.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
    query = module.q_b_norm(query)
    query = apply_rotary_pos_emb(query, cos, sin)
    key_value = module.kv_norm(module.kv_proj(hidden_states)).view(*hidden_shape)
    key_value = key_value.transpose(1, 2)
    key_value = apply_rotary_pos_emb(key_value, cos, sin)

    global_position_ids = cp_state["position_ids"]
    compressed_kv = None
    block_bias = None
    if module.compressor is not None:
        global_hidden_states = gather_tensor(hidden_states, group, 1)
        global_q_residual = gather_tensor(q_residual, group, 1)
        compressed_kv, block_bias = module.compressor(
            global_hidden_states,
            global_q_residual,
            global_position_ids,
            None,
            module.layer_idx,
        )
    kwargs["_llmtuner_prepared_compressed_kv"] = compressed_kv
    attention_mask = build_raw_attention_mask(
        cp_state["attention_mask"],
        global_position_ids,
        query_indices,
        hidden_states.shape[0],
        query.dtype,
        module.sliding_window,
    )
    if compressed_kv is not None:
        if block_bias is None:
            attention_mask = F.pad(
                attention_mask, (0, compressed_kv.shape[2]), value=0.0
            )
        else:
            if query_indices.numel() != global_position_ids.shape[-1]:
                rank_start = int(query_indices[0])
                block_bias = block_bias.narrow(-2, rank_start, query_indices.numel())
            attention_mask = torch.cat(
                [attention_mask, block_bias.to(attention_mask.dtype)], dim=-1
            )
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation,
        eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        module,
        query,
        key_value,
        key_value,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        sliding_window=module.sliding_window,
        s_aux=module.sinks,
        **kwargs,
    )
    attn_output = apply_rotary_pos_emb(
        attn_output.transpose(1, 2), cos, -sin
    ).transpose(1, 2)
    grouped = attn_output.reshape(*input_shape, module.config.o_groups, -1)
    grouped = module.o_a_proj(grouped).flatten(2)
    return module.o_b_proj(grouped), attn_weights


def compressor_attention_math(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    position_ids: torch.Tensor,
    *,
    compress_csa_windows: Callable[..., Any],
    select_csa_windows: Callable[..., Any],
    compress_hca_windows: Callable[..., Any],
    attention: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, None]:
    """Project, compress and merge stateless V4 CP attention in model order.

    The parallel adapter validates inputs and binds group-aware window/attention
    callbacks. This function neither selects communication groups nor installs
    forwards. Adapted from Transformers 5.9.0 (Apache-2.0).
    """
    batch, length, _ = hidden_states.shape
    shape = (batch, length, -1, module.head_dim)
    cos, sin = position_embeddings[module.rope_layer_type]
    q_residual = module.q_a_norm(module.q_a_proj(hidden_states))
    query = module.q_b_norm(module.q_b_proj(q_residual).view(shape).transpose(1, 2))
    query = apply_rotary_pos_emb(query, cos, sin)
    kv = module.kv_norm(module.kv_proj(hidden_states)).view(shape).transpose(1, 2)
    kv = apply_rotary_pos_emb(kv, cos, sin)
    selected = None
    if module.layer_type == "compressed_sparse_attention":
        compressed, _ = compress_csa_windows(module.compressor, hidden_states)
        selected = select_csa_windows(
            module.compressor.indexer,
            hidden_states,
            q_residual,
            position_ids,
        )
    elif module.layer_type == "heavily_compressed_attention":
        compressed, _ = compress_hca_windows(module.compressor, hidden_states)
    else:
        compressed = kv[:, :, :0]
    output = attention(
        query,
        kv,
        compressed,
        sinks=module.sinks,
        compress_rate=(
            module.compressor.compress_rate if module.compressor is not None else None
        ),
        sliding_window=module.sliding_window,
        scaling=module.scaling,
        dropout=module.attention_dropout if module.training else 0.0,
        training=module.training,
        selected_windows=selected,
    )
    output = apply_rotary_pos_emb(output.transpose(1, 2), cos, -sin).transpose(1, 2)
    grouped = output.reshape(batch, length, module.config.o_groups, -1)
    return module.o_b_proj(module.o_a_proj(grouped).flatten(2)), None


def compressor_decoder_forward(
    module: torch.nn.Module,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    use_cache: bool = False,
    past_key_values: Any = None,
    **kwargs: Any,
) -> MoeModelOutputWithPast:
    """Stateless V4 decoder for an installed compressor CP attention stack.

    Adapted from Transformers 5.9.0 DeepseekV4Model.forward (Apache-2.0).
    Each attention owns global causal masks; the decoder never creates a cache.
    The installer must validate layer types and coordinate input failures.
    """
    if use_cache or past_key_values is not None or attention_mask is not None:
        raise ValueError(
            "Compressor CP decoder requires stateless internally masked training"
        )
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Provide exactly one of input_ids or inputs_embeds")
    if position_ids is None:
        raise ValueError("Compressor CP decoder requires global position_ids")
    hidden = module.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    position_embeddings = {
        name: module.rotary_emb(hidden, position_ids=position_ids, layer_type=name)
        for name in ("main", "compress")
    }
    hidden = hidden.unsqueeze(2).expand(-1, -1, module.config.hc_mult, -1).contiguous()
    for layer in module.layers:
        hidden = layer(
            hidden,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=None,
            input_ids=input_ids,
            past_key_values=None,
            **kwargs,
        )
    hidden = module.norm(module.hc_head(hidden))
    return MoeModelOutputWithPast(last_hidden_state=hidden, past_key_values=None)
