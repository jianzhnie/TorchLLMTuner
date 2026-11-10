"""DeepSeek V4 attention masks in global context-parallel coordinates."""

import torch


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
    import torch.nn.functional as F
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

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
