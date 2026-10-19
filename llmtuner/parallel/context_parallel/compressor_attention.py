# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""Ulysses attention over sliding KV and rank-owned HCA/CSA entries.

The caller projects local queries/sliding KV and supplies compressed windows.
Outputs follow the HF [batch, local_sequence, heads, dim] layout.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn import functional as F

from ...accelerator.collectives import synchronize_local_errors
from .attention import HeadToSeq, SeqToHead
from .compression_windows import CompressionWindowExchange


def compressor_ulysses_attention(
    query,
    sliding_kv,
    compressed_kv,
    *,
    sinks,
    compress_rate,
    sliding_window,
    scaling,
    group,
    dropout=0.0,
    training=True,
    selected_windows=None,
):
    """Attend with a per-head sink and globally indexed compressed causality.

    Query is BHSD; both KV inputs are shared single-head BHSD. Complete Compressor
    windows are owned by their final token's rank, with unequal counts allowed.
    Padding, CSA indexing and KV cache require separate model-level contracts.
    """
    dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    with synchronize_local_errors("Compressor attention input validation", group=group):
        if query.dtype not in dtypes:
            raise ValueError(
                "Compressor attention requires floating point query tensors"
            )
        for value in (sliding_kv, compressed_kv):
            if value.dtype != query.dtype or value.device != query.device:
                raise ValueError("Compressor query and KV must share dtype and device")
        if sinks.device != query.device:
            raise ValueError("Compressor sinks must be on the query device")
        batch, heads, length, width = query.shape
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        if heads % size or length == 0:
            raise ValueError(
                "Compressor Ulysses requires divisible heads and nonempty shards"
            )
        if (compress_rate is not None and compress_rate <= 0) or sliding_window <= 0:
            raise ValueError(
                "Compressor compression rate and sliding window must be positive"
            )
        first = rank * length // compress_rate if compress_rate is not None else 0
        last = (rank + 1) * length // compress_rate if compress_rate is not None else 0
        if sliding_kv.shape != (batch, 1, length, width):
            raise ValueError("Compressor sliding KV must be shared [B, 1, local_S, D]")
        if compressed_kv.shape != (batch, 1, last - first, width):
            raise ValueError(
                "Compressor compressed KV does not match global window ownership"
            )
        if sinks.shape != (heads,):
            raise ValueError("Compressor sinks must contain one value per query head")
        if selected_windows is not None:
            total = length * size // compress_rate if compress_rate is not None else 0
            if (
                selected_windows.ndim != 3
                or selected_windows.shape[:2] != (batch, length)
                or selected_windows.dtype != torch.long
                or selected_windows.device != query.device
                or torch.any(selected_windows < -1)
                or torch.any(selected_windows >= total)
            ):
                raise ValueError(
                    "CSA selection must contain local global-window indices"
                )
    metadata = torch.tensor(
        [
            batch,
            heads,
            length,
            width,
            0 if compress_rate is None else compress_rate,
            sliding_window,
            dtypes.index(query.dtype),
            -1 if selected_windows is None else selected_windows.shape[-1],
        ],
        device=query.device,
        dtype=torch.long,
    )
    peers = [torch.empty_like(metadata) for _ in range(size)]
    dist.all_gather(peers, metadata, group=group)
    if any(not torch.equal(peer, metadata) for peer in peers):
        raise ValueError(
            "Compressor attention requires matching CP shapes and window settings"
        )
    local_heads = heads // size
    query = SeqToHead.apply(query, group)
    sliding = SeqToHead.apply(sliding_kv.expand(-1, heads, -1, -1).contiguous(), group)
    # Every peer receives this owner's entries for its query heads. Shared KV is
    # transmitted once per destination, with backward summing all head consumers.
    owned = compressed_kv[:, 0].transpose(0, 1).contiguous()
    send = owned.repeat(size, 1, 1)
    counts = [
        ((peer + 1) * length // compress_rate - peer * length // compress_rate)
        if compress_rate is not None
        else 0
        for peer in range(size)
    ]
    received = CompressionWindowExchange.apply(
        send, [last - first] * size, counts, group
    )
    compressed = received.transpose(0, 1).unsqueeze(1)
    full_length = length * size
    positions = torch.arange(full_length, device=query.device)
    sliding_allowed = (positions[None, :] <= positions[:, None]) & (
        positions[None, :] > positions[:, None] - sliding_window
    )
    indices = torch.arange(compressed.shape[2], device=query.device)
    compressed_allowed = (
        indices[None, :] < (positions[:, None] + 1) // compress_rate
        if compress_rate is not None
        else torch.zeros(full_length, 0, device=query.device, dtype=torch.bool)
    )
    if selected_windows is not None:
        # Selection is discrete metadata, shared across heads but query-sharded.
        selections = [torch.empty_like(selected_windows) for _ in range(size)]
        dist.all_gather(selections, selected_windows.contiguous(), group=group)
        selected = torch.cat(selections, dim=1)
        visible = torch.zeros(
            batch,
            full_length,
            compressed.shape[2] + 1,
            device=query.device,
            dtype=torch.bool,
        )
        safe = torch.where(selected >= 0, selected, compressed.shape[2])
        visible.scatter_(-1, safe, True)
        compressed_allowed = compressed_allowed[None] & visible[..., :-1]
    else:
        compressed_allowed = compressed_allowed[None].expand(batch, -1, -1)
    allowed = torch.cat(
        (sliding_allowed[None].expand(batch, -1, -1), compressed_allowed), dim=-1
    )
    kv = torch.cat((sliding, compressed.expand(-1, local_heads, -1, -1)), dim=2)
    logits = (query @ kv.transpose(-1, -2)) * scaling
    logits = logits.masked_fill(~allowed[:, None], float("-inf"))
    sink = sinks[rank * local_heads : (rank + 1) * local_heads]
    sink = sink[None, :, None, None].expand(batch, -1, full_length, 1)
    # Sink contributes to the softmax denominator but has zero value.
    logits = torch.cat((logits, sink), dim=-1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    weights = logits.softmax(dim=-1)[..., :-1]
    weights = F.dropout(weights, p=dropout, training=training)
    output = weights.to(kv.dtype) @ kv
    return HeadToSeq.apply(output, group).transpose(1, 2).contiguous()


def compressor_attention_forward(group):
    """Build an instance forward for stateless DeepSeek V4 compressor attention.

    Projection/rotary/output math follows Transformers 5.9.0
    modeling_deepseek_v4.py (Apache-2.0). This factory does not install itself;
    the model integration must enforce contiguous inputs and supported layers.
    """
    from .compression_windows import (
        compress_csa_windows,
        compress_hca_windows,
        select_csa_windows,
    )

    def forward(
        module,
        hidden_states,
        position_embeddings,
        position_ids,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            apply_rotary_pos_emb,
        )

        with synchronize_local_errors(
            "Compressor module input validation", group=group
        ):
            if module.layer_type not in (
                "heavily_compressed_attention",
                "compressed_sparse_attention",
                "sliding_attention",
            ):
                raise ValueError("Compressor CP forward requires HCA or CSA layers")
            if past_key_values is not None or attention_mask is not None:
                raise ValueError(
                    "Compressor CP constructs its own masks and rejects KV cache"
                )
            batch, length, _ = hidden_states.shape
            expected = torch.arange(length, device=hidden_states.device)
            expected = expected + dist.get_rank(group) * length
            if position_ids.shape != (batch, length) or not torch.equal(
                position_ids, expected[None].expand(batch, -1)
            ):
                raise ValueError(
                    "Compressor CP requires contiguous global position_ids"
                )
        shape = (batch, length, -1, module.head_dim)
        cos, sin = position_embeddings[module.rope_layer_type]
        q_residual = module.q_a_norm(module.q_a_proj(hidden_states))
        query = module.q_b_norm(module.q_b_proj(q_residual).view(shape).transpose(1, 2))
        query = apply_rotary_pos_emb(query, cos, sin)
        kv = module.kv_norm(module.kv_proj(hidden_states)).view(shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)
        selected = None
        if module.layer_type == "compressed_sparse_attention":
            compressed, _ = compress_csa_windows(
                module.compressor, hidden_states, group=group
            )
            selected = select_csa_windows(
                module.compressor.indexer,
                hidden_states,
                q_residual,
                position_ids,
                group=group,
            )
        elif module.layer_type == "heavily_compressed_attention":
            compressed, _ = compress_hca_windows(
                module.compressor, hidden_states, group=group
            )
        else:
            compressed = kv[:, :, :0]
        output = compressor_ulysses_attention(
            query,
            kv,
            compressed,
            sinks=module.sinks,
            compress_rate=(
                module.compressor.compress_rate
                if module.compressor is not None
                else None
            ),
            sliding_window=module.sliding_window,
            scaling=module.scaling,
            group=group,
            dropout=module.attention_dropout if module.training else 0.0,
            training=module.training,
            selected_windows=selected,
        )
        output = apply_rotary_pos_emb(output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = output.reshape(batch, length, module.config.o_groups, -1)
        return module.o_b_proj(module.o_a_proj(grouped).flatten(2)), None

    return forward


def compressor_decoder_forward(
    module,
    input_ids=None,
    inputs_embeds=None,
    position_ids=None,
    attention_mask=None,
    use_cache=False,
    past_key_values=None,
    **kwargs,
):
    """Stateless V4 decoder for an installed compressor CP attention stack.

    Adapted from Transformers 5.9.0 DeepseekV4Model.forward (Apache-2.0).
    Each attention owns global causal masks; the decoder never creates a cache.
    The installer must validate layer types and coordinate input failures.
    """
    from transformers.modeling_outputs import MoeModelOutputWithPast

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


def apply_compressor_context_parallel(model, *, cp_mesh, cfg):
    """Install stateless HCA/CSA CP after validating the complete local decoder."""
    from types import MethodType

    decoder = model.model.model
    layers = list(decoder.layers)
    if cfg.cp_config.context_parallel_strategy != "ulysses":
        raise ValueError(
            "DeepSeek V4 Compressor CP currently requires strategy='ulysses'"
        )
    if cfg.tp_size != 1 or cfg.pp_size != 1:
        raise ValueError(
            "DeepSeek V4 Compressor CP TP/PP composition is not implemented"
        )
    if cfg.cp_config.context_parallel_load_balancer is not None:
        raise ValueError(
            "Compressor CP requires contiguous shards with no load balancer"
        )
    if getattr(decoder.config, "attn_mask_type", "causal") != "causal":
        raise ValueError("Compressor CP does not support packed document attention")
    if not layers or any(
        getattr(layer.self_attn, "layer_type", None)
        not in (
            "heavily_compressed_attention",
            "compressed_sparse_attention",
            "sliding_attention",
        )
        for layer in layers
    ):
        raise ValueError("Compressor CP encountered an unsupported attention layer")
    for layer in layers:
        attention = layer.self_attn
        if attention.num_heads % cp_mesh.size():
            raise ValueError(
                "Compressor CP requires query heads divisible by the CP degree"
            )
        if attention.attention_dropout != 0:
            raise ValueError(
                "Compressor CP dropout random-stream equivalence is not implemented"
            )
    group = cp_mesh.get_group()
    for layer in layers:
        attention = layer.self_attn
        attention.forward = MethodType(compressor_attention_forward(group), attention)
    decoder.forward = MethodType(compressor_decoder_forward, decoder)
    model.set_cp_mesh(cp_mesh, strategy="compressor_ulysses")
    return model
