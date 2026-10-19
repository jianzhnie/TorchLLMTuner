# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""Redistribute contiguous CP token shards into complete compression windows.

Window ownership follows the rank containing its final token. This preserves
cross-shard windows without replicating the full input on every CP rank.
The caller still owns compressor math, global window positions and attention.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ...accelerator.collectives import synchronize_local_errors


class CompressionWindowExchange(torch.autograd.Function):
    """Variable-size token exchange whose backward restores source ownership."""

    @staticmethod
    def forward(ctx, tokens, send_sizes, receive_sizes, group):
        ctx.send_sizes = send_sizes
        ctx.receive_sizes = receive_sizes
        ctx.group = group
        output = tokens.new_empty((sum(receive_sizes), *tokens.shape[1:]))
        dist.all_to_all_single(
            output, tokens.contiguous(), receive_sizes, send_sizes, group=group
        )
        return output

    @staticmethod
    def backward(ctx, gradient):
        output = gradient.new_empty((sum(ctx.send_sizes), *gradient.shape[1:]))
        dist.all_to_all_single(
            output,
            gradient.contiguous(),
            ctx.send_sizes,
            ctx.receive_sizes,
            group=ctx.group,
        )
        return output, None, None, None


def exchange_compression_windows(
    local_tokens: torch.Tensor, *, window_size: int, group: dist.ProcessGroup
) -> tuple[torch.Tensor, int]:
    """Return ``[batch, owned_windows, window_size, width]`` and first window ID.

    Inputs must be equal-length contiguous ``[batch, local_sequence, width]``
    shards in CP group rank order. The final incomplete global window is omitted.
    A rank can own zero windows even when it contributes tokens to a peer.
    """
    dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    with synchronize_local_errors("Compression window input validation", group=group):
        if local_tokens.dtype not in dtypes:
            raise ValueError("Compression windows require floating point tokens")
        if local_tokens.ndim != 3 or local_tokens.shape[1] == 0:
            raise ValueError("Compression windows require nonempty [B, S, D] shards")
        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or window_size <= 0
        ):
            raise ValueError("Compression window_size must be a positive integer")
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    batch, length, width = local_tokens.shape
    # Agree on shapes before choosing variable all-to-all split sizes. Mismatched
    # sizes would otherwise fail inside communication with asymmetric errors.
    metadata = torch.tensor(
        [batch, length, width, window_size, dtypes.index(local_tokens.dtype)],
        device=local_tokens.device,
        dtype=torch.long,
    )
    peers = [torch.empty_like(metadata) for _ in range(size)]
    dist.all_gather(peers, metadata, group=group)
    if any(not torch.equal(peer, metadata) for peer in peers):
        raise ValueError("Compression windows require equal shard shapes/window_size")

    first = rank * length // window_size
    last = (rank + 1) * length // window_size
    own_start, own_end = first * window_size, last * window_size
    source_start, source_end = rank * length, (rank + 1) * length
    send_sizes, receive_sizes, pieces = [], [], []
    for peer in range(size):
        destination_start = (peer * length // window_size) * window_size
        destination_end = ((peer + 1) * length // window_size) * window_size
        start = max(source_start, destination_start)
        end = min(source_end, destination_end)
        count = max(0, end - start)
        send_sizes.append(count)
        if count:
            pieces.append(local_tokens[:, start - source_start : end - source_start])
        receive_sizes.append(
            max(0, min(own_end, (peer + 1) * length) - max(own_start, peer * length))
        )
    # Retain an autograd edge even for a source containing only discarded tail.
    packed = torch.cat(pieces, dim=1) if pieces else local_tokens[:, :0]
    received = CompressionWindowExchange.apply(
        packed.transpose(0, 1).contiguous(), send_sizes, receive_sizes, group
    )
    windows = received.transpose(0, 1).reshape(batch, last - first, window_size, width)
    return windows, first


def compress_hca_windows(module, hidden_states, *, group):
    """Compute rank-owned HCA entries with global rotary positions.

    Adapted from Transformers 5.9.0 modeling_deepseek_v4.py,
    DeepseekV4HCACompressor.forward (Apache-2.0).
    Matches Transformers DeepSeek V4 stateless HCA compression. The attention
    adapter must exchange these entries and construct bias from global window
    IDs; this function does not provide KV cache or CSA overlap semantics.
    """
    # Project on the source owner, then move only the features needed by each
    # complete window. Parameter gradients remain local until DP/CP reduction.
    projected = torch.cat(
        (module.kv_proj(hidden_states), module.gate_proj(hidden_states)), dim=-1
    )
    windows, first = exchange_compression_windows(
        projected, window_size=module.compress_rate, group=group
    )
    kv, gate = windows.chunk(2, dim=-1)
    gate = gate + module.position_bias.to(gate.dtype)
    entries = module.kv_norm(
        (kv * gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)).sum(dim=2)
    )
    positions = (
        torch.arange(
            first, first + entries.shape[1], device=entries.device, dtype=torch.long
        )
        * module.compress_rate
    )
    positions = positions.unsqueeze(0).expand(entries.shape[0], -1)
    cos, sin = module.rotary_emb(
        entries, position_ids=positions, layer_type=module.rope_layer_type
    )
    # Keep the model's exact partial-RoPE convention. Import only when this
    # model-specific path is called, not when generic CP helpers are imported.
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        apply_rotary_pos_emb,
    )

    return apply_rotary_pos_emb(entries.unsqueeze(1), cos, sin), first


def compress_csa_windows(module, hidden_states, *, group):
    """Compute CSA entries with the preceding window's Ca across CP owners.

    Adapted from Transformers 5.9.0 DeepseekV4CSACompressor.forward.
    Returns rank-owned entries and global first ID; indexer selection is separate.
    """
    projected = torch.cat(
        (module.kv_proj(hidden_states), module.gate_proj(hidden_states)), dim=-1
    )
    windows, first = exchange_compression_windows(
        projected, window_size=module.compress_rate, group=group
    )
    kv, gate = windows.chunk(2, dim=-1)
    gate = gate + module.position_bias.to(gate.dtype)
    batch, count, ratio, _ = kv.shape
    width = module.head_dim
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    length = hidden_states.shape[1]
    total = length * size // ratio
    last = first + count
    # Only the final Ca window may be consumed on a different owner. Empty
    # owners can be skipped because destination follows the final-token rule.
    destination = (last + 1) * ratio - 1
    destination = destination // length
    send_sizes = [0] * size
    receive_sizes = [0] * size
    send = torch.cat((kv[:, -1:, :, :width], gate[:, -1:, :, :width]), dim=-1)
    if count and last < total and destination != rank:
        send_sizes[destination] = 1
    else:
        send = send[:, :0]
    if count and first > 0:
        source = (first * ratio - 1) // length
        if source != rank:
            receive_sizes[source] = 1
    received = CompressionWindowExchange.apply(
        send.transpose(0, 1).contiguous(), send_sizes, receive_sizes, group
    ).transpose(0, 1)
    initial_kv = kv.new_zeros((batch, 1, ratio, width))
    initial_gate = gate.new_full((batch, 1, ratio, width), float("-inf"))
    if received.shape[1]:
        initial_kv, initial_gate = received.chunk(2, dim=-1)
    # Preserve the collective's backward participation for every rank, including
    # ranks with no entries and no halo contribution.
    zero = received.sum() * 0
    previous_kv = torch.cat((initial_kv, kv[:, :-1, :, :width]), dim=1)[:, :count]
    previous_gate = torch.cat((initial_gate, gate[:, :-1, :, :width]), dim=1)[:, :count]
    combined_kv = torch.cat((previous_kv, kv[..., width:]), dim=2)
    combined_gate = torch.cat((previous_gate, gate[..., width:]), dim=2)
    entries = (
        module.kv_norm(
            (
                combined_kv
                * combined_gate.softmax(dim=2, dtype=torch.float32).to(
                    combined_kv.dtype
                )
            ).sum(dim=2)
        )
        + zero
    )
    positions = torch.arange(first, last, device=entries.device) * ratio
    positions = positions[None].expand(batch, -1)
    cos, sin = module.rotary_emb(
        entries, position_ids=positions, layer_type=module.rope_layer_type
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        apply_rotary_pos_emb,
    )

    return apply_rotary_pos_emb(entries.unsqueeze(1), cos, sin), first


@torch.no_grad()
def select_csa_windows(module, hidden_states, q_residual, position_ids, *, group):
    """Run the discrete CSA indexer against all globally compressed keys.

    Follows Transformers 5.9.0 DeepseekV4Indexer.forward. Selection indices have
    no gradient, matching HF topk indices; compression for attention stays live.
    """
    with synchronize_local_errors("CSA indexer input validation", group=group):
        batch, length, _ = hidden_states.shape
        expected = torch.arange(length, device=hidden_states.device)
        expected = expected + dist.get_rank(group) * length
        if position_ids.shape != (batch, length) or not torch.equal(
            position_ids, expected[None].expand(batch, -1)
        ):
            raise ValueError("CSA indexer requires contiguous global position_ids")
        if q_residual.shape != (batch, length, module.q_b_proj.in_features):
            raise ValueError("CSA indexer q_residual shape does not match local input")
    entries, _ = compress_csa_windows(module, hidden_states, group=group)
    batch, _, _, width = entries.shape
    length = hidden_states.shape[1]
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    first = rank * length // module.compress_rate
    last = (rank + 1) * length // module.compress_rate
    counts = [
        (peer + 1) * length // module.compress_rate
        - peer * length // module.compress_rate
        for peer in range(size)
    ]
    # Variable-size gather through all-to-all handles zero-entry owners without
    # padding fabricated keys into the score/topk axis.
    send = entries[:, 0].transpose(0, 1).contiguous().repeat(size, 1, 1)
    received = entries.new_empty((sum(counts), batch, width))
    dist.all_to_all_single(received, send, counts, [last - first] * size, group=group)
    keys = received.transpose(0, 1)
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        apply_rotary_pos_emb,
    )

    cos, sin = module.rotary_emb(
        hidden_states, position_ids=position_ids, layer_type=module.rope_layer_type
    )
    query = module.q_b_proj(q_residual).view(batch, length, -1, module.head_dim)
    query = apply_rotary_pos_emb(query.transpose(1, 2), cos, sin).transpose(1, 2)
    scores = torch.matmul(query.float(), keys.transpose(-1, -2).float().unsqueeze(1))
    scores = scores.relu() * module.softmax_scale
    weights = module.weights_proj(hidden_states).float() * module.weights_scaling
    scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
    total = keys.shape[1]
    threshold = (position_ids + 1) // module.compress_rate
    indices = torch.arange(total, device=scores.device)
    scores = scores.masked_fill(indices[None, None] >= threshold[..., None], -torch.inf)
    selected = scores.topk(min(module.index_topk, total), dim=-1).indices
    return torch.where(selected >= threshold[..., None], -1, selected)
