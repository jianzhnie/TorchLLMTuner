# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""Redistribute contiguous CP token shards into complete compression windows.

Window ownership follows the rank containing its final token. This preserves
cross-shard windows without replicating the full input on every CP rank.
The caller still owns compressor math, global window positions and attention.
"""

from __future__ import annotations

from functools import partial

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


def exchange_previous_window(tail, *, first, count, window_size, local_length, group):
    """Exchange the preceding complete window, skipping empty owners."""
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    total = local_length * size // window_size
    last = first + count
    destination = ((last + 1) * window_size - 1) // local_length
    send_sizes = [0] * size
    receive_sizes = [0] * size
    if count and last < total and destination != rank:
        send_sizes[destination] = 1
    else:
        tail = tail[:, :0]
    if count and first > 0:
        source = (first * window_size - 1) // local_length
        if source != rank:
            receive_sizes[source] = 1
    return CompressionWindowExchange.apply(
        tail.transpose(0, 1).contiguous(), send_sizes, receive_sizes, group
    ).transpose(0, 1)


def compress_hca_windows(module, hidden_states, *, group):
    """Bind complete-window communication to model-owned HCA math."""
    return module.compress_context_parallel(
        hidden_states,
        exchange_windows=partial(exchange_compression_windows, group=group),
    )


def compress_csa_windows(module, hidden_states, *, group):
    """Bind window and halo communication to model-owned CSA math."""
    return module.compress_context_parallel(
        hidden_states,
        exchange_windows=partial(exchange_compression_windows, group=group),
        exchange_previous_window=partial(exchange_previous_window, group=group),
    )


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
    return module.select_context_parallel_windows(
        hidden_states, q_residual, position_ids, keys
    )
