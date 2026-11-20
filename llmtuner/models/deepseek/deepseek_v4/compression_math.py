# Copyright 2026 The DeepSeek-AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""Shared CSA compressor and indexer mathematics with injected communication."""

import torch

from .overlap import assemble_overlap_windows
from .rotary import apply_rotary_pos_emb


def compress_overlap(
    self, hidden_states: torch.Tensor, *, cache, cache_key: str
) -> torch.Tensor:
    """Compress CSA windows using the caller's independent cache namespace.

    The indexer and attention compressor own separate projections and caches,
    but share overlap assembly, normalization and rotary position handling.
    Incomplete windows stay in the cache until enough tokens arrive.
    """
    batch = hidden_states.shape[0]
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    ratio = self.compress_rate
    if cache is None:
        usable = (kv.shape[1] // ratio) * ratio
        chunk_kv, chunk_gate, first_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_position = cache.store_compression_weights(
            cache_key, kv, gate
        )
    if chunk_kv.shape[1] > 0:
        windows = chunk_kv.shape[1] // ratio
        chunk_kv = chunk_kv.view(batch, windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, windows, ratio, -1) + self.position_bias
        combined_kv, combined_gate = assemble_overlap_windows(
            chunk_kv, chunk_gate, self.head_dim
        )
        if cache is not None:
            prior_kv, prior_gate = cache.update_overlap_state(
                cache_key, chunk_kv, chunk_gate, self.head_dim
            )
            if prior_kv is not None:
                combined_kv[:, 0, :ratio] = prior_kv.to(combined_kv.dtype)
                combined_gate[:, 0, :ratio] = prior_gate.to(combined_gate.dtype)
        weights = combined_gate.softmax(dim=2, dtype=torch.float32).to(
            combined_kv.dtype
        )
        compressed = self.kv_norm((combined_kv * weights).sum(dim=2))
        positions = torch.arange(windows, device=compressed.device)
        positions = (positions * ratio + first_position).unsqueeze(0).expand(batch, -1)
        cosine, sine = self.rotary_emb(
            compressed, position_ids=positions, layer_type=self.rope_layer_type
        )
        compressed = apply_rotary_pos_emb(
            compressed.unsqueeze(1), cosine, sine
        ).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
    if cache is not None:
        compressed = cache.update_compressor_states(cache_key, compressed)
    return compressed


def compress_overlap_context_parallel(
    self, hidden_states, *, exchange_windows, exchange_previous_window
):
    """Compress owned CSA windows with explicit communication callbacks."""
    projected = torch.cat(
        (self.kv_proj(hidden_states), self.gate_proj(hidden_states)), dim=-1
    )
    windows, first = exchange_windows(projected, window_size=self.compress_rate)
    kv, gate = windows.chunk(2, dim=-1)
    gate = gate + self.position_bias.to(gate.dtype)
    batch, count, ratio, _ = kv.shape
    width = self.head_dim
    received = exchange_previous_window(
        torch.cat((kv[:, -1:, :, :width], gate[:, -1:, :, :width]), dim=-1),
        first=first,
        count=count,
        window_size=ratio,
        local_length=hidden_states.shape[1],
    )
    last = first + count
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
        self.kv_norm(
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
    cos, sin = self.rotary_emb(
        entries, position_ids=positions, layer_type=self.rope_layer_type
    )
    return apply_rotary_pos_emb(entries.unsqueeze(1), cos, sin), first
