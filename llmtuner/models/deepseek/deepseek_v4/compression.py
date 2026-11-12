# Copyright 2026 The DeepSeek-AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Transformers 5.11 modeling_deepseek_v4.
"""DeepSeek V4 window compression and compressed-slot validity."""

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as reference
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from .indexer import DeepseekV4Indexer
from .normalization import DeepseekV4RMSNorm
from .overlap import assemble_overlap_windows
from .rotary import DeepseekV4RotaryEmbedding, apply_rotary_pos_emb


class DeepseekV4CSACompressor(reference.DeepseekV4CSACompressor):
    """Compress overlapping windows and mask slots selected by the indexer."""

    def __init__(self, config: DeepseekV4Config) -> None:
        nn.Module.__init__(self)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(
            torch.empty(self.compress_rate, 2 * self.head_dim)
        )
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden_states.shape
        cache = (
            past_key_values.layers[layer_idx] if past_key_values is not None else None
        )
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        ratio = self.compress_rate
        if cache is None:
            usable = (kv.shape[1] // ratio) * ratio
            chunk_kv, chunk_gate, first_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_position = cache.store_compression_weights(
                "compressor", kv, gate
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
                    "compressor", chunk_kv, chunk_gate, self.head_dim
                )
                if prior_kv is not None:
                    combined_kv[:, 0, :ratio] = prior_kv.to(combined_kv.dtype)
                    combined_gate[:, 0, :ratio] = prior_gate.to(combined_gate.dtype)
            weights = combined_gate.softmax(dim=2, dtype=torch.float32).to(
                combined_kv.dtype
            )
            compressed = self.kv_norm((combined_kv * weights).sum(dim=2))
            positions = torch.arange(windows, device=compressed.device)
            positions = (
                (positions * ratio + first_position).unsqueeze(0).expand(batch, -1)
            )
            cosine, sine = self.rotary_emb(
                compressed, position_ids=positions, layer_type=self.rope_layer_type
            )
            compressed = apply_rotary_pos_emb(
                compressed.unsqueeze(1), cosine, sine
            ).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
        if cache is not None:
            compressed = cache.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)
        selected = self.indexer(
            hidden_states, q_residual, position_ids, past_key_values, layer_idx
        )
        entries = compressed_kv.shape[2]
        valid = selected >= 0
        # Route invalid picks into one extra slot, then remove it from the mask.
        safe_indices = torch.where(valid, selected, torch.full_like(selected, entries))
        bias = compressed_kv.new_full((batch, 1, sequence, entries + 1), float("-inf"))
        bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        return compressed_kv, bias[..., :entries]


class DeepseekV4HCACompressor(reference.DeepseekV4HCACompressor):
    """Compress complete windows and preserve caller-owned streaming history."""

    def __init__(self, config: DeepseekV4Config) -> None:
        nn.Module.__init__(self)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(
            torch.empty(self.compress_rate, self.head_dim)
        )
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch = hidden_states.shape[0]
        cache = (
            past_key_values.layers[layer_idx] if past_key_values is not None else None
        )
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_position = cache.store_compression_weights(
                "compressor", kv, gate
            )
        if chunk_kv.shape[1] > 0:
            windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, windows, self.compress_rate, -1)
            chunk_gate = chunk_gate + self.position_bias
            weights = chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)
            compressed = self.kv_norm((chunk_kv * weights).sum(dim=2))
            positions = torch.arange(windows, device=compressed.device)
            positions = (positions * self.compress_rate + first_position).unsqueeze(0)
            positions = positions.expand(batch, -1)
            cosine, sine = self.rotary_emb(
                compressed, position_ids=positions, layer_type=self.rope_layer_type
            )
            compressed = apply_rotary_pos_emb(
                compressed.unsqueeze(1), cosine, sine
            ).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
        if cache is not None:
            compressed = cache.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)
        entries = compressed_kv.shape[2]
        sequence = position_ids.shape[1]
        if sequence == 1 or entries == 0:
            return compressed_kv, None
        entry_ids = torch.arange(entries, device=compressed_kv.device)
        threshold = (position_ids + 1) // self.compress_rate
        bias = compressed_kv.new_zeros((batch, 1, sequence, entries))
        bias = bias.masked_fill(
            entry_ids.view(1, 1, 1, -1) >= threshold.unsqueeze(1).unsqueeze(-1),
            float("-inf"),
        )
        return compressed_kv, bias
