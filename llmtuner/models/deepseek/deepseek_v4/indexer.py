# Copyright 2026 The DeepSeek-AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Transformers 5.11 modeling_deepseek_v4.
"""DeepSeek V4 compressed-key index scoring and causal top-k selection."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as reference
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from .normalization import DeepseekV4RMSNorm
from .overlap import assemble_overlap_windows
from .rotary import DeepseekV4RotaryEmbedding, apply_rotary_pos_emb


class DeepseekV4IndexerScorer(reference.DeepseekV4IndexerScorer):
    """Score compressed keys with FP32 dot products and learned head weights."""

    def __init__(self, config: DeepseekV4Config) -> None:
        nn.Module.__init__(self)
        self.softmax_scale = config.index_head_dim**-0.5
        self.weights_scaling = config.index_n_heads**-0.5
        self.weights_proj = nn.Linear(
            config.hidden_size, config.index_n_heads, bias=False
        )

    def forward(
        self,
        query: torch.Tensor,
        compressed_kv: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(
            query.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1)
        )
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling
        return (scores * weights.unsqueeze(-1)).sum(dim=2)


class DeepseekV4Indexer(reference.DeepseekV4Indexer):
    """Compress overlapping windows and select causally valid key blocks."""

    def __init__(self, config: DeepseekV4Config) -> None:
        nn.Module.__init__(self)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(
            torch.empty(self.compress_rate, 2 * self.head_dim)
        )
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.head_dim, bias=False
        )
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.scorer = DeepseekV4IndexerScorer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
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
                "indexer", kv, gate
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
                    "indexer", chunk_kv, chunk_gate, self.head_dim
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
        compressed_kv = (
            compressed
            if cache is None
            else cache.update_compressor_states("indexer", compressed)
        )
        cosine, sine = self.rotary_emb(
            hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type
        )
        query = self.q_b_proj(q_residual).view(batch, sequence, -1, self.head_dim)
        query = apply_rotary_pos_emb(query.transpose(1, 2), cosine, sine).transpose(
            1, 2
        )
        scores = self.scorer(query, compressed_kv, hidden_states)
        entries = compressed_kv.shape[1]
        top_k = min(self.index_topk, entries)
        if entries > 0:
            threshold = (position_ids + 1) // ratio
            entry_ids = torch.arange(entries, device=scores.device)
            future = entry_ids.view(1, 1, -1) >= threshold.unsqueeze(-1)
            scores = scores.masked_fill(future, float("-inf"))
            selected = scores.topk(top_k, dim=-1).indices
            # Early queries may have fewer valid blocks than the requested k.
            return torch.where(
                selected >= threshold.unsqueeze(-1),
                torch.full_like(selected, -1),
                selected,
            )
        return scores.topk(top_k, dim=-1).indices
