# Copyright 2026 The DeepSeek-AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Transformers 5.11 modeling_deepseek_v4.
"""DeepSeek V4 shared-KV attention and grouped low-rank output math."""

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as reference
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from .compression import DeepseekV4CSACompressor, DeepseekV4HCACompressor
from .normalization import DeepseekV4RMSNorm, DeepseekV4UnweightedRMSNorm
from .rotary import apply_rotary_pos_emb


def validate_output_geometry(config: DeepseekV4Config) -> None:
    """Reject output grouping that would truncate the attention head storage."""
    for name in ("num_attention_heads", "head_dim", "o_groups", "o_lora_rank"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"DeepSeek V4 {name} must be a positive integer")
    if config.num_attention_heads * config.head_dim % config.o_groups:
        raise ValueError("DeepSeek V4 attention width must be divisible by o_groups")


class DeepseekV4GroupedLinear(reference.DeepseekV4GroupedLinear):
    """Project head groups independently with native two-dimensional storage."""

    def __init__(
        self,
        in_features_per_group: int,
        out_features: int,
        n_groups: int,
        bias: bool = False,
    ) -> None:
        for name, value in (
            ("in_features_per_group", in_features_per_group),
            ("out_features", out_features),
            ("n_groups", n_groups),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"Grouped projection {name} must be a positive integer"
                )
        if out_features % n_groups:
            raise ValueError(
                "Grouped projection out_features must divide into n_groups"
            )
        nn.Linear.__init__(self, in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim < 2 or hidden_states.shape[-2:] != (
            self.n_groups,
            self.in_features,
        ):
            raise ValueError(
                "Grouped projection expects trailing dimensions "
                f"({self.n_groups}, {self.in_features}), "
                f"got {tuple(hidden_states.shape)}"
            )
        input_shape = hidden_states.shape[:-2]
        width = hidden_states.shape[-1]
        weight = self.weight.view(self.n_groups, -1, width).transpose(1, 2)
        values = hidden_states.reshape(-1, self.n_groups, width).transpose(0, 1)
        output = torch.bmm(values, weight).transpose(0, 1)
        return output.reshape(
            *input_shape, self.n_groups, self.out_features // self.n_groups
        )


class DeepseekV4Attention(reference.DeepseekV4Attention):
    """Own projections, partial rotary, cache and compressed-KV mask assembly."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        validate_output_geometry(config)
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.rope_layer_type = (
            "main" if self.layer_type == "sliding_attention" else "compress"
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.head_dim, bias=False
        )
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(
            config.o_groups * config.o_lora_rank, config.hidden_size, bias=False
        )
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        self.compressor = (
            {
                "heavily_compressed_attention": DeepseekV4HCACompressor,
                "compressed_sparse_attention": DeepseekV4CSACompressor,
            }[self.layer_type](config)
            if self.layer_type != "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cosine, sine = position_embeddings[self.rope_layer_type]
        query_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(query_residual).view(*hidden_shape).transpose(1, 2)
        query = apply_rotary_pos_emb(self.q_b_norm(query), cosine, sine)
        kv = (
            self.kv_norm(self.kv_proj(hidden_states))
            .view(*hidden_shape)
            .transpose(1, 2)
        )
        kv = apply_rotary_pos_emb(kv, cosine, sine)
        if past_key_values is not None:
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]
        block_bias = None
        if self.compressor is not None:
            compressed, block_bias = self.compressor(
                hidden_states,
                query_residual,
                position_ids,
                past_key_values,
                self.layer_idx,
            )
            kv = torch.cat([kv, compressed], dim=2)
        if (
            isinstance(attention_mask, torch.Tensor)
            and kv.shape[2] > attention_mask.shape[-1]
        ):
            # Compressed slots carry their own causal/indexer validity.
            attention_mask = (
                torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
                if block_bias is not None
                else F.pad(
                    attention_mask,
                    (0, kv.shape[2] - attention_mask.shape[-1]),
                    value=0.0,
                )
            )
        attention = reference.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, reference.eager_attention_forward
        )
        output, weights = attention(
            self,
            query,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )
        # Shared values carry RoPE too; undo it at the query position before
        # grouped output projection. Preserve the native layout transposes.
        output = apply_rotary_pos_emb(output.transpose(1, 2), cosine, -sine).transpose(
            1, 2
        )
        grouped = output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped), weights
