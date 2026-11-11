# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Math/layout reference: Transformers 5.11.0 modeling_qwen3_moe.
"""Experiment-owned Qwen3-MoE math; no training-framework dependencies."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = hidden.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * normalized.to(hidden.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        parameters = config.rope_parameters
        if parameters["rope_type"] != "default":
            raise ValueError("Custom experiment implements default RoPE only")
        dim = config.head_dim
        if dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        inverse = 1.0 / (
            parameters["rope_theta"] ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inverse, persistent=False)

    @torch.no_grad()
    def forward(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Match reference arithmetic: evaluate frequencies in FP32, then cast
        # cos/sin to activation dtype. Module.to controls buffer storage dtype.
        inverse = self.inv_freq[None, :, None].float().expand(positions.shape[0], -1, 1)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            frequencies = (inverse @ positions[:, None, :].float()).transpose(1, 2)
            angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)


def rotate_half(hidden: torch.Tensor) -> torch.Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self, hidden_states, position_embeddings, attention_mask=None, **kwargs
    ):
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = (tensor.unsqueeze(1) for tensor in position_embeddings)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        dropout = self.attention_dropout if self.training else 0.0
        if self.config._attn_implementation == "eager":
            weights = query @ key.transpose(-2, -1) * self.scaling
            if attention_mask is None:
                causal = torch.ones(
                    query.shape[-2],
                    key.shape[-2],
                    dtype=torch.bool,
                    device=query.device,
                ).tril()
                weights = weights.masked_fill(~causal, torch.finfo(query.dtype).min)
            else:
                weights = weights + attention_mask
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attended = F.dropout(weights, p=dropout, training=self.training) @ value
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout,
                is_causal=attention_mask is None,
                scale=self.scaling,
            )
        attended = (
            attended.transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1)
        )
        return self.o_proj(attended), None


class DenseMLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Router(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden.reshape(-1, self.hidden_dim), self.weight)
        scores, indices = logits.float().softmax(-1).topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(-1, keepdim=True)
        return logits, scores.to(logits.dtype), indices


class Experts(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = nn.SiLU()

    def forward(
        self, hidden: torch.Tensor, indices: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        with torch.no_grad():
            mask = F.one_hot(indices, num_classes=self.num_experts).permute(2, 1, 0)
            hits = mask.sum((-1, -2)).gt(0).nonzero().flatten()
        for expert in hits:
            slot, token = torch.where(mask[expert])
            gate, up = F.linear(hidden[token], self.gate_up_proj[expert]).chunk(2, -1)
            result = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
            result = result * scores[token, slot, None]
            output.index_add_(0, token, result.to(output.dtype))
        return output


class SparseMoE(nn.Module):
    router_logits_in_output = False
    expert_layout = "fused_gate_up"

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.experts = Experts(config)
        self.gate = Router(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        _, scores, indices = self.gate(flat)
        return self.experts(flat, indices, scores).reshape(shape)


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, index: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, index)
        sparse = (
            index not in config.mlp_only_layers
            and config.num_experts > 0
            and (index + 1) % config.decoder_sparse_step == 0
        )
        self.mlp = SparseMoE(config) if sparse else DenseMLP(config)
        self.moe_block_name = "mlp" if sparse else None
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        attended, _ = self.self_attn(
            self.input_layernorm(hidden), position_embeddings=position_embeddings
        )
        hidden = hidden + attended
        return hidden + self.mlp(self.post_attention_layernorm(hidden))
