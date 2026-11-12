# Copyright 2026 The DeepSeek-AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Transformers 5.11 modeling_deepseek_v4.
"""DeepSeek V4 learned and token-table routing semantics."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as reference
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config


def validate_routing_config(config: DeepseekV4Config) -> None:
    """Reject invalid expert selection before router state is allocated."""
    for name in ("num_local_experts", "num_experts_per_tok"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"DeepSeek V4 {name} must be a positive integer")
    if config.num_experts_per_tok > config.num_local_experts:
        raise ValueError("DeepSeek V4 top-k cannot exceed num_local_experts")
    if config.scoring_func not in ACT2FN:
        raise ValueError(
            f"DeepSeek V4 unsupported scoring_func: {config.scoring_func!r}"
        )


class DeepseekV4TopKRouter(reference.DeepseekV4TopKRouter):
    """Select with correction bias, but weight using uncorrected scores."""

    def __init__(self, config: DeepseekV4Config) -> None:
        validate_routing_config(config)
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.num_experts), persistent=True
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = torch.topk(
            scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False
        ).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(reference.DeepseekV4HashRouter):
    """Select from a persistent token table while retaining learned scores."""

    def __init__(self, config: DeepseekV4Config) -> None:
        validate_routing_config(config)
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer(
            "tid2eid",
            torch.zeros(config.vocab_size, self.top_k, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices
