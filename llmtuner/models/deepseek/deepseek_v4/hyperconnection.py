# Copyright 2026 The DeepSeek-AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Transformers 5.11 modeling_deepseek_v4.
"""DeepSeek V4 residual stream mixing and final stream collapse."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as reference
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from .normalization import DeepseekV4UnweightedRMSNorm


class DeepseekV4HyperConnection(reference.DeepseekV4HyperConnection):
    """Map residual streams to expansion, Sinkhorn mixing and collapsed input."""

    def __init__(self, config: DeepseekV4Config) -> None:
        nn.Module.__init__(self)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(
        self, hidden_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        streams = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        widths = [streams, streams, streams * streams]
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split(widths, dim=-1)
        pre_b, post_b, comb_b = self.base.split(widths)
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        logits = comb_w.view(*comb_w.shape[:-1], streams, streams) * comb_scale
        logits = logits + comb_b.view(streams, streams)
        combination = torch.softmax(logits, dim=-1) + self.hc_eps
        # Preserve native column-first normalization and exact epsilon placement.
        combination = combination / (
            combination.sum(dim=-2, keepdim=True) + self.hc_eps
        )
        for _ in range(self.hc_sinkhorn_iters - 1):
            combination = combination / (
                combination.sum(dim=-1, keepdim=True) + self.hc_eps
            )
            combination = combination / (
                combination.sum(dim=-2, keepdim=True) + self.hc_eps
            )
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
        return post, combination, collapsed.to(hidden_streams.dtype)


class DeepseekV4HyperHead(reference.DeepseekV4HyperHead):
    """Collapse residual streams before the model's final normalization."""

    def __init__(self, config: DeepseekV4Config) -> None:
        nn.Module.__init__(self)
        self.hc_mult = config.hc_mult
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * config.hidden_size)
        )
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(hidden_streams.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = (
            torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
            + self.eps
        )
        return (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
