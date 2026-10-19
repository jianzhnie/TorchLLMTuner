# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and
# HuggingFace Inc. team. All rights reserved.
#
# The multi-head latent attention, MoE gating and sparse MoE block in this file are
# adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py). They have been
# extensively modified and extended for the Kimi-Linear architecture.
#
# Licensing Information:
# - Code adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py)
#   is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Trainable Kimi expert routing, retaining upstream expert checkpoint names.

Adapted from FSDPTurbo 0af4eca models/kimi/modeling_kimi_linear.py.
The local reference is differentiable; EP conversion is a separate migration.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

from ..common.activation import SituGLU
from .kimi_linear_config import KimiLinearConfig
from .normalization import KimiRMSNorm


class KimiMoEGate(nn.Module):
    """
    MoEGate adapted from Deepseek-V3.
    Parameter correspondences:
        num_experts -> n_routed_experts
        num_experts_per_token -> num_experts_per_tok
        num_expert_group -> n_group
        moe_router_activation_func -> scoring_func
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        for name, value in (
            ("num_experts", self.num_experts),
            ("num_experts_per_token", self.top_k),
            ("num_expert_group", self.num_expert_group),
            ("topk_group", self.topk_group),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Kimi {name} must be a positive integer")
        if (
            self.num_experts % self.num_expert_group
            or not 1 <= self.topk_group <= self.num_expert_group
        ):
            raise ValueError(
                "Kimi expert groups must divide num_experts and contain topk_group"
            )
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("Kimi num_experts_per_token must be in [1, num_experts]")
        if self.num_expert_group > self.topk_group:
            per_group = self.num_experts // self.num_expert_group
            if per_group < 2 or self.top_k > per_group * self.topk_group:
                raise ValueError(
                    "Kimi retained groups must contain top_k and two experts per group"
                )

        # topk selection algorithm
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim)),
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.reshape(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(
                f"Unsupported MoE scoring function: {self.moe_router_activation_func}",
            )

        # select top-k experts
        scores = scores.reshape(bsz * seq_len, self.num_experts)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            group_scores = (
                scores_for_choice.reshape(
                    bsz * seq_len,
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )  # [n, num_expert_group]
            group_idx = torch.topk(
                group_scores,
                k=self.topk_group,
                dim=-1,
                sorted=False,
            )[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, num_expert_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, num_expert_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len,
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .reshape(bsz * seq_len, self.num_experts)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(
                ~score_mask.bool(), float("-inf")
            )  # [n, e]
        else:
            tmp_scores = scores_for_choice
        _, topk_idx = torch.topk(
            tmp_scores,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )
        topk_weight = scores.gather(1, topk_idx)

        # norm gate to sum 1
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        # must multiply the scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


class KimiExpertMLP(nn.Module):
    def __init__(self, config, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.situ = config.hidden_act == "situ"
        self.act_fn = (
            SituGLU(
                (
                    1.0
                    if config.activation_situ_beta is None
                    else config.activation_situ_beta
                ),
                config.activation_situ_linear_beta,
            )
            if self.situ
            else ACT2FN[config.hidden_act]
        )

    def forward(self, x):
        gate, up = self.w1(x), self.w3(x)
        return self.w2(self.act_fn(gate, up) if self.situ else self.act_fn(gate) * up)


class KimiSparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = KimiMoEGate(config)
        width = (
            config.hidden_size
            if config.routed_expert_hidden_size is None
            else config.routed_expert_hidden_size
        )
        self.experts = nn.ModuleList(
            [
                KimiExpertMLP(config, width, config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        if config.routed_expert_hidden_size is not None:
            self.routed_expert_down_proj = nn.Linear(
                config.hidden_size, width, bias=False
            )
            self.routed_expert_up_proj = nn.Linear(
                width, config.hidden_size, bias=False
            )
            if config.latent_moe_use_norm:
                self.routed_expert_norm = KimiRMSNorm(width, config.rms_norm_eps)
        self.shared_experts = None
        if config.num_shared_experts:
            from copy import deepcopy

            from .model import KimiMLP

            shared_config = deepcopy(config)
            shared_config.intermediate_size = (
                config.moe_intermediate_size * config.num_shared_experts
            )
            self.shared_experts = KimiMLP(shared_config)

    def forward(self, hidden_states):
        ids, scores = self.gate(hidden_states)
        inputs = hidden_states.reshape(-1, hidden_states.shape[-1])
        if hasattr(self, "routed_expert_down_proj"):
            inputs = self.routed_expert_down_proj(inputs)
        output = torch.zeros_like(inputs)
        for index, expert in enumerate(self.experts):
            tokens, slots = (ids == index).nonzero(as_tuple=True)
            values = expert(inputs.index_select(0, tokens))
            values = values * scores[tokens, slots, None].to(values.dtype)
            output = output.index_add(0, tokens, values)
        if hasattr(self, "routed_expert_norm"):
            output = self.routed_expert_norm(output)
        if hasattr(self, "routed_expert_up_proj"):
            output = self.routed_expert_up_proj(output)
        output = output.reshape_as(hidden_states)
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output
