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
from copy import deepcopy
from typing import Any

import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from transformers.activations import SiLUActivation

from ....errors import UnsupportedCombinationError
from ...common.activation import ActivationFn, SituGLU, SwiGLU
from ...common.moe.experts import GroupedExperts, RoutedExperts
from .configuration import KimiLinearConfig
from .mlp import KimiExpertMLP as KimiExpertMLP
from .mlp import KimiMLP
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

    def __init__(self, config: KimiLinearConfig) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.validate_routing()

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

    def validate_routing(self) -> None:
        """Validate live selection geometry before allocation or EP conversion."""
        if self.moe_router_activation_func not in ("sigmoid", "softmax"):
            raise ValueError(
                "Kimi moe_router_activation_func must be sigmoid or softmax"
            )
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

    def validate_storage(self) -> None:
        """Check that live router tensors match the declared selection geometry."""
        self.validate_routing()
        if tuple(self.weight.shape) != (self.num_experts, self.gating_dim):
            raise ValueError(
                "Kimi router weight must match expert count and input width"
            )
        if tuple(self.e_score_correction_bias.shape) != (self.num_experts,):
            raise ValueError(
                "Kimi router correction bias must have one value per expert"
            )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token-wise expert IDs and differentiable FP32 route weights."""
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.reshape(-1, h)
        # Explicit casts alone do not prevent autocast from lowering linear.
        # Keep routing precision consistent with the native EP router.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
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


class KimiSparseMoE(nn.Module):
    """Per-expert reference math, including latent and shared expert paths."""

    def __init__(self, config: KimiLinearConfig) -> None:
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
            shared_config = deepcopy(config)
            shared_config.intermediate_size = (
                config.moe_intermediate_size * config.num_shared_experts
            )
            self.shared_experts = KimiMLP(shared_config)

    def validate_expert_geometry(self) -> None:
        """Validate the live router and three-projection expert layout."""
        self.gate.validate_storage()
        count = len(self.experts)
        if count != self.config.num_experts or self.gate.num_experts != count:
            raise ValueError("Kimi EP router and expert count must agree")
        if not count:
            raise ValueError("Kimi EP requires at least one expert")
        first = self.experts[0].w1
        width = first.in_features
        down = getattr(self, "routed_expert_down_proj", None)
        up = getattr(self, "routed_expert_up_proj", None)
        if (down is None) != (up is None):
            raise ValueError("Kimi EP latent projections must be paired")
        if down is None:
            if width != self.gate.gating_dim:
                raise ValueError("Kimi EP expert input must match router input width")
        elif tuple(down.weight.shape) != (width, self.gate.gating_dim) or tuple(
            up.weight.shape
        ) != (self.gate.gating_dim, width):
            raise ValueError("Kimi EP latent projection geometry is inconsistent")
        norm = getattr(self, "routed_expert_norm", None)
        if norm is not None and tuple(norm.weight.shape) != (width,):
            raise ValueError("Kimi EP latent norm must match expert width")
        expected_shapes = {
            "w1": (first.out_features, first.in_features),
            "w2": (first.in_features, first.out_features),
            "w3": (first.out_features, first.in_features),
        }
        for name, expected in expected_shapes.items():
            if any(
                tuple(getattr(expert, name).weight.shape) != expected
                for expert in self.experts
            ):
                raise ValueError(
                    f"Kimi EP inconsistent expert projection geometry for {name}"
                )

    def validate_grouped_activation(self) -> None:
        """Check that one grouped activation preserves every live expert's math."""
        first_expert = self.experts[0] if self.experts else None
        if first_expert is None:
            raise ValueError("Kimi EP requires at least one expert")
        for expert in self.experts:
            if expert.situ:
                valid = type(expert.act_fn) is SituGLU
            else:
                valid = type(expert.act_fn) is SiLUActivation
            if expert.situ and valid:
                if not math.isfinite(expert.act_fn.beta) or expert.act_fn.beta <= 0:
                    raise ValueError("SituGLU beta must be finite and positive")
                linear_beta = expert.act_fn.linear_beta
                if linear_beta is not None and (
                    not math.isfinite(linear_beta) or linear_beta < 0
                ):
                    raise ValueError(
                        "SituGLU linear_beta must be finite and nonnegative"
                    )
            if not valid:
                raise UnsupportedCombinationError(
                    "Kimi EP expert activation is not supported"
                )
            if (
                expert.situ != first_expert.situ
                or type(expert.act_fn) is not type(first_expert.act_fn)
                or (
                    expert.situ
                    and (expert.act_fn.beta, expert.act_fn.linear_beta)
                    != (first_expert.act_fn.beta, first_expert.act_fn.linear_beta)
                )
            ):
                raise UnsupportedCombinationError(
                    "Kimi EP requires uniform expert activations"
                )

    def build_grouped_activation(self) -> ActivationFn:
        """Construct the shared gated activation from live expert semantics."""
        self.validate_grouped_activation()
        first = self.experts[0]
        if first.situ:
            return SituGLU(first.act_fn.beta, first.act_fn.linear_beta)
        return SwiGLU()

    def expert_parallel_math_signature(self) -> tuple[object, ...]:
        """Snapshot routing, activation and latent norm math for plan validation.

        Tensor identity/layout is owned by ParallelPlan. Mutable scalar math
        needs its own snapshot even when the original modules are reused.
        """
        gate = self.gate
        return (
            gate.top_k,
            gate.moe_router_activation_func,
            gate.moe_renormalize,
            gate.routed_scaling_factor,
            gate.num_expert_group,
            gate.topk_group,
            tuple(
                (
                    expert.situ,
                    type(expert.act_fn),
                    expert.act_fn.beta if expert.situ else None,
                    expert.act_fn.linear_beta if expert.situ else None,
                )
                for expert in self.experts
            ),
            (
                self.routed_expert_norm.variance_epsilon
                if hasattr(self, "routed_expert_norm")
                else None
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ids, scores = self.gate(hidden_states)
        inputs = hidden_states.reshape(-1, hidden_states.shape[-1])
        if hasattr(self, "routed_expert_down_proj"):
            inputs = self.routed_expert_down_proj(inputs)
        compute_dtype = torch.promote_types(inputs.dtype, scores.dtype)
        output = torch.zeros_like(inputs, dtype=compute_dtype)
        for index, expert in enumerate(self.experts):
            tokens, slots = (ids == index).nonzero(as_tuple=True)
            values = expert(inputs.index_select(0, tokens))
            # Keep weighted outputs and accumulation in routing precision;
            # casting each expert separately introduces BF16 rounding errors.
            values = values.to(compute_dtype) * scores[tokens, slots, None]
            output = output.index_add(0, tokens, values)
        output = output.to(inputs.dtype)
        if hasattr(self, "routed_expert_norm"):
            output = self.routed_expert_norm(output)
        if hasattr(self, "routed_expert_up_proj"):
            output = self.routed_expert_up_proj(output)
        output = output.reshape_as(hidden_states)
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output


class KimiRoutedExperts(RoutedExperts):
    """Route in latent space, then normalize/project the combined result.

    The gate and shared expert still consume the original model-width input.
    Latent normalization belongs after weighted combine, never inside experts.
    """

    def __init__(
        self, experts: GroupedExperts, dispatcher: Any, source: KimiSparseMoE
    ) -> None:
        super().__init__(experts, dispatcher)
        for name in (
            "routed_expert_down_proj",
            "routed_expert_norm",
            "routed_expert_up_proj",
        ):
            module = getattr(source, name, None)
            if module is not None:
                setattr(self, name, module)

    def forward(
        self,
        x_TD: torch.Tensor,
        scores_TK: torch.Tensor,
        ids_TK: torch.Tensor,
        counts_E: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self, "routed_expert_down_proj"):
            x_TD = self.routed_expert_down_proj(x_TD)
        output = super().forward(x_TD, scores_TK, ids_TK, counts_E)
        if hasattr(self, "routed_expert_norm"):
            output = self.routed_expert_norm(output)
        if hasattr(self, "routed_expert_up_proj"):
            output = self.routed_expert_up_proj(output)
        return output
