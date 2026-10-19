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
"""Kimi Delta Attention training projections and recurrent CPU reference.

Adapted from FSDPTurbo 0af4eca models/kimi/modeling_kimi_linear.py.
Accelerator execution uses optional production KDA kernels, loaded on demand.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .causal_convolution import KimiCausalConvolution
from .kimi_linear_config import KimiLinearConfig
from .normalization import KimiGatedRMSNorm


def recurrent_kda(q, k, v, g, beta, A_log, dt_bias, lower_bound=None):
    """CPU training reference; full heads, zero initial state, no cache."""
    dtype = q.dtype
    q, k = [
        (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6))
        .to(dtype)
        .float()
        for x in (q, k)
    ]
    raw = g.float() + dt_bias.float().view(g.shape[-2:])
    rate = A_log.float().exp().view(-1, 1)
    decay = (
        lower_bound * torch.sigmoid(rate * raw)
        if lower_bound is not None
        else -rate * F.softplus(raw)
    ).exp()
    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1])
    outputs = []
    for t in range(q.shape[1]):
        state = state * decay[:, t].unsqueeze(-1)
        key = k[:, t]
        residual = v[:, t].float() - (key.unsqueeze(-1) * state).sum(-2)
        update = residual * beta[:, t].float().sigmoid().unsqueeze(-1)
        state = state + key.unsqueeze(-1) * update.unsqueeze(-2)
        outputs.append((q[:, t].unsqueeze(-1) * state).sum(-2) * q.shape[-1] ** -0.5)
    return torch.stack(outputs, dim=1).to(dtype)


class KimiDeltaAttention(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        settings = config.linear_attn_config
        self.head_dim = self.head_k_dim = settings["head_dim"]
        self.num_heads = settings["num_heads"]
        width = self.head_dim * self.num_heads
        for name in ("q", "k", "v"):
            setattr(
                self, f"{name}_proj", nn.Linear(config.hidden_size, width, bias=False)
            )
            setattr(
                self,
                f"{name}_conv1d",
                KimiCausalConvolution(width, settings["short_conv_kernel_size"]),
            )
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, width, bias=False)
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.num_heads).uniform_(1, 16).log())
        self.dt_bias = nn.Parameter(torch.zeros(width))
        self.use_full_rank_gate = settings.get("use_full_rank_gate", False)
        self.gate_lower_bound = settings.get("gate_lower_bound")
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(config.hidden_size, width, bias=False)
        else:
            self.g_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, width, bias=False)
        self.o_norm = KimiGatedRMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(width, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, cache_params=None, **kwargs):
        if cache_params is not None:
            raise ValueError("Kimi KDA training does not accept inference cache")
        if hidden_states.ndim != 3:
            raise ValueError("Kimi KDA requires [batch, sequence, hidden] input")
        batch, length, _ = hidden_states.shape
        boundaries = kwargs.get("cu_seqlens")
        if boundaries is not None:
            if (
                batch != 1
                or boundaries.ndim != 1
                or boundaries.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    "Packed KDA requires batch=1 and integer 1D boundaries"
                )
            offsets = boundaries.tolist()
            if (
                len(offsets) < 2
                or offsets[0] != 0
                or offsets[-1] != length
                or any(b <= a for a, b in zip(offsets[:-1], offsets[1:], strict=True))
            ):
                raise ValueError("Packed KDA boundaries must increase from 0 to length")
            # Both the convolution history and delta state reset per document.
            return torch.cat(
                [
                    self.forward(
                        hidden_states[:, start:end],
                        attention_mask=(
                            None
                            if attention_mask is None
                            else attention_mask[:, start:end]
                        ),
                    )
                    for start, end in zip(offsets[:-1], offsets[1:], strict=True)
                ],
                dim=1,
            )
        if attention_mask is not None:
            if (
                attention_mask.shape != hidden_states.shape[:2]
                or attention_mask.device != hidden_states.device
                or attention_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "KDA padding mask must be boolean [batch, sequence] "
                    "on the input device, with True for valid tokens"
                )
            rows = []
            for row in range(batch):
                indices = attention_mask[row].nonzero().flatten()
                valid = hidden_states[row : row + 1].index_select(1, indices)
                values = self.forward(valid)
                rows.append(
                    torch.zeros_like(hidden_states[row : row + 1]).index_copy(
                        1, indices, values
                    )
                )
            return torch.cat(rows, dim=0)
        if length == 0:
            # No kernel launch for an empty row; retain all gradient paths.
            zero = sum(parameter.sum() * 0 for parameter in self.parameters())
            return hidden_states + zero

        def project(name):
            return getattr(self, f"{name}_conv1d")(
                getattr(self, f"{name}_proj")(hidden_states)
            ).view(batch, length, self.num_heads, self.head_dim)

        q, k, v = (project(name) for name in ("q", "k", "v"))
        g = self.f_b_proj(self.f_a_proj(hidden_states)).view_as(q)
        beta = self.b_proj(hidden_states).float()
        if hidden_states.device.type == "cpu":
            output = recurrent_kda(
                q, k, v, g, beta, self.A_log, self.dt_bias, self.gate_lower_bound
            )
        else:
            if hidden_states.device.type == "npu":
                from triton_ascend_kernels.attention.fla.kda import chunk_kda
            else:
                from fla.ops.kda import chunk_kda
            output, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta.sigmoid(),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=False,
                state_v_first=False,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
            )
        gate = (
            self.g_proj(hidden_states)
            if self.use_full_rank_gate
            else self.g_b_proj(self.g_a_proj(hidden_states))
        ).view_as(output)
        output = self.o_norm(output, gate)
        return self.o_proj(output.reshape(batch, length, -1))
