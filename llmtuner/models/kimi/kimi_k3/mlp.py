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
"""Dense, shared and routed Kimi feed-forward components.

Projection names preserve the dense and expert checkpoint layouts.
"""

from torch import nn
from transformers.activations import ACT2FN

from ...common.activation import SituGLU


def build_kimi_activation(config) -> nn.Module:
    """Construct a fresh activation for a dense or individual-expert MLP."""
    if config.hidden_act != "situ":
        return ACT2FN[config.hidden_act]
    beta = 1.0 if config.activation_situ_beta is None else config.activation_situ_beta
    return SituGLU(beta, config.activation_situ_linear_beta)


class KimiMLP(nn.Module):
    def __init__(self, config):
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
        self.situ = config.hidden_act == "situ"
        self.act_fn = build_kimi_activation(config)

    def forward(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(
            self.act_fn(gate, up) if self.situ else self.act_fn(gate) * up
        )


class KimiExpertMLP(nn.Module):
    def __init__(self, config, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.situ = config.hidden_act == "situ"
        self.act_fn = build_kimi_activation(config)

    def forward(self, x):
        gate, up = self.w1(x), self.w3(x)
        return self.w2(self.act_fn(gate, up) if self.situ else self.act_fn(gate) * up)
