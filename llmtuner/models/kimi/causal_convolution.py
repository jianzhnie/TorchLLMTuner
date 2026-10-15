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
"""Depthwise causal convolution for Kimi KDA training.

Parameter layout matches FSDPTurbo ShortConvolution (channels, 1, kernel).
The ATen path preserves document boundaries without optional kernel imports.
KCP applies its state-exchange kernel directly to these same parameters.
"""

import torch
from torch import nn
from torch.nn import functional as F


class KimiCausalConvolution(nn.Conv1d):
    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        super().__init__(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            bias=False,
            padding=kernel_size - 1,
        )
        self.activation = "silu"

    def forward(
        self, x: torch.Tensor, *, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.in_channels:
            raise ValueError("Kimi convolution requires [batch, sequence, channels]")
        if cu_seqlens is None:
            if x.shape[1] == 0:
                # Keep both parameter and input gradient paths for empty input.
                return x + self.weight.sum() * 0
            output = F.conv1d(
                x.transpose(1, 2),
                self.weight,
                self.bias,
                padding=self.padding,
                groups=self.groups,
            )[..., : x.shape[1]].transpose(1, 2)
        else:
            if x.shape[0] != 1 or cu_seqlens.ndim != 1:
                raise ValueError(
                    "Packed Kimi convolution requires batch=1 and 1D boundaries"
                )
            if cu_seqlens.dtype not in (torch.int32, torch.int64):
                raise ValueError("Packed boundaries must use an integer dtype")
            boundaries = cu_seqlens.tolist()
            if (
                len(boundaries) < 2
                or boundaries[0] != 0
                or boundaries[-1] != x.shape[1]
                or any(
                    b <= a for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
                )
            ):
                raise ValueError(
                    "Packed boundaries must increase from zero to sequence length"
                )
            # Each document starts with zero convolution history.
            return torch.cat(
                [
                    self.forward(x[:, start:end])
                    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
                ],
                dim=1,
            )
        return F.silu(output)
