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
"""Kimi text decoder integration for the HF training wrapper.

Adapted from FSDPTurbo 0af4eca models/kimi/modeling_kimi_linear.py.
Dense training is implemented here; sparse experts and attention residuals
require their own integration before the corresponding configs can run.
"""

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .delta_attention import KimiDeltaAttention
from .kimi_linear_config import KimiLinearConfig
from .latent_attention import KimiLatentAttention
from .normalization import KimiRMSNorm


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
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class KimiDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_linear_attn = config.is_kda_layer(layer_idx)
        self.self_attn = (
            KimiDeltaAttention if self.is_linear_attn else KimiLatentAttention
        )(config, layer_idx)
        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            from .moe import KimiSparseMoE

            self.block_sparse_moe = KimiSparseMoE(config)
        else:
            self.mlp = KimiMLP(config)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), attention_mask=attention_mask, **kwargs
        )
        feed_forward = getattr(self, "block_sparse_moe", None)
        if feed_forward is None:
            feed_forward = self.mlp
        return hidden_states + feed_forward(
            self.post_attention_layernorm(hidden_states)
        )


class KimiTextDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [KimiDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        use_cache=False,
        valid_token_mask=None,
        **kwargs,
    ):
        if use_cache:
            raise ValueError("Kimi training decoder does not support inference cache")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        hidden = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        valid_tokens = valid_token_mask
        if valid_tokens is not None and (
            valid_tokens.shape != hidden.shape[:2]
            or valid_tokens.dtype != torch.bool
            or valid_tokens.device != hidden.device
        ):
            raise ValueError(
                "valid_token_mask must be boolean [batch, sequence] on input device"
            )
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            if attention_mask.shape != hidden.shape[:2]:
                raise ValueError("Kimi padding mask must match [batch, sequence]")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("Kimi padding mask must contain only zero or one")
            valid_tokens = attention_mask.to(device=hidden.device, dtype=torch.bool)
        boundaries = None
        if position_ids is not None and hidden.shape[0] == 1:
            resets = position_ids[0] == 0
            if valid_tokens is not None:
                resets = resets & valid_tokens[0]
            starts = resets.nonzero().flatten()
            if starts.numel() and starts[0] != 0:
                starts = torch.cat((starts.new_zeros(1), starts))
            if starts.numel() > 1:
                boundaries = torch.cat((starts, starts.new_tensor([hidden.shape[1]])))
        for layer in self.layers:
            if layer.is_linear_attn:
                hidden = layer(
                    hidden, attention_mask=valid_tokens, cu_seqlens=boundaries
                )
            else:
                mask = attention_mask
                if mask is None or valid_tokens is not None:
                    length = hidden.shape[1]
                    allowed = torch.ones(
                        length, length, dtype=torch.bool, device=hidden.device
                    ).tril()
                    if boundaries is not None:
                        documents = resets.long().cumsum(0)
                        allowed &= documents[:, None] == documents[None, :]
                    allowed = allowed[None, None]
                    if valid_tokens is not None:
                        allowed = allowed & valid_tokens[:, None, None, :]
                    mask = torch.zeros_like(allowed, dtype=hidden.dtype).masked_fill(
                        ~allowed, float("-inf")
                    )
                hidden = layer(hidden, attention_mask=mask)
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden))


class KimiLinearForCausalLM(PreTrainedModel):
    config_class = KimiLinearConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _supports_attention_backend = True
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        if config.attn_res_block_size is not None:
            raise ValueError("Kimi attention residual integration remains incomplete")
        super().__init__(config)
        self.model = KimiTextDecoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        hidden = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs
        ).last_hidden_state
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
