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
"""Explicit Kimi text decoder and model-owned training contract.

Adapted from FSDPTurbo 0af4eca models/kimi/modeling_kimi_linear.py.
Dense, local sparse-expert and attention-residual training are implemented.
Native all-to-all EP conversion lives in parallel/expert_parallel; hybrid KCP
and pipeline transfer of attention-residual history require further integration.
"""

from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from ....errors import UnsupportedCombinationError
from ...common.cast_linear import build_lm_head
from ...common.input_preparation import prepare_text_inputs
from ...common.moe.layer import MoE
from ...configuration import (
    validate_attention_backend,
    validate_native_execution_policy,
)
from ...contracts import DecoderStructure
from .attention_residual import attention_residual
from .configuration import KimiLinearConfig
from .delta_attention import KimiDeltaAttention
from .latent_attention import KimiLatentAttention
from .mlp import KimiMLP as KimiMLP
from .moe import KimiMoEGate, KimiRoutedExperts, KimiSparseMoE
from .normalization import KimiRMSNorm


class KimiDecoderLayer(nn.Module):
    """Own KDA/MLA selection, residual math and the dense or sparse MLP slot."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_res_block_size = config.attn_res_block_size
        if self.attn_res_block_size is not None:
            for name in ("self_attention_res", "mlp_res"):
                setattr(
                    self,
                    f"{name}_norm",
                    KimiRMSNorm(config.hidden_size, config.rms_norm_eps),
                )
                setattr(
                    self, f"{name}_proj", nn.Linear(config.hidden_size, 1, bias=False)
                )
        self.is_linear_attn = config.is_kda_layer(layer_idx)
        self.self_attn = (
            KimiDeltaAttention if self.is_linear_attn else KimiLatentAttention
        )(config, layer_idx)
        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiSparseMoE(config)
            self.moe_block_name = "block_sparse_moe"
        else:
            self.mlp = KimiMLP(config)
            self.moe_block_name = None
        self.input_layernorm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        block_residual: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.attn_res_block_size is not None:
            shape = hidden_states.shape
            prefix = hidden_states
            if block_residual.shape[1]:
                hidden_states = attention_residual(
                    prefix.reshape(-1, shape[-1]),
                    block_residual,
                    self.self_attention_res_proj.weight,
                    self.self_attention_res_norm.weight,
                    self.self_attention_res_norm.variance_epsilon,
                ).reshape(shape)
            if self.layer_idx % self.attn_res_block_size == 0:
                block_residual = torch.cat(
                    (block_residual, prefix.reshape(-1, shape[-1]).unsqueeze(1)), 1
                )
                prefix = None
            output = self.self_attn(
                self.input_layernorm(hidden_states),
                attention_mask=attention_mask,
                **kwargs,
            )
            prefix = output if prefix is None else prefix + output
            hidden_states = attention_residual(
                prefix.reshape(-1, shape[-1]),
                block_residual,
                self.mlp_res_proj.weight,
                self.mlp_res_norm.weight,
                self.mlp_res_norm.variance_epsilon,
            ).reshape(shape)
            residual = prefix
        else:
            hidden_states = hidden_states + self.self_attn(
                self.input_layernorm(hidden_states),
                attention_mask=attention_mask,
                **kwargs,
            )
            residual = hidden_states
        # Consume the declared slot, including a replacement installed by EP.
        # Do not retain a second registered alias to the feed-forward module.
        feed_forward = getattr(self, self.moe_block_name or "mlp")
        hidden_states = residual + feed_forward(
            self.post_attention_layernorm(hidden_states)
        )
        if self.attn_res_block_size is not None:
            return hidden_states, block_residual
        return hidden_states


class KimiTextDecoder(nn.Module):
    """Model-owned embedding, hybrid decoder stack and final normalization."""

    def __init__(self, config: KimiLinearConfig) -> None:
        validate_native_execution_policy(config, model_name="Kimi")
        validate_attention_backend(config)
        super().__init__()
        self.config = config
        if config.attn_res_block_size is not None:
            if (
                not isinstance(config.attn_res_block_size, int)
                or isinstance(config.attn_res_block_size, bool)
                or config.attn_res_block_size <= 0
            ):
                raise ValueError("attn_res_block_size must be a positive integer")
            self.output_attn_res_norm = KimiRMSNorm(
                config.hidden_size, config.rms_norm_eps
            )
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            KimiDecoderLayer(config, index) for index in range(config.num_hidden_layers)
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
        if use_cache or kwargs.get("past_key_values") is not None:
            raise ValueError("Kimi training decoder does not support inference cache")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        hidden = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        if position_ids is not None and (
            position_ids.shape != hidden.shape[:2]
            or position_ids.device != hidden.device
            or position_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "Kimi position_ids must be integer [batch, sequence] on input device"
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
            mask_tokens = attention_mask.to(device=hidden.device, dtype=torch.bool)
            if valid_tokens is not None and not torch.equal(valid_tokens, mask_tokens):
                raise ValueError("Kimi attention_mask and valid_token_mask disagree")
            valid_tokens = mask_tokens
        if position_ids is not None and hidden.shape[0] > 1:
            resets = position_ids[:, 1:] == 0
            if valid_tokens is not None:
                resets = resets & valid_tokens[:, 1:]
            if bool(resets.any()):
                raise ValueError(
                    "Kimi packed document resets require a single batch row; "
                    "batched recurrence boundaries are not implemented"
                )
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
        blocks = hidden.new_zeros(hidden.shape[0] * hidden.shape[1], 0, hidden.shape[2])
        for layer in self.layers:
            if layer.is_linear_attn:
                hidden = layer(
                    hidden,
                    attention_mask=valid_tokens,
                    cu_seqlens=boundaries,
                    block_residual=blocks,
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
                hidden = layer(hidden, attention_mask=mask, block_residual=blocks)
            if self.config.attn_res_block_size is not None:
                hidden, blocks = hidden
        if self.config.attn_res_block_size is not None:
            hidden = attention_residual(
                hidden.reshape(-1, hidden.shape[-1]),
                blocks,
                self.output_attn_res_proj.weight,
                self.output_attn_res_norm.weight,
                self.output_attn_res_norm.variance_epsilon,
            ).reshape_as(hidden)
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden))


class KimiLinearForCausalLM(PreTrainedModel):
    """Explicit causal model tree with Kimi-specific initialization and loss."""

    decoder_structure = DecoderStructure(
        decoder="model",
        embedding="model.embed_tokens",
        blocks="model.layers",
        norm="model.norm",
        head="lm_head",
        attention="self_attn",
    )

    config_class = KimiLinearConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _supports_attention_backend = True
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: KimiLinearConfig) -> None:
        validate_native_execution_policy(config, model_name="Kimi")
        validate_attention_backend(config)
        if getattr(config, "experts_implementation", "native") not in (
            "native",
            "eager",
        ):
            raise ValueError(
                "Kimi local experts support experts_implementation native/eager; "
                "other backends require a model-owned expert implementation."
            )
        super().__init__(config)
        self.model = KimiTextDecoder(config)
        self.lm_head = build_lm_head(
            config.hidden_size,
            config.vocab_size,
            compute_dtype=getattr(config, "compute_dtype", None),
        )
        self.post_init()

    @property
    def enable_weight_tying(self) -> bool:
        """Report actual Parameter identity rather than configuration intent."""
        weight = getattr(self.lm_head, "weight", None)
        return weight is not None and weight is getattr(
            self.model.embed_tokens, "weight", None
        )

    @property
    def layers(self) -> nn.ModuleList:
        """Expose the model-owned decoder layers to parallel executors."""
        return self.model.layers

    @property
    def tok_embeddings(self) -> nn.Embedding:
        """Return the model-owned token embedding."""
        return self.model.embed_tokens

    @property
    def norm(self) -> nn.Module:
        """Return the model-owned final normalization."""
        return self.model.norm

    def set_cp_mesh(
        self,
        mesh: Any,
        *,
        load_balancer: str | None = None,
        strategy: str = "kv_allgather",
        mask_mesh: Any = None,
    ) -> None:
        """Record CP metadata; KDA validates recurrence constraints later."""
        self.cp_mesh = mesh
        self._cp_load_balancer = load_balancer
        self._cp_mask_mesh = mesh if mask_mesh is None else mask_mesh
        self._cp_strategy = strategy

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize missing linear/embedding weights without overwriting loads."""
        if isinstance(module, nn.Linear | nn.Embedding):
            # A partially loaded module may still reach this callback. The
            # parameter flag protects both normal sampling and padding reset.
            if getattr(module.weight, "_is_hf_initialized", False):
                return
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def materialization_buffers(
        self, device: torch.device
    ) -> dict[tuple[nn.Module, str], torch.Tensor]:
        """Rebuild non-checkpoint MoE counters after meta allocation.

        EP conversion creates these buffers on the model-owned MoE blocks. They
        are runtime state, so memory materialization receives their values from
        the model instead of inferring names or semantics globally.
        """
        buffers: dict[tuple[nn.Module, str], torch.Tensor] = {}
        for module in self.modules():
            if not isinstance(module, MoE):
                continue
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.is_meta:
                    buffers[module, name] = torch.zeros(
                        buffer.shape, dtype=buffer.dtype, device=device
                    )
        return buffers

    def initialize_parameters(self) -> None:
        """Initialize allocated storage, including constructor-owned parameters.

        ``to_empty`` discards norm, convolution, router and KDA constructor
        values as well as linear weights. Restore those explicitly here without
        changing eager construction or its RNG order.
        """
        if any(parameter.is_meta for parameter in self.parameters()):
            raise RuntimeError(
                "Kimi parameters must be materialized before initialization"
            )
        if any(isinstance(module, KimiRoutedExperts) for module in self.modules()):
            raise TypeError(
                "Converted Kimi experts require an ownership-aware initialization "
                "contract; initialize before conversion or load checkpoint weights"
            )
        self.apply(self._init_weights)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, KimiRMSNorm):
                    nn.init.ones_(module.weight)
                elif isinstance(module, nn.Conv1d):
                    module.reset_parameters()
                elif isinstance(module, KimiMoEGate):
                    module.reset_parameters()
                    nn.init.zeros_(module.e_score_correction_bias)
                elif isinstance(module, KimiDeltaAttention):
                    rates = torch.empty_like(module.A_log, dtype=torch.float32)
                    module.A_log.copy_(rates.uniform_(1, 16).log_())
                    nn.init.zeros_(module.dt_bias)

    def prepare_inputs(
        self,
        batch: Any,
        *,
        parallel_dims: Any,
        max_context_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Expose Kimi's text input contract to the Trainer."""
        inputs, labels, extra = prepare_text_inputs(
            self,
            batch,
            parallel_dims=parallel_dims,
            max_context_length=max_context_length,
            preserve_batch_rows=True,
        )
        if "input_batch_shape" in extra and "attention_masks" in extra:
            raise UnsupportedCombinationError(
                "Kimi packed document input currently requires a single batch row"
            )
        return inputs, labels, extra

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        positions=None,
        attention_masks=None,
        padding_mask=None,
        skip_lm_head: bool = False,
        input_batch_shape: tuple[int, int] | None = None,
        **kwargs,
    ):
        if positions is not None and kwargs.get("position_ids") is not None:
            raise ValueError("Specify only one of positions and position_ids")
        if attention_masks is not None and kwargs.get("attention_mask") is not None:
            raise ValueError("Specify only one of attention_masks and attention_mask")
        flat_input = input_ids is not None and input_ids.ndim == 1
        if input_batch_shape is not None:
            if not flat_input:
                raise ValueError("input_batch_shape requires flat Trainer tokens")
            input_ids = input_ids.reshape(input_batch_shape)
            if labels is not None:
                labels = labels.reshape(input_batch_shape)
        elif flat_input:
            input_ids = input_ids.unsqueeze(0)
        if positions is not None:
            if positions.ndim == 1 and input_batch_shape is not None:
                positions = positions.reshape(input_batch_shape)
            kwargs["position_ids"] = (
                positions.unsqueeze(0) if positions.ndim == 1 else positions
            )
        if attention_masks is not None:
            kwargs["attention_mask"] = attention_masks
        elif padding_mask is not None:
            valid_tokens = ~padding_mask
            if valid_tokens.ndim == 1 and input_batch_shape is not None:
                valid_tokens = valid_tokens.reshape(input_batch_shape)
            kwargs["valid_token_mask"] = (
                valid_tokens.unsqueeze(0) if valid_tokens.ndim == 1 else valid_tokens
            )
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        hidden = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs
        ).last_hidden_state
        if skip_lm_head:
            return hidden.reshape(-1, hidden.shape[-1]) if flat_input else hidden
        logits = self.lm_head(hidden)
        loss_logits = logits
        if flat_input:
            logits = logits.reshape(-1, logits.shape[-1])
        loss = None
        if labels is not None:
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
            loss = nn.functional.cross_entropy(
                loss_logits[:, :-1].float().reshape(-1, loss_logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
        )
