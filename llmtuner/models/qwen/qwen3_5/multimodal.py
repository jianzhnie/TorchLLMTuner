# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
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
"""Qwen3.5 model with portable core operators and the Turbo forward and loss."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from ...execution import ModelExecutionCapabilities
from ..checkpoint import qwen35_multimodal_configuration
from ..ops import (
    flash_attention_forward,
    gated_attention_inputs,
    gated_delta_forward,
    rms_norm,
    vision_attention_forward,
)


def _attention_interface(implementation: str) -> Callable:
    if implementation == "flash_attention_2":
        return flash_attention_forward
    return ALL_ATTENTION_FUNCTIONS.get_interface(
        implementation, hf.eager_attention_forward
    )


class Qwen3_5RMSNormGated(hf.Qwen3_5RMSNormGated):
    """Apply the norm-before-gate convention used by the training kernels."""

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize and scale hidden states before applying the SiLU gate."""
        normalized = rms_norm(hidden_states, self.weight, self.variance_epsilon)
        return normalized * F.silu(gate)


class Qwen3_5RMSNorm(hf.Qwen3_5RMSNorm):
    """Apply Qwen's 1 + weight scale through the common normalization operator."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize hidden states with the Qwen residual scale."""
        use_eager = hidden_states.device.type != "npu"
        weight = 1.0 + (self.weight.float() if use_eager else self.weight)
        return rms_norm(
            hidden_states,
            weight,
            self.eps,
        ).to(hidden_states.dtype)


class Qwen3_5GatedDeltaNet(hf.Qwen3_5GatedDeltaNet):
    """GDN computation with explicit convolution and recurrent cache state."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.gdn_implementation = getattr(config, "gdn_implementation", "eager")
        self.causal_conv1d_implementation = getattr(
            config, "causal_conv1d_implementation", "eager"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute one GDN layer, retaining HF Cache continuation semantics."""
        hidden_states = hf.apply_mask_to_padding_states(hidden_states, attention_mask)
        return gated_delta_forward(
            self,
            hidden_states,
            cache_params=cache_params,
            cu_seqlens=kwargs.get("cu_seqlens"),
        )


class Qwen3_5Attention(hf.Qwen3_5Attention):
    """Text attention with explicit Turbo backend selection and Qwen gating."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor = None,
        past_key_values: Cache = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project Q/K/V, update KV cache, and apply gated attention."""
        if kwargs.get("position_ids") is not None and kwargs["position_ids"].ndim == 3:
            kwargs["position_ids"] = kwargs["position_ids"][0]
        input_shape = hidden_states.shape[:-1]
        query_states, key_states, value_states, gate = gated_attention_inputs(
            self, hidden_states, position_embeddings, past_key_values
        )

        attention_interface = _attention_interface(self.config._attn_implementation)
        total_seq_len = query_states.shape[2]
        attention_kwargs = dict(
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=True,
            **kwargs,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            total_seq_len=total_seq_len,
            seq_split_lens=None,
            **attention_kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3_5VisionAttention(hf.Qwen3_5VisionAttention):
    """Packed bidirectional attention for independent image and video frames."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Attend within each sequence delimited by cumulative lengths."""
        return vision_attention_forward(
            self,
            _attention_interface,
            hidden_states,
            cu_seqlens,
            position_embeddings,
            **kwargs,
        )


class Qwen3_5DecoderLayer(hf.Qwen3_5DecoderLayer):
    """Decoder layer that assembles FSDPTurbo attention / GDN / MLP / norm sublayers."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        hf.GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.block_type = self.layer_type
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type!r}")
        self.mlp = hf.Qwen3_5MLP(config, config.intermediate_size)
        self.moe_block_name = None
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen3_5TextModel(hf.Qwen3_5TextModel):
    """Text model assembled from explicit Qwen component subclasses."""

    def __init__(self, config: Qwen3_5TextConfig) -> None:
        hf.Qwen3_5PreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Preserve HF initialization guards and Tensor/DTensor layouts for GDN."""
        if isinstance(module, hf.Qwen3_5GatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log,
                torch.empty_like(module.A_log, dtype=torch.float32)
                .uniform_(0.01, 16)
                .log_(),
            )
        else:
            super()._init_weights(module)


class Qwen3_5VisionBlock(hf.Qwen3_5VisionBlock):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        hf.GradientCheckpointingLayer.__init__(self)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config)
        self.mlp = hf.Qwen3_5VisionMLP(config)


class Qwen3_5Model(hf.Qwen3_5Model):
    """Multimodal wrapper that assembles FSDPTurbo text and vision submodels."""

    def __init__(self, config: Qwen3_5Config) -> None:
        hf.Qwen3_5PreTrainedModel.__init__(self, config)
        self.visual = hf.Qwen3_5VisionModel._from_config(config.vision_config)
        self.visual.blocks = nn.ModuleList(
            [
                Qwen3_5VisionBlock(config.vision_config)
                for _ in range(config.vision_config.depth)
            ]
        )
        self.language_model = Qwen3_5TextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class Qwen3_5ForConditionalGeneration(hf.Qwen3_5ForConditionalGeneration):
    @property
    def checkpoint_configuration(self) -> dict[str, object]:
        return qwen35_multimodal_configuration(self.config)

    @property
    def execution_capabilities(self) -> ModelExecutionCapabilities:
        return ModelExecutionCapabilities(
            hidden_states=True, hidden_size=self.config.text_config.hidden_size
        )

    @property
    def vocab_size(self) -> int:
        return self.config.text_config.vocab_size

    def __init__(self, config: Qwen3_5Config) -> None:
        hf.Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *args: Any,
        skip_lm_head: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Keep HF logits behavior and expose uncached full hidden states."""
        if not skip_lm_head:
            return super().forward(input_ids, *args, **kwargs)
        if args:
            raise TypeError("Hidden-state output requires keyword model arguments")
        if (
            kwargs.get("past_key_values") is not None
            or kwargs.get("cache_position") is not None
            or kwargs.get("use_cache", False)
        ):
            raise ValueError("Hidden-state training output does not support KV cache")
        # These options belong to the LM head; hidden output keeps every row
        # and always returns a tensor, independently of the public output format.
        for name in ("labels", "logits_to_keep", "return_dict"):
            kwargs.pop(name, None)
        kwargs["use_cache"] = False
        outputs = self.model(input_ids=input_ids, return_dict=True, **kwargs)
        return outputs.last_hidden_state


__all__ = ["Qwen3_5ForConditionalGeneration"]
