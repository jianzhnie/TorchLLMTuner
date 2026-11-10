# Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from Transformers modeling_qwen3_5_moe.
"""Explicit Qwen3.5 MoE text-model composition.

The family owns the GDN/full-attention schedule and decoder tree. Reference
math components and execution APIs are reused while device and numerical
policies remain explicit, separately installed transformations.
"""

from typing import Any

from torch import Tensor, nn
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as reference
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)


class Qwen3_5MoeDecoderLayer(reference.Qwen3_5MoeDecoderLayer):
    """Own the token-mixer selection and residual-block composition."""

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        # The reference forward renamed this field; retain both spellings
        # while the supported HF components straddle that API change.
        self.block_type = self.layer_type
        if self.layer_type == "linear_attention":
            self.linear_attn = reference.Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = reference.Qwen3_5MoeAttention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type!r}")
        self.mlp = reference.Qwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = reference.Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = reference.Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states
        output = self.mlp(self.post_attention_layernorm(hidden_states))
        if isinstance(output, tuple):
            output, _ = output
        return hidden_states + output


class Qwen3_5MoeTextModel(reference.Qwen3_5MoeTextModel):
    """Construct the text embedding, hybrid decoder stack and output norm."""

    @classmethod
    def is_custom_code(cls) -> bool:
        # Use the guarded reference initializer, including container-owned
        # initialization of router weights stored on a child module.
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return reference.Qwen3_5MoeTextModel._can_set_experts_implementation()

    def __init__(self, config: Qwen3_5MoeTextConfig):
        reference.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            Qwen3_5MoeDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = reference.Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = reference.Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3_5MoeForCausalLM(reference.Qwen3_5MoeForCausalLM):
    """Model-owned text LM; vision-language execution is a separate contract."""

    @classmethod
    def is_custom_code(cls) -> bool:
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return reference.Qwen3_5MoeForCausalLM._can_set_experts_implementation()

    def __init__(self, config: Qwen3_5MoeTextConfig):
        reference.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.model = Qwen3_5MoeTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()
