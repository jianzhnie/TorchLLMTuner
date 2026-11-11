# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from Transformers modeling_qwen3_moe.
"""Explicit Qwen3 MoE assembly; HF supplies reusable execution components.

Keep child order and post_init boundaries identical to the reference model:
they determine parameter initialization, RNG consumption and checkpoint names.
Device policies and parallel transformations are installed by their own layers.
"""

from typing import Any

from torch import Tensor, nn
from transformers.models.qwen3_moe import modeling_qwen3_moe as reference
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class Qwen3MoeDecoderLayer(reference.Qwen3MoeDecoderLayer):
    """Own decoder composition and execution using reusable math components."""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.self_attn = reference.Qwen3MoeAttention(config, layer_idx)
        if layer_idx not in config.mlp_only_layers and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = reference.Qwen3MoeSparseMoeBlock(config)
            self.mlp.router_logits_in_output = False
            self.mlp.expert_layout = "fused_gate_up"
            self.moe_block_name = "mlp"
        else:
            self.moe_block_name = None
            self.mlp = reference.Qwen3MoeMLP(
                config, intermediate_size=config.intermediate_size
            )
        self.input_layernorm = reference.Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = reference.Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3MoeModel(reference.Qwen3MoeModel):
    """Model-owned embedding, decoder stack and final normalization."""

    @classmethod
    def is_custom_code(cls) -> bool:
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return reference.Qwen3MoeModel._can_set_experts_implementation()

    def __init__(self, config: Qwen3MoeConfig):
        reference.Qwen3MoePreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            Qwen3MoeDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = reference.Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = reference.Qwen3MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3MoeForCausalLM(reference.Qwen3MoeForCausalLM):
    """Explicit causal model tree, preserving reference state and loss APIs."""

    @classmethod
    def is_custom_code(cls) -> bool:
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # HF detects the decorator by reading the defining model file. Our
        # experts are the reference components, so preserve their capability.
        return reference.Qwen3MoeForCausalLM._can_set_experts_implementation()

    def __init__(self, config: Qwen3MoeConfig):
        reference.Qwen3MoePreTrainedModel.__init__(self, config)
        self.model = Qwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()
