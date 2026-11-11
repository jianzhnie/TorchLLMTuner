# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from Transformers modeling_qwen3_moe.
"""Explicit Qwen3 MoE assembly; HF supplies reusable execution components.

Keep child order and post_init boundaries identical to the reference model:
they determine parameter initialization, RNG consumption and checkpoint names.
Device policies and parallel transformations are installed by their own layers.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from torch import Tensor, nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention as HFQwen3MoeAttention,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer as HFQwen3MoeDecoderLayer,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeMLP as HFQwen3MoeMLP,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeModel as HFQwen3MoeModel,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoePreTrainedModel as HFQwen3MoePreTrainedModel,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeRMSNorm as HFQwen3MoeRMSNorm,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeRotaryEmbedding as HFQwen3MoeRotaryEmbedding,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock as HFQwen3MoeSparseMoeBlock,
)

from ...contracts import DecoderStructure, MoEBlockContract


class Qwen3MoeDecoderLayer(HFQwen3MoeDecoderLayer):
    """Own decoder composition and execution using reusable math components."""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.self_attn = HFQwen3MoeAttention(config, layer_idx)
        if layer_idx not in config.mlp_only_layers and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = HFQwen3MoeSparseMoeBlock(config)
            self.mlp.moe_contract = MoEBlockContract(
                router="gate",
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                route_norm=config.norm_topk_prob,
                # Optional extension slot used by conversion/checkpoint callers;
                # native Qwen3 has no shared expert, but a caller may attach one.
                shared_expert="shared_expert",
            )
            self.mlp.router_logits_in_output = False
            self.mlp.expert_layout = "fused_gate_up"
            self.moe_block_name = "mlp"
        else:
            self.moe_block_name = None
            self.mlp = HFQwen3MoeMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = HFQwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = HFQwen3MoeRMSNorm(
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


class Qwen3MoeModel(HFQwen3MoeModel):
    """Model-owned embedding, decoder stack and final normalization."""

    @classmethod
    def is_custom_code(cls) -> bool:
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return HFQwen3MoeModel._can_set_experts_implementation()

    def __init__(self, config: Qwen3MoeConfig) -> None:
        HFQwen3MoePreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            Qwen3MoeDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = HFQwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HFQwen3MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3MoeForCausalLM(HFQwen3MoeForCausalLM):
    """Explicit causal model tree, preserving reference state and loss APIs."""

    decoder_structure = DecoderStructure(
        decoder="model",
        embedding="model.embed_tokens",
        blocks="model.layers",
        norm="model.norm",
        head="lm_head",
        attention="self_attn",
    )

    @classmethod
    def is_custom_code(cls) -> bool:
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # HF detects the decorator by reading the defining model file. Our
        # experts are the reference components, so preserve their capability.
        return HFQwen3MoeForCausalLM._can_set_experts_implementation()

    def __init__(self, config: Qwen3MoeConfig) -> None:
        HFQwen3MoePreTrainedModel.__init__(self, config)
        self.model = Qwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    @property
    def tp_config(self) -> Mapping[str, str]:
        """Declare exact targets relative to this instance, not HF patterns.

        Embedding/head stay replicated by default, preserving tied weights and
        existing training behavior. Vocabulary sharding remains an explicit
        parallel override. Resolve again after PP changes local ownership.
        """
        plan = {"lm_head": "replicated_with_grad_allreduce"}
        for block_id, layer in self.model.layers.named_children():
            prefix = f"model.layers.{block_id}"
            for projection in ("q_proj", "k_proj", "v_proj"):
                plan[f"{prefix}.self_attn.{projection}"] = "colwise"
            for norm in ("q_norm", "k_norm"):
                plan[f"{prefix}.self_attn.{norm}"] = "replicated_with_grad_allreduce"
            plan[f"{prefix}.self_attn.o_proj"] = "rowwise"
            if layer.moe_block_name is not None:
                plan[f"{prefix}.mlp.experts.gate_up_proj"] = "packed_colwise"
                plan[f"{prefix}.mlp.experts.down_proj"] = "rowwise"
            else:
                for projection in ("gate_proj", "up_proj"):
                    plan[f"{prefix}.mlp.{projection}"] = "colwise"
                plan[f"{prefix}.mlp.down_proj"] = "rowwise"
        return MappingProxyType(plan)

    @property
    def enable_weight_tying(self) -> bool:
        """Report actual Parameter identity rather than configuration intent."""
        return self.lm_head.weight is self.model.embed_tokens.weight

    def validate_parallel(self, *, tp_degree: int, ep_degree: int) -> None:
        """Validate the model's full declared plan before any sharding.

        TP+EP delegates routed expert sharding to EP in the current engine;
        expert intermediate width therefore need not divide TP in that case.
        This validates the default full plan, not arbitrary partial overrides.
        """
        for name, degree in (("TP", tp_degree), ("EP", ep_degree)):
            if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
                raise ValueError(f"{name} degree must be a positive integer")
        config = self.config
        for name in ("num_attention_heads", "num_key_value_heads"):
            if getattr(config, name) % tp_degree:
                raise ValueError(f"{name} must be divisible by TP degree")
        for layer in self.model.layers:
            if layer.moe_block_name is None:
                if config.intermediate_size % tp_degree:
                    raise ValueError("Dense width must be divisible by TP degree")
                continue
            if config.num_experts % ep_degree:
                raise ValueError("Expert count must be divisible by EP degree")
            if ep_degree == 1 and config.moe_intermediate_size % tp_degree:
                raise ValueError("Expert width must be divisible by TP degree")
            experts = layer.mlp.experts
            expected_gate_up = (
                config.num_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            )
            expected_down = (
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            )
            if tuple(experts.gate_up_proj.shape) != expected_gate_up:
                raise ValueError("Qwen3 gate/up storage differs from declared geometry")
            if tuple(experts.down_proj.shape) != expected_down:
                raise ValueError("Qwen3 down storage differs from declared geometry")
