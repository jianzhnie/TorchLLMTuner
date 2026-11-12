# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Layout/initialization reference: Transformers 5.11.0 modeling_qwen3_moe.
"""Model-owned Qwen3-MoE structure and parallel contracts."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from .layers import DecoderLayer, Experts, RMSNorm, RotaryEmbedding, Router


@torch.no_grad()
def initialize_module(module: nn.Module, *, std: float) -> None:
    """Initialize in child order, preserving the Qwen3 reference RNG contract."""
    if isinstance(module, nn.Linear | nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            module.weight[module.padding_idx].zero_()
    elif isinstance(module, Experts):
        nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
        nn.init.normal_(module.down_proj, mean=0.0, std=std)
    elif isinstance(module, Router):
        nn.init.normal_(module.weight, mean=0.0, std=std)


class DecoderStack(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.apply(
            lambda module: initialize_module(module, std=config.initializer_range)
        )

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(tokens)
        rotary = self.rotary_emb(hidden, positions)
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=rotary)
        return self.norm(hidden)


class Qwen3MoeForCausalLM(nn.Module):
    """Experiment-owned causal model; config is the only Transformers dependency."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("Custom Qwen3-MoE requires hidden_act='silu'")
        if config.use_sliding_window:
            raise ValueError(
                "Custom experiment does not implement sliding-window attention"
            )
        if config._attn_implementation not in (None, "eager", "sdpa"):
            raise ValueError("Custom experiment supports eager and SDPA attention only")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Attention heads must be divisible by KV heads")
        if (
            config.num_experts > 0
            and not 1 <= config.num_experts_per_tok <= config.num_experts
        ):
            raise ValueError("Routing top-k must be between one and num_experts")
        if config.router_aux_loss_coef != 0:
            raise ValueError("Custom experiment requires router_aux_loss_coef=0")
        config._attn_implementation = config._attn_implementation or "sdpa"
        self.config = config
        self.model = DecoderStack(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        initialize_module(self.lm_head, std=config.initializer_range)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        if use_cache:
            raise ValueError("Custom experiment does not implement a KV cache")
        hidden = self.model(input_ids, position_ids)
        return SimpleNamespace(logits=self.lm_head(hidden))


@dataclass(frozen=True, slots=True)
class ModelContract:
    """Structural and mathematical facts consumed by parallel assembly."""

    architecture: str
    tp_plan: Mapping[str, str]
    expert_layout: str
    targets: tuple[tuple[str, str, str], ...]
    vocab_size: int


class Qwen3MoeModel(nn.Module):
    """Expose a small stable contract without hiding the model's forward math."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.model = Qwen3MoeForCausalLM(config)
        tp_plan = {
            "model.model.embed_tokens": "embedding_rowwise",
            "model.lm_head": "replicated_with_grad_allreduce",
        }
        for index, layer in enumerate(self.layers):
            prefix = f"model.model.layers.{index}"
            for projection in ("q_proj", "k_proj", "v_proj"):
                tp_plan[f"{prefix}.self_attn.{projection}"] = "colwise"
            tp_plan[f"{prefix}.self_attn.o_proj"] = "rowwise"
            if layer.moe_block_name is not None:
                tp_plan[f"{prefix}.mlp.experts.gate_up_proj"] = "packed_colwise"
                tp_plan[f"{prefix}.mlp.experts.down_proj"] = "rowwise"
            else:
                tp_plan[f"{prefix}.mlp.gate_proj"] = "colwise"
                tp_plan[f"{prefix}.mlp.up_proj"] = "colwise"
                tp_plan[f"{prefix}.mlp.down_proj"] = "rowwise"
        self.contract = ModelContract(
            architecture="qwen3_moe",
            tp_plan=MappingProxyType(tp_plan),
            expert_layout="fused_gate_up",
            vocab_size=config.vocab_size,
            targets=tuple(
                (f"model.model.layers.{i}", "decoder", "sequence_shard")
                for i in range(config.num_hidden_layers)
            )
            + tuple(
                (f"model.model.layers.{i}.mlp", "moe", "fused_gate_up")
                for i, layer in enumerate(self.layers)
                if layer.moe_block_name is not None
            )
            + (
                ("model.model.embed_tokens", "embedding", "vocab_shard"),
                ("model.model.norm", "norm", "replicated"),
                ("model.lm_head", "head", "replicated"),
            ),
        )

    def validate_parallel(self, *, tp_degree: int, ep_degree: int) -> None:
        """Keep Qwen3 geometry and packed expert constraints with the model."""
        config = self.model.config
        for name in ("num_attention_heads", "num_key_value_heads"):
            if getattr(config, name) % tp_degree:
                raise ValueError(f"{name} must be divisible by TP degree")
        if config.moe_intermediate_size % tp_degree:
            raise ValueError("Expert width must be divisible by TP degree")
        if config.num_experts % ep_degree:
            raise ValueError("Expert count must be divisible by EP degree")
        for layer in self.layers:
            if layer.moe_block_name is None:
                continue
            experts = layer.mlp.experts
            expected = (
                config.num_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            )
            if tuple(experts.gate_up_proj.shape) != expected:
                raise ValueError(
                    "Qwen3 gate/up storage differs from its declared geometry"
                )
            if tuple(experts.down_proj.shape) != (
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            ):
                raise ValueError(
                    "Qwen3 down storage differs from its declared geometry"
                )

    @property
    def tp_config(self) -> Mapping[str, str]:
        return self.contract.tp_plan

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.model.layers

    @property
    def tok_embeddings(self) -> nn.Embedding:
        return self.model.model.embed_tokens

    @property
    def norm(self) -> nn.Module:
        return self.model.model.norm

    @property
    def lm_head(self) -> nn.Linear:
        return self.model.lm_head

    @property
    def enable_weight_tying(self) -> bool:
        return self.lm_head.weight is self.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        ).logits

    def preprocess_inputs(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Default protocol hook; specialized models may shard or pack here."""
        return batch

    def parallelize(self, config, *, world_size: int):
        from .assembly import assemble_model

        return assemble_model(self, config, world_size=world_size)

    def apply_expert_parallel(self, mesh) -> None:
        """Hook for an EP dispatcher; eager MoE needs routing-aware dispatch."""
        del mesh
        raise NotImplementedError(
            "Qwen3-MoE eager experts require an EP dispatcher implementation"
        )
