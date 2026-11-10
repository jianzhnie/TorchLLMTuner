"""Explicit DeepSeek V4 subclasses with optional EP and CP training paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as reference
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention as HFDeepseekV4Attention,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer as HFDeepseekV4DecoderLayer,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts as HFDeepseekV4Experts,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Model as HFDeepseekV4Model,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4SparseMoeBlock as HFDeepseekV4SparseMoeBlock,
)

from ....parallel.expert_parallel.token_dispatch import (
    AllToAllTokenDispatcher,
    materialize,
)
from .context_parallel import deepseek_v4_cp_attention_forward


class DeepseekV4Experts(HFDeepseekV4Experts):
    """HF expert parameters with an explicit all-to-all grouped-GEMM path."""

    def native_eager_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *,
        ep_group,
        fixed_router: bool = False,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        if top_k_weights.shape != top_k_index.shape:
            top_k_weights = top_k_weights.gather(1, top_k_index)
        if fixed_router:
            tokens, top_k = top_k_index.shape
            top_k_index = (
                torch.arange(tokens, device=hidden_states.device).unsqueeze(1)
                + torch.arange(top_k, device=hidden_states.device)
            ) % self.num_global_experts
            top_k_weights = torch.full_like(top_k_weights, 1.0 / top_k)

        dispatcher = getattr(self, "_llmtuner_ep_dispatcher", None)
        if (
            dispatcher is None
            or dispatcher.ep_group is not ep_group
            or dispatcher.top_k != top_k_index.shape[1]
        ):
            dispatcher = AllToAllTokenDispatcher(
                self.num_global_experts, top_k_index.shape[1]
            )
            dispatcher.set_ep_group(ep_group=ep_group)
            self._llmtuner_ep_dispatcher = dispatcher
        counts = torch.bincount(
            top_k_index.reshape(-1), minlength=self.num_global_experts
        )
        pending = dispatcher.begin_dispatch(
            hidden_states, top_k_weights, top_k_index, counts
        )
        routed, local_counts, metadata = dispatcher.finish_dispatch(pending)
        received_scores = materialize(
            dispatcher._dispatch_token_exchange(
                metadata.topk_scores_experts_sorted_N.reshape(-1, 1),
                metadata.output_splits,
                metadata.input_splits,
            )
        )[metadata.permuted_indices]
        gate_up_proj = (
            self.gate_up_proj.to_local()
            if isinstance(self.gate_up_proj, DTensor)
            else self.gate_up_proj
        )
        down_proj = (
            self.down_proj.to_local()
            if isinstance(self.down_proj, DTensor)
            else self.down_proj
        )
        offsets = local_counts.cumsum(0, dtype=torch.int32)
        gate_up = torch._grouped_mm(
            routed, gate_up_proj.transpose(-2, -1), offs=offsets
        ).float()
        gate, up = gate_up.chunk(2, dim=-1)
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        intermediate = self.act_fn(gate) * up
        intermediate = intermediate * received_scores.float()
        expert_output = torch._grouped_mm(
            intermediate.to(hidden_states.dtype),
            down_proj.transpose(-2, -1),
            offs=offsets,
        )
        output = dispatcher._unpermute(
            expert_output.float(), metadata.input_shape, metadata.permuted_indices
        )
        output = materialize(
            dispatcher._combine_token_exchange(
                output, metadata.input_splits, metadata.output_splits
            )
        )
        order = torch.argsort(top_k_index.reshape(-1), stable=True)
        inverse = torch.argsort(order)
        output = output[inverse].reshape(
            hidden_states.shape[0], top_k_index.shape[1], hidden_states.shape[-1]
        )
        return output.sum(dim=1).view(*original_shape)


class DeepseekV4SparseMoeBlock(HFDeepseekV4SparseMoeBlock):
    """Assemble the explicit routed-expert subclass under the native router."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        nn.Module.__init__(self)
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.gate = (
            reference.DeepseekV4HashRouter(config)
            if self.is_hash
            else reference.DeepseekV4TopKRouter(config)
        )
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = reference.DeepseekV4MLP(config)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.reshape(-1, hidden_dim)
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return (routed + self.shared_experts(residual)).to(hidden_states.dtype)


class DeepseekV4DecoderLayer(HFDeepseekV4DecoderLayer):
    """Native decoder layer with the explicit MoE block."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.self_attn = reference.DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = reference.DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = reference.DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = reference.DeepseekV4HyperConnection(config)
        self.ffn_hc = reference.DeepseekV4HyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        post, combination, collapsed = self.attn_hc(hidden_states)
        attention_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        # Sinkhorn output is not symmetric: mixing consumes its transpose.
        hidden_states = post.to(dtype).unsqueeze(-1) * attention_output.unsqueeze(
            -2
        ) + torch.matmul(combination.to(dtype).transpose(-1, -2), hidden_states)
        post, combination, collapsed = self.ffn_hc(hidden_states)
        output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        return post.to(dtype).unsqueeze(-1) * output.unsqueeze(-2) + torch.matmul(
            combination.to(dtype).transpose(-1, -2), hidden_states
        )


class DeepseekV4Model(HFDeepseekV4Model):
    """Native DeepSeek V4 model with explicit layer and CP entry points."""

    context_parallel_attention_class = HFDeepseekV4Attention

    @classmethod
    def is_custom_code(cls) -> bool:
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return HFDeepseekV4Model._can_set_experts_implementation()

    def __init__(self, config: DeepseekV4Config):
        reference.DeepseekV4PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = reference.DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = reference.DeepseekV4RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.hc_head = reference.DeepseekV4HyperHead(config)
        self.post_init()

    def context_parallel_attention_forward(
        self,
        attention_module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: object,
        query_indices: torch.Tensor,
        cp_state: dict[str, object],
        group: object,
        gather_tensor: Callable[[torch.Tensor, object, int], torch.Tensor],
        kwargs: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return deepseek_v4_cp_attention_forward(
            attention_module,
            hidden_states,
            position_embeddings,
            query_indices,
            cp_state,
            group,
            gather_tensor,
            kwargs,
        )


class DeepseekV4ForCausalLM(HFDeepseekV4ForCausalLM):
    """Explicit DeepSeek V4 model chain with the native causal-LM contract."""

    @classmethod
    def is_custom_code(cls) -> bool:
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return True

    def __init__(self, config: DeepseekV4Config):
        reference.DeepseekV4PreTrainedModel.__init__(self, config)
        self.model = DeepseekV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()


__all__ = [
    "DeepseekV4Config",
    "DeepseekV4DecoderLayer",
    "DeepseekV4Experts",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4SparseMoeBlock",
]
