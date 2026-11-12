# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Constructors and forward math adapted from Transformers modeling_qwen3_moe.
"""Model-owned Qwen3 MoE assembly and execution with reusable HF components.

Keep child order and post_init boundaries identical to the reference model:
they determine parameter initialization, RNG consumption and checkpoint names.
Device policies and parallel transformations are installed by their own layers.
"""

import inspect
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import module as module_hooks
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations import use_experts_implementation
from transformers.integrations.hub_kernels import use_kernelized_func
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention as HFQwen3MoeAttention,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer as HFQwen3MoeDecoderLayer,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts as HFQwen3MoeExperts,
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
    Qwen3MoeRotaryEmbedding as HFQwen3MoeRotaryEmbedding,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock as HFQwen3MoeSparseMoeBlock,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeTopKRouter as HFQwen3MoeTopKRouter,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    load_balancing_loss_func,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ....datasets.contracts import Batch
from ....errors import UnsupportedCombinationError
from ...common.attention.backend_names import CP_SDPA_BACKEND
from ...common.attention.masks import build_dense_attention_mask
from ...common.cast_linear import build_lm_head
from ...common.input_preparation import prepare_text_inputs
from ...configuration import configure_reference_experts, validate_attention_backend
from ...contracts import DecoderStructure, MoEBlockContract
from ..routing import Qwen3MoeFP32TopKRouter, validate_router_compute_dtype
from .normalization import build_rms_norm, validate_norm_backend
from .rotary import (
    Qwen3FP32RotaryEmbedding,
    apply_fp32_rotary,
    validate_rotary_precision,
)


def validate_attention_geometry(config: Qwen3MoeConfig) -> None:
    """Validate model-owned GQA/RoPE geometry before any parameter allocation."""
    for name in ("hidden_size", "num_attention_heads", "num_key_value_heads"):
        value = getattr(config, name, None)
        if type(value) is not int or value < 1:
            raise ValueError(f"Qwen3 {name} must be a positive integer")
    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("Qwen3 attention heads must be divisible by KV heads")
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    if type(head_dim) is not int or head_dim < 1:
        raise ValueError("Qwen3 head_dim must be a positive integer")
    if head_dim % 2:
        raise ValueError("Qwen3 full rotary attention requires an even head_dim")


def validate_qkv_projection(config: Qwen3MoeConfig) -> None:
    """Reject unknown projection math before allocating or initializing weights."""
    if getattr(config, "qkv_projection", "separate") not in ("separate", "fused"):
        raise ValueError("Qwen3 qkv_projection must be separate or fused")


@use_kernelized_func(apply_rotary_pos_emb)
class Qwen3MoeAttention(HFQwen3MoeAttention):
    """Own attention execution and explicit QKV, norm and rotary policies."""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        projection = getattr(config, "qkv_projection", "separate")
        validate_attention_geometry(config)
        validate_qkv_projection(config)
        validate_norm_backend(config)
        validate_rotary_precision(config)
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        backend = getattr(config, "norm_backend", "native")
        self.q_norm = build_rms_norm(
            self.head_dim, config.rms_norm_eps, backend=backend
        )
        self.k_norm = build_rms_norm(
            self.head_dim, config.rms_norm_eps, backend=backend
        )
        self.sliding_window = getattr(config, "sliding_window", None)
        self._llmtuner_fused_qkv = projection == "fused"

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None]:
        """Execute Q/K/V, normalization, rotary, cache and attention in model order.

        Concatenation is differentiable and does not replace checkpoint storage.
        Low-precision input gradients can differ from three separate GEMMs.
        """
        fp32_rotary = getattr(self.config, "rotary_compute_dtype", None) == "float32"
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        projections = (self.q_proj, self.k_proj, self.v_proj)
        if self._llmtuner_fused_qkv:
            if (
                module_hooks._global_forward_pre_hooks
                or module_hooks._global_forward_hooks
                or module_hooks._global_backward_pre_hooks
                or module_hooks._global_backward_hooks
            ):
                raise ValueError(
                    "Qwen3 fused QKV cannot bypass global module hooks; "
                    "use separate QKV"
                )
            # A fused GEMM bypasses Linear.__call__. Refuse modifications that
            # would otherwise silently disappear, including backward hooks.
            for projection in projections:
                if (
                    type(projection).forward is not nn.Linear.forward
                    or "forward" in projection.__dict__
                    or type(projection).__call__ is not nn.Module.__call__
                    or type(projection)._call_impl is not nn.Module._call_impl
                    or "_call_impl" in projection.__dict__
                    or projection._compiled_call_impl is not None
                    or projection._forward_pre_hooks
                    or projection._forward_hooks
                    or projection._backward_pre_hooks
                    or projection._backward_hooks
                ):
                    raise ValueError(
                        "Qwen3 fused QKV cannot bypass modified projection "
                        "forward or hooks; use separate QKV"
                    )
            has_bias = tuple(projection.bias is not None for projection in projections)
            if any(has_bias) and not all(has_bias):
                raise ValueError(
                    "Qwen3 fused QKV requires biases on all three projections "
                    "or none; use separate QKV for mixed bias layouts"
                )
            weight = torch.cat([projection.weight for projection in projections])
            bias = (
                torch.cat([projection.bias for projection in projections])
                if self.q_proj.bias is not None
                else None
            )
            query, key, value = F.linear(hidden_states, weight, bias).split(
                [projection.out_features for projection in projections], dim=-1
            )
            query = self.q_norm(query.reshape(shape)).transpose(1, 2)
            key = self.k_norm(key.reshape(shape)).transpose(1, 2)
            value = value.reshape(shape).transpose(1, 2)
        else:
            # Preserve native projection/norm ordering, including module hooks.
            query = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
            key = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
            value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        rotary = apply_fp32_rotary if fp32_rotary else apply_rotary_pos_emb
        query, key = rotary(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        if self.config._attn_implementation == CP_SDPA_BACKEND:
            kernel = getattr(self, "_titan_cp_sdpa_kernel", None)
            if kernel is None:
                raise RuntimeError("Qwen3 CP attention requires its installed kernel")
            output = kernel(
                query,
                key,
                value,
                module=self,
                block_mask=attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
            weights = None
        else:
            interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            output, weights = interface(
                self,
                query,
                key,
                value,
                attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        output = output.reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.o_proj(output), weights


class Qwen3MoeMLP(HFQwen3MoeMLP):
    """Own dense gated-MLP storage and math, preserving checkpoint names."""

    def __init__(
        self, config: Qwen3MoeConfig, intermediate_size: int | None = None
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


@use_experts_implementation
class Qwen3MoeExperts(HFQwen3MoeExperts):
    """Own packed storage and eager math, retaining explicit backend dispatch.

    Gate and up occupy consecutive halves. Routing scores multiply after the
    down projection; accumulation follows expert order in activation dtype.
    Reference inheritance preserves initialization classification only.
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
        nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor
    ) -> Tensor:
        output = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            active_experts = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        for expert in active_experts:
            expert_idx = expert[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate, up = F.linear(
                hidden_states[token_idx], self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            values = F.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            values = values * top_k_weights[token_idx, top_k_pos, None]
            output.index_add_(0, token_idx, values.to(output.dtype))
        return output


class Qwen3MoeTopKRouter(HFQwen3MoeTopKRouter):
    """Own native routing; FP32 softmax does not imply FP32 gate GEMM.

    Preserve sorted top-k and normalization without an added epsilon. The
    explicit FP32 routing policy has separate arithmetic and selection order.
    Inheritance retains reference initialization and output-capture contracts.
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tokens = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(tokens, self.weight)
        probabilities = F.softmax(logits, dtype=torch.float, dim=-1)
        scores, indices = torch.topk(probabilities, self.top_k, dim=-1)
        if self.norm_topk_prob:
            scores /= scores.sum(dim=-1, keepdim=True)
        return logits, scores.to(logits.dtype), indices


@torch.no_grad()
def initialize_qwen3_weights(model: PreTrainedModel, module: nn.Module) -> None:
    """Initialize generic components then model-owned packed experts/router.

    Keep HF's initialization-aware writes: post_init and meta materialization
    rely on their flags to avoid reinitializing loaded or shared parameters.
    """
    PreTrainedModel._init_weights(model, module)
    std = model.config.initializer_range
    if isinstance(module, HFQwen3MoeExperts):
        init.normal_(module.gate_up_proj, mean=0.0, std=std)
        init.normal_(module.down_proj, mean=0.0, std=std)
    elif isinstance(module, HFQwen3MoeTopKRouter):
        init.normal_(module.weight, mean=0.0, std=std)


class Qwen3MoeSparseMoeBlock(HFQwen3MoeSparseMoeBlock):
    """Own routing and expert execution without changing native initialization."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        nn.Module.__init__(self)
        self.experts = Qwen3MoeExperts(config)
        router_type = (
            Qwen3MoeFP32TopKRouter
            if getattr(config, "router_compute_dtype", None) == "float32"
            else Qwen3MoeTopKRouter
        )
        self.gate = router_type(config)
        self.moe_contract = MoEBlockContract(
            router="gate",
            activation=config.hidden_act,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            route_norm=config.norm_topk_prob,
            # Native Qwen3 has no shared expert in its forward mathematics.
            # Extensions must declare their own layout and execution semantics.
            shared_expert=None,
        )
        self.router_logits_in_output = False
        self.expert_layout = "fused_gate_up"

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Route flattened tokens; experts apply selected scores after down-proj."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        tokens = hidden_states.view(-1, hidden_dim)
        _, scores, selected = self.gate(tokens)
        output = self.experts(tokens, selected, scores)
        return output.reshape(batch_size, sequence_length, hidden_dim)


class Qwen3MoeDecoderLayer(HFQwen3MoeDecoderLayer):
    """Own decoder composition and execution using reusable math components."""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        if layer_idx not in config.mlp_only_layers and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
            self.moe_block_name = "mlp"
        else:
            self.moe_block_name = None
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = build_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            backend=getattr(config, "norm_backend", "native"),
        )
        self.post_attention_layernorm = build_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            backend=getattr(config, "norm_backend", "native"),
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

    def _init_weights(self, module: nn.Module) -> None:
        initialize_qwen3_weights(self, module)

    def __init__(self, config: Qwen3MoeConfig) -> None:
        validate_attention_geometry(config)
        validate_qkv_projection(config)
        validate_rotary_precision(config)
        validate_norm_backend(config)
        validate_router_compute_dtype(config)
        validate_attention_backend(config)
        configure_reference_experts(config)
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
        self.norm = build_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            backend=getattr(config, "norm_backend", "native"),
        )
        rotary_type = (
            Qwen3FP32RotaryEmbedding
            if getattr(config, "rotary_compute_dtype", None) == "float32"
            else HFQwen3MoeRotaryEmbedding
        )
        self.rotary_emb = rotary_type(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> MoeModelOutputWithPast:
        """Execute the declared decoder stack in model-owned order.

        Transformers supplies mask/cache utilities and output capture; the
        model owns embedding, position construction, block execution and norm.
        Iterate registered blocks directly so stable global IDs need not start
        at zero after a structural transformation. This alone does not provide
        the separate pipeline-stage input/output contract.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Attention gathers TP token shards before QKV. Mask and rotary
        # metadata describe that gathered sequence, not the local residual.
        first_block = next(iter(self.layers), None)
        tp_group = (
            getattr(first_block.self_attn, "_tp_seq_group", None)
            if first_block is not None
            else None
        )
        attention_length = inputs_embeds.shape[1] * (
            tp_group.size() if tp_group is not None else 1
        )
        mask_inputs = inputs_embeds
        if attention_length != inputs_embeds.shape[1]:
            # Mask utilities read shape/device/dtype only. Expand a view instead
            # of gathering or allocating a second hidden-state tensor.
            mask_inputs = inputs_embeds[:, :1].expand(-1, attention_length, -1)
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(attention_length, device=inputs_embeds.device)
                + past_seen_tokens
            ).unsqueeze(0)
        mask_function = (
            create_causal_mask
            if self.config.sliding_window is None
            else create_sliding_window_causal_mask
        )
        if self.config._attn_implementation == CP_SDPA_BACKEND:
            # CP kernels construct global causality from rank ownership. A
            # local HF mask would hide remote keys or use the wrong offsets.
            if attention_mask is not None or past_key_values is not None or use_cache:
                raise UnsupportedCombinationError(
                    "Qwen3 SDPA CP currently requires unpadded causal input "
                    "without an explicit mask or cache"
                )
            causal_mask = None
        else:
            causal_mask = mask_function(
                config=self.config,
                inputs_embeds=mask_inputs,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return MoeModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values,
        )


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
        # Preserve the reference capability API across supported HF versions;
        # local experts use the same explicit implementation decorator.
        return HFQwen3MoeForCausalLM._can_set_experts_implementation()

    def _init_weights(self, module: nn.Module) -> None:
        initialize_qwen3_weights(self, module)

    def __init__(self, config: Qwen3MoeConfig) -> None:
        validate_attention_geometry(config)
        validate_qkv_projection(config)
        validate_rotary_precision(config)
        validate_norm_backend(config)
        validate_router_compute_dtype(config)
        validate_attention_backend(config)
        configure_reference_experts(config)
        HFQwen3MoePreTrainedModel.__init__(self, config)
        self.model = Qwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = build_lm_head(
            config.hidden_size,
            config.vocab_size,
            compute_dtype=getattr(config, "compute_dtype", None),
        )
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    pipeline_runtime_modules = ("model.rotary_emb",)

    def validate_pipeline_partition(self, assignments: list[list[str]]) -> None:
        """Validate exact stage ownership without changing modules or RNG."""
        if self.enable_weight_tying:
            raise UnsupportedCombinationError("Qwen3 PP requires untied embedding/head")
        if self.config.output_router_logits:
            raise UnsupportedCombinationError("Qwen3 PP does not return router logits")
        required = [f"model.layers.{key}" for key in self.model.layers._modules]
        observed = []
        for index, targets in enumerate(assignments):
            expected = set(self.pipeline_runtime_modules)
            if index == 0:
                expected.add("model.embed_tokens")
            if index == len(assignments) - 1:
                expected.update(("model.norm", "lm_head"))
            blocks = [target for target in targets if target in required]
            expected.update(blocks)
            if set(targets) != expected:
                raise ValueError(f"Qwen3 PP stage {index} has invalid module ownership")
            observed.extend(blocks)
        if observed != required:
            raise ValueError("Qwen3 PP must assign every decoder block once in order")

    def configure_pipeline_stage(self, *, first: bool, last: bool) -> None:
        """Select model-owned stage tensor I/O after generic module pruning."""
        self._pipeline_stage = (first, last)

    def forward_pipeline_stage(
        self,
        inputs: Tensor,
        *,
        positions: Tensor | None,
        attention_mask: Tensor | None,
        padding_mask: Tensor | None,
        skip_lm_head: bool,
    ) -> Tensor:
        """Run tokens on the first stage and hidden states on later stages."""
        first, last = self._pipeline_stage
        flat = inputs.ndim == (1 if first else 2)
        if not first and not inputs.is_floating_point():
            raise TypeError("Non-first Qwen3 PP stages require hidden states")
        if flat:
            inputs = inputs.unsqueeze(0)
        if positions is not None and positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if attention_mask is None and padding_mask is not None:
            attention_mask = ~padding_mask
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
        output = self.model(
            input_ids=inputs if first else None,
            inputs_embeds=None if first else inputs,
            position_ids=positions,
            attention_mask=attention_mask,
            use_cache=False,
            output_router_logits=False,
        ).last_hidden_state
        if last and not skip_lm_head:
            output = self.lm_head(output)
        # Stage sends must own storage, including an Identity-only stage.
        return (output.squeeze(0) if flat else output).clone()

    @property
    def layers(self) -> nn.ModuleList:
        """Expose decoder blocks for parallel execution without inspection."""
        return self.model.layers

    @property
    def tok_embeddings(self) -> nn.Embedding:
        """Return the model-owned token embedding."""
        return self.model.embed_tokens

    @property
    def norm(self) -> nn.Module:
        """Return the model-owned final normalization."""
        return self.model.norm

    def materialization_buffers(
        self, device: torch.device
    ) -> dict[tuple[nn.Module, str], Tensor]:
        """Rebuild model-owned RoPE buffers without changing parameters or RNG.

        Constructor-device frequencies preserve the established CPU-then-move
        policy. The memory layer only receives final storage, never RoPE rules.
        Re-resolve after any operation that replaces the rotary module.
        """
        if getattr(self.config, "rotary_init_device", "constructor") != "constructor":
            raise ValueError(
                "Qwen3-MoE buffer reconstruction requires constructor RoPE"
            )
        rotary = self.model.rotary_emb
        if type(rotary) not in (HFQwen3MoeRotaryEmbedding, Qwen3FP32RotaryEmbedding):
            raise TypeError("Qwen3 materialization requires its declared rotary type")
        with torch.device("cpu"):
            if "device" in inspect.signature(HFQwen3MoeRotaryEmbedding).parameters:
                fresh = HFQwen3MoeRotaryEmbedding(self.config, device="cpu")
            else:
                fresh = HFQwen3MoeRotaryEmbedding(self.config)
        return {
            (rotary, name): value.to(device=device, dtype=rotary._buffers[name].dtype)
            for name, value in fresh.named_buffers(recurse=False)
        }

    def initialize_parameters(self) -> None:
        """Initialize allocated parameters in the established module traversal order.

        This is the post-materialization initializer, not eager construction
        replay. It retains the previous apply(_init_weights) RNG contract.
        Invoke before optimizer construction or loading checkpoint parameters.
        """
        if any(parameter.is_meta for parameter in self.parameters()):
            raise RuntimeError(
                "Qwen3 parameters must be materialized before initialization"
            )
        for layer in self.model.layers:
            if (
                layer.moe_block_name is not None
                and type(layer.mlp) is not Qwen3MoeSparseMoeBlock
            ):
                raise TypeError(
                    "Converted experts require their own initialization contract"
                )
        self.apply(self._init_weights)

    @property
    def tp_config(self) -> Mapping[str, str]:
        """Declare exact targets relative to this instance, not HF patterns.

        Embedding/head stay replicated by default, preserving tied weights and
        existing training behavior. Vocabulary sharding remains an explicit
        parallel override. Resolve again after PP changes local ownership.
        """
        plan = (
            {"lm_head": "replicated_with_grad_allreduce"}
            if not hasattr(self, "_pipeline_stage") or self._pipeline_stage[1]
            else {}
        )
        for block_id, layer in self.model.layers._modules.items():
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
        weight = getattr(self.lm_head, "weight", None)
        return weight is not None and weight is getattr(
            self.model.embed_tokens, "weight", None
        )

    def set_cp_mesh(
        self,
        mesh: Any,
        *,
        load_balancer: str | None = None,
        strategy: str = "kv_allgather",
        mask_mesh: Any = None,
    ) -> None:
        """Record the CP execution contract for input preparation and kernels."""
        self.cp_mesh = mesh
        self._cp_load_balancer = load_balancer
        self._cp_mask_mesh = mesh if mask_mesh is None else mask_mesh
        self._cp_strategy = strategy

    def prepare_inputs(
        self,
        batch: Any,
        *,
        parallel_dims: Any,
        max_context_length: int | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Keep independent batch rows isolated when adapting to flat tokens."""
        tokens = batch.input_ids if isinstance(batch, Batch) else batch["input"]
        multirow = tokens.ndim == 2 and tokens.shape[0] > 1
        if (
            multirow
            and parallel_dims is not None
            and any(
                parallel_dims.get_optional_mesh(axis) is not None
                for axis in ("cp", "tp")
            )
        ):
            raise UnsupportedCombinationError(
                "Qwen3 multirow input requires a row-aware mask/sharding contract "
                "for CP/TP; flattening before sharding is unsupported"
            )
        inputs, labels, extra = prepare_text_inputs(
            self,
            batch,
            parallel_dims=parallel_dims,
            max_context_length=max_context_length,
        )
        if multirow:
            rows, width = tokens.shape
            positions = extra.get("positions")
            if positions is None:
                positions = torch.arange(width, device=tokens.device).repeat(rows)
                extra["positions"] = positions
            mask = extra.get("attention_masks")
            if mask is None:
                mask = build_dense_attention_mask(
                    positions,
                    dtype=next(self.parameters()).dtype,
                    block_causal=False,
                )
            row_ids = torch.arange(rows, device=tokens.device).repeat_interleave(width)
            mask = mask.masked_fill(row_ids[:, None] != row_ids[None, :], float("-inf"))
            if "padding_mask" in extra:
                mask = mask.masked_fill(extra["padding_mask"][None, :], float("-inf"))
            extra["attention_masks"] = mask
        return inputs, labels, extra

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        positions: Tensor | None = None,
        attention_masks: Tensor | None = None,
        padding_mask: Tensor | None = None,
        skip_lm_head: bool = False,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool | None = None,
        labels: Tensor | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | Tensor = 0,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> MoeCausalLMOutputWithPast | Tensor | tuple[Any, ...]:
        """Own decoder execution, head selection and training loss semantics.

        Flat-token training is adapted before decoder execution. Compute loss
        on batched logits, then remove the synthetic batch axis for callers.
        Hidden-only execution deliberately avoids both the head and LM losses.
        This training entry point disables cache writes, including when a
        caller requests use_cache; decoder-only caching is a separate API.
        """
        if positions is not None and position_ids is not None:
            raise ValueError("Specify only one of positions and position_ids")
        if attention_masks is not None and attention_mask is not None:
            raise ValueError("Specify only one of attention_masks and attention_mask")
        if hasattr(self, "_pipeline_stage"):
            if input_ids is None or inputs_embeds is not None or labels is not None:
                raise ValueError("Qwen3 PP requires a stage tensor and external loss")
            if past_key_values is not None or use_cache or output_router_logits:
                raise UnsupportedCombinationError(
                    "Qwen3 PP does not support cache/router outputs"
                )
            return self.forward_pipeline_stage(
                input_ids,
                positions=positions if positions is not None else position_ids,
                attention_mask=attention_masks
                if attention_masks is not None
                else attention_mask,
                padding_mask=padding_mask,
                skip_lm_head=skip_lm_head,
            )
        flat_input = input_ids is not None and input_ids.ndim == 1
        if flat_input:
            input_ids = input_ids.unsqueeze(0)
        if positions is not None and positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if attention_masks is not None:
            attention_mask = attention_masks
        elif padding_mask is not None and attention_mask is None:
            keep_mask = (~padding_mask).to(torch.long)
            attention_mask = (
                keep_mask.unsqueeze(0) if keep_mask.ndim == 1 else keep_mask
            )
        if positions is not None:
            position_ids = positions
        output_router_logits = (
            self.config.output_router_logits
            if output_router_logits is None
            else output_router_logits
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_router_logits=output_router_logits,
            return_dict=True,
            **kwargs,
        )
        if skip_lm_head:
            hidden = outputs.last_hidden_state
            return hidden.squeeze(0) if flat_input else hidden
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        loss = (
            None
            if labels is None
            else self.loss_function(logits, labels, self.vocab_size, **kwargs)
        )
        aux_loss = None
        if output_router_logits:
            # Connectivity masks may be 4-D (packed documents or flat rows).
            # Router balancing needs 2-D token validity, not attention edges.
            router_mask = None
            if padding_mask is not None:
                router_mask = ~padding_mask
                if router_mask.ndim == 1:
                    router_mask = router_mask.unsqueeze(0)
            elif attention_mask is not None and attention_mask.ndim == 2:
                router_mask = attention_mask
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                router_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)
        output = MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits.squeeze(0) if flat_input else logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
        if return_dict is None:
            return_dict = self.config.return_dict
        return output if return_dict else output.to_tuple()

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
        validate_attention_geometry(config)
        for name in ("num_attention_heads", "num_key_value_heads"):
            if getattr(config, name) % tp_degree:
                raise ValueError(f"{name} must be divisible by TP degree")
        for layer in self.model.layers:
            if layer.moe_block_name is None:
                if config.intermediate_size % tp_degree:
                    raise ValueError("Dense width must be divisible by TP degree")
                for name, shape in (
                    ("gate_proj", (config.intermediate_size, config.hidden_size)),
                    ("up_proj", (config.intermediate_size, config.hidden_size)),
                    ("down_proj", (config.hidden_size, config.intermediate_size)),
                ):
                    projection = getattr(layer.mlp, name)
                    if tuple(projection.weight.shape) != shape:
                        raise ValueError(
                            f"Qwen3 dense {name} storage differs from declared geometry"
                        )
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
            if tuple(layer.mlp.gate.weight.shape) != (
                config.num_experts,
                config.hidden_size,
            ):
                raise ValueError("Qwen3 router storage differs from declared geometry")
        # The generic shard resolver checks divisibility, not Qwen3's complete
        # Q/K/V/O geometry. Bind the model's declaration to actual storage,
        # including meta tensors, before either TP or EP-only transforms it.
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        query_width = config.num_attention_heads * head_dim
        kv_width = config.num_key_value_heads * head_dim
        for layer in self.model.layers:
            for name, shape in (
                ("q_proj", (query_width, config.hidden_size)),
                ("k_proj", (kv_width, config.hidden_size)),
                ("v_proj", (kv_width, config.hidden_size)),
                ("o_proj", (config.hidden_size, query_width)),
            ):
                projection = getattr(layer.self_attn, name)
                if tuple(projection.weight.shape) != shape:
                    raise ValueError(
                        f"Qwen3 {name} storage differs from declared geometry"
                    )
                expected_bias = (shape[0],) if config.attention_bias else None
                actual_bias = (
                    tuple(projection.bias.shape)
                    if projection.bias is not None
                    else None
                )
                if actual_bias != expected_bias:
                    raise ValueError(
                        f"Qwen3 {name} bias differs from declared geometry"
                    )
            for name in ("q_norm", "k_norm"):
                if tuple(getattr(layer.self_attn, name).weight.shape) != (head_dim,):
                    raise ValueError(
                        f"Qwen3 {name} storage differs from declared geometry"
                    )
        # PP deliberately removes non-owned vocabulary modules. Validate the
        # stage's declared roles, not Identity placeholders from other stages.
        first, last = getattr(self, "_pipeline_stage", (True, True))
        for name, module, shape, owned in (
            (
                "embedding",
                self.tok_embeddings,
                (config.vocab_size, config.hidden_size),
                first,
            ),
            ("norm", self.norm, (config.hidden_size,), last),
            ("head", self.lm_head, (config.vocab_size, config.hidden_size), last),
        ):
            if owned and tuple(module.weight.shape) != shape:
                raise ValueError(f"Qwen3 {name} storage differs from declared geometry")
