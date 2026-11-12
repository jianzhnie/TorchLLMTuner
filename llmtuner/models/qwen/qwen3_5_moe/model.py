# Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Constructors and decoder execution adapted from Transformers modeling_qwen3_5_moe.
"""Explicit Qwen3.5 MoE text-model composition.

The family owns the GDN/full-attention schedule and decoder tree. Reference
math components and execution APIs are reused. Router/head precision is chosen
at construction; parallel and device transformations retain their own owners.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations import use_experts_implementation
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as reference
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ....errors import UnsupportedCombinationError
from ...common.cast_linear import build_lm_head
from ...common.input_preparation import prepare_text_inputs
from ...configuration import (
    configure_reference_experts,
    validate_attention_backend,
    validate_native_execution_policy,
)
from ...contracts import DecoderStructure, MoEBlockContract
from ..routing import (
    Qwen3_5MoeFP32TopKRouter,
    Qwen3_5MoeTopKRouter,
    validate_router_compute_dtype,
)
from .gated_delta import Qwen3_5MoeGatedDeltaNet, Qwen3_5MoeRMSNormGated
from .rotary import Qwen3_5MoeTextRotaryEmbedding


def validate_model_geometry(config: Qwen3_5MoeTextConfig) -> None:
    """Reject invalid hybrid schedules and head groups before allocating weights."""
    if (
        type(config.num_hidden_layers) is not int
        or config.num_hidden_layers < 0
        or not isinstance(config.layer_types, (list, tuple))
        or len(config.layer_types) != config.num_hidden_layers
    ):
        raise ValueError("Qwen3.5 layer_types must match num_hidden_layers")
    if any(
        kind not in ("full_attention", "linear_attention")
        for kind in config.layer_types
    ):
        raise ValueError("Qwen3.5 layer_types contains an unsupported layer type")
    for name in ("vocab_size", "hidden_size"):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError(f"Qwen3.5 {name} must be a positive integer")
    if config.num_hidden_layers:
        for name in (
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ValueError(f"Qwen3.5 {name} must be a positive integer")
        if config.num_experts_per_tok > config.num_experts:
            raise ValueError("Qwen3.5 top-k cannot exceed the expert count")
    if "full_attention" in config.layer_types:
        for name in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ValueError(f"Qwen3.5 {name} must be a positive integer")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Qwen3.5 attention heads must form complete KV groups")
    if "linear_attention" in config.layer_types:
        for name in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        ):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ValueError(f"Qwen3.5 {name} must be a positive integer")
        if config.linear_num_value_heads % config.linear_num_key_heads:
            raise ValueError("Qwen3.5 GDN value heads must form complete key groups")


def validate_recurrent_mask(
    config: Qwen3_5MoeTextConfig,
    attention_mask: Tensor | None = None,
    *,
    is_causal: bool | None = None,
) -> None:
    """Keep native recurrent state separate from pairwise attention masks.

    GDN accepts a two-dimensional token keep mask, not document boundaries.
    A block-causal mask cannot reset its convolution and recurrent state.
    Check metadata only, without synchronizing device tensors.
    """
    if "linear_attention" not in config.layer_types:
        return
    # Native GDN always uses causal convolution and recurrence. Do not accept
    # metadata requesting another mathematical operation and then ignore it.
    configured_causality = getattr(config, "is_causal", None)
    if any(
        value is not None and value is not True
        for value in (configured_causality, is_causal)
    ):
        raise UnsupportedCombinationError(
            "Qwen3.5 native GDN requires causal execution; "
            "is_causal must be True or None"
        )
    if getattr(config, "attn_mask_type", "causal") == "block_causal":
        raise UnsupportedCombinationError(
            "Qwen3.5 native GDN packed documents require convolution and "
            "recurrent-state resets; block-causal attention alone is insufficient"
        )
    if attention_mask is not None and (
        not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2
    ):
        raise UnsupportedCombinationError(
            "Qwen3.5 native GDN requires a [batch, sequence] token keep mask; "
            "pairwise attention masks cannot reset recurrent state"
        )


def validate_linear_attention_policy(config: Qwen3_5MoeTextConfig) -> None:
    """Reject removed runtime policies until model-owned implementations exist."""
    validate_recurrent_mask(config)
    backend = getattr(config, "linear_attention_backend", "native")
    policy = getattr(config, "gdn_numerical_policy", "native")
    if backend != "native" or policy != "native":
        raise ValueError(
            f"Qwen3.5 linear attention {backend!r}/{policy!r} is not implemented "
            "by the explicit model; only native/native is supported"
        )
    if getattr(config, "gdn_allow_right_padding", False) or getattr(
        config, "gdn_deterministic_reduction", False
    ):
        raise ValueError(
            "Qwen3.5 custom GDN padding/reduction needs its explicit backend"
        )


@torch.no_grad()
def initialize_delta_weights(module: nn.Module) -> bool:
    """Own native decay initialization and the low-precision log-zero guard.

    FP32 keeps the established draw order, including reference partial-load
    behavior. Low-precision missing decay is drawn in FP32 and clamped before
    log; an already loaded low-precision decay consumes no RNG.
    """
    if not isinstance(module, reference.Qwen3_5MoeGatedDeltaNet):
        return False
    init.ones_(module.dt_bias)
    if module.A_log.dtype not in (torch.float16, torch.bfloat16):
        init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        return True
    # External loading marks individual tensors before initializing missing
    # state. Preserve loaded decay values and avoid consuming RNG for them.
    if not getattr(module.A_log, "_is_hf_initialized", False):
        decay = torch.empty_like(module.A_log, dtype=torch.float32).uniform_(0, 16)
        decay.clamp_min_(torch.finfo(torch.float32).tiny).log_()
        init.copy_(module.A_log, decay)
    return True


class Qwen3_5MoeRMSNorm(reference.Qwen3_5MoeRMSNorm):
    """Zero-centered RMSNorm with FP32 normalization and affine math."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        nn.Module.__init__(self)
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: Tensor) -> Tensor:
        values = hidden_states.float()
        normalized = values * torch.rsqrt(
            values.pow(2).mean(-1, keepdim=True) + self.eps
        )
        # Casting before the affine multiply changes low-precision rounding.
        return (normalized * (1.0 + self.weight.float())).type_as(hidden_states)


class Qwen3_5MoeMLP(reference.Qwen3_5MoeMLP):
    """Own the shared expert's gated MLP and stable projection names."""

    def __init__(self, config: Qwen3_5MoeTextConfig, intermediate_size: int) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


@use_experts_implementation
class Qwen3_5MoeExperts(reference.Qwen3_5MoeExperts):
    """Own packed expert math while retaining explicit kernel dispatch.

    Gate/up occupy consecutive halves. Scores multiply after down projection,
    before expert-order index_add in activation dtype. Reference inheritance
    preserves the initialization classification, not forward delegation.
    """

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
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
            mask = nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            active = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        for expert in active:
            index = expert[0]
            slots, tokens = torch.where(mask[index])
            gate, up = nn.functional.linear(
                hidden_states[tokens], self.gate_up_proj[index]
            ).chunk(2, dim=-1)
            values = nn.functional.linear(self.act_fn(gate) * up, self.down_proj[index])
            values = values * top_k_weights[tokens, slots, None]
            output.index_add_(0, tokens, values.to(output.dtype))
        return output


class Qwen3_5MoeSparseMoeBlock(reference.Qwen3_5MoeSparseMoeBlock):
    """Reference MoE math with an explicit tensor-only output contract."""

    router_logits_in_output = False

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        nn.Module.__init__(self)
        router_type = (
            Qwen3_5MoeFP32TopKRouter
            if getattr(config, "router_compute_dtype", None) == "float32"
            else Qwen3_5MoeTopKRouter
        )
        self.gate = router_type(config)
        self.experts = Qwen3_5MoeExperts(config)
        self.shared_expert = Qwen3_5MoeMLP(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        self.moe_contract = MoEBlockContract(
            router="gate",
            activation=config.hidden_act,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            route_norm=True,
            shared_expert="shared_expert",
            shared_expert_gate="shared_expert_gate",
        )
        self.expert_layout = "fused_gate_up"

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Own routed/shared expert order and apply router scores inside experts."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        tokens = hidden_states.view(-1, hidden_dim)
        shared = self.shared_expert(tokens)
        _, scores, selected = self.gate(tokens)
        routed = self.experts(tokens, selected, scores)
        shared = torch.sigmoid(self.shared_expert_gate(tokens)) * shared
        return (routed + shared).reshape(batch_size, sequence_length, hidden_dim)


def initialize_model_weights(module: nn.Module, std: float) -> bool:
    """Initialize text projections, embeddings, norms and packed experts.

    Transformers' guarded initialization primitives preserve already loaded
    tensors. Keep gate/up before down and initialize the gate at its parent
    block, matching the established traversal and RNG order.
    """
    if isinstance(module, nn.Linear | nn.Conv1d):
        # The established generic projection policy treats zero as the
        # default deviation; packed experts below use the configured value.
        if module.weight is not None:
            init.normal_(module.weight, mean=0.0, std=std or 0.02)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        init.normal_(module.weight, mean=0.0, std=std or 0.02)
        # Slicing loses the loaded-tensor marker used by guarded primitives.
        if module.padding_idx is not None and not getattr(
            module.weight, "_is_hf_initialized", False
        ):
            init.zeros_(module.weight[module.padding_idx])
    elif isinstance(module, Qwen3_5MoeTextRotaryEmbedding):
        module.initialize_frequencies()
    elif isinstance(module, Qwen3_5MoeRMSNormGated):
        init.ones_(module.weight)
    elif isinstance(module, Qwen3_5MoeRMSNorm):
        init.zeros_(module.weight)
    elif isinstance(module, Qwen3_5MoeExperts):
        init.normal_(module.gate_up_proj, mean=0.0, std=std)
        init.normal_(module.down_proj, mean=0.0, std=std)
    elif isinstance(module, Qwen3_5MoeSparseMoeBlock):
        init.normal_(module.gate.weight, mean=0.0, std=std)
    else:
        return False
    return True


class Qwen3_5MoeAttention(reference.Qwen3_5MoeAttention):
    """Own query/gate packing and gated full-attention execution.

    Reference kernel dispatch and rotary math are reused without global patches.
    Parameter names and construction order retain external checkpoint semantics.
    """

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
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
            2 * config.num_attention_heads * self.head_dim,
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
        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, 2 * self.head_dim),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = reference.apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        attention = reference.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, reference.eager_attention_forward
        )
        output, weights = attention(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        output = output.reshape(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate)
        return self.o_proj(output), weights


class Qwen3_5MoeDecoderLayer(reference.Qwen3_5MoeDecoderLayer):
    """Own the token-mixer selection and residual-block composition."""

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        # The reference forward renamed this field; retain both spellings
        # while the supported HF components straddle that API change.
        self.block_type = self.layer_type
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
            # Convolution and recurrence must see the complete token history.
            # Parallel consumers use this declaration rather than vendor types.
            self.linear_attn.requires_full_token_history = True
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type!r}")
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.moe_block_name = "mlp"
        self.input_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
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

    def _init_weights(self, module: nn.Module) -> None:
        if not initialize_delta_weights(module) and not initialize_model_weights(
            module, self.config.initializer_range
        ):
            super()._init_weights(module)

    @classmethod
    def is_custom_code(cls) -> bool:
        # Keep loaded-tensor guards active for both reference leaf components
        # and family-owned initialization using the same guarded primitives.
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return reference.Qwen3_5MoeTextModel._can_set_experts_implementation()

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        validate_model_geometry(config)
        validate_native_execution_policy(config, model_name="Qwen3.5")
        validate_linear_attention_policy(config)
        validate_router_compute_dtype(config)
        validate_attention_backend(config)
        configure_reference_experts(config)
        reference.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            Qwen3_5MoeDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
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
    ) -> reference.Qwen3_5MoeModelOutputWithPast:
        """Own hybrid layer execution and gathered-sequence mask geometry."""
        validate_recurrent_mask(
            self.config, attention_mask, is_causal=kwargs.get("is_causal")
        )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        first = next(iter(self.layers), None)
        mixer = (
            (
                first.linear_attn
                if first.layer_type == "linear_attention"
                else first.self_attn
            )
            if first is not None
            else None
        )
        tp_group = getattr(mixer, "_tp_seq_group", None)
        length = inputs_embeds.shape[1] * (
            tp_group.size() if tp_group is not None else 1
        )
        mask_inputs = inputs_embeds
        if length != inputs_embeds.shape[1]:
            mask_inputs = inputs_embeds[:, :1].expand(-1, length, -1)
        if position_ids is None:
            offset = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = torch.arange(length, device=inputs_embeds.device) + offset
            position_ids = position_ids.view(1, 1, -1).expand(
                4, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(4, -1, -1)
        text_positions = None
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_positions, position_ids = position_ids[0], position_ids[1:]
        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=mask_inputs,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_positions,
        )
        masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if "linear_attention" in self.config.layer_types:
            # Keep the reference component API difference local to this family.
            recurrent_mask = getattr(reference, "create_recurrent_attention_mask", None)
            masks["linear_attention"] = (
                recurrent_mask(**mask_kwargs)
                if recurrent_mask is not None
                else self._update_linear_attn_mask(attention_mask, past_key_values)
            )
        hidden = inputs_embeds
        rotary = self.rotary_emb(hidden, position_ids)
        for layer in self.layers:
            hidden = layer(
                hidden,
                position_embeddings=rotary,
                attention_mask=masks[layer.layer_type],
                position_ids=text_positions,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        return reference.Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=self.norm(hidden), past_key_values=past_key_values
        )


class Qwen3_5MoeForCausalLM(reference.Qwen3_5MoeForCausalLM):
    """Model-owned text LM; vision-language execution is a separate contract."""

    decoder_structure = DecoderStructure(
        decoder="model",
        embedding="model.embed_tokens",
        blocks="model.layers",
        norm="model.norm",
        head="lm_head",
    )

    def _init_weights(self, module: nn.Module) -> None:
        if not initialize_delta_weights(module) and not initialize_model_weights(
            module, self.config.initializer_range
        ):
            super()._init_weights(module)

    @classmethod
    def is_custom_code(cls) -> bool:
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return reference.Qwen3_5MoeForCausalLM._can_set_experts_implementation()

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        validate_model_geometry(config)
        validate_native_execution_policy(config, model_name="Qwen3.5")
        validate_linear_attention_policy(config)
        validate_router_compute_dtype(config)
        validate_attention_backend(config)
        configure_reference_experts(config)
        reference.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.model = Qwen3_5MoeTextModel(config)
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

    @property
    def tp_config(self) -> Mapping[str, str]:
        """Declare TP ownership for every supported Qwen3.5 projection."""
        plan: dict[str, str] = {"lm_head": "replicated_with_grad_allreduce"}
        # PP may retain global block IDs in the local ModuleList. Use the
        # registered names so resolved targets keep their canonical FQNs.
        for block_id, layer in self.layers._modules.items():
            prefix = f"model.layers.{block_id}"
            if layer.layer_type == "full_attention":
                for projection in ("q_proj", "k_proj", "v_proj"):
                    plan[f"{prefix}.self_attn.{projection}"] = "colwise"
                plan[f"{prefix}.self_attn.o_proj"] = "rowwise"
            elif layer.layer_type == "linear_attention":
                for projection in (
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                ):
                    plan[f"{prefix}.linear_attn.{projection}"] = (
                        "replicated_with_grad_allreduce"
                    )
                plan[f"{prefix}.linear_attn.out_proj"] = (
                    "replicated_with_grad_allreduce"
                )
            else:
                raise ValueError(f"Unsupported Qwen3.5 layer type {layer.layer_type!r}")
            plan[f"{prefix}.mlp.experts.gate_up_proj"] = "packed_colwise"
            plan[f"{prefix}.mlp.experts.down_proj"] = "rowwise"
            plan[f"{prefix}.mlp.shared_expert.gate_proj"] = "colwise"
            plan[f"{prefix}.mlp.shared_expert.up_proj"] = "colwise"
            plan[f"{prefix}.mlp.shared_expert.down_proj"] = "rowwise"
            plan[f"{prefix}.mlp.shared_expert_gate"] = "replicated_with_grad_allreduce"
        return MappingProxyType(plan)

    def validate_parallel(self, *, tp_degree: int, ep_degree: int) -> None:
        """Validate native MoE storage before the full declared parallel plan.

        Routed expert width is sharded by TP only without EP; shared expert
        width always follows TP. GDN projections remain replicated and do not
        inherit the full-attention head divisibility rules.
        """
        for name, degree in (("TP", tp_degree), ("EP", ep_degree)):
            if type(degree) is not int or degree < 1:
                raise ValueError(f"{name} degree must be a positive integer")
        config = self.config
        if config.num_experts % ep_degree:
            raise ValueError("Expert count must be divisible by EP degree")
        if ep_degree == 1 and config.moe_intermediate_size % tp_degree:
            raise ValueError("Expert width must be divisible by TP degree")
        if config.shared_expert_intermediate_size % tp_degree:
            raise ValueError("Shared expert width must be divisible by TP degree")
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                head_dim = config.head_dim
                query_width = config.num_attention_heads * head_dim
                kv_width = config.num_key_value_heads * head_dim
                # Qwen3.5 packs a query and output gate in q_proj. Generic
                # divisibility alone cannot detect a missing gate half.
                for name, shape in (
                    ("q_proj", (2 * query_width, config.hidden_size)),
                    ("k_proj", (kv_width, config.hidden_size)),
                    ("v_proj", (kv_width, config.hidden_size)),
                    ("o_proj", (config.hidden_size, query_width)),
                ):
                    projection = getattr(layer.self_attn, name)
                    if tuple(projection.weight.shape) != shape:
                        raise ValueError(
                            f"Qwen3.5 {name} storage differs from declared geometry"
                        )
                    bias = projection.bias
                    if (bias is not None) != config.attention_bias or (
                        bias is not None and tuple(bias.shape) != (shape[0],)
                    ):
                        raise ValueError(
                            f"Qwen3.5 {name} bias differs from declared geometry"
                        )
                for name in ("q_norm", "k_norm"):
                    if tuple(getattr(layer.self_attn, name).weight.shape) != (
                        head_dim,
                    ):
                        raise ValueError(
                            f"Qwen3.5 {name} storage differs from declared geometry"
                        )
            elif layer.layer_type == "linear_attention":
                key_width = config.linear_num_key_heads * config.linear_key_head_dim
                value_width = (
                    config.linear_num_value_heads * config.linear_value_head_dim
                )
                conv_width = 2 * key_width + value_width
                attention = layer.linear_attn
                for name, shape in (
                    ("in_proj_qkv", (conv_width, config.hidden_size)),
                    ("in_proj_z", (value_width, config.hidden_size)),
                    ("in_proj_b", (config.linear_num_value_heads, config.hidden_size)),
                    ("in_proj_a", (config.linear_num_value_heads, config.hidden_size)),
                    ("out_proj", (config.hidden_size, value_width)),
                    ("conv1d", (conv_width, 1, config.linear_conv_kernel_dim)),
                    ("norm", (config.linear_value_head_dim,)),
                ):
                    if tuple(getattr(attention, name).weight.shape) != shape:
                        raise ValueError(
                            f"Qwen3.5 GDN {name} storage differs from declared geometry"
                        )
                for name in ("dt_bias", "A_log"):
                    if tuple(getattr(attention, name).shape) != (
                        config.linear_num_value_heads,
                    ):
                        raise ValueError(
                            f"Qwen3.5 GDN {name} storage differs from declared geometry"
                        )
            for name in ("input_layernorm", "post_attention_layernorm"):
                if tuple(getattr(layer, name).weight.shape) != (config.hidden_size,):
                    raise ValueError(
                        f"Qwen3.5 {name} storage differs from declared geometry"
                    )
            block = layer.mlp
            for name, parameter, shape in (
                (
                    "gate/up",
                    block.experts.gate_up_proj,
                    (
                        config.num_experts,
                        2 * config.moe_intermediate_size,
                        config.hidden_size,
                    ),
                ),
                (
                    "down",
                    block.experts.down_proj,
                    (
                        config.num_experts,
                        config.hidden_size,
                        config.moe_intermediate_size,
                    ),
                ),
                ("router", block.gate.weight, (config.num_experts, config.hidden_size)),
                (
                    "shared gate",
                    block.shared_expert_gate.weight,
                    (1, config.hidden_size),
                ),
            ):
                if tuple(parameter.shape) != shape:
                    raise ValueError(
                        f"Qwen3.5 {name} storage differs from declared geometry"
                    )
            for name, shape in (
                (
                    "gate_proj",
                    (config.shared_expert_intermediate_size, config.hidden_size),
                ),
                (
                    "up_proj",
                    (config.shared_expert_intermediate_size, config.hidden_size),
                ),
                (
                    "down_proj",
                    (config.hidden_size, config.shared_expert_intermediate_size),
                ),
            ):
                if tuple(getattr(block.shared_expert, name).weight.shape) != shape:
                    raise ValueError(
                        f"Qwen3.5 shared {name} storage differs from declared geometry"
                    )

        for name, module, shape in (
            (
                "embedding",
                self.tok_embeddings,
                (config.vocab_size, config.hidden_size),
            ),
            ("norm", self.norm, (config.hidden_size,)),
            ("head", self.lm_head, (config.vocab_size, config.hidden_size)),
        ):
            if tuple(module.weight.shape) != shape:
                raise ValueError(
                    f"Qwen3.5 {name} storage differs from declared geometry"
                )

    def materialization_buffers(
        self, device: torch.device
    ) -> dict[tuple[nn.Module, str], Tensor]:
        """Rebuild the model-owned text rotary buffers after meta allocation."""
        rotary = self.model.rotary_emb
        with torch.device("cpu"):
            fresh = Qwen3_5MoeTextRotaryEmbedding(self.config, device="cpu")
        return {
            (rotary, name): value.to(device=device, dtype=rotary._buffers[name].dtype)
            for name, value in fresh.named_buffers(recurse=False)
        }

    def initialize_parameters(self) -> None:
        """Initialize allocated parameters in the model's traversal order."""
        if any(parameter.is_meta for parameter in self.parameters()):
            raise RuntimeError(
                "Qwen3.5 parameters must be materialized before initialization"
            )
        rotary = self.model.rotary_emb

        def initialize(module: nn.Module) -> None:
            # Default frequencies were constructed/materialized by their owner.
            # Recomputing them here can replace device-specific rounding with
            # CPU values. Dynamic RoPE retains its native reset lifecycle.
            if module is rotary and rotary.rope_type == "default":
                return
            self._init_weights(module)

        self.apply(initialize)

    def prepare_inputs(
        self,
        batch: Any,
        *,
        parallel_dims: Any,
        max_context_length: int | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Expose Qwen3.5 text input preparation to the Trainer."""
        validate_recurrent_mask(self.config)
        inputs, labels, extra = prepare_text_inputs(
            self,
            batch,
            parallel_dims=parallel_dims,
            max_context_length=max_context_length,
            preserve_batch_rows=True,
        )
        return inputs, labels, extra

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        positions: Tensor | None = None,
        attention_masks: Tensor | None = None,
        padding_mask: Tensor | None = None,
        skip_lm_head: bool = False,
        input_batch_shape: tuple[int, int] | None = None,
        labels: Tensor | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | Tensor = 0,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Own logits, losses and output assembly after Trainer input adaptation."""
        # Do not silently override reference-style inputs with Trainer aliases.
        # Metadata-only checks avoid device reads in the training hot path.
        if positions is not None and kwargs.get("position_ids") is not None:
            raise ValueError("Specify only one of positions and position_ids")
        if attention_masks is not None and kwargs.get("attention_mask") is not None:
            raise ValueError("Specify only one of attention_masks and attention_mask")
        validate_recurrent_mask(
            self.config,
            attention_masks
            if attention_masks is not None
            else kwargs.get("attention_mask"),
            is_causal=kwargs.get("is_causal"),
        )
        flat_input = input_ids is not None and input_ids.ndim == 1
        if input_batch_shape is not None:
            if not flat_input:
                raise ValueError("input_batch_shape requires flat Trainer tokens")
            input_ids = input_ids.reshape(input_batch_shape)
            if labels is not None and labels.ndim == 1:
                # The causal loss shifts within each row, never across samples.
                labels = labels.reshape(input_batch_shape)
        elif flat_input:
            input_ids = input_ids.unsqueeze(0)
        if positions is not None:
            if positions.ndim == 1:
                # A sequence-length vector remains broadcastable across rows;
                # a full flattened batch carries independent row positions.
                positions = (
                    positions.reshape(input_batch_shape)
                    if input_batch_shape is not None
                    and positions.numel() == input_ids.numel()
                    else positions.unsqueeze(0)
                )
            kwargs["position_ids"] = positions
        if attention_masks is not None:
            kwargs["attention_mask"] = attention_masks
        elif padding_mask is not None and kwargs.get("attention_mask") is None:
            keep_mask = (~padding_mask).to(torch.long)
            if keep_mask.ndim == 1:
                keep_mask = (
                    keep_mask.reshape(input_batch_shape)
                    if input_batch_shape is not None
                    else keep_mask.unsqueeze(0)
                )
            kwargs["attention_mask"] = keep_mask
        kwargs["use_cache"] = False
        if skip_lm_head:
            hidden = self.model(
                input_ids=input_ids, return_dict=True, **kwargs
            ).last_hidden_state
            return hidden.reshape(-1, hidden.shape[-1]) if flat_input else hidden
        output_router_logits = (
            self.config.output_router_logits
            if output_router_logits is None
            else output_router_logits
        )
        outputs = self.model(
            input_ids=input_ids,
            output_router_logits=output_router_logits,
            return_dict=True,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        # Decoder inputs were named arguments of the reference LM API, not
        # loss-function extras. Preserve that boundary for custom losses.
        loss_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name
            not in {
                "attention_mask",
                "position_ids",
                "past_key_values",
                "inputs_embeds",
                "use_cache",
            }
        }
        loss = (
            None
            if labels is None
            else self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)
        )
        aux_loss = None
        if output_router_logits:
            # Pairwise connectivity is not token validity. Keep router
            # statistics independent of causal/packed attention edge layout.
            router_mask = None
            if padding_mask is not None:
                router_mask = ~padding_mask
                if router_mask.ndim == 1:
                    router_mask = router_mask.reshape(
                        outputs.last_hidden_state.shape[:2]
                    )
            elif (
                kwargs.get("attention_mask") is not None
                and kwargs["attention_mask"].ndim == 2
            ):
                router_mask = kwargs["attention_mask"]
            aux_loss = reference.load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                router_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)
        if flat_input:
            logits = logits.reshape(-1, logits.shape[-1])
        output = reference.MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
        use_dict = self.config.return_dict if return_dict is None else return_dict
        return output if use_dict else output.to_tuple()
