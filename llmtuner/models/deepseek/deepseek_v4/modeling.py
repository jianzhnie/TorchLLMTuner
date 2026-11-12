# Copyright 2026 The DeepSeek-AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Decoder execution adapted from Transformers 5.11 modeling_deepseek_v4.
"""Explicit DeepSeek V4 subclasses with optional EP and CP training paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations import use_experts_implementation
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as reference
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
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
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ...common.cast_linear import build_lm_head
from ...common.input_preparation import prepare_text_inputs
from ...configuration import (
    configure_reference_experts,
    validate_attention_backend,
    validate_native_execution_policy,
)
from ...contracts import DecoderStructure
from .attention import DeepseekV4Attention, validate_output_geometry
from .compression import DeepseekV4CSACompressor, DeepseekV4HCACompressor
from .context_parallel import deepseek_v4_cp_attention_forward
from .hyperconnection import DeepseekV4HyperConnection, DeepseekV4HyperHead
from .indexer import DeepseekV4Indexer
from .normalization import DeepseekV4RMSNorm, DeepseekV4UnweightedRMSNorm
from .rotary import DeepseekV4RotaryEmbedding
from .routing import (
    DeepseekV4HashRouter,
    DeepseekV4TopKRouter,
    validate_routing_config,
)


def validate_layer_schedule(config: DeepseekV4Config) -> None:
    """Validate complete decoder schedules before allocating any model tensors."""
    attention_types = {
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    }
    for name, supported in (
        ("layer_types", attention_types),
        ("mlp_layer_types", {"hash_moe", "moe"}),
    ):
        schedule = getattr(config, name)
        if len(schedule) != config.num_hidden_layers:
            raise ValueError(f"DeepSeek V4 {name} must cover every decoder layer")
        for index, kind in enumerate(schedule):
            if kind not in supported:
                raise ValueError(f"DeepSeek V4 unsupported {name}[{index}]: {kind!r}")
    for kind in set(config.layer_types) - {"sliding_attention"}:
        rate = config.compress_rates.get(kind)
        if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
            raise ValueError(f"DeepSeek V4 compress_rates[{kind!r}] must be positive")


def initialize_model_weights(module: nn.Module, std: float) -> None:
    """Initialize explicitly declared V4 modules with native RNG and load guards."""
    if isinstance(module, nn.Linear):
        init.normal_(module.weight, mean=0.0, std=std or 0.02)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        init.normal_(module.weight, mean=0.0, std=std or 0.02)
        if module.padding_idx is not None and not getattr(
            module.weight, "_is_hf_initialized", False
        ):
            init.zeros_(module.weight[module.padding_idx])
    elif isinstance(module, DeepseekV4RMSNorm):
        init.ones_(module.weight)
    elif isinstance(module, DeepseekV4UnweightedRMSNorm):
        pass
    elif isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
        init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, DeepseekV4TopKRouter):
            init.zeros_(module.e_score_correction_bias)
        else:
            init.zeros_(module.tid2eid)
    elif isinstance(module, DeepseekV4Experts):
        init.normal_(module.gate_up_proj, mean=0.0, std=std)
        init.normal_(module.down_proj, mean=0.0, std=std)
    elif isinstance(module, DeepseekV4Attention):
        init.zeros_(module.sinks)
    elif isinstance(module, DeepseekV4HyperConnection):
        init.normal_(module.fn, mean=0.0, std=std)
        init.zeros_(module.base)
        init.ones_(module.scale)
    elif isinstance(module, DeepseekV4HyperHead):
        init.normal_(module.hc_fn, mean=0.0, std=std)
        init.zeros_(module.hc_base)
        init.ones_(module.hc_scale)
    elif isinstance(
        module, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)
    ):
        init.zeros_(module.position_bias)
    elif isinstance(module, DeepseekV4RotaryEmbedding):
        for kind in module.layer_types:
            initializer = (
                module.compute_default_rope_parameters
                if module.rope_type[kind] == "default"
                else ROPE_INIT_FUNCTIONS[module.rope_type[kind]]
            )
            frequencies, _ = initializer(module.config, layer_type=kind)
            init.copy_(getattr(module, f"{kind}_inv_freq"), frequencies)
            init.copy_(getattr(module, f"{kind}_original_inv_freq"), frequencies)


@use_experts_implementation
class DeepseekV4Experts(HFDeepseekV4Experts):
    """Own packed gate/up, asymmetric SwiGLU clamp and post-down scoring."""

    def __init__(self, config: DeepseekV4Config) -> None:
        nn.Module.__init__(self)
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Backend implementations also consume this model-owned clamp contract.
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate.clamp(max=self.limit)) * up.clamp(
            min=-self.limit, max=self.limit
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            active = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        for expert in active:
            expert = expert[0]
            slot, token = torch.where(mask[expert])
            values = self._apply_gate(
                F.linear(hidden_states[token], self.gate_up_proj[expert])
            )
            values = F.linear(values, self.down_proj[expert])
            values = values * top_k_weights[token, slot, None]
            output.index_add_(0, token, values.to(output.dtype))
        return output


class DeepseekV4MLP(reference.DeepseekV4MLP):
    """Shared expert with separate projections and asymmetric SwiGLU clamp."""

    def __init__(self, config: DeepseekV4Config) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.limit)
        up = self.up_proj(hidden_states).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(self.act_fn(gate) * up)


class DeepseekV4SparseMoeBlock(HFDeepseekV4SparseMoeBlock):
    """Assemble the explicit routed-expert subclass under the native router."""

    router_logits_in_output = False

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.gate = (
            DeepseekV4HashRouter(config)
            if self.is_hash
            else DeepseekV4TopKRouter(config)
        )
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

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

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.moe_block_name = "mlp"
        self.input_layernorm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

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

    context_parallel_attention_class = DeepseekV4Attention

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        initialize_model_weights(module, self.config.initializer_range)

    @classmethod
    def is_custom_code(cls) -> bool:
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return HFDeepseekV4Model._can_set_experts_implementation()

    def __init__(self, config: DeepseekV4Config) -> None:
        validate_native_execution_policy(config, model_name="DeepSeek V4")
        validate_output_geometry(config)
        validate_layer_schedule(config)
        validate_routing_config(config)
        validate_attention_backend(config)
        configure_reference_experts(config)
        reference.DeepseekV4PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            DeepseekV4DecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.hc_head = DeepseekV4HyperHead(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> reference.MoeModelOutputWithPast:
        """Own hyperconnection stream expansion, layer execution and reduction."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        # Preserve the native distinction between internal attention state and
        # the caller-owned cache returned by this invocation.
        return_cache = past_key_values if use_cache else None
        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_key_values.get_seq_length()
            ).unsqueeze(0)
        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        hidden_states = (
            inputs_embeds.unsqueeze(2)
            .expand(-1, -1, self.config.hc_mult, -1)
            .contiguous()
        )
        position_embeddings = {
            kind: self.rotary_emb(
                inputs_embeds, position_ids=position_ids, layer_type=kind
            )
            for kind in ("main", "compress")
        }
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        hidden_states = self.norm(self.hc_head(hidden_states))
        return reference.MoeModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=return_cache
        )

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

    decoder_structure = DecoderStructure(
        decoder="model",
        embedding="model.embed_tokens",
        blocks="model.layers",
        norm="model.norm",
        head="lm_head",
        attention="self_attn",
    )

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        initialize_model_weights(module, self.config.initializer_range)

    @classmethod
    def is_custom_code(cls) -> bool:
        return False

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return True

    def __init__(self, config: DeepseekV4Config) -> None:
        validate_native_execution_policy(config, model_name="DeepSeek V4")
        validate_output_geometry(config)
        validate_layer_schedule(config)
        validate_routing_config(config)
        validate_attention_backend(config)
        configure_reference_experts(config)
        reference.DeepseekV4PreTrainedModel.__init__(self, config)
        self.model = DeepseekV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = build_lm_head(
            config.hidden_size,
            config.vocab_size,
            compute_dtype=getattr(config, "compute_dtype", None),
        )
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
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

    def materialization_buffers(
        self, device: torch.device
    ) -> dict[tuple[nn.Module, str], torch.Tensor]:
        """Rebuild DeepSeek rotary and hash-routing buffers after meta allocation."""
        buffers: dict[tuple[nn.Module, str], torch.Tensor] = {}
        for module in self.modules():
            if isinstance(module, DeepseekV4RotaryEmbedding):
                with torch.device("cpu"):
                    fresh = DeepseekV4RotaryEmbedding(self.config)
                for name, value in fresh.named_buffers(recurse=False):
                    buffers[module, name] = value.to(
                        device=device, dtype=module._buffers[name].dtype
                    )
            elif isinstance(module, DeepseekV4HashRouter):
                with torch.device("cpu"):
                    fresh = DeepseekV4HashRouter(self.config)
                buffers[module, "tid2eid"] = fresh.tid2eid.to(
                    device=device, dtype=module._buffers["tid2eid"].dtype
                )
        return buffers

    def initialize_parameters(self) -> None:
        """Initialize fresh training weights and a balanced hash-routing table.

        The reference initializer zeros the persistent table because released
        weights supply it. Fresh random training has no such checkpoint: assign
        consecutive slots modulo the expert count, without consuming extra RNG.
        This entry point runs before checkpoint loading, never after it.
        """
        if any(parameter.is_meta for parameter in self.parameters()):
            raise RuntimeError(
                "DeepSeek V4 parameters must be materialized before initialization"
            )
        self.apply(self._init_weights)
        self.initialize_training_buffers()

    def initialize_training_buffers(self) -> None:
        """Populate fresh-training routing state without reinitializing weights.

        Call only for random training, before loading any saved state. Keeping
        this separate avoids replaying the eager constructor's weight RNG.
        """
        routers = [
            module
            for module in self.modules()
            if isinstance(module, DeepseekV4HashRouter)
        ]
        for router in routers:
            if router.tid2eid.is_meta:
                raise RuntimeError("Hash routing buffers must be materialized")
            if not 0 < router.top_k <= router.num_experts:
                raise ValueError("Hash router top-k must be within the expert count")
        with torch.no_grad():
            for router in routers:
                table = router.tid2eid
                slots = torch.arange(table.numel(), device=table.device)
                table.copy_((slots % router.num_experts).reshape(table.shape))

    @property
    def context_parallel_backend(self) -> str:
        """Select DeepSeek's model-owned compressor CP implementation."""
        return "compressor"

    def set_cp_mesh(
        self,
        mesh: Any,
        *,
        load_balancer: str | None = None,
        strategy: str = "kv_allgather",
        mask_mesh: Any = None,
    ) -> None:
        """Record the mesh used by the model-owned DeepSeek CP attention path."""
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Expose DeepSeek's text input contract to the Trainer."""
        return prepare_text_inputs(
            self,
            batch,
            parallel_dims=parallel_dims,
            max_context_length=max_context_length,
            preserve_batch_rows=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        attention_masks: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        skip_lm_head: bool = False,
        input_batch_shape: tuple[int, int] | None = None,
        labels: torch.Tensor | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Accept the flat-token Trainer contract and preserve HF outputs."""
        if positions is not None and kwargs.get("position_ids") is not None:
            raise ValueError("Specify only one of positions and position_ids")
        if attention_masks is not None and kwargs.get("attention_mask") is not None:
            raise ValueError("Specify only one of attention_masks and attention_mask")
        flat_input = input_ids is not None and input_ids.ndim == 1
        if input_batch_shape is not None:
            if not flat_input:
                raise ValueError("input_batch_shape requires flat Trainer tokens")
            input_ids = input_ids.reshape(input_batch_shape)
            if labels is not None and labels.ndim == 1:
                # Causal shifting must stop at each independent sample boundary.
                labels = labels.reshape(input_batch_shape)
        elif flat_input:
            input_ids = input_ids.unsqueeze(0)
        if positions is not None:
            if (
                positions.ndim == 1
                and input_batch_shape is not None
                and positions.numel() == input_ids.numel()
            ):
                positions = positions.reshape(input_batch_shape)
            kwargs["position_ids"] = (
                positions.unsqueeze(0) if positions.ndim == 1 else positions
            )
        if attention_masks is not None:
            kwargs["attention_mask"] = attention_masks
        elif padding_mask is not None and kwargs.get("attention_mask") is None:
            keep_mask = (~padding_mask).to(torch.long)
            if keep_mask.ndim == 1 and input_batch_shape is not None:
                keep_mask = keep_mask.reshape(input_batch_shape)
            kwargs["attention_mask"] = (
                keep_mask.unsqueeze(0) if keep_mask.ndim == 1 else keep_mask
            )
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
        # Decoder's named inputs are not custom loss-function extras.
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
            aux_loss = reference.load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                kwargs.get("attention_mask"),
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
        return_dict = self.config.return_dict if return_dict is None else return_dict
        return output if return_dict else output.to_tuple()


__all__ = [
    "DeepseekV4Config",
    "DeepseekV4DecoderLayer",
    "DeepseekV4Experts",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4SparseMoeBlock",
]
