# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Constructors and forward math adapted from Transformers modeling_qwen3_moe.
"""Model-owned Qwen3 MoE assembly and execution with reusable HF components.

Keep child order and post_init boundaries identical to the reference model:
they determine parameter initialization, RNG consumption and checkpoint names.
Device policies and parallel transformations are installed by their own layers.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from torch import Tensor
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
)

from experiments.pipeline_parallel.model_execution import PipelineStageIO
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.contracts import DecoderStructure
from llmtuner.models.qwen.qwen3.model import (
    Qwen3MoeForCausalLM as BaseQwen3MoeForCausalLM,
)
from llmtuner.models.qwen.qwen3.model import (
    validate_attention_geometry,
)


class Qwen3MoeForCausalLM(BaseQwen3MoeForCausalLM):
    """Explicit causal model tree, preserving reference state and loss APIs."""

    decoder_structure = DecoderStructure(
        decoder="model",
        embedding="model.embed_tokens",
        blocks="model.layers",
        norm="model.norm",
        head="lm_head",
        attention="self_attn",
    )

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
        self.pipeline_stage_io = PipelineStageIO(
            first=first,
            last=last,
            hidden_size=self.config.hidden_size,
            activation_dtype=next(self.parameters()).dtype,
        )

    def forward_pipeline_stage(
        self,
        inputs: Tensor,
        *,
        positions: Tensor | None,
        attention_mask: Tensor | None,
        padding_mask: Tensor | None,
    ) -> Tensor:
        """Run tokens on the first stage and hidden states on later stages."""
        first, last = self.pipeline_stage_io.first, self.pipeline_stage_io.last
        self.pipeline_stage_io.validate_input(inputs)
        token_shape = tuple(inputs.shape if first else inputs.shape[:-1])
        flat = inputs.ndim == (1 if first else 2)
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
            padding_mask=padding_mask,
        ).last_hidden_state
        if last:
            output = self.lm_head(output)
        # Stage sends must own storage, including an Identity-only stage.
        output = (output.squeeze(0) if flat else output).clone()
        self.pipeline_stage_io.validate_output(
            output,
            token_shape=token_shape,
            output_width=(
                self.vocab_size if last else self.pipeline_stage_io.hidden_size
            ),
        )
        return output

    @property
    def tp_config(self) -> Mapping[str, str]:
        """Declare exact targets relative to this instance, not HF patterns.

        Embedding/head stay replicated by default, preserving tied weights and
        existing training behavior. Vocabulary sharding remains an explicit
        parallel override. Resolve again after PP changes local ownership.
        """
        plan = (
            {"lm_head": "replicated_with_grad_allreduce"}
            if not hasattr(self, "pipeline_stage_io") or self.pipeline_stage_io.last
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
        if hasattr(self, "pipeline_stage_io"):
            if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise UnsupportedCombinationError(
                    "Qwen3 PP must preserve every token row; logits_to_keep must be 0"
                )
            if kwargs:
                raise UnsupportedCombinationError(
                    f"Qwen3 PP does not support extra forward options: {sorted(kwargs)}"
                )
            if skip_lm_head:
                raise UnsupportedCombinationError(
                    "Qwen3 PP requires its declared stage output; "
                    "hidden-only requests are unsupported"
                )
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
            )
        return super().forward(
            input_ids,
            positions=positions,
            attention_masks=attention_masks,
            padding_mask=padding_mask,
            skip_lm_head=skip_lm_head,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_router_logits=output_router_logits,
            logits_to_keep=logits_to_keep,
            return_dict=return_dict,
            **kwargs,
        )

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
        stage_io = getattr(self, "pipeline_stage_io", None)
        first, last = (stage_io.first, stage_io.last) if stage_io else (True, True)
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
