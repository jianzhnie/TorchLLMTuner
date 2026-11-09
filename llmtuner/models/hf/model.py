"""A thin wrapper that plugs a HuggingFace model into a training loop.

This is the whole model layer: build a HF ``ForCausalLM`` from its config, give
the training loop a uniform ``(tokens, positions) -> logits`` entry point, and
expose the decoder's parts under the names the rest of the framework expects
(``tok_embeddings`` / ``layers`` / ``norm`` / ``lm_head`` / ``rotary_emb``).

Deliberately thin. An earlier version of this file (torchtitan's
``experiments/transformers_modeling_backend/model.py``) carried a 230-line
monkey-patch of ``PreTrainedModel._init_weights`` that re-derived every weight
with torchtitan's depth-scaled scheme, plus a ``PretrainedConfig`` subclass that
synced torchtitan and HF attribute names. Both existed to make checkpoints
bit-compatible with torchtitan's native models. llmtuner trains HF models with
HF's own initialization, so neither is needed -- ``from_config`` produces a
fully initialized model and that is the one we train.

What replaces them is per-forward work the HF model does not do on its own:

* attention is routed through flex attention so packed-document masking can be
  expressed (see ``get_attention_masks``),
* RoPE is driven by explicit ``positions`` rather than an ``arange``, because
  packed samples reset their positions at each document boundary.

Forward shape: this wrapper is a *decoder wrapper*, not a CausalLM. It runs the
decoder and applies ``lm_head`` itself, so ``self.model.model`` is the bare text
stack. Callers pass flat ``(T,)`` token and position tensors; the batch dim is
added and removed internally. ``forward(..., skip_lm_head=True)`` returns the
hidden states instead, for the trainer's chunked-loss path.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from ...accelerator.collectives import synchronize_local_errors
from ...accelerator.distributed import get_rank
from ...datasets.contracts import Batch, TrainerBatch
from ...parallel.topology import ParallelTopology
from ...utils.logging import get_logger
from ..common.moe.layer import collect_moe_layers
from .attention_backend import (
    ATTENTION_BACKEND,
    CP_SDPA_BACKEND,
    EAGER_FLEX_BACKEND,
    available_attention_backend,
    register_attention_backend,
    uses_deepseek_sparse_attention,
)
from .attention_masks import (
    build_dense_attention_mask,
    get_attention_masks,
    get_cp_attention_masks,
    mask_dtype,
)
from .expert_kernels import stabilize_moe_experts, stabilize_moe_router
from .factory import resolve_model_class, unwrap_text_config
from .gdn_numerics import (
    configure_gdn_numerics,
    validate_gdn_numerical_batch,
    validate_gdn_numerical_config,
    validate_gdn_numerical_sequence,
)
from .input_preparation import preprocess_inputs
from .linear_attention import (
    apply_linear_attention_backend,
    validate_linear_attention_batch,
    validate_linear_attention_config,
    validate_linear_attention_sequence,
)
from .materialization import validate_rotary_initialization
from .model_setup import (
    apply_cast_lm_head,
    configure_experts_implementation,
    resolve_submodule_name,
    validate_head_config,
)
from .normalization_setup import stabilize_residual_norm_gradients
from .qkv_projection import (
    apply_fp32_rotary_projection,
    apply_fused_qkv_projection,
)

logger = get_logger(__name__)

__all__ = ["HFTransformerModel"]


class HFTransformerModel(nn.Module):
    """A HF decoder stack behind a uniform training forward.

    Args:
        config: the HF config to instantiate (see ``build_model_config``).
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        config = unwrap_text_config(config)
        validate_linear_attention_config(config)
        validate_gdn_numerical_config(config)
        validate_rotary_initialization(config)
        # DSA models take a dense additive mask instead of a flex BlockMask;
        # resolved once here rather than per forward. The mask family is decided
        # in ``get_attention_masks``, which is the one place that knows both the
        # model and the batch's positions.
        self.uses_dsa = uses_deepseek_sparse_attention(config)
        validate_head_config(config)
        config._attn_implementation = (
            "eager"
            if config.model_type == "deepseek_v4"
            else available_attention_backend()
        )
        attention_backend = getattr(config, "attention_backend", "auto")
        if attention_backend == "flex":
            config._attn_implementation = ATTENTION_BACKEND
        elif attention_backend == "eager_flex":
            if config.model_type != "qwen3":
                raise ValueError(
                    "model.attention_backend='eager_flex' requires dense Qwen3"
                )
            config._attn_implementation = EAGER_FLEX_BACKEND
        register_attention_backend()

        model_cls = resolve_model_class(config)
        configure_experts_implementation(config, model_cls=model_cls)
        self.model = model_cls(config=config)
        configure_gdn_numerics(self.model)
        apply_linear_attention_backend(self.model)
        stabilize_residual_norm_gradients(self.model)
        stabilize_moe_router(self.model)
        stabilize_moe_experts(self.model)
        self.model.config._attn_implementation = config._attn_implementation

        if getattr(config, "qkv_projection", "separate") != "separate":
            apply_fused_qkv_projection(self.model)
        elif getattr(config, "rotary_compute_dtype", None) == "float32":
            apply_fp32_rotary_projection(self.model)

        apply_cast_lm_head(self.model, config)

        self.cp_mesh = None
        self._cp_load_balancer = None
        self._cp_strategy = "kv_allgather"

        # The decoder is the text stack; lm_head is its sibling on the CausalLM.
        # Stored with object.__setattr__ on purpose: a plain ``self._decoder = ...``
        # would go through nn.Module.__setattr__, which registers it in
        # ``_modules`` and makes every parameter appear twice in the state dict
        # (once under ``model.``, once under ``_decoder.``).
        #
        # The attribute names are resolved once here; the accessors below are
        # properties so a pipeline stage can swap a part out in place.
        object.__setattr__(self, "_decoder", self.model.model)
        object.__setattr__(
            self,
            "_embed_name",
            resolve_submodule_name(
                self._decoder, ("embed_tokens", "wte"), component="token embedding"
            ),
        )
        object.__setattr__(
            self,
            "_norm_name",
            resolve_submodule_name(
                self._decoder,
                ("norm", "final_layernorm", "ln_f"),
                component="final norm",
            ),
        )

    # -- parts, under the names the rest of the framework uses ------------------

    @property
    def tok_embeddings(self) -> nn.Module:
        return getattr(self._decoder, self._embed_name)

    @tok_embeddings.setter
    def tok_embeddings(self, value: nn.Module) -> None:
        setattr(self._decoder, self._embed_name, value)

    @property
    def layers(self) -> nn.ModuleList:
        return self._decoder.layers

    @layers.setter
    def layers(self, value) -> None:
        self._decoder.layers = value

    @property
    def norm(self) -> nn.Module:
        return getattr(self._decoder, self._norm_name)

    @norm.setter
    def norm(self, value: nn.Module) -> None:
        setattr(self._decoder, self._norm_name, value)

    @property
    def vocab_size(self) -> int | None:
        """The vocabulary the ``lm_head`` was built for, or ``None`` if unnamed.

        Read by the loss seam: a vocab-parallel cross-entropy slices the
        shards against the *global* class count, and the only value that keeps
        a replicated head on the plain path is the one the head was built with
        -- the HF config's, which for a hub id or a local checkpoint directory
        need not equal the ``ModelConfig.vocab_size`` field.
        """
        return getattr(self.model.config, "vocab_size", None)

    @property
    def lm_head(self) -> nn.Module | None:
        return getattr(self.model, "lm_head", None)

    @lm_head.setter
    def lm_head(self, value: nn.Module | None) -> None:
        self.model.lm_head = value

    @property
    def rotary_emb(self) -> nn.Module | None:
        return getattr(self._decoder, "rotary_emb", None)

    @rotary_emb.setter
    def rotary_emb(self, value: nn.Module | None) -> None:
        self._decoder.rotary_emb = value

    @property
    def uses_flex_attention(self) -> bool:
        """Whether the decoder routes attention through the flex kernel.

        Read by the compile step: flex has only an inductor lowering, so a
        non-inductor compile backend must either scoop the flex region
        (``aot_eager`` + regional_inductor) or be rejected.
        """
        return self.model.config._attn_implementation == ATTENTION_BACKEND

    @property
    def enable_weight_tying(self) -> bool:
        """Whether ``lm_head`` and the embedding share one ``Parameter``.

        Read by FSDP, which must not let one Parameter be owned by two FSDP
        units; the two modules are wrapped together when this is true.

        Compared by identity rather than by ``config.tie_word_embeddings``: that
        flag records intent, and a model may carry an unshared head anyway (or
        share one the flag does not mention). Identity is also exactly the check
        FSDP2 itself performs, so the answer here matches what FSDP will do.
        """
        embed, head = self.tok_embeddings, self.lm_head
        if head is None:
            return False
        weight = getattr(embed, "weight", None)
        return weight is not None and weight is getattr(head, "weight", None)

    # -- HF integration hooks --------------------------------------------------

    def set_cp_mesh(
        self,
        mesh,
        *,
        load_balancer: str | None = None,
        strategy: str = "kv_allgather",
        mask_mesh=None,
    ) -> None:
        """Record the CP mesh so logit dumps can tag their CP coordinate.

        Also records the CP load-balancer type and strategy: the trainer shards
        the batch with the balancer, and the forward's BlockMask handling must
        match the strategy -- kv_allgather attends gathered K/V against a
        Q-sharded mask, while ulysses reassembles the full sequence in the
        all-to-all and needs a packed corpus's document mask full-length.
        With dual CP axes, ``mask_mesh`` selects the KV axis because Ulysses
        assembles queries within each KV shard before attention.
        """
        self.cp_mesh = mesh
        # Dual CP gathers queries within Ulysses, so masks shard only over KV.
        # Token inputs and positions still shard over the total CP mesh.
        self._cp_mask_mesh = mesh if mask_mesh is None else mask_mesh
        self._cp_load_balancer = load_balancer
        self._cp_strategy = strategy

    @property
    def tp_config(self) -> dict[str, str]:
        """HF's TP plan, with patterns rewritten to THIS wrapper's module paths.

        HF states its plan relative to the model it ships -- ``layers.*.q_proj``,
        plus a ``model.``-prefixed variant for families that nest one level
        deeper. The parallel layer, however, walks *this* wrapper, whose
        ``named_modules`` paths all sit under ``model.`` because that is the
        attribute the HF CausalLM is held in. Prefixing every pattern with
        ``model.`` maps one spelling onto the other, so HF's own two variants
        both resolve here and no call site has to know either layout.

        Without this the plan is simply not found (the attribute lives on the
        inner HF model) and ``apply_tp`` silently shards nothing -- a replicated
        run that looks like a working one. That failure is what makes this
        translation load-bearing rather than cosmetic.

        Returns ``{}`` when the model ships no plan. The parallel entry point
        rejects an enabled TP run without a plan or explicit pattern overrides.

        Reads the inner model's ``_tp_config`` first and falls back to its
        ``tp_config`` property only when the attribute carries nothing. HF builds
        the instance ``_tp_config`` from ``config.base_model_tp_config`` plus each
        child module's own plan, which is the TP plan this layer wants; the
        property additionally switches to ``_ep_config`` when the config asks for
        expert parallelism, and those are EP specs, not TP ones. The fallback is
        for a model that exposes the plan through the property alone. Versions
        exposing ``_tp_plan`` / ``tp_plan`` are supported through the same
        pattern translation.
        """
        plan = getattr(self.model, "_tp_config", None) or {}
        if not plan:
            plan = getattr(self.model, "tp_config", None) or {}
        if not plan:
            plan = getattr(self.model, "_tp_plan", None) or {}
        if not plan:
            plan = getattr(self.model, "tp_plan", None) or {}
        return {f"model.{pattern}": spec for pattern, spec in plan.items()}

    def named_children(self):
        """Present the decoder's parts as direct children.

        ``nn.Module.named_children`` would yield exactly one child (``self.model``),
        so the parallelism layer, which walks children, would see a single opaque
        blob. Yielding the parts here is what lets it address the decoder's pieces
        (``layers.*`` and friends) directly.

        This does NOT flatten ``state_dict`` keys: the state dict is built from
        ``_modules``, which still holds everything under ``self.model``, so keys
        keep their ``model.`` prefix. Only the child *iteration* is reshaped.
        The registered parent follows the standard parts so recursive walkers
        also discover architecture-specific state. Consumers selecting parts
        must preserve this structural parent rather than replace it.
        """
        yield "tok_embeddings", self.tok_embeddings
        yield "layers", self.layers
        yield "norm", self.norm
        if self.lm_head is not None:
            yield "lm_head", self.lm_head
        if self.rotary_emb is not None:
            yield "rotary_emb", self.rotary_emb
        # Preserve traversal of architecture-specific parameters and buffers.
        # FSDP's visited-module set skips the parts already yielded above and
        # discovers remaining state through its actual registered parent.
        yield "model", self.model

    def preprocess_inputs(
        self,
        input_dict: Batch | TrainerBatch,
        *,
        parallel_dims: ParallelTopology | None,
        max_context_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Normalize and shard a batch through the model input pipeline."""
        config = self.model.config
        numerical_policy = getattr(config, "gdn_numerical_policy", "native")
        allow_right_padding = getattr(config, "gdn_allow_right_padding", False)
        if getattr(config, "linear_attention_backend", "native") == "npu_triton":
            validate_linear_attention_batch(
                input_dict,
                allow_right_padding=allow_right_padding,
                validate_metadata=numerical_policy == "native",
            )
        if numerical_policy != "native":
            validate_gdn_numerical_batch(
                input_dict,
                allow_right_padding=allow_right_padding,
            )
        return preprocess_inputs(
            self,
            input_dict,
            parallel_dims=parallel_dims,
            max_context_length=max_context_length,
        )

    def get_attention_masks(self, positions: torch.Tensor) -> BlockMask | torch.Tensor:
        return get_attention_masks(self, positions)

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        attention_masks=None,
        padding_mask: torch.Tensor | None = None,
        skip_lm_head: bool = False,
    ) -> torch.Tensor:
        """Run the decoder over one packed sequence and return logits.

        Args:
            input_ids: ``(T,)`` flat token ids. Under CP, this rank's sequence
                shard -- ``(T/cp,)``. Under pipeline parallelism, a non-first
                stage receives the previous stage's output instead: ``(T, H)``
                hidden states, detected by ``tok_embeddings`` having been split
                out (replaced by an ``nn.Identity``).
            positions: ``(T,)`` per-token positions, resetting at document
                boundaries. Drives RoPE. Defaults to ``arange``, which is correct
                only when the sequence is a single document. Under CP, the
                matching shard of the full positions.
            attention_masks: a prebuilt BlockMask for flex or a dense tensor
                for packed Ulysses CP SDPA. Ulysses retains the full sequence
                mask; KV all-gather flex uses a Q-sharded BlockMask.
            padding_mask: ``(T,)`` boolean, true for padding, sharded exactly
                like ``input_ids``. Staged onto every swapped-in MoE block
                (``MoE.set_padding_mask``) so the blocks' load-balancing
                statistics skip padding tokens; the decoder's arithmetic is
                unaffected. The HF decoder layer's fixed
                ``self.mlp(hidden_states)`` call signature is why this is a
                staged side channel rather than a threaded argument.
            skip_lm_head: return the ``(T, H)`` hidden states instead of logits.
                The chunked-loss path uses this so the trainer can run lm_head +
                cross-entropy per sequence chunk (see
                ``components.loss.chunked_lm_head_cross_entropy``) rather than
                materialize the full ``T * V`` logits.
        """
        config = self.model.config
        numerical_policy = getattr(config, "gdn_numerical_policy", "native")
        allow_right_padding = getattr(config, "gdn_allow_right_padding", False)
        if getattr(config, "linear_attention_backend", "native") == "npu_triton":
            validate_linear_attention_sequence(
                input_ids,
                positions,
                attention_masks,
                padding_mask,
                allow_right_padding=allow_right_padding,
                validate_metadata=numerical_policy == "native",
            )
        if numerical_policy != "native":
            validate_gdn_numerical_sequence(
                input_ids,
                positions,
                attention_masks,
                padding_mask,
                allow_right_padding=allow_right_padding,
            )
        if isinstance(self.tok_embeddings, nn.Identity):
            # Non-first pipeline stage: the input IS the previous stage's
            # hidden states, so the embedding lookup is skipped by feeding the
            # decoder ``inputs_embeds`` directly. PP does not shard the
            # sequence, so the local ``arange`` positions default stays right.
            decoder_input = {"inputs_embeds": input_ids.unsqueeze(0)}
        else:
            decoder_input = {"input_ids": input_ids.unsqueeze(0)}
        local_seq_len = input_ids.shape[0]
        if positions is None:
            positions = torch.arange(local_seq_len, device=input_ids.device)

        if self.cp_mesh is not None and self._cp_strategy == "compressor_ulysses":
            with synchronize_local_errors(
                "HCA CP wrapper input", group=self.cp_mesh.get_group()
            ):
                if padding_mask is not None and padding_mask.any():
                    raise ValueError("HCA CP padding is not yet implemented")
                if attention_masks is not None:
                    raise ValueError("HCA CP constructs global causal masks internally")
        kwargs = self._apply_attention(positions, attention_masks)
        if (
            self.cp_mesh is None
            and positions.numel() != local_seq_len
            and self.model.config._attn_implementation == "sdpa"
            and kwargs.get("attention_mask") is None
        ):
            # HF builds masks before attention gathers the TP token shards.
            # Build against the full attention sequence, preserving HF's causal
            # and sliding-window rules in eager and compiled execution alike.
            mask_inputs = input_ids.new_empty(
                (1, positions.numel(), 1), dtype=torch.float32
            )
            mask_kwargs = {
                "config": self.model.config,
                "inputs_embeds": mask_inputs,
                "attention_mask": None,
                "past_key_values": None,
                "position_ids": positions.unsqueeze(0),
            }
            masks = {"full_attention": create_causal_mask(**mask_kwargs)}
            if getattr(self._decoder, "has_sliding_layers", False):
                masks["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs
                )
            if hasattr(self._decoder, "has_sliding_layers"):
                kwargs["attention_mask"] = masks
            elif getattr(self.model.config, "sliding_window", None) is not None:
                kwargs["attention_mask"] = create_sliding_window_causal_mask(
                    **mask_kwargs
                )
            else:
                kwargs["attention_mask"] = masks["full_attention"]

        # Preserve the collator's positions, including restarted padded suffix.
        # A real HF 2D validity mask prevents its mask builder from mistaking
        # that suffix for another packed document and reaches GDN validators.
        # Per-call kwargs (also retained by AC) avoid mutable module mask state.
        if allow_right_padding:
            if padding_mask is not None and bool(padding_mask.any()):
                if kwargs.get("attention_mask") is not None:
                    raise ValueError("GDN right padding requires native HF 2D masks")
                kwargs["attention_mask"] = (~padding_mask).unsqueeze(0)

        # Stage the padding mask on every MoE block -- including ``None``, so
        # a mask from a previous microbatch can never survive into this one.
        # Full/selective checkpoint contexts retain each forward's mask for
        # replay even when another microbatch re-stages these blocks.
        for moe in collect_moe_layers(self):
            moe.set_padding_mask(padding_mask)

        if self.model.config.model_type == "kimi_linear" and padding_mask is not None:
            kwargs["valid_token_mask"] = (~padding_mask).unsqueeze(0)

        # A HF decoder expects a batch dim; the wrapper's contract is flat.
        hidden_states = self._decoder(
            **decoder_input,
            position_ids=positions.unsqueeze(0),
            use_cache=False,
            **kwargs,
        ).last_hidden_state.squeeze(0)

        if (
            self.lm_head is not None
            and not isinstance(self.lm_head, nn.Identity)
            and not skip_lm_head
        ):
            logits = self.lm_head(hidden_states)
        else:
            # Non-final pipeline stage, or a chunked-loss forward: the output
            # must own its storage rather than be a view of the decoder's
            # ``last_hidden_state`` (squeeze above), because split-backward
            # schedules (ZBVZeroBubble) call ``detach_()`` on stage outputs,
            # which views do not support. The chunked-loss caller does not
            # detach in place, but shares the branch so both skip reasons stay
            # one code path -- the clone is one T*H copy per forward.
            logits = hidden_states.clone()

        # Debug hook, off by default: set ``model.logit_dump_dir`` to a
        # directory to dump this rank's logits on every forward (used by
        # numerical comparison harnesses; not a config surface).
        if getattr(self, "logit_dump_dir", None) is not None:
            self._maybe_dump_logits(self.logit_dump_dir, logits)

        return logits

    def _apply_attention(
        self, positions: torch.Tensor, attention_masks
    ) -> dict[str, Any]:
        """Select the decoder's mask representation for the active backend.

        Build only the representation consumed by the backend: Flex uses a
        BlockMask, while tensor backends use a dense document mask for packed
        input or let the decoder construct its own causal mask.
        """
        if self.cp_mesh is not None and self._cp_strategy == "compressor_ulysses":
            if attention_masks is not None:
                raise ValueError("HCA CP constructs global causal masks internally")
            return {"attention_mask": None}

        if self.cp_mesh is not None and self._cp_strategy == "kcp":
            if getattr(self.model.config, "attn_mask_type", "causal") != "causal":
                raise ValueError("KCP does not support packed document attention")
            if attention_masks is not None:
                raise ValueError(
                    "KCP constructs causal masks from local token validity"
                )
            # KDA consumes local validity; the MLA Ulysses callable reconstructs
            # global causality after exchange. A flex BlockMask would bypass
            # that contract and describe a different query/key layout.
            return {"attention_mask": None}

        if self.model.config._attn_implementation == CP_SDPA_BACKEND:
            if isinstance(attention_masks, torch.Tensor):
                return {"attention_mask": attention_masks}
            if (
                getattr(self.model.config, "_cp_sdpa_dual", False)
                or self._cp_strategy == "kv_allgather"
            ) and getattr(
                self.model.config, "attn_mask_type", "causal"
            ) == "block_causal":
                raise ValueError(
                    "Packed KV SDPA CP requires the full preprocessed document mask"
                )
            # The CP sdpa kernel (context_parallel/sdpa_attention.py) builds the
            # mask it needs from the CP offsets: this rank's queries against the
            # gathered full sequence. Handing the decoder a BlockMask here would
            # both waste the build and invite a Q-sharded mask into a call that
            # already attended the full sequence.
            #
            # Exception: the *ulysses* sdpa kernel attends the full sequence
            # after its head swap, so a packed corpus's document mask applies
            # unsharded -- exactly as it does for flex, and it is the one thing
            # that mask cannot be derived from a length. Preprocess it before
            # positions are sharded; local positions lose document boundaries. The
            # kernel adds no causal mask, only an optional sliding window.
            if (
                getattr(self.model.config, "_cp_sdpa_ulysses", False)
                and getattr(self.model.config, "attn_mask_type", "causal")
                == "block_causal"
            ):
                raise ValueError(
                    "Packed Ulysses SDPA CP requires the full preprocessed "
                    "document mask"
                )
            return {"attention_mask": None}

        if self.cp_mesh is None and isinstance(attention_masks, torch.Tensor):
            if self.model.config._attn_implementation != ATTENTION_BACKEND:
                return {"attention_mask": attention_masks}

        if attention_masks is None and (
            self.cp_mesh is not None
            or self.model.config._attn_implementation == ATTENTION_BACKEND
        ):
            if self.cp_mesh is None:
                attention_masks = self.get_attention_masks(positions=positions)
            else:
                attention_masks = get_cp_attention_masks(self, positions)

        # is_causal is the flex-only spelling: it selects which mod the BlockMask
        # runs, so it is withheld from every other backend (see below).
        if self.model.config._attn_implementation == ATTENTION_BACKEND:
            return {"attention_mask": attention_masks, "is_causal": False}

        # Every other backend wants a tensor mask (or none), so a packed corpus
        # -- the one case whose mask is not a function of the length alone --
        # is materialized as a dense additive ``[1, 1, T, T]`` mask: causal AND
        # same-document, which is what the BlockMask computes for flex. That is
        # O(T**2) memory per forward (32 MiB of bf16 at T=4096), so flex remains
        # the better path where it exists; this fallback is what lets a packed
        # corpus run on a backend without it (NPU). DeepSeek-style sparse
        # attention already gets the same dense mask for the same reason (see
        # ``get_attention_masks``), so the two paths cannot disagree about what
        # "block_causal" allows.
        #
        # Packing is detected from ``attn_mask_type``, not from scanning
        # ``positions`` for restarts: a restart scan misses a boundary after a
        # length-1 document (positions ``[0, 0, ...]`` never descend), which
        # would let sdpa run plain causal attention across the boundary with no
        # complaint. The flag is derived from the corpus at config build time
        # (``build_model_config_for``), so it cannot disagree with the batch.
        #
        # The mask is built at ``positions``' length, which under TP sequence
        # parallelism is the *gathered* length the TP boundary attends over.
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            return {
                "attention_mask": build_dense_attention_mask(
                    positions,
                    dtype=mask_dtype(self),
                    block_causal=True,
                )
            }

        # A single causal document. Hand the decoder NOTHING and let HF follow its
        # own path: it builds a 4D mask only when it must (sliding-window layers,
        # padding) and otherwise returns None, and its sdpa wrapper derives
        # ``is_causal`` from whether a mask is present. Passing ``is_causal=True``
        # ourselves would double up with the mask HF does build -- sdpa rejects
        # ``attn_mask`` together with ``is_causal=True`` -- which is what an
        # earlier version of this method got wrong.
        return {"attention_mask": None}

    def _maybe_dump_logits(self, dump_dir: str, logits: torch.Tensor) -> None:
        """Append this rank's logits (one entry per forward) for numerical tests."""
        rank = get_rank()
        cp_coord = self.cp_mesh.get_local_rank() if self.cp_mesh is not None else 0
        recs = getattr(self, "_logit_dump_recs", None)
        if recs is None:
            recs = self._logit_dump_recs = []
        recs.append((cp_coord, logits.detach().float().cpu()))
        torch.save(recs, os.path.join(dump_dir, f"logits_rank{rank}.pt"))

    def __setattr__(self, name, value) -> None:
        """Route property-backed names through their setters.

        ``nn.Module.__setattr__`` registers modules directly, which would bypass
        the property setters defined above (``self.layers = ...`` would create a
        new child instead of replacing the decoder's).
        """
        prop = getattr(type(self), name, None)
        if isinstance(prop, property) and prop.fset is not None:
            prop.fset(self, value)
            return
        super().__setattr__(name, value)
