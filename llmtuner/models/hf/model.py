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
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.attention.flex_attention import BlockMask, and_masks
from transformers.configuration_utils import PretrainedConfig

from llmtuner.errors import UnsupportedCombinationError

from ...accelerator import distributed
from ...components.loss import next_token_targets
from ...datasets.contracts import Batch, TrainerBatch
from ...parallel.context_parallel.input_sharding import (
    shard_attention_mask_for_cp,
    shard_batch_for_cp,
    shard_batch_for_tp,
    shard_padding_mask_for_cp,
    shard_padding_mask_for_tp,
)
from ...parallel.topology import ParallelTopology
from ...utils.logging import get_logger
from ..common.attention.masks import (
    build_dense_attention_mask,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from ..common.moe.layer import collect_moe_layers
from .attention_backend import (
    ATTENTION_BACKEND,
    available_attention_backend,
    register_attention_backend,
    uses_deepseek_sparse_attention,
)
from .factory import resolve_model_class, unwrap_text_config
from .model_setup import (
    apply_cast_lm_head,
    configure_experts_implementation,
    resolve_submodule_name,
    stabilize_residual_norm_gradients,
    validate_head_config,
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
        register_attention_backend()

        model_cls = resolve_model_class(config)
        configure_experts_implementation(config, model_cls=model_cls)
        self.model = model_cls(config=config)
        stabilize_residual_norm_gradients(self.model)
        self.model.config._attn_implementation = config._attn_implementation

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

    def _shard_for_cp(
        self,
        cp_mesh: DeviceMesh | None,
        *,
        packed: bool,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        positions: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        BlockMask | torch.Tensor | None,
    ]:
        """Steps 3-4 of ``preprocess_inputs``: mask full-length, then CP-shard.

        Returns ``(inputs, labels, positions, padding_mask, attention_masks)``;
        the last is ``None`` when no mask was built (CP on but not packed, or
        the attention backend consumes masks on its own).
        """
        if cp_mesh is None:
            attention_masks = None
            if positions is not None:
                mask = self.get_attention_masks(positions=positions)
                if self.model.config._attn_implementation == ATTENTION_BACKEND:
                    attention_masks = mask
            return inputs, labels, positions, padding_mask, attention_masks

        if positions is None:
            # The forward's own ``arange`` default would restart at 0 on
            # every rank; the shard needs positions that describe the whole
            # sequence.
            positions = torch.arange(inputs.numel(), device=inputs.device)
        # A causal-only mask (a single document) can be rebuilt from this
        # rank's positions shard, which is what ``_get_cp_attention_masks``
        # does. Packed cannot: ``positions`` is about to be sharded and the
        # document structure is not recoverable from a shard of it, so the
        # full-length mask is built first. What happens to it next depends
        # on the strategy: kv_allgather attends gathered full-length K/V
        # against local queries, so the mask is Q-sharded to match (the GQA
        # head count still divides by cp -- sharding Q does not change how
        # many Q heads a rank owns); ulysses all-to-all's the FULL sequence
        # onto every rank before attention, so the document mask stays
        # full-length and unsharded -- the varlen semantics, where the
        # document structure is global metadata that the token shard must
        # not cut. Both decisions are config-keyed, hence rank-symmetric.
        attention_masks = None
        if packed:
            attention_masks = self.get_attention_masks(positions=positions)
            if self._cp_strategy != "ulysses":
                attention_masks = shard_attention_mask_for_cp(
                    attention_masks,
                    getattr(self, "_cp_mask_mesh", cp_mesh),
                    self._cp_load_balancer,
                )
            if self.model.config._attn_implementation != ATTENTION_BACKEND:
                attention_masks = None
        inputs, labels, positions = shard_batch_for_cp(
            inputs,
            labels,
            positions,
            cp_mesh,
            load_balancer=self._cp_load_balancer,
        )
        if padding_mask is not None:
            # The mask describes the same token stream, so it takes the
            # same shard (load-balancer rearrangement included).
            padding_mask = shard_padding_mask_for_cp(
                padding_mask, cp_mesh, self._cp_load_balancer
            )
        return inputs, labels, positions, padding_mask, attention_masks

    def _shard_for_tp(
        self,
        tp_mesh: DeviceMesh | None,
        *,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        positions: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Step 4b of ``preprocess_inputs``: the TP (sequence-parallel) shard.

        Only the token-carrying tensors are cut; positions keep the CP-shard
        (or, with CP off, full) length because attention and RoPE see the
        sequence after the in-projection all-gather.
        """
        if tp_mesh is None:
            return inputs, labels, positions, padding_mask
        # Sequence parallelism premise (step 4b): cut the token-carrying
        # tensors along the TP axis. Positions are NOT cut -- after the
        # in-projection all-gather, RoPE and attention see the assembled
        # sequence, so they keep the CP-shard (or, with CP off, full) length.
        # Synthesize them full-length when the batch did not carry any: the
        # forward's own ``arange`` default would be sized to the TP-sharded
        # input and restart at 0 on every rank.
        if positions is None:
            positions = torch.arange(inputs.numel(), device=inputs.device)
        if padding_mask is not None:
            # Cut BEFORE the batch tensors: the mask's length still matches
            # the pre-shard ``inputs``, which is what the divisibility
            # check must measure.
            padding_mask = shard_padding_mask_for_tp(padding_mask, tp_mesh)
        inputs, labels = shard_batch_for_tp(inputs, labels, tp_mesh)
        return inputs, labels, positions, padding_mask

    def preprocess_inputs(
        self,
        input_dict: Batch | TrainerBatch,
        *,
        parallel_dims: ParallelTopology | None,
        max_context_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Turn a dataloader batch into ``(inputs, labels, extra_kwargs)``.

        The seam torchtitan's trainer calls, and the reason it is on the model
        rather than in the loop: every step below is a statement about *this*
        architecture's input contract, and one of them needs the full-length
        positions that only exist before the sequence is sharded.

        1. **Normalize the batch shape.** The two loaders disagree about what a
           batch is -- the synthetic one yields ``(B, T)`` rows of one document
           each, a Grain one a flat packed stream -- so they are reconciled
           here. ``num_valid_tokens`` is *not* read: the trainer pops it before
           this call, because the loss denominator has to be the pre-shard count
           and reduced across DP before the first backward.
        2. **Collapse the batch dim**, turning ``(B, T)`` into the flat ``(B*T,)``
           the forward takes.
        3. **Build the attention mask**, when the batch carries ``positions``.
           This has to happen before step 5: the document structure of a packed
           batch is a global property, so the mask is built over the FULL
           sequence and only then Q-sharded.
        4. **Shard for context parallelism**, positions included, so RoPE
           follows each token to its rank.
        4b. **Shard for tensor parallelism** (sequence parallelism): the fused
           TP GEMMs all-gather the sequence inside each projection, so the
           forward must be entered holding only this rank's ``T / tp`` slice --
           otherwise the gather concatenates ``tp`` copies of the full sequence
           and every sharded weight gradient comes out ``tp`` times too large.
           Only the token-carrying tensors (``inputs``, ``labels``) are cut:
           after the in-projection gather, attention and RoPE see the assembled
           sequence (the CP shard, or the full sequence with CP off), so
           ``positions`` and the mask keep their CP/full length. This matches
           torchtitan's joint ``(CP, TP)`` sequence sharding
           (``hf_sharding.py``'s ``PartitionSpec(DP, (CP, TP), None)``): the CP
           shard composed with a contiguous TP slice of it is exactly the joint
           (CP outer, TP inner) split.
        5. **Return the leftover dict as ``extra_kwargs``.** Those are splatted
           into ``forward``, so anything left here must be one of its keyword
           parameters -- ``positions``, ``attention_masks`` and
           ``padding_mask``, and nothing else.

        ``positions`` is optional: the synthetic source has none and the forward
        falls back to its own ``arange``, which is right for a single document
        but must not be relied on for a packed one. The CP load-balancer
        setting is latched onto this wrapper by the context-parallel entry
        point (see ``set_cp_mesh``).

        The reference also takes ``max_num_documents``, to size a fixed-shape
        varlen mask for CUDA graph capture. It is not accepted here because
        there is no path for it to take: llmtuner's CP path goes through
        ``create_attention_mask`` (a flex ``BlockMask``), which rebuilds from
        the positions shard and takes no capacity bound, and the packing
        collator already caps segments at one row. Accepting it would mean a
        parameter the trainer passes and this method silently discards.

        Tensors arrive on the trainer's device; the trainer moves them before
        calling, so this is pure structure and stays device-free.
        """
        extra_kwargs: dict[str, Any] = {}
        positions = None
        padding_mask = None

        if isinstance(input_dict, Batch):
            # Rows are independent documents of length T.
            labels = next_token_targets(
                input_dict.labels.reshape(-1), seq_len=input_dict.labels.shape[-1]
            )
            inputs = input_dict.input_ids
        else:
            # This wrapper calls a text decoder, with no vision encoder or
            # media/three-axis RoPE input path. Never train on placeholders
            # while silently discarding the media they describe.
            for field in (
                "pixel_values",
                "pixel_values_videos",
                "grid_thw",
                "grid_thw_videos",
                "mrope_positions",
            ):
                if input_dict.get(field) is not None:
                    raise UnsupportedCombinationError(
                        f"HFTransformerModel cannot consume {field}: its text "
                        "decoder does not implement multimodal inputs. Use a "
                        "text dataset or a model adapter that consumes media "
                        "and its positional metadata."
                    )
            # Packed stream: the collator already shifted and masked the labels
            # at every document boundary, so for this path the shift is a read.
            inputs, labels = input_dict["input"], input_dict["labels"]
            positions = input_dict.get("positions")
            # True-for-padding, from the collator; consumed by the MoE blocks'
            # load-balancing statistics (see forward's ``padding_mask``).
            padding_mask = input_dict.get("padding_mask")

        inputs, labels = inputs.reshape(-1), labels.reshape(-1)
        if positions is not None:
            positions = positions.reshape(-1)
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1).to(torch.bool)

        for name, value in (
            ("labels", labels),
            ("positions", positions),
            ("padding_mask", padding_mask),
        ):
            if value is not None and value.numel() != inputs.numel():
                raise ValueError(
                    f"{name} has {value.numel()} tokens, but input has "
                    f"{inputs.numel()}; batch token fields must have equal lengths"
                )

        # Whether the mask must be prebuilt full-length: a packed corpus
        # (``block_causal``) carries a document structure that is not
        # recoverable from a CP positions shard, so the mask must exist before
        # the shard below. This is decided by ``attn_mask_type``, NOT by
        # scanning ``positions`` for restarts: a restart scan misses the two
        # shapes a real packed corpus produces -- a document that fills the
        # whole row (the packing collator splits overlong documents into
        # single-document rows, whose positions are a plain arange) and a
        # boundary after a length-1 document (positions ``[0, 0, ...]`` have no
        # descending edge). The first would crash the CP forward for lack of a
        # prebuilt mask; the second would disarm the sdpa packed-guard and let
        # attention cross the boundary silently.
        packed = positions is not None and (
            getattr(self.model.config, "attn_mask_type", "causal") == "block_causal"
        )

        # Built before the CP shard, always from the FULL-length positions --
        # which is exactly why this lives here and not in the loop: after the
        # shard below, no rank holds a positions vector that can describe the
        # document structure. The prebuilt mask is then handed to the forward
        # below where the kernel can consume it (the CP/ulysses path keys its
        # dispatch off its Q length); otherwise ``_apply_attention`` rebuilds
        # one from ``positions``.
        cp_mesh = (
            None if parallel_dims is None else parallel_dims.get_optional_mesh("cp")
        )
        inputs, labels, positions, padding_mask, attention_masks = self._shard_for_cp(
            cp_mesh,
            packed=packed,
            inputs=inputs,
            labels=labels,
            positions=positions,
            padding_mask=padding_mask,
        )
        if attention_masks is not None:
            extra_kwargs["attention_masks"] = attention_masks

        tp_mesh = (
            None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")
        )
        inputs, labels, positions, padding_mask = self._shard_for_tp(
            tp_mesh,
            inputs=inputs,
            labels=labels,
            positions=positions,
            padding_mask=padding_mask,
        )

        if positions is not None:
            extra_kwargs["positions"] = positions
        if padding_mask is not None:
            extra_kwargs["padding_mask"] = padding_mask
        return inputs, labels, extra_kwargs

    def get_attention_masks(self, positions: torch.Tensor) -> BlockMask | torch.Tensor:
        """Build this batch's attention mask: a flex BlockMask, or a dense one.

        ``attn_mask_type`` selects between plain causal and causal-plus-same-
        document. The latter is the packed path: samples share one sequence, so
        attention must not cross a document boundary (positions reset to 0
        there). Both cases return a BlockMask -- with no mask at all flex would
        compute full attention.

        A DSA model gets a dense 4D additive mask instead: its own attention
        code reads the mask as a tensor (see ``uses_dsa``). ``attn_mask_type``
        means the same thing there -- plain causal vs causal-and-same-document
        -- so the two families cannot disagree about what "block_causal" allows.
        """
        block_causal = (
            getattr(self.model.config, "attn_mask_type", "causal") == "block_causal"
        )
        if self.uses_dsa:
            return build_dense_attention_mask(
                positions,
                dtype=self.tok_embeddings.weight.dtype,
                block_causal=block_causal,
            )
        if block_causal:
            mask_mod = and_masks(
                get_causal_mask_mod(),
                get_document_mask_mod(positions),
            )
        else:
            mask_mod = get_causal_mask_mod()

        num_tokens = positions.shape[0]
        return create_attention_mask(
            mask_mod,
            B=1,
            H=None,
            Q_LEN=num_tokens,
            KV_LEN=num_tokens,
            device=positions.device,
            BLOCK_SIZE=128,
            separate_full_blocks=True,
        )

    def _get_cp_attention_masks(self, positions: torch.Tensor) -> BlockMask:
        """Build the BlockMask for a CP forward: full-length, strategy-shaped.

        Under CP, ``positions`` is this rank's shard of the sequence (possibly
        load-balancer-rearranged), so it cannot describe the full document
        structure: the mask is built over the FULL sequence. kv_allgather then
        needs it sharded along its Q axis, matching how the CP kernel's
        gathered K/V stay full-length. Ulysses needs it as built: its
        all-to-all reassembles the full sequence before attention, and the
        kernel would only discard a Q-sharded mask and rebuild this very one.
        ``get_attention_masks`` builds the full mask from an arange -- valid
        because only the causal mod is taken here.

        Packed batches (``block_causal``) cannot take this path: the document
        mask needs the full positions, which only the caller has. Build the
        full-length mask with ``get_attention_masks(full_positions)`` and pass
        it as ``attention_masks`` -- Q-sharded by ``shard_attention_mask_for_cp``
        for kv_allgather, full-length and unsharded for ulysses.
        """
        if self.uses_dsa:
            # A dense mask is built, not modded, and CP shards masks by
            # rewriting a BlockMask's indices (shard_attention_mask_for_cp takes
            # a BlockMask and nothing else). Sharding the dense tensor instead
            # would mean slicing Q and keeping KV full by hand -- unverified,
            # and silently wrong if the load balancer permutes the shard, so it
            # is refused rather than guessed at.
            raise UnsupportedCombinationError(
                "Context parallel with a DSA (dense-mask) model is not "
                "implemented: the CP mask path shards a flex BlockMask, and a "
                "dense additive mask has to be sliced by hand. Unlock by "
                "sharding the dense mask Q-axis (load balancer included) and "
                "re-validating numerics on multiple ranks."
            )
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            raise ValueError(
                "Context parallel with packed sequences needs a prebuilt mask: "
                "build the full-length BlockMask with get_attention_masks from "
                "the FULL positions and pass it to forward as attention_masks "
                "-- Q-sharded with shard_attention_mask_for_cp under "
                "'kv_allgather', or full-length (unsharded) under 'ulysses', "
                "whose all-to-all reassembles the full sequence on every rank. "
                "The positions this forward receives are already CP-sharded and "
                "cannot describe the full document structure."
            )
        cp_size = self.cp_mesh.size()
        full_len = positions.shape[0] * cp_size
        full_positions = torch.arange(full_len, device=positions.device)
        mask = self.get_attention_masks(positions=full_positions)
        if self._cp_strategy == "ulysses":
            return mask
        return shard_attention_mask_for_cp(
            mask, getattr(self, "_cp_mask_mesh", self.cp_mesh), self._cp_load_balancer
        )

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
            attention_masks: a prebuilt BlockMask. Only the flex backend consumes
                it (see ``_apply_attention``); with sdpa the decoder is left to
                its own causal default. Under CP, a full-length mask already
                Q-sharded by ``shard_attention_mask_for_cp``.
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
            from ...accelerator.collectives import synchronize_local_errors

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
            from transformers.masking_utils import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )

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
        """Cross the ROLE-IN / CONVENTION-OUT seam: decide what to hand the decoder.

        Roles are fixed: this wrapper ALWAYS routes through an attention
        implementation, and that implementation is ALWAYS fed a mask describing
        how tokens may attend -- flex consumes it as a ``BlockMask``. How a
        *backend* wants its mask is a different question (sdpa wants a boolean
        tensor, and derives causality from the mask's presence), so it is settled
        here and nowhere else.

        Consequently the mask is built unconditionally, even on the backend that
        ends up discarding it: not building it would make the two paths differ in
        more than the backend, and would hide the packing gap this fallback
        leaves open.
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

        if attention_masks is None:
            if self.cp_mesh is None:
                attention_masks = self.get_attention_masks(positions=positions)
            else:
                attention_masks = self._get_cp_attention_masks(positions)

        # is_causal is the flex-only spelling: it selects which mod the BlockMask
        # runs, so it is withheld from every other backend (see below).
        if self.model.config._attn_implementation == ATTENTION_BACKEND:
            return {"attention_mask": attention_masks, "is_causal": False}

        # Every other backend wants a tensor mask (or none). There is no generic
        # conversion from a BlockMask, and one is not wanted: the only llmtuner
        # case that needs a tensor mask is packing, which the flex fallback
        # cannot express anyway. Fail loudly rather than run with the mask
        # silently dropped -- that would turn the packed path into full attention.
        #
        # Packing is detected from ``attn_mask_type``, not from scanning
        # ``positions`` for restarts: a restart scan misses a boundary after a
        # length-1 document (positions ``[0, 0, ...]`` never descend), which
        # would let sdpa run plain causal attention across the boundary with no
        # complaint. The flag is derived from the corpus at config build time
        # (``build_model_config_for``), so it cannot disagree with the batch.
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            raise ValueError(
                f"Attention backend {self.model.config._attn_implementation!r} "
                "cannot express the mask for a packed sequence: this run's "
                "corpus is packed (attn_mask_type='block_causal'), and a "
                "tensor-mask backend has no way to keep attention from "
                "crossing a document boundary. Sequence packing requires CUDA "
                "and the flex attention kernel."
            )

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
        rank = distributed.get_rank()
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
