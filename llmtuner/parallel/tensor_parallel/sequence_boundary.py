"""TP sequence communication boundaries installed without renaming model state."""

from __future__ import annotations

import torch
from torch.distributed import ProcessGroup

from ...accelerator.linear_collectives import all_gather_along, reduce_scatter_along
from .. import compatibility


class FeatureParallelSequenceBoundary:
    """Explicit type for feature-sharded expert execution between TP collectives."""

    _tp_seq_group: ProcessGroup


class GatherSequenceFirst:
    """Mixin that all-gathers the TP sequence shard before HF attention runs.

    Installed by ``apply_tp`` via a ``__class__`` swap (not a module wrapper),
    so module paths, ``state_dict`` keys and later attach points (``apply_cp``'s
    the CP attention kernel) are all untouched. ``hidden_states`` arrives as this
    rank's ``[B, T / tp, K]`` sequence shard and is gathered along the sequence
    dim to the length the inner forward expects -- the full sequence, or the CP
    shard when CP is on (the gather spans the TP group only, and a TP group
    collectively holds exactly one CP shard).

    The gather's autograd dual is a reduce-scatter, which sums each rank's
    input-gradient contribution back to its own token shard -- the exact dual
    the sequence-parallel layout needs.
    """

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        gathered = all_gather_along(hidden_states, -2, self._tp_seq_group)
        original = self.__dict__.get("_tp_attention_original_forward")
        return (
            original(gathered, *args, **kwargs)
            if original is not None
            else super().forward(gathered, *args, **kwargs)
        )


class ReplicatedAttentionSequenceBoundary:
    """Run unsharded attention on the gathered sequence, return local tokens.

    Some HF plans shard only MLPs (e.g. DeepSeek MLA). Attention still needs
    the full sequence and RoPE positions. Its output is replicated, so slice
    tokens without summing; the input gather sums partial gradients in backward.
    Replicated attention parameter gradients are reduced by the trainer.

    Qwen3.5 GDN uses the same boundary with all projections replicated: both
    the causal convolution and recurrent scan require the full token history.
    Its attention_mask must describe the gathered sequence (as supplied by HF),
    and is forwarded unchanged. This boundary adds no parameter-gradient hooks;
    the trainer performs the single replicated-parameter gradient reduction.
    """

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        gathered = all_gather_along(hidden_states, -2, self._tp_seq_group)
        original = self.__dict__.get("_tp_attention_original_forward")
        output = (
            original(gathered, *args, **kwargs)
            if original is not None
            else super().forward(gathered, *args, **kwargs)
        )
        attention = output[0] if isinstance(output, tuple) else output
        local = attention.chunk(self._tp_seq_group.size(), dim=-2)[
            self._tp_seq_group.rank()
        ].contiguous()
        return (local, *output[1:]) if isinstance(output, tuple) else local


class TPMoeSequenceBoundary(FeatureParallelSequenceBoundary):
    """Mixin that brackets a HF MoE block with the TP sequence collectives.

    The MoE block under TP is the dense colwise/rowwise pair with the feature
    shard kept internal: the input arrives as this rank's ``(B, T / tp, D)``
    sequence shard and is all-gathered (backward: reduce-scatter of the input
    gradient); the block then runs on the full token stream with its expert
    weights sharded on the F dim (``shard_experts_for_tp``), producing an
    output that is partial over the TP group; the boundary reduce-scatter sums
    the partials and returns a ``(B, T / tp, D)`` sequence shard (backward:
    all-gather of the output gradient). Weight layout and collectives are the
    same dual pair the dense TP realizers use.

    The router is deliberately untouched: its weight stays replicated, every
    rank computes the identical routing on the gathered stream, and the
    trainer's ``allreduce_replicated_tp_grads`` sums its gradient (each rank's
    copy earns a different partial through the sharded expert outputs).

    Installed by ``__class__`` swap (like ``GatherSequenceFirst``), so module
    paths, ``state_dict`` keys and later attach points are all untouched.
    """

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        gathered = all_gather_along(hidden_states, -2, self._tp_seq_group)
        original = self.__dict__.get("_tp_moe_original_forward")
        out = (
            original(gathered, *args, **kwargs)
            if original is not None
            else super().forward(gathered, *args, **kwargs)
        )
        # This probe sits between the two collectives; it is only safe because
        # the block's output TYPE is structural -- identical on every rank of
        # the TP group -- so either all ranks raise here or none do. Never let
        # a data-dependent type reach this check.
        if not isinstance(out, torch.Tensor):
            compatibility.tp_moe_non_tensor_output(self, out)
        return reduce_scatter_along(out, -2, self._tp_seq_group)
