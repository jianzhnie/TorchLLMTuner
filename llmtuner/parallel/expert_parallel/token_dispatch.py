"""Local and all-to-all token dispatch with per-call combine metadata.

Adapted from torchtitan ``models/common/token_dispatcher.py``. Local dispatch
sorts assignments by expert; all-to-all also transfers tokens to expert ranks
using explicit process groups. With no EP group, it follows the local path.
Combine reverses permutation and communication, applies routing scores, and
scatters outputs back to their original tokens.

TorchAO padded permutation lives in ``torchao_dispatch.py``. Domino reuses the
all-to-all path for separate token slices. Backend gaps and adaptation details
are recorded in ``docs/llmtuner_upstream_map.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single

from ...accelerator import distributed
from ...models.common.scatter_add import deterministic_scatter_add
from ...utils.logging import get_logger

logger = get_logger(__name__)

# torch ships two functional all-to-alls. ``all_to_all_single`` has NO autograd
# kernel -- calling it on a tensor that needs grad makes torch warn that the
# backward "may lead to silently incorrect behavior", and in practice the
# exchange simply drops the gradient. ``all_to_all_single_autograd`` is the
# differentiable one, whose backward is the reciprocal exchange.
#
# The count exchange below is under ``torch.no_grad()`` and may use either; the
# dispatch and combine exchanges carry gradients -- back to the expert weights
# and forward into the router scores -- so they must use the differentiable
# form. On a build that lacks it, EP training cannot move the experts at all,
# which is why the fallback below says so out loud instead of quietly training
# an untrained expert stack.
try:
    from torch.distributed._functional_collectives import (
        all_to_all_single_autograd as _differentiable_all_to_all,
    )
except ImportError:  # pragma: no cover - depends on the torch build
    _differentiable_all_to_all = None


def _grad_carrying_all_to_all(
    tensor: torch.Tensor,
    output_splits: list[int] | None,
    input_splits: list[int] | None,
    *,
    group,
) -> torch.Tensor:
    """One all-to-all that backprop can flow through; see the note above."""
    if _differentiable_all_to_all is None:
        logger.warning(
            "This torch has no all_to_all_single_autograd, so the EP dispatch "
            "and combine exchanges cannot carry gradients: the expert weights "
            "will not be updated. Run with expert_parallel_size=1, or upgrade "
            "torch."
        )
        return all_to_all_single(tensor, output_splits, input_splits, group=group)
    return _differentiable_all_to_all(tensor, output_splits, input_splits, group=group)


__all__ = [
    "LocalDispatchMetadata",
    "AllToAllDispatchMetadata",
    "LocalTokenDispatcher",
    "AllToAllTokenDispatcher",
]


def materialize(tensor: torch.Tensor) -> torch.Tensor:
    """Force an async collective result to be ready.

    ``all_to_all_single`` returns immediately, leaving the transfer running; the
    count exchange's output is needed on the host (to build split-size lists)
    before the data all-to-all can be launched, so it must be waited on first.
    """
    wait = getattr(tensor, "wait", None)
    return wait() if callable(wait) else tensor


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """What ``LocalTokenDispatcher.dispatch()`` hands to ``combine()``."""

    token_indices_experts_sorted_N: torch.Tensor  # noqa: N815
    topk_scores_experts_sorted_N: torch.Tensor  # noqa: N815


@dataclass(frozen=True, kw_only=True)
class AllToAllDispatchMetadata(LocalDispatchMetadata):
    """The local metadata plus what the all-to-all needs to be reversed."""

    input_shape: tuple  # for _unpermute
    permuted_indices: torch.Tensor  # for _unpermute
    input_splits: list[int]
    output_splits: list[int]


@dataclass(frozen=True, kw_only=True)
class DispatchCounts:
    """Receive counts and host split sizes for one token exchange."""

    counts: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]


@dataclass(frozen=True, kw_only=True)
class PendingAllToAllDispatch:
    """Launched token exchange and the metadata needed to consume it.

    Keep this object alive until completion: each slice owns its tensors and
    split sizes independently, including when receive sizes differ by rank.
    """

    routed_input: torch.Tensor
    token_indices: torch.Tensor
    scores: torch.Tensor
    counts: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]


class LocalTokenDispatcher:
    """Token dispatcher for EP=1: local reordering only.

    Not an ``nn.Module`` -- a dispatcher owns no parameters or buffers.

    Args:
        num_experts: total expert count (E).
        top_k: experts each token is routed to (K).
    """

    def __init__(self, num_experts: int, top_k: int) -> None:
        self.num_experts = num_experts
        self.top_k = top_k

    def set_ep_group(self, *, ep_group: dist.ProcessGroup | None) -> None:
        """No-op for the EP=1 dispatcher. Subclasses override."""
        del ep_group

    def _local_reorder(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder tokens by expert assignment for local expert computation.

        Groups tokens by expert index via a stable argsort, so within an expert
        the original token order is preserved. Routing scores are applied to the
        expert outputs in ``combine``, after expert computation.

        Args:
            x_TD: ``(T, D)`` input tokens.
            topk_scores_TK: ``(T, K)`` routing scores.
            topk_expert_ids_TK: ``(T, K)`` expert indices.

        Returns:
            routed_input_ND: ``(N, D)``, N = T*K, tokens in expert-sorted order.
            token_indices_experts_sorted_N: ``(N,)`` token-to-original mapping.
            topk_scores_experts_sorted_N: ``(N,)`` scores in expert-sorted order.
        """
        if x_TD.ndim != 2:
            raise ValueError("Token inputs must have shape (tokens, hidden_dim)")
        routing_shape = (x_TD.shape[0], self.top_k)
        if (
            topk_scores_TK.shape != routing_shape
            or topk_expert_ids_TK.shape != routing_shape
        ):
            raise ValueError(
                "Routing scores and expert IDs must both have shape "
                f"{routing_shape}, matching the token count and top_k"
            )
        # N = T*K: one entry per (token, chosen expert) pair.
        token_indices_experts_sorted_N = torch.argsort(
            topk_expert_ids_TK.reshape(-1), stable=True
        )
        topk_scores_experts_sorted_N = topk_scores_TK.reshape(-1)[
            token_indices_experts_sorted_N
        ]
        token_indices_experts_sorted_N = token_indices_experts_sorted_N // self.top_k
        routed_input_ND = x_TD[token_indices_experts_sorted_N]

        return (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert assignment for local expert computation.

        Args:
            x_TD: ``(T, D)`` all input tokens.
            topk_scores_TK: ``(T, K)`` routing scores.
            topk_expert_ids_TK: ``(T, K)`` expert indices per token.
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert.

        Returns:
            routed_input_RD: ``(R, D)`` with ``R = sum(num_local_tokens_per_
                expert_E)``, tokens sorted by expert index.
            num_local_tokens_per_expert_E: ``(E,)``, unchanged.
            metadata: for ``combine()``.
        """
        # R = N: with no EP there is no all-to-all, so nothing is dropped.
        (
            routed_input_RD,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)
        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        )
        return routed_input_RD, num_local_tokens_per_expert_E, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Weight the expert outputs by their routing score and scatter them home.

        The score is applied here rather than in ``dispatch`` so it multiplies
        the expert output, which is what the routing weight means.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs.
            metadata: from ``dispatch()``.
            x_TD: ``(T, D)`` original input tokens.

        Returns:
            out_TD: ``(T, D)`` combined output.
        """
        out_TD = torch.zeros_like(x_TD)

        # Promote low-precision expert outputs for score multiplication while
        # preserving float64 outputs used by high-precision models.
        compute_dtype = torch.promote_types(routed_output_RD.dtype, torch.float32)
        routed_output_RD = (
            routed_output_RD.to(compute_dtype)
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(routed_output_RD.dtype)

        dim = x_TD.shape[-1]
        # A token routed to K experts appears K times at its own index, so this
        # scatter_add has duplicate indices -- hence the deterministic variant.
        out_TD = deterministic_scatter_add(
            out_TD,
            metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, dim),
            routed_output_RD,
        )
        return out_TD


class BaseEPTokenDispatcher(LocalTokenDispatcher, ABC):
    """Base for EP dispatchers: owns the EP group and its wiring.

    Args:
        num_experts: total expert count (E). Each rank holds only a slice of
            them: the count is narrowed to the local experts by the all-to-all,
            and by the weight layout the EP swap installs.
        top_k: experts each token is routed to.
    """

    def __init__(self, num_experts: int, top_k: int) -> None:
        super().__init__(num_experts, top_k)
        # ``None`` means EP is off and every method falls back to local.
        self.ep_group: dist.ProcessGroup | None = None

    def set_ep_group(self, *, ep_group: dist.ProcessGroup | None) -> None:
        """Install the EP group used by dispatch/combine."""
        self.ep_group = ep_group

    @abstractmethod
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Move each token to the rank that owns the expert it chose."""
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: object,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Return expert outputs to the tokens that asked for them."""
        raise NotImplementedError


class AllToAllTokenDispatcher(BaseEPTokenDispatcher):
    """EP>1: reorder locally, then all-to-all tokens to their experts' ranks.

    Dispatch moves a token to whichever rank holds the expert it picked;
    combine reverses that and scatters the results back. Both halves are the
    same collective with the split sizes swapped.
    """

    def _token_count_exchange(
        self,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Tell every rank how many tokens you are sending it per expert.

        Each rank contributes a ``(ep_size, e)`` block of counts; the all-to-all
        gathers one column-block from every rank, giving each rank the counts it
        will receive -- and, transposed, the counts it must send.

        Split sizes are omitted, so every rank sends an equal-size buffer: the
        blocks are all ``(ep_size, e)`` and therefore already aligned.
        """
        assert self.ep_group is not None
        ep_size = distributed.get_world_size(self.ep_group)
        return all_to_all_single(
            num_local_tokens_per_expert_E.view(ep_size, -1),
            None,
            None,
            group=self.ep_group,
        )

    def _sync_token_count_exchange(
        self,
        num_local_tokens_per_expert_E: torch.Tensor,
        num_global_tokens_per_local_expert_EP_e: torch.Tensor,
        ep_size: int,
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        """Wait for the counts, then materialize the split-size lists.

        Remote counts (``output_splits``) must be known before the variable-size
        data all-to-all is launched; local counts are ours already and only need
        to reach the host.
        """
        num_global_tokens_per_local_expert_EP_e = materialize(
            num_global_tokens_per_local_expert_EP_e
        )
        num_global_tokens_per_local_expert_E = (
            num_global_tokens_per_local_expert_EP_e.reshape(-1)
        )
        input_splits = (
            num_local_tokens_per_expert_E.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        # NOTE: this incurs a device-to-host sync.
        output_splits = (
            num_global_tokens_per_local_expert_E.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
        )
        return (
            num_global_tokens_per_local_expert_E,
            input_splits.tolist(),
            output_splits.tolist(),
        )

    def _dispatch_token_exchange(
        self,
        routed_input_ND: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        """Send routed tokens to the ranks holding their chosen experts."""
        assert self.ep_group is not None
        return _grad_carrying_all_to_all(
            routed_input_ND, output_splits, input_splits, group=self.ep_group
        )

    def _combine_token_exchange(
        self,
        routed_output_RD: torch.Tensor,
        input_splits: list[int],
        output_splits: list[int],
    ) -> torch.Tensor:
        """Send expert outputs back, reversing the dispatch exchange."""
        assert self.ep_group is not None
        # The split lists are swapped relative to dispatch: what was received
        # there is sent here.
        return _grad_carrying_all_to_all(
            routed_output_RD, input_splits, output_splits, group=self.ep_group
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AllToAllDispatchMetadata]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        Args:
            x_TD: ``(T, D)`` local token shard.
            topk_scores_TK: ``(T, K)`` routing scores.
            topk_expert_ids_TK: ``(T, K)`` expert indices.
            num_local_tokens_per_expert_E: ``(E,)`` counts for this shard.

        Returns:
            routed_input_RD: tokens in expert-major order for local experts.
            num_tokens_per_local_expert_e: ``(num_local_experts,)`` counts.
            metadata: for ``combine()``.
        """
        # EP=1: nothing crosses ranks, so defer to the local path.
        if self.ep_group is None:
            return LocalTokenDispatcher.dispatch(
                self,
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )

        return self.finish_dispatch(
            self.begin_dispatch(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )
        )

    def prepare_dispatch_counts(
        self,
        counts: torch.Tensor,
    ) -> list[DispatchCounts]:
        """Exchange all slices' expert counts in one collective.

        Input is (slices, global_experts). Pack destination rank first so every
        peer receives all slices for its local experts. Materialize host split
        sizes once, before launching any token exchange.
        """
        if self.ep_group is None:
            raise ValueError("prepare_dispatch_counts requires an EP process group")
        ep_size = distributed.get_world_size(self.ep_group)
        if (
            counts.ndim != 2
            or counts.shape[0] < 1
            or counts.shape[1] != self.num_experts
        ):
            raise ValueError("Slice counts must have shape (slices, num_experts)")
        if self.num_experts % ep_size:
            raise ValueError("num_experts must be divisible by the EP world size")
        num_slices = counts.shape[0]
        with torch.no_grad():
            send_counts = counts.reshape(num_slices, ep_size, -1)
            received = (
                materialize(
                    all_to_all_single(
                        send_counts.transpose(0, 1).contiguous().reshape(ep_size, -1),
                        None,
                        None,
                        group=self.ep_group,
                    )
                )
                .reshape(ep_size, num_slices, -1)
                .transpose(0, 1)
            )
            # A blocking host copy makes both lists ready for Python consumers;
            # no second slice count collective can serialize a token transfer.
            splits = torch.stack((send_counts.sum(-1), received.sum(-1)))
            host_splits = splits.cpu().tolist()
            return [
                DispatchCounts(
                    counts=received[i].reshape(-1),
                    input_splits=host_splits[0][i],
                    output_splits=host_splits[1][i],
                )
                for i in range(num_slices)
            ]

    def begin_dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        prepared_counts: DispatchCounts | None = None,
    ) -> PendingAllToAllDispatch:
        """Launch an EP token exchange without consuming its token result.

        Count exchange still synchronizes to obtain host split sizes. Callers
        can prepare multiple slices before running expert computation; actual
        stream scheduling and overlap require separate accelerator validation.
        """
        if self.ep_group is None:
            raise ValueError("begin_dispatch requires an EP process group")
        ep_size = distributed.get_world_size(self.ep_group)
        # _local_reorder yields (N, D), N = T*K; the all-to-all below yields
        # (R, D) with R != N, since each rank ends up with only its own experts'
        # tokens.
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        if prepared_counts is None:
            with torch.no_grad():
                exchanged_counts = self._token_count_exchange(
                    num_local_tokens_per_expert_E
                )
                received_counts, input_splits, output_splits = (
                    self._sync_token_count_exchange(
                        num_local_tokens_per_expert_E, exchanged_counts, ep_size
                    )
                )
            prepared_counts = DispatchCounts(
                counts=received_counts,
                input_splits=input_splits,
                output_splits=output_splits,
            )
        num_global_tokens_per_local_expert_E = prepared_counts.counts
        input_splits_list = prepared_counts.input_splits
        output_splits_list = prepared_counts.output_splits

        routed_input_RD = self._dispatch_token_exchange(
            routed_input_ND,
            output_splits_list,
            input_splits_list,
        )
        return PendingAllToAllDispatch(
            routed_input=routed_input_RD,
            token_indices=token_indices_experts_sorted_N,
            scores=topk_scores_experts_sorted_N,
            counts=num_global_tokens_per_local_expert_E,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )

    def finish_dispatch(
        self,
        pending: PendingAllToAllDispatch,
    ) -> tuple[torch.Tensor, torch.Tensor, AllToAllDispatchMetadata]:
        """Consume a launched exchange and arrange tokens by local expert."""
        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        ) = self._permute(pending.routed_input, pending.counts)
        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=pending.token_indices,
            topk_scores_experts_sorted_N=pending.scores,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=pending.input_splits,
            output_splits=pending.output_splits,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    def _permute(
        self,
        routed_input_RD: torch.Tensor,
        num_global_tokens_per_local_expert_E: torch.Tensor,
    ) -> tuple[tuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder received tokens from rank-major to expert-major layout.

        ``num_global_tokens_per_local_expert_E`` arrives as an ``(EP, e)`` count
        matrix; this builds the index map that regroups it by local expert.
        """
        assert self.ep_group is not None
        ep_size = distributed.get_world_size(self.ep_group)
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size
        device = num_global_tokens_per_local_expert_E.device
        total = routed_input_RD.shape[0]

        t_mat = num_global_tokens_per_local_expert_E.view(ep_size, e)

        # Where each (r, e) segment starts in the input (rank-major order).
        input_starts = (
            num_global_tokens_per_local_expert_E.cumsum(0)
            - num_global_tokens_per_local_expert_E
        ).view(ep_size, e)

        # Transpose to expert-major (e, EP) and flatten.
        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        # output[p] = input[input_starts[seg] + (p - output_starts[seg])].
        seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(
            segment_lens, output_size=total
        )
        output_starts = segment_lens.cumsum(0) - segment_lens
        permuted_indices = (
            input_starts[seg_ids]
            + torch.arange(seg_ids.shape[0], device=device)
            - output_starts[seg_ids]
        )

        num_global_tokens_per_local_expert_e = t_mat.sum(0)
        return (
            routed_input_RD.shape,
            routed_input_RD[permuted_indices, :],
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        )

    def _unpermute(
        self,
        routed_output_RD: torch.Tensor,
        input_shape: tuple,
        permuted_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the expert-major reordering."""
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Unpermute, all-to-all back, weight by score, and scatter home.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs in expert-major order.
            metadata: from ``dispatch()``.
            x_TD: ``(T, D)`` original input tokens.

        Returns:
            out_TD: ``(T, D)`` combined local output.
        """
        # EP=1: nothing to send back.
        if self.ep_group is None:
            return LocalTokenDispatcher.combine(
                self,
                routed_output_RD,
                metadata,
                x_TD,
            )

        routed_output_RD = self._unpermute(
            routed_output_RD, metadata.input_shape, metadata.permuted_indices
        )
        routed_output_RD = self._combine_token_exchange(
            routed_output_RD,
            metadata.input_splits,
            metadata.output_splits,
        )

        # The tokens are back on their home ranks; weighting by score and
        # scattering home is exactly the local dispatcher's combine.
        return LocalTokenDispatcher.combine(
            self,
            materialize(routed_output_RD),
            metadata,
            x_TD,
        )
