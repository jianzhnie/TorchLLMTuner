"""TorchAO padded token dispatch for quantized grouped expert computation.

Extends the shared all-to-all dispatcher with padded permutation and its inverse.
The optional torchao package is imported only when constructing the dispatcher.
"""

from __future__ import annotations

import torch

from ...accelerator import distributed
from .token_dispatch import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)

__all__ = ["TORCHAO_INSTALL_HINT", "TorchAOTokenDispatcher"]

TORCHAO_INSTALL_HINT = (
    "ep_token_dispatcher='torchao' requires the optional `torchao` package, "
    "which is not installed. Install it with `pip install torchao`, or use "
    "the default ep_token_dispatcher='alltoall'. torchao's `permute_and_pad` "
    "pads each expert's token group to a multiple of pad_multiple for "
    "FP8/MXFP8 quantized grouped GEMMs."
)


class TorchAOTokenDispatcher(AllToAllTokenDispatcher):
    """All-to-all dispatch with token-group padding for quantized grouped GEMMs.

    Identical to ``AllToAllTokenDispatcher`` except ``_permute``/``_unpermute``
    delegate to torchao's ``permute_and_pad``, which reorders tokens to
    expert-major order AND pads each expert's token group to a multiple of
    ``pad_multiple`` -- the alignment FP8/MXFP8 quantized grouped GEMM kernels
    require (16 for FP8, 32 for MXFP8). The padding shows up in
    ``num_tokens_per_local_expert_e`` and must be stripped by the matching
    ``_unpermute``. ``GroupedExperts`` consumes the returned padded counts
    with either its grouped GEMM or per-expert loop; padding is removed before
    scores are applied in combine.

    Invariants (unchanged from the all-to-all dispatcher):

    * ``dispatch(x_TD(T, D), topk_scores_TK(T, K), topk_expert_ids_TK(T, K),
      num_local_tokens_per_expert_E(E,))`` returns tokens in expert-major
      order for the local experts, per-local-expert counts (padded), and
      metadata consumed only by ``combine``.
    * ``combine`` inverts dispatch exactly: unpermute (strip padding),
      all-to-all back, weight by score, deterministic scatter home.

    torchao is an optional dependency: constructing this class without it
    raises ``ImportError`` with an install hint. With ``ep_group=None`` it
    skips the all-to-all and applies only the local padded permute (the EP=1
    debug/numerics path upstream supports).

    The adapter trims unused allocation capacity beyond the padded expert
    segments, keeping the expert compute contract exact. Resolving that size
    materializes the padded count sum on the host.
    """

    def __init__(self, num_experts: int, top_k: int, *, pad_multiple: int) -> None:
        super().__init__(num_experts, top_k)
        if (
            not isinstance(pad_multiple, int)
            or isinstance(pad_multiple, bool)
            or pad_multiple < 1
        ):
            raise ValueError(
                f"pad_multiple must be a positive integer, got {pad_multiple!r}"
            )
        self.pad_multiple = pad_multiple
        try:
            from torchao.prototype.moe_training.ep.permute import permute_and_pad
        except ImportError as exc:
            raise ImportError(TORCHAO_INSTALL_HINT) from exc
        self._permute_and_pad = permute_and_pad

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AllToAllDispatchMetadata]:
        """Dispatch, padding each local expert's token group to ``pad_multiple``."""
        if self.ep_group is not None:
            return super().dispatch(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )

        # EP=1: no all-to-all. Locally reorder to expert-sorted order, then
        # apply the padded permute so the quantized grouped GEMM sees groups
        # aligned to pad_multiple.
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_tokens_per_local_expert_padded_e,
        ) = self._permute(routed_input_ND, num_local_tokens_per_expert_E)

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            # Unused in the EP=1 combine path (no all-to-all to reverse).
            input_splits=[],
            output_splits=[],
        )
        return routed_input_RD, num_tokens_per_local_expert_padded_e, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Strip the padding, then run the local score + scatter home."""
        if self.ep_group is not None:
            return super().combine(routed_output_RD, metadata, x_TD)

        # EP=1: _unpermute removes the padding and recovers expert-sorted
        # order; the rest is exactly the local dispatcher's combine.
        assert isinstance(metadata, AllToAllDispatchMetadata)
        routed_output_RD = self._unpermute(
            routed_output_RD, metadata.input_shape, metadata.permuted_indices
        )
        return LocalTokenDispatcher.combine(self, routed_output_RD, metadata, x_TD)

    def _permute(
        self,
        routed_input_RD: torch.Tensor,
        num_global_tokens_per_local_expert_E: torch.Tensor,
    ) -> tuple[tuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Padded expert-major permute via torchao's ``permute_and_pad``."""
        # ep_size=1 when EP is disabled: permute_and_pad then only pads token
        # groups (rank-major == expert-major for a single rank).
        ep_size = (
            1 if self.ep_group is None else distributed.get_world_size(self.ep_group)
        )
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size

        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_padded_e,
            _group_offsets,
        ) = self._permute_and_pad(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
            ep_size,
            e,
            self.pad_multiple,
        )
        covered_rows = int(num_global_tokens_per_local_expert_padded_e.sum().item())
        if covered_rows > routed_input_RD.shape[0]:
            raise ValueError("TorchAO padded counts exceed the routed buffer capacity")
        return (
            input_shape,
            routed_input_RD[:covered_rows],
            permuted_indices[:covered_rows],
            num_global_tokens_per_local_expert_padded_e,
        )

    def _unpermute(
        self,
        routed_output_RD: torch.Tensor,
        input_shape: tuple,
        permuted_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the padded permute and strip the sentinel row it added."""
        return super()._unpermute(routed_output_RD, input_shape, permuted_indices)[:-1]
