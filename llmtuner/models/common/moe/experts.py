"""The expert weights, and the routed-expert module that consumes them.

``GroupedExperts`` is vendored from torchtitan ``models/common/moe.py`` (that
class only); ``RoutedExperts`` is the wrapper that pairs it with a token
dispatcher, and is what the model's MoE block calls. Removals from both: the
nested ``Config`` dataclasses, ``torch_remat``, and the ``spmd_types``
type-checking blocks -- none of them affect the arithmetic.

Shape legend (Noam Shazeer's convention, scoped to this file):

* ``E`` -- number of experts (a local shard of it under EP).
* ``D`` -- model dimension.
* ``F`` -- expert hidden dimension (the per-expert FFN width).
* ``R`` -- routed tokens assigned to the experts on this rank.
* ``O`` / ``I`` -- the grouped-GEMM weight's out/in feature roles. Kept as
  roles rather than model dims because the stored weights are ``(F, D)`` for
  the up/gate projections and ``(D, F)`` for the down one.

Weights are three stacked tensors rather than a ``ModuleList`` of small
modules. That is what makes the expert computation a single grouped GEMM
instead of E separate ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....accelerator.capabilities import has
from ..activation import SwiGLU

if TYPE_CHECKING:
    from ....parallel.expert_parallel.token_dispatch import LocalTokenDispatcher

__all__ = [
    "ExpertErrorCapture",
    "GroupedExperts",
    "RoutedExperts",
]


@dataclass
class ExpertErrorCapture:
    """Carry a local compute error through FSDP post-forward hooks.

    This opaque object survives input pytree casting by identity. The caller
    must check error before consuming the returned expert output.
    """

    error: BaseException | None = None


class GroupedExperts(nn.Module):
    """All experts' weights as three ``(E, O, I)`` tensors.

    Args:
        dim: model dimension (``D``).
        hidden_dim: expert hidden dimension (``F``).
        num_experts: number of experts (``E``).
        activation_fn: the gated activation; defaults to SwiGLU.
        use_grouped_mm: run the expert GEMMs as one ``torch._grouped_mm``
            instead of a per-expert loop. ``None`` (the default) decides by
            probing the op, which is what a run wants; pass a bool to pin it,
            which is what a test wants. See ``_grouped_mm`` for the trade-off.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        *,
        activation_fn: nn.Module | None = None,
        use_grouped_mm: bool | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("dim", dim),
            ("hidden_dim", hidden_dim),
            ("num_experts", num_experts),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be positive integer, got {value!r}")
        if use_grouped_mm is not None and not isinstance(use_grouped_mm, bool):
            raise TypeError("use_grouped_mm must be bool or None")
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.use_grouped_mm = (
            has("torch_grouped_mm") if use_grouped_mm is None else use_grouped_mm
        )
        self.activation_fn = activation_fn if activation_fn is not None else SwiGLU()

        # ``w1``/``w3`` are the gate and up projections, ``w2`` the down one.
        # Empty rather than random: the weights are filled by the caller (the
        # EP swap copies them out of the HF block; a from-scratch build would
        # run an initializer of its own).
        self.w1_EFD = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w3_EFD = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2_EDF = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        *,
        error_capture: ExpertErrorCapture | None = None,
    ) -> torch.Tensor:
        """Compute experts, optionally deferring a local error until after hooks.

        Only the dispatcher may opt into capture after validating the local
        compute boundary. The failed output is an unused detached input placeholder:
        callers must propagate error_capture.error before combine or backward.
        Pre/post-forward hook errors are outside this compute-only boundary.
        """
        if error_capture is None:
            return self._compute(x_RD, num_tokens_per_expert_E)
        error_capture.error = None
        try:
            if any(
                type(parameter) not in (torch.Tensor, nn.Parameter)
                for parameter in self.parameters(recurse=False)
            ):
                raise RuntimeError(
                    "Expert error capture requires materialized local weights"
                )
            return self._compute(x_RD, num_tokens_per_expert_E)
        except BaseException as error:
            error_capture.error = error
            return x_RD.detach()

    def _compute(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Run every expert over the tokens routed to it.

        ``x_RD`` holds each expert's tokens contiguously, in expert order -- the
        dispatcher's job is to produce that layout. ``num_tokens_per_expert_E``
        gives nonnegative integer segment lengths whose sum must cover all rows.
        """
        if x_RD.ndim != 2 or x_RD.shape[1] != self.dim:
            raise ValueError(
                f"Expert inputs must have shape (routed_tokens, {self.dim}), "
                f"got {tuple(x_RD.shape)}"
            )
        if num_tokens_per_expert_E.shape != (self.num_experts,):
            raise ValueError(
                f"Expert token counts must have shape ({self.num_experts},), "
                f"got {tuple(num_tokens_per_expert_E.shape)}"
            )
        if num_tokens_per_expert_E.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("Expert token counts must use an integer dtype")
        torch._assert_async(
            (num_tokens_per_expert_E >= 0).all(),
            "Expert token counts must be nonnegative",
        )
        torch._assert_async(
            num_tokens_per_expert_E.sum() == x_RD.shape[0],
            "Expert token counts must sum to the number of routed tokens",
        )
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)

        gate_RF = self._grouped_mm(x_RD, self.w1_EFD, offsets_E)
        up_RF = self._grouped_mm(x_RD, self.w3_EFD, offsets_E)
        h_RF = self.activation_fn(gate_RF, up_RF)
        return self._grouped_mm(h_RF, self.w2_EDF, offsets_E).type_as(x_RD)

    def _grouped_mm(
        self,
        x_TD: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        """Project contiguous token segments with each expert's weight.

        ``offsets_E`` contains cumulative segment ends; expert zero starts at
        zero. BF16 activations use torch._grouped_mm when enabled. Other dtypes
        use per-expert F.linear to avoid reducing their precision to BF16.

        Empty input still runs an empty linear operation to retain the input
        and weight gradient paths required by distributed dispatch backward.
        """
        if x_TD.shape[0] == 0:
            # A rank with no expert tokens must still traverse dispatch in
            # backward. Empty linear keeps the input and weight graph alive
            # and produces zero parameter gradients without any expert work.
            return F.linear(x_TD, weight_EOI[0].to(x_TD.dtype))

        if self.use_grouped_mm and x_TD.dtype == torch.bfloat16:
            return torch._grouped_mm(
                x_TD.bfloat16(),
                weight_EOI.bfloat16().transpose(-2, -1),
                offs=offsets_E,
            ).type_as(x_TD)

        counts = offsets_E.clone()
        counts[1:] = offsets_E[1:] - offsets_E[:-1]
        out_TD = x_TD.new_empty(x_TD.shape[0], weight_EOI.shape[1])
        start = 0
        for expert, count in enumerate(counts.tolist()):
            # Counts are data-dependent segment sizes, not Python branch
            # conditions. Empty linear segments preserve zero expert gradients.
            torch._check(count >= 0)
            end = start + count
            torch._check(end <= x_TD.shape[0])
            segment = torch.narrow(x_TD, 0, start, count)
            out_TD[start:end] = F.linear(segment, weight_EOI[expert])
            start = end
        return out_TD


class RoutedExperts(nn.Module):
    """The dispatch/combine pair wrapped around the grouped expert weights.

    Split out from ``MoE`` so the routing (which is model-specific) and the
    expert compute (which is not) can vary independently.

    Args:
        grouped_experts: the expert weights.
        dispatcher: moves tokens to the ranks holding their experts.
    """

    def __init__(
        self,
        grouped_experts: GroupedExperts,
        dispatcher: LocalTokenDispatcher,
    ) -> None:
        super().__init__()
        self.inner_experts = grouped_experts
        self.token_dispatcher = dispatcher

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, run them, and combine the results."""
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        routed_output_RD = self.inner_experts(
            routed_input_RD, num_global_tokens_per_local_expert_e
        )
        return self.token_dispatcher.combine(
            routed_output_RD,
            metadata,
            x_TD,
        )
