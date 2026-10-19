"""Two-slice EP dispatch, expert execution and combine orchestration.

Adapted from FSDPTurbo distributed/expert_parallel/domino_dispatcher.py.
CUDA slice 1 dispatch and slice 0 combine are scheduled on a communication
stream alongside expert computation. Target-device validation is pending.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llmtuner.models.common.moe.experts import RoutedExperts
from llmtuner.parallel.expert_parallel.token_dispatch import (
    AllToAllTokenDispatcher,
    DispatchCounts,
)

if TYPE_CHECKING:
    from llmtuner.models.common.moe.experts import GroupedExperts
    from llmtuner.parallel.expert_parallel.token_dispatch import LocalTokenDispatcher


class DominoRoutedExperts(RoutedExperts):
    """Run routing and expert computation in two token slices.

    Each slice owns its dispatch metadata through combine, so variable all to
    all receive sizes cannot leak between slices. This is the semantic core of
    FSDPTurbo Domino. CUDA uses a dedicated dispatch/combine stream; other
    devices preserve the two-slice collective and checkpoint contract.
    """

    def __init__(
        self,
        grouped_experts: GroupedExperts,
        dispatcher: LocalTokenDispatcher,
    ) -> None:
        super().__init__(grouped_experts, dispatcher)
        # Runtime streams are device-specific and are not checkpoint state.
        self._comm_streams: dict[torch.device, torch.cuda.Stream] = {}

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        del num_local_tokens_per_expert_E
        # Validate the complete call before either slice starts communicating.
        # A truncated routing tensor may fit slice 0 and fail only after peers
        # have entered the second exchange.
        if x_TD.ndim != 2:
            raise ValueError("Token inputs must have shape (tokens, hidden_dim)")
        routing_shape = (x_TD.shape[0], self.token_dispatcher.top_k)
        if (
            topk_scores_TK.shape != routing_shape
            or topk_expert_ids_TK.shape != routing_shape
        ):
            raise ValueError(
                "Routing scores and expert IDs must both have shape "
                f"{routing_shape}, matching the token count and top_k"
            )
        # Every rank executes both exchanges, including ranks with zero or
        # one token. Rank-local early returns would strand their peers.
        split_at = (x_TD.shape[0] + 1) // 2
        slices = []
        for token_slice in (slice(0, split_at), slice(split_at, None)):
            slice_input_TD = x_TD[token_slice]
            slice_scores_TK = topk_scores_TK[token_slice]
            slice_expert_ids_TK = topk_expert_ids_TK[token_slice]
            slice_counts_E = torch.bincount(
                slice_expert_ids_TK.reshape(-1),
                minlength=self.token_dispatcher.num_experts,
            )
            slices.append(
                (slice_input_TD, slice_scores_TK, slice_expert_ids_TK, slice_counts_E)
            )

        dispatcher = self.token_dispatcher
        if (
            isinstance(dispatcher, AllToAllTokenDispatcher)
            and dispatcher.ep_group is not None
        ):
            # Resolve every host split size before token transfers, avoiding a
            # second count exchange between the two asynchronous data launches.
            prepared_counts = dispatcher.prepare_dispatch_counts(
                torch.stack([args[3] for args in slices])
            )
            if x_TD.is_cuda:
                return self._forward_cuda(slices, prepared_counts, dispatcher)
            pending = [
                dispatcher.begin_dispatch(*args, prepared_counts=counts)
                for args, counts in zip(slices, prepared_counts, strict=True)
            ]
            outputs = []
            for args, exchange in zip(slices, pending, strict=True):
                routed_input, counts, metadata = dispatcher.finish_dispatch(exchange)
                routed_output = self.inner_experts(routed_input, counts)
                outputs.append(dispatcher.combine(routed_output, metadata, args[0]))
        else:
            outputs = [
                super(DominoRoutedExperts, self).forward(*args) for args in slices
            ]
        return torch.cat(outputs, dim=0)

    def _forward_cuda(
        self,
        slices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        prepared_counts: list[DispatchCounts],
        dispatcher: AllToAllTokenDispatcher,
    ) -> torch.Tensor:
        """Pipeline dispatch1/expert0 and combine0/expert1 on two streams.

        All ranks enqueue dispatch0, dispatch1, combine0, combine1 in that order.
        Native autograd operations retain their forward stream and provide the
        cross-stream backward dependencies; no detached communication buffers
        or custom backward scheduling are introduced here.
        """
        device = slices[0][0].device
        compute_stream = torch.cuda.current_stream(device)
        comm_stream = self._comm_streams.get(device)
        if comm_stream is None:
            comm_stream = torch.cuda.Stream(device=device)
            self._comm_streams[device] = comm_stream

        first = dispatcher.begin_dispatch(
            *slices[0], prepared_counts=prepared_counts[0]
        )
        input0, counts0, metadata0 = dispatcher.finish_dispatch(first)

        # Wait for slice/count producers, then launch slice 1 on the comm stream.
        # This wait is queued before expert0, so it cannot serialize expert0
        # behind slice 1 communication.
        comm_stream.wait_stream(compute_stream)
        for tensor in (*slices[1], prepared_counts[1].counts):
            tensor.record_stream(comm_stream)
        with torch.cuda.stream(comm_stream):
            second = dispatcher.begin_dispatch(
                *slices[1], prepared_counts=prepared_counts[1]
            )
            input1, counts1, metadata1 = dispatcher.finish_dispatch(second)
            ready = comm_stream.record_event()

        expert_output0 = self.inner_experts(input0, counts0)
        # Communication consumes expert0 only after its producer completes,
        # while the caller can proceed with expert1 after dispatch1's event.
        comm_stream.wait_stream(compute_stream)
        for tensor in (
            expert_output0,
            slices[0][0],
            metadata0.token_indices_experts_sorted_N,
            metadata0.topk_scores_experts_sorted_N,
            metadata0.permuted_indices,
        ):
            tensor.record_stream(comm_stream)
        with torch.cuda.stream(comm_stream):
            output0 = dispatcher.combine(expert_output0, metadata0, slices[0][0])

        compute_stream.wait_event(ready)
        for tensor in (input1, counts1):
            tensor.record_stream(compute_stream)
        expert_output1 = self.inner_experts(input1, counts1)
        comm_stream.wait_stream(compute_stream)
        expert_output1.record_stream(comm_stream)
        with torch.cuda.stream(comm_stream):
            output1 = dispatcher.combine(expert_output1, metadata1, slices[1][0])
            combined = comm_stream.record_event()

        compute_stream.wait_event(combined)
        for tensor in (output0, output1):
            tensor.record_stream(compute_stream)
        return torch.cat((output0, output1), dim=0)
