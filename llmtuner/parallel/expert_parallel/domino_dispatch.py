"""Two-slice EP dispatch, expert execution and combine orchestration.

Adapted from FSDPTurbo distributed/expert_parallel/domino_dispatcher.py.
CUDA slice 1 dispatch and slice 0 combine are scheduled on a communication
stream alongside expert computation. Target-device validation is pending.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.nn.modules import module as module_hooks

from llmtuner.accelerator.collectives import all_reduce, synchronize_local_errors
from llmtuner.models.common.activation import SituGLU, SwiGLU
from llmtuner.models.common.moe.experts import (
    ExpertErrorCapture,
    GroupedExperts,
    RoutedExperts,
)
from llmtuner.parallel.expert_parallel.token_dispatch import (
    AllToAllTokenDispatcher,
    DispatchCounts,
    materialize,
)

if TYPE_CHECKING:
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
        self.error_group: dist.ProcessGroup | None = None
        self._comm_streams: dict[torch.device, torch.cuda.Stream] = {}

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        del num_local_tokens_per_expert_E
        dispatcher = self.token_dispatcher
        preparation = (
            synchronize_local_errors(
                "Domino input preparation",
                group=self.error_group
                if self.error_group is not None
                else dispatcher.ep_group,
            )
            if isinstance(dispatcher, AllToAllTokenDispatcher)
            and dispatcher.ep_group is not None
            else nullcontext()
        )
        # Only local tensor work belongs here; expert hooks may communicate.
        with preparation:
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
            if topk_expert_ids_TK.dtype != torch.int64:
                raise ValueError("Expert IDs must have dtype torch.int64")
            if topk_expert_ids_TK.numel() and (
                (topk_expert_ids_TK < 0).any()
                or (topk_expert_ids_TK >= self.token_dispatcher.num_experts).any()
            ):
                raise ValueError("Expert IDs must be in [0, num_experts)")
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
                    (
                        slice_input_TD,
                        slice_scores_TK,
                        slice_expert_ids_TK,
                        slice_counts_E,
                    )
                )

        if (
            isinstance(dispatcher, AllToAllTokenDispatcher)
            and dispatcher.ep_group is not None
        ):
            coordinate_experts = self._can_coordinate_experts(x_TD, dispatcher)
            # Resolve every host split size before token transfers, avoiding a
            # second count exchange between the two asynchronous data launches.
            prepared_counts = dispatcher.prepare_dispatch_counts(
                torch.stack([args[3] for args in slices])
            )
            if x_TD.is_cuda:
                return self._forward_cuda(
                    slices,
                    prepared_counts,
                    dispatcher,
                    coordinate_experts=coordinate_experts,
                )
            pending = []
            try:
                for args, counts in zip(slices, prepared_counts, strict=True):
                    pending.append(
                        dispatcher.begin_dispatch(*args, prepared_counts=counts)
                    )
                outputs = []
                for args, exchange in zip(slices, pending, strict=True):
                    routed_input, counts, metadata = dispatcher.finish_dispatch(
                        exchange
                    )
                    routed_output = self._run_experts(
                        routed_input,
                        counts,
                        dispatcher,
                        coordinate_experts=coordinate_experts,
                    )
                    outputs.append(dispatcher.combine(routed_output, metadata, args[0]))
            except BaseException as error:
                # Both token transfers precede expert computation. Even if
                # expert0 fails, retain and wait for every launched transfer
                # before discarding its buffers. This starts no new collective
                # and does not claim to recover peers already in combine.
                cleanup_error = None
                for exchange in pending:
                    try:
                        materialize(exchange.routed_input)
                    except BaseException as caught:
                        cleanup_error = caught
                if cleanup_error is not None:
                    raise error from cleanup_error
                raise

        else:
            outputs = [
                super(DominoRoutedExperts, self).forward(*args) for args in slices
            ]
        return torch.cat(outputs, dim=0)

    def _can_coordinate_experts(
        self, inputs: torch.Tensor, dispatcher: AllToAllTokenDispatcher
    ) -> bool:
        """Coordinate failures only for the built-in, communication-free body.

        Known FSDP hooks run outside the captured compute body.
        Other hooks, DTensor computation, compile and custom activations may
        communicate and are excluded from this local error boundary.
        All EP ranks agree on eligibility before entering the optional boundary;
        inspecting only this rank would give peers different collective orders.
        """
        experts = self.inner_experts
        activation = getattr(experts, "activation_fn", None)
        fsdp_experts = type(experts).__bases__ == (FSDPModule, GroupedExperts)
        hooked_fsdp = False
        if fsdp_experts:
            state = experts._get_fsdp_state()
            hooked_fsdp = experts in state._modules
            # Grouped FSDP stores a MultiHandle and wraps its state callbacks.
            # Check both registry ownership and the callback chain: merely
            # accepting any functools-wrapped hook could admit communication.
            for hooks, handle, expected in (
                (
                    experts._forward_pre_hooks,
                    state._pre_forward_hook_handle,
                    state._pre_forward,
                ),
                (
                    experts._forward_hooks,
                    state._post_forward_hook_handle,
                    state._post_forward,
                ),
            ):
                handles = getattr(handle, "handles", (handle,))
                ids = {getattr(item, "id", None) for item in handles}
                if len(hooks) != 1 or not set(hooks).issubset(ids):
                    hooked_fsdp = False
                    break
                callback = next(iter(hooks.values()))
                seen = set()
                while (
                    callback != expected
                    and callback is not None
                    and id(callback) not in seen
                ):
                    seen.add(id(callback))
                    callback = getattr(callback, "__wrapped__", None)
                if callback != expected:
                    hooked_fsdp = False
                    break
        local = (
            (type(experts) is GroupedExperts or fsdp_experts)
            and type(activation) in (SwiGLU, SituGLU)
            and getattr(experts.forward, "__func__", None) is GroupedExperts.forward
            and getattr(experts._compute, "__func__", None) is GroupedExperts._compute
            and getattr(experts._grouped_mm, "__func__", None)
            is GroupedExperts._grouped_mm
            and getattr(activation.forward, "__func__", None)
            is type(activation).forward
            and not module_hooks._global_forward_hooks
            and not module_hooks._global_forward_pre_hooks
            and all(
                (
                    (module is experts and hooked_fsdp)
                    or (not module._forward_hooks and not module._forward_pre_hooks)
                )
                and getattr(module, "_compiled_call_impl", None) is None
                for module in experts.modules()
            )
            and (
                hooked_fsdp
                or all(
                    type(parameter) in (torch.Tensor, nn.Parameter)
                    for parameter in experts.parameters()
                )
            )
            and type(inputs) is torch.Tensor
        )
        eligible = torch.tensor(int(local), device=inputs.device, dtype=torch.int32)
        all_reduce(
            eligible,
            op="min",
            group=self.error_group
            if self.error_group is not None
            else dispatcher.ep_group,
        )
        return bool(eligible.item())

    def _run_experts(
        self,
        inputs: torch.Tensor,
        counts: torch.Tensor,
        dispatcher: AllToAllTokenDispatcher,
        *,
        coordinate_experts: bool,
    ) -> torch.Tensor:
        # Eligibility was agreed before launching either data transfer. A
        # blocking reduction here before compute would serialize dispatch1
        # ahead of expert0, and combine0 ahead of expert1, defeating overlap.
        if not coordinate_experts:
            return self.inner_experts(inputs, counts)
        capture = ExpertErrorCapture()
        # The known local body captures compute errors. Let FSDP's normal
        # post-forward hooks complete before coordinating across error groups.
        # Communication-hook failures remain outside the local-error boundary.
        output = self.inner_experts(inputs, counts, error_capture=capture)
        with synchronize_local_errors(
            "Domino expert computation",
            group=self.error_group
            if self.error_group is not None
            else dispatcher.ep_group,
        ):
            if capture.error is not None:
                raise capture.error
        return output

    def _forward_cuda(
        self,
        slices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        prepared_counts: list[DispatchCounts],
        dispatcher: AllToAllTokenDispatcher,
        *,
        coordinate_experts: bool,
    ) -> torch.Tensor:
        """Pipeline dispatch1/expert0 and combine0/expert1 on two streams.

        All ranks enqueue dispatch0, dispatch1, combine0, combine1 in that order.
        Eligibility is agreed before dispatch0. For local bodies, error status
        is coordinated after each expert and before combine.
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

        first = second = None
        output0 = output1 = None
        try:
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

            expert_output0 = self._run_experts(
                input0, counts0, dispatcher, coordinate_experts=coordinate_experts
            )
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
            expert_output1 = self._run_experts(
                input1, counts1, dispatcher, coordinate_experts=coordinate_experts
            )
            comm_stream.wait_stream(compute_stream)
            expert_output1.record_stream(comm_stream)
            with torch.cuda.stream(comm_stream):
                output1 = dispatcher.combine(expert_output1, metadata1, slices[1][0])
                combined = comm_stream.record_event()

            compute_stream.wait_event(combined)
            for tensor in (output0, output1):
                tensor.record_stream(compute_stream)
            return torch.cat((output0, output1), dim=0)
        except BaseException as error:
            # Consume already enqueued dispatch/combine results on their stream.
            # No replacement collective is launched on the failing path.
            cleanup_error = None
            comm_stream.wait_stream(compute_stream)
            with torch.cuda.stream(comm_stream):
                pending_tensors = [
                    exchange.routed_input
                    for exchange in (first, second)
                    if exchange is not None
                ]
                for tensor in (*pending_tensors, output0, output1):
                    if tensor is None:
                        continue
                    try:
                        materialize(tensor)
                    except BaseException as caught:
                        cleanup_error = caught
            compute_stream.wait_stream(comm_stream)
            if cleanup_error is not None:
                raise error from cleanup_error
            raise
