"""MC2 fused dispatch/projection/activation/projection/combine orchestration.

Adapted from FSDPTurbo distributed/expert_parallel/dispatcher_mc2.py. The
stacked expert parameters retain llmtuner's checkpoint layout; fused gate/up
weights are assembled differentiably for each call.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.errors import EnvironmentUnsupportedError
from llmtuner.models.common.moe.experts import GroupedExperts, RoutedExperts
from llmtuner.parallel.expert_parallel.mc2_ops import (
    AllToAllGroupedMatmul,
    GroupedMatmulAllToAll,
    load_mc2_ops,
    mc2_communication_context,
)
from llmtuner.parallel.expert_parallel.token_dispatch import (
    AllToAllTokenDispatcher,
    LocalDispatchMetadata,
    LocalTokenDispatcher,
)

_MC2_FORWARD_CALLS = {"fused": 0, "fallback": 0}


def reset_mc2_dispatch_metrics() -> None:
    for key in _MC2_FORWARD_CALLS:
        _MC2_FORWARD_CALLS[key] = 0


def mc2_dispatch_metrics() -> dict[str, int | float]:
    calls = dict(_MC2_FORWARD_CALLS)
    total = sum(calls.values())
    return {
        **calls,
        "fused_ratio": calls["fused"] / total if total else 0.0,
    }


class MC2GroupedExperts(GroupedExperts):
    """Own fused compute and weights so ordinary module/FSDP hooks run."""

    def forward(
        self,
        inputs,
        ops,
        comm_name,
        world_size,
        send_counts,
        recv_counts,
        group=None,
    ) -> torch.Tensor:
        gate_up_weights = torch.cat((self.w1_EFD, self.w3_EFD), dim=1)
        gate_up = AllToAllGroupedMatmul.apply(
            inputs,
            gate_up_weights,
            ops,
            comm_name,
            world_size,
            send_counts,
            recv_counts,
            group,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self.activation_fn(gate, up)
        return GroupedMatmulAllToAll.apply(
            activated,
            self.w2_EDF,
            ops,
            comm_name,
            world_size,
            recv_counts,
            send_counts,
            group,
        )


class MC2RoutedExperts(RoutedExperts):
    """Use fused MC2 when every local expert group is nonempty."""

    def __init__(
        self, grouped_experts: MC2GroupedExperts, dispatcher: AllToAllTokenDispatcher
    ) -> None:
        if not isinstance(grouped_experts, MC2GroupedExperts):
            raise TypeError("MC2 requires MC2GroupedExperts to own fused computation")
        load_mc2_ops()
        if not isinstance(dispatcher, AllToAllTokenDispatcher):
            raise EnvironmentUnsupportedError(
                "MC2 requires an all-to-all EP dispatcher"
            )
        super().__init__(grouped_experts, dispatcher)
        self.error_group: torch.distributed.ProcessGroup | None = None

    def _unfused_forward(self, inputs, scores, expert_ids, counts, prepared_counts):
        pending = self.token_dispatcher.begin_dispatch(
            inputs,
            scores,
            expert_ids,
            counts,
            prepared_counts=prepared_counts,
        )
        dispatched, received_counts, metadata = self.token_dispatcher.finish_dispatch(
            pending
        )
        output = GroupedExperts.forward(self.inner_experts, dispatched, received_counts)
        return self.token_dispatcher.combine(output, metadata, inputs)

    def forward(
        self,
        inputs: torch.Tensor,
        scores: torch.Tensor,
        expert_ids: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        dispatcher = self.token_dispatcher
        if dispatcher.ep_group is None:
            raise EnvironmentUnsupportedError("MC2 requires an EP process group")
        # Local input failures must be agreed before peers exchange counts or
        # enter a fused collective. Expert/FSDP calls remain outside this region.
        with synchronize_local_errors(
            "MC2 input preparation",
            group=self.error_group
            if self.error_group is not None
            else dispatcher.ep_group,
        ):
            if inputs.dtype not in (torch.float16, torch.bfloat16):
                raise EnvironmentUnsupportedError(
                    "MC2 requires float16 or bfloat16 inputs"
                )
            routed, indices, sorted_scores = dispatcher._local_reorder(
                inputs, scores, expert_ids
            )
            if expert_ids.dtype != torch.int64:
                raise ValueError("Expert IDs must have dtype torch.int64")
            if expert_ids.numel() and (
                (expert_ids < 0).any() or (expert_ids >= dispatcher.num_experts).any()
            ):
                raise ValueError("Expert IDs must be in [0, num_experts)")
            if counts.shape != (dispatcher.num_experts,) or counts.dtype != torch.int64:
                raise ValueError("MC2 counts must be int64 with shape (num_experts,)")
            expected_counts = torch.bincount(
                expert_ids.reshape(-1), minlength=dispatcher.num_experts
            )
            # Fused kernels consume Python counts. Validate the same host
            # snapshot used below before any rank enters count exchange.
            host_counts, host_expected = (
                torch.stack((counts, expected_counts)).cpu().tolist()
            )
            if host_counts != host_expected:
                raise ValueError("MC2 counts must match the expert ID histogram")
        ops, comm_name, world_size = mc2_communication_context(
            dispatcher.ep_group, inputs.device
        )
        prepared = dispatcher.prepare_dispatch_counts(counts.unsqueeze(0))[0]
        send_counts = host_counts
        recv_counts = prepared.counts.cpu().tolist()
        needs_unfused = any(count == 0 for count in (*send_counts, *recv_counts))
        if dist.is_initialized() and world_size > 1:
            # Every rank must enter the same forward and backward collectives.
            fallback = torch.tensor(
                int(needs_unfused), dtype=torch.int32, device=inputs.device
            )
            dist.all_reduce(fallback, op=dist.ReduceOp.MAX, group=dispatcher.ep_group)
            needs_unfused = bool(fallback.item())
        _MC2_FORWARD_CALLS["fallback" if needs_unfused else "fused"] += 1
        if needs_unfused:
            return self._unfused_forward(inputs, scores, expert_ids, counts, prepared)
        combined = self.inner_experts(
            routed,
            ops,
            comm_name,
            world_size,
            send_counts,
            recv_counts,
            dispatcher.ep_group,
        )
        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=indices,
            topk_scores_experts_sorted_N=sorted_scores,
        )
        return LocalTokenDispatcher.combine(dispatcher, combined, metadata, inputs)
