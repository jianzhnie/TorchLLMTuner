"""MC2 fused dispatch/projection/activation/projection/combine orchestration.

Adapted from FSDPTurbo distributed/expert_parallel/dispatcher_mc2.py. The
stacked expert parameters retain llmtuner's checkpoint layout; fused gate/up
weights are assembled differentiably for each call.
"""

from __future__ import annotations

import torch

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


class MC2RoutedExperts(RoutedExperts):
    """Execute Ascend fused collectives; no unfused or local fallback."""

    def __init__(
        self, grouped_experts: GroupedExperts, dispatcher: AllToAllTokenDispatcher
    ) -> None:
        load_mc2_ops()
        if not isinstance(dispatcher, AllToAllTokenDispatcher):
            raise EnvironmentUnsupportedError(
                "MC2 requires an all-to-all EP dispatcher"
            )
        super().__init__(grouped_experts, dispatcher)

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
        ops, comm_name, world_size = mc2_communication_context(
            dispatcher.ep_group, inputs.device
        )
        if inputs.dtype not in (torch.float16, torch.bfloat16):
            raise EnvironmentUnsupportedError("MC2 requires float16 or bfloat16 inputs")
        routed, indices, sorted_scores = dispatcher._local_reorder(
            inputs, scores, expert_ids
        )
        prepared = dispatcher.prepare_dispatch_counts(counts.unsqueeze(0))[0]
        send_counts = counts.cpu().tolist()
        recv_counts = prepared.counts.cpu().tolist()
        experts = self.inner_experts
        gate_up_weights = torch.cat((experts.w1_EFD, experts.w3_EFD), dim=1)
        gate_up = AllToAllGroupedMatmul.apply(
            routed,
            gate_up_weights,
            ops,
            comm_name,
            world_size,
            send_counts,
            recv_counts,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = experts.activation_fn(gate, up)
        combined = GroupedMatmulAllToAll.apply(
            activated,
            experts.w2_EDF,
            ops,
            comm_name,
            world_size,
            recv_counts,
            send_counts,
        )
        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=indices,
            topk_scores_experts_sorted_N=sorted_scores,
        )
        return LocalTokenDispatcher.combine(dispatcher, combined, metadata, inputs)
