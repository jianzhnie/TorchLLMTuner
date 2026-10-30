"""Explicit torchtitan BF16 expert arithmetic across the all-to-all dispatcher."""

from __future__ import annotations

import torch

from ...models.common.moe.experts import GroupedExperts, RoutedExperts
from .token_dispatch import AllToAllTokenDispatcher, materialize


class NpuBaselineAllToAllTokenDispatcher(AllToAllTokenDispatcher):
    """Match the baseline's BF16 reduction for repeated token gradients."""

    def _permute_tokens(self, inputs, token_indices, expert_ids):
        if inputs.device.type != "npu":
            raise TypeError("torchtitan NPU token permutation requires NPU inputs")
        if inputs.shape[0] == 0:
            return inputs[token_indices]
        import torch_npu

        # Indexing has identical forward values but can round a repeated-index
        # input gradient differently from the baseline permutation kernel.
        routed, _ = torch_npu.npu_moe_token_permute(inputs, expert_ids)
        return routed


class NpuBaselineGroupedExperts(GroupedExperts):
    """Keep the existing parameter layout and FSDP module-call lifecycle."""

    def forward(self, inputs, counts, *, routing_scores):
        if inputs.device.type != "npu" or inputs.dtype != torch.bfloat16:
            raise TypeError("torchtitan EP experts require BF16 NPU inputs")
        import torch_npu

        offsets = counts.cumsum(0, dtype=torch.int32)
        packed = torch.cat((self.w1_EFD, self.w3_EFD), dim=1)
        projected = torch._grouped_mm(inputs, packed.transpose(-2, -1), offs=offsets)
        hidden = torch_npu.npu_swiglu(projected, dim=-1)
        hidden = hidden * routing_scores.to(hidden.dtype)
        return torch._grouped_mm(hidden, self.w2_EDF.transpose(-2, -1), offs=offsets)


class NpuBaselineRoutedExperts(RoutedExperts):
    """Transfer scores with tokens, then sum outputs in original top-k order.

    The explicit torchtitan path weights the BF16 activation before down GEMM.
    Generic dispatchers apply scores after returning outputs to their owners;
    reusing that combine here would apply them twice and change rounding.
    """

    def forward(self, inputs, scores, expert_ids, counts):
        dispatcher = self.token_dispatcher
        if type(dispatcher) is not NpuBaselineAllToAllTokenDispatcher:
            raise TypeError("torchtitan NPU EP requires the alltoall dispatcher")
        routed, local_counts, metadata = dispatcher.dispatch(
            inputs, scores, expert_ids, counts
        )
        received_scores = materialize(
            dispatcher._dispatch_token_exchange(
                metadata.topk_scores_experts_sorted_N.reshape(-1, 1),
                metadata.output_splits,
                metadata.input_splits,
            )
        )
        received_scores = received_scores[metadata.permuted_indices]
        outputs = self.inner_experts(
            routed, local_counts, routing_scores=received_scores
        )
        outputs = dispatcher._unpermute(
            outputs, metadata.input_shape, metadata.permuted_indices
        )
        outputs = materialize(
            dispatcher._combine_token_exchange(
                outputs, metadata.input_splits, metadata.output_splits
            )
        )
        order = torch.argsort(expert_ids.reshape(-1), stable=True)
        inverse = torch.argsort(order)
        # A rank may own experts but originate no tokens. Inferring -1 is
        # ambiguous for a zero-element output, even after a valid exchange.
        outputs = outputs[inverse].reshape(
            inputs.shape[0], scores.shape[1], inputs.shape[-1]
        )
        return outputs.sum(dim=1)
