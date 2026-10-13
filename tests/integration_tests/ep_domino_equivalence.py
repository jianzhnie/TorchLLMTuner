"""Verify two-slice Domino EP forward and backward against local experts.

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/ep_domino_equivalence.py

Odd, unequal rank token counts and zero receives exercise split bookkeeping.
"""

from itertools import product

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from llmtuner.models.common.moe.experts import (
    DominoRoutedExperts,
    GroupedExperts,
    RoutedExperts,
)
from llmtuner.parallel.expert_parallel.token_dispatch import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        generator = torch.Generator().manual_seed(19)
        for routed_class, zero_receives, recompute in product(
            (RoutedExperts, DominoRoutedExperts), (False, True), (False, True)
        ):
            reference_experts = GroupedExperts(8, 12, 4, use_grouped_mm=False).double()
            with torch.no_grad():
                for parameter in reference_experts.parameters():
                    parameter.copy_(
                        torch.randn(
                            parameter.shape, generator=generator, dtype=torch.float64
                        )
                        * 0.1
                    )
            local_experts = GroupedExperts(8, 12, 2, use_grouped_mm=False).double()
            with torch.no_grad():
                for target, source in zip(
                    local_experts.parameters(),
                    reference_experts.parameters(),
                    strict=True,
                ):
                    target.copy_(source[rank * 2 : (rank + 1) * 2])
            reference = RoutedExperts(reference_experts, LocalTokenDispatcher(4, 2))
            dispatcher = AllToAllTokenDispatcher(4, 2)
            dispatcher.wire_meshes(ep_group=dist.group.WORLD)
            distributed_experts = routed_class(local_experts, dispatcher)
            calls = []
            if recompute:
                original_forward = distributed_experts.forward

                def counted_forward(
                    *args, original=original_forward, invocations=calls, **kwargs
                ):
                    invocations.append(True)
                    return original(*args, **kwargs)

                distributed_experts.forward = counted_forward
                distributed_experts = checkpoint_wrapper(distributed_experts)
            inputs = torch.randn(12, 8, generator=generator, dtype=torch.float64)
            scores = torch.softmax(
                torch.randn(12, 2, generator=generator, dtype=torch.float64), dim=-1
            )
            expert_ids = torch.tensor(
                [[0, 1], [2, 0], [1, 3], [0, 2], [3, 1], [2, 3]] * 2
            )
            if zero_receives:
                expert_ids = torch.tensor([[0, 1]] * 12)
            weights = torch.randn(12, 8, generator=generator, dtype=torch.float64)
            reference_inputs = inputs.clone().requires_grad_()
            reference_scores = scores.clone().requires_grad_()
            expected = reference(
                reference_inputs,
                reference_scores,
                expert_ids,
                torch.bincount(expert_ids.flatten(), minlength=4),
            )
            (expected * weights).sum().backward()
            token_slice = slice(0, 5) if rank == 0 else slice(5, 12)
            local_inputs = inputs[token_slice].clone().requires_grad_()
            local_scores = scores[token_slice].clone().requires_grad_()
            local_ids = expert_ids[token_slice]
            actual = distributed_experts(
                local_inputs,
                local_scores,
                local_ids,
                torch.bincount(local_ids.flatten(), minlength=4),
            )
            (actual * weights[token_slice]).sum().backward()
            if recompute:
                assert len(calls) == 2, "Checkpoint backward must actually recompute"
            torch.testing.assert_close(
                actual, expected[token_slice], rtol=1e-9, atol=1e-9
            )
            torch.testing.assert_close(
                local_inputs.grad,
                reference_inputs.grad[token_slice],
                rtol=1e-9,
                atol=1e-9,
            )
            torch.testing.assert_close(
                local_scores.grad,
                reference_scores.grad[token_slice],
                rtol=1e-9,
                atol=1e-9,
            )
            for got, want in zip(
                local_experts.parameters(), reference_experts.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    got.grad, want.grad[rank * 2 : (rank + 1) * 2], rtol=1e-9, atol=1e-9
                )
            if rank == 0:
                print(
                    f"{routed_class.__name__} zero_receives={zero_receives} "
                    f"recompute={recompute}: "
                    "forward and gradients match",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
