"""Ascend MC2 forward/backward parity with independent CPU dense experts.

PYTHONPATH=. torchrun --nproc_per_node=2 tests/integration_tests/ep_mc2_equivalence.py
Requires torch_npu, HCCL, the three fused MC2 operators and two Ascend devices.
CPU adapter emulation cannot substitute for this test.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from llmtuner.models.common.moe.experts import GroupedExperts, RoutedExperts
from llmtuner.parallel.expert_parallel.mc2_dispatch import (
    MC2GroupedExperts,
    MC2RoutedExperts,
)
from llmtuner.parallel.expert_parallel.mc2_ops import load_mc2_ops
from llmtuner.parallel.expert_parallel.token_dispatch import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)


def main():
    ops = load_mc2_ops()
    ops.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", ops.npu.current_device())
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        for zero_receives, empty_rank, recompute in (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ):
            torch.manual_seed(19)
            reference_experts = GroupedExperts(16, 32, 4, use_grouped_mm=False).half()
            for parameter in reference_experts.parameters():
                torch.nn.init.normal_(parameter, std=0.05)
            local_experts = (
                MC2GroupedExperts(16, 32, 2, use_grouped_mm=False).half().to(device)
            )
            with torch.no_grad():
                for local, full in zip(
                    local_experts.parameters(),
                    reference_experts.parameters(),
                    strict=True,
                ):
                    local.copy_(full[rank * 2 : (rank + 1) * 2])
            dispatcher = AllToAllTokenDispatcher(4, 2)
            dispatcher.set_ep_group(ep_group=dist.group.WORLD)
            model = MC2RoutedExperts(local_experts, dispatcher)
            if recompute:
                model = checkpoint_wrapper(model)
            reference = RoutedExperts(reference_experts, LocalTokenDispatcher(4, 2))
            inputs = torch.randn(12, 16, dtype=torch.float16)
            scores = torch.softmax(torch.randn(12, 2), -1).half()
            ids = torch.tensor([[0, 1], [2, 3], [1, 2], [0, 3]] * 3)
            if zero_receives:
                ids[:] = torch.tensor([0, 1])
            cotangent = torch.randn_like(inputs)
            ref_input = inputs.clone().requires_grad_()
            ref_scores = scores.clone().requires_grad_()
            expected = reference(
                ref_input, ref_scores, ids, torch.bincount(ids.flatten(), minlength=4)
            )
            (expected * cotangent).float().sum().backward()
            boundary = 0 if empty_rank else 5
            token_slice = slice(0, boundary) if rank == 0 else slice(boundary, 12)
            local_input = inputs[token_slice].to(device).requires_grad_()
            local_scores = scores[token_slice].to(device).requires_grad_()
            local_ids = ids[token_slice].to(device)
            actual = model(
                local_input,
                local_scores,
                local_ids,
                torch.bincount(local_ids.flatten(), minlength=4),
            )
            (actual * cotangent[token_slice].to(device)).float().sum().backward()
            for got, want in (
                (actual, expected[token_slice]),
                (local_input.grad, ref_input.grad[token_slice]),
                (local_scores.grad, ref_scores.grad[token_slice]),
            ):
                torch.testing.assert_close(got.cpu(), want, rtol=1e-2, atol=5e-4)
            for got, want in zip(
                local_experts.parameters(), reference_experts.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    got.grad.cpu(),
                    want.grad[rank * 2 : (rank + 1) * 2],
                    rtol=1e-2,
                    atol=5e-4,
                )
            if rank == 0:
                print(
                    f"MC2 zero_receives={zero_receives} empty_rank={empty_rank} "
                    f"recompute={recompute}: forward and all gradients match",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
