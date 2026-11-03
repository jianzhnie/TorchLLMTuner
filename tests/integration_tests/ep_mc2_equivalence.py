"""Ascend MC2 forward/backward parity with independent CPU dense experts.

PYTHONPATH=. torchrun --nproc_per_node=8 tests/integration_tests/ep_mc2_equivalence.py
Requires torch_npu, HCCL, the three fused MC2 operators and Ascend devices.
Set MC2_TEST_DTYPE=bfloat16 to exercise the BF16 path. CPU adapter emulation
cannot substitute for this test.
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
    mc2_dispatch_metrics,
    reset_mc2_dispatch_metrics,
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
        world_size = dist.get_world_size()
        assert world_size >= 2
        rank = dist.get_rank()
        num_experts = world_size * 2
        num_tokens = world_size * world_size
        dtype = (
            torch.bfloat16
            if os.environ.get("MC2_TEST_DTYPE") == "bfloat16"
            else torch.float16
        )
        tolerance = (
            {"rtol": 2e-2, "atol": 2e-2}
            if dtype == torch.bfloat16
            else {"rtol": 1e-2, "atol": 5e-4}
        )
        reset_mc2_dispatch_metrics()
        expected_calls = {"fused": 0, "fallback": 0}
        for zero_receives, empty_rank, recompute in (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, True),
        ):
            torch.manual_seed(19)
            reference_experts = GroupedExperts(
                16, 32, num_experts, use_grouped_mm=False
            ).to(dtype)
            for parameter in reference_experts.parameters():
                torch.nn.init.normal_(parameter, std=0.05)
            local_experts = MC2GroupedExperts(16, 32, 2, use_grouped_mm=False).to(
                dtype=dtype, device=device
            )
            with torch.no_grad():
                for local, full in zip(
                    local_experts.parameters(),
                    reference_experts.parameters(),
                    strict=True,
                ):
                    local.copy_(full[rank * 2 : (rank + 1) * 2])
            dispatcher = AllToAllTokenDispatcher(num_experts, 2)
            dispatcher.set_ep_group(ep_group=dist.group.WORLD)
            model = MC2RoutedExperts(local_experts, dispatcher)
            if recompute:
                model = checkpoint_wrapper(model)
            reference = RoutedExperts(
                reference_experts, LocalTokenDispatcher(num_experts, 2)
            )
            inputs = torch.randn(num_tokens, 16, dtype=dtype)
            scores = torch.softmax(torch.randn(num_tokens, 2), -1).to(dtype)
            ids = (
                torch.arange(num_tokens * 2, dtype=torch.int64).reshape(-1, 2)
                % num_experts
            )
            if zero_receives:
                ids[:] = torch.tensor([0, 1])
            cotangent = torch.randn_like(inputs)
            ref_input = inputs.clone().requires_grad_()
            ref_scores = scores.clone().requires_grad_()
            expected = reference(
                ref_input,
                ref_scores,
                ids,
                torch.bincount(ids.flatten(), minlength=num_experts),
            )
            (expected * cotangent).float().sum().backward()
            shards = torch.tensor_split(
                torch.arange(num_tokens), world_size - 1 if empty_rank else world_size
            )
            token_indices = (
                inputs.new_empty(0, dtype=torch.int64)
                if empty_rank and rank == 0
                else shards[rank - 1 if empty_rank else rank]
            )
            local_input = inputs[token_indices].to(device).requires_grad_()
            local_scores = scores[token_indices].to(device).requires_grad_()
            local_ids = ids[token_indices].to(device)
            if not zero_receives and not empty_rank:
                local_counts = torch.bincount(
                    local_ids.flatten(), minlength=num_experts
                )
                assert torch.all(local_counts > 0), (
                    "the nominal fused case must route tokens to every expert "
                    "on each source rank"
                )
            actual = model(
                local_input,
                local_scores,
                local_ids,
                torch.bincount(local_ids.flatten(), minlength=num_experts),
            )
            selected = "fallback" if zero_receives or empty_rank else "fused"
            (actual * cotangent[token_indices].to(device)).float().sum().backward()
            expected_calls[selected] += 2 if recompute else 1
            metrics = mc2_dispatch_metrics()
            assert metrics["fused"] == expected_calls["fused"], metrics
            assert metrics["fallback"] == expected_calls["fallback"], metrics
            for got, want in (
                (actual, expected[token_indices]),
                (local_input.grad, ref_input.grad[token_indices]),
                (local_scores.grad, ref_scores.grad[token_indices]),
            ):
                torch.testing.assert_close(got.cpu(), want, **tolerance)
            for got, want in zip(
                local_experts.parameters(), reference_experts.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    got.grad.cpu(),
                    want.grad[rank * 2 : (rank + 1) * 2],
                    **tolerance,
                )
            if rank == 0:
                print(
                    f"MC2 dtype={dtype} zero_receives={zero_receives} "
                    f"empty_rank={empty_rank} recompute={recompute}: "
                    "forward and all gradients match",
                    flush=True,
                )
        metrics = mc2_dispatch_metrics()
        assert metrics["fused"] == 3 and metrics["fallback"] == 4, metrics
        assert metrics["fused_ratio"] == 3 / 7, metrics
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
