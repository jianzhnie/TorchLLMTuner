"""Check real padded EP dispatch against an analytical expert reference.

Run with torchrun --nproc_per_node=2 and a real torchao runtime on PYTHONPATH.
Each expert multiplies tokens by its global expert index plus one. This exposes
misrouting, padding leakage, score errors, and incorrect all-to-all backward.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from llmtuner.parallel.expert_parallel.torchao_dispatch import TorchAOTokenDispatcher
from tests.caps import require_env
from tests.integration_tests.harness import init_distributed


def main() -> None:
    require_env("dtensor")
    device = init_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"requires two ranks, got {world}"
    num_experts = 8
    local_experts = num_experts // world
    dispatcher = TorchAOTokenDispatcher(num_experts, 2, pad_multiple=16)
    dispatcher.set_ep_group(ep_group=dist.group.WORLD)
    try:
        for skewed in (False, True):
            generator = torch.Generator().manual_seed(42 + rank)
            tokens = 13 + rank * 6
            inputs = torch.randn(tokens, 32, generator=generator).to(device)
            inputs.requires_grad_()
            scores = torch.rand(tokens, 2, generator=generator).to(device)
            scores = (scores / scores.sum(dim=-1, keepdim=True)).detach()
            scores.requires_grad_()
            expert_ids = torch.randint(
                num_experts, (tokens, 2), generator=generator
            ).to(device)
            if skewed:
                expert_ids.fill_(0)
                expert_ids[:, 1] = num_experts - 1
            counts = torch.bincount(expert_ids.flatten(), minlength=num_experts)
            routed, padded_counts, metadata = dispatcher.dispatch(
                inputs, scores, expert_ids, counts
            )
            assert (padded_counts % 16 == 0).all().item()
            assert routed.shape[0] == int(padded_counts.sum().item())
            factors = torch.arange(
                rank * local_experts + 1,
                (rank + 1) * local_experts + 1,
                device=device,
                dtype=inputs.dtype,
            )
            routed_factors = torch.repeat_interleave(factors, padded_counts)
            assert routed.shape[0] >= routed_factors.numel()
            routed_factors = torch.nn.functional.pad(
                routed_factors, (0, routed.shape[0] - routed_factors.numel())
            )
            actual = dispatcher.combine(
                routed * routed_factors.unsqueeze(-1), metadata, inputs
            )
            reference_inputs = inputs.detach().clone().requires_grad_()
            reference_scores = scores.detach().clone().requires_grad_()
            reference = reference_inputs * (reference_scores * (expert_ids + 1)).sum(
                dim=-1, keepdim=True
            )
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
            probe = torch.randn(tokens, 32, generator=generator).to(device)
            (actual * probe).sum().backward()
            (reference * probe).sum().backward()
            for observed, expected in (
                (inputs.grad, reference_inputs.grad),
                (scores.grad, reference_scores.grad),
            ):
                assert observed is not None and expected is not None
                assert torch.isfinite(observed).all().item()
                torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-5)
            print(
                f"PASS rank={rank} device={device} skewed={skewed} "
                f"padded_counts={padded_counts.tolist()} output/input_grad/score_grad",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
