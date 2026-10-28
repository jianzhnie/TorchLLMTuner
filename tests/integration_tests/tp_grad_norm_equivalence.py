"""Global clipping for local TP shards and TP-replicated parameters.

Run with ``PYTHONPATH=. torchrun --standalone --nproc_per_node=2
tests/integration_tests/tp_grad_norm_equivalence.py``.
"""

import math

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.accelerator.collectives import clip_grad_norm_
from tests.integration_tests.harness import backend, init_distributed, moved


def main() -> None:
    device = init_distributed()
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    mesh = init_device_mesh(device.type, (2,), mesh_dim_names=("tp",))

    replicated = moved(torch.nn.Parameter(torch.zeros(1)))
    sharded = moved(torch.nn.Parameter(torch.zeros(1)))
    replicated.grad = torch.tensor([12.0], device=device)
    sharded.grad = torch.tensor([3.0 if rank == 0 else 4.0], device=device)
    norm = clip_grad_norm_(
        [replicated, sharded],
        max_norm=6.5,
        foreach=False,
        tp_mesh=mesh,
        tp_sharded_parameters=[sharded],
    )
    # The replicated contribution is counted once, plus both TP shards.
    torch.testing.assert_close(norm, torch.tensor(13.0, device=device))
    torch.testing.assert_close(replicated.grad, torch.tensor([6.0], device=device))
    torch.testing.assert_close(
        sharded.grad, torch.tensor([1.5 if rank == 0 else 2.0], device=device)
    )

    # EP may consume the same physical ranks as TP. Expert parameters are
    # counted through EP only; dense TP shards still need their TP reduction.
    expert = moved(torch.nn.Parameter(torch.zeros(1)))
    replicated.grad = torch.tensor([12.0], device=device)
    sharded.grad = torch.tensor([3.0 if rank == 0 else 4.0], device=device)
    expert.grad = torch.tensor([5.0 if rank == 0 else 12.0], device=device)
    norm = clip_grad_norm_(
        [replicated, sharded, expert],
        max_norm=0.0,
        foreach=False,
        tp_mesh=mesh,
        tp_sharded_parameters=[sharded, expert],
        ep_mesh=mesh,
        expert_parameters=[expert],
    )
    torch.testing.assert_close(norm, torch.tensor(math.sqrt(338), device=device))
    torch.testing.assert_close(
        expert.grad, torch.tensor([5.0 if rank == 0 else 12.0], device=device)
    )

    # The documented string infinity norm must work through the TP branch.
    sharded.grad = torch.tensor([3.0 if rank == 0 else 4.0], device=device)
    norm = clip_grad_norm_(
        [sharded],
        max_norm=0.0,
        norm_type="inf",
        tp_mesh=mesh,
        tp_sharded_parameters=[sharded],
    )
    torch.testing.assert_close(norm, torch.tensor(4.0, device=device))

    # Both ranks must reject a nonfinite global norm after joining reductions.
    # Case one overflows only at the SUM; case two has one bad local shard.
    for gradient in (2e38, float("inf") if rank == 0 else 1.0):
        sharded.grad = torch.tensor([gradient], device=device)
        before = sharded.grad.clone()
        try:
            clip_grad_norm_(
                [sharded],
                max_norm=1.0,
                norm_type=1.0,
                error_if_nonfinite=True,
                tp_mesh=mesh,
                tp_sharded_parameters=[sharded],
            )
        except RuntimeError as exc:
            assert "non-finite" in str(exc), str(exc)
        else:
            raise AssertionError("nonfinite global norm was accepted")
        torch.testing.assert_close(sharded.grad, before)
        dist.barrier()

    if rank == 0:
        print(f"device={device.type} backend={backend()}")
        print("all checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
