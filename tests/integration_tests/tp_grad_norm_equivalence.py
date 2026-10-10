"""Global clipping for local TP shards and TP-replicated parameters.

Run with ``PYTHONPATH=. torchrun --standalone --nproc_per_node=2
tests/integration_tests/tp_grad_norm_equivalence.py``.
"""

import math

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.accelerator.collectives import clip_grad_norm_


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))

    replicated = torch.nn.Parameter(torch.zeros(1))
    sharded = torch.nn.Parameter(torch.zeros(1))
    replicated.grad = torch.tensor([12.0])
    sharded.grad = torch.tensor([3.0 if rank == 0 else 4.0])
    norm = clip_grad_norm_(
        [replicated, sharded],
        max_norm=6.5,
        foreach=False,
        tp_mesh=mesh,
        tp_sharded_parameters=[sharded],
    )
    # The replicated contribution is counted once, plus both TP shards.
    torch.testing.assert_close(norm, torch.tensor(13.0))
    torch.testing.assert_close(replicated.grad, torch.tensor([6.0]))
    torch.testing.assert_close(
        sharded.grad, torch.tensor([1.5 if rank == 0 else 2.0])
    )

    # EP may consume the same physical ranks as TP. Expert parameters are
    # counted through EP only; dense TP shards still need their TP reduction.
    expert = torch.nn.Parameter(torch.zeros(1))
    replicated.grad = torch.tensor([12.0])
    sharded.grad = torch.tensor([3.0 if rank == 0 else 4.0])
    expert.grad = torch.tensor([5.0 if rank == 0 else 12.0])
    norm = clip_grad_norm_(
        [replicated, sharded, expert],
        max_norm=0.0,
        foreach=False,
        tp_mesh=mesh,
        tp_sharded_parameters=[sharded, expert],
        ep_mesh=mesh,
        expert_parameters=[expert],
    )
    torch.testing.assert_close(norm, torch.tensor(math.sqrt(338)))
    torch.testing.assert_close(expert.grad, torch.tensor([5.0 if rank == 0 else 12.0]))

    if rank == 0:
        print("all checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
