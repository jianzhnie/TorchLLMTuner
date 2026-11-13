"""Verify shared parameters across model parts enter the TP collective once.

Run with two CPU/Gloo ranks:

    LLMTUNER_TEST_DEVICE=cpu PYTHONPATH=. torchrun --standalone \
        --nproc_per_node=2 tests/integration_tests/shared_parameter_tp_reduction.py

On an NPU host, omit ``LLMTUNER_TEST_DEVICE`` and select two free devices to
exercise the same check through HCCL.
Four ranks additionally cover DP2 x TP2 DTensor gradients with empty DP shards.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from llmtuner.trainer.runtime.gradients import allreduce_replicated_tp_grads
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size in (2, 4), f"expected 2 or 4 ranks, got {world_size}"

    shared = nn.Parameter(torch.zeros(1, device=device))
    shared.grad = torch.tensor([float(rank + 1)], device=device)
    parts = []
    for name in ("first", "second"):
        part = nn.Module()
        part.register_parameter(name, shared)
        parts.append(part)

    trainer = SimpleNamespace(
        parallel_dims=SimpleNamespace(
            tp_enabled=True,
            get_optional_mesh=lambda axis: SimpleNamespace(
                get_group=lambda: dist.group.WORLD
            ),
        ),
        model_parts=parts,
    )
    allreduce_replicated_tp_grads(
        trainer.model_parts, parallel_dims=trainer.parallel_dims
    )

    expected = torch.tensor(
        [sum(range(1, world_size + 1))], dtype=shared.dtype, device=device
    )
    torch.testing.assert_close(shared.grad, expected, rtol=0, atol=0)
    # A rank-local unused parameter must still join its peer's reduction.
    shared.grad = torch.tensor([5.0], device=device) if rank == 0 else None
    allreduce_replicated_tp_grads(
        trainer.model_parts, parallel_dims=trainer.parallel_dims
    )
    torch.testing.assert_close(
        shared.grad, torch.tensor([5.0], device=device), rtol=0, atol=0
    )
    shared.grad = None
    allreduce_replicated_tp_grads(
        trainer.model_parts, parallel_dims=trainer.parallel_dims
    )
    assert shared.grad is None
    # TP peers share a DP shard; four ranks also exercise empty DP ownership.
    mesh = init_device_mesh(
        device.type, (world_size // 2, 2), mesh_dim_names=("dp", "tp")
    )
    weight = nn.Parameter(
        distribute_tensor(torch.ones(1, device=device), mesh, [Shard(0), Replicate()])
    )
    part = nn.Module()
    part.register_parameter("weight", weight)
    trainer.model_parts = [part, part]
    tp_mesh = mesh["tp"]
    trainer.parallel_dims.get_optional_mesh = lambda axis: tp_mesh
    for missing_rank in (0, 1):
        weight.grad = (
            None
            if tp_mesh.get_local_rank() == missing_rank
            else torch.ones_like(weight) * (rank + 1)
        )
        allreduce_replicated_tp_grads(
            trainer.model_parts, parallel_dims=trainer.parallel_dims
        )
        assert isinstance(weight.grad, DTensor)
        assert weight.grad.device_mesh == weight.device_mesh
        assert weight.grad.placements == weight.placements
        contributing_rank = (rank // 2) * 2 + (1 - missing_rank)
        torch.testing.assert_close(
            weight.grad.to_local(),
            torch.full_like(weight.to_local(), contributing_rank + 1),
            rtol=0,
            atol=0,
        )
    weight.grad = None
    allreduce_replicated_tp_grads(
        trainer.model_parts, parallel_dims=trainer.parallel_dims
    )
    assert weight.grad is None
    if rank == 0:
        print(
            "PASS shared/DTensor TP gradients, missing participation and empty shards"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
