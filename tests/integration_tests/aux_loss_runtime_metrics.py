"""PP layer means exclude auxiliary losses from older model constructions.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/aux_loss_runtime_metrics.py
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.models.common.aux_loss import (
    AuxLoss,
    collect_aux_loss_metrics,
    rollup_aux_loss_metrics,
)


def main():
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("pp",))

        # Previous models intentionally differ by rank. Current model has three
        # layers globally, each reporting the same independent raw metric 2.
        class PreviousRankZero(AuxLoss):
            pass

        class PreviousRankOne(AuxLoss):
            pass

        previous_type = PreviousRankZero if rank == 0 else PreviousRankOne
        previous = torch.nn.ModuleList(
            [previous_type(coeff=1.0) for _ in range(5 + rank)]
        )
        current = torch.nn.ModuleList([AuxLoss(coeff=1.0) for _ in range(rank + 1)])
        for loss in current:
            loss.instance_acc.fill_(2.0)
        topology = SimpleNamespace(
            get_optional_mesh=lambda name: mesh if name == "pp" else None
        )
        rollup_aux_loss_metrics([current])
        metrics = collect_aux_loss_metrics(topology, model_parts=[current])
        assert metrics == {"aux_loss/mean": 2.0}
        assert len(previous) == 5 + rank
        assert all(loss.instance_acc.item() == 0 for loss in current)
        # A stage without any current AuxLoss must join key and count collectives.
        if rank == 1:
            AuxLoss._group_counts.clear()
        empty_stage = torch.nn.ModuleList([AuxLoss(coeff=1.0)] if rank == 0 else [])
        for loss in empty_stage:
            loss.instance_acc.fill_(3.0)
        rollup_aux_loss_metrics([empty_stage])
        assert collect_aux_loss_metrics(topology, model_parts=[empty_stage]) == {
            "aux_loss/mean": 3.0
        }
        dist.barrier()
        print(f"rank={rank}: current PP layer mean=2, old model counts excluded")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
