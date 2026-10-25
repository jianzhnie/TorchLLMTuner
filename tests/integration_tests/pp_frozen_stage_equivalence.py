"""Compare frozen PP stages against independent dense AdamW updates.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/pp_frozen_stage_equivalence.py

CPU/Gloo, actual ComposablePipelineStage and 1F1B, frozen first/last stages.
This verifies the optimizer/scheduler boundary, not full Trainer recovery.
"""

import copy
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining.schedules import Schedule1F1B

from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.parallel.pipeline_parallel.partition import ComposablePipelineStage


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("pp",))
        for frozen_stage in (0, 1):
            torch.manual_seed(290)
            dense = torch.nn.Sequential(
                torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)
            ).double()
            dense[frozen_stage].requires_grad_(False)
            local = torch.nn.Sequential()
            local.add_module(str(rank), copy.deepcopy(dense[rank]))
            frozen = {
                name: parameter.detach().clone()
                for name, parameter in local.named_parameters()
            }
            optimizer = OptimizersContainer(
                OptimizerConfig(implementation="for-loop", learning_rate=0.01),
                model_parts=[local],
                pipeline_mesh=mesh,
            )
            scheduler = build_lr_scheduler(
                LRSchedulerConfig(warmup_steps=2),
                optimizers=list(optimizer),
                training_steps=3,
            )
            reference_optimizer = torch.optim.AdamW(
                [p for p in dense.parameters() if p.requires_grad],
                lr=0.01,
                weight_decay=0,
            )
            reference_scheduler = build_lr_scheduler(
                LRSchedulerConfig(warmup_steps=2),
                optimizers=[reference_optimizer],
                training_steps=3,
            )
            stage = ComposablePipelineStage(
                local, rank, 2, torch.device("cpu"), group=mesh.get_group()
            )

            # Sum microbatch losses with a full-batch denominator.
            def loss(output, target):
                return (output - target).square().sum() / 8

            schedule = Schedule1F1B(stage, 2, loss_fn=loss, scale_grads=False)
            for step in range(3):
                optimizer.zero_grad()
                reference_optimizer.zero_grad()
                inputs = torch.randn(4, 3, dtype=torch.float64)
                targets = torch.randn(4, 2, dtype=torch.float64)
                expected = dense(inputs)
                loss(expected, targets).backward()
                losses = []
                output = schedule.step(
                    *([inputs] if rank == 0 else []),
                    target=targets if rank == 1 else None,
                    losses=losses if rank == 1 else None,
                )
                if rank == 1:
                    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(
                        sum(losses), loss(expected, targets), rtol=1e-12, atol=1e-12
                    )
                for actual, reference in zip(
                    local.parameters(), dense[rank].parameters(), strict=True
                ):
                    if reference.requires_grad:
                        torch.testing.assert_close(
                            actual.grad, reference.grad, rtol=1e-12, atol=1e-12
                        )
                    else:
                        assert actual.grad is None
                optimizer.step()
                scheduler.step()
                reference_optimizer.step()
                reference_scheduler.step()
                assert scheduler.state_dict() == {"last_epoch": step + 1}
                for name, actual in local.named_parameters():
                    reference = dense.get_parameter(name)
                    torch.testing.assert_close(
                        actual, reference, rtol=1e-12, atol=1e-12
                    )
                    if rank == frozen_stage:
                        torch.testing.assert_close(actual, frozen[name], rtol=0, atol=0)
            dist.barrier()
            print(
                f"rank={rank} frozen_stage={frozen_stage}: dense oracle passed",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
