"""Real DCP async round-trip and group teardown on two device-selected ranks.

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/async_checkpoint_lifecycle_equivalence.py

Uses the production optimizer/scheduler containers. Add --fsdp for sharded
parameters and --async-mode async_with_pinned_mem for process uploads with
pinned staging. The harness selects CPU/Gloo or NPU/HCCL. Cross-node runs
require a shared TMPDIR; accelerator staging requires an accelerator run.
"""

import argparse
from contextlib import nullcontext
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import CheckpointConfig, LRSchedulerConfig, OptimizerConfig
from tests.integration_tests.harness import device as harness_device
from tests.integration_tests.harness import init_distributed, moved


def step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    inputs = moved(torch.arange(12).float().reshape(4, 3) / 11)
    model(inputs).square().mean().backward()
    optimizer.step()
    scheduler.step()


def manager_for(model, optimizer, scheduler, folder, *, async_mode="async"):
    return DistributedCheckpointManager(
        CheckpointConfig(
            enable=True, interval=1, keep_latest_k=0, async_mode=async_mode
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={},
        folder=folder,
    )


def assert_group_released(group):
    try:
        dist.get_process_group_ranks(group)
    except KeyError:
        return
    raise AssertionError("Checkpoint manager left its Gloo group registered")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument(
        "--async-mode", choices=("async", "async_with_pinned_mem"), default="async"
    )
    args = parser.parse_args()
    device = init_distributed()
    active = []
    try:
        assert dist.get_world_size() == 2
        mesh = None
        if args.fsdp:
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh(device.type, (2,), mesh_dim_names=("dp_shard",))

        def build_model():
            model = moved(torch.nn.Linear(3, 2))
            if mesh is not None:
                from torch.distributed.fsdp import fully_shard

                fully_shard(model, mesh=mesh)
            return model

        rank = dist.get_rank()
        context = TemporaryDirectory() if rank == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            root = paths[0]
            for cycle in range(2):
                torch.manual_seed(42 + cycle)
                model = build_model()
                optimizer = OptimizersContainer(
                    OptimizerConfig(learning_rate=0.01, implementation="for-loop"),
                    model_parts=[model],
                )
                scheduler = build_lr_scheduler(
                    LRSchedulerConfig(decay_ratio=1.0),
                    optimizers=optimizer.optimizers,
                    training_steps=8,
                )
                step(model, optimizer, scheduler)
                saved_model = {k: v.clone() for k, v in model.state_dict().items()}
                folder = f"{root}/cycle_{cycle}"
                manager = manager_for(
                    model, optimizer, scheduler, folder, async_mode=args.async_mode
                )
                active.append(manager)
                group = manager.pg
                assert group is not None
                assert manager.save(1)
                # A newer native model-only export must not hide the resume.
                assert manager.save(2, last_step=True)
                manager.close()
                manager.close()
                assert_group_released(group)
                assert manager.save_future is None
                dist.barrier()
                step(model, optimizer, scheduler)
                resumed_model = build_model()
                resumed_optimizer = OptimizersContainer(
                    OptimizerConfig(learning_rate=0.3, implementation="for-loop"),
                    model_parts=[resumed_model],
                )
                resumed_scheduler = build_lr_scheduler(
                    LRSchedulerConfig(decay_ratio=1.0),
                    optimizers=resumed_optimizer.optimizers,
                    training_steps=8,
                )
                resumed = manager_for(
                    resumed_model,
                    resumed_optimizer,
                    resumed_scheduler,
                    folder,
                    async_mode=args.async_mode,
                )
                active.append(resumed)
                resumed_group = resumed.pg
                assert resumed._find_load_step() == 1
                assert resumed.load(-1)
                torch.testing.assert_close(resumed_model.state_dict(), saved_model)
                resumed.close()
                assert_group_released(resumed_group)
                # Verify the restored optimizer and schedule by continuing one step.
                step(resumed_model, resumed_optimizer, resumed_scheduler)
                torch.testing.assert_close(
                    resumed_model.state_dict(), model.state_dict(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    resumed_optimizer.state_dict(),
                    optimizer.state_dict(),
                    rtol=0,
                    atol=0,
                )
                assert resumed_scheduler.state_dict() == scheduler.state_dict()
                probe = torch.tensor(rank + 1, device=harness_device())
                dist.all_reduce(probe)
                assert probe.item() == 3
                if rank == 0:
                    print(
                        f"cycle {cycle}: {args.async_mode} fsdp={args.fsdp} "
                        "resume and group teardown match"
                    )
            dist.barrier()
    finally:
        for manager in active:
            manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
