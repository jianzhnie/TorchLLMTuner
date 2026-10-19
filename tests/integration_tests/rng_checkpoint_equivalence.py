"""Rank-local RNG/DCP dropout resume, using real AdamW and CPU DDP/FSDP.

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/rng_checkpoint_equivalence.py

Each rank uses different Python, NumPy and torch streams. Checks resumed loss,
parameters and optimizer state exactly after two additional stochastic updates.
Runs sync and async saves; advances training after staging and before close.
Custom FSDP relocates both linear units to the root hook, producing three
hook-owned contexts. Uses the actual DistributedCheckpointManager; Trainer
construction and topology-changing restores are not covered.
This covers default CPU streams, not private loader or DTensor generators.
"""

import random
from contextlib import nullcontext
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel

from llmtuner.components.checkpointer.checkpoint_keys import RNG_STATE
from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.checkpointer.random_state import RNGCheckpointState
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.parallel.fully_shard_parallel.fine_grained import (
    fully_shard_fine_grained,
)


def runtime(seed, mesh=None, *, custom=False):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.Dropout(0.5), torch.nn.Linear(8, 2)
    )
    if mesh is None:
        execution_model = DistributedDataParallel(model)
    else:
        shard = fully_shard_fine_grained if custom else fully_shard
        for layer in model:
            if isinstance(layer, torch.nn.Linear):
                if custom:
                    shard(layer, mesh=mesh, hook_module=model)
                else:
                    shard(layer, mesh=mesh)
        shard(model, mesh=mesh)
        if custom:
            state = model._get_fsdp_state()
            model(torch.ones(4, 4)).sum().backward()
            assert len(state.communication_contexts) == 3
            model.zero_grad(set_to_none=True)
        assert isinstance(model[0].weight, DTensor)
        assert model[0].weight.to_local().shape[0] == 4
        execution_model = model
    optimizer = OptimizersContainer(
        OptimizerConfig(learning_rate=0.01, implementation="for-loop"),
        model_parts=[model],
    )
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(), optimizers=optimizer.optimizers, training_steps=4
    )
    return model, execution_model, optimizer, scheduler


def manager(runtime, folder, *, async_mode="disabled"):
    model, _, optimizer, scheduler = runtime
    return DistributedCheckpointManager(
        CheckpointConfig(
            enable=True, interval=1, keep_latest_k=0, async_mode=async_mode
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={RNG_STATE: RNGCheckpointState(torch.device("cpu"))},
        folder=folder,
    )


def update(runtime):
    _, execution_model, optimizer, scheduler = runtime
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.arange(16).float().reshape(4, 4) / 16
    inputs = inputs * (0.5 + random.random() + np.random.random())
    loss = execution_model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach().clone()


def main():
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        context = TemporaryDirectory() if rank == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            mesh = init_device_mesh("cpu", (2,))
            for backend, mode in (
                ("ddp", "disabled"),
                ("ddp", "async"),
                ("fsdp", "disabled"),
                ("fsdp", "async"),
                ("custom", "disabled"),
                ("custom", "async"),
            ):
                selected_mesh = mesh if backend != "ddp" else None
                original = runtime(42, selected_mesh, custom=backend == "custom")
                random.seed(100 + rank)
                np.random.seed(200 + rank)
                torch.manual_seed(300 + rank)
                for _ in range(2):
                    update(original)
                folder = f"{paths[0]}/{backend}_{mode}"
                writer = manager(original, folder, async_mode=mode)
                try:
                    assert writer.save(2)
                    if mode == "async":
                        assert writer.save_future is not None
                    writer.maybe_wait_for_staging()
                    expected_losses = [update(original) for _ in range(2)]
                finally:
                    writer.close()
                resumed = runtime(123, selected_mesh, custom=backend == "custom")
                reader = manager(resumed, folder, async_mode=mode)
                try:
                    assert reader.load(2)
                finally:
                    reader.close()
                actual_losses = [update(resumed) for _ in range(2)]
                torch.testing.assert_close(
                    actual_losses, expected_losses, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    resumed[0].state_dict(), original[0].state_dict(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    resumed[2].state_dict(), original[2].state_dict(), rtol=0, atol=0
                )
                assert resumed[3].state_dict() == original[3].state_dict()
                dist.barrier()
                if rank == 0:
                    print(f"{backend}/{mode}: rank-distinct dropout resume exact")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
