"""FSDP AdamW/DCP resume matches uninterrupted and dense-reference training.

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/fsdp_checkpoint_equivalence.py

Runs native/custom adapters with shard and replicate meshes on CPU/Gloo,
including shard-to-replicate and replicate-to-shard checkpoint restoration.
Custom here means the current native adapter, not the pending relocated-hook
engine. Uses rank-distinct batches, a decaying LR, and fresh resume objects.
Checks loss, all gradients, final parameters, Adam moments/counters, and LR.
Target-device kernels and async checkpoint transports are not covered.
"""

from contextlib import nullcontext
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from fsdp_mesh_equivalence import SEQ, _build_model, _data, _full, _loss_sum
from torch.distributed.tensor import DTensor

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.components.optimizer.state_dict import get_flat_optim_state_dict
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.topology import ParallelTopology

STEPS = 6
RESUME_STEP = 2
LR = 0.01
ATOL = 1e-5


def runtime(cfg, topology, *, seed=0, learning_rate=LR):
    model = apply_fsdp(_build_model(seed), cfg, topology)
    weight = model.tok_embeddings.weight
    assert isinstance(weight, DTensor), "FSDP did not transform embedding weight"
    expected_rows = weight.shape[0] // cfg.data_parallel_shard_size
    assert weight.to_local().shape[0] == expected_rows, "Unexpected shard layout"
    optimizer = OptimizersContainer(
        OptimizerConfig(
            learning_rate=learning_rate,
            weight_decay=0.1,
            implementation="for-loop",
        ),
        model_parts=[model],
    )
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(decay_ratio=1.0),
        optimizers=optimizer.optimizers,
        training_steps=STEPS,
    )
    return model, optimizer, scheduler


def checkpoint_manager(model, optimizer, scheduler, folder):
    return DistributedCheckpointManager(
        CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={},
        folder=folder,
    )


def update(model, optimizer, scheduler, step, rank):
    optimizer.zero_grad(set_to_none=True)
    loss = _loss_sum(model, *_data(step, rank)) / (2 * SEQ)
    loss.backward()
    gradients = {name: _full(p.grad).clone() for name, p in model.named_parameters()}
    optimizer.step()
    scheduler.step()
    reduced = loss.detach().clone()
    dist.all_reduce(reduced)
    return reduced, gradients


def topology_for(mode, implementation):
    cfg = ParallelConfig(
        data_parallel_shard_size=2 if mode == "shard" else 1,
        data_parallel_replicate_size=2 if mode == "ddp" else 1,
    )
    cfg.fsdp_config.fsdp_implementation = implementation
    topology = ParallelTopology.from_config(cfg, 2)
    topology.build_device_mesh()
    return cfg, topology


def full_optimizer_state(optimizer):
    return {key: _full(value) for key, value in optimizer.state_dict().items()}


def check_case(mode, resume_mode, implementation, root, rank):
    cfg, topology = topology_for(mode, implementation)
    model, optimizer, scheduler = runtime(cfg, topology)
    reference = _build_model()
    reference_optimizer = torch.optim.AdamW(
        reference.named_parameters(), lr=LR, weight_decay=0.1, foreach=False
    )
    resumed = None
    folder = f"{root}/{mode}_{resume_mode}_{implementation}"
    resume_atol = 0 if mode == resume_mode else ATOL
    for step in range(STEPS):
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = sum(
            _loss_sum(reference, *_data(step, r)) for r in range(2)
        ) / (2 * SEQ)
        reference_loss.backward()
        # The schedule includes one virtual stable step so the final real
        # update has nonzero LR: update zero uses LR, then linear decay.
        for group in reference_optimizer.param_groups:
            group["lr"] = LR * (1 - step / STEPS)
        reference_optimizer.step()
        loss, gradients = update(model, optimizer, scheduler, step, rank)
        torch.testing.assert_close(loss, reference_loss.detach(), rtol=0, atol=ATOL)
        for name, param in reference.named_parameters():
            torch.testing.assert_close(gradients[name], param.grad, rtol=0, atol=ATOL)
        if resumed is not None:
            resumed_loss, resumed_gradients = update(*resumed, step, rank)
            torch.testing.assert_close(resumed_loss, loss, rtol=0, atol=resume_atol)
            torch.testing.assert_close(
                resumed_gradients, gradients, rtol=0, atol=resume_atol
            )
        if step + 1 == RESUME_STEP:
            manager = checkpoint_manager(model, optimizer, scheduler, folder)
            try:
                assert manager.save(RESUME_STEP)
            finally:
                manager.close()
            dist.barrier()
            # Different weights and base LR make a missing restore observable.
            resume_cfg, resume_topology = topology_for(resume_mode, implementation)
            resumed = runtime(resume_cfg, resume_topology, seed=123, learning_rate=0.3)
            loader = checkpoint_manager(*resumed, folder)
            try:
                assert loader.load(RESUME_STEP)
            finally:
                loader.close()
            assert resumed[2].state_dict() == scheduler.state_dict()
            assert resumed[2].get_metrics() == scheduler.get_metrics()
            torch.testing.assert_close(
                {name: _full(p) for name, p in resumed[0].named_parameters()},
                {name: _full(p) for name, p in model.named_parameters()},
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                full_optimizer_state(resumed[1]),
                full_optimizer_state(optimizer),
                rtol=0,
                atol=0,
            )

    resumed_model, resumed_optimizer, resumed_scheduler = resumed
    full_params = {name: _full(p) for name, p in model.named_parameters()}
    torch.testing.assert_close(
        full_params, dict(reference.named_parameters()), rtol=0, atol=ATOL
    )
    torch.testing.assert_close(
        {name: _full(p) for name, p in resumed_model.named_parameters()},
        full_params,
        rtol=0,
        atol=resume_atol,
    )
    state = full_optimizer_state(optimizer)
    torch.testing.assert_close(
        full_optimizer_state(resumed_optimizer), state, rtol=0, atol=resume_atol
    )
    reference_state = get_flat_optim_state_dict(reference_optimizer)
    for key, value in state.items():
        if key.startswith("state."):
            torch.testing.assert_close(
                _full(value), reference_state[key], rtol=0, atol=ATOL
            )
    assert (
        scheduler.state_dict()
        == resumed_scheduler.state_dict()
        == {"last_epoch": STEPS}
    )
    if rank == 0:
        print(
            f"{mode}->{resume_mode}/{implementation}: "
            f"AdamW resume atol={resume_atol}; dense oracle atol={ATOL}"
        )


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        context = TemporaryDirectory() if rank == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            for mode in ("shard", "ddp"):
                for resume_mode in ("shard", "ddp"):
                    for implementation in ("native", "custom"):
                        check_case(mode, resume_mode, implementation, paths[0], rank)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
