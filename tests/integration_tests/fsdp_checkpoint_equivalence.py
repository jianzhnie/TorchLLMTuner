"""FSDP AdamW/DCP resume matches uninterrupted and dense-reference training.

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/fsdp_checkpoint_equivalence.py

Runs native/custom engines with shard and replicate meshes on CPU/Gloo,
including shard-to-replicate and replicate-to-shard checkpoint restoration.
The custom (fine-grained) engine needs torch 2.14 private APIs; where they are
absent its cases skip with a printed note and only the native engine runs.
Custom here uses default module hooks; relocated hooks and multiple contexts
are not exercised. Uses rank-distinct batches, a decaying LR, and fresh objects.
Checks loss, all gradients, final parameters, Adam moments/counters, and LR.
Injects a rank-zero failure after a threaded same-step rewrite has written its
shards; a fresh runtime must still restore the original published checkpoint.
Target-device kernels and pinned-memory checkpoint transports are not covered.
Append ``--fsspec`` to exercise file:// URIs through the real fsspec adapter.
``--checkpoint-root s3://bucket/prefix`` (or gs://) uses a unique subdirectory
under that root and retains artifacts for inspection; requires the corresponding
fsspec driver and credentials. Cloud transports are not validated by local runs.
``--process-save-cpu`` uses the real DCP save subprocess with shared CPU staging;
pinning, async device staging and nonblocking device copies are disabled. This
validates worker serialization/I/O, not pinned-memory device transport.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
from fsdp_mesh_equivalence import SEQ, _build_model, _data, _full, _loss_sum
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.tensor import DTensor

from llmtuner.components.checkpointer import distributed_checkpoint
from llmtuner.components.checkpointer.checkpoint_writer import (
    GenerationFileSystemWriter,
)
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
from llmtuner.utils.seeding import set_deterministic_algorithms
from tests.caps import CAPS
from tests.integration_tests.harness import backend, device, init_distributed

STEPS = 6
RESUME_STEP = 2
LR = 0.01
ATOL = 1e-5


class _FailAfterWritingGenerationWriter(GenerationFileSystemWriter):
    """Serializable fault injector; execute the failure inside the save worker."""

    def _write_data(self, *args, **kwargs):
        result = super()._write_data(*args, **kwargs)
        if dist.get_rank() == 0:
            raise OSError("injected rank-zero failure after writing shards")
        return result


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


def checkpoint_manager(
    model,
    optimizer,
    scheduler,
    folder,
    *,
    async_mode="disabled",
    process_save_cpu=False,
):
    manager = DistributedCheckpointManager(
        CheckpointConfig(
            enable=True,
            interval=1,
            keep_latest_k=0,
            async_mode="async_with_pinned_mem" if process_save_cpu else async_mode,
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={},
        folder=folder,
    )
    if process_save_cpu:
        manager.stager = DefaultStager(
            StagingOptions(
                use_pinned_memory=False,
                use_shared_memory=True,
                use_async_staging=False,
                use_non_blocking_copy=False,
            )
        )
    return manager


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


def fail_same_step_rewrite(cfg, topology, folder, rank, *, process_save_cpu=False):
    replacement = runtime(cfg, topology, seed=987, learning_rate=0.7)
    manager = checkpoint_manager(
        *replacement, folder, async_mode="async", process_save_cpu=process_save_cpu
    )
    write_data = FileSystemWriter._write_data

    def fail_after_writing(self, *args, **kwargs):
        result = write_data(self, *args, **kwargs)
        if rank == 0:
            raise OSError("injected rank-zero failure after writing shards")
        return result

    try:
        injection = (
            patch.object(
                distributed_checkpoint,
                "GenerationFileSystemWriter",
                _FailAfterWritingGenerationWriter,
            )
            if process_save_cpu
            else patch.object(FileSystemWriter, "_write_data", fail_after_writing)
        )
        with injection:
            try:
                manager.save(RESUME_STEP)
                manager.maybe_wait_for_saving()
            except CheckpointException as error:
                assert "injected rank-zero failure" in str(error)
            else:
                raise AssertionError("The failed rewrite must fail on every rank")
    finally:
        manager.close()


def check_case(
    mode, resume_mode, implementation, root, rank, *, process_save_cpu=False
):
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
            manager = checkpoint_manager(
                model, optimizer, scheduler, folder, process_save_cpu=process_save_cpu
            )
            try:
                assert manager.save(RESUME_STEP)
            finally:
                manager.close()
            dist.barrier()
            fail_same_step_rewrite(
                cfg, topology, folder, rank, process_save_cpu=process_save_cpu
            )
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
            f"AdamW resume atol={resume_atol}; dense oracle atol={ATOL} "
            f"[device={device().type} backend={backend()}]"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group()
    location.add_argument(
        "--fsspec", action="store_true", help="Use file:// checkpoint URIs"
    )
    location.add_argument(
        "--checkpoint-root", help="Use a unique subfolder under this checkpoint root"
    )
    parser.add_argument(
        "--process-save-cpu",
        action="store_true",
        help="Exercise the save subprocess with shared CPU staging (no device pinning)",
    )
    args = parser.parse_args()
    init_distributed()
    # The same-layout cases below assert bit-equality between the continuous
    # run and the resumed one (``resume_atol = 0``), which is only a statement
    # about the checkpoint if the kernels themselves are deterministic: the
    # default NPU reductions are not, and differ by ~1 ULP between two runs
    # with identical state.
    set_deterministic_algorithms()
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        context = TemporaryDirectory() if rank == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            checkpoint_root = Path(paths[0]).as_uri() if args.fsspec else paths[0]
            if args.checkpoint_root:
                subfolder = f"equivalence-{Path(paths[0]).name}"
                checkpoint_root = f"{args.checkpoint_root.rstrip('/')}/{subfolder}"
            # The custom (fine-grained) engine imports torch 2.14 private APIs
            # at module scope; on older torch only the native implementation
            # can run. Gate instead of failing the whole sweep.
            implementations = ["native"]
            if CAPS["fine_grained_fsdp"]():
                implementations.append("custom")
            elif rank == 0:
                print("SKIP custom implementation: fine_grained_fsdp unavailable")
            for mode in ("shard", "ddp"):
                for resume_mode in ("shard", "ddp"):
                    for implementation in implementations:
                        check_case(
                            mode,
                            resume_mode,
                            implementation,
                            checkpoint_root,
                            rank,
                            process_save_cpu=args.process_save_cpu,
                        )
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
