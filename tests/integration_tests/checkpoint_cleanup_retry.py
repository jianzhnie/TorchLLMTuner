"""Real Gloo groups and staging workers survive interrupted cleanup.

Run with torchrun --nproc_per_node=2. Inject failures in executor.shutdown or
before/after group destruction; ownership, collectives and close retries are real.
"""

from datetime import timedelta
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.trainer.trainer import Trainer


def check_cleanup(error_type, failing_rank):
    with TemporaryDirectory() as root:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        manager = DistributedCheckpointManager(
            CheckpointConfig(enable=True, keep_latest_k=0, async_mode="async"),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            states={},
            folder=root,
        )
        group = manager.pg
        assert group is not None
        stager = DefaultStager(
            StagingOptions(
                use_pinned_memory=False,
                use_shared_memory=False,
                use_async_staging=True,
            )
        )
        manager.stager = stager
        executor = stager._staging_executor
        assert executor is not None
        executor.submit(lambda: None).result()
        trainer = Trainer.__new__(Trainer)
        trainer.checkpointer = manager
        released = []
        for name in ("metrics", "dataloader", "gc_handler"):
            setattr(
                trainer,
                name,
                SimpleNamespace(close=lambda name=name: released.append(name)),
            )
        failure = error_type("interrupted real staging executor shutdown")
        try:
            interrupted = failing_rank is None or dist.get_rank() == failing_rank
            if interrupted:
                with patch.object(executor, "shutdown", side_effect=failure):
                    caught = None
                    try:
                        trainer.close()
                    except BaseException as error:
                        caught = error
                    assert caught is failure
            assert trainer.checkpointer is manager
            assert manager._initialized
            assert manager.stager is stager
            assert manager.pg is group
            assert any(thread.is_alive() for thread in executor._threads)
            expected = ["metrics", "dataloader", "gc_handler"] if interrupted else []
            assert released == expected
            # The retained owned group must remain usable, not just referenced.
            value = torch.tensor(dist.get_rank() + 1)
            dist.all_reduce(value, group=group)
            assert value.item() == 3
            dist.barrier()
            trainer.close()
            assert trainer.checkpointer is None
            assert manager.stager is None and manager.pg is None
            assert not manager._initialized
            assert all(not thread.is_alive() for thread in executor._threads)
            try:
                dist.get_process_group_ranks(group)
            except KeyError:
                pass
            else:
                raise AssertionError("closed checkpoint group is still registered")
            trainer.close()
            assert released == ["metrics", "dataloader", "gc_handler"]
            dist.barrier()
        finally:
            stager.close()
            manager.close()


def check_group_cleanup(error_type, failing_rank):
    with TemporaryDirectory() as root:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        manager = DistributedCheckpointManager(
            CheckpointConfig(enable=True, keep_latest_k=0, async_mode="async"),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            states={},
            folder=root,
        )
        group = manager.pg
        failure = error_type("failure before real group destruction")
        destroy = dist.destroy_process_group
        calls = []

        def interrupted_destroy(target):
            calls.append(target)
            raise failure

        try:
            if failing_rank is None or dist.get_rank() == failing_rank:
                with patch.object(dist, "destroy_process_group", interrupted_destroy):
                    try:
                        manager.close()
                    except BaseException as error:
                        assert error is failure
                    else:
                        raise AssertionError("group destruction failure was swallowed")
                assert calls == [group]
            assert manager.pg is group
            assert manager._initialized
            assert dist.get_process_group_ranks(group) == [0, 1]
            value = torch.tensor(dist.get_rank() + 1)
            dist.all_reduce(value, group=group)
            assert value.item() == 3
            dist.barrier()
            with patch.object(dist, "destroy_process_group", wraps=destroy) as cleanup:
                manager.close()
                manager.close()
                assert cleanup.call_count == 1
            assert manager.pg is None and not manager._initialized
            try:
                dist.get_process_group_ranks(group)
            except KeyError:
                pass
            else:
                raise AssertionError("retry left group registered")
            dist.barrier()
        finally:
            # Also release a group whose reference the broken implementation lost.
            try:
                dist.get_process_group_ranks(group)
            except KeyError:
                pass
            else:
                destroy(group)
            manager.pg = None
            manager.close()


def check_completed_group_cleanup(error_type, failing_rank, failing_name):
    with TemporaryDirectory() as root:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        manager = DistributedCheckpointManager(
            CheckpointConfig(enable=True, keep_latest_k=0, async_mode="async"),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            states={},
            folder=root,
        )
        manager.staging_pg = dist.new_group(backend="gloo")
        groups = [manager.staging_pg, manager.pg]
        failing_group = getattr(manager, failing_name)
        failure = error_type("failure after real group destruction")
        destroy = dist.destroy_process_group
        calls = []
        interrupted = failing_rank is None or dist.get_rank() == failing_rank

        def completed_destroy(target):
            calls.append(target)
            destroy(target)
            if interrupted and target is failing_group:
                raise failure

        try:
            with patch.object(dist, "destroy_process_group", completed_destroy):
                caught = None
                try:
                    manager.close()
                except BaseException as error:
                    caught = error
                assert caught is (failure if interrupted else None)
                assert manager.staging_pg is None and manager.pg is None
                assert calls == groups
                manager.close()
                manager.close()
                assert calls == groups
                assert not manager._initialized
            for group in groups:
                try:
                    dist.get_process_group_ranks(group)
                except KeyError:
                    pass
                else:
                    raise AssertionError("completed shutdown left a registered group")
            dist.barrier()
        finally:
            manager.close()


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    try:
        assert dist.get_world_size() == 2
        for failing_rank in (None, 0, 1):
            for error_type in (KeyboardInterrupt, SystemExit, OSError):
                check_cleanup(error_type, failing_rank)
                check_group_cleanup(error_type, failing_rank)
                for name in ("staging_pg", "pg"):
                    check_completed_group_cleanup(error_type, failing_rank, name)
        if dist.get_rank() == 0:
            print(
                "Checkpoint cleanup retains live groups and releases each group once."
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
