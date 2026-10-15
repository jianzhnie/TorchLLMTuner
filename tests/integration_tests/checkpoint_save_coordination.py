"""Writer construction failures preserve recovery points and permit retries.

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    tests/integration_tests/checkpoint_save_coordination.py

CPU/Gloo, synchronous and threaded async DCP, real local checkpoint storage.
"""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

from llmtuner.components.checkpointer import distributed_checkpoint as checkpoint
from llmtuner.config.checkpoint_config import CheckpointConfig


def check_stager_failure(root, failing_rank):
    rank = dist.get_rank()
    folder = root / f"stager-{failing_rank}"
    torch.manual_seed(13)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = checkpoint.DistributedCheckpointManager(
        CheckpointConfig(
            enable=True,
            interval=1,
            keep_latest_k=2,
            async_mode="async_with_pinned_mem",
            last_save_model_only=False,
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        states={},
        folder=str(folder),
    )
    failure = OSError("injected stager construction failure")

    def cpu_stager(options):
        # Exercise the production constructor call with actual requested device
        # options, then adapt to this CPU runner. No device-pinning evidence.
        assert options.use_pinned_memory and options.use_non_blocking_copy
        return DefaultStager(
            StagingOptions(
                use_pinned_memory=False,
                use_shared_memory=True,
                use_async_staging=False,
                use_non_blocking_copy=False,
            )
        )

    def build_stager(options):
        if rank == failing_rank:
            raise failure
        return cpu_stager(options)

    try:
        # A forced full final save is synchronous and leaves no stager behind.
        before = {key: value.clone() for key, value in model.state_dict().items()}
        assert manager.save(1, last_step=True)
        assert manager.stager is None
        with torch.no_grad():
            model.weight.add_(3)
        with (
            patch.object(checkpoint, "DefaultStager", build_stager),
            patch.object(dcp, "async_save") as upload,
        ):
            try:
                manager.save(2)
            except OSError as error:
                assert rank == failing_rank and error is failure
            except RuntimeError as error:
                assert rank != failing_rank
                assert (
                    str(error)
                    == "Checkpoint stager construction failed on another rank"
                )
            else:
                raise AssertionError("asymmetric stager failure ignored")
            upload.assert_not_called()
        assert (manager.stager is None) == (rank == failing_rank)
        assert manager.save_future is None and manager.staging_future is None
        retained_stager = manager.stager
        dist.barrier()
        target = {key: torch.empty_like(value) for key, value in before.items()}
        dcp.load(target, checkpoint_id=folder / "checkpoint" / "step-1")
        torch.testing.assert_close(target, before, rtol=0, atol=0)
        with patch.object(checkpoint, "DefaultStager", cpu_stager):
            assert manager.save(2)
        if retained_stager is not None:
            assert manager.stager is retained_stager
        manager.maybe_wait_for_staging()
        manager.maybe_wait_for_saving()
        dcp.load(target, checkpoint_id=folder / "checkpoint" / "step-2")
        torch.testing.assert_close(target, model.state_dict(), rtol=0, atol=0)
    finally:
        manager.close()
    assert manager.stager is None
    dist.barrier()
    print(
        f"rank={rank} stager failed_rank={failing_rank}: PROCESS retry passed",
        flush=True,
    )


def check_staging_failure(root, failing_rank, asynchronous):
    rank = dist.get_rank()
    folder = root / f"staging-{failing_rank}-{asynchronous}"
    torch.manual_seed(14)
    model = torch.nn.Linear(2, 2)
    failure = OSError("injected staging copy failure")

    class FailOnceStager(DefaultStager):
        def __init__(self, options):
            super().__init__(
                StagingOptions(
                    use_pinned_memory=False,
                    use_shared_memory=True,
                    use_async_staging=True,
                    use_non_blocking_copy=False,
                )
            )
            self.fail = rank == failing_rank

        def stage(self, state):
            if self.fail:
                self.fail = False
                if asynchronous:

                    def fail_copy():
                        raise failure

                    return self._staging_executor.submit(fail_copy)
                raise failure
            return super().stage(state)

    optimizer = torch.optim.AdamW(model.parameters())
    manager = checkpoint.DistributedCheckpointManager(
        CheckpointConfig(
            enable=True,
            interval=1,
            keep_latest_k=2,
            async_mode="async_with_pinned_mem",
            last_save_model_only=False,
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        states={},
        folder=str(folder),
    )
    try:
        before = {key: value.clone() for key, value in model.state_dict().items()}
        assert manager.save(1, last_step=True)
        with torch.no_grad():
            model.weight.add_(4)
        with patch.object(checkpoint, "DefaultStager", FailOnceStager):
            assert manager.save(2)
        for wait in (manager.maybe_wait_for_staging, manager.maybe_wait_for_saving):
            try:
                wait()
            except OSError as error:
                assert rank == failing_rank and error is failure
            except RuntimeError as error:
                assert rank != failing_rank
                assert str(error) == "Checkpoint staging failed on another rank"
            else:
                raise AssertionError("failed staging reported success")
        assert manager.save_future is None and manager.staging_future is None
        assert not (folder / "checkpoint" / "step-2" / ".metadata").exists()
        target = {key: torch.empty_like(value) for key, value in before.items()}
        dcp.load(target, checkpoint_id=folder / "checkpoint" / "step-1")
        torch.testing.assert_close(target, before, rtol=0, atol=0)
        assert manager.save(2)
        manager.maybe_wait_for_staging()
        manager.maybe_wait_for_saving()
        dcp.load(target, checkpoint_id=folder / "checkpoint" / "step-2")
        torch.testing.assert_close(target, model.state_dict(), rtol=0, atol=0)
    finally:
        manager.close()
    dist.barrier()
    print(
        f"rank={rank} staging async={asynchronous} failed_rank={failing_rank}: "
        "PROCESS failure and retry passed",
        flush=True,
    )


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        with TemporaryDirectory() as local_root:
            roots = [local_root if rank == 0 else None]
            dist.broadcast_object_list(roots)
            root = Path(roots[0])
            for mode in ("disabled", "async"):
                for failing_rank in (0, 1):
                    folder = root / f"{mode}-{failing_rank}"
                    torch.manual_seed(12)
                    model = torch.nn.Linear(2, 2)
                    optimizer = torch.optim.AdamW(model.parameters())
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lambda _: 1
                    )
                    manager = checkpoint.DistributedCheckpointManager(
                        CheckpointConfig(
                            enable=True, interval=1, keep_latest_k=2, async_mode=mode
                        ),
                        model_parts=[model],
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        states={},
                        folder=str(folder),
                    )
                    try:
                        before = {
                            key: value.clone()
                            for key, value in model.state_dict().items()
                        }
                        assert manager.save(1)
                        manager.maybe_wait_for_saving()
                        failure = OSError("injected writer initialization failure")
                        original_writer = checkpoint.GenerationFileSystemWriter

                        def build_writer(
                            *args,
                            failing_rank=failing_rank,
                            failure=failure,
                            original_writer=original_writer,
                            **kwargs,
                        ):
                            if rank == failing_rank:
                                raise failure
                            return original_writer(*args, **kwargs)

                        with torch.no_grad():
                            model.weight.add_(2)
                        with patch.object(
                            checkpoint, "GenerationFileSystemWriter", build_writer
                        ):
                            try:
                                manager.save(2)
                            except OSError as error:
                                assert rank == failing_rank and error is failure
                            except RuntimeError as error:
                                assert rank != failing_rank
                                assert str(error) == (
                                    "Checkpoint writer construction failed "
                                    "on another rank"
                                )
                            else:
                                raise AssertionError(
                                    "asymmetric writer failure ignored"
                                )
                        dist.barrier()
                        assert manager.save_future is None
                        target = {
                            key: torch.empty_like(value)
                            for key, value in before.items()
                        }
                        dcp.load(target, checkpoint_id=folder / "checkpoint" / "step-1")
                        torch.testing.assert_close(target, before, rtol=0, atol=0)
                        assert manager.save(2)
                        manager.maybe_wait_for_saving()
                        dcp.load(target, checkpoint_id=folder / "checkpoint" / "step-2")
                        torch.testing.assert_close(
                            target, model.state_dict(), rtol=0, atol=0
                        )
                    finally:
                        manager.close()
                    dist.barrier()
                    print(
                        f"rank={rank} {mode} failed_rank={failing_rank}: "
                        "old snapshot and retry passed",
                        flush=True,
                    )
            for failing_rank in (0, 1):
                check_stager_failure(root, failing_rank)
            for asynchronous in (False, True):
                for failing_rank in (0, 1):
                    check_staging_failure(root, failing_rank, asynchronous)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
