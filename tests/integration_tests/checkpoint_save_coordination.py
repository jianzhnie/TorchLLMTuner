"""Writer construction failures preserve recovery points and permit retries.

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    tests/integration_tests/checkpoint_save_coordination.py

CPU/Gloo, synchronous and threaded async DCP, real local checkpoint storage.
"""

from concurrent.futures import Future
from dataclasses import replace
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


def check_local_preparation_failure(root):
    from llmtuner.components.checkpointer.random_state import RNGCheckpointState

    for category in ("lr_scheduler", "rng_state"):
        for failing_rank in (0, 1):
            for error_type in (ValueError, KeyboardInterrupt, SystemExit):
                torch.manual_seed(131)
                model = torch.nn.Linear(2, 2)
                optimizer = torch.optim.AdamW(model.parameters())
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
                rng = RNGCheckpointState(torch.device("cpu"))
                manager = checkpoint.DistributedCheckpointManager(
                    CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
                    model_parts=[model],
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    states={"rng_state": rng},
                    folder=str(
                        root
                        / f"prepare-{category}-{failing_rank}-{error_type.__name__}"
                    ),
                )
                target = scheduler if category == "lr_scheduler" else rng
                original = target.state_dict
                failure = error_type("injected local checkpoint preparation failure")

                def prepare(
                    *, original=original, failure=failure, failing_rank=failing_rank
                ):
                    if dist.get_rank() == failing_rank:
                        raise failure
                    return original()

                try:
                    assert manager.save(1)
                    with (
                        patch.object(target, "state_dict", prepare),
                        patch.object(
                            checkpoint.dcp, "load", wraps=checkpoint.dcp.load
                        ) as reader,
                    ):
                        caught = None
                        try:
                            manager.load(1)
                        except BaseException as error:
                            caught = error
                        if dist.get_rank() == failing_rank:
                            assert caught is failure
                        else:
                            assert isinstance(caught, RuntimeError), caught
                            phase = (
                                "preflight preparation"
                                if category == "rng_state"
                                else "state preparation"
                            )
                            assert str(caught) == (
                                f"Checkpoint local {phase} failed on another rank"
                            ), str(caught)
                        if category == "rng_state":
                            reader.assert_not_called()
                        else:
                            # RNG compatibility metadata is read first; the
                            # failed scheduler preparation must prevent full IO.
                            assert reader.call_count == 1
                            assert set(reader.call_args.args[0]) == {"rng_state"}
                    dist.barrier()
                    assert manager.load(1)
                finally:
                    manager.close()


def check_trainer_restore_failure(root):
    from llmtuner.trainer.trainer import Trainer

    for failing_rank in (0, 1):
        for error_type in (ValueError, KeyboardInterrupt, SystemExit):
            torch.manual_seed(130)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            trainer = Trainer.__new__(Trainer)
            trainer.step, trainer.ntokens_seen = 1, 2
            manager = checkpoint.DistributedCheckpointManager(
                CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
                model_parts=[model],
                optimizer=optimizer,
                lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
                states={"train_state": trainer},
                folder=str(
                    root / f"trainer-restore-{failing_rank}-{error_type.__name__}"
                ),
            )
            original = trainer.load_state_dict
            failure = error_type("injected trainer state restoration failure")

            def restore(
                state, *, original=original, failure=failure, failing_rank=failing_rank
            ):
                if dist.get_rank() == failing_rank:
                    raise failure
                original(state)

            try:
                assert manager.save(1)
                trainer.step, trainer.ntokens_seen = 7, 9
                with patch.object(trainer, "load_state_dict", restore):
                    caught = None
                    try:
                        manager.load(1)
                    except BaseException as error:
                        caught = error
                    if dist.get_rank() == failing_rank:
                        assert caught is failure
                    else:
                        assert isinstance(caught, RuntimeError), caught
                        assert str(caught) == (
                            "Checkpoint train_state restoration failed on another rank"
                        ), str(caught)
                dist.barrier()
                assert manager.load(1)
                assert trainer.state_dict() == {"step": 1, "ntokens_seen": 2}
            finally:
                manager.close()


def check_final_path_visibility(root):
    for visible_rank in (0, 1):
        torch.manual_seed(129)
        model = torch.nn.Linear(2, 2)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters())
        manager = checkpoint.DistributedCheckpointManager(
            CheckpointConfig(
                enable=True, interval=1, keep_latest_k=0, last_save_model_only=True
            ),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            states={},
            folder=str(root / f"final-visibility-{visible_rank}"),
        )
        try:
            assert manager.save(1)
            with torch.no_grad():
                model.weight.add_(2)
            after = {key: value.clone() for key, value in model.state_dict().items()}
            with patch.object(
                manager,
                "_is_resumable_checkpoint",
                return_value=dist.get_rank() == visible_rank,
            ):
                destination = manager._create_final_checkpoint_id(1)
                destinations = [None] * dist.get_world_size()
                dist.all_gather_object(destinations, destination)
                expected = str(Path(manager.folder) / "step-1" / "model-export")
                assert destinations == [expected] * dist.get_world_size(), destinations
                assert manager.save(1, last_step=True)
            target = {key: torch.empty_like(value) for key, value in before.items()}
            dcp.load(target, checkpoint_id=Path(manager.folder) / "step-1")
            torch.testing.assert_close(target, before, atol=0, rtol=0)
            dcp.load(target, checkpoint_id=expected)
            torch.testing.assert_close(target, after, atol=0, rtol=0)
            dist.barrier()
        finally:
            manager.close()


def check_final_path_failure(root):
    rank = dist.get_rank()
    for failing_rank in (0, 1):
        for error_type in (OSError, KeyboardInterrupt, SystemExit):
            torch.manual_seed(128)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            manager = checkpoint.DistributedCheckpointManager(
                CheckpointConfig(
                    enable=True,
                    interval=1,
                    keep_latest_k=0,
                    last_save_model_only=True,
                ),
                model_parts=[model],
                optimizer=optimizer,
                lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
                states={},
                folder=str(root / f"final-path-{failing_rank}-{error_type.__name__}"),
            )
            failure = error_type("injected final checkpoint metadata probe failure")
            probe = manager._is_resumable_checkpoint

            def inspect(
                path, *, failure=failure, probe=probe, failing_rank=failing_rank
            ):
                if rank == failing_rank:
                    raise failure
                return probe(path)

            try:
                assert manager.save(1)
                expected = {
                    key: value.clone() for key, value in model.state_dict().items()
                }
                with (
                    patch.object(manager, "_is_resumable_checkpoint", inspect),
                    patch.object(manager, "_save_checkpoint") as writer,
                ):
                    caught = None
                    try:
                        manager.save(1, last_step=True)
                    except BaseException as error:
                        caught = error
                    if rank == failing_rank:
                        assert caught is failure
                    else:
                        assert isinstance(caught, RuntimeError), caught
                        assert str(caught) == (
                            "Checkpoint final destination failed on another rank"
                        )
                    writer.assert_not_called()
                dist.barrier()
                assert manager.save(1, last_step=True)
                with torch.no_grad():
                    model.weight.add_(1)
                assert manager.load(1)
                torch.testing.assert_close(model.state_dict(), expected, atol=0, rtol=0)
                assert (
                    Path(manager.folder) / "step-1" / "model-export" / ".metadata"
                ).is_file()
            finally:
                manager.close()


def check_retention_preflight(root):
    rank = dist.get_rank()
    for error_type in (OSError, KeyboardInterrupt, SystemExit):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        manager = checkpoint.DistributedCheckpointManager(
            CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            states={},
            folder=str(root / f"retention-{error_type.__name__}"),
        )
        failure = error_type("injected retention scan failure")

        def purge(*, failure=failure, **kwargs):
            if rank == 0:
                raise failure

        try:
            with (
                patch.object(manager, "_purge_stale_checkpoints", purge),
                patch.object(manager, "_save_checkpoint") as writer,
            ):
                try:
                    manager.save(1)
                except error_type as error:
                    assert rank == 0 and error is failure
                except RuntimeError as error:
                    assert rank == 1 and str(error) == (
                        "Checkpoint retention preflight failed on another rank"
                    )
                else:
                    raise AssertionError("Retention failure was not coordinated")
                writer.assert_not_called()
            dist.barrier()
            assert manager.save(1)
            assert manager.load(1)
        finally:
            manager.close()


def check_completed_retention(root):
    rank = dist.get_rank()
    for last_step in (False, True):
        torch.manual_seed(120)
        model = torch.nn.Linear(2, 2)
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters())
        manager = checkpoint.DistributedCheckpointManager(
            CheckpointConfig(
                enable=True, interval=1, keep_latest_k=0, last_save_model_only=False
            ),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            states={},
            folder=str(root / f"completed-retention-{last_step}"),
        )
        failure = OSError("injected post-publication retention failure")
        original = manager._purge_stale_checkpoints

        def purge(*, completed=False, failure=failure, original=original, **kwargs):
            if completed and rank == 0:
                raise failure
            return original(completed=completed, **kwargs)

        try:
            with patch.object(manager, "_purge_stale_checkpoints", purge):
                try:
                    manager.save(1, last_step=last_step)
                except OSError as error:
                    assert rank == 0 and error is failure
                except RuntimeError as error:
                    assert rank == 1 and str(error) == (
                        "Checkpoint completed retention failed on another rank"
                    )
                else:
                    raise AssertionError("Completed retention failure not coordinated")
            dist.barrier()
            with torch.no_grad():
                model.weight.add_(1)
            assert manager.load(1)
            torch.testing.assert_close(model.state_dict(), expected, atol=0, rtol=0)
            assert manager.save(2)
        finally:
            manager.close()


def check_pending_save_failure(root):
    rank = dist.get_rank()
    for phase in ("future", "retention"):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        manager = checkpoint.DistributedCheckpointManager(
            CheckpointConfig(
                enable=True, interval=1, keep_latest_k=0, async_mode="async"
            ),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            states={},
            folder=str(root / f"pending-{phase}"),
        )
        failure = OSError(f"injected completed {phase} failure")
        pending = Future()
        if rank == 0 and phase == "future":
            pending.set_exception(failure)
        else:
            pending.set_result(None)
        manager.save_future = pending
        manager._pending_retention = {"saving_step": 0}

        def purge(*, failure=failure, phase=phase, **kwargs):
            if rank == 0 and phase == "retention":
                raise failure

        try:
            with (
                patch.object(manager, "_purge_stale_checkpoints", purge),
                patch.object(manager, "_save_checkpoint") as writer,
            ):
                try:
                    manager.save(1)
                except OSError as error:
                    assert rank == 0 and error is failure
                except RuntimeError as error:
                    assert rank == 1 and str(error) == (
                        "Checkpoint pending save failed on another rank"
                    )
                else:
                    raise AssertionError("Pending save failure not coordinated")
                writer.assert_not_called()
            assert manager.save_future is None
            assert manager._pending_retention is None
            dist.barrier()
            assert manager.save(1)
            manager.maybe_wait_for_saving()
            assert manager.load(1)
        finally:
            manager.close()


def check_load_wait_failure(root):
    rank = dist.get_rank()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = checkpoint.DistributedCheckpointManager(
        CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        states={},
        folder=str(root / "load-wait"),
    )
    try:
        assert manager.save(1)
        for method, phase in (
            ("maybe_wait_for_staging", "Checkpoint load staging"),
            ("maybe_wait_for_saving", "Checkpoint pending save"),
        ):
            for error_type in (OSError, KeyboardInterrupt, SystemExit):
                failure = error_type("injected load wait failure")

                def wait(*, failure=failure):
                    if rank == 0:
                        raise failure

                with (
                    patch.object(manager, method, wait),
                    patch.object(manager, "_resolve_load_source") as resolve,
                ):
                    try:
                        manager.load(1)
                    except error_type as error:
                        assert rank == 0 and error is failure
                    except RuntimeError as error:
                        assert (
                            rank == 1
                            and str(error) == f"{phase} failed on another rank"
                        )
                    else:
                        raise AssertionError("Load wait failure not coordinated")
                    resolve.assert_not_called()
                dist.barrier()
                assert manager.load(1)
    finally:
        manager.close()


def check_save_argument_agreement(root):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = checkpoint.DistributedCheckpointManager(
        CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        states={},
        folder=str(root / "argument-agreement"),
    )
    try:
        for step, last in ((dist.get_rank() + 1, False), (1, dist.get_rank() == 0)):
            # Intercept the backend so the unfixed case fails promptly rather
            # than entering different collective sequences.
            with patch.object(manager, "_save", return_value=True) as save:
                try:
                    manager.save(step, last_step=last)
                except RuntimeError as error:
                    assert "inconsistent save arguments" in str(error)
                else:
                    raise AssertionError("rank-divergent save arguments accepted")
                save.assert_not_called()
            dist.barrier()
        for field, value in (("interval", 2), ("load_only", True)):
            original = getattr(manager, field)
            if dist.get_rank() == 0:
                setattr(manager, field, value)
            try:
                with patch.object(manager, "_save", return_value=True) as save:
                    try:
                        manager.save(3)
                    except RuntimeError as error:
                        assert "inconsistent save policy" in str(error)
                    else:
                        raise AssertionError("rank-divergent save policy accepted")
                    save.assert_not_called()
                dist.barrier()
            finally:
                setattr(manager, field, original)
        for field, value in (
            ("folder", str(root / "other-destination")),
            ("last_save_model_only", not manager.last_save_model_only),
            ("last_save_in_hf", not manager.last_save_in_hf),
            ("export_dtype", torch.float16),
            ("async_mode", checkpoint.DistributedCheckpointAsyncMode.ASYNC),
        ):
            original = getattr(manager, field)
            if dist.get_rank() == 0:
                setattr(manager, field, value)
            try:
                with patch.object(manager, "_save", return_value=True) as save:
                    try:
                        manager.save(1, last_step=True)
                    except RuntimeError as error:
                        assert "inconsistent save configuration" in str(error)
                    else:
                        raise AssertionError(
                            "rank-divergent save configuration accepted"
                        )
                    save.assert_not_called()
                dist.barrier()
            finally:
                setattr(manager, field, original)
        # Agreement must allow an actual publication after rejected calls.
        assert manager.save(1, last_step=True)
        dist.barrier()
    finally:
        manager.close()


def check_load_preflight(root):
    rank = dist.get_rank()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = checkpoint.DistributedCheckpointManager(
        CheckpointConfig(enable=True, last_save_model_only=False, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        states={},
        folder=str(root / "load-preflight"),
    )
    try:
        assert manager.save(1, last_step=True)
        saved = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            model.weight.add_(3)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        for method in ("_resolve_load_source", "_states_to_load"):
            for failing_rank in (0, 1):
                failure = OSError(f"injected {method} failure")
                original = getattr(manager, method)

                def preflight(
                    *args,
                    failing_rank=failing_rank,
                    failure=failure,
                    original=original,
                    **kwargs,
                ):
                    if rank == failing_rank:
                        raise failure
                    return original(*args, **kwargs)

                with (
                    patch.object(manager, method, preflight),
                    patch.object(manager, "_load_checkpoint") as load,
                ):
                    try:
                        manager.load(1)
                    except OSError as error:
                        assert rank == failing_rank and error is failure
                    except RuntimeError as error:
                        assert rank != failing_rank
                        assert str(error) == (
                            "Checkpoint load preflight failed on another rank"
                        )
                    else:
                        raise AssertionError("load preflight failure ignored")
                    load.assert_not_called()
                torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
                dist.barrier()
        source = manager._resolve_load_source(1)
        assert source is not None
        for mismatch in ("missing", "path", "model_only", "format", "states"):
            alternate = source
            if rank == 0:
                if mismatch == "missing":
                    alternate = None
                elif mismatch == "path":
                    alternate = replace(
                        source, checkpoint_id=source.checkpoint_id + "-other"
                    )
                elif mismatch == "model_only":
                    alternate = replace(source, model_only=True)
                elif mismatch == "format":
                    alternate = replace(source, from_hf=True)
            selected_states = manager._states_to_load(model_only=False)
            if rank == 0 and mismatch == "states":
                selected_states = {"model": manager.states["model"]}
            with (
                patch.object(manager, "_resolve_load_source", return_value=alternate),
                patch.object(manager, "_states_to_load", return_value=selected_states),
                patch.object(manager, "_load_checkpoint") as load,
            ):
                try:
                    manager.load(1)
                except RuntimeError as error:
                    assert "inconsistent restore sources or state categories" in str(
                        error
                    )
                else:
                    raise AssertionError(f"{mismatch}: inconsistent restore accepted")
                load.assert_not_called()
            torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
            dist.barrier()
        with (
            patch.object(manager, "_resolve_load_source", return_value=None),
            patch.object(manager, "_load_checkpoint") as load,
        ):
            assert not manager.load()
            load.assert_not_called()
        dist.barrier()
        assert manager.load(1)
        torch.testing.assert_close(model.state_dict(), saved, rtol=0, atol=0)
        print(f"rank={rank}: load preflight failures and real DCP retry passed")
    finally:
        manager.close()


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


def check_staging_failure(root, failing_rank, asynchronous, error_type):
    rank = dist.get_rank()
    folder = root / f"staging-{failing_rank}-{asynchronous}-{error_type.__name__}"
    torch.manual_seed(14)
    model = torch.nn.Linear(2, 2)
    failure = error_type("injected staging copy failure")

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
        for index, wait in enumerate(
            (manager.maybe_wait_for_staging, manager.maybe_wait_for_saving)
        ):
            try:
                wait()
            except BaseException as error:
                if rank == failing_rank:
                    if index == 0 or isinstance(failure, Exception):
                        assert error is failure
                    else:
                        assert isinstance(error, RuntimeError)
                        assert error.__cause__ is failure
                else:
                    assert isinstance(error, RuntimeError)
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
        f"rank={rank} staging async={asynchronous} failed_rank={failing_rank} "
        f"{error_type.__name__}: "
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
                        # Exceed the horizon with an actual publication. Earlier
                        # saves only fill it and never exercise the purge worker.
                        with torch.no_grad():
                            model.bias.add_(3)
                        assert manager.save(3)
                        manager.maybe_wait_for_saving()
                    finally:
                        manager.close()
                    dist.barrier()
                    assert not manager.purge_thread.is_alive()
                    assert sorted(
                        path.name for path in (folder / "checkpoint").glob("step-*")
                    ) == ["step-2", "step-3"]
                    dcp.load(target, checkpoint_id=folder / "checkpoint" / "step-3")
                    torch.testing.assert_close(
                        target, model.state_dict(), rtol=0, atol=0
                    )
                    print(
                        f"rank={rank} {mode} failed_rank={failing_rank}: "
                        "old snapshot, retry and completed retention passed",
                        flush=True,
                    )
            check_local_preparation_failure(root)
            check_trainer_restore_failure(root)
            check_final_path_visibility(root)
            check_final_path_failure(root)
            check_retention_preflight(root)
            check_completed_retention(root)
            check_pending_save_failure(root)
            check_load_wait_failure(root)
            check_save_argument_agreement(root)
            check_load_preflight(root)
            for failing_rank in (0, 1):
                check_stager_failure(root, failing_rank)
            for asynchronous in (False, True):
                for failing_rank in (0, 1):
                    for error_type in (OSError, KeyboardInterrupt, SystemExit):
                        check_staging_failure(
                            root, failing_rank, asynchronous, error_type
                        )
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
