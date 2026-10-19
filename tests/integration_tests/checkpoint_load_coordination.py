"""Two CPU/Gloo ranks: local load reporting and GC failures stay coordinated.

Run with torchrun --nproc_per_node=2. Uses real DCP save/load and AdamW state;
also checks asymmetric invalid save/load arguments before staging or source IO.
Reporting/GC cases interrupt only a selected logger or cleanup call.
"""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer import base, distributed_checkpoint
from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.utils.garbage_collection import GarbageCollection
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        with TemporaryDirectory() if rank == 0 else nullcontext(None) as root:
            roots = [root]
            dist.broadcast_object_list(roots, src=0)
            torch.manual_seed(123)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
            model(torch.ones(2, 2)).square().sum().backward()
            optimizer.step()
            scheduler.step()
            states = (model, optimizer, scheduler)
            manager = DistributedCheckpointManager(
                CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
                model_parts=[model],
                optimizer=optimizer,
                lr_scheduler=scheduler,
                states={},
                folder=roots[0],
            )
            try:
                assert manager.save(1)
                saved = deepcopy([state.state_dict() for state in states])
                for invalid_step in (None, True, -2, 1.5):
                    for failing_rank in (0, 1):
                        before = deepcopy([state.state_dict() for state in states])
                        try:
                            manager.load(invalid_step if rank == failing_rank else 1)
                        except ValueError as error:
                            assert rank == failing_rank
                            assert "checkpoint load step must be an integer" in str(
                                error
                            )
                        except RuntimeError as error:
                            assert rank != failing_rank
                            assert str(error) == (
                                "Checkpoint load arguments failed on another rank"
                            )
                        else:
                            raise AssertionError("Invalid load argument accepted")
                        assert_state_equal(
                            [state.state_dict() for state in states], before
                        )
                        dist.barrier()
                        assert manager.load(1)
                        assert_state_equal(
                            [state.state_dict() for state in states], saved
                        )
                for argument, invalid in (
                    ("step", None),
                    ("step", True),
                    ("step", -1),
                    ("step", 1.5),
                    ("last_step", None),
                    ("last_step", 1),
                    ("last_step", "false"),
                ):
                    for failing_rank in (0, 1):
                        before = deepcopy([state.state_dict() for state in states])
                        step = (
                            invalid
                            if argument == "step" and rank == failing_rank
                            else 1
                        )
                        last = (
                            invalid
                            if argument == "last_step" and rank == failing_rank
                            else False
                        )
                        try:
                            manager.save(step, last_step=last)
                        except ValueError as error:
                            assert rank == failing_rank
                            assert f"checkpoint save {argument}" in str(error)
                        except RuntimeError as error:
                            assert rank != failing_rank
                            assert str(error) == (
                                "Checkpoint save arguments failed on another rank"
                            )
                        else:
                            raise AssertionError("Invalid save argument accepted")
                        assert_state_equal(
                            [state.state_dict() for state in states], before
                        )
                        dist.barrier()
                        assert manager.load(1)
                        assert_state_equal(
                            [state.state_dict() for state in states], saved
                        )
                for phase in (
                    "reporting",
                    "completion reporting",
                    "final reporting",
                    "final model preparation",
                    "final state reporting",
                ):
                    for failing_rank in (0, 1):
                        for error_type in (OSError, KeyboardInterrupt, SystemExit):
                            manager.last_save_model_only = (
                                phase != "final state reporting"
                            )
                            failure = error_type("injected save reporting failure")
                            original_info = distributed_checkpoint.logger.info

                            def save_report(
                                message,
                                *args,
                                phase=phase,
                                failure=failure,
                                failing_rank=failing_rank,
                                original_info=original_info,
                                **kwargs,
                            ):
                                prefix = {
                                    "reporting": "Saving the checkpoint.",
                                    "completion reporting": (
                                        "Finished saving the checkpoint"
                                    ),
                                    "final reporting": "Last step checkpoint completed",
                                    "final model preparation": (
                                        "Saving a model only checkpoint"
                                    ),
                                    "final state reporting": "Saving a full checkpoint",
                                }[phase]
                                if (
                                    str(message).startswith(prefix)
                                    and rank == failing_rank
                                ):
                                    raise failure
                                return original_info(message, *args, **kwargs)

                            with patch.object(
                                distributed_checkpoint.logger, "info", save_report
                            ):
                                caught = None
                                try:
                                    manager.save(1, last_step=phase.startswith("final"))
                                except BaseException as error:
                                    caught = error
                                # Observe both outcomes before another protocol.
                                outcomes = [None, None]
                                dist.all_gather_object(outcomes, type(caught).__name__)
                                assert all(name != "NoneType" for name in outcomes), (
                                    outcomes
                                )
                                if rank == failing_rank:
                                    assert caught is failure
                                else:
                                    assert isinstance(caught, RuntimeError), caught
                                    boundary = (
                                        f"Checkpoint {phase}"
                                        if phase
                                        in {
                                            "final model preparation",
                                            "final state reporting",
                                        }
                                        else f"Checkpoint save {phase}"
                                    )
                                    assert str(caught) == (
                                        f"{boundary} failed on another rank"
                                    )
                            assert_state_equal(
                                [state.state_dict() for state in states], saved
                            )
                            assert manager.load(1)
                            dist.barrier()
                for phase in ("reporting", "cleanup", "completion reporting"):
                    for failing_rank in (0, 1):
                        for error_type in (OSError, KeyboardInterrupt, SystemExit):
                            with torch.no_grad():
                                model.weight.add_(1)
                            before = deepcopy([state.state_dict() for state in states])
                            failure = error_type("injected local load failure")
                            original_info = base.logger.info
                            original_collect = GarbageCollection.collect

                            def report(
                                message,
                                *args,
                                phase=phase,
                                failing_rank=failing_rank,
                                failure=failure,
                                original_info=original_info,
                                **kwargs,
                            ):
                                prefix = (
                                    "Loading the checkpoint"
                                    if phase == "reporting"
                                    else "Finished loading the checkpoint"
                                )
                                if (
                                    str(message).startswith(prefix)
                                    and rank == failing_rank
                                ):
                                    raise failure
                                return original_info(message, *args, **kwargs)

                            def collect(
                                reason,
                                *,
                                failing_rank=failing_rank,
                                failure=failure,
                                original_collect=original_collect,
                                **kwargs,
                            ):
                                if rank == failing_rank:
                                    raise failure
                                return original_collect(reason, **kwargs)

                            hook = (
                                patch.object(
                                    GarbageCollection, "collect", staticmethod(collect)
                                )
                                if phase == "cleanup"
                                else patch.object(base.logger, "info", report)
                            )
                            with (
                                hook,
                                patch.object(
                                    manager,
                                    "_load_checkpoint",
                                    wraps=manager._load_checkpoint,
                                ) as load,
                            ):
                                caught = None
                                try:
                                    manager.load(1)
                                except BaseException as error:
                                    caught = error
                                if rank == failing_rank:
                                    assert caught is failure
                                else:
                                    assert isinstance(caught, RuntimeError), caught
                                    assert str(caught) == (
                                        f"Checkpoint load {phase} "
                                        "failed on another rank"
                                    )
                                assert load.call_count == (phase != "reporting")
                            assert_state_equal(
                                [state.state_dict() for state in states],
                                before if phase == "reporting" else saved,
                            )
                            dist.barrier()
                            assert manager.load(1)
                            assert_state_equal(
                                [state.state_dict() for state in states], saved
                            )
                            dist.barrier()
                if rank == 0:
                    print(
                        "Load report/GC failures coordinate; states and retries pass."
                    )
            finally:
                manager.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
