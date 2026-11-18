"""A rank exit preserves full Trainer state for fresh-process recovery.

Run reference, exit, then resume with the same CHECKPOINT_TEST_ROOT, using two
CPU/Gloo ranks for each phase. The exit launcher must fail with rank code 17.
CHECKPOINT_EXIT_RANK selects rank 0 or 1. The reference uses a separate run folder.
Checks all registered state, rank-local RNG draws, losses and two further updates.
Set CHECKPOINT_TEST_ASYNC=1 in all phases to exit with real step-3 shard files
written but metadata publication blocked. Fresh peers must discover step-2.
"""

import os
import random
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemWriter

from llmtuner.accelerator.device import device_type
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.checkpoint_scenarios import build_config, train_steps
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal
from tests.integration_tests.trainer_ema_process_resume import snapshot


def advance(trainer, count):
    """Exercise all saved RNG streams alongside real data and optimizer updates."""
    source = trainer.data_iterator()
    updates = []
    for _ in range(count):
        draws = (random.random(), float(np.random.random()), torch.rand(4))
        loss = train_steps(trainer, source, 1)[0]
        updates.append({"draws": draws, "loss": loss})
    return updates


def exit_during_upload(trainer, folder, exiting_rank):
    """Stop after real payload IO, before the background writer can publish."""
    entered, release = Event(), Event()
    write_data = FileSystemWriter.write_data
    previous_metadata = (folder / "step-2/.metadata").read_bytes()

    def blocked_write(writer, plan, planner):
        result = write_data(writer, plan, planner)
        assert plan.items, "Each rank must write actual training state"
        entered.set()
        if not release.wait(60):
            raise TimeoutError("Publication was not released before process exit")
        return result

    with patch.object(FileSystemWriter, "write_data", blocked_write):
        assert trainer.checkpointer.save(3)
        assert entered.wait(30), "Async worker did not finish payload writing"
        future = trainer.checkpointer.save_future
        assert future is not None and not future.done()
        assert list((folder / "step-3").glob("*.distcp"))
        assert not (folder / "step-3/.metadata").exists()
        assert (folder / "step-2/.metadata").read_bytes() == previous_metadata
        # Both main threads agree that their own writer is still blocked.
        # The owned async IO group is separate from this default group.
        dist.barrier()
        print(
            f"rank={trainer.rank}: payload written; step-3 unpublished; step-2 intact",
            flush=True,
        )
        os._exit(17 if trainer.rank == exiting_rank else 0)


def main():
    if device_type != "cpu":
        raise RuntimeError("Hide accelerators before running this CPU/Gloo test")
    root = Path(os.environ["CHECKPOINT_TEST_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    phase = os.environ.get("CHECKPOINT_TEST_PHASE", "exit")
    exiting_rank = int(os.environ.get("CHECKPOINT_EXIT_RANK", "1"))
    if phase not in {"reference", "exit", "resume"} or exiting_rank not in (0, 1):
        raise ValueError("Expected reference|exit|resume and CHECKPOINT_EXIT_RANK=0|1")
    async_upload = os.environ.get("CHECKPOINT_TEST_ASYNC", "0") == "1"
    mode = "async" if async_upload else "sync"
    run = root / (
        f"reference-{mode}"
        if phase == "reference"
        else f"interrupted-{mode}-rank-{exiting_rank}"
    )
    config = build_config(str(run))
    config.checkpoint.interval = 1
    config.checkpoint.async_mode = "async" if async_upload else "disabled"
    trainer = Trainer(config)
    rank = trainer.rank
    assert dist.get_world_size() == 2
    random.seed(731 + rank)
    np.random.seed(811 + rank)
    torch.manual_seed(911 + rank)
    reference_path = root / f"reference-{mode}-rank-{rank}.pt"
    folder = run / "checkpoint"
    try:
        if phase == "reference":
            advance(trainer, 2)
            boundary = snapshot(trainer)
            assert boundary["ema"]
            assert any("exp_avg" in key for key in boundary["optimizer"])
            updates = advance(trainer, 2)
            torch.save(
                {"boundary": boundary, "updates": updates, "final": snapshot(trainer)},
                reference_path,
            )
            print(f"rank={rank}: full-state continuous reference saved", flush=True)
            return

        expected = torch.load(reference_path, weights_only=False)
        if phase == "exit":
            advance(trainer, 2)
            assert trainer.checkpointer.save(2)
            trainer.checkpointer.maybe_wait_for_saving()
            assert_state_equal(snapshot(trainer), expected["boundary"], "before_exit")
            if async_upload:
                assert_state_equal(
                    advance(trainer, 1), expected["updates"][:1], "before_upload"
                )
                exit_during_upload(trainer, folder, exiting_rank)
            dist.barrier()
            if rank == exiting_rank:
                os._exit(17)
            try:
                trainer.checkpointer.save(3)
            except BaseException as error:
                print(f"rank={rank}: observed peer exit: {error}", flush=True)
            else:
                raise AssertionError("Save accepted a dead peer")
            assert (folder / "step-2" / ".metadata").is_file()
            assert not (folder / "step-3" / ".metadata").exists()
            print(f"rank={rank}: full step-2 survived; step-3 unpublished", flush=True)
            os._exit(0)

        assert trainer.step == 0
        assert trainer.checkpointer._find_load_step() == 2
        assert trainer.checkpointer.load(-1)
        assert_state_equal(snapshot(trainer), expected["boundary"], "loaded")
        assert trainer.checkpointer.load(2)
        assert_state_equal(snapshot(trainer), expected["boundary"], "reloaded")
        assert_state_equal(advance(trainer, 1), expected["updates"][:1], "first_update")
        # The live synthetic loader forbids rewinding consumed input. Reject an
        # older checkpoint before changing any training state, then continue.
        before_rejected_load = snapshot(trainer)
        try:
            trainer.checkpointer.load(2)
        except ValueError as error:
            assert "already yielded 3" in str(error)
        else:
            raise AssertionError("Live dataloader accepted a backward cursor")
        assert_state_equal(snapshot(trainer), before_rejected_load, "rejected_rewind")
        assert_state_equal(
            advance(trainer, 1), expected["updates"][1:], "second_update"
        )
        assert_state_equal(snapshot(trainer), expected["final"], "continued_state")
        assert trainer.checkpointer.save(4)
        assert trainer.checkpointer.load(4)
        assert_state_equal(snapshot(trainer), expected["final"], "republished_state")
        print(
            f"rank={rank}: full-state fresh-process resume and replay passed",
            flush=True,
        )
    finally:
        try:
            trainer.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
