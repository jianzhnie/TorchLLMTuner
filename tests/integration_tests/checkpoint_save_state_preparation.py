"""Real DCP rejects rank-local state preparation failures before starting I/O.

Run with torchrun --nproc_per_node=2 on CPU/Gloo. Covers real scheduler/RNG
callbacks, synchronous/threaded saves, full final saves, and subsequent retries.
"""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer import distributed_checkpoint as checkpoint
from llmtuner.components.checkpointer.random_state import RNGCheckpointState
from llmtuner.config.checkpoint_config import CheckpointConfig


def check_failure(root, category, failing_rank, mode):
    torch.manual_seed(326)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    rng = RNGCheckpointState(torch.device("cpu"))
    folder = root / f"{category}-{failing_rank}-{mode}"
    manager = checkpoint.DistributedCheckpointManager(
        CheckpointConfig(
            enable=True,
            interval=1,
            keep_latest_k=0,
            async_mode="async" if mode == "async" else "disabled",
            last_save_model_only=False,
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={"rng_state": rng},
        folder=str(folder),
    )
    folder = Path(manager.folder)
    target = scheduler if category == "lr_scheduler" else rng
    original = target.state_dict
    failure = OSError("local save state preparation failed")

    def prepare():
        if dist.get_rank() == failing_rank:
            raise failure
        return original()

    try:
        with (
            patch.object(target, "state_dict", prepare),
            patch.object(checkpoint.dcp, "save", wraps=checkpoint.dcp.save) as save,
            patch.object(
                checkpoint.dcp, "async_save", wraps=checkpoint.dcp.async_save
            ) as async_save,
        ):
            try:
                manager.save(2, last_step=mode == "final")
            except BaseException as error:
                if dist.get_rank() == failing_rank:
                    assert error is failure
                else:
                    assert isinstance(error, RuntimeError), error
                    assert (
                        "Checkpoint local save preparation failed on another rank"
                        in str(error)
                    ), error
            else:
                raise AssertionError("Peer accepted failed state preparation")
            save.assert_not_called()
            async_save.assert_not_called()
        assert manager.save_future is None
        assert not (folder / "step-2" / ".metadata").exists()
        dist.barrier()
        assert manager.save(2, last_step=mode == "final")
        manager.maybe_wait_for_saving()
        assert (folder / "step-2" / ".metadata").exists()
        assert manager.load(2)
        dist.barrier()
    finally:
        manager.close()


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        with TemporaryDirectory(prefix="llmtuner-state-preparation-326-") as local:
            paths = [local if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(paths)
            for category in ("lr_scheduler", "rng_state"):
                for failing_rank in (0, 1):
                    for mode in ("sync", "async", "final"):
                        check_failure(Path(paths[0]), category, failing_rank, mode)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
