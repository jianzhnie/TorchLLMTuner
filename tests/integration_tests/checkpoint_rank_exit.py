"""A rank exit preserves the last checkpoint; fresh peers can resume it.

Run the failure phase with CHECKPOINT_TEST_PHASE=exit and expect torchrun to
fail. Then run with CHECKPOINT_TEST_PHASE=resume and the same shared
CHECKPOINT_TEST_ROOT. Both phases require two CPU/Gloo ranks.
"""

import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.config.checkpoint_config import CheckpointConfig


def build_manager(model, folder):
    optimizer = torch.optim.AdamW(model.parameters())
    return DistributedCheckpointManager(
        CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        states={},
        folder=str(folder),
    )


def main():
    root = Path(os.environ["CHECKPOINT_TEST_ROOT"])
    phase = os.environ.get("CHECKPOINT_TEST_PHASE", "exit")
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    torch.manual_seed(117)
    model = torch.nn.Linear(3, 2)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    manager = build_manager(model, root / "checkpoint")
    folder = root / "checkpoint" / "checkpoint"
    try:
        if phase == "exit":
            assert manager.save(1)
            dist.barrier()
            if rank == int(os.environ.get("CHECKPOINT_EXIT_RANK", "1")):
                os._exit(17)
            try:
                manager.save(2)
                manager.maybe_wait_for_saving()
            except BaseException as error:
                print(f"rank={rank}: observed peer exit: {error}", flush=True)
            assert (folder / "step-1" / ".metadata").is_file()
            assert not (folder / "step-2" / ".metadata").exists()
            print(f"rank={rank}: step-1 survived; step-2 was not published", flush=True)
            os._exit(0)
        if phase != "resume":
            raise ValueError(f"unknown CHECKPOINT_TEST_PHASE: {phase}")
        assert manager.load(1)
        torch.testing.assert_close(model.state_dict(), initial, atol=0, rtol=0)
        assert manager.load(1)
        with torch.no_grad():
            model.weight.add_(2)
        assert manager.save(2)
        with torch.no_grad():
            model.weight.zero_()
        assert manager.load(2)
        expected = {key: value.clone() for key, value in initial.items()}
        expected["weight"].add_(2)
        torch.testing.assert_close(model.state_dict(), expected, atol=0, rtol=0)
        print(f"rank={rank}: repeated load and fresh-process resume passed", flush=True)
    finally:
        manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
