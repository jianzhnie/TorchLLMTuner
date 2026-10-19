"""Two-rank optional-backend prewarm failures and retry agreement.

Run with torchrun --nproc_per_node=2 on CPU/Gloo. The manager and state callback
are explicit substitutes; this verifies orchestration, not optional-package I/O.
"""

from datetime import timedelta
from types import SimpleNamespace

import torch.distributed as dist

from llmtuner.components.checkpointer.torch_checkpointing_backend import (
    TorchCheckpointingManager,
)


def check_failure(failing_rank, phase):
    manager = TorchCheckpointingManager.__new__(TorchCheckpointingManager)
    manager.enable = True
    manager.load_only = False
    manager.interval = 10
    manager.enable_first_step_checkpoint = False
    manager._prewarmed = False
    manager._initialized = True
    failure = OSError("local prewarm failed")
    fail = True
    calls = []

    class State:
        def state_dict(self):
            if fail and phase == "state" and dist.get_rank() == failing_rank:
                raise failure
            return {}

        def load_state_dict(self, state):
            pass

    def prewarm(state):
        calls.append("prewarm")
        if fail and phase == "backend" and dist.get_rank() == failing_rank:
            raise failure

    manager.states = {"custom": State()}
    manager._manager = SimpleNamespace(prewarm_staging=prewarm)
    try:
        try:
            manager.save(2)
        except BaseException as error:
            if dist.get_rank() == failing_rank:
                assert error is failure
            else:
                assert isinstance(error, RuntimeError), error
                assert "failed on another rank" in str(error)
        else:
            raise AssertionError("Peer accepted failed prewarm")
        assert not manager._prewarmed
        dist.barrier()
        fail = False
        assert manager.save(2) is False
        assert manager._prewarmed
        count = len(calls)
        assert manager.save(3) is False
        assert len(calls) == count
        dist.barrier()
    finally:
        # This test owns no backend resources or retention thread.
        manager._initialized = False


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        for phase in ("state", "backend"):
            for rank in (0, 1):
                check_failure(rank, phase)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
