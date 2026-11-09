"""Two-rank CPU/Gloo payload failures must stop peers before backend save.

Run: torchrun --standalone --nproc_per_node=2 \
    tests/integration_tests/checkpoint_payload_coordination.py
"""

from collections import UserDict
from datetime import timedelta

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer.backends.torch_checkpointing_backend import (
    TorchCheckpointingManager,
)


class CollectiveState:
    def __init__(self, mode):
        self.mode = mode

    def state_dict(self):
        value = torch.tensor(1)
        dist.all_reduce(value)
        assert value.item() == dist.get_world_size()
        if self.mode == "callback" and dist.get_rank() == 0:
            raise ValueError("rank-local callback failure")
        device = "meta" if self.mode == "payload" and dist.get_rank() == 0 else "cpu"
        return UserDict({"nested": [torch.ones(2, device=device)]})

    def load_state_dict(self, state):
        raise NotImplementedError


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        manager = TorchCheckpointingManager.__new__(TorchCheckpointingManager)
        manager._cpu_synchronous_save = True
        for mode in ("callback", "payload"):
            try:
                manager._save_payload({"custom": CollectiveState(mode)})
            except (ValueError, RuntimeError) as error:
                if dist.get_rank() == 0:
                    assert isinstance(error, ValueError)
                else:
                    assert "Checkpoint payload construction" in str(error)
            else:
                raise AssertionError(f"rank accepted invalid {mode}")
            # Both ranks must reach this collective after the rejected save.
            dist.barrier()
        payload = manager._save_payload({"custom": CollectiveState("valid")})
        assert payload["custom"]["nested"][0].device.type == "cpu"
        if dist.get_rank() == 0:
            print("PASS: callback/payload rank-local failures coordinated; retry valid")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
