"""NPU staging event ordering with actual async copies and changing payloads.

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    tests/integration_tests/npu_checkpoint_staging_events.py

Requires torch_checkpointing 0.1.0. Each rank stages 128 changing tensors through
the real DefaultStager. This verifies completion and copy correctness, not pin
memory effectiveness or end-to-end checkpoint durability.
"""

from concurrent.futures import Future
from types import SimpleNamespace

import torch
import torch.distributed as dist

from tests.caps import require_env

require_env("torch_npu")

from torch_checkpointing.staging import CheckpointStagerConfig, DefaultStager

from llmtuner.accelerator.device import get_current_device, set_device
from llmtuner.components.checkpointer.storage.checkpointing_staging import (
    adapt_npu_staging,
)


def main() -> None:
    device = get_current_device()
    if device.type != "npu":
        raise RuntimeError("This acceptance requires a real NPU")
    set_device(device)
    dist.init_process_group("hccl")
    try:
        rank = dist.get_rank()
        stager = DefaultStager(
            CheckpointStagerConfig(
                use_async_staging=True,
                use_pinned_memory=False,
                use_shared_memory=False,
                use_non_blocking_copy=True,
            )
        )
        try:
            manager = SimpleNamespace(_saver=SimpleNamespace(_checkpoint_stager=stager))
            assert adapt_npu_staging(manager)
            assert not adapt_npu_staging(manager)
            original = torch.arange(257, dtype=torch.float32) + rank * 1000
            payload = original.to(device)
            for step in range(128):
                payload.add_(1)
                metadata = {"rank": rank, "step": step}
                future = stager.stage(
                    {"model": payload, "metadata": metadata},
                    keys_not_requiring_copy=("metadata",),
                )
                assert isinstance(future, Future)
                staged = future.result(timeout=60)
                assert staged["metadata"] is metadata
                assert staged["model"].device.type == "cpu"
                torch.testing.assert_close(
                    staged["model"], original + step + 1, rtol=0, atol=0
                )
            print(f"PASS rank={rank}: 128 real async staging copies", flush=True)
        finally:
            stager.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
