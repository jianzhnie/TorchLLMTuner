"""Trainer snapshot safety with delayed CPU staging and a mutable forward buffer.

Run full/resume in separate two-rank torchrun invocations with a shared WORKDIR:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/async_snapshot_equivalence.py full WORKDIR
    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/async_snapshot_equivalence.py resume WORKDIR

Uses the existing HF Trainer/AdamW/EMA recovery oracle with PP1/FSDP2. Staging
is held until the manager's staging wait, exposing forward-buffer races without
timing assumptions. Real DefaultStager shared CPU copies and PROCESS uploads
are used. Pinning/nonblocking device copies are disabled; no device guarantee.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import torch
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.tensor import DTensor

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.pp_checkpoint_equivalence import _run


class DelayedCPUStager:
    """Delay actual staging until its synchronization boundary is reached."""

    should_synchronize_after_execute = False

    def __init__(self):
        self.ready = Event()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.stager = DefaultStager(
            StagingOptions(
                use_pinned_memory=False,
                use_shared_memory=True,
                use_async_staging=False,
                use_non_blocking_copy=False,
            )
        )
        self.buffer_storage = None

    def stage(self, state):
        self.ready.clear()
        expected = state["snapshot_forward_count"].clone()

        def copy_snapshot():
            if not self.ready.wait(timeout=60):
                raise TimeoutError("Trainer did not reach the staging wait")
            staged = self.stager.stage(state)
            counter = staged["snapshot_forward_count"]
            local = counter._local_tensor if isinstance(counter, DTensor) else counter
            torch.testing.assert_close(counter, expected, rtol=0, atol=0)
            assert local.is_shared(), "Snapshot must use actual shared CPU storage"
            storage = local.untyped_storage().data_ptr()
            reused = self.buffer_storage is not None
            if reused:
                assert storage == self.buffer_storage, "Staging cache was not reused"
            self.buffer_storage = storage
            print(
                f"[snapshot] count={counter.item()} shared=True cache_reused={reused}"
            )
            return staged

        return self.executor.submit(copy_snapshot)

    def synchronize_staging(self):
        self.ready.set()

    def close(self):
        self.ready.set()
        self.executor.shutdown(wait=True)
        self.stager.close()


def main():
    if os.environ.get("LLMTUNER_TEST_PP_SIZE", "1") != "1":
        raise ValueError("The mutable root-buffer scenario requires PP1")
    os.environ["LLMTUNER_TEST_PP_SIZE"] = "1"
    os.environ["LLMTUNER_TEST_ASYNC_MODE"] = "async_with_pinned_mem"
    model_init = HFTransformerModel.__init__
    model_forward = HFTransformerModel.forward
    trainer_init = Trainer.__init__
    wait_staging = DistributedCheckpointManager._maybe_wait_for_staging

    def initialize_model(self, *args, **kwargs):
        model_init(self, *args, **kwargs)
        self.register_buffer(
            "snapshot_forward_count", torch.zeros((), dtype=torch.int64, device="cpu")
        )

    def forward_with_buffer(self, *args, **kwargs):
        self.snapshot_forward_count.add_(1)
        return model_forward(self, *args, **kwargs)

    def initialize_trainer(self, *args, **kwargs):
        trainer_init(self, *args, **kwargs)
        self.checkpointer.stager = DelayedCPUStager()
        if sys.argv[1] == "full":
            # A first snapshot populates the cache; step-2 reuses its storage.
            assert self.checkpointer.save(0)

    def release_staging(self):
        if isinstance(self.stager, DelayedCPUStager):
            self.stager.ready.set()
        return wait_staging(self)

    with (
        patch.object(HFTransformerModel, "__init__", initialize_model),
        patch.object(HFTransformerModel, "forward", forward_with_buffer),
        patch.object(Trainer, "__init__", initialize_trainer),
        patch.object(
            DistributedCheckpointManager, "_maybe_wait_for_staging", release_staging
        ),
    ):
        _run()


if __name__ == "__main__":
    main()
