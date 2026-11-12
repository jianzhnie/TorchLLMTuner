"""Trainer snapshot safety with delayed CPU staging and a mutable forward buffer.

Run full/resume in separate two-rank torchrun invocations with a shared WORKDIR:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/async_snapshot_equivalence.py full WORKDIR
    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/async_snapshot_equivalence.py resume WORKDIR

Uses the explicit Qwen3 Trainer/AdamW/EMA recovery oracle with PP1/FSDP2. Staging
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

from llmtuner.components.checkpointer.backends import distributed_checkpoint
from llmtuner.components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.pp_checkpoint_equivalence import _run


class DelayedCPUStager:
    """Delay actual staging until its synchronization boundary is reached."""

    should_synchronize_after_execute = False

    def __init__(self, options: StagingOptions | None = None) -> None:
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
        self.counts: list[int] = []
        self.closed = False

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
            self.counts.append(int(counter.item()))
            print(f"[snapshot] count={counter.item()} shared=True delayed=True")
            return staged

        return self.executor.submit(copy_snapshot)

    def synchronize_staging(self):
        self.ready.set()

    def close(self):
        self.ready.set()
        self.executor.shutdown(wait=True)
        self.stager.close()
        self.closed = True


def main():
    if os.environ.get("LLMTUNER_TEST_PP_SIZE", "1") != "1":
        raise ValueError("The mutable root-buffer scenario requires PP1")
    os.environ["LLMTUNER_TEST_PP_SIZE"] = "1"
    os.environ["LLMTUNER_TEST_ASYNC_MODE"] = "async_with_pinned_mem"
    stagers: list[DelayedCPUStager] = []

    def make_stager(options):
        stager = DelayedCPUStager(options)
        stagers.append(stager)
        return stager

    model_init = Qwen3MoeForCausalLM.__init__
    model_forward = Qwen3MoeForCausalLM.forward
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
        if sys.argv[1] == "full":
            # Exercise replacement of a used stager before the step-2 save.
            assert self.checkpointer.save(0)

    def release_staging(self):
        if isinstance(self.stager, DelayedCPUStager):
            self.stager.ready.set()
        return wait_staging(self)

    with (
        patch.object(distributed_checkpoint, "DefaultStager", make_stager),
        patch.object(Qwen3MoeForCausalLM, "__init__", initialize_model),
        patch.object(Qwen3MoeForCausalLM, "forward", forward_with_buffer),
        patch.object(Trainer, "__init__", initialize_trainer),
        patch.object(
            DistributedCheckpointManager, "_maybe_wait_for_staging", release_staging
        ),
    ):
        _run()
    if sys.argv[1] == "full":
        assert len(stagers) >= 2, "Expected manager-owned stager replacement"
        assert any(count > 0 for stager in stagers for count in stager.counts), (
            "The training checkpoint must execute delayed staging after forward"
        )
    assert all(stager.closed for stager in stagers), "Stager resources leaked"


if __name__ == "__main__":
    main()
