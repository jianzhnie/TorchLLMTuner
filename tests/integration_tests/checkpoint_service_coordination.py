"""Real two-rank coordination around optional-backend local service failure.

Run with torchrun --nproc_per_node=2 on CPU/Gloo. Backend construction/storage
are substitutes: this tests manager-owned retention threads and cleanup, not
optional-package I/O or subprocesses.
"""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist

from llmtuner.components.checkpointer import torch_checkpointing_backend as tc
from llmtuner.config.checkpoint_config import CheckpointConfig


def check_failure(failing_rank, phase):
    calls = []
    backend_manager = SimpleNamespace(close=lambda: calls.append("close"))
    failure = OSError("local checkpoint construction failed")

    def build():
        if phase == "build" and dist.get_rank() == failing_rank:
            raise failure
        return backend_manager

    def create_storage():
        if phase == "storage" and dist.get_rank() == failing_rank:
            raise failure
        return object()

    config = SimpleNamespace(
        storage_config=SimpleNamespace(create_storage=create_storage),
        build=build,
    )

    def require_backend():
        if phase == "dependency" and dist.get_rank() == failing_rank:
            raise failure
        return object()

    original_info = tc.logger.info

    def report(*args, **kwargs):
        if phase == "logging" and dist.get_rank() == failing_rank:
            raise failure
        original_info(*args, **kwargs)

    original_start = tc.threading.Thread.start

    def start(thread):
        if phase == "thread" and dist.get_rank() == failing_rank:
            raise failure
        original_start(thread)

    manager = tc.TorchCheckpointingManager.__new__(tc.TorchCheckpointingManager)
    try:
        with (
            patch.object(tc, "require_torch_checkpointing", require_backend),
            patch.object(tc, "async_save_config", lambda backend: object()),
            patch.object(tc, "default_backend_config", lambda *a, **kw: config),
            patch.object(
                tc,
                "BackendCheckpointStorage",
                lambda storage: SimpleNamespace(remove=lambda path: None),
            ),
            patch.object(tc.logger, "info", report),
            patch.object(tc.threading.Thread, "start", start),
        ):
            try:
                manager.__init__(
                    CheckpointConfig(enable=True, keep_latest_k=2),
                    model_parts=[],
                    optimizer=None,
                    lr_scheduler=None,
                    states={},
                    folder="/tmp/llmtuner-services-321",
                )
            except BaseException as error:
                if dist.get_rank() == failing_rank:
                    assert error is failure
                else:
                    assert isinstance(error, RuntimeError), error
                    assert "failed on another rank" in str(error)
            else:
                raise AssertionError("Peer accepted failed construction")
        expected_calls = (
            []
            if phase == "dependency"
            or (phase in {"build", "storage"} and dist.get_rank() == failing_rank)
            else ["close"]
        )
        assert calls == expected_calls
        assert not manager._initialized
        if phase in {"logging", "thread"}:
            assert not manager.purge_thread.is_alive()
        manager.close()
        assert calls == expected_calls
        dist.barrier()
    finally:
        manager.close()


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        for phase in ("build", "storage", "dependency", "logging", "thread"):
            for failing_rank in (0, 1):
                check_failure(failing_rank, phase)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
