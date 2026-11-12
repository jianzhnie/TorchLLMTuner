"""Choose copy capability independently of background checkpoint upload."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmtuner.components.checkpointer.backends import distributed_checkpoint
from llmtuner.config.checkpoint_config import CheckpointConfig


@pytest.mark.parametrize("accelerator_available", [False, True])
def test_nonblocking_staging_requires_available_accelerator(
    tmp_path, monkeypatch, accelerator_available
):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    options = []

    def make_stager(config):
        options.append(config)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        torch.accelerator, "is_available", lambda: accelerator_available
    )
    monkeypatch.setattr(distributed_checkpoint, "DefaultStager", make_stager)
    manager = distributed_checkpoint.DistributedCheckpointManager(
        CheckpointConfig(
            enable=True, interval=1, keep_latest_k=0, async_mode="async_with_pinned_mem"
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
        states={},
        folder=str(tmp_path),
    )
    upload = Future()
    upload.set_result(None)
    monkeypatch.setattr(manager, "_save_checkpoint", lambda *args, **kwargs: upload)
    try:
        assert manager.save(1)
        assert len(options) == 1
        assert options[0].use_non_blocking_copy is accelerator_available
        assert options[0].use_async_staging
        assert options[0].use_shared_memory
        assert options[0].use_pinned_memory
    finally:
        manager.close()
