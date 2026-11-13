"""Data services use explicit configuration, topology and observation inputs."""

from types import SimpleNamespace

import pytest
import torch

import llmtuner.trainer.runtime.validation as validation_runtime
from llmtuner.config.validation_config import ValidationConfig
from llmtuner.trainer.runtime.batch import batch_generator, prepare_microbatch


def test_batch_generator_accounts_without_trainer():
    tokens = []
    durations = []
    metrics = SimpleNamespace(
        add_tokens=tokens.append, add_data_loading_time=durations.append
    )
    batch = {"labels": torch.ones(2, 3)}
    iterator = batch_generator([batch], metrics=metrics)
    try:
        assert next(iterator) is batch
    finally:
        iterator.close()
    assert tokens == [6]
    assert len(durations) == 1
    assert durations[0] >= 0


@pytest.mark.parametrize("steps", [-1, 3])
def test_validation_loader_uses_explicit_batch_ownership(monkeypatch, steps):
    cfg = SimpleNamespace(global_batch_size=8, max_seq_len=16)
    requested_axes = []

    def mesh(axis, *, include_singleton_axes):
        requested_axes.append((axis, include_singleton_axes))
        return SimpleNamespace(get_local_rank=lambda: 1, size=lambda: 2)

    topology = SimpleNamespace(pp_enabled=True, get_optional_mesh=mesh)
    calls = []
    loader = object()

    def build(config, **kwargs):
        calls.append((config, kwargs))
        return loader

    monkeypatch.setattr(validation_runtime, "build_dataloader", build)
    validation = ValidationConfig(steps=steps)
    assert (
        validation_runtime.create_validation_dataloader(cfg, topology, validation)
        is loader
    )
    assert requested_axes == [("batch", True)]
    assert calls == [
        (
            cfg,
            {
                "dp_rank": 1,
                "dp_world_size": 2,
                "num_tokens_per_batch": 64,
                "repeat": steps != -1,
                "dataset": validation.dataset,
            },
        )
    ]


def test_microbatch_preparation_preserves_callback_and_host_ownership():
    labels = torch.tensor([[1, -100, 3], [4, 5, -100]])
    batch = {
        "input": torch.ones_like(labels),
        "labels": labels,
        "padding_mask": torch.tensor([[False, False, False], [False, True, True]]),
    }
    calls = []

    def valid_tokens(value):
        calls.append(value)
        return 2

    prepared, tokens = prepare_microbatch(batch, valid_token_counter=valid_tokens)

    assert calls == [batch]
    assert prepared["batch"] is batch
    assert prepared["num_valid_tokens"] == 2
    assert prepared["num_routing_tokens"] == 4
    assert tokens == 6
    assert set(batch) == {"input", "labels", "padding_mask"}
    assert batch["labels"] is labels
