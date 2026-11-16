"""Data services use explicit configuration, topology and observation inputs."""

from types import SimpleNamespace

import pytest
import torch

import llmtuner.trainer.runtime.validation as validation_runtime
from llmtuner.config.validation_config import ValidationConfig
from llmtuner.trainer.runtime import batch as batch_runtime
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

    topology = SimpleNamespace(get_optional_mesh=mesh)
    calls = []
    loader = object()

    def build(config, **kwargs):
        calls.append((config, kwargs))
        return loader

    monkeypatch.setattr(batch_runtime, "build_dataset_dataloader", build)
    validation = ValidationConfig(steps=steps)
    assert (
        validation_runtime.create_validation_dataloader(cfg, topology, validation)
        is loader
    )
    assert requested_axes == [("dp", True)]
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


@pytest.mark.parametrize("invalid_count", [True, -1, 5, 1.5])
@pytest.mark.parametrize("phase", ["training", "validation"])
def test_token_accounting_rejects_invalid_callback_before_metrics(invalid_count, phase):
    batch = {
        "input": torch.ones(4, dtype=torch.long),
        "labels": torch.ones(4, dtype=torch.long),
    }
    observed = []

    def counter(batch):
        return invalid_count

    with pytest.raises(ValueError, match="num_valid_tokens"):
        if phase == "training":
            prepare_microbatch(batch, valid_token_counter=counter)
        else:
            validation_runtime.count_validation_batch_tokens(
                batch,
                metrics=SimpleNamespace(add_tokens=observed.append),
                token_counter=counter,
                device=torch.device("cpu"),
                dp_mesh=None,
            )
    assert observed == []


@pytest.mark.parametrize("phase", ["training", "validation"])
@pytest.mark.parametrize("malformation", ["dtype", "shape", "missing_input"])
def test_token_accounting_validates_fields_before_callback(phase, malformation):
    batch = {
        "input": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.ones(2, 3, dtype=torch.long),
    }
    if malformation == "dtype":
        batch["labels"] = batch["labels"].float()
        expected_error = TypeError
    elif malformation == "shape":
        batch["labels"] = batch["labels"].reshape(3, 2)
        expected_error = ValueError
    else:
        del batch["input"]
        expected_error = KeyError
    calls = []

    def counter(batch):
        calls.append("counter")
        return 6

    with pytest.raises(expected_error):
        if phase == "training":
            prepare_microbatch(batch, valid_token_counter=counter)
        else:
            validation_runtime.count_validation_batch_tokens(
                batch,
                metrics=SimpleNamespace(add_tokens=lambda _: calls.append("metrics")),
                token_counter=counter,
                device=torch.device("cpu"),
                dp_mesh=None,
            )
    assert calls == []
