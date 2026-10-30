"""The observer preserves batches, metrics and the training call count."""

import json
from types import SimpleNamespace

import torch

from examples.npu_training_audit import AuditedNPUTrainer
from llmtuner import Trainer


def test_observer_records_the_exact_consumed_batch(tmp_path, monkeypatch):
    observer = object.__new__(AuditedNPUTrainer)
    observer.step = 7
    observer._audit_steps = tmp_path / "steps.jsonl"
    batch = {
        "input": torch.tensor([1, 2]),
        "labels": torch.tensor([2, -100]),
        "positions": torch.tensor([0, 1]),
        "padding_mask": torch.tensor([False, True]),
        "num_valid_tokens": 1,
    }
    calls = []

    def step(self, iterator):
        assert next(iterator) is batch
        calls.append(self.step)
        return {"loss": 0.5}

    syncs = []
    monkeypatch.setattr(Trainer, "train_step", step)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: syncs.append(1)),
        raising=False,
    )
    assert observer.train_step(iter([batch])) == {"loss": 0.5}
    assert calls == [7]
    assert len(syncs) == 2
    record = json.loads(observer._audit_steps.read_text())
    assert record["step"] == 7
    assert record["batches"][0]["valid_tokens"] == 1
    assert record["batches"][0]["fields"]["input"]["shape"] == [2]
    assert record["seconds"] >= 0
