"""The NPU staging fix changes only the supported manager instance."""

import sys
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from llmtuner.components.checkpointer.storage import checkpointing_staging as staging


@pytest.mark.parametrize("device,release", [("cpu", "0.1.0"), ("npu", "0.2.0")])
def test_staging_adapter_preserves_other_environments(monkeypatch, device, release):
    monkeypatch.setattr(staging, "device_type", device)
    monkeypatch.setattr(staging, "version", lambda name: release)
    target = SimpleNamespace(_stage=object())
    original = target._stage
    manager = SimpleNamespace(_saver=SimpleNamespace(_checkpoint_stager=target))
    assert not staging.adapt_npu_staging(manager)
    assert target._stage is original


def test_staging_adapter_skips_when_distribution_metadata_is_missing(monkeypatch):
    monkeypatch.setattr(staging, "device_type", "npu")

    def missing_distribution(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(staging, "version", missing_distribution)
    target = SimpleNamespace(_stage=object())
    original = target._stage
    manager = SimpleNamespace(_saver=SimpleNamespace(_checkpoint_stager=target))

    assert not staging.adapt_npu_staging(manager)
    assert target._stage is original


def test_staging_adapter_waits_once_and_preserves_payload_and_instance(monkeypatch):
    calls = []

    class DefaultStager:
        _accelerator_available = True

        def _stage(self, state, event, excluded):
            calls.append(("stage", state, event, excluded))
            return state

    target = DefaultStager()
    other = DefaultStager()
    monkeypatch.setattr(staging, "device_type", "npu")
    monkeypatch.setattr(staging, "version", lambda name: "0.1.0")
    monkeypatch.setitem(
        sys.modules,
        "torch_checkpointing.staging",
        SimpleNamespace(DefaultStager=DefaultStager),
    )
    manager = SimpleNamespace(_saver=SimpleNamespace(_checkpoint_stager=target))
    assert staging.adapt_npu_staging(manager)
    assert not staging.adapt_npu_staging(manager)
    assert other._stage.__func__ is DefaultStager._stage
    event = SimpleNamespace(synchronize=lambda: calls.append("wait"))
    payload = {"model": object()}
    excluded = ["metadata"]
    assert target._stage(payload, event, excluded) is payload
    assert calls == ["wait", ("stage", payload, event, excluded)]

    calls.clear()
    assert target._stage(payload, None) is payload
    assert calls == [("stage", payload, None, ())]


def test_staging_wait_failure_does_not_start_copy(monkeypatch):
    class DefaultStager:
        _accelerator_available = True

        def _stage(self, *args):
            pytest.fail("A failed producer wait must not start staging")

    target = DefaultStager()
    target._staged_state_dict = {"previous": object()}
    monkeypatch.setattr(staging, "device_type", "npu")
    monkeypatch.setattr(staging, "version", lambda name: "0.1.0")
    monkeypatch.setitem(
        sys.modules,
        "torch_checkpointing.staging",
        SimpleNamespace(DefaultStager=DefaultStager),
    )
    manager = SimpleNamespace(_saver=SimpleNamespace(_checkpoint_stager=target))
    assert staging.adapt_npu_staging(manager)

    def fail_wait():
        raise RuntimeError("producer failure")

    with pytest.raises(RuntimeError, match="producer failure"):
        target._stage({}, SimpleNamespace(synchronize=fail_wait))
    assert target._staged_state_dict is None
