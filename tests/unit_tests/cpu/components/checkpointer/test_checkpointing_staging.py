"""The NPU staging fix changes only the supported manager instance."""

from types import SimpleNamespace

import pytest

from llmtuner.components.checkpointer.storage import checkpointing_staging as staging


@pytest.mark.parametrize("device,release", [("cpu", "0.1.0"), ("npu", "0.2.0")])
def test_staging_adapter_preserves_other_environments(monkeypatch, device, release):
    monkeypatch.setattr(staging, "device_type", device)
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
    monkeypatch.setattr(staging, "DefaultStager", DefaultStager)
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
    monkeypatch.setattr(staging, "DefaultStager", DefaultStager)
    manager = SimpleNamespace(_saver=SimpleNamespace(_checkpoint_stager=target))
    assert staging.adapt_npu_staging(manager)

    def fail_wait():
        raise RuntimeError("producer failure")

    with pytest.raises(RuntimeError, match="producer failure"):
        target._stage({}, SimpleNamespace(synchronize=fail_wait))
    assert target._staged_state_dict is None


def test_worker_adapter_binds_only_target_and_captures_rank_device(monkeypatch):
    class DefaultStager:
        _accelerator_available = True

        def _stage_on_worker(self, *args):
            raise AssertionError("Upstream worker must not execute")

        def stage(self, *args):
            raise AssertionError("CUDA staging must not execute")

        def _stage(self, *args):
            raise AssertionError("CUDA copy must not execute")

        def _init_worker_thread(self):
            raise AssertionError("CUDA worker init must not execute")

    monkeypatch.setattr(staging, "device_type", "npu")
    monkeypatch.setattr(staging, "DefaultStager", DefaultStager)
    calls = []
    monkeypatch.setattr(
        staging.torch,
        "npu",
        SimpleNamespace(current_device=lambda: 3, set_device=calls.append),
        raising=False,
    )
    target, other = DefaultStager(), DefaultStager()
    manager = SimpleNamespace(_saver=SimpleNamespace(_checkpoint_stager=target))
    assert staging.adapt_npu_staging(manager)
    assert not staging.adapt_npu_staging(manager)
    assert target._llmtuner_npu_device == 3
    assert target.stage.__func__ is staging._npu_stage
    assert target._stage.__func__ is staging._npu_copy_stage
    target._init_worker_thread()
    assert calls == [3]
    assert other.stage.__func__ is DefaultStager.stage
