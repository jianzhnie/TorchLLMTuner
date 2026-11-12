"""Pipeline RNG compatibility must not modify process-global torch functions."""

import contextlib

import pytest
import torch

from llmtuner.errors import EnvironmentUnsupportedError
from llmtuner.parallel.pipeline_parallel.microbatch_drivers import bind_stage_rng


class Schedule:
    def _initialize_pp_stages(self):
        with torch.random.fork_rng(devices=[]):
            return torch.rand(3)


def test_private_metadata_namespace_preserves_global_rng_and_function():
    schedule = Schedule()
    original = torch.random.fork_rng
    before = torch.get_rng_state().clone()
    bind_stage_rng(schedule)
    bound = schedule._initialize_pp_stages
    first = bound()
    second = bound()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.random.fork_rng is original
    bind_stage_rng(schedule)
    assert schedule._initialize_pp_stages is bound


class FailingSchedule:
    def _initialize_pp_stages(self):
        with torch.random.fork_rng(devices=[]):
            torch.rand(5)
            raise RuntimeError("metadata failure")


def test_metadata_exception_restores_rng_without_global_patch():
    schedule = FailingSchedule()
    original = torch.random.fork_rng
    before = torch.get_rng_state().clone()
    bind_stage_rng(schedule)
    with pytest.raises(RuntimeError, match="metadata failure"):
        schedule._initialize_pp_stages()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.random.fork_rng is original
    assert FailingSchedule._initialize_pp_stages.__globals__["torch"] is torch


class DeviceSchedule:
    def _initialize_pp_stages(self, devices, **kwargs):
        with torch.random.fork_rng(devices=devices, **kwargs):
            pass


def test_private_namespace_selects_stage_device_and_preserves_explicit_type(
    monkeypatch,
):
    calls = []

    @contextlib.contextmanager
    def record(**kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr(torch.random, "fork_rng", record)
    schedule = DeviceSchedule()
    untouched = DeviceSchedule()
    bind_stage_rng(schedule)
    schedule._initialize_pp_stages([torch.device("npu:1")])
    schedule._initialize_pp_stages([], device_type="cpu", enabled=False)
    assert calls == [
        {"devices": [torch.device("npu:1")], "device_type": "npu"},
        {"devices": [], "device_type": "cpu", "enabled": False},
    ]
    assert torch.random.fork_rng is record
    assert (
        untouched._initialize_pp_stages.__func__ is DeviceSchedule._initialize_pp_stages
    )


def test_mixed_accelerator_metadata_is_rejected_before_rng_access():
    schedule = DeviceSchedule()
    original = torch.random.fork_rng
    bind_stage_rng(schedule)
    with pytest.raises(ValueError, match="one accelerator type"):
        schedule._initialize_pp_stages([torch.device("cuda:0"), torch.device("npu:0")])
    assert torch.random.fork_rng is original


def test_unknown_metadata_callable_has_explicit_compatibility_error():
    schedule = DeviceSchedule()
    schedule._initialize_pp_stages = lambda: None
    with pytest.raises(EnvironmentUnsupportedError, match="Python metadata"):
        bind_stage_rng(schedule)
