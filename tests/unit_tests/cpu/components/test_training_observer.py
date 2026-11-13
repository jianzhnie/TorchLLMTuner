"""Training-only scopes and bounded steady statistics."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import llmtuner.trainer.runtime.training_observer as observer_module
from llmtuner.config.observability_config import MetricsConfig
from llmtuner.models.common.moe.experts import GroupedExperts, RoutedExperts
from llmtuner.parallel.expert_parallel.token_dispatch import LocalTokenDispatcher
from llmtuner.trainer.runtime.training_observer import (
    DispatchTimer,
    TrainingObserver,
    install_dispatch_timers,
    percentile,
)


def test_percentile_interpolates_nonuniform_samples():
    assert percentile([10.0, 1.0, 3.0], 0.5) == 3.0
    assert percentile([10.0, 1.0, 3.0], 0.9) == pytest.approx(8.6)


def test_dispatch_timing_does_not_change_autograd_and_drains_events():
    timer = DispatchTimer()
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    with timer.measure(value.device):
        output = value.square().sum()
    output.backward()
    torch.testing.assert_close(value.grad, torch.tensor([2.0, 4.0]))
    assert timer.consume_seconds() > 0
    assert timer.consume_seconds() == 0


def test_observer_excludes_warmup_and_between_step_work(monkeypatch):
    metrics = SimpleNamespace(
        config=MetricsConfig(metrics_warmup_steps=1, steady_window_steps=2),
        ntokens_since_last_log=0,
        valid_tokens_since_last_log=0,
        data_loading_times=[],
        _non_data_parallel_size=1,
        num_flops_per_token=1000,
        gpu_peak_flops=1e6,
    )
    observer = TrainingObserver(metrics, torch.device("cpu"))
    clock = iter([0, 10, 100, 102, 200, 204, 400, 406])
    monkeypatch.setattr(
        "llmtuner.trainer.runtime.training_observer.time.perf_counter",
        lambda: next(clock),
    )
    results = []
    for _ in range(4):
        observer.begin()
        metrics.ntokens_since_last_log += 100
        metrics.valid_tokens_since_last_log += 30
        metrics.data_loading_times.append(0.5)
        results.append(observer.end())
    assert "steady/samples" not in results[0]
    assert results[1]["steady/step_p50_seconds"] == 2
    assert results[-1]["steady/samples"] == 2
    assert results[-1]["steady/step_p50_seconds"] == 5
    assert results[-1]["steady/step_p90_seconds"] == pytest.approx(5.8)
    assert results[-1]["steady/tps_per_device"] == 20
    assert results[-1]["steady/mfu_percent"] == 2
    assert results[-1]["train/global_valid_targets"] == 30
    assert results[-1]["train/valid_target_fraction"] == 0.3


def test_steady_throughput_weights_time_and_counts_across_unequal_steps():
    metrics = SimpleNamespace(
        config=MetricsConfig(metrics_warmup_steps=1, steady_window_steps=2),
        num_flops_per_token=1000,
        gpu_peak_flops=0,
    )
    observer = TrainingObserver(metrics, torch.device("cpu"))
    assert observer.record_steady_step(seconds=100, tokens=10000, world=2) == {}
    observer.record_steady_step(seconds=1, tokens=100, world=2)
    result = observer.record_steady_step(seconds=3, tokens=900, world=2)
    assert result["steady/tps_per_device"] == 125
    assert "steady/mfu_percent" not in result

    result = observer.record_steady_step(seconds=2, tokens=100, world=2)
    assert result["steady/samples"] == 2
    assert result["steady/tps_per_device"] == 100
    assert result["steady/step_p50_seconds"] == 2.5


@pytest.mark.parametrize(
    "options", [{"metrics_warmup_steps": -1}, {"steady_window_steps": 0}]
)
def test_invalid_sampling_config_is_rejected(options):
    with pytest.raises(ValueError):
        MetricsConfig(**options)


def test_observer_uses_injected_group_for_errors_and_reductions(monkeypatch):
    group = object()
    coordinated = []
    reductions = []

    @contextmanager
    def coordinate(label, *, group):
        coordinated.append(group)
        yield

    def world_size(actual_group):
        assert actual_group is group
        return 2

    def reduce(value, op=None, *, group):
        reductions.append(group)
        if value.dtype == torch.int64:
            value.mul_(2)

    monkeypatch.setattr(observer_module, "synchronize_local_errors", coordinate)
    monkeypatch.setattr(observer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(observer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(observer_module.dist, "get_world_size", world_size)
    monkeypatch.setattr(observer_module.dist, "all_reduce", reduce)
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(observer_module.time, "perf_counter", lambda: next(clock))
    metrics = SimpleNamespace(
        config=MetricsConfig(metrics_warmup_steps=0, steady_window_steps=2),
        ntokens_since_last_log=0,
        valid_tokens_since_last_log=0,
        data_loading_times=[],
        _non_data_parallel_size=1,
        num_flops_per_token=1000,
        gpu_peak_flops=1e6,
    )
    observer = TrainingObserver(metrics, torch.device("cpu"), group=group)
    observer.begin()
    metrics.ntokens_since_last_log = 100
    metrics.valid_tokens_since_last_log = 40
    result = observer.end()

    assert coordinated == [group, group]
    assert reductions == [group, group]
    assert result["train/global_tokens"] == 200
    assert result["train/global_valid_targets"] == 80
    assert result["train/tps_per_device"] == 50


def test_dispatch_installation_deduplicates_shared_parts_and_reuses_timers():
    experts = RoutedExperts(GroupedExperts(8, 12, 4), LocalTokenDispatcher(4, 2))
    parts = [torch.nn.ModuleList([experts]), torch.nn.ModuleList([experts])]

    timers = install_dispatch_timers(parts)
    timers[0].cpu_seconds = 1.25

    assert len(timers) == 1
    assert experts.dispatch_observer is timers[0]
    assert install_dispatch_timers(parts) == timers
    assert timers[0].consume_seconds() == 1.25


def test_dispatch_installation_validates_all_owners_before_mutation():
    experts = [
        RoutedExperts(GroupedExperts(8, 12, 4), LocalTokenDispatcher(4, 2))
        for _ in range(2)
    ]
    custom = object()
    experts[1].dispatch_observer = custom

    with pytest.raises(ValueError, match="different observer"):
        install_dispatch_timers([torch.nn.ModuleList(experts)])

    assert experts[0].dispatch_observer is None
    assert experts[1].dispatch_observer is custom
