"""Training-only scopes and bounded steady statistics."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.config.observability_config import MetricsConfig
from llmtuner.trainer.runtime.training_observer import (
    DispatchTimer,
    TrainingObserver,
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
