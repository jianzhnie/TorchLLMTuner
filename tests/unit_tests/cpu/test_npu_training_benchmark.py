"""Check timing aggregation and call boundaries; these are not NPU benchmarks."""

from types import SimpleNamespace

import pytest
import torch

from examples.npu_training_benchmark import BenchmarkNPUTrainer, summarize_timings
from llmtuner.trainer.trainer import Trainer


def records(durations):
    return [
        {"step": step, "seconds": seconds, "metrics": {"loss": 0.5}}
        for step, seconds in enumerate(durations, 1)
    ]


def test_uses_slowest_rank_per_step_and_discards_warmup():
    result = summarize_timings(
        [records([100, 1, 5]), records([200, 3, 2])], steps=3, warmup=1
    )
    assert result["step_seconds_max_across_ranks"] == [3, 5]
    assert result["mean_step_seconds"] == 4
    assert result["p50_step_seconds"] == 4
    assert result["p90_step_seconds"] == pytest.approx(4.8)
    assert result["world_size"] == 2
    assert result["measured_steps"] == 2


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_durations_even_in_warmup(duration):
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_timings([records([duration, 2])], steps=2, warmup=1)


@pytest.mark.parametrize("steps", [[1, 3], [1, 1], [2, 3], [1]])
def test_rejects_incomplete_or_duplicate_steps(steps):
    rows = records([1] * len(steps))
    for row, step in zip(rows, steps, strict=True):
        row["step"] = step
    with pytest.raises(ValueError, match="every completed step"):
        summarize_timings([rows], steps=2, warmup=0)


@pytest.mark.parametrize("warmup", [-1, 2, 3])
def test_rejects_warmup_without_measured_steps(warmup):
    with pytest.raises(ValueError, match="warmup"):
        summarize_timings([records([1, 2])], steps=2, warmup=warmup)


def test_rejects_empty_rank_list_and_nonfinite_metrics():
    with pytest.raises(ValueError, match="ranks"):
        summarize_timings([], steps=2, warmup=0)
    rows = records([1, 2])
    rows[0]["metrics"]["loss"] = float("nan")
    with pytest.raises(ValueError, match="metrics"):
        summarize_timings([rows], steps=2, warmup=0)


def test_single_measured_step_and_nonreporting_rank():
    rows = records([9, 2])
    rows[1]["metrics"] = None
    result = summarize_timings([rows], steps=2, warmup=1)
    assert result["p10_step_seconds"] == result["p90_step_seconds"] == 2


def test_timer_preserves_iterator_and_metrics_without_auditing(monkeypatch):
    observer = object.__new__(BenchmarkNPUTrainer)
    observer.step = 2
    observer.benchmark_warmup = 1
    observer.benchmark_rows = []
    batch = object()
    iterator = iter([batch])
    metrics = {"loss": 0.5}
    calls = []

    def train_step(self, observed):
        assert observed is iterator
        assert next(observed) is batch
        calls.append("train")
        return metrics

    times = iter([10.0, 13.0])
    monkeypatch.setattr(Trainer, "train_step", train_step)
    monkeypatch.setattr(
        "examples.npu_training_benchmark.perf_counter", lambda: next(times)
    )
    monkeypatch.setattr(
        "examples.npu_training_benchmark.dist.barrier",
        lambda: calls.append("barrier"),
    )
    monkeypatch.setattr(
        "examples.npu_training_benchmark.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: calls.append("sync"),
            reset_peak_memory_stats=lambda: calls.append("reset"),
        ),
        raising=False,
    )
    assert observer.train_step(iterator) is metrics
    assert calls == ["sync", "barrier", "sync", "reset", "train", "sync"]
    assert observer.benchmark_rows == [{"step": 2, "seconds": 3.0, "metrics": metrics}]


def test_failed_step_does_not_receive_a_success_record(monkeypatch):
    observer = object.__new__(BenchmarkNPUTrainer)
    observer.step = 1
    observer.benchmark_warmup = 1
    observer.benchmark_rows = []

    def fail(self, iterator):
        raise RuntimeError("training failed")

    monkeypatch.setattr(Trainer, "train_step", fail)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False
    )
    with pytest.raises(RuntimeError, match="training failed"):
        observer.train_step(iter([]))
    assert observer.benchmark_rows == []


@pytest.mark.parametrize("invalid", ["save", "resume", "validation", "profiler"])
def test_rejects_runs_that_mix_training_with_other_work(tmp_path, invalid):
    cfg = SimpleNamespace(
        steps=3,
        checkpoint=SimpleNamespace(load_only=True, load_step=-1, folder="checkpoint"),
        validation=None,
        profiler=SimpleNamespace(enable_profiling=False, enable_memory_snapshot=False),
        dump_folder=str(tmp_path),
    )
    if invalid == "save":
        cfg.checkpoint.load_only = False
    elif invalid == "resume":
        cfg.checkpoint.load_step = 1
    elif invalid == "validation":
        cfg.validation = object()
    else:
        cfg.profiler.enable_profiling = True
    with pytest.raises(ValueError):
        BenchmarkNPUTrainer(cfg, warmup=1)


@pytest.mark.parametrize("existing", ["benchmark.json", "checkpoint/step-50"])
def test_rejects_existing_reports_and_implicit_resume(tmp_path, existing):
    path = tmp_path / existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    cfg = SimpleNamespace(
        steps=3,
        checkpoint=SimpleNamespace(load_only=True, load_step=-1, folder="checkpoint"),
        validation=None,
        profiler=SimpleNamespace(enable_profiling=False, enable_memory_snapshot=False),
        dump_folder=str(tmp_path),
    )
    with pytest.raises(ValueError, match="fresh"):
        BenchmarkNPUTrainer(cfg, warmup=1)
