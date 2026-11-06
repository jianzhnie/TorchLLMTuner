"""Check timing aggregation and call boundaries; these are not NPU benchmarks."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from examples.npu_training_benchmark import (
    BenchmarkNPUTrainer,
    compile_counter_delta,
    summarize_resident_memory,
    summarize_timings,
)
from llmtuner.components.metrics import (
    compute_utilization,
    configured_tokens_per_step,
)
from llmtuner.trainer.trainer import Trainer


@pytest.mark.parametrize("failed_release", [None, "close", "destroy", "both"])
def test_initial_provenance_failure_closes_trainer_and_process_group(
    monkeypatch, failed_release
):
    from examples import npu_training_benchmark as benchmark

    calls = []

    def release(name):
        calls.append(name)
        if failed_release in (name, "both"):
            raise RuntimeError(f"{name} failed")

    monkeypatch.setattr(
        benchmark.torch,
        "npu",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )
    monkeypatch.setattr(benchmark.torch, "set_default_dtype", lambda dtype: None)
    monkeypatch.setattr(benchmark, "qwen3_30b_npu_config", lambda: object())
    monkeypatch.setattr(
        benchmark,
        "BenchmarkNPUTrainer",
        lambda *args, **kwargs: SimpleNamespace(close=lambda: release("close")),
    )
    monkeypatch.setattr(
        benchmark,
        "synchronize_local_errors",
        lambda label: nullcontext(),
    )

    def fail_metadata():
        raise RuntimeError("provenance failed")

    monkeypatch.setattr(benchmark, "runtime_metadata", fail_metadata)
    monkeypatch.setattr(
        benchmark.dist,
        "destroy_process_group",
        lambda: release("destroy"),
    )
    with pytest.raises(RuntimeError, match="provenance failed"):
        benchmark.main()
    assert calls == ["close", "destroy"]


def test_trainer_construction_failure_still_destroys_process_group(monkeypatch):
    from examples import npu_training_benchmark as benchmark

    calls = []
    monkeypatch.setattr(
        benchmark.torch,
        "npu",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )
    monkeypatch.setattr(benchmark.torch, "set_default_dtype", lambda dtype: None)
    monkeypatch.setattr(benchmark, "qwen3_30b_npu_config", lambda: object())

    def fail_construction(*args, **kwargs):
        raise RuntimeError("trainer construction failed")

    monkeypatch.setattr(benchmark, "BenchmarkNPUTrainer", fail_construction)
    monkeypatch.setattr(
        benchmark.dist,
        "destroy_process_group",
        lambda: calls.append("destroy"),
    )
    with pytest.raises(RuntimeError, match="trainer construction failed"):
        benchmark.main()
    assert calls == ["destroy"]


def test_local_report_failure_is_coordinated_before_gather(monkeypatch):
    from examples import npu_training_benchmark as benchmark

    calls = []
    boundaries = []

    @contextmanager
    def coordinated(label):
        boundaries.append(label)
        try:
            yield
        finally:
            boundaries.pop()

    def fail_memory_report():
        assert boundaries == ["Benchmark local report collection"]
        raise RuntimeError("memory report failed")

    trainer = SimpleNamespace(
        metrics=SimpleNamespace(num_flops_per_token=1, gpu_peak_flops=1),
        device="npu:0",
        benchmark_rows=[],
        train=lambda: calls.append("train"),
        close=lambda: calls.append("close"),
    )
    config = SimpleNamespace(
        parallel=SimpleNamespace(
            ep_config=SimpleNamespace(ep_token_dispatcher="alltoall")
        ),
        activation_swap=False,
    )
    monkeypatch.setattr(
        benchmark.torch,
        "npu",
        SimpleNamespace(
            is_available=lambda: True, max_memory_allocated=fail_memory_report
        ),
        raising=False,
    )
    monkeypatch.setattr(benchmark.torch, "set_default_dtype", lambda dtype: None)
    monkeypatch.setattr(benchmark, "qwen3_30b_npu_config", lambda: config)
    monkeypatch.setattr(benchmark, "BenchmarkNPUTrainer", lambda *a, **kw: trainer)
    monkeypatch.setattr(benchmark, "runtime_metadata", lambda: {})
    monkeypatch.setattr(benchmark, "synchronize_local_errors", coordinated)
    monkeypatch.setattr(benchmark.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(benchmark.dist, "get_backend", lambda: "hccl")
    monkeypatch.setattr(benchmark.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        benchmark.dist,
        "all_gather_object",
        lambda *args: pytest.fail("gather entered after local report failure"),
    )
    monkeypatch.setattr(
        benchmark.dist, "destroy_process_group", lambda: calls.append("destroy")
    )
    with pytest.raises(RuntimeError, match="memory report failed"):
        benchmark.main()
    assert calls == ["train", "close", "destroy"]
    assert boundaries == []


def records(durations):
    return [
        {"step": step, "seconds": seconds, "metrics": {"loss": 0.5}}
        for step, seconds in enumerate(durations, 1)
    ]


def test_compile_counter_delta_is_scoped_to_one_step():
    before = {"frames": {"layer": 3}, "recompile_reasons": {"shape": 2}}
    after = {"frames": {"layer": 4}, "recompile_reasons": {"shape": 2}}
    assert compile_counter_delta(before, after) == {
        "frames": {"layer": 1},
        "recompile_reasons": {"shape": 0},
    }


@pytest.mark.parametrize("accumulation", [1, 4])
def test_compute_utilization_counts_all_accumulated_batches(accumulation):
    config = SimpleNamespace(
        global_batch_size=8,
        max_seq_len=4096,
        gradient_accumulation_steps=accumulation,
    )
    tokens = configured_tokens_per_step(config)
    assert tokens == 32768 * accumulation
    result = compute_utilization(
        tokens_per_second=tokens / 2,
        flops_per_token=6 * 7_000_000_000,
        peak_flops_per_device=376e12,
        world_size=8,
    )
    assert result["tflops"] == pytest.approx(688.128 * accumulation)
    assert result["utilization_percent"] == pytest.approx(
        100 * 688.128 * accumulation / 3008
    )


def test_compute_utilization_uses_global_peak_compute():
    assert compute_utilization(
        tokens_per_second=1_000_000,
        flops_per_token=376_000_000,
        peak_flops_per_device=376e12,
        world_size=8,
    ) == {
        "tflops": 376.0,
        "peak_tflops": 3008.0,
        "utilization_percent": 12.5,
    }


def test_compute_utilization_does_not_guess_unknown_geometry():
    assert compute_utilization(
        tokens_per_second=1_000_000,
        flops_per_token=0,
        peak_flops_per_device=376e12,
        world_size=8,
    ) == {"tflops": None, "peak_tflops": None, "utilization_percent": None}


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
    monkeypatch.setattr(
        "examples.npu_training_benchmark.compile_counter_snapshot", lambda: {}
    )
    assert observer.train_step(iterator) is metrics
    assert calls == ["sync", "barrier", "sync", "reset", "train", "sync"]
    assert observer.benchmark_rows == [
        {"step": 2, "seconds": 3.0, "metrics": metrics, "compile_counters": {}}
    ]


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


def test_resident_memory_counts_local_parameters_gradients_and_optimizer_state():
    model = torch.nn.Linear(2, 3, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    result = summarize_resident_memory(model, [optimizer])

    expected = model.weight.numel() * model.weight.element_size()
    assert expected > 0
    assert result["parameters"] == expected
    assert result["gradients"] == expected
    assert result["optimizer_state"]["exp_avg"] == expected
    assert result["optimizer_state"]["exp_avg_sq"] == expected
    optimizer.zero_grad(set_to_none=True)
    assert summarize_resident_memory(model, [optimizer])["gradients"] == 0


def test_source_metadata_identifies_dirty_and_untracked_code(tmp_path):
    import subprocess

    from examples.npu_training_benchmark import source_metadata

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    source = tmp_path / "llmtuner"
    source.mkdir()
    tracked = source / "model.py"
    tracked.write_text("version = 1\n")
    git("add", ".")
    git("commit", "-m", "initial")
    clean = source_metadata(tmp_path)
    assert clean["git_dirty"] is False
    tracked.write_text("version = 2\n")
    changed = source_metadata(tmp_path)
    assert changed["git_dirty"] is True
    assert changed["git_commit"] == clean["git_commit"]
    assert changed["git_diff_sha256"] != clean["git_diff_sha256"]
    git("add", ".")
    assert source_metadata(tmp_path)["git_diff_sha256"] == changed["git_diff_sha256"]
    extra = source / "kernel.py"
    extra.write_text("version = 3\n")
    first = source_metadata(tmp_path)["untracked_source_sha256"]
    extra.write_text("version = 4\n")
    second = source_metadata(tmp_path)["untracked_source_sha256"]
    assert first["llmtuner/kernel.py"] != second["llmtuner/kernel.py"]

    native_source = source / "custom_kernel.CPP"
    native_source.write_text("void kernel() {}\n")
    native_hashes = source_metadata(tmp_path)["untracked_source_sha256"]
    assert "llmtuner/custom_kernel.CPP" in native_hashes
    config = tmp_path / "examples" / "npu_config.YML"
    config.parent.mkdir()
    config.write_text("sequence_length: 4096\n")
    config_hashes = source_metadata(tmp_path)["untracked_source_sha256"]
    assert "examples/npu_config.YML" in config_hashes

    binary = tmp_path / "llmtuner" / "weights.bin"
    binary.write_bytes(b"\x00\xff\x80")
    git("add", ".")
    git("commit", "-m", "add binary fixture")
    binary_clean = source_metadata(tmp_path)
    binary.write_bytes(b"\x00\xfe\x81")
    binary_changed = source_metadata(tmp_path)
    assert binary_changed["git_dirty"] is True
    assert binary_changed["git_diff_sha256"] != binary_clean["git_diff_sha256"]


def test_runtime_provenance_rejects_midrun_source_or_environment_change():
    from examples.npu_training_benchmark import validate_runtime_consistency

    before = {"git_commit": "abc", "git_diff_sha256": "123", "packages": {"torch": "2"}}
    validate_runtime_consistency(before, dict(before))
    for key, value in (
        ("git_commit", "def"),
        ("git_diff_sha256", "456"),
        ("packages", {"torch": "3"}),
    ):
        with pytest.raises(RuntimeError, match=key):
            validate_runtime_consistency(before, {**before, key: value})


def test_accelerator_provenance_includes_device_soc_and_cann(tmp_path, monkeypatch):
    from examples import npu_training_benchmark as benchmark

    version_file = tmp_path / "opp" / "version.info"
    version_file.parent.mkdir()
    version_file.write_text('Version="9.1.0"\n')
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "stale"))
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path))
    assert benchmark.cann_version() == "9.1.0"

    monkeypatch.setattr(benchmark, "device_type", "npu")
    monkeypatch.setattr(
        benchmark,
        "device_module",
        SimpleNamespace(
            current_device=lambda: 3,
            get_device_name=lambda device: f"Ascend910_{device}",
            get_soc_version=lambda: 251,
        ),
    )
    assert benchmark.accelerator_metadata() == {
        "type": "npu",
        "name": "Ascend910_3",
        "soc_version": "251",
        "cann_version": "9.1.0",
    }


def test_rank_runtime_validation_allows_hosts_but_rejects_code_drift():
    from examples.npu_training_benchmark import validate_rank_runtimes

    reference = {
        "git_commit": "abc",
        "git_diff_sha256": "123",
        "untracked_source_sha256": {},
        "packages": {"torch": "2"},
        "python": "3.12",
        "accelerator": {
            "type": "npu",
            "name": "Ascend910_9392",
            "soc_version": "251",
            "cann_version": "9.1.0",
        },
        "platform": "host-a",
    }
    other = {**reference, "platform": "host-b"}
    validate_rank_runtimes([{"rank": 1, "runtime": other}], reference)
    for key in ("git_commit", "git_diff_sha256", "python", "accelerator"):
        with pytest.raises(RuntimeError, match=f"rank 1 runtime mismatch: {key}"):
            validate_rank_runtimes(
                [{"rank": 1, "runtime": {**other, key: "different"}}], reference
            )
    with pytest.raises(RuntimeError, match="untracked_source_sha256"):
        validate_rank_runtimes(
            [{"rank": 2, "runtime": {**other, "untracked_source_sha256": {"x": "y"}}}],
            reference,
        )
