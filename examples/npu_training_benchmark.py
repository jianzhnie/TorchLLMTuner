"""Measure synchronized train_step durations without input/gradient auditing.

Run with the Qwen3 NPU example's environment and LLMTUNER_SAVE_CHECKPOINT=0.
LLMTUNER_BENCHMARK_WARMUP selects excluded steps (default 10). Each run needs
a fresh dump folder. Results describe this run; they do not establish matched
torchtitan-npu throughput or an acceptance threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import torch.distributed as dist
from torch._dynamo.utils import counters
from torch.distributed.tensor import DTensor

from examples.train_qwen3_30b_npu import qwen3_30b_npu_config
from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.accelerator.device import device_module, device_type
from llmtuner.components.compute_metrics import (
    compute_utilization,
    configured_tokens_per_step,
    token_throughput,
)
from llmtuner.trainer.trainer import Trainer

_PROVENANCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".json",
    ".py",
    ".pyi",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}


def summarize_timings(rank_rows: list[list[dict]], *, steps: int, warmup: int) -> dict:
    """Use the slowest rank per step, after checking complete rank coverage."""
    if not rank_rows or not 0 <= warmup < steps:
        raise ValueError("Require ranks and 0 <= warmup < steps")
    for rows in rank_rows:
        if [row["step"] for row in rows] != list(range(1, steps + 1)):
            raise ValueError("Benchmark requires every completed step on every rank")
        for row in rows:
            if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
                raise ValueError("Step durations must be finite and positive")
            if row["metrics"] is not None and not all(
                math.isfinite(value) for value in row["metrics"].values()
            ):
                raise ValueError("Benchmark metrics must be finite")
    durations = [
        max(rows[index]["seconds"] for rows in rank_rows)
        for index in range(warmup, steps)
    ]
    quantiles = torch.tensor(durations, dtype=torch.float64, device="cpu").quantile(
        torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device="cpu")
    )
    return {
        "warmup_steps": warmup,
        "measured_steps": len(durations),
        "world_size": len(rank_rows),
        "step_seconds_max_across_ranks": durations,
        "mean_step_seconds": mean(durations),
        "p10_step_seconds": quantiles[0].item(),
        "p50_step_seconds": quantiles[1].item(),
        "p90_step_seconds": quantiles[2].item(),
    }


def compile_counter_snapshot() -> dict:
    return {
        name: dict(counters[name])
        for name in ("frames", "recompile_reasons", "aot_autograd", "inductor")
    }


def compile_counter_delta(before: dict, after: dict) -> dict:
    return {
        name: {
            key: value - before.get(name, {}).get(key, 0)
            for key, value in values.items()
        }
        for name, values in after.items()
    }


class BenchmarkNPUTrainer(Trainer):
    def __init__(self, config, *, warmup: int):
        if not 0 <= warmup < config.steps:
            raise ValueError("Require 0 <= benchmark warmup < training.steps")
        if not config.checkpoint.load_only or config.checkpoint.load_step != -1:
            raise ValueError("Benchmark requires load-only initial weights, no resume")
        if config.validation is not None or (
            config.profiler.enable_profiling or config.profiler.enable_memory_snapshot
        ):
            raise ValueError("Benchmark requires validation and profiler disabled")
        folder = Path(config.dump_folder)
        if (folder / "benchmark.json").exists() or any(
            (folder / config.checkpoint.folder).glob("step-*")
        ):
            raise ValueError("Benchmark requires a fresh checkpoint/report directory")
        self.benchmark_warmup = warmup
        self.benchmark_rows = []
        super().__init__(config)

    def train_step(self, data_iterator):
        torch.npu.synchronize()
        if dist.is_initialized():
            dist.barrier()
        torch.npu.synchronize()
        if self.step == self.benchmark_warmup + 1:
            torch.npu.reset_peak_memory_stats()
        compile_before = compile_counter_snapshot()
        start = perf_counter()
        metrics = super().train_step(data_iterator)
        torch.npu.synchronize()
        seconds = perf_counter() - start
        self.benchmark_rows.append(
            {
                "step": self.step,
                "seconds": seconds,
                "metrics": metrics,
                "compile_counters": compile_counter_delta(
                    compile_before, compile_counter_snapshot()
                ),
            }
        )
        return metrics


def source_metadata(repo_root: Path) -> dict:
    """Identify tracked edits and untracked training code without saving contents."""

    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout

    patch = git("diff", "HEAD", "--binary", "--no-ext-diff")
    untracked = {}
    untracked_files = git("ls-files", "--others", "--exclude-standard", "-z")
    for name in untracked_files.decode("utf-8").split("\0"):
        if not name or not name.startswith(("llmtuner/", "examples/", "tests/")):
            continue
        path = repo_root / name
        if path.is_file() and path.suffix.lower() in _PROVENANCE_SUFFIXES:
            untracked[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "git_commit": git("rev-parse", "HEAD").decode("ascii").strip(),
        "git_dirty": bool(patch or untracked),
        "git_diff_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked_source_sha256": untracked,
    }


def cann_version() -> str | None:
    """Read CANN's installed version without invoking a shell command."""
    roots = dict.fromkeys(
        Path(value)
        for name in (
            "ASCEND_HOME_PATH",
            "ASCEND_TOOLKIT_HOME",
            "ASCEND_TOOLKIT_LATEST_HOME",
        )
        if (value := os.environ.get(name))
    )
    for toolkit_root in roots:
        for path in (
            toolkit_root / "opp" / "version.info",
            toolkit_root / "version.info",
            toolkit_root / "version.cfg",
        ):
            try:
                lines = path.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                key, separator, value = line.partition("=")
                if separator and key.strip() == "Version":
                    return value.strip().strip('"') or None
    return None


def accelerator_metadata() -> dict:
    """Describe the active accelerator and its device-stack version."""
    if device_type != "npu":
        return {"type": device_type, "name": device_type}
    device_index = device_module.current_device()
    return {
        "type": device_type,
        "name": device_module.get_device_name(device_index),
        "soc_version": str(device_module.get_soc_version()),
        "cann_version": cann_version(),
    }


def runtime_metadata() -> dict:
    repo_root = Path(__file__).resolve().parents[1]

    def package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "accelerator": accelerator_metadata(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "torch-npu", "transformers", "datasets")
        },
        **source_metadata(repo_root),
    }


def validate_runtime_consistency(before: dict, after: dict) -> None:
    """Do not publish timings under a source/environment identity that changed."""
    if before != after:
        changed = sorted(
            key
            for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)
        )
        raise RuntimeError(
            "Benchmark runtime changed during execution: " + ", ".join(changed)
        )


def validate_rank_runtimes(records: list[dict], reference: dict) -> None:
    """Require matching code and dependencies while permitting distinct hosts."""
    for rank in records:
        for key in (
            "git_commit",
            "git_diff_sha256",
            "untracked_source_sha256",
            "packages",
            "python",
            "accelerator",
        ):
            if rank["runtime"][key] != reference[key]:
                raise RuntimeError(
                    f"Benchmark rank {rank['rank']} runtime mismatch: {key}"
                )


def resident_nbytes(tensor: torch.Tensor) -> int:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.numel() * tensor.element_size()


def summarize_resident_memory(model, optimizers) -> dict:
    parameters = list(model.parameters())
    optimizer_states: dict[str, int] = {}
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_states[name] = optimizer_states.get(name, 0) + (
                        resident_nbytes(value)
                    )
    return {
        "parameters": sum(resident_nbytes(parameter) for parameter in parameters),
        "gradients": sum(
            resident_nbytes(parameter.grad)
            for parameter in parameters
            if parameter.grad is not None
        ),
        "optimizer_state": dict(sorted(optimizer_states.items())),
    }


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Benchmark requires a visible Ascend NPU")
    config = qwen3_30b_npu_config()
    warmup = int(os.environ.get("LLMTUNER_BENCHMARK_WARMUP", "10"))
    torch.set_default_dtype(torch.bfloat16)
    trainer = None
    try:
        trainer = BenchmarkNPUTrainer(config, warmup=warmup)
        with synchronize_local_errors("Benchmark initial runtime provenance"):
            runtime_before = runtime_metadata()
            num_flops_per_token = trainer.metrics.num_flops_per_token
            peak_flops_per_device = trainer.metrics.gpu_peak_flops
        if config.parallel.ep_config.ep_token_dispatcher == "mc2":
            from llmtuner.parallel.expert_parallel.mc2_dispatch import (
                reset_mc2_dispatch_metrics,
            )

            reset_mc2_dispatch_metrics()
        if config.activation_swap:
            from llmtuner.memory.activation_swap import reset_activation_swap_metrics

            reset_activation_swap_metrics()
        trainer.train()
        with synchronize_local_errors("Benchmark runtime provenance"):
            runtime_after = runtime_metadata()
            validate_runtime_consistency(runtime_before, runtime_after)
        with synchronize_local_errors("Benchmark local report collection"):
            local = {
                "rank": dist.get_rank(),
                "device": str(trainer.device),
                "backend": str(dist.get_backend()),
                "runtime": runtime_before,
                "steps": trainer.benchmark_rows,
                "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                "peak_host_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
                "model_dtypes": sorted(
                    {str(parameter.dtype) for parameter in trainer.model.parameters()}
                ),
                "default_dtype": str(torch.get_default_dtype()),
                "resident_bytes": summarize_resident_memory(
                    trainer.model, trainer.optimizer.optimizers
                ),
                "residency_scope": (
                    "rank-local logical tensor bytes; DTensor uses to_local(); "
                    "gradients and optimizer state are snapshots after the final step, "
                    "not allocator peaks or activation/workspace measurements"
                ),
                "optimizer_defaults": [
                    optimizer.defaults for optimizer in trainer.optimizer.optimizers
                ],
            }
            if config.parallel.ep_config.ep_token_dispatcher == "mc2":
                from llmtuner.parallel.expert_parallel.mc2_dispatch import (
                    mc2_dispatch_metrics,
                )

                local["mc2_dispatch"] = mc2_dispatch_metrics()
            if config.activation_swap:
                from llmtuner.memory.activation_swap import activation_swap_metrics

                local["activation_swap"] = activation_swap_metrics()
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        with synchronize_local_errors("Benchmark report"):
            validate_rank_runtimes(gathered, runtime_before)
            report = summarize_timings(
                [rank["steps"] for rank in gathered],
                steps=config.steps,
                warmup=warmup,
            )
            report.update(
                timing_scope=(
                    "synchronized full train_step including data fetch, forward, "
                    "backward, optimizer and internal metrics; excludes outer "
                    "logging, GC, checkpoint, validation, profiler and startup; "
                    "rank barrier occurs before the timed interval"
                ),
                configured_tokens_per_second=(
                    token_throughput(
                        configured_tokens_per_step(config), report["mean_step_seconds"]
                    )
                ),
                token_definition=(
                    "global_batch_size * max_seq_len * gradient_accumulation_steps; "
                    "configured padded tokens, not valid loss tokens"
                ),
                peak_allocated_bytes=max(
                    rank["peak_allocated_bytes"] for rank in gathered
                ),
                peak_host_rss_bytes=max(
                    rank["peak_host_rss_bytes"] for rank in gathered
                ),
                runtime=runtime_before if dist.get_rank() == 0 else None,
                rank_records=gathered,
                configuration=asdict(config),
                acceptance="measurement only; matched baseline/standard still required",
            )
            report["compute"] = compute_utilization(
                tokens_per_second=report["configured_tokens_per_second"],
                flops_per_token=num_flops_per_token,
                peak_flops_per_device=peak_flops_per_device,
                world_size=dist.get_world_size(),
            )
            if dist.get_rank() == 0:
                Path(config.dump_folder).mkdir(parents=True, exist_ok=True)
                (Path(config.dump_folder) / "benchmark.json").write_text(
                    json.dumps(report, indent=2, default=str, allow_nan=False) + "\n"
                )
    finally:
        active_error = sys.exc_info()[1]
        cleanup_error = None
        releases = [] if trainer is None else [trainer.close]
        releases.append(dist.destroy_process_group)
        for release in releases:
            try:
                release()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if active_error is None and cleanup_error is not None:
            raise cleanup_error


if __name__ == "__main__":
    main()
