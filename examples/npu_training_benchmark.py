"""Measure synchronized train_step durations without input/gradient auditing.

Run with the Qwen3 NPU example's environment and LLMTUNER_SAVE_CHECKPOINT=0.
LLMTUNER_BENCHMARK_WARMUP selects excluded steps (default 10). Each run needs
a fresh dump folder. Results describe this run; they do not establish matched
torchtitan-npu throughput or an acceptance threshold.
"""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import subprocess
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
from llmtuner.trainer.trainer import Trainer


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


def compute_utilization(
    *,
    tokens_per_second: float,
    flops_per_token: int,
    peak_flops_per_device: float,
    world_size: int,
) -> dict[str, float | None]:
    """Convert global throughput into TFLOPS and peak-compute utilization."""
    if flops_per_token <= 0 or peak_flops_per_device <= 0 or world_size <= 0:
        return {"tflops": None, "peak_tflops": None, "utilization_percent": None}
    tflops = flops_per_token * tokens_per_second / 1e12
    peak_tflops = peak_flops_per_device * world_size / 1e12
    return {
        "tflops": tflops,
        "peak_tflops": peak_tflops,
        "utilization_percent": 100 * tflops / peak_tflops,
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


def runtime_metadata() -> dict:
    repo_root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    def package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "torch-npu", "transformers", "datasets")
        },
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


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
    trainer = BenchmarkNPUTrainer(config, warmup=warmup)
    num_flops_per_token = trainer.metrics.num_flops_per_token
    peak_flops_per_device = trainer.metrics.gpu_peak_flops
    try:
        if config.parallel.ep_config.ep_token_dispatcher == "mc2":
            from llmtuner.parallel.expert_parallel.mc2_dispatch import (
                reset_mc2_dispatch_metrics,
            )

            reset_mc2_dispatch_metrics()
        if config.activation_swap:
            from llmtuner.memory.activation_swap import reset_activation_swap_metrics

            reset_activation_swap_metrics()
        trainer.train()
        local = {
            "rank": dist.get_rank(),
            "device": str(trainer.device),
            "backend": str(dist.get_backend()),
            "steps": trainer.benchmark_rows,
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
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
                    config.global_batch_size
                    * config.max_seq_len
                    / report["mean_step_seconds"]
                ),
                token_definition="configured padded tokens, not valid loss tokens",
                peak_allocated_bytes=max(
                    rank["peak_allocated_bytes"] for rank in gathered
                ),
                peak_host_rss_bytes=max(
                    rank["peak_host_rss_bytes"] for rank in gathered
                ),
                runtime=runtime_metadata() if dist.get_rank() == 0 else None,
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
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
