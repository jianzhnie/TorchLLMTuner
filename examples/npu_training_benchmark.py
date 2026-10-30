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
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import torch.distributed as dist

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
        start = perf_counter()
        metrics = super().train_step(data_iterator)
        torch.npu.synchronize()
        seconds = perf_counter() - start
        self.benchmark_rows.append(
            {"step": self.step, "seconds": seconds, "metrics": metrics}
        )
        return metrics


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Benchmark requires a visible Ascend NPU")
    config = qwen3_30b_npu_config()
    warmup = int(os.environ.get("LLMTUNER_BENCHMARK_WARMUP", "10"))
    torch.set_default_dtype(torch.bfloat16)
    trainer = BenchmarkNPUTrainer(config, warmup=warmup)
    try:
        trainer.train()
        local = {
            "rank": dist.get_rank(),
            "device": str(trainer.device),
            "backend": str(dist.get_backend()),
            "steps": trainer.benchmark_rows,
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "optimizer_defaults": [
                optimizer.defaults for optimizer in trainer.optimizer.optimizers
            ],
        }
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
                rank_records=gathered,
                configuration=asdict(config),
                acceptance="measurement only; matched baseline/standard still required",
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
