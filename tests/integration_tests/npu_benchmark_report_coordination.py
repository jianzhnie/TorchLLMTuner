"""Two real NPU training ranks reject a rank-local benchmark report failure.

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    tests/integration_tests/npu_benchmark_report_coordination.py /tmp/benchmark-fault

Only the recipe and report fault are injected. Training, optimizer updates,
runtime provenance, error coordination and process-group cleanup are real.
This small-model failure test does not measure target-model performance.
"""

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from examples import npu_training_benchmark as benchmark
from llmtuner.config import (
    CheckpointConfig,
    LLMTunerConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)


def local_copy(parameter):
    if isinstance(parameter, DTensor):
        parameter = parameter.to_local()
    return parameter.detach().cpu().clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--failing-rank", type=int, choices=(0, 1), default=0)
    options = parser.parse_args()
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2
    folder = options.directory
    config = LLMTunerConfig(
        model=ModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        parallel=ParallelConfig(data_parallel_shard_size=2),
        training=TrainingConfig(
            global_batch_size=2,
            max_seq_len=8,
            steps=2,
            dump_folder=str(folder),
            checkpoint_config=CheckpointConfig(enable=False, load_only=True),
        ),
    )
    trainers = []
    initial = []
    trainer_type = benchmark.BenchmarkNPUTrainer
    summarize = benchmark.summarize_resident_memory
    failure = RuntimeError(f"injected report failure on rank {options.failing_rank}")

    def build_trainer(*args, **kwargs):
        trainer = trainer_type(*args, **kwargs)
        trainers.append(trainer)
        initial.extend(local_copy(p) for p in trainer.model.parameters())
        return trainer

    def collect_report(*args, **kwargs):
        if rank == options.failing_rank:
            raise failure
        return summarize(*args, **kwargs)

    previous_dtype = torch.get_default_dtype()
    try:
        with (
            patch.object(benchmark, "qwen3_30b_npu_config", lambda: config),
            patch.object(benchmark, "BenchmarkNPUTrainer", build_trainer),
            patch.object(benchmark, "summarize_resident_memory", collect_report),
            patch.dict(os.environ, {"LLMTUNER_BENCHMARK_WARMUP": "0"}),
        ):
            try:
                benchmark.main()
            except RuntimeError as error:
                if rank == options.failing_rank:
                    assert error is failure
                else:
                    assert str(error) == (
                        "Benchmark local report collection failed on another rank"
                    ), str(error)
            else:
                raise AssertionError("Benchmark accepted failed report collection")
    finally:
        torch.set_default_dtype(previous_dtype)

    assert not dist.is_initialized(), "benchmark leaked its process group"
    assert len(trainers) == 1
    trainer = trainers[0]
    assert [row["step"] for row in trainer.benchmark_rows] == [1, 2]
    assert any(
        not torch.equal(local_copy(parameter), before)
        for parameter, before in zip(trainer.model.parameters(), initial, strict=True)
    ), "report fault was not preceded by real parameter updates"
    assert not (folder / "benchmark.json").exists()
    print(f"PASS rank={rank}: trained, rejected report failure, destroyed group")


if __name__ == "__main__":
    main()
