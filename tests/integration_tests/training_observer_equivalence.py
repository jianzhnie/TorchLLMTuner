"""torchrun --standalone --nproc_per_node=2 this_file.py [--device npu]."""

import argparse
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from llmtuner.config.observability_config import MetricsConfig
from llmtuner.trainer.runtime.training_observer import TrainingObserver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    args = parser.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl" if args.device == "npu" else "gloo")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world >= 2
        metrics = SimpleNamespace(
            config=MetricsConfig(metrics_warmup_steps=1),
            ntokens_since_last_log=0,
            valid_tokens_since_last_log=0,
            data_loading_times=[],
            _non_data_parallel_size=1,
            num_flops_per_token=1000,
            gpu_peak_flops=376e12,
        )
        device = torch.device(args.device)
        observer = TrainingObserver(metrics, device)
        for index in range(3):
            observer.begin()
            metrics.ntokens_since_last_log += 100 * (rank + 1)
            metrics.valid_tokens_since_last_log += 10 * (rank + 1)
            time.sleep(0.05 * (rank + 1))
            result = observer.end()
            expected = 100 * world * (world + 1) // 2
            assert result["train/global_tokens"] == expected
            assert result["train/global_valid_targets"] == expected // 10
            assert result["train/slowest_rank_step_seconds"] >= 0.05 * world
            assert ("steady/samples" in result) == (index > 0)
            values = torch.tensor(
                [result["train/global_tps"]], device=device, dtype=torch.float32
            )
            reference = values.clone()
            dist.broadcast(reference, 0)
            torch.testing.assert_close(values, reference, rtol=0, atol=0)
        # TP/CP/PP duplicate streams are removed before dividing over devices.
        metrics._non_data_parallel_size = world
        observer.begin()
        metrics.ntokens_since_last_log += 100
        metrics.valid_tokens_since_last_log += 25
        result = observer.end()
        assert result["train/global_tokens"] == 100
        assert result["train/global_valid_targets"] == 25
        if rank == 0:
            print("PASS: asymmetric global targets, slowest rank, warmup, replication")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
