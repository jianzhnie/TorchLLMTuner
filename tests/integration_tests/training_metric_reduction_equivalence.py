"""Check training metrics with actual topology groups and independent sums.

Run with PYTHONPATH=. torchrun --nproc_per_node=4. Explicit scalar shard
losses test reporting only; no model transformations or backward are covered.
"""

import os

import torch
import torch.distributed as dist

from llmtuner.accelerator.collectives import all_reduce
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.runtime.reductions import resolve_training_reduction_meshes
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.harness import init_distributed


def main():
    device = init_distributed()
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    try:
        assert dist.get_world_size() == 4
        for cp, tp in ((2, 2), (2, 1), (1, 2)):
            topology = ParallelTopology.from_config(
                ParallelConfig(
                    context_parallel_kv_allgather_size=cp,
                    tensor_parallel_size=tp,
                    data_parallel_shard_size=-1,
                ),
                4,
            )
            meshes = resolve_training_reduction_meshes(topology)
            dp = meshes.dp_mesh.get_local_rank() if meshes.dp_mesh is not None else 0
            coordinate = dist.get_rank() % (cp * tp)
            for empty_dp in (False, True):
                counts = [
                    0 if empty_dp and d == 0 else 4 * (d + 1)
                    for d in range(4 // (cp * tp))
                ]
                shard_sums = [
                    [
                        0.0 if count == 0 else float((d + 1) * (i + 1))
                        for i in range(cp * tp)
                    ]
                    for d, count in enumerate(counts)
                ]
                trainer = Trainer.__new__(Trainer)
                trainer.device = device
                trainer.parallel_dims = topology
                trainer.ntokens_seen = 7
                trainer.model_parts = []
                metrics = (
                    trainer._step_metrics(
                        accumulated_loss=torch.tensor(
                            shard_sums[dp][coordinate], dtype=dtype, device=device
                        ),
                        global_valid_tokens=torch.tensor(
                            sum(counts), dtype=dtype, device=device
                        ),
                        local_valid_tokens=counts[dp],
                        local_valid_tokens_tensor=torch.tensor(
                            counts[dp], dtype=dtype, device=device
                        ),
                        reduction_meshes=meshes,
                        grad_norm=torch.tensor(2.0, device=device),
                        lr_metrics={"lr/SGD": 0.1},
                    )
                    if sum(counts)
                    else None
                )
                if metrics is None:
                    continue  # globally empty steps are rejected before reporting
                reference = os.environ.get(
                    "LLMTUNER_TEST_METRICS_REFERENCE", "sum_then_divide"
                )
                if reference == "per_rank":
                    expected_tensor = torch.tensor(
                        shard_sums[dp][coordinate], dtype=dtype, device=device
                    ) / sum(counts)
                    all_reduce(expected_tensor, group=meshes.loss_mesh.get_group())
                    expected = expected_tensor.item()
                elif reference == "sum_then_divide":
                    expected = sum(map(sum, shard_sums)) / sum(counts)
                else:
                    raise ValueError(f"Unknown metrics reference: {reference}")
                maximum = max(
                    sum(losses) / count if count else 0
                    for losses, count in zip(shard_sums, counts, strict=True)
                )
                torch.testing.assert_close(
                    metrics,
                    {
                        "loss": expected,
                        "max_loss": maximum,
                        "grad_norm": 2.0,
                        "n_tokens_seen": 28.0,
                        "lr/SGD": 0.1,
                    },
                )
                dist.barrier()
                if dist.get_rank() == 0:
                    print(f"CP{cp}/TP{tp} {empty_dp=}: independent metrics passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
