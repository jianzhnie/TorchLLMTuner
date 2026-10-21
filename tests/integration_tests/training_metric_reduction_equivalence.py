"""Check training metrics with actual topology groups and independent sums.

Run with PYTHONPATH=. torchrun --nproc_per_node=4. Explicit scalar shard
losses test reporting only; no model transformations or backward are covered.
"""

import torch
import torch.distributed as dist

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.runtime.reductions import resolve_training_reduction_meshes
from llmtuner.trainer.trainer import Trainer


def main():
    dist.init_process_group("gloo")
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
                trainer.device = torch.device("cpu")
                trainer.parallel_dims = topology
                trainer.ntokens_seen = 7
                trainer.model_parts = []
                metrics = (
                    trainer._step_metrics(
                        accumulated_loss=torch.tensor(
                            shard_sums[dp][coordinate], dtype=torch.float64
                        ),
                        global_valid_tokens=torch.tensor(
                            sum(counts), dtype=torch.float64
                        ),
                        local_valid_tokens=counts[dp],
                        local_valid_tokens_tensor=torch.tensor(
                            counts[dp], dtype=torch.float64
                        ),
                        reduction_meshes=meshes,
                        grad_norm=torch.tensor(2.0),
                        lr_metrics={"lr/SGD": 0.1},
                    )
                    if sum(counts)
                    else None
                )
                if metrics is None:
                    continue  # globally empty steps are rejected before reporting
                expected = sum(map(sum, shard_sums)) / sum(counts)
                maximum = max(
                    sum(losses) / count if count else 0
                    for losses, count in zip(shard_sums, counts, strict=True)
                )
                assert metrics == {
                    "loss": expected,
                    "max_loss": maximum,
                    "grad_norm": 2.0,
                    "n_tokens_seen": 28.0,
                    "lr/SGD": 0.1,
                }, metrics
                dist.barrier()
                if dist.get_rank() == 0:
                    print(f"CP{cp}/TP{tp} {empty_dp=}: independent metrics passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
