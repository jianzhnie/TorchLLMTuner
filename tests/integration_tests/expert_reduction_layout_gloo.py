"""Real Gloo split-layout transport and rank-local preparation failure checks.

Run from the repository root with torchrun --standalone --nproc-per-node=2
--module tests.integration_tests.expert_reduction_layout_gloo. This validates
mechanics only, not NPU stream behavior, HCCL rounding or target-model precision.
"""

from datetime import timedelta

import torch
import torch.distributed as dist

from llmtuner.parallel.fully_shard_parallel.expert_reduction_layout import (
    SplitExpertReduceScatter,
)


class Communication:
    def __init__(self):
        self.calls = 0

    def __call__(self, output, inputs, group, op, async_op=False):
        self.calls += 1
        return dist.reduce_scatter_tensor(
            output, inputs, group=group, op=op, async_op=async_op
        )


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == 2
        # Each destination block contains a prefix, packed experts and suffix.
        block = 43
        source = torch.arange(world * block, dtype=torch.float32) + rank * 1000
        original = source.clone()
        expected = sum(
            torch.arange(world * block, dtype=torch.float32) + peer * 1000
            for peer in range(world)
        )[rank * block : (rank + 1) * block]
        output = torch.full((block,), -1.0)
        comm = Communication()
        workspace = (source.numel() + output.numel()) * source.element_size()

        def adapter(budget):
            return SplitExpertReduceScatter(
                comm,
                world_size=world,
                offset=3,
                experts=2,
                width=2,
                hidden=3,
                workspace_limit_bytes=budget,
                output_numel=block,
            )

        adapter(workspace)(output, source, dist.group.WORLD, dist.ReduceOp.SUM)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert torch.equal(source, original) and comm.calls == 1
        previous = output.clone()
        # Only rank 1 fails locally. Neither rank may enter reduce-scatter.
        try:
            adapter(workspace - 1 if rank == 1 else workspace)(
                output, source, dist.group.WORLD, dist.ReduceOp.SUM
            )
        except (ValueError, RuntimeError) as error:
            assert (
                "workspace exceeds" in str(error)
                if rank == 1
                else "failed on another rank" in str(error)
            ), str(error)
        else:
            raise AssertionError("rank-local budget violation was accepted")
        assert comm.calls == 1 and torch.equal(output, previous)
        assert torch.equal(source, original)
        # A fresh valid call proves that the failed preparation left the group
        # usable, rather than merely letting both processes terminate.
        adapter(workspace)(output, source, dist.group.WORLD, dist.ReduceOp.SUM)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert comm.calls == 2
        dist.barrier()
        print(f"SPLIT_LAYOUT_GLOO_PASS rank={rank}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
