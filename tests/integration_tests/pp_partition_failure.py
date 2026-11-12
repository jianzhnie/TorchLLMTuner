"""Coordinate a rank-local PP ownership failure before mesh access or splitting.

Run with torchrun --standalone --nproc_per_node=2 -m
 tests.integration_tests.pp_partition_failure.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.config.parallel_config import ParallelConfig, PipelineParallelConfig
from llmtuner.parallel.pipeline_parallel.pipeline_parallel import apply_pp
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed()
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This check requires exactly two ranks")
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        if dist.get_rank() == 1:
            model.layers[1].weight = model.layers[0].weight
        parameters = tuple(model.parameters())
        values = tuple(parameter.detach().clone() for parameter in parameters)
        rng = torch.get_rng_state().clone()

        def forbidden_mesh(name):
            raise AssertionError("Invalid global ownership reached mesh access")

        topology = SimpleNamespace(
            pp=2, dp_replicate=1, dp_shard=1, get_mesh=forbidden_mesh
        )
        cfg = ParallelConfig(
            pipeline_parallel_size=2,
            pp_config=PipelineParallelConfig(
                num_pp_microbatches=2,
                module_fqns_per_model_part=[["layers.0"], ["layers.1"]],
            ),
        )
        try:
            apply_pp(
                model,
                parallel_dims=topology,
                cfg=cfg,
                device=device,
                global_batch_size=2,
                error_group=dist.group.WORLD,
            )
        except RuntimeError as error:
            expected = (
                "shared parameter"
                if dist.get_rank() == 1
                else "Pipeline partition validation failed on another rank"
            )
            if expected not in str(error):
                raise
        else:
            raise AssertionError("Invalid PP ownership was accepted")
        assert torch.equal(torch.get_rng_state(), rng)
        for current, original, value in zip(
            model.parameters(), parameters, values, strict=True
        ):
            assert current is original
            torch.testing.assert_close(current, value, rtol=0, atol=0)
        dist.barrier()
        if dist.get_rank() == 0:
            print("PP rank-local ownership failure coordinated before mesh/split")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
