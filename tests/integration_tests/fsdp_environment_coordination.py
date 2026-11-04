"""Coordinate rank-local custom FSDP import failures after mesh construction."""

import argparse
import importlib
from dataclasses import replace

import torch.distributed as dist
from torch.distributed.tensor.debug import CommDebugMode

from llmtuner.config import ParallelConfig
from llmtuner.config.parallel_config import FSDPConfig
from llmtuner.errors import EnvironmentUnsupportedError
from llmtuner.parallel.topology import ParallelTopology
from tests.integration_tests.harness import init_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp-size", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    init_distributed()
    module = importlib.import_module(
        "llmtuner.parallel.fully_shard_parallel.fully_shard_parallel"
    )
    original = module.load_custom_fsdp_engine
    try:
        world_size = dist.get_world_size()
        assert world_size == 2 * args.pp_size
        rank = dist.get_rank()
        cfg = ParallelConfig(
            data_parallel_shard_size=2,
            pipeline_parallel_size=args.pp_size,
            fsdp_config=FSDPConfig(fsdp_implementation="custom"),
        )
        dims = ParallelTopology.from_config(cfg, world_size=world_size)
        dims.build_device_mesh()
        failure = EnvironmentUnsupportedError("injected rank-local FSDP import failure")

        def resolve():
            if rank == 0:
                raise failure
            return None

        # Fault injection only: this test does not execute the private engine.
        module.load_custom_fsdp_engine = resolve
        caught = None
        try:
            module.validate_fsdp_environment(cfg, dims)
        except (EnvironmentUnsupportedError, RuntimeError) as error:
            caught = error
        outcomes = [None] * world_size
        dist.all_gather_object(outcomes, (type(caught).__name__, str(caught)))
        assert outcomes[0][0] == "EnvironmentUnsupportedError", outcomes
        for kind, message in outcomes[1:]:
            assert kind == "RuntimeError", outcomes
            assert "failed on another rank" in message, outcomes
        if rank == 0:
            assert caught is failure
        module.load_custom_fsdp_engine = lambda: None
        module.validate_fsdp_environment(cfg, dims)
        # Native assembly must not acquire the optional import or a new
        # coordination collective, even with active DP and pipeline meshes.
        module.load_custom_fsdp_engine = resolve
        native = replace(cfg, fsdp_config=FSDPConfig(fsdp_implementation="native"))
        with CommDebugMode() as debug:
            module.validate_fsdp_environment(native, dims)
        assert debug.get_total_counts() == 0, debug.get_comm_counts()
        dist.barrier()
        print(f"PASS custom FSDP environment failure coordination/retry rank={rank}")
    finally:
        module.load_custom_fsdp_engine = original
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
