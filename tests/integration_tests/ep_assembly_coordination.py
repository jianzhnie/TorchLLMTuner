"""One invalid EP assembly rank must stop its successful peer before training.

Run with two ranks:
    LLMTUNER_TEST_DEVICE=cpu PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
        tests/integration_tests/ep_assembly_coordination.py
"""

import torch
import torch.distributed as dist

from llmtuner.config import ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.parallel.expert_parallel import expert_parallel
from tests.integration_tests.harness import init_distributed


def main() -> None:
    init_distributed()
    rank = dist.get_rank()
    cfg = ParallelConfig(expert_parallel_size=2)
    if rank == 1:
        # Keep the peer's local conversion successful without introducing model
        # compute into a test whose subject is the real error collective.
        expert_parallel.swap_moe_blocks = lambda *args, **kwargs: 0
    try:
        expert_parallel.apply_ep(torch.nn.Linear(2, 2), cfg, ep_group=dist.group.WORLD)
    except Exception as error:
        if rank == 0:
            assert isinstance(error, UnsupportedCombinationError), error
        else:
            assert isinstance(error, RuntimeError), error
            assert "EP assembly validation failed on another rank" in str(error)
    else:
        raise AssertionError(f"rank {rank} continued after EP assembly failure")
    dist.barrier()
    if rank == 0:
        print("EP assembly failure coordination passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
