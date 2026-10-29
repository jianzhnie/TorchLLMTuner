"""Joint checkpoint preflight must not involve unrelated pipeline ranks.

Run with torchrun --standalone --nproc-per-node=4.
"""

import torch.distributed as dist
from torch import nn

from llmtuner.parallel.expert_parallel.checkpointing import EP_GROUP_ATTR
from llmtuner.parallel.tensor_parallel.checkpointing import _coordinate_chunk_errors


def main():
    dist.init_process_group("gloo")
    try:
        group = dist.new_group([0, 1])
        rank = dist.get_rank()
        if rank in (0, 1):
            model = nn.Module()
            setattr(model, EP_GROUP_ATTR, group)
            expected = ValueError if rank == 0 else RuntimeError
            try:
                with _coordinate_chunk_errors(
                    [model], "Joint subset", include_expert_groups=True
                ):
                    if rank == 0:
                        raise ValueError("invalid local chunk")
            except expected:
                pass
            else:
                raise AssertionError("checkpoint failure was not propagated")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
