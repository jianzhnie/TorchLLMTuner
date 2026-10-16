"""Two CPU/Gloo ranks: compiled TP collective gradients match analytic values.

Run with torchrun --nproc_per_node=2. Resolve capabilities before compilation,
as apply_tp does, and exercise the production wrappers with fullgraph capture.
"""

from datetime import timedelta

import torch
import torch.distributed as dist

from llmtuner.parallel.tensor_parallel.linear_ops import (
    all_gather_along,
    functional_collectives,
    reduce_scatter_along,
)


def gather_loss(inputs):
    return all_gather_along(inputs, 0, dist.group.WORLD).square().sum()


def scatter_loss(inputs):
    return reduce_scatter_along(inputs, 0, dist.group.WORLD).square().sum()


dist.init_process_group("gloo", timeout=timedelta(seconds=30))
try:
    assert dist.get_world_size() == 2
    functional_collectives()
    for operation, rows, gradient in ((gather_loss, 2, 4), (scatter_loss, 4, 4)):
        compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
        for forward in (operation, compiled):
            inputs = torch.ones(rows, 3, requires_grad=True)
            forward(inputs).backward()
            torch.testing.assert_close(
                inputs.grad, torch.full_like(inputs, gradient), rtol=0, atol=0
            )
        print(
            f"rank {dist.get_rank()}: {operation.__name__} gradients exact", flush=True
        )
finally:
    dist.destroy_process_group()
