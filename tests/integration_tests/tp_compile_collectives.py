"""Two ranks: compiled TP collective gradients match analytic values.

Run with torchrun --nproc_per_node=2. Resolve capabilities before compilation,
as apply_tp does, and exercise the production wrappers with fullgraph capture.
"""

import os
from functools import partial

import torch
import torch.distributed as dist

from llmtuner.accelerator.linear_collectives import (
    all_gather_along,
    reduce_scatter_along,
    resolve_autograd_collectives,
)
from tests.integration_tests.harness import init_distributed


def gather_loss(inputs, axis=0):
    return all_gather_along(inputs, axis, dist.group.WORLD).square().sum()


def scatter_loss(inputs, axis=0):
    return reduce_scatter_along(inputs, axis, dist.group.WORLD).square().sum()


device = init_distributed()
try:
    assert dist.get_world_size() == 2
    resolve_autograd_collectives()
    compile_backend = os.environ.get("LLMTUNER_TEST_COMPILE_BACKEND", "aot_eager")
    # Rank-dependent, asymmetric values expose axis permutations that an
    # all-ones gradient check cannot detect. Compare outputs and nonlinear
    # backward results against eager on every supported axis.
    for operation in (all_gather_along, reduce_scatter_along):
        for axis in (0, 1, -2, -1):

            def forward(inputs, operation=operation, axis=axis):
                return operation(inputs, axis, dist.group.WORLD)

            compiled = torch.compile(forward, backend=compile_backend, fullgraph=True)
            source = (
                torch.arange(24, device=device, dtype=torch.float32).reshape(4, 6)
                + 31 * dist.get_rank()
            )
            eager_input = source.clone().requires_grad_()
            compiled_input = source.clone().requires_grad_()
            expected = forward(eager_input)
            actual = compiled(compiled_input)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.square().sum().backward()
            actual.square().sum().backward()
            torch.testing.assert_close(
                compiled_input.grad, eager_input.grad, rtol=0, atol=0
            )
    for operation, rows, gradient in ((gather_loss, 2, 4), (scatter_loss, 4, 4)):
        for axis in (0, 1, -1):
            eager = partial(operation, axis=axis)
            compiled = torch.compile(eager, backend=compile_backend, fullgraph=True)
            for forward in (eager, compiled):
                inputs = torch.ones(rows, 4, device=device, requires_grad=True)
                forward(inputs).backward()
                torch.testing.assert_close(
                    inputs.grad, torch.full_like(inputs, gradient), rtol=0, atol=0
                )
            print(
                f"rank {dist.get_rank()}: backend={compile_backend} "
                f"{operation.__name__} axis={axis} gradients exact",
                flush=True,
            )
finally:
    dist.destroy_process_group()
