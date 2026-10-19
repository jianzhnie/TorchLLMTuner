"""Two Gloo ranks: positive/negative TP axes, batched masks and autograd."""

import torch
import torch.distributed as dist

from llmtuner.parallel.tensor_parallel.linear_ops import (
    all_gather_along,
    reduce_scatter_along,
)


def main():
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        sources = [torch.arange(6).reshape(2, 3).double() + 10 * i for i in range(2)]
        for axis in (0, 1, -2, -1):
            canonical = axis if axis >= 0 else axis + 2
            x = sources[rank].clone().requires_grad_()
            actual = all_gather_along(x, axis, dist.group.WORLD)
            expected = torch.cat(sources, dim=canonical)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            actual.sum().backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, 2), atol=0, rtol=0)
            masks = [source.remainder(3) == 0 for source in sources]
            mask = all_gather_along(masks[rank], axis, dist.group.WORLD)
            torch.testing.assert_close(
                mask, torch.cat(masks, canonical), atol=0, rtol=0
            )

            inputs = [
                torch.arange(16).reshape(4, 4).double() + 10 * i for i in range(2)
            ]
            x = inputs[rank].clone().requires_grad_()
            actual = reduce_scatter_along(x, axis, dist.group.WORLD)
            expected = sum(inputs).chunk(2, canonical)[rank]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            actual.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)
        if rank == 0:
            print("TP positive/negative axes, batched masks and gradients match.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
