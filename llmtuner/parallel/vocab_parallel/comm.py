"""Autograd-covered variable-size all-to-all primitives."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single_autograd

__all__ = ["alltoallv", "alltoallv_with_gradient_divide_factor"]


def alltoallv(
    input_: torch.Tensor,
    group: dist.ProcessGroup | None,
    input_splits: Sequence[int],
    output_splits: Sequence[int],
) -> torch.Tensor:
    """Exchange variable-sized leading-dimension chunks over ``group``."""
    if group is None or dist.get_world_size(group) == 1:
        return input_
    return all_to_all_single_autograd(
        input_.contiguous(), list(output_splits), list(input_splits), group
    )


def alltoallv_with_gradient_divide_factor(
    input_: torch.Tensor,
    group: dist.ProcessGroup | None,
    input_splits: Sequence[int],
    output_splits: Sequence[int],
    gradient_divide_factor: float,
) -> torch.Tensor:
    """Exchange chunks and divide their incoming gradient by a positive factor."""
    output = alltoallv(input_, group, input_splits, output_splits)
    if gradient_divide_factor != 1.0 and output.requires_grad:
        output.register_hook(lambda grad: grad / gradient_divide_factor)
    return output
