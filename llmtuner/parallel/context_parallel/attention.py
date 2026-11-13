"""Differentiable sequence/head redistribution shared by CP backends."""

from __future__ import annotations

import torch
import torch.distributed as dist

_SEQ_DIM = 2
_HEAD_DIM = 1


def cp_all_to_all(
    x_BHSD: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """One CP all-to-all: split ``scatter_dim`` across the group, join ``gather_dim``.

    Chunk ``r`` of the scatter dim goes to rank ``r``; the pieces that come
    back are concatenated along the gather dim in rank order. With
    ``scatter_dim=1, gather_dim=2`` this is the ulysses token-to-head swap
    ``(b, h, s/cp, d) -> (b, h/cp, s, d)``; swapping the two dims is the exact
    inverse, which is also each direction's backward.
    """
    cp = group.size()
    shape = x_BHSD.shape
    if shape[scatter_dim] % cp != 0:
        raise ValueError(
            f"Ulysses all-to-all scatters dim {scatter_dim} of size "
            f"{shape[scatter_dim]} across cp={cp} ranks, which does not divide "
            "evenly. For the head axis this means the model's head count must "
            "be a multiple of the CP degree."
        )
    split = (
        shape[:scatter_dim] + (cp, shape[scatter_dim] // cp) + shape[scatter_dim + 1 :]
    )
    # Chunk r of the scatter dim is bound for rank r, so it leads.
    x = x_BHSD.reshape(split).movedim(scatter_dim, 0).contiguous()
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=group)
    # y[r] is rank r's piece; merge the group axis into the gather dim so the
    # pieces concatenate in rank order.
    y = y.movedim(0, gather_dim)
    merged = (
        y.shape[:gather_dim]
        + (y.shape[gather_dim] * y.shape[gather_dim + 1],)
        + y.shape[gather_dim + 2 :]
    )
    return y.reshape(merged)


class SeqToHead(torch.autograd.Function):
    """``(b, h, s/cp, d) -> (b, h/cp, s, d)``; the backward is the inverse swap."""

    @staticmethod
    def forward(ctx, x_BHSD, group):
        ctx.group = group
        return cp_all_to_all(x_BHSD, group, scatter_dim=_HEAD_DIM, gather_dim=_SEQ_DIM)

    @staticmethod
    def backward(ctx, grad_BHSD):
        return (
            cp_all_to_all(
                grad_BHSD, ctx.group, scatter_dim=_SEQ_DIM, gather_dim=_HEAD_DIM
            ),
            None,
        )


class HeadToSeq(torch.autograd.Function):
    """``(b, h/cp, s, d) -> (b, h, s/cp, d)``; the backward is the inverse swap."""

    @staticmethod
    def forward(ctx, x_BHSD, group):
        ctx.group = group
        return cp_all_to_all(x_BHSD, group, scatter_dim=_SEQ_DIM, gather_dim=_HEAD_DIM)

    @staticmethod
    def backward(ctx, grad_BHSD):
        return (
            cp_all_to_all(
                grad_BHSD, ctx.group, scatter_dim=_HEAD_DIM, gather_dim=_SEQ_DIM
            ),
            None,
        )
