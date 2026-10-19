# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Ascend fused EP communication/GEMM and its explicit autograd inverses.

Adapted from FSDPTurbo ops/npu/grouped_matmul_mc2.py. Shared experts remain
outside these operators in llmtuner's MoE layer. torch_npu is loaded on use.
"""

from __future__ import annotations

import importlib

import torch
import torch.distributed as dist

from llmtuner.errors import EnvironmentUnsupportedError


def load_mc2_ops():
    """Require the real fused operators; never substitute unfused collectives."""
    try:
        ops = importlib.import_module("torch_npu")
    except ImportError as exc:
        raise ImportError(
            "MC2 requires torch_npu matching your PyTorch and Ascend runtime"
        ) from exc
    required = ("npu_alltoallv_gmm", "npu_gmm_alltoallv", "npu_grouped_matmul")
    missing = [name for name in required if not callable(getattr(ops, name, None))]
    if missing:
        raise EnvironmentUnsupportedError(
            "MC2 requires torch_npu fused operators: " + ", ".join(missing)
        )
    return ops


def mc2_communication_context(group: dist.ProcessGroup, device: torch.device):
    """Resolve the HCCL name using the group's global rank, as upstream does."""
    if device.type != "npu":
        raise EnvironmentUnsupportedError("MC2 requires NPU tensors and an HCCL group")
    ops = load_mc2_ops()
    rank = dist.get_global_rank(group, dist.get_rank(group))
    backend = group._get_backend(device)
    get_name = getattr(backend, "get_hccl_comm_name", None)
    if not callable(get_name):
        raise EnvironmentUnsupportedError("MC2 requires HCCL get_hccl_comm_name")
    return ops, get_name(rank), dist.get_world_size(group)


def _expert_counts(counts: list[int], world_size: int, device: torch.device):
    # Wire counts are rank-major; grouped weight gradients need expert totals.
    return (
        torch.tensor(counts, dtype=torch.int64, device=device)
        .reshape(world_size, -1)
        .sum(0)
    )


class AllToAllGroupedMatmul(torch.autograd.Function):
    """Fuse dispatch and projection; backward fuses its inverse input gradient."""

    @staticmethod
    def forward(
        ctx, inputs, weights, ops, comm_name, world_size, send_counts, recv_counts
    ):
        output, _, permuted_input = ops.npu_alltoallv_gmm(
            inputs,
            weights,
            comm_name,
            world_size,
            send_counts,
            recv_counts,
            trans_gmm_weight=True,
            permute_out_flag=True,
        )
        ctx.save_for_backward(weights, permuted_input)
        ctx.ops = ops
        ctx.comm_name = comm_name
        ctx.world_size = world_size
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weights, permuted_input = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, _ = ctx.ops.npu_gmm_alltoallv(
            grad_output,
            weights,
            ctx.comm_name,
            ctx.world_size,
            ctx.recv_counts,
            ctx.send_counts,
        )
        grad_weights = ctx.ops.npu_grouped_matmul(
            [grad_output.T],
            [permuted_input],
            bias=None,
            group_list=_expert_counts(ctx.recv_counts, ctx.world_size, weights.device),
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        return grad_input, grad_weights, None, None, None, None, None


class GroupedMatmulAllToAll(torch.autograd.Function):
    """Fuse projection and combine; backward dispatches output gradients."""

    @staticmethod
    def forward(
        ctx, inputs, weights, ops, comm_name, world_size, send_counts, recv_counts
    ):
        output, _ = ops.npu_gmm_alltoallv(
            inputs,
            weights,
            comm_name,
            world_size,
            send_counts,
            recv_counts,
            trans_gmm_weight=True,
        )
        ctx.save_for_backward(inputs, weights)
        ctx.ops = ops
        ctx.comm_name = comm_name
        ctx.world_size = world_size
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        grad_input, _, permuted_gradient = ctx.ops.npu_alltoallv_gmm(
            grad_output.contiguous(),
            weights,
            ctx.comm_name,
            ctx.world_size,
            ctx.recv_counts,
            ctx.send_counts,
            permute_out_flag=True,
        )
        grad_weights = ctx.ops.npu_grouped_matmul(
            [permuted_gradient.T],
            [inputs],
            bias=None,
            group_list=_expert_counts(ctx.send_counts, ctx.world_size, weights.device),
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        return grad_input, grad_weights, None, None, None, None, None
