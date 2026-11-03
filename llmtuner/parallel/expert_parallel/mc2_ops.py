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


def _counts_matrix(counts: list[int], world_size: int, num_experts: int):
    return torch.tensor(counts, dtype=torch.int64).reshape(world_size, num_experts)


def _expert_major_to_source_major(tensor: torch.Tensor, counts: torch.Tensor):
    expert_segments = tensor.split(counts.sum(0).tolist())
    source_segments: list[list[torch.Tensor]] = [[] for _ in range(counts.shape[0])]
    for expert, segment in enumerate(expert_segments):
        for source, piece in enumerate(segment.split(counts[:, expert].tolist())):
            source_segments[source].append(piece)
    return torch.cat([piece for row in source_segments for piece in row])


def _source_major_to_expert_major(tensor: torch.Tensor, counts: torch.Tensor):
    source_segments = tensor.split(counts.sum(1).tolist())
    expert_segments: list[list[torch.Tensor]] = [[] for _ in range(counts.shape[1])]
    for source, segment in enumerate(source_segments):
        for expert, piece in enumerate(segment.split(counts[source].tolist())):
            expert_segments[expert].append(piece)
    return torch.cat([piece for row in expert_segments for piece in row])


def _grouped_projection_backward(grad_output, weights, inputs, counts):
    expert_counts = counts.sum(0).tolist()
    output_segments = grad_output.split(expert_counts)
    input_segments = inputs.split(expert_counts)
    grad_inputs, grad_weights = [], []
    for weight, output_piece, input_piece in zip(
        weights, output_segments, input_segments, strict=True
    ):
        grad_inputs.append(output_piece @ weight)
        grad_weights.append(output_piece.T @ input_piece)
    grad_input = (
        torch.cat(grad_inputs)
        if grad_inputs
        else grad_output.new_empty((0, weights.shape[-1]))
    )
    return grad_input, torch.stack(grad_weights)


def _dispatch_projection_backward(
    grad_output, weights, permuted_input, send_counts, recv_counts, world_size, group
):
    send = _counts_matrix(send_counts, world_size, weights.shape[0])
    recv = _counts_matrix(recv_counts, world_size, weights.shape[0])
    grad_input, grad_weights = _grouped_projection_backward(
        grad_output, weights, permuted_input, recv
    )
    source_major = _expert_major_to_source_major(grad_input, recv)
    if world_size == 1 or not dist.is_initialized():
        return source_major, grad_weights
    result = source_major.new_empty((sum(send_counts), source_major.shape[-1]))
    dist.all_to_all_single(
        result,
        source_major.contiguous(),
        output_split_sizes=send.sum(1).tolist(),
        input_split_sizes=recv.sum(1).tolist(),
        group=group,
    )
    return result, grad_weights


def _combine_projection_backward(
    grad_output, weights, inputs, send_counts, recv_counts, world_size, group
):
    send = _counts_matrix(send_counts, world_size, weights.shape[0])
    recv = _counts_matrix(recv_counts, world_size, weights.shape[0])
    if world_size == 1 or not dist.is_initialized():
        expert_major = _source_major_to_expert_major(grad_output, recv)
    else:
        received = grad_output.new_empty((sum(send_counts), grad_output.shape[-1]))
        dist.all_to_all_single(
            received,
            grad_output.contiguous(),
            output_split_sizes=send.sum(1).tolist(),
            input_split_sizes=recv.sum(1).tolist(),
            group=group,
        )
        expert_major = _source_major_to_expert_major(received, send)
    return _grouped_projection_backward(expert_major, weights, inputs, send)


class AllToAllGroupedMatmul(torch.autograd.Function):
    """Fuse dispatch and projection; backward fuses its inverse input gradient."""

    @staticmethod
    def forward(
        ctx,
        inputs,
        weights,
        ops,
        comm_name,
        world_size,
        send_counts,
        recv_counts,
        group=None,
        use_reference_backward=False,
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
        ctx.group = group
        ctx.use_reference_backward = use_reference_backward
        ctx.num_inputs = len(ctx.needs_input_grad)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weights, permuted_input = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if ctx.use_reference_backward or (
            ctx.world_size == 1 and any(count == 0 for count in ctx.recv_counts)
        ):
            grad_input, grad_weights = _dispatch_projection_backward(
                grad_output,
                weights,
                permuted_input,
                ctx.send_counts,
                ctx.recv_counts,
                ctx.world_size,
                ctx.group,
            )
            return (grad_input, grad_weights) + (None,) * (ctx.num_inputs - 2)
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
        return (grad_input, grad_weights) + (None,) * (ctx.num_inputs - 2)


class GroupedMatmulAllToAll(torch.autograd.Function):
    """Fuse projection and combine; backward dispatches output gradients."""

    @staticmethod
    def forward(
        ctx,
        inputs,
        weights,
        ops,
        comm_name,
        world_size,
        send_counts,
        recv_counts,
        group=None,
        use_reference_backward=False,
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
        ctx.group = group
        ctx.use_reference_backward = use_reference_backward
        ctx.num_inputs = len(ctx.needs_input_grad)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        if ctx.use_reference_backward or (
            ctx.world_size == 1 and any(count == 0 for count in ctx.send_counts)
        ):
            grad_input, grad_weights = _combine_projection_backward(
                grad_output.contiguous(),
                weights,
                inputs,
                ctx.send_counts,
                ctx.recv_counts,
                ctx.world_size,
                ctx.group,
            )
            return (grad_input, grad_weights) + (None,) * (ctx.num_inputs - 2)
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
        return (grad_input, grad_weights) + (None,) * (ctx.num_inputs - 2)
