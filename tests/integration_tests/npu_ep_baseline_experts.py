"""Score-aware EP output and VJP against real baseline GMM on 2/4/8 ranks."""

import argparse
import importlib
import os

import torch
import torch.distributed as dist

from llmtuner.parallel.expert_parallel.npu_baseline_experts import (
    NpuBaselineAllToAllTokenDispatcher,
    NpuBaselineGroupedExperts,
    NpuBaselineRoutedExperts,
)
from tests.caps import require_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--baseline-dispatch", action="store_true")
    parser.add_argument("--empty-owner", action="store_true")
    parser.add_argument("--random-routing", action="store_true")
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument("--empty-origin", action="store_true")
    origin.add_argument("--uneven-origin", action="store_true")
    parser.add_argument("--top-k", type=int, choices=(2, 8), default=2)
    args = parser.parse_args()
    require_env("torch_npu")
    importlib.import_module("torchtitan_npu")
    from torchtitan_npu.converters.kernels.gmm import _run_experts_grouped_mm

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        world = dist.get_world_size()
        assert world in (2, 4, 8)
        rank = dist.get_rank()
        for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
            backend.allow_hf32 = False
        torch.manual_seed(37)
        experts = 16 if args.top_k == 8 else 4
        assert experts % world == 0, "expert count must divide evenly across ranks"
        tokens = 64 if args.random_routing else 16
        weights = [
            (
                torch.randn(experts, 128, 128, device="npu", dtype=torch.bfloat16) * 0.1
            ).requires_grad_()
            for _ in range(3)
        ]
        block = NpuBaselineGroupedExperts(128, 128, experts // world).to(
            device="npu", dtype=torch.bfloat16
        )
        with torch.no_grad():
            for parameter, full in zip(block.parameters(), weights, strict=True):
                parameter.copy_(full.chunk(world)[rank])
        dispatcher = NpuBaselineAllToAllTokenDispatcher(experts, args.top_k)
        dispatcher.set_ep_group(ep_group=dist.group.WORLD)
        model = NpuBaselineRoutedExperts(block, dispatcher)
        torch.manual_seed(91 + rank)
        inputs = torch.randn(tokens, 128, device="npu", dtype=torch.bfloat16)
        scores = torch.rand(tokens, args.top_k, device="npu")
        ids = torch.tensor([[0, 2], [1, 3]], device="npu").repeat(8, 1)
        if args.random_routing or args.top_k != 2:
            ids = (
                torch.rand(tokens, experts, device="npu")
                .topk(args.top_k, dim=1)
                .indices
            )
        if args.empty_owner:
            ids = (
                torch.rand(tokens, experts // 2, device="npu")
                .topk(args.top_k, dim=1)
                .indices
            )
        counts = torch.nn.functional.one_hot(ids, experts).sum((0, 1))
        global_inputs = [torch.empty_like(inputs) for _ in range(world)]
        global_scores = [torch.empty_like(scores) for _ in range(world)]
        dist.all_gather(global_inputs, inputs)
        dist.all_gather(global_scores, scores)
        global_expert_ids = [torch.empty_like(ids) for _ in range(world)]
        dist.all_gather(global_expert_ids, ids)
        sizes = [tokens] * world
        if args.empty_origin:
            sizes[-1] = 0
        if args.uneven_origin:
            sizes = [max(1, tokens - i * 7) for i in range(world)]
        global_inputs = [
            value[:size] for value, size in zip(global_inputs, sizes, strict=True)
        ]
        global_scores = [
            value[:size] for value, size in zip(global_scores, sizes, strict=True)
        ]
        global_expert_ids = [
            value[:size] for value, size in zip(global_expert_ids, sizes, strict=True)
        ]
        inputs, scores, ids = (
            inputs[: sizes[rank]],
            scores[: sizes[rank]],
            ids[: sizes[rank]],
        )
        counts = torch.nn.functional.one_hot(ids, experts).sum((0, 1))
        x = torch.cat(global_inputs).requires_grad_()
        score = torch.cat(global_scores).requires_grad_()
        global_ids = torch.cat(global_expert_ids)
        order = torch.argsort(global_ids.flatten(), stable=True)
        routed = x[order // args.top_k]
        routed_scores = score.flatten()[order, None]
        w1, w3, w2 = weights
        expected = (
            _run_experts_grouped_mm(
                torch.cat((w1, w3), dim=1),
                w2,
                None,
                routed,
                torch.nn.functional.one_hot(global_ids, experts).sum((0, 1)),
                routed_scores=routed_scores,
            )[torch.argsort(order)]
            .reshape(sum(sizes), args.top_k, 128)
            .sum(1)
        )
        local_x = inputs.clone().requires_grad_()
        local_scores = scores.clone().requires_grad_()
        if args.checkpoint:
            from torch.utils.checkpoint import checkpoint

            observed = checkpoint(
                model, local_x, local_scores, ids, counts, use_reentrant=False
            )
        else:
            observed = model(local_x, local_scores, ids, counts)
        torch.manual_seed(53)
        probe = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(expected, (x, score, *weights), probe)
        actual_grads = torch.autograd.grad(
            observed,
            (local_x, local_scores, *block.parameters()),
            probe.split(sizes)[rank],
        )
        torch.testing.assert_close(
            observed, expected.split(sizes)[rank], atol=0, rtol=0
        )
        for actual, reference in zip(actual_grads[:2], expected_grads[:2], strict=True):
            torch.testing.assert_close(
                actual, reference.split(sizes)[rank], atol=0, rtol=0
            )
        for actual, reference in zip(actual_grads[2:], expected_grads[2:], strict=True):
            torch.testing.assert_close(
                actual, reference.chunk(world)[rank], atol=0, rtol=0
            )
        if args.baseline_dispatch:
            from types import SimpleNamespace

            import torch_npu
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor.parallel import parallelize_module
            from torchtitan.models.common.moe import GroupedExperts
            from torchtitan_npu.converters.kernels.gmm import NpuGroupedExpertConverter
            from torchtitan_npu.converters.kernels.moe_dispatch import NpuExpertParallel

            root = torch.nn.Module()
            root.experts = GroupedExperts.Config(
                dim=128, hidden_dim=128, num_experts=experts
            ).build()
            root.to(device="npu", dtype=torch.bfloat16)
            NpuGroupedExpertConverter(SimpleNamespace(name="qwen3")).convert(root)
            with torch.no_grad():
                root.experts.w13.copy_(torch.cat((w1, w3), dim=1))
                root.experts.w2.copy_(w2)
            mesh = init_device_mesh("npu", (world,), mesh_dim_names=("ep",))
            parallelize_module(root.experts, mesh, NpuExpertParallel())
            baseline_x = inputs.clone().requires_grad_()
            baseline_scores = scores.clone().requires_grad_()
            routed_x, indices = torch_npu.npu_moe_token_permute(baseline_x, ids)
            routed_scores, _ = torch_npu.npu_moe_token_permute(
                baseline_scores.flatten()[:, None], ids.reshape(-1, 1)
            )
            baseline_output = root.experts(routed_x, counts, routed_scores)
            baseline_output = (
                torch_npu.npu_moe_token_unpermute(baseline_output, indices, None)
                .reshape(sizes[rank], args.top_k, 128)
                .sum(1)
            )
            baseline_grads = torch.autograd.grad(
                baseline_output,
                (baseline_x, baseline_scores, root.experts.w13, root.experts.w2),
                probe.split(sizes)[rank],
            )
            torch.testing.assert_close(observed, baseline_output, atol=0, rtol=0)
            for actual, reference in zip(
                actual_grads[:2], baseline_grads[:2], strict=True
            ):
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            packed_grad = baseline_grads[2].to_local()
            gate_grad, up_grad = packed_grad.chunk(2, dim=1)
            down_grad = baseline_grads[3].to_local()
            for actual, reference in zip(
                actual_grads[2:], (gate_grad, up_grad, down_grad), strict=True
            ):
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            print(f"PASS rank={rank} actual baseline EP hooks output and all VJPs")
        print(f"PASS rank={rank} score-aware EP baseline output and all VJPs exact")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
