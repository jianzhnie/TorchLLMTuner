"""Compare non-empty fused MC2 with its empty-group fallback on Ascend."""

import os
import time

import torch
import torch.distributed as dist

from llmtuner.parallel.expert_parallel.mc2_dispatch import (
    MC2GroupedExperts,
    MC2RoutedExperts,
    mc2_dispatch_metrics,
    reset_mc2_dispatch_metrics,
)
from llmtuner.parallel.expert_parallel.mc2_ops import load_mc2_ops
from llmtuner.parallel.expert_parallel.token_dispatch import AllToAllTokenDispatcher


def main() -> None:
    ops = load_mc2_ops()
    ops.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", ops.npu.current_device())
    dist.init_process_group("hccl")
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        num_experts = world_size * 2
        num_tokens = world_size * 64
        model = MC2GroupedExperts(128, 256, 2).to(dtype=torch.bfloat16, device=device)
        dispatcher = AllToAllTokenDispatcher(num_experts, 2)
        dispatcher.set_ep_group(ep_group=dist.group.WORLD)
        model = MC2RoutedExperts(model, dispatcher)
        inputs = torch.randn(num_tokens, 128, dtype=torch.bfloat16, device=device)
        scores = torch.softmax(torch.randn(num_tokens, 2, device=device), -1)
        ids = torch.arange(num_tokens * 2, device=device).reshape(-1, 2) % num_experts
        cotangent = torch.randn_like(inputs)
        for fused in (True, False):
            local_ids = (
                ids if fused else torch.tensor([0, 1], device=device).expand_as(ids)
            )
            counts = torch.bincount(local_ids.flatten(), minlength=num_experts)
            reset_mc2_dispatch_metrics()
            durations = []
            for step in range(25):
                model.zero_grad()
                if step == 5:
                    torch.npu.synchronize()
                    started = time.perf_counter()
                output = model(inputs, scores, local_ids, counts)
                (output * cotangent).float().sum().backward()
                torch.npu.synchronize()
                if step >= 5:
                    durations.append(time.perf_counter() - started)
                    started = time.perf_counter()
            metrics = mc2_dispatch_metrics()
            expected = "fused" if fused else "fallback"
            assert metrics[expected] == 25, metrics
            elapsed = sum(durations) / len(durations)
            print(
                f"MC2 {expected}: {elapsed * 1000:.3f} ms/step rank={rank}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
