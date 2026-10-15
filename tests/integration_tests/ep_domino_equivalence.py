"""Verify two-slice Domino EP forward and backward against local experts.

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/ep_domino_equivalence.py

Odd, unequal rank token counts and zero receives exercise split bookkeeping.
Routing scores and expert IDs use noncontiguous storage on both ranks.
CUDA: add --device cuda; use --trace-dir outputs/domino-traces to export
per-rank profiler traces. Inspect NCCL dispatch1 and expert0 kernel intervals
on separate streams; trace export alone does not prove overlap. Add
--nondefault-stream to exercise a caller stream different from the default.
"""

import argparse
import os
from contextlib import nullcontext
from itertools import product
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from llmtuner.models.common.moe.experts import (
    GroupedExperts,
    RoutedExperts,
)
from llmtuner.parallel.expert_parallel.domino_dispatch import DominoRoutedExperts
from llmtuner.parallel.expert_parallel.token_dispatch import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--nondefault-stream", action="store_true")
    args = parser.parse_args()
    if args.nondefault_stream and args.device != "cuda":
        parser.error("--nondefault-stream requires --device cuda")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requires CUDA devices and NCCL")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    profiler = None
    try:
        if args.trace_dir is not None:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(activities=activities, record_shapes=True)
            profiler.start()
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        caller_stream = (
            torch.cuda.Stream(device=device) if args.nondefault_stream else None
        )
        generator = torch.Generator().manual_seed(19)
        for routed_class, zero_receives, recompute, small_rank in product(
            (RoutedExperts, DominoRoutedExperts),
            (False, True),
            (False, True),
            (None, 0, 1),
        ):
            reference_experts = (
                GroupedExperts(8, 12, 4, use_grouped_mm=False).double().to(device)
            )
            with torch.no_grad():
                for parameter in reference_experts.parameters():
                    parameter.copy_(
                        torch.randn(
                            parameter.shape, generator=generator, dtype=torch.float64
                        )
                        * 0.1
                    )
            local_experts = (
                GroupedExperts(8, 12, 2, use_grouped_mm=False).double().to(device)
            )
            with torch.no_grad():
                for target, source in zip(
                    local_experts.parameters(),
                    reference_experts.parameters(),
                    strict=True,
                ):
                    target.copy_(source[rank * 2 : (rank + 1) * 2])
            reference = RoutedExperts(reference_experts, LocalTokenDispatcher(4, 2))
            dispatcher = AllToAllTokenDispatcher(4, 2)
            dispatcher.set_ep_group(ep_group=dist.group.WORLD)
            distributed_experts = routed_class(local_experts, dispatcher)
            calls = []
            if recompute:
                original_forward = distributed_experts.forward

                def counted_forward(
                    *args, original=original_forward, invocations=calls, **kwargs
                ):
                    invocations.append(True)
                    return original(*args, **kwargs)

                distributed_experts.forward = counted_forward
                distributed_experts = checkpoint_wrapper(distributed_experts)
            inputs = torch.randn(12, 8, generator=generator, dtype=torch.float64)
            scores = torch.softmax(
                torch.randn(12, 2, generator=generator, dtype=torch.float64), dim=-1
            )
            expert_ids = torch.tensor(
                [[0, 1], [2, 0], [1, 3], [0, 2], [3, 1], [2, 3]] * 2
            )
            if zero_receives:
                expert_ids = torch.tensor([[0, 1]] * 12)
            weights = torch.randn(12, 8, generator=generator, dtype=torch.float64)
            inputs, scores, expert_ids, weights = (
                tensor.to(device) for tensor in (inputs, scores, expert_ids, weights)
            )
            reference_inputs = inputs.clone().requires_grad_()
            reference_scores = scores.clone().requires_grad_()
            expected = reference(
                reference_inputs,
                reference_scores,
                expert_ids,
                torch.bincount(expert_ids.flatten(), minlength=4),
            )
            (expected * weights).sum().backward()
            token_slice = slice(0, 5) if rank == 0 else slice(5, 12)
            if small_rank is not None:
                token_slice = (
                    slice(0, small_rank) if rank == 0 else slice(small_rank, 12)
                )
            local_inputs = inputs[token_slice].clone().requires_grad_()
            local_scores = (
                scores[token_slice].T.contiguous().T.detach().requires_grad_()
            )
            local_ids = expert_ids[token_slice].T.contiguous().T
            if local_inputs.shape[0] > 1:
                assert (
                    not local_scores.is_contiguous() and not local_ids.is_contiguous()
                )
            local_counts = torch.bincount(local_ids.flatten(), minlength=4)
            if caller_stream is not None:
                default_stream = torch.cuda.current_stream(device)
                caller_stream.wait_stream(default_stream)
                for tensor in (
                    local_inputs,
                    local_scores,
                    local_ids,
                    local_counts,
                    weights,
                    *local_experts.parameters(),
                ):
                    tensor.record_stream(caller_stream)
            context = (
                torch.cuda.stream(caller_stream)
                if caller_stream is not None
                else nullcontext()
            )
            with context:
                actual = distributed_experts(
                    local_inputs,
                    local_scores,
                    local_ids,
                    local_counts,
                )
                (actual * weights[token_slice]).sum().backward()
            if caller_stream is not None:
                default_stream.wait_stream(caller_stream)
                for tensor in (
                    actual,
                    local_inputs.grad,
                    local_scores.grad,
                    *(parameter.grad for parameter in local_experts.parameters()),
                ):
                    tensor.record_stream(default_stream)
            if recompute:
                assert len(calls) == 2, "Checkpoint backward must actually recompute"
            torch.testing.assert_close(
                actual, expected[token_slice], rtol=1e-9, atol=1e-9
            )
            torch.testing.assert_close(
                local_inputs.grad,
                reference_inputs.grad[token_slice],
                rtol=1e-9,
                atol=1e-9,
            )
            torch.testing.assert_close(
                local_scores.grad,
                reference_scores.grad[token_slice],
                rtol=1e-9,
                atol=1e-9,
            )
            for got, want in zip(
                local_experts.parameters(), reference_experts.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    got.grad, want.grad[rank * 2 : (rank + 1) * 2], rtol=1e-9, atol=1e-9
                )
            if rank == 0:
                print(
                    f"{routed_class.__name__} zero_receives={zero_receives} "
                    f"recompute={recompute}: "
                    f"small_rank={small_rank}: "
                    f"device={device} nondefault_stream={args.nondefault_stream}: "
                    "forward and gradients match",
                    flush=True,
                )
    finally:
        if profiler is not None:
            profiler.stop()
            args.trace_dir.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(
                str(args.trace_dir / f"rank{dist.get_rank()}.json")
            )
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
