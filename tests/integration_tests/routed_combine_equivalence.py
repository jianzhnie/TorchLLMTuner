"""Check portable routed combine numerics, VJP and deterministic replay.

Run as a module on the available accelerator, or with LLMTUNER_TEST_DEVICE=cpu.
The independent reference gathers expert-sorted rows back into token/top-k order
and sums slots; it does not use scatter or index_add.
"""

from itertools import product

import torch

from llmtuner.models.qwen.ops import unpermute
from llmtuner.parallel.expert_parallel.token_dispatch import (
    LocalDispatchMetadata,
    LocalTokenDispatcher,
)
from tests.integration_tests.harness import device


def check_case(dev, dtype, top_k, deterministic, tokens):
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(91)
    hidden = 32
    ids = torch.randint(0, 16, (tokens, top_k), device=dev)
    order = ids.flatten().argsort(stable=True)
    inverse = order.argsort()
    values = torch.randn(tokens * top_k, hidden, device=dev, dtype=dtype) * 0.1
    scores = torch.rand(tokens, top_k, device=dev, dtype=torch.float32)
    probe = torch.randn(tokens, hidden, device=dev, dtype=dtype)
    # Native low-precision accumulation and reductions may round differently.
    tolerance = 2e-3 if dtype == torch.bfloat16 else 2e-6
    for implementation in ("qwen", "dispatcher", "dispatcher_fp32"):
        previous = None
        for _ in range(3):
            routed = values.detach().clone().requires_grad_()
            probabilities = scores.detach().clone().requires_grad_()
            if implementation == "qwen":
                actual = unpermute(routed, order, probs=probabilities)
                weighted = routed * probabilities.flatten()[order, None].to(dtype)
            else:
                dispatcher = LocalTokenDispatcher(16, top_k)
                dispatcher.accumulate_in_fp32 = implementation == "dispatcher_fp32"
                metadata = LocalDispatchMetadata(
                    token_indices_experts_sorted_N=order // top_k,
                    topk_scores_experts_sorted_N=probabilities.flatten()[order],
                )
                actual = dispatcher.combine(
                    routed, metadata, values.new_empty(tokens, hidden)
                )
                weighted = routed.float() * probabilities.flatten()[order, None]
                if not dispatcher.accumulate_in_fp32:
                    weighted = weighted.to(dtype)
            expected = (
                weighted[inverse]
                .reshape(tokens, top_k, hidden)
                .float()
                .sum(1)
                .to(dtype)
            )
            actual_grad = torch.autograd.grad(
                actual, (routed, probabilities), probe, retain_graph=True
            )
            expected_grad = torch.autograd.grad(
                expected, (routed, probabilities), probe
            )
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)
            for result, reference in zip(actual_grad, expected_grad, strict=True):
                torch.testing.assert_close(result, reference, atol=tolerance, rtol=0)
            replay = (actual.detach(), *actual_grad)
            if deterministic and previous is not None:
                for result, reference in zip(replay, previous, strict=True):
                    torch.testing.assert_close(result, reference, atol=0, rtol=0)
            previous = replay
        print(
            f"PASS {implementation} {dev.type} {dtype} k={top_k} "
            f"tokens={tokens} deterministic={deterministic}",
            flush=True,
        )


def main():
    dev = device()
    if dev.type != "cpu":
        torch.get_device_module(dev).set_device(dev)
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for dtype, top_k, deterministic, tokens in product(
            (torch.float32, torch.bfloat16), (2, 8), (False, True), (0, 64)
        ):
            check_case(dev, dtype, top_k, deterministic, tokens)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


if __name__ == "__main__":
    main()
