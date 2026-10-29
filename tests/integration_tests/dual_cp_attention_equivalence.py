"""Dual CP collective forward/backward against independent dense SDPA.

Run: PYTHONPATH=. torchrun --nproc_per_node=4 \
    tests/integration_tests/dual_cp_attention_equivalence.py

CPU/Gloo float64 or accelerator float32, KV2/Ulysses2. Substitutes attention
with SDPA; actual SeqToHead, KV all-gather and HeadToSeq autograd run unchanged.
Includes GQA ratio 3 (Q6/KV2), yielding three local query heads after Ulysses.
Does not validate production flex masks or flex kernels. Set
LLMTUNER_TEST_DUAL_CP_BACKEND=sdpa to test DualCPSdpaKernel without substitution.
LLMTUNER_TEST_DUAL_CP_SLIDING_WINDOW optionally adds a sliding-window mask.
"""

import os
from itertools import product
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

from llmtuner.parallel.context_parallel.attention import DualCPFlexKernel
from llmtuner.parallel.context_parallel.sdpa_attention import DualCPSdpaKernel
from llmtuner.parallel.topology import ParallelTopology
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed()
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    tolerance = 1e-10 if device.type == "cpu" else 1e-5
    try:
        assert dist.get_world_size() == 4
        rank = dist.get_rank()
        topology = ParallelTopology(
            dp_replicate=1,
            dp_shard=1,
            cp=4,
            tp=1,
            pp=1,
            ep=1,
            world_size=4,
            cp_kv=2,
            cp_ulysses=2,
        )
        topology.build_device_mesh()
        backend = os.environ.get("LLMTUNER_TEST_DUAL_CP_BACKEND", "flex")
        if backend not in {"flex", "sdpa"}:
            raise ValueError(f"Unknown dual CP test backend: {backend}")
        window = os.environ.get("LLMTUNER_TEST_DUAL_CP_SLIDING_WINDOW")
        sliding_window = int(window) if window is not None else None
        kernel_type = DualCPSdpaKernel if backend == "sdpa" else DualCPFlexKernel
        kernel = kernel_type(
            kv_mesh=topology.get_optional_mesh("cp_kv"),
            ulysses_mesh=topology.get_optional_mesh("cp_ulysses"),
        )
        for scenario, (query_heads, kv_heads) in product(
            ("noncausal", "causal", "packed"), ((4, 4), (4, 2), (6, 2))
        ):
            torch.manual_seed(123)
            reference = [
                torch.randn(1, heads, 16, 4, dtype=dtype).to(device).requires_grad_()
                for heads in (query_heads, kv_heads, kv_heads)
            ]
            local = [
                tensor.detach()[:, :, rank * 4 : (rank + 1) * 4]
                .clone()
                .requires_grad_()
                for tensor in reference
            ]
            mask = torch.ones(16, 16, dtype=torch.bool)
            if scenario != "noncausal":
                mask = mask.tril()
            if scenario == "packed":
                documents = torch.repeat_interleave(
                    torch.arange(3), torch.tensor([5, 6, 5])
                )
                mask &= documents[:, None] == documents[None, :]
            mask = mask.to(device)
            expected_mask = mask
            if sliding_window is not None:
                indices = torch.arange(16, device=device)
                rows = indices[:, None]
                cols = indices[None, :]
                sliding = (cols <= rows) & (rows - cols < sliding_window)
                expected_mask = mask & sliding
            expected = F.scaled_dot_product_attention(
                *reference, attn_mask=expected_mask, enable_gqa=True
            )

            def attention(module, q, k, v, block_mask, kwargs, *, mask=expected_mask):
                query_start = rank // 2 * 8
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask[query_start : query_start + 8],
                    enable_gqa=True,
                )

            if backend == "sdpa":
                kernel.packed = scenario == "packed"
                supplied_mask = None if scenario == "causal" else mask[None, None]
                actual = kernel(
                    *local,
                    module=None,
                    block_mask=supplied_mask,
                    sliding_window=sliding_window,
                ).transpose(1, 2)
            else:
                with patch(
                    "llmtuner.parallel.context_parallel.attention.run_flex", attention
                ):
                    actual = kernel(*local, module=None).transpose(1, 2)
            torch.testing.assert_close(
                actual,
                expected[:, :, rank * 4 : (rank + 1) * 4],
                atol=tolerance,
                rtol=tolerance,
            )
            expected.square().sum().backward()
            actual.square().sum().backward()
            for actual_input, reference_input in zip(local, reference, strict=True):
                assert actual_input.grad is not None
                assert torch.isfinite(actual_input.grad).all().item()
                assert actual_input.grad.abs().max().item() > tolerance
                torch.testing.assert_close(
                    actual_input.grad,
                    reference_input.grad[:, :, rank * 4 : (rank + 1) * 4],
                    atol=tolerance,
                    rtol=tolerance,
                )
            if rank == 0:
                print(
                    f"PASS dual CP backend={backend} mask={scenario} "
                    f"Q_heads={query_heads} "
                    f"KV_heads={kv_heads} sliding_window={sliding_window}: "
                    f"output/QKV grads device={device} tol={tolerance:g}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
