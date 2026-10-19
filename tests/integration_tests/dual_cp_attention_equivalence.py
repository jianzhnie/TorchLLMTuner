"""Dual CP collective forward/backward against independent dense SDPA.

Run: PYTHONPATH=. torchrun --nproc_per_node=4 \
    tests/integration_tests/dual_cp_attention_equivalence.py

CPU/Gloo float64, KV2/Ulysses2. Substitutes only the attention computation
with SDPA; actual SeqToHead, KV all-gather and HeadToSeq autograd run unchanged.
Includes GQA ratio 3 (Q6/KV2), yielding three local query heads after Ulysses.
Does not validate production flex masks or target-device kernels.
"""

from itertools import product
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

from llmtuner.parallel.context_parallel.attention import DualCPFlexKernel
from llmtuner.parallel.topology import ParallelTopology


def main() -> None:
    dist.init_process_group("gloo")
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
        kernel = DualCPFlexKernel(
            kv_mesh=topology.get_optional_mesh("cp_kv"),
            ulysses_mesh=topology.get_optional_mesh("cp_ulysses"),
        )
        for scenario, (query_heads, kv_heads) in product(
            ("noncausal", "causal", "packed"), ((4, 4), (4, 2), (6, 2))
        ):
            torch.manual_seed(123)
            reference = [
                torch.randn(1, heads, 16, 4, dtype=torch.float64, requires_grad=True)
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
            expected = F.scaled_dot_product_attention(
                *reference, attn_mask=mask, enable_gqa=True
            )

            def attention(module, q, k, v, block_mask, kwargs, *, mask=mask):
                query_start = rank // 2 * 8
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask[query_start : query_start + 8],
                    enable_gqa=True,
                )

            with patch(
                "llmtuner.parallel.context_parallel.attention.run_flex", attention
            ):
                actual = kernel(*local, module=None).transpose(1, 2)
            torch.testing.assert_close(
                actual,
                expected[:, :, rank * 4 : (rank + 1) * 4],
                atol=1e-10,
                rtol=1e-10,
            )
            expected.square().sum().backward()
            actual.square().sum().backward()
            for actual_input, reference_input in zip(local, reference, strict=True):
                torch.testing.assert_close(
                    actual_input.grad,
                    reference_input.grad[:, :, rank * 4 : (rank + 1) * 4],
                    atol=1e-10,
                    rtol=1e-10,
                )
            if rank == 0:
                print(
                    f"PASS dual CP mask={scenario} Q_heads={query_heads} "
                    f"KV_heads={kv_heads}: "
                    "output/QKV grads",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
