"""CP compression window ownership vs complete-sequence windows, CPU/Gloo.

Run from the repository root:
    torchrun --nproc_per_node=2 \
        tests/integration_tests/compression_window_equivalence.py
This validates token exchange, not a complete compressor/attention integration.
"""

from tests.caps import require_env

require_env("spmd_types")

import torch
import torch.distributed as dist

from llmtuner.parallel.context_parallel.compression_windows import (
    compress_csa_windows,
    compress_hca_windows,
    exchange_compression_windows,
    select_csa_windows,
)


def run_case(length, window_size):
    size, rank = dist.get_world_size(), dist.get_rank()
    torch.manual_seed(47)
    full = torch.randn(2, length * size, 5, requires_grad=True)
    local = full.detach()[:, rank * length : (rank + 1) * length].clone()
    local.requires_grad_()
    actual, first = exchange_compression_windows(
        local, window_size=window_size, group=dist.group.WORLD
    )
    count = full.shape[1] // window_size
    expected = full[:, : count * window_size].reshape(2, count, window_size, 5)
    owned = (rank + 1) * length // window_size - first
    torch.testing.assert_close(
        actual, expected[:, first : first + owned], atol=0, rtol=0
    )
    # Weighted, nonlinear per-window calculation verifies inverse communication,
    # including dropped tail and ranks that receive no complete windows.
    weights = torch.arange(window_size, dtype=full.dtype)[None, None, :, None] + 1
    actual_loss = (actual.square() * weights).sum()
    reference_loss = (expected.square() * weights).sum()
    actual_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(
        local.grad, full.grad[:, rank * length : (rank + 1) * length], atol=0, rtol=0
    )
    print(
        f"rank {rank}: length={length}, window={window_size}, owned={owned} passed",
        flush=True,
    )


def run_hca_case(length):
    from copy import deepcopy

    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4HCACompressor,
    )

    config = DeepseekV4Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_groups=2,
        o_lora_rank=4,
        layer_types=["heavily_compressed_attention"],
        mlp_layer_types=["moe"],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 4,
        },
    )
    torch.manual_seed(47)
    model = DeepseekV4HCACompressor(config)
    with torch.no_grad():
        model.position_bias.normal_()
    reference = deepcopy(model)
    rank, size = dist.get_rank(), dist.get_world_size()
    full = torch.randn(2, length * size, 16, requires_grad=True)
    local = full.detach()[:, rank * length : (rank + 1) * length].clone()
    local.requires_grad_()
    actual, first = compress_hca_windows(model, local, group=dist.group.WORLD)
    positions = torch.arange(full.shape[1])[None].expand(2, -1)
    expected, _ = reference(full, torch.empty(0), positions, None, 0)
    torch.testing.assert_close(
        actual, expected[:, :, first : first + actual.shape[2]], atol=1e-6, rtol=1e-6
    )
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(
        local.grad,
        full.grad[:, rank * length : (rank + 1) * length],
        atol=2e-6,
        rtol=2e-6,
    )
    for (name, parameter), (_, wanted) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        dist.all_reduce(parameter.grad)
        torch.testing.assert_close(
            parameter.grad, wanted.grad, atol=3e-6, rtol=3e-6, msg=name
        )
    print(
        f"rank {rank}: real HCA length={length}, first_window={first} passed",
        flush=True,
    )


def run_csa_case(length):
    from copy import deepcopy
    from unittest.mock import patch

    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4CSACompressor,
    )

    config = DeepseekV4Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=8,
        partial_rotary_factor=0.5,
        q_lora_rank=8,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=8,
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["moe"],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 4,
        },
    )
    torch.manual_seed(71)
    model = DeepseekV4CSACompressor(config)
    with torch.no_grad():
        model.position_bias.normal_()
    reference = deepcopy(model)
    rank, size = dist.get_rank(), dist.get_world_size()
    full = torch.randn(2, length * size, 16, requires_grad=True)
    local = (
        full.detach()[:, rank * length : (rank + 1) * length].clone().requires_grad_()
    )
    actual, first = compress_csa_windows(model, local, group=dist.group.WORLD)
    positions = torch.arange(length * size)[None].expand(2, -1)
    # Exclude discrete indexer selection from this compression-only comparison.
    # HF's entire Ca/Cb compression and RoPE computation is unchanged.
    selection = torch.empty(2, length * size, 0, dtype=torch.long)
    with patch.object(reference.indexer, "forward", return_value=selection):
        expected, _ = reference(full, torch.empty(0), positions, None, 0)
    torch.testing.assert_close(
        actual, expected[:, :, first : first + actual.shape[2]], atol=1e-6, rtol=1e-6
    )
    upstream = torch.randn_like(expected)
    actual.backward(upstream[:, :, first : first + actual.shape[2]])
    if expected.requires_grad:
        expected.backward(upstream)
    else:
        full.grad = torch.zeros_like(full)
    torch.testing.assert_close(
        local.grad,
        full.grad[:, rank * length : (rank + 1) * length],
        atol=3e-6,
        rtol=3e-6,
    )
    for (name, p), (_, r) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if name.startswith("indexer."):
            assert p.grad is None and r.grad is None
            continue
        dist.all_reduce(p.grad)
        if r.grad is None:
            assert torch.count_nonzero(p.grad) == 0, name
            continue
        if not torch.allclose(p.grad, r.grad, atol=4e-6, rtol=4e-6):
            print(
                name,
                "maxdiff",
                (p.grad - r.grad).abs().max().item(),
                "reference max",
                r.grad.abs().max().item(),
                "finite",
                torch.isfinite(p.grad).all().item(),
                flush=True,
            )
        torch.testing.assert_close(p.grad, r.grad, atol=4e-6, rtol=4e-6, msg=name)
    print(f"rank {rank}: CSA overlap length={length} passed", flush=True)

    # Now exercise the real indexer; no replacement or mocked selection.
    with torch.no_grad():
        model.indexer.position_bias.normal_()
    query_residual = torch.randn(2, length * size, config.q_lora_rank)
    for topk in (1, 3):
        model.indexer.index_topk = topk
        wanted_indices = model.indexer(
            full.detach(), query_residual, positions, None, 0
        )
        actual_indices = select_csa_windows(
            model.indexer,
            local.detach(),
            query_residual[:, rank * length : (rank + 1) * length],
            positions[:, rank * length : (rank + 1) * length],
            group=dist.group.WORLD,
        )
        torch.testing.assert_close(
            actual_indices,
            wanted_indices[:, rank * length : (rank + 1) * length],
            atol=0,
            rtol=0,
        )
    print(f"rank {rank}: real CSA indexer length={length} passed", flush=True)


def main():
    dist.init_process_group("gloo")
    try:
        for length, window_size in ((3, 4), (4, 4), (5, 3), (2, 7), (1, 2), (5, 1)):
            run_case(length, window_size)
        for length in (3, 4, 5):
            run_hca_case(length)
        for length in (3, 4, 5, 1):
            run_csa_case(length)
        # Agree on metadata before data movement; do not let incompatible split
        # sizes or dtypes enter all-to-all on only a subset of ranks.
        for mismatch in ("length", "dtype"):
            rank = dist.get_rank()
            length = 3 + rank if mismatch == "length" else 3
            dtype = torch.float64 if mismatch == "dtype" and rank else torch.float32
            try:
                exchange_compression_windows(
                    torch.ones(1, length, 4, dtype=dtype),
                    window_size=4,
                    group=dist.group.WORLD,
                )
            except ValueError as error:
                assert "equal shard" in str(error)
            else:
                raise AssertionError("Incompatible shards must fail on every rank")
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
