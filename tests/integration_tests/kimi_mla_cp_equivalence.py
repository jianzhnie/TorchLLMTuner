"""KCP MLA Ulysses branch vs full-sequence eager MLA, CPU/Gloo FP32.

Run with torchrun --nproc_per_node=2. Does not execute KDA Ascend kernels.
"""

from copy import deepcopy

from tests.caps import require_env

require_env("spmd_types")

import torch
import torch.distributed as dist

from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
from llmtuner.models.kimi.latent_attention import KimiLatentAttention
from llmtuner.parallel.context_parallel.kcp import kcp_kda_forward, kcp_mla_attention


def run_case(query_rank, gate, padding, additive):
    torch.manual_seed(42)
    config = KimiLinearConfig(
        hidden_size=12,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=query_rank,
        kv_lora_rank=4,
        qk_nope_head_dim=3,
        qk_rope_head_dim=1,
        v_head_dim=3,
        mla_use_nope=True,
        mla_use_output_gate=gate,
    )
    config._attn_implementation = "eager"
    model = KimiLatentAttention(config, 0)
    reference = deepcopy(model)
    model._kcp_attention = kcp_mla_attention(dist.group.WORLD)
    rank, size = dist.get_rank(), dist.get_world_size()
    local_length, length = 3, 3 * size
    inputs = torch.randn(1, length, 12)
    gradient = torch.randn_like(inputs)
    valid = torch.ones(1, length, dtype=torch.bool)
    if padding:
        valid[:, 0] = False
        valid[:, -1] = False
    full_mask = torch.ones(length, length, dtype=torch.bool).tril()[None, None]
    full_mask = full_mask & valid[:, None, None, :]
    source = inputs.clone().requires_grad_()
    shard = slice(rank * local_length, (rank + 1) * local_length)
    local = inputs[:, shard].clone().requires_grad_()
    local_mask = torch.ones(local_length, local_length, dtype=torch.bool).tril()
    local_mask = local_mask[None, None] & valid[:, None, None, shard]
    if additive:
        full_mask = torch.zeros_like(full_mask, dtype=inputs.dtype).masked_fill(
            ~full_mask, float("-inf")
        )
        local_mask = torch.zeros_like(local_mask, dtype=inputs.dtype).masked_fill(
            ~local_mask, float("-inf")
        )
    expected = reference(source, attention_mask=full_mask)
    actual = model(local, attention_mask=local_mask)
    torch.testing.assert_close(actual, expected[:, shard], atol=1e-6, rtol=1e-6)
    actual.backward(gradient[:, shard])
    expected.backward(gradient)
    torch.testing.assert_close(local.grad, source.grad[:, shard], atol=2e-6, rtol=2e-6)
    for (name, parameter), (_, wanted) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        dist.all_reduce(parameter.grad)
        torch.testing.assert_close(
            parameter.grad, wanted.grad, atol=3e-6, rtol=3e-6, msg=name
        )
    print(
        f"rank {rank}: MLA q_lora={query_rank}, gate={gate}, "
        f"padding={padding}, additive={additive} passed",
        flush=True,
    )


def main():
    dist.init_process_group("gloo")
    try:
        for query_rank in (None, 4):
            for gate in (False, True):
                for padding in (False, True):
                    for additive in (False, True):
                        run_case(query_rank, gate, padding, additive)
        # A mask error on one rank must be shared before peers enter all-to-all.
        query = torch.randn(1, 4, 3, 4)
        mask = torch.ones(1, 1, 3, 3, dtype=torch.bool).tril()
        if dist.get_rank() == 0:
            mask[..., 0, 2] = True
        try:
            kcp_mla_attention(dist.group.WORLD)(
                None, query, query, query, mask, scaling=0.5
            )
        except (ValueError, RuntimeError) as error:
            assert "causal padding" in str(error) or "another rank" in str(error)
        else:
            raise AssertionError("Invalid mask must fail on every CP rank")
        dist.barrier()
        print(
            f"rank {dist.get_rank()}: rank-local invalid mask coordinated", flush=True
        )
        from llmtuner.models.kimi.delta_attention import KimiDeltaAttention

        config = KimiLinearConfig(
            hidden_size=8,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [],
                "head_dim": 4,
                "num_heads": 2,
                "short_conv_kernel_size": 3,
            },
        )
        module = KimiDeltaAttention(config, 0)
        mask = torch.ones(1, 3, dtype=torch.bool)
        if dist.get_rank() == 0:
            mask[0, 1] = False
        try:
            kcp_kda_forward(type(module).forward, dist.group.WORLD)(
                module, torch.randn(1, 3, 8), attention_mask=mask
            )
        except (ValueError, RuntimeError) as error:
            assert "compacted" in str(error) or "another rank" in str(error)
        else:
            raise AssertionError("KDA padding must fail on every CP rank")
        dist.barrier()
        print(
            f"rank {dist.get_rank()}: KDA padding coordinated before kernels",
            flush=True,
        )
        try:
            kcp_kda_forward(type(module).forward, dist.group.WORLD)(
                module, torch.randn(1, 3 + dist.get_rank(), 8)
            )
        except ValueError as error:
            assert "equal local sequence lengths" in str(error)
        else:
            raise AssertionError("Unequal KDA shards must fail before kernels")
        dist.barrier()
        print(
            f"rank {dist.get_rank()}: unequal KDA shards rejected before kernels",
            flush=True,
        )
        # Locally valid modules can still disagree on state-message geometry.
        if dist.get_rank() == 0:
            for name in ("q", "k", "v"):
                convolution = getattr(module, f"{name}_conv1d")
                convolution.weight = torch.nn.Parameter(
                    torch.randn(*convolution.weight.shape[:-1], 4)
                )
        try:
            kcp_kda_forward(type(module).forward, dist.group.WORLD)(
                module, torch.randn(1, 3, 8)
            )
        except ValueError as error:
            assert "matching convolution kernel size" in str(error)
        else:
            raise AssertionError(
                "Different KDA state geometry must fail before kernels"
            )
        dist.barrier()
        print(f"rank {dist.get_rank()}: KDA geometry mismatch rejected", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
