"""Check KV × Ulysses CP token ordering and mask wiring on four CPU ranks.

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/cp_dual_axis_equivalence.py

Uses production eager flex forward; CPU flex backward is unavailable.
"""

from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

from llmtuner.config import LLMTunerConfig, ModelConfig, ParallelConfig, TrainingConfig
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.context_parallel.attention import (
    DualCPFlexKernel,
    HeadToSeq,
    SeqToHead,
)
from llmtuner.parallel.context_parallel.context_parallel import apply_cp
from llmtuner.parallel.topology import ParallelTopology
from tests.integration_tests.cp_ulysses_varlen_equivalence import (
    _ref_forward,
    _uncompiled_create_attention_mask,
)


def check_collective_gradients(topology):
    """Dense SDPA substitutes only attention; production collectives keep backward."""
    group = topology.get_mesh("cp")
    rank = group.get_local_rank()
    gen = torch.Generator().manual_seed(9)
    full = [
        torch.randn(1, 4, 512, 8, generator=gen, dtype=torch.float64) for _ in range(3)
    ]
    weights = torch.randn(1, 4, 512, 8, generator=gen, dtype=torch.float64)
    ref = [x.clone().requires_grad_() for x in full]
    expected = F.scaled_dot_product_attention(*ref, is_causal=True)
    (expected * weights).sum().backward()
    local = [
        x[:, :, rank * 128 : (rank + 1) * 128].clone().requires_grad_() for x in full
    ]
    kv_rank = topology.get_mesh("cp_kv").get_local_rank()
    mask = torch.arange(512)[None, :] <= (torch.arange(256)[:, None] + kv_rank * 256)

    def dense(module, q, k, v, block_mask, kwargs):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    kernel = DualCPFlexKernel(
        kv_mesh=topology.get_mesh("cp_kv"),
        ulysses_mesh=topology.get_mesh("cp_ulysses"),
    )
    # Reproduce the former gather-before-exchange order: its key sequence is
    # permuted relative to the causal mask, so this reference must distinguish it.
    with torch.no_grad():
        old_k, old_v = kernel._flex_cp_allgather(
            local[1].contiguous(), local[2].contiguous(), 2, kernel._kv_pg_name
        )
        old_q = SeqToHead.apply(local[0], kernel._ulysses_group)
        old_k = SeqToHead.apply(old_k, kernel._ulysses_group)
        old_v = SeqToHead.apply(old_v, kernel._ulysses_group)
        old_out = HeadToSeq.apply(
            dense(None, old_q, old_k, old_v, None, {}).contiguous(),
            kernel._ulysses_group,
        )
        old_difference = (
            (old_out - expected[:, :, rank * 128 : (rank + 1) * 128]).abs().max()
        )
        dist.all_reduce(old_difference, op=dist.ReduceOp.MAX)
        assert old_difference > 1e-3
    with patch("llmtuner.parallel.context_parallel.attention.run_flex", dense):
        actual = kernel(*local, module=None).transpose(1, 2)
        (actual * weights[:, :, rank * 128 : (rank + 1) * 128]).sum().backward()
    torch.testing.assert_close(
        actual, expected[:, :, rank * 128 : (rank + 1) * 128], rtol=1e-9, atol=1e-9
    )
    for got, want in zip(local, ref, strict=True):
        assert got.grad.abs().max() > 0
        torch.testing.assert_close(
            got.grad,
            want.grad[:, :, rank * 128 : (rank + 1) * 128],
            rtol=1e-9,
            atol=1e-9,
        )
    if dist.get_rank() == 0:
        print("dual CP collective gradients match dense reference", flush=True)


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 4
        cfg = LLMTunerConfig(
            model=ModelConfig(
                model_name_or_path="qwen3",
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
            ),
            parallel=ParallelConfig(
                context_parallel_kv_allgather_size=2,
                context_parallel_ulysses_size=2,
                context_parallel_load_balancer=None,
            ),
            training=TrainingConfig(max_seq_len=512, steps=1),
        )
        topology = ParallelTopology.from_config(cfg.parallel, 4)
        topology.build_device_mesh()
        mesh = topology.get_mesh("cp")
        check_collective_gradients(topology)
        generator = torch.Generator().manual_seed(31)
        ids = torch.randint(128, (512,), generator=generator)
        labels = torch.randint(128, (512,), generator=generator)
        for packed in (False, True):
            positions = (
                torch.cat([torch.arange(n) for n in (73, 182, 257)])
                if packed
                else torch.arange(512)
            )
            torch.manual_seed(0)
            model = HFTransformerModel(build_model_config_for(cfg)).double().eval()
            torch.manual_seed(0)
            reference = HFTransformerModel(build_model_config_for(cfg)).double().eval()
            model.model.config._attn_implementation = "flex_torchtitan"
            if packed:
                model.model.config.attn_mask_type = "block_causal"
            apply_cp(model, mesh=mesh, cfg=cfg.parallel, parallel_dims=topology)
            with (
                patch(
                    "llmtuner.models.hf.model.create_attention_mask",
                    _uncompiled_create_attention_mask,
                ),
                torch.no_grad(),
            ):
                inputs, local_labels, extra = model.preprocess_inputs(
                    {"input": ids, "labels": labels, "positions": positions},
                    parallel_dims=topology,
                )
                output = model(inputs, **extra)
                expected = _ref_forward(reference, ids, positions, document=packed)
            mask = extra.get("attention_masks")
            if packed:
                assert tuple(mask.seq_lengths) == (256, 512)
            rank = topology.get_mesh("cp").get_local_rank()
            torch.testing.assert_close(local_labels, labels.chunk(4)[rank])
            parts = [torch.empty_like(output) for _ in range(4)]
            dist.all_gather(parts, output.contiguous())
            actual = torch.cat(parts)
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
            if dist.get_rank() == 0:
                difference = (actual - expected).abs().max().item()
                print(
                    f"dual CP packed={packed}: max diff {difference:.3e}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
