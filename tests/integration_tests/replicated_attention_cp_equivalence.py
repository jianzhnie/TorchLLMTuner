"""Replicated MLA attention under TP=2 and Ulysses CP=2.

Run: PYTHONPATH=. torchrun --nproc_per_node=4
    tests/integration_tests/replicated_attention_cp_equivalence.py

Checks full-model forward on CPU/Gloo. Two attention heads suffice because
TP shards only MLPs; Ulysses splits the two replicated heads. CPU flex does
not support backward, so this script does not establish CP gradient equivalence.
"""

import torch
import torch.distributed as dist
from transformers import AutoConfig

from llmtuner.config import ParallelConfig
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.context_parallel import apply_cp
from llmtuner.parallel.tensor_parallel import apply_tp
from llmtuner.parallel.topology import ParallelTopology


def build_model():
    torch.manual_seed(0)
    config = AutoConfig.for_model(
        "deepseek_v3",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=32,
        n_group=1,
        topk_group=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        first_k_dense_replace=1,
        max_position_embeddings=512,
        experts_implementation="eager",
    )
    return HFTransformerModel(config).to(torch.float64).eval()


def main():
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    cfg = ParallelConfig(
        tensor_parallel_size=2,
        context_parallel_ulysses_size=2,
        context_parallel_strategy="ulysses",
        context_parallel_load_balancer=None,
    )
    dims = ParallelTopology.from_config(cfg, 4)
    dims.build_device_mesh()
    model, reference = build_model(), build_model()
    apply_tp(model, dims.spmd_dense_mesh(), cfg)
    model.model.config._attn_implementation = "flex_torchtitan"
    apply_cp(model, dims.spmd_dense_mesh(), cfg, parallel_dims=dims)
    ids = torch.randint(128, (256,), generator=torch.Generator().manual_seed(7))
    inputs, _, extra = model.preprocess_inputs(
        {"input": ids, "labels": ids}, parallel_dims=dims
    )
    assert inputs.numel() == 64
    with torch.no_grad():
        logits = model(inputs, **extra)
        expected = reference(ids)
    offset = dims.get_mesh("cp").get_local_rank() * 128
    offset += dims.get_mesh("tp").get_local_rank() * 64
    difference = (logits - expected[offset : offset + 64]).abs().max()
    dist.all_reduce(difference, op=dist.ReduceOp.MAX)
    # HF DeepSeek RMSNorm uses fp32 internally; SDPA and eager flex use
    # different arithmetic orders, so float64 weights do not remove fp32 noise.
    assert difference.item() < 1e-7, difference.item()
    if dist.get_rank() == 0:
        print(f"replicated MLA TP=2 CP=2 logits max diff = {difference.item():.3e}")
        print("all checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
