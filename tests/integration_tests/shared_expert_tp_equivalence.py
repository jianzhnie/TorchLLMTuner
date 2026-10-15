"""shared-expert x TP check: sharded shared expert equals the unsharded model.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \\
        tests/integration_tests/shared_expert_tp_equivalence.py

A tiny offline DeepSeek-V3 (with a shared expert) runs apply_tp at tp=2.
The shared expert shards featurewise with no collectives -- inside the MoE
sequence boundary the token stream is already gathered, so gate/up shard on
the output features and down on the input features, leaving a partial output
that the boundary reduce-scatter sums. The whole model's logits and the
replicated attention/shared-expert gradients must match the unsharded reference
computed in-process before apply_tp.

Environment note: needs torch >= 2.12 and a transformers that builds
deepseek_v3 offline. Verified on torch 2.14 with two CPU/Gloo ranks;
GPU/NPU execution is not covered.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig

from llmtuner.config import ParallelConfig
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.tensor_parallel import apply_tp
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.trainer import Trainer

VOCAB = 128
TOKENS = 64
TOL = 1e-9
# HF DeepSeek RMSNorm computes in float32 even for float64 model weights;
# backward summation order across token shards therefore has fp32 noise.
GRAD_TOL = 1e-6


def _config() -> AutoConfig:
    return AutoConfig.for_model(
        "deepseek_v3",
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=3,
        num_key_value_heads=3,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        max_position_embeddings=256,
        experts_implementation="eager",
    )


def _model() -> HFTransformerModel:
    torch.manual_seed(0)
    return HFTransformerModel(_config()).to(torch.float64).eval()


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    parallel = ParallelConfig(tensor_parallel_size=world)
    dims = ParallelTopology.from_config(parallel, world)
    dims.build_device_mesh()
    tp_mesh = dims.get_mesh("tp")

    model = _model()
    ref = _model()

    ids = torch.randint(VOCAB, (TOKENS,), generator=torch.Generator().manual_seed(7))
    positions = torch.arange(TOKENS)
    with torch.no_grad():
        ref_logits = ref(ids, positions=positions)

    apply_tp(model, tp_mesh, parallel)

    failures: list[str] = []

    # Non-vacuity: the shared expert's weights are halved on every rank.
    sparse_layers = [
        (index, layer)
        for index, layer in enumerate(model.layers)
        if isinstance(getattr(layer.mlp, "shared_experts", None), torch.nn.Module)
    ]
    if not sparse_layers:
        failures.append("no shared expert found to shard -- check is vacuous")
    for index, layer in sparse_layers:
        gate = layer.mlp.shared_experts.gate_proj
        ref_gate = ref.layers[index].mlp.shared_experts.gate_proj
        if gate.weight.shape[0] * world != ref_gate.weight.shape[0]:
            failures.append(
                f"rank {rank}: shared gate rows {gate.weight.shape[0]}, "
                f"reference {ref_gate.weight.shape[0]} -- not sharded"
            )

    with torch.no_grad():
        inputs, _, extra = model.preprocess_inputs(
            {"input": ids, "labels": ids}, parallel_dims=dims
        )
        assert inputs.numel() == TOKENS // world
        local = model(inputs, **extra)
        full = torch.cat([torch.empty_like(local) for _ in range(world)], dim=-2)
        dist.all_gather_into_tensor(full, local, group=tp_mesh.get_group())
    diff = (full - ref_logits).abs().max().item()
    if diff > TOL:
        failures.append(f"rank {rank}: tp-joined logits diff {diff:.3e}")

    # The replicated attention boundary must also preserve backward semantics.
    F.cross_entropy(ref(ids, positions=positions), ids, reduction="sum").backward()
    local_labels = ids.chunk(world)[rank]
    F.cross_entropy(model(inputs, **extra), local_labels, reduction="sum").backward()
    trainer = Trainer.__new__(Trainer)
    trainer.parallel_dims = dims
    trainer.model_parts = [model]
    trainer._allreduce_replicated_tp_grads()
    grad_diff = 0.0
    for index, layer in enumerate(model.layers):
        reference = ref.layers[index]
        for name, parameter in layer.self_attn.named_parameters():
            expected = reference.self_attn.get_parameter(name).grad
            assert parameter.grad is not None and expected is not None
            grad_diff = max(grad_diff, (parameter.grad - expected).abs().max().item())
        if hasattr(layer.mlp, "shared_experts"):
            for name, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                gradient = getattr(layer.mlp.shared_experts, name).weight.grad
                expected = getattr(reference.mlp.shared_experts, name).weight.grad
                assert gradient is not None and expected is not None
                expected = expected.chunk(world, dim=dim)[rank]
                grad_diff = max(grad_diff, (gradient - expected).abs().max().item())
    if grad_diff > GRAD_TOL:
        failures.append(
            f"rank {rank}: attention/shared-expert gradient diff {grad_diff:.3e}"
        )

    if rank == 0:
        print(f"shared-expert tp={world} tokens={TOKENS} tol={TOL:.0e}")
        print(f"tp-joined vs reference max diff = {diff:.3e}")
        print(f"attention/shared-expert gradient max diff = {grad_diff:.3e}")
        for f in failures:
            print(f"  FAIL {f}")
        print("all checks passed" if not failures else "CHECKS FAILED")
    verdict = torch.tensor(len(failures), dtype=torch.int64)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    assert int(verdict) == 0, f"{int(verdict)} check(s) failed -- see rank 0 output"


if __name__ == "__main__":
    main()
