"""Kimi all-to-all EP against independent full per-expert computation.

Run: torchrun --nproc_per_node=2 tests/integration_tests/kimi_ep_equivalence.py
CPU/Gloo, FP32, TP/PP/CP1; covers latent/shared/SituGLU and uneven/empty tokens.
"""

from copy import deepcopy

from tests.caps import require_env

require_env("spmd_types")

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
from llmtuner.models.kimi.moe import KimiSparseMoE
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks


def run_case(*, latent, activation, empty):
    torch.manual_seed(17)
    config = KimiLinearConfig(
        hidden_size=8,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=12,
        num_shared_experts=1,
        num_expert_group=2,
        topk_group=1,
        hidden_act=activation,
        routed_expert_hidden_size=4 if latent else None,
        latent_moe_use_norm=latent,
    )
    model = nn.Module()
    layer = nn.Module()
    source = KimiSparseMoE(config).eval()
    reference = deepcopy(source)
    layer.block_sparse_moe = source
    model.layers = nn.ModuleList([layer])
    swap_hf_moe_blocks(model, ep_group=dist.group.WORLD)
    converted = layer.block_sparse_moe
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(91 + rank)
    count = 0 if empty and rank == 0 else 3 + rank * 2
    inputs = torch.randn(1, count, 8, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    actual, expected = converted(inputs), reference(other)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(inputs.grad, other.grad, atol=2e-6, rtol=2e-6)
    # Each EP shard sees expert assignments from every input rank. Sum the
    # independent reference gradients before selecting this rank's expert slice.
    for parameter in reference.experts.parameters():
        dist.all_reduce(parameter.grad)
    local = config.num_experts // size
    for source_name, target_name in (
        ("w1", "w1_EFD"),
        ("w2", "w2_EDF"),
        ("w3", "w3_EFD"),
    ):
        full = torch.stack(
            [getattr(expert, source_name).weight.grad for expert in reference.experts]
        )
        actual_grad = getattr(converted.routed_experts.inner_experts, target_name).grad
        torch.testing.assert_close(
            actual_grad, full[rank * local : (rank + 1) * local], atol=2e-6, rtol=2e-6
        )
    # Replicated parameters have rank-local gradients before outer FSDP/DP
    # reduction; compare them to the rank-local reference, including latent ops.
    torch.testing.assert_close(
        converted.router.gate.weight.grad, reference.gate.weight.grad
    )
    for path in (
        "shared_experts",
        "routed_expert_down_proj",
        "routed_expert_norm",
        "routed_expert_up_proj",
    ):
        original = getattr(reference, path, None)
        target = getattr(
            converted if path == "shared_experts" else converted.routed_experts,
            path,
            None,
        )
        if original is not None:
            for left, right in zip(
                target.parameters(), original.parameters(), strict=True
            ):
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-6)
    # Export full EP weights to the original individual-expert schema, then
    # reconstruct another independently converted/sharded model on each rank.
    state = ModelCheckpointState(model).state_dict()
    adapter = HFTransformerStateDictAdapter(config, None, kimi_expert_parallel=True)
    exported = adapter.to_hf({f"model.{key}": value for key, value in state.items()})
    reference_state = {
        f"layers.0.block_sparse_moe.{key}": value
        for key, value in reference.state_dict().items()
    }
    assert exported.keys() == reference_state.keys()
    for key in exported:
        torch.testing.assert_close(exported[key], reference_state[key], atol=0, rtol=0)
    fresh = nn.Module()
    fresh_layer = nn.Module()
    fresh_layer.block_sparse_moe = KimiSparseMoE(config).eval()
    fresh.layers = nn.ModuleList([fresh_layer])
    swap_hf_moe_blocks(fresh, ep_group=dist.group.WORLD)
    imported = {
        key.removeprefix("model."): value
        for key, value in adapter.from_hf(exported).items()
    }
    ModelCheckpointState(fresh).load_state_dict(imported)
    torch.testing.assert_close(
        fresh_layer.block_sparse_moe(inputs), actual, atol=0, rtol=0
    )
    dist.barrier()
    if rank == 0:
        print(
            f"passed latent={latent} activation={activation} empty_rank0={empty}",
            flush=True,
        )


if __name__ == "__main__":
    dist.init_process_group("gloo")
    try:
        for latent in (False, True):
            for activation in ("silu", "situ"):
                for empty in (False, True):
                    run_case(latent=latent, activation=activation, empty=empty)
    finally:
        dist.destroy_process_group()
