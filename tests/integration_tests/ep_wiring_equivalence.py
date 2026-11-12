"""EP wiring check: ep=2 through ``apply_ep`` must match the explicit model.

Run under torchrun with 2 ranks:

    torchrun --nproc_per_node=2 -m tests.integration_tests.ep_wiring_equivalence

Where ``ep_equivalence.py`` checks the dispatcher arithmetic on hand-built
MoEs, this checks the WIRING end to end: a real (tiny, offline, randomly
initialized) qwen3_moe goes through ``apply_ep`` with a 2-rank EP group,
and each rank's forward over its own token shard must reproduce
(a) the unmodified explicit model's forward and (b) the EP=1 swap's forward.

Non-vacuity: under EP=2 each rank must hold only half the experts, and the
slice must be THIS rank's slice -- rank 1 holding experts 0-3 instead of 4-7
is exactly the wiring bug the weight check catches.

The EP=2 model also compares all-to-all backward and one SGD update to an
independent full-batch reference under CPU/Gloo. The aux-loss
injection (a forward-time accumulation plus a backward-time gradient) is
exercised on the EP=1 swap, where the backward is plain local compute.

Everything runs in float64; routing is fp32 in both implementations (see
tests/test_ep_model_replacement.py), so the tolerance sits above the fp32 routing noise.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.models.common.moe.layer import MoE
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel.expert_parallel import apply_ep
from llmtuner.parallel.expert_parallel.model_replacement import swap_moe_blocks

NUM_EXPERTS = 8
TOKENS = 40
TOL = 1e-6


def _config() -> AutoConfig:
    cfg = AutoConfig.for_model(
        "qwen3_moe",
        vocab_size=128,
        hidden_size=64,
        head_dim=16,
        intermediate_size=128,
        moe_intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=256,
    )
    # transformers 5.x defaults MoE experts to its grouped_mm integration, which
    # this torch build does not support in float64 -- the reference model here
    # runs in fp64, so force the eager expert path.
    cfg._experts_implementation = "eager"
    return cfg


def _model(seed: int = 0) -> Qwen3MoeForCausalLM:
    """Deterministically initialized tiny qwen3_moe; identical on every rank."""
    torch.manual_seed(seed)
    return Qwen3MoeForCausalLM(_config()).to(torch.float64).eval()


def _data(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Each rank owns its own token shard -- that is what makes EP a form of
    # parallelism (see tests/ep_equivalence.py for why sharing the shard
    # would double-count).
    g = torch.Generator().manual_seed(1000 + rank)
    ids = torch.randint(128, (TOKENS,), generator=g)
    return ids, torch.arange(TOKENS)


def _ref_gate_up(block) -> torch.Tensor:
    """Read the explicit model's declared packed expert storage."""
    return block.experts.gate_up_proj


def check_distributed_update(
    ref: Qwen3MoeForCausalLM,
    parallel: Qwen3MoeForCausalLM,
    ids: torch.Tensor,
    positions: torch.Tensor,
    group: dist.ProcessGroup,
    failures: list[str],
) -> tuple[float, float]:
    """Compare real all-to-all backward and one update to a global batch.

    EP owns expert slices; other gradients need a SUM across token owners.
    The reference keeps rank inputs in separate batch rows, so attention does
    not connect the last token of one rank to the first token of another.
    """
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    rows = [torch.empty_like(ids) for _ in range(world)]
    dist.all_gather(rows, ids, group=group)
    reference_logits = ref(
        torch.stack(rows), position_ids=positions.expand(world, -1)
    ).logits
    parallel_logits = parallel(ids, positions=positions).logits
    (reference_logits.square().sum() / (world * TOKENS)).backward()
    (parallel_logits.square().sum() / (world * TOKENS)).backward()
    expected = dict(ref.named_parameters())
    max_gradient_diff = 0.0
    max_update_diff = 0.0
    nonzero_expert_gradients = 0
    nonzero_router_gradients = 0
    update_pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, parameter in parallel.named_parameters():
        assert parameter.grad is not None, name
        if ".routed_experts.inner_experts." in name:
            prefix, role = name.split("routed_experts.inner_experts.")
            assert role in ("w1_EFD", "w3_EFD", "w2_EDF"), name
            packed = role in ("w1_EFD", "w3_EFD")
            reference = expected[
                prefix + "experts." + ("gate_up_proj" if packed else "down_proj")
            ]
            value = reference.grad
            assert value is not None, name
            nonzero_expert_gradients += int(parameter.grad.count_nonzero())
            if packed:
                half = 0 if role == "w1_EFD" else 1
                value = value.chunk(2, dim=1)[half]
            wanted_gradient = value.chunk(world, dim=0)[rank]
            wanted_weight = reference.detach()
            if packed:
                wanted_weight = wanted_weight.chunk(2, dim=1)[half]
            wanted_weight = wanted_weight.chunk(world, dim=0)[rank]
        else:
            dist.all_reduce(parameter.grad, group=group)
            reference = expected[name.replace(".mlp.router.gate.", ".mlp.gate.")]
            assert reference.grad is not None, name
            wanted_gradient = reference.grad
            wanted_weight = reference.detach()
            if ".mlp.router.gate." in name:
                nonzero_router_gradients += int(parameter.grad.count_nonzero())
        difference = (parameter.grad - wanted_gradient).abs().max().item()
        max_gradient_diff = max(max_gradient_diff, difference)
        if not torch.isfinite(parameter.grad).all() or difference > TOL:
            failures.append(f"rank {rank}: {name} gradient diff {difference:.3e}")
        update_pairs.append((name, parameter, wanted_weight))
    # Use the real optimizer on both independently constructed models. Views
    # retained above follow the reference's in-place parameter update.
    torch.optim.SGD(ref.parameters(), lr=0.05, foreach=False).step()
    torch.optim.SGD(parallel.parameters(), lr=0.05, foreach=False).step()
    for name, parameter, wanted_weight in update_pairs:
        difference = (parameter - wanted_weight).abs().max().item()
        max_update_diff = max(max_update_diff, difference)
        if not torch.isfinite(parameter).all() or difference > TOL:
            failures.append(f"rank {rank}: {name} update diff {difference:.3e}")
    assert nonzero_expert_gradients > 0
    assert nonzero_router_gradients > 0
    return max_gradient_diff, max_update_diff


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    ep_mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("ep",))
    ep_group = ep_mesh.get_group()
    failures: list[str] = []

    ref = _model()
    ep1 = _model()
    swap_moe_blocks(ep1)
    ep2 = _model()
    apply_ep(ep2, ParallelConfig(expert_parallel_size=2), ep_group=ep_group)

    ids, positions = _data(rank)
    with torch.no_grad():
        ref_logits = ref(ids, positions=positions).logits
        ep1_logits = ep1(ids, positions=positions).logits
        ep2_logits = ep2(ids, positions=positions).logits

    diff_hf = (ep2_logits - ref_logits).abs().max().item()
    diff_ep1 = (ep2_logits - ep1_logits).abs().max().item()
    if diff_hf > TOL:
        failures.append(f"rank {rank}: EP=2 vs explicit diff {diff_hf:.3e}")
    if diff_ep1 > TOL:
        failures.append(f"rank {rank}: EP=2 vs EP=1 diff {diff_ep1:.3e}")

    # Non-vacuity: each rank holds only its own half of the experts, and the
    # slice matches the reference weights for exactly those experts.
    num_local = NUM_EXPERTS // world
    for layer_idx, (layer_ep2, layer_ref) in enumerate(
        zip(ep2.layers, ref.layers, strict=True)
    ):
        moe = layer_ep2.mlp
        assert isinstance(moe, MoE)
        grouped = moe.routed_experts.inner_experts
        if grouped.num_experts != num_local:
            failures.append(
                f"rank {rank} layer {layer_idx}: holds {grouped.num_experts} "
                f"experts, want {num_local} -- EP did not shard"
            )
        lo = rank * num_local
        # transformers 5.x keeps all experts fused in gate_up_proj (E, 2F, D);
        # 4.x keeps one module per expert. Read in HF's own terms either way,
        # so the comparison stays anchored on what HF computes with.
        gate_up = _ref_gate_up(layer_ref.mlp)
        f = gate_up.shape[1] // 2
        for e in range(num_local):
            hf_w = gate_up[lo + e, :f]
            if not torch.equal(grouped.w1_EFD[e], hf_w):
                failures.append(
                    f"rank {rank} layer {layer_idx}: local expert {e} is not "
                    f"global expert {lo + e}"
                )

    gradient_diff, update_diff = check_distributed_update(
        ref, ep2, ids, positions, ep_group, failures
    )

    # Aux loss under training mode: the metric accumulates in the forward and
    # the gradient reaches the router on backward. Run on the EP=1 swap: what
    # is pinned here is the injection, which is dispatcher-independent.
    AuxLoss.set_step_denominator(torch.tensor(float(TOKENS - 1)))
    ep1.train()
    ep1(ids, positions=positions).logits.sum().backward()
    aux = ep1.layers[0].mlp.router.aux_loss
    if aux.instance_acc.item() <= 0:
        failures.append(f"rank {rank}: aux loss metric not accumulated")
    gate_grad = ep1.layers[0].mlp.router.gate.weight.grad
    if gate_grad is None or gate_grad.abs().max().item() == 0:
        failures.append(f"rank {rank}: no gradient reached the router")

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    differences = torch.tensor([diff_hf, diff_ep1, gradient_diff, update_diff])
    dist.all_reduce(differences, op=dist.ReduceOp.MAX)
    diff_hf, diff_ep1, gradient_diff, update_diff = differences.tolist()
    if rank == 0:
        print(f"tokens={TOKENS} experts={NUM_EXPERTS} ep_size={world} tol={TOL:.0e}")
        print(f"EP=2 vs explicit  max abs diff = {diff_hf:.3e}")
        print(f"EP=2 vs EP=1 max abs diff = {diff_ep1:.3e}")
        print(f"EP=2 gradient max abs diff = {gradient_diff:.3e}")
        print(f"EP=2 update   max abs diff = {update_diff:.3e}")
        print(f"aux metric (rank 0)       = {aux.instance_acc.item():.4f}")
        print(f"failed ranks = {int(local_ok.item())}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "EP wiring equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
