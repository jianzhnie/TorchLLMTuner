"""EP+FSDP check: expert weights must shard on the sparse mesh, not the dense one.

Run under torchrun with 4 ranks (from the repo root):

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/ep_fsdp_equivalence.py

Add ``--converted-root-expert`` to register a root alias to the local down
projection after real EP conversion and before FSDP. This checks shared identity,
sparse placement and expert gradients through root-owned FSDP management.

Add ``--external-expert-alias`` to check that an external full-expert down
projection alias is rejected before installing EP-sharded replacements.

Add ``--shared-projection`` to share the two MoE layers' Q projection before
parallel transforms and compare its accumulated gradient with the dense reference.

Add ``--pattern-fsdp`` to exercise pattern-selected FSDP units with the same
expert mesh, loss, and gradient checks.
Add ``--fsdp-implementation custom`` to exercise custom parameter groups and
hook-owned communication buffers; default module hooks are used here.
Add --domino-expert-failure to inject one rank's local expert projection error,
require all four ranks to abort before combine, reset the FSDP root, and compare
the same runtime's healthy retry against the independent dense reference.
Append --expert-fsdp to select each expert itself as an FSDP pattern unit
and verify post-forward state before and after root reset. --shared-experts
shares all weights across the two layers, requiring a grouped FSDP state.
--relocated-expert-hooks moves those hooks to the common root (custom only),
exercising multiple communication contexts and incomplete-forward cleanup.

Regression pin for the missing ``moe_enabled``/``moe`` flags on the EP swap:
without them, FSDP's MoE branch never fires and the expert weights -- each
rank holding a DIFFERENT slice of the experts -- are sharded as dense
parameters over the dense DP mesh, mixing ranks of different EP coordinates
into one FSDP group. The forward then all-gathers mismatched shards and the
output is garbage, silently.

Setup: dp_shard=4, ep=2 (so efsdp=2) over a tiny offline qwen3_moe, through
the real ``parallelize_hf_transformers`` entry point. Each rank runs its own
quarter of a global batch; FSDP's gradient reduction (sum, division disabled)
turns the per-rank partials into full-batch gradients. The whole thing is
compared against a single-process reference over the full batch: loss, router
gradients, per-expert weight gradients.

Non-vacuity: the expert parameters must be DTensors on the edp (efsdp) mesh --
size 2 here -- while dense parameters sit on the size-4 dense mesh. That mesh
assignment is exactly what the missing flags broke.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.distributed.tensor import DTensor
from transformers import AutoConfig

from llmtuner.accelerator.collectives import all_reduce
from llmtuner.accelerator.spmd import spmd_context
from llmtuner.config import ParallelConfig
from llmtuner.config.parallel_config import ExpertParallelConfig, FSDPConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.models.common.moe.layer import MoE
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.expert_parallel.expert_parallel import apply_ep
from llmtuner.parallel.parallelize import parallelize_hf_transformers
from llmtuner.parallel.topology import ParallelTopology

NUM_EXPERTS = 8
TOP_K = 2
DIM = 64
EXPERT_HIDDEN = 48
TOKENS_PER_RANK = 16
TOL = 1e-4


def _config() -> AutoConfig:
    # experts_implementation="eager": transformers 5.x defaults to grouped_mm,
    # which is bf16-only and undispatchable on CPU (see test_ep_model_replacement.py).
    return AutoConfig.for_model(
        "qwen3_moe",
        vocab_size=128,
        hidden_size=DIM,
        intermediate_size=128,
        moe_intermediate_size=EXPERT_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        norm_topk_prob=True,
        max_position_embeddings=256,
        experts_implementation="eager",
    )


def _model(
    seed: int = 0, *, shared_projection: bool = False, shared_experts: bool = False
) -> HFTransformerModel:
    """Deterministically initialized tiny qwen3_moe; identical on every rank."""
    torch.manual_seed(seed)
    model = HFTransformerModel(_config()).float().eval()
    if shared_projection:
        model.layers[1].self_attn.q_proj.weight = model.layers[
            0
        ].self_attn.q_proj.weight
    if shared_experts:
        first, second = (layer.mlp.experts for layer in model.layers)
        second.gate_up_proj = first.gate_up_proj
        second.down_proj = first.down_proj
    return model


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    """The global batch ``(dp, T)``; identical on every rank, row-sliced later.

    One row per rank: dp shards a batch, and each row is an independent
    sequence (RoPE makes the position values semantic, so a row's positions
    restart at 0 -- slicing one long sequence across ranks would not be dp).
    """
    ids = torch.randint(
        128, (4, TOKENS_PER_RANK), generator=torch.Generator().manual_seed(1000)
    )
    labels = torch.randint(
        128, (4, TOKENS_PER_RANK), generator=torch.Generator().manual_seed(2000)
    )
    return ids, labels


def _loss_sum(model, ids_T, labels_T) -> torch.Tensor:
    """Summed CE over one packed 1-D sequence (the wrapper's input shape)."""
    positions = torch.arange(ids_T.shape[0])
    return F.cross_entropy(model(ids_T, positions=positions), labels_T, reduction="sum")


def _full(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern-fsdp", action="store_true")
    parser.add_argument("--fixed-router", action="store_true")
    parser.add_argument("--domino-expert-failure", action="store_true")
    parser.add_argument("--expert-fsdp", action="store_true")
    parser.add_argument("--shared-experts", action="store_true")
    parser.add_argument("--relocated-expert-hooks", action="store_true")
    parser.add_argument("--shared-projection", action="store_true")
    parser.add_argument("--external-expert-alias", action="store_true")
    parser.add_argument("--converted-root-expert", action="store_true")
    parser.add_argument(
        "--fsdp-implementation", choices=("native", "custom"), default="native"
    )
    args = parser.parse_args()
    if args.relocated_expert_hooks and (
        not args.expert_fsdp or args.fsdp_implementation != "custom"
    ):
        parser.error("--relocated-expert-hooks requires --expert-fsdp and custom FSDP")
    if args.fsdp_implementation == "custom":
        from llmtuner.parallel.fully_shard_parallel.fine_grained_param_group import (
            FineGrainedFSDPParamGroup,
        )
        from llmtuner.parallel.fully_shard_parallel.fine_grained_state import (
            FineGrainedFSDPState,
        )
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4, f"this check assumes 4 ranks, got {world}"

    cfg = ParallelConfig(
        data_parallel_shard_size=4,
        expert_parallel_size=2,
        ep_config=ExpertParallelConfig(
            fixed_router=args.fixed_router,
            router_aux_loss_coef=0.0 if args.fixed_router else None,
            ep_token_dispatcher="domino" if args.domino_expert_failure else "alltoall",
        ),
        fsdp_config=FSDPConfig(
            fsdp_implementation=args.fsdp_implementation,
            fsdp_apply_modules={
                "model.model.layers.{*}.mlp.routed_experts.inner_experts": {}
            }
            if args.expert_fsdp
            else ({"model.model.layers.{*}": {}} if args.pattern_fsdp else {}),
        ),
    )
    parallel_dims = ParallelTopology.from_config(cfg, world)
    parallel_dims.build_device_mesh()
    loss_mesh = parallel_dims.get_mesh("loss")

    if args.external_expert_alias:
        model = _model()
        original = model.layers[0].mlp
        source = original.experts.down_proj
        model.register_parameter("external_expert", source)
        try:
            parallelize_hf_transformers(
                model,
                cfg=cfg,
                mesh=parallel_dims.spmd_dense_mesh(),
                parallel_dims=parallel_dims,
                device=torch.device("cpu"),
            )
        except UnsupportedCombinationError as error:
            assert "external_expert" in str(error)
        else:
            raise AssertionError("EP-sharded external aliases must not be split")
        assert model.layers[0].mlp is original
        assert model.external_expert is source
        if rank == 0:
            print("EP external expert alias rejected before installation", flush=True)
        dist.destroy_process_group()
        return

    failures: list[str] = []

    # -- single-process full-batch reference (computed identically per rank) ---
    # The wrapper takes one packed 1-D sequence per call, so the reference runs
    # the batch's rows one at a time and sums -- the same total as one batched
    # call, and exactly what dp sharding splits.
    ids_all, labels_all = _data()
    ref = _model(
        shared_projection=args.shared_projection, shared_experts=args.shared_experts
    )
    if args.fixed_router:
        from tests.integration_tests.fixed_router_ep_equivalence import (
            explicit_reference,
        )

        for layer in ref.layers:
            block = layer.mlp

            def fixed_forward(inputs, original=block):
                flat = inputs.reshape(-1, inputs.shape[-1])
                output, _, _ = explicit_reference(
                    original, flat, torch.zeros(flat.shape[0], dtype=torch.bool)
                )
                return output.reshape_as(inputs)

            block.forward = fixed_forward
    with spmd_context(None):
        ref_losses = [
            _loss_sum(ref, ids_all[r], labels_all[r]) for r in range(ids_all.shape[0])
        ]
    sum(ref_losses).backward()
    ref_loss = sum(loss.item() for loss in ref_losses)

    # -- ep=2 + dp_shard=4 through the real orchestration entry point ----------
    def apply_ep_with_alias(model, **kwargs):
        model = apply_ep(model, **kwargs)
        if args.shared_experts:
            first, second = (
                layer.mlp.routed_experts.inner_experts for layer in model.layers
            )
            second.w1_EFD = first.w1_EFD
            second.w3_EFD = first.w3_EFD
            second.w2_EDF = first.w2_EDF
        if args.relocated_expert_hooks:
            cfg.fsdp_config.fsdp_apply_modules = {
                f"model.model.layers.{index}.mlp.routed_experts.inner_experts": {
                    "hook_module": model
                }
                for index in range(len(model.layers))
            }
        if args.converted_root_expert:
            model.register_parameter(
                "external_expert",
                model.layers[0].mlp.routed_experts.inner_experts.w2_EDF,
            )
        return model

    with patch("llmtuner.parallel.parallelize.apply_ep", apply_ep_with_alias):
        model = parallelize_hf_transformers(
            _model(
                shared_projection=args.shared_projection,
                shared_experts=args.shared_experts,
            ),
            cfg=cfg,
            mesh=parallel_dims.spmd_dense_mesh(),
            parallel_dims=parallel_dims,
            device=torch.device("cpu"),
        )
    if args.converted_root_expert:
        expert = model.layers[0].mlp.routed_experts.inner_experts.w2_EDF
        assert model.external_expert is expert
        assert isinstance(expert, DTensor)
        assert expert.device_mesh.size() == 2

    if args.shared_projection:
        assert model.layers[0].self_attn.q_proj.weight is (
            model.layers[1].self_attn.q_proj.weight
        )
        if not args.converted_root_expert:
            assert model.layers[0]._get_fsdp_state() is (
                model.layers[1]._get_fsdp_state()
            )

    if args.domino_expert_failure:
        routed = model.layers[0].mlp.routed_experts
        assert routed.error_group.size() == world
        original_run = routed._run_experts
        original_linear = F.linear
        injected = False
        # Exercise training-only observations on the failing first MoE block.
        # Preserve prior accumulation to detect accidental clearing as well.
        model.train()
        AuxLoss.set_step_denominator(
            torch.tensor(world * TOKENS_PER_RANK), model_parts=[model]
        )
        failed_moe = model.layers[0].mlp
        failed_moe.tokens_per_expert_E.fill_(3)
        prior_counts = failed_moe.tokens_per_expert_E.clone()

        def fail_linear(*args, **kwargs):
            nonlocal injected
            if rank == 0 and not injected:
                injected = True
                raise ValueError("injected FSDP-contained expert failure")
            return original_linear(*args, **kwargs)

        def run_with_fault(*args, **kwargs):
            assert kwargs.get("coordinate_experts") is True
            with patch("torch.nn.functional.linear", fail_linear):
                return original_run(*args, **kwargs)

        try:
            with patch.object(routed, "_run_experts", run_with_fault):
                with spmd_context(parallel_dims):
                    _loss_sum(model, ids_all[rank], labels_all[rank])
        except (ValueError, RuntimeError) as error:
            if rank == 0:
                assert isinstance(error, ValueError), str(error)
                assert "injected FSDP-contained expert failure" in str(error)
            else:
                assert "Domino expert computation failed on another rank" in str(error)
        else:
            raise AssertionError(f"rank {rank}: expert failure did not reach this rank")
        torch.testing.assert_close(
            failed_moe.tokens_per_expert_E, prior_counts, rtol=0, atol=0
        )
        model.eval()
        if args.expert_fsdp:
            expert_state = routed.inner_experts._get_fsdp_state()
            if args.shared_experts:
                assert (
                    expert_state
                    is model.layers[
                        1
                    ].mlp.routed_experts.inner_experts._get_fsdp_state()
                )
            if args.relocated_expert_hooks:
                assert expert_state.hook_module() is model
                assert expert_state._training_state == TrainingState.FORWARD
                assert len(model._get_fsdp_state().communication_contexts) >= 2
            elif args.shared_experts:
                assert routed.inner_experts not in expert_state._modules_to_run_forward
                assert (
                    model.layers[1].mlp.routed_experts.inner_experts
                    in expert_state._modules_to_run_forward
                )
            else:
                assert expert_state._training_state == TrainingState.IDLE
        dist.barrier()
        # PyTorch 2.14 requires explicit root reset after an aborted iteration.
        model.reset_iter_state()
        if args.expert_fsdp:
            assert expert_state._training_state == TrainingState.IDLE
            assert not expert_state._modules_to_run_forward
            if args.relocated_expert_hooks:
                for context in model._get_fsdp_state().communication_contexts:
                    assert context.all_gather_state is None
                    assert not context.reduce_scatter_states
                    assert not context.post_forward_order
        model.zero_grad(set_to_none=True)
        assert all(parameter.grad is None for parameter in model.parameters())
        dist.barrier()
        if rank == 0:
            print(
                "Domino EP2/FSDP2: all four ranks aborted; root reset complete",
                flush=True,
            )

    with spmd_context(parallel_dims):
        loss_sum = _loss_sum(model, ids_all[rank], labels_all[rank])
    loss_sum.backward()

    all_reduce(loss_sum, group=loss_mesh.get_group())
    got_loss = float(loss_sum.detach())
    if abs(got_loss - ref_loss) > TOL:
        failures.append(f"rank {rank}: loss {got_loss:.6f} vs reference {ref_loss:.6f}")

    if args.shared_projection:
        actual = _full(model.layers[0].self_attn.q_proj.weight.grad)
        expected = ref.layers[0].self_attn.q_proj.weight.grad
        difference = (actual - expected).abs().max().item()
        if difference > TOL:
            failures.append(f"rank {rank}: shared Q gradient diff {difference:.3e}")

    num_local = NUM_EXPERTS // cfg.ep_size
    # Sparse mesh layout (dp_replicate, efsdp, ep): ep coordinate is fastest.
    ep_rank = rank % cfg.ep_size
    lo_e = ep_rank * num_local
    for layer_idx, (layer, ref_layer) in enumerate(
        zip(model.layers, ref.layers, strict=True)
    ):
        moe = layer.mlp
        if not isinstance(moe, MoE):
            failures.append(f"rank {rank} layer {layer_idx}: mlp was not swapped")
            continue

        # Non-vacuity, the BUG-1 pin: the flags FSDP's MoE branch reads, and
        # the mesh each parameter actually landed on.
        if getattr(layer, "moe_enabled", False) is not True or layer.moe is not moe:
            failures.append(
                f"rank {rank} layer {layer_idx}: moe_enabled/moe flags missing"
            )
        grouped = moe.routed_experts.inner_experts
        if args.fsdp_implementation == "custom":
            state = (
                grouped._get_fsdp_state()
                if args.expert_fsdp
                else model._get_fsdp_state()
                if args.converted_root_expert
                and (layer_idx == 0 or args.shared_projection)
                else layer._get_fsdp_state()
            )
            assert isinstance(state, FineGrainedFSDPState)
            assert len(state._fsdp_param_groups) == (1 if args.expert_fsdp else 2)
            assert all(
                isinstance(group, FineGrainedFSDPParamGroup)
                for group in state._fsdp_param_groups
            )
        w1 = grouped.w1_EFD
        gate_w = moe.router.gate.weight
        if not isinstance(w1, DTensor) or w1.device_mesh.size() != 2:
            failures.append(
                f"rank {rank} layer {layer_idx}: expert weights not on the "
                f"efsdp mesh (got {type(w1).__name__}, mesh "
                f"{None if not isinstance(w1, DTensor) else w1.device_mesh.size()})"
            )
        if not isinstance(gate_w, DTensor) or gate_w.device_mesh.size() != 4:
            gate_mesh = (
                gate_w.device_mesh.size() if isinstance(gate_w, DTensor) else None
            )
            failures.append(
                f"rank {rank} layer {layer_idx}: router weight not on the "
                f"dense dp mesh (mesh {gate_mesh})"
            )

        if isinstance(w1, DTensor):
            expert_ranks = w1.device_mesh.mesh.reshape(-1).tolist()
            expected_ranks = list(range(ep_rank, world, cfg.ep_size))
            if expert_ranks != expected_ranks:
                failures.append(
                    f"rank {rank}: expert mesh ranks {expert_ranks}, "
                    f"want {expected_ranks}"
                )
        if isinstance(gate_w, DTensor):
            dense_ranks = gate_w.device_mesh.mesh.reshape(-1).tolist()
            if dense_ranks != list(range(world)):
                failures.append(f"rank {rank}: dense mesh ranks {dense_ranks}")

        ref_mlp = ref_layer.mlp
        if args.fixed_router:
            assert moe.router.fixed_router
            assert gate_w.grad is None
            assert ref_mlp.gate.weight.grad is None
        else:
            gate_diff = (
                (_full(gate_w.grad) - ref_mlp.gate.weight.grad).abs().max().item()
            )
            if gate_diff > TOL:
                failures.append(
                    f"rank {rank} layer {layer_idx}: router grad diff {gate_diff:.3e}"
                )

        gate_up_grad = ref_mlp.experts.gate_up_proj.grad  # (E, 2F, D)
        down_grad = ref_mlp.experts.down_proj.grad  # (E, D, F)
        checks = [
            (
                "w1",
                _full(grouped.w1_EFD.grad),
                gate_up_grad[lo_e : lo_e + num_local, :EXPERT_HIDDEN],
            ),
            (
                "w3",
                _full(grouped.w3_EFD.grad),
                gate_up_grad[lo_e : lo_e + num_local, EXPERT_HIDDEN:],
            ),
            ("w2", _full(grouped.w2_EDF.grad), down_grad[lo_e : lo_e + num_local]),
        ]
        for name, got, want in checks:
            diff = (got - want).abs().max().item()
            if diff > TOL:
                failures.append(
                    f"rank {rank} layer {layer_idx}: {name} grad diff {diff:.3e}"
                )

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"tokens/rank={TOKENS_PER_RANK} experts={NUM_EXPERTS} "
            f"ep=2 dp_shard=4 (efsdp=2) tol={TOL:.0e} "
            f"implementation={args.fsdp_implementation} pattern={args.pattern_fsdp}"
        )
        print(f"loss: ep+fsdp={got_loss:.6f} ref={ref_loss:.6f}")
        print(f"failed ranks = {int(local_ok.item())}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "EP+FSDP equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
