"""PP x EP check: a 2-stage pipeline with 2-way EP must match single-process.

Run under torchrun with 4 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/pp_ep_equivalence.py

Add `--activation-checkpoint full` or `--activation-checkpoint selective`
to exercise decoder checkpoint replay. `--aux-coefficient 0.3` magnifies
the router auxiliary gradient relative to the default coefficient 0.001.
Add `--compile-backend aot_eager` to compile every local stage while retaining
the independent eager, uncheckpointed reference and existing tolerances.

The pipeline (pp=2) composes with expert parallelism (ep=2): the sparse mesh
carries a pp axis, so each stage's ranks form their own EP group, and the MoE
swap runs per stage chunk (parallelize.py's PP per-part path, stage order
from the STAGES table). A tiny qwen3_moe trains for a few steps on the
synthetic corpus; the per-step losses must track a single-process,
single-stage reference computed in-process. Each step also compares clipped
gradients and post-AdamW parameters, materializing dense DTensors and merging
expert shards in EP rank order.

Non-vacuity: the model is actually split (each rank holds half the layers),
each MoE block holds only this stage's half of the experts, and the reference
is computed independently rather than copied from the pipeline run.

Environment note: needs torch >= 2.12 for the pipelining/SPMD surface.
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.components.loss import IGNORE_INDEX, cross_entropy_loss
from llmtuner.config import (
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.config.compile_config import CompileConfig
from llmtuner.config.parallel_config import PipelineParallelConfig
from llmtuner.datasets.synthetic import RandomTokenSource
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.models.common.moe.layer import MoE
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.expert_parallel import swap_hf_moe_blocks
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn

STEPS = 4
MICROBATCHES = 4
GLOBAL_BATCH = 8
SEQ = 32
VOCAB = 128
NUM_EXPERTS = 8
SEED = 42
PP = 2
EP = 2
WORLD = 4  # pp * dp_shard = 2 * 2; the sparse region (2 ranks/stage) fits ep=2

# Looser than the dense PP check: EP's dispatch permutes tokens, so the fp32
# summation order differs from the reference's; nothing here is reassociated
# beyond that.
TOL = 1e-4


def _cfg(
    *,
    activation_checkpoint: str = "none",
    aux_coefficient: float = 0.001,
    compile_backend: str | None = None,
) -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            arch_overrides={
                "num_experts": NUM_EXPERTS,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 48,
                "norm_topk_prob": True,
                "router_aux_loss_coef": aux_coefficient,
            },
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=PP,
            expert_parallel_size=EP,
            data_parallel_shard_size=-1,
            pp_config=PipelineParallelConfig(
                pipeline_parallel_schedule="1F1B", num_pp_microbatches=MICROBATCHES
            ),
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=STEPS,
            seed=SEED,
            deterministic=True,
            compile=compile_backend is not None,
            compile_config=CompileConfig(backend=compile_backend or "inductor"),
            activation_checkpoint_mode=activation_checkpoint,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def _reference_trajectory(
    cfg: LLMTunerConfig, initial_states: list[dict[str, torch.Tensor]]
) -> tuple[list[float], list[dict[str, tuple[torch.Tensor, torch.Tensor]]]]:
    """Run the same steps on one process from the pipeline's initial state.

    PP intentionally derives a distinct initialization seed per stage and EP
    stores only a local expert slice. Rebuild the EP=1 swapped model and merge
    those actual stage states before training the oracle; constructing a fresh
    model from ``cfg.seed`` would compare different parameters.
    """
    torch.manual_seed(cfg.seed)
    model = HFTransformerModel(build_model_config_for(cfg))
    swap_hf_moe_blocks(model)
    merged: dict[str, torch.Tensor] = {}
    for key in {key for state in initial_states for key in state}:
        values = [state[key] for state in initial_states if key in state]
        if key.endswith(("w1_EFD", "w3_EFD", "w2_EDF")):
            # Each PP stage appears on EP ranks consecutively in the mesh; the
            # local expert shards therefore concatenate in rank order.
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = values[0]
    model.load_state_dict(merged, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=cfg.optimizer.implementation == "fused",
        foreach=cfg.optimizer.implementation == "foreach",
    )
    # DP splits the global batch across EP ranks before PP forms its local
    # microbatches. Attention sees one row per forward on each rank here.
    rows_per_mb = cfg.global_batch_size // (MICROBATCHES * EP)
    batches = iter(
        RandomTokenSource(
            seed=cfg.seed,
            vocab_size=cfg.vocab_size,
            batch_size=cfg.global_batch_size,
            seq_len=cfg.max_seq_len,
        )
    )
    losses = []
    snapshots = []
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        batch = next(batches)
        AuxLoss.set_step_denominator(torch.tensor(batch.input_ids.numel()))
        targets = model.preprocess_inputs(batch, parallel_dims=None)[1].reshape(
            cfg.global_batch_size, cfg.max_seq_len
        )
        num_valid = int((targets != IGNORE_INDEX).sum())
        loss_sum = None
        for mb in range(MICROBATCHES * EP):
            row = slice(mb * rows_per_mb, (mb + 1) * rows_per_mb)
            logits = model(batch.input_ids[row].reshape(-1))
            loss = cross_entropy_loss(logits, targets[row].reshape(-1))
            (loss / num_valid).backward()
            loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
        clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        optimizer.step()
        snapshots.append(
            {
                name: (parameter.grad.detach().clone(), parameter.detach().clone())
                for name, parameter in model.named_parameters()
            }
        )
        losses.append(float(loss_sum / num_valid))
    return losses, snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activation-checkpoint", choices=("none", "full", "selective"), default="none"
    )
    parser.add_argument("--aux-coefficient", type=float, default=0.001)
    parser.add_argument("--compile-backend", choices=("aot_eager",))
    args = parser.parse_args()
    cfg = _cfg(
        activation_checkpoint=args.activation_checkpoint,
        aux_coefficient=args.aux_coefficient,
        compile_backend=args.compile_backend,
    )
    failures: list[str] = []

    trainer = Trainer(cfg)
    rank = trainer.rank
    assert trainer.world_size == WORLD, (
        f"this check assumes {WORLD} ranks, got {trainer.world_size}"
    )

    # -- non-vacuity: the model is split, and the experts are sharded --------
    num_layers_held = sum(len(part.layers) for part in trainer.model_parts)
    if num_layers_held >= cfg.num_hidden_layers:
        failures.append(
            f"rank {rank}: holds {num_layers_held} of {cfg.num_hidden_layers} "
            "layers -- the model was not split"
        )
    num_local = NUM_EXPERTS // EP
    for part_idx, part in enumerate(trainer.model_parts):
        for layer_idx, layer in enumerate(part.layers):
            moe = layer.mlp
            assert isinstance(moe, MoE), (
                f"rank {rank} part {part_idx} layer {layer_idx}: "
                "the swap did not install llmtuner MoE blocks"
            )
            if args.aux_coefficient:
                assert moe.router.aux_loss is not None
                assert moe.router.aux_loss.coeff == args.aux_coefficient
            held = moe.routed_experts.inner_experts.num_experts
            if held != num_local:
                failures.append(
                    f"rank {rank} part {part_idx} layer {layer_idx}: holds "
                    f"{held} experts, want {num_local} -- EP did not shard"
                )

    # Capture the actual PP initialization before the first optimizer step.
    # Ranks 0/1 hold stage 0's complementary expert shards; ranks 2/3 hold
    # stage 1's. Dense weights are replicated inside each stage.
    local_initial_state = {}
    for part in trainer.model_parts:
        for key, value in part.state_dict().items():
            # FSDP state_dict values are DTensors. Materialize each dense
            # stage's weights collectively before object gathering; pickling
            # the DTensor itself retains its process mesh and can hang.
            if isinstance(value, DTensor):
                value = value.full_tensor()
            local_initial_state[canonical_fqn(key)] = value.detach().cpu().clone()
    initial_states = [None] * trainer.world_size
    dist.all_gather_object(initial_states, local_initial_state)

    # -- the trajectory ------------------------------------------------------
    data_iterator = trainer.data_iterator()
    pp_ep_losses = []
    actual_snapshots = []
    for _step in range(STEPS):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        if args.compile_backend is not None:
            from torch._dynamo.eval_frame import OptimizedModule
            from torch._dynamo.utils import counters

            assert all(
                isinstance(part, OptimizedModule) for part in trainer.model_parts
            )
            assert counters["stats"]["unique_graphs"] > 0
        assert metrics is not None  # log_freq=1: every step reports
        pp_ep_losses.append(metrics["loss"])
        local = {}
        for part in trainer.model_parts:
            for name, parameter in part.named_parameters():
                values = []
                for value in (parameter.grad, parameter):
                    assert value is not None, name
                    if isinstance(value, DTensor):
                        value = value.full_tensor()
                    values.append(value.detach().cpu().clone())
                local[canonical_fqn(name)] = tuple(values)
        gathered = [None] * WORLD
        dist.all_gather_object(gathered, local)
        merged = {}
        for name in {key for state in gathered for key in state}:
            entries = [state[name] for state in gathered if name in state]
            if name.endswith(("w1_EFD", "w3_EFD", "w2_EDF")):
                merged[name] = tuple(
                    torch.cat([entry[i] for entry in entries]) for i in range(2)
                )
            else:
                for entry in entries[1:]:
                    for left, right in zip(entries[0], entry, strict=True):
                        torch.testing.assert_close(left, right, rtol=TOL, atol=TOL)
                merged[name] = entries[0]
        actual_snapshots.append(merged)
    trainer.close()

    reference, expected_snapshots = _reference_trajectory(cfg, initial_states)
    for step, (actual, expected) in enumerate(
        zip(actual_snapshots, expected_snapshots, strict=True)
    ):
        assert actual.keys() == expected.keys()
        for name in expected:
            for kind, left, right in zip(
                ("gradient", "parameter"), actual[name], expected[name], strict=True
            ):
                try:
                    torch.testing.assert_close(left, right, rtol=TOL, atol=TOL)
                except AssertionError as error:
                    failures.append(
                        f"rank {rank} step {step + 1} {name} {kind}: {error}"
                    )
    for step, (got, want) in enumerate(zip(pp_ep_losses, reference, strict=True)):
        if abs(got - want) > TOL:
            failures.append(f"rank {rank} step {step + 1}: loss {got} vs {want}")

    if rank == 0:
        print(f"pp={PP} ep={EP} world={WORLD} steps={STEPS} tol={TOL:.0e}")
        print(f"pp+ep losses = {[f'{x:.6f}' for x in pp_ep_losses]}")
        print(f"reference    = {[f'{x:.6f}' for x in reference]}")
        for f in failures:
            print(f"  FAIL {f}")
        print("all checks passed" if not failures else "CHECKS FAILED")
    verdict = torch.tensor(len(failures), dtype=torch.int64)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    assert int(verdict) == 0, f"{int(verdict)} check(s) failed -- see rank 0 output"


if __name__ == "__main__":
    main()
