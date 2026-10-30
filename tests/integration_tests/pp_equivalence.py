"""PP>1 check: a split pipeline must train like the unpipelined model.

Run under torchrun with 2 ranks, one schedule per invocation:

    torchrun --nproc_per_node=2 tests/integration_tests/pp_equivalence.py
    torchrun --nproc_per_node=2 \\
        tests/integration_tests/pp_equivalence.py Interleaved1F1B
    torchrun --nproc_per_node=2 tests/integration_tests/pp_equivalence.py ZBVZeroBubble

Append an accumulation count after the schedule (for example 1F1B 2).
Set LLMTUNER_TEST_OPTIMIZER_GROUPS=1 for embedding/head/stage-specific rules.
Use four ranks with LLMTUNER_TEST_WORLD_SIZE=4 and
LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom for PP2 × custom FSDP2.
Set LLMTUNER_TEST_DP_REPLICATE_SIZE=2 to test replicated DP independently.
For eight-rank HSDP2x2 x PP2, also set LLMTUNER_TEST_GLOBAL_BATCH=16
so every DP rank has at least four rows for the four microbatches.
Set LLMTUNER_TEST_AC=full or selective to compare checkpointed stages
against the uncheckpointed dense reference.
Set LLMTUNER_TEST_REQUIRE_RECOMPUTE=1 with full/selective AC to require
nonzero original-forward and recompute calls per local decoder at every step.
Set LLMTUNER_TEST_CP_STRATEGY=ulysses, LLMTUNER_TEST_ULYSSES_SIZE=2 and
LLMTUNER_TEST_CP_SIZE=1 for the single-axis Ulysses path.
The dense reference uses the same per-DP-rank microbatch boundaries.

A tiny offline qwen3 (random init, fixed seed) goes through the real
``Trainer`` with ``pp=2``, 4 micro-batches per step, for 4 optimizer steps,
under either a single-stage schedule (1F1B: one stage per rank), a looped
one (Interleaved1F1B: two virtual stages per rank, over a deeper model so
every stage holds at least one layer), or a V-block one (ZBVZeroBubble: two
stages per rank, paired front-to-back so rank 0 holds the first AND the last
stage). The loss on the rank holding the last stage must track, step for
step, a reference computed on the whole model with no pipeline.

The reference applies the same row chunking and input preprocessing on one
process, preserving independent rows' attention masks and positions -- same
batches, same per-micro-batch summed CE, same clip and AdamW -- so any
divergence is attributable to the pipeline machinery (stage split, p2p
activations/grads, schedule, cross-stage grad-norm reduction), not to a
different attention pattern.

Non-vacuity: each rank must hold only its own stages' layers, and the per-rank
parameter counts must sum to the whole model's.

Everything runs in fp32 on CPU/gloo. The tolerance sits above the only
intended difference: the gradient norm is reduced per stage and combined,
which is a different floating-point summation order than one whole-model norm.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.components.loss import IGNORE_INDEX, cross_entropy_loss
from llmtuner.components.metrics import get_metrics_rank
from llmtuner.config import (
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    ParamGroupConfig,
    TrainingConfig,
)
from llmtuner.config.parallel_config import (
    ContextParallelConfig,
    FSDPConfig,
    PipelineParallelConfig,
)
from llmtuner.datasets.synthetic import RandomTokenSource
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.harness import device as harness_device
from tests.integration_tests.harness import moved, moved_batch

STEPS = 4
MICROBATCHES = 4
GLOBAL_BATCH = int(os.environ.get("LLMTUNER_TEST_GLOBAL_BATCH", "8"))
SEQ = int(os.environ.get("LLMTUNER_TEST_SEQ_LEN", "32"))
VOCAB = 128
SEED = 42

# fp32; the only sanctioned divergence is the grad-norm summation order (per
# stage, then combined, vs one whole-model norm), which sits at ~1e-7 relative.
TOL = 1e-5


# Per-schedule scenario: a looped or V schedule runs two virtual stages per
# rank (4 stages over pp=2), so the model is deepened to give every stage at
# least one layer. The reference trajectory is schedule-independent -- it
# applies the same row chunking either way -- so only the split changes.
SCENARIOS = {
    # schedule: (num_hidden_layers, stages_per_rank)
    "1F1B": (2, 1),
    "Interleaved1F1B": (6, 2),
    # V-block layout: rank r takes stages (r, num_stages-1-r), so the last
    # stage -- and the loss -- sits on rank 0 (``get_metrics_rank``), not the
    # last rank.
    "ZBVZeroBubble": (6, 2),
}

# Schedules whose V layout pairs the first and last stage onto rank 0.
V_SCHEDULES = {"ZBVZeroBubble"}


def _cfg(schedule: str) -> LLMTunerConfig:
    num_layers, _ = SCENARIOS[schedule]
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",  # offline: AutoConfig.for_model("qwen3", ...)
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=2,
            context_parallel_kv_allgather_size=int(
                os.environ.get("LLMTUNER_TEST_CP_SIZE", "1")
            ),
            context_parallel_ulysses_size=int(
                os.environ.get("LLMTUNER_TEST_ULYSSES_SIZE", "1")
            ),
            cp_config=ContextParallelConfig(
                context_parallel_strategy=os.environ.get(
                    "LLMTUNER_TEST_CP_STRATEGY", "kv_allgather"
                ),
                context_parallel_load_balancer=os.environ.get(
                    "LLMTUNER_TEST_CP_LOAD_BALANCER"
                ),
            ),
            data_parallel_shard_size=-1,
            data_parallel_replicate_size=int(
                os.environ.get("LLMTUNER_TEST_DP_REPLICATE_SIZE", "1")
            ),
            fsdp_config=FSDPConfig(
                fsdp_implementation=os.environ.get(
                    "LLMTUNER_TEST_FSDP_IMPLEMENTATION", "native"
                )
            ),
            pp_config=PipelineParallelConfig(
                pipeline_parallel_schedule=schedule, num_pp_microbatches=MICROBATCHES
            ),
        ),
        optimizer=OptimizerConfig(
            learning_rate=3e-4,
            weight_decay=0.0,
            param_groups=(
                [
                    ParamGroupConfig(
                        pattern=pattern,
                        optimizer_name="AdamW",
                        optimizer_kwargs={"lr": 3e-4, "weight_decay": 0.0},
                    )
                    for pattern in (r"embed_tokens\.weight$", r"lm_head\.weight$", ".*")
                ]
                if os.environ.get("LLMTUNER_TEST_OPTIMIZER_GROUPS") == "1"
                else []
            ),
        ),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=STEPS,
            seed=SEED,
            deterministic=True,
            activation_checkpoint_mode=os.environ.get("LLMTUNER_TEST_AC", "none"),
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def _reference_trajectory(
    cfg: LLMTunerConfig,
    initial_state: dict[str, torch.Tensor],
    *,
    dp_size: int,
) -> tuple[
    list[float], list[float], list[dict[str, tuple[torch.Tensor, torch.Tensor]]]
]:
    """The same training step with no pipeline: same chunks, one process.

    Mirrors the trainer's step arithmetic exactly -- the same per-row target
    shift (``preprocess_inputs``), the same summed CE per micro-batch, the
    same clip and optimizer -- with the micro-batch loop unrolled locally
    instead of being driven through a schedule.
    """
    torch.manual_seed(cfg.seed)
    # Run the oracle on the same device as the pipeline arm: the property
    # under test is pipeline == no-pipeline, and a CPU oracle would mix
    # cross-device kernel rounding into the tolerance budget.
    model = moved(HFTransformerModel(build_model_config_for(cfg)))
    # PP deliberately derives a different initialization seed per stage.
    # Rebuild the single-process oracle from those actual initial weights;
    # otherwise even the first forward compares different models.
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=cfg.optimizer.implementation == "fused",
        foreach=cfg.optimizer.implementation == "foreach",
    )
    # Each DP rank splits its own rows into microbatches. The dense oracle
    # visits all those chunks with the same independent attention boundaries.
    reference_microbatches = dp_size * MICROBATCHES
    rows_per_mb = cfg.global_batch_size // reference_microbatches

    batches = iter(
        RandomTokenSource(
            seed=cfg.seed,
            vocab_size=cfg.vocab_size,
            batch_size=cfg.global_batch_size,
            seq_len=cfg.max_seq_len,
        )
    )
    losses = []
    norms = []
    snapshots = []
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        step_batches = [
            moved_batch(next(batches)) for _ in range(cfg.gradient_accumulation_steps)
        ]
        target_batches = [
            model.preprocess_inputs(batch, parallel_dims=None)[1].reshape(
                cfg.global_batch_size, cfg.max_seq_len
            )
            for batch in step_batches
        ]
        num_valid = sum(
            int((targets != IGNORE_INDEX).sum()) for targets in target_batches
        )
        loss_sum = None
        for batch, targets in zip(step_batches, target_batches, strict=True):
            for mb in range(reference_microbatches):
                row = slice(mb * rows_per_mb, (mb + 1) * rows_per_mb)
                # Preserve independent row boundaries and their position/mask
                # metadata, just as the pipeline's microbatch preprocessing does.
                inputs, mb_targets, kwargs = model.preprocess_inputs(
                    {"input": batch.input_ids[row], "labels": targets[row]},
                    parallel_dims=None,
                )
                logits = model(inputs, **kwargs)
                loss = cross_entropy_loss(logits, mb_targets)
                (loss / num_valid).backward()
                loss_sum = (
                    loss.detach() if loss_sum is None else loss_sum + loss.detach()
                )

        norm = clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        norms.append(float(norm))
        optimizer.step()
        snapshots.append(
            {
                name: (
                    parameter.grad.detach().cpu().clone(),
                    parameter.detach().cpu().clone(),
                )
                for name, parameter in model.named_parameters()
            }
        )
        losses.append(float(loss_sum / num_valid))
    return losses, norms, snapshots


def main() -> None:
    # ``Trainer`` starts the process group for the device llmtuner selects, so
    # this script only borrows the device to place its own cross-rank tallies.
    local_device = harness_device()
    schedule = sys.argv[1] if len(sys.argv) > 1 else "1F1B"
    if schedule not in SCENARIOS:
        raise ValueError(
            f"unknown schedule {schedule!r}; expected one of {sorted(SCENARIOS)}"
        )
    _, stages_per_rank = SCENARIOS[schedule]
    cfg = _cfg(schedule)
    if len(sys.argv) > 2:
        cfg.training.gradient_accumulation_steps = int(sys.argv[2])
        if cfg.gradient_accumulation_steps < 1:
            raise ValueError("accumulation steps must be positive")
    failures: list[str] = []

    # Trainer init owns the process group (torchrun env); pp=2 over 2 ranks.
    trainer = Trainer(cfg)
    if cfg.training.activation_checkpoint_mode in {"full", "selective"}:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        assert all(
            isinstance(layer, CheckpointWrapper)
            for part in trainer.model_parts
            for layer in part.layers
        ), "requested AC must wrap every local decoder layer"
    rank = trainer.rank
    assert trainer.world_size == int(os.environ.get("LLMTUNER_TEST_WORLD_SIZE", "2")), (
        f"unexpected world size: {trainer.world_size}"
    )
    pp_rank = trainer.parallel_dims.get_mesh("pp").get_local_rank()
    _, dp_size = trainer.dp_rank_world_size()
    replicas = trainer.world_size // cfg.parallel.pp_size
    assert (
        trainer.parallel_dims.dp_replicate == cfg.parallel.data_parallel_replicate_size
    )
    if trainer.parallel_dims.dp_replicate > 1:
        for part in trainer.model_parts:
            parameters = list(part.parameters())
            assert parameters
            for parameter in parameters:
                assert isinstance(parameter, DTensor)
                mesh = parameter.device_mesh
                assert "dp_replicate" in mesh.mesh_dim_names
                replica_axis = mesh.mesh_dim_names.index("dp_replicate")
                assert (
                    mesh.size(replica_axis) == cfg.parallel.data_parallel_replicate_size
                )
                assert isinstance(parameter.placements[replica_axis], Replicate)
    if trainer.parallel_dims.dp_shard > 1:
        sharded_parameters = [
            parameter
            for part in trainer.model_parts
            for parameter in part.parameters()
            if isinstance(parameter, DTensor)
            and parameter.to_local().numel() < parameter.numel()
        ]
        assert sharded_parameters, "no parameter was actually FSDP-sharded"

    # -- non-vacuity: this rank holds its stages and only their layers -------
    assert len(trainer.model_parts) == stages_per_rank
    num_layers_held = sum(len(part.layers) for part in trainer.model_parts)
    if num_layers_held >= cfg.num_hidden_layers:
        failures.append(
            f"rank {rank}: holds {num_layers_held} of {cfg.num_hidden_layers} "
            "layers -- the model was not split"
        )
    # Stage layout: single-stage and looped schedules put the first stage on
    # rank 0 and the last on rank 1; a V schedule pairs them front-to-back, so
    # rank 0 holds both and rank 1 neither.
    if schedule in V_SCHEDULES:
        want_first = want_last = pp_rank == 0
    else:
        want_first = pp_rank == 0
        want_last = not want_first
    if (trainer.pp_has_first_stage, trainer.pp_has_last_stage) != (
        want_first,
        want_last,
    ):
        failures.append(
            f"rank {rank}: has_first={trainer.pp_has_first_stage} "
            f"has_last={trainer.pp_has_last_stage} -- expected "
            f"({want_first}, {want_last}) for a 2-rank {schedule} pipeline"
        )
    # Both layouts keep the first stage first and the last stage last in the
    # rank's part list ("loop" stages ascend; the single-stage case has one).
    if trainer.pp_has_first_stage and isinstance(
        trainer.model_parts[0].tok_embeddings, nn.Identity
    ):
        failures.append(f"rank {rank}: first stage lost its embedding")
    if trainer.pp_has_last_stage and isinstance(
        trainer.model_parts[-1].lm_head, nn.Identity
    ):
        failures.append(f"rank {rank}: last stage lost its lm_head")

    # The stages' parameter sets are disjoint and cover the whole model.
    local_numel = sum(
        p.numel() for part in trainer.model_parts for p in part.parameters()
    )
    total_numel = torch.tensor([local_numel], device=local_device)
    dist.all_reduce(total_numel, op=dist.ReduceOp.SUM)
    reference_numel = sum(
        p.numel() for p in HFTransformerModel(build_model_config_for(cfg)).parameters()
    )
    if total_numel.item() != reference_numel * replicas:
        failures.append(
            f"rank {rank}: stage parameters sum to {total_numel.item()}, "
            f"the replicated logical model has {reference_numel * replicas}"
        )

    if os.environ.get("LLMTUNER_TEST_TRACE_STATE_MESH") == "1":
        for part in trainer.model_parts:
            for name, value in part.state_dict().items():
                if isinstance(value, DTensor):
                    mesh = value.device_mesh
                    groups = [
                        dist.get_process_group_ranks(mesh.get_group(axis))
                        for axis in range(mesh.ndim)
                    ]
                    print(
                        f"STATE_MESH rank={rank} name={name} "
                        f"shape={tuple(value.shape)} "
                        f"local={tuple(value.to_local().shape)} "
                        f"mesh={mesh.mesh.tolist()} groups={groups} "
                        f"placements={value.placements}",
                        flush=True,
                    )
    local_initial_state = {
        key: (value.full_tensor() if isinstance(value, DTensor) else value)
        .detach()
        .cpu()
        .clone()
        for part in trainer.model_parts
        for key, value in part.state_dict().items()
    }
    stage_states = [None] * trainer.world_size
    dist.all_gather_object(stage_states, local_initial_state)
    initial_state = {
        key: value for stage_state in stage_states for key, value in stage_state.items()
    }

    # -- the trajectory ------------------------------------------------------
    data_iterator = trainer.data_iterator()
    pp_losses = []
    pp_norms = []
    pp_snapshots = []
    for step in range(STEPS):
        trainer.step += 1
        replay_counts = []
        replay_handles = []
        if os.environ.get("LLMTUNER_TEST_REQUIRE_RECOMPUTE") == "1":
            from llmtuner.models.common.aux_loss import in_checkpoint_recompute

            assert cfg.training.activation_checkpoint_mode in {"full", "selective"}
            for part in trainer.model_parts:
                for layer in part.layers:
                    counts = [0, 0]
                    replay_counts.append(counts)

                    def count_forward(module, inputs, counts=counts):
                        counts[int(in_checkpoint_recompute.get())] += 1

                    replay_handles.append(
                        layer._checkpoint_wrapped_module.register_forward_pre_hook(
                            count_forward
                        )
                    )
        try:
            metrics = trainer.train_step(data_iterator)
        finally:
            for handle in replay_handles:
                handle.remove()
        if os.environ.get("LLMTUNER_TEST_REQUIRE_RECOMPUTE") == "1":
            assert replay_counts and all(
                forward > 0 and replay > 0 for forward, replay in replay_counts
            ), f"decoder forward/recompute counts: {replay_counts}"
            print(
                f"rank {rank}: step {trainer.step} decoder "
                f"forward/recompute counts={replay_counts}",
                flush=True,
            )
        assert metrics is not None  # log_freq=1: every step reports
        pp_snapshots.append(
            {
                canonical_fqn(name): tuple(
                    (value.full_tensor() if isinstance(value, DTensor) else value)
                    .detach()
                    .cpu()
                    .clone()
                    for value in (parameter.grad, parameter)
                )
                for part in trainer.model_parts
                for name, parameter in part.named_parameters()
            }
        )
        if step == 0:
            active_grads = sum(
                p.grad is not None
                for part in trainer.model_parts
                for p in part.parameters()
            )
            if active_grads == 0:
                failures.append(f"rank {rank}: PP backward produced no gradients")
        # Every rank must report the same cumulative token count. It describes
        # the data the step read, not the layer slice this stage holds, so a
        # count taken from a stage's own (sharded, micro-batched) slice would
        # disagree here -- and it is logged beside a loss normalized by the
        # whole batch, so the two would contradict each other. Reduced inside
        # the loop because the process group is gone once the run ends.
        tally = torch.tensor([metrics["n_tokens_seen"]], device=local_device)
        dist.all_reduce(tally, op=dist.ReduceOp.MAX)
        if int(tally[0]) != metrics["n_tokens_seen"]:
            failures.append(
                f"rank {rank}: token count {metrics['n_tokens_seen']} at step "
                f"{step + 1}, another rank reported {int(tally[0])}"
            )
        if trainer.pp_has_last_stage:
            # Only the last stage holds a real loss; other stages carry the
            # sentinel, which is never logged (the metrics rank is the
            # last-stage rank -- rank 1, or rank 0 for a V layout).
            pp_losses.append(metrics["loss"])
            pp_norms.append(metrics["grad_norm"])

    reference, reference_norms, reference_snapshots = _reference_trajectory(
        cfg, initial_state, dp_size=dp_size
    )
    # The all_reduce below runs on the harness backend (hccl on NPU), which
    # cannot take CPU tensors -- allocate the tally on the device. float32:
    # NPU has no fp64 collectives, and a diagnostic max needs no more.
    maximum_state_errors = torch.zeros(2, dtype=torch.float32, device=harness_device())
    for step, (actual, expected) in enumerate(
        zip(pp_snapshots, reference_snapshots, strict=True), 1
    ):
        for name, tensors in actual.items():
            for index, (label, got, want) in enumerate(
                zip(
                    ("clipped gradient", "parameter"),
                    tensors,
                    expected[name],
                    strict=True,
                )
            ):
                difference = (got - want).abs()
                maximum_state_errors[index] = max(
                    float(maximum_state_errors[index]), float(difference.max())
                )
                if not torch.allclose(got, want, rtol=TOL, atol=TOL):
                    largest = int(difference.argmax())
                    actual_gradient = tensors[0].flatten()[largest]
                    reference_gradient = expected[name][0].flatten()[largest]
                    failures.append(
                        f"rank {rank}: step {step} {name} {label} differs "
                        f"(max abs={float(difference.max()):.3e}, "
                        f"gradient at max={float(actual_gradient):.9e}/"
                        f"{float(reference_gradient):.9e})"
                    )

    # The real losses live on the rank holding the last stage; collect them on
    # rank 0 for the report. That rank is the last one for single-stage and
    # looped layouts, rank 0 itself for a V layout -- exactly what
    # ``get_metrics_rank`` resolves.
    metrics_rank = get_metrics_rank(
        parallel_dims=trainer.parallel_dims, pp_schedule=schedule
    )
    assert pp_losses or rank != metrics_rank
    gathered = [pp_losses]
    dist.broadcast_object_list(gathered, src=metrics_rank)
    pp_losses = gathered[0]
    gathered_norms = [pp_norms]
    dist.broadcast_object_list(gathered_norms, src=metrics_rank)
    pp_norms = gathered_norms[0]

    # Every rank has both series now (the reference is computed locally and
    # identically on each), so every rank runs the comparison.
    max_diff = 0.0
    for step, (got, want) in enumerate(zip(pp_losses, reference, strict=True), 1):
        diff = abs(got - want)
        max_diff = max(max_diff, diff)
        if not torch.isclose(torch.tensor(got), torch.tensor(want), rtol=TOL, atol=TOL):
            failures.append(
                f"rank {rank}: step {step} loss {got:.6f} vs reference "
                f"{want:.6f} (diff {diff:.3e})"
            )
    for step, (got, want) in enumerate(zip(pp_norms, reference_norms, strict=True), 1):
        if not torch.isclose(torch.tensor(got), torch.tensor(want), rtol=TOL, atol=TOL):
            failures.append(
                f"rank {rank}: step {step} grad norm {got:.6f} vs reference {want:.6f}"
            )

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0], device=local_device)
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)
    dist.all_reduce(maximum_state_errors, op=dist.ReduceOp.MAX)
    rank_failures = [None] * trainer.world_size
    dist.all_gather_object(rank_failures, failures)

    if rank == 0:
        print(
            f"world={trainer.world_size} dp_shard={trainer.parallel_dims.dp_shard} "
            f"dp_replicate={trainer.parallel_dims.dp_replicate} "
            f"cp={trainer.parallel_dims.cp} "
            f"cp_kv={trainer.parallel_dims.cp_kv} "
            f"cp_ulysses={trainer.parallel_dims.cp_ulysses} "
            f"fsdp={cfg.parallel.fsdp_config.fsdp_implementation} "
            f"ac={cfg.training.activation_checkpoint_mode}"
        )
        print(
            f"pp=2 schedule={schedule} steps={STEPS} microbatches={MICROBATCHES} "
            f"global_batch={GLOBAL_BATCH} seq={SEQ} tol={TOL:.0e}"
        )
        print(f"reference losses = {[f'{x:.6f}' for x in reference]}")
        print(f"pp losses        = {[f'{x:.6f}' for x in pp_losses]}")
        print(f"reference norms  = {[f'{x:.6f}' for x in reference_norms]}")
        print(f"pp norms         = {[f'{x:.6f}' for x in pp_norms]}")
        print(f"max abs diff     = {max_diff:.3e}")
        print(f"max gradient diff = {float(maximum_state_errors[0]):.3e}")
        print(f"max parameter diff = {float(maximum_state_errors[1]):.3e}")
        print(f"failed ranks     = {int(local_ok.item())}")
        for errors in rank_failures:
            for error in errors:
                print(f"  FAIL {error}")
        if local_ok.item() == 0:
            print("all checks passed")

    assert local_ok.item() == 0, "PP equivalence check failed"
    trainer.checkpointer.close()
    trainer.metrics.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
