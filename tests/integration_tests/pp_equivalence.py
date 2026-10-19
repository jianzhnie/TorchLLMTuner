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
The dense reference uses the same per-DP-rank microbatch boundaries.

A tiny offline qwen3 (random init, fixed seed) goes through the real
``Trainer`` with ``pp=2``, 4 micro-batches per step, for 4 optimizer steps,
under either a single-stage schedule (1F1B: one stage per rank), a looped
one (Interleaved1F1B: two virtual stages per rank, over a deeper model so
every stage holds at least one layer), or a V-block one (ZBVZeroBubble: two
stages per rank, paired front-to-back so rank 0 holds the first AND the last
stage). The loss on the rank holding the last stage must track, step for
step, a reference computed on the whole model with no pipeline.

What the reference is, and why it is not just the ``pp=1`` trainer: the
trainer's non-PP body flattens the whole batch into ONE causal sequence, while
the PP body runs each micro-batch as its own sequence (pipeline stages are
separate forward calls; attention cannot cross a micro-batch boundary). The
reference therefore applies the *same* row chunking on one process -- same
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
from torch.distributed.tensor import DTensor

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

STEPS = 4
MICROBATCHES = 4
GLOBAL_BATCH = 8
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
            cp_config=ContextParallelConfig(
                context_parallel_load_balancer=os.environ.get(
                    "LLMTUNER_TEST_CP_LOAD_BALANCER"
                )
            ),
            data_parallel_shard_size=-1,
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
    model = HFTransformerModel(build_model_config_for(cfg))
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
        step_batches = [next(batches) for _ in range(cfg.gradient_accumulation_steps)]
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
                logits = model(batch.input_ids[row].reshape(-1))
                loss = cross_entropy_loss(logits, targets[row].reshape(-1))
                (loss / num_valid).backward()
                loss_sum = (
                    loss.detach() if loss_sum is None else loss_sum + loss.detach()
                )

        norm = clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        norms.append(float(norm))
        optimizer.step()
        snapshots.append(
            {
                name: (parameter.grad.detach().clone(), parameter.detach().clone())
                for name, parameter in model.named_parameters()
            }
        )
        losses.append(float(loss_sum / num_valid))
    return losses, norms, snapshots


def main() -> None:
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
    rank = trainer.rank
    assert trainer.world_size == int(os.environ.get("LLMTUNER_TEST_WORLD_SIZE", "2")), (
        f"unexpected world size: {trainer.world_size}"
    )
    pp_rank = trainer.parallel_dims.get_mesh("pp").get_local_rank()
    _, dp_size = trainer.dp_rank_world_size()
    replicas = trainer.world_size // cfg.parallel.pp_size

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
    total_numel = torch.tensor([local_numel])
    dist.all_reduce(total_numel, op=dist.ReduceOp.SUM)
    reference_numel = sum(
        p.numel() for p in HFTransformerModel(build_model_config_for(cfg)).parameters()
    )
    if total_numel.item() != reference_numel * replicas:
        failures.append(
            f"rank {rank}: stage parameters sum to {total_numel.item()}, "
            f"the replicated logical model has {reference_numel * replicas}"
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
        metrics = trainer.train_step(data_iterator)
        assert metrics is not None  # log_freq=1: every step reports
        pp_snapshots.append(
            {
                name: tuple(
                    (value.full_tensor() if isinstance(value, DTensor) else value)
                    .detach()
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
        tally = torch.tensor([metrics["n_tokens_seen"]])
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
    maximum_state_errors = torch.zeros(2, dtype=torch.float64)
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
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)
    dist.all_reduce(maximum_state_errors, op=dist.ReduceOp.MAX)
    rank_failures = [None] * trainer.world_size
    dist.all_gather_object(rank_failures, failures)

    if rank == 0:
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
