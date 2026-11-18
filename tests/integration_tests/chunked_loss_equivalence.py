"""Chunked loss check: the chunked trainer must train like the plain reference.

Run under torchrun with 2 ranks (from the repo root):

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/chunked_loss_equivalence.py

A tiny explicit Qwen3 dense decoder (random init, fixed seed) uses ``Trainer``
with ``chunked_loss_num_chunks=3`` (uneven chunks over the 128-token per-rank
batch) for 4 optimizer steps and is compared against a single-process,
full-logits reference over the same global batches -- the loss trajectory AND
the final parameters.

Two variants, selected by env var:

* default (``CHUNKED_LOSS_EQ_TP`` unset): dp_shard=2, so the chunked backward
  composes with FSDP2 -- the root-owned head accumulates gradients across
  chunks before the decoder backward reduces the combined gradients.
  Parameters are compared through ``full_tensor()``.
* ``CHUNKED_LOSS_EQ_TP=2``: tp=2, dp_shard=1. Each rank's hidden states are a
  T/tp sequence shard (the sequence-parallelism premise); chunking that local
  shard composes with the sum reduction, and the replicated lm_head's
  token-partial gradient is summed by the trainer's
  ``allreduce_replicated_tp_grads``. Production checkpoint layout metadata
  reconstructs all gradients and final weights for comparison with the reference.
  One row per microbatch respects the Qwen3 TP contract; accumulation retains
  the effective batch of eight rows.

Set CHUNKED_LOSS_EQ_TIED=1 to share embedding/head parameters. Runs in FP32
on the selected harness device (CPU/gloo or NPU/HCCL).
Set CHUNKED_LOSS_EQ_ACCUMULATION=2 and CHUNKED_LOSS_EQ_MASKED=1 to test
unequal sequence lengths and ignored-target counts across accumulation groups.
Set CHUNKED_LOSS_EQ_PADDING=1 for right-padded rows with differing lengths;
the reference removes padding and executes each row independently.
Set CHUNKED_LOSS_EQ_UNPAD_REFERENCE=0 for models whose compression windows
depend on padded positions; compare ordinary loss on the same padded layout.
Add CHUNKED_LOSS_EQ_PADDING_SIDE=left to reset positions after a padded prefix.
Add CHUNKED_LOSS_EQ_EMPTY_ROWS=1 to include rows containing only padding.
Set CHUNKED_LOSS_EQ_REJECT_EMPTY=1 to enable EMA and verify that a fully
masked window is rejected; only the count of consumed tokens may advance.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.components.loss import IGNORE_INDEX, cross_entropy_loss
from llmtuner.config import (
    CheckpointConfig,
    EMAConfig,
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.datasets.contracts import Batch, TrainerBatch
from llmtuner.datasets.synthetic import RandomTokenSource
from llmtuner.models.configuration import build_model_config_for
from llmtuner.models.registry import build_registered_model
from llmtuner.parallel.tensor_parallel.checkpointing import gather_tensor_parallel_state
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal
from tests.integration_tests.harness import (
    device as harness_device,
)
from tests.integration_tests.harness import (
    moved,
    moved_batch,
)
from tests.integration_tests.trainer_ema_process_resume import snapshot

STEPS = 4
GLOBAL_BATCH = 8
SEQ = 32
VOCAB = 128
SEED = 42
NUM_CHUNKS = 3

TP = int(os.environ.get("CHUNKED_LOSS_EQ_TP", "1"))
DP_SHARD = 2 // TP
# Qwen3 TP rejects multirow flattening; retain the effective batch via accumulation.
MICRO_BATCH = GLOBAL_BATCH if TP == 1 else 1
ACCUMULATION = int(os.environ.get("CHUNKED_LOSS_EQ_ACCUMULATION", "1"))
MASKED = os.environ.get("CHUNKED_LOSS_EQ_MASKED") == "1"
PADDING = os.environ.get("CHUNKED_LOSS_EQ_PADDING") == "1"
# Compression/recurrent models may assign meaning to physical padded positions.
# Keep the stronger unpadding oracle by default for padding-invariant models.
UNPAD_REFERENCE = os.environ.get("CHUNKED_LOSS_EQ_UNPAD_REFERENCE", "1") == "1"
PADDING_SIDE = os.environ.get("CHUNKED_LOSS_EQ_PADDING_SIDE", "right")
EMPTY_ROWS = os.environ.get("CHUNKED_LOSS_EQ_EMPTY_ROWS") == "1"
REJECT_EMPTY = os.environ.get("CHUNKED_LOSS_EQ_REJECT_EMPTY") == "1"
if PADDING_SIDE not in {"left", "right"}:
    raise ValueError("CHUNKED_LOSS_EQ_PADDING_SIDE must be left or right")

# fp32; the sanctioned divergences are summation orders (per-chunk CE sums,
# per-chunk reduce-scatters), which sit at ~1e-6 relative.
LOSS_TOL = 1e-4
PARAM_TOL = 1e-4
GRAD_ATOL = 1e-5
GRAD_RTOL = 1e-4


def _cfg() -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            experts_implementation="eager",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            arch_overrides={
                "tie_word_embeddings": os.environ.get("CHUNKED_LOSS_EQ_TIED") == "1",
                "head_dim": 16,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32,
                "mlp_only_layers": [0, 1],
            },
        ),
        parallel=ParallelConfig(
            tensor_parallel_size=TP,
            data_parallel_shard_size=DP_SHARD,
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=MICRO_BATCH,
            max_seq_len=SEQ,
            steps=STEPS,
            seed=SEED,
            deterministic=True,
            chunked_loss_num_chunks=NUM_CHUNKS,
            gradient_accumulation_steps=ACCUMULATION * GLOBAL_BATCH // MICRO_BATCH,
            metrics_config=MetricsConfig(log_freq=1),
            ema_config=EMAConfig(decay=0.75) if REJECT_EMPTY else None,
            checkpoint_config=CheckpointConfig(enable=REJECT_EMPTY, interval=100),
            dump_folder=(
                "outputs/refactoring_validation/chunked-empty-window/"
                + os.environ.get("MASTER_PORT", "local")
            ),
        ),
    )


def scenario_batches(source: Iterable[Batch]) -> Iterator[Batch | TrainerBatch]:
    """Apply the same deterministic loss-mask scenario before DP preparation."""
    for index, batch in enumerate(source):
        if PADDING:
            tokens = batch.input_ids
            positions = torch.arange(tokens.shape[-1]).expand_as(tokens)
            # Token-derived lengths agree before and after loader DP slicing.
            lengths = SEQ - 2 - 2 * ((tokens[:, 0] + index) % 3)
            if EMPTY_ROWS:
                lengths = lengths.masked_fill((tokens[:, 0] + index) % 4 == 0, 0)
            if PADDING_SIDE == "left":
                positions = positions - (SEQ - lengths[:, None])
                padding = positions < 0
                positions = positions.clamp_min(0)
            else:
                padding = positions >= lengths[:, None]
            labels = batch.labels.roll(-1, dims=-1).clone()
            labels[padding | (positions >= lengths[:, None] - 1)] = IGNORE_INDEX
            if MASKED:
                labels[labels.remainder(2 + index % 3) == 0] = IGNORE_INDEX
            yield {
                "input": tokens,
                "labels": labels,
                "positions": positions,
                "padding_mask": padding,
            }
            continue
        if not MASKED:
            yield batch
            continue
        length = SEQ - 2 * (index % 3)
        labels = batch.labels[:, :length].clone()
        # Different groups and DP rows retain different target counts.
        labels[labels.remainder(2 + index % 3) == 0] = IGNORE_INDEX
        yield Batch(input_ids=batch.input_ids[:, :length], labels=labels)


def _reference_trajectory(
    cfg: LLMTunerConfig,
) -> tuple[list[float], dict, list[dict[str, torch.Tensor | None]]]:
    """The same training, un-chunked and unparallelized: one process, full batch.

    Mirrors the trainer's non-PP step arithmetic exactly. As in
    tp_fsdp_equivalence.py, dp sharding turns the global batch into per-group
    flattened sequences, so the reference runs the DP groups as SEPARATE
    forwards and sums -- a single flat forward over the union would give the
    second group's rows different positions and a different causal boundary.
    """
    torch.manual_seed(cfg.seed)
    model = moved(build_registered_model(build_model_config_for(cfg)))
    initialize_buffers = getattr(model, "initialize_training_buffers", None)
    if callable(initialize_buffers):
        initialize_buffers()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=cfg.optimizer.implementation == "fused",
        foreach=cfg.optimizer.implementation == "foreach",
    )
    batches = scenario_batches(
        RandomTokenSource(
            seed=cfg.seed,
            vocab_size=cfg.vocab_size,
            batch_size=cfg.global_batch_size,
            seq_len=cfg.max_seq_len,
        )
    )
    rows_per_dp = cfg.global_batch_size // DP_SHARD
    losses = []
    gradients = []
    empty_rows = 0
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        num_valid = 0
        loss_sum = None
        for _microbatch in range(cfg.gradient_accumulation_steps):
            batch = next(batches)
            if PADDING and UNPAD_REFERENCE:
                # Independent oracle: no padding mask or padded token reaches
                # reference attention, unlike the flattened Trainer path.
                num_valid += int((batch["labels"] != IGNORE_INDEX).sum())
                for row in range(batch["input"].shape[0]):
                    keep = ~batch["padding_mask"][row]
                    if not keep.any():
                        empty_rows += 1
                        continue
                    inputs = moved(batch["input"][row, keep])
                    targets = moved(batch["labels"][row, keep])
                    logits = model(inputs).logits
                    loss = cross_entropy_loss(logits, targets)
                    loss_sum = loss if loss_sum is None else loss_sum + loss
                continue
            if PADDING:
                empty_rows += int(batch["padding_mask"].all(dim=-1).sum())
                batch = {name: moved(value) for name, value in batch.items()}
            else:
                batch = moved_batch(batch)
            num_valid += int(
                (
                    model.prepare_inputs(batch, parallel_dims=None)[1] != IGNORE_INDEX
                ).sum()
            )
            for d in range(DP_SHARD):
                rows = slice(d * rows_per_dp, (d + 1) * rows_per_dp)
                sub = (
                    {name: value[rows] for name, value in batch.items()}
                    if PADDING
                    else Batch(
                        input_ids=batch.input_ids[rows], labels=batch.labels[rows]
                    )
                )
                inputs, targets, kwargs = model.prepare_inputs(sub, parallel_dims=None)
                logits = model(inputs, **kwargs).logits
                loss = cross_entropy_loss(logits, targets)
                loss_sum = loss if loss_sum is None else loss_sum + loss
        # Normalized BEFORE backward, as the trainer does.
        (loss_sum / num_valid).backward()
        clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        gradients.append(
            {
                name: (
                    parameter.grad.detach().clone()
                    if parameter.grad is not None
                    else None
                )
                for name, parameter in model.named_parameters()
            }
        )
        optimizer.step()
        losses.append(float(loss_sum.detach() / num_valid))
    if PADDING and EMPTY_ROWS:
        assert empty_rows > 0, "The padding scenario must exercise empty rows"
    params = {name: p.detach().clone() for name, p in model.named_parameters()}
    return losses, params, gradients


def _full(t: torch.Tensor) -> torch.Tensor:
    """Global tensor at this rank: redistribute the FSDP shard."""
    return t.full_tensor() if isinstance(t, DTensor) else t.detach()


def global_parameters(
    model: torch.nn.Module, *, gradients: bool = False
) -> dict[str, torch.Tensor | None]:
    state = {
        canonical_fqn(name): parameter.grad if gradients else parameter.detach()
        for name, parameter in model.named_parameters()
    }
    assert len(state) == sum(1 for _ in model.parameters()), "Parameter name collision"
    # None gradients remain visible to the comparison, rather than being skipped.
    state = gather_tensor_parallel_state(state, model_parts=[model])
    return {
        name: _full(value) if value is not None else None
        for name, value in state.items()
    }


def check_empty_window_rejection(trainer: Trainer) -> None:
    """Both entry points reject empty input without publishing an update."""
    cfg = trainer.cfg
    shape = (MICRO_BATCH // DP_SHARD, SEQ)
    empty = {
        "input": torch.zeros(shape, dtype=torch.long),
        "labels": torch.full(shape, IGNORE_INDEX, dtype=torch.long),
        "padding_mask": torch.ones(shape, dtype=torch.bool),
    }
    consumed_tokens = empty["input"].numel() // TP * cfg.gradient_accumulation_steps
    for outer_loop in (False, True):
        before = snapshot(trainer)
        assert "ema" in before and before["optimizer"]
        with ExitStack() as stack:
            if outer_loop:
                stack.enter_context(
                    patch.object(
                        trainer,
                        "data_iterator",
                        lambda: iter([empty] * cfg.gradient_accumulation_steps),
                    )
                )
                stack.enter_context(
                    patch.object(cfg.training, "steps", trainer.step + 1)
                )
                save = stack.enter_context(patch.object(trainer.checkpointer, "save"))
            try:
                if outer_loop:
                    trainer._run_training()
                else:
                    trainer.train_step(iter([empty] * cfg.gradient_accumulation_steps))
            except RuntimeError as error:
                assert "zero valid tokens" in str(error), str(error)
            else:
                raise AssertionError("Fully masked accumulation window was accepted")
            if outer_loop:
                save.assert_not_called()
        # Consumed input is counted even when rejected. Only the outer loop
        # advances the attempted step; all other checkpointed state stays fixed.
        before["train_state"]["ntokens_seen"] += consumed_tokens
        before["train_state"]["step"] += int(outer_loop)
        assert_state_equal(snapshot(trainer), before)
    if trainer.rank == 0:
        print("empty window rejected; only consumed-token count advanced")
        print("outer loop retained attempted step; checkpoint save not called")


def main() -> None:
    cfg = _cfg()
    failures: list[str] = []

    # Trainer init owns the process group (torchrun env).
    trainer = Trainer(cfg)
    try:
        assert trainer.model_parts[0].enable_weight_tying == (
            os.environ.get("CHUNKED_LOSS_EQ_TIED") == "1"
        )
        rank = trainer.rank
        world = trainer.world_size
        assert world == TP * DP_SHARD == 2, f"this check assumes 2 ranks, got {world}"

        reference, ref_params, ref_gradients = _reference_trajectory(cfg)

        # -- the trajectory, through the real Trainer, chunked ---------------------
        data_iterator = iter(scenario_batches(trainer.data_iterator()))
        losses = []
        max_grad_diff = 0.0
        for step in range(STEPS):
            trainer.step += 1
            metrics = trainer.train_step(data_iterator)
            model = trainer.model_parts[0]
            assert (model.lm_head.weight is model.tok_embeddings.weight) == (
                os.environ.get("CHUNKED_LOSS_EQ_TIED") == "1"
            ), "Chunked updates must preserve embedding/head parameter identity"
            assert metrics is not None  # log_freq=1: every step reports
            losses.append(metrics["loss"])
            actual_gradients = global_parameters(model, gradients=True)
            assert actual_gradients.keys() == ref_gradients[step].keys()
            for name, actual in actual_gradients.items():
                expected = ref_gradients[step][name]
                if actual is None and expected is None:
                    continue
                if actual is None:
                    failures.append(f"rank {rank}: step {step + 1} missing grad {name}")
                    continue
                if expected is None:
                    failures.append(
                        f"rank {rank}: step {step + 1} unexpected grad {name}"
                    )
                    continue
                max_grad_diff = max(
                    max_grad_diff, (actual - expected).abs().max().item()
                )
                if not torch.allclose(actual, expected, atol=GRAD_ATOL, rtol=GRAD_RTOL):
                    failures.append(
                        f"rank {rank}: step {step + 1} gradient {name} differs"
                    )

        max_loss_diff = 0.0
        for step, (got, want) in enumerate(zip(losses, reference, strict=True), 1):
            diff = abs(got - want)
            max_loss_diff = max(max_loss_diff, diff)
            if not math.isfinite(diff) or diff > LOSS_TOL:
                failures.append(
                    f"rank {rank}: step {step} loss {got:.6f} vs reference "
                    f"{want:.6f} (diff {diff:.3e})"
                )

        # Both TP and FSDP must actually shard parameters and preserve updates.
        max_param_diff = 0.0
        model = trainer.model_parts[0]
        actual_params = global_parameters(model)
        assert actual_params.keys() == ref_params.keys()
        saw_shard = False
        for name, parameter in model.named_parameters():
            name = canonical_fqn(name)
            local = (
                parameter.to_local() if isinstance(parameter, DTensor) else parameter
            )
            want = ref_params[name]
            saw_shard |= local.numel() < want.numel()
            full = actual_params[name]
            if full.shape != want.shape:
                failures.append(f"rank {rank}: param {name} shape {full.shape}")
                continue
            diff = (full - want).abs().max().item()
            max_param_diff = max(max_param_diff, diff)
            if not math.isfinite(diff) or diff > PARAM_TOL:
                failures.append(f"rank {rank}: param {name} diff {diff:.3e}")
        if not saw_shard:
            failures.append(f"rank {rank}: no parameter was sharded -- vacuous")

        if REJECT_EMPTY:
            check_empty_window_rejection(trainer)

        # Every rank must agree that every check passed, not just report its own.
        local_ok = torch.tensor([0.0 if not failures else 1.0], device=harness_device())
        dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

        if rank == 0:
            print(
                f"tp={TP} dp_shard={DP_SHARD} chunks={NUM_CHUNKS} steps={STEPS} "
                f"microbatch={MICRO_BATCH} seq={SEQ} "
                f"accumulation={cfg.gradient_accumulation_steps} masked={MASKED} "
                f"padding={PADDING} padding_side={PADDING_SIDE} "
                f"empty_rows={EMPTY_ROWS} unpad_reference={UNPAD_REFERENCE}"
            )
            print(f"reference losses = {[f'{x:.6f}' for x in reference]}")
            print(f"chunked losses   = {[f'{x:.6f}' for x in losses]}")
            print(f"loss   max abs diff = {max_loss_diff:.3e}")
            print(f"params max abs diff = {max_param_diff:.3e}")
            print(f"gradients max abs diff = {max_grad_diff:.3e}")
            print(f"failed ranks     = {int(local_ok.item())}")
            for f in failures:
                print(f"  FAIL {f}")
            if not failures:
                print("all checks passed")

        assert local_ok.item() == 0, "chunked loss equivalence check failed"
    finally:
        try:
            trainer.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
