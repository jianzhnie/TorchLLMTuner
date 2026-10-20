"""PP x validation check: the schedule's eval driver runs the pass.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \\
        tests/integration_tests/pp_validation_equivalence.py

A tiny qwen3 trains under pp=2 with validation enabled (freq=1, steps=2).
Every train step is followed by a validation pass driven through
``schedule.eval`` (upstream's Validator seam). The check pins: the pass runs
on both ranks without hanging, the logged validation loss is finite and
positive, matches an unpipelined model with the same microbatch boundaries,
and the model returns to train mode afterwards. Optimizer state, scheduler
progress, training token counts and existing gradients stay unchanged.

Environment note: needs torch >= 2.12 (the pipelining surface; the eval
driver is gated by the ``pipelining_microbatch_drivers`` capability). Verified
on torch 2.14 with two CPU/Gloo ranks; GPU/NPU execution is not covered.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from llmtuner.components.loss import IGNORE_INDEX, next_token_targets
from llmtuner.config import (
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
    ValidationConfig,
)
from llmtuner.config.parallel_config import PipelineParallelConfig
from llmtuner.datasets.synthetic import RandomTokenSource
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.harness import device as harness_device

STEPS = 2
MICROBATCHES = 2
GLOBAL_BATCH = 4
SEQ = 32
VOCAB = 128
PP = 2
WORLD = 2
VALIDATION_STEPS = 2
TOL = 1e-5


def _cfg() -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=PP,
            data_parallel_shard_size=-1,
            pp_config=PipelineParallelConfig(
                pipeline_parallel_schedule="1F1B", num_pp_microbatches=MICROBATCHES
            ),
        ),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=STEPS,
            seed=42,
            deterministic=True,
            metrics_config=MetricsConfig(log_freq=1),
            validation_config=ValidationConfig(freq=1, steps=VALIDATION_STEPS),
        ),
    )


@torch.no_grad()
def _reference_validation(trainer: Trainer) -> float:
    """Reconstruct current stage weights and independently score the same corpus."""
    local_state = {
        name: tensor.detach().cpu().clone()
        for part in trainer.model_parts
        for name, tensor in part.state_dict().items()
    }
    stage_states = [None] * WORLD
    dist.all_gather_object(stage_states, local_state)
    state = {}
    for stage_state in stage_states:
        for name, tensor in stage_state.items():
            if name in state:
                torch.testing.assert_close(state[name], tensor, rtol=0, atol=0)
            state[name] = tensor
    # Isolate reference initialization from the training process's RNG state.
    with torch.random.fork_rng(devices=[]):
        model = HFTransformerModel(build_model_config_for(trainer.cfg)).eval()
    model.load_state_dict(state, strict=True)
    batches = iter(
        RandomTokenSource(
            seed=trainer.cfg.seed,
            vocab_size=VOCAB,
            batch_size=GLOBAL_BATCH,
            seq_len=SEQ,
        )
    )
    loss_sum = torch.zeros(())
    valid_tokens = 0
    for _ in range(VALIDATION_STEPS):
        batch = next(batches)
        targets = next_token_targets(batch.labels, seq_len=SEQ).reshape(
            GLOBAL_BATCH, SEQ
        )
        valid_tokens += int((targets != IGNORE_INDEX).sum())
        for inputs, labels in zip(
            batch.input_ids.chunk(MICROBATCHES),
            targets.chunk(MICROBATCHES),
            strict=True,
        ):
            logits = model(inputs.reshape(-1))
            loss_sum += F.cross_entropy(
                logits.float(), labels.reshape(-1), reduction="sum"
            )
    return float(loss_sum) / valid_tokens


def main() -> None:
    cfg = _cfg()
    failures: list[str] = []

    trainer = Trainer(cfg)
    rank = trainer.rank
    assert trainer.world_size == WORLD, (
        f"this check assumes {WORLD} ranks, got {trainer.world_size}"
    )

    # Capture the validation reports through the metrics seam.
    reported: list[float] = []
    original = trainer.metrics.log_validation

    def _capture(loss: float, step: int) -> None:
        reported.append(loss)
        original(loss=loss, step=step)

    trainer.metrics.log_validation = _capture

    data_iterator = trainer.data_iterator()
    expected: list[float] = []
    for _ in range(STEPS):
        trainer.step += 1
        trainer.train_step(data_iterator)
        if trainer.should_validate(trainer.step):
            expected.append(_reference_validation(trainer))
            optimizer_state = deepcopy(trainer.optimizer.state_dict())
            scheduler_state = deepcopy(trainer.lr_scheduler.state_dict())
            tokens_seen = trainer.ntokens_seen
            gradients = {
                name: None if parameter.grad is None else parameter.grad.clone()
                for part in trainer.model_parts
                for name, parameter in part.named_parameters()
            }
            trainer.validate(trainer.step)
            torch.testing.assert_close(
                trainer.optimizer.state_dict(), optimizer_state, rtol=0, atol=0
            )
            assert trainer.lr_scheduler.state_dict() == scheduler_state
            assert trainer.ntokens_seen == tokens_seen
            torch.testing.assert_close(
                {
                    name: parameter.grad
                    for part in trainer.model_parts
                    for name, parameter in part.named_parameters()
                },
                gradients,
                rtol=0,
                atol=0,
            )
    trainer.close()

    # The pass ran once per step (freq=1), on the rank(s) that hold the loss.
    if trainer.pp_has_last_stage:
        if len(reported) != STEPS:
            failures.append(
                f"rank {rank}: {len(reported)} validation reports, want {STEPS}"
            )
        if any(not (0.0 < loss < float("inf")) for loss in reported):
            failures.append(f"rank {rank}: non-finite validation loss {reported}")
        if len(reported) == len(expected):
            if any(
                abs(actual - ref) > TOL
                for actual, ref in zip(reported, expected, strict=True)
            ):
                failures.append(
                    f"rank {rank}: validation losses {reported} differ from {expected}"
                )
    if any(not part.training for part in trainer.model_parts):
        failures.append(f"rank {rank}: a model part is still in eval mode")

    if trainer.pp_has_last_stage:
        print(f"pp={PP} validation freq=1 steps={STEPS}")
        print(f"reported losses = {reported}")
        print(f"reference losses = {expected}")
        if len(reported) == len(expected) and reported:
            max_diff = max(abs(a - b) for a, b in zip(reported, expected, strict=True))
            print(f"max abs diff = {max_diff:.3e}")
        for f in failures:
            print(f"  FAIL {f}")
        print("all checks passed" if not failures else "CHECKS FAILED")
    # The verdict crosses an HCCL group when the trainer runs on the accelerator,
    # so it has to live on that device; a CPU tensor here is a backend error.
    verdict = torch.tensor(len(failures), dtype=torch.int64, device=harness_device())
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    dist.destroy_process_group()
    assert int(verdict) == 0, f"{int(verdict)} check(s) failed -- see last stage output"


if __name__ == "__main__":
    main()
