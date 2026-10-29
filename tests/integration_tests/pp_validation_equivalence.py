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

Set LLMTUNER_TEST_PP_SCHEDULE=Interleaved1F1B to exercise two local stages
per rank with six decoder layers and four microbatches.
Set LLMTUNER_TEST_WORLD_SIZE=4 with four ranks for PP2 x native FSDP2.
Set LLMTUNER_TEST_CP_SIZE=2 and LLMTUNER_TEST_ULYSSES_SIZE=2 with
LLMTUNER_TEST_WORLD_SIZE=8 and LLMTUNER_TEST_SEQ_LEN=256 for dual CP.
Set LLMTUNER_TEST_DP_REPLICATE_SIZE=2 to use a replica axis within DP.

Needs torch >= 2.12; the eval driver is capability-gated.
"""

from __future__ import annotations

import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, Shard

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

SCHEDULE = os.environ.get("LLMTUNER_TEST_PP_SCHEDULE", "1F1B")
if SCHEDULE not in ("1F1B", "Interleaved1F1B"):
    raise ValueError("this check supports 1F1B or Interleaved1F1B")
INTERLEAVED = SCHEDULE == "Interleaved1F1B"
WORLD = int(os.environ.get("LLMTUNER_TEST_WORLD_SIZE", "2"))
if WORLD not in (2, 4, 8):
    raise ValueError("this check supports two, four or eight ranks")
PP = 2
CP_KV = int(os.environ.get("LLMTUNER_TEST_CP_SIZE", "1"))
CP_ULYSSES = int(os.environ.get("LLMTUNER_TEST_ULYSSES_SIZE", "1"))
if CP_KV < 1 or CP_ULYSSES < 1 or WORLD % (PP * CP_KV * CP_ULYSSES):
    raise ValueError("world size must be divisible by PP times both CP degrees")
DP = WORLD // (PP * CP_KV * CP_ULYSSES)
DP_REPLICATE = int(os.environ.get("LLMTUNER_TEST_DP_REPLICATE_SIZE", "1"))
if DP_REPLICATE < 1 or DP % DP_REPLICATE:
    raise ValueError("DP size must be divisible by the replica degree")
DP_SHARD = DP // DP_REPLICATE
STEPS = 2
MICROBATCHES = 4 if INTERLEAVED else 2
GLOBAL_BATCH = MICROBATCHES * 2 * DP
SEQ = int(os.environ.get("LLMTUNER_TEST_SEQ_LEN", "32"))
VOCAB = 128
VALIDATION_STEPS = 2
TOL = 1e-5


def _cfg() -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6 if INTERLEAVED else 2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=PP,
            context_parallel_kv_allgather_size=CP_KV,
            context_parallel_ulysses_size=CP_ULYSSES,
            data_parallel_replicate_size=DP_REPLICATE,
            data_parallel_shard_size=-1,
            pp_config=PipelineParallelConfig(
                pipeline_parallel_schedule=SCHEDULE,
                num_pp_microbatches=MICROBATCHES,
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
    local_state = {}
    for part in trainer.model_parts:
        for name, tensor in part.state_dict().items():
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            local_state[name] = tensor.detach().cpu().clone()
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
            batch.input_ids.chunk(MICROBATCHES * DP),
            targets.chunk(MICROBATCHES * DP),
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
    assert len(trainer.model_parts) == (2 if INTERLEAVED else 1)
    assert trainer.parallel_dims.dp_shard == DP_SHARD
    assert trainer.parallel_dims.dp_replicate == DP_REPLICATE
    assert trainer.parallel_dims.cp == CP_KV * CP_ULYSSES
    if CP_KV > 1 and CP_ULYSSES > 1 and harness_device().type == "npu":
        from llmtuner.models.hf.attention_backend import CP_SDPA_BACKEND
        from llmtuner.parallel.context_parallel.sdpa_attention import DualCPSdpaKernel

        assert all(
            part.model.config._attn_implementation == CP_SDPA_BACKEND
            and part.model.config._cp_sdpa_dual
            and any(isinstance(module, DualCPSdpaKernel) for module in part.modules())
            for part in trainer.model_parts
        )
    if DP_REPLICATE > 1:
        for part in trainer.model_parts:
            parameters = list(part.parameters())
            assert parameters
            for parameter in parameters:
                assert isinstance(parameter, DTensor)
                mesh = parameter.device_mesh
                assert "dp_replicate" in mesh.mesh_dim_names
                replica_axis = mesh.mesh_dim_names.index("dp_replicate")
                assert mesh.size(replica_axis) == DP_REPLICATE
                assert isinstance(parameter.placements[replica_axis], Replicate)
    if DP_SHARD > 1:
        from torch.distributed.fsdp import FSDPModule

        for part in trainer.model_parts:
            assert isinstance(part, FSDPModule)
            sharded_parameters = [
                parameter
                for parameter in part.parameters()
                if isinstance(parameter, DTensor)
                and any(
                    isinstance(placement, Shard) for placement in parameter.placements
                )
            ]
            assert sharded_parameters
            assert any(
                parameter.to_local().numel() < parameter.numel()
                for parameter in sharded_parameters
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
        print(f"pp={PP} schedule={SCHEDULE} validation freq=1 steps={STEPS}")
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
