"""PP checkpoint round-trip: a resumed pp=2 run must match the uninterrupted one.

Run under torchrun with 2 ranks, in two phases that share a workdir:

    D=$(mktemp -d)
    torchrun --nproc_per_node=2 \\
        tests/integration_tests/pp_checkpoint_equivalence.py full "$D"
    torchrun --nproc_per_node=2 \\
        tests/integration_tests/pp_checkpoint_equivalence.py resume "$D"

Append ``--packed`` to both commands to use a checkpointable local JSONL corpus
with independently packed pipeline microbatches. On CPU that variant uses the
explicit dense-mask SDPA substitute from the real-corpus harness, rather than
the production CUDA flex kernel.

Set ``LLMTUNER_TEST_LEGACY_LAYOUT=1`` for the full phase to save the historical
loader format without batch layout metadata, then resume normally to exercise
the manager's strict planner migration on both ranks.

Set ``LLMTUNER_TEST_RESUME_LR=0.003`` on the resume phase to verify that saved
optimizer learning rates override fresh construction values before scheduling.

Phase ``full`` trains a tiny offline qwen3 for N+M optimizer steps with
``pp=2`` (schedule 1F1B, 4 micro-batches per step) through the real
``Trainer``, saving a full DCP checkpoint -- model, FQN-keyed optimizer state,
and the trainer's step/token counters -- at step N, and records the
uninterrupted loss trajectory to the workdir.

Phase ``resume`` builds a brand-new Trainer against the same workdir, loads
the step-N checkpoint, and continues for M steps. The resumed losses must
match the uninterrupted run's last M losses BITWISE: the optimizer's Adam
moments enter the update, so a checkpoint that restored weights but not
moments (or moments keyed to the wrong stage's parameters) would diverge
visibly, while a correct round-trip reproduces the uninterrupted run's
arithmetic exactly.

The FQN keying is what pp > 1 exercises that a single-rank round-trip does
not: every stage's optimizer numbers its own parameters from 0, so positional
keys collide across ranks in one shared checkpoint (the flat FQN-keyed
optimizer state dict is what prevents that).

The synthetic random corpus is registered in the checkpoint's ``states`` under
``DATALOADER`` (its cursor is ``_position``, restored by directly generating
from the saved absolute batch index), so the resume phase reads exactly the batches the
uninterrupted run consumed at steps N+1..N+M with no manual fast-forward.

Everything runs in fp32 on CPU/gloo. The caller owns the workdir and its cleanup.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.config import (
    CheckpointConfig,
    DataloaderConfig,
    EMAConfig,
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.trainer.trainer import Trainer

SPLIT_STEP = 2  # N: checkpoint taken after this many steps
EXTRA_STEPS = 2  # M: steps trained after the checkpoint (both runs)
TOTAL_STEPS = SPLIT_STEP + EXTRA_STEPS
MICROBATCHES = 4
GLOBAL_BATCH = 8
SEQ = 32
VOCAB = 128
SEED = 42

FULL_LOSSES_FILE = "losses_full.json"


def _cfg(workdir: str, *, packed: bool = False) -> LLMTunerConfig:
    data_root = os.path.join(workdir, f"data-rank-{os.environ.get('RANK', '0')}")
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",  # offline: AutoConfig.for_model("qwen3", ...)
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=2,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=MICROBATCHES,
            # -1 derives the shard degree from the world size; with pp=2 on 2
            # ranks that leaves dp=1, so both stages see the whole batch.
            data_parallel_shard_size=-1,
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=TOTAL_STEPS,
            seed=SEED,
            deterministic=True,
            dump_folder=workdir,
            checkpoint_config=CheckpointConfig(
                enable=True,
                folder="checkpoint",
                interval=SPLIT_STEP,
                # No retention policy to exercise: keep the purge thread out.
                keep_latest_k=0,
            ),
            metrics_config=MetricsConfig(log_freq=1),
            ema_config=EMAConfig(
                decay=None if os.environ.get("LLMTUNER_TEST_EMA") == "dynamic" else 0.5,
                half_life_fraction=0.5,
            )
            if os.environ.get("LLMTUNER_TEST_EMA") in {"1", "dynamic"}
            else None,
            dataloader_config=(
                DataloaderConfig(
                    dataset="local_jsonl",
                    tokenizer_path=os.path.join(data_root, "tokenizer"),
                    dataset_path=os.path.join(data_root, "rows.jsonl"),
                    shuffle=False,
                )
                if packed
                else DataloaderConfig()
            ),
        ),
    )


def _train_steps(trainer: Trainer, data_iterator, num_steps: int) -> list[float]:
    """Drive ``num_steps`` optimizer steps; return the last-stage losses."""
    losses = []
    for _ in range(num_steps):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        assert metrics is not None  # log_freq=1: every step reports
        if trainer.pp_has_last_stage:
            losses.append(metrics["loss"])
    return losses


def _phase_full(workdir: str, *, packed: bool = False) -> None:
    trainer = Trainer(_cfg(workdir, packed=packed))
    assert trainer.world_size == 2, (
        f"this check assumes 2 ranks, got {trainer.world_size}"
    )

    data_iterator = trainer.data_iterator()
    losses = _train_steps(trainer, data_iterator, SPLIT_STEP)

    # The interval policy agrees this is a checkpointing step; a full state
    # (not the model-only last-step export) because ``last_step`` is False.
    if os.environ.get("LLMTUNER_TEST_LEGACY_LAYOUT") == "1":
        original_state_dict = trainer.dataloader.state_dict

        def legacy_state_dict():
            state = original_state_dict()
            state.pop("batch_layout", None)
            return state

        with patch.object(trainer.dataloader, "state_dict", legacy_state_dict):
            saved = trainer.checkpointer.save(SPLIT_STEP)
    else:
        saved = trainer.checkpointer.save(SPLIT_STEP)
    assert saved, "the split-step checkpoint was not written"

    losses += _train_steps(trainer, data_iterator, EXTRA_STEPS)
    torch.save(
        {
            "model": {
                key: value.detach().cpu().clone()
                for part in trainer.model_parts
                for key, value in part.state_dict().items()
            },
            "ntokens_seen": trainer.ntokens_seen,
            "ema": trainer.ema.state_dict() if trainer.ema is not None else {},
        },
        os.path.join(workdir, f"final-state-rank-{trainer.rank}.pt"),
    )

    checkpoint_dir = os.path.join(workdir, "checkpoint", f"step-{SPLIT_STEP}")
    if trainer.pp_has_last_stage:
        with open(os.path.join(workdir, FULL_LOSSES_FILE), "w") as f:
            json.dump(losses, f)
        assert os.path.isfile(os.path.join(checkpoint_dir, ".metadata")), (
            f"no DCP checkpoint at {checkpoint_dir}"
        )

    print(
        f"[full] rank {trainer.rank}: trained {TOTAL_STEPS} steps, "
        f"saved step-{SPLIT_STEP} checkpoint, losses={losses}",
        flush=True,
    )
    dist.barrier()
    trainer.close()
    dist.destroy_process_group()


def _phase_resume(workdir: str, *, packed: bool = False) -> None:
    cfg = _cfg(workdir, packed=packed)
    if "LLMTUNER_TEST_RESUME_LR" in os.environ:
        cfg.optimizer.learning_rate = float(os.environ["LLMTUNER_TEST_RESUME_LR"])
    trainer = Trainer(cfg)
    assert trainer.world_size == 2, (
        f"this check assumes 2 ranks, got {trainer.world_size}"
    )
    failures: list[str] = []

    loaded = trainer.checkpointer.load(-1)
    if not loaded:
        failures.append(f"rank {trainer.rank}: no checkpoint was loaded")
    if trainer.step != SPLIT_STEP:
        failures.append(
            f"rank {trainer.rank}: resumed at step {trainer.step}, "
            f"expected {SPLIT_STEP}"
        )

    # Non-vacuity: the optimizer must have come back warm -- a cold Adam (no
    # or zeroed moments) is exactly the failure a loss match would then catch,
    # so assert it directly for a clearer message. The container's state dict is
    # flat and FQN-keyed, so the moments are the ``state.<fqn>.exp_avg`` entries.
    optim_state = trainer.optimizer.state_dict()
    moments = {
        key: value
        for key, value in optim_state.items()
        if key.startswith("state.") and key.endswith("exp_avg")
    }
    if not moments:
        failures.append(f"rank {trainer.rank}: optimizer state is empty")
    elif not any(float(moment.abs().sum()) > 0 for moment in moments.values()):
        failures.append(f"rank {trainer.rank}: optimizer moments are all zero")

    # The loader's read position came back with the checkpoint (DATALOADER is
    # registered in ``states``), so step N+1 reads what the uninterrupted run
    # read -- no manual fast-forward.
    data_iterator = trainer.data_iterator()

    losses = _train_steps(trainer, data_iterator, EXTRA_STEPS)
    final_state = torch.load(
        os.path.join(workdir, f"final-state-rank-{trainer.rank}.pt"),
        map_location="cpu",
        weights_only=True,
    )
    actual_parameters = {
        key: value.detach().cpu()
        for part in trainer.model_parts
        for key, value in part.state_dict().items()
    }
    if actual_parameters.keys() != final_state["model"].keys():
        failures.append(f"rank {trainer.rank}: final model keys differ")
    else:
        for key, actual in actual_parameters.items():
            if not torch.equal(actual, final_state["model"][key]):
                failures.append(f"rank {trainer.rank}: final parameter {key} differs")
    if trainer.ntokens_seen != final_state["ntokens_seen"]:
        failures.append(f"rank {trainer.rank}: restored token counter differs")
    if trainer.ema is not None:
        actual_ema = trainer.ema.state_dict()
        expected_ema = final_state["ema"]
        if actual_ema.keys() != expected_ema.keys():
            failures.append(f"rank {trainer.rank}: final EMA keys differ")
        else:
            for key, actual in actual_ema.items():
                expected = expected_ema[key]
                if isinstance(actual, torch.Tensor):
                    if not torch.equal(actual.cpu(), expected.cpu()):
                        failures.append(f"rank {trainer.rank}: EMA {key} differs")
                elif actual != expected:
                    failures.append(f"rank {trainer.rank}: EMA metadata {key} differs")

    with open(os.path.join(workdir, FULL_LOSSES_FILE)) as f:
        full_losses = json.load(f)
    uninterrupted_tail = full_losses[SPLIT_STEP:]

    # The real losses live on the last-stage rank; collect them everywhere.
    gathered = [losses]
    dist.broadcast_object_list(gathered, src=trainer.world_size - 1)
    resumed_losses = gathered[0]

    max_diff = 0.0
    for offset, (got, want) in enumerate(
        zip(resumed_losses, uninterrupted_tail, strict=True), 1
    ):
        diff = abs(got - want)
        max_diff = max(max_diff, diff)
        if got != want:  # bitwise: a correct round-trip reproduces the arithmetic
            failures.append(
                f"rank {trainer.rank}: step {SPLIT_STEP + offset} loss {got!r} "
                f"vs uninterrupted {want!r} (diff {diff:.3e})"
            )

    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if trainer.rank == 0:
        print(
            f"pp=2 checkpoint round-trip: split={SPLIT_STEP} "
            f"extra={EXTRA_STEPS} microbatches={MICROBATCHES}"
        )
        print(f"uninterrupted losses = {[f'{x:.6f}' for x in full_losses]}")
        print(f"resumed losses       = {[f'{x:.6f}' for x in resumed_losses]}")
        print(f"max abs diff         = {max_diff:.3e}")
        for f_ in failures:
            print(f"  FAIL {f_}")
        if not failures:
            print("all checks passed")

    dist.barrier()
    trainer.close()
    dist.destroy_process_group()

    assert local_ok.item() == 0, "PP checkpoint round-trip check failed"


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else None
    workdir = sys.argv[2] if len(sys.argv) > 2 else None
    if phase not in ("full", "resume") or not workdir:
        raise SystemExit(
            "usage: torchrun --nproc_per_node=2 "
            "tests/integration_tests/pp_checkpoint_equivalence.py "
            "{full|resume} WORKDIR [--packed]"
        )
    packed = sys.argv[3:] == ["--packed"]
    if sys.argv[3:] and not packed:
        raise SystemExit("only --packed is accepted after WORKDIR")
    run_phase = _phase_full if phase == "full" else _phase_resume
    os.makedirs(workdir, exist_ok=True)
    if packed:
        from llmtuner.accelerator.device import device_type
        from llmtuner.models.hf.model import HFTransformerModel
        from tests.integration_tests.pp_real_corpus_equivalence import (
            _cpu_packed_attention,
            _write_corpus,
        )

        if phase == "full":
            data_root = os.path.join(
                workdir, f"data-rank-{os.environ.get('RANK', '0')}"
            )
            os.makedirs(data_root, exist_ok=True)
            _write_corpus(data_root)
        if device_type == "cpu":
            with patch.object(
                HFTransformerModel, "_apply_attention", _cpu_packed_attention
            ):
                run_phase(workdir, packed=True)
            return
    run_phase(workdir, packed=packed)


if __name__ == "__main__":
    main()
