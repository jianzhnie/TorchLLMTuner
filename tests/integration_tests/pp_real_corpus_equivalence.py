"""PP x real-corpus check: a 2-stage pipeline over packed local_jsonl data.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \\
        tests/integration_tests/pp_real_corpus_equivalence.py

The loader independently packs four microbatches before grouping a schedule
step. The reference uses the same document masks and a full-step valid-token
denominator. Flat packed streams supplied externally remain unsupported for
multiple pipeline microbatches.

Finite validation exhausts the corpus, including a fully padded microbatch in
the last schedule step. CPU uses dense-mask SDPA as an explicit test substitute
for the production flex kernel; GPU/NPU execution requires separate validation.
For PP2 × KV-CP2, set LLMTUNER_TEST_CP_SIZE=2 and run four ranks.
Set LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom to exercise the custom engine.
Set LLMTUNER_TEST_PP_SCHEDULE=Interleaved1F1B to exercise two virtual stages
per rank over six decoder layers with the same packed-corpus reference.
Set LLMTUNER_TEST_LIMO_PATH and LLMTUNER_TEST_TOKENIZER_PATH to use the first
two original LIMO question/solution pairs with the real Qwen tokenizer.
This arm tests plain-text causal packing, not SFT prompt masking or pretrained
30B weights; the decoder remains a randomly initialized small test model.
That CPU scenario retains actual CP masks/collectives and substitutes only
attention arithmetic, with sequence length 256 and a proportionally larger corpus.

"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.accelerator.tensor_tree import move_tensor_tree
from llmtuner.components.loss import cross_entropy_loss
from llmtuner.config import (
    DataloaderConfig,
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
    ValidationConfig,
)
from llmtuner.config.parallel_config import FSDPConfig, PipelineParallelConfig
from llmtuner.datasets.loader_builder import build_dataloader
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.trainer.trainer import Trainer
from tests.data_fixtures import write_tokenizer

STEPS = 3
MICROBATCHES = 4
GLOBAL_BATCH = 8
CP = int(os.environ.get("LLMTUNER_TEST_CP_SIZE", "1"))
SEQ = 256 if CP > 1 else 64
LIMO_PATH = os.environ.get("LLMTUNER_TEST_LIMO_PATH")
TOKENIZER_PATH = os.environ.get("LLMTUNER_TEST_TOKENIZER_PATH")
if LIMO_PATH and not TOKENIZER_PATH:
    raise ValueError("LLMTUNER_TEST_LIMO_PATH requires LLMTUNER_TEST_TOKENIZER_PATH")
VOCAB = 151936 if LIMO_PATH else 128
SEED = 42
PP = 2
SCHEDULE = os.environ.get("LLMTUNER_TEST_PP_SCHEDULE", "1F1B")
if SCHEDULE not in {"1F1B", "Interleaved1F1B"}:
    raise ValueError(f"unsupported packed-corpus PP schedule: {SCHEDULE!r}")
WORLD = PP * CP
NUM_ROWS = 48 * (SEQ // 64)  # Keep the finite corpus above a full token batch.
TOL = 1e-5


def _cpu_packed_attention(self, positions, attention_masks):
    """CPU test substitute: expand the production mask, then use HF SDPA."""
    from torch.nn.attention.flex_attention import create_mask

    if attention_masks is None:
        attention_masks = self.get_attention_masks(positions=positions)
    mask = create_mask(
        attention_masks.mask_mod,
        B=1,
        H=1,
        Q_LEN=positions.numel(),
        KV_LEN=positions.numel(),
        device=str(positions.device),
    )
    return {"attention_mask": mask}


def _write_corpus(root: str) -> tuple[str, str]:
    """A tiny tokenizer and a JSONL corpus of multi-document rows."""
    import os

    tokenizer_path = os.path.join(root, "tokenizer")
    corpus_path = os.path.join(root, "rows.jsonl")
    if LIMO_PATH:
        with open(LIMO_PATH, "rb") as source:
            original = source.read()
        rows = original.splitlines()[:2]
        assert len(rows) == 2, "LIMO check requires two original records"
        with open(corpus_path, "w") as handle:
            for raw in rows:
                record = json.loads(raw)
                assert record["question"] and record["solution"]
                handle.write(
                    json.dumps({"text": record["question"] + "\n" + record["solution"]})
                    + "\n"
                )
        print(
            f"real_corpus=LIMO rows=0:2 source_sha256="
            f"{hashlib.sha256(original).hexdigest()} tokenizer={TOKENIZER_PATH} "
            "objective=plain_text_causal auxiliary_small_model=True",
            flush=True,
        )
        return TOKENIZER_PATH, corpus_path
    write_tokenizer(tokenizer_path)
    with open(corpus_path, "w") as handle:
        for i in range(NUM_ROWS):
            handle.write(
                json.dumps({"text": f"w{i} " + "lorem ipsum " * (i % 5 + 1)}) + "\n"
            )
    return tokenizer_path, corpus_path


def _cfg(tokenizer_path: str, corpus_path: str) -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6 if SCHEDULE == "Interleaved1F1B" else 4,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=PP,
            context_parallel_kv_allgather_size=CP,
            data_parallel_shard_size=-1,
            fsdp_config=FSDPConfig(
                fsdp_implementation=os.environ.get(
                    "LLMTUNER_TEST_FSDP_IMPLEMENTATION", "native"
                )
            ),
            pp_config=PipelineParallelConfig(
                pipeline_parallel_schedule=SCHEDULE, num_pp_microbatches=MICROBATCHES
            ),
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=STEPS,
            seed=SEED,
            deterministic=True,
            metrics_config=MetricsConfig(log_freq=1),
            validation_config=ValidationConfig(freq=1, steps=-1),
            dataloader_config=DataloaderConfig(
                dataset="local_jsonl",
                tokenizer_path=tokenizer_path,
                dataset_path=corpus_path,
                shuffle=False,
            ),
        ),
    )


def _reference_trajectory(
    cfg: LLMTunerConfig,
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[float], dict[str, torch.Tensor], float, bool]:
    """The same steps in one process over the same loader, unsharded."""
    torch.manual_seed(cfg.seed)
    model = HFTransformerModel(build_model_config_for(cfg))
    model.load_state_dict(initial_state, strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=cfg.optimizer.implementation == "fused",
        foreach=cfg.optimizer.implementation == "foreach",
    )
    loader = build_dataloader(
        cfg,
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=GLOBAL_BATCH * SEQ,
    )
    losses = []
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        batch = next(iter(loader))
        # The packed batch carries per-token positions; drop the trainer's
        # bookkeeping int before the forward, as the trainer does.
        batch = dict(batch)
        num_valid = batch.pop("num_valid_tokens")
        batch = move_tensor_tree(batch, device=device)
        loss_sum = None
        for mb in range(MICROBATCHES):
            row = slice(mb, mb + 1)
            microbatch = {k: v[row] for k, v in batch.items()}
            inputs, labels, extra = model.preprocess_inputs(
                microbatch, parallel_dims=None
            )
            logits = model(inputs, **extra)
            loss = cross_entropy_loss(logits, labels)
            (loss / num_valid).backward()
            loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
        clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        optimizer.step()
        losses.append(float(loss_sum / num_valid))
    loader.close()
    model.eval()
    validation_loader = build_dataloader(
        cfg,
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=GLOBAL_BATCH * SEQ,
        repeat=False,
    )
    validation_sum = 0.0
    validation_tokens = 0
    saw_padding_microbatch = False
    try:
        with torch.no_grad():
            for batch in validation_loader:
                validation_tokens += batch["num_valid_tokens"]
                batch = move_tensor_tree(
                    {k: v for k, v in batch.items() if k != "num_valid_tokens"},
                    device=device,
                )
                for mb in range(MICROBATCHES):
                    microbatch = {k: v[mb : mb + 1] for k, v in batch.items()}
                    saw_padding_microbatch |= bool(microbatch["padding_mask"].all())
                    inputs, labels, extra = model.preprocess_inputs(
                        microbatch, parallel_dims=None
                    )
                    validation_sum += float(
                        cross_entropy_loss(model(inputs, **extra), labels)
                    )
    finally:
        validation_loader.close()
    assert validation_tokens > 0
    return (
        losses,
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
        validation_sum / validation_tokens,
        saw_padding_microbatch,
    )


def _run(root: str) -> None:
    tokenizer_path, corpus_path = _write_corpus(root)
    cfg = _cfg(tokenizer_path, corpus_path)
    failures: list[str] = []

    trainer = Trainer(cfg)
    rank = trainer.rank
    assert trainer.world_size == WORLD, (
        f"this check assumes {WORLD} ranks, got {trainer.world_size}"
    )
    local_state = {
        name: (value.full_tensor() if isinstance(value, DTensor) else value)
        .detach()
        .cpu()
        .clone()
        for part in trainer.model_parts
        for name, value in part.state_dict().items()
    }
    stage_states = [None] * WORLD
    dist.all_gather_object(stage_states, local_state)
    initial_state = {
        name: value for state in stage_states for name, value in state.items()
    }

    # -- non-vacuity: the model is split, and the corpus is really packed -----
    num_layers_held = sum(len(part.layers) for part in trainer.model_parts)
    expected_parts = 2 if SCHEDULE == "Interleaved1F1B" else 1
    assert len(trainer.model_parts) == expected_parts, (
        SCHEDULE,
        len(trainer.model_parts),
    )
    if num_layers_held >= cfg.num_hidden_layers:
        failures.append(
            f"rank {rank}: holds {num_layers_held} of {cfg.num_hidden_layers} "
            "layers -- the model was not split"
        )
    loader_state = trainer.dataloader.state_dict()
    probe = next(iter(trainer.dataloader))
    trainer.dataloader.load_state_dict(loader_state)
    data_iterator = trainer.data_iterator()
    if probe["labels"].shape != (MICROBATCHES, GLOBAL_BATCH * SEQ // MICROBATCHES):
        failures.append(
            f"rank {rank}: unexpected microbatch shape {probe['labels'].shape}"
        )
    if "positions" not in probe:
        failures.append(f"rank {rank}: packed batch carries no positions")
    positions = probe["positions"]
    if positions is not None and int((positions == 0).sum()) < 2:
        failures.append(
            f"rank {rank}: a packed row should restart positions more than "
            f"once, got {int((positions == 0).sum())} restarts"
        )

    # -- the trajectory ------------------------------------------------------
    pp_losses = []
    for _step in range(STEPS):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        assert metrics is not None  # log_freq=1: every step reports
        pp_losses.append(metrics["loss"])
    validation_reports = []

    def capture_validation(*, loss, step):
        validation_reports.append(loss)

    trainer.metrics.log_validation = capture_validation
    trainer.validate(trainer.step)
    if any(not part.training for part in trainer.model_parts):
        failures.append(f"rank {rank}: validation did not restore training mode")
    # A real schedule must unwind on every stage when only the final stage's
    # metric sink fails. It must also restore each module's training mode/RNG.
    saved_modes = {
        module: module.training
        for part in trainer.model_parts
        for module in part.modules()
    }
    saved_rng = torch.get_rng_state().clone()
    sink_error = OSError("pipeline validation sink failed")

    def fail_validation_sink(*, loss, step):
        raise sink_error

    trainer.metrics.log_validation = fail_validation_sink
    try:
        trainer.validate(trainer.step)
    except OSError as error:
        assert trainer.pp_has_last_stage and error is sink_error
    except RuntimeError as error:
        assert not trainer.pp_has_last_stage
        assert "Validation result reporting" in str(error)
    else:
        raise AssertionError("pipeline validation sink failure did not propagate")
    finally:
        trainer.metrics.log_validation = capture_validation
    assert all(module.training == mode for module, mode in saved_modes.items())
    torch.testing.assert_close(torch.get_rng_state(), saved_rng, rtol=0, atol=0)
    dist.barrier()
    print(f"rank={rank}: actual PP validation sink failure coordinated", flush=True)
    local_final_state = {
        name: (value.full_tensor() if isinstance(value, DTensor) else value)
        .detach()
        .cpu()
        .clone()
        for part in trainer.model_parts
        for name, value in part.state_dict().items()
    }
    trainer.close()

    reference, final_state, reference_validation, saw_padding = _reference_trajectory(
        cfg, initial_state, trainer.device
    )
    if not saw_padding:
        failures.append(
            f"rank {rank}: finite corpus did not exercise a padded microbatch"
        )
    if trainer.pp_has_last_stage:
        if len(validation_reports) != 1:
            failures.append(f"rank {rank}: expected one finite validation report")
        elif (
            not math.isfinite(validation_reports[0])
            or not math.isfinite(reference_validation)
            or abs(validation_reports[0] - reference_validation) > TOL
        ):
            failures.append(
                f"rank {rank}: validation {validation_reports[0]} "
                f"vs {reference_validation}"
            )
    max_parameter_diff = 0.0
    for name, actual in local_final_state.items():
        diff = float((actual - final_state[name]).abs().max())
        max_parameter_diff = max(max_parameter_diff, diff)
        if not math.isfinite(diff) or diff > TOL:
            failures.append(f"rank {rank}: final parameter {name} diff {diff:.3e}")
    max_diff = torch.tensor(max_parameter_diff, device=trainer.device)
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    for step, (got, want) in enumerate(zip(pp_losses, reference, strict=True)):
        if not math.isfinite(got) or not math.isfinite(want) or abs(got - want) > TOL:
            failures.append(f"rank {rank} step {step + 1}: loss {got} vs {want}")

    verdict = torch.tensor(len(failures), dtype=torch.int64, device=trainer.device)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    rank_failures = [None] * WORLD
    dist.all_gather_object(rank_failures, failures)
    if rank == 0:
        if trainer.device.type == "cpu":
            print("CPU attention uses dense-mask SDPA; CUDA flex is not verified.")
        print(
            f"pp={PP} cp={CP} schedule={SCHEDULE} "
            f"corpus={'LIMO-prefix' if LIMO_PATH else 'generated'}(packed) "
            f"steps={STEPS} tol={TOL:.0e}"
        )
        print(f"pp losses  = {[f'{x:.6f}' for x in pp_losses]}")
        print(f"reference  = {[f'{x:.6f}' for x in reference]}")
        print(f"final parameter max diff = {float(max_diff):.3e}")
        print(f"finite validation reference = {reference_validation:.6f}")
        for errors in rank_failures:
            for error in errors:
                print(f"  FAIL {error}")
        print("all checks passed" if int(verdict) == 0 else "CHECKS FAILED")
    assert int(verdict) == 0, f"{int(verdict)} check(s) failed -- see rank 0 output"


def main() -> None:
    from llmtuner.accelerator.device import device_type

    with tempfile.TemporaryDirectory(prefix="pp_real_corpus_") as root:
        if device_type == "cpu" and CP > 1:
            from tests.integration_tests.pp_cp_equivalence import cpu_cp_attention

            with cpu_cp_attention():
                _run(root)
        elif device_type == "cpu":
            with patch.object(
                HFTransformerModel, "_apply_attention", _cpu_packed_attention
            ):
                _run(root)
        else:
            _run(root)


if __name__ == "__main__":
    main()
