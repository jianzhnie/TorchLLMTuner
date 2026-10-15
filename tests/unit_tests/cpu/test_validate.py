"""The validation loop: config rules, feasibility gates, and the pass itself.

Everything here runs single-process on CPU: the pass's two loud errors (zero
batches, zero valid tokens), the token-normalized reporting loss, the
eval/train mode bracketing, and the build-time rejection of the combinations
that cannot terminate cleanly (``steps=-1`` with DP > 1 or an infinite corpus,
and finite distributed datasets). These unit tests exercise reductions in
their single-rank identity form; validation_reduction_equivalence.py separately
checks real four-rank CPU/Gloo DP x CP and DP x TP reduction semantics.
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from llmtuner.components.loss import IGNORE_INDEX
from llmtuner.config import LLMTunerConfig, TrainingConfig, ValidationConfig


@pytest.mark.parametrize("failed", [False, True])
def test_validation_preserves_training_random_streams(monkeypatch, failed):
    from llmtuner.trainer.runtime import validation as validation_runtime

    trainer, _ = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=1), loader=_ListLoader([])
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(4))
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)

    def random_pass(*args):
        random.random()
        np.random.random()
        torch.rand(17)
        if failed:
            raise RuntimeError("validation failed")

    monkeypatch.setattr(validation_runtime, "run_validation_loop", random_pass)
    try:
        if failed:
            with pytest.raises(RuntimeError, match="validation failed"):
                trainer.validate(1)
        else:
            trainer.validate(1)
        assert random.random() == expected[0]
        assert np.random.random() == expected[1]
        torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def test_validation_finishes_staging_before_forward_mutates_weights(monkeypatch):
    from llmtuner.trainer.runtime import validation as validation_runtime

    trainer, _ = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=1), loader=_ListLoader([])
    )
    embedding = torch.nn.Embedding(2, 2, max_norm=1.0)
    with torch.no_grad():
        embedding.weight.fill_(10)
    expected = embedding.weight.detach().clone()
    snapshot = {}
    trainer.model_parts = [embedding]
    trainer.checkpointer = SimpleNamespace(
        maybe_wait_for_staging=lambda: snapshot.update(
            weight=embedding.weight.detach().clone()
        )
    )
    monkeypatch.setattr(
        validation_runtime,
        "run_validation_loop",
        lambda *args: embedding(torch.tensor([0, 1])),
    )
    trainer.validate(1)
    torch.testing.assert_close(snapshot["weight"], expected, rtol=0, atol=0)


def _trainer_cls():
    """The trainer, imported lazily: on torch < 2.10 its import chain
    (DTensor, pipelining, distributed.checkpoint internals) fails at
    collection, and the tests that need it skip rather than error. The config
    and gating tests above do not need it and always run."""
    return pytest.importorskip(
        "llmtuner.trainer.trainer",
        reason="trainer import needs a newer torch than this host has",
    ).Trainer


VOCAB = 8
SEQ_LEN = 4


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf")])
def test_validation_rejects_nonfinite_loss_before_logging(loss):
    from llmtuner.trainer.runtime.validation import finalize_validation

    logged = []
    trainer = SimpleNamespace(
        metrics=SimpleNamespace(log_validation=lambda **kwargs: logged.append(kwargs))
    )
    with pytest.raises(FloatingPointError, match="not finite at step 7"):
        finalize_validation(
            trainer,
            num_steps=1,
            accumulated_loss=torch.tensor(loss),
            total_global_valid_tokens=torch.tensor(2),
            loss_mesh=None,
            step=7,
        )
    assert logged == []


class _ListLoader:
    """The smallest object with the dataloader surface the pass uses."""

    def __init__(self, batches: list[dict[str, Any]]):
        self.batches = batches
        self.closed = False

    def __iter__(self):
        return iter(self.batches)

    def close(self) -> None:
        self.closed = True


class _EchoModel(torch.nn.Module):
    """Deterministic logits, so the expected loss is computable in the test."""

    def __init__(self) -> None:
        super().__init__()
        self.saw_eval_mode: bool | None = None
        self.saw_num_valid_tokens_kwarg = False

    def preprocess_inputs(self, batch, **kwargs):
        if "num_valid_tokens" in batch:
            self.saw_num_valid_tokens_kwarg = True
        labels = batch["labels"]
        return batch["input_ids"].reshape(-1), labels.reshape(-1), {}

    def forward(self, inputs, **kwargs):
        self.saw_eval_mode = not self.training
        logits = torch.zeros(inputs.numel(), VOCAB)
        logits[:, 0] = 1.0
        logits[:, 1] = 0.5
        return logits


def _batch(valid_tokens: int) -> dict[str, Any]:
    labels = torch.ones(1, SEQ_LEN, dtype=torch.long)
    labels[0, valid_tokens:] = IGNORE_INDEX
    return {
        "input_ids": torch.ones(1, SEQ_LEN, dtype=torch.long),
        "labels": labels,
        "num_valid_tokens": valid_tokens,
    }


def _make_trainer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation: ValidationConfig | None,
    loader: _ListLoader,
    model: _EchoModel | None = None,
):
    Trainer = _trainer_cls()
    trainer = Trainer.__new__(Trainer)
    cfg = LLMTunerConfig()
    cfg.training.validation_config = validation
    trainer.cfg = cfg
    trainer.parallel_dims = None
    trainer.device = torch.device("cpu")
    trainer.model = model if model is not None else _EchoModel()
    trainer.model_parts = [trainer.model]
    trainer.checkpointer = None
    trainer.ntokens_seen = 7
    logged: dict[str, Any] = {}
    trainer.metrics = SimpleNamespace(
        add_tokens=lambda n: None,
        log_validation=lambda loss, step: logged.update(loss=loss, step=step),
    )
    monkeypatch.setattr(
        "llmtuner.trainer.runtime.validation.build_dataloader", lambda *a, **k: loader
    )
    return trainer, logged


def _expected_loss(batches: list[dict[str, Any]]) -> float:
    loss_sum = sum(
        float(
            _trainer_cls().loss_sum(
                _EchoModel()(b["input_ids"]), b["labels"].reshape(-1)
            )
        )
        for b in batches
    )
    valid = sum(b["num_valid_tokens"] for b in batches)
    return loss_sum / valid


# -- config rules ------------------------------------------------------------


def test_validation_freq_must_be_positive() -> None:
    with pytest.raises(ValueError, match="validation.freq"):
        ValidationConfig(freq=0)


def test_validation_steps_must_be_positive_or_neg1() -> None:
    for steps in (0, -2):
        with pytest.raises(ValueError, match="validation.steps"):
            ValidationConfig(steps=steps)
    ValidationConfig(steps=-1)
    ValidationConfig(steps=3)


def test_validation_defaults_to_off() -> None:
    # TrainingConfig alone: LLMTunerConfig() also builds ParallelConfig,
    # whose schedule check imports torch.distributed.pipelining (absent on
    # older torch), which is not what this assertion is about.
    assert TrainingConfig().validation is None


# -- build-time feasibility ---------------------------------------------------


def test_feasibility_accepts_pipeline_parallelism() -> None:
    """PP x validation composes since 2026-10-02: the schedule's eval driver
    (upstream Validator's seam) drives the pass; see evaluate_pipeline_batch."""
    Trainer = _trainer_cls()
    Trainer.check_validation_feasibility(
        ValidationConfig(),
        dp_world_size=1,
        training_dataset="local_jsonl",
    )


def test_feasibility_rejects_steps_neg1_when_dp_gt_1() -> None:
    Trainer = _trainer_cls()
    with pytest.raises(ValueError, match="validation collectives"):
        Trainer.check_validation_feasibility(
            ValidationConfig(steps=-1),
            dp_world_size=2,
            training_dataset="local_jsonl",
        )


def test_feasibility_rejects_steps_neg1_on_the_infinite_corpus() -> None:
    Trainer = _trainer_cls()
    with pytest.raises(ValueError, match="infinite synthetic source"):
        Trainer.check_validation_feasibility(
            ValidationConfig(steps=-1),
            dp_world_size=1,
            training_dataset="random",
        )
    # The override corpus is what the pass reads, so it is the one checked.
    with pytest.raises(ValueError, match="infinite synthetic source"):
        Trainer.check_validation_feasibility(
            ValidationConfig(steps=-1, dataset="random"),
            dp_world_size=1,
            training_dataset="local_jsonl",
        )


def test_feasibility_accepts_the_terminating_combinations() -> None:
    Trainer = _trainer_cls()
    Trainer.check_validation_feasibility(
        ValidationConfig(steps=-1),
        dp_world_size=1,
        training_dataset="local_jsonl",
    )
    Trainer.check_validation_feasibility(
        ValidationConfig(steps=10),
        dp_world_size=8,
        training_dataset="random",
    )


# -- gating -------------------------------------------------------------------


def test_should_validate() -> None:
    Trainer = _trainer_cls()

    def gating_trainer(validation: ValidationConfig | None):
        trainer = Trainer.__new__(Trainer)
        cfg = LLMTunerConfig()
        cfg.training.validation_config = validation
        trainer.cfg = cfg
        return trainer

    trainer = gating_trainer(ValidationConfig(freq=10))
    assert trainer.should_validate(1)
    assert not trainer.should_validate(2)
    assert trainer.should_validate(10)
    assert trainer.should_validate(20)

    trainer_off = gating_trainer(None)
    assert not trainer_off.should_validate(1)
    assert not trainer_off.should_validate(10)


# -- the pass -------------------------------------------------------------------


def test_validate_reports_token_normalized_loss_and_restores_train_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [_batch(3), _batch(2)]
    loader = _ListLoader(batches)
    model = _EchoModel()
    model.train()
    trainer, logged = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=-1), loader=loader, model=model
    )

    trainer.validate(step=10)

    assert logged["step"] == 10
    assert logged["loss"] == pytest.approx(_expected_loss(batches))
    # The pass ran in eval mode and the model is back in train mode after.
    assert model.saw_eval_mode is True
    assert model.training
    # The bookkeeping int never reaches the model forward.
    assert not model.saw_num_valid_tokens_kwarg
    # Validation is a pure observer of the checkpointed training counter.
    assert trainer.ntokens_seen == 7
    assert loader.closed


class _FakePPSchedule:
    """Records the eval call; each microbatch's loss is a known constant."""

    def __init__(self, per_mb_loss: float):
        self.per_mb_loss = per_mb_loss
        self.eval_calls: list[dict[str, Any]] = []

    def eval(self, *, arg_mbs, kwarg_mbs, target_mbs, losses):
        self.eval_calls.append(
            {"arg_mbs": arg_mbs, "kwarg_mbs": kwarg_mbs, "target_mbs": target_mbs}
        )
        assert losses is not None
        n = len(target_mbs)
        losses.extend(torch.tensor(self.per_mb_loss) for _ in range(n))


def test_validate_pp_drives_schedule_eval_and_reports_normalized_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under PP the pass runs the schedule's eval driver, with the loss
    denominator pinned to 1 so the division happens once at the end."""
    Trainer = _trainer_cls()
    import llmtuner.trainer.runtime.validation as validate_mod

    batches = [_batch(2), _batch(2)]
    loader = _ListLoader(batches)
    trainer = Trainer.__new__(Trainer)
    cfg = LLMTunerConfig()
    cfg.training.validation_config = ValidationConfig(steps=-1)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")

    class _Dims:
        pp_enabled = True
        dp_cp_enabled = False
        tp_enabled = False

        def get_optional_mesh(self, name, **kwargs):
            if name == "batch":
                return SimpleNamespace(get_local_rank=lambda: 0, size=lambda: 1)
            return None

    trainer.parallel_dims = _Dims()
    trainer.model = None
    trainer.model_parts = [_EchoModel()]
    trainer.checkpointer = None
    trainer.pp_has_first_stage = True
    trainer.pp_has_last_stage = True
    trainer.pp_schedule = _FakePPSchedule(per_mb_loss=6.0)
    logged: dict[str, Any] = {}
    trainer.metrics = SimpleNamespace(
        add_tokens=lambda n: None,
        log_validation=lambda loss, step: logged.update(loss=loss, step=step),
    )
    monkeypatch.setattr(validate_mod, "build_dataloader", lambda *a, **k: loader)
    monkeypatch.setattr(validate_mod, "spmd_context", lambda dims: nullcontext())

    trainer.validate(step=3)

    # Two batches, each split into num_pp_microbatches microbatches (default 1).
    assert len(trainer.pp_schedule.eval_calls) == 2
    call = trainer.pp_schedule.eval_calls[0]
    assert call["arg_mbs"] is not None and call["target_mbs"] is not None
    # Denominator pinned to 1 during eval.
    assert float(trainer.pp_schedule._llmtuner_global_valid_tokens) == 1.0
    # Raw sums: 6.0 per microbatch x 1 microbatch x 2 batches, over 4 valid
    # tokens (2 per batch).
    assert logged["loss"] == pytest.approx(6.0 * 2 / 4)
    assert logged["step"] == 3


def test_validate_positive_steps_bounds_the_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [_batch(2), _batch(2), _batch(2)]
    loader = _ListLoader(batches)
    trainer, logged = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=1), loader=loader
    )

    trainer.validate(step=10)

    assert logged["loss"] == pytest.approx(_expected_loss(batches[:1]))


def test_validate_raises_on_zero_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _ListLoader([])
    model = _EchoModel()
    model.train()
    trainer, _ = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=-1), loader=loader, model=model
    )

    with pytest.raises(ValueError, match="zero batches"):
        trainer.validate(step=1)

    assert model.training
    assert loader.closed


def test_validate_raises_on_zero_valid_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _ListLoader([_batch(0)])
    trainer, _ = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=-1), loader=loader
    )

    with pytest.raises(ValueError, match="zero valid tokens"):
        trainer.validate(step=1)

    assert loader.closed


@pytest.mark.parametrize("fail", [False, True])
def test_validate_restores_mixed_module_modes(monkeypatch, fail) -> None:
    from llmtuner.trainer.runtime import validation as validation_runtime

    model = _EchoModel()
    model.add_module("frozen", torch.nn.Dropout())
    model.train()
    model.frozen.eval()
    trainer, _ = _make_trainer(
        monkeypatch,
        validation=ValidationConfig(steps=1),
        loader=_ListLoader([_batch(2)]),
        model=model,
    )

    def run_validation_loop(active_trainer, validation, step):
        assert active_trainer is trainer
        assert not model.training
        assert not model.frozen.training
        if fail:
            raise RuntimeError("validation failed")

    monkeypatch.setattr(validation_runtime, "run_validation_loop", run_validation_loop)
    if fail:
        with pytest.raises(RuntimeError, match="validation failed"):
            trainer.validate(step=1)
    else:
        trainer.validate(step=1)
    assert model.training
    assert not model.frozen.training


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("original", ["missing", None, "causal", "block_causal"])
@pytest.mark.parametrize("max_documents", [None, 1])
@pytest.mark.parametrize(
    "dataset,expected", [("local_jsonl", "block_causal"), ("random", "causal")]
)
def test_validation_override_restores_training_attention_mask(
    monkeypatch, fail, original, max_documents, dataset, expected
):
    from llmtuner.trainer.runtime import validation as validation_runtime

    trainer, _ = _make_trainer(
        monkeypatch,
        validation=ValidationConfig(steps=1, dataset=dataset),
        loader=_ListLoader([]),
    )
    trainer.cfg.dataloader.max_num_documents = max_documents
    if max_documents == 1:
        expected = "causal"
    config = SimpleNamespace()
    if original != "missing":
        config.attn_mask_type = original
    trainer.model.model = SimpleNamespace(config=config)
    other_part = _EchoModel()
    other_part.model = SimpleNamespace(config=config)
    trainer.model_parts.append(other_part)
    kernel = torch.nn.Module()
    kernel.packed = True
    trainer.model._titan_flex_kernel = kernel

    def run_loop(active_trainer, validation, step):
        assert config.attn_mask_type == expected
        assert kernel.packed == (expected == "block_causal")
        if fail:
            raise RuntimeError("validation failed")

    monkeypatch.setattr(validation_runtime, "run_validation_loop", run_loop)
    if fail:
        with pytest.raises(RuntimeError, match="validation failed"):
            trainer.validate(1)
    else:
        trainer.validate(1)
    if original == "missing":
        assert not hasattr(config, "attn_mask_type")
    else:
        assert config.attn_mask_type == original
    assert kernel.packed is True


def test_kcp_rejects_packed_validation_before_loop(monkeypatch):
    from llmtuner.trainer.runtime import validation as validation_runtime

    trainer, _ = _make_trainer(
        monkeypatch,
        validation=ValidationConfig(steps=1, dataset="local_jsonl"),
        loader=_ListLoader([]),
    )
    trainer.cfg.parallel.context_parallel_kcp_size = 2
    trainer.cfg.parallel.cp_config.context_parallel_strategy = "kcp"

    def unexpected(*args):
        pytest.fail("KCP packed validation must fail before constructing a loader")

    monkeypatch.setattr(validation_runtime, "run_validation_loop", unexpected)
    with pytest.raises(ValueError, match="KCP.*packed"):
        trainer.validate(1)
    assert trainer.model.training


@pytest.mark.parametrize("forward_fails", [False, True])
def test_validation_preserves_forward_error_when_close_fails(
    monkeypatch, forward_fails
):
    loader = _ListLoader([_batch(2)])

    def close():
        loader.closed = True
        raise RuntimeError("close failed")

    loader.close = close
    model = _EchoModel()
    original = ValueError("forward failed")
    if forward_fails:

        def forward(*args, **kwargs):
            raise original

        model.forward = forward
    trainer, _ = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=1), loader=loader, model=model
    )
    with pytest.raises(
        ValueError if forward_fails else RuntimeError,
        match="forward failed" if forward_fails else "close failed",
    ) as caught:
        trainer.validate(step=1)
    if forward_fails:
        assert caught.value is original
    assert loader.closed
    assert model.training


@pytest.mark.parametrize("rows", [0, 1, 3])
@pytest.mark.parametrize("packed_dict", [False, True])
def test_pipeline_microbatch_split_rejects_incomplete_rows(rows, packed_dict):
    from llmtuner.datasets.contracts import Batch
    from llmtuner.trainer.runtime.pipeline import split_pipeline_microbatches

    inputs = torch.ones(rows, SEQ_LEN, dtype=torch.long)
    labels = inputs.clone()
    batch = (
        {"input_ids": inputs, "labels": labels}
        if packed_dict
        else Batch(input_ids=inputs, labels=labels)
    )
    with pytest.raises(ValueError, match="Pipeline batch"):
        split_pipeline_microbatches(batch, num_microbatches=2)


@pytest.mark.parametrize("iteration_fails", [False, True])
def test_pipeline_validation_cleanup_preserves_iteration_error(
    monkeypatch, iteration_fails
):
    import llmtuner.trainer.runtime.validation as validation_module

    original = ValueError("iteration failed")
    closed = []

    class Loader:
        def __iter__(self):
            if iteration_fails:
                raise original
            return iter([])

        def close(self):
            closed.append(True)
            raise RuntimeError("close failed")

    monkeypatch.setattr(validation_module, "require", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        validation_module, "resolve_loss_reduction_meshes", lambda dims: (None, None)
    )
    monkeypatch.setattr(
        validation_module, "create_validation_dataloader", lambda *args: Loader()
    )
    trainer = SimpleNamespace(
        parallel_dims=SimpleNamespace(pp_enabled=True), device=torch.device("cpu")
    )
    with pytest.raises(
        ValueError if iteration_fails else RuntimeError,
        match="iteration failed" if iteration_fails else "close failed",
    ) as caught:
        validation_module.run_validation_loop(trainer, ValidationConfig(), 1)
    if iteration_fails:
        assert caught.value is original
    assert closed == [True]


@pytest.mark.parametrize("pipeline", [False, True])
def test_validation_closes_loader_after_token_allocation_failure(monkeypatch, pipeline):
    import llmtuner.trainer.runtime.validation as validation_module

    original = RuntimeError("token allocation failed")
    closed = []

    class Loader:
        def close(self):
            closed.append(True)
            raise ValueError("cleanup failed")

    def fail_allocation(*args, **kwargs):
        raise original

    monkeypatch.setattr(validation_module, "require", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        validation_module, "resolve_loss_reduction_meshes", lambda dims: (None, None)
    )
    monkeypatch.setattr(
        validation_module, "create_validation_dataloader", lambda *args: Loader()
    )
    monkeypatch.setattr(validation_module.torch, "zeros", fail_allocation)
    trainer = SimpleNamespace(
        parallel_dims=SimpleNamespace(pp_enabled=pipeline), device=torch.device("cpu")
    )
    with pytest.raises(RuntimeError, match="token allocation failed") as caught:
        validation_module.run_validation_loop(trainer, ValidationConfig(), 1)
    assert caught.value is original
    assert closed == [True]
