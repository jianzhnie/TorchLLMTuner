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
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from llmtuner.components.loss import IGNORE_INDEX, cross_entropy_loss
from llmtuner.config import LLMTunerConfig, TrainingConfig, ValidationConfig
from llmtuner.models.common.validation import temporary_attention_mask
from llmtuner.models.execution import ModelExecutionCapabilities
from llmtuner.trainer.runtime.engine import TrainingEngine
from llmtuner.trainer.runtime.validation import (
    ValidationDataLoaderResources,
    check_validation_feasibility,
)


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

    def random_pass(*args, **kwargs):
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
    trainer.training_engine.model_parts = trainer.model_parts
    trainer.checkpointer = SimpleNamespace(
        maybe_wait_for_staging=lambda: snapshot.update(
            weight=embedding.weight.detach().clone()
        )
    )
    monkeypatch.setattr(
        validation_runtime,
        "run_validation_loop",
        lambda *args, **kwargs: embedding(torch.tensor([0, 1])),
    )
    trainer.validate(1)
    torch.testing.assert_close(snapshot["weight"], expected, rtol=0, atol=0)


def _trainer_cls():
    """The trainer, imported lazily: on torch < 2.10 its import chain
    (DTensor and distributed.checkpoint internals) fails at
    collection, and the tests that need it skip rather than error. The config
    and gating tests above do not need it and always run."""
    return pytest.importorskip(
        "llmtuner.trainer.trainer",
        reason="trainer import needs a newer torch than this host has",
    ).Trainer


VOCAB = 8
SEQ_LEN = 4


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf")])
def test_validation_rejects_nonfinite_loss_before_logging(loss, monkeypatch):
    from llmtuner.trainer.runtime.validation import finalize_validation

    logged = []
    error_group = object()
    coordinated = []

    @contextmanager
    def coordinate(label, *, group):
        coordinated.append(group)
        yield

    monkeypatch.setattr(
        "llmtuner.trainer.runtime.validation.synchronize_local_errors", coordinate
    )
    trainer = SimpleNamespace(
        error_group=error_group,
        metrics=SimpleNamespace(log_validation=lambda **kwargs: logged.append(kwargs)),
    )
    with pytest.raises(FloatingPointError, match="not finite at step 7"):
        finalize_validation(
            metrics=trainer.metrics,
            error_group=trainer.error_group,
            num_steps=1,
            accumulated_loss=torch.tensor(loss),
            total_global_valid_tokens=torch.tensor(2),
            loss_mesh=None,
            step=7,
        )
    assert logged == []
    assert coordinated == [error_group, error_group]


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

    execution_capabilities = ModelExecutionCapabilities()

    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(config=SimpleNamespace())
        self.saw_eval_mode: bool | None = None
        self.saw_num_valid_tokens_kwarg = False

    def validation_context(self, *, mask_type):
        return temporary_attention_mask(self.model.config, mask_type=mask_type)

    def prepare_inputs(self, batch, *, parallel_dims):
        if "num_valid_tokens" in batch:
            self.saw_num_valid_tokens_kwarg = True
        labels = batch["labels"]
        return batch["input"].reshape(-1), labels.reshape(-1), {}

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
        "input": torch.ones(1, SEQ_LEN, dtype=torch.long),
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
    trainer.training_engine = TrainingEngine(
        model_parts=trainer.model_parts,
        parallel_dims=trainer.parallel_dims,
        device=trainer.device,
        chunked_loss_num_chunks=1,
        param_context=trainer.param_context,
        loss_sum=cross_entropy_loss,
    )
    trainer.checkpointer = None
    trainer.ntokens_seen = 7
    logged: dict[str, Any] = {}
    trainer.metrics = SimpleNamespace(
        add_tokens=lambda n: None,
        log_validation=lambda loss, step: logged.update(loss=loss, step=step),
    )
    monkeypatch.setattr(
        "llmtuner.trainer.runtime.validation.build_rank_dataloader",
        lambda *a, **k: loader,
    )
    return trainer, logged


def _expected_loss(batches: list[dict[str, Any]]) -> float:
    loss_sum = sum(
        float(cross_entropy_loss(_EchoModel()(b["input"]), b["labels"].reshape(-1)))
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
    assert TrainingConfig().validation is None


# -- build-time feasibility ---------------------------------------------------


def test_feasibility_rejects_steps_neg1_when_dp_gt_1() -> None:
    with pytest.raises(ValueError, match="validation collectives"):
        check_validation_feasibility(
            ValidationConfig(steps=-1),
            dp_world_size=2,
            training_dataset="local_jsonl",
        )


def test_feasibility_rejects_steps_neg1_on_the_infinite_corpus() -> None:
    with pytest.raises(ValueError, match="infinite synthetic source"):
        check_validation_feasibility(
            ValidationConfig(steps=-1),
            dp_world_size=1,
            training_dataset="random",
        )
    # The override corpus is what the pass reads, so it is the one checked.
    with pytest.raises(ValueError, match="infinite synthetic source"):
        check_validation_feasibility(
            ValidationConfig(steps=-1, dataset="random"),
            dp_world_size=1,
            training_dataset="local_jsonl",
        )


def test_feasibility_accepts_the_terminating_combinations() -> None:
    check_validation_feasibility(
        ValidationConfig(steps=-1),
        dp_world_size=1,
        training_dataset="local_jsonl",
    )
    check_validation_feasibility(
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
def test_validate_restores_existing_and_absent_gradients(monkeypatch, fail):
    from llmtuner.trainer.runtime import validation as validation_runtime

    model = _EchoModel()
    model.register_parameter("existing", torch.nn.Parameter(torch.ones(3)))
    model.register_parameter("absent", torch.nn.Parameter(torch.ones(2)))
    existing_gradient = torch.tensor([2.0, 3.0, 4.0])
    model.existing.grad = existing_gradient
    other_part = torch.nn.Linear(3, 2, bias=False)
    other_gradient = torch.ones_like(other_part.weight)
    other_part.weight.grad = other_gradient
    trainer, _ = _make_trainer(
        monkeypatch,
        validation=ValidationConfig(steps=1),
        loader=_ListLoader([]),
        model=model,
    )
    trainer.model_parts.append(other_part)

    def clear_gradients(*args, **kwargs):
        model.existing.grad = None
        model.absent.grad = torch.ones_like(model.absent)
        other_part.weight.grad = None
        if fail:
            raise RuntimeError("metadata initialization failed")

    monkeypatch.setattr(validation_runtime, "run_validation_loop", clear_gradients)
    if fail:
        with pytest.raises(RuntimeError, match="metadata initialization failed"):
            trainer.validate(1)
    else:
        trainer.validate(1)
    assert model.existing.grad is existing_gradient
    assert model.absent.grad is None
    assert other_part.weight.grad is other_gradient
    torch.testing.assert_close(model.existing.grad, torch.tensor([2.0, 3.0, 4.0]))


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

    def run_validation_loop(active_engine, **kwargs):
        assert active_engine is trainer.training_engine
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


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_validate_restores_modes_after_partial_eval_setup(monkeypatch, error_type):
    from llmtuner.trainer.runtime import validation as validation_runtime

    original = error_type("eval setup failed")

    class FailingEval(torch.nn.Module):
        def train(self, mode=True):
            super().train(mode)
            if not mode:
                raise original
            return self

    model = _EchoModel()
    model.add_module("frozen", torch.nn.Dropout())
    model.add_module("failing", FailingEval())
    model.add_module("untouched", torch.nn.Dropout())
    model.frozen.eval()
    trainer, _ = _make_trainer(
        monkeypatch,
        validation=ValidationConfig(steps=1, dataset="local_jsonl"),
        loader=_ListLoader([]),
        model=model,
    )
    config = SimpleNamespace(attn_mask_type="causal")
    model.model = SimpleNamespace(config=config)
    modes = {module: module.training for module in model.modules()}

    def unexpected(*args, **kwargs):
        pytest.fail("validation loop ran after eval setup failed")

    monkeypatch.setattr(validation_runtime, "run_validation_loop", unexpected)
    with pytest.raises(error_type) as caught:
        trainer.validate(1)

    assert caught.value is original
    assert {module: module.training for module in model.modules()} == modes
    assert config.attn_mask_type == "causal"


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

    def run_loop(active_engine, **kwargs):
        assert config.attn_mask_type == expected
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


def test_kcp_rejects_packed_validation_before_loop(monkeypatch):
    from llmtuner.trainer.runtime import validation as validation_runtime

    trainer, _ = _make_trainer(
        monkeypatch,
        validation=ValidationConfig(steps=1, dataset="local_jsonl"),
        loader=_ListLoader([]),
    )
    trainer.cfg.parallel.context_parallel_kcp_size = 2
    trainer.cfg.parallel.cp_config.context_parallel_strategy = "kcp"

    def unexpected(*args, **kwargs):
        pytest.fail("KCP packed validation must fail before constructing a loader")

    monkeypatch.setattr(validation_runtime, "run_validation_loop", unexpected)
    with pytest.raises(ValueError, match="KCP.*packed"):
        trainer.validate(1)
    assert trainer.model.training


@pytest.mark.parametrize("forward_fails", [False, True])
@pytest.mark.parametrize("report_error", [None, OSError, KeyboardInterrupt, SystemExit])
def test_validation_preserves_forward_error_when_close_fails(
    monkeypatch, forward_fails, report_error
):
    from llmtuner.trainer.runtime import validation as validation_runtime

    if report_error is not None:

        def report(*args, **kwargs):
            raise report_error("validation cleanup logger failed")

        monkeypatch.setattr(
            validation_runtime,
            "get_logger",
            lambda name: SimpleNamespace(exception=report),
        )

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


def test_validation_closes_loader_after_token_allocation_failure(monkeypatch):
    import llmtuner.trainer.runtime.validation as validation_module

    original = RuntimeError("token allocation failed")
    closed = []

    class Loader:
        def close(self):
            closed.append(True)
            raise ValueError("cleanup failed")

    def fail_allocation(*args, **kwargs):
        raise original

    monkeypatch.setattr(
        validation_module, "resolve_loss_reduction_meshes", lambda dims: (None, None)
    )
    monkeypatch.setattr(
        validation_module, "create_validation_dataloader", lambda *args: Loader()
    )
    monkeypatch.setattr(validation_module.torch, "zeros", fail_allocation)
    trainer = SimpleNamespace(
        error_group=None,
        cfg=None,
        parallel_dims=None,
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="token allocation failed") as caught:
        validation_module.run_validation_loop(
            trainer,
            loader_resources=ValidationDataLoaderResources(),
            cfg=trainer.cfg,
            metrics=None,
            token_counter=lambda batch: 0,
            validation=ValidationConfig(),
            step=1,
        )
    assert caught.value is original
    assert closed == [True]


@pytest.mark.parametrize("pass_failed", [False, True])
def test_validation_loader_owner_retries_failed_close(pass_failed):
    calls = []

    class Loader:
        def close(self):
            calls.append("close")
            if len(calls) == 1:
                raise RuntimeError("temporary close failure")

    resources = ValidationDataLoaderResources()
    loader = Loader()
    resources.acquire(loader)
    if pass_failed:
        resources.close_dataloader(loader, failed=True)
    else:
        with pytest.raises(RuntimeError, match="temporary close failure"):
            resources.close_dataloader(loader, failed=False)
    resources.close()
    resources.close()
    assert calls == ["close", "close"]
