from contextlib import contextmanager

import pytest

import llmtuner.components.checkpointer.core.state_adaptation as state_adaptation


class _Adapter:
    kimi_expert_parallel = False

    def from_hf(self, state):
        return {key.removeprefix("hf."): value for key, value in state.items()}


class _ModelState:
    def __init__(self, error=None):
        self.error = error
        self.validated = None

    def validate_state_dict(self, state):
        self.validated = state
        if self.error is not None:
            raise self.error


def test_hf_state_adaptation_coordinates_local_conversion_and_validation(
    monkeypatch,
):
    phases = []

    @contextmanager
    def coordinate(phase):
        phases.append(phase)
        yield

    monkeypatch.setattr(state_adaptation, "synchronize_local_errors", coordinate)
    model_state = _ModelState()
    weight = object()
    converted = state_adaptation.adapt_hf_model_state(
        _Adapter(), model_state, {"hf.weight": weight}
    )

    assert converted == {"weight": weight}
    assert model_state.validated is converted
    assert phases == [
        "HF checkpoint state adaptation",
        "HF checkpoint model validation",
    ]


def test_hf_state_adaptation_preserves_kimi_collective_boundary(monkeypatch):
    phases = []

    @contextmanager
    def coordinate(phase):
        phases.append(phase)
        yield

    class KimiAdapter(_Adapter):
        kimi_expert_parallel = True

    monkeypatch.setattr(state_adaptation, "synchronize_local_errors", coordinate)
    model_state = _ModelState()
    converted = state_adaptation.adapt_hf_model_state(
        KimiAdapter(), model_state, {"hf.weight": 1}
    )

    assert converted == {"weight": 1}
    assert model_state.validated is converted
    assert phases == ["HF checkpoint model validation"]


def test_hf_state_adaptation_propagates_destination_validation_failure():
    failure = ValueError("invalid destination")

    with pytest.raises(ValueError, match="invalid destination") as caught:
        state_adaptation.adapt_hf_model_state(
            _Adapter(), _ModelState(error=failure), {"hf.weight": 1}
        )

    assert caught.value is failure
