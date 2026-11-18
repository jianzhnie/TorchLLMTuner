from contextlib import contextmanager

import pytest

import llmtuner.components.checkpointer.core.state_adaptation as state_adaptation


class _Adapter:
    coordinates_state_conversion = False

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
    def coordinate(phase, *, group=None):
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
    def coordinate(phase, *, group=None):
        phases.append(phase)
        yield

    class KimiAdapter(_Adapter):
        coordinates_state_conversion = True

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


@pytest.mark.parametrize("coordinates_state_conversion", [False, True])
def test_failed_conversion_never_validates_destination(coordinates_state_conversion):
    failure = ValueError("invalid HF state")

    class FailedAdapter:
        def from_hf(self, state):
            raise failure

    adapter = FailedAdapter()
    adapter.coordinates_state_conversion = coordinates_state_conversion
    destination = _ModelState()
    with pytest.raises(ValueError, match="invalid HF state") as caught:
        state_adaptation.adapt_hf_model_state(adapter, destination, {"hf.weight": 1})

    assert caught.value is failure
    assert destination.validated is None


@pytest.mark.parametrize("coordinates_conversion", [False, True])
def test_explicit_error_group_reaches_local_boundaries(
    monkeypatch, coordinates_conversion
):
    expected_group = object()
    phases = []

    @contextmanager
    def coordinate(phase, *, group=None):
        assert group is expected_group
        phases.append(phase)
        yield

    adapter = _Adapter()
    adapter.coordinates_state_conversion = coordinates_conversion
    destination = _ModelState()
    monkeypatch.setattr(state_adaptation, "synchronize_local_errors", coordinate)
    converted = state_adaptation.adapt_hf_model_state(
        adapter, destination, {"hf.weight": 3}, error_group=expected_group
    )
    assert destination.validated is converted
    assert converted == {"weight": 3}
    expected = ["HF checkpoint model validation"]
    if not coordinates_conversion:
        expected.insert(0, "HF checkpoint state adaptation")
    assert phases == expected


def test_rng_peer_validation_failure_prevents_local_restore(monkeypatch):
    calls = []

    class RNG:
        def validate_state_dict(self, state):
            calls.append("validate")

        def load_state_dict(self, state):
            pytest.fail("A peer rejected RNG; local generators must stay unchanged")

    @contextmanager
    def coordinate(phase):
        yield
        if phase.endswith("RNG validation"):
            raise RuntimeError("peer rejected RNG")

    monkeypatch.setattr(state_adaptation, "synchronize_local_errors", coordinate)
    with pytest.raises(RuntimeError, match="peer rejected RNG"):
        state_adaptation.restore_rng_state({"rng_state": RNG()}, {"rng_state": {}})
    assert calls == ["validate"]


def test_requested_rng_missing_from_checkpoint_rejects_before_restoration(monkeypatch):
    phases = []

    class RNG:
        def validate_state_dict(self, state):
            pytest.fail("Missing RNG must not be validated or restored")

        load_state_dict = validate_state_dict

    @contextmanager
    def coordinate(phase):
        phases.append(phase)
        yield

    monkeypatch.setattr(state_adaptation, "synchronize_local_errors", coordinate)
    with pytest.raises(KeyError, match="rng_state"):
        state_adaptation.restore_rng_state({"rng_state": RNG()}, {})
    assert phases == ["Checkpoint RNG validation"]


@pytest.mark.parametrize("failure_phase", [None, "reader", "validation"])
def test_hf_restore_writes_model_only_after_successful_read_and_validation(
    failure_phase,
):
    failure = ValueError("invalid HF restore")
    calls = []

    class Adapter(_Adapter):
        def to_hf(self, state):
            return {"hf.weight": state["weight"]}

        def from_hf(self, state):
            calls.append("convert")
            return super().from_hf(state)

    class ModelState(_ModelState):
        def load_state_dict(self, state):
            assert self.validated is state
            calls.append(("load", state))

    model = ModelState(error=failure if failure_phase == "validation" else None)

    def restore():
        with state_adaptation.hf_model_restore(
            Adapter(), model, {"weight": 1}
        ) as destination:
            calls.append("read")
            if failure_phase == "reader":
                raise failure
            destination["hf.weight"] = 2

    if failure_phase is None:
        restore()
        assert calls == ["read", "convert", ("load", {"weight": 2})]
    else:
        with pytest.raises(ValueError, match="invalid HF restore") as caught:
            restore()
        assert caught.value is failure
        assert calls == (["read"] if failure_phase == "reader" else ["read", "convert"])


@pytest.mark.parametrize("export_indices", [None, {"export.weight": 2}])
@pytest.mark.parametrize("legacy_interface", [False, True])
def test_hf_export_uses_export_conversion_and_explicit_index(
    export_indices, legacy_interface
):
    from types import SimpleNamespace

    def unexpected(state):
        pytest.fail("Export must use its model-owned conversion")

    adapter = SimpleNamespace(
        to_hf_export=lambda state: {"export.weight": state["weight"]},
        hf_export_fqn_to_index_mapping=export_indices,
    )
    if legacy_interface:
        adapter.to_hf = unexpected
        adapter.fqn_to_index_mapping = {"import.weight": 1}
    converted, indices = state_adaptation.prepare_hf_export(adapter, {"weight": 3})
    assert converted == {"export.weight": 3}
    assert indices is export_indices


def test_hf_export_supports_adapter_without_separate_export_interface():
    from types import SimpleNamespace

    adapter = SimpleNamespace(
        to_hf=lambda state: {"hf.weight": state["weight"]},
        fqn_to_index_mapping={"hf.weight": 1},
    )
    converted, indices = state_adaptation.prepare_hf_export(adapter, {"weight": 3})
    assert converted == {"hf.weight": 3}
    assert indices is adapter.fqn_to_index_mapping


def test_restore_callbacks_load_dependencies_and_leave_rng_for_last():
    calls = []

    class State:
        def __init__(self, name):
            self.name = name

        def state_dict(self):
            return {}

        def load_state_dict(self, value):
            calls.append((self.name, value))

    states = {
        name: State(name)
        for name in ("lr_scheduler", "rng_state", "optimizer", "model", "ema")
    }
    destination = {name: object() for name in states}
    state_adaptation.restore_checkpoint_callbacks(states, destination)
    assert calls == [
        (name, destination[name])
        for name in ("model", "optimizer", "lr_scheduler", "ema")
    ]


def test_restore_callbacks_reject_reader_replacement_of_non_stateful_value():
    target = {}
    state_adaptation.restore_checkpoint_callbacks(
        {"custom": target}, {"custom": target}
    )
    with pytest.raises(TypeError, match="Cannot restore non-Stateful"):
        state_adaptation.restore_checkpoint_callbacks(
            {"custom": target}, {"custom": target}, loaded={"custom": {}}
        )
