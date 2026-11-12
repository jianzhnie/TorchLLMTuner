"""Storage allocation has no model inspection or implicit initialization."""

import pytest
import torch
from torch import nn

from llmtuner.memory.materialization import materialize_meta_model


def meta_model():
    with torch.device("meta"):
        model = nn.Module()
        model.input = nn.Embedding(4, 2)
        model.output = nn.Linear(2, 4, bias=False)
        model.output.weight = model.input.weight
        model.register_buffer("counter", torch.zeros(2), persistent=False)
    return model


def test_allocation_preserves_parameter_identity_rng_and_shared_storage():
    model = meta_model()
    parameter = model.input.weight
    counter = torch.zeros(2)
    rng = torch.get_rng_state().clone()
    materialize_meta_model(
        model, torch.device("cpu"), runtime_buffers={(model, "counter"): counter}
    )
    assert model.input.weight is model.output.weight is parameter
    assert not parameter.is_meta
    assert model.counter is counter
    assert torch.equal(rng, torch.get_rng_state())


def test_unknown_runtime_buffer_fails_before_allocation():
    model = meta_model()
    parameter = model.input.weight
    with pytest.raises(RuntimeError, match="non-checkpoint buffers"):
        materialize_meta_model(model, torch.device("cpu"))
    assert model.input.weight is parameter
    assert parameter.is_meta
    assert model.counter.is_meta


@pytest.mark.parametrize("kind", ["wrong_shape", "wrong_dtype", "meta", "foreign"])
def test_invalid_buffer_reconstruction_fails_before_allocation(kind):
    model = meta_model()
    owner = nn.Module() if kind == "foreign" else model
    value = torch.zeros(3 if kind == "wrong_shape" else 2)
    if kind == "wrong_dtype":
        value = value.double()
    if kind == "meta":
        value = value.to("meta")
    with pytest.raises(ValueError, match="buffer"):
        materialize_meta_model(
            model, torch.device("cpu"), runtime_buffers={(owner, "counter"): value}
        )
    assert model.input.weight.is_meta


def test_runtime_buffer_aliases_are_reconstructed_once():
    model = meta_model()
    model.register_buffer("alias", model.counter, persistent=False)
    value = torch.zeros(2)
    materialize_meta_model(
        model, torch.device("cpu"), runtime_buffers={(model, "counter"): value}
    )
    assert model.counter is model.alias is value


def test_conflicting_runtime_alias_values_fail_before_allocation():
    model = meta_model()
    model.register_buffer("alias", model.counter, persistent=False)
    with pytest.raises(ValueError, match="Conflicting runtime buffer aliases"):
        materialize_meta_model(
            model,
            torch.device("cpu"),
            runtime_buffers={
                (model, "counter"): torch.zeros(2),
                (model, "alias"): torch.ones(2),
            },
        )
    assert model.input.weight.is_meta
    assert model.counter is model.alias


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
def test_distinct_meta_storage_views_reject_before_materialization(kind):
    with torch.device("meta"):
        model = nn.Module()
        model.weight = nn.Parameter(torch.empty(8))
        view = model.weight.detach()[2:6]
        if kind == "parameter":
            model.view = nn.Parameter(view)
        else:
            model.register_buffer("view", view)
    parameter = model.weight
    alias = model.view
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="distinct tensor views"):
        materialize_meta_model(model, torch.device("cpu"))
    assert model.weight is parameter and model.view is alias
    assert parameter.is_meta and alias.is_meta
    assert parameter.untyped_storage()._cdata == alias.untyped_storage()._cdata
    assert torch.equal(torch.get_rng_state(), rng)


def test_parameter_cannot_replace_runtime_buffer_ownership():
    model = meta_model()
    parameter = model.input.weight
    buffer = model.counter
    names = tuple(model.state_dict())
    with pytest.raises(ValueError, match="runtime buffer"):
        materialize_meta_model(
            model,
            torch.device("cpu"),
            runtime_buffers={(model, "counter"): nn.Parameter(torch.zeros(2))},
        )
    assert model.input.weight is parameter and parameter.is_meta
    assert model.counter is buffer and buffer.is_meta
    assert "counter" not in model._parameters
    assert tuple(model.state_dict()) == names
