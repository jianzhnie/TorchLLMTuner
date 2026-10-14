"""Model checkpoint state aggregation, cache lifetime, and extra state."""

import pytest
import torch
from torch import nn

from llmtuner.components.checkpointer.model_state import ModelCheckpointState


def test_model_checkpoint_state_keeps_tensor_storage_stable_across_calls() -> None:
    """Stable storage is what lets async DCP reuse its pinned host buffers."""
    model = nn.Linear(4, 4)
    wrapper = ModelCheckpointState(model)

    first = wrapper.state_dict()
    storages = {k: v.untyped_storage().data_ptr() for k, v in first.items()}
    with torch.no_grad():
        model.weight.add_(1)

    second = wrapper.state_dict()

    assert {k: v.untyped_storage().data_ptr() for k, v in second.items()} == storages
    assert torch.equal(second["weight"], model.weight)


def test_model_checkpoint_state_refreshes_and_restores_extra_state() -> None:
    class WithExtraState(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.counter = 1

        def get_extra_state(self):
            return {"counter": self.counter}

        def set_extra_state(self, state):
            self.counter = state["counter"]

    model = WithExtraState()
    state = ModelCheckpointState(model)
    assert state.state_dict()["_extra_state"] == {"counter": 1}
    model.counter = 2
    saved = state.state_dict()
    assert saved["_extra_state"] == {"counter": 2}
    model.counter = 3
    state.load_state_dict(saved)
    assert model.counter == 2
    assert state.state_dict()["_extra_state"] == {"counter": 2}


@pytest.mark.parametrize("change", ["add", "remove"])
def test_model_structure_changes_fail_before_cache_refresh(change) -> None:
    model = nn.Linear(2, 2)
    model.register_buffer("counter", torch.zeros(1))
    state = ModelCheckpointState(model)
    original_keys = set(state.cached_state_dict)
    if change == "add":
        model.register_buffer("new_counter", torch.ones(1))
    else:
        del model.counter
    with pytest.raises(RuntimeError, match="checkpoint keys changed"):
        state.state_dict()
    assert set(state.cached_state_dict) == original_keys


def test_checkpoint_owns_model_part_collection() -> None:
    first = nn.ModuleDict({"first": nn.Linear(2, 2)})
    second = nn.ModuleDict({"second": nn.Linear(2, 2)})
    parts = [first, second]
    state = ModelCheckpointState(parts)
    saved = {key: value.clone() for key, value in state.state_dict().items()}
    parts.clear()
    with torch.no_grad():
        first["first"].weight.zero_()
        second["second"].weight.zero_()
    state.load_state_dict(saved)
    assert torch.equal(first["first"].weight, saved["first.weight"])
    assert torch.equal(second["second"].weight, saved["second.weight"])
    assert set(state.state_dict()) == set(saved)


@pytest.mark.parametrize("parts", [[], ()])
def test_empty_model_parts_preserve_manager_compatibility(parts) -> None:
    state = ModelCheckpointState(parts)
    assert state.state_dict() == {}
    state.load_state_dict({})


def test_invalid_model_part_is_rejected_before_shard_inspection() -> None:
    with pytest.raises(TypeError, match="nn.Module"):
        ModelCheckpointState([nn.Linear(2, 2), object()])


def test_overlapping_model_parts_are_rejected_instead_of_overwriting() -> None:
    with pytest.raises(ValueError, match="overlap at part 1.*bias.*weight"):
        ModelCheckpointState([nn.Linear(2, 2), nn.Linear(2, 2)])


def test_buffer_key_collision_fails_before_refreshing_checkpoint_cache() -> None:
    first = nn.ModuleDict({"first": nn.Linear(2, 2)})
    second = nn.ModuleDict({"second": nn.Linear(2, 2)})
    first.register_buffer("counter", torch.tensor(1))
    state = ModelCheckpointState([first, second])
    cached = state.cached_state_dict.copy()
    second.register_buffer("counter", torch.tensor(2))

    with pytest.raises(ValueError, match="overlap at part 1.*counter"):
        state.state_dict()
    assert state.cached_state_dict.keys() == cached.keys()
    assert all(state.cached_state_dict[key] is value for key, value in cached.items())


@pytest.mark.parametrize("invalid", ["missing", "shape", "type"])
def test_model_restore_validates_all_parts_before_copying_weights(invalid):
    first = nn.ModuleDict({"first": nn.Linear(2, 2)})
    second = nn.ModuleDict({"second": nn.Linear(2, 2)})
    state = ModelCheckpointState([first, second])
    original = {key: value.clone() for key, value in state.state_dict().items()}
    saved = {key: value + 1 for key, value in original.items()}
    if invalid == "missing":
        del saved["second.weight"]
    elif invalid == "shape":
        saved["second.weight"] = torch.zeros(3, 2)
    else:
        saved["second.weight"] = None
    with pytest.raises(ValueError, match="second.weight"):
        state.load_state_dict(saved)
    torch.testing.assert_close(state.state_dict(), original, rtol=0, atol=0)


def test_model_restore_allows_auxiliary_checkpoint_categories():
    model = nn.Linear(2, 2)
    state = ModelCheckpointState(model)
    saved = {key: value.clone() + 1 for key, value in state.state_dict().items()}
    state.load_state_dict({**saved, "optimizer": {"step": 2}})
    torch.testing.assert_close(state.state_dict(), saved, rtol=0, atol=0)


def test_model_restore_keeps_legacy_scalar_buffer_compatibility():
    model = nn.Module()
    model.register_buffer("counter", torch.tensor(0))
    state = ModelCheckpointState(model)
    state.load_state_dict({"counter": torch.tensor([3])})
    assert model.counter.ndim == 0
    assert model.counter.item() == 3
