"""Registered ownership inspection must not execute model checkpoint hooks."""

import torch
from torch import nn

from llmtuner.models.registered_state import (
    named_extra_state_owners,
    named_persistent_buffers,
    registered_state_paths,
)


def test_registered_state_preserves_aliases_without_executing_hooks():
    class Stateful(nn.Module):
        def get_extra_state(self):
            raise AssertionError("Ownership inspection invoked an extra-state getter")

    child = Stateful()
    child.weight = nn.Parameter(torch.ones(2))
    child.register_buffer("saved", torch.zeros(2))
    child.register_buffer("alias", child.saved)
    child.register_buffer("scratch", torch.zeros(2), persistent=False)
    child.register_buffer("absent", None)
    model = nn.ModuleDict({"first": child, "second": child})

    def reject_hook(*args):
        raise AssertionError("Ownership inspection invoked a state hook")

    child.register_state_dict_pre_hook(reject_hook)
    assert dict(named_persistent_buffers(model)).keys() == {
        "first.saved",
        "first.alias",
        "second.saved",
        "second.alias",
    }
    assert all(buffer is child.saved for _, buffer in named_persistent_buffers(model))
    assert dict(named_extra_state_owners(model)) == {
        "first._extra_state": child,
        "second._extra_state": child,
    }
    assert set(registered_state_paths(model)) == {
        "first.weight",
        "second.weight",
        "first.saved",
        "first.alias",
        "second.saved",
        "second.alias",
        "first._extra_state",
        "second._extra_state",
    }
