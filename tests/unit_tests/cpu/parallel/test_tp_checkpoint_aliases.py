"""Every serialized alias of a TP slice must use the same global layout."""

import pytest
import torch
from torch import nn

from llmtuner.parallel.tensor_parallel import checkpointing as tp


@pytest.mark.parametrize("shared_module", [False, True])
def test_aliases_round_trip_global_tp_state(monkeypatch, shared_module):
    model = nn.Module()
    model.first = nn.Linear(3, 2, bias=False)
    model.second = model.first if shared_module else nn.Linear(3, 2, bias=False)
    model.second.weight = model.first.weight
    group = object()
    if shared_module:
        model.first._tp_checkpoint_layouts = {"weight": (0, False, group)}
    else:
        model._tp_checkpoint_layouts = {"first.weight": (0, False, group)}
    monkeypatch.setattr(tp.dist, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(tp.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(tp.dist, "get_rank", lambda group: 0)

    def reports(output, report, group):
        output[:] = [report, report]

    def gather(output, value, group):
        output[0].copy_(value)
        output[1].copy_(value + 10)

    monkeypatch.setattr(tp.dist, "all_gather_object", reports)
    monkeypatch.setattr(tp.dist, "all_gather", gather)
    local = model.state_dict()
    full = tp.gather_tensor_parallel_state(local, model_parts=[model])
    expected = torch.cat((local["first.weight"], local["first.weight"] + 10))
    for name in local:
        torch.testing.assert_close(full[name], expected, rtol=0, atol=0)
    restored = tp.load_tensor_parallel_state(full, model_parts=[model])
    for name in local:
        torch.testing.assert_close(restored[name], local[name], rtol=0, atol=0)


def test_conflicting_alias_layouts_are_rejected():
    model = nn.Module()
    model.first = nn.Parameter(torch.ones(2, 3))
    model.second = model.first
    group = object()
    model._tp_checkpoint_layouts = {
        "first": (0, False, group),
        "second": (1, False, group),
    }
    with pytest.raises(ValueError, match="Shared parameter.*conflicting"):
        tp._index_layouts([model])


def test_stale_layout_path_is_rejected():
    model = nn.Linear(3, 2, bias=False)
    model._tp_checkpoint_layouts = {"removed.weight": (0, False, object())}
    with pytest.raises(ValueError, match="no live parameter"):
        tp._index_layouts([model])
