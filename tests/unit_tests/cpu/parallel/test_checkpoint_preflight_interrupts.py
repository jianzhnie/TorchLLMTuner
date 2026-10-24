"""Local interruption must be reported before checkpoint tensor collectives."""

import pytest
import torch
from torch import nn

from llmtuner.parallel.expert_parallel import checkpointing as ep
from llmtuner.parallel.tensor_parallel import checkpointing as tp


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("operation", ["tp_save", "tp_load", "ep_save"])
def test_checkpoint_preflight_reports_interrupt_before_reraising(
    monkeypatch, operation, error_type
):
    model = nn.Linear(2, 2, bias=False)
    group = object()
    original = error_type("interrupted local checkpoint planning")
    reports = []

    def fail(*args):
        raise original

    def exchange(output, report, *, group):
        reports.append(report)
        output[:] = [report, (True, [])]

    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1])
    monkeypatch.setattr(torch.distributed, "all_gather_object", exchange)
    state = {"weight": torch.ones(2, 2)}
    if operation.startswith("tp"):
        model._tp_checkpoint_layouts = {"weight": (0, False, group)}
        monkeypatch.setattr(tp, "_index_layouts", fail)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda _: 2)
        call = (
            tp.load_tensor_parallel_state
            if operation == "tp_load"
            else tp.gather_tensor_parallel_state
        )
        kwargs = {"model_parts": [model]}
    else:
        monkeypatch.setattr(ep, "index_checkpoint_parameters", fail)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(ep, "get_world_size", lambda _: 2)
        call = ep.gather_expert_state
        kwargs = {"model_parts": [model], "shard_map": {"weight": group}}
    with pytest.raises(error_type) as caught:
        call(state, **kwargs)
    assert caught.value is original
    assert reports == [(False, [])]


@pytest.mark.parametrize("loading", [False, True])
@pytest.mark.parametrize("failure", ["invalid_dimension", "peer_error", "manifest"])
def test_tp_checkpoint_preflight_rejects_invalid_plan_before_tensor_collectives(
    monkeypatch, loading, failure
):
    model = nn.Linear(2, 2, bias=False)
    group = object()
    dimension = 2 if failure == "invalid_dimension" else 0
    model._tp_checkpoint_layouts = {"weight": (dimension, False, group)}
    reports = []

    def exchange(output, report, *, group):
        reports.append(report)
        peer = report
        if failure == "peer_error":
            peer = (False, [])
        elif failure == "manifest":
            peer = (True, [])
        output[:] = [report, peer]

    def unexpected_collective(*args, **kwargs):
        pytest.fail("Invalid checkpoint plan reached tensor communication")

    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1])
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", exchange)
    monkeypatch.setattr(torch.distributed, "all_gather", unexpected_collective)
    monkeypatch.setattr(torch.distributed, "get_rank", unexpected_collective)
    call = tp.load_tensor_parallel_state if loading else tp.gather_tensor_parallel_state
    state = {"weight": torch.ones(4 if loading else 2, 2)}
    expected_type, expected_message = {
        "invalid_dimension": (ValueError, "Invalid TP checkpoint dimension"),
        "peer_error": (
            RuntimeError,
            "TP checkpoint preparation failed on another rank",
        ),
        "manifest": (
            ValueError,
            "TP checkpoint entries or layouts differ across ranks",
        ),
    }[failure]
    with pytest.raises(expected_type, match=expected_message):
        call(state, model_parts=[model])
    assert len(reports) == 1
    assert reports[0][0] is (failure != "invalid_dimension")
