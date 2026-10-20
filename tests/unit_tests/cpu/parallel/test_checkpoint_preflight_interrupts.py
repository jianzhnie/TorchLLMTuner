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
