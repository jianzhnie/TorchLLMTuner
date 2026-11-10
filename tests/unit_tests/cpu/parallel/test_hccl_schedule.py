from types import SimpleNamespace

import pytest
import torch.distributed as dist
from torch.distributed.pipelining.schedules import Schedule1F1B

from llmtuner.errors import EnvironmentUnsupportedError, UnsupportedCombinationError
from llmtuner.parallel.pipeline_parallel.hccl_schedule import (
    hccl_1f1b_schedule_class,
    ordered_p2p,
)


@pytest.mark.parametrize("rank,expected", [(0, "ssrr"), (2, "rrss")])
def test_ordered_submission_preserves_message_contract(monkeypatch, rank, expected):
    calls = []

    def send(tensor, **kwargs):
        calls.append(("s", tensor, kwargs))
        return tensor

    def recv(tensor, **kwargs):
        calls.append(("r", tensor, kwargs))
        return tensor

    monkeypatch.setattr(dist, "isend", send)
    monkeypatch.setattr(dist, "irecv", recv)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    group = object()
    ops = [
        SimpleNamespace(
            op=op,
            tensor=index,
            group=group,
            tag=10 + index,
            peer=2 if rank == 0 else 0,
            group_peer=1 if rank == 0 else 0,
        )
        for index, op in enumerate([send, recv, send, recv])
    ]
    works = ordered_p2p(ops)
    assert "".join(call[0] for call in calls) == expected
    assert [tensor for kind, tensor, _ in calls if kind == "s"] == [0, 2]
    assert [tensor for kind, tensor, _ in calls if kind == "r"] == [1, 3]
    assert works == [call[1] for call in calls]
    for kind, tensor, kwargs in calls:
        assert kwargs == {
            "group": group,
            "tag": 10 + tensor,
            "group_dst" if kind == "s" else "group_src": 1 if rank == 0 else 0,
        }


def test_multiple_peers_rejected_before_submission():
    ops = [SimpleNamespace(peer=peer, op=dist.isend) for peer in (1, 2)]
    with pytest.raises(UnsupportedCombinationError, match="one peer"):
        ordered_p2p(ops)
    assert ordered_p2p([]) == []


def test_different_groups_rejected_before_submission():
    ops = [SimpleNamespace(peer=1, group=object(), op=dist.isend) for _ in range(2)]
    with pytest.raises(UnsupportedCombinationError, match="one group"):
        ordered_p2p(ops)


def test_schedule_binding_keeps_upstream_code_and_globals_unchanged():
    original = Schedule1F1B._step_microbatches
    original_submit = original.__globals__["_batch_p2p"]
    adapted = hccl_1f1b_schedule_class()._step_microbatches
    assert adapted.__code__ is original.__code__
    assert adapted.__defaults__ is original.__defaults__
    assert adapted.__globals__ is not original.__globals__
    assert adapted.__globals__["_batch_p2p"] is ordered_p2p
    assert original.__globals__["_batch_p2p"] is original_submit
    assert Schedule1F1B._step_microbatches is original


def test_changed_upstream_boundary_fails_explicitly(monkeypatch):
    def changed(self, *args):
        raise AssertionError("The incompatible schedule must not execute")

    hccl_1f1b_schedule_class.cache_clear()
    monkeypatch.setattr(Schedule1F1B, "_step_microbatches", changed)
    with pytest.raises(EnvironmentUnsupportedError, match="compatibility adapter"):
        hccl_1f1b_schedule_class()
