"""NPU pipeline metadata uses sequential size/payload object transport."""

from types import SimpleNamespace

import torch

from llmtuner.parallel.pipeline_parallel.partition import ComposablePipelineStage


def test_npu_metadata_preserves_peer_group_and_object(monkeypatch):
    stage = SimpleNamespace(
        device=SimpleNamespace(type="npu", index=0),
        group=object(),
        _resolve_peer_global_rank=lambda index: index + 4,
    )
    calls = []

    def send(objects, **kwargs):
        calls.append((objects, kwargs))

    def recv(objects, **kwargs):
        calls.append((list(objects), kwargs))
        objects[0] = {"shape": (1, 8, 16)}

    monkeypatch.setattr(torch.distributed, "send_object_list", send)
    monkeypatch.setattr(torch.distributed, "recv_object_list", recv)
    payload = {"shape": (1, 8, 16)}
    ComposablePipelineStage._send_meta(stage, payload, 1)
    assert ComposablePipelineStage._recv_meta(stage, 0) == payload
    assert calls[0][0] == [payload]
    assert calls[0][1]["dst"] == 5
    assert calls[1][1]["src"] == 4
    for _, kwargs in calls:
        assert kwargs["group"] is stage.group
        assert kwargs["device"] == stage.device
        assert kwargs["use_batch"] is False
