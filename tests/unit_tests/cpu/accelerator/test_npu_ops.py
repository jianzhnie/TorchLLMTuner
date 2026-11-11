"""Ascend bindings preserve vendor API contracts without model dependencies."""

from types import SimpleNamespace

import torch

from llmtuner.accelerator import npu_ops


def test_fusion_attention_normalizes_sequence_length_containers(monkeypatch):
    calls = []

    def fake_attention(*args, **kwargs):
        calls.append(kwargs)
        return args[0], None

    monkeypatch.setattr(npu_ops, "_require_npu", lambda *args: None)
    monkeypatch.setattr(
        npu_ops, "torch_npu", SimpleNamespace(npu_fusion_attention=fake_attention)
    )
    query = torch.empty(8, 2, 16)
    lengths = torch.tensor([0, 4, 8], dtype=torch.int32)
    for q_lengths, k_lengths in ((lengths, lengths), ([0, 4, 8], [0, 4, 8])):
        output = npu_ops.fusion_attention(
            query,
            query,
            query,
            head_num=2,
            scale=0.25,
            q_cu_seqlens=q_lengths,
            k_cu_seqlens=k_lengths,
        )
        assert output is query
    assert [call["actual_seq_qlen"] for call in calls] == [[0, 4, 8]] * 2
    assert [call["actual_seq_kvlen"] for call in calls] == [[0, 4, 8]] * 2
    assert all(call["input_layout"] == "TND" for call in calls)
