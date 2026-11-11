"""Kernel registration and backend contracts."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.kernel import get_kernel, register_kernel
from llmtuner.kernel.npu import basic as npu_basic
from llmtuner.kernel.ops.attention import fusion_attention
from llmtuner.kernel.ops.moe import moe_token_permute
from llmtuner.kernel.ops.rotary import rotary_mul


def test_missing_backend_requires_explicit_cpu_fallback():
    with pytest.raises(KeyError, match="not available"):
        get_kernel("rms_norm", "meta")
    assert get_kernel("rms_norm", "meta", fallback_to_cpu=True) is get_kernel(
        "rms_norm", "cpu"
    )


def test_duplicate_registration_does_not_replace_backend():
    def first():
        pass

    def second():
        pass

    register_kernel("duplicate_test", "cpu")(first)
    with pytest.raises(ValueError, match="already registered"):
        register_kernel("duplicate_test", "cpu")(second)
    assert get_kernel("duplicate_test", "cpu") is first


def test_fusion_attention_normalizes_sequence_length_containers(monkeypatch):
    calls = []

    def fake_attention(*args, **kwargs):
        calls.append(kwargs)
        return args[0], None

    monkeypatch.setattr(npu_basic, "_require_npu", lambda *args: None)
    monkeypatch.setattr(
        npu_basic, "torch_npu", SimpleNamespace(npu_fusion_attention=fake_attention)
    )
    query = torch.empty(8, 2, 16)
    lengths = torch.tensor([0, 4, 8], dtype=torch.int32)
    for q_lengths, k_lengths in ((lengths, lengths), ([0, 4, 8], [0, 4, 8])):
        output = npu_basic.fusion_attention_npu(
            query,
            query,
            query,
            head_num=2,
            scale=0.25,
            q_cu_seqlens=q_lengths,
            k_cu_seqlens=k_lengths,
        )
        assert output is query
    assert [call["actual_seq_qlen"] for call in calls] == [[4, 8]] * 2
    assert [call["actual_seq_kvlen"] for call in calls] == [[4, 8]] * 2


def test_cpu_moe_permutation_preserves_stable_expert_order():
    values = torch.arange(24).reshape(4, 6)
    ids = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 0]])
    actual = moe_token_permute(values, ids, use_eager=True)
    assignments = torch.argsort(ids.reshape(-1), stable=True)
    torch.testing.assert_close(actual, values[assignments // 2])


def test_cpu_interleaved_rotary_matches_complex_expression():
    values = torch.randn(3, 2, 8, dtype=torch.float64)
    angles = torch.randn(3, 1, 4, dtype=torch.float64)
    cosine = angles.cos().repeat_interleave(2, dim=-1)
    sine = angles.sin().repeat_interleave(2, dim=-1)
    actual = rotary_mul(values, cosine, sine, rotary_mode="interleave", use_eager=True)
    expected = torch.view_as_real(
        torch.view_as_complex(values.reshape(3, 2, 4, 2))
        * torch.polar(torch.ones_like(angles), angles)
    ).flatten(-2)
    torch.testing.assert_close(actual, expected)


def test_cpu_packed_attention_uses_document_boundaries():
    query = torch.randn(6, 2, 4, dtype=torch.float64)
    key = torch.randn(6, 2, 4, dtype=torch.float64)
    value = torch.randn(6, 2, 4, dtype=torch.float64)
    actual = fusion_attention(
        query,
        key,
        value,
        head_num=2,
        scale=0.5,
        q_cu_seqlens=[0, 2, 6],
        k_cu_seqlens=[0, 2, 6],
        use_eager=True,
    )
    expected = torch.empty_like(actual)
    for start, end in ((0, 2), (2, 6)):
        probabilities = torch.softmax(
            query[start:end].transpose(0, 1)
            @ key[start:end].transpose(0, 1).transpose(-2, -1)
            * 0.5,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        expected[start:end] = (
            probabilities @ value[start:end].transpose(0, 1)
        ).transpose(0, 1)
    torch.testing.assert_close(actual, expected)
