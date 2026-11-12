"""Device selection and call contracts, without claiming accelerator execution."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.errors import EnvironmentUnsupportedError
from llmtuner.kernel.ops import varlen_attention as kernels
from llmtuner.models.kimi.kimi_k3 import multimodal


def test_cpu_flash_path_is_rejected_without_silent_reference():
    values = torch.randn(3, 2, 4)
    with pytest.raises(EnvironmentUnsupportedError, match="does not support cpu"):
        kernels.varlen_attention(values, values, values, None, None, None, None)


def test_missing_cuda_runtime_preserves_cause(monkeypatch):
    cause = ImportError("missing FlashAttention")
    monkeypatch.setattr(kernels, "flash_attn_varlen_func", None)
    monkeypatch.setattr(kernels, "FLASH_ATTN_IMPORT_ERROR", cause)
    query = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    with pytest.raises(EnvironmentUnsupportedError, match="flash_attn") as exc:
        kernels.varlen_attention(query, query, query, None, None, None, None)
    assert exc.value.__cause__ is cause


def test_npu_dispatch_preserves_packed_contract(monkeypatch):
    query = SimpleNamespace(device=SimpleNamespace(type="npu"), shape=(7, 2, 4))
    offsets = torch.tensor([0, 3, 7], dtype=torch.int32)
    output = torch.randn(7, 2, 4)
    calls = []

    def fusion(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    monkeypatch.setattr(kernels, "fusion_attention", fusion)
    actual = kernels.varlen_attention(query, query, query, offsets, offsets, 4, 4)
    assert actual is output
    assert calls == [
        (
            (query, query, query),
            dict(head_num=2, scale=0.5, q_cu_seqlens=offsets, k_cu_seqlens=offsets),
        )
    ]


def test_model_owns_output_reshape_and_preserves_gradient(monkeypatch):
    values = torch.randn(7, 2, 4, requires_grad=True)
    calls = []

    def attention(*args, **kwargs):
        calls.append((args, kwargs))
        return args[0] * 2

    monkeypatch.setattr(multimodal, "varlen_attention", attention)
    offsets = torch.tensor([0, 3, 7], dtype=torch.int32)
    result = multimodal.multihead_attention(
        values, values, values, offsets, offsets, 4, 4, deterministic=True
    )
    torch.testing.assert_close(result, (values * 2).flatten(-2))
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 2))
    assert calls[0][1] == {"deterministic": True}
    assert calls[0][0][3] is offsets
