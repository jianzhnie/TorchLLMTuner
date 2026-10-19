"""KCP adapter argument plumbing; no device or state-exchange claims."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmtuner.parallel.context_parallel.kcp import kcp_kda_forward


@pytest.mark.parametrize("tuple_result", [False, True])
def test_kcp_preserves_context_and_kernel_gate_contract(monkeypatch, tuple_result):
    context = object()
    calls = []

    def build_context(**kwargs):
        assert kwargs["cu_seqlens"].tolist() == [0, 8]
        assert kwargs["conv1d_kernel_size"] == 3
        return context

    def convolution(**kwargs):
        assert kwargs["cp_context"] is context
        assert kwargs["initial_state"] is None
        assert kwargs["output_final_state"] is False
        calls.append("conv")
        value = kwargs["x"]
        return (value, None) if tuple_result else value

    def attention(**kwargs):
        assert kwargs["cp_context"] is context
        assert kwargs["use_beta_sigmoid_in_kernel"] is False
        assert kwargs["use_gate_in_kernel"] is True
        assert kwargs["safe_gate"] is True
        assert kwargs["lower_bound"] == -3
        assert kwargs["disable_recompute"] is True
        torch.testing.assert_close(kwargs["beta"], torch.full((1, 4, 2), 0.5))
        calls.append("attention")
        value = kwargs["v"]
        return (value, None) if tuple_result else value

    monkeypatch.setattr(
        "llmtuner.parallel.context_parallel.kcp._kcp_runtime",
        lambda: (convolution, attention, build_context),
    )
    module = nn.Module()
    for name in ("q", "k", "v"):
        setattr(module, f"{name}_proj", nn.Identity())
        conv = nn.Conv1d(4, 4, 3, groups=4)
        conv.activation = "silu"
        setattr(module, f"{name}_conv1d", conv)
    module.f_a_proj = module.f_b_proj = nn.Identity()
    module.b_proj = nn.Linear(4, 2, bias=False)
    nn.init.zeros_(module.b_proj.weight)
    module.g_proj = module.o_proj = nn.Identity()
    module.o_norm = lambda output, gate: output
    module.head_k_dim = module.head_dim = module.num_heads = 2
    module.use_full_rank_gate = True
    module.gate_lower_bound = -3
    module.A_log = module.dt_bias = torch.zeros(2)
    module.config = SimpleNamespace(skip_kda_recompute=True)
    inputs = torch.randn(1, 4, 4, requires_grad=True)
    forward = kcp_kda_forward(lambda: None, SimpleNamespace(size=lambda: 2))
    output = forward(module, inputs)
    torch.testing.assert_close(output, inputs)
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
    assert calls == ["conv", "conv", "conv", "attention"]


@pytest.mark.parametrize(
    "mask", [torch.tensor([[False, True, True]]), torch.tensor([[True, False, True]])]
)
def test_kcp_rejects_padding_before_runtime(monkeypatch, mask):
    def unexpected():
        pytest.fail("Padding must be rejected before optional kernel loading")

    monkeypatch.setattr(
        "llmtuner.parallel.context_parallel.kcp._kcp_runtime", unexpected
    )
    module = nn.Module()
    forward = kcp_kda_forward(lambda: None, None)
    with pytest.raises(ValueError, match="compacted convolution/recurrent state"):
        forward(module, torch.randn(1, 3, 4), attention_mask=mask)


def test_zeroing_padding_is_not_compacted_kda_state():
    from llmtuner.models.kimi.delta_attention import KimiDeltaAttention
    from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig

    torch.manual_seed(41)
    config = KimiLinearConfig(
        hidden_size=8,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    module = KimiDeltaAttention(config, 0)
    inputs = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, True, True]])
    expected = module(inputs, attention_mask=mask)
    zeroed = module(inputs * mask[..., None]) * mask[..., None]
    assert (expected[:, mask[0]] - zeroed[:, mask[0]]).abs().max() > 1e-4


def test_kcp_rejects_inconsistent_local_convolution_kernels(monkeypatch):
    def unexpected():
        pytest.fail("Invalid local geometry must fail before kernel loading")

    monkeypatch.setattr(
        "llmtuner.parallel.context_parallel.kcp._kcp_runtime", unexpected
    )
    module = nn.Module()
    for name, size in (("q", 3), ("k", 4), ("v", 3)):
        setattr(module, f"{name}_conv1d", nn.Conv1d(4, 4, size, groups=4))
    module.head_k_dim = 4
    module.gate_lower_bound = None
    with pytest.raises(ValueError, match="matching Q/K/V convolution kernels"):
        kcp_kda_forward(lambda: None, None)(module, torch.randn(1, 3, 4))
