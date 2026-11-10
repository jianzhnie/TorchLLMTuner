"""Opt-in score placement preserves native HF ownership and router policy."""

import copy

import pytest
import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)

from llmtuner.config.model_config import ModelConfig
from llmtuner.models.hf.expert_kernels import (
    stabilize_moe_experts,
)
from llmtuner.models.hf.model_setup import configure_experts_implementation
from llmtuner.models.qwen3_5.experts import (
    _forward_pre_down_grouped_experts,
    _validate_pre_down_grouped_inputs,
)


def model(implementation):
    config = AutoConfig.for_model(
        "qwen3_5_moe_text",
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    config.experts_implementation = implementation
    configure_experts_implementation(config, model_cls=object)
    holder = nn.Module()
    holder.config = config
    holder.moe = Qwen3_5MoeSparseMoeBlock(config).to(torch.bfloat16)
    for p in holder.parameters():
        nn.init.normal_(p, std=0.2)
    return holder


def test_opt_in_preserves_parameters_shared_and_router():
    assert ModelConfig().experts_implementation == "native"
    assert ModelConfig(experts_implementation="pre_down")
    holder = model("pre_down")
    before = dict(holder.named_parameters())
    router = holder.moe.gate.forward.__func__
    shared = holder.moe.shared_expert.forward.__func__
    shared_gate = holder.moe.shared_expert_gate
    stabilize_moe_experts(holder)
    after = dict(holder.named_parameters())
    assert before.keys() == after.keys()
    assert all(after[k] is p for k, p in before.items())
    assert holder.moe.gate.forward.__func__ is router
    assert holder.moe.shared_expert.forward.__func__ is shared
    assert holder.moe.shared_expert_gate is shared_gate
    assert holder.config._experts_implementation == "eager"


def test_native_unchanged_and_invalid_family_rejected():
    holder = model("native")
    before = holder.moe.experts.forward.__func__
    stabilize_moe_experts(holder)
    assert holder.moe.experts.forward.__func__ is before
    holder.config.experts_implementation = "pre_down"
    holder.config.model_type = "qwen3_moe"
    with pytest.raises(ValueError, match="Qwen3.5"):
        configure_experts_implementation(holder.config, model_cls=object)
    with pytest.raises(ValueError, match="Qwen3.5"):
        stabilize_moe_experts(holder)


def test_forward_and_local_vjp_match_independent_pre_down_expression():
    torch.manual_seed(42)
    holder = model("pre_down")
    oracle = copy.deepcopy(holder.moe.experts)
    stabilize_moe_experts(holder)
    inputs = torch.randn(3, 16, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [2, 3], [1, 2]])
    weights = torch.rand(3, 2, dtype=torch.float32)
    dy = torch.randn_like(inputs)
    x = inputs.clone().requires_grad_()
    score = weights.clone().requires_grad_()
    actual = holder.moe.experts(x, ids, score)
    actual.backward(dy)
    ox = inputs.clone().requires_grad_()
    os = weights.clone().requires_grad_()
    rows = []
    for token in range(3):
        contributions = []
        for slot in range(2):
            expert = ids[token, slot]
            gate, up = nn.functional.linear(
                ox[token], oracle.gate_up_proj[expert]
            ).chunk(2, -1)
            hidden = nn.functional.silu(gate) * up
            weighted = (hidden.float() * os[token, slot]).to(torch.bfloat16)
            contributions.append(
                nn.functional.linear(weighted, oracle.down_proj[expert])
            )
        rows.append(contributions[0] + contributions[1])
    expected = torch.stack(rows)
    expected.backward(dy)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for left, right in ((x.grad, ox.grad), (score.grad, os.grad)):
        torch.testing.assert_close(left, right, atol=0.005, rtol=0.02)
    for name, p in holder.moe.experts.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=0.005, rtol=0.02
        )
    with pytest.raises(TypeError, match="BF16"):
        holder.moe.experts(inputs.float(), ids, weights)


def test_cpu_pre_down_keeps_loop_without_grouped_operator(monkeypatch):
    holder = model("pre_down")
    stabilize_moe_experts(holder)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU pre_down must keep the existing loop")

    monkeypatch.setattr(torch, "_grouped_mm", forbidden)
    result = holder.moe.experts(
        torch.randn(3, 16, dtype=torch.bfloat16),
        torch.tensor([[0, 1], [2, 3], [1, 2]]),
        torch.rand(3, 2),
    )
    assert result.shape == (3, 16)


def test_grouped_logical_vjp_empty_experts_and_parameter_ownership(monkeypatch):
    """CPU stub verifies dispatch/autograd wiring, not native NPU grouped MM."""
    torch.manual_seed(42)
    holder = model("pre_down")
    oracle = copy.deepcopy(holder.moe.experts)
    stabilize_moe_experts(holder)
    before = dict(holder.named_parameters())
    calls = []

    def grouped_stub(left, right, *, offs):
        calls.append((right, offs.clone()))
        pieces, start = [], 0
        for expert, end in enumerate(offs.tolist()):
            pieces.append(
                nn.functional.linear(left[start:end], right[expert].transpose(-2, -1))
            )
            start = end
        return torch.cat(pieces)

    monkeypatch.setattr(torch, "_grouped_mm", grouped_stub)
    # Expert 3 is empty; offsets must retain its zero-length segment.
    ids = torch.tensor([[0, 1], [2, 0], [1, 2], [2, 1]])
    x = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=True)
    scores = torch.rand(4, 2, requires_grad=True)
    dy = torch.randn_like(x)
    output = _forward_pre_down_grouped_experts(holder.moe.experts, x, ids, scores)
    output.backward(dy)
    ox = x.detach().clone().requires_grad_()
    os = scores.detach().clone().requires_grad_()
    expected = []
    for token in range(4):
        contributions = []
        for slot in range(2):
            expert = ids[token, slot]
            gate_weight, up_weight = oracle.gate_up_proj[expert].chunk(2, 0)
            gate = nn.functional.linear(ox[token], gate_weight)
            up = nn.functional.linear(ox[token], up_weight)
            hidden = (nn.functional.silu(gate) * up).float()
            hidden = (hidden * os[token, slot]).bfloat16()
            contributions.append(nn.functional.linear(hidden, oracle.down_proj[expert]))
        expected.append(contributions[0] + contributions[1])
    expected = torch.stack(expected)
    expected.backward(dy)
    for actual, reference in (
        (output, expected),
        (x.grad, ox.grad),
        (scores.grad, os.grad),
    ):
        torch.testing.assert_close(actual, reference, atol=0.005, rtol=0.02)
    for name, parameter in holder.moe.experts.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(oracle.named_parameters())[name].grad,
            atol=0.005,
            rtol=0.02,
        )
        assert torch.count_nonzero(parameter.grad[3]) == 0
    assert len(calls) == 3
    assert calls[0][1].tolist() == [2, 5, 8, 8]
    assert before.keys() == dict(holder.named_parameters()).keys()
    assert all(dict(holder.named_parameters())[k] is p for k, p in before.items())
    assert calls[0][0].untyped_storage().data_ptr() == (
        holder.moe.experts.gate_up_proj.untyped_storage().data_ptr()
    )
    # Simulate parameter materialization: a second forward must resolve fresh views.
    holder.moe.experts.gate_up_proj = nn.Parameter(
        holder.moe.experts.gate_up_proj.detach().clone()
    )
    _forward_pre_down_grouped_experts(holder.moe.experts, x, ids, scores)
    assert calls[3][0].untyped_storage().data_ptr() == (
        holder.moe.experts.gate_up_proj.untyped_storage().data_ptr()
    )
    assert calls[3][0].untyped_storage().data_ptr() != (
        calls[0][0].untyped_storage().data_ptr()
    )


@pytest.mark.parametrize(
    "case,exception,match",
    [
        ("activation_dtype", TypeError, "BF16"),
        ("weight_dtype", TypeError, "BF16"),
        ("ids_dtype", TypeError, "int64"),
        ("score_dtype", TypeError, "FP32/BF16"),
        ("activation_layout", ValueError, "contiguous"),
        ("weight_layout", ValueError, "contiguous"),
        ("misaligned", ValueError, "aligned"),
        ("geometry", ValueError, "geometry"),
        ("routing_shape", ValueError, "IDs/scores"),
        ("empty_tokens", ValueError, "nonempty"),
    ],
)
def test_grouped_contract_rejects_unsupported_inputs(case, exception, match):
    experts = model("pre_down").moe.experts
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [2, 3], [1, 2]])
    scores = torch.rand(3, 2)
    if case == "activation_dtype":
        x = x.float()
    elif case == "weight_dtype":
        experts.float()
    elif case == "ids_dtype":
        ids = ids.int()
    elif case == "score_dtype":
        scores = scores.double()
    elif case == "activation_layout":
        x = torch.randn(16, 3, dtype=torch.bfloat16).t()
    elif case == "weight_layout":
        experts.down_proj = nn.Parameter(
            experts.down_proj.transpose(1, 2).contiguous().transpose(1, 2)
        )
    elif case == "misaligned":
        x = torch.empty(49, dtype=torch.bfloat16)[1:].view(3, 16)
    elif case == "geometry":
        experts.down_proj = nn.Parameter(torch.empty(4, 16, 7, dtype=torch.bfloat16))
    elif case == "routing_shape":
        scores = scores[:, :1]
    elif case == "empty_tokens":
        x, ids, scores = x[:0], ids[:0], scores[:0]
    with pytest.raises(exception, match=match):
        _validate_pre_down_grouped_inputs(experts, x, ids, scores)
