"""MoE scoring placement and BF16 backward arithmetic."""

import pytest
import torch


def test_torchtitan_moe_scores_before_down_projection(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.models.hf.model_setup import forward_torchtitan_moe

    torch.manual_seed(37)
    inputs = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    gate_up = torch.randn(1, 12, 8, dtype=torch.bfloat16, requires_grad=True)
    down = torch.randn(1, 8, 6, dtype=torch.bfloat16, requires_grad=True)
    scores = torch.tensor([0.17, 0.53, 0.89], requires_grad=True)
    indices = torch.arange(3)

    def dispatch(hidden, routing_scores, expert_ids, counts):
        return (
            hidden,
            counts,
            SimpleNamespace(
                topk_scores_experts_sorted_N=routing_scores,
                token_indices_experts_sorted_N=indices,
            ),
        )

    def grouped_mm(hidden, weight, *, offs):
        assert offs.tolist() == [3]
        return hidden @ weight[0]

    monkeypatch.setattr(torch, "_grouped_mm", grouped_mm)
    module = SimpleNamespace(
        gate=lambda hidden: (None, scores, torch.zeros(3, 1, dtype=torch.long)),
        experts=SimpleNamespace(
            num_experts=1,
            gate_up_proj=gate_up,
            down_proj=down,
            act_fn=torch.nn.functional.silu,
        ),
        _llmtuner_local_dispatcher=SimpleNamespace(dispatch=dispatch),
    )
    actual = forward_torchtitan_moe(module, inputs.unsqueeze(0))
    gate, up = (inputs @ gate_up[0].T).chunk(2, dim=-1)
    hidden = torch.nn.functional.silu(gate) * up
    expected = (hidden * scores[:, None].bfloat16()) @ down[0].T
    torch.testing.assert_close(actual[0], expected, atol=0, rtol=0)
    after_down = ((hidden @ down[0].T).float() * scores[:, None]).bfloat16()
    assert not torch.equal(expected, after_down)
    parameters = (inputs, gate_up, down, scores)
    probe = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, parameters, probe, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, parameters, probe[0])
    for observed, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)


def test_torchtitan_moe_rejects_non_silu_activation():
    from types import SimpleNamespace

    from llmtuner.models.hf.model_setup import stabilize_moe_experts

    model = SimpleNamespace(
        config=SimpleNamespace(experts_implementation="torchtitan", hidden_act="gelu")
    )
    with pytest.raises(ValueError, match="hidden_act='silu'"):
        stabilize_moe_experts(model)
