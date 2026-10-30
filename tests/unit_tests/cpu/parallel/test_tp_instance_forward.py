"""Instance kernel overrides must retain TP sequence collectives."""

import torch
from torch import nn


def test_tp_boundary_preserves_instance_kernel_and_collective_order(monkeypatch):
    import importlib
    from types import MethodType

    implementation = importlib.import_module(
        "llmtuner.parallel.tensor_parallel.tensor_parallel"
    )
    model = nn.Module()
    block = nn.Module()
    block.experts = nn.Module()
    block.experts.gate_up_proj = nn.Parameter(torch.randn(4, 16, 16))
    block.experts.down_proj = nn.Parameter(torch.randn(4, 16, 8))
    model.add_module("block", block)
    events = []

    def kernel(self, hidden_states):
        events.append("kernel")
        assert hidden_states.shape[-2] == 8
        return hidden_states * 2

    def gather(value, dim, group):
        events.append("gather")
        return torch.cat((value, value), dim=dim)

    def scatter(value, dim, group):
        events.append("scatter")
        return value.chunk(2, dim=dim)[0]

    block.forward = MethodType(kernel, block)
    keys = set(model.state_dict())
    implementation.realize_moe_blocks_tp(
        [("block", block)], tp_size=2, tp_rank=0, group=object()
    )
    monkeypatch.setattr(implementation, "all_gather_along", gather)
    monkeypatch.setattr(implementation, "reduce_scatter_along", scatter)
    value = torch.randn(1, 4, 16, requires_grad=True)
    result = block(value)
    assert events == ["gather", "kernel", "scatter"]
    torch.testing.assert_close(result, value * 2)
    result.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 2))
    assert set(model.state_dict()) == keys


def test_attention_instance_kernel_survives_tp_assembly(monkeypatch):
    import importlib
    from types import MethodType, SimpleNamespace

    from llmtuner.config.parallel_config import ParallelConfig

    implementation = importlib.import_module(
        "llmtuner.parallel.tensor_parallel.tensor_parallel"
    )
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    axis = SimpleNamespace(
        device_type="cpu",
        size=lambda: 2,
        get_local_rank=lambda: 0,
        get_group=lambda: group,
    )
    for replicated in (False, True):
        model = nn.Module()
        attention = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            attention.add_module(name, nn.Linear(8, 8, bias=False))
        model.add_module("self_attn", attention)
        model.add_module("mlp", nn.Linear(8, 8, bias=False))
        calls = []

        def kernel(self, hidden_states, calls=calls):
            calls.append(hidden_states.shape[-2])
            return hidden_states * 2, None

        attention.forward = MethodType(kernel, attention)
        plan = (
            {"mlp": "colwise"}
            if replicated
            else {
                "self_attn.q_proj": "colwise",
                "self_attn.k_proj": "colwise",
                "self_attn.v_proj": "colwise",
                "self_attn.o_proj": "rowwise",
            }
        )
        implementation.apply_tp(
            model,
            {"tp": axis},
            ParallelConfig(tensor_parallel_size=2),
            plan=plan,
        )
        monkeypatch.setattr(
            implementation,
            "all_gather_along",
            lambda x, dim, group: torch.cat((x, x), dim=dim),
        )
        inputs = torch.randn(1, 4, 8, requires_grad=True)
        output, auxiliary = attention(inputs)
        assert calls == [8]
        assert auxiliary is None
        expected = inputs * 2 if replicated else torch.cat((inputs, inputs), dim=1) * 2
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        torch.testing.assert_close(
            inputs.grad, torch.full_like(inputs, 2 if replicated else 4)
        )
