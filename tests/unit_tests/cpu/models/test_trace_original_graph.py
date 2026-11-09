"""Primary diagnostic VJPs must execute without internal observation hooks."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]


def functions(filename, names):
    path = ROOT / "tests" / "integration_tests" / filename
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == len(names)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class OriginalModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False).to(torch.bfloat16)

    def forward(self, x, **kwargs):
        # Assert this at execution time, not by inspecting diagnostic source.
        for module in self.modules():
            assert not module._forward_hooks
            assert not module._forward_pre_hooks
            assert not module._backward_hooks
        return self.projection(x) + x * x


def oracle(model, x, dy):
    model.zero_grad(set_to_none=True)
    leaf = x.detach().clone().requires_grad_()
    output = model(leaf)
    output.backward(dy)
    return (
        output.detach().clone(),
        leaf.grad.clone(),
        {key: value.grad.clone() for key, value in model.named_parameters()},
    )


@pytest.mark.parametrize("hf", [True, False])
def test_layer_original_capture_matches_plain_backward_without_hooks(hf):
    capture = functions(
        "qwen35_decoder_layer_trace.py", {"forward_layer", "capture_original"}
    )["capture_original"]
    model = OriginalModule()
    x = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    output, dx, grads = oracle(model, x, dy)
    observed, actual_grads = capture(model, hf=hf, x=x, cotangent=dy)
    assert torch.equal(observed["output"], output)
    assert torch.equal(observed["dx"], dx)
    assert all(torch.equal(actual_grads[key], value) for key, value in grads.items())
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("hf", [True, False])
def test_moe_primary_has_no_router_probe_and_matches_original_backward(hf):
    run = functions("qwen35_moe_module_trace.py", {"run_module"})["run_module"]
    model = OriginalModule()  # Deliberately has no gate/router probe interface.
    x = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    output, dx, grads = oracle(model, x, dy)
    observed, actual_grads = run(
        model,
        hf=hf,
        x=x,
        dy=dy,
        scores=torch.rand(1, 3, 2),
        ids=torch.zeros(1, 3, 2, dtype=torch.long),
        routed_only=False,
    )
    assert observed.keys() == {"output", "dx"}
    assert torch.equal(observed["output"], output)
    assert torch.equal(observed["dx"], dx)
    assert all(torch.equal(actual_grads[key], value) for key, value in grads.items())


def test_bf16_router_replay_is_separate_from_original_result():
    run = functions("qwen35_moe_module_trace.py", {"run_module"})["run_module"]

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2, 4, dtype=torch.bfloat16))

        def forward(self, x):
            logits = nn.functional.linear(x, self.weight)
            scores = logits.float().softmax(-1)
            ids = torch.arange(2).expand_as(scores)
            return logits, scores, ids

    class BranchedMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Router()
            self.shared = nn.Linear(4, 4, bias=False).bfloat16()
            self.routed = nn.Linear(4, 4, bias=False).bfloat16()
            self.observed_modes = []

        def forward(self, x):
            logits, scores, _ = self.gate(x)
            self.observed_modes.append(
                (bool(self.gate._forward_hooks), scores.retains_grad)
            )
            return (
                self.shared(x)
                + self.routed(x) * scores[..., :1].to(x.dtype)
                + x * logits[..., :1]
            )

    model = BranchedMoE()
    x = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    kwargs = dict(
        hf=True,
        x=x,
        dy=torch.randn_like(x),
        scores=torch.rand(1, 3, 2),
        ids=torch.zeros(1, 3, 2, dtype=torch.long),
        routed_only=False,
    )
    original, original_grads = run(model, **kwargs)
    saved_dx = original["dx"].clone()
    saved_grads = {key: value.clone() for key, value in original_grads.items()}
    observed, _ = run(model, **kwargs, observe_router=True)
    assert model.observed_modes == [(False, False), (True, True)]
    assert original.keys() == {"output", "dx"}
    assert {"dscore", "weights", "ids", "logits"} <= observed.keys()
    assert torch.equal(original["dx"], saved_dx)
    assert all(
        torch.equal(original_grads[key], value) for key, value in saved_grads.items()
    )
    assert not model.gate._forward_hooks
    # Scheduling-sensitive replay dx is reported, never required to be exact.
    assert observed["dx"].dtype == original["dx"].dtype == torch.bfloat16


def test_merge_interventions_only_change_order_and_view_edges():
    namespace = functions(
        "qwen35_moe_merge_trace.py",
        {"forward_reference_order", "forward_reference_views"},
    )
    events = []

    class Branch(nn.Module):
        def __init__(self, name, out_dim=4):
            super().__init__()
            self.name = name
            self.linear = nn.Linear(4, out_dim, bias=False).bfloat16()

        def forward(self, x, *args):
            events.append((self.name, x.ndim, x.untyped_storage().data_ptr()))
            value = self.linear(x)
            if self.name == "router":
                value = value.reshape(-1, 2)
                return value, value.float().softmax(-1), torch.zeros_like(value).long()
            if self.name == "experts":
                return value * args[1][:, :1].to(value.dtype)
            return value

    model = nn.Module()
    model.gate = Branch("router", 2)
    model.experts = Branch("experts")
    model.shared_expert = Branch("shared")
    model.shared_expert_gate = Branch("shared_gate", 1)
    parameters = dict(model.named_parameters())
    x = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    outputs = []
    for name in ("forward_reference_order", "forward_reference_views"):
        events.clear()
        leaf = x.clone().requires_grad_()
        output = namespace[name](model, leaf)
        outputs.append(output.detach())
        output.backward(torch.ones_like(output))
        assert [event[0] for event in events] == [
            "router",
            "experts",
            "shared",
            "shared_gate",
        ]
        assert [event[1] for event in events] == (
            [2, 2, 2, 2] if name == "forward_reference_order" else [3, 2, 3, 3]
        )
        assert all(event[2] == leaf.untyped_storage().data_ptr() for event in events)
        assert leaf.grad.dtype == torch.bfloat16 and torch.isfinite(leaf.grad).all()
        assert all(
            dict(model.named_parameters())[k] is p for k, p in parameters.items()
        )
        assert all(p.grad is not None for p in model.parameters())
        assert all(
            not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()
        )
        model.zero_grad(set_to_none=True)
    assert torch.equal(*outputs)


def test_decoder_moe_order_switch_preserves_native_and_parameter_identity():
    namespace = functions("qwen35_decoder_layer_trace.py", {"configure_moe_graph"})
    forward = functions("qwen35_moe_merge_trace.py", {"forward_reference_order"})[
        "forward_reference_order"
    ]
    imports = []

    def import_diagnostic(name):
        imports.append(name)
        return SimpleNamespace(forward_reference_order=forward)

    namespace.update(
        importlib=SimpleNamespace(import_module=import_diagnostic),
        MethodType=MethodType,
    )
    configure = namespace["configure_moe_graph"]
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.shared_expert = nn.Linear(4, 4)
    layer.mlp.gate = nn.Linear(4, 2)
    before = dict(layer.named_parameters())
    native = layer.mlp.forward.__func__
    configure(layer, "native")
    assert not imports and layer.mlp.forward.__func__ is native
    configure(layer, "reference-order")
    assert imports == ["tests.integration_tests.qwen35_moe_merge_trace"]
    assert layer.mlp.forward.__func__ is forward
    assert all(dict(layer.named_parameters())[key] is p for key, p in before.items())
    with pytest.raises(ValueError, match="Unsupported"):
        configure(layer, "invalid")


def test_gdn_diagnostic_switches_preserve_parameters_and_cast_vjp():
    namespace = functions(
        "qwen35_decoder_layer_trace.py",
        {
            "configure_gdn_graph",
            "forward_split_qkv",
            "forward_gated_norm_reference_cast",
        },
    )
    namespace["MethodType"] = MethodType

    class Qwen3_5MoeRMSNormGated(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
            self.variance_epsilon = 1e-6

    layer = nn.Module()
    layer.linear_attn = nn.Module()
    gdn = layer.linear_attn
    gdn.key_dim, gdn.value_dim = 4, 8
    gdn.in_proj_qkv = nn.Linear(4, 16, bias=False).bfloat16()
    gdn.norm = Qwen3_5MoeRMSNormGated()
    before = dict(layer.named_parameters())
    namespace["configure_gdn_graph"](layer, "native", "native")
    assert "forward" not in gdn.in_proj_qkv.__dict__
    assert "forward" not in gdn.norm.__dict__
    x = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    expected = gdn.in_proj_qkv(x)
    namespace["configure_gdn_graph"](layer, "separate", "reference-order")
    torch.testing.assert_close(gdn.in_proj_qkv(x), expected, atol=0.005, rtol=0.02)
    assert all(dict(layer.named_parameters())[key] is p for key, p in before.items())
    leaf = x.detach().clone().requires_grad_()
    gate = torch.randn_like(leaf, requires_grad=True)
    dy = torch.randn_like(leaf)
    actual = gdn.norm(leaf, gate)
    actual_grads = torch.autograd.grad(actual, (leaf, gate, gdn.norm.weight), dy)
    weight = gdn.norm.weight.detach().clone().requires_grad_()
    fp32_input = leaf.float()
    normalized = fp32_input * torch.rsqrt(
        fp32_input.square().mean(-1, keepdim=True) + 1e-6
    )
    oracle = (weight.float() * normalized).bfloat16()
    oracle = (oracle * nn.functional.silu(gate.float())).bfloat16()
    expected_grads = torch.autograd.grad(oracle, (leaf, gate, weight), dy)
    assert torch.equal(actual, oracle)
    assert all(
        torch.equal(a, b) for a, b in zip(actual_grads, expected_grads, strict=True)
    )
