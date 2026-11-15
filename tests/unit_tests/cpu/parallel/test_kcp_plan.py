"""Pure KCP binding and installation failure boundaries, without kernel claims."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmtuner.models.contracts import DecoderStructure
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.latent_attention import KimiLatentAttention
from llmtuner.parallel.context_parallel import kcp


class KDA(nn.Module):
    """Declared KDA layout for metadata resolution, not KDA execution."""

    context_parallel_layout = "kda_state"

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))

    def validate_context_parallel_inputs(self, *args, **kwargs):
        raise AssertionError("Metadata checks must not execute model input validation")

    def forward_context_parallel(self, *args, **kwargs):
        raise AssertionError("Metadata checks must not execute model computation")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight


def declared_kda_model(*attentions: nn.Module) -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleList()
    for attention in attentions:
        layer = nn.Module()
        layer.self_attn = attention
        model.layers.append(layer)
    model.decoder_structure = DecoderStructure(
        decoder="",
        embedding="embedding",
        blocks="layers",
        norm="norm",
        head="head",
        attention="self_attn",
    )
    return model


def test_kcp_resolution_preserves_aliases_parameters_forward_and_rng(monkeypatch):
    module = KDA()
    model = declared_kda_model(module, module)
    parameters = tuple(model.parameters())
    before = torch.get_rng_state().clone()
    forward = module.forward

    def reject_runtime():
        pytest.fail("Pure resolution must not load KCP runtime")

    monkeypatch.setattr(kcp, "_kcp_runtime", reject_runtime)
    plan = kcp.resolve_kcp_plan(model, degree=2)
    assert len(plan.targets) == 1
    assert plan.targets[0].path == "layers.0.self_attn"
    assert plan.targets[0].module is module
    assert plan.targets[0].layout == "kda_state"
    assert model.layers[1].self_attn is module
    assert module.forward == forward
    assert torch.equal(before, torch.get_rng_state())
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    plan.validate_binding(model)


def test_kcp_stale_plan_rejected_before_loading_kernels(monkeypatch):
    model = declared_kda_model(KDA())
    plan = kcp.resolve_kcp_plan(model, degree=2)
    model.layers[0].self_attn = KDA()
    monkeypatch.setattr(
        kcp, "_kcp_runtime", lambda: pytest.fail("Stale plan reached runtime")
    )
    mesh = SimpleNamespace(size=lambda: 2)
    with pytest.raises(ValueError, match="Stale plan"):
        kcp.apply_kcp(model, mesh, resolved_plan=plan)


def test_kcp_changed_mask_rejected_before_loading_kernels(monkeypatch):
    model = declared_kda_model(KDA())
    model.config = SimpleNamespace(attn_mask_type="causal")
    plan = kcp.resolve_kcp_plan(model, degree=2)
    model.config.attn_mask_type = "block_causal"
    original = model.layers[0].self_attn.forward
    monkeypatch.setattr(
        kcp, "_kcp_runtime", lambda: pytest.fail("Invalid mask reached runtime")
    )
    mesh = SimpleNamespace(
        size=lambda: 2,
        get_group=lambda: pytest.fail("Invalid mask reached group acquisition"),
    )
    with pytest.raises(ValueError, match="packed document attention"):
        kcp.apply_kcp(model, mesh, resolved_plan=plan)
    assert model.layers[0].self_attn.forward == original


def test_kcp_wrapper_construction_failure_leaves_all_forwards_unchanged(monkeypatch):
    model = declared_kda_model(KDA(), KDA())
    original = [layer.self_attn.forward for layer in model.layers]
    calls = []
    monkeypatch.setattr(kcp, "_kcp_runtime", lambda: ())

    def wrap(forward, group):
        calls.append(forward)
        if len(calls) == 2:
            raise RuntimeError("second wrapper failed")
        return lambda module, value: value

    monkeypatch.setattr(kcp, "kcp_kda_forward", wrap)
    mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: object())
    with pytest.raises(RuntimeError, match="second wrapper failed"):
        kcp.apply_kcp(model, mesh, resolved_plan=kcp.resolve_kcp_plan(model, degree=2))
    assert len(calls) == 2
    assert [layer.self_attn.forward for layer in model.layers] == original


def test_declared_mla_only_pipeline_part_binds_without_kda(monkeypatch):
    config = KimiLinearConfig(
        hidden_size=12,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=3,
        qk_rope_head_dim=1,
        v_head_dim=3,
        mla_use_nope=True,
    )
    model = nn.Module()
    block = nn.Module()
    block.self_attn = KimiLatentAttention(config, 0)
    model.layers = nn.ModuleList([block])
    model.decoder_structure = DecoderStructure(
        decoder="",
        embedding="embedding",
        blocks="layers",
        norm="norm",
        head="head",
        attention="self_attn",
    )
    plan = kcp.resolve_kcp_plan(model, degree=2)
    assert [(target.path, target.layout) for target in plan.targets] == [
        ("layers.0.self_attn", "mla_ulysses")
    ]
    monkeypatch.setattr(
        kcp, "_kcp_runtime", lambda: pytest.fail("MLA must not require KDA kernels")
    )
    installed = object()
    monkeypatch.setattr(kcp, "kcp_mla_attention", lambda group: installed)
    mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: object())
    assert kcp.apply_kcp(model, mesh, resolved_plan=plan) == 0
    assert block.self_attn._kcp_attention is installed


@pytest.mark.parametrize("layout", ["kda_state", "mla_ulysses", None, "standard"])
def test_explicit_attention_selects_model_declared_layout(layout):
    class DeclaredAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_parallel_layout = layout
            self.num_heads = 4
            self.weight = nn.Parameter(torch.ones(2))

    model = nn.Module()
    block = nn.Module()
    block.self_attn = DeclaredAttention()
    model.layers = nn.ModuleList([block])
    model.decoder_structure = DecoderStructure(
        decoder="",
        embedding="embedding",
        blocks="layers",
        norm="norm",
        head="head",
        attention="self_attn",
    )
    parameter = block.self_attn.weight
    forward = block.self_attn.forward
    rng = torch.get_rng_state().clone()
    if layout in ("kda_state", "mla_ulysses"):
        plan = kcp.resolve_kcp_plan(model, degree=2)
        assert [(target.path, target.layout) for target in plan.targets] == [
            ("layers.0.self_attn", layout)
        ]
    else:
        with pytest.raises(ValueError, match="declared Kimi KDA/MLA"):
            kcp.resolve_kcp_plan(model, degree=2)
    assert block.self_attn.weight is parameter
    assert block.self_attn.forward == forward
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "changed_role"])
def test_invalid_declared_kcp_plan_rejected_before_runtime(monkeypatch, invalid):
    model = nn.Module()
    model.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.self_attn = KDA()
        layer.self_attn.context_parallel_layout = "kda_state"
        model.layers.append(layer)
    model.decoder_structure = DecoderStructure(
        decoder="",
        embedding="embedding",
        blocks="layers",
        norm="norm",
        head="head",
        attention="self_attn",
    )
    plan = kcp.resolve_kcp_plan(model, degree=2)
    if invalid == "missing":
        plan = replace(plan, targets=plan.targets[:-1])
        message = "cover all declared"
    elif invalid == "duplicate":
        plan = replace(plan, targets=(*plan.targets, plan.targets[0]))
        message = "duplicate attention"
    else:
        model.layers[-1].self_attn.context_parallel_layout = "mla_ulysses"
        message = "Stale KCP attention layout"
    original = tuple(layer.self_attn.forward for layer in model.layers)
    monkeypatch.setattr(
        kcp, "_kcp_runtime", lambda: pytest.fail("Invalid plan reached KCP kernels")
    )
    with pytest.raises(ValueError, match=message):
        kcp.apply_kcp(model, SimpleNamespace(size=lambda: 2), resolved_plan=plan)
    assert tuple(layer.self_attn.forward for layer in model.layers) == original
