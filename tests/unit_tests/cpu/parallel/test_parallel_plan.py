"""The minimal immutable plan shared by resolution and execution."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from llmtuner.models.contracts import DecoderStructure
from llmtuner.parallel import ParallelPlan, PlanTarget


def test_plan_counts_resolved_targets_without_tensor_payloads() -> None:
    plan = ParallelPlan(
        stage="tp",
        targets=(
            PlanTarget("model.layers.0.mlp", "dense"),
            PlanTarget("model.layers.1.mlp", "moe"),
            PlanTarget("model.embed_tokens", "vocab"),
        ),
        degree=2,
        implementation="functional",
    )
    assert plan.count("dense") == 1
    assert plan.count("moe") == 1
    assert plan.count("vocab") == 1
    with pytest.raises(FrozenInstanceError):
        plan.degree = 4


def test_plan_snapshots_caller_owned_containers_without_copying_parameters():
    model = nn.Linear(2, 2)
    target = PlanTarget("", "dense", model)
    targets = [target]
    operations = ["shard"]
    parameters = [[name, value] for name, value in model.named_parameters()]
    plan = ParallelPlan(
        "tp",
        targets,
        2,
        "functional",
        operations=operations,
        parameter_bindings=parameters,
    )
    targets.clear()
    operations[0] = "other"
    parameters[0][1] = nn.Parameter(torch.zeros_like(model.weight))
    parameters.clear()
    assert plan.targets == (target,)
    assert plan.operations == ("shard",)
    assert plan.parameter_bindings[0][1] is model.weight
    plan.validate_binding(model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": ""},
        {"degree": 0},
        {"degree": True},
        {"degree": 1.5},
        {"degree": "2"},
        {"degree": None},
        {"stage": " "},
        {"stage": 1},
        {"implementation": " "},
        {"implementation": 1},
        {"degrees": (2, False)},
        {"degrees": (2, 0)},
        {"degrees": (2, 1.5)},
        {"implementation": ""},
    ],
)
def test_plan_rejects_incomplete_execution_metadata(kwargs) -> None:
    values = {
        "stage": "tp",
        "targets": (),
        "degree": 2,
        "implementation": "functional",
        **kwargs,
    }
    with pytest.raises(ValueError):
        ParallelPlan(**values)


def test_plan_target_allows_the_root_module_path() -> None:
    target = PlanTarget("", "moe")
    assert target.path == ""
    assert target.kind == "moe"


@pytest.mark.parametrize("path", [None, 3, ".layer", "layer.", "layer..mlp", " "])
def test_plan_target_rejects_malformed_module_paths(path) -> None:
    with pytest.raises(ValueError, match="module path"):
        PlanTarget(path, "dense")


def test_plan_reports_removed_target_as_stale_binding() -> None:
    model = nn.Sequential(nn.Linear(2, 2))
    plan = ParallelPlan(
        stage="tp",
        targets=(PlanTarget("0", "dense", model[0]),),
        degree=2,
        implementation="functional",
    )
    del model[0]
    with pytest.raises(ValueError, match="Stale plan binding: 0"):
        plan.validate_binding(model)


@pytest.mark.parametrize("token_history", [False, True])
def test_plan_rejects_unregistered_target_alias(token_history: bool) -> None:
    model = nn.Module()
    model.layer = nn.Linear(2, 2)
    model.layer.requires_full_token_history = True
    plan = ParallelPlan(
        "cp",
        (PlanTarget("layer", "attention", model.layer),),
        2,
        "sdpa",
        token_history_requirements=(("layer", True),) if token_history else (),
    )
    plan.validate_binding(model)
    layer = model.layer
    del model.layer
    # get_submodule follows this alias although it has no checkpoint ownership.
    object.__setattr__(model, "layer", layer)
    assert model.get_submodule("layer") is layer
    with pytest.raises(ValueError, match="Stale plan"):
        plan.validate_binding(model)


def test_plan_preserves_registered_alias_target_binding() -> None:
    model = nn.Module()
    model.layer = nn.Linear(2, 2)
    model.alias = model.layer
    plan = ParallelPlan(
        "tp", (PlanTarget("alias", "dense", model.layer),), 2, "functional"
    )
    plan.validate_binding(model)
    assert model.alias.weight is model.layer.weight


@pytest.mark.parametrize(
    "change", ["shape", "dtype", "stride", "offset", "grad", "device"]
)
def test_plan_rejects_buffer_metadata_change_with_same_identity(change) -> None:
    model = nn.Module()
    model.register_buffer("cache", torch.arange(6.0).reshape(2, 3))
    plan = ParallelPlan(
        stage="tp",
        targets=(),
        degree=2,
        implementation="functional",
        buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
    )
    original = model.cache
    if change == "shape":
        model.cache.resize_(3, 2)
    elif change == "dtype":
        model.cache.data = model.cache.double()
    elif change == "stride":
        model.cache.data = model.cache.t().contiguous().t()
    elif change == "offset":
        model.cache.data = torch.arange(9.0)[3:].reshape(2, 3)
    elif change == "grad":
        model.cache.requires_grad_(True)
    else:
        # TensorImpl replacement preserves Python identity during materialization.
        torch.utils.swap_tensors(model.cache, torch.empty(2, 3, device="meta"))
    assert model.cache is original
    with pytest.raises(ValueError, match="Stale plan buffer metadata: cache"):
        plan.validate_binding(model)


def test_plan_allows_buffer_value_update_without_layout_change() -> None:
    model = nn.Module()
    model.register_buffer("router_bias", torch.zeros(4))
    plan = ParallelPlan(
        stage="ep",
        targets=(),
        degree=2,
        implementation="alltoall",
        buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
    )
    model.router_bias.add_(torch.arange(4.0))
    plan.validate_binding(model)


def test_plan_preserves_nonzero_offset_views_during_value_updates() -> None:
    model = nn.Module()
    model.weight = nn.Parameter(torch.arange(9.0)[1:].reshape(2, 4))
    model.register_buffer("cache", torch.arange(7.0)[3:])
    plan = ParallelPlan(
        stage="tp",
        targets=(),
        degree=2,
        implementation="functional",
        parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
        buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
    )
    assert model.weight.storage_offset() == 1
    assert model.cache.storage_offset() == 3
    with torch.no_grad():
        model.weight.add_(2)
        model.cache.mul_(2)
    plan.validate_binding(model)
    torch.testing.assert_close(model.weight, torch.arange(3.0, 11.0).reshape(2, 4))
    torch.testing.assert_close(model.cache, torch.arange(3.0, 7.0) * 2)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("initially_shared", [False, True])
def test_plan_rejects_changed_storage_aliases(device, initially_shared):
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty(4, device=device))
    model.register_buffer(
        "cache",
        model.weight.detach() if initially_shared else torch.empty(4, device=device),
    )
    plan = ParallelPlan(
        stage="tp",
        targets=(),
        degree=2,
        implementation="functional",
        parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
        buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
    )
    original = model.cache
    model.cache.data = (
        model.cache.clone() if initially_shared else model.weight.detach()
    )
    assert model.cache is original
    with pytest.raises(ValueError, match="Stale plan storage aliases"):
        plan.validate_binding(model)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_plan_allows_storage_reallocation_with_alias_relation_preserved(device):
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty(4, device=device))
    model.register_buffer("cache", model.weight.detach())
    plan = ParallelPlan(
        stage="tp",
        targets=(),
        degree=2,
        implementation="functional",
        parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
        buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
    )
    replacement = torch.ones(4, device=device)
    model.weight.data = replacement
    model.cache.data = replacement
    plan.validate_binding(model)


@pytest.mark.parametrize("change", ["replace", "remove", "foreign"])
def test_minimal_plan_validates_target_ownership_without_global_snapshot(change):
    model = nn.Sequential(nn.Linear(4, 4))
    target = model[0]
    plan = ParallelPlan(
        stage="tp",
        targets=(PlanTarget("0", "dense", target),),
        degree=2,
        implementation="functional",
    )
    plan.validate_binding(model)
    if change == "replace":
        model[0] = nn.Linear(4, 4)
    elif change == "remove":
        del model[0]
    else:
        model = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(ValueError, match="Stale plan binding"):
        plan.validate_binding(model)


@pytest.mark.parametrize("change", ["replace", "remove", "invalid"])
def test_plan_rejects_changed_explicit_decoder_structure(change):
    model = nn.Module()
    model.decoder_structure = DecoderStructure(
        decoder="model",
        embedding="model.embed",
        blocks="model.layers",
        norm="model.norm",
        head="head",
        attention="self_attn",
    )
    plan = ParallelPlan(
        stage="tp",
        targets=(),
        degree=2,
        implementation="functional",
        module_bindings=tuple(model.named_modules(remove_duplicate=False)),
    )
    if change == "replace":
        model.decoder_structure = DecoderStructure(
            decoder="model",
            embedding="model.embed",
            blocks="model.layers",
            norm="model.norm",
            head="other_head",
            attention="self_attn",
        )
    elif change == "remove":
        del model.decoder_structure
    else:
        model.decoder_structure = "invalid"
    with pytest.raises(ValueError, match="Stale plan decoder structure"):
        plan.validate_binding(model)


def test_plan_rejects_decoder_declaration_added_after_resolution():
    model = nn.Module()
    plan = ParallelPlan(
        stage="tp",
        targets=(),
        degree=2,
        implementation="functional",
        module_bindings=tuple(model.named_modules(remove_duplicate=False)),
    )
    model.decoder_structure = DecoderStructure(
        decoder="", embedding="embed", blocks="layers", norm="norm", head="head"
    )
    with pytest.raises(ValueError, match="Stale plan decoder structure"):
        plan.validate_binding(model)


def test_plan_accepts_equivalent_decoder_declaration_without_tensor_changes():
    model = nn.Module()
    declaration = dict(
        decoder="model",
        embedding="model.embed",
        blocks="model.layers",
        norm="model.norm",
        head="head",
        attention="self_attn",
    )
    model.decoder_structure = DecoderStructure(**declaration)
    plan = ParallelPlan(
        stage="ep",
        targets=(),
        degree=2,
        implementation="alltoall",
        module_bindings=tuple(model.named_modules(remove_duplicate=False)),
    )
    model.decoder_structure = DecoderStructure(**declaration)
    plan.validate_binding(model)


@pytest.mark.parametrize("replacement", ["instance", "class", "receiver"])
def test_plan_rejects_changed_forward_without_parameter_changes(
    monkeypatch, replacement
):
    class Projection(nn.Linear):
        pass

    model = nn.Sequential(Projection(3, 4))
    plan = ParallelPlan(
        "tp",
        (),
        2,
        "native",
        module_bindings=tuple(model.named_modules(remove_duplicate=False)),
        parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
    )
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    if replacement == "instance":
        monkeypatch.setattr(model[0], "forward", lambda inputs: inputs)
    elif replacement == "class":
        monkeypatch.setattr(Projection, "forward", lambda self, inputs: inputs)
    else:
        monkeypatch.setattr(model[0], "forward", model.forward)
    with pytest.raises(ValueError, match="Stale plan forward binding: 0"):
        plan.validate_binding(model)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


def test_plan_accepts_unchanged_bound_forward_and_inplace_weight_update():
    model = nn.Sequential(nn.Linear(3, 4))
    plan = ParallelPlan(
        "tp",
        (),
        2,
        "native",
        module_bindings=tuple(model.named_modules(remove_duplicate=False)),
        parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
    )
    with torch.no_grad():
        model[0].weight.add_(1)
    plan.validate_binding(model)
    plan.validate_binding(model)
