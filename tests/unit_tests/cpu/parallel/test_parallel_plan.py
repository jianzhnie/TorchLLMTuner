"""The minimal immutable plan shared by resolution and execution."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

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
