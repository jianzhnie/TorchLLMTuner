"""Model and parallel stage declarations must agree before execution."""

import pytest
import torch

from llmtuner.models.execution import ModelExecutionCapabilities
from llmtuner.parallel.execution import (
    ExecutionLayout,
    resolve_execution_layout,
)


@pytest.mark.parametrize("chunked", [False, True])
def test_hidden_capability_requires_structure_only_for_chunked_loss(chunked):
    model = torch.nn.Linear(2, 2)
    model.execution_capabilities = ModelExecutionCapabilities(hidden_states=True)
    options = dict(
        tp_size=1,
        cp_size=1,
        tp_plan=None,
        chunked_loss=chunked,
        compile_enabled=False,
        activation_swap=False,
    )
    if chunked:
        with pytest.raises(TypeError, match="Chunked loss.*DecoderStructure"):
            resolve_execution_layout(model, **options)
    else:
        assert not resolve_execution_layout(model, **options).chunked_loss


def test_chunked_assembly_rejects_missing_structure_before_stages(monkeypatch):
    from llmtuner.config.parallel_config import ParallelConfig
    from llmtuner.parallel.parallelize import parallelize_model

    model = torch.nn.Linear(2, 2)
    model.execution_capabilities = ModelExecutionCapabilities(hidden_states=True)
    parameters = tuple(model.parameters())

    def unexpected_transform(*args, **kwargs):
        pytest.fail("Missing chunked declaration reached model transformation")

    monkeypatch.setattr("llmtuner.parallel.parallelize.apply_tp", unexpected_transform)
    with pytest.raises(TypeError, match="Chunked loss.*DecoderStructure"):
        parallelize_model(
            model,
            cfg=ParallelConfig(),
            mesh=None,
            parallel_dims=None,
            chunked_loss_num_chunks=2,
        )
    assert all(
        actual is expected
        for actual, expected in zip(model.parameters(), parameters, strict=True)
    )


@pytest.mark.parametrize("declaration", [{"head": "head"}, "head", 1])
def test_execution_resolution_rejects_malformed_structure_declaration(declaration):
    model = torch.nn.Linear(2, 2)
    model.execution_capabilities = ModelExecutionCapabilities()
    model.decoder_structure = declaration
    with pytest.raises(TypeError, match="DecoderStructure declaration"):
        resolve_execution_layout(
            model,
            tp_size=1,
            cp_size=1,
            tp_plan=None,
            chunked_loss=False,
            compile_enabled=False,
            activation_swap=False,
        )


def test_execution_resolution_requires_registered_output_projection():
    from llmtuner.models.contracts import DecoderStructure

    model = torch.nn.Module()
    model.execution_capabilities = ModelExecutionCapabilities()
    model.decoder_structure = DecoderStructure(
        decoder="", embedding="embedding", blocks="blocks", norm="norm", head="head"
    )
    # A module exposed as an ordinary attribute has no checkpoint ownership.
    object.__setattr__(model, "head", torch.nn.Linear(2, 2))
    with pytest.raises(AttributeError, match="not registered"):
        resolve_execution_layout(
            model,
            tp_size=1,
            cp_size=1,
            tp_plan=None,
            chunked_loss=False,
            compile_enabled=False,
            activation_swap=False,
        )


@pytest.mark.parametrize("change", ["add", "remove", "reorder", "empty_add"])
def test_materialization_refresh_rejects_registration_changes_without_mutating_layout(
    change,
):
    from torch import nn

    from llmtuner.parallel.assembly import ModelPartLayout

    model = nn.Module() if change == "empty_add" else nn.Linear(2, 2)
    layout = ModelPartLayout()
    layout.refresh(model)
    bindings = dict(layout.parameter_bindings)
    replacements = dict(layout.replacements.parameter_targets)
    if change in {"add", "empty_add"}:
        model.extra = nn.Parameter(torch.ones(2))
    elif change == "remove":
        del model.bias
    else:
        weight = model.weight
        del model.weight
        model.weight = weight

    with pytest.raises(ValueError, match="registration paths or order"):
        layout.refresh(model)

    assert tuple(layout.parameter_bindings) == tuple(bindings)
    assert all(
        layout.parameter_bindings[key] is value for key, value in bindings.items()
    )
    assert dict(layout.replacements.parameter_targets) == replacements


@pytest.mark.parametrize("change", ["split", "merge"])
def test_materialization_refresh_rejects_changed_alias_ownership(change):
    from torch import nn

    from llmtuner.parallel.assembly import ModelPartLayout

    model = nn.Module()
    model.first = nn.Parameter(torch.ones(2))
    model.second = model.first if change == "split" else nn.Parameter(torch.ones(2))
    layout = ModelPartLayout()
    layout.refresh(model)
    bindings = dict(layout.parameter_bindings)
    replacements = dict(layout.replacements.parameter_targets)
    if change == "split":
        model.second = nn.Parameter(torch.zeros(2))
    else:
        model.second = model.first

    with pytest.raises(ValueError, match="split a parameter alias|merged distinct"):
        layout.refresh(model)

    assert layout.parameter_bindings.keys() == bindings.keys()
    assert all(
        layout.parameter_bindings[name] is value for name, value in bindings.items()
    )
    assert dict(layout.replacements.parameter_targets) == replacements


def test_materialization_refresh_preserves_aliases_and_composes_replacements():
    from torch import nn

    from llmtuner.parallel.assembly import ModelPartLayout

    model = nn.Module()
    model.first = nn.Parameter(torch.ones(2))
    model.alias = model.first
    model.second = nn.Parameter(torch.ones(2))
    original_first, original_second = model.first, model.second
    layout = ModelPartLayout()
    layout.refresh(model)
    for _ in range(2):
        model.first = nn.Parameter(torch.zeros(2))
        model.alias = model.first
        model.second = nn.Parameter(torch.zeros(2))
        layout.refresh(model)
        layout.validate(model)

    assert layout.replacements.parameter_targets[original_first] == (model.first,)
    assert layout.replacements.parameter_targets[original_second] == (model.second,)
    assert layout.parameter_aliases[model.first] == ("first", "alias")


def test_layout_rejects_sequence_axes_outside_captured_topology():
    from torch import nn

    from llmtuner.parallel.assembly import ModelPartLayout
    from llmtuner.parallel.topology import ParallelTopologySnapshot

    model = nn.Linear(2, 2)
    model.execution_capabilities = ModelExecutionCapabilities()
    execution = ExecutionLayout(
        output="logits",
        sequence_shard_axes=("tp",),
        vocabulary_sharded=False,
        vocabulary_size=None,
        capabilities=model.execution_capabilities,
        chunked_loss=False,
    )
    layout = ModelPartLayout(execution=execution, topology=ParallelTopologySnapshot())
    layout.refresh(model)
    with pytest.raises(ValueError, match="sequence layout disagrees with topology"):
        layout.validate(model)


def test_local_parts_cannot_mix_topology_snapshots():
    from torch import nn

    from llmtuner.parallel.assembly import (
        ModelPartLayout,
        validate_model_part_layouts,
    )
    from llmtuner.parallel.topology import ParallelTopologySnapshot

    parts = [nn.Linear(2, 2), nn.Linear(2, 2)]
    layouts = (
        ModelPartLayout(topology=ParallelTopologySnapshot()),
        ModelPartLayout(topology=ParallelTopologySnapshot(dp_shard=2, world_size=2)),
    )
    for model, layout in zip(parts, layouts, strict=True):
        layout.refresh(model)
    with pytest.raises(ValueError, match="same declared topology"):
        validate_model_part_layouts(parts, layouts)
