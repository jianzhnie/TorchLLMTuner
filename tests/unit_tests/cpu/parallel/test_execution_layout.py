"""Model and parallel stage declarations must agree before execution."""

import pytest
import torch

from llmtuner.models.execution import ModelExecutionCapabilities
from llmtuner.parallel.execution import (
    ExecutionLayout,
)


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
        ModelPartLayout(topology=ParallelTopologySnapshot(world_size=2)),
    )
    for model, layout in zip(parts, layouts, strict=True):
        layout.refresh(model)
    with pytest.raises(ValueError, match="same declared topology"):
        validate_model_part_layouts(parts, layouts)
