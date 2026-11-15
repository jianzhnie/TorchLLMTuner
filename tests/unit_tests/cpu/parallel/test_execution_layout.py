"""Model and parallel stage declarations must agree before execution."""

from dataclasses import replace

import pytest
import torch

from llmtuner.models.execution import ModelExecutionCapabilities, PipelineStageIO
from llmtuner.parallel.execution import (
    ExecutionLayout,
    validate_execution_chain,
    validate_local_pipeline_layouts,
)


def stage_layout(*, first, last, hidden_size, dtype=torch.float32):
    return ExecutionLayout(
        output="logits" if last else "pipeline_hidden",
        sequence_shard_axes=(),
        vocabulary_sharded=False,
        vocabulary_size=32,
        capabilities=ModelExecutionCapabilities(hidden_states=True, hidden_size=8),
        chunked_loss=False,
        pipeline_io=PipelineStageIO(first, last, hidden_size, dtype),
    )


@pytest.mark.parametrize("first,last", [(True, False), (False, False), (False, True)])
def test_pipeline_stage_rejects_width_conflicting_with_model(first, last):
    with pytest.raises(ValueError, match="disagrees with model hidden size"):
        stage_layout(first=first, last=last, hidden_size=4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matching_model_width_does_not_bypass_adjacent_dtype_validation(dtype):
    first = stage_layout(first=True, last=False, hidden_size=8)
    last = stage_layout(first=False, last=True, hidden_size=8, dtype=dtype)
    if dtype == torch.float32:
        validate_execution_chain((first, last))
    else:
        with pytest.raises(ValueError, match="activation dtype"):
            validate_execution_chain((first, last))


@pytest.mark.parametrize("has_first,has_last", [(False, True), (True, False)])
def test_local_pipeline_layouts_reject_wrong_owner_flags(has_first, has_last):
    layouts = (
        stage_layout(first=True, last=False, hidden_size=8),
        stage_layout(first=False, last=True, hidden_size=8),
    )
    with pytest.raises(ValueError, match="owner disagrees"):
        validate_local_pipeline_layouts(
            layouts, has_first_stage=has_first, has_last_stage=has_last
        )


def test_local_pipeline_layouts_reject_missing_io_on_logits_part():
    layout = replace(
        stage_layout(first=False, last=True, hidden_size=8), pipeline_io=None
    )
    with pytest.raises(ValueError, match="stage I/O declarations"):
        validate_local_pipeline_layouts(
            (layout,), has_first_stage=False, has_last_stage=True
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_local_pipeline_layouts_check_dtype_without_requiring_global_endpoints(dtype):
    layouts = (
        stage_layout(first=False, last=False, hidden_size=8),
        stage_layout(first=False, last=False, hidden_size=8, dtype=dtype),
    )
    if dtype == torch.float32:
        validate_local_pipeline_layouts(
            layouts, has_first_stage=False, has_last_stage=False
        )
    else:
        with pytest.raises(ValueError, match="activation dtype"):
            validate_local_pipeline_layouts(
                layouts, has_first_stage=False, has_last_stage=False
            )


@pytest.mark.parametrize("role", ["first", "last"])
def test_local_pipeline_ownership_requires_unique_endpoint(role):
    layout = stage_layout(first=role == "first", last=role == "last", hidden_size=8)
    with pytest.raises(ValueError, match="owner disagrees"):
        validate_local_pipeline_layouts(
            (layout, layout),
            has_first_stage=role == "first",
            has_last_stage=role == "last",
        )


def test_local_pipeline_layouts_reject_different_sequence_partitioning():
    first = stage_layout(first=True, last=False, hidden_size=8)
    last = replace(
        stage_layout(first=False, last=True, hidden_size=8),
        sequence_shard_axes=("tp",),
    )
    with pytest.raises(ValueError, match="sequence shard axes"):
        validate_local_pipeline_layouts(
            (first, last), has_first_stage=True, has_last_stage=True
        )


@pytest.mark.parametrize("changed", [None, "capabilities", "pipeline_io", "vocabulary"])
def test_transformed_model_must_preserve_assembled_execution_contract(changed):
    layout = stage_layout(first=False, last=True, hidden_size=8)
    model = torch.nn.Module()
    model.execution_capabilities = layout.capabilities
    model.pipeline_stage_io = layout.pipeline_io
    model.vocab_size = layout.vocabulary_size
    if changed == "capabilities":
        model.execution_capabilities = ModelExecutionCapabilities()
    elif changed == "pipeline_io":
        model.pipeline_stage_io = replace(layout.pipeline_io, first=True)
    elif changed == "vocabulary":
        model.vocab_size += 1
    if changed is None:
        layout.validate_model(model)
    else:
        with pytest.raises(ValueError, match="Transformed model"):
            layout.validate_model(model)


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
        ModelPartLayout(topology=ParallelTopologySnapshot(pp=2, world_size=2)),
    )
    for model, layout in zip(parts, layouts, strict=True):
        layout.refresh(model)
    with pytest.raises(ValueError, match="same declared topology"):
        validate_model_part_layouts(parts, layouts)
