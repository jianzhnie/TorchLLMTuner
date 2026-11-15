"""Local PP metadata and schedule doubles for assembly-only unit checks."""

from types import SimpleNamespace

import torch

from llmtuner.models.execution import PipelineStageIO
from llmtuner.parallel import parallelize
from llmtuner.parallel.pipeline_parallel.pipeline_parallel import PipelineLoss


def stub_pipeline(monkeypatch, parts):
    """Declare a complete local chain while replacing only its communication."""
    stages = []
    for index, part in enumerate(parts):
        first, last = index == 0, index == len(parts) - 1
        configure = getattr(part, "configure_pipeline_stage", None)
        if callable(configure):
            configure(first=first, last=last)
        else:
            part.pipeline_stage_io = PipelineStageIO(
                first=first, last=last, hidden_size=2, activation_dtype=torch.float32
            )
        stages.append(SimpleNamespace(submod=part, stage_index=index))
    group = object()
    mesh = SimpleNamespace(size=lambda: 1, get_group=lambda: group)

    def gather(destination, value, *, group):
        assert group is mesh.get_group()
        destination[:] = [value]

    monkeypatch.setattr(parallelize.dist, "all_gather_object", gather)
    topology = SimpleNamespace(
        pp_enabled=True,
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=2,
        ep=1,
        world_size=2,
        cp_kv=1,
        cp_ulysses=1,
        get_optional_mesh=lambda name: None,
        get_mesh=lambda name: mesh,
        spmd_dense_mesh=lambda: None,
    )
    schedule = SimpleNamespace(
        _stages=stages,
        _has_backward=True,
        _loss_fn=PipelineLoss(),
        _step_microbatches=lambda *args, **kwargs: None,
        eval=lambda **kwargs: None,
    )
    return stages, topology, schedule
