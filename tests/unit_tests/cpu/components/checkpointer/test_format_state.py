"""Native format records participate in isolated restore preflight."""

import pytest

from llmtuner.components.checkpointer.core.checkpoint_keys import CHECKPOINT_FORMAT
from llmtuner.components.checkpointer.core.format_state import CheckpointFormatState
from llmtuner.components.checkpointer.core.state_adaptation import (
    prepare_local_state_preflight,
    validate_local_state_preflight,
)
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.errors import ConfigError


@pytest.mark.parametrize("version", [True, 0, 1, 3, "2", None])
def test_preflight_rejects_unsupported_format_without_mutating_writer(version):
    record = CheckpointFormatState()
    selected, destination = prepare_local_state_preflight({CHECKPOINT_FORMAT: record})
    destination[CHECKPOINT_FORMAT]["version"] = version
    with pytest.raises(ValueError, match="format version"):
        validate_local_state_preflight(selected, destination)
    assert record.state_dict() == {
        "name": "llmtuner.native",
        "version": CheckpointFormatState.VERSION,
        "model_contracts": "[]",
        "topology": "null",
    }


def test_native_format_name_is_required():
    record = CheckpointFormatState()
    with pytest.raises(ValueError, match="native training format"):
        record.load_state_dict({"name": "external", "version": 1})


def test_native_format_preflight_is_mandatory():
    with pytest.raises(ConfigError, match="cannot be excluded"):
        CheckpointConfig(exclude_from_loading=[CHECKPOINT_FORMAT])


def test_dcp_full_restore_rejects_unversioned_checkpoint(tmp_path):
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer.storage.load_planner import (
        CheckpointLoadPlanner,
    )

    dcp.save({"weight": torch.ones(2)}, checkpoint_id=tmp_path)
    destination = {
        CHECKPOINT_FORMAT: CheckpointFormatState().state_dict(),
        "weight": torch.zeros(2),
    }
    with pytest.raises(CheckpointException, match="no complete format record"):
        dcp.load(destination, checkpoint_id=tmp_path, planner=CheckpointLoadPlanner())
    assert not destination["weight"].any()
    model_only = {"weight": torch.zeros(2)}
    dcp.load(model_only, checkpoint_id=tmp_path, planner=CheckpointLoadPlanner())
    torch.testing.assert_close(model_only["weight"], torch.ones(2))


@pytest.mark.parametrize("category", ["rng_state", "dataloader"])
def test_versioned_restore_cannot_use_legacy_missing_metadata_fallback(
    tmp_path, category
):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer.storage.load_planner import (
        CheckpointLoadPlanner,
    )

    saved = {CHECKPOINT_FORMAT: CheckpointFormatState().state_dict()}
    if category == "dataloader":
        saved[category] = {"version": 1, "cursor": 7}
    dcp.save(saved, checkpoint_id=tmp_path)
    destination = {CHECKPOINT_FORMAT: CheckpointFormatState().state_dict()}
    destination[category] = (
        {"cpu": "required RNG record"}
        if category == "rng_state"
        else {"version": 1, "cursor": 0, "batch_layout": {"tokens": 32}}
    )
    with pytest.raises(CheckpointException, match=category):
        dcp.load(destination, checkpoint_id=tmp_path, planner=CheckpointLoadPlanner())


def test_model_configuration_is_snapshotted_and_validated_before_restore():
    from torch import nn

    from llmtuner.models.execution import ModelExecutionCapabilities

    model = nn.Module()
    model.execution_capabilities = ModelExecutionCapabilities()
    model.checkpoint_configuration = {
        "route_scale": 1.0,
        "rope": {"theta": 10000.0},
    }
    saved = CheckpointFormatState.capture([model])
    model.checkpoint_configuration["rope"]["theta"] = 1000000.0
    target = CheckpointFormatState.capture([model])
    with pytest.raises(ValueError, match="model execution contracts differ"):
        target.validate_state_dict(saved.state_dict())
    # Mutating the live config does not retroactively change a captured record.
    assert saved.state_dict() != target.state_dict()


def test_standalone_checkpoint_configuration_does_not_require_execution_contract():
    from torch import nn

    model = nn.Linear(2, 2)
    model.checkpoint_configuration = {"activation": "silu"}
    saved = CheckpointFormatState.capture([model]).state_dict()
    model.checkpoint_configuration = {"activation": "gelu"}
    target = CheckpointFormatState.capture([model])
    with pytest.raises(ValueError, match="model execution contracts differ"):
        target.validate_state_dict(saved)


def test_format_record_captures_assembly_topology_without_live_aliases():
    import json

    from torch import nn

    from llmtuner.parallel.assembly import ModelPartLayout
    from llmtuner.parallel.topology import ParallelTopologySnapshot

    layout = ModelPartLayout(topology=ParallelTopologySnapshot())
    record = CheckpointFormatState.capture([nn.Linear(2, 2)], runtime_layouts=(layout,))
    saved = record.state_dict()
    layout.topology = ParallelTopologySnapshot(tp=2, world_size=2)
    assert json.loads(saved["topology"])["tp"] == 1
    assert record.state_dict() == saved
    # Provenance alone does not establish a supported topology transition.
    other = CheckpointFormatState.capture([nn.Linear(2, 2)], runtime_layouts=(layout,))
    other.validate_state_dict(saved)


@pytest.mark.parametrize("topology", [None, "{", "{}", '"tp"', '{"tp": 2}'])
def test_format_preflight_rejects_missing_or_malformed_topology(topology):
    record = CheckpointFormatState()
    saved = record.state_dict()
    if topology is None:
        saved.pop("topology")
    else:
        saved["topology"] = topology
    with pytest.raises(ValueError, match="source topology record"):
        record.validate_state_dict(saved)


def test_format_preflight_rejects_inconsistent_topology_degrees():
    import json
    from dataclasses import asdict

    from llmtuner.parallel.topology import ParallelTopologySnapshot

    record = CheckpointFormatState()
    saved = record.state_dict()
    degrees = asdict(ParallelTopologySnapshot())
    degrees["tp"] = 2
    saved["topology"] = json.dumps(degrees)
    with pytest.raises(ValueError, match="source topology record"):
        record.validate_state_dict(saved)


def test_format_capture_rejects_inconsistent_local_topology():
    from torch import nn

    from llmtuner.parallel.assembly import ModelPartLayout
    from llmtuner.parallel.topology import ParallelTopologySnapshot

    layouts = (
        ModelPartLayout(topology=ParallelTopologySnapshot()),
        ModelPartLayout(topology=None),
    )
    with pytest.raises(ValueError, match="different topologies"):
        CheckpointFormatState.capture(
            [nn.Linear(2, 2), nn.Linear(2, 2)], runtime_layouts=layouts
        )


def test_format_capture_rejects_cross_rank_topology_disagreement(monkeypatch):
    from torch import nn

    from llmtuner.components.checkpointer.core import format_state
    from llmtuner.parallel.assembly import ModelPartLayout
    from llmtuner.parallel.topology import ParallelTopologySnapshot

    monkeypatch.setattr(format_state, "get_world_size", lambda: 2)

    def gather(peers, local):
        peers[:] = [local, (local[0], ParallelTopologySnapshot(tp=2, world_size=2))]

    monkeypatch.setattr(format_state.dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="topology declarations differ across ranks"):
        CheckpointFormatState.capture(
            [nn.Linear(2, 2)],
            runtime_layouts=(ModelPartLayout(topology=ParallelTopologySnapshot()),),
        )
