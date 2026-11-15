"""Strict DCP rejection of incomplete synthetic loader records."""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from llmtuner.components.checkpointer.storage.load_planner import CheckpointLoadPlanner
from tests.caps import require_env

require_env("dcp")


@pytest.mark.parametrize("saved_steps", [7, {"0": 7, "1": 9}])
def test_pre_layout_cursor_load_rejects_missing_metadata(tmp_path, saved_steps):
    checkpoint = tmp_path / "legacy"
    dcp.save(
        {
            "dataloader": {"dp_world_size": 2, "steps": saved_steps},
            "weight": torch.ones(3),
        },
        checkpoint_id=checkpoint,
    )
    target = {
        "dataloader": {
            "dp_world_size": 2,
            "steps": {"0": 0},
            "batch_layout": {"seed": 42, "seq_len": 32},
        },
        "weight": torch.zeros(3),
    }
    with pytest.raises(CheckpointException, match="dataloader"):
        dcp.load(target, checkpoint_id=checkpoint, planner=CheckpointLoadPlanner())
    torch.testing.assert_close(target["weight"], torch.zeros(3), rtol=0, atol=0)


def test_partial_rank_cursor_layout_remains_strict(tmp_path):
    checkpoint = tmp_path / "partial"
    dcp.save(
        {"dataloader": {"steps": {"0": 7}, "batch_layout": {"seed": 42}}},
        checkpoint_id=checkpoint,
    )
    target = {
        "dataloader": {
            "steps": {"0": 0},
            "batch_layout": {"seed": 42, "seq_len": 32},
        }
    }
    with pytest.raises(CheckpointException, match="batch_layout.seq_len"):
        dcp.load(target, checkpoint_id=checkpoint, planner=CheckpointLoadPlanner())


def test_flat_destination_rejects_missing_layout_without_mutating_live_state(tmp_path):
    checkpoint = tmp_path / "flat"
    dcp.save(
        {"dataloader": {"steps": {"0": 7}}, "weight": torch.ones(3)},
        checkpoint_id=checkpoint,
    )
    target = {
        "dataloader.steps.0": 0,
        "dataloader.batch_layout.seed": 42,
        "dataloader.batch_layout.seq_len": 32,
        "weight": torch.zeros(3),
    }
    with pytest.raises(CheckpointException, match="batch_layout"):
        dcp.load(
            target,
            checkpoint_id=checkpoint,
            planner=CheckpointLoadPlanner(flatten_state_dict=False),
        )
    assert target["dataloader.steps.0"] == 0
    torch.testing.assert_close(target["weight"], torch.zeros(3), rtol=0, atol=0)
