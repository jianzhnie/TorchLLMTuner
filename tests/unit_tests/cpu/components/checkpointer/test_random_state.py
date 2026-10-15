"""Default RNG checkpoint round-trip and incompatible-state rejection."""

import pickle
import random

import numpy as np
import pytest
import torch

from llmtuner.components.checkpointer.random_state import RNGCheckpointState
from llmtuner.utils.seeding import preserve_rng_state


def test_rng_checkpoint_restores_all_cpu_streams():
    with preserve_rng_state(torch.device("cpu")):
        state = RNGCheckpointState(torch.device("cpu"))
        saved = state.state_dict()
        expected = (random.random(), np.random.random(), torch.rand(5))
        state.load_state_dict(saved)
        assert random.random() == expected[0]
        assert np.random.random() == expected[1]
        torch.testing.assert_close(torch.rand(5), expected[2], rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,value", [("version", 2), ("world_size", 2), ("device_type", "cuda")]
)
def test_rng_checkpoint_rejects_incompatible_metadata_without_changing_streams(
    field, value
):
    with preserve_rng_state(torch.device("cpu")):
        state = RNGCheckpointState(torch.device("cpu"))
        saved = state.state_dict()
        payload = pickle.loads(saved[state.key])
        payload[field] = value
        cpu_before = torch.get_rng_state().clone()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        with pytest.raises(ValueError):
            state.load_state_dict({state.key: pickle.dumps(payload)})
        assert torch.equal(torch.get_rng_state(), cpu_before)
        assert random.getstate() == python_before
        actual = np.random.get_state()
        assert actual[0] == numpy_before[0]
        np.testing.assert_array_equal(actual[1], numpy_before[1])
        assert actual[2:] == numpy_before[2:]


def test_rng_checkpoint_empty_state_is_noop():
    state = RNGCheckpointState(torch.device("cpu"))
    before = torch.get_rng_state().clone()
    state.load_state_dict({})
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("changed_axis", ["tp", "pp", "ep", "cp", "cp_kv"])
def test_rng_restore_rejects_changed_parallel_degrees(changed_axis):
    from types import SimpleNamespace

    degrees = dict(
        dp_replicate=1, dp_shard=2, cp=1, tp=1, pp=1, ep=1, cp_kv=1, cp_ulysses=1
    )
    topology = SimpleNamespace(**degrees)
    state = RNGCheckpointState(torch.device("cpu"), topology=topology)
    saved = state.state_dict()
    # Capture ownership: editing the caller's topology cannot edit the payload.
    setattr(topology, changed_axis, 2)
    target = RNGCheckpointState(torch.device("cpu"), topology=topology)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="parallel topology changed"):
        target.load_state_dict(saved)
    assert torch.equal(torch.get_rng_state(), before)
    state.load_state_dict(saved)


def test_legacy_dcp_without_rng_keeps_live_stream_and_restores_model(tmp_path):
    import torch.distributed.checkpoint as dcp

    from llmtuner.components.checkpointer.load_planner import (
        CheckpointLoadPlanner,
    )

    dcp.save({"weight": torch.ones(2)}, checkpoint_id=tmp_path)
    rng = RNGCheckpointState(torch.device("cpu"))
    target = {"weight": torch.zeros(2), "rng_state": rng.state_dict()}
    before = torch.get_rng_state().clone()
    planner = CheckpointLoadPlanner()
    dcp.load(target, checkpoint_id=tmp_path, planner=planner)
    assert not planner.has_rng_state
    assert torch.equal(target["weight"], torch.ones(2))
    assert torch.equal(torch.get_rng_state(), before)


def test_dcp_with_rng_rejects_missing_rank_instead_of_treating_it_as_legacy(tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer.load_planner import (
        CheckpointLoadPlanner,
    )

    rng = RNGCheckpointState(torch.device("cpu"))
    payload = rng.state_dict()[rng.key]
    dcp.save(
        {"weight": torch.ones(2), "rng_state": {"rank_1": payload}},
        checkpoint_id=tmp_path,
    )
    target = {"weight": torch.zeros(2), "rng_state": rng.state_dict()}
    before = torch.get_rng_state().clone()
    with pytest.raises(CheckpointException, match="rng_state.rank_0"):
        dcp.load(target, checkpoint_id=tmp_path, planner=CheckpointLoadPlanner())
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(target["weight"], torch.zeros(2))


@pytest.mark.parametrize("field", ["python", "numpy", "cpu"])
def test_invalid_host_rng_payload_preserves_all_live_streams(field):
    with preserve_rng_state(torch.device("cpu")):
        checkpoint = RNGCheckpointState(torch.device("cpu"))
        payload = pickle.loads(checkpoint.state_dict()[checkpoint.key])
        payload[field] = (
            torch.zeros(3, dtype=torch.uint8) if field == "cpu" else ("invalid",)
        )
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        before_cpu = torch.get_rng_state().clone()
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            checkpoint.load_state_dict({checkpoint.key: pickle.dumps(payload)})
        assert random.getstate() == before_python
        after_numpy = np.random.get_state()
        assert after_numpy[0] == before_numpy[0]
        np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
        assert after_numpy[2:] == before_numpy[2:]
        assert torch.equal(torch.get_rng_state(), before_cpu)


@pytest.mark.parametrize("field", ["version", "world_size"])
@pytest.mark.parametrize("value", [True, 1.0])
def test_rng_checkpoint_rejects_noninteger_metadata_without_rewinding(field, value):
    with preserve_rng_state(torch.device("cpu")):
        checkpoint = RNGCheckpointState(torch.device("cpu"))
        payload = pickle.loads(checkpoint.state_dict()[checkpoint.key])
        payload[field] = value
        random.random()
        np.random.random()
        torch.rand(3)
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        before_cpu = torch.get_rng_state().clone()
        with pytest.raises(ValueError):
            checkpoint.load_state_dict({checkpoint.key: pickle.dumps(payload)})
        assert random.getstate() == before_python
        after_numpy = np.random.get_state()
        np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
        assert after_numpy[2:] == before_numpy[2:]
        assert torch.equal(torch.get_rng_state(), before_cpu)


@pytest.mark.parametrize("state", [None, [], "", [1], "invalid"])
def test_rng_checkpoint_rejects_non_dictionary_state(state):
    checkpoint = RNGCheckpointState(torch.device("cpu"))
    with pytest.raises(ValueError, match="dictionary"):
        checkpoint.load_state_dict(state)


@pytest.mark.parametrize("payload", [None, [], "", [1], "invalid"])
def test_rng_checkpoint_rejects_non_dictionary_payload(payload):
    checkpoint = RNGCheckpointState(torch.device("cpu"))
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="payload must be a dictionary"):
        checkpoint.load_state_dict({checkpoint.key: pickle.dumps(payload)})
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "axis", ["dp_replicate", "dp_shard", "cp", "tp", "pp", "ep", "cp_kv", "cp_ulysses"]
)
@pytest.mark.parametrize("value", [True, 1.0])
def test_rng_checkpoint_rejects_noninteger_parallel_degree(axis, value):
    from types import SimpleNamespace

    degrees = dict(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, cp_kv=1, cp_ulysses=1
    )
    with preserve_rng_state(torch.device("cpu")):
        checkpoint = RNGCheckpointState(
            torch.device("cpu"), topology=SimpleNamespace(**degrees)
        )
        payload = pickle.loads(checkpoint.state_dict()[checkpoint.key])
        payload["parallel_degrees"][axis] = value
        torch.rand(3)
        before = torch.get_rng_state().clone()
        with pytest.raises(ValueError, match="parallel topology"):
            checkpoint.load_state_dict({checkpoint.key: pickle.dumps(payload)})
        assert torch.equal(torch.get_rng_state(), before)
