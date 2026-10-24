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


def test_rng_checkpoint_validation_does_not_advance_live_stream():
    checkpoint = RNGCheckpointState(torch.device("cpu"))
    saved = checkpoint.state_dict()
    torch.rand(7)
    before = torch.get_rng_state().clone()
    checkpoint.validate_state_dict(saved)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("field,value", [("world_size", 2), ("version", 2)])
def test_manager_rejects_incompatible_rng_before_mutating_training_state(
    tmp_path, field, value
):
    from copy import deepcopy

    from llmtuner.components.checkpointer.distributed_checkpoint import (
        DistributedCheckpointManager,
    )
    from llmtuner.config.checkpoint_config import CheckpointConfig

    with preserve_rng_state(torch.device("cpu")):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        rng = RNGCheckpointState(torch.device("cpu"))
        manager = DistributedCheckpointManager(
            CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=scheduler,
            states={"rng_state": rng},
            folder=str(tmp_path),
        )
        try:
            model(torch.ones(2, 2)).sum().backward()
            optimizer.step()
            scheduler.step()
            saved_rng = rng.state_dict()
            payload = pickle.loads(saved_rng[rng.key])
            payload[field] = value

            # Keep the normal manager save path; inject only saved metadata.
            class IncompatibleRNG:
                def state_dict(self):
                    return {rng.key: pickle.dumps(payload)}

                def load_state_dict(self, state):
                    raise AssertionError("Save-only incompatible RNG fixture")

            manager.states["rng_state"] = IncompatibleRNG()
            assert manager.save(1)
            manager.states["rng_state"] = rng
            optimizer.zero_grad()
            model(torch.full((2, 2), 3.0)).sum().backward()
            optimizer.step()
            scheduler.step()
            before_model = deepcopy(model.state_dict())
            before_optimizer = deepcopy(optimizer.state_dict())
            before_scheduler = deepcopy(scheduler.state_dict())
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            before_cpu = torch.get_rng_state().clone()
            with pytest.raises(ValueError, match="RNG checkpoint"):
                manager.load()
            torch.testing.assert_close(model.state_dict(), before_model, rtol=0, atol=0)
            torch.testing.assert_close(
                optimizer.state_dict(), before_optimizer, rtol=0, atol=0
            )
            assert scheduler.state_dict() == before_scheduler
            assert random.getstate() == before_python
            after_numpy = np.random.get_state()
            np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
            assert after_numpy[0] == before_numpy[0]
            assert after_numpy[2:] == before_numpy[2:]
            assert torch.equal(torch.get_rng_state(), before_cpu)
        finally:
            manager.close()


@pytest.mark.parametrize("mismatch", ["dp_degree", "batch_layout", "steps"])
def test_manager_rejects_invalid_data_cursor_before_model_or_optimizer_load(
    tmp_path, mismatch
):
    from copy import deepcopy

    from llmtuner.components.checkpointer.distributed_checkpoint import (
        DistributedCheckpointManager,
    )
    from llmtuner.config.checkpoint_config import CheckpointConfig
    from llmtuner.datasets.synthetic import RandomTokenDataLoader

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loader = RandomTokenDataLoader(seed=42, vocab_size=8, batch_size=2, seq_len=4)
    manager = DistributedCheckpointManager(
        CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={"dataloader": loader},
        folder=str(tmp_path),
    )
    try:
        saved = loader.state_dict()
        if mismatch == "dp_degree":
            saved["dp_world_size"] = 2
        elif mismatch == "batch_layout":
            saved["batch_layout"]["seq_len"] = 8
        else:
            saved["steps"] = -1
        # Real DCP save, with only the loader metadata deliberately malformed.
        from unittest.mock import patch

        with patch.object(loader, "state_dict", return_value=saved):
            assert manager.save(1)
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        next(iter(loader))
        before_model = deepcopy(model.state_dict())
        before_optimizer = deepcopy(optimizer.state_dict())
        before_scheduler = deepcopy(scheduler.state_dict())
        before_loader = deepcopy(loader.state_dict())
        with pytest.raises(ValueError):
            manager.load()
        torch.testing.assert_close(model.state_dict(), before_model, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state_dict(), before_optimizer, rtol=0, atol=0
        )
        assert scheduler.state_dict() == before_scheduler
        assert loader.state_dict() == before_loader
    finally:
        manager.close()


@pytest.mark.parametrize("field", ["step", "ntokens_seen"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_manager_rejects_invalid_trainer_counters_before_live_state_changes(
    tmp_path, field, value
):
    from copy import deepcopy

    from llmtuner.components.checkpointer.distributed_checkpoint import (
        DistributedCheckpointManager,
    )
    from llmtuner.config.checkpoint_config import CheckpointConfig
    from llmtuner.trainer.trainer import Trainer

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer = Trainer.__new__(Trainer)
    trainer.step = 1
    trainer.ntokens_seen = 8
    setattr(trainer, field, value)
    manager = DistributedCheckpointManager(
        CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={"train_state": trainer},
        folder=str(tmp_path),
    )
    try:
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        assert manager.save(1)
        trainer.step = 2
        trainer.ntokens_seen = 16
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        before_model = deepcopy(model.state_dict())
        before_optimizer = deepcopy(optimizer.state_dict())
        before_scheduler = deepcopy(scheduler.state_dict())
        before_trainer = trainer.state_dict()
        with pytest.raises(ValueError, match=f"trainer {field}"):
            manager.load()
        torch.testing.assert_close(model.state_dict(), before_model, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state_dict(), before_optimizer, rtol=0, atol=0
        )
        assert scheduler.state_dict() == before_scheduler
        assert trainer.state_dict() == before_trainer
    finally:
        manager.close()


@pytest.mark.parametrize("epoch", [-1, True, 1.5, 5])
def test_manager_rejects_invalid_scheduler_progress_before_live_writes(tmp_path, epoch):
    from copy import deepcopy
    from unittest.mock import patch

    from llmtuner.components.checkpointer.distributed_checkpoint import (
        DistributedCheckpointManager,
    )
    from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
    from llmtuner.config.checkpoint_config import CheckpointConfig
    from llmtuner.config.lr_scheduler_config import LRSchedulerConfig

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(), optimizers=[optimizer], training_steps=4
    )
    manager = DistributedCheckpointManager(
        CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={},
        folder=str(tmp_path),
    )
    try:
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        with patch.object(scheduler, "state_dict", return_value={"last_epoch": epoch}):
            assert manager.save(1)
        model(torch.full((2, 2), 3.0)).sum().backward()
        optimizer.step()
        scheduler.step()
        before_model = deepcopy(model.state_dict())
        before_optimizer = deepcopy(optimizer.state_dict())
        before_scheduler = deepcopy(scheduler.state_dict())
        with pytest.raises(ValueError, match="scheduler last_epoch"):
            manager.load()
        torch.testing.assert_close(model.state_dict(), before_model, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state_dict(), before_optimizer, rtol=0, atol=0
        )
        assert scheduler.state_dict() == before_scheduler
    finally:
        manager.close()
