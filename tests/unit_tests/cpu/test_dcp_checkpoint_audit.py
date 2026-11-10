"""Independent contracts and real DCP files, including damaged .metadata."""

import copy
import json
import pickle
from dataclasses import asdict, replace

import grain.python as grain
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex

from llmtuner.components.checkpointer.core.random_state import RNGCheckpointState
from llmtuner.components.optimizer.state_dict import get_flat_optim_state_dict
from llmtuner.datasets.dataloader import GrainDataLoader
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.audit_dcp_checkpoint import (
    AuditContract,
    TensorSpec,
    audit_checkpoint,
    validate_chunks,
)


def spec(tensor):
    return TensorSpec(tuple(tensor.shape), str(tensor.dtype))


@pytest.fixture
def production_state():
    parameter = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "param_names": ["weight"]}],
        foreach=False,
        fused=False,
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameter]["step"].fill_(100)
    optim = get_flat_optim_state_dict(optimizer)
    trainer = object.__new__(Trainer)
    trainer.step, trainer.ntokens_seen = 100, 800
    degrees = dict(
        dp_replicate=1, dp_shard=8, cp=1, tp=1, pp=1, ep=1, cp_kv=1, cp_ulysses=1
    )
    layout = dict(
        num_tokens_per_batch=32,
        max_context_length=32,
        num_rows_per_batch=1,
        max_num_documents=1,
        data_identity="independent-test-corpus-and-tokenizer-digest",
    )
    # Use production serializers and an actual Grain iterator without training
    # or creating a distributed process group. Cursor schema comes from a NEW
    # iterator template, not either checkpoint under audit.
    loader = object.__new__(GrainDataLoader)
    loader._dp_world_size = 8
    loader._batch_layout = layout
    loader._iterator = iter(grain.MapDataset.range(1000).to_iter_dataset())
    cursor_template, _ = flatten_state_dict(loader._iterator.get_state())
    data = {}
    for rank in range(8):
        loader._rank_id = f"dp_rank_{rank}"
        data.update(loader.state_dict())
    rng = {}
    for rank in range(8):
        serializer = object.__new__(RNGCheckpointState)
        serializer.device = torch.device("cpu")
        serializer.rank, serializer.world_size = rank, 8
        serializer.key = f"rank_{rank}"
        serializer.parallel_degrees = degrees
        rng.update(serializer.state_dict())
    loader._iterator.close()
    state = dict(
        model={"weight": parameter.detach()},
        optimizer=optim,
        train_state=trainer.state_dict(),
        lr_scheduler={"last_epoch": 100},
        dataloader=data,
        rng_state=rng,
    )
    # Build only from the live recipe objects, before either DCP save/read.
    contract = AuditContract(
        world_size=8,
        step=100,
        ntokens_seen=800,
        device_type="cpu",
        parallel_degrees=degrees,
        expected_fqns={"weight": spec(parameter)},
        optimizer_tensors={
            k: spec(v) for k, v in optim.items() if isinstance(v, torch.Tensor)
        },
        optimizer_values={
            k: v for k, v in optim.items() if not isinstance(v, torch.Tensor)
        },
        optimizer_kinds={"weight": "AdamW"},
        optimizer_steps={"weight": 100},
        data_layout=layout,
        cursor_schema={k: type(v).__name__ for k, v in cursor_template.items()},
        ema=None,
        ema_values={},
    )
    return state, contract


def save(path, state):
    dcp.save(state, checkpoint_id=path, no_dist=True)
    return path


def damage(path, transform):
    file = path / ".metadata"
    metadata = pickle.loads(file.read_bytes())
    transform(metadata)
    file.write_bytes(pickle.dumps(metadata))


def test_real_production_serializers_pass(tmp_path, production_state):
    state, contract = production_state
    path = save(tmp_path / "valid", state)
    result = audit_checkpoint(path, contract)
    assert result["keys"] > 30
    assert "structural" in result["scope"]


@pytest.mark.parametrize(
    "missing",
    [
        "model.weight",
        "optimizer.state.weight.exp_avg",
        "optimizer.param_groups.weight.betas",
        "train_state.ntokens_seen",
        "dataloader.dp_rank_7",
        "dataloader.batch_layout.data_identity",
        "rng_state.rank_7",
        "lr_scheduler.last_epoch",
    ],
)
def test_common_omissions_in_both_checkpoints_fail(tmp_path, production_state, missing):
    state, contract = production_state
    flat, _ = flatten_state_dict(state)
    for key in list(flat):
        if key == missing or key.startswith(missing + "."):
            del flat[key]
    for name in ("continuous", "resumed"):
        path = save(tmp_path / name, flat)
        with pytest.raises(ValueError, match="schema mismatch"):
            audit_checkpoint(path, contract)


@pytest.mark.parametrize("fault", ["hole", "empty", "overlap", "bounds", "ndim"])
def test_real_dcp_corrupt_chunk_geometry_fails(tmp_path, production_state, fault):
    state, contract = production_state
    path = save(tmp_path / fault, state)

    def mutate(metadata):
        tensor = metadata.state_dict_metadata["model.weight"]
        if fault == "empty":
            tensor.chunks = []
        elif fault == "hole":
            tensor.chunks[0].sizes = torch.Size([1, 3])
        elif fault == "overlap":
            tensor.chunks.append(copy.deepcopy(tensor.chunks[0]))
        elif fault == "bounds":
            tensor.chunks[0].offsets = torch.Size([1, 0])
        else:
            tensor.chunks[0].sizes = torch.Size([6])

    damage(path, mutate)
    with pytest.raises(ValueError, match="coverage|overlapping|bounds|ndim"):
        audit_checkpoint(path, contract)


def test_legal_repartitioned_metadata_passes(tmp_path, production_state):
    state, contract = production_state
    # Real records, with a different legal partition than the model state.
    state["half0"], state["half1"] = state["model"]["weight"].chunk(2)
    path = save(tmp_path / "partitioned", state)

    def repartition(metadata):
        tensor = metadata.state_dict_metadata["model.weight"]
        tensor.chunks = [
            ChunkStorageMetadata(torch.Size([0, 0]), torch.Size([1, 3])),
            ChunkStorageMetadata(torch.Size([1, 0]), torch.Size([1, 3])),
        ]
        for index in list(metadata.storage_data):
            if index.fqn == "model.weight":
                del metadata.storage_data[index]
        for rank in range(2):
            old = MetadataIndex(f"half{rank}", torch.Size([0, 0]))
            new = MetadataIndex("model.weight", torch.Size([rank, 0]), rank)
            metadata.storage_data[new] = metadata.storage_data.pop(old)
            del metadata.state_dict_metadata[f"half{rank}"]

    damage(path, repartition)
    assert audit_checkpoint(path, contract)["keys"] > 0
    loaded = {"model.weight": torch.empty(2, 3)}
    dcp.load(loaded, checkpoint_id=path, no_dist=True)
    torch.testing.assert_close(loaded["model.weight"], state["model"]["weight"])


def test_json_contract_roundtrip(tmp_path, production_state):
    state, contract = production_state
    restored = AuditContract.from_dict(json.loads(json.dumps(asdict(contract))))
    path = save(tmp_path / "json", state)
    assert audit_checkpoint(path, restored)["keys"] > 0


def test_native_flat_model_namespace(tmp_path, production_state):
    state, contract = production_state
    weights = state.pop("model")
    state.update({f"model.{k}": v for k, v in weights.items()})
    # Native backend merges wrapper FQNs at top level, not under a model root.
    contract = replace(
        contract,
        model_prefix="",
        expected_fqns={f"model.{k}": v for k, v in contract.expected_fqns.items()},
        optimizer_kinds={"model.weight": "AdamW"},
    )
    for field in ("optimizer_tensors", "optimizer_values", "optimizer_steps"):
        values = getattr(contract, field)
        renamed = {
            k.replace("state.weight.", "state.model.weight.").replace(
                "param_groups.weight.", "param_groups.model.weight."
            )
            if k != "weight"
            else "model.weight": v
            for k, v in values.items()
        }
        contract = replace(contract, **{field: renamed})
    state["optimizer"] = {
        k.replace("state.weight.", "state.model.weight.").replace(
            "param_groups.weight.", "param_groups.model.weight."
        )
        if k != "weight"
        else "model.weight": v
        for k, v in state["optimizer"].items()
    }
    assert audit_checkpoint(save(tmp_path / "flat", state), contract)["keys"] > 0


@pytest.mark.parametrize("field", ["device", "python", "cpu", "world_size"])
def test_rng_payload_missing_field_fails(tmp_path, production_state, field):
    state, contract = production_state
    payload = pickle.loads(state["rng_state"]["rank_7"])
    del payload[field]
    state["rng_state"]["rank_7"] = pickle.dumps(payload)
    path = save(tmp_path / "rng", state)
    with pytest.raises(ValueError, match="RNG schema"):
        audit_checkpoint(path, contract)


@pytest.mark.parametrize("key", ["train_state", "lr_scheduler", "optimizer"])
def test_wrong_step100_fails(tmp_path, production_state, key):
    state, contract = production_state
    if key == "optimizer":
        state[key]["state.weight.step"].fill_(50)
    else:
        state[key]["last_epoch" if key == "lr_scheduler" else "step"] = 50
    path = save(tmp_path / key, state)
    with pytest.raises(ValueError, match="mismatch"):
        audit_checkpoint(path, contract)


def test_explicit_ema_missing_or_unexpected_fails(tmp_path, production_state):
    state, contract = production_state
    path = save(tmp_path / "without", state)
    enabled = replace(
        contract, ema={"state.weight.ema_params": contract.expected_fqns["weight"]}
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        audit_checkpoint(path, enabled)
    state["ema"] = {"state.weight.ema_params": state["model"]["weight"]}
    path = save(tmp_path / "with", state)
    with pytest.raises(ValueError, match="schema mismatch"):
        audit_checkpoint(path, contract)
    assert audit_checkpoint(path, enabled)["keys"] > 0


@pytest.mark.parametrize("fault", ["missing_index", "missing_file", "truncated"])
def test_real_storage_corruption_fails(tmp_path, production_state, fault):
    state, contract = production_state
    path = save(tmp_path / fault, state)
    metadata = dcp.FileSystemReader(path).read_metadata()
    index = next(i for i in metadata.storage_data if i.fqn == "model.weight")
    if fault == "missing_index":
        damage(path, lambda m: m.storage_data.pop(index))
    else:
        file = path / metadata.storage_data[index].relative_path
        if fault == "missing_file":
            file.unlink()
        else:
            file.write_bytes(file.read_bytes()[:10])
    with pytest.raises((ValueError, FileNotFoundError)):
        audit_checkpoint(path, contract)


def test_control_memory_preflight_rejects(tmp_path, production_state):
    state, contract = production_state
    path = save(tmp_path / "memory", state)
    with pytest.raises(ValueError, match="memory preflight"):
        audit_checkpoint(path, replace(contract, max_control_bytes=1))


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_real_scalar_and_empty_geometry(tmp_path, shape):
    path = save(tmp_path / "geometry", {"weight": torch.empty(shape)})
    metadata = dcp.FileSystemReader(path).read_metadata()
    validate_chunks("weight", metadata.state_dict_metadata["weight"])


def test_incomplete_optimizer_contract_itself_is_rejected(production_state):
    _, contract = production_state
    tensors = dict(contract.optimizer_tensors)
    del tensors["state.weight.exp_avg_sq"]
    with pytest.raises(ValueError, match="contract missing"):
        replace(contract, optimizer_tensors=tensors).validate()


def test_scheduler_schema_extension_is_explicit(tmp_path, production_state):
    state, contract = production_state
    state["lr_scheduler"]["extra_progress"] = 123
    path = save(tmp_path / "scheduler", state)
    with pytest.raises(ValueError, match="schema mismatch"):
        audit_checkpoint(path, contract)
    extended = replace(
        contract, scheduler_values={"last_epoch": 100, "extra_progress": 123}
    )
    assert audit_checkpoint(path, extended)["keys"] > 0


@pytest.mark.parametrize("fault", ["extra_byte", "missing_byte", "alias_record"])
def test_real_control_storage_indices_are_strict(tmp_path, production_state, fault):
    state, contract = production_state
    path = save(tmp_path / fault, state)

    def mutate(metadata):
        index = MetadataIndex("train_state.step")
        if fault == "missing_byte":
            del metadata.storage_data[index]
        elif fault == "extra_byte":
            metadata.storage_data[
                MetadataIndex("train_state.step", torch.Size([0]))
            ] = metadata.storage_data[index]
        else:
            metadata.storage_data[MetadataIndex("train_state.ntokens_seen")] = (
                metadata.storage_data[index]
            )

    damage(path, mutate)
    with pytest.raises(ValueError, match="storage indices|overlapping storage"):
        audit_checkpoint(path, contract)


def global_rows_state(state, contract):
    from llmtuner.datasets.packing.global_rows import GlobalRowsIterDataset

    schema = None
    values = {}
    for rank in range(8):
        graph = GlobalRowsIterDataset(
            grain.MapDataset.range(1000).to_iter_dataset(),
            global_rows=8,
            dp_rank=rank,
            dp_world_size=8,
        )
        iterator = iter(graph)
        cursor = iterator.get_state()
        iterator.close()
        state["dataloader"][f"dp_rank_{rank}"] = cursor
        flat, _ = flatten_state_dict(cursor)
        schema = {k: type(v).__name__ for k, v in flat.items()}
        values[f"dp_rank_{rank}"] = {"geometry": (8, rank, 8)}
    return replace(contract, cursor_schema=schema, cursor_rank_values=values)


def test_global_rows_geometry_real_dcp_roundtrip(tmp_path, production_state):
    state, contract = production_state
    contract = global_rows_state(state, contract)
    contract = AuditContract.from_dict(json.loads(json.dumps(asdict(contract))))
    audit_checkpoint(save(tmp_path / "global_rows", state), contract)


@pytest.mark.parametrize("geometry", [(8, 0, 8), (8, 1, 4), (True, 1, 8), (8, True, 8)])
def test_global_rows_wrong_rank_geometry_rejected(tmp_path, production_state, geometry):
    state, contract = production_state
    contract = global_rows_state(state, contract)
    state["dataloader"]["dp_rank_1"]["geometry"] = geometry
    with pytest.raises(ValueError, match="cursor value"):
        audit_checkpoint(save(tmp_path / "bad_geometry", state), contract)


def test_tuple_cursor_without_independent_value_rejected(production_state):
    state, contract = production_state
    contract = global_rows_state(state, contract)
    with pytest.raises(ValueError, match="explicit per-rank"):
        replace(contract, cursor_rank_values=None).validate()
