"""CPU evidence contracts without importing the reference/NPU Trainer entry."""

import ast
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Replicate, Shard

ENTRY = (
    Path(__file__).resolve().parents[3]
    / "integration_tests/qwen35_reference_training.py"
)


def helpers():
    names = {
        "tensor_info",
        "boundary",
        "from_hf_group",
        "canonical_native_fqn",
        "state_fingerprint",
        "checkpoint_state_fingerprint",
        "local_shard",
        "model_adam_manifest",
        "shared_head_tensor",
        "verify_initial_slice",
    }
    nodes = [
        node
        for node in ast.parse(ENTRY.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == len(names)
    namespace = {
        "torch": torch,
        "hashlib": hashlib,
        "json": json,
        "pickle": pickle,
        "np": np,
        "Any": Any,
        "Path": Path,
        "safe_open": safe_open,
        "DTensor": DTensor,
        "Replicate": Replicate,
        "Shard": Shard,
        "EXPERT_SUFFIXES": (".experts.gate_up_proj", ".experts.down_proj"),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(ENTRY), "exec"), namespace
    )
    return namespace


def test_genuine_head_boundary_and_no_source_mutation(tmp_path):
    h = helpers()
    head = torch.arange(12).reshape(4, 3).bfloat16()
    save_file({"lm_head.weight": head}, tmp_path / "head.safetensors")
    loaded = h["shared_head_tensor"](tmp_path, {"lm_head.weight": "head.safetensors"})
    assert torch.equal(head, loaded)

    class Adapter:
        def from_hf(self, state):
            assert state["lm_head.weight"] is loaded
            return {key: value for key, value in state.items()}

    source = {"model.layers.0.weight": torch.ones(2, 3, dtype=torch.bfloat16)}
    result = h["from_hf_group"](Adapter(), source, loaded)
    assert set(result) == {"model.language_model.layers.0.weight"}
    assert set(source) == {"model.layers.0.weight"}
    with pytest.raises(AssertionError):
        h["from_hf_group"](Adapter(), source)
    with pytest.raises(AssertionError):
        h["from_hf_group"](Adapter(), source, torch.empty(0))


def test_exact_initial_slice_empty_owners_dtype_and_corruption():
    verify = helpers()["verify_initial_slice"]
    weight = torch.ones(1, 4, dtype=torch.bfloat16)
    for rank in range(8):
        shard = weight[min(rank, 1) : min(rank, 1) + 1]
        assert verify(shard.clone(), shard, "gate")["empty_local_shard"] == (rank > 0)
    with pytest.raises(AssertionError, match="differs"):
        verify(weight + 1, weight, "gate")
    with pytest.raises(AssertionError):
        verify(weight.float(), weight, "gate")
    with pytest.raises(ValueError, match="Nonfinite"):
        helpers()["tensor_info"](torch.tensor([float("nan")]))


def test_real_adam_resume_manifest_roundtrip_and_corruption(tmp_path):
    h = helpers()
    model = torch.nn.Linear(4, 3, bias=False).bfloat16()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, foreach=False, fused=False
    )
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    baseline = h["model_adam_manifest"](model, [optimizer])
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    restored = torch.nn.Linear(4, 3, bias=False).bfloat16()
    other = torch.optim.AdamW(
        restored.parameters(), lr=3e-4, foreach=False, fused=False
    )
    payload = torch.load(path, weights_only=True)
    restored.load_state_dict(payload["model"])
    other.load_state_dict(payload["optimizer"])
    assert h["model_adam_manifest"](restored, [other]) == baseline
    state = other.state[restored.weight]
    for key in ("step", "exp_avg", "exp_avg_sq"):
        original = state[key].clone()
        state[key].add_(1)
        assert h["model_adam_manifest"](restored, [other]) != baseline
        state[key].copy_(original)
        assert h["model_adam_manifest"](restored, [other]) == baseline
    restored.weight.data.add_(1)
    assert h["model_adam_manifest"](restored, [other]) != baseline


def test_manifest_rejects_missing_ownership_and_wrong_moment_layout():
    h = helpers()
    model = torch.nn.Linear(4, 3, bias=False).bfloat16()
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.state[model.weight]["exp_avg"] = torch.zeros(2)
    with pytest.raises(AssertionError):
        h["model_adam_manifest"](model, [optimizer])
    with pytest.raises(AssertionError, match="coverage"):
        h["model_adam_manifest"](model, [])


def test_pickle_rng_fingerprint_tracks_values_not_storage_identity():
    fingerprint = helpers()["checkpoint_state_fingerprint"]
    a = {"rank_0": pickle.dumps({"cpu": torch.arange(16, dtype=torch.uint8)})}
    b = {"rank_0": pickle.dumps({"cpu": torch.arange(16, dtype=torch.uint8)})}
    assert fingerprint("rng_state", a) == fingerprint("rng_state", b)
    b = {"rank_0": pickle.dumps({"cpu": torch.zeros(16, dtype=torch.uint8)})}
    assert fingerprint("rng_state", a) != fingerprint("rng_state", b)


@pytest.mark.parametrize("coordinate_type", [list, tuple])
def test_dim0_layout_geometry_and_empty_ownership_without_collectives(coordinate_type):
    # Exercise the verifier's metadata arithmetic; this is not a simulated
    # distributed/FSDP execution or a replacement for a device acceptance test.
    h = helpers()

    class Mesh:
        shape = (8,)

        def __init__(self, rank):
            self.rank = rank

        def get_coordinate(self):
            return coordinate_type([self.rank])

    class LayoutTensor:
        shape = (1, 4)
        placements = (Shard(0),)

        def __init__(self, rank):
            self.device_mesh = Mesh(rank)
            self.local = torch.ones(1 if rank == 0 else 0, 4)

        def numel(self):
            return 4

        def to_local(self):
            return self.local

    h["DTensor"] = LayoutTensor
    for rank in range(8):
        local, layout = h["local_shard"](LayoutTensor(rank))
        assert layout["offset"] == min(rank, 1)
        assert layout["global_shape"] == [1, 4]
        assert layout["coordinate"] == [rank]
        assert layout == json.loads(json.dumps(layout))
        assert local.shape == (1 if rank == 0 else 0, 4)
    malformed = LayoutTensor(1)
    malformed.local = torch.ones(1, 4)
    with pytest.raises(AssertionError, match="local shard shape"):
        h["local_shard"](malformed)
    malformed = LayoutTensor(0)
    malformed.placements = (Shard(1),)
    with pytest.raises(AssertionError):
        h["local_shard"](malformed)
