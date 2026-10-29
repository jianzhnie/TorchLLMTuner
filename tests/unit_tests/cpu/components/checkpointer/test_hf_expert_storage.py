import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint.metadata import MetadataIndex

from llmtuner.components.checkpointer.hf_expert_storage import (
    GroupedExpertHuggingFaceStorageReader,
)
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter


def _checkpoint(tmp_path, *, missing=False):
    weights = {}
    for expert in range(3):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = (4, 2) if projection == "down_proj" else (2, 4)
            key = f"model.layers.0.mlp.experts.{expert}.{projection}.weight"
            weights[key] = torch.arange(8, dtype=torch.float32).reshape(shape) + (
                expert * 100
                + {"gate_proj": 0, "up_proj": 10, "down_proj": 20}[projection]
            )
    if missing:
        weights.pop("model.layers.0.mlp.experts.2.up_proj.weight")
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
    )
    return weights


@pytest.mark.parametrize("thread_count", [1, 2])
@pytest.mark.parametrize("indexed", [False, True])
def test_individual_hf_experts_load_into_fused_tensors(tmp_path, thread_count, indexed):
    weights = _checkpoint(tmp_path)
    if not indexed:
        (tmp_path / "model.safetensors.index.json").unlink()
    else:
        weight_map = {}
        for expert in range(3):
            filename = f"expert-{expert}.safetensors"
            subset = {
                key: value
                for key, value in weights.items()
                if f".experts.{expert}." in key
            }
            save_file(subset, str(tmp_path / filename))
            weight_map.update(dict.fromkeys(subset, filename))
        (tmp_path / "model.safetensors").unlink()
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    config = SimpleNamespace(model_type="qwen3_moe", num_experts=3)
    adapter = HFTransformerStateDictAdapter(config, str(tmp_path))
    prefix = "model.layers.0.mlp.experts."
    destination = {
        prefix + "gate_up_proj": torch.empty(3, 4, 4),
        prefix + "down_proj": torch.empty(3, 4, 2),
    }
    wrapper = {"model." + key: value for key, value in destination.items()}
    assert set(adapter.to_hf(wrapper)) == set(destination)
    dcp.load(
        destination,
        storage_reader=adapter.get_hf_storage_reader(
            str(tmp_path), thread_count=thread_count
        ),
    )
    for expert in range(3):
        for projection, target_slice in (
            ("gate_proj", slice(0, 2)),
            ("up_proj", slice(2, 4)),
        ):
            torch.testing.assert_close(
                destination[prefix + "gate_up_proj"][expert, target_slice],
                weights[f"{prefix}{expert}.{projection}.weight"],
                atol=0,
                rtol=0,
            )
        torch.testing.assert_close(
            destination[prefix + "down_proj"][expert],
            weights[f"{prefix}{expert}.down_proj.weight"],
            atol=0,
            rtol=0,
        )


def test_individual_hf_experts_reject_missing_projection(tmp_path):
    _checkpoint(tmp_path, missing=True)
    reader = GroupedExpertHuggingFaceStorageReader(str(tmp_path), num_experts=3)
    with pytest.raises(ValueError, match="Incomplete individual HF experts"):
        reader.read_metadata()


def test_individual_hf_experts_load_native_ep_stacks(tmp_path):
    weights = _checkpoint(tmp_path)
    prefix = "model.layers.0.mlp."
    gate = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    weights[prefix + "gate.weight"] = gate
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
    )
    destination = {
        prefix + "routed_experts.inner_experts." + role: torch.full(shape, float("nan"))
        for role, shape in (
            ("w1_EFD", (3, 2, 4)),
            ("w3_EFD", (3, 2, 4)),
            ("w2_EDF", (3, 4, 2)),
        )
    }
    adapter = HFTransformerStateDictAdapter(
        SimpleNamespace(model_type="qwen3_moe", num_experts=3),
        str(tmp_path),
        qwen3_expert_parallel=True,
    )
    wrapped = {"model." + key: value for key, value in destination.items()}
    wrapped["model." + prefix + "router.gate.weight"] = torch.full_like(
        gate, float("nan")
    )
    hf_state = adapter.to_hf(wrapped)
    assert prefix + "gate.weight" in hf_state
    assert prefix + "router.gate.weight" not in hf_state
    assert set(adapter.from_hf(hf_state)) == set(wrapped)
    dcp.load(hf_state, storage_reader=adapter.get_hf_storage_reader(str(tmp_path)))
    restored = adapter.from_hf(hf_state)
    torch.testing.assert_close(
        restored["model." + prefix + "router.gate.weight"], gate, atol=0, rtol=0
    )
    for projection, role in (
        ("gate_proj", "w1_EFD"),
        ("up_proj", "w3_EFD"),
        ("down_proj", "w2_EDF"),
    ):
        for expert in range(3):
            torch.testing.assert_close(
                destination[prefix + "routed_experts.inner_experts." + role][expert],
                weights[f"{prefix}experts.{expert}.{projection}.weight"],
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("native_ep_layout", [False, True])
def test_individual_hf_experts_reject_mixed_layout(tmp_path, native_ep_layout):
    weights = _checkpoint(tmp_path)
    weights["model.layers.0.mlp.experts.gate_up_proj"] = torch.zeros(3, 4, 4)
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
    )
    reader = GroupedExpertHuggingFaceStorageReader(
        str(tmp_path), num_experts=3, native_ep_layout=native_ep_layout
    )
    with pytest.raises(ValueError, match="fused"):
        reader.read_metadata()


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_individual_hf_expert_reads_only_requested_slice(tmp_path, projection):
    from safetensors import safe_open

    weights = _checkpoint(tmp_path)
    reader = GroupedExpertHuggingFaceStorageReader(str(tmp_path), num_experts=3)
    metadata = reader.read_metadata()
    reader.set_up_storage_reader(metadata, is_coordinator=True)
    prefix = "model.layers.0.mlp.experts."
    target = "down_proj" if projection == "down_proj" else "gate_up_proj"
    chunk_offset = (1, 2 if projection == "up_proj" else 0, 0)
    request = SimpleNamespace(
        storage_index=MetadataIndex(prefix + target, offset=torch.Size(chunk_offset)),
        storage_offsets=torch.Size((0, 1, 1)),
        lengths=torch.Size((1, 1, 1)),
    )
    destination = torch.full((1, 1, 1), float("nan"))
    committed = []
    planner = SimpleNamespace(
        resolve_tensor=lambda request: destination,
        commit_tensor=lambda request, tensor: committed.append(tensor),
    )
    with safe_open(str(tmp_path / "model.safetensors"), framework="pt") as tensors:
        reader._process_read_request(tensors, request, planner)
    expected = weights[f"{prefix}1.{projection}.weight"][1:2, 1:2].unsqueeze(0)
    torch.testing.assert_close(destination, expected, atol=0, rtol=0)
    assert len(committed) == 1


def test_individual_hf_experts_reject_duplicate_numeric_index(tmp_path):
    weights = _checkpoint(tmp_path)
    weights["model.layers.0.mlp.experts.01.gate_proj.weight"] = weights[
        "model.layers.0.mlp.experts.1.gate_proj.weight"
    ].clone()
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
    )
    reader = GroupedExpertHuggingFaceStorageReader(str(tmp_path), num_experts=3)
    with pytest.raises(ValueError, match="Duplicate individual HF expert index"):
        reader.read_metadata()
