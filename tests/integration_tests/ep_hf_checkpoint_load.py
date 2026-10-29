"""Two CPU/Gloo ranks: individual HF weights restore native EP ownership."""

import json
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from llmtuner.components.checkpointer.hf_storage import save_hf_checkpoint
from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from llmtuner.parallel.expert_parallel.checkpointing import (
    expert_shard_map,
    mark_experts_ep_sharded,
    restore_sharded_expert_state,
    shard_expert_state,
)


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        paths = [tempfile.mkdtemp(prefix="ep-hf-load-") if rank == 0 else None]
        dist.broadcast_object_list(paths)
        path = Path(paths[0])
        prefix = "model.layers.0.mlp."
        weights = {prefix + "gate.weight": torch.arange(12.0).reshape(4, 3)}
        roles = {"gate_proj": "w1_EFD", "up_proj": "w3_EFD", "down_proj": "w2_EDF"}
        for expert in range(4):
            for index, projection in enumerate(roles):
                shape = (3, 2) if projection == "down_proj" else (2, 3)
                weights[f"{prefix}experts.{expert}.{projection}.weight"] = (
                    torch.arange(6.0).reshape(shape) + 100 * expert + 10 * index
                )
        if rank == 0:
            save_file(weights, str(path / "model.safetensors"))
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
            )
        dist.barrier()
        model = nn.Module()
        model.model = nn.Module()
        model.model.model = nn.Module()
        model.model.model.layers = nn.ModuleList([nn.Module()])
        mlp = nn.Module()
        model.model.model.layers[0].mlp = mlp
        mlp.router = nn.Module()
        mlp.router.gate = nn.Linear(3, 4, bias=False)
        mlp.routed_experts = nn.Module()
        grouped = nn.Module()
        mlp.routed_experts.inner_experts = grouped
        for projection, role in roles.items():
            shape = (2, 3, 2) if projection == "down_proj" else (2, 2, 3)
            grouped.register_parameter(
                role, nn.Parameter(torch.full(shape, float("nan")))
            )
        with torch.no_grad():
            mlp.router.gate.weight.fill_(float("nan"))
        mark_experts_ep_sharded(grouped, dist.group.WORLD)
        sharded = shard_expert_state(
            model.state_dict(), shard_map=expert_shard_map([model]), model_parts=[model]
        )
        expert_key = "model." + prefix + "routed_experts.inner_experts.w1_EFD"
        chunk = sharded[expert_key]
        assert chunk.shape == (4, 2, 3)
        assert chunk.chunks[0][0] == (rank * 2, 0, 0)
        assert chunk.chunks[0][1].data_ptr() == grouped.w1_EFD.data_ptr()
        hf_template = HFTransformerStateDictAdapter(
            SimpleNamespace(model_type="qwen3_moe", num_experts=4),
            str(path),
            qwen3_expert_parallel=True,
        )
        dcp.load(
            hf_template.to_hf(sharded),
            storage_reader=hf_template.get_hf_storage_reader(str(path)),
        )
        for projection, role in roles.items():
            expected = torch.stack(
                [
                    weights[f"{prefix}experts.{e}.{projection}.weight"]
                    for e in range(rank * 2, rank * 2 + 2)
                ]
            )
            torch.testing.assert_close(getattr(grouped, role), expected, atol=0, rtol=0)
        checkpoint_path = str(path / "ep-sharded")
        expected_local = {
            key: value.clone() for key, value in model.state_dict().items()
        }
        dcp.save(sharded, checkpoint_id=checkpoint_path)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float("nan"))
        dcp.load(sharded, checkpoint_id=checkpoint_path)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected_local[key], atol=0, rtol=0)
        # A replicated destination also reconstructs every expert from EP chunks.
        complete = {
            key: torch.empty(value.shape, dtype=value.dtype)
            for key, value in sharded.items()
        }
        dcp.load(complete, checkpoint_id=checkpoint_path)
        for projection, role in roles.items():
            key = "model." + prefix + "routed_experts.inner_experts." + role
            expected = torch.stack(
                [weights[f"{prefix}experts.{e}.{projection}.weight"] for e in range(4)]
            )
            torch.testing.assert_close(complete[key], expected, atol=0, rtol=0)
        with torch.no_grad():
            for parameter in grouped.parameters():
                parameter.fill_(float("nan"))
        restored = restore_sharded_expert_state(
            complete,
            local_state=model.state_dict(),
            shard_map=expert_shard_map([model]),
            model_parts=[model],
        )
        model.load_state_dict(restored)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected_local[key], atol=0, rtol=0)
        wrapper = ModelCheckpointState(model)
        local_wrapper = ModelCheckpointState(model, sharded_expert_parallel=True)
        dcp.save({"model": local_wrapper}, checkpoint_id=str(path / "wrapper-shards"))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float("nan"))
        dcp.load({"model": local_wrapper}, checkpoint_id=str(path / "wrapper-shards"))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected_local[key], atol=0, rtol=0)
        local_wrapper.load_state_dict(complete)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected_local[key], atol=0, rtol=0)
        adapter = HFTransformerStateDictAdapter(
            SimpleNamespace(model_type="qwen3_moe", num_experts=4),
            str(path),
            qwen3_expert_parallel=True,
        )
        local_export = adapter.to_hf_export(local_wrapper.state_dict())
        save_hf_checkpoint(
            local_export,
            str(path / "local-hf-export"),
            adapter.hf_export_fqn_to_index_mapping,
        )
        published = json.loads(
            (path / "local-hf-export" / "model.safetensors.index.json").read_text()
        )["weight_map"]
        assert set(published) == set(weights)
        for key, filename in published.items():
            with safe_open(
                str(path / "local-hf-export" / filename), framework="pt"
            ) as file:
                torch.testing.assert_close(
                    file.get_tensor(key), weights[key], atol=0, rtol=0
                )
        local_destination = adapter.to_hf(local_wrapper.state_dict())
        dcp.load(
            local_destination,
            storage_reader=adapter.get_hf_storage_reader(str(path / "local-hf-export")),
        )
        local_wrapper.load_state_dict(adapter.from_hf(local_destination))
        destination = adapter.to_hf(wrapper.state_dict())
        dcp.load(destination, storage_reader=adapter.get_hf_storage_reader(str(path)))
        wrapper.load_state_dict(adapter.from_hf(destination))
        exported = adapter.to_hf_export(wrapper.state_dict())
        assert set(exported) == set(weights)
        for key, expected_weight in weights.items():
            torch.testing.assert_close(exported[key], expected_weight, atol=0, rtol=0)
        if rank == 0:
            save_file(exported, str(path / "export.safetensors"))
        assert set(adapter.hf_export_fqn_to_index_mapping) == set(weights)
        save_hf_checkpoint(
            exported, str(path / "hf-export"), adapter.hf_export_fqn_to_index_mapping
        )
        reloaded = {
            key: torch.full_like(value, float("nan"))
            for key, value in destination.items()
        }
        dcp.load(
            reloaded,
            storage_reader=adapter.get_hf_storage_reader(str(path / "hf-export")),
        )
        wrapper.load_state_dict(adapter.from_hf(reloaded))
        torch.testing.assert_close(
            mlp.router.gate.weight, weights[prefix + "gate.weight"], atol=0, rtol=0
        )
        for projection, role in roles.items():
            expected = torch.stack(
                [
                    weights[f"{prefix}experts.{expert}.{projection}.weight"]
                    for expert in range(rank * 2, rank * 2 + 2)
                ]
            )
            torch.testing.assert_close(getattr(grouped, role), expected, atol=0, rtol=0)
        dist.barrier()
        if rank == 0:
            print("PASS: HF load restores EP2 ownership; standard HF export is exact")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
