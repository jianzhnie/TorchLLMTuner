"""Eight CPU/Gloo ranks: dense DP4/TP2 and sparse FSDP2/EP4 DCP restore."""

import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from llmtuner.parallel.expert_parallel.checkpointing import mark_experts_ep_sharded
from llmtuner.parallel.expert_parallel.expert_parallel import apply_ep
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        assert dist.get_world_size() == 8
        dense = init_device_mesh("cpu", (4, 2), mesh_dim_names=("dp", "tp"))
        sparse = init_device_mesh("cpu", (2, 4), mesh_dim_names=("efsdp", "ep"))
        dense_full = torch.arange(128.0).reshape(16, 8)
        expert_full = torch.arange(512.0).reshape(8, 8, 8)
        model = nn.Module()
        model.dense = nn.Parameter(
            distribute_tensor(
                dense_full.chunk(2, 1)[dense["tp"].get_local_rank()].contiguous(),
                dense["dp"],
                [Shard(0)],
            )
        )
        model._tp_checkpoint_layouts = {"dense": (1, False, dense["tp"].get_group())}
        model.experts = nn.Module()
        model.experts.weight = nn.Parameter(
            distribute_tensor(
                expert_full.chunk(4, 0)[sparse["ep"].get_local_rank()].contiguous(),
                sparse["efsdp"],
                [Shard(1)],
            )
        )
        mark_experts_ep_sharded(model.experts, sparse["ep"].get_group())
        wrapper = ModelCheckpointState(
            model, sharded_tensor_parallel=True, sharded_expert_parallel=True
        )
        state = wrapper.state_dict()
        assert state["dense"].chunks[0][0] == (
            dense["dp"].get_local_rank() * 4,
            dense["tp"].get_local_rank() * 4,
        )
        assert state["experts.weight"].chunks[0][0] == (
            sparse["ep"].get_local_rank() * 2,
            sparse["efsdp"].get_local_rank() * 4,
            0,
        )
        expected = {
            key: value.to_local().clone() for key, value in model.state_dict().items()
        }
        paths = [
            tempfile.mkdtemp(prefix="tp2-ep4-dp-") if dist.get_rank() == 0 else None
        ]
        dist.broadcast_object_list(paths)
        dcp.save({"model": wrapper}, checkpoint_id=paths[0])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float("nan"))
        dcp.load({"model": wrapper}, checkpoint_id=paths[0])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.to_local(), expected[key], atol=0, rtol=0)
        complete = {
            "model": {
                "dense": torch.empty_like(dense_full),
                "experts.weight": torch.empty_like(expert_full),
            }
        }
        dcp.load(complete, checkpoint_id=paths[0])
        torch.testing.assert_close(
            complete["model"]["dense"], dense_full, atol=0, rtol=0
        )
        torch.testing.assert_close(
            complete["model"]["experts.weight"], expert_full, atol=0, rtol=0
        )
        from transformers import AutoConfig

        config = AutoConfig.for_model(
            "qwen3_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_tok=2,
            max_position_embeddings=256,
            experts_implementation="eager",
        )
        torch.manual_seed(42)
        actual = HFTransformerModel(config).float().eval()
        hf_weights = {}
        for key, value in actual.state_dict().items():
            key = key.removeprefix("model.")
            if key.endswith(".experts.gate_up_proj"):
                prefix = key.removesuffix("gate_up_proj")
                for expert, stacked in enumerate(value):
                    gate, up = stacked.chunk(2, 0)
                    hf_weights[f"{prefix}{expert}.gate_proj.weight"] = gate.contiguous()
                    hf_weights[f"{prefix}{expert}.up_proj.weight"] = up.contiguous()
            elif key.endswith(".experts.down_proj"):
                prefix = key.removesuffix("down_proj")
                for expert, weight in enumerate(value):
                    hf_weights[f"{prefix}{expert}.down_proj.weight"] = (
                        weight.contiguous()
                    )
            else:
                hf_weights[key] = value.clone()
        hf_path = Path(paths[0]) / "hf"
        if dist.get_rank() == 0:
            hf_path.mkdir()
            save_file(hf_weights, str(hf_path / "model.safetensors"))
        dist.barrier()
        parallel = ParallelConfig(
            data_parallel_shard_size=4,
            tensor_parallel_size=2,
            expert_parallel_size=4,
        )
        actual = apply_tp(actual, dense["tp"], parallel)
        actual = apply_ep(actual, parallel, ep_group=sparse["ep"].get_group())
        actual_wrapper = ModelCheckpointState(
            actual, sharded_tensor_parallel=True, sharded_expert_parallel=True
        )
        actual_expected = {
            key: value.clone() for key, value in actual.state_dict().items()
        }
        dcp.save({"model": actual_wrapper}, checkpoint_id=paths[0] + "-actual")
        with torch.no_grad():
            for parameter in actual.parameters():
                parameter.fill_(float("nan"))
        dcp.load({"model": actual_wrapper}, checkpoint_id=paths[0] + "-actual")
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, actual_expected[key], atol=0, rtol=0)
        adapter = HFTransformerStateDictAdapter(
            config, str(hf_path), qwen3_expert_parallel=True
        )
        with torch.no_grad():
            for parameter in actual.parameters():
                parameter.fill_(float("nan"))
        destination = adapter.to_hf(actual_wrapper.state_dict())
        dcp.load(
            destination, storage_reader=adapter.get_hf_storage_reader(str(hf_path))
        )
        actual_wrapper.load_state_dict(adapter.from_hf(destination))
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, actual_expected[key], atol=0, rtol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
