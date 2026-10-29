"""Two NPU ranks: partial FSDP meta materialization with CPU offload."""

import torch
import torch.distributed as dist
from transformers import AutoConfig

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.hf.materialization import materialize_meta_model
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.runtime.assembly import _fsdp_managed_parameters
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed()
    try:
        assert dist.get_world_size() == 2
        config = AutoConfig.for_model(
            "qwen3",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8,
        )
        with torch.device("meta"):
            model = HFTransformerModel(config)
        layer_paths = [
            name
            for name, module in model.named_modules()
            if type(module).__name__ == "Qwen3DecoderLayer"
        ]
        parallel = ParallelConfig(data_parallel_shard_size=2)
        parallel.fsdp_config.fsdp_cpu_offload = True
        parallel.fsdp_config.fsdp_cpu_offload_pin_memory = False
        parallel.fsdp_config.fsdp_apply_modules = {layer_paths[0]: {}}
        parallel.fsdp_config.fsdp_ignored_modules = [layer_paths[1]]
        topology = ParallelTopology(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=2,
        )
        topology.build_device_mesh()
        apply_fsdp(model, parallel, topology)
        managed = _fsdp_managed_parameters(model)
        assert managed
        unmanaged = set(model.parameters()) - managed
        assert unmanaged
        materialize_meta_model(
            model,
            device,
            parameter_device=torch.device("cpu"),
            offloaded_parameters=managed,
        )
        assert _fsdp_managed_parameters(model) == managed
        assert all(parameter.device.type == "cpu" for parameter in managed)
        assert all(parameter.device == device for parameter in unmanaged)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.01)
        tokens = torch.arange(8, device=device) % 32
        output = model(tokens)
        assert output.device == device
        assert torch.isfinite(output).all().item()
        output.float().mean().backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        dist.barrier()
        if dist.get_rank() == 0:
            print("PASS: partial FSDP meta CPU-offload forward/backward on NPU")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
