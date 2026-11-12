"""Two NPU ranks: partial FSDP meta materialization with CPU offload."""

import torch
import torch.distributed as dist
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.memory.materialization import materialize_meta_model
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.runtime.assembly import _fsdp_managed_parameters
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed()
    try:
        assert dist.get_world_size() == 2
        config = Qwen3MoeConfig(
            head_dim=8,
            num_experts=4,
            moe_intermediate_size=16,
            mlp_only_layers=[0, 1],
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8,
        )
        with torch.device("meta"):
            model = Qwen3MoeForCausalLM(config)
        layer_paths = [
            f"{model.decoder_structure.blocks}.{index}"
            for index in range(config.num_hidden_layers)
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
            runtime_buffers=model.materialization_buffers(device),
        )
        assert _fsdp_managed_parameters(model) == managed
        assert all(parameter.device.type == "cpu" for parameter in managed)
        assert all(parameter.device == device for parameter in unmanaged)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.01)
        transfers = {"h2d_bytes": 0, "d2h_bytes": 0}
        original_to = torch.Tensor.to

        def measured_to(self, *args, **kwargs):
            result = original_to(self, *args, **kwargs)
            if result.device != self.device:
                key = "h2d_bytes" if self.device.type == "cpu" else "d2h_bytes"
                transfers[key] += self.numel() * self.element_size()
            return result

        torch.Tensor.to = measured_to
        try:
            tokens = torch.arange(8, device=device) % 32
            output = model(tokens).logits
            assert output.device == device
            assert torch.isfinite(output).all().item()
            output.float().mean().backward()
        finally:
            torch.Tensor.to = original_to
        assert transfers["h2d_bytes"] > 0
        assert transfers["d2h_bytes"] > 0
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                "PASS: partial FSDP meta CPU-offload forward/backward on NPU; "
                f"transfers={transfers}"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
