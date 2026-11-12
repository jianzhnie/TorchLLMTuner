"""CPU/Gloo custom FSDP, full AC and prefetch versus unmodified HF Qwen3.

Run with PYTHONPATH=. torchrun --nproc_per_node=2 from the repository root.
Three layers, two distinct rank batches and three momentum-SGD updates.
The explicit Qwen3 candidate declares every layer dense; HF Qwen3 remains the
independent reference. Pass --implementation native for a separate native FSDP
check; it does not establish custom FSDP support.
This checks eager FP32 numerics, not device overlap, compile or checkpoint I/O.
"""

import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import DefaultDeviceType
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig

from llmtuner.accelerator.device import device_type
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.memory.activation_checkpoint import wrap_full
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.utils.module_names import canonical_fqn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=("custom", "native"), default="custom"
    )
    args = parser.parse_args()
    if device_type != "cpu":
        raise RuntimeError("This oracle requires a CPU-only environment")
    # torch-npu selects NPU as checkpoint's default even with devices hidden.
    DefaultDeviceType.set_device_type("cpu")
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        torch.manual_seed(42)
        model = Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=128,
                attention_dropout=0.0,
                mlp_only_layers=[0, 1, 2],
                num_experts=4,
                num_experts_per_tok=2,
            )
        )
        reference = Qwen3ForCausalLM(Qwen3Config(**model.config.to_dict()))
        reference.load_state_dict(model.state_dict(), strict=True)
        assert all(layer.moe_block_name is None for layer in model.layers)
        executions = [0] * len(model.layers)
        for i, layer in enumerate(model.layers):

            def count_forward(module, inputs, index=i):
                executions[index] += 1

            layer.register_forward_pre_hook(count_forward)
            model.layers[i] = wrap_full(layer)
        cfg = ParallelConfig(
            data_parallel_shard_size=2,
            fsdp_config=FSDPConfig(
                fsdp_implementation=args.implementation,
                fsdp_forward_prefetch_count=2,
                fsdp_backward_prefetch_count=2,
            ),
        )
        topology = ParallelTopology.from_config(cfg, 2)
        topology.build_device_mesh()
        model = apply_fsdp(model, cfg, topology)
        assert any(
            isinstance(parameter, DTensor)
            and parameter.to_local().numel() < parameter.numel()
            for parameter in model.parameters()
        ), "FSDP must actually shard parameters"
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.05, momentum=0.9, foreach=False, fused=False
        )
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.05, momentum=0.9, foreach=False, fused=False
        )
        ref_params = dict(reference.named_parameters())
        assert {
            canonical_fqn(name) for name, _ in model.named_parameters()
        } == ref_params.keys()
        for name, parameter in model.named_parameters():
            key = canonical_fqn(name)
            torch.testing.assert_close(
                parameter.full_tensor(), ref_params[key], rtol=0, atol=0
            )
        for step in range(3):
            before_executions = executions.copy()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            inputs = (torch.arange(16).reshape(2, 8) + step * 3) % 32
            targets = (inputs + 1) % 32
            expected = [
                reference(inputs[r : r + 1]).logits.squeeze(0) for r in range(2)
            ]
            actual = model(inputs[rank]).logits
            torch.testing.assert_close(actual, expected[rank], atol=1e-6, rtol=1e-5)
            loss = F.cross_entropy(actual, targets[rank], reduction="sum") / 16
            ref_loss = (
                sum(
                    F.cross_entropy(expected[r], targets[r], reduction="sum")
                    for r in range(2)
                )
                / 16
            )
            loss.backward()
            ref_loss.backward()
            assert all(
                after - before >= 2
                for before, after in zip(before_executions, executions, strict=True)
            ), f"AC must recompute each decoder: {before_executions} -> {executions}"
            for name, p in model.named_parameters():
                key = canonical_fqn(name)
                torch.testing.assert_close(
                    p.grad.full_tensor(),
                    ref_params[key].grad,
                    atol=1e-6,
                    rtol=1e-5,
                    msg=lambda msg, key=key: f"{key}: {msg}",
                )
            optimizer.step()
            reference_optimizer.step()
            for name, p in model.named_parameters():
                key = canonical_fqn(name)
                torch.testing.assert_close(
                    p.full_tensor(),
                    ref_params[key],
                    atol=1e-6,
                    rtol=1e-5,
                    msg=lambda msg, key=key: f"{key}: {msg}",
                )
                torch.testing.assert_close(
                    optimizer.state[p]["momentum_buffer"].full_tensor(),
                    reference_optimizer.state[ref_params[key]]["momentum_buffer"],
                    atol=1e-6,
                    rtol=1e-5,
                    msg=lambda msg, key=key: f"momentum {key}: {msg}",
                )
            print(
                f"rank={rank} step={step + 1}: "
                f"{args.implementation} explicit dense output/all gradients/"
                "momentum update passed",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
