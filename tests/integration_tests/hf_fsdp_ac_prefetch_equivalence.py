"""CPU/Gloo custom FSDP, full AC and prefetch versus unmodified HF Qwen3.

Run with PYTHONPATH=. torchrun --nproc_per_node=2 from the repository root.
Three layers, two distinct rank batches and three momentum-SGD updates.
This checks eager FP32 numerics, not device overlap, compile or checkpoint I/O.
"""

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Qwen3Config

from llmtuner.accelerator.device import device_type
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.memory.activation_checkpoint import wrap_full
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.utils.module_names import canonical_fqn


def main() -> None:
    if device_type != "cpu":
        raise RuntimeError("This oracle requires a CPU-only environment")
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        torch.manual_seed(42)
        model = HFTransformerModel(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=2,
                num_key_value_heads=2,
                attention_dropout=0.0,
            )
        )
        reference = copy.deepcopy(model.model)
        for i, layer in enumerate(model.layers):
            model.layers[i] = wrap_full(layer)
        cfg = ParallelConfig(
            data_parallel_shard_size=2,
            fsdp_config=FSDPConfig(
                fsdp_implementation="custom",
                fsdp_forward_prefetch_count=2,
                fsdp_backward_prefetch_count=2,
            ),
        )
        topology = ParallelTopology.from_config(cfg, 2)
        topology.build_device_mesh()
        model = apply_fsdp(model, cfg, topology)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.05, momentum=0.9
        )
        ref_params = dict(reference.named_parameters())
        assert {
            canonical_fqn(name).removeprefix("model.")
            for name, _ in model.named_parameters()
        } == ref_params.keys()
        for name, parameter in model.named_parameters():
            key = canonical_fqn(name).removeprefix("model.")
            torch.testing.assert_close(
                parameter.full_tensor(), ref_params[key], rtol=0, atol=0
            )
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            inputs = (torch.arange(16).reshape(2, 8) + step * 3) % 32
            targets = (inputs + 1) % 32
            expected = [
                reference(inputs[r : r + 1]).logits.squeeze(0) for r in range(2)
            ]
            actual = model(inputs[rank])
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
            for name, p in model.named_parameters():
                key = canonical_fqn(name).removeprefix("model.")
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
                key = canonical_fqn(name).removeprefix("model.")
                torch.testing.assert_close(
                    p.full_tensor(),
                    ref_params[key],
                    atol=1e-6,
                    rtol=1e-5,
                    msg=lambda msg, key=key: f"{key}: {msg}",
                )
            print(
                f"rank={rank} step={step + 1}: "
                "HF dense output/all gradients/momentum update passed",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
