"""Production TP2/EP2/FSDP2 assembly versus the direct ordered stage calls.

Run on four ranks. Compares three updates with asymmetric inputs and two
accumulation groups. Exact equality checks assembly-only changes; this is not
an independent mathematical oracle or a performance benchmark.
"""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import DefaultDeviceType
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.accelerator.spmd import spmd_context
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.memory.activation_checkpoint import apply_ac
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel import token_dispatch
from llmtuner.parallel.expert_parallel.expert_parallel import apply_ep
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.parallelize import parallelize_model
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp
from llmtuner.parallel.topology import ParallelTopology, build_device_mesh
from llmtuner.trainer.runtime.gradients import allreduce_replicated_tp_grads
from tests.integration_tests.harness import init_distributed


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Compare the actual optimizer shard without introducing a gather."""
    return value.to_local() if isinstance(value, DTensor) else value


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--ac", choices=("none", "full"), default="none")
    args = parser.parse_args()
    device = init_distributed(timeout=timedelta(seconds=120))
    previous_checkpoint_device = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type(device.type)
    try:
        if dist.get_world_size() != 4:
            raise ValueError("This check requires four ranks for TP2/EP2/FSDP2")
        cfg = ParallelConfig(
            tensor_parallel_size=2,
            expert_parallel_size=2,
            data_parallel_shard_size=2,
        )
        topology = ParallelTopology.from_config(cfg, dist.get_world_size())
        mesh = build_device_mesh(topology)
        ep_group = topology.get_mesh("ep").get_group()
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.0,
            attn_implementation="eager",
            experts_implementation="eager",
        )
        torch.manual_seed(173)
        direct = Qwen3MoeForCausalLM(config).to(device)
        planned = deepcopy(direct)
        direct = apply_tp(direct, mesh, cfg)
        direct = apply_ep(direct, cfg, ep_group=ep_group)
        direct = apply_ac(direct, args.ac)
        direct = apply_fsdp(direct, cfg=cfg, parallel_dims=topology)
        planned = parallelize_model(
            planned,
            cfg=cfg,
            mesh=mesh,
            parallel_dims=topology,
            activation_checkpoint=args.ac,
        )
        for model in (direct, planned):
            assert any(isinstance(p, DTensor) for p in model.parameters())
            for layer in model.layers:
                experts = layer.mlp.routed_experts.inner_experts
                assert experts.w1_EFD.shape[0] == 2
                assert layer.self_attn.q_proj.weight.shape[0] == 16
        optimizers = [
            torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, foreach=False)
            for m in (direct, planned)
        ]
        stream = topology.get_mesh("batch").get_local_rank()
        with patch.object(
            token_dispatch,
            "_grad_carrying_all_to_all",
            wraps=token_dispatch._grad_carrying_all_to_all,
        ) as exchange:
            for step in range(3):
                outputs = []
                for model, optimizer in zip((direct, planned), optimizers, strict=True):
                    optimizer.zero_grad(set_to_none=True)
                    group_outputs = []
                    for group in range(2):
                        tokens = (
                            torch.arange(8, device=device) + 3 * stream + group + step
                        ) % 32
                        local, _, metadata = model.prepare_inputs(
                            {
                                "input": tokens,
                                "labels": tokens,
                                "positions": torch.arange(8, device=device),
                            },
                            parallel_dims=topology,
                        )
                        with spmd_context(topology):
                            output = model(local, **metadata).logits
                            (output.float().square().sum() / 512).backward()
                        group_outputs.append(output.detach().clone())
                    allreduce_replicated_tp_grads([model], parallel_dims=topology)
                    outputs.append(group_outputs)
                with synchronize_local_errors("Assembly gradient comparison"):
                    for left, right in zip(*outputs, strict=True):
                        assert left.numel() and left.count_nonzero()
                        torch.testing.assert_close(left, right, atol=0, rtol=0)
                    nonzero = 0
                    for (name, left), (other, right) in zip(
                        direct.named_parameters(),
                        planned.named_parameters(),
                        strict=True,
                    ):
                        assert name == other
                        assert left.grad is not None and right.grad is not None, name
                        torch.testing.assert_close(
                            local_tensor(left.grad),
                            local_tensor(right.grad),
                            atol=0,
                            rtol=0,
                            msg=name,
                        )
                        nonzero += int(bool(local_tensor(left.grad).count_nonzero()))
                    assert nonzero > 0
                for optimizer in optimizers:
                    optimizer.step()
                with synchronize_local_errors("Assembly update comparison"):
                    for (name, left), (_, right) in zip(
                        direct.named_parameters(),
                        planned.named_parameters(),
                        strict=True,
                    ):
                        torch.testing.assert_close(
                            local_tensor(left),
                            local_tensor(right),
                            atol=0,
                            rtol=0,
                            msg=name,
                        )
                if dist.get_rank() == 0:
                    print(f"PASS: TP2/EP2/FSDP2 AC={args.ac} exact update {step + 1}")
            assert exchange.call_count > 0
    finally:
        try:
            dist.destroy_process_group()
        finally:
            DefaultDeviceType.set_device_type(previous_checkpoint_device)


if __name__ == "__main__":
    main()
