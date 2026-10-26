"""Check fused EP references through actual EP2/FSDP assembly on four ranks.

PYTHONPATH=. torchrun --nproc_per_node=4 \
    tests/integration_tests/ep_fsdp_pattern_references.py

Append --ignore-modules to exclude the complete original expert subtree.
CPU/Gloo, custom FSDP. Verify copied values, exclusion ownership and relocated
hook identity. Append --train for three-step dense gradient/update equivalence
with the Trainer exclusion reducer (no aux loss, AC or compile).
Use eight ranks and --replicate for dp_replicate2/efsdp2/EP2.
"""

import argparse
import copy
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoConfig

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.parallelize import parallelize_hf_transformers
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.trainer import Trainer


def check_training(model, reference, dims):
    """Compare real EP dispatch and Trainer reduction with full HF experts."""
    grouped = model.layers[0].mlp.routed_experts.inner_experts
    ep_rank = dims.get_mesh("ep").get_local_rank()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dense_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    runtime = SimpleNamespace(parallel_dims=dims, model_parts=[model])
    world = dist.get_world_size()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)
        losses = []
        for rank in range(world):
            inputs = (torch.arange(8) + rank * 5 + step * 3) % 32
            expected = reference(inputs)
            losses.append(expected.square().sum() / world)
            if rank == dist.get_rank():
                output = model(inputs)
                torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
                (output.square().sum() / world).backward()
        sum(losses).backward()
        Trainer._allreduce_fsdp_excluded_grads(runtime)
        source = reference.layers[0].mlp.experts
        weights = source.gate_up_proj.detach().chunk(2, dim=0)[ep_rank].chunk(2, dim=1)
        gradients = source.gate_up_proj.grad.chunk(2, dim=0)[ep_rank].chunk(2, dim=1)
        expert_values = {
            "w1_EFD": (weights[0], gradients[0]),
            "w3_EFD": (weights[1], gradients[1]),
            "w2_EDF": (
                source.down_proj.detach().chunk(2, dim=0)[ep_rank],
                source.down_proj.grad.chunk(2, dim=0)[ep_rank],
            ),
        }
        reference_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            if ".routed_experts.inner_experts." in name:
                expected_gradient = expert_values[name.rsplit(".", 1)[1]][1]
            else:
                source_name = name.replace(".mlp.router.gate.", ".mlp.gate.")
                expected_gradient = reference_params[source_name].grad
            gradient = parameter.grad
            if isinstance(gradient, DTensor):
                gradient = gradient.full_tensor()
            torch.testing.assert_close(
                gradient,
                expected_gradient,
                atol=1e-6,
                rtol=1e-6,
                msg=lambda message, name=name: f"{name}\n{message}",
            )
        expected_norm = torch.linalg.vector_norm(
            torch.cat(
                [p.grad.flatten() for p in reference.parameters() if p.grad is not None]
            )
        )
        norm = clip_grad_norm_(
            model.parameters(),
            0.5,
            ep_mesh=dims.get_mesh("ep"),
            expert_parameters=list(grouped.parameters()),
        )
        torch.testing.assert_close(norm, expected_norm, atol=1e-6, rtol=1e-6)
        factor = (0.5 / (expected_norm + 1e-6)).clamp(max=1)
        for parameter in reference.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)
        optimizer.step()
        dense_optimizer.step()
        # Slice after the update; the views above still point at source weights.
        for name, parameter in model.named_parameters():
            if ".routed_experts.inner_experts." in name:
                expected_value = expert_values[name.rsplit(".", 1)[1]][0]
            else:
                expected_value = reference_params[
                    name.replace(".mlp.router.gate.", ".mlp.gate.")
                ]
            value = (
                parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            )
            torch.testing.assert_close(
                value,
                expected_value,
                atol=1e-6,
                rtol=1e-6,
                msg=lambda message, name=name: f"{name}\n{message}",
            )
        print(f"rank={dist.get_rank()} EP2/FSDP2 step={step}: dense reference passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ignore-modules", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--replicate", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        world = dist.get_world_size()
        assert world == (8 if args.replicate else 4)
        torch.manual_seed(310)
        model = HFTransformerModel(
            AutoConfig.for_model(
                "qwen3_moe",
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                moe_intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_experts=4,
                num_experts_per_tok=2,
                experts_implementation="eager",
            )
        )
        reference = copy.deepcopy(model) if args.train else None
        source = model.layers[0].mlp.experts
        original = source.gate_up_proj
        expected = original.detach().clone()
        expected_down = source.down_proj.detach().clone()
        path = next(name for name, module in model.named_modules() if module is source)
        cfg = ParallelConfig(
            expert_parallel_size=2,
            data_parallel_replicate_size=2 if args.replicate else 1,
            fsdp_config=FSDPConfig(
                fsdp_implementation="custom",
                fsdp_apply_modules=(
                    {"": {}}
                    if args.ignore_modules
                    else {path: {"ignored_params": {original}, "hook_module": source}}
                ),
                fsdp_ignored_modules=[path] if args.ignore_modules else [],
            ),
        )
        if args.train:
            # The independent objective below contains only squared logits.
            cfg.ep_config.router_aux_loss_coef = 0.0
        dims = ParallelTopology.from_config(cfg, world_size=world)
        dims.build_device_mesh()
        ep_rank = dims.get_mesh("ep").get_local_rank()
        if args.replicate:
            assert dims.get_mesh("dp_replicate").size() == 2
            assert dims.get_mesh("efsdp").size() == 2
            assert dims.get_mesh("dp").size() == 8
        parallelize_hf_transformers(
            model, cfg=cfg, mesh=dims.spmd_dense_mesh(), parallel_dims=dims
        )
        grouped = model.layers[0].mlp.routed_experts.inner_experts
        assert not isinstance(grouped.w1_EFD, DTensor)
        assert not isinstance(grouped.w3_EFD, DTensor)
        if args.ignore_modules:
            assert not isinstance(grouped.w2_EDF, DTensor)
            torch.testing.assert_close(
                grouped.w2_EDF, expected_down.chunk(2, dim=0)[ep_rank], atol=0, rtol=0
            )
            assert cfg.fsdp_config.fsdp_ignored_modules == [path]
        else:
            assert isinstance(grouped.w2_EDF, DTensor)
            assert grouped._get_fsdp_state().hook_module() is grouped
            assert cfg.fsdp_config.fsdp_apply_modules[path]["ignored_params"] == {
                original
            }
            assert cfg.fsdp_config.fsdp_apply_modules[path]["hook_module"] is source
        gate, up = expected.chunk(2, dim=0)[ep_rank].chunk(2, dim=1)
        torch.testing.assert_close(grouped.w1_EFD, gate, atol=0, rtol=0)
        torch.testing.assert_close(grouped.w3_EFD, up, atol=0, rtol=0)
        if args.train:
            check_training(model, reference, dims)
        print(
            f"rank={dist.get_rank()} EP references and FSDP ownership: passed "
            f"ignore_modules={args.ignore_modules}"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
