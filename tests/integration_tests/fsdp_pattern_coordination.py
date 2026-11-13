"""Rank-local pattern failures stop peers before FSDP mutation.

Run with torchrun --standalone --nproc_per_node=2. Faults are real invalid
configuration or Parameter references, not mocked sharding implementations.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.fully_shard_parallel.pattern import apply_fsdp_by_pattern
from llmtuner.parallel.topology import ParallelTopology


class SeparateBlock(nn.Module):
    fsdp_separate_parameter_names = ("scale",)

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4))
        self.projection = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, inputs):
        return self.projection(inputs) * self.scale.to(inputs.dtype)


class Decoder(nn.Module):
    enable_weight_tying = False

    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Identity()
        self.norm = nn.Identity()
        self.lm_head = nn.Identity()
        self.layers = nn.ModuleList([SeparateBlock()])

    def forward(self, inputs):
        return self.layers[0](inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device, int(os.environ.get("LOCAL_RANK", 0)))
    if args.device == "npu":
        torch.npu.set_device(device)
    dist.init_process_group(
        "hccl" if args.device == "npu" else "gloo", timeout=timedelta(seconds=60)
    )
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        mesh = init_device_mesh(args.device, (2,))
        for fault in ("missing", "foreign", "invalid_policy", "shared", "reshard"):
            torch.manual_seed(73)
            model = nn.Sequential(nn.Linear(4, 4)).to(device)
            if fault == "shared":
                model.append(nn.Linear(4, 4).to(device))
                model[1].weight = model[0].weight
            parameters = list(model.parameters())
            values = [parameter.detach().clone() for parameter in parameters]
            options = {"0": {}}
            if rank == 0:
                if fault == "missing":
                    options = {"missing": {}}
                elif fault == "foreign":
                    options["0"]["ignored_params"] = {nn.Parameter(torch.ones(4))}
                elif fault == "invalid_policy":
                    options["0"]["offload_policy"] = object()
                elif fault == "shared":
                    options = {"0": {"reshard_after_forward": False}, "1": {}}
                else:
                    options["0"]["reshard_after_forward"] = 3
            cfg = ParallelConfig(fsdp_config=FSDPConfig(fsdp_apply_modules=options))
            caught = None
            try:
                apply_fsdp_by_pattern(
                    model,
                    dp_mesh=mesh,
                    cfg=cfg,
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                    error_group=dist.group.WORLD,
                )
            except (ValueError, TypeError, RuntimeError) as error:
                caught = error
            assert caught is not None
            if rank == 0:
                assert isinstance(
                    caught, TypeError if fault == "invalid_policy" else ValueError
                )
            else:
                assert type(caught) is RuntimeError
                assert "FSDP pattern validation failed on another rank" in str(caught)
            assert not any(isinstance(module, FSDPModule) for module in model.modules())
            for original, actual, value in zip(
                parameters, model.parameters(), values, strict=True
            ):
                assert actual is original
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            # All peers can retry with a valid plan, and execute real FSDP
            # forward/backward rather than merely reaching another barrier.
            cfg = ParallelConfig(fsdp_config=FSDPConfig(fsdp_apply_modules={"0": {}}))
            apply_fsdp_by_pattern(
                model,
                dp_mesh=mesh,
                cfg=cfg,
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                error_group=dist.group.WORLD,
            )
            assert isinstance(model[0].weight, DTensor)
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.set_force_sum_reduction_for_comms(True)
            model(torch.ones(2, 4, device=device) + rank).square().sum().backward()
            gradient = model[0].weight.grad.to_local()
            assert torch.isfinite(gradient).all() and gradient.count_nonzero() > 0
            dist.barrier()
            print(f"rank={rank} {fault}: coordinated rejection and retry passed")
        cfg = ParallelConfig(
            data_parallel_shard_size=2,
            fsdp_config=FSDPConfig(fsdp_apply_modules={"0": {}}),
        )
        topology = ParallelTopology.from_config(cfg, world_size=2)
        topology.build_device_mesh()
        model = nn.Sequential(nn.Linear(4, 4)).to(device)
        if rank == 0:
            model[0].bias = nn.Parameter(model[0].bias.bfloat16())
        original = list(model.parameters())
        caught = None
        try:
            apply_fsdp(model, cfg, topology)
        except (ValueError, RuntimeError) as error:
            caught = error
        assert caught is not None
        if rank == 0:
            assert type(caught) is ValueError
            assert "single parameter dtype" in str(caught)
        else:
            assert "FSDP parameter dtype validation failed on another rank" in str(
                caught
            )
        assert all(a is b for a, b in zip(original, model.parameters(), strict=True))
        assert not any(isinstance(module, FSDPModule) for module in model.modules())
        model.float()
        apply_fsdp(model, cfg, topology)
        model(torch.ones(2, 4, device=device)).square().sum().backward()
        assert model[0].weight.grad.to_local().count_nonzero() > 0
        print(f"rank={rank} dtype: coordinated rejection and retry passed")
        decoder_cfg = ParallelConfig(data_parallel_shard_size=2)
        for fault in ("external_alias", "nested"):
            model = Decoder().to(device)
            if rank == 0:
                if fault == "external_alias":
                    model.register_parameter("external", model.layers[0].scale)
                else:
                    model.layers[0].child = SeparateBlock().to(device)
            parameters = list(model.parameters())
            caught = None
            try:
                apply_fsdp(model, decoder_cfg, topology)
            except (ValueError, RuntimeError) as error:
                caught = error
            assert caught is not None
            if rank == 0:
                assert type(caught) is ValueError
            else:
                assert "Decoder FSDP plan validation failed on another rank" in str(
                    caught
                )
            assert not any(isinstance(module, FSDPModule) for module in model.modules())
            assert all(
                a is b for a, b in zip(parameters, model.parameters(), strict=True)
            )
            model = Decoder().to(device)
            apply_fsdp(model, decoder_cfg, topology)
            result = model(torch.ones(2, 4, device=device, dtype=torch.bfloat16))
            result.float().square().sum().backward()
            gradient = model.layers[0].scale.grad.to_local()
            assert torch.isfinite(gradient).all() and gradient.count_nonzero() > 0
            dist.barrier()
            print(f"rank={rank} decoder {fault}: rejection and retry passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
