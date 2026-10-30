"""Diagnose converted baseline MoE with production sequence-parallel hooks.

This diagnostic records differences; exit zero means the probe completed,
not that cross-framework numerical acceptance passed. No baseline source is patched.
"""

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from tests.caps import require_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_env("torch_npu")
    importlib.import_module("torchtitan_npu")
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
    from torchtitan_npu.converters.kernels.gmm import (
        NpuGroupedExpertConverter,
        _run_experts_grouped_mm,
    )
    from torchtitan_npu.converters.kernels.moe_dispatch import NpuMoeDispatchConverter
    from torchtitan_npu.models.common.moe_parallelize import (
        apply_sequence_sharded_moe_ep_tp,
    )

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
            backend.allow_hf32 = False
        torch.manual_seed(37)
        config = MoE.Config(
            num_experts=4,
            experts=GroupedExperts.Config(dim=32, hidden_dim=64, num_experts=4),
            router=TokenChoiceTopKRouter.Config(
                num_experts=4,
                gate=Linear.Config(in_features=32, out_features=4, bias=False),
                top_k=2,
                score_func="softmax",
                route_norm=True,
                _debug_force_load_balance=True,
            ),
            load_balance_coeff=None,
        )
        root = nn.Module()
        block = nn.Module()
        block.moe_enabled = True
        block.moe = config.build()
        root.layers = nn.ModuleDict({"0": block})
        root.to(device="npu", dtype=torch.bfloat16)
        spec = SimpleNamespace(name="qwen3")
        NpuGroupedExpertConverter(spec).convert(root)
        NpuMoeDispatchConverter(spec).convert(root)
        with torch.no_grad():
            for parameter in root.parameters():
                parameter.normal_(0, 0.1)
        oracle = copy.deepcopy(block.moe)

        def feature_partial_reference(experts, inputs, counts, scores=None):
            # Independent same-token feature partitions preserve TP rounding.
            # Only the oracle is altered; the measured baseline stays intact.
            gate, up = experts.w13.chunk(2, dim=1)
            partials = [
                _run_experts_grouped_mm(
                    torch.cat((gate.chunk(2, 1)[i], up.chunk(2, 1)[i]), 1),
                    experts.w2.chunk(2, 2)[i].contiguous(),
                    None,
                    inputs,
                    counts,
                    routed_scores=scores,
                    activation_fn=experts._expert_activation_fn,
                )
                for i in range(2)
            ]
            return partials[0] + partials[1]

        oracle.experts.forward = MethodType(feature_partial_reference, oracle.experts)
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("tp",))
        apply_sequence_sharded_moe_ep_tp(root, mesh, None, None, None)
        expert_shapes = []
        block.moe.experts.register_forward_pre_hook(
            lambda module, inputs: expert_shapes.append(list(inputs[0].shape))
        )
        shared = torch.randn(1, 8, 32, device="npu", dtype=torch.bfloat16)
        report = []
        for distinct in (False, True):
            local = shared + rank if distinct else shared
            inputs = DTensor.from_local(local, mesh, [Shard(1)], run_check=False)
            with torch.no_grad():
                actual = block.moe(inputs)
                if isinstance(actual, DTensor):
                    actual = actual.to_local()
                expected = oracle(local)
            delta = (actual.float() - expected.float()).abs()
            if not distinct:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            report.append(
                dict(
                    distinct_tokens=distinct,
                    exact=torch.equal(actual, expected),
                    max_ae=delta.max().item(),
                    mean_ae=delta.mean().item(),
                    expert_input_shape=expert_shapes[-1],
                )
            )
        output = args.output.with_name(args.output.stem + f"-rank{rank}.json")
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"rank={rank}: {report}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
