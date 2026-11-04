"""Diagnose converted MoE with the plugin's sequence-sharded parallel plan.

This diagnostic records differences; exit zero means the probe completed,
not that cross-framework numerical acceptance passed. No baseline source is patched.
Use --production-plan for the upstream llama4 MoE plan called by Qwen3.
The default checks the plugin's separate sequence-sharded plan. Neither probe
reproduces the complete production model assembly.
"""

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)

from tests.caps import require_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicated-input-control", action="store_true")
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument("--production-plan", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--detach-router-scores", action="store_true")
    parser.add_argument("--local-router-control", action="store_true")
    parser.add_argument("--parameter-gradients", action="store_true")
    parser.add_argument("--sequence-ep-control", action="store_true")
    parser.add_argument("--reload-canonical-state", action="store_true")
    parser.add_argument("--canonical-packed-weight-shards", action="store_true")
    args = parser.parse_args()
    if args.production_plan and args.replicated_input_control:
        parser.error("production plan and replicated-input control are separate probes")
    if args.expert_parallel and args.replicated_input_control:
        parser.error("replicated-input control currently covers TP-only experts")
    if args.parameter_gradients and (not args.backward or args.detach_router_scores):
        parser.error("parameter gradients require --backward with router scores")
    if (
        args.parameter_gradients
        and args.expert_parallel
        and not args.sequence_ep_control
    ):
        parser.error("EP parameter gradient reference requires --sequence-ep-control")
    if args.sequence_ep_control and not (
        args.expert_parallel and args.production_plan and args.local_router_control
    ):
        parser.error("sequence EP control requires production EP and local router")
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

    plan_module = importlib.import_module(
        "torchtitan.models.llama4.parallelize"
        if args.production_plan
        else "torchtitan_npu.models.common.moe_parallelize"
    )
    apply_plan = (
        plan_module.apply_moe_ep_tp
        if args.production_plan
        else apply_sequence_sharded_moe_ep_tp
    )
    source_paths = {
        "probe": Path(__file__),
        "parallel_plan": Path(plan_module.__file__),
        "gmm": Path(
            importlib.import_module("torchtitan_npu.converters.kernels.gmm").__file__
        ),
        "dispatch": Path(
            importlib.import_module(
                "torchtitan_npu.converters.kernels.moe_dispatch"
            ).__file__
        ),
        "router_parallel": Path(
            importlib.import_module("torchtitan.distributed.tensor_parallel").__file__
        ),
    }
    sources = {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in source_paths.items()
    }
    if args.local_router_control:
        if not args.production_plan:
            parser.error("local router control requires --production-plan")

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == (4 if args.expert_parallel else 2)
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
        if args.detach_router_scores:
            # Diagnostic only: isolate the expert-to-input Jacobian from the
            # router-score path in both implementations without changing values.
            for router in (block.moe.router, oracle.router):
                original_forward = router.forward

                def detached_scores(*inputs, _forward=original_forward, **kwargs):
                    scores, indices, counts = _forward(*inputs, **kwargs)
                    return scores.detach(), indices, counts

                router.forward = detached_scores

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

        if not args.expert_parallel:
            oracle.experts.forward = MethodType(
                feature_partial_reference, oracle.experts
            )
        full_mesh = init_device_mesh(
            "npu",
            (2, 2) if args.expert_parallel else (2,),
            mesh_dim_names=("dp", "tp") if args.expert_parallel else ("tp",),
        )
        mesh = full_mesh["tp"]
        ep_mesh = full_mesh["dp", "tp"]._flatten("ep") if args.expert_parallel else None
        if args.replicated_input_control:
            # Diagnostic only: feed every feature shard the same full sequence,
            # then select the original local sequence at the output boundary.
            # Do not patch the installed baseline or call this production parity.
            from torch.distributed.tensor.parallel import (
                PrepareModuleInputOutput,
                parallelize_module,
            )
            from torchtitan.distributed.expert_parallel import TensorParallel

            parallelize_module(
                block,
                mesh,
                {
                    "moe": PrepareModuleInputOutput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                        use_local_input=True,
                        output_layouts=(Replicate(),),
                        desired_output_layouts=(Shard(1),),
                    )
                },
            )
            parallelize_module(block.moe.experts, mesh, TensorParallel())
        else:
            if args.expert_parallel:
                from torchtitan_npu.converters.kernels.moe_dispatch import (
                    NpuMoeDispatchParallelizePlanUpdater,
                )

                real_parallelize = plan_module.parallelize_module

                def converted_parallelize(*args, **kwargs):
                    plan = kwargs["parallelize_plan"]
                    if isinstance(plan, dict) and "moe" in plan:
                        if args_namespace.sequence_ep_control:
                            from torch.distributed.tensor.parallel import (
                                PrepareModuleInputOutput,
                            )

                            plan = {
                                **plan,
                                "moe": PrepareModuleInputOutput(
                                    input_layouts=(Shard(1),),
                                    desired_input_layouts=(Shard(1),),
                                    use_local_input=True,
                                    output_layouts=(Shard(1),),
                                    desired_output_layouts=(Shard(1),),
                                ),
                            }
                    kwargs["parallelize_plan"] = (
                        NpuMoeDispatchParallelizePlanUpdater.update(plan)
                    )
                    return real_parallelize(*args, **kwargs)

                # Reproduce the converter container's production plan update.
                args_namespace = args
                plan_module.parallelize_module = converted_parallelize
                try:
                    apply_plan(root, mesh, ep_mesh, None, None)
                finally:
                    plan_module.parallelize_module = real_parallelize
            else:
                apply_plan(root, mesh, None, None, None)
        if args.local_router_control:
            gate = block.moe.router.gate
            # Isolate the DTensor gate boundary: local computation preserves
            # partial weight gradients and leaves input reduction to MoE.
            # This is a diagnostic replacement, not production acceptance.
            gate._forward_pre_hooks.clear()
            gate._forward_hooks.clear()

            def local_gate(module, inputs):
                weight = module.weight.to_local(grad_placements=(Partial(),))
                return torch.nn.functional.linear(inputs, weight, None)

            gate.forward = MethodType(local_gate, gate)
        if args.reload_canonical_state:
            # Exercise the ordinary global coordinates used by HF/DCP load,
            # rather than assuming initialization's local packing is retained.
            with torch.no_grad():
                for name, parameter in block.moe.named_parameters():
                    reference = dict(oracle.named_parameters())[name]
                    incoming = distribute_tensor(
                        reference.detach(),
                        parameter.device_mesh,
                        parameter.placements,
                    )
                    parameter.copy_(incoming)
                    torch.testing.assert_close(
                        parameter.full_tensor(), reference, rtol=0, atol=0
                    )
        expert_shapes = []
        block.moe.experts.register_forward_pre_hook(
            lambda module, inputs: expert_shapes.append(list(inputs[0].shape))
        )
        shared = torch.randn(1, 8, 32, device="npu", dtype=torch.bfloat16)
        report = []
        for distinct in (False, True):
            local = shared + rank if distinct else shared
            local = local.detach().requires_grad_(args.backward)
            inputs = DTensor.from_local(local, mesh, [Shard(1)], run_check=False)
            with torch.set_grad_enabled(args.backward):
                actual = block.moe(inputs)
                if isinstance(actual, DTensor):
                    actual = actual.to_local()
                expected = oracle(local)
            delta = (actual.float() - expected.float()).abs()
            if args.replicated_input_control or (
                not distinct and not args.production_plan
            ):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            report.append(
                dict(
                    distinct_tokens=distinct,
                    plan=apply_plan.__module__ + "." + apply_plan.__name__,
                    replicated_input_control=args.replicated_input_control,
                    detached_router_scores=args.detach_router_scores,
                    local_router_control=args.local_router_control,
                    sequence_ep_control=args.sequence_ep_control,
                    reloaded_canonical_state=args.reload_canonical_state,
                    canonical_packed_weight_shards=args.canonical_packed_weight_shards,
                    sources=sources,
                    exact=torch.equal(actual, expected),
                    max_ae=delta.max().item(),
                    mean_ae=delta.mean().item(),
                    tp_degree_rescaled_max_ae=(
                        (actual.float() / mesh.size() - expected.float())
                        .abs()
                        .max()
                        .item()
                    ),
                    expert_input_shape=expert_shapes[-1],
                )
            )
            if args.backward:
                # A fixed nonzero linear probe isolates the Jacobian from the
                # already-known forward scale error. Do not alter model output.
                probe = torch.linspace(
                    0.5, 1.5, actual.numel(), device=actual.device
                ).reshape_as(actual)
                observed_grad = torch.autograd.grad(
                    (actual.float() * probe).sum(),
                    local,
                    retain_graph=args.parameter_gradients,
                )[0]
                reference_grad = torch.autograd.grad(
                    (expected.float() * probe).sum(),
                    local,
                    retain_graph=args.parameter_gradients,
                )[0]
                assert torch.isfinite(observed_grad).all()
                assert torch.isfinite(reference_grad).all()
                assert observed_grad.count_nonzero() > 0
                assert reference_grad.count_nonzero() > 0
                report[-1]["input_gradient"] = {
                    "max_ae": (observed_grad.float() - reference_grad.float())
                    .abs()
                    .max()
                    .item(),
                    "tp_degree_rescaled_max_ae": (
                        observed_grad.float() / mesh.size() - reference_grad.float()
                    )
                    .abs()
                    .max()
                    .item(),
                }
                if args.parameter_gradients:
                    parameters = dict(block.moe.named_parameters())
                    reference_parameters = dict(oracle.named_parameters())
                    assert parameters.keys() == reference_parameters.keys()
                    observed = torch.autograd.grad(
                        (actual.float() * probe).sum(), tuple(parameters.values())
                    )
                    references = torch.autograd.grad(
                        (expected.float() * probe).sum(),
                        tuple(reference_parameters.values()),
                    )
                    differences = {}
                    for (name, parameter), gradient, reference in zip(
                        parameters.items(), observed, references, strict=True
                    ):
                        # The distributed model consumes both sequence shards.
                        # Sum independent local-loss references over the owning
                        # mesh before slicing features/experts. Router weights
                        # are replicated only within their TP group.
                        reference = reference.detach().clone().contiguous()
                        owner_mesh = parameter.device_mesh
                        dist.all_reduce(reference, group=owner_mesh.get_group())
                        if isinstance(gradient, DTensor):
                            gradient = gradient.redistribute(
                                placements=parameter.placements
                            ).to_local()
                        if args.expert_parallel and name.startswith("experts."):
                            reference = reference.chunk(owner_mesh.size(), 0)[
                                owner_mesh.get_local_rank()
                            ]
                        elif name == "experts.w13":
                            if args.canonical_packed_weight_shards:
                                reference = reference.chunk(2, 1)[rank]
                            else:
                                gate, up = reference.chunk(2, 1)
                                reference = torch.cat(
                                    (gate.chunk(2, 1)[rank], up.chunk(2, 1)[rank]), 1
                                )
                        elif name == "experts.w2":
                            reference = reference.chunk(2, 2)[rank]
                        assert gradient.shape == reference.shape
                        assert torch.isfinite(gradient).all()
                        assert gradient.count_nonzero() > 0
                        assert reference.count_nonzero() > 0
                        error = gradient.float() - reference.float()
                        differences[name] = {
                            "max_ae": error.abs().max().item(),
                            "reference_max_abs": reference.abs().max().item(),
                            "relative_l2": (
                                error.norm() / reference.float().norm()
                            ).item(),
                        }
                    report[-1]["parameter_gradients"] = differences
            if args.sequence_ep_control:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if args.backward:
                    torch.testing.assert_close(
                        observed_grad, reference_grad, rtol=0, atol=0
                    )
        for name, path in source_paths.items():
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == sources[name]["sha256"]
            ), f"probe source changed during execution: {path}"
        output = args.output.with_name(args.output.stem + f"-rank{rank}.json")
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"rank={rank}: {report}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
