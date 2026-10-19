"""Public TP2/EP2 assembly with CPU MC2 operator emulation.

Run with PYTHONPATH=. torchrun --nproc_per_node=2. Dense TP and EP transport
are real; NPU fused kernels, FSDP and compile are not covered.
LLMTUNER_MC2_BOUNDARY_DIAGNOSTICS=1 captures layer boundaries and FP64
references for actual backward operands. Adding LLMTUNER_MC2_SPLIT_GRAD_DIAGNOSTIC=1
changes only the emulated input gradient to branchwise rounding as a causal
probe; a pass in that mode is not production MC2 validation.
LLMTUNER_MC2_FP64_REFERENCE=1 also evaluates an unmodified, unsharded HF
model from the same quantized weights and reports complete replicated-gradient
errors; those diagnostics do not change the original acceptance assertions.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.config.parallel_config import ExpertParallelConfig, ParallelConfig
from llmtuner.parallel.expert_parallel import mc2_dispatch
from llmtuner.parallel.expert_parallel.expert_parallel import apply_ep
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.ep_wiring_equivalence import _model
from tests.integration_tests.mc2_distributed_adapter import DistributedOps


class GradientDiagnosticOps(DistributedOps):
    """Inspect real gate/up input-gradient operands without changing results."""

    def project(self, inputs, weights, counts, transpose):
        output = super().project(inputs, weights, counts, transpose)
        if not transpose and weights.shape[1:] == (96, 64):
            split_outputs = []
            oracle_outputs = []
            for piece, weight in zip(inputs.split(counts), weights, strict=True):
                gate, up = piece.chunk(2, dim=1)
                gate_weight, up_weight = weight.chunk(2, dim=0)
                split_outputs.append(gate @ gate_weight + up @ up_weight)
                oracle_outputs.append(
                    gate.double() @ gate_weight.double()
                    + up.double() @ up_weight.double()
                )
            split = torch.cat(split_outputs)
            oracle = torch.cat(oracle_outputs)
            print(
                f"rank {dist.get_rank()}: actual gate/up backward "
                f"fused_vs_split={(output - split).abs().max().item()} "
                f"fused_oracle={(output.double() - oracle).abs().max().item()} "
                f"split_oracle={(split.double() - oracle).abs().max().item()} "
                f"single_round_equal={torch.equal(output, oracle.half())}",
                flush=True,
            )
            if os.environ.get("LLMTUNER_MC2_SPLIT_GRAD_DIAGNOSTIC") == "1":
                return split
        return output


def hf_parameter_view(parameters, key, rank, *, gradient):
    """Map this fixed Qwen3 fixture's TP/EP leaves to independent HF slices."""
    if ".router.gate.weight" in key:
        key = key.replace(".router.gate.weight", ".gate.weight")
    for suffix, source, half in (
        ("w1_EFD", "gate_up_proj", 0),
        ("w3_EFD", "gate_up_proj", 1),
        ("w2_EDF", "down_proj", None),
    ):
        marker = f"routed_experts.inner_experts.{suffix}"
        if marker in key:
            parameter = parameters[key.replace(marker, f"experts.{source}")]
            value = parameter.grad if gradient else parameter
            if half is not None:
                value = value.chunk(2, dim=1)[half]
            return value.chunk(2, dim=0)[rank]
    parameter = parameters[key]
    value = parameter.grad if gradient else parameter
    if any(f".{name}.weight" in key for name in ("q_proj", "k_proj", "v_proj")):
        return value.chunk(2, dim=0)[rank]
    if ".o_proj.weight" in key:
        return value.chunk(2, dim=1)[rank]
    return value


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        ops = (
            GradientDiagnosticOps()
            if os.environ.get("LLMTUNER_MC2_BOUNDARY_DIAGNOSTICS") == "1"
            else DistributedOps()
        )
        models = []
        for dispatcher in ("alltoall", "mc2"):
            cfg = ParallelConfig(
                tensor_parallel_size=2,
                expert_parallel_size=2,
                ep_config=ExpertParallelConfig(ep_token_dispatcher=dispatcher),
            )
            model = _model().half()
            apply_tp(model, mesh, cfg)
            # EP deferral must retain full F, not a packed TP projection shard.
            assert model.layers[0].mlp.experts.gate_up_proj.shape == (8, 96, 64)
            with patch.object(mc2_dispatch, "load_mc2_ops", lambda: ops):
                apply_ep(model, cfg, ep_group=dist.group.WORLD)
            assert model.layers[0].mlp.routed_experts.inner_experts.w1_EFD.shape == (
                4,
                48,
                64,
            )
            assert any(
                type(module).__name__
                in {
                    "ColumnParallelLinear",
                    "RowParallelLinear",
                    "ColwiseLinearNoGather",
                }
                for module in model.modules()
            )
            models.append(model)
        ids = torch.arange(16) % 128
        positions = torch.arange(16)
        local_ids = ids.chunk(2)[rank]
        outputs = []
        boundaries = []
        routes = []
        if os.environ.get("LLMTUNER_MC2_FP64_REFERENCE") == "1":
            for model in models:
                route = {}
                routes.append(route)
                for index, layer in enumerate(model.layers):

                    def capture_routes(
                        module, args, output, *, index=index, route=route
                    ):
                        route[index] = output[1].detach().clone()

                    layer.mlp.router.register_forward_hook(capture_routes)
        if os.environ.get("LLMTUNER_MC2_BOUNDARY_DIAGNOSTICS") == "1":
            for model in models:
                captured = {}
                boundaries.append(captured)
                for index, layer in enumerate(model.layers):

                    def capture(
                        module, args, output, *, index=index, captured=captured
                    ):
                        captured[index] = (args[0], output)
                        args[0].retain_grad()
                        output.retain_grad()

                    layer.mlp.routed_experts.register_forward_hook(capture)
        for model in models:
            with (
                patch.object(mc2_dispatch, "load_mc2_ops", lambda: ops),
                patch.object(
                    mc2_dispatch,
                    "mc2_communication_context",
                    lambda *args: (ops, "test", 2),
                ),
            ):
                output = model(local_ids, positions=positions)
                output.float().square().sum().backward()
            runtime = SimpleNamespace(
                parallel_dims=SimpleNamespace(get_optional_mesh=lambda name: mesh),
                model_parts=[model],
            )
            Trainer._allreduce_replicated_tp_grads(runtime)
            outputs.append(output)
        torch.testing.assert_close(outputs[1], outputs[0], atol=5e-4, rtol=1e-2)
        if boundaries:
            for index in boundaries[0]:
                reference_input, reference_output = boundaries[0][index]
                fused_input, fused_output = boundaries[1][index]
                for label, actual, expected in (
                    ("input", fused_input, reference_input),
                    ("output", fused_output, reference_output),
                    ("input_grad", fused_input.grad, reference_input.grad),
                    ("output_grad", fused_output.grad, reference_output.grad),
                ):
                    difference = (actual - expected).abs().max().item()
                    print(
                        f"rank {rank}: layer {index} {label} diff={difference}",
                        flush=True,
                    )
        if os.environ.get("LLMTUNER_MC2_FP64_REFERENCE") == "1":
            # Recreate the exact quantized initial weights, then evaluate the
            # unmodified HF graph without TP, EP or fused custom autograd.
            oracle = _model().half().double()
            oracle_routes = {}
            for index, layer in enumerate(oracle.layers):

                def capture_hf_routes(module, args, output, *, index=index):
                    oracle_routes[index] = output[2].detach().clone()

                layer.mlp.gate.register_forward_hook(capture_hf_routes)
            oracle_output = oracle(ids, positions=positions)
            oracle_output.float().square().sum().backward()
            oracle_parameters = dict(oracle.named_parameters())
            for route in routes:
                for index, selected in route.items():
                    torch.testing.assert_close(
                        selected, oracle_routes[index].chunk(2)[rank], atol=0, rtol=0
                    )
            print(
                f"rank {rank}: FP64 router choices identical in both layers", flush=True
            )
            compared = 0
            for dispatcher, model, output in zip(
                ("alltoall", "mc2"), models, outputs, strict=True
            ):
                local_oracle = oracle_output.chunk(2)[rank]
                print(
                    f"rank {rank}: FP64 {dispatcher} output max_error="
                    f"{(output.double() - local_oracle).abs().max().item()}",
                    flush=True,
                )
                for key, parameter in model.named_parameters():
                    expected_weight = hf_parameter_view(
                        oracle_parameters, key, rank, gradient=False
                    )
                    torch.testing.assert_close(
                        parameter.double(), expected_weight, atol=0, rtol=0
                    )
                    expected_gradient = hf_parameter_view(
                        oracle_parameters, key, rank, gradient=True
                    )
                    assert parameter.grad is not None and expected_gradient is not None
                    assert parameter.grad.shape == expected_gradient.shape, key
                    error = parameter.grad.double() - expected_gradient
                    relative_norm = error.norm() / expected_gradient.norm().clamp_min(
                        1e-30
                    )
                    print(
                        f"rank {rank}: FP64 {dispatcher} {key} "
                        f"max_error={error.abs().max().item()} "
                        f"relative_norm={relative_norm.item()}",
                        flush=True,
                    )
                    compared += 1
            assert compared == sum(len(list(model.parameters())) for model in models)
            print(
                f"rank {rank}: FP64 mapped all {compared} parameter views", flush=True
            )
        left = dict(models[0].named_parameters())
        right = dict(models[1].named_parameters())
        assert left.keys() == right.keys()
        failures = []
        output_difference = (outputs[1] - outputs[0]).abs().max().item()
        print(
            f"rank {rank}: output max diff {output_difference}",
            flush=True,
        )
        for key in left:
            assert left[key].grad is not None, key
            assert right[key].grad is not None, key
            try:
                torch.testing.assert_close(
                    right[key].grad,
                    left[key].grad,
                    atol=5e-4,
                    rtol=1e-2,
                    msg=lambda msg, key=key: f"{key}: {msg}",
                )
            except AssertionError as error:
                failures.append(str(error))
                print(f"rank {rank}: {error}", flush=True)
        assert not failures, f"{len(failures)} parameter gradients differ"
        dist.barrier()
        if rank == 0:
            mode = (
                "split-gradient diagnostic"
                if os.environ.get("LLMTUNER_MC2_SPLIT_GRAD_DIAGNOSTIC") == "1"
                else "fused adapter"
            )
            print(f"TP2/EP2 {mode}: outputs and all parameter gradients passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
