"""Independent dense oracle with explicit CP-local FP32 weight reduction.

Run on four CPU/Gloo ranks with the Trainer harness arguments, for example
``native aux adam full fp32_router forward_trajectory``. Only the dense oracle
changes: QKV and expert weight gradients sum two FP32 CP-local contributions.
The production model, optimizer, original assertions and tolerances stay intact.
This fixture is specific to contiguous CP2/TP2 token and feature partitions;
it does not prove target-device, arbitrary topology or exact gradient equivalence.
"""

import runpy
import sys
from types import MethodType

import torch
import torch.nn.functional as F

import tests.integration_tests.tp_ep_cp_equivalence as fixture


class CPLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        x = x.reshape(-1, x.shape[-1])
        flat = dy.reshape(-1, dy.shape[-1])
        parts = []
        for feature_grad in flat.chunk(2, dim=-1):
            local = [
                (g.T @ a).float()
                for g, a in zip(
                    feature_grad.chunk(2, dim=0), x.chunk(2, dim=0), strict=True
                )
            ]
            parts.append((local[0] + local[1]).to(weight.dtype))
        return (dy @ weight), torch.cat(parts, dim=0)


class ExpertLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, first_half):
        ctx.save_for_backward(x, weight, first_half)
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, dy):
        x, weight, first_half = ctx.saved_tensors
        left = (dy[first_half].T @ x[first_half]).float()
        right = (dy[~first_half].T @ x[~first_half]).float()
        return dy @ weight, (left + right).to(weight.dtype), None


original = fixture._build
count = 0


def validate_reference_gradients():
    """Check custom backward against independent local autograd contributions."""
    generator = torch.Generator().manual_seed(115)
    x = torch.randn(8, 6, generator=generator, dtype=torch.float64).requires_grad_()
    weight = torch.randn(4, 6, generator=generator, dtype=torch.float64)
    weight.requires_grad_()
    dy = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    for first in (torch.arange(8) < 4, torch.ones(8).bool(), torch.zeros(8).bool()):
        output = ExpertLinear.apply(x, weight, first)
        actual = torch.autograd.grad(output, (x, weight), dy)
        expected_input = torch.autograd.grad(F.linear(x, weight), x, dy)[0]
        local = [
            torch.autograd.grad(F.linear(x[mask], weight), weight, dy[mask])[0]
            for mask in (first, ~first)
        ]
        torch.testing.assert_close(output, F.linear(x, weight), atol=0, rtol=0)
        torch.testing.assert_close(actual[0], expected_input, atol=0, rtol=0)
        torch.testing.assert_close(
            actual[1], (local[0].float() + local[1].float()).double(), atol=0, rtol=0
        )
    actual = torch.autograd.grad(CPLinear.apply(x, weight), (x, weight), dy)
    expected_weight = []
    for feature in (slice(0, 2), slice(2, 4)):
        local = [
            torch.autograd.grad(
                F.linear(x[rows], weight[feature]), weight, dy[rows, feature]
            )[0][feature]
            for rows in (slice(0, 4), slice(4, 8))
        ]
        expected_weight.append((local[0].float() + local[1].float()).double())
    torch.testing.assert_close(actual[0], expected_input, atol=0, rtol=0)
    torch.testing.assert_close(actual[1], torch.cat(expected_weight), atol=0, rtol=0)


def build(*args, **kwargs):
    global count
    cfg = args[0]
    if (
        cfg.parallel.tensor_parallel_size != 2
        or cfg.parallel.context_parallel_kv_allgather_size != 2
        or cfg.parallel.expert_parallel_size != 2
        or cfg.parallel.cp_config.context_parallel_load_balancer is not None
        or cfg.parallel.fsdp_config.fsdp_reduce_dtype != "float32"
    ):
        raise ValueError(
            "Rounded reference requires contiguous TP2/KVCP2/EP2 "
            "with float32 FSDP reduction"
        )
    model = original(*args, **kwargs)
    count += 1
    if count == 2:
        for layer in model.layers:
            for name in ("q_proj", "k_proj", "v_proj"):
                module = getattr(layer.self_attn, name)
                assert module.bias is None

                def forward(self, x):
                    return CPLinear.apply(x, self.weight)

                module.forward = MethodType(forward, module)
        for layer in model.layers:

            def experts_forward(self, hidden, selected, scores):
                output = torch.zeros_like(hidden)
                for expert in range(self.num_experts):
                    rows, slots = (selected == expert).nonzero(as_tuple=True)
                    gate_weight, up_weight = self.gate_up_proj[expert].chunk(2, dim=0)
                    x = hidden[rows]
                    half = rows < hidden.shape[0] // 2
                    gate = ExpertLinear.apply(x, gate_weight, half)
                    up = ExpertLinear.apply(x, up_weight, half)
                    value = ExpertLinear.apply(
                        F.silu(gate) * up, self.down_proj[expert], half
                    )
                    output = output.index_add(
                        0, rows, value * scores[rows, slots, None]
                    )
                return output

            layer.mlp.experts.forward = MethodType(experts_forward, layer.mlp.experts)
        print(
            "INDEPENDENT_CP_QKV_EXPERT_REFERENCE: CP2 token halves, "
            "TP2 feature halves, local FP64 GEMM then FP32 SUM",
            flush=True,
        )
    return model


if __name__ == "__main__":
    if len(sys.argv) <= 5 or sys.argv[5] != "fp32_router":
        raise ValueError("Rounded reference requires the explicit fp32_router option")
    validate_reference_gradients()
    fixture._build = build
    try:
        runpy.run_module(
            "tests.integration_tests.trainer_tp_ep_cp_equivalence", run_name="__main__"
        )
        assert count == 2, "Expected one production model and one dense oracle"
    finally:
        fixture._build = original
