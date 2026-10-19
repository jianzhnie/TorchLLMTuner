"""Two-rank CPU/Gloo root MoE TP and reapplication equivalence.

Run with torchrun --nproc_per_node=2; optionally pass --router-fp64.
The small HF-style block includes routed and shared experts. Compare production
TP boundaries against dense outputs/weight gradients and an independent partial
expert input-gradient oracle. FP32 router backward rounds each rank's partial
before summation, so FP64 dense input-gradient equality is only asserted with
--router-fp64. Neither mode establishes target-device kernel correctness.
"""

import argparse
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import functional as F

from llmtuner.config import ParallelConfig
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--router-fp64", action="store_true")
ARGS = PARSER.parse_args()


class Experts(nn.Module):
    def __init__(self, count, dim, hidden):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(count, 2 * hidden, dim))
        self.down_proj = nn.Parameter(torch.randn(count, dim, hidden))


class SharedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SharedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(16, 4, bias=False)
        self.experts = Experts(4, 16, 8)
        self.top_k = 2
        self.num_experts = 4
        self.shared_experts = SharedMLP()
        self._tp_config = {
            "experts": "moe_tp_experts",
            "shared_experts.gate_proj": "colwise",
            "shared_experts.up_proj": "colwise",
            "shared_experts.down_proj": "rowwise",
        }

    def forward(self, x):
        flat = x.reshape(-1, 16)
        logits = self.gate(flat)
        scores = torch.softmax(logits if ARGS.router_fp64 else logits.float(), -1)
        weights, ids = scores.topk(self.top_k, dim=-1)
        routing = torch.zeros_like(scores).scatter(-1, ids, weights)
        output = torch.zeros_like(flat)
        for expert in range(self.num_experts):
            selected = routing[:, expert] != 0
            gate, up = F.linear(
                flat[selected], self.experts.gate_up_proj[expert]
            ).chunk(2, -1)
            value = F.linear(F.silu(gate) * up, self.experts.down_proj[expert])
            output[selected] = (
                output[selected] + value * routing[selected, expert : expert + 1]
            )
        return output.reshape_as(x) + self.shared_experts(x)


dist.init_process_group("gloo")
try:
    rank = dist.get_rank()
    torch.manual_seed(83)
    reference = SharedBlock().double()
    model = copy.deepcopy(reference)
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
    cfg = ParallelConfig(tensor_parallel_size=2)
    full_x = torch.randn(1, 8, 16, dtype=torch.float64, requires_grad=True)
    expected = reference(full_x)
    expected.square().sum().backward()
    # Independent partial-expert graph: slice reference weights directly,
    # without using any production sharding or TP communication function.
    oracle_x = full_x.detach().clone().requires_grad_()
    logits = reference.gate(oracle_x.reshape(-1, 16))
    scores = torch.softmax(logits if ARGS.router_fp64 else logits.float(), dim=-1)
    weights, ids = scores.topk(2, dim=-1)
    routing = torch.zeros_like(scores).scatter(-1, ids, weights)
    partial = torch.zeros_like(oracle_x.reshape(-1, 16))
    f = 8
    for expert in range(4):
        selected = routing[:, expert] != 0
        w = reference.experts.gate_up_proj[expert]
        gate = torch.nn.functional.linear(
            oracle_x.reshape(-1, 16)[selected], w[:f].chunk(2, 0)[rank]
        )
        up = torch.nn.functional.linear(
            oracle_x.reshape(-1, 16)[selected], w[f:].chunk(2, 0)[rank]
        )
        value = torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up,
            reference.experts.down_proj[expert].chunk(2, 1)[rank],
        )
        partial[selected] = (
            partial[selected] + value * routing[selected, expert : expert + 1]
        )
    shared = reference.shared_experts
    gate = torch.nn.functional.linear(
        oracle_x, shared.gate_proj.weight.chunk(2, 0)[rank]
    )
    up = torch.nn.functional.linear(oracle_x, shared.up_proj.weight.chunk(2, 0)[rank])
    partial = partial.reshape_as(oracle_x) + torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up, shared.down_proj.weight.chunk(2, 1)[rank]
    )
    (oracle_grad,) = torch.autograd.grad(
        partial, oracle_x, grad_outputs=2 * expected.detach()
    )
    dist.all_reduce(oracle_grad)
    error = (oracle_grad - full_x.grad).abs().max().item()
    print(
        f"rank={rank} partial vs dense input gradient max error={error:.9g}",
        flush=True,
    )
    if ARGS.router_fp64:
        torch.testing.assert_close(oracle_grad, full_x.grad, rtol=1e-9, atol=1e-10)
    apply_tp(model, mesh, cfg)
    for iteration in range(2):
        model.zero_grad(set_to_none=True)
        if iteration:
            before = list(model.parameters())
            apply_tp(model, mesh, cfg)
            assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
        local_x = full_x.detach().chunk(2, dim=1)[rank].clone().requires_grad_()
        actual = model(local_x)
        torch.testing.assert_close(
            actual, expected.detach().chunk(2, dim=1)[rank], rtol=1e-10, atol=1e-10
        )
        actual.square().sum().backward()
        torch.testing.assert_close(
            local_x.grad, oracle_grad.chunk(2, dim=1)[rank], rtol=1e-9, atol=1e-10
        )
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(model.shared_experts, name)
            baseline = getattr(reference.shared_experts, name)
            dim = 1 if name == "down_proj" else 0
            torch.testing.assert_close(
                projection.weight.grad,
                baseline.weight.grad.chunk(2, dim=dim)[rank],
                rtol=1e-9,
                atol=1e-10,
            )
        print(
            f"rank={rank} iteration={iteration}: output/input/shared gradients passed",
            flush=True,
        )
finally:
    dist.destroy_process_group()
