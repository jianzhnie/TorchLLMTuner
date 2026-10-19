"""Two-rank CPU/Gloo root MoE TP and reapplication equivalence.

Run with torchrun --nproc_per_node=2; optionally pass --router-fp64.
The small HF-style block includes routed and shared experts. Compare production
TP boundaries against dense outputs/weight gradients and an independent partial
expert gradient oracle. Production Trainer performs replicated router reduction.
FP32 router backward rounds each rank's partial
before summation, so FP64 dense input-gradient equality is only asserted with
--router-fp64. Neither mode establishes target-device kernel correctness.
"""

import argparse
import copy
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import functional as F

from llmtuner.config import ParallelConfig
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp
from llmtuner.trainer.trainer import Trainer

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--router-fp64", action="store_true")
PARSER.add_argument("--tied-shared-projections", action="store_true")
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
    if ARGS.tied_shared_projections:
        reference.shared_experts.up_proj.weight = (
            reference.shared_experts.gate_proj.weight
        )
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
    reference_parameters = dict(reference.named_parameters())
    oracle_grad, *parameter_grads = torch.autograd.grad(
        partial,
        (oracle_x, *reference_parameters.values()),
        grad_outputs=2 * expected.detach(),
    )
    oracle_parameters = dict(zip(reference_parameters, parameter_grads, strict=True))
    dist.all_reduce(oracle_parameters["gate.weight"])
    dist.all_reduce(oracle_grad)
    error = (oracle_grad - full_x.grad).abs().max().item()
    print(
        f"rank={rank} partial vs dense input gradient max error={error:.9g}",
        flush=True,
    )
    if ARGS.router_fp64:
        torch.testing.assert_close(oracle_grad, full_x.grad, rtol=1e-9, atol=1e-10)
    apply_tp(model, mesh, cfg)
    if ARGS.tied_shared_projections:
        assert (
            model.shared_experts.up_proj.weight is model.shared_experts.gate_proj.weight
        )
    trainer = SimpleNamespace(
        model_parts=[model],
        parallel_dims=SimpleNamespace(get_optional_mesh=lambda name: mesh["tp"]),
    )
    expected_gradients = {}
    for name, gradient in oracle_parameters.items():
        if name == "experts.gate_up_proj":
            gate, up = gradient.chunk(2, dim=1)
            gradient = torch.cat(
                [gate.chunk(2, dim=1)[rank], up.chunk(2, dim=1)[rank]], dim=1
            )
        elif name == "experts.down_proj":
            gradient = gradient.chunk(2, dim=2)[rank]
        elif name.startswith("shared_experts."):
            dimension = 1 if "down_proj" in name else 0
            gradient = gradient.chunk(2, dim=dimension)[rank]
        expected_gradients[name] = gradient
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
        Trainer._allreduce_replicated_tp_grads(trainer)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.grad, expected_gradients[name], rtol=1e-9, atol=1e-10
            )
            if ARGS.router_fp64 and name == "gate.weight":
                torch.testing.assert_close(
                    parameter.grad,
                    reference.gate.weight.grad,
                    rtol=1e-9,
                    atol=1e-10,
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
            f"rank={rank} iteration={iteration}: output and all gradients passed",
            flush=True,
        )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.step()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter,
            before[name] - 0.001 * expected_gradients[name],
            rtol=1e-9,
            atol=1e-10,
        )
    print(f"rank={rank}: all parameter SGD updates passed", flush=True)
    # Exercise the production packed/shared checkpoint transforms with real DCP.
    import torch.distributed.checkpoint as dcp

    from llmtuner.components.checkpointer.model_state import ModelCheckpointState
    from llmtuner.components.optimizer.optimizer import OptimizersContainer
    from llmtuner.config.optimizer_config import OptimizerConfig

    adam = OptimizersContainer(
        OptimizerConfig(implementation="for-loop"), model_parts=[model]
    )
    adam.step()
    source_model = ModelCheckpointState(model)
    saved_optimizer = copy.deepcopy(adam.state_dict())
    saved_model = copy.deepcopy(source_model.state_dict())
    paths = [tempfile.mkdtemp(prefix="llmtuner-tp-moe-dcp-") if rank == 0 else None]
    dist.broadcast_object_list(paths)
    dcp.save({"model": source_model, "optimizer": adam}, checkpoint_id=paths[0])
    fresh = copy.deepcopy(reference)
    apply_tp(fresh, mesh, cfg)
    target_model = ModelCheckpointState(fresh)
    target_optimizer = OptimizersContainer(
        OptimizerConfig(implementation="for-loop"), model_parts=[fresh]
    )
    dcp.load(
        {"model": target_model, "optimizer": target_optimizer}, checkpoint_id=paths[0]
    )
    torch.testing.assert_close(target_model.state_dict(), saved_model, rtol=0, atol=0)
    torch.testing.assert_close(
        target_optimizer.state_dict(), saved_optimizer, rtol=0, atol=0
    )
    for name, parameter in fresh.named_parameters():
        torch.testing.assert_close(
            parameter, dict(model.named_parameters())[name], rtol=0, atol=0
        )
        parameter.grad = dict(model.named_parameters())[name].grad.detach().clone()
    if ARGS.tied_shared_projections:
        assert (
            fresh.shared_experts.gate_proj.weight is fresh.shared_experts.up_proj.weight
        )
    adam.step()
    target_optimizer.step()
    for name, parameter in fresh.named_parameters():
        torch.testing.assert_close(
            parameter, dict(model.named_parameters())[name], rtol=0, atol=0
        )
    print(
        f"rank={rank}: packed/shared DCP state and next Adam update passed", flush=True
    )
finally:
    dist.destroy_process_group()
