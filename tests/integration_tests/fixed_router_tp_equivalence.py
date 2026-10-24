"""Fixed-router EP1 x feature-TP conversion versus full-expert arithmetic.

Run with PYTHONPATH=. torchrun --nproc_per_node=2 on CPU/Gloo.
Append --compile for the targeted production aot_eager matrix, or --fullgraph
for production per-block compilation. Every case must produce a graph to pass.
This tests the lower-level transformation API, including an explicit shared
expert. Full Trainer resume is covered by trainer_ema_process_resume.py;
target-device validation remains separate.
"""

import argparse
from itertools import product

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from llmtuner.config.compile_config import CompileConfig
from llmtuner.memory.activation_checkpoint import wrap_full
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks
from llmtuner.parallel.model_compile import apply_compile
from llmtuner.parallel.tensor_parallel.checkpointing import (
    gather_tensor_parallel_state,
    load_tensor_parallel_state,
)
from llmtuner.parallel.tensor_parallel.linear_ops import functional_collectives
from llmtuner.parallel.tensor_parallel.tensor_parallel import realize_moe_blocks_tp
from tests.integration_tests.ep_wiring_equivalence import _model
from tests.integration_tests.fixed_router_ep_equivalence import explicit_reference


class SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(611)
        self.gate_proj = nn.Linear(64, 32, bias=False, dtype=torch.float64)
        self.up_proj = nn.Linear(64, 32, bias=False, dtype=torch.float64)
        self.down_proj = nn.Linear(32, 64, bias=False, dtype=torch.float64)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 0.03
                )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEDecoder(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block

    def forward(self, x, **kwargs):
        return self.mlp(x, **kwargs)


class MoEModel(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = nn.ModuleList([layer])

    def forward(self, x, **kwargs):
        return self.layers[0](x, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compile", action="store_true", help="Exercise aot_eager graphs"
    )
    parser.add_argument(
        "--fullgraph", action="store_true", help="Require one graph; implies --compile"
    )
    args = parser.parse_args()
    args.compile = args.compile or args.fullgraph
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        # apply_tp resolves the version-dependent API before tracing.
        functional_collectives()
        registry_size = getattr(
            torch._C._distributed_c10d, "_get_work_registry_size", None
        )
        for (
            recompute,
            batched,
            staged,
            coefficient,
            all_padding,
            shared_name,
            block_registration,
        ) in product(
            (False, True),
            (False, True),
            (False, True),
            (0.0, 0.03),
            (False, True),
            (None, "shared_expert", "shared_experts"),
            (False, True, "experts"),
        ):
            if args.compile and (
                not batched
                or not staged
                or all_padding
                or block_registration != "experts"
            ):
                continue
            if args.compile:
                torch._dynamo.reset()
                torch._dynamo.utils.counters.clear()
            pending_before = registry_size() if registry_size is not None else None
            reference = _model().train().layers[0].mlp
            target = _model().train()
            block = target.layers[0].mlp
            if shared_name is not None:
                setattr(block, shared_name, SharedExpert())
                setattr(reference, shared_name, SharedExpert())
            block_path = next(
                name for name, module in target.named_modules() if module is block
            )
            realize_moe_blocks_tp(
                [("layers.0.mlp", block)],
                tp_size=2,
                tp_rank=rank,
                group=dist.group.WORLD,
            )
            target._tp_checkpoint_layouts = {
                f"{block_path}.experts.gate_up_proj": (1, True, dist.group.WORLD),
                f"{block_path}.experts.down_proj": (2, False, dist.group.WORLD),
            }
            if shared_name is not None:
                for name, dimension in (
                    ("gate_proj", 0),
                    ("up_proj", 0),
                    ("down_proj", 1),
                ):
                    target._tp_checkpoint_layouts[
                        f"{block_path}.{shared_name}.{name}.weight"
                    ] = (dimension, False, dist.group.WORLD)
            if block_registration == "experts":
                block.experts._tp_checkpoint_layouts = {
                    "gate_up_proj": (1, True, dist.group.WORLD),
                    "down_proj": (2, False, dist.group.WORLD),
                }
                target._tp_checkpoint_layouts = {
                    key: value
                    for key, value in target._tp_checkpoint_layouts.items()
                    if ".experts." not in key
                }
            elif block_registration:
                block._tp_checkpoint_layouts = {
                    key.removeprefix(f"{block_path}."): value
                    for key, value in target._tp_checkpoint_layouts.items()
                }
                del target._tp_checkpoint_layouts
            swap_hf_moe_blocks(
                target,
                fixed_router=True,
                tp_enabled=True,
                router_aux_loss_coef=coefficient,
            )
            block = target.layers[0].mlp
            assert block._tp_moe_boundary
            assert block.routed_experts.inner_experts.w1_EFD.shape == (8, 24, 64)
            generator = torch.Generator().manual_seed(199)
            shape = (2, 8, 64) if batched else (8, 64)
            inputs = torch.randn(shape, generator=generator, dtype=torch.float64)
            weights = torch.randn(shape, generator=generator, dtype=torch.float64)
            mask = (torch.arange(inputs.numel() // 64) % 3 == 0).reshape(shape[:-1])
            if all_padding:
                mask.fill_(True)
            expected_inputs = inputs.clone().requires_grad_()
            actual_inputs = inputs.chunk(2, dim=-2)[rank].contiguous().requires_grad_()
            local_mask = mask.chunk(2, dim=-1)[rank].contiguous()
            denominator = (~mask).sum()
            AuxLoss.set_step_denominator(denominator, model_parts=[block])
            expected, auxiliary, counts = explicit_reference(
                reference, expected_inputs.reshape(-1, 64), mask.reshape(-1)
            )
            expected = expected.reshape(shape)
            if shared_name is not None:
                shared = getattr(reference, shared_name)
                expected = expected + F.linear(
                    F.silu(F.linear(expected_inputs, shared.gate_proj.weight))
                    * F.linear(expected_inputs, shared.up_proj.weight),
                    shared.down_proj.weight,
                )
            expected_loss = (expected * weights).sum()
            if coefficient:
                expected_loss = (
                    expected_loss + auxiliary * coefficient / denominator.clamp_min(1)
                )
            expected_loss = expected_loss + sum(
                p.sum() * 0 for p in reference.experts.parameters()
            )
            expected_loss.backward()
            if staged:
                block.set_padding_mask(local_mask.reshape(-1))
            router_calls = []
            hook = (
                None
                if args.compile
                else block.router.register_forward_pre_hook(
                    lambda *args, calls=router_calls: calls.append(True)
                )
            )
            runner = wrap_full(block) if recompute else block
            if args.compile:
                decoder = MoEDecoder(block)
                if recompute:
                    decoder = wrap_full(decoder)
                runner = apply_compile(
                    MoEModel(decoder),
                    compile_config=CompileConfig(
                        backend="aot_eager", per_block=args.fullgraph
                    ),
                )
                assert torch._dynamo.config.capture_scalar_outputs
            kwargs = {} if staged else {"padding_mask": local_mask}
            actual = runner(actual_inputs, **kwargs)
            (actual * weights.chunk(2, dim=-2)[rank]).sum().backward()
            if hook is not None:
                hook.remove()
            if args.compile:
                graph_count = torch._dynamo.utils.counters["stats"]["unique_graphs"]
                assert graph_count > 0, (
                    f"No compiled graph: {recompute=}, {coefficient=}, {shared_name=}; "
                    f"counters={dict(torch._dynamo.utils.counters)}"
                )
            if not args.compile:
                assert len(router_calls) == (2 if recompute else 1)
            assert block._pending_padding_mask is None or torch.equal(
                block._pending_padding_mask, local_mask.reshape(-1)
            )
            if coefficient:
                dist.all_reduce(block.router.gate.weight.grad)
                torch.testing.assert_close(
                    block.router.gate.weight.grad,
                    reference.gate.weight.grad,
                    atol=1e-7,
                    rtol=1e-6,
                )
                torch.testing.assert_close(
                    block.router.aux_loss.instance_acc,
                    (auxiliary.detach() / denominator.clamp_min(1)).float(),
                    atol=1e-7,
                    rtol=1e-6,
                )
            torch.testing.assert_close(
                actual, expected.chunk(2, dim=-2)[rank], atol=1e-10, rtol=1e-10
            )
            torch.testing.assert_close(
                actual_inputs.grad,
                expected_inputs.grad.chunk(2, dim=-2)[rank],
                atol=1e-10,
                rtol=1e-10,
            )
            torch.testing.assert_close(
                block.tokens_per_expert_E, counts.float(), atol=0, rtol=0
            )
            grouped = block.routed_experts.inner_experts
            gate_grad, up_grad = reference.experts.gate_up_proj.grad.chunk(2, dim=1)
            gradients = (gate_grad, up_grad, reference.experts.down_proj.grad)
            names = ("w1_EFD", "w3_EFD", "w2_EDF")
            expected_updates = []
            for name, gradient in zip(names, gradients, strict=True):
                dimension = 2 if name == "w2_EDF" else 1
                parameter = getattr(grouped, name)
                wanted = gradient.chunk(2, dim=dimension)[rank]
                torch.testing.assert_close(
                    parameter.grad, wanted, atol=1e-10, rtol=1e-10
                )
                expected_updates.append(parameter.detach().clone() - 0.01 * wanted)
            if shared_name is not None:
                for name, dimension in (
                    ("gate_proj", 0),
                    ("up_proj", 0),
                    ("down_proj", 1),
                ):
                    actual_parameter = getattr(block.shared_experts, name).weight
                    expected_parameter = getattr(shared, name).weight
                    torch.testing.assert_close(
                        actual_parameter.grad,
                        expected_parameter.grad.chunk(2, dim=dimension)[rank],
                        atol=1e-10,
                        rtol=1e-10,
                    )
                    assert (
                        f"shared_experts.{name}.weight" in block.tp_sharded_param_names
                    )
            torch.optim.SGD(block.parameters(), lr=0.01).step()
            for name, wanted in zip(names, expected_updates, strict=True):
                torch.testing.assert_close(
                    getattr(grouped, name), wanted, atol=1e-10, rtol=1e-10
                )
            local_state = target.state_dict()
            full_state = gather_tensor_parallel_state(local_state, model_parts=[target])
            original_gate, original_up = reference.experts.gate_up_proj.detach().chunk(
                2, dim=1
            )
            source_weights = (
                original_gate,
                original_up,
                reference.experts.down_proj.detach(),
            )
            for name, source, gradient in zip(
                names, source_weights, gradients, strict=True
            ):
                key = f"{block_path}.routed_experts.inner_experts.{name}"
                dimension = 2 if name == "w2_EDF" else 1
                assert full_state[key].shape[dimension] == 48
                torch.testing.assert_close(
                    full_state[key], source - 0.01 * gradient, atol=1e-10, rtol=1e-10
                )
            if shared_name is not None:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    parameter = getattr(shared, name).weight
                    key = f"{block_path}.shared_experts.{name}.weight"
                    torch.testing.assert_close(
                        full_state[key],
                        parameter.detach() - 0.01 * parameter.grad,
                        atol=1e-10,
                        rtol=1e-10,
                    )
            restored = load_tensor_parallel_state(full_state, model_parts=[target])
            for key, wanted in local_state.items():
                torch.testing.assert_close(restored[key], wanted, atol=0, rtol=0)
            dist.barrier()
            if registry_size is not None:
                assert registry_size() == pending_before, (
                    f"Collective work leaked: {recompute=}, {batched=}, {staged=}, "
                    f"{coefficient=}, {all_padding=}, {shared_name=}, "
                    f"{block_registration=}"
                )
            if rank == 0:
                print(
                    f"fixed TP {recompute=} {batched=} "
                    f"{staged=} {coefficient=} {all_padding=} "
                    f"{shared_name=} {block_registration=}: passed",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
