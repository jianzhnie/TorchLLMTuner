"""Public fixed-router EP wiring versus explicit per-expert arithmetic.

Run with PYTHONPATH=. torchrun --nproc_per_node=2 on CPU/Gloo. Tests
local EP1 and EP2 alltoall/Domino, full activation checkpoint replay, padding,
empty sources and empty destination experts. This does not validate device overlap.
"""

from itertools import product

import torch
import torch.distributed as dist
import torch.nn.functional as F

from llmtuner.config.parallel_config import ExpertParallelConfig, ParallelConfig
from llmtuner.memory.activation_checkpoint import wrap_full
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.parallel.expert_parallel.expert_parallel import apply_ep
from tests.integration_tests.ep_wiring_equivalence import _model


def explicit_reference(block, inputs, mask):
    """Use original HF weights and independent fixed selection/FFN equations."""
    count, dim = inputs.shape
    experts, top_k = 8, 2
    ids = (torch.arange(count)[:, None] + torch.arange(top_k)[None, :]) % experts
    scores = F.linear(inputs.float(), block.gate.weight.float()).softmax(-1)
    selected = scores.new_full((count, top_k), 1.0 / top_k)
    output = inputs * 0
    for expert in range(experts):
        token, slot = torch.where(ids == expert)
        gate, up = block.experts.gate_up_proj[expert].chunk(2, dim=0)
        hidden = F.silu(F.linear(inputs[token], gate)) * F.linear(inputs[token], up)
        values = F.linear(hidden, block.experts.down_proj[expert])
        output = output.index_add(0, token, values * selected[token, slot, None])
    counts = torch.bincount(ids[~mask].flatten(), minlength=experts)
    frequencies = counts.float() / counts.sum().clamp_min(1) * experts
    probabilities = F.normalize(scores, p=1, dim=-1) * ~mask[:, None]
    raw_aux = (frequencies * probabilities.sum(0)).sum()
    assert output.shape == (count, dim)
    return output, raw_aux, counts


def main():
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        for dispatcher, recompute, scenario, coefficient, ep_size in product(
            ("alltoall", "domino"),
            (False, True),
            ("unequal", "empty", "sparse", "all_padding"),
            (0.0, 0.03),
            (1, 2),
        ):
            if ep_size == 1 and dispatcher != "alltoall":
                continue
            reference = _model().train().layers[0].mlp
            target = _model().train()
            config = ParallelConfig(
                expert_parallel_size=ep_size,
                ep_config=ExpertParallelConfig(
                    fixed_router=True,
                    ep_token_dispatcher=dispatcher,
                    router_aux_loss_coef=coefficient,
                ),
            )
            apply_ep(target, config, ep_group=dist.group.WORLD)
            block = target.layers[0].mlp
            assert block.router.fixed_router
            num_local = 8 // ep_size
            expert_start = rank * num_local if ep_size > 1 else 0
            expert_slice = slice(expert_start, expert_start + num_local)
            assert block.routed_experts.inner_experts.w1_EFD.shape == (
                num_local,
                48,
                64,
            )
            tokens = {
                "unequal": (5, 7),
                "empty": (0, 3),
                "sparse": (1, 2),
                "all_padding": (3, 5),
            }[scenario][rank]
            generator = torch.Generator().manual_seed(731 + rank)
            inputs = torch.randn(tokens, 64, generator=generator, dtype=torch.float64)
            weights = torch.randn(tokens, 64, generator=generator, dtype=torch.float64)
            mask = torch.arange(tokens) % 3 == 0
            if scenario == "all_padding":
                mask.fill_(True)
            denominator = (~mask).sum()
            dist.all_reduce(denominator)
            AuxLoss.set_step_denominator(denominator, model_parts=[block])
            expected_inputs = inputs.clone().requires_grad_()
            actual_inputs = inputs.clone().requires_grad_()
            expected, auxiliary, counts = explicit_reference(
                reference, expected_inputs, mask
            )
            expected_loss = (expected * weights).sum()
            if coefficient:
                expected_loss = (
                    expected_loss + auxiliary * coefficient / denominator.clamp_min(1)
                )
            # Give unused reference experts explicit zero gradients too.
            expected_loss = expected_loss + sum(
                p.sum() * 0 for p in reference.experts.parameters()
            )
            expected_loss.backward()
            invocations = []
            hook = block.router.register_forward_pre_hook(
                lambda *args, calls=invocations: calls.append(True)
            )
            try:
                runner = wrap_full(block) if recompute else block
                actual = runner(actual_inputs, padding_mask=mask)
                (actual * weights).sum().backward()
            finally:
                hook.remove()
            assert len(invocations) == (2 if recompute else 1)
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
            # Routing runs in FP32 even when the expert arithmetic is FP64.
            torch.testing.assert_close(
                actual_inputs.grad, expected_inputs.grad, atol=1e-7, rtol=1e-6
            )
            torch.testing.assert_close(
                block.router.gate.weight.grad,
                reference.gate.weight.grad,
                atol=1e-7,
                rtol=1e-6,
            )
            torch.testing.assert_close(
                block.tokens_per_expert_E, counts.float(), atol=0, rtol=0
            )
            if coefficient:
                torch.testing.assert_close(
                    block.router.aux_loss.instance_acc,
                    (auxiliary.detach() / denominator.clamp_min(1)).float(),
                    atol=1e-7,
                    rtol=1e-6,
                )
            else:
                assert block.router.aux_loss is None
                assert block.router.gate.weight.grad is None
                assert reference.gate.weight.grad is None
            gate_grad, up_grad = reference.experts.gate_up_proj.grad.chunk(2, dim=1)
            expected_gradients = (gate_grad, up_grad, reference.experts.down_proj.grad)
            actual_parameters = (
                block.routed_experts.inner_experts.w1_EFD,
                block.routed_experts.inner_experts.w3_EFD,
                block.routed_experts.inner_experts.w2_EDF,
            )
            gate_weight, up_weight = reference.experts.gate_up_proj.detach().chunk(
                2, dim=1
            )
            reference_weights = (
                gate_weight,
                up_weight,
                reference.experts.down_proj.detach(),
            )
            expected_updates = []
            for parameter, source, wanted in zip(
                actual_parameters, reference_weights, expected_gradients, strict=True
            ):
                source = source[expert_slice]
                torch.testing.assert_close(parameter, source, atol=0, rtol=0)
                wanted = wanted.contiguous().clone()
                if ep_size > 1:
                    dist.all_reduce(wanted)
                torch.testing.assert_close(
                    parameter.grad,
                    wanted[expert_slice],
                    atol=1e-10,
                    rtol=1e-10,
                )
                expected_updates.append(source - 0.01 * wanted[expert_slice])
            expected_gate = reference.gate.weight.detach().clone()
            if reference.gate.weight.grad is not None:
                expected_gate -= 0.01 * reference.gate.weight.grad
            torch.optim.SGD(block.parameters(), lr=0.01).step()
            for parameter, expected_update in zip(
                actual_parameters, expected_updates, strict=True
            ):
                torch.testing.assert_close(
                    parameter, expected_update, atol=1e-10, rtol=1e-10
                )
            torch.testing.assert_close(
                block.router.gate.weight, expected_gate, atol=1e-7, rtol=1e-6
            )
            dist.barrier()
            if rank == 0:
                print(
                    f"fixed router {dispatcher=} {recompute=} "
                    f"{scenario=} {coefficient=} {ep_size=}: passed",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
