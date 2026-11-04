"""Real EP conversion under no_grad preserves legacy expert freezing and gradients.

Run with torchrun --standalone --nproc_per_node=2; the shared harness selects
NPU/HCCL by default, or CPU/Gloo with LLMTUNER_TEST_DEVICE=cpu.
"""

import copy

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks
from tests.integration_tests.harness import init_distributed


class LegacyBlock(nn.Module):
    def __init__(self, names):
        super().__init__()
        self.top_k = 2
        self.norm_topk_prob = True
        self.gate = nn.Linear(8, 4, bias=False)
        self.gate.weight.requires_grad_(False)
        self.experts = nn.ModuleList()
        self.names = names
        for _ in range(4):
            expert = nn.Module()
            expert.act_fn = nn.SiLU()
            for name, shape in zip(names, ((8, 12), (8, 12), (12, 8)), strict=True):
                setattr(expert, name, nn.Linear(*shape, bias=False))
            getattr(expert, names[1]).weight.requires_grad_(False)
            self.experts.append(expert)

    def forward(self, inputs):
        scores, ids = self.gate(inputs).softmax(-1).topk(self.top_k, dim=-1)
        scores = scores / scores.sum(-1, keepdim=True)
        result = torch.zeros_like(inputs)
        for index, expert in enumerate(self.experts):
            gate, up, down = (getattr(expert, name) for name in self.names)
            output = down(torch.nn.functional.silu(gate(inputs)) * up(inputs))
            coefficient = (scores * (ids == index)).sum(-1, keepdim=True)
            result = result + output * coefficient
        return result


def main():
    device = init_distributed()
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        for names in (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2")):
            torch.manual_seed(31)
            reference = LegacyBlock(names).to(device)
            mixed_model = nn.Module()
            mixed_model.layers = nn.ModuleList([nn.Module()])
            mixed_block = copy.deepcopy(reference)
            mixed_model.layers[0].mlp = mixed_block
            getattr(mixed_block.experts[1], names[0]).to(dtype=torch.bfloat16)
            original_parameters = tuple(mixed_model.parameters())
            try:
                swap_hf_moe_blocks(
                    mixed_model, ep_group=dist.group.WORLD, router_aux_loss_coef=0.0
                )
            except UnsupportedCombinationError as error:
                assert "different dtypes" in str(error), str(error)
            else:
                raise AssertionError(
                    "heterogeneous expert dtypes were silently stacked"
                )
            assert mixed_model.layers[0].mlp is mixed_block
            assert all(
                before is after
                for before, after in zip(
                    original_parameters, mixed_model.parameters(), strict=True
                )
            )
            for reordered in (False, True):
                shared_model = nn.Module()
                shared_model.layers = nn.ModuleList([nn.Module(), nn.Module()])
                first = copy.deepcopy(reference)
                second = copy.deepcopy(reference)
                members = list(first.experts)
                second.experts = nn.ModuleList(members[::-1] if reordered else members)
                shared_model.layers[0].mlp = first
                shared_model.layers[1].mlp = second
                try:
                    swap_hf_moe_blocks(
                        shared_model,
                        ep_group=dist.group.WORLD,
                        router_aux_loss_coef=0.0,
                    )
                except UnsupportedCombinationError as error:
                    assert reordered and "expert order" in str(error)
                    assert shared_model.layers[0].mlp is first
                    assert shared_model.layers[1].mlp is second
                else:
                    assert not reordered
                    grouped = [
                        layer.mlp.routed_experts.inner_experts
                        for layer in shared_model.layers
                    ]
                    for name in ("w1_EFD", "w3_EFD", "w2_EDF"):
                        assert getattr(grouped[0], name) is getattr(grouped[1], name)
            model = nn.Module()
            layer = nn.Module()
            layer.mlp = copy.deepcopy(reference)
            model.layers = nn.ModuleList([layer])
            original = layer.mlp
            second_projection = getattr(original.experts[1], names[0])
            untied = second_projection.weight
            second_projection.weight = getattr(original.experts[0], names[0]).weight
            try:
                swap_hf_moe_blocks(model, ep_group=dist.group.WORLD)
            except UnsupportedCombinationError as error:
                assert "shared individual expert parameter" in str(error)
            else:
                raise AssertionError("expert parameter aliases were silently copied")
            assert layer.mlp is original
            assert (
                second_projection.weight
                is getattr(original.experts[0], names[0]).weight
            )
            second_projection.weight = untied
            with torch.no_grad():
                assert (
                    swap_hf_moe_blocks(
                        model, ep_group=dist.group.WORLD, router_aux_loss_coef=0.0
                    )
                    == 1
                )
            converted = model.layers[0].mlp
            grouped = converted.routed_experts.inner_experts
            assert tuple(p.requires_grad for p in grouped.parameters()) == (
                True,
                False,
                True,
            )
            inputs = torch.randn(16, 8, device=device, requires_grad=True)
            cotangent = torch.randn_like(inputs)
            expected = reference(inputs)
            (expected * cotangent).sum().backward()
            local = inputs.detach().chunk(2)[rank].clone().requires_grad_()
            actual = converted(local)
            (actual * cotangent.chunk(2)[rank]).sum().backward()
            tolerance = {"rtol": 1e-4, "atol": 1e-5}
            torch.testing.assert_close(actual, expected.chunk(2)[rank], **tolerance)
            torch.testing.assert_close(
                local.grad, inputs.grad.chunk(2)[rank], **tolerance
            )
            for name, parameter in zip(names, grouped.parameters(), strict=True):
                if parameter.requires_grad:
                    expected_grad = torch.stack(
                        [getattr(e, name).weight.grad for e in reference.experts]
                    ).chunk(2)[rank]
                    assert torch.count_nonzero(expected_grad) > 0
                    torch.testing.assert_close(
                        parameter.grad, expected_grad, **tolerance
                    )
                else:
                    assert parameter.grad is None
            print(
                f"PASS rank={rank} device={device} legacy={names}: "
                "freezing and gradients"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
