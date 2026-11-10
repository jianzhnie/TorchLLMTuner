"""Official-source MoE adapter validation; requires an explicit importable module.

KIMI_REFERENCE_MODULE must identify the reviewed official source module. This
test does not substitute local architecture classes or exercise KDA kernels.
"""

import os
from copy import deepcopy
from importlib import import_module
from itertools import product

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.kimi.kimi_k3.official_moe_training import (
    forward_kimi_training,
    install_kimi_moe_training,
)


def main():
    device = torch.device(os.environ.get("KIMI_TEST_DEVICE", "cpu"))
    if device.type == "npu":
        if torch_npu is None or not torch.npu.is_available():
            raise RuntimeError("KIMI_TEST_DEVICE=npu requires available torch-npu")
        torch.npu.set_device(device)
    elif device.type != "cpu":
        raise ValueError("KIMI_TEST_DEVICE must select CPU or NPU")
    official = import_module(os.environ["KIMI_REFERENCE_MODULE"])
    for dtype, latent, activation, grouped in product(
        (torch.float32, torch.bfloat16), (False, True), ("silu", "situ"), (False, True)
    ):
        torch.manual_seed(19)
        config = official.KimiLinearConfig(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=2 if grouped else 3,
            num_expert_group=2 if grouped else 1,
            topk_group=1,
            moe_intermediate_size=12,
            num_shared_experts=1,
            routed_expert_hidden_size=4 if latent else None,
            latent_moe_use_norm=latent,
            hidden_act=activation,
        )
        module = official.KimiSparseMoeBlock(config).to(device=device, dtype=dtype)
        module.gate.e_score_correction_bias.data.zero_()
        reference = deepcopy(module)
        reference.eval()
        inputs = torch.randn(1, 31, 8, dtype=dtype).to(device).requires_grad_()
        other = inputs.detach().clone().requires_grad_()
        keys = tuple(module.state_dict())
        with torch.no_grad():
            expected_inference = reference(inputs)
        original_parameters = tuple(module.parameters())
        assert install_kimi_moe_training(module, moe_type=type(module)) == 1
        assert install_kimi_moe_training(module, moe_type=type(module)) == 1
        assert all(
            before is after
            for before, after in zip(
                original_parameters, module.parameters(), strict=True
            )
        )
        actual = module(inputs)
        # Official sorting may reorder GEMM rows. Kernels may round those rows
        # differently; BF16 tolerance is explicit, not a bitwise claim.
        tolerance = 5e-3 if dtype == torch.bfloat16 else 1e-6
        torch.testing.assert_close(
            actual, expected_inference, atol=tolerance, rtol=tolerance
        )
        ids, scores = reference.gate(other)
        flat = other.reshape(-1, 8)
        if latent:
            flat = reference.routed_expert_down_proj(flat)
        # Keep expert GEMM batches identical: unused rows otherwise change
        # BF16 weight-gradient rounding, especially through latent projections.
        # Assemble expert-major outputs then independently invert the assignment.
        assignment = ids.flatten().argsort(stable=True)
        counts = torch.bincount(ids.flatten(), minlength=config.num_experts)
        expert_outputs = []
        start = 0
        for expert, count in zip(reference.experts, counts.tolist(), strict=True):
            token_ids = assignment[start : start + count] // ids.shape[1]
            expert_outputs.append(expert(flat.index_select(0, token_ids)))
            start += count
        sorted_outputs = torch.cat(expert_outputs)
        selected = sorted_outputs.index_select(0, assignment.argsort()).reshape(
            *ids.shape, flat.shape[-1]
        )
        expected = (selected.to(scores.dtype) * scores.unsqueeze(-1)).sum(1).to(dtype)
        if latent:
            expected = reference.routed_expert_up_proj(
                reference.routed_expert_norm(expected)
            )
        expected = expected.reshape_as(other) + reference.shared_experts(other)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        dy = torch.randn_like(actual)
        actual.backward(dy)
        expected.backward(dy)
        torch.testing.assert_close(
            inputs.grad, other.grad, atol=tolerance, rtol=tolerance
        )
        for (name, param), (_, ref) in zip(
            module.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert (param.grad is None) == (ref.grad is None), name
            if ref.grad is not None:
                assert torch.isfinite(param.grad).all(), name
                torch.testing.assert_close(
                    param.grad, ref.grad, atol=tolerance, rtol=tolerance, msg=name
                )
        assert torch.count_nonzero(module.gate.weight.grad) > 0
        assert tuple(module.state_dict()) == keys
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, foreach=False)
        ref_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1e-3, foreach=False
        )
        optimizer.step()
        ref_optimizer.step()
        for param, ref in zip(module.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(param, ref, atol=tolerance, rtol=tolerance)
        print(
            f"PASS official MoE device={device} dtype={dtype} latent={latent} "
            f"act={activation} grouped={grouped} backward/AdamW"
        )
        # Empty batches must preserve shape and a differentiable path without
        # accidentally routing tokens or manufacturing parameter updates.
        empty = inputs.detach()[:, :0].requires_grad_()
        empty_output = forward_kimi_training(module, empty)
        assert empty_output.shape == empty.shape
        empty_output.sum().backward()
        assert empty.grad is not None
        module.ep_size = 2
        try:
            forward_kimi_training(module, inputs.detach())
        except UnsupportedCombinationError:
            pass
        else:
            raise AssertionError("Sharded official experts must be rejected")
        untouched = deepcopy(reference)
        invalid = deepcopy(reference)
        invalid.ep_size = 2
        container = torch.nn.ModuleList([untouched, invalid])
        original_forward = untouched.forward.__func__
        try:
            install_kimi_moe_training(container, moe_type=type(reference))
        except UnsupportedCombinationError:
            assert untouched.forward.__func__ is original_forward
        else:
            raise AssertionError("Installation must reject partial expert ownership")


if __name__ == "__main__":
    main()
