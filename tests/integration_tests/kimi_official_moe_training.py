"""Official-source MoE adapter validation; requires an explicit importable module.

KIMI_REFERENCE_MODULE must identify the reviewed official source module. This
test does not substitute local architecture classes or exercise KDA kernels.
"""

import os
from copy import deepcopy
from importlib import import_module
from itertools import product

import torch

from llmtuner.models.hf.kimi_training import forward_kimi_training


def main():
    official = import_module(os.environ["KIMI_REFERENCE_MODULE"])
    for dtype, latent, activation in product(
        (torch.float32, torch.bfloat16), (False, True), ("silu", "situ")
    ):
        torch.manual_seed(19)
        config = official.KimiLinearConfig(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=3,
            moe_intermediate_size=12,
            num_shared_experts=1,
            routed_expert_hidden_size=4 if latent else None,
            latent_moe_use_norm=latent,
            hidden_act=activation,
        )
        module = official.KimiSparseMoeBlock(config).to(dtype=dtype)
        module.gate.e_score_correction_bias.data.zero_()
        reference = deepcopy(module)
        inputs = torch.randn(1, 31, 8, dtype=dtype, requires_grad=True)
        other = inputs.detach().clone().requires_grad_()
        keys = tuple(module.state_dict())
        actual = forward_kimi_training(module, inputs)
        module.eval()
        with torch.no_grad():
            expected_inference = module(inputs)
        torch.testing.assert_close(actual, expected_inference, atol=0, rtol=0)
        ids, scores = reference.gate(other)
        flat = other.reshape(-1, 8)
        if latent:
            flat = reference.routed_expert_down_proj(flat)
        # All-expert enumeration gives an independent differentiable reference.
        all_outputs = torch.stack([expert(flat) for expert in reference.experts], 1)
        selected = all_outputs.gather(
            1, ids.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
        )
        expected = (selected.to(scores.dtype) * scores.unsqueeze(-1)).sum(1).to(dtype)
        if latent:
            expected = reference.routed_expert_up_proj(
                reference.routed_expert_norm(expected)
            )
        expected = expected.reshape_as(other) + reference.shared_experts(other)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        dy = torch.randn_like(actual)
        actual.backward(dy)
        expected.backward(dy)
        torch.testing.assert_close(inputs.grad, other.grad)
        for (name, param), (_, ref) in zip(
            module.named_parameters(), reference.named_parameters(), strict=True
        ):
            if ref.grad is not None:
                torch.testing.assert_close(param.grad, ref.grad, msg=name)
        assert torch.count_nonzero(module.gate.weight.grad) > 0
        assert tuple(module.state_dict()) == keys
        print(f"PASS official MoE dtype={dtype} latent={latent} act={activation}")


if __name__ == "__main__":
    main()
