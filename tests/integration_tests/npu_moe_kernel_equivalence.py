"""Check local BF16 experts against the active NPU baseline kernel formula.

Run explicitly with Python on an idle NPU. This is a kernel contract check,
not a full Trainer or distributed numerical acceptance test.
"""

from types import SimpleNamespace

from tests.caps import require_env

require_env("torch_npu")

import torch
import torch_npu

from llmtuner.models.hf.model_setup import forward_torchtitan_moe
from llmtuner.parallel.expert_parallel.token_dispatch import LocalTokenDispatcher


def main():
    torch.npu.set_device(0)
    torch.manual_seed(37)
    experts, top_k, width, intermediate = 4, 2, 32, 32
    inputs = torch.randn(2, 4, width, device="npu", dtype=torch.bfloat16)
    inputs.requires_grad_()
    gate_up = torch.randn(
        experts, intermediate * 2, width, device="npu", dtype=torch.bfloat16
    ).requires_grad_()
    down = torch.randn(
        experts, width, intermediate, device="npu", dtype=torch.bfloat16
    ).requires_grad_()
    scores = torch.rand(8, top_k, device="npu", requires_grad=True)
    indices = torch.tensor(
        [[0, 1], [2, 3], [1, 3], [0, 2]] * 2, device="npu", dtype=torch.int64
    )
    module = SimpleNamespace(
        gate=lambda hidden: (None, scores, indices),
        experts=SimpleNamespace(
            num_experts=experts,
            gate_up_proj=gate_up,
            down_proj=down,
            act_fn=torch.nn.functional.silu,
        ),
        _llmtuner_local_dispatcher=LocalTokenDispatcher(experts, top_k),
    )
    observed = forward_torchtitan_moe(module, inputs)
    routed, permutation = torch_npu.npu_moe_token_permute(
        inputs.reshape(-1, width), indices
    )
    routed_scores, _ = torch_npu.npu_moe_token_permute(
        scores.reshape(-1, 1), indices.reshape(-1, 1)
    )
    counts = torch.nn.functional.one_hot(indices, num_classes=experts).sum((0, 1))
    offsets = counts.cumsum(0, dtype=torch.int32)
    projected = torch._grouped_mm(routed, gate_up.transpose(-2, -1), offs=offsets)
    hidden = torch_npu.npu_swiglu(projected, dim=-1)
    hidden = hidden * routed_scores.to(hidden.dtype)
    output = torch._grouped_mm(hidden, down.transpose(-2, -1), offs=offsets)
    reference = torch_npu.npu_moe_token_unpermute(output, permutation, None)
    reference = reference.view(8, top_k, width).sum(1).reshape_as(observed)
    torch.testing.assert_close(observed, reference, atol=0, rtol=0)
    parameters = inputs, gate_up, down, scores
    probe = torch.randn_like(observed)
    actual_gradients = torch.autograd.grad(
        observed, parameters, probe, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(reference, parameters, probe)
    for name, actual, expected in zip(
        ("input", "gate_up", "down", "scores"),
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=name)
    print("PASS BF16 local MoE NPU output and all input/parameter/score gradients")


if __name__ == "__main__":
    main()
