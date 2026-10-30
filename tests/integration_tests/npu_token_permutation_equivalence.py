"""Check large BF16 token-permutation VJPs against the baseline NPU kernel."""

import torch

from llmtuner.parallel.expert_parallel.npu_baseline_experts import (
    NpuBaselineAllToAllTokenDispatcher,
)
from tests.caps import require_env


def main():
    require_env("torch_npu")
    import torch_npu

    torch.npu.set_device(0)
    torch.manual_seed(91)
    torch.use_deterministic_algorithms(True)
    dispatcher = NpuBaselineAllToAllTokenDispatcher(128, 8)
    for tokens, width in ((64, 128), (4096, 2048), (0, 2048)):
        inputs = torch.randn(
            tokens, width, device="npu", dtype=torch.bfloat16, requires_grad=True
        )
        ids = torch.rand(tokens, 128, device="npu").topk(8, dim=1).indices
        scores = torch.rand(tokens, 8, device="npu")
        actual, indices, _ = dispatcher._local_reorder(inputs, scores, ids)
        if tokens:
            expected, _ = torch_npu.npu_moe_token_permute(inputs, ids)
        else:
            expected = inputs[indices]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        upstream = torch.randn_like(actual) * 1e-5
        actual_grad = torch.autograd.grad(actual, inputs, upstream)[0]
        expected_grad = torch.autograd.grad(expected, inputs, upstream)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
        print(f"PASS tokens={tokens} width={width}: permutation and VJP exact")


if __name__ == "__main__":
    main()
