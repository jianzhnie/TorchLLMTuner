"""Compare NPU and advanced-index MoE permutation backward on repeated tokens."""

import argparse
import json
from pathlib import Path

import torch

from tests.caps import require_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_env("torch_npu")
    import torch_npu

    torch.npu.set_device(0)
    torch.use_deterministic_algorithms(True)
    rows = []
    for topk in (2, 8):
        torch.manual_seed(91)
        values = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
        ids = torch.rand(64, 16, device="npu").topk(topk, dim=1).indices
        order = torch.argsort(ids.flatten(), stable=True)
        native_input = values.clone().requires_grad_()
        indexed_input = values.clone().requires_grad_()
        native, _ = torch_npu.npu_moe_token_permute(native_input, ids)
        indexed = indexed_input[order // topk]
        torch.testing.assert_close(indexed, native, atol=0, rtol=0)
        probe = torch.randn_like(native)
        native_grad = torch.autograd.grad(native, native_input, probe)[0]
        indexed_grad = torch.autograd.grad(indexed, indexed_input, probe)[0]
        delta = (native_grad.float() - indexed_grad.float()).abs()
        rows.append(
            {
                "topk": topk,
                "forward_exact": True,
                "backward_exact": torch.equal(native_grad, indexed_grad),
                "backward_max_ae": delta.max().item(),
                "different": (native_grad != indexed_grad).sum().item(),
            }
        )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
