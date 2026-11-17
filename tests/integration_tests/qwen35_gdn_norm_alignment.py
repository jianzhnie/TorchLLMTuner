"""Compare production gated norm outputs and VJPs with installed Transformers.

The 93-token case zeros output cotangents after token 93; it does not validate
attention masks or full GDN execution. Run explicitly on CPU or an idle NPU.
"""

import argparse

import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNormGated as ReferenceGatedNorm,
)

from llmtuner.models.qwen.qwen3_5_moe.gated_delta import Qwen3_5MoeRMSNormGated


def run(device: torch.device) -> None:
    generator = torch.Generator().manual_seed(42)
    shape = (1, 128, 32, 128)
    for dtype in (torch.float32, torch.bfloat16):
        for valid_tokens in (93, 128):
            reference = ReferenceGatedNorm(128).to(device=device, dtype=dtype)
            actual = Qwen3_5MoeRMSNormGated(128).to(device=device, dtype=dtype)
            with torch.no_grad():
                reference.weight.copy_(torch.linspace(0.7, 1.4, 128))
                actual.load_state_dict(reference.state_dict(), strict=True)
            inputs = [
                torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
                for _ in range(3)
            ]
            inputs[2][:, valid_tokens:] = 0
            results = []
            for module in (reference, actual):
                x, gate = [
                    value.detach().clone().requires_grad_() for value in inputs[:2]
                ]
                output = module(x, gate)
                gradients = torch.autograd.grad(
                    output, (x, gate, module.weight), inputs[2]
                )
                results.append((output.detach(), *gradients))
            for name, expected, observed in zip(
                ("output", "dx", "dgate", "dweight"), *results, strict=True
            ):
                assert torch.isfinite(observed).all(), name
                assert observed.count_nonzero(), name
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            print(f"PASS device={device} dtype={dtype} valid_tokens={valid_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or npu:<index>")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    run(device)


if __name__ == "__main__":
    main()
