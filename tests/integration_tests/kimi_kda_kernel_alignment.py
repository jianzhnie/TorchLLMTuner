"""Real Ascend KDA core versus K3-precision CPU recurrence.

This component test does not establish full-model or distributed acceptance.
Run on an idle NPU with the optional Triton-Ascend KDA package installed.
"""

from itertools import product

import torch
import torch_npu  # noqa: F401

from llmtuner.kernel.ops.kda import npu_chunk_kda
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.delta_attention import (
    KimiDeltaAttention,
    recurrent_kda,
)


def check_module():
    """Include projections, convolution, full-rank gate and corrected norm."""
    for full_rank, lower_bound in product((False, True), (None, -5.0)):
        torch.manual_seed(19)
        config = KimiLinearConfig(
            hidden_size=128,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [],
                "head_dim": 64,
                "num_heads": 2,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": full_rank,
                "gate_lower_bound": lower_bound,
            },
        )
        local = KimiDeltaAttention(config, 0).to("npu", dtype=torch.bfloat16)
        for name in ("A_log", "dt_bias"):
            parameter = getattr(local, name)
            parameter.data = parameter.data.float()
        reference = KimiDeltaAttention(config, 0).to(dtype=torch.bfloat16)
        for name in ("A_log", "dt_bias"):
            parameter = getattr(reference, name)
            parameter.data = parameter.data.float()
        reference.load_state_dict(local.state_dict())
        inputs = torch.randn(1, 17, 128, dtype=torch.bfloat16, requires_grad=True)
        other = inputs.detach().to("npu").requires_grad_()
        expected, actual = reference(inputs), local(other)
        torch.testing.assert_close(actual.cpu(), expected, atol=5e-3, rtol=2e-2)
        upstream = torch.randn_like(expected)
        actual.backward(upstream.to("npu"))
        expected.backward(upstream)
        torch.testing.assert_close(other.grad.cpu(), inputs.grad, atol=2e-2, rtol=5e-2)
        for (name, parameter), (ref_name, ref) in zip(
            local.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == ref_name
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            difference = (parameter.grad.cpu() - ref.grad).abs()
            allowed = 2e-2 + 5e-2 * ref.grad.abs()
            print(
                f"module full_rank={full_rank} lower_bound={lower_bound} "
                f"parameter={name} gradient_max_abs={difference.max().item()} "
                f"outside_tolerance={(difference > allowed).sum().item()}",
                flush=True,
            )
            torch.testing.assert_close(
                parameter.grad.cpu(), ref.grad, atol=2e-2, rtol=5e-2, msg=name
            )
        print(
            f"PASS NPU KDA module full_rank={full_rank} lower_bound={lower_bound}",
            flush=True,
        )


def main():
    if npu_chunk_kda is None:
        raise RuntimeError("Triton-Ascend KDA is required; no fallback permitted")
    torch.npu.set_device(0)
    # Predeclared BF16 kernel tolerances, separately for outputs and gradients.
    for length, lower_bound in product((17, 128), (None, -5.0)):
        torch.manual_seed(19)
        shape = (1, length, 2, 64)
        values = [torch.randn(shape, dtype=torch.bfloat16) * 0.2 for _ in range(4)]
        values += [
            torch.randn(1, length, 2, dtype=torch.float32),
            torch.empty(2).uniform_(1, 16).log(),
            torch.zeros(128),
        ]
        device_args = [value.to("npu").requires_grad_() for value in values]
        cpu_args = [value.clone().requires_grad_() for value in values]
        q, k, v, g, beta, rate, bias = device_args
        actual, _ = npu_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta.sigmoid(),
            A_log=rate,
            dt_bias=bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=lower_bound is not None,
            lower_bound=lower_bound,
            transpose_state_layout=False,
        )
        expected = recurrent_kda(*cpu_args, lower_bound=lower_bound)
        assert actual.device.type == "npu"
        assert torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual.cpu(), expected, atol=5e-3, rtol=2e-2)
        upstream = torch.randn_like(expected)
        actual.backward(upstream.to("npu"))
        expected.backward(upstream)
        for name, param, reference in zip(
            ("q", "k", "v", "g", "beta", "A_log", "dt_bias"),
            device_args,
            cpu_args,
            strict=True,
        ):
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert torch.count_nonzero(param.grad) > 0, name
            difference = (param.grad.cpu() - reference.grad).abs().max().item()
            print(
                f"length={length} lower_bound={lower_bound} "
                f"gradient={name} max_abs={difference}",
                flush=True,
            )
            torch.testing.assert_close(
                param.grad.cpu(), reference.grad, atol=2e-2, rtol=5e-2, msg=name
            )
        print(f"PASS NPU KDA length={length} lower_bound={lower_bound}", flush=True)
    check_module()


if __name__ == "__main__":
    main()
