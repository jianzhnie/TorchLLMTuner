"""Compare local Kimi components with real K3 and Transformers modules.

KIMI_REFERENCE_MODULE selects an explicitly reviewed official K3 source module.
KIMI_TEST_DEVICE defaults to cpu; npu requires torch-npu and idle hardware.
This is component acceptance, not full-model or throughput acceptance.
"""

import os
from functools import partial
from importlib import import_module
from itertools import product

import torch
from transformers.models.kimi_linear.configuration_kimi_linear import (
    KimiLinearConfig as NativeConfig,
)
from transformers.models.kimi_linear.modeling_kimi_linear import (
    KimiLinearDeltaAttention as NativeDeltaAttention,
)
from transformers.models.kimi_linear.modeling_kimi_linear import (
    KimiLinearRMSNormGated,
    chunk_kimi_delta_attention,
)

try:
    import torch_npu
except ImportError:
    torch_npu = None

from llmtuner.models.kimi.kimi_k3.attention_residual import attention_residual
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.delta_attention import (
    KimiDeltaAttention,
    recurrent_kda,
)
from llmtuner.models.kimi.kimi_k3.latent_attention import KimiLatentAttention
from llmtuner.models.kimi.kimi_k3.moe import KimiSparseMoE
from llmtuner.models.kimi.kimi_k3.normalization import KimiGatedRMSNorm
from llmtuner.models.kimi.kimi_k3.official_moe_training import forward_kimi_training


def check_native_kda():
    """FP32 CPU reference comparison for the shared unbounded/low-rank subset."""
    torch.manual_seed(19)
    config = KimiLinearConfig(
        hidden_size=8,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    native_config = NativeConfig(
        hidden_size=8,
        num_hidden_layers=1,
        linear_num_heads=2,
        linear_head_dim=4,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
        rms_norm_eps=config.rms_norm_eps,
    )
    local = KimiDeltaAttention(config, 0)
    reference = NativeDeltaAttention(native_config, 0)
    state = local.state_dict()
    mapped = {}
    for name, value in state.items():
        if "_conv1d." in name:
            continue
        target = (
            "forget_gate." + name
            if name in {"A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight"}
            else name
        )
        mapped[target] = value.reshape(1, 1, 2, 1) if name == "A_log" else value
    mapped["conv1d.weight"] = torch.cat(
        [state[name + "_conv1d.weight"] for name in ("q", "k", "v")]
    )
    reference.load_state_dict(mapped, strict=True)
    inputs = torch.randn(1, 7, 8, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    actual, expected = local(inputs), reference(other)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(inputs.grad, other.grad, atol=3e-6, rtol=3e-6)
    ref_params = dict(reference.named_parameters())
    for name, parameter in local.named_parameters():
        if "_conv1d." in name:
            index = ("q", "k", "v").index(name[0])
            gradient = ref_params["conv1d.weight"].grad.chunk(3)[index]
        else:
            target = (
                "forget_gate." + name if "forget_gate." + name in ref_params else name
            )
            gradient = ref_params[target].grad.reshape_as(parameter)
        torch.testing.assert_close(
            parameter.grad, gradient, atol=3e-6, rtol=3e-6, msg=name
        )
    print("PASS native KDA FP32 CPU shared subset", flush=True)


def check_native_kda_precision_boundaries():
    """Isolate K3 versus native BF16 rounding using the real HF CPU reference.

    Feeding rounded Q/K and FP32 beta to the HF core is a controlled experiment,
    not acceptance of its unmodified BF16 module or the K3 NPU kernel.
    """
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(19)
        args = [
            torch.randn(1, 7, 2, 4, dtype=dtype, requires_grad=True) for _ in range(4)
        ]
        beta = torch.randn(1, 7, 2, dtype=dtype, requires_grad=True)
        rate = torch.randn(2, requires_grad=True)
        bias = torch.randn(8, requires_grad=True)
        query, key, value, raw = args
        all_args = [*args, beta, rate, bias]
        decay = -rate.exp().view(1, 1, 2, 1) * torch.nn.functional.softplus(
            raw.float() + bias.view(2, 4)
        )
        actual = recurrent_kda(*all_args)
        upstream = torch.ones_like(actual)
        gradients = torch.autograd.grad(actual, all_args, upstream, retain_graph=True)
        for rounded_qk, fp32_beta in product((False, True), repeat=2):
            queries, keys = query, key
            if rounded_qk:
                queries, keys = [
                    (
                        tensor.float()
                        * (tensor.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt()
                    ).to(dtype)
                    for tensor in (query, key)
                ]
            expected, _ = chunk_kimi_delta_attention(
                queries,
                keys,
                value,
                decay,
                beta.float().sigmoid() if fp32_beta else beta.sigmoid(),
                use_qk_l2norm_in_kernel=not rounded_qk,
            )
            refs = torch.autograd.grad(expected, all_args, upstream, retain_graph=True)
            output_diff = (actual - expected).abs().max().item()
            gradient_diff = max(
                (left - right).abs().max().item()
                for left, right in zip(gradients, refs, strict=True)
            )
            if rounded_qk and fp32_beta:
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
                for left, right in zip(gradients, refs, strict=True):
                    assert torch.isfinite(left).all()
                    torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-6)
            print(
                f"KDA precision {dtype} rounded_qk={rounded_qk} "
                f"fp32_beta={fp32_beta} output_max={output_diff} "
                f"gradient_max={gradient_diff}",
                flush=True,
            )


def compare(local, reference, inputs, local_forward=None, ref_forward=None):
    """Load identical tensors and compare outputs, input and named gradients."""
    reference.load_state_dict(local.state_dict(), strict=True)
    other = inputs.detach().clone().requires_grad_()
    actual = local(inputs) if local_forward is None else local_forward(inputs)
    expected = reference(other) if ref_forward is None else ref_forward(other)
    tolerance = 5e-3 if inputs.dtype == torch.bfloat16 else 3e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(inputs.grad, other.grad, atol=tolerance, rtol=tolerance)
    refs = dict(reference.named_parameters())
    for name, parameter in local.named_parameters():
        ref = refs[name]
        assert (parameter.grad is None) == (ref.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(
                parameter.grad, ref.grad, atol=tolerance, rtol=tolerance, msg=name
            )


def main():
    official = import_module(os.environ["KIMI_REFERENCE_MODULE"])
    device = torch.device(os.environ.get("KIMI_TEST_DEVICE", "cpu"))
    if device.type == "npu":
        if torch_npu is None or not torch.npu.is_available():
            raise RuntimeError("NPU tests require available torch-npu")
        torch.npu.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Only CPU/NPU accepted")
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(42)
        local = KimiGatedRMSNorm(16).to(device=device, dtype=dtype)
        reference = KimiLinearRMSNormGated(16).to(device=device, dtype=dtype)
        with torch.no_grad():
            local.weight.copy_(torch.randn_like(local.weight))
        gate = torch.randn(2, 5, 3, 16, dtype=dtype, device=device, requires_grad=True)
        other_gate = gate.detach().clone().requires_grad_()
        inputs = torch.randn_like(gate).requires_grad_()
        compare(
            local,
            reference,
            inputs,
            partial(local, gate=gate),
            partial(reference, gate=other_gate),
        )
        torch.testing.assert_close(gate.grad, other_gate.grad, atol=0, rtol=0)
        print(f"PASS native gated RMSNorm {device} {dtype}", flush=True)

    for dtype, latent, activation, grouped in product(
        (torch.float32, torch.bfloat16), (False, True), ("silu", "situ"), (False, True)
    ):
        torch.manual_seed(19)
        settings = dict(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=2,
            num_expert_group=2 if grouped else 1,
            topk_group=1,
            moe_intermediate_size=12,
            num_shared_experts=1,
            routed_expert_hidden_size=4 if latent else None,
            latent_moe_use_norm=latent,
            hidden_act=activation,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
        )
        local = KimiSparseMoE(KimiLinearConfig(**settings)).to(
            device=device, dtype=dtype
        )
        reference = official.KimiSparseMoeBlock(
            official.KimiLinearConfig(**settings)
        ).to(device=device, dtype=dtype)
        inputs = torch.randn(1, 31, 8, dtype=dtype, device=device, requires_grad=True)
        compare(
            local,
            reference,
            inputs,
            ref_forward=partial(forward_kimi_training, reference),
        )
        print(
            f"PASS K3 MoE {device} {dtype} latent={latent} "
            f"{activation} grouped={grouped}",
            flush=True,
        )

    for query_rank, output_gate in product((None, 4), (False, True)):
        settings = dict(
            hidden_size=12,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=query_rank,
            kv_lora_rank=4,
            qk_nope_head_dim=3,
            qk_rope_head_dim=1,
            v_head_dim=3,
            mla_use_nope=True,
            mla_use_output_gate=output_gate,
        )
        config = KimiLinearConfig(**settings)
        ref_config = official.KimiLinearConfig(**settings)
        config._attn_implementation = ref_config._attn_implementation = "eager"
        local = KimiLatentAttention(config, 0).to(device)
        reference = official.KimiMLAAttention(ref_config, 0).to(device)
        inputs = torch.randn(1, 5, 12, device=device, requires_grad=True)
        mask = torch.full((1, 1, 5, 5), float("-inf"), device=device).triu(1)
        compare(
            local,
            reference,
            inputs,
            partial(local, attention_mask=mask),
            partial(reference, attention_mask=mask),
        )
        print(f"PASS K3 MLA rank={query_rank} gate={output_gate}", flush=True)

    norm = official.KimiRMSNorm(12).to(device)
    proj = torch.nn.Linear(12, 1, bias=False).to(device)
    prefix = torch.randn(6, 12, device=device, requires_grad=True)
    blocks = torch.randn(6, 3, 12, device=device, requires_grad=True)
    actual = attention_residual(
        prefix, blocks, proj.weight, norm.weight, norm.variance_epsilon
    )
    expected = official._apply_attn_res(prefix, blocks, proj, norm)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    values = (prefix, blocks, proj.weight, norm.weight)
    upstream = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, values, upstream)
    refs = torch.autograd.grad(expected, values, upstream)
    for gradient, ref in zip(grads, refs, strict=True):
        torch.testing.assert_close(gradient, ref, atol=0, rtol=0)
    print("PASS K3 attention residual", flush=True)


if __name__ == "__main__":
    if os.environ.get("KIMI_NATIVE_KDA_ONLY") == "1":
        # Run in a separate environment without FLA: the native HF reference
        # fallback is differentiable CPU PyTorch, while FLA requires a device.
        check_native_kda()
        check_native_kda_precision_boundaries()
    else:
        main()
