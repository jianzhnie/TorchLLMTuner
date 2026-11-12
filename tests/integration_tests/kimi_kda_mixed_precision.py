"""Two-rank NPU KDA: BF16 projections, FP32 decay, FSDP forward/backward/update.

Run with torchrun --nproc_per_node=2; requires Triton-Ascend KDA kernels.
This small-model regression is not official K3 numerical acceptance.
"""

import os
from argparse import ArgumentParser
from copy import deepcopy
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.memory.activation_checkpoint import apply_ac
from llmtuner.models.configuration import build_model_config, resolve_model_class
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    apply_fsdp_to_decoder,
)
from llmtuner.parallel.fully_shard_parallel.policies import (
    disable_fsdp_gradient_division,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--ac", choices=("none", "full"), default="full")
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(42)
        torch.set_default_dtype(torch.bfloat16)
        config = build_model_config(
            "kimi_linear",
            seq_len=128,
            arch_overrides={
                "vocab_size": 128,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "linear_attn_config": {
                    "kda_layers": [1],
                    "full_attn_layers": [],
                    "head_dim": 64,
                    "num_heads": 2,
                    "short_conv_kernel_size": 4,
                },
            },
        )
        config.attn_mask_type = "causal"
        model = resolve_model_class(config)(config).to("npu")
        model.initialize_parameters()
        reference = deepcopy(model)
        apply_ac(model, args.ac)
        names = set(model.state_dict())
        mesh = init_device_mesh("npu", (dist.get_world_size(),))
        apply_fsdp_to_decoder(
            model,
            mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )
        disable_fsdp_gradient_division(model)
        assert set(model.state_dict()) == names
        decay_names = set()
        for name, parameter in model.named_parameters():
            if name.endswith((".A_log", ".dt_bias")):
                decay_names.add(name)
                assert parameter.dtype == torch.float32
        assert decay_names, "The test must exercise mixed-precision KDA parameters"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        ref_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1e-3, foreach=False
        )
        for step in range(2):
            tokens = (torch.arange(128, device="npu") + dist.get_rank() + step) % 128
            positions = torch.arange(128, device="npu")
            actual = model(tokens, positions=positions).logits
            expected = reference(tokens, positions=positions).logits
            if dist.get_rank() == 0:
                max_abs = (actual - expected).abs().max().item()
                print(
                    f"step={step} logits_max_abs={max_abs}",
                    flush=True,
                )
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.02)
            actual.float().square().mean().backward()
            expected.float().square().mean().backward()
            for parameter in reference.parameters():
                # The framework disables FSDP gradient division: the trainer
                # owns loss normalization. Match its FP32 SUM contract here.
                reduced = parameter.grad.float()
                dist.all_reduce(reduced)
                parameter.grad.copy_(reduced)
            pairs = list(
                zip(model.named_parameters(), reference.named_parameters(), strict=True)
            )
            # Check decay parameters and internal projections before embeddings
            # so a failure reports which part of the backward chain diverged.
            pairs.sort(key=lambda pair: "embed_tokens" in pair[0][0])
            for (name, parameter), (ref_name, ref) in pairs:
                assert name.replace("_checkpoint_wrapped_module.", "") == ref_name
                grad = parameter.grad.full_tensor()
                assert torch.isfinite(grad).all()
                if name.endswith((".A_log", ".dt_bias")):
                    assert torch.count_nonzero(grad) > 0
                if dist.get_rank() == 0:
                    print(
                        f"step={step} {name} grad_max_abs="
                        f"{(grad - ref.grad).abs().max().item()}",
                        flush=True,
                    )
                torch.testing.assert_close(
                    grad,
                    ref.grad,
                    atol=0.003,
                    rtol=0.03,
                    msg=lambda detail, name=name: f"{name}: {detail}",
                )
            optimizer.step()
            ref_optimizer.step()
            for name, parameter in model.named_parameters():
                if name.endswith((".A_log", ".dt_bias")):
                    assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
            for parameter, ref in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    parameter.full_tensor(), ref, atol=0.003, rtol=0.02
                )
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
        if dist.get_rank() == 0:
            print(
                "PASS: mixed-dtype KDA FSDP, gradients, two AdamW updates, stable FQNs"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
