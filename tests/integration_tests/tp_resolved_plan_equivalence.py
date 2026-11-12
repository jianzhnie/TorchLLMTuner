"""Compare direct TP with pre-resolved TP using real distributed kernels.

Run explicitly with torchrun --standalone --nproc_per_node=2. The resolver
guard forbids a second search; it does not replace communication or compute.
"""

import argparse
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as ReferenceQwen35Moe,
)
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as ReferenceQwen3Moe,
)

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp, resolve_tp_plan
from llmtuner.parallel.topology import ParallelTopology, build_device_mesh
from tests.integration_tests.harness import init_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-mode", choices=("causal", "padding", "packed"), default="causal"
    )
    parser.add_argument("--model", choices=("qwen3", "qwen35"), default="qwen3")
    args = parser.parse_args()
    device = init_distributed()
    try:
        world = dist.get_world_size()
        cfg = ParallelConfig(tensor_parallel_size=world, data_parallel_shard_size=1)
        topology = ParallelTopology.from_config(cfg, world)
        mesh = build_device_mesh(topology)
        torch.manual_seed(1729)
        dtype = torch.bfloat16 if device.type == "npu" else torch.float32
        if args.model == "qwen3":
            config = Qwen3MoeConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                moe_intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                num_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=64,
                router_aux_loss_coef=0.0,
                attention_dropout=0.0,
                experts_implementation="eager",
                attn_implementation="eager",
            )
            model_type, reference_type = Qwen3MoeForCausalLM, ReferenceQwen3Moe
        else:
            config = Qwen3_5MoeTextConfig(
                vocab_size=64,
                hidden_size=32,
                num_hidden_layers=2,
                layer_types=["linear_attention", "full_attention"],
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=128,
                linear_num_key_heads=1,
                linear_num_value_heads=2,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                num_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=16,
                shared_expert_intermediate_size=16,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.5,
                    "mrope_section": [11, 11, 10],
                },
                experts_implementation="eager",
                attn_implementation="eager",
            )
            model_type, reference_type = Qwen3_5MoeForCausalLM, ReferenceQwen35Moe
        config.attn_mask_type = (
            "block_causal" if args.input_mode == "packed" else "causal"
        )
        direct = model_type(config).to(device=device, dtype=dtype)
        planned = model_type(config).to(device=device, dtype=dtype)
        planned.load_state_dict(direct.state_dict(), strict=True)
        reference = reference_type(config).to(device=device, dtype=dtype)
        reference.load_state_dict(direct.state_dict(), strict=True)
        with synchronize_local_errors("TP preflight equivalence"):
            resolved = resolve_tp_plan(planned, cfg=cfg)
        direct = apply_tp(direct, mesh, cfg)
        replacements = []
        with (
            patch(
                "llmtuner.parallel.tensor_parallel.tensor_parallel.resolve_tp_plan",
                side_effect=AssertionError("Executor must consume the resolved plan"),
            ),
            patch(
                "llmtuner.parallel.tensor_parallel.tensor_parallel.looks_like_attention",
                side_effect=AssertionError("Executor must not inspect attention"),
            ),
            patch(
                "llmtuner.parallel.tensor_parallel.tensor_parallel._resolve_shared_expert",
                side_effect=AssertionError("Executor must use bound shared experts"),
            ),
        ):
            planned = apply_tp(
                planned,
                mesh,
                cfg,
                resolved_plan=resolved,
                on_replacement=lambda modules, parameters: replacements.append(
                    (modules, parameters)
                ),
            )
        tokens = torch.arange(8, device=device).reshape(1, -1)
        positions = torch.arange(8, device=device).reshape(1, -1)
        batch = {"input": tokens, "labels": tokens, "positions": positions}
        reference_mask = None
        if args.input_mode != "causal":
            padding = torch.tensor([False] * 7 + [True], device=device).reshape(1, -1)
            batch["padding_mask"] = padding
            reference_mask = (~padding).long()
        if args.input_mode == "packed":
            positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]], device=device)
            batch["positions"] = positions
            documents = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], device=device)
            causal = torch.ones(8, 8, dtype=torch.bool, device=device).tril()
            allowed = causal & (documents[:, None] == documents[None, :])
            allowed &= (~padding).reshape(1, 8)
            reference_mask = torch.zeros(8, 8, dtype=dtype, device=device)
            reference_mask.masked_fill_(~allowed, float("-inf"))
            reference_mask = reference_mask[None, None]
        with torch.no_grad():
            expected_logits = reference(
                tokens, position_ids=positions, attention_mask=reference_mask
            ).logits
            expected_logits = expected_logits.chunk(world, dim=1)[dist.get_rank()]
            expected_logits = expected_logits.reshape(-1, config.vocab_size)
        outputs = []
        for model in (direct, planned):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
            local, local_labels, metadata = model.prepare_inputs(
                batch, parallel_dims=topology
            )
            assert local.numel() == local_labels.numel() == 8 // world
            assert metadata["positions"].numel() == 8
            if args.input_mode != "causal":
                assert metadata["padding_mask"].numel() == 8
            logits = model(local, **metadata).logits
            rtol, atol = (2e-2, 2e-3) if dtype == torch.bfloat16 else (1e-5, 1e-6)
            torch.testing.assert_close(logits, expected_logits, rtol=rtol, atol=atol)
            if args.input_mode == "causal":
                with torch.no_grad():
                    inferred_positions = model(local).logits
                torch.testing.assert_close(logits, inferred_positions, rtol=0, atol=0)
            logits.float().square().mean().backward()
            optimizer.step()
            outputs.append(logits.detach())
        with synchronize_local_errors("TP preflight result comparison"):
            assert len(replacements) == 1
            modules, parameters = replacements[0]
            assert modules and parameters
            current = {id(parameter) for parameter in planned.parameters()}
            for source, targets in parameters.items():
                assert id(source) not in current
                assert len(targets) == 1 and id(targets[0]) in current
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            assert outputs[0].numel() > 0 and outputs[0].count_nonzero() > 0
            nonzero_gradients = 0
            for (name, expected), (other, actual) in zip(
                direct.named_parameters(), planned.named_parameters(), strict=True
            ):
                assert name == other
                torch.testing.assert_close(expected, actual, atol=0, rtol=0, msg=name)
                assert (expected.grad is None) == (actual.grad is None)
                if expected.grad is not None:
                    torch.testing.assert_close(
                        expected.grad, actual.grad, atol=0, rtol=0, msg=name
                    )
                    nonzero_gradients += int(bool(expected.grad.count_nonzero()))
            assert nonzero_gradients > 0
        if dist.get_rank() == 0:
            print("Pre-resolved TP: logits, gradients and updates exactly equal")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
