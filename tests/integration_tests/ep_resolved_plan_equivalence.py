"""Compare direct and pre-resolved EP using real distributed dispatch.

Run explicitly with two idle devices:
    torchrun --standalone --nproc_per_node=2 -m \
        tests.integration_tests.ep_resolved_plan_equivalence

For joint TP2/EP2, use four ranks and append --tp-degree 2. TP executes before
EP in both arms; the planned arm consumes resolved targets for both stages.

CPU/Gloo uses LLMTUNER_TEST_DEVICE=cpu and foreach=False optimizers.
At least two ranks and actual gradient-carrying all-to-all calls are required.

Guards reject repeated routing/layout discovery after parsing. Exact equality
is required between two arms of the same EP implementation, not against an
unsharded model or a different kernel. This is a small-model acceptance test.
It compares transformation paths, not full Trainer gradient normalization or
independent mathematical correctness of the shared implementations.
"""

from argparse import ArgumentParser
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel import token_dispatch
from llmtuner.parallel.expert_parallel.model_replacement import (
    resolve_ep_plan,
    swap_moe_blocks,
)
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp, resolve_tp_plan
from llmtuner.parallel.topology import ParallelTopology, build_device_mesh
from tests.integration_tests.harness import init_distributed


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tp-degree", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    device = init_distributed(timeout=timedelta(seconds=120))
    try:
        if dist.get_world_size() < 2:
            raise ValueError("EP communication acceptance requires at least two ranks")
        ep_degree = dist.get_world_size() // args.tp_degree
        if dist.get_world_size() % args.tp_degree or ep_degree < 2:
            raise ValueError("Joint TP/EP acceptance requires TP degree times EP>=2")
        topology = None
        mesh = None
        ep_group = dist.group.WORLD
        if args.tp_degree > 1:
            cfg = ParallelConfig(
                tensor_parallel_size=args.tp_degree,
                expert_parallel_size=ep_degree,
                data_parallel_shard_size=ep_degree,
            )
            topology = ParallelTopology.from_config(cfg, dist.get_world_size())
            mesh = build_device_mesh(topology)
            ep_group = topology.get_mesh("ep").get_group()
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.0,
            experts_implementation="eager",
            attn_implementation="eager",
        )
        torch.manual_seed(173)
        direct = Qwen3MoeForCausalLM(config).to(device)
        planned = Qwen3MoeForCausalLM(config).to(device)
        planned.load_state_dict(direct.state_dict(), strict=True)
        if mesh is not None:
            with synchronize_local_errors("Joint TP pure plan", group=dist.group.WORLD):
                tp_plan = resolve_tp_plan(planned, cfg=cfg)
            direct = apply_tp(direct, mesh, cfg)
            planned = apply_tp(planned, mesh, cfg, resolved_plan=tp_plan)
        with synchronize_local_errors("EP pure plan", group=dist.group.WORLD):
            plan = resolve_ep_plan(
                planned, degree=ep_degree, tp_enabled=args.tp_degree > 1
            )
        swap_moe_blocks(direct, ep_group=ep_group, tp_enabled=args.tp_degree > 1)
        with (
            patch(
                "llmtuner.parallel.expert_parallel.model_replacement.find_moe_candidates",
                side_effect=AssertionError("EP execution rediscovered decoder slots"),
            ),
            patch(
                "llmtuner.parallel.expert_parallel.model_replacement._legacy_inspection",
                side_effect=AssertionError("EP execution rediscovered projections"),
            ),
            patch(
                "llmtuner.parallel.expert_parallel.block_conversion.resolve_packed_experts",
                side_effect=AssertionError("EP execution rediscovered packed layout"),
            ),
            patch(
                "llmtuner.parallel.expert_parallel.block_conversion.resolve_moe_routing",
                side_effect=AssertionError("EP execution rediscovered routing"),
            ),
        ):
            swap_moe_blocks(
                planned,
                ep_group=ep_group,
                tp_enabled=args.tp_degree > 1,
                resolved_plan=plan,
            )
        for model in (direct, planned):
            for layer in model.layers:
                experts = layer.mlp.routed_experts.inner_experts
                assert experts.w1_EFD.shape[0] == config.num_experts // ep_degree
                if mesh is not None:
                    assert layer.self_attn.q_proj.weight.shape[0] == (
                        config.num_attention_heads * config.head_dim // args.tp_degree
                    )
        # Asymmetric per-rank tokens exercise actual expert exchange.
        stream_rank = (
            dist.get_rank()
            if topology is None
            else topology.get_mesh("batch").get_local_rank()
        )
        tokens = (torch.arange(8, device=device) + 3 * stream_rank) % 32
        positions = torch.arange(8, device=device)
        optimizers = [
            torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
            for model in (direct, planned)
        ]
        with patch.object(
            token_dispatch,
            "_grad_carrying_all_to_all",
            wraps=token_dispatch._grad_carrying_all_to_all,
        ) as exchange:
            for _ in range(3):
                outputs = []
                for model, optimizer in zip((direct, planned), optimizers, strict=True):
                    optimizer.zero_grad(set_to_none=True)
                    if topology is None:
                        local, metadata = tokens, {"positions": positions}
                    else:
                        local, labels, metadata = model.prepare_inputs(
                            {"input": tokens, "labels": tokens, "positions": positions},
                            parallel_dims=topology,
                        )
                        assert local.numel() == labels.numel() == 8 // args.tp_degree
                    output = model(local, **metadata).logits
                    (output.float().square().mean()).backward()
                    optimizer.step()
                    outputs.append(output.detach())
                with synchronize_local_errors(
                    "EP exact comparison", group=dist.group.WORLD
                ):
                    assert outputs[0].numel() and outputs[0].count_nonzero()
                    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
                    nonzero = 0
                    for (name, left), (other, right) in zip(
                        direct.named_parameters(),
                        planned.named_parameters(),
                        strict=True,
                    ):
                        assert name == other
                        torch.testing.assert_close(
                            left, right, atol=0, rtol=0, msg=name
                        )
                        assert (left.grad is None) == (right.grad is None)
                        if left.grad is not None:
                            torch.testing.assert_close(
                                left.grad, right.grad, atol=0, rtol=0, msg=name
                            )
                            nonzero += int(bool(left.grad.count_nonzero()))
                    assert nonzero > 0
            with synchronize_local_errors(
                "EP communication coverage", group=dist.group.WORLD
            ):
                assert exchange.call_count > 0, (
                    "EP did not exchange gradient-carrying tokens"
                )
        if dist.get_rank() == 0:
            print(
                f"Pre-resolved TP{args.tp_degree}/EP{ep_degree}: "
                "3-step logits, gradients and updates exactly equal"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
