"""Compare direct and pre-resolved EP using real distributed dispatch.

Run explicitly with two idle devices:
    torchrun --standalone --nproc_per_node=2 -m \
        tests.integration_tests.ep_resolved_plan_equivalence

For joint TP2/EP2, use four ranks and append --tp-degree 2. TP executes before
EP in both arms; the planned arm consumes resolved targets for both stages.
--shared-experts ties the two source experts modules. --resume-model saves after
step one, cold-loads a fresh converted model and compares two subsequent steps.
This mode synchronizes replica gradients before updating checkpointed weights.
SGD is stateless by default; --resume-optimizer selects production AdamW and
also checks its cold-loaded moments/counters and subsequent updates.
--resume-ema additionally checks unique ownership, restored averages and an
independent lerp formula each step. --fsdp with --tp-degree 2 additionally
shards the joint model using native FSDP; --custom-fsdp selects custom FSDP.
RNG recovery and independent-process
restart remain outside this script.

CPU/Gloo uses LLMTUNER_TEST_DEVICE=cpu and foreach=False optimizers.
At least two ranks and actual gradient-carrying all-to-all calls are required.

Guards reject repeated routing/layout discovery after parsing. Exact equality
is required between two arms of the same EP implementation, not against an
unsharded model or a different kernel. This is a small-model acceptance test.
It compares transformation paths, not full Trainer gradient normalization or
independent mathematical correctness of the shared implementations.
"""

from argparse import ArgumentParser
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.components.checkpointer.core.model_state import ModelCheckpointState
from llmtuner.components.optimizer.ema import EMA
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel import token_dispatch
from llmtuner.parallel.expert_parallel.model_replacement import (
    swap_moe_blocks,
)
from llmtuner.parallel.expert_parallel.plan import resolve_ep_plan
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp, resolve_tp_plan
from llmtuner.parallel.topology import ParallelTopology, build_device_mesh
from llmtuner.trainer.runtime.gradients import (
    allreduce_fsdp_excluded_grads,
    allreduce_replicated_tp_grads,
    expert_parameter_ids,
)
from tests.integration_tests.harness import init_distributed


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tp-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--fsdp", action="store_true", help="Shard the joint TP/EP model"
    )
    parser.add_argument(
        "--custom-fsdp", action="store_true", help="Use custom FSDP with --fsdp"
    )
    parser.add_argument(
        "--shared-experts",
        action="store_true",
        help="Share one packed experts module across both decoder blocks",
    )
    parser.add_argument(
        "--resume-model", action="store_true", help="Cold-load model after step one"
    )
    parser.add_argument(
        "--resume-optimizer", action="store_true", help="Include production AdamW state"
    )
    parser.add_argument(
        "--resume-ema", action="store_true", help="Include EMA averages"
    )
    args = parser.parse_args()
    if args.resume_optimizer and not args.resume_model:
        parser.error("--resume-optimizer requires --resume-model")
    if args.resume_ema and not args.resume_optimizer:
        parser.error("--resume-ema requires --resume-optimizer")
    if args.fsdp and args.tp_degree != 2:
        parser.error("--fsdp requires --tp-degree 2")
    if args.custom_fsdp and not args.fsdp:
        parser.error("--custom-fsdp requires --fsdp")
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
                fsdp_config=FSDPConfig(
                    fsdp_implementation="custom" if args.custom_fsdp else "native"
                ),
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

        def build_model():
            model = Qwen3MoeForCausalLM(config).to(device)
            if args.shared_experts:
                model.layers[1].mlp.experts = model.layers[0].mlp.experts
            return model

        def convert_planned(model):
            if mesh is not None:
                with synchronize_local_errors(
                    "Joint TP pure plan", group=dist.group.WORLD
                ):
                    tp_plan = resolve_tp_plan(model, cfg=cfg)
                model = apply_tp(model, mesh, cfg, resolved_plan=tp_plan)
            with synchronize_local_errors("EP pure plan", group=dist.group.WORLD):
                plan = resolve_ep_plan(
                    model, degree=ep_degree, tp_enabled=args.tp_degree > 1
                )
            with (
                patch(
                    "llmtuner.parallel.expert_parallel.plan.find_moe_candidates",
                    side_effect=AssertionError(
                        "EP execution rediscovered decoder slots"
                    ),
                ),
                patch(
                    "llmtuner.parallel.expert_parallel.bindings.resolve_packed_experts",
                    side_effect=AssertionError(
                        "EP execution rediscovered packed layout"
                    ),
                ),
                patch(
                    "llmtuner.parallel.expert_parallel.bindings.resolve_moe_routing",
                    side_effect=AssertionError("EP execution rediscovered routing"),
                ),
            ):
                swap_moe_blocks(
                    model,
                    ep_group=ep_group,
                    tp_enabled=args.tp_degree > 1,
                    resolved_plan=plan,
                )
            return apply_fsdp(model, cfg, topology) if args.fsdp else model

        direct = build_model()
        planned = build_model()
        planned.load_state_dict(direct.state_dict(), strict=True)
        if mesh is not None:
            direct = apply_tp(direct, mesh, cfg)
        swap_moe_blocks(direct, ep_group=ep_group, tp_enabled=args.tp_degree > 1)
        if args.fsdp:
            direct = apply_fsdp(direct, cfg, topology)
        planned = convert_planned(planned)
        for model in (direct, planned):
            if args.shared_experts:
                first = model.layers[0].mlp.routed_experts.inner_experts
                second = model.layers[1].mlp.routed_experts.inner_experts
                assert first is second
                for name in ("w1_EFD", "w3_EFD", "w2_EDF"):
                    assert getattr(first, name) is getattr(second, name)
            for layer in model.layers:
                experts = layer.mlp.routed_experts.inner_experts
                assert experts.w1_EFD.shape[0] == config.num_experts // ep_degree
                if args.fsdp:
                    assert isinstance(experts.w1_EFD, DTensor)
                    assert experts.w1_EFD.to_local().numel() < experts.w1_EFD.numel()
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

        def build_optimizer(model):
            if args.custom_fsdp:
                from llmtuner.parallel.fully_shard_parallel.fine_grained.fine_grained_state import (  # noqa: E501
                    FineGrainedFSDPState,
                )

                assert isinstance(model._get_fsdp_state(), FineGrainedFSDPState)
            if args.resume_optimizer:
                return OptimizersContainer(
                    OptimizerConfig(learning_rate=0.01, implementation="for-loop"),
                    model_parts=[model],
                )
            return torch.optim.SGD(
                model.parameters(), lr=0.01, foreach=False, fused=False
            )

        optimizers = [build_optimizer(model) for model in (direct, planned)]

        def build_ema(model):
            if not args.resume_ema:
                return None
            ema = EMA(model_parts=[model], decay=0.5)
            tracked = [
                parameter
                for inner in ema.optimizers
                for group in inner.param_groups
                for parameter in group["params"]
            ]
            assert len(tracked) == len({id(parameter) for parameter in tracked})
            assert len(tracked) == sum(p.requires_grad for p in model.parameters())
            return ema

        averages = [build_ema(model) for model in (direct, planned)]

        def local_tensor(value):
            return value.to_local() if isinstance(value, DTensor) else value

        with patch.object(
            token_dispatch,
            "_grad_carrying_all_to_all",
            wraps=token_dispatch._grad_carrying_all_to_all,
        ) as exchange:
            for step in range(3):
                outputs = []
                for model, optimizer, ema in zip(
                    (direct, planned), optimizers, averages, strict=True
                ):
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
                    if args.resume_model:
                        # DCP treats dense weights as EP-replicated. The raw
                        # transformation comparison otherwise updates them with
                        # rank-local gradients, which is not a resumable layout.
                        if topology is not None:
                            allreduce_fsdp_excluded_grads(
                                [model], parallel_dims=topology
                            )
                            allreduce_replicated_tp_grads(
                                [model], parallel_dims=topology
                            )
                        else:
                            expert_ids = expert_parameter_ids([model])
                            for parameter in model.parameters():
                                if (
                                    id(parameter) not in expert_ids
                                    and parameter.grad is not None
                                ):
                                    dist.all_reduce(parameter.grad, group=ep_group)
                    before = deepcopy(ema.state_dict()) if ema is not None else None
                    optimizer.step()
                    if ema is not None:
                        ema.step(step + 1)
                        weights = ModelCheckpointState(model).state_dict()
                        for key, value in ema.state_dict().items():
                            fqn = key.removeprefix("state.").removesuffix(".ema_params")
                            torch.testing.assert_close(
                                value,
                                before[key].lerp(weights[fqn], 0.5),
                                atol=0,
                                rtol=0,
                            )
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
                            local_tensor(left),
                            local_tensor(right),
                            atol=0,
                            rtol=0,
                            msg=name,
                        )
                        assert (left.grad is None) == (right.grad is None)
                        if left.grad is not None:
                            torch.testing.assert_close(
                                local_tensor(left.grad),
                                local_tensor(right.grad),
                                atol=0,
                                rtol=0,
                                msg=name,
                            )
                            nonzero += int(
                                bool(local_tensor(left.grad).count_nonzero())
                            )
                    assert nonzero > 0
                if args.resume_optimizer:
                    states = [optimizer.state_dict() for optimizer in optimizers]
                    torch.testing.assert_close(states[0], states[1], atol=0, rtol=0)
                if args.resume_ema:
                    states = [ema.state_dict() for ema in averages]
                    torch.testing.assert_close(states[0], states[1], atol=0, rtol=0)
                if args.resume_model and step == 0:
                    context = (
                        TemporaryDirectory() if dist.get_rank() == 0 else nullcontext()
                    )
                    with context as folder:
                        paths = [folder]
                        dist.broadcast_object_list(paths, src=0)
                        state = {"model": ModelCheckpointState(planned)}
                        if args.resume_optimizer:
                            state["optimizer"] = optimizers[1]
                        if args.resume_ema:
                            state["ema"] = averages[1]
                        dcp.save(state, checkpoint_id=paths[0])
                        resumed = convert_planned(build_model())
                        with torch.no_grad():
                            for parameter in resumed.parameters():
                                parameter.fill_(float("nan"))
                        resumed_optimizer = build_optimizer(resumed)
                        resumed_ema = build_ema(resumed)
                        destination = {"model": ModelCheckpointState(resumed)}
                        if args.resume_optimizer:
                            assert all(
                                not inner.state
                                for inner in resumed_optimizer.optimizers
                            )
                            destination["optimizer"] = resumed_optimizer
                        if args.resume_ema:
                            destination["ema"] = resumed_ema
                        dcp.load(destination, checkpoint_id=paths[0])
                        if args.shared_experts:
                            assert (
                                resumed.layers[0].mlp.routed_experts.inner_experts
                                is resumed.layers[1].mlp.routed_experts.inner_experts
                            )
                        torch.testing.assert_close(
                            {
                                key: local_tensor(value)
                                for key, value in resumed.state_dict().items()
                            },
                            {
                                key: local_tensor(value)
                                for key, value in planned.state_dict().items()
                            },
                            atol=0,
                            rtol=0,
                        )
                        if args.resume_optimizer:
                            torch.testing.assert_close(
                                resumed_optimizer.state_dict(),
                                optimizers[1].state_dict(),
                                atol=0,
                                rtol=0,
                            )
                        if args.resume_ema:
                            torch.testing.assert_close(
                                resumed_ema.state_dict(),
                                averages[1].state_dict(),
                                atol=0,
                                rtol=0,
                            )
                        planned = resumed
                        optimizers[1] = resumed_optimizer
                        averages[1] = resumed_ema
                        dist.barrier()
            with synchronize_local_errors(
                "EP communication coverage", group=dist.group.WORLD
            ):
                assert exchange.call_count > 0, (
                    "EP did not exchange gradient-carrying tokens"
                )
        if dist.get_rank() == 0:
            print(
                f"Pre-resolved TP{args.tp_degree}/EP{ep_degree} "
                f"shared_experts={args.shared_experts}: "
                f"resume_model={args.resume_model}: "
                f"resume_optimizer={args.resume_optimizer}: "
                f"resume_ema={args.resume_ema}: "
                f"fsdp={args.fsdp}: "
                f"custom_fsdp={args.custom_fsdp}: "
                "3-step logits, gradients and updates exactly equal"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
