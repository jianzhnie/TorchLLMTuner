"""Two-rank explicit EP versus independent unsharded Transformers math.

Run with torchrun --standalone --nproc_per_node=2 -m
 tests.integration_tests.explicit_ep_reference_equivalence.
CPU validation uses LLMTUNER_TEST_DEVICE=cpu. This small FP32 test declares
rtol=1e-5 / atol=1e-6 for different expert GEMM/accumulation decompositions.
Use four ranks and --tp-degree 2 for TP2/EP2 with two independent batch streams.
The production sparse mesh overlaps TP: EP peers route disjoint sequence
shards from the same full input. Replicated TP gradients use the runtime
reduction; the oracle remains a complete, unsharded model for that input.
Append --ac full to checkpoint the candidate decoder layers after TP/EP.
"""

from argparse import ArgumentParser
from collections.abc import Iterator
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.checkpoint import DefaultDeviceType
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as ReferenceModel,
)

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.memory.activation_checkpoint import apply_ac
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel.model_replacement import swap_moe_blocks
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp
from llmtuner.parallel.topology import ParallelTopology, build_device_mesh
from llmtuner.trainer.runtime.gradients import allreduce_replicated_tp_grads
from tests.integration_tests.harness import init_distributed


def mapped_parameters(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
    *,
    ep_group: dist.ProcessGroup,
    tp_rank: int = 0,
    tp_degree: int = 1,
) -> Iterator[tuple[str, torch.nn.Parameter, torch.nn.Parameter, tuple[slice, ...]]]:
    """Map independent packed reference storage to rank-local native experts."""
    actual = dict(candidate.named_parameters())
    rank, size = dist.get_rank(ep_group), dist.get_world_size(ep_group)
    for name, parameter in reference.named_parameters():
        if name.endswith(".experts.gate_up_proj"):
            root = name.removesuffix(".experts.gate_up_proj")
            local = parameter.shape[0] // size
            width = parameter.shape[1] // 2
            for suffix, start in (("w1_EFD", 0), ("w3_EFD", width)):
                yield (
                    name + ":" + suffix,
                    actual[f"{root}.routed_experts.inner_experts.{suffix}"],
                    parameter,
                    (
                        slice(rank * local, (rank + 1) * local),
                        slice(start, start + width),
                    ),
                )
        elif name.endswith(".experts.down_proj"):
            root = name.removesuffix(".experts.down_proj")
            local = parameter.shape[0] // size
            yield (
                name,
                actual[f"{root}.routed_experts.inner_experts.w2_EDF"],
                parameter,
                (slice(rank * local, (rank + 1) * local),),
            )
        else:
            target = name.replace(".mlp.gate.weight", ".mlp.router.gate.weight")
            index = (slice(None),)
            if tp_degree > 1:
                if any(
                    f".{projection}.weight" in name
                    for projection in ("q_proj", "k_proj", "v_proj")
                ):
                    width = parameter.shape[0] // tp_degree
                    index = (slice(tp_rank * width, (tp_rank + 1) * width),)
                elif ".o_proj.weight" in name:
                    width = parameter.shape[1] // tp_degree
                    index = (slice(None), slice(tp_rank * width, (tp_rank + 1) * width))
            yield name, actual[target], parameter, index


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tp-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ac", choices=("none", "full"), default="none")
    args = parser.parse_args()
    device = init_distributed(timeout=timedelta(seconds=120))
    previous_checkpoint_device = DefaultDeviceType.get_device_type()
    # CPU-only checkpoint calls otherwise inherit torch-npu's process default
    # and attempt to initialize an unavailable NPU during recomputation.
    DefaultDeviceType.set_device_type(device.type)
    try:
        if dist.get_world_size() != 2 * args.tp_degree:
            raise ValueError("This acceptance requires two ranks per TP degree")
        topology = None
        ep_group = dist.group.WORLD
        tp_rank = 0
        if args.tp_degree > 1:
            parallel = ParallelConfig(
                tensor_parallel_size=2,
                expert_parallel_size=2,
                data_parallel_shard_size=2,
            )
            topology = ParallelTopology.from_config(parallel, dist.get_world_size())
            mesh = build_device_mesh(topology)
            ep_group = topology.get_mesh("ep").get_group()
            tp_rank = topology.get_mesh("tp").get_local_rank()
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
        )
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        torch.manual_seed(173)
        reference = ReferenceModel(deepcopy(config)).to(device)
        candidate = Qwen3MoeForCausalLM(deepcopy(config)).to(device)
        candidate.load_state_dict(reference.state_dict(), strict=True)
        if topology is not None:
            candidate = apply_tp(candidate, mesh, parallel)
        swap_moe_blocks(
            candidate,
            ep_group=ep_group,
            router_aux_loss_coef=0.0,
            tp_enabled=args.tp_degree > 1,
        )
        pairs = list(
            mapped_parameters(
                reference,
                candidate,
                ep_group=ep_group,
                tp_rank=tp_rank,
                tp_degree=args.tp_degree,
            )
        )
        state_keys = tuple(candidate.state_dict())
        candidate = apply_ac(candidate, args.ac)
        assert tuple(candidate.state_dict()) == state_keys
        if args.ac == "full":
            assert all(
                hasattr(layer, "_checkpoint_wrapped_module")
                for layer in candidate.layers
            )
        assert {id(pair[1]) for pair in pairs} == {
            id(p) for p in candidate.parameters()
        }
        optimizers = [
            torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
            for model in (reference, candidate)
        ]
        stream_rank = (
            dist.get_rank()
            if topology is None
            else topology.get_mesh("batch").get_local_rank()
        )
        tokens = ((torch.arange(8, device=device) + 3 * stream_rank) % 32)[None]
        for step in range(3):
            outputs = []
            for model, optimizer in zip(
                (reference, candidate), optimizers, strict=True
            ):
                optimizer.zero_grad(set_to_none=True)
                if model is candidate and topology is not None:
                    local, _, metadata = model.prepare_inputs(
                        {
                            "input": tokens,
                            "labels": tokens,
                            "positions": torch.arange(8, device=device)[None],
                        },
                        parallel_dims=topology,
                    )
                    output = model(local, **metadata).logits
                else:
                    output = model(tokens, use_cache=False).logits
                (
                    output.float().square().sum() / (tokens.numel() * config.vocab_size)
                ).backward()
                outputs.append(output.detach())
            # EP-only peers own distinct full inputs: sum their expert losses.
            # In this TP2/EP2 topology EP peers instead own disjoint TP token
            # shards of one input, already represented once by the full oracle.
            if topology is not None:
                allreduce_replicated_tp_grads(
                    SimpleNamespace(parallel_dims=topology, model_parts=[candidate])
                )
            for name, parameter in reference.named_parameters():
                if ".experts." in name and topology is None:
                    assert parameter.grad is not None
                    dist.all_reduce(parameter.grad)
            with synchronize_local_errors(
                "EP reference gradients", group=dist.group.WORLD
            ):
                expected_output = outputs[0].chunk(args.tp_degree, dim=1)[tp_rank]
                torch.testing.assert_close(
                    outputs[1].reshape_as(expected_output),
                    expected_output,
                    rtol=1e-5,
                    atol=1e-6,
                )
                assert outputs[1].numel() and outputs[1].count_nonzero()
                nonzero = expert_nonzero = router_nonzero = 0
                for name, actual, expected, index in pairs:
                    assert actual.grad is not None and expected.grad is not None, name
                    torch.testing.assert_close(
                        actual.grad,
                        expected.grad[index],
                        rtol=1e-5,
                        atol=1e-6,
                        msg=name,
                    )
                    has_gradient = int(bool(actual.grad.count_nonzero()))
                    nonzero += has_gradient
                    if ".experts." in name:
                        expert_nonzero += has_gradient
                    if name.endswith(".mlp.gate.weight"):
                        router_nonzero += has_gradient
                assert nonzero > 0 and expert_nonzero > 0 and router_nonzero > 0
            for optimizer in optimizers:
                optimizer.step()
            with synchronize_local_errors(
                "EP reference updates", group=dist.group.WORLD
            ):
                for name, actual, expected, index in pairs:
                    torch.testing.assert_close(
                        actual, expected[index], rtol=1e-5, atol=1e-6, msg=name
                    )
            if dist.get_rank() == 0:
                print(
                    f"TP{args.tp_degree}/EP2 AC={args.ac} reference step {step + 1}: "
                    "logits, gradients, updates passed"
                )
    finally:
        try:
            dist.destroy_process_group()
        finally:
            DefaultDeviceType.set_device_type(previous_checkpoint_device)


if __name__ == "__main__":
    main()
