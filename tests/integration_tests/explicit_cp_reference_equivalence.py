"""Compare explicit Qwen3 CP training with an independent unsharded reference.

Run with torchrun --standalone --nproc_per_node=2 -m
 tests.integration_tests.explicit_cp_reference_equivalence. Exercises unpadded
causal input, real sequence communication and three optimizer updates.
Add --tp-degree 2 with four ranks to verify TP=2 x CP=2, including packed
expert shard layouts and production replicated-parameter gradient reduction.
"""

import argparse
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as ReferenceModel,
)

from llmtuner.config.parallel_config import ContextParallelConfig, ParallelConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.context_parallel.context_parallel import apply_cp
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp
from llmtuner.parallel.topology import ParallelTopology, build_device_mesh
from llmtuner.trainer.runtime.gradients import allreduce_replicated_tp_grads
from tests.integration_tests.harness import init_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", choices=("kv_allgather", "ulysses"), default="kv_allgather"
    )
    parser.add_argument("--tp-degree", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    device = init_distributed()
    try:
        size, rank = dist.get_world_size(), dist.get_rank()
        if size != 2 * args.tp_degree:
            raise ValueError("This check requires CP=2 times the requested TP degree")
        parallel = ParallelConfig(
            tensor_parallel_size=args.tp_degree,
            context_parallel_kv_allgather_size=2
            if args.strategy == "kv_allgather"
            else 1,
            context_parallel_ulysses_size=2 if args.strategy == "ulysses" else 1,
            cp_config=ContextParallelConfig(
                context_parallel_strategy=args.strategy,
                context_parallel_load_balancer=None,
            ),
        )
        topology = ParallelTopology.from_config(parallel, size)
        full_mesh = build_device_mesh(topology)
        mesh = topology.get_optional_mesh("cp")
        tp_mesh = topology.get_optional_mesh("tp")
        assert mesh is not None
        assert (tp_mesh is not None) == (args.tp_degree > 1)
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2 * args.tp_degree,
            head_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
            mlp_only_layers=[1],
            tie_word_embeddings=False,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        torch.manual_seed(29)
        model = Qwen3MoeForCausalLM(deepcopy(config)).to(device)
        reference = ReferenceModel(deepcopy(config)).to(device)
        reference.load_state_dict(model.state_dict(), strict=True)
        model = apply_tp(model, full_mesh, parallel)
        apply_cp(model, mesh=full_mesh, cfg=parallel, parallel_dims=topology)
        runtime = SimpleNamespace(parallel_dims=topology, model_parts=[model])
        layouts = getattr(model, "_tp_checkpoint_layouts", {})

        def reference_shard(name: str, value: torch.Tensor) -> torch.Tensor:
            if name not in layouts:
                return value
            axis, packed, _ = layouts[name]
            coordinate = tp_mesh.get_local_rank()
            if packed:
                halves = value.chunk(2, dim=axis)
                return torch.cat(
                    [
                        half.chunk(args.tp_degree, dim=axis)[coordinate]
                        for half in halves
                    ],
                    dim=axis,
                )
            return value.chunk(args.tp_degree, dim=axis)[coordinate]

        calls = []
        kernels = {
            id(
                layer.self_attn._titan_cp_sdpa_kernel
            ): layer.self_attn._titan_cp_sdpa_kernel
            for layer in model.layers
        }
        for kernel in kernels.values():
            kernel.register_forward_hook(lambda *args: calls.append(1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.01, foreach=False
        )
        for step in range(3):
            tokens = (torch.arange(12, device=device) * 7 + step + 1) % 32
            inputs, _, kwargs = model.prepare_inputs(
                {"input": tokens, "labels": tokens}, parallel_dims=topology
            )
            logits = model(inputs, **kwargs).logits
            expected = reference(tokens.unsqueeze(0), use_cache=False).logits[0]
            local = tokens.numel() // size
            tp_coordinate = tp_mesh.get_local_rank() if tp_mesh is not None else 0
            offset = (mesh.get_local_rank() * args.tp_degree + tp_coordinate) * local
            torch.testing.assert_close(
                logits, expected[offset : offset + local], rtol=1e-5, atol=1e-6
            )
            coefficients = torch.arange(1, 13, device=device).float()[:, None]
            loss = (
                logits.square() * coefficients[offset : offset + local]
            ).sum() / expected.numel()
            expected_loss = (expected.square() * coefficients).mean()
            loss.backward()
            expected_loss.backward()
            allreduce_replicated_tp_grads(runtime)
            expected_parameters = dict(reference.named_parameters())
            nonzero = 0
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, group=mesh.get_group())
                target = expected_parameters[name]
                expected_grad = (
                    target.grad if target.grad is not None else torch.zeros_like(target)
                )
                torch.testing.assert_close(
                    parameter.grad,
                    reference_shard(name, expected_grad),
                    rtol=1e-4,
                    atol=2e-6,
                )
                nonzero += bool(parameter.grad.count_nonzero())
            if nonzero == 0 or len(calls) != (step + 1) * len(model.layers):
                raise AssertionError("CP kernel execution or gradients were vacuous")
            before = model.tok_embeddings.weight.detach().clone()
            optimizer.step()
            if torch.equal(before, model.tok_embeddings.weight):
                raise AssertionError("CP training did not update embedding parameters")
            reference_optimizer.step()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    parameter,
                    reference_shard(name, expected_parameters[name]),
                    rtol=1e-5,
                    atol=1e-6,
                )
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            if rank == 0:
                print(
                    f"TP={args.tp_degree} CP=2 reference step {step + 1}: "
                    "logits, gradients, updates passed"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
