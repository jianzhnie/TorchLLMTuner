"""Exercise explicit Qwen3 on a real two-rank 1F1B pipeline.

Uses production partitioning/scheduling and an independent unsharded reference.
Run with torchrun --standalone --nproc_per_node=2 -m
 tests.integration_tests.explicit_pp_reference_equivalence.
Append --ac full to checkpoint each rank's local decoder after partitioning.
"""

from argparse import ArgumentParser
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import DefaultDeviceType
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as ReferenceModel,
)

from llmtuner.config.parallel_config import ParallelConfig, PipelineParallelConfig
from llmtuner.memory.activation_checkpoint import apply_ac
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.pipeline_parallel.pipeline_parallel import (
    apply_pp,
    build_pipeline_schedule,
)
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.harness import init_distributed


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--ac", choices=("none", "full"), default="none")
    args = parser.parse_args()
    device = init_distributed()
    previous_checkpoint_device = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type(device.type)
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This pipeline check requires exactly two ranks")
        mesh = init_device_mesh(device.type, (2,), mesh_dim_names=("pp",))
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            mlp_only_layers=[1],
            tie_word_embeddings=False,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        torch.manual_seed(61)
        model = Qwen3MoeForCausalLM(deepcopy(config)).to(device)
        reference = ReferenceModel(deepcopy(config)).to(device)
        reference.load_state_dict(model.state_dict(), strict=True)
        cfg = ParallelConfig(
            pipeline_parallel_size=2,
            pp_config=PipelineParallelConfig(num_pp_microbatches=2),
        )
        topology = SimpleNamespace(
            pp=2,
            dp_replicate=1,
            dp_shard=1,
            get_mesh=lambda name: mesh,
        )
        stages, parts, first, last = apply_pp(
            model,
            parallel_dims=topology,
            cfg=cfg,
            device=device,
            global_batch_size=2,
        )
        assert len(parts) == 1
        part = parts[0]
        keys = tuple(part.state_dict())
        parameters = tuple(part.parameters())
        apply_ac(part, args.ac)
        assert tuple(part.state_dict()) == keys
        assert all(a is b for a, b in zip(parameters, part.parameters(), strict=True))
        if args.ac == "full":
            assert len(part.layers)
            assert all(
                hasattr(layer, "_checkpoint_wrapped_module") for layer in part.layers
            )
        schedule = build_pipeline_schedule(stages, cfg=cfg)
        optimizer = torch.optim.SGD(part.parameters(), lr=0.01, foreach=False)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.01, foreach=False
        )
        for step in range(3):
            tokens = ((torch.arange(12, device=device) * 5 + step + 1) % 32).reshape(
                2, 6
            )
            labels = (tokens + 3) % 32
            expected = reference(tokens, use_cache=False).logits
            loss = F.cross_entropy(
                expected.flatten(0, 1), labels.flatten(), reduction="mean"
            )
            loss.backward()
            schedule._llmtuner_global_valid_tokens = torch.tensor(12.0, device=device)
            losses = [] if last else None
            actual = schedule.step(
                *((tokens,) if first else ()),
                target=labels if last else None,
                losses=losses,
            )
            if last:
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(
                    torch.stack(losses).sum(), loss, rtol=1e-5, atol=1e-6
                )
            targets = dict(reference.named_parameters())
            nonzero = 0
            for name, parameter in part.named_parameters():
                target = targets[canonical_fqn(name)]
                torch.testing.assert_close(
                    parameter.grad, target.grad, rtol=1e-5, atol=1e-6
                )
                nonzero += parameter.grad is not None and bool(
                    parameter.grad.count_nonzero()
                )
            if not nonzero:
                raise AssertionError("Pipeline backward produced no gradients")
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in part.named_parameters():
                torch.testing.assert_close(
                    parameter, targets[canonical_fqn(name)], rtol=1e-5, atol=1e-6
                )
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            if dist.get_rank() == 0:
                print(
                    f"PP AC={args.ac} reference step {step + 1}: "
                    "gradients and updates passed"
                )
    finally:
        try:
            dist.destroy_process_group()
        finally:
            DefaultDeviceType.set_device_type(previous_checkpoint_device)


if __name__ == "__main__":
    main()
