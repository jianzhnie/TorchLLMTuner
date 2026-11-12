"""Real NPU sharding/communication and three-step unsharded training oracle.

Run with torchrun -m experiments.mini-llmtuner.tests.distributed. Each rank
checks its reconstructed gradients and weights, not only rank zero's loss.
"""

import argparse

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from ..components import CrossEntropyLoss, NoOpScheduler
from ..config import ExperimentConfig, ParallelConfig, load_model_config
from ..data import SyntheticData
from ..engine import TrainingEngine
from ..llama import LlamaModel
from ..model import Qwen3Model, Qwen3MoeModel
from ..parallel import ParallelRuntime, local_tensor
from ..train import select_device
from ..trainer import ExperimentTrainer


class HuggingFaceReference(torch.nn.Module):
    """Independent HF forward/math with names matching the experiment wrapper."""

    def __init__(self, architecture="qwen3_moe"):
        super().__init__()
        model_class = (
            LlamaForCausalLM if architecture == "llama" else Qwen3MoeForCausalLM
        )
        self.model = model_class(load_model_config(None, architecture))

    def preprocess_inputs(self, batch):
        return batch

    @property
    def layers(self):
        return self.model.model.layers

    def forward(self, input_ids, *, position_ids):
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        ).logits


def gather(value: torch.Tensor, group, axis: int, packed: bool = False) -> torch.Tensor:
    parts = [torch.empty_like(value) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, value.contiguous(), group=group)
    if packed:
        halves = [part.chunk(2, dim=axis) for part in parts]
        return torch.cat(
            (
                torch.cat([x[0] for x in halves], axis),
                torch.cat([x[1] for x in halves], axis),
            ),
            axis,
        )
    return torch.cat(parts, dim=axis)


def full_parameter(name, value, trainer):
    runtime = trainer.assembly.runtime
    mapping = next(x for x in runtime.parameter_mappings if x.canonical_name == name)
    tensor = value.full_tensor() if isinstance(value, DTensor) else value.detach()
    if mapping.source.ep_axis is not None:
        tensor = gather(tensor, runtime.ep_group, 0)
    if mapping.source.tp_axis is not None:
        if name.endswith("gate_up_proj"):
            tensor = gather(tensor, runtime.tp_group, 1, packed=True)
        elif name.endswith("down_proj") and ".experts." in name:
            tensor = gather(tensor, runtime.tp_group, 2)
        else:
            path = name.rsplit(".", 1)[0]
            layout = trainer.model.contract.tp_plan[path]
            tensor = gather(tensor, runtime.tp_group, 0 if layout == "colwise" else 1)
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--fsdp-size", type=int, default=1)
    parser.add_argument("--empty-owner", action="store_true")
    parser.add_argument(
        "--architecture", choices=("qwen3_moe", "qwen3", "llama"), default="qwen3_moe"
    )
    parser.add_argument("--hf-reference", action="store_true")
    args = parser.parse_args()
    if args.hf_reference and args.architecture not in ("qwen3_moe", "llama"):
        parser.error("HF oracle supports qwen3_moe and llama")
    if args.architecture != "qwen3_moe" and (args.ep_size > 1 or args.empty_owner):
        parser.error("Dense Qwen3 does not support EP or empty-expert checks")
    model_class = {
        "qwen3": Qwen3Model,
        "qwen3_moe": Qwen3MoeModel,
        "llama": LlamaModel,
    }[args.architecture]
    device = select_device("npu")
    dist.init_process_group("hccl")
    trainer = None
    try:
        config = ExperimentConfig(
            parallel=ParallelConfig(
                tensor_parallel_size=args.tp_size,
                expert_parallel_size=args.ep_size,
                data_parallel_shard_size=args.fsdp_size,
            ),
            gradient_accumulation_steps=2,
            architecture=args.architecture,
        )
        bad = model_class(load_model_config(None, args.architecture)).to(device)
        parameters = tuple(bad.parameters())
        if dist.get_rank() == 0:
            bad.model.config.num_attention_heads += 1
        caught = None
        try:
            bad.parallelize(config.parallel, world_size=dist.get_world_size())
        except (ValueError, RuntimeError) as error:
            caught = error
        assert caught is not None, "Rank-local geometry failure was not coordinated"
        if dist.get_rank() != 0:
            assert "preflight failed on another rank" in str(caught)
        assert all(a is b for a, b in zip(parameters, bad.parameters(), strict=True))
        del bad
        torch.manual_seed(config.seed)
        reference = (
            HuggingFaceReference(args.architecture)
            if args.hf_reference
            else model_class(load_model_config(None, args.architecture))
        ).to(device)
        source = model_class(load_model_config(None, args.architecture)).to(device)
        source.load_state_dict(reference.state_dict())
        trainer = ExperimentTrainer(
            config,
            device,
            model=source,
            optimizer_factory=lambda params, lr: torch.optim.SGD(
                params, lr=lr, momentum=0.9, foreach=False, fused=False
            ),
        )
        runtime = trainer.assembly.runtime
        optimizer = torch.optim.SGD(
            reference.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            foreach=False,
            fused=False,
        )
        oracle = TrainingEngine(
            reference, optimizer, NoOpScheduler(), CrossEntropyLoss(), ParallelRuntime()
        )
        streams = [SyntheticData(64, config, device) for _ in range(runtime.data_size)]
        for rank, stream in enumerate(streams):
            stream.stream_rank, stream.stream_size = rank, runtime.data_size
        # Assert real ownership, including DTensor FSDP storage and TP expert axes.
        if args.fsdp_size > 1:
            assert isinstance(trainer.model, FSDPModule)
            assert all(isinstance(p, DTensor) for p in trainer.model.parameters())
            assert any(
                p.to_local().numel() < p.numel() for p in trainer.model.parameters()
            )
        if args.architecture == "qwen3_moe":
            experts = trainer.model.layers[0].mlp.experts
            assert experts.gate_up_proj.shape == (
                4 // args.ep_size,
                32 // args.tp_size,
                32,
            )
            assert experts.down_proj.shape == (
                4 // args.ep_size,
                32,
                16 // args.tp_size,
            )
        errors = {"loss": 0.0, "gradient": 0.0, "weight": 0.0, "grad_norm": 0.0}
        for _step in range(3):
            if args.empty_owner:
                # Equal logits route every token to the same top-k IDs, so EP
                # has an entirely empty owner. Keep the real router/autograd.
                with torch.no_grad():
                    for model in (reference, trainer.model):
                        for layer in model.layers:
                            layer.mlp.gate.weight.zero_()
            batches, merged = [], []
            for _ in range(2):
                group = [stream.next_batch() for stream in streams]
                for index, batch in enumerate(group):
                    batch["labels"][:, -(index + 1) :] = -100
                batches.append(group[runtime.data_rank])
                merged.append(
                    {key: torch.cat([b[key] for b in group]) for key in group[0]}
                )
            expected = oracle.train_step(merged)
            before = [
                local_tensor(p).detach().clone() for p in trainer.model.parameters()
            ]
            actual = trainer.engine.train_step(batches)
            assert any(
                not torch.equal(local_tensor(p), previous)
                for p, previous in zip(trainer.model.parameters(), before, strict=True)
            ), "No local parameter update occurred"
            assert actual.targets == expected.targets > 0
            torch.testing.assert_close(
                actual.loss_sum, expected.loss_sum, rtol=2e-5, atol=2e-5
            )
            torch.testing.assert_close(
                actual.grad_norm, expected.grad_norm, rtol=2e-5, atol=2e-5
            )
            errors["loss"] = max(
                errors["loss"], abs(actual.loss_sum - expected.loss_sum)
            )
            errors["grad_norm"] = max(
                errors["grad_norm"], abs(actual.grad_norm - expected.grad_norm)
            )
            nonzero = 0
            for (name, parameter), (other, baseline) in zip(
                trainer.model.named_parameters(),
                reference.named_parameters(),
                strict=True,
            ):
                assert name == other
                for key, value, target in (
                    ("weight", parameter, baseline),
                    ("gradient", parameter.grad, baseline.grad),
                ):
                    assert value is not None and target is not None, name
                    full = full_parameter(name, value, trainer)
                    mapping = next(
                        x
                        for x in trainer.assembly.parameter_mappings
                        if x.canonical_name == name
                    )
                    torch.testing.assert_close(
                        local_tensor(value),
                        mapping.select_local(target),
                        rtol=2e-5,
                        atol=2e-5,
                        msg=f"ownership:{key}:{name}",
                    )
                    torch.testing.assert_close(
                        full, target, rtol=2e-5, atol=2e-5, msg=f"{key}:{name}"
                    )
                    errors[key] = max(
                        errors[key],
                        float((full.detach() - target.detach()).abs().max()),
                    )
                nonzero += int(bool(baseline.grad.count_nonzero()))
            assert nonzero > 0
        if args.ep_size > 1:
            assert all(
                layer.mlp.dispatcher.exchanges >= 12 for layer in trainer.model.layers
            )
            if args.empty_owner:
                empties = torch.tensor(
                    sum(
                        layer.mlp.dispatcher.empty_batches
                        for layer in trainer.model.layers
                    ),
                    device=device,
                )
                dist.all_reduce(empties)
                assert int(empties) > 0, "Empty-owner case did not run"
        if dist.get_rank() == 0:
            print(
                f"PASS NPU FSDP{args.fsdp_size}/TP{args.tp_size}/EP{args.ep_size}: "
                f"3 steps x 2 unequal-target microbatches; max_abs={errors}",
                flush=True,
            )
    finally:
        if trainer is not None:
            trainer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
