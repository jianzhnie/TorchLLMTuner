"""Actual Trainer DP2 excluded-embedding replica regression.

Run with two CPU ranks using torchrun. Compare three AdamW steps with an
independent unsharded model, explicit shifted CE, accumulation and clipping.
Use four ranks and arguments "custom ep" for EP2/efsdp2, excluded expert
gate/up projections and independently computed nonzero router auxiliary loss.
Append ac and/or compile to exercise full checkpointing and aot_eager.
Use four ranks and append hsdp (without ep) for replicate2/shard2.
Append masked for unequal valid-label counts, including a fully masked DP rank;
prompt tokens still contribute to the independent router objective.
"""

import copy
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.datasets.contracts import Batch
from llmtuner.trainer.runtime import assembly
from llmtuner.trainer.trainer import Trainer

original_apply = assembly._apply_parallelism
references = []


def exclude(trainer, model, **kwargs):
    if trainer.cfg.parallel.ep_size > 1:
        source = model.layers[0].mlp.experts
        path = next(name for name, module in model.named_modules() if module is source)
        trainer.cfg.parallel.fsdp_config.fsdp_apply_modules = {
            path: {"ignored_params": {source.gate_up_proj}, "hook_module": source}
        }
    else:
        path = next(
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Embedding)
        )
        trainer.cfg.parallel.fsdp_config.fsdp_apply_modules = {"": {}}
        trainer.cfg.parallel.fsdp_config.fsdp_ignored_modules = [path]
    references.append(copy.deepcopy(model))
    original_apply(trainer, model, **kwargs)


def reference_parameter(reference, name, dims, ep_enabled):
    """Map HF fused expert projections to the physical EP slice independently."""
    name = ".".join(
        part
        for part in name.split(".")
        if part not in {"_orig_mod", "_checkpoint_wrapped_module"}
    )
    if ep_enabled and ".routed_experts.inner_experts." in name:
        source = reference.layers[0].mlp.experts
        ep_rank = dims.get_mesh("ep").get_local_rank()
        projection = name.rsplit(".", 1)[1]
        if projection == "w2_EDF":
            parameter = source.down_proj
            return (
                parameter.detach().chunk(2, dim=0)[ep_rank],
                parameter.grad.chunk(2, dim=0)[ep_rank],
            )
        index = {"w1_EFD": 0, "w3_EFD": 1}[projection]
        parameter = source.gate_up_proj
        return (
            parameter.detach().chunk(2, dim=0)[ep_rank].chunk(2, dim=1)[index],
            parameter.grad.chunk(2, dim=0)[ep_rank].chunk(2, dim=1)[index],
        )
    parameter = dict(reference.named_parameters())[
        name.replace(".mlp.router.gate.", ".mlp.gate.")
    ]
    return parameter, parameter.grad


def main():
    implementation = sys.argv[1] if len(sys.argv) > 1 else "native"
    ep_enabled = "ep" in sys.argv[2:]
    hsdp = "hsdp" in sys.argv[2:]
    if hsdp and ep_enabled:
        raise ValueError("This four-rank HSDP scenario requires EP1")
    world = 4 if ep_enabled or hsdp else 2
    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe" if ep_enabled else "qwen3",
            experts_implementation="eager" if ep_enabled else "native",
            arch_overrides={
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
            }
            if ep_enabled
            else {},
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        parallel=ParallelConfig(
            data_parallel_shard_size=2 if hsdp else world,
            data_parallel_replicate_size=2 if hsdp else 1,
            expert_parallel_size=2 if ep_enabled else 1,
        ),
        training=TrainingConfig(
            global_batch_size=2 * world,
            max_seq_len=8,
            steps=3,
            max_norm=0.5,
            gradient_accumulation_steps=2,
            seed=42,
        ),
    )
    cfg.parallel.fsdp_config.fsdp_implementation = implementation
    cfg.training.activation_checkpoint_mode = "full" if "ac" in sys.argv else "none"
    cfg.training.compile = "compile" in sys.argv
    cfg.training.compile_config.backend = "aot_eager"
    if ep_enabled:
        if implementation != "custom":
            raise ValueError("EP exclusion regression requires custom FSDP")
        cfg.parallel.ep_config.router_aux_loss_coef = 0.03
    trainer = None
    try:
        with patch.object(assembly, "_apply_parallelism", exclude):
            trainer = Trainer(cfg)
        assert dist.get_world_size() == world
        if hsdp:
            assert trainer.parallel_dims.get_mesh("dp_replicate").size() == 2
            assert trainer.parallel_dims.get_mesh("dp_shard").size() == 2
            assert any(
                isinstance(parameter, DTensor) and parameter.device_mesh.shape == (2, 2)
                for parameter in trainer.model.parameters()
            )
        if ep_enabled:
            grouped = trainer.model.layers[0].mlp.routed_experts.inner_experts
            weight = grouped.w1_EFD
            assert isinstance(grouped.w2_EDF, DTensor)
            replica_group = trainer.parallel_dims.get_mesh("efsdp").get_group()
        else:
            weight = next(
                module.weight
                for module in trainer.model.modules()
                if isinstance(module, torch.nn.Embedding)
            )
            replica_group = None
        assert not isinstance(weight, DTensor)
        assert any(isinstance(p, DTensor) for p in trainer.model.parameters())
        if ep_enabled:
            from llmtuner.models.common.aux_loss import AuxLoss

            assert any(
                isinstance(module, AuxLoss) and module.coeff == 0.03
                for module in trainer.model.modules()
            )
        reference = references.pop()
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
            foreach=False,
        )
        captured_router_outputs = []
        if ep_enabled:
            reference.layers[0].mlp.gate.register_forward_hook(
                lambda module, inputs, output: captured_router_outputs.append(output)
            )
        generators = [
            torch.Generator().manual_seed(400 + rank) for rank in range(world)
        ]
        for step in range(1, 4):
            trainer.step = step
            all_batches = [
                [
                    Batch(
                        input_ids=torch.randint(
                            32, (1, cfg.max_seq_len), generator=generator
                        ),
                        labels=torch.randint(
                            32, (1, cfg.max_seq_len), generator=generator
                        ),
                    )
                    for _ in range(2)
                ]
                for generator in generators
            ]
            if "masked" in sys.argv:
                for rank, batches in enumerate(all_batches):
                    for index, batch in enumerate(batches):
                        # Rank zero has no CE targets. Other ranks have
                        # different prompt lengths in each accumulation group.
                        prefix = cfg.max_seq_len if rank == 0 else rank + index + 1
                        batch.labels[:, :prefix] = -100
            valid_tokens = sum(
                int((batch.labels[:, 1:] != -100).sum())
                for batches in all_batches
                for batch in batches
            )
            assert valid_tokens > 0
            reference_optimizer.zero_grad(set_to_none=True)
            for batches in all_batches:
                for batch in batches:
                    logits = reference(batch.input_ids.flatten())
                    # Each independent row excludes its final next-token target.
                    loss = (
                        F.cross_entropy(
                            logits[:-1].float(), batch.labels[0, 1:], reduction="sum"
                        )
                        / valid_tokens
                    )
                    if ep_enabled:
                        router_logits, _, selected = captured_router_outputs.pop()
                        probabilities = router_logits.float().softmax(dim=-1)
                        counts = torch.bincount(selected.flatten(), minlength=4).float()
                        frequencies = counts * (4 / selected.numel())
                        auxiliary = (frequencies * probabilities.sum(dim=0)).sum()
                        loss = loss + 0.03 * auxiliary / (2 * world * cfg.max_seq_len)
                    loss.backward()
            expected_norm = torch.linalg.vector_norm(
                torch.cat(
                    [
                        p.grad.flatten()
                        for p in reference.parameters()
                        if p.grad is not None
                    ]
                )
            )
            factor = (0.5 / (expected_norm + 1e-6)).clamp(max=1)
            for parameter in reference.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
            for group in reference_optimizer.param_groups:
                group["lr"] = trainer.optimizer.optimizers[0].param_groups[0]["lr"]
            trainer.train_step(iter(all_batches[dist.get_rank()]))
            for name, parameter in trainer.model.named_parameters():
                expected = reference_parameter(
                    reference, name, trainer.parallel_dims, ep_enabled
                )
                gradient = parameter.grad
                if isinstance(gradient, DTensor):
                    gradient = gradient.full_tensor()
                torch.testing.assert_close(gradient, expected[1], atol=1e-6, rtol=1e-6)
            reference_optimizer.step()
            for name, parameter in trainer.model.named_parameters():
                expected = reference_parameter(
                    reference, name, trainer.parallel_dims, ep_enabled
                )
                value = (
                    parameter.full_tensor()
                    if isinstance(parameter, DTensor)
                    else parameter
                )
                torch.testing.assert_close(value, expected[0], atol=1e-6, rtol=1e-6)
            for value in (weight.grad, weight.detach()):
                replicas = [
                    torch.empty_like(value)
                    for _ in range(dist.get_world_size(replica_group))
                ]
                dist.all_gather(replicas, value, group=replica_group)
                for replica in replicas[1:]:
                    torch.testing.assert_close(
                        replicas[0], replica, atol=1e-6, rtol=1e-6
                    )
            print(
                f"rank={dist.get_rank()} step={step}: "
                "excluded gradient and weight replicas agree",
                flush=True,
            )

        if cfg.training.compile:
            assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
    finally:
        if trainer is not None:
            trainer.close()

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
