"""Coordinate rank-local CP plan errors across dense and pipeline assembly.

Run two ranks, or four with --pp-size 2. --schedule Interleaved1F1B checks
multiple local chunks. An FSDP entry sentinel makes rejection failures finite;
afterwards a fresh real Trainer must complete two updates without that sentinel.
This small dense fixture checks failure boundaries, not target-model acceptance.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.observability_config import MetricsConfig
from llmtuner.config.parallel_config import (
    ContextParallelConfig,
    ParallelConfig,
    PipelineParallelConfig,
)
from llmtuner.config.training_config import TrainingConfig
from llmtuner.models.configuration import build_model_config_for
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel import parallelize
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.harness import init_distributed


def configuration(pp_size: int, schedule: str) -> LLMTunerConfig:
    layers = 6 if schedule == "Interleaved1F1B" else 2
    microbatches = 4 if schedule == "Interleaved1F1B" else 2
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            arch_overrides={
                "head_dim": 8,
                "num_experts": 4,
                "moe_intermediate_size": 16,
                "mlp_only_layers": list(range(layers)),
            },
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=pp_size,
            context_parallel_ulysses_size=2,
            cp_config=ContextParallelConfig(context_parallel_strategy="ulysses"),
            pp_config=PipelineParallelConfig(
                pipeline_parallel_schedule=schedule,
                num_pp_microbatches=microbatches,
            ),
        ),
        training=TrainingConfig(
            global_batch_size=microbatches if pp_size > 1 else 1,
            max_seq_len=32,
            steps=2,
            seed=97,
            metrics_config=MetricsConfig(log_freq=1),
            checkpoint_config=CheckpointConfig(enable=False),
        ),
    )


def reject_fault(cfg, topology, device, *, fault: str, faulty_rank: int) -> None:
    """Both the faulty rank and every peer must stop before FSDP entry."""
    torch.manual_seed(97)
    model = Qwen3MoeForCausalLM(build_model_config_for(cfg)).to(device)
    if dist.get_rank() == faulty_rank:
        if fault == "attention":
            model.decoder_structure = replace(
                model.decoder_structure, attention="missing_attention"
            )
        else:
            model.config.num_attention_heads = 3
    parameters = tuple(model.parameters())
    values = tuple(parameter.detach().clone() for parameter in parameters)
    implementation = model.config._attn_implementation
    entered_fsdp = False

    def forbidden_fsdp(*args, **kwargs):
        nonlocal entered_fsdp
        entered_fsdp = True
        raise AssertionError("CP failure reached FSDP")

    caught = None
    with patch.object(parallelize, "apply_fsdp", forbidden_fsdp):
        try:
            parallelize.parallelize_model(
                model,
                cfg=cfg.parallel,
                mesh=topology.spmd_dense_mesh(),
                parallel_dims=topology,
                device=device,
                global_batch_size=cfg.global_batch_size,
            )
        except (ValueError, RuntimeError, AssertionError) as error:
            caught = error
    outcomes = [None] * dist.get_world_size()
    dist.all_gather_object(outcomes, (type(caught).__name__, str(caught), entered_fsdp))
    for rank, (kind, message, entered) in enumerate(outcomes):
        assert not entered, outcomes
        if rank == faulty_rank:
            assert kind == "ValueError", outcomes
            assert (
                "missing_attention" if fault == "attention" else "divisible"
            ) in message
        else:
            assert kind == "RuntimeError", outcomes
            assert "CP plan validation failed on another rank" in message, outcomes
    assert model.config._attn_implementation == implementation
    assert all(
        current is previous
        for current, previous in zip(model.parameters(), parameters, strict=True)
    )
    for parameter, value in zip(parameters, values, strict=True):
        torch.testing.assert_close(parameter, value, rtol=0, atol=0)
    assert all(
        not hasattr(module, "_titan_cp_sdpa_kernel") for module in model.modules()
    )
    print(f"rank={dist.get_rank()} fault={fault} faulty_rank={faulty_rank} coordinated")


def local(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--schedule", choices=("1F1B", "Interleaved1F1B"), default="1F1B"
    )
    args = parser.parse_args()
    device = init_distributed(timeout=timedelta(seconds=60))
    try:
        assert dist.get_world_size() == 2 * args.pp_size
        cfg = configuration(args.pp_size, args.schedule)
        topology = ParallelTopology.from_config(cfg.parallel, dist.get_world_size())
        topology.build_device_mesh()
        for faulty_rank in (0, dist.get_world_size() - 1):
            for fault in ("attention", "heads"):
                reject_fault(
                    cfg, topology, device, fault=fault, faulty_rank=faulty_rank
                )
        trainer = Trainer(cfg)
        try:
            iterator = trainer.data_iterator()
            parameters = [p for part in trainer.model_parts for p in part.parameters()]
            assert parameters
            for step in range(1, 3):
                before = [local(p).detach().clone() for p in parameters]
                trainer.step = step
                metrics = trainer.train_step(iterator)
                if metrics is not None:
                    assert math.isfinite(metrics["loss"]) and metrics["loss"] > 0
                    assert (
                        math.isfinite(metrics["grad_norm"]) and metrics["grad_norm"] > 0
                    )
                assert any(
                    not torch.equal(local(p), value)
                    for p, value in zip(parameters, before, strict=True)
                ), "Fresh Trainer did not update local parameters"
            print(
                f"rank={trainer.rank} pp={args.pp_size} cp=2 {args.schedule} "
                "fresh real Trainer completed two updates"
            )
        finally:
            trainer.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
