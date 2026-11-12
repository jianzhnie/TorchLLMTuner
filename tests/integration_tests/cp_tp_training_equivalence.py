"""Compare actual TP2 x CP2 Trainer updates with independent dense training.

Run four ranks with ``kv_allgather`` or ``ulysses`` as the first argument.
Both CPU and NPU use the explicit model and production SDPA CP kernels.
Single-row input preserves the explicit model's supported CP/TP boundary.
This small FP32 fixture does not establish target-model or P4 acceptance.
"""

from __future__ import annotations

import sys
from itertools import chain

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.observability_config import MetricsConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.parallel_config import (
    ContextParallelConfig,
    FSDPConfig,
    ParallelConfig,
)
from llmtuner.config.training_config import TrainingConfig
from llmtuner.datasets.synthetic import RandomTokenSource
from llmtuner.models.configuration import build_model_config_for
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.tensor_parallel.tensor_parallel import (
    ColumnParallelLinear,
    ColwiseLinearNoGather,
    RowParallelLinear,
)
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.harness import backend, moved_batch

STEPS = 3
ATOL = 1e-5
RTOL = 1e-4


def configuration(strategy: str) -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            arch_overrides={
                "head_dim": 16,
                "num_experts": 4,
                "moe_intermediate_size": 32,
                "mlp_only_layers": [0, 1],
            },
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            tensor_parallel_size=2,
            cp_config=ContextParallelConfig(context_parallel_strategy=strategy),
            context_parallel_kv_allgather_size=2 if strategy == "kv_allgather" else 1,
            context_parallel_ulysses_size=2 if strategy == "ulysses" else 1,
            data_parallel_shard_size=1,
            fsdp_config=FSDPConfig(fsdp_implementation="native"),
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            # Explicit Qwen3 requires a single row for CP/TP input sharding.
            global_batch_size=1,
            max_seq_len=128,
            steps=STEPS,
            seed=42,
            deterministic=True,
            metrics_config=MetricsConfig(log_freq=1),
            checkpoint_config=CheckpointConfig(enable=False),
        ),
    )


def full(value: torch.Tensor) -> torch.Tensor:
    return value.full_tensor() if isinstance(value, DTensor) else value


def main() -> None:
    strategy = sys.argv[1] if len(sys.argv) > 1 else "kv_allgather"
    if strategy not in {"kv_allgather", "ulysses"}:
        raise ValueError("Expected kv_allgather or ulysses")
    cfg = configuration(strategy)
    trainer = Trainer(cfg)
    try:
        assert trainer.world_size == 4
        assert trainer.parallel_dims.tp == trainer.parallel_dims.cp == 2
        assert trainer.dp_rank_world_size() == (0, 1)
        model = trainer.model_parts[0]
        layouts = {}
        for name, module in model.named_modules():
            if isinstance(module, ColumnParallelLinear | ColwiseLinearNoGather):
                layouts[f"{name}.weight"] = 0
            elif isinstance(module, RowParallelLinear):
                layouts[f"{name}.weight"] = 1
        assert layouts, "TP projections were not installed"
        tp_rank = trainer.parallel_dims.get_mesh("tp").get_local_rank()

        torch.manual_seed(cfg.seed)
        oracle = Qwen3MoeForCausalLM(build_model_config_for(cfg)).to(trainer.device)
        optimizer = torch.optim.AdamW(
            oracle.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=cfg.optimizer.implementation == "fused",
            foreach=cfg.optimizer.implementation == "foreach",
        )
        oracle_parameters = dict(oracle.named_parameters())

        def expected(name: str, *, gradient: bool) -> torch.Tensor:
            parameter = oracle_parameters[canonical_fqn(name)]
            value = parameter.grad if gradient else parameter
            assert value is not None, name
            axis = layouts.get(name)
            return value if axis is None else value.chunk(2, dim=axis)[tp_rank]

        # Shared seed alone does not prove identical initialization.
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                full(parameter), expected(name, gradient=False), atol=0, rtol=0
            )
        initial_parameters = {
            name: full(parameter).detach().clone()
            for name, parameter in model.named_parameters()
        }
        reference_batches = iter(
            RandomTokenSource(
                seed=cfg.seed,
                vocab_size=cfg.vocab_size,
                batch_size=cfg.global_batch_size,
                seq_len=cfg.max_seq_len,
            )
        )
        iterator = trainer.data_iterator()
        first_training_batch = next(iterator)
        first_reference_batch = next(reference_batches)
        if isinstance(first_training_batch, dict):
            raise AssertionError("synthetic training expected a Batch object")
        torch.testing.assert_close(
            first_training_batch.input_ids,
            first_reference_batch.input_ids,
            atol=0,
            rtol=0,
            msg="trainer and reference received different token batches",
        )
        torch.testing.assert_close(
            first_training_batch.labels,
            first_reference_batch.labels,
            atol=0,
            rtol=0,
            msg="trainer and reference received different label batches",
        )
        reference_inputs, reference_labels, reference_kwargs = oracle.prepare_inputs(
            moved_batch(first_reference_batch), parallel_dims=None
        )
        cp_rank = trainer.parallel_dims.get_mesh("cp").get_local_rank()
        expected_inputs = reference_inputs.chunk(2)[cp_rank].chunk(2)[tp_rank]
        expected_labels = reference_labels.chunk(2)[cp_rank].chunk(2)[tp_rank]
        actual_inputs, actual_labels, actual_kwargs = model.prepare_inputs(
            moved_batch(first_training_batch), parallel_dims=trainer.parallel_dims
        )
        torch.testing.assert_close(
            actual_inputs,
            expected_inputs,
            atol=0,
            rtol=0,
            msg="TP/CP preprocessing produced different input tokens",
        )
        torch.testing.assert_close(
            actual_labels,
            expected_labels,
            atol=0,
            rtol=0,
            msg="TP/CP preprocessing produced different target tokens",
        )
        reference_positions = reference_kwargs.get("positions")
        if reference_positions is None:
            reference_positions = torch.arange(
                reference_inputs.numel(), device=reference_inputs.device
            )
        expected_positions = reference_positions.chunk(2)[cp_rank]
        torch.testing.assert_close(
            actual_kwargs["positions"],
            expected_positions,
            atol=0,
            rtol=0,
            msg="TP/CP preprocessing produced different rotary positions",
        )
        iterator = chain((first_training_batch,), iterator)
        reference_batches = chain((first_reference_batch,), reference_batches)
        for step in range(1, STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            inputs, labels, kwargs = oracle.prepare_inputs(
                moved_batch(next(reference_batches)), parallel_dims=None
            )
            logits = oracle(inputs, **kwargs).logits
            reference_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
            reference_loss.backward()
            reference_norm = clip_grad_norm_(
                oracle.parameters(), max_norm=cfg.max_norm, foreach=True
            )
            assert torch.isfinite(reference_loss) and torch.isfinite(reference_norm)
            assert reference_norm > 0, "reference backward produced no gradients"
            trainer.step = step
            metrics = trainer.train_step(iterator)
            assert metrics is not None
            for key, want in (
                ("loss", reference_loss.detach()),
                ("grad_norm", reference_norm),
            ):
                torch.testing.assert_close(
                    torch.tensor(metrics[key], device=trainer.device),
                    want,
                    atol=ATOL,
                    rtol=RTOL,
                    msg=lambda error, step=step, key=key: f"step={step} {key}: {error}",
                )
            optimizer.step()
            checked = 0
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                for value, gradient in ((parameter.grad, True), (parameter, False)):
                    torch.testing.assert_close(
                        full(value),
                        expected(name, gradient=gradient),
                        atol=ATOL,
                        rtol=RTOL,
                        msg=lambda error, step=step, name=name: (
                            f"step={step} {name}: {error}"
                        ),
                    )
                checked += 1
            assert checked == len(oracle_parameters)
            assert any(
                not torch.equal(full(parameter), initial_parameters[name])
                for name, parameter in model.named_parameters()
            ), "Trainer did not update any parameter"
            print(
                f"rank={trainer.rank} strategy={strategy} step={step} "
                f"device={trainer.device.type} backend={backend()} "
                f"parameters={checked} loss/norm/gradients/updates passed",
                flush=True,
            )
    finally:
        trainer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
