"""Compare actual TP2 x CP2 Trainer updates with independent dense training.

Run four ranks with ``kv_allgather`` or ``ulysses`` as the first argument.
NPU uses the production attention path. CPU explicitly substitutes dense-mask
attention and only checks the training and collective contracts.
This small FP32 fixture does not establish target-model or P4 acceptance.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext

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
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.tensor_parallel.tensor_parallel import (
    ColumnParallelLinear,
    ColwiseLinearNoGather,
    RowParallelLinear,
)
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn
from tests.caps import require_env
from tests.integration_tests.harness import backend, device, moved_batch
from tests.integration_tests.pp_cp_equivalence import cpu_cp_attention

STEPS = 3
ATOL = 1e-5
RTOL = 1e-4


def configuration(strategy: str) -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="llama",
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
            global_batch_size=4,
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
    require_env("dtensor")
    strategy = sys.argv[1] if len(sys.argv) > 1 else "kv_allgather"
    if strategy not in {"kv_allgather", "ulysses"}:
        raise ValueError("Expected kv_allgather or ulysses")
    cfg = configuration(strategy)
    context = cpu_cp_attention() if device().type == "cpu" else nullcontext()
    with context:
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
            oracle = HFTransformerModel(build_model_config_for(cfg)).to(trainer.device)
            optimizer = torch.optim.AdamW(
                oracle.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
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
            reference_batches = iter(
                RandomTokenSource(
                    seed=cfg.seed,
                    vocab_size=cfg.vocab_size,
                    batch_size=cfg.global_batch_size,
                    seq_len=cfg.max_seq_len,
                )
            )
            iterator = trainer.data_iterator()
            for step in range(1, STEPS + 1):
                optimizer.zero_grad(set_to_none=True)
                inputs, labels, _ = oracle.preprocess_inputs(
                    moved_batch(next(reference_batches)), parallel_dims=None
                )
                logits = oracle(inputs)
                reference_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
                )
                reference_loss.backward()
                reference_norm = clip_grad_norm_(
                    oracle.parameters(), max_norm=cfg.max_norm, foreach=True
                )
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
                        msg=lambda error, step=step, key=key: (
                            f"step={step} {key}: {error}"
                        ),
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
