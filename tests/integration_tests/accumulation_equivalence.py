"""Unequal-token accumulation must match an independently normalized loss sum.

Run: PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/accumulation_equivalence.py

Exercise actual Trainer.train_step, device-selected FSDP, three AdamW updates.
The Trainer selects CPU/Gloo or NPU/HCCL from the installed runtime.
Use --nproc_per_node=4 for HSDP replicate=2/shard=2; the manual reference
reduces every backward, while train_step reduces replicas only on the last.
Masks differ across ranks and accumulation groups; compare reported loss,
post-backward gradients and updated parameters against a manual loss-sum run.
Keep separate forwards: Batch rows flatten into one causal stream, so joining
microbatches changes attention context and is not an equivalent reference.
Set ACCUMULATION_SWAP=1 on NPU to swap only the accumulated path's activations.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.accelerator.distributed import get_comm_device
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.datasets.contracts import Batch
from llmtuner.trainer.trainer import Trainer


def config(accumulation: int) -> LLMTunerConfig:
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    if world_size not in (2, 4):
        raise ValueError("accumulation test requires 2 FSDP or 4 HSDP ranks")
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        parallel=ParallelConfig(
            data_parallel_shard_size=2,
            data_parallel_replicate_size=world_size // 2,
        ),
        optimizer=OptimizerConfig(learning_rate=1e-3),
        training=TrainingConfig(
            global_batch_size=2 * world_size,
            max_seq_len=8,
            steps=3,
            seed=42,
            max_norm=0,
            gradient_accumulation_steps=accumulation,
            activation_swap=(
                accumulation > 1 and os.environ.get("ACCUMULATION_SWAP") == "1"
            ),
        ),
    )


def local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def main() -> None:
    swap = os.environ.get("ACCUMULATION_SWAP") == "1"
    if swap:
        from llmtuner.memory.activation_swap import reset_activation_swap_metrics

        reset_activation_swap_metrics()
    accumulated = Trainer(config(2))
    combined = None
    try:
        combined = Trainer(config(1))
        # Explicit identical initial state isolates accumulation from runtime RNG.
        combined.model.load_state_dict(accumulated.model.state_dict())
        assert dist.get_world_size() in (2, 4)
        for step in range(1, 4):
            generator = torch.Generator().manual_seed(100 * step + dist.get_rank())
            batches = []
            for index in range(2):
                tokens = torch.randint(32, (2, 8), generator=generator)
                labels = torch.randint(32, (2, 8), generator=generator)
                labels[:, : 1 + index + 2 * dist.get_rank()] = -100
                batches.append(Batch(input_ids=tokens, labels=labels))
            accumulated.step = combined.step = step
            combined.optimizer.zero_grad(set_to_none=True)
            denominator = torch.tensor(
                sum(int((batch.labels[:, 1:] != -100).sum()) for batch in batches),
                dtype=torch.int64,
                device=get_comm_device(),
            )
            dist.all_reduce(denominator)
            sums = []
            for batch in batches:
                inputs, labels, kwargs = combined.preprocess(batch)
                logits = combined.model(inputs, **kwargs)
                # Independent cross entropy and denominator, outside train_step.
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 32), labels.reshape(-1), reduction="sum"
                )
                (loss / denominator).backward()
                sums.append(loss.detach())
            reference_loss = torch.stack(sums).sum() / denominator
            dist.all_reduce(reference_loss)
            combined.optimizer.step()
            combined.lr_scheduler.step()
            metrics = accumulated.train_step(iter(batches))
            torch.testing.assert_close(
                torch.tensor(metrics["loss"], device=reference_loss.device),
                reference_loss,
                atol=2e-6,
                rtol=2e-5,
            )
            expected = dict(combined.model.named_parameters())
            for name, parameter in accumulated.model.named_parameters():
                torch.testing.assert_close(
                    local(parameter.grad),
                    local(expected[name].grad),
                    atol=2e-6,
                    rtol=2e-5,
                )
                torch.testing.assert_close(
                    local(parameter),
                    local(expected[name]),
                    atol=2e-6,
                    rtol=2e-5,
                )
        if swap:
            from llmtuner.memory.activation_swap import activation_swap_metrics

            transfers = activation_swap_metrics()
            assert transfers["copies_d2h"] > 0, transfers
            assert transfers["copies_h2d"] > 0, transfers
            assert transfers["host_bytes_current"] == 0, transfers
            print(f"PASS accumulated activation swap: {transfers}")
        if dist.get_rank() == 0:
            print("PASS unequal-token accumulation: losses, gradients, AdamW updates")
    finally:
        if combined is not None:
            combined.close()
        accumulated.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
