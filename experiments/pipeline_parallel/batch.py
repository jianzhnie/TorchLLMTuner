"""Trainer batch loading, token accounting, and device-side preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.distributed import ProcessGroup

from experiments.pipeline_parallel.loader_builder import (
    build_dataloader as build_dataset_dataloader,
)
from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.config.training_config import resolve_batch_size_per_rank
from llmtuner.datasets.contracts import (
    BaseDataLoader,
    Batch,
    TrainerBatch,
)
from llmtuner.models.execution import validate_prepared_model_inputs
from llmtuner.trainer.runtime.batch import (
    AccumulationBatch,
    batch_generator,
    count_routing_tokens,
    count_valid_tokens,
    fetch_batch,
    move_batch_to_device,
    prepare_microbatch,
    validated_batch_token_counts,
)

if TYPE_CHECKING:
    from experiments.pipeline_parallel.topology import ParallelTopology
    from llmtuner.config.llmtuner_config import LLMTunerConfig


__all__ = [
    "AccumulationBatch",
    "batch_generator",
    "build_rank_dataloader",
    "count_routing_tokens",
    "count_valid_tokens",
    "fetch_batch",
    "move_batch_to_device",
    "prepare_microbatch",
    "prepare_model_inputs",
    "resolve_data_parallel_rank",
    "validated_batch_token_counts",
]


def resolve_data_parallel_rank(
    parallel_dims: ParallelTopology | None,
) -> tuple[int, int]:
    """Return the loader rank, using PP's stage-shared batch view when needed."""
    if parallel_dims is None:
        return 0, 1
    mesh = parallel_dims.get_optional_mesh(
        "batch" if parallel_dims.pp_enabled else "dp", include_singleton_axes=True
    )
    return mesh.get_local_rank(), mesh.size()


def build_rank_dataloader(
    cfg: LLMTunerConfig,
    *,
    parallel_dims: ParallelTopology | None,
    repeat: bool = True,
    dataset: str | None = None,
) -> BaseDataLoader:
    """Build a loader using shared DP ownership and per-rank token geometry."""
    dp_rank, dp_world_size = resolve_data_parallel_rank(parallel_dims)
    batch_size = resolve_batch_size_per_rank(cfg.global_batch_size, dp_world_size)
    return build_dataset_dataloader(
        cfg,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        num_tokens_per_batch=batch_size * cfg.max_seq_len,
        repeat=repeat,
        dataset=dataset,
    )


def prepare_model_inputs(
    model: torch.nn.Module,
    batch: Batch | TrainerBatch,
    *,
    device: torch.device,
    parallel_dims: Any,
    error_group: ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Connect a batch to the explicit model's input contract.

    Collator token counts are not model inputs. Unread accumulation groups stay
    on the host until this function consumes them.
    """
    if isinstance(batch, dict):
        batch = {
            key: value for key, value in batch.items() if key != "num_valid_tokens"
        }
    with synchronize_local_errors("Training batch preprocessing", group=error_group):
        prepare_inputs = getattr(model, "prepare_inputs", None)
        if not callable(prepare_inputs):
            raise TypeError(
                f"{type(model).__name__} must implement "
                "prepare_inputs(batch, parallel_dims=...)"
            )
        prepared = prepare_inputs(
            move_batch_to_device(batch, device),
            parallel_dims=parallel_dims,
        )
        return validate_prepared_model_inputs(prepared)
