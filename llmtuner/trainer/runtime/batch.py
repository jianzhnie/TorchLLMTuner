"""Trainer batch loading, token accounting, and device-side preprocessing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed import ProcessGroup

from ...accelerator.collectives import all_reduce, synchronize_local_errors
from ...accelerator.distributed import get_comm_device, get_world_size
from ...components.loss import IGNORE_INDEX
from ...datasets.contracts import (
    BaseDataLoader,
    Batch,
    DataloaderExhaustedError,
    TrainerBatch,
)
from ...datasets.loader_builder import build_dataloader as build_dataset_dataloader

if TYPE_CHECKING:
    from ...components.metrics import MetricsProcessor
    from ...config.llmtuner_config import LLMTunerConfig
    from ...parallel.topology import ParallelTopology


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


def resolve_batch_size_per_rank(cfg: LLMTunerConfig, dp_world_size: int) -> int:
    """Divide the global batch without silently dropping a remainder."""
    global_batch_size = cfg.global_batch_size
    if global_batch_size % dp_world_size != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"the number of data-parallel ranks ({dp_world_size}); each rank "
            f"reads global_batch_size // dp_world_size samples, and the "
            f"remainder would be dropped silently."
        )
    return global_batch_size // dp_world_size


def build_training_dataloader(
    cfg: LLMTunerConfig, *, parallel_dims: ParallelTopology | None
) -> BaseDataLoader:
    """Build the recoverable loader for this rank's explicit data ownership."""
    dp_rank, dp_world_size = resolve_data_parallel_rank(parallel_dims)
    batch_size = resolve_batch_size_per_rank(cfg, dp_world_size)
    return build_dataset_dataloader(
        cfg,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        num_tokens_per_batch=batch_size * cfg.max_seq_len,
    )


def fetch_batch(
    data_iterator: Iterator[Batch | TrainerBatch],
    *,
    group: ProcessGroup | None = None,
) -> Batch | TrainerBatch:
    """Fetch one host batch, coordinating exhaustion/errors before model work.

    All group members must call this in the same order (WORLD by default).
    A failed fetch takes precedence over exhaustion. A blocked loader is outside
    this contract; original fetch exceptions remain on the rank that raised them.
    """
    error: BaseException | None = None
    status = 2  # ready; 1 means exhausted, 0 means a fetch/structure failure
    try:
        batch = next(data_iterator)
        labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
        labels.numel()
        if not isinstance(batch, Batch) and batch.get("num_valid_tokens") is not None:
            count_valid_tokens(batch)
    except (DataloaderExhaustedError, StopIteration) as ex:
        error = ex
        status = 1
    except BaseException as ex:
        error = ex
        status = 0

    # Every model axis participates: a PP stage or sequence peer that
    # exits alone leaves the others entering unmatched model collectives.
    # Coordinate before yielding, also before token denominator reductions.
    if get_world_size(group) > 1:
        ready = torch.tensor(status, device=get_comm_device(group), dtype=torch.int32)
        all_reduce(ready, op="min", group=group)
        status = int(ready.item())
    if status == 0:
        if error is not None and not isinstance(
            error, DataloaderExhaustedError | StopIteration
        ):
            raise error
        raise RuntimeError("Data fetch failed on another rank")
    if status == 1:
        raise DataloaderExhaustedError() from error
    return batch


def batch_generator(
    data_iterable: Iterable[Batch | TrainerBatch],
    *,
    metrics: MetricsProcessor,
    group: ProcessGroup | None = None,
) -> Iterator[Batch | TrainerBatch]:
    """Account for each fetch and yield the unchanged host batch.

    Throughput counts all labels, independently of the loss denominator.
    Raise DataloaderExhaustedError on exhaustion so accumulation cannot train
    on a partial step. Coordinate fetch failures across all initialized ranks
    before yielding a batch; this does not handle failures inside model execution."""
    data_iterator = iter(data_iterable)
    while True:
        data_load_start = perf_counter()
        batch = fetch_batch(data_iterator, group=group)
        # Fetch has already coordinated loader errors. Keep local metric
        # callbacks in a separate boundary before peers account tokens or
        # fetch another batch; the loader collective must stay outside it.
        with synchronize_local_errors("Training fetch accounting", group=group):
            labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
            metrics.add_tokens(labels.numel())
            metrics.add_data_loading_time(perf_counter() - data_load_start)
        yield batch


def count_valid_tokens(batch: Batch | TrainerBatch) -> int:
    """Count loss-bearing labels before CP/TP sharding.

    Synthetic Batch targets need next-token shifting. Packed dict labels are
    already aligned; use the collator's count when present, otherwise recount.
    The caller reduces this count across DP only."""
    if isinstance(batch, Batch):
        seq_len = batch.labels.shape[-1]
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        # Each row's first label is never a next-token target; its final
        # prediction is ignored. Counting the remaining labels needs no shift.
        return int((batch.labels[..., 1:] != IGNORE_INDEX).sum())
    num_valid_tokens = batch.get("num_valid_tokens")
    if num_valid_tokens is None:
        # Recounted rather than permissively defaulted, so a dict that
        # silently lacks the key still produces a correct denominator.
        labels = batch["labels"]
        num_valid_tokens = int((labels != IGNORE_INDEX).sum())
    if (
        isinstance(num_valid_tokens, bool)
        or not isinstance(num_valid_tokens, int)
        or not 0 <= num_valid_tokens <= batch["labels"].numel()
    ):
        raise ValueError(
            "num_valid_tokens must be an integer between 0 and the number of labels"
        )
    return num_valid_tokens


def count_routing_tokens(batch: Batch | TrainerBatch) -> int:
    """Count tokens seen by the router, including prompt labels but not padding."""
    if isinstance(batch, Batch):
        return batch.input_ids.numel()
    padding_mask = batch.get("padding_mask")
    if padding_mask is not None:
        return int((~padding_mask.to(torch.bool)).sum())
    return batch["input"].numel()


def prepare_microbatch(
    batch: Batch | TrainerBatch,
    *,
    valid_token_counter: Callable[[Batch | TrainerBatch], int] = count_valid_tokens,
    error_group: ProcessGroup | None = None,
) -> tuple[dict[str, Any], int]:
    """Return host-batch counts and the unsharded label count without mutation.

    The caller owns cumulative training counters and sequence-replica correction.
    Coordinate every local count before callers commit any training state.
    """
    with synchronize_local_errors("Training token accounting", group=error_group):
        labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
        num_tokens = labels.numel()
        num_valid_tokens = valid_token_counter(batch)
        num_routing_tokens = count_routing_tokens(batch)
    return (
        {
            "batch": batch,
            "num_valid_tokens": num_valid_tokens,
            "num_routing_tokens": num_routing_tokens,
        },
        num_tokens,
    )


def move_batch_to_device(
    batch: Batch | TrainerBatch, device: torch.device
) -> Batch | TrainerBatch:
    """Move top-level tensors at consumption, preserving batch metadata."""
    if isinstance(batch, dict):
        return {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
    return Batch(
        input_ids=batch.input_ids.to(device, non_blocking=True),
        labels=batch.labels.to(device, non_blocking=True),
    )


def prepare_model_inputs(
    model: torch.nn.Module,
    batch: Batch | TrainerBatch,
    *,
    device: torch.device,
    parallel_dims: Any,
    max_context_length: int,
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
    prepare_inputs = getattr(model, "prepare_inputs", None)
    if not callable(prepare_inputs):
        raise TypeError(
            f"{type(model).__name__} must implement "
            "prepare_inputs(batch, parallel_dims=..., max_context_length=...)"
        )
    with synchronize_local_errors("Training batch preprocessing", group=error_group):
        return prepare_inputs(
            move_batch_to_device(batch, device),
            parallel_dims=parallel_dims,
            max_context_length=max_context_length,
        )
