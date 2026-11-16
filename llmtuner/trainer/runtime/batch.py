"""Trainer batch loading, token accounting, and device-side preprocessing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypedDict

import torch
from torch.distributed import ProcessGroup
from torch.utils._pytree import tree_map_only

from ...accelerator.collectives import all_reduce, synchronize_local_errors
from ...accelerator.distributed import get_comm_device, get_world_size
from ...components.loss import IGNORE_INDEX
from ...config.training_config import resolve_batch_size_per_rank
from ...datasets.contracts import (
    BaseDataLoader,
    Batch,
    DataloaderExhaustedError,
    TrainerBatch,
    validate_token_alignment,
    validated_token_fields,
)
from ...datasets.loader_builder import build_dataloader as build_dataset_dataloader
from ...models.execution import validate_prepared_model_inputs

if TYPE_CHECKING:
    from ...components.metrics import MetricsProcessor
    from ...config.llmtuner_config import LLMTunerConfig
    from ...parallel.topology import ParallelTopology


def resolve_data_parallel_rank(
    parallel_dims: ParallelTopology | None,
) -> tuple[int, int]:
    """Return the rank and size of the data-parallel loader group."""
    if parallel_dims is None:
        return 0, 1
    mesh = parallel_dims.get_optional_mesh("dp", include_singleton_axes=True)
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

    # Every model axis participates: a sequence peer that
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
    return _validate_valid_token_count(num_valid_tokens, batch["labels"].numel())


def _validate_valid_token_count(count: int, label_count: int) -> int:
    """Apply the same denominator bounds to collators and injected counters."""
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= label_count
    ):
        raise ValueError(
            "num_valid_tokens must be an integer between 0 and the number of labels"
        )
    return count


def validated_batch_token_counts(
    batch: Batch | TrainerBatch,
    *,
    token_counter: Callable[[Batch | TrainerBatch], int],
) -> tuple[int, int]:
    """Validate raw fields and local counts before any token reduction.

    This operation is local; training and validation own their respective
    distributed error boundaries and must call it inside those boundaries.
    """
    _, labels = validated_token_fields(batch)
    num_tokens = labels.numel()
    num_valid_tokens = _validate_valid_token_count(token_counter(batch), num_tokens)
    return num_tokens, num_valid_tokens


def count_routing_tokens(batch: Batch | TrainerBatch) -> int:
    """Count tokens seen by the router, including prompt labels but not padding."""
    if isinstance(batch, Batch):
        return batch.input_ids.numel()
    padding_mask = batch.get("padding_mask")
    if padding_mask is not None:
        validate_token_alignment(batch["input"], padding_mask, name="padding_mask")
        return int((~padding_mask.to(torch.bool)).sum())
    return batch["input"].numel()


class AccumulationBatch(TypedDict):
    """Host batch and unsharded counts for one gradient-accumulation group.

    Valid targets normalize the LM loss; routing tokens normalize auxiliary
    losses and include tokens whose LM targets are ignored.
    """

    batch: Batch | TrainerBatch
    num_valid_tokens: int
    num_routing_tokens: int


def prepare_microbatch(
    batch: Batch | TrainerBatch,
    *,
    valid_token_counter: Callable[[Batch | TrainerBatch], int] = count_valid_tokens,
    error_group: ProcessGroup | None = None,
) -> tuple[AccumulationBatch, int]:
    """Return host-batch counts and the unsharded label count without mutation.

    The caller owns cumulative training counters and sequence-replica correction.
    Coordinate every local count before callers commit any training state.
    """
    with synchronize_local_errors("Training token accounting", group=error_group):
        num_tokens, num_valid_tokens = validated_batch_token_counts(
            batch, token_counter=valid_token_counter
        )
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
    """Move input tensor leaves at consumption, preserving nested metadata.

    Dict/list/tuple containers follow PyTorch's pytree registrations. Opaque
    metadata remains unchanged; models own its interpretation and validation.
    """

    def move(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device, non_blocking=True)

    if isinstance(batch, dict):
        return tree_map_only(torch.Tensor, move, batch)
    return Batch(input_ids=move(batch.input_ids), labels=move(batch.labels))


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
