"""Trainer batch loading, token accounting, and device-side preprocessing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

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
    from ..trainer import Trainer


def dp_rank_world_size(self: Trainer) -> tuple[int, int]:
    """Return the rank and size of the loader's data-parallel axis.

    PP uses the batch mesh so every stage reads the same shard. Without a
    topology, including before mesh initialization, use a single rank."""
    parallel_dims = getattr(self, "parallel_dims", None)
    if parallel_dims is None:
        return 0, 1
    # Both views span DP replicate * shard. PP's batch view excludes stages,
    # ensuring each stage reads the same shard of the global batch.
    mesh = parallel_dims.get_optional_mesh(
        "batch" if parallel_dims.pp_enabled else "dp", include_singleton_axes=True
    )
    return mesh.get_local_rank(), mesh.size()


def batch_size_per_rank(self: Trainer, dp_world_size: int) -> int:
    """Divide the configured global batch across DP ranks without dropping rows."""
    global_batch_size = self.cfg.global_batch_size
    if global_batch_size % dp_world_size != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"the number of data-parallel ranks ({dp_world_size}); each rank "
            f"reads global_batch_size // dp_world_size samples, and the "
            f"remainder would be dropped silently."
        )
    return global_batch_size // dp_world_size


def build_dataloader(self: Trainer) -> BaseDataLoader:
    """Build a checkpointable loader with this rank's token budget."""
    dp_rank, dp_world_size = self.dp_rank_world_size()
    batch_size_per_rank = self.batch_size_per_rank(dp_world_size)
    return build_dataset_dataloader(
        self.cfg,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        # Per rank, not global: the Grain loader splits every dataset's
        # rows across ``dp_world_size`` ranks itself, so this many tokens
        # per rank is this many tokens per rank of the global batch.
        num_tokens_per_batch=batch_size_per_rank * self.cfg.max_seq_len,
    )


def data_iterator(self: Trainer) -> Iterator[Batch | TrainerBatch]:
    """Iterate the initialized loader; reject missing runtime assembly."""
    if self.dataloader is None:
        raise RuntimeError("Trainer dataloader has not been initialized")
    return iter(self.dataloader)


def fetch_batch(
    data_iterator: Iterator[Batch | TrainerBatch],
) -> Batch | TrainerBatch:
    """Fetch one host batch, coordinating exhaustion/errors before model work.

    All world ranks must call this in the same order. A failed fetch takes
    precedence over exhaustion. Model execution and a blocked loader are outside
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
    if get_world_size() > 1:
        ready = torch.tensor(status, device=get_comm_device(), dtype=torch.int32)
        all_reduce(ready, op="min")
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
    self: Trainer, data_iterable: Iterable[Batch | TrainerBatch]
) -> Iterator[Batch | TrainerBatch]:
    """Account for each fetch and yield the unchanged host batch.

    Throughput counts all labels, independently of the loss denominator.
    Raise DataloaderExhaustedError on exhaustion so accumulation cannot train
    on a partial step. Coordinate fetch failures across all initialized ranks
    before yielding a batch; this does not handle failures inside model execution."""
    data_iterator = iter(data_iterable)
    while True:
        data_load_start = perf_counter()
        batch = fetch_batch(data_iterator)
        # Fetch has already coordinated loader errors. Keep local metric
        # callbacks in a separate boundary before peers account tokens or
        # fetch another batch; the loader collective must stay outside it.
        with synchronize_local_errors("Training fetch accounting"):
            labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
            self.metrics.add_tokens(labels.numel())
            self.metrics.add_data_loading_time(perf_counter() - data_load_start)
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


def microbatch(self: Trainer, batch: Batch | TrainerBatch) -> dict[str, Any]:
    """Attach valid-token and routing-token counts to one host batch.

    Throughput divides by CP * TP because those ranks read the same batch.
    Keep tensors on the host until preprocess consumes this accumulation group."""
    labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
    # Count this rank's share of the sequence, not the whole batch: every
    # rank of a CP/TP group reads the *same* batch from the loader, and the
    # sequence is cut across the group later (``shard_batch_for_cp`` /
    # ``shard_batch_for_tp``). Counting the full batch here and then summing
    # over the dp*cp*tp loss mesh in ``train_step`` would report a corpus
    # cp * tp times its true size; counting the share makes that sum
    # reconstruct the tokens actually read.
    parallel_dims = getattr(self, "parallel_dims", None)
    sequence_shards = (
        1 if parallel_dims is None else parallel_dims.cp * parallel_dims.tp
    )
    with synchronize_local_errors("Training token accounting"):
        num_valid_tokens = self.count_valid_tokens(batch)
        num_routing_tokens = count_routing_tokens(batch)
    # Commit accounting only after both counters have been validated.
    self.ntokens_seen += labels.numel() // sequence_shards

    return {
        "batch": batch,
        "num_valid_tokens": num_valid_tokens,
        "num_routing_tokens": num_routing_tokens,
    }


def to_device(self: Trainer, batch: Batch | TrainerBatch) -> Batch | TrainerBatch:
    """Move a consumption group's top-level tensors, preserving batch metadata."""
    if isinstance(batch, dict):
        return {
            key: value.to(self.device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
    return Batch(
        input_ids=batch.input_ids.to(self.device, non_blocking=True),
        labels=batch.labels.to(self.device, non_blocking=True),
    )


def preprocess(
    self: Trainer, batch: Batch | TrainerBatch
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Move the batch and ask the stage's canonical model to prepare inputs.

    Remove collator token-count metadata before model preprocessing. Transfer
    occurs at consumption so unread accumulation groups stay in host memory."""
    if isinstance(batch, dict):
        batch = {
            key: value for key, value in batch.items() if key != "num_valid_tokens"
        }
    # HF preprocessing builds local masks and slices already-replicated inputs;
    # attention collectives start only in model execution after this boundary.
    with synchronize_local_errors("Training batch preprocessing"):
        result = self.example_model.preprocess_inputs(
            self.to_device(batch),
            parallel_dims=self.parallel_dims,
            max_context_length=self.cfg.max_seq_len,
        )
    return result
