"""Batch, prefetch, and checkpoint already composed Grain dataset graphs.

Adapted from torchtitan ``components/data/loader.py``. The loader builder owns
source selection, shuffle, repetition, and DP sharding. GrainDataLoader owns
collation, thread prefetch, and one resumable iterator. Checkpoints validate DP
degree and batch layout before restoring its cursor; failed restoration closes
the iterator while preserving the original error.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import grain.python as grain
import torch
from grain import experimental as grain_experimental

from ..components.tokenizer.base import BaseTokenizer
from ..utils.logging import get_logger
from ..utils.seeding import preserve_rng_state
from .collators import CollatorFactory, TextCollator
from .composition import as_iter_dataset
from .contracts import (
    BaseDataLoader,
    DatasetBuildContext,
    GrainDataset,
    TrainerBatch,
    require_positive_integer,
    require_same_dp_degree,
    validate_dp_coordinates,
    validate_loader_iteration,
)

__all__ = [
    "GrainDataLoader",
]


class GrainDataLoader(BaseDataLoader):
    """Batches and checkpoints one composed Grain dataset graph."""

    def __init__(
        self,
        dataset: GrainDataset,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        collator: CollatorFactory = TextCollator,
        repeat: bool = True,
        read_options: grain.ReadOptions | None = None,
        num_prefetch_batches: int = 2,
        max_num_documents: int | None = None,
        data_identity: str | None = None,
    ) -> None:
        """Batch an already composed graph without changing its data ownership.

        The builder owns shuffle, repetition and DP sharding. Here ``repeat``
        controls dropping partial batches and the finite-DP safety check.
        Runtime context configures the collator and map-to-iterator reads;
        max_num_documents also belongs to the checkpoint's batch-layout identity.
        Prefetch depth must be a positive integer, matching DataloaderConfig.
        ``data_identity`` is an optional stable fingerprint supplied by the
        graph owner. When set, restoration requires the same saved identity;
        a checkpoint without it is rejected rather than assumed equivalent.
        """
        validate_dp_coordinates(dp_rank=dp_rank, dp_world_size=dp_world_size)
        if data_identity is not None and (
            not isinstance(data_identity, str) or not data_identity.strip()
        ):
            raise ValueError("data_identity must be a nonempty string or None")
        require_positive_integer("num_prefetch_batches", num_prefetch_batches)
        validate_loader_iteration(repeat=repeat, dp_world_size=dp_world_size)
        if read_options is None:
            read_options = grain.ReadOptions()
        # Graph ownership is resolved by the builder. The loader records its
        # rank for checkpoint lookup; it cannot infer ownership from a graph.
        self._dp_world_size = dp_world_size
        self._rank_id = f"dp_rank_{dp_rank}"

        # TODO(data-finite-dp): Support finite distributed datasets with a global
        # remainder policy. Simple map datasets can truncate or pad before DP
        # sharding; filtered, mixed, packed, and streaming datasets need
        # coordinated exhaustion so every rank runs the same number of steps.
        context = DatasetBuildContext(
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            num_tokens_per_batch=num_tokens_per_batch,
            read_options=read_options,
            max_num_documents=max_num_documents,
        )

        batch_collator = collator(context=context)
        num_rows_per_batch = batch_collator.num_rows_per_batch()
        require_positive_integer("num_rows_per_batch", num_rows_per_batch)
        self._batch_layout = {
            "num_tokens_per_batch": num_tokens_per_batch,
            "max_context_length": max_context_length,
            "num_rows_per_batch": num_rows_per_batch,
            "max_num_documents": max_num_documents,
        }
        # The graph's owner supplies identity: a loader cannot infer source
        # contents, tokenizer behavior or recipe semantics from a Grain graph.
        if data_identity is not None:
            self._batch_layout["data_identity"] = data_identity

        # TODO(data-multiprocessing): CPU-heavy processing should use multiple
        # processes rather than only threads. Grain can divide map-style data among
        # workers, but packing and mixing map data with a stream produce an iterable
        # before the loader sees it. Investigate an earlier boundary where one
        # shared worker pool processes samples, instead of creating a pool per
        # dataset or packing per worker.
        dataset = as_iter_dataset(dataset, context=context)

        # Batch and collate samples.
        dataset = dataset.batch(
            num_rows_per_batch,
            drop_remainder=repeat,
            batch_fn=batch_collator,
        )

        # Queue completed batches while the trainer consumes the previous batch.
        dataset = grain_experimental.ThreadPrefetchIterDataset(
            dataset, prefetch_buffer_size=num_prefetch_batches
        )
        self._dataset = dataset
        self._iterator = iter(dataset)

    def __iter__(self) -> Iterator[TrainerBatch]:
        return self._iterator

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dp_world_size": self._dp_world_size,
            "batch_layout": self._batch_layout.copy(),
            # Grain may return its own mutable cursor dictionary. Checkpoint
            # readers fill destinations in place, so never expose that alias.
            self._rank_id: deepcopy(self._iterator.get_state()),
        }

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Check metadata and restore a disposable cursor, leaving the live one alone.

        This checks what the graph's set_state accepts. Restoring packing state
        may replay source records even without fetching a batch here. Deferred
        read errors and external source side effects are outside this check.
        """
        if not isinstance(state_dict, dict):
            raise ValueError("GrainDataLoader checkpoint state must be a dictionary")
        if not state_dict:
            return
        version = state_dict.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version != 1:
            raise ValueError(f"unsupported GrainDataLoader state version {version!r}")
        require_same_dp_degree(state_dict.get("dp_world_size"), self._dp_world_size)
        if self._rank_id not in state_dict:
            raise ValueError(
                f"checkpoint is missing dataloader state for {self._rank_id}"
            )
        saved_layout = state_dict.get("batch_layout")
        if saved_layout is None:
            raise ValueError("checkpoint is missing dataloader batch_layout")
        if saved_layout != self._batch_layout:
            raise ValueError(
                "cannot resume after changing dataloader batch layout: "
                f"checkpoint has {saved_layout}, "
                f"current loader has {self._batch_layout}"
            )
        with preserve_rng_state(torch.device("cpu")):
            probe = iter(self._dataset)
            try:
                probe.set_state(deepcopy(state_dict[self._rank_id]))
            except BaseException:
                try:
                    probe.close()
                except BaseException:
                    try:
                        get_logger(__name__).exception(
                            "Dataloader probe cleanup failed "
                            "after cursor validation failed"
                        )
                    except BaseException:
                        pass  # Preserve the original cursor validation error.
                raise
            else:
                probe.close()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        if not state_dict:
            return
        try:
            self._iterator.set_state(state_dict[self._rank_id])
        except BaseException:
            try:
                self.close()
            except BaseException:
                try:
                    get_logger(__name__).exception(
                        "Dataloader cleanup failed after state restoration failed"
                    )
                except BaseException:
                    pass  # Preserve the original cursor restoration error.
            raise

    def close(self) -> None:
        self._iterator.close()
