"""Shared data contracts: build context, iteration policy, samples, and batches.

Processors, collators, and loaders import these types without depending on the
recipe composition or loader assembly modules. Runtime context and iteration
policy are adapted from torchtitan ``components/data/types.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import grain.python as grain
import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful

from ..components.tokenizer.base import BaseTokenizer

__all__ = [
    "BaseDataLoader",
    "Batch",
    "DataloaderExhaustedError",
    "DatasetBuildContext",
    "DatasetIterationPolicy",
    "GrainDataset",
    "SampleProcessor",
    "SampleProcessorFactory",
    "TextSequence",
    "TrainerBatch",
    "require_positive_integer",
    "require_same_dp_degree",
    "validate_dp_coordinates",
    "validate_loader_iteration",
]

# Model inputs, aligned labels, and num_valid_tokens metadata. Collators count
# valid labels while constructing batches; the trainer removes this count
# before model preprocessing. Synthetic Batch targets are shifted separately.
TrainerBatch: TypeAlias = dict[str, Any]
GrainDataset: TypeAlias = grain.MapDataset | grain.IterDataset


class DataloaderExhaustedError(Exception):
    """Signal exhausted training data without PEP 479's StopIteration wrapping."""


class BaseDataLoader(Stateful, ABC):
    """Checkpointable batch iterator with optional resource cleanup."""

    @abstractmethod
    def __iter__(self) -> Iterator[Batch | TrainerBatch]: ...

    def close(self) -> None:
        pass


def require_positive_integer(name: str, value: int) -> None:
    """Require a positive integer size, rejecting booleans and non-integers.

    The data pipeline's one shape check, shared by the dataclasses below and by
    the builders that take the same numbers as plain parameters (the loader's
    ``max_num_documents``, the packing builders' ``num_packing_bins``, the
    synthetic loader's ``dp_world_size``). It used to be three copied checks --
    with the message text spelled out at each site, so the wording could drift
    between the two entry points for the same value.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive integer, got {value!r}")


def require_same_dp_degree(saved_dp_world_size: int, dp_world_size: int) -> None:
    """Reject checkpoint slices written under another data-parallel degree."""
    require_positive_integer("checkpoint dp_world_size", saved_dp_world_size)
    if saved_dp_world_size != dp_world_size:
        raise ValueError(
            "cannot resume after changing the effective data-parallel degree, "
            f"checkpoint has dp_world_size={saved_dp_world_size}, "
            f"current run has dp_world_size={dp_world_size}"
        )


def validate_dp_coordinates(*, dp_rank: int, dp_world_size: int) -> None:
    """Validate integer rank ownership before constructing a data stream."""
    require_positive_integer("dp_world_size", dp_world_size)
    if (
        not isinstance(dp_rank, int)
        or isinstance(dp_rank, bool)
        or not 0 <= dp_rank < dp_world_size
    ):
        raise ValueError(
            f"dp_rank must be an integer in [0, {dp_world_size}), got {dp_rank!r}"
        )


def validate_loader_iteration(*, repeat: bool, dp_world_size: int) -> None:
    """Reject loader policies that can exhaust DP ranks at different steps.

    Finite distributed loading needs coordinated exhaustion or a global
    remainder policy; dropping a local partial batch alone is insufficient.
    """
    if not isinstance(repeat, bool):
        raise TypeError(f"repeat must be bool, got {repeat!r}")
    if dp_world_size > 1 and not repeat:
        raise ValueError(
            "repeat=False with data parallelism can exhaust ranks at different "
            "steps and hang collectives; use repeat=True with a trainer-"
            "controlled step count"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetBuildContext:
    """Runtime values shared while building the data pipeline."""

    tokenizer: BaseTokenizer
    max_context_length: int
    num_tokens_per_batch: int
    read_options: grain.ReadOptions
    max_num_documents: int | None = None

    def __post_init__(self) -> None:
        require_positive_integer("max_context_length", self.max_context_length)
        require_positive_integer("num_tokens_per_batch", self.num_tokens_per_batch)
        if self.max_num_documents is not None:
            require_positive_integer("max_num_documents", self.max_num_documents)


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetIterationPolicy:
    """Controls dataset order, repetition, and data-parallel ownership."""

    seed: int
    shuffle: bool
    repeat: bool
    dp_rank: int
    dp_world_size: int
    streaming_shuffle_buffer_size: int

    def __post_init__(self) -> None:
        for name in ("shuffle", "repeat"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool, got {value!r}")
        validate_dp_coordinates(dp_rank=self.dp_rank, dp_world_size=self.dp_world_size)
        require_positive_integer(
            "streaming_shuffle_buffer_size", self.streaming_shuffle_buffer_size
        )


@dataclass
class Batch:
    """One micro-batch, on CPU: ``input_ids`` and ``labels`` of shape ``(B, T)``."""

    input_ids: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSequence:
    """Next-token-aligned text preserved through composition and packing.

    NOTE: It is the dataset's processor responsibility to shift tokens into
    aligned input and label pairs. The trainer does not do it.
    """

    input_ids: np.ndarray
    """Tokens provided to the model."""
    labels: np.ndarray
    """Target for each input token, with `IGNORE_INDEX` where loss is disabled."""
    positions: np.ndarray | None = None
    """Per-token positions; `None` until packing or collation materializes them."""
    padding_mask: np.ndarray | None = None
    """Per-token mask that is true for padding; `None` before padding."""

    def __post_init__(self) -> None:
        for name in ("input_ids", "labels", "positions", "padding_mask"):
            value = getattr(self, name)
            if value is not None and value.ndim != 1:
                raise ValueError(
                    f"TextSequence.{name} must be one-dimensional, got {value.shape}"
                )
        lengths = [len(self.input_ids), len(self.labels)]
        if self.positions is not None:
            lengths.append(len(self.positions))
        if self.padding_mask is not None:
            lengths.append(len(self.padding_mask))
        if len(set(lengths)) != 1:
            raise ValueError(
                f"TextSequence fields must have equal lengths, got {lengths}"
            )


class SampleProcessor(ABC):
    """Row processor using Grain-provided deterministic randomness.

    A subclass takes whatever it needs from ``context`` -- the tokenizer, the
    context length -- and encodes it in ``__init__``; ``__call__`` is the
    per-row transform. Both halves run in the data pipeline, so anything read
    here must already be set up by the time :func:`build_dataset` runs.
    """

    def __init__(self, *, context: DatasetBuildContext) -> None:
        del context

    @abstractmethod
    def __call__(
        self,
        sample: Any,
        rng: np.random.Generator,
    ) -> Any: ...


class SampleProcessorFactory(Protocol):
    """Create a row processor with the current dataset build context.

    Processor classes and partials binding processor options both satisfy this
    contract. Leave ``context`` unbound; dataset construction supplies it.
    """

    def __call__(self, *, context: DatasetBuildContext) -> SampleProcessor: ...
