"""Shared data contracts: build context, iteration policy, samples, and batches.

Processors, collators, and loaders import these types without depending on the
recipe composition or loader assembly modules. Runtime context and iteration
policy are adapted from torchtitan ``components/data/types.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import grain.python as grain
import numpy as np
import torch

from ..components.tokenizer.base import BaseTokenizer

__all__ = [
    "Batch",
    "DatasetBuildContext",
    "DatasetIterationPolicy",
    "SampleProcessor",
    "TextSequence",
    "require_positive",
]


def require_positive(name: str, value: int) -> None:
    """Raise unless ``value`` is positive, naming ``name``.

    The data pipeline's one shape check, shared by the dataclasses below and by
    the builders that take the same numbers as plain parameters (the loader's
    ``max_num_documents``, the packing builders' ``num_packing_bins``, the
    synthetic loader's ``dp_world_size``). It used to be three copied checks --
    with the message text spelled out at each site, so the wording could drift
    between the two entry points for the same value.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetBuildContext:
    """Runtime values shared while building the data pipeline."""

    tokenizer: BaseTokenizer
    max_context_length: int
    num_tokens_per_batch: int
    read_options: grain.ReadOptions
    max_num_documents: int | None = None

    def __post_init__(self) -> None:
        require_positive("max_context_length", self.max_context_length)
        require_positive("num_tokens_per_batch", self.num_tokens_per_batch)
        if self.max_num_documents is not None:
            require_positive("max_num_documents", self.max_num_documents)


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
        require_positive("dp_world_size", self.dp_world_size)
        if not 0 <= self.dp_rank < self.dp_world_size:
            raise ValueError(
                f"dp_rank must be in [0, {self.dp_world_size}), got {self.dp_rank}"
            )
        require_positive(
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
