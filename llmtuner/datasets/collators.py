"""Conversion from dataset rows to trainer batches.

Vendored from torchtitan ``components/data/collators.py``. A collator is
constructed directly with the build context and reads what it needs from there.
The padding rules are unchanged: padded positions are
``arange % max_context_length``, not zeros, so a padded row looks like a fresh
document rather than a continuation of the previous one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol

import torch

from ..accelerator.device import should_use_pin_memory
from ..components.loss import IGNORE_INDEX
from .contracts import (
    DatasetBuildContext,
    TextSequence,
    TrainerBatch,
    require_positive_integer,
)

__all__ = [
    "Collator",
    "CollatorFactory",
    "HAS_PIN_MEMORY",
    "PipelineTextCollator",
    "TextCollator",
]


# Page-locked batches let the trainer issue an async host-to-device copy; a copy
# out of pageable memory is synchronous whatever ``non_blocking`` says. There has
# to be an accelerator to pin for -- allocating with ``pin_memory=True`` raises
# without one -- so CPU-only runs fall back to ordinary pageable memory.
HAS_PIN_MEMORY = should_use_pin_memory()


class Collator(ABC):
    """Row-to-batch conversion."""

    @abstractmethod
    def __call__(self, rows: Sequence[Any]) -> TrainerBatch: ...

    def num_rows_per_batch(self) -> int:
        """Return the number of dataset rows consumed by one trainer batch."""
        return 1


class CollatorFactory(Protocol):
    """Construct a collator after recipe-specific options have been bound."""

    def __call__(self, *, context: DatasetBuildContext) -> Collator: ...


class TextCollator(Collator):
    """Pack text rows into a padded batch, pinning memory when supported."""

    def __init__(self, *, context: DatasetBuildContext) -> None:
        self._num_tokens_per_batch = context.num_tokens_per_batch
        self._max_context_length = context.max_context_length

    def __call__(self, rows: Sequence[TextSequence]) -> TrainerBatch:
        if not rows:
            raise ValueError("text batch must contain at least one row")
        num_tokens = sum(len(row.input_ids) for row in rows)
        if num_tokens > self._num_tokens_per_batch:
            raise ValueError("text rows exceed the configured token batch")

        size = self._num_tokens_per_batch
        input_ids = torch.zeros(size, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY)
        positions = torch.zeros(size, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY)
        labels = torch.full(
            (size,), IGNORE_INDEX, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY
        )
        padding_mask = torch.ones(size, dtype=torch.bool, pin_memory=HAS_PIN_MEMORY)

        torch.cat(
            [torch.as_tensor(row.input_ids) for row in rows],
            out=input_ids[:num_tokens],
        )
        torch.cat(
            [torch.as_tensor(row.labels) for row in rows],
            out=labels[:num_tokens],
        )
        torch.cat(
            [
                (
                    torch.arange(len(row.input_ids))
                    if row.positions is None
                    else torch.as_tensor(row.positions)
                )
                for row in rows
            ],
            out=positions[:num_tokens],
        )
        torch.cat(
            [
                (
                    torch.zeros(len(row.input_ids), dtype=torch.bool)
                    if row.padding_mask is None
                    else torch.as_tensor(row.padding_mask, dtype=torch.bool)
                )
                for row in rows
            ],
            out=padding_mask[:num_tokens],
        )

        pad_len = self._num_tokens_per_batch - num_tokens
        if pad_len:
            torch.arange(pad_len, out=positions[num_tokens:])
            positions[num_tokens:].remainder_(self._max_context_length)

        # Enforce the batch contract even for custom recipes that supply a
        # padding mask without already masking labels. The output owns its
        # storage, so this does not mutate the source TextSequence.
        labels.masked_fill_(padding_mask, IGNORE_INDEX)

        return {
            "input": input_ids,
            "labels": labels,
            "positions": positions,
            "padding_mask": padding_mask,
            "num_valid_tokens": int((labels != IGNORE_INDEX).sum()),
        }


class PipelineTextCollator(Collator):
    """Group independently packed rows into one pipeline schedule step."""

    def __init__(self, *, context: DatasetBuildContext, num_microbatches: int) -> None:
        require_positive_integer("num_microbatches", num_microbatches)
        if context.num_tokens_per_batch % num_microbatches:
            raise ValueError("token batch must be divisible by num_microbatches")
        self._num_microbatches = num_microbatches
        self._collator = TextCollator(
            context=replace(
                context,
                num_tokens_per_batch=context.num_tokens_per_batch // num_microbatches,
            )
        )

    def num_rows_per_batch(self) -> int:
        return self._num_microbatches

    def __call__(self, rows: Sequence[TextSequence]) -> TrainerBatch:
        if not 0 < len(rows) <= self._num_microbatches:
            raise ValueError("pipeline batch must contain 1..num_microbatches rows")
        batches = [self._collator([row]) for row in rows]
        # Finite validation may end before a complete schedule step. Keep the
        # schedule shapes fixed with fully masked rows, without repeating data.
        if len(batches) < self._num_microbatches:
            empty_row = TextSequence(
                input_ids=rows[0].input_ids[:0],
                labels=rows[0].labels[:0],
            )
            for _ in range(self._num_microbatches - len(batches)):
                batches.append(self._collator([empty_row]))
        result: TrainerBatch = {}
        for key in ("input", "labels", "positions", "padding_mask"):
            template = batches[0][key]
            output = torch.empty(
                (self._num_microbatches, *template.shape),
                dtype=template.dtype,
                pin_memory=HAS_PIN_MEMORY,
            )
            torch.stack([batch[key] for batch in batches], out=output)
            result[key] = output
        result["num_valid_tokens"] = sum(batch["num_valid_tokens"] for batch in batches)
        return result
