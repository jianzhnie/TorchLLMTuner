"""Grain iterators that splice documents across fixed-length row boundaries.

``DocumentAwareConcatThenSplitIterDataset`` fills rows while capping the number
of document segments per row, carrying a checkpointable remainder; the flat-map
transform and ``next_document_chunk_end`` cut documents at context-sized or
document boundaries.
"""

from __future__ import annotations

from typing import Any

import grain.python as grain
import numpy as np

from ...components.loss import IGNORE_INDEX
from ..dataset import (
    TextSequence,
)


class DocumentAwareConcatThenSplitIterDataset(grain.IterDataset):
    """Concat-then-split packing with a document-segment capacity."""

    def __init__(
        self,
        parent: grain.IterDataset,
        *,
        max_num_documents_per_row: int,
        max_context_length: int,
        num_tokens_per_row: int,
    ) -> None:
        super().__init__(parent)
        self._max_num_documents_per_row = max_num_documents_per_row
        self._max_context_length = max_context_length
        self._num_tokens_per_row = num_tokens_per_row

    def __iter__(self) -> grain.DatasetIterator:
        return DocumentAwareConcatThenSplitIterator(
            iter(self._parent),
            max_num_documents_per_row=self._max_num_documents_per_row,
            max_context_length=self._max_context_length,
            num_tokens_per_row=self._num_tokens_per_row,
        )


class DocumentAwareConcatThenSplitIterator(grain.DatasetIterator):
    """Build fixed-size rows while preserving document order and remainders."""

    def __init__(
        self,
        parent: grain.DatasetIterator,
        *,
        max_num_documents_per_row: int,
        max_context_length: int,
        num_tokens_per_row: int,
    ) -> None:
        super().__init__(parent)
        self._max_num_documents_per_row = max_num_documents_per_row
        self._max_context_length = max_context_length
        self._num_tokens_per_row = num_tokens_per_row
        self._remainder: TextSequence | None = None
        self._remainder_parent_state: dict[str, Any] | None = None
        self._remainder_offset = 0
        self._finished = False

    def __next__(self) -> TextSequence:
        self._assert_not_closed()
        if self._finished:
            raise StopIteration

        input_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        position_parts: list[np.ndarray] = []
        num_tokens = 0

        while (
            num_tokens < self._num_tokens_per_row
            and len(position_parts) < self._max_num_documents_per_row
        ):
            segment = self._take_segment(self._num_tokens_per_row - num_tokens)
            if segment is None:
                break
            input_seg, label_seg, position_seg, mask_seg = segment
            input_parts.append(input_seg)
            label_parts.append(label_seg)
            position_parts.append(position_seg)
            mask_parts.append(mask_seg)
            num_tokens += len(input_seg)

        if not input_parts:
            raise StopIteration

        input_ids = np.concatenate(input_parts)
        labels = np.concatenate(label_parts)
        positions = np.concatenate(position_parts)
        padding_mask = np.concatenate(mask_parts)
        pad_len = self._num_tokens_per_row - num_tokens
        if pad_len:
            padding_positions = (
                np.arange(pad_len, dtype=positions.dtype) % self._max_context_length
            )
            input_ids = np.pad(input_ids, (0, pad_len))
            labels = np.pad(labels, (0, pad_len), constant_values=IGNORE_INDEX)
            positions = np.concatenate((positions, padding_positions))
            padding_mask = np.pad(padding_mask, (0, pad_len), constant_values=True)

        return TextSequence(
            input_ids=input_ids,
            labels=labels,
            positions=positions,
            padding_mask=padding_mask,
        )

    def _take_segment(
        self, available_tokens: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Cut the next document-bounded segment off the remainder stream.

        Returns ``(input_ids, labels, positions, padding_mask)`` for the
        segment, or ``None`` when the parent iterator is exhausted
        (``self._finished`` set). Empty sequences are skipped internally.
        """
        while self._remainder is None:
            parent_state = self._parent.get_state()
            try:
                sequence = next(self._parent)
            except StopIteration:
                self._finished = True
                return None
            if len(sequence.input_ids) == 0:
                continue
            self._remainder = sequence
            self._remainder_parent_state = parent_state
            self._remainder_offset = 0

        sequence = self._remainder
        source_positions = (
            None if sequence.positions is None else np.asarray(sequence.positions)
        )
        segment_end = next_document_chunk_end(
            num_tokens=len(sequence.input_ids),
            positions=source_positions,
            start=self._remainder_offset,
            max_context_length=self._max_context_length,
        )
        num_segment_tokens = min(
            segment_end - self._remainder_offset,
            available_tokens,
        )

        token_slice = slice(
            self._remainder_offset,
            self._remainder_offset + num_segment_tokens,
        )
        # Padding from an upstream packer (e.g. first-fit inside
        # concat-then-split) must stay marked as padding; only a source
        # without a mask is treated as all real tokens.
        source_mask = getattr(sequence, "padding_mask", None)
        mask_seg = (
            np.zeros(num_segment_tokens, dtype=np.bool_)
            if source_mask is None
            else np.asarray(source_mask[token_slice], dtype=np.bool_)
        )
        self._remainder_offset += num_segment_tokens
        if self._remainder_offset == len(sequence.input_ids):
            self._remainder = None
            self._remainder_parent_state = None
            self._remainder_offset = 0
        return (
            np.asarray(sequence.input_ids[token_slice]),
            np.asarray(sequence.labels[token_slice]),
            np.arange(num_segment_tokens, dtype=np.int64),
            mask_seg,
        )

    def get_state(self) -> dict[str, Any]:
        if self._remainder is None:
            parent_state = self._parent.get_state()
        else:
            assert self._remainder_parent_state is not None
            parent_state = self._remainder_parent_state
        return {
            "parent": parent_state,
            "has_remainder": self._remainder is not None,
            "remainder_offset": self._remainder_offset,
            "finished": self._finished,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._parent.set_state(state["parent"])
        self._remainder = None
        self._remainder_parent_state = None
        self._remainder_offset = 0
        self._finished = state["finished"]
        if state["has_remainder"]:
            self._remainder_parent_state = state["parent"]
            try:
                self._remainder = next(self._parent)
            except StopIteration:
                # The recorded remainder came from the parent's last element:
                # the state restored above is already past it, so there is no
                # remainder to rebuild -- the next __next__ simply raises.
                self._remainder = None
                self._remainder_parent_state = None
                self._finished = True
                return
            self._remainder_offset = state["remainder_offset"]


class SplitTextSequenceDocuments(grain.experimental.FlatMapTransform):
    """Expose context-sized document chunks to Grain's native packing limit."""

    def __init__(
        self,
        *,
        max_context_length: int,
    ) -> None:
        self._max_context_length = max_context_length
        self.max_fan_out = max_context_length

    def flat_map(self, element: TextSequence) -> list[TextSequence]:
        if len(element.input_ids) == 0:
            return []

        positions = None if element.positions is None else np.asarray(element.positions)
        chunks = []
        chunk_start = 0
        while chunk_start < len(element.input_ids):
            chunk_end = next_document_chunk_end(
                num_tokens=len(element.input_ids),
                positions=positions,
                start=chunk_start,
                max_context_length=self._max_context_length,
            )
            chunks.append(
                TextSequence(
                    input_ids=np.asarray(element.input_ids[chunk_start:chunk_end]),
                    labels=np.asarray(element.labels[chunk_start:chunk_end]),
                    positions=np.arange(chunk_end - chunk_start, dtype=np.int64),
                    padding_mask=(
                        None
                        if element.padding_mask is None
                        else np.asarray(element.padding_mask[chunk_start:chunk_end])
                    ),
                )
            )
            chunk_start = chunk_end
        return chunks


def next_document_chunk_end(
    *,
    num_tokens: int,
    positions: np.ndarray | None,
    start: int,
    max_context_length: int,
) -> int:
    """Return the next document boundary or context-sized chunk boundary."""
    end = min(start + max_context_length, num_tokens)
    if positions is not None:
        next_starts = np.flatnonzero(positions[start + 1 : end] == 0)
        if next_starts.size:
            end = start + 1 + int(next_starts[0])
    return end

