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
from ...utils.logging import get_logger
from ..contracts import TextSequence, require_positive_integer


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
        require_positive_integer("max_num_documents_per_row", max_num_documents_per_row)
        require_positive_integer("max_context_length", max_context_length)
        require_positive_integer("num_tokens_per_row", num_tokens_per_row)
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
        require_positive_integer("max_num_documents_per_row", max_num_documents_per_row)
        require_positive_integer("max_context_length", max_context_length)
        require_positive_integer("num_tokens_per_row", num_tokens_per_row)
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
            input_parts.append(segment.input_ids)
            label_parts.append(segment.labels)
            position_parts.append(segment.positions)
            mask_parts.append(segment.padding_mask)
            num_tokens += len(segment.input_ids)

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

    def _take_segment(self, available_tokens: int) -> TextSequence | None:
        """Cut the next document-bounded segment off the remainder stream.

        Returns a sequence with explicit positions and padding mask for the
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
        return TextSequence(
            input_ids=np.asarray(sequence.input_ids[token_slice]),
            labels=np.asarray(sequence.labels[token_slice]),
            positions=np.arange(num_segment_tokens, dtype=np.int64),
            padding_mask=mask_seg,
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
        has_remainder = state["has_remainder"]
        finished = state["finished"]
        offset = state["remainder_offset"]
        if not isinstance(has_remainder, bool) or not isinstance(finished, bool):
            raise ValueError("packing checkpoint flags must be booleans")
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or (not has_remainder and offset != 0)
            or (has_remainder and finished)
        ):
            raise ValueError("packing checkpoint has inconsistent remainder state")
        previous_parent_state = self._parent.get_state()
        remainder = None
        try:
            self._parent.set_state(state["parent"])
            if has_remainder:
                try:
                    remainder = next(self._parent)
                except StopIteration as error:
                    raise ValueError(
                        "packing checkpoint remainder is missing from the parent source"
                    ) from error
                if not 0 < offset < len(remainder.input_ids):
                    raise ValueError(
                        "packing checkpoint remainder offset is outside "
                        "the restored sample"
                    )
        except BaseException:
            # Replaying the remainder advances the parent. Invalid state must
            # not change the live stream's next token or consume its sample.
            try:
                self._parent.set_state(previous_parent_state)
            except BaseException:
                get_logger(__name__).exception(
                    "Packing cursor rollback failed; closing the iterator"
                )
                try:
                    self.close()
                except BaseException:
                    get_logger(__name__).exception(
                        "Packing iterator cleanup failed after rollback failure"
                    )
            raise

        self._remainder = remainder
        self._remainder_parent_state = state["parent"] if has_remainder else None
        self._remainder_offset = offset
        self._finished = finished


class SplitTextSequenceDocuments(grain.experimental.FlatMapTransform):
    """Expose context-sized document chunks to Grain's native packing limit."""

    def __init__(
        self,
        *,
        max_context_length: int,
    ) -> None:
        require_positive_integer("max_context_length", max_context_length)
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
