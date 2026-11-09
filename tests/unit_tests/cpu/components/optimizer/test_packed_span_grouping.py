"""Packed transport grouping preserves ordering and empty parameter slots."""

from llmtuner.components.optimizer.flex_shard._optimizer_reshard_schedule import (
    _packed_spans_by_parameter,
    _PackedSpan,
    _TensorRegion,
)


def test_grouping_preserves_interleaved_spans_and_empty_parameters():
    spans = [_PackedSpan(_TensorRegion((0,), (2,)), offset) for offset in range(5)]
    indexed = list(zip((2, 0, 2, 3, 0), spans, strict=True))
    assert _packed_spans_by_parameter(indexed, 5) == (
        (spans[1], spans[4]),
        (),
        (spans[0], spans[2]),
        (spans[3],),
        (),
    )
    assert _packed_spans_by_parameter([], 2) == ((), ())
    assert _packed_spans_by_parameter([], 0) == ()
