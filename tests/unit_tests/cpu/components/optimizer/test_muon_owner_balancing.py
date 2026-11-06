"""Owner selection keeps keyed ties and cumulative load deterministic."""

import pytest

from llmtuner.components.optimizer.flex_shard.muon_redistribution import (
    _balance_loads_across_partitions,
)


@pytest.mark.parametrize(
    "loads,cumulative,expected",
    [
        ([], (9, 0), ((), (9, 0))),
        (
            [(4, 1, "b"), (4, 1, "a"), (2, 5, "c"), (0, 0, "d")],
            (9, 0),
            ((0, 1, 1, 0), (13, 6)),
        ),
        (
            [(5, 0, "a"), (0, 99, "b"), (0, 2, "c")],
            (0, 0),
            ((0, 1, 1), (5, 0)),
        ),
    ],
)
def test_owner_balancing_preserves_load_priority_and_empty_input(
    loads, cumulative, expected
):
    assert (
        _balance_loads_across_partitions(
            loads, initial_cumulative_primary_loads=cumulative
        )
        == expected
    )


def test_owner_balancing_uses_stable_keys_independent_of_input_order():
    loads = [(4, 1, "b"), (4, 1, "a"), (2, 5, "c"), (0, 0, "d")]
    expected = {"a": 1, "b": 0, "c": 1, "d": 0}
    for order in (loads, list(reversed(loads))):
        owners, cumulative = _balance_loads_across_partitions(
            order, initial_cumulative_primary_loads=(9, 0)
        )
        assert dict(zip((key for _, _, key in order), owners, strict=True)) == expected
        assert cumulative == (13, 6)
