"""CPU checks for target shifting, vocabulary bounds, GC, and RNG utilities."""

from __future__ import annotations

import os

import pytest
import torch

from llmtuner.components.loss import (
    IGNORE_INDEX,
    next_token_targets,
    vocab_shard_bounds,
)
from llmtuner.utils.garbage_collection import GarbageCollection


@pytest.mark.parametrize("debug", [0, 1, "false", None])
def test_gc_rejects_non_boolean_debug_before_disabling_collector(monkeypatch, debug):
    from llmtuner.utils import garbage_collection

    monkeypatch.setattr(
        garbage_collection.gc,
        "disable",
        lambda: pytest.fail("invalid debug option must not change GC state"),
    )
    with pytest.raises(TypeError, match="debug must be bool"):
        GarbageCollection(debug=debug)


# -- vocab_shard_bounds -------------------------------------------------------


def test_the_last_shard_absorbs_the_remainder() -> None:
    """``V=10, tp=3`` -> ``[0,4) [4,8) [8,10)``: even except the final chunk."""
    assert vocab_shard_bounds(10, 3, 0) == (0, 4)
    assert vocab_shard_bounds(10, 3, 1) == (4, 8)
    assert vocab_shard_bounds(10, 3, 2) == (8, 10)


def test_a_size_wider_than_the_vocabulary_yields_empty_not_negative() -> None:
    """The clamp exists so callers reject an empty shard rather than slicing backwards.

    Without the ``min(global_vocab_size, ...)``, rank 7 of a 8-way split of a
    4-token vocabulary would get ``start=4, end=... `` past the end and a
    negative length. The callers (the sharded cross-entropy) turn the empty
    slice into a loud error; this only guarantees the bounds are sane.
    """
    start, end = vocab_shard_bounds(4, 8, 7)
    assert start == 4
    assert end >= start, "end must never precede start"
    assert end == start, "rank 7 owns nothing"


def test_sharding_is_total_and_disjoint() -> None:
    """Every vocabulary entry is owned by exactly one rank -- no gaps, no overlap."""
    vocab, tp = 10, 3
    owned: list[int] = []
    for rank in range(tp):
        start, end = vocab_shard_bounds(vocab, tp, rank)
        owned.extend(range(start, end))
    assert owned == list(range(vocab))


@pytest.mark.parametrize(
    "vocab,tp,rank,match",
    [
        (0, 2, 0, "global_vocab_size must be >= 1"),
        (-1, 2, 0, "global_vocab_size must be >= 1"),
        (10, 0, 0, "tp_world_size must be >= 1"),
        (10, 2, 2, "is outside"),
        (10, 2, -1, "is outside"),
    ],
)
def test_bad_bounds_arguments_are_rejected(
    vocab: int, tp: int, rank: int, match: str
) -> None:
    """A zero/negative size or an out-of-range rank is a config bug, not a slice."""
    with pytest.raises(ValueError, match=match):
        vocab_shard_bounds(vocab, tp, rank)


# -- next_token_targets -------------------------------------------------------


def test_the_last_position_of_every_row_is_ignored() -> None:
    """The shift is within a row: the boundary token predicts nothing.

    Positions ``[2, 3, -100] [5, 6, -100]``: row 0's last slot would predict
    token 4, which belongs to the *next* document, so it is masked rather than
    letting the model train on a prediction it had no context for.
    """
    labels = torch.tensor([1, 2, 3, 4, 5, 6])
    targets = next_token_targets(labels, seq_len=3)
    assert targets.tolist() == [2, 3, IGNORE_INDEX, 5, 6, IGNORE_INDEX]


def test_the_shift_does_not_cross_a_row_boundary() -> None:
    """A flat shift would place 4 at index 3; the row-wise one must not."""
    labels = torch.tensor([1, 2, 3, 4, 5, 6])
    targets = next_token_targets(labels, seq_len=3).view(2, 3)
    assert targets[0].tolist() == [2, 3, IGNORE_INDEX]
    assert targets[1].tolist() == [5, 6, IGNORE_INDEX]
    # Index 3 of the flat stream is the first slot of row 1, which predicts 5 --
    # if the implementation had shifted flat, it would hold 4.
    assert targets[1][0].item() == 5


def test_labels_the_source_tensor_is_not_mutated() -> None:
    """``targets`` is built with ``full_like``; the caller's labels survive."""
    labels = torch.tensor([1, 2, 3, 4])
    before = labels.clone()
    next_token_targets(labels, seq_len=2)
    assert torch.equal(labels, before)


def test_a_ragged_label_count_is_rejected() -> None:
    """5 labels cannot be whole rows of 2 -- silently dropping one hides a data bug."""
    with pytest.raises(ValueError, match="not a whole number of rows"):
        next_token_targets(torch.zeros(5, dtype=torch.long), seq_len=2)


# -- GarbageCollection --------------------------------------------------------


def test_closed_collector_cannot_run(monkeypatch):
    collector = GarbageCollection()
    collector.close()

    def fail_if_collected(*args, **kwargs):
        pytest.fail("closed collector triggered manual collection")

    monkeypatch.setattr(collector, "collect", fail_if_collected)
    with pytest.raises(RuntimeError, match="closed"):
        collector.run(1000)


@pytest.mark.parametrize("frequency", [True, 1.5, float("nan"), "2", None])
def test_gc_rejects_invalid_frequency_before_disabling_collection(
    frequency, monkeypatch
):
    import gc

    def unexpected_disable():
        pytest.fail("invalid frequency must not change global GC state")

    monkeypatch.setattr(gc, "disable", unexpected_disable)
    with pytest.raises(ValueError, match="positive integer"):
        GarbageCollection(gc_freq=frequency)


def test_a_non_positive_collection_frequency_is_rejected() -> None:
    """0 would make the modulo-by-frequency either a ZeroDivision or a no-op."""
    with pytest.raises(ValueError, match="gc_freq must be a positive integer"):
        GarbageCollection(gc_freq=0)
    with pytest.raises(ValueError, match="gc_freq must be a positive integer"):
        GarbageCollection(gc_freq=-1)


def test_a_positive_frequency_disables_the_automatic_collector() -> None:
    """The class owns collection timing, so CPython's own trigger must be off."""
    import gc as _gc

    _gc.enable()
    collector = GarbageCollection(gc_freq=1000)
    try:
        assert collector.gc_freq == 1000
        assert _gc.isenabled() is False, "the collector takes over scheduling"
    finally:
        # Do not leak the disabled state into whichever test runs next.
        _gc.enable()


@pytest.mark.parametrize("initially_enabled", [True, False])
def test_collection_close_restores_initial_gc_state(initially_enabled) -> None:
    import gc

    original = gc.isenabled()
    try:
        (gc.enable if initially_enabled else gc.disable)()
        collector = GarbageCollection()
        assert not gc.isenabled()
        collector.close()
        assert gc.isenabled() == initially_enabled
        collector.close()
        assert gc.isenabled() == initially_enabled
    finally:
        (gc.enable if original else gc.disable)()


@pytest.mark.parametrize("seq_len", [0, -1])
def test_target_shift_rejects_nonpositive_sequence_length(seq_len):
    with pytest.raises(ValueError, match="seq_len must be >= 1"):
        next_token_targets(torch.tensor([1, 2]), seq_len=seq_len)


def test_target_shift_accepts_noncontiguous_labels():
    labels = torch.arange(12).reshape(3, 4).t()
    targets = next_token_targets(labels, seq_len=3)
    expected = torch.full_like(labels, IGNORE_INDEX)
    expected[:, :-1] = labels[:, 1:]
    torch.testing.assert_close(targets, expected)


@pytest.mark.parametrize("seed", [-(2**63), 2**32, 2**64 - 1, -1])
def test_seeding_supports_pytorch_seed_range(seed):
    import random

    import numpy as np

    from llmtuner.utils.seeding import seed_everything

    seed_everything(seed, deterministic=False)
    first = (random.random(), np.random.random(), torch.rand(3))
    seed_everything(seed, deterministic=False)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("enabled", [True, False])
def test_gc_initialization_failure_restores_automatic_collection(
    monkeypatch, error_type, enabled
):
    import gc

    original = gc.isenabled()

    def fail(reason):
        raise error_type("collection failed")

    monkeypatch.setattr(GarbageCollection, "collect", staticmethod(fail))
    try:
        (gc.enable if enabled else gc.disable)()
        with pytest.raises(error_type, match="collection failed"):
            GarbageCollection()
        assert gc.isenabled() is enabled
    finally:
        (gc.enable if original else gc.disable)()


def test_seed_flags_require_boolean_values():
    from llmtuner.utils.seeding import seed_everything

    with pytest.raises(TypeError, match="deterministic"):
        seed_everything(1, deterministic=1)
    with pytest.raises(TypeError, match="must be bool"):
        seed_everything(1, deterministic=False, detect_anomaly=0)


@pytest.mark.parametrize("seed", [True, 1.5, "1", -(2**63) - 1, 2**64])
def test_invalid_seed_does_not_change_rng_or_environment(seed):
    import random

    import numpy as np

    from llmtuner.utils.seeding import seed_everything

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    hash_seed = os.environ.get("PYTHONHASHSEED")

    with pytest.raises(ValueError, match="seed must be an integer"):
        seed_everything(seed, deterministic=False)

    assert random.getstate() == python_state
    actual_numpy_state = np.random.get_state()
    assert actual_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy_state[1], numpy_state[1])
    assert actual_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert os.environ.get("PYTHONHASHSEED") == hash_seed


def test_seed_configuration_disables_anomaly_detection_for_subsequent_run():
    from llmtuner.utils.seeding import seed_everything

    enabled = torch.is_anomaly_enabled()
    check_nan = torch.is_anomaly_check_nan_enabled()
    try:
        seed_everything(1, deterministic=False, detect_anomaly=True)
        assert torch.is_anomaly_enabled()
        assert not torch.is_anomaly_check_nan_enabled()
        seed_everything(1, deterministic=False, detect_anomaly=False)
        assert not torch.is_anomaly_enabled()
    finally:
        torch.autograd.set_detect_anomaly(enabled, check_nan=check_nan)
