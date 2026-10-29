"""Repeated map shards must reject all-filtered input before infinite iteration."""

from tests.caps import require_env

require_env("grain")

import random

import grain.python as grain
import numpy as np
import pytest
import torch

from llmtuner.datasets.composition import apply_iteration_policy
from llmtuner.errors import ConfigError
from tests.data_fixtures import make_policy


@pytest.mark.parametrize("shuffle", [False, True])
def test_repeated_all_filtered_map_is_rejected(shuffle):
    dataset = grain.MapDataset.range(7).filter(lambda value: False)
    with pytest.raises(ConfigError, match="no usable samples.*rank 0/1, 7 rows"):
        apply_iteration_policy(
            dataset, iteration_policy=make_policy(shuffle=shuffle, repeat=True)
        )


def test_empty_dp_shard_is_rejected_even_when_another_rank_has_data():
    dataset = grain.MapDataset.range(8).filter(lambda value: value >= 4)
    with pytest.raises(ConfigError, match="rank 0/2, 4 rows"):
        apply_iteration_policy(
            dataset,
            iteration_policy=make_policy(dp_rank=0, dp_world_size=2, repeat=True),
        )
    actual = apply_iteration_policy(
        dataset, iteration_policy=make_policy(dp_rank=1, dp_world_size=2, repeat=True)
    )
    assert [actual[index] for index in range(8)] == [4, 5, 6, 7] * 2


def test_finite_all_filtered_map_remains_finite_without_rejection():
    actual = apply_iteration_policy(
        grain.MapDataset.range(7).filter(lambda value: False),
        iteration_policy=make_policy(repeat=False),
    )
    assert len(actual) == 7
    assert list(actual.to_iter_dataset()) == []


def test_sparse_valid_map_retains_index_and_repeat_order():
    actual = apply_iteration_policy(
        grain.MapDataset.range(2048).filter(lambda value: value == 2047),
        iteration_policy=make_policy(repeat=True),
    )
    assert actual[0] is None
    assert actual[2047] == 2047
    assert actual[2048] is None
    assert actual[4095] == 2047


def test_preflight_preserves_python_numpy_and_torch_rng():
    def process(value):
        random.random()
        np.random.random()
        torch.rand(1)
        return value

    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    apply_iteration_policy(
        grain.MapDataset.range(4).map(process),
        iteration_policy=make_policy(repeat=True),
    )
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_preflight_stops_after_first_usable_sample():
    calls = []

    def process(value):
        calls.append(value)
        return value if value == 2 else None

    apply_iteration_policy(
        grain.MapDataset.range(8).map(process),
        iteration_policy=make_policy(repeat=True),
    )
    assert calls == [0, 1, 2]
