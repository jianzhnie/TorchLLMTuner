import pickle

import pytest
import torch
from torch.distributed.tensor import Shard

from llmtuner.config.optimizer_layout import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
)


def test_bucket_config_normalizes_pattern_sequence():
    config = BucketConfig(patterns=["model.layers.*", "output.*"], name="weights")

    assert config.patterns == ("model.layers.*", "output.*")
    assert config.name == "weights"


@pytest.mark.parametrize("patterns", ["weight", (), ("",), ("weight", 1)])
def test_bucket_config_rejects_invalid_patterns(patterns):
    with pytest.raises(ValueError, match="BucketConfig.patterns"):
        BucketConfig(patterns=patterns)


def test_bucket_config_rejects_non_string_name():
    with pytest.raises(ValueError, match="BucketConfig.name"):
        BucketConfig(patterns=("weight",), name=None)


@pytest.mark.parametrize("dim_size", [-1, 1.5, True])
def test_block_shard_rejects_invalid_dimension_size(dim_size):
    with pytest.raises(ValueError, match="dimension size"):
        BlockShard(dim=0, block_sizes=(2,)).num_blocks(dim_size)


@pytest.mark.parametrize("block_index", [-1, 1.5, False])
def test_block_shard_rejects_invalid_block_index(block_index):
    with pytest.raises(ValueError, match="block index"):
        BlockShard(dim=0, block_sizes=(2,)).block_start(block_index)


def test_compute_layout_rejects_non_string_shard_order_names():
    with pytest.raises(ValueError, match="non-empty mesh axis names"):
        ComputeLayout(
            shardings_by_mesh_axis={"dp": Shard(0), "tp": Shard(0)},
            shard_order_by_tensor_dim={0: (["invalid"], "tp")},
            compute_dtype=torch.bfloat16,
        )


def test_compute_layout_rejects_empty_mesh_axis_name():
    with pytest.raises(ValueError, match="mesh axis names must not be empty"):
        ComputeLayout(shardings_by_mesh_axis={"": Shard(0)})


def test_compute_layout_pickle_round_trip_preserves_immutable_mappings():
    original = ComputeLayout(
        shardings_by_mesh_axis={"dp": Shard(0), "tp": Shard(0)},
        shard_order_by_tensor_dim={0: ("dp", "tp")},
        compute_dtype=torch.bfloat16,
    )

    restored = pickle.loads(pickle.dumps(original))

    assert restored == original
    assert restored.shardings_by_mesh_axis == original.shardings_by_mesh_axis
    assert restored.shard_order_by_tensor_dim == original.shard_order_by_tensor_dim


def test_equivalent_layouts_have_equal_hashes_independent_of_mapping_order():
    shardings = {"dp": Shard(0), "tp": Shard(0), "ep": Shard(1), "cp": Shard(1)}
    first = ComputeLayout(
        shardings_by_mesh_axis=shardings,
        shard_order_by_tensor_dim={0: ("dp", "tp"), 1: ("ep", "cp")},
    )
    second = ComputeLayout(
        shardings_by_mesh_axis=dict(reversed(list(shardings.items()))),
        shard_order_by_tensor_dim={1: ("ep", "cp"), 0: ("dp", "tp")},
    )

    assert first == second
    assert hash(first) == hash(second)
    assert {first: "planned"}[second] == "planned"
