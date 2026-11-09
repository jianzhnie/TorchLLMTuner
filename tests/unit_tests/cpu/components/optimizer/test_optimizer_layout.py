import pickle

import pytest
import torch
from torch.distributed.tensor import Shard

from llmtuner.components.optimizer.flex_shard.optimizer_reshard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
)


def test_bucket_config_normalizes_pattern_sequence():
    config = BucketConfig(patterns=["model.layers.*", "output.*"], name="weights")

    assert config.patterns == ("model.layers.*", "output.*")
    assert config.name == "weights"


@pytest.mark.parametrize("patterns", [(), ("",), ("", "weight")])
def test_bucket_config_preserves_upstream_empty_pattern_semantics(patterns):
    config = BucketConfig(patterns=patterns, name=None)
    assert config.patterns == patterns
    assert config.name is None


def test_block_shard_boundary_arithmetic_matches_upstream():
    blocks = BlockShard(dim=0, block_sizes=(2, 4))
    assert blocks.num_blocks(0) == 0
    assert blocks.num_blocks(12) == 4
    assert [blocks.block_start(i) for i in range(5)] == [0, 2, 6, 8, 12]
    with pytest.raises(ValueError, match="divisible"):
        blocks.num_blocks(5)


def test_compute_layout_preserves_unused_empty_axis_declaration():
    config = ComputeLayout(shardings_by_mesh_axis={"dp": Shard(0), "": Shard(0)})
    assert config.shardings_by_mesh_axis[""] == Shard(0)


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
