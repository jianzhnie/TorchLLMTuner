"""Check ownership using norms that have an independently known total."""

import math

import pytest

from tests.integration_tests.reconstruct_gradient_norms import reconstruct


def record(name, norm, *, tp_sharded=False, placements=(), mesh_ranks=()):
    return {
        "part": 0,
        "name": name,
        "local_l2_norm": norm,
        "tp_sharded": tp_sharded,
        "placements": list(placements),
        "mesh_ranks": list(mesh_ranks),
    }


def test_shards_and_replicas_contribute_exactly_once():
    # DP2 x TP2: the replicated tensor has DP-shard norms 3 and 4,
    # and the TP-sharded tensor has four disjoint unit-norm pieces.
    ranks = [
        [
            record("replicated", 3 if rank < 2 else 4, placements=("S(0)",)),
            record("sharded", 1, tp_sharded=True, placements=("S(0)",)),
            record("excluded", 12),
        ]
        for rank in range(4)
    ]
    for rank, records in enumerate(ranks):
        for row in records:
            if row["placements"]:
                row["mesh_ranks"] = [rank % 2, rank % 2 + 2]
    result = reconstruct(ranks, dp=2, tp=2)
    assert result["l2_norm"] == pytest.approx(math.sqrt(25 + 4 + 144))
    assert [p["owners"] for p in result["parameters"]] == [2, 4, 1]


def test_replica_drift_is_reported_without_averaging():
    result = reconstruct([[record("weight", 3)], [record("weight", 4)]], dp=1, tp=2)
    assert result["l2_norm"] == 3
    assert result["parameters"][0]["max_replica_norm_spread"] == 1


@pytest.mark.parametrize("placement", ["P", "Partial(sum)"])
def test_partial_gradient_cannot_be_reconstructed_as_a_shard(placement):
    with pytest.raises(AssertionError, match="unsupported gradient placement"):
        reconstruct([[record("weight", 1, placements=(placement,))]], dp=1, tp=1)


def test_wrong_mesh_cannot_silently_use_rank_arithmetic():
    ranks = [
        [record("weight", 1, placements=("S(0)",), mesh_ranks=(0, 1))] for _ in range(4)
    ]
    with pytest.raises(AssertionError, match="unsupported gradient mesh"):
        reconstruct(ranks, dp=2, tp=2)
