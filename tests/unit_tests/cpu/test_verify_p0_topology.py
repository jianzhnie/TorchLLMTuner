"""Reject real group drift, rather than trusting configured parallel degrees."""

import json
from pathlib import Path

import pytest

from tests.integration_tests.verify_p0_topology import verify


def write_dp8(root: Path):
    for rank in range(8):
        row = {
            "rank": rank,
            "dp_coordinates": [rank, 8],
            "meshes": {
                "dense": {
                    "local_groups": {
                        "pp": [rank],
                        "dp_replicate": [rank],
                        "dp_shard": list(range(8)),
                        "cp": [rank],
                        "tp": [rank],
                    }
                },
                "sparse": {
                    "local_groups": {
                        "pp": [rank],
                        "dp_replicate": [rank],
                        "efsdp": list(range(8)),
                        "ep": [rank],
                    }
                },
            },
        }
        (root / f"rank{rank}_mesh.json").write_text(json.dumps(row))


def test_observed_dp8_groups_pass(tmp_path):
    write_dp8(tmp_path)
    assert verify(tmp_path, "dp8")["expert_fsdp"] == 8


@pytest.mark.parametrize("field", ["tp", "dp_shard", "ep", "efsdp", "coordinate"])
def test_group_or_data_coordinate_drift_fails(tmp_path, field):
    write_dp8(tmp_path)
    path = tmp_path / "rank3_mesh.json"
    row = json.loads(path.read_text())
    if field == "coordinate":
        row["dp_coordinates"] = [2, 8]
    else:
        mesh = "dense" if field in ("tp", "dp_shard") else "sparse"
        row["meshes"][mesh]["local_groups"][field] = [0]
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError):
        verify(tmp_path, "dp8")


@pytest.mark.parametrize(
    ("layout", "dp", "tp", "sparse_rows"),
    [
        ("dp4_tp2", 4, 2, [list(range(8))]),
        ("dp4_tp2_ep4", 4, 2, [[0, 4], [1, 5], [2, 6], [3, 7]]),
        ("dp8_ep8", 8, 1, [[rank] for rank in range(8)]),
    ],
)
def test_required_tp_ep_layouts_use_explicit_group_membership(
    tmp_path, layout, dp, tp, sparse_rows
):
    # Explicit expert-FSDP columns exercise EP's reuse of the same eight
    # devices; EP must not multiply the dense world size.
    for rank in range(8):
        expert_group = next(group for group in sparse_rows if rank in group)
        ep_size = 8 // len(expert_group)
        ep_start = rank - rank % ep_size
        row = {
            "rank": rank,
            "dp_coordinates": [rank // tp, dp],
            "meshes": {
                "dense": {
                    "local_groups": {
                        "pp": [rank],
                        "dp_replicate": [rank],
                        "dp_shard": list(range(rank % tp, 8, tp)),
                        "cp": [rank],
                        "tp": [rank] if tp == 1 else [rank // 2 * 2, rank // 2 * 2 + 1],
                    }
                },
                "sparse": {
                    "local_groups": {
                        "pp": [rank],
                        "dp_replicate": [rank],
                        "efsdp": expert_group,
                        "ep": list(range(ep_start, ep_start + ep_size)),
                    }
                },
            },
        }
        (tmp_path / f"rank{rank}_mesh.json").write_text(json.dumps(row))
    result = verify(tmp_path, layout)
    assert result["expert_fsdp"] == len(sparse_rows[0])
