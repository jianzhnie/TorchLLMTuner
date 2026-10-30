"""Workload audit must reject missing steps, changed inputs and ownership."""

import json

import pytest

from tests.integration_tests.verify_training_workload import verify


def fixture(tmp_path):
    batch = {"fields": {"input": "hash"}, "valid_tokens": 2}
    reference = {"degrees": {"4": [{"step": 1, "ranks": [batch] * 4}]}}
    for rank in range(4):
        (tmp_path / f"rank{rank}_mesh.json").write_text(
            json.dumps({"rank": rank, "dp_coordinates": [rank, 4]})
        )
        (tmp_path / f"rank{rank}_steps.jsonl").write_text(
            json.dumps({"step": 1, "batches": [batch]}) + "\n"
        )
    return reference


def test_complete_workload_passes(tmp_path):
    assert verify(fixture(tmp_path), tmp_path, world=4, steps=1)["rank_steps"] == 4


@pytest.mark.parametrize("failure", ["input", "tokens", "missing", "ownership"])
def test_workload_corruption_fails(tmp_path, failure):
    reference = fixture(tmp_path)
    path = tmp_path / "rank0_steps.jsonl"
    record = json.loads(path.read_text())
    if failure == "missing":
        path.write_text("")
    elif failure == "ownership":
        (tmp_path / "rank0_mesh.json").write_text(
            json.dumps({"rank": 0, "dp_coordinates": [1, 4]})
        )
    else:
        if failure == "tokens":
            record["batches"][0]["valid_tokens"] = 3
        else:
            record["batches"][0]["fields"]["input"] = "changed"
        path.write_text(json.dumps(record) + "\n")
    with pytest.raises(AssertionError):
        verify(reference, tmp_path, world=4, steps=1)


def test_uneven_replication_cannot_pass_complete_ownership(tmp_path):
    reference = fixture(tmp_path)
    # Every DP coordinate exists, but rank0's data is replicated three times
    # while the other coordinates have fewer replicas. Set coverage alone
    # would incorrectly accept this topology.
    for rank in range(4, 8):
        coordinate = [0, 0, 1, 2][rank - 4]
        (tmp_path / f"rank{rank}_mesh.json").write_text(
            json.dumps({"rank": rank, "dp_coordinates": [coordinate, 4]})
        )
        (tmp_path / f"rank{rank}_steps.jsonl").write_text(
            (tmp_path / "rank0_steps.jsonl").read_text()
        )
    with pytest.raises(AssertionError, match="uneven replicated DP ownership"):
        verify(reference, tmp_path, world=8, steps=1)
