"""Reject incomplete or divergent training observations."""

import json

import pytest

from tests.integration_tests.compare_training_audits import compare


def write_audit(root, steps):
    root.mkdir()
    (root / "rank0_mesh.json").write_text(json.dumps({"rank": 0}))
    records = [
        {
            "step": step,
            "seconds": 1.0,
            "metrics": {"loss": 0.5, "grad_norm": 1.0},
            "batches": [
                {
                    "valid_tokens": 2,
                    "fields": dict.fromkeys(
                        ["input", "labels", "positions", "padding_mask"], "hash"
                    ),
                }
            ],
        }
        for step in steps
    ]
    (root / "rank0_steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )


def test_exact_resume_observations_pass(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    write_audit(left, [1, 2, 3])
    write_audit(right, [2, 3])
    result = compare(left, right, world=1, total=3, split=1)
    assert result["rank_step_comparisons"] == 2


@pytest.mark.parametrize("failure", ["missing", "nan", "input", "metric", "mesh"])
def test_divergent_or_incomplete_audit_fails(tmp_path, failure):
    left, right = tmp_path / "a", tmp_path / "b"
    write_audit(left, [1, 2, 3])
    write_audit(right, [2, 3])
    path = right / "rank0_steps.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "missing":
        rows.pop()
    elif failure == "nan":
        rows[0]["metrics"]["loss"] = float("nan")
    elif failure == "input":
        rows[0]["batches"][0]["fields"]["input"] = "different"
    elif failure == "metric":
        rows[0]["metrics"]["loss"] = 0.6
    else:
        (right / "rank0_mesh.json").write_text(json.dumps({"rank": 1}))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(AssertionError):
        compare(left, right, world=1, total=3, split=1)
