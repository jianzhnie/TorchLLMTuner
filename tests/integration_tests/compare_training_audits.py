"""Check per-rank continuous/resumed training observations, with no tolerances.

Input audits describe real Trainer calls. This checks loss/gradient norm and
input continuity; final checkpoint equality is a separate required check.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_steps(path: Path, expected: range):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step"] for row in records] == list(expected), f"step coverage: {path}"
    for row in records:
        metrics = row["metrics"]
        assert metrics and {"loss", "grad_norm"} <= metrics.keys(), path
        assert all(math.isfinite(value) for value in metrics.values()), path
        assert math.isfinite(row["seconds"]) and row["seconds"] > 0, path
        assert row["batches"], path
        for batch in row["batches"]:
            assert batch["valid_tokens"] >= 0, path
            assert set(batch["fields"]) == {
                "input",
                "labels",
                "positions",
                "padding_mask",
            }, path
    return {row["step"]: row for row in records}


def compare(continuous: Path, resumed: Path, *, world: int, total: int, split: int):
    assert world > 0 and 0 < split < total
    comparisons = 0
    for rank in range(world):
        left = read_steps(continuous / f"rank{rank}_steps.jsonl", range(1, total + 1))
        right = read_steps(
            resumed / f"rank{rank}_steps.jsonl", range(split + 1, total + 1)
        )
        a = json.loads((continuous / f"rank{rank}_mesh.json").read_text())
        b = json.loads((resumed / f"rank{rank}_mesh.json").read_text())
        assert a == b, f"rank{rank} resumed topology changed"
        for step, observed in right.items():
            reference = left[step]
            assert observed["batches"] == reference["batches"], (rank, step, "inputs")
            assert observed["metrics"] == reference["metrics"], (rank, step, "metrics")
            comparisons += 1
    return {
        "status": "passed",
        "scope": "same-layout actual rank groups, input batches, finite exact metrics",
        "rank_step_comparisons": comparisons,
        "continuous_steps": total,
        "resumed_from": split,
        "required_separate_check": "final full training checkpoint equality",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuous", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--split", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.continuous,
        args.resumed,
        world=args.world,
        total=args.total,
        split=args.split,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
