"""Match every actual rank's host input audit to the canonical corpus stream."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def verify(reference, audit: Path, *, world: int, steps: int):
    assert world > 0 and steps > 0
    expected = reference["degrees"]
    checked = 0
    ownership = Counter()
    degrees = set()
    for rank in range(world):
        mesh = json.loads((audit / f"rank{rank}_mesh.json").read_text())
        assert mesh["rank"] == rank
        dp_rank, degree = mesh["dp_coordinates"]
        assert degree in (4, 8) and 0 <= dp_rank < degree
        degrees.add(degree)
        ownership[dp_rank] += 1
        rows = [
            json.loads(line)
            for line in (audit / f"rank{rank}_steps.jsonl").read_text().splitlines()
        ]
        assert [row["step"] for row in rows] == list(range(1, steps + 1))
        for row in rows:
            target = expected[str(degree)][row["step"] - 1]
            assert target["step"] == row["step"]
            assert row["batches"] == [target["ranks"][dp_rank]], (rank, row["step"])
            checked += 1
    assert len(degrees) == 1, "inconsistent DP degrees across ranks"
    assert set(ownership) == set(range(degree)), "incomplete DP ownership"
    assert world % degree == 0 and set(ownership.values()) == {world // degree}, (
        "uneven replicated DP ownership"
    )
    return {
        "status": "passed",
        "rank_steps": checked,
        "scope": "actual host input/labels/positions/padding and valid-token counts",
        "note": "does not prove model numerics or formal throughput",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        json.loads(args.reference.read_text()),
        args.audit,
        world=args.world,
        steps=args.steps,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
