"""Verify actual rank-group observations for the four required eight-rank layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYOUTS = {
    "dp8": (8, 1, 1),
    "dp4_tp2": (4, 2, 1),
    "dp4_tp2_ep4": (4, 2, 4),
    "dp8_ep8": (8, 1, 8),
}


def verify(audit: Path, layout: str):
    dp, tp, ep = LAYOUTS[layout]
    for rank in range(8):
        row = json.loads((audit / f"rank{rank}_mesh.json").read_text())
        assert row["rank"] == rank
        assert row["dp_coordinates"] == [rank // tp, dp], rank
        dense = row["meshes"]["dense"]["local_groups"]
        sparse = row["meshes"]["sparse"]["local_groups"]
        assert dense == {
            "pp": [rank],
            "dp_replicate": [rank],
            "dp_shard": list(range(rank % tp, 8, tp)),
            "cp": [rank],
            "tp": list(range(rank // tp * tp, (rank // tp + 1) * tp)),
        }, (rank, "dense")
        assert sparse == {
            "pp": [rank],
            "dp_replicate": [rank],
            "efsdp": list(range(rank % ep, 8, ep)),
            "ep": list(range(rank // ep * ep, (rank // ep + 1) * ep)),
        }, (rank, "sparse")
    return {
        "status": "passed",
        "layout": layout,
        "world": 8,
        "dense_dp_shard": dp,
        "tp": tp,
        "ep": ep,
        "expert_fsdp": 8 // ep,
        "scope": "actual dense/sparse process group membership on every rank",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--layout", choices=LAYOUTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.audit, args.layout)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
