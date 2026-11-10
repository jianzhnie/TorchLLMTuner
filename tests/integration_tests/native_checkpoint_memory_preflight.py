"""Payload-free admission for trusted torch.save native checkpoint records.

This estimates process retention across a sequential audit, not just its current
record pair. Pickle object expansion and allocator behavior are not hard bounded.
No torch.load or pickle decoding is performed by this preflight.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

GIB = 1024**3


def record_bytes(path: Path) -> int:
    """Include uncompressed ZIP bytes; reject legacy/unknown serialization."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlink record: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Unsupported non-ZIP torch serialization: {path}")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError(f"Duplicate ZIP entries: {path}")
        return max(path.stat().st_size, sum(entry.file_size for entry in entries))


def preflight(pairs: list[tuple[Path, Path]], budget_bytes: int) -> dict:
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ValueError("budget_bytes must be a positive integer")
    if not pairs:
        raise ValueError("No record pairs")
    sizes = [sum(record_bytes(path) for path in pair) for pair in pairs]
    # Charge all previously visited mapped/retained records, plus worst-case
    # current pair expansion, finite/equality temporaries, and interpreter reserve.
    estimate = 2 * GIB + 2 * sum(sizes) + 8 * max(sizes)
    report = {
        "record_pairs": len(pairs),
        "aggregate_record_bytes": sum(sizes),
        "largest_pair_bytes": max(sizes),
        "estimated_bytes": estimate,
        "budget_bytes": budget_bytes,
        "estimate_is_hard_bound": False,
        "payload_loads": 0,
        "admitted": estimate <= budget_bytes,
        "formula": "2GiB + 2*aggregate_record_bytes + 8*largest_pair_bytes",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--budget-gib", type=int, default=32)
    args = parser.parse_args()
    names = sorted(path.name for path in args.left.glob("*.pt"))
    if set(names) != {path.name for path in args.right.glob("*.pt")}:
        raise ValueError("Record sets differ")
    report = preflight(
        [(args.left / name, args.right / name) for name in names],
        args.budget_gib * GIB,
    )
    print(json.dumps(report, indent=2))
    if not report["admitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
