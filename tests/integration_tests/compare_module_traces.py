"""Compare local module trace tensors without silently dropping missing evidence."""

import argparse
import json
from pathlib import Path

import torch


def compare_traces(baseline, candidate):
    records = []
    for key in sorted(baseline.keys() | candidate.keys()):
        left, right = baseline.get(key), candidate.get(key)
        if not isinstance(left, torch.Tensor) and not isinstance(right, torch.Tensor):
            continue
        record = {"key": key}
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            record.update(status="missing_tensor", exact=False)
        elif left.shape != right.shape:
            record.update(
                status="shape_mismatch",
                exact=False,
                baseline_shape=list(left.shape),
                candidate_shape=list(right.shape),
            )
        else:
            difference = (left.double() - right.double()).abs()
            record.update(
                status="compared",
                exact=torch.equal(left, right),
                dtype_equal=left.dtype == right.dtype,
                finite=bool(torch.isfinite(difference).all()),
                max_ae=difference.max().item() if difference.numel() else 0.0,
                mean_ae=difference.mean().item() if difference.numel() else 0.0,
                different=int((left != right).sum()),
                count=left.numel(),
            )
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.ranks < 1 or args.threads < 1:
        parser.error("ranks and threads must be positive")
    torch.set_num_threads(args.threads)
    report = []
    for rank in range(args.ranks):
        paths = [
            root / f"rank{rank}" / "module-trace-step-1.pt"
            for root in (args.baseline, args.candidate)
        ]
        traces = [torch.load(p, weights_only=True, mmap=True) for p in paths]
        records = compare_traces(*traces)
        report.append({"rank": rank, "tensors": records})
        print(rank, [(r["key"], r.get("max_ae", r["status"])) for r in records])
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
