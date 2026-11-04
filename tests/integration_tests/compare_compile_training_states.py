"""Compare local diagnostic snapshots; this does not validate checkpoint resume.

Run with ``python -m tests.integration_tests.compare_compile_training_states
EXPECTED ACTUAL --world-size 8 --step 2``. All tensors must match exactly.
"""

import argparse
from pathlib import Path

import torch


def compare_training_states(
    expected: Path, actual: Path, *, world_size: int, step: int
) -> int:
    """Require complete rank coverage, schemas and exact finite tensor values."""
    if world_size < 1 or step < 1:
        raise ValueError("world_size and step must be positive")
    tensors = 0

    def compare(left, right, path):
        nonlocal tensors
        if isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                raise ValueError(f"{path}: snapshot keys differ")
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor):
                raise ValueError(f"{path}: expected a tensor")
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError(f"{path}: nonfinite tensor")
            try:
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            except AssertionError as error:
                raise AssertionError(f"{path}: {error}") from error
            tensors += 1
        elif left is not None or right is not None:
            raise ValueError(f"{path}: unsupported snapshot value")

    for rank in range(world_size):
        filename = f"rank{rank}-step{step}.pt"
        left, right = (
            torch.load(folder / filename, map_location="cpu", weights_only=True)
            for folder in (expected, actual)
        )
        for state in (left, right):
            if set(state) != {"logits", "candidate", "reference"}:
                raise ValueError(f"{filename}: incomplete snapshot")
            if not isinstance(state["logits"], torch.Tensor):
                raise ValueError(f"{filename}: missing logits")
            for label in ("candidate", "reference"):
                if not state[label]:
                    raise ValueError(f"{filename}: empty {label} state")
                for parameter in state[label].values():
                    if set(parameter) != {"weight", "gradient"}:
                        raise ValueError(f"{filename}: incomplete parameter state")
                    if not isinstance(parameter["weight"], torch.Tensor):
                        raise ValueError(f"{filename}: missing parameter weight")
        compare(left, right, filename)
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    count = compare_training_states(
        args.expected, args.actual, world_size=args.world_size, step=args.step
    )
    print(
        f"Exact finite tensor comparison passed: "
        f"{args.world_size} ranks, {count} tensors"
    )


if __name__ == "__main__":
    main()
