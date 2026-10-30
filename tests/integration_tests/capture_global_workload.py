"""Capture canonical CPU corpus fingerprints for actual training audit checks.

One global loader provides the reference for DP4 and DP8 local batches. This
checks data ownership, not model numerics, and does not use a process group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tests.caps import require_env

require_env("grain")

from examples.train_qwen3_30b_npu import qwen3_30b_npu_config
from llmtuner.components.loss import IGNORE_INDEX
from llmtuner.datasets.loader_builder import build_dataloader, corpus_data_identity

FIELDS = ("input", "labels", "positions", "padding_mask")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    cfg = qwen3_30b_npu_config()
    cfg.dataloader.global_context_rows = True
    assert cfg.global_batch_size == 8
    loader = build_dataloader(
        cfg,
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=cfg.max_seq_len * cfg.global_batch_size,
    )
    report = {
        "scope": "CPU canonical workload reference; no model or NPU acceptance",
        "data_identity": corpus_data_identity(
            cfg, dataset=cfg.dataloader.dataset, repeat=True
        ),
        "degrees": {"4": [], "8": []},
    }
    try:
        iterator = iter(loader)
        for step in range(1, args.steps + 1):
            batch = next(iterator)
            for world in (4, 8):
                local = cfg.max_seq_len * 8 // world
                ranks = []
                for rank in range(world):
                    fields = {}
                    for key in FIELDS:
                        value = batch[key][
                            rank * local : (rank + 1) * local
                        ].contiguous()
                        fields[key] = {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "sha256": hashlib.sha256(
                                value.numpy().tobytes()
                            ).hexdigest(),
                        }
                    labels = batch["labels"][rank * local : (rank + 1) * local]
                    ranks.append(
                        {
                            "fields": fields,
                            "valid_tokens": int((labels != IGNORE_INDEX).sum()),
                        }
                    )
                assert sum(r["valid_tokens"] for r in ranks) == int(
                    batch["num_valid_tokens"]
                )
                report["degrees"][str(world)].append({"step": step, "ranks": ranks})
    finally:
        loader.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Captured {args.steps} global batches for DP4/DP8", flush=True)


if __name__ == "__main__":
    main()
