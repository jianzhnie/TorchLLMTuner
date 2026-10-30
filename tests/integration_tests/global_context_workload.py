"""Audit actual corpus batches across DP degrees without constructing a model.

Run inside the training environment with the target tokenizer and LIMO mounted:
    python -m tests.integration_tests.global_context_workload --output report.json
This is a CPU data audit, not NPU or cross-framework acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tests.caps import require_env

require_env("grain")

from examples.train_qwen3_30b_npu import qwen3_30b_npu_config
from llmtuner.datasets.loader_builder import build_dataloader


def fingerprint(batch):
    return {
        key: hashlib.sha256(batch[key].contiguous().numpy().tobytes()).hexdigest()
        for key in ("input", "labels", "positions", "padding_mask")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("at least two steps are required for independent recovery")
    cfg = qwen3_30b_npu_config()
    cfg.dataloader.global_context_rows = True
    assert cfg.global_batch_size == 8
    report = {"scope": "CPU actual LIMO loader audit", "steps": [], "degrees": [4, 8]}
    reference = None
    for world in (8, 4):
        loaders = []
        resumed = None
        try:
            for rank in range(world):
                loaders.append(
                    build_dataloader(
                        cfg,
                        dp_rank=rank,
                        dp_world_size=world,
                        num_tokens_per_batch=cfg.max_seq_len * 8 // world,
                    )
                )
            resumed = build_dataloader(
                cfg,
                dp_rank=0,
                dp_world_size=world,
                num_tokens_per_batch=cfg.max_seq_len * 8 // world,
            )
            iterators = [iter(loader) for loader in loaders]
            observed = []
            for step in range(args.steps):
                batches = [next(iterator) for iterator in iterators]
                rows = []
                for batch in batches:
                    for offset in range(8 // world):
                        row = {
                            key: batch[key].reshape(-1, cfg.max_seq_len)[offset]
                            for key in ("input", "labels", "positions", "padding_mask")
                        }
                        rows.append(fingerprint(row))
                valid = sum(int(batch["num_valid_tokens"]) for batch in batches)
                observed.append({"step": step + 1, "valid_tokens": valid, "rows": rows})
                if step == 0:
                    resumed.load_state_dict(loaders[0].state_dict())
                    replay = iter(resumed)
                else:
                    assert fingerprint(next(replay)) == fingerprint(batches[0])
            if reference is None:
                reference = observed
            else:
                assert observed == reference, "DP-dependent effective workload"
            report["steps"].append(
                {"dp": world, "batches": observed, "resume": "exact"}
            )
        finally:
            for loader in [*loaders, resumed]:
                if loader is not None:
                    loader.close()
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
