"""Reconstruct diagnostic L2 norms for the verified DP8 and DP4TP2 layouts.

Only single-axis Shard/Replicate gradient placements are supported. Reports are
diagnostic: no cross-layout or cross-framework acceptance threshold is applied.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path

from tests.integration_tests.verify_p0_topology import verify


def reconstruct(rank_records: list[list[dict]], *, dp: int, tp: int) -> dict:
    assert dp > 0 and tp > 0 and len(rank_records) == dp * tp
    parameters = []
    for records in rank_records:
        by_name = {(row["part"], row["name"]): row for row in records}
        assert len(by_name) == len(records), "duplicate gradient parameter"
        parameters.append(by_name)
    names = parameters[0].keys()
    assert names and all(row.keys() == names for row in parameters)
    squared_norms = []
    details = []
    for name in names:
        rows = [rank[name] for rank in parameters]
        assert len({row["tp_sharded"] for row in rows}) == 1
        assert all(row["placements"] == rows[0]["placements"] for row in rows)
        placements = rows[0]["placements"]
        assert len(placements) <= 1, "unsupported multi-axis gradient placement"
        placement = placements[0] if placements else "R"
        sharded = placement.startswith(("S(", "Shard(dim="))
        assert sharded or placement in ("R", "Replicate()"), (
            f"unsupported gradient placement: {placement}"
        )
        owners: dict[tuple[int, int], list[float]] = {}
        for rank, row in enumerate(rows):
            if placements:
                expected_mesh = list(range(rank % tp, dp * tp, tp))
                assert row["mesh_ranks"] == expected_mesh, (
                    f"unsupported gradient mesh for rank {rank}: "
                    f"{row['mesh_ranks']}; expected {expected_mesh}"
                )
            else:
                assert row["mesh_ranks"] == [], "plain gradient has a mesh"
            value = row["local_l2_norm"]
            assert math.isfinite(value) and value >= 0, name
            owner = (
                rank // tp if sharded else 0,
                rank % tp if row["tp_sharded"] else 0,
            )
            owners.setdefault(owner, []).append(value)
        # Keep one copy of a replicated contribution. Report discrepancies
        # between replicas separately rather than averaging them away.
        power = math.fsum(values[0] ** 2 for values in owners.values())
        squared_norms.append(power)
        details.append(
            {
                "part": name[0],
                "name": name[1],
                "l2_norm": math.sqrt(power),
                "owners": len(owners),
                "max_replica_norm_spread": max(
                    max(values) - min(values) for values in owners.values()
                ),
            }
        )
    return {"l2_norm": math.sqrt(math.fsum(squared_norms)), "parameters": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--layout", choices=("dp8", "dp4_tp2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    topology = verify(args.audit, args.layout)
    results = []
    with ExitStack() as stack:
        gradients = [
            stack.enter_context((args.audit / f"rank{rank}_gradients.jsonl").open())
            for rank in range(8)
        ]
        metrics = [
            stack.enter_context((args.audit / f"rank{rank}_steps.jsonl").open())
            for rank in range(8)
        ]
        for step, lines in enumerate(zip(*gradients, *metrics, strict=True), 1):
            rows = [json.loads(line) for line in lines]
            assert all(row["step"] == step for row in rows), "step coverage differs"
            observed = [row["metrics"]["grad_norm"] for row in rows[8:]]
            assert all(math.isfinite(value) and value >= 0 for value in observed)
            result = reconstruct(
                [row["parameters"] for row in rows[:8]],
                dp=topology["dense_dp_shard"],
                tp=topology["tp"],
            )
            result.update(
                step=step,
                trainer_l2_norm=observed[0],
                absolute_difference=abs(result["l2_norm"] - observed[0]),
                trainer_rank_norm_spread=max(observed) - min(observed),
            )
            results.append(result)
    assert results, "empty diagnostic"
    args.output.write_text(
        json.dumps(
            {"status": "diagnostic_only", "topology": topology, "steps": results},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
