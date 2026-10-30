"""Optional rank-local observations of the real NPU example's training steps.

The synchronized timings include batch loading and fingerprint overhead, but
exclude checkpoint saving and validation outside train_step. They are not an
uninstrumented throughput baseline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist

from llmtuner import Trainer


class AuditedNPUTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self._audit_path = Path(config.dump_folder) / "audit"
        self._audit_path.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank()
        topology = {}
        views = {
            **self.parallel_dims._global_meshes,
            **self.parallel_dims._single_axis_meshes,
            **{
                "/".join(names): mesh
                for names, mesh in self.parallel_dims._multi_axis_meshes.items()
            },
        }
        for name, mesh in views.items():
            topology[name] = {
                "ranks": mesh.mesh.cpu().tolist(),
                "axes": list(mesh.mesh_dim_names),
                "local_groups": {
                    axis: dist.get_process_group_ranks(mesh.get_group(axis))
                    for axis in mesh.mesh_dim_names
                },
            }
        (self._audit_path / f"rank{rank}_mesh.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "dp_coordinates": self.dp_rank_world_size(),
                    "meshes": topology,
                },
                indent=2,
            )
            + "\n"
        )
        # Append supports a restart in the same dump folder; each record names
        # the completed global step. Failed steps never receive a success row.
        self._audit_steps = self._audit_path / f"rank{rank}_steps.jsonl"

    def train_step(self, data_iterator):
        observed = []

        def batches():
            while True:
                batch = next(data_iterator)
                fields = {}
                for name in ("input", "labels", "positions", "padding_mask"):
                    value = batch[name].detach().cpu().contiguous()
                    fields[name] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                    }
                observed.append(
                    {"fields": fields, "valid_tokens": int(batch["num_valid_tokens"])}
                )
                yield batch

        torch.npu.synchronize()
        start = perf_counter()
        metrics = super().train_step(batches())
        torch.npu.synchronize()
        elapsed = perf_counter() - start
        record = {
            "step": self.step,
            "seconds": elapsed,
            "batches": observed,
            "metrics": metrics,
            "timing_scope": (
                "train_step including input audit; excludes save/validation"
            ),
        }
        with self._audit_steps.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return metrics
