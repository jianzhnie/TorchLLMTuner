"""Observe local pre-clipping gradients in the real 30B Trainer.

Run with the same environment as examples.train_qwen3_30b_npu, replacing its
torchrun module with tests.integration_tests.npu_tp_gradient_diagnostic.
Use a fresh dump folder. Timings include instrumentation and are not throughput
evidence. Local norms require layout-aware reconstruction before comparison.
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from examples.npu_training_audit import AuditedNPUTrainer
from examples.train_qwen3_30b_npu import qwen3_30b_npu_config
from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.parallel.tensor_parallel.tensor_parallel import tp_sharded_param_ids


@torch.no_grad()
def local_gradient_records(model_parts, tp_ids: set[int]) -> list[dict]:
    records = []
    norms = []
    for part_index, part in enumerate(model_parts):
        for name, parameter in part.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            local = gradient.to_local() if isinstance(gradient, DTensor) else gradient
            records.append(
                {
                    "part": part_index,
                    "name": name,
                    "tp_sharded": id(parameter) in tp_ids,
                    "shape": list(gradient.shape),
                    "local_shape": list(local.shape),
                    "dtype": str(local.dtype),
                    "placements": [str(p) for p in gradient.placements]
                    if isinstance(gradient, DTensor)
                    else [],
                    "mesh_axes": list(gradient.device_mesh.mesh_dim_names)
                    if isinstance(gradient, DTensor)
                    else [],
                    "mesh_ranks": gradient.device_mesh.mesh.tolist()
                    if isinstance(gradient, DTensor)
                    else [],
                }
            )
            norms.append(torch.linalg.vector_norm(local, dtype=torch.float32))
    if norms:
        # One host transfer after local reductions; no extra norm collectives.
        values = torch.stack(norms).cpu().tolist()
        for record, value in zip(records, values, strict=True):
            record["local_l2_norm"] = value
    return records


class GradientDiagnosticTrainer(AuditedNPUTrainer):
    def __init__(self, config):
        super().__init__(config)
        self._gradient_output = (
            self._audit_path / f"rank{dist.get_rank()}_gradients.jsonl"
        )
        with synchronize_local_errors("Gradient diagnostic output creation"):
            # Per-rank exclusive creation avoids a shared-directory race.
            self._gradient_output.touch(exist_ok=False)

    def _clip_and_check_finite(self, **kwargs):
        with synchronize_local_errors("Pre-clipping gradient diagnostic"):
            records = local_gradient_records(
                self.model_parts, tp_sharded_param_ids(self.model_parts)
            )
            with self._gradient_output.open("a") as stream:
                stream.write(
                    json.dumps(
                        {"step": self.step, "parameters": records}, allow_nan=False
                    )
                    + "\n"
                )
        return super()._clip_and_check_finite(**kwargs)


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Gradient diagnostic requires a visible Ascend NPU")
    config = qwen3_30b_npu_config()
    parallel = config.parallel
    if parallel.fsdp_config.fsdp_cpu_offload:
        raise ValueError("This diagnostic does not support CPU-offloaded gradients")
    if (
        parallel.expert_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.cp_size != 1
    ):
        raise ValueError("This diagnostic requires EP=PP=CP=1")
    if not os.environ.get("LLMTUNER_DUMP_FOLDER"):
        raise ValueError("Set LLMTUNER_DUMP_FOLDER to a fresh diagnostic directory")
    torch.set_default_dtype(torch.bfloat16)
    GradientDiagnosticTrainer(config).train()


if __name__ == "__main__":
    main()
