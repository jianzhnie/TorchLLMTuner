"""Capture only specified canonical expert chunks and their real optimizer state."""

import json
import re
from pathlib import Path

from torch.distributed.tensor import DTensor, Shard

from tests.integration_tests.alignment_parameter_hash import canonical_parameter_views


def _local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def targeted_expert_records(parameters, names, trainer, selection):
    targets = json.loads(Path(selection).read_text())
    if len({tuple(t) for t in targets}) != len(targets):
        raise ValueError("target chunks must have unique canonical coordinates")
    for canonical, expert, start, end in targets:
        if not re.fullmatch(r"layers\.\d+\.experts\.(gate|up|down)", canonical):
            raise ValueError("target capture requires canonical expert projections")
        if not (0 <= expert < 128 and 0 <= start < end):
            raise ValueError("invalid target chunk coordinates")
    container = getattr(trainer, "optimizers", getattr(trainer, "optimizer", None))
    optimizers = list(getattr(container, "optimizers", [container]))
    records = {}
    for parameter in parameters:
        name = names[id(parameter)]
        if not any(token in name for token in ("inner_experts.", "mlp.experts.")):
            continue
        offset = 0
        if isinstance(parameter, DTensor):
            if parameter.device_mesh.ndim != 1 or parameter.placements != (Shard(0),):
                raise ValueError("target capture requires one-dimensional Shard(0)")
            length, offset = parameter.placements[0]._local_shard_size_and_offset(
                parameter.shape[0],
                parameter.device_mesh.size(),
                parameter.device_mesh.get_local_rank(),
            )
            if length != parameter.to_local().shape[0]:
                raise ValueError("target capture ownership mismatch")
        values = {"parameter": _local(parameter.detach())}
        if parameter.grad is not None:
            values["gradient"] = _local(parameter.grad.detach())
        states = [
            optimizer.state[parameter]
            for optimizer in optimizers
            if optimizer is not None and parameter in optimizer.state
        ]
        if len(states) > 1:
            raise ValueError("target parameter belongs to multiple optimizer states")
        state = states[0] if states else {}
        for field in ("exp_avg", "exp_avg_sq"):
            if field in state:
                if state[field].shape != parameter.shape:
                    raise ValueError("optimizer moment geometry differs from parameter")
                values[field] = _local(state[field].detach())
        views = {
            field: dict(canonical_parameter_views(name, value))
            for field, value in values.items()
        }
        for canonical, expert, start, end in targets:
            if canonical not in views["parameter"]:
                continue
            projection = views["parameter"][canonical]
            index = expert - offset
            if not 0 <= index < projection.shape[0]:
                continue
            if end > projection.shape[1]:
                raise ValueError("target chunk exceeds canonical projection rows")
            key = f"{canonical}/expert{expert}/rows{start}:{end}"
            if key in records:
                raise ValueError("target coordinate has duplicate parameter ownership")
            records[key] = {
                "source_parameter": name,
                "global_shape": tuple(parameter.shape),
                "local_shape": tuple(_local(parameter).shape),
                "expert_offset": offset,
                "optimizer_state_present": bool(states),
                "optimizer_step": state["step"].detach().cpu().clone()
                if "step" in state
                else None,
                **{
                    field: projections[canonical][index, start:end]
                    .detach()
                    .cpu()
                    .clone()
                    for field, projections in views.items()
                },
            }
    return records
