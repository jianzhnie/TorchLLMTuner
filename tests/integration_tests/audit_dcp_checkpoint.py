"""Audit trusted native DCP files against an independently supplied contract.

This is a structural audit, not a numerical comparison or a restore test.
Only small control states are loaded. Model/moment payloads are never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    MetadataIndex,
    TensorStorageMetadata,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class AuditContract:
    """Obtain these fields from the recipe/live schema, never either checkpoint.

    expected_fqns includes model parameters AND persistent buffers. Optimizer
    tensors/options are separate explicit FQN-keyed production state schemas;
    this supports mixed optimizers without guessing state from model weights.
    cursor_schema maps flattened rank-relative Grain cursor paths to leaf types.
    ema=None explicitly disables EMA; otherwise it lists every EMA tensor/option.
    """

    world_size: int
    step: int
    ntokens_seen: int
    device_type: str
    parallel_degrees: dict[str, int]
    expected_fqns: dict[str, TensorSpec]
    optimizer_tensors: dict[str, TensorSpec]
    optimizer_values: dict[str, Any]
    optimizer_kinds: dict[str, str]
    optimizer_steps: dict[str, int]
    data_layout: dict[str, Any]
    cursor_schema: dict[str, str]
    ema: dict[str, TensorSpec] | None
    ema_values: dict[str, Any]
    model_prefix: str = "model."
    scheduler_values: dict[str, Any] | None = None
    max_control_bytes: int = 16 * 1024**2
    cursor_rank_values: dict[str, dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AuditContract:
        value = dict(value)
        for field in ("expected_fqns", "optimizer_tensors", "ema"):
            if value[field] is not None:
                value[field] = {
                    key: TensorSpec(tuple(spec["shape"]), spec["dtype"])
                    for key, spec in value[field].items()
                }
        for key in ("optimizer_values", "ema_values"):
            for name, item in value[key].items():
                if name.endswith((".betas", ".ns_coefficients")):
                    value[key][name] = tuple(
                        tuple(v) if isinstance(v, list) else v for v in item
                    )
        for rank_values in (value.get("cursor_rank_values") or {}).values():
            for key, item in rank_values.items():
                if value["cursor_schema"].get(key) == "tuple":
                    rank_values[key] = tuple(item)
        return cls(**value)

    def validate(self) -> None:
        for name in ("world_size", "step", "ntokens_seen", "max_control_bytes"):
            value = getattr(self, name)
            require(type(value) is int and value > 0, f"invalid contract {name}")
        axes = {
            "dp_replicate",
            "dp_shard",
            "cp",
            "tp",
            "pp",
            "ep",
            "cp_kv",
            "cp_ulysses",
        }
        require(set(self.parallel_degrees) == axes, "contract topology axes")
        require(
            all(type(v) is int and v > 0 for v in self.parallel_degrees.values()),
            "contract topology degrees",
        )
        p = self.parallel_degrees
        require(
            p["dp_replicate"] * p["dp_shard"] * p["cp"] * p["tp"] * p["pp"]
            == self.world_size,
            "contract world_size/topology mismatch",
        )
        require(bool(self.expected_fqns), "empty expectedFQNs contract")
        require(
            bool(self.optimizer_tensors) and bool(self.optimizer_values),
            "empty optimizer contract",
        )
        require(bool(self.optimizer_kinds), "optimizer ownership contract required")
        for fqn, kind in self.optimizer_kinds.items():
            require(fqn in self.expected_fqns, f"unknown optimizer FQN {fqn}")
            require(kind in {"Adam", "AdamW", "DistMuon"}, "unsupported optimizer")
            names = (
                ("momentum_buffer",)
                if kind == "DistMuon"
                else ("exp_avg", "exp_avg_sq", "step")
            )
            options = (
                (
                    "lr",
                    "weight_decay",
                    "momentum",
                    "nesterov",
                    "eps",
                    "ns_steps",
                    "ns_coefficients",
                    "adjust_lr_fn",
                )
                if kind == "DistMuon"
                else (
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "amsgrad",
                    "maximize",
                    "foreach",
                    "capturable",
                    "differentiable",
                    "fused",
                )
            )
            if kind != "DistMuon" and self.optimizer_values.get(
                f"param_groups.{fqn}.amsgrad"
            ):
                names = (*names, "max_exp_avg_sq")
            for name in names:
                key = f"state.{fqn}.{name}"
                require(key in self.optimizer_tensors, f"contract missing {key}")
                require(
                    self.optimizer_tensors[key].shape
                    == (() if name == "step" else self.expected_fqns[fqn].shape),
                    f"contract optimizer shape {key}",
                )
            if kind != "DistMuon":
                require(
                    fqn in self.optimizer_steps
                    and type(self.optimizer_steps[fqn]) is int
                    and self.optimizer_steps[fqn] >= 0,
                    f"contract optimizer step missing {fqn}",
                )
            for name in options:
                require(
                    f"param_groups.{fqn}.{name}" in self.optimizer_values,
                    f"contract missing optimizer option {fqn}.{name}",
                )
        layout = {
            "num_tokens_per_batch",
            "max_context_length",
            "num_rows_per_batch",
            "max_num_documents",
            "data_identity",
        }
        require(layout <= set(self.data_layout), "incomplete production data layout")
        require(bool(self.cursor_schema), "independent Grain cursor schema required")
        require(
            all(
                v in {"int", "float", "bool", "str", "bytes", "NoneType", "tuple"}
                for v in self.cursor_schema.values()
            ),
            "unsupported cursor leaf type",
        )
        rank_values = self.cursor_rank_values or {}
        ranks = {
            f"dp_rank_{r}"
            for r in range(
                self.parallel_degrees["dp_replicate"]
                * self.parallel_degrees["dp_shard"]
            )
        }
        require(set(rank_values) <= ranks, "unknown cursor rank")
        for rank in ranks:
            values = rank_values.get(rank, {})
            require(set(values) <= set(self.cursor_schema), "unknown cursor value")
            for key, kind in self.cursor_schema.items():
                if kind == "tuple":
                    require(
                        key in values
                        and type(values[key]) is tuple
                        and all(type(v) is int for v in values[key]),
                        "tuple cursor requires explicit per-rank integer tuple",
                    )
        require(
            self.ema is not None or not self.ema_values,
            "EMA disabled but options supplied",
        )
        require(self.ema is None or bool(self.ema), "empty enabled EMA contract")
        if self.scheduler_values is not None:
            require(
                self.scheduler_values.get("last_epoch") == self.step,
                "contract scheduler last_epoch mismatch",
            )


def validate_chunks(key: str, spec: TensorStorageMetadata) -> None:
    """Disjoint in-bounds boxes with equal total volume cover the full tensor."""
    shape = tuple(spec.size)
    require(all(type(s) is int and s >= 0 for s in shape), f"{key}: invalid size")
    boxes = []
    volume = 0
    for chunk in spec.chunks:
        offsets, sizes = tuple(chunk.offsets), tuple(chunk.sizes)
        require(len(offsets) == len(sizes) == len(shape), f"{key}: chunk ndim")
        require(
            all(
                type(o) is int and type(s) is int and o >= 0 and s >= 0 and o + s <= n
                for o, s, n in zip(offsets, sizes, shape, strict=True)
            ),
            f"{key}: chunk out of bounds",
        )
        count = math.prod(sizes)
        if not count:
            continue
        for old_offsets, old_sizes in boxes:
            overlap = all(
                a < b + t and b < a + s
                for a, s, b, t in zip(
                    offsets, sizes, old_offsets, old_sizes, strict=True
                )
            )
            require(not overlap, f"{key}: overlapping chunks")
        boxes.append((offsets, sizes))
        volume += count
    require(volume == math.prod(shape), f"{key}: incomplete chunk coverage")


def validate_checkpoint_storage(path: Path, metadata) -> dict[str, int]:
    """Validate geometry, exact storage indices and file ranges before payload IO."""
    saved = metadata.state_dict_metadata
    largest_tensor = 0
    largest_chunk = 0
    for key, spec in saved.items():
        if isinstance(spec, TensorStorageMetadata):
            validate_chunks(key, spec)
            element_bytes = torch.empty((), dtype=spec.properties.dtype).element_size()
            largest_tensor = max(largest_tensor, math.prod(spec.size) * element_bytes)
            largest_chunk = max(
                largest_chunk,
                max(
                    (math.prod(chunk.sizes) * element_bytes for chunk in spec.chunks),
                    default=0,
                ),
            )
        else:
            require(
                isinstance(spec, BytesStorageMetadata),
                f"{key}: unsupported storage metadata",
            )
    require(isinstance(metadata.storage_data, dict), "missing storage index mapping")
    largest_record = 0
    wanted_indices = set()
    for key, spec in saved.items():
        if isinstance(spec, TensorStorageMetadata):
            wanted_indices.update(
                MetadataIndex(key, chunk.offsets) for chunk in spec.chunks
            )
        else:
            wanted_indices.add(MetadataIndex(key))
    require(
        set(metadata.storage_data) == wanted_indices, "missing or extra storage indices"
    )
    ranges_by_file: dict[Path, list[tuple[int, int]]] = {}
    for index, info in metadata.storage_data.items():
        require(index.fqn in saved, "storage index absent from schema")
        require(
            type(info.offset) is int and type(info.length) is int,
            f"{index.fqn}: storage offset/length must be integers",
        )
        file = (path / info.relative_path).resolve()
        require(file.is_relative_to(path.resolve()), "storage path escapes checkpoint")
        require(
            file.is_file(), f"{index.fqn}: missing or non-regular storage file {file}"
        )
        require(
            info.offset >= 0
            and info.length > 0
            and info.offset + info.length <= file.stat().st_size,
            f"{index.fqn}: invalid storage file range",
        )
        ranges = ranges_by_file.setdefault(file, [])
        end = info.offset + info.length
        require(
            not any(info.offset < old_end and start < end for start, old_end in ranges),
            f"{index.fqn}: overlapping storage record ranges",
        )
        ranges.append((info.offset, end))
        largest_record = max(largest_record, info.length)
    return {
        "largest_logical_tensor_bytes": largest_tensor,
        "largest_serialized_record_bytes": largest_record,
        "largest_decoded_chunk_bytes": largest_chunk,
    }


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    # Contract layout is a dictionary tree; cursor paths are already flat DCP FQNs.
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def _validate_rng(payload: Any, contract: AuditContract, key: str) -> None:
    require(type(payload) is bytes, f"{key}: RNG payload must be bytes")
    state = pickle.loads(payload)
    fields = {
        "version",
        "world_size",
        "device_type",
        "parallel_degrees",
        "python",
        "numpy",
        "cpu",
        "device",
    }
    require(type(state) is dict and set(state) == fields, f"{key}: RNG schema")
    for field, expected in {
        "version": 1,
        "world_size": contract.world_size,
        "device_type": contract.device_type,
        "parallel_degrees": contract.parallel_degrees,
    }.items():
        require(
            type(state[field]) is type(expected) and state[field] == expected,
            f"{key}: RNG {field}",
        )
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(state["numpy"])
    torch.Generator(device="cpu").set_state(state["cpu"])
    device = state["device"]
    if contract.device_type == "cpu":
        require(device is None, f"{key}: CPU device RNG must be None")
    else:
        require(
            isinstance(device, torch.Tensor)
            and device.device.type == "cpu"
            and device.dtype == torch.uint8
            and device.ndim == 1
            and device.numel() > 0,
            f"{key}: invalid device RNG tensor",
        )


def audit_checkpoint(path: Path, contract: AuditContract) -> dict[str, Any]:
    contract.validate()
    reader = dcp.FileSystemReader(path)
    metadata = reader.read_metadata()
    saved = metadata.state_dict_metadata
    tensors = {
        f"{contract.model_prefix}{k}": v for k, v in contract.expected_fqns.items()
    }
    tensors.update({f"optimizer.{k}": v for k, v in contract.optimizer_tensors.items()})
    if contract.ema is not None:
        tensors.update({f"ema.{k}": v for k, v in contract.ema.items()})
    expected = {
        "train_state.step": contract.step,
        "train_state.ntokens_seen": contract.ntokens_seen,
        **{
            f"lr_scheduler.{k}": v
            for k, v in (
                contract.scheduler_values
                if contract.scheduler_values is not None
                else {"last_epoch": contract.step}
            ).items()
        },
        "dataloader.version": 1,
        "dataloader.dp_world_size": contract.parallel_degrees["dp_replicate"]
        * contract.parallel_degrees["dp_shard"],
        **{f"optimizer.{k}": v for k, v in contract.optimizer_values.items()},
        **{
            f"dataloader.batch_layout.{k}": v
            for k, v in _flatten(contract.data_layout).items()
        },
        **{f"ema.{k}": v for k, v in contract.ema_values.items()},
    }
    cursors = {
        f"dataloader.dp_rank_{rank}.{key}": kind
        for rank in range(expected["dataloader.dp_world_size"])
        for key, kind in contract.cursor_schema.items()
    }
    rng = {f"rng_state.rank_{rank}" for rank in range(contract.world_size)}
    wanted = set(tensors) | set(expected) | set(cursors) | rng
    require(
        len(wanted) == len(tensors) + len(expected) + len(cursors) + len(rng),
        "contract key categories overlap",
    )
    require(
        set(saved) == wanted,
        f"schema mismatch: missing={sorted(wanted - set(saved))}; "
        f"unexpected={sorted(set(saved) - wanted)}",
    )
    largest_tensor = 0
    for key, spec in saved.items():
        if isinstance(spec, TensorStorageMetadata):
            validate_chunks(key, spec)
            require(key in tensors, f"{key}: unexpected tensor storage")
            required = tensors[key]
            require(
                tuple(spec.size) == required.shape
                and str(spec.properties.dtype) == required.dtype,
                f"{key}: tensor shape/dtype mismatch",
            )
            largest_tensor = max(
                largest_tensor,
                math.prod(spec.size)
                * torch.empty((), dtype=spec.properties.dtype).element_size(),
            )
            for chunk in spec.chunks:
                require(
                    MetadataIndex(key, chunk.offsets) in metadata.storage_data,
                    f"{key}: missing tensor storage index",
                )
        else:
            require(
                isinstance(spec, BytesStorageMetadata) and key not in tensors,
                f"{key}: unexpected byte storage",
            )
    storage_inventory = validate_checkpoint_storage(path, metadata)
    largest_record = storage_inventory["largest_serialized_record_bytes"]
    steps = {
        f"optimizer.state.{fqn}.step": step
        for fqn, step in contract.optimizer_steps.items()
    }
    controls = set(expected) | set(cursors) | rng | set(steps)
    sizes = {}
    for key in controls:
        entries = [v for i, v in metadata.storage_data.items() if i.fqn == key]
        require(len(entries) == 1, f"{key}: missing/duplicate storage index")
        require(
            entries[0].length <= contract.max_control_bytes,
            f"{key}: control record exceeds memory preflight limit",
        )
        sizes[key] = entries[0].length
    groups, group, total = [], [], 0
    for key in sorted(controls):
        if group and total + sizes[key] > contract.max_control_bytes:
            groups.append(group)
            group, total = [], 0
        group.append(key)
        total += sizes[key]
    if group:
        groups.append(group)
    for keys in groups:
        state = {}
        for key in keys:
            if key in steps:
                require(
                    key in tensors and tensors[key].shape == (),
                    f"{key}: scalar optimizer step required",
                )
                state[key] = torch.empty((), dtype=saved[key].properties.dtype)
            else:
                require(
                    isinstance(saved[key], BytesStorageMetadata),
                    f"{key}: control storage",
                )
                state[key] = None
        dcp.load(
            state,
            storage_reader=reader,
            planner=DefaultLoadPlanner(
                flatten_state_dict=False, flatten_sharded_tensors=False
            ),
            no_dist=True,
        )
        for key, value in state.items():
            if key in steps:
                require(value.item() == steps[key], f"{key}: optimizer step mismatch")
            elif key in expected:
                require(
                    type(value) is type(expected[key]) and value == expected[key],
                    f"{key}: control value mismatch",
                )
            elif key in cursors:
                require(
                    type(value).__name__ == cursors[key], f"{key}: cursor leaf type"
                )
                _, rank, relative = key.split(".", 2)
                values = (contract.cursor_rank_values or {}).get(rank, {})
                if relative in values:
                    if cursors[key] == "tuple":
                        require(
                            all(type(v) is int for v in value),
                            f"{key}: cursor value must be an integer tuple",
                        )
                    require(value == values[relative], f"{key}: cursor value")
            else:
                _validate_rng(value, contract, key)
        del state
    return {
        "scope": "structural integrity only; not numerical equality or restore",
        "keys": len(saved),
        "largest_logical_tensor_bytes": largest_tensor,
        "largest_serialized_record_bytes": largest_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = AuditContract.from_dict(json.loads(args.contract.read_text()))
    print(json.dumps(audit_checkpoint(args.checkpoint, contract), indent=2))


if __name__ == "__main__":
    main()
