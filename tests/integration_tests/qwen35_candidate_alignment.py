"""Actual candidate Trainer, full Qwen3.5/LIMO, shared initialization, 100 steps.

Run later in the existing NPU environment, from the repository root::
    torchrun --standalone --nproc-per-node=8 -m \
      tests.integration_tests.qwen35_candidate_alignment \
      --model /original/HF/model --data /original/LIMO \
      --shared-hf /tmp/tlt_qwen35_shared_hf_v1 \
      --output /scratch/candidate --artifacts /scratch/candidate-observations

Independent full-state resume adds --resume /scratch/candidate/checkpoint/step-50
and --verify-batches-from /scratch/candidate-observations, with NEW output and
artifact directories. The horizon remains 100, including the scheduler.

Uses recipe build_config and Trainer.train/train_step without a new training
loop. The original loader remains registered in the checkpointer; only its
consumption iterator is observed. Every host batch is saved with a deterministic
content hash. Resume checks every consumed batch against the continuous run.
Step50 additionally records model/Adam local-shard hashes and the actual
scheduler/RNG/loader/train-state payload fingerprints observed during native
synchronous DCP saving. Resume compares them immediately after checkpoint load,
before iterator construction; this is the saved-state boundary, not a claim
that subsequent iterator/profiler setup cannot consume RNG. Existing step51
batch replay checks remain in place. No checkpoint objects or loop are replaced.
Every step records actual loss, global valid targets, preclip norm, applied LR,
and full-logit hash/statistics. Full logits and reduced preclip gradient/local
updated-weight shards are dumped only at selected steps (default 1,50,100).
Full logits require unchunked CE, so this tool explicitly disables chunked loss.
Production GDN/router/experts/rotary numerical policies are explicit opt-ins,
recorded in the contract: defaults are native GDN/experts and None router/rotary
precision. Selected policies change arithmetic through existing production
adapters; the actual recipe/Trainer loop remains in use. No dependency
installation or reference execution is performed by this entry.
Numerical cross-framework acceptance remains pending the reference comparison.
Before Trainer construction, disk preflight combines checkpoint/observation
budgets by filesystem and requires the upper estimate plus 20% (at least 32 GiB)
headroom. Startup source identity is fixed before runtime imports; per-step and
final records flag subsequent entry-file edits. Versions include torch-npu.
"""

from __future__ import annotations

# Startup source capture must precede runtime imports.
# ruff: noqa: E402
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import shutil
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

# Capture before optional/runtime imports or long checkpoint hashing can run.
ENTRY_STARTUP_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Replicate, Shard

from examples.qwen3_5.train import build_config
from llmtuner import Trainer
from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.components.checkpointer.core.checkpoint_keys import DATALOADER
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.datasets.contracts import TrainerBatch
from llmtuner.utils.module_names import canonical_fqn


def state_fingerprint(value: Any) -> str:
    """Value-based fingerprints, independent of tensor pickle/storage identity."""

    def describe(item):
        if isinstance(item, torch.Tensor):
            return {
                "tensor": tensor_info(item, empty_local_shard=not bool(item.numel()))
            }
        if isinstance(item, np.ndarray):
            return {
                "numpy": str(item.dtype),
                "shape": list(item.shape),
                "sha256": hashlib.sha256(item.tobytes()).hexdigest(),
            }
        if isinstance(item, np.generic):
            return describe(item.item())
        if isinstance(item, dict):
            entries = [(describe(key), describe(value)) for key, value in item.items()]
            return {
                "dict": sorted(
                    entries, key=lambda pair: json.dumps(pair[0], sort_keys=True)
                )
            }
        if isinstance(item, tuple | list):
            return {type(item).__name__: [describe(v) for v in item]}
        if isinstance(item, bytes):
            return {"bytes_sha256": hashlib.sha256(item).hexdigest()}
        if item is None or isinstance(item, str | bool | int | float):
            return {type(item).__name__: item}
        raise TypeError(f"Unsupported checkpoint fingerprint state: {type(item)}")

    encoded = json.dumps(describe(value), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_state_fingerprint(category: str, state: Any) -> str:
    # RNGCheckpointState stores a trusted local pickle with tensor payloads.
    # Decode it so tensor storage IDs never affect equality across processes.
    if category == "rng_state":
        state = {key: pickle.loads(value) for key, value in state.items()}
    return state_fingerprint(state)


def model_adam_manifest(model, optimizers) -> dict:
    names = {
        parameter: canonical_fqn(name) for name, parameter in model.named_parameters()
    }
    result = {"model": {}, "adam": {}, "param_groups": []}
    for parameter, name in names.items():
        local, layout = local_shard(parameter)
        result["model"][name] = {
            **layout,
            **tensor_info(local, empty_local_shard=not bool(local.numel())),
        }
    for index, optimizer in enumerate(optimizers):
        assert isinstance(optimizer, torch.optim.AdamW)
        for group in optimizer.param_groups:
            metadata = {key: value for key, value in group.items() if key != "params"}
            result["param_groups"].append(
                {
                    "optimizer": index,
                    "parameters": [names[p] for p in group["params"]],
                    "options_sha256": state_fingerprint(metadata),
                }
            )
            for parameter in group["params"]:
                state = optimizer.state[parameter]
                assert {"step", "exp_avg", "exp_avg_sq"} <= state.keys()
                entries = {}
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        local, layout = local_shard(value)
                        entries[key] = {
                            **layout,
                            **tensor_info(
                                local, empty_local_shard=not bool(local.numel())
                            ),
                        }
                    else:
                        entries[key] = {"value_sha256": state_fingerprint(value)}
                result["adam"][f"{index}:{names[parameter]}"] = entries
    return result


@contextmanager
def observe_checkpoint_payloads(states, fingerprints):
    """Observe existing state_dict calls, returning each original payload unchanged."""
    restored = []
    try:
        with synchronize_local_errors("Alignment checkpoint observer setup"):
            for category in ("lr_scheduler", DATALOADER, "train_state", "rng_state"):
                state = states[category]
                original = state.state_dict
                had_override = "state_dict" in vars(state)
                old_override = vars(state).get("state_dict")

                def observed(
                    original=original,
                    category=category,
                    state=state,
                    had_override=had_override,
                    old_override=old_override,
                ):
                    # Some schedulers serialize their __dict__. Never let the
                    # observation method itself leak into the checkpoint payload.
                    hook = vars(state)["state_dict"]
                    if had_override:
                        state.state_dict = old_override
                    else:
                        del state.state_dict
                    try:
                        payload = original()
                    finally:
                        state.state_dict = hook
                    digest = checkpoint_state_fingerprint(category, payload)
                    assert fingerprints.setdefault(category, digest) == digest, (
                        f"Checkpoint callback state changed within save: {category}"
                    )
                    return payload

                restored.append((state, had_override, old_override))
                state.state_dict = observed
        yield
    finally:
        primary = sys.exc_info()[1]
        try:
            with synchronize_local_errors("Alignment checkpoint observer cleanup"):
                cleanup_error = None
                for state, had_override, old_override in reversed(restored):
                    try:
                        if had_override:
                            state.state_dict = old_override
                        elif "state_dict" in vars(state):
                            del state.state_dict
                    except BaseException as error:
                        if cleanup_error is None:
                            cleanup_error = error
                if cleanup_error is not None:
                    raise cleanup_error
        except BaseException as cleanup_error:
            if primary is None:
                raise
            # Retain the original coordinated save/setup exception, while
            # reporting cleanup errors without masking its traceback.
            try:
                print(
                    f"Checkpoint observer cleanup also failed: {cleanup_error}",
                    file=sys.stderr,
                )
            except BaseException:
                pass


def initial_hf_key(name: str) -> str:
    """Pure FSDP/AC wrapper naming only; EP layouts are intentionally unsupported."""
    name = canonical_fqn(name)
    assert name.startswith("model."), f"Unexpected candidate parameter: {name}"
    return name[len("model.") :]


def verify_initial_slice(actual, expected, key: str) -> dict:
    assert actual.dtype == expected.dtype == torch.bfloat16, key
    assert actual.shape == expected.shape and torch.equal(actual, expected), (
        f"Loaded initial shard differs: {key}"
    )
    return tensor_info(actual, empty_local_shard=not bool(actual.numel()))


def verify_initial_shards(model, shared: Path) -> dict:
    index = json.loads((shared / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    grouped = {}
    seen = set()
    for name, parameter in model.named_parameters():
        key = initial_hf_key(name)
        assert key in weight_map and key not in seen, key
        assert isinstance(parameter, DTensor), f"Unsharded initial parameter: {name}"
        local, layout = local_shard(parameter)
        grouped.setdefault(weight_map[key], []).append((key, local, layout))
        seen.add(key)
    assert seen == set(weight_map), "Initial checkpoint/model key coverage differs"
    records = {}
    for filename, entries in grouped.items():
        assert Path(filename).name == filename, filename
        with safe_open(shared / filename, framework="pt", device="cpu") as tensors:
            for key, local, layout in entries:
                view = tensors.get_slice(key)
                assert list(view.get_shape()) == layout["global_shape"], key
                start = layout["offset"]
                # Read only this rank's dim-0 shard, including legitimate empty
                # shards for [1,D] gates. Never materialize a full model/layer.
                expected = view[start : start + local.shape[0]]
                actual = local.detach().cpu()
                records[key] = {
                    **layout,
                    **verify_initial_slice(actual, expected, key),
                }
    return {"exact": True, "tensor_count": len(records), "tensors": records}


def verify_runtime_precision(model) -> dict:
    assert all(p.dtype == torch.bfloat16 for p in model.parameters()), (
        "Expected BF16 parameter storage"
    )
    states = set()
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        state = module._get_fsdp_state()
        if id(state) in states:
            continue
        states.add(id(state))
        policy = state._mp_policy
        assert policy.param_dtype == torch.bfloat16
        assert policy.reduce_dtype == torch.float32
        assert policy.cast_forward_inputs is False
        group = state._fsdp_param_group
        if group is not None:
            assert group.mp_policy == policy
            assert group.gradient_divide_factor == 1.0
    assert states, "No runtime FSDP states found"
    return {
        "fsdp_states": len(states),
        "parameter_dtype": "torch.bfloat16",
        "reduce_dtype": "torch.float32",
        "cast_forward_inputs": False,
        "gradient_divide_factor": 1.0,
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def cpu_batch(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Expected original CPU loader batch")
        return value.detach().contiguous().clone()
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: cpu_batch(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(cpu_batch(v) for v in value)
    if value is None or isinstance(value, str | bool | int | float):
        return value
    raise TypeError(f"Unsupported raw batch value: {type(value)}")


def tensor_info(value: torch.Tensor, *, empty_local_shard: bool = False) -> dict:
    if (not value.numel() and not empty_local_shard) or value.layout != torch.strided:
        raise ValueError("Expected nonempty dense tensor")
    value = value.detach().cpu().contiguous()
    nonzero = False
    for chunk in value.reshape(-1).split(1024 * 1024):
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("Nonfinite observed tensor")
        nonzero |= bool(torch.count_nonzero(chunk))
    digest = hashlib.sha256()
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return {
        "sha256": digest.hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nonzero": nonzero,
        "bytes": value.numel() * value.element_size(),
        "empty_local_shard": not bool(value.numel()),
    }


def batch_hash(value: Any) -> str:
    def descriptor(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return {"tensor": tensor_info(item)}
        if isinstance(item, dict):
            return {k: descriptor(item[k]) for k in sorted(item)}
        if isinstance(item, tuple | list):
            return {type(item).__name__: [descriptor(v) for v in item]}
        return {type(item).__name__: item}

    encoded = json.dumps(descriptor(value), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def local_shard(value: torch.Tensor) -> tuple[torch.Tensor, dict]:
    assert value.numel() > 0, "Empty global parameter/gradient"
    if not isinstance(value, DTensor):
        return value, {"global_shape": list(value.shape), "replicated": True}
    axes = [
        axis
        for axis, placement in enumerate(value.placements)
        if isinstance(placement, Shard)
    ]
    assert len(axes) == 1, "Expected pure FSDP: one sharded mesh axis"
    axis = axes[0]
    assert value.placements[axis].dim == 0
    assert all(isinstance(p, Shard | Replicate) for p in value.placements)
    assert value.device_mesh.shape[axis] == 8
    coordinate = value.device_mesh.get_coordinate()
    chunk_size = (value.shape[0] + 7) // 8
    start = min(chunk_size * coordinate[axis], value.shape[0])
    expected = list(value.shape)
    expected[0] = min(chunk_size, value.shape[0] - start)
    local = value.to_local()
    assert list(local.shape) == expected, "Unexpected FSDP local shard shape"
    # E.g. [1,D] shared_expert_gate legitimately has empty shards on ranks 1..7.
    return local, {
        "global_shape": list(value.shape),
        "shard_dim": 0,
        "offset": start,
        "mesh_shape": list(value.device_mesh.shape),
        # JSON checkpoints encode sequences as lists; keep the in-memory
        # manifest identical when comparing a freshly restored shard.
        "coordinate": list(coordinate),
        "placements": [str(p) for p in value.placements],
    }


class BatchRecorder:
    """Wrap consumption only: checkpointer still owns the ORIGINAL loader."""

    def __init__(self, source: Iterator, trainer: AlignmentTrainer):
        self.source, self.trainer = source, trainer

    def __iter__(self) -> BatchRecorder:
        return self

    def __next__(self) -> TrainerBatch:
        batch = next(self.source)
        trainer = self.trainer
        if not isinstance(batch, dict):
            raise TypeError("LIMO recipe must yield a native TrainerBatch dictionary")
        raw = cpu_batch(batch)
        digest = batch_hash(raw)
        index = len(trainer.batch_records)
        name = f"step-{trainer.step:05d}-batch-{index:03d}.pt"
        if trainer.options.verify_batches_from is not None:
            path = (
                trainer.options.verify_batches_from
                / trainer.rank_dir.name
                / "batches"
                / name
            )
            baseline = torch.load(path, map_location="cpu", weights_only=True)
            if batch_hash(baseline["batch"]) != baseline["sha256"]:
                raise ValueError(f"Corrupt baseline batch: {path}")
            if baseline["sha256"] != digest:
                raise ValueError(f"Resume data mismatch (including step51): {path}")
        path = trainer.rank_dir / "batches" / name
        with path.open("xb") as stream:
            torch.save({"batch": raw, "sha256": digest}, stream)
        trainer.batch_records.append(
            {
                "file": str(path.relative_to(trainer.rank_dir)),
                "sha256": digest,
                "valid_targets": trainer.count_valid_tokens(batch),
            }
        )
        return batch


class AlignmentTrainer(Trainer):
    """Observation hooks only; optimizer ordering and loop remain in Trainer."""

    def __init__(self, cfg: Any, options: argparse.Namespace, contract: dict):
        self.options, self.contract = options, contract
        self.rank_dir = options.artifacts / f"rank-{dist.get_rank():05d}"
        with synchronize_local_errors("Alignment artifact directories"):
            self.rank_dir.mkdir(parents=True, exist_ok=False)
            (self.rank_dir / "batches").mkdir()
        super().__init__(cfg)
        with synchronize_local_errors("Alignment artifact setup"):
            write_json(
                self.rank_dir / "run.json",
                {
                    "contract": contract,
                    "training_output": str(Path(options.output).resolve()),
                    "resume": str(options.resume) if options.resume else None,
                    "snapshot_steps": options.snapshot_steps,
                    "logits_steps": options.logits_steps,
                    "acceptance": "Candidate only; reference acceptance pending",
                    "disk_estimate": disk_estimate(options),
                    "entry_source": entry_source_state(),
                },
            )
            assert {
                "model",
                "optimizer",
                "lr_scheduler",
                "train_state",
                "rng_state",
                DATALOADER,
            } <= self.checkpointer.states.keys()
            assert self.checkpointer.states[DATALOADER] is self.dataloader
            assert self._chunked_loss_num_chunks == 1 and len(self.model_parts) == 1
            dims = self.parallel_dims
            assert dims.dp_shard == 8 and dims.dp_replicate == 1
            assert (dims.tp, dims.cp, dims.pp, dims.ep) == (1, 1, 1, 1)
            text = self.model.model.config
            assert (text.num_hidden_layers, text.hidden_size, text.num_experts) == (
                40,
                2048,
                256,
            )
            assert sum(p.numel() for p in self.model.parameters()) == 34_660_610_688
            if options.verify_batches_from is not None:
                original = json.loads(
                    (
                        options.verify_batches_from / self.rank_dir.name / "run.json"
                    ).read_text()
                )
                assert original["contract"] == contract, (
                    "Continuous/resume contract differs"
                )
                assert (
                    options.resume.resolve()
                    == (
                        Path(original["training_output"]) / "checkpoint" / "step-50"
                    ).resolve()
                ), "Resume must use the continuous run's own step50"
        original_load = self.checkpointer.load

        def checked_load(*args, **kwargs):
            loaded = original_load(*args, **kwargs)
            with synchronize_local_errors("Alignment initial/resume verification"):
                if options.resume:
                    baseline_path = (
                        options.verify_batches_from
                        / self.rank_dir.name
                        / "checkpoint-step-00050.json"
                    )
                    baseline = json.loads(baseline_path.read_text())
                    assert baseline["step"] == 50
                    assert (
                        Path(baseline["checkpoint"]).resolve()
                        == options.resume.resolve()
                    )
                    assert (
                        file_hash(options.resume / ".metadata")
                        == baseline["metadata_sha256"]
                    )
                    assert baseline["model_adam"] == model_adam_manifest(
                        self.model, self.optimizer
                    ), "Restored model/Adam local shards or groups differ from step50"
                    fingerprints = {
                        category: checkpoint_state_fingerprint(
                            category, self.checkpointer.states[category].state_dict()
                        )
                        for category in (
                            "lr_scheduler",
                            DATALOADER,
                            "train_state",
                            "rng_state",
                        )
                    }
                    assert fingerprints == baseline["state_fingerprints"], (
                        "Restored scheduler/RNG/loader/train state differs from "
                        "saved boundary"
                    )
                    write_json(
                        self.rank_dir / "resume-step50-exact.json",
                        {
                            "exact": True,
                            "before_data_iterator": True,
                            "baseline_sha256": file_hash(baseline_path),
                            "state_fingerprints": fingerprints,
                        },
                    )
                expected = 50 if options.resume else 0
                assert loaded and self.step == expected, (loaded, self.step, expected)
                assert self.lr_scheduler.state_dict()["last_epoch"] == expected
                precision = verify_runtime_precision(self.model)
                source_identity = verify_source_identity(contract)
                assert self.model.model.config._attn_implementation == "sdpa"
                write_json(
                    self.rank_dir / "loaded-source-identity.json", source_identity
                )
                initial_verification = None
                if not options.resume:
                    initial_verification = verify_initial_shards(
                        self.model, options.shared_hf
                    )
                    write_json(
                        self.rank_dir / "initial-shards-exact.json",
                        initial_verification,
                    )
                for optimizer in self.optimizer:
                    assert isinstance(optimizer, torch.optim.AdamW)
                    for group in optimizer.param_groups:
                        assert group["lr"] == options.lr
                        assert tuple(group["betas"]) == (0.9, 0.999)
                        assert group["eps"] == 1e-8 and group["weight_decay"] == 0
                        assert not group["fused"] and not group["foreach"]
                    if options.resume:
                        for group in optimizer.param_groups:
                            for parameter in group["params"]:
                                state = optimizer.state[parameter]
                                assert {"step", "exp_avg", "exp_avg_sq"} <= state.keys()
                                assert int(state["step"]) == 50
                                for key in ("exp_avg", "exp_avg_sq"):
                                    moment, _ = local_shard(state[key])
                                    tensor_info(
                                        moment,
                                        empty_local_shard=not bool(moment.numel()),
                                    )
                    else:
                        assert not optimizer.state, "Fresh run has stale Adam state"
                self.validate_weights()
                write_json(
                    self.rank_dir / "loaded.json",
                    {
                        "loaded": True,
                        "step": self.step,
                        "scheduler": self.lr_scheduler.state_dict(),
                        "parameter_dtypes": sorted(
                            {str(p.dtype) for p in self.model.parameters()}
                        ),
                        "runtime_precision": precision,
                        "initial_shards_exact": initial_verification is not None,
                        "adam_moment_dtypes": sorted(
                            {
                                str(state[key].dtype)
                                for optimizer in self.optimizer
                                for state in optimizer.state.values()
                                for key in ("exp_avg", "exp_avg_sq")
                                if key in state
                            }
                        ),
                        "original_loader_checkpointed": self.checkpointer.states[
                            DATALOADER
                        ]
                        is self.dataloader,
                    },
                )
            return loaded

        self.checkpointer.load = checked_load
        original_save = self.checkpointer.save

        def checked_save(curr_step, last_step=False):
            if curr_step != 50:
                return original_save(curr_step, last_step=last_step)
            # Only this backend's verified synchronous native DCP boundary is
            # supported. Never claim an async/live-state snapshot is identical.
            assert type(self.checkpointer).__module__.endswith(
                "backends.distributed_checkpoint"
            )
            assert self.checkpointer.async_mode.value == "disabled"
            fingerprints = {}
            with synchronize_local_errors("Alignment step50 pre-save manifest"):
                before = model_adam_manifest(self.model, self.optimizer)
            with observe_checkpoint_payloads(self.checkpointer.states, fingerprints):
                saved = original_save(curr_step, last_step=last_step)
            with synchronize_local_errors("Alignment step50 saved-boundary manifest"):
                assert saved, "Step50 checkpoint was not saved"
                assert before == model_adam_manifest(self.model, self.optimizer), (
                    "Model/Adam changed during save; boundary cannot be certified"
                )
                assert set(fingerprints) == {
                    "lr_scheduler",
                    DATALOADER,
                    "train_state",
                    "rng_state",
                }
                checkpoint = Path(self.checkpointer.folder) / "step-50"
                metadata = checkpoint / ".metadata"
                assert metadata.is_file(), (
                    "Synchronous checkpoint metadata not published"
                )
                write_json(
                    self.rank_dir / "checkpoint-step-00050.json",
                    {
                        "step": 50,
                        "checkpoint": str(checkpoint.resolve()),
                        "metadata_sha256": file_hash(metadata),
                        "model_adam": before,
                        "state_fingerprints": fingerprints,
                        "boundary": (
                            "Actual state_dict payloads during synchronous DCP save"
                        ),
                    },
                )
            return saved

        self.checkpointer.save = checked_save

    def data_iterator(self) -> Iterator[TrainerBatch]:
        assert self.checkpointer.states[DATALOADER] is self.dataloader
        return BatchRecorder(super().data_iterator(), self)

    def validate_weights(self) -> None:
        nonzero = False
        for parameter in self.model.parameters():
            local, _ = local_shard(parameter)
            # CPU transfer is one local tensor at a time, never a full model.
            nonzero |= tensor_info(local, empty_local_shard=not bool(local.numel()))[
                "nonzero"
            ]
        assert nonzero, "Entire local model is zero"

    def dump_shards(self, kind: str) -> None:
        groups: dict[str, list] = {}
        for name, parameter in self.model.named_parameters():
            parts = name.split(".")
            group = parts[parts.index("layers") + 1] if "layers" in parts else "outer"
            groups.setdefault(group, []).append((name, parameter))
        folder = self.rank_dir / f"step-{self.step:05d}-{kind}"
        folder.mkdir(exist_ok=False)
        index, any_nonzero = {}, False
        for group, parameters in groups.items():
            tensors, layouts = {}, {}
            for name, parameter in parameters:
                value = parameter.grad if kind == "gradients-preclip" else parameter
                assert value is not None, f"Missing {kind}: {name}"
                value, layout = local_shard(value)
                cpu = value.detach().cpu().contiguous().clone()
                layouts[name] = {
                    **layout,
                    **tensor_info(cpu, empty_local_shard=not bool(cpu.numel())),
                }
                any_nonzero |= layouts[name]["nonzero"]
                tensors[name] = cpu
            path = folder / f"{group}.pt"
            with path.open("xb") as stream:
                torch.save(tensors, stream)
            index[path.name] = {"sha256": file_hash(path), "tensors": layouts}
            del tensors, cpu, value
        assert any_nonzero, f"Entire {kind} local snapshot is zero"
        write_json(
            folder / "index.json", {"kind": kind, "step": self.step, "files": index}
        )

    def before_gradient_clipping(self) -> None:
        super().before_gradient_clipping()
        with synchronize_local_errors("Alignment preclip snapshot"):
            assert not self.preclip_seen, "Duplicate preclip hook"
            self.preclip_seen = True
            if self.step in self.options.snapshot_steps:
                self.dump_shards("gradients-preclip")

    @contextmanager
    def param_context(self):
        captures = []

        def capture(module, inputs, output):
            with synchronize_local_errors("Alignment original logits capture"):
                assert not captures, (
                    "Duplicate outer forward/AC capture; refusing overwrite"
                )
                assert isinstance(output, torch.Tensor) and output.shape[-1] == 248320
                cpu = output.detach().cpu().contiguous()
                info = tensor_info(cpu)
                info.update(min=float(cpu.min()), max=float(cpu.max()))
                assert info["nonzero"], "Entire logits tensor is zero"
                if self.step in self.options.logits_steps:
                    filename = (
                        f"step-{self.step:05d}-logits-{len(self.logit_records):03d}.pt"
                    )
                    path = self.rank_dir / filename
                    with path.open("xb") as stream:
                        torch.save(cpu, stream)
                    info["file"] = path.name
                captures.append(info)

        with super().param_context():
            handle = self.model.register_forward_hook(capture)
            try:
                yield
            finally:
                handle.remove()
        assert len(captures) == 1, "No original forward logits captured"
        self.logit_records.extend(captures)

    def forward_backward_step(self, microbatch, *, global_valid_tokens):
        count = int(global_valid_tokens.detach().cpu())
        assert count > 0
        self.denominators.append(count)
        loss = super().forward_backward_step(
            microbatch, global_valid_tokens=global_valid_tokens
        )
        assert torch.isfinite(loss).all()
        self.local_losses.append(float(loss.detach().cpu()))
        return loss

    def train_step(self, data_iterator):
        self.batch_records, self.logit_records = [], []
        self.denominators, self.local_losses = [], []
        self.preclip_seen = False
        applied_lr = self.lr_scheduler.get_metrics()
        result = super().train_step(data_iterator)
        with synchronize_local_errors("Alignment step metrics/snapshot"):
            assert result is not None and {"loss", "grad_norm"} <= result.keys()
            assert self.preclip_seen
            assert result["loss"] > 0 and result["grad_norm"] > 0
            expected = self.cfg.gradient_accumulation_steps
            assert (
                len(self.batch_records)
                == len(self.logit_records)
                == len(self.local_losses)
                == expected
            )
            assert (
                len(self.denominators) == expected and len(set(self.denominators)) == 1
            )
            assert all(value == self.options.lr for value in applied_lr.values())
            assert all(result[key] == value for key, value in applied_lr.items())
            assert all(math.isfinite(value) for value in result.values())
            assert self.lr_scheduler.state_dict()["last_epoch"] == self.step
            if self.step in self.options.snapshot_steps:
                self.dump_shards("weights-after-update")
            write_json(
                self.rank_dir / f"step-{self.step:05d}.json",
                {
                    "step": self.step,
                    "global_valid_targets": self.denominators[0],
                    "local_valid_targets": sum(
                        v["valid_targets"] for v in self.batch_records
                    ),
                    "local_loss_sum": sum(self.local_losses),
                    "metrics": result,
                    "applied_lr": applied_lr,
                    "batches": self.batch_records,
                    "logits": self.logit_records,
                    "preclip_hook_seen": True,
                    "entry_source": entry_source_state(),
                },
            )
        # Check the observed host counts against the Trainer's actual denominator.
        total = torch.tensor(
            sum(v["valid_targets"] for v in self.batch_records), device=self.device
        )
        dist.all_reduce(total)
        assert int(total) == self.denominators[0], (
            "Global target normalization mismatch"
        )
        return result


def make_config(args: argparse.Namespace):
    recipe = argparse.Namespace(
        model=str(args.model),
        data=str(args.data),
        output=args.output,
        layers=None,
        steps=100,
        batch=args.batch,
        sequence_length=args.sequence_length,
        compile_backend="none",
        linear_attention_backend=getattr(args, "linear_attention_backend", "native"),
        router_compute_dtype=getattr(args, "router_compute_dtype", None),
        experts_implementation=getattr(args, "experts_implementation", "native"),
        gdn_numerical_policy=getattr(args, "gdn_numerical_policy", "native"),
        gdn_allow_right_padding=getattr(args, "gdn_allow_right_padding", False),
        gdn_deterministic_reduction=getattr(args, "gdn_deterministic_reduction", False),
        rotary_compute_dtype=getattr(args, "rotary_compute_dtype", None),
        rotary_cache_dtype=getattr(args, "rotary_cache_dtype", None),
        checkpoint_interval=50,
        load_step=-1,
    )
    cfg = build_config(recipe)
    cfg.optimizer = OptimizerConfig(
        optimizer_type="AdamW",
        learning_rate=args.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.0,
        implementation="for-loop",
        lr_scheduler_config=LRSchedulerConfig(
            total_steps=100, warmup_steps=0, decay_ratio=0.0, min_lr_factor=0.0
        ),
    )
    cfg.training.gradient_accumulation_steps = args.accumulation
    cfg.training.chunked_loss_num_chunks = 1
    cfg.training.grad_norm_precision = "float32"
    cfg.training.max_norm = 1.0
    checkpoint = cfg.training.checkpoint_config
    checkpoint.initial_load_path = str(args.resume or args.shared_hf)
    checkpoint.initial_load_in_hf = args.resume is None
    checkpoint.initial_load_model_only = args.resume is None
    checkpoint.last_save_model_only = False
    checkpoint.async_mode = "disabled"
    checkpoint.keep_latest_k = 0
    return cfg


def hf_source_paths() -> dict:
    # Defining modules, not package __init__ or a repository reference copy.
    from_names = (
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
        "transformers.modeling_utils",
        "transformers.masking_utils",
        "transformers.integrations.sdpa_attention",
        "transformers.utils.generic",
        "transformers.utils.output_capturing",
        "transformers.activations",
        "transformers.initialization",
        "transformers.modeling_rope_utils",
        "transformers.modeling_layers",
        "transformers.cache_utils",
        "transformers.modeling_attn_mask_utils",
        "transformers.utils.import_utils",
    )
    paths = {
        name: Path(importlib.import_module(name).__file__).resolve()
        for name in from_names
    }
    # Experts/kernel decorators and dispatch are version-owned integration
    # modules. Include this small complete tree to cover their transitive calls.
    integration = Path(
        importlib.import_module("transformers.integrations").__file__
    ).parent
    paths.update(
        {
            "transformers.integrations/"
            + str(path.relative_to(integration)): path.resolve()
            for path in integration.rglob("*.py")
        }
    )
    return paths


def verify_source_identity(contract: dict) -> dict:
    root = Path(__file__).resolve().parents[2]
    assert file_hash(Path(__file__)) == ENTRY_STARTUP_SHA256 == contract["entry_sha256"]
    current_tree = {
        str(path.relative_to(root)) for path in (root / "llmtuner").rglob("*.py")
    }
    assert current_tree == set(contract["numerical_sources_sha256"]), (
        "Framework source tree added/removed files"
    )
    for name, digest in contract["numerical_sources_sha256"].items():
        assert file_hash(root / name) == digest, f"Changed source: {name}"
    hf_paths = hf_source_paths()
    assert hf_paths.keys() == contract["hf_sources_sha256"].keys(), (
        "HF source tree changed"
    )
    for name, path in hf_paths.items():
        assert file_hash(path) == contract["hf_sources_sha256"][name], name
    assert (
        file_hash(Path(Trainer.__init__.__code__.co_filename))
        == contract["trainer_sha256"]
    )
    assert (
        file_hash(Path(build_config.__code__.co_filename)) == contract["recipe_sha256"]
    )
    return {
        "source_contract_version": 2,
        "verified": True,
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
    }


def source_contract(args: argparse.Namespace) -> dict:
    shared = args.shared_hf
    assert importlib.metadata.version("transformers").startswith("5.13.")
    index = json.loads((shared / "model.safetensors.index.json").read_text())
    assert len(index["weight_map"]) == 693
    assert index["metadata"]["total_size"] == 69_321_221_376
    provenance_path = shared / "initialization_provenance.json"
    assert index["metadata"]["initialization_provenance_sha256"] == file_hash(
        provenance_path
    )
    provenance = json.loads(provenance_path.read_text())
    assert provenance["config"]["original_sha256"] == file_hash(
        args.model / "config.json"
    )
    assert provenance["config"]["text_config_sha256"] == file_hash(
        shared / "config.json"
    )
    # Streaming hashes only; no initial tensors are deserialized by this entry.
    assert set(index["weight_map"].values()) == set(provenance["shards"])
    for name, metadata in provenance["shards"].items():
        assert Path(name).name == name and name.endswith(".safetensors")
        assert file_hash(shared / name) == metadata["sha256"], (
            f"Changed shared shard: {name}"
        )
    tokenizer = {
        p.name: file_hash(p)
        for p in sorted(args.model.iterdir())
        if p.is_file()
        and (p.name.startswith(("tokenizer", "special_tokens", "chat_template")))
    }
    assert tokenizer, "Missing original tokenizer assets"
    root = Path(__file__).resolve().parents[2]
    numerical_sources = (
        "llmtuner/models/hf/attention_backend.py",
        "llmtuner/parallel/expert_parallel/token_dispatch.py",
        "llmtuner/models/common/scatter_add.py",
        "llmtuner/models/hf/expert_kernels.py",
        "llmtuner/models/hf/gdn_numerics.py",
        "llmtuner/models/hf/padding_metadata.py",
        "llmtuner/models/common/moe/gate.py",
        "llmtuner/models/hf/qkv_projection.py",
        "llmtuner/models/hf/normalization_setup.py",
        "llmtuner/models/hf/normalization.py",
        "llmtuner/models/hf/residual_normalization.py",
        "llmtuner/models/hf/linear_attention.py",
        "llmtuner/models/hf/model.py",
        "llmtuner/models/hf/model_setup.py",
        "llmtuner/models/hf/materialization.py",
        "llmtuner/models/hf/state_dict_adapter.py",
        "llmtuner/accelerator/gated_delta.py",
        "llmtuner/accelerator/collectives.py",
        "llmtuner/parallel/fully_shard_parallel/fully_shard_parallel.py",
        "llmtuner/parallel/fully_shard_parallel/plan.py",
        "llmtuner/parallel/fully_shard_parallel/policies.py",
        "llmtuner/trainer/runtime/assembly.py",
        "llmtuner/trainer/runtime/batch.py",
        "llmtuner/trainer/runtime/reductions.py",
        "llmtuner/trainer/runtime/gradients.py",
        "llmtuner/components/loss.py",
        "llmtuner/components/optimizer/optimizer.py",
        "llmtuner/memory/activation_checkpoint.py",
        "llmtuner/components/checkpointer/core/base.py",
        "llmtuner/components/checkpointer/core/random_state.py",
        "llmtuner/components/checkpointer/backends/distributed_checkpoint.py",
        "llmtuner/datasets/dataloader.py",
    )
    # Capture complete local behavior trees, including transitive data/runtime
    # helpers, rather than only the public entry files.
    source_paths = set(numerical_sources)
    source_paths.update(
        str(path.relative_to(root)) for path in (root / "llmtuner").rglob("*.py")
    )
    helper_hashes = {name: file_hash(root / name) for name in sorted(source_paths)}
    import_sources = hf_source_paths()
    external_hashes = {name: file_hash(path) for name, path in import_sources.items()}
    return {
        "source_contract_version": 2,
        "resolved_attention_backend": "sdpa",
        "hf_sources_sha256": external_hashes,
        "horizon": 100,
        "world_size": 8,
        "geometry": "35B-A3B full 40 layers",
        "shared_index_sha256": file_hash(shared / "model.safetensors.index.json"),
        "data_sha256": file_hash(args.data),
        "tokenizer_hashes": tokenizer,
        "global_batch_size_per_accumulation": args.batch,
        "accumulation": args.accumulation,
        "sequence_length": args.sequence_length,
        "linear_attention_backend": getattr(args, "linear_attention_backend", "native"),
        "router_compute_dtype": getattr(args, "router_compute_dtype", None),
        "experts_implementation": getattr(args, "experts_implementation", "native"),
        "gdn_numerical_policy": getattr(args, "gdn_numerical_policy", "native"),
        "gdn_allow_right_padding": getattr(args, "gdn_allow_right_padding", False),
        "gdn_deterministic_reduction": getattr(
            args, "gdn_deterministic_reduction", False
        ),
        "rotary_compute_dtype": getattr(args, "rotary_compute_dtype", None),
        "rotary_cache_dtype": getattr(args, "rotary_cache_dtype", None),
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_num_documents": 1,
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "implementation": "for-loop",
        },
        "scheduler": "constant; warmup=0; decay_ratio=0; horizon=100",
        "precision": "BF16 parameters, FP32 FSDP reduction/gradnorm",
        "activation_checkpoint": "full",
        "max_norm": 1.0,
        "deterministic": True,
        "compile": False,
        "normalization": "sum token CE / global valid targets; FSDP gradient divide=1",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-npu", "transformers")
        },
        "entry_sha256": ENTRY_STARTUP_SHA256,
        "trainer_sha256": file_hash(Path(Trainer.__init__.__code__.co_filename)),
        "recipe_sha256": file_hash(Path(build_config.__code__.co_filename)),
        "numerical_sources_sha256": helper_hashes,
    }


def disk_estimate(args: argparse.Namespace) -> dict:
    # Logical aggregate across all eight ranks; filesystem overhead is additional.
    params = 34_660_610_688
    steps = [s for s in args.snapshot_steps if not args.resume or s > 50]
    logit_steps = [s for s in args.logits_steps if not args.resume or s > 50]
    logits_elements = (
        len(logit_steps)
        * args.batch
        * args.accumulation
        * args.sequence_length
        * 248320
    )
    checkpoints = 1 if args.resume else 2
    # Inputs/labels/positions int64 plus bool padding, torch.save/JSON overhead.
    batches = (50 if args.resume else 100) * args.accumulation
    batch_metadata_bytes = (
        batches * args.batch * args.sequence_length * 25
        + batches * 8 * 16 * 1024
        + 512 * 1024**2
    )
    return {
        "scope": "aggregate all 8 ranks, excludes filesystem/staging overhead",
        "snapshot_steps_executed": steps,
        "full_logits_steps_executed": logit_steps,
        "observations_bytes_lower": len(steps) * params * 4 + logits_elements * 2,
        "observations_bytes_upper": len(steps) * params * 6 + logits_elements * 4,
        "batch_and_metadata_bytes_upper": batch_metadata_bytes,
        "full_checkpoints": checkpoints,
        "checkpoint_bytes_lower": checkpoints * params * 6,
        "checkpoint_bytes_upper": checkpoints * params * 10,
        "basis": "BF16 weights; BF16..FP32 grads/moments/logits; no compression",
    }


def entry_source_state() -> dict:
    """Keep the launched source identity even if the file is edited mid-run."""
    try:
        current = file_hash(Path(__file__))
        return {
            "startup_sha256": ENTRY_STARTUP_SHA256,
            "on_disk_sha256": current,
            "changed_since_startup": current != ENTRY_STARTUP_SHA256,
        }
    except OSError as error:
        return {
            "startup_sha256": ENTRY_STARTUP_SHA256,
            "on_disk_sha256": None,
            "changed_since_startup": True,
            "read_error": str(error),
        }


def record_failure(path: Path | None, error: BaseException) -> None:
    """Failure reporting must never replace the primary exception (e.g. ENOSPC)."""
    try:
        if path is None:
            return  # Fresh output ownership has not yet been established.
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            path,
            {
                "error": f"{type(error).__name__}: {error}",
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
                "entry_source": entry_source_state(),
            },
        )
    except BaseException as reporting_error:
        try:
            print(
                f"Could not write {path}: {reporting_error}; "
                f"preserving original {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        except BaseException:
            pass  # Even a closed stderr must not obscure the original exception.


def disk_preflight(args: argparse.Namespace) -> dict:
    """Combine destinations on the same filesystem; do not double-count free space."""
    estimate = disk_estimate(args)
    budgets = (
        (Path(args.output).resolve(), estimate["checkpoint_bytes_upper"]),
        (
            args.artifacts,
            estimate["observations_bytes_upper"]
            + estimate["batch_and_metadata_bytes_upper"],
        ),
    )
    filesystems: dict[int, dict] = {}
    for destination, budget in budgets:
        anchor = destination
        while not anchor.exists():
            anchor = anchor.parent
        device = anchor.stat().st_dev
        usage = shutil.disk_usage(anchor)
        entry = filesystems.setdefault(
            device,
            {
                "device": device,
                "existing_path": str(anchor),
                "destinations": {},
                "estimated_bytes": 0,
                "free_bytes": usage.free,
            },
        )
        entry["destinations"][str(destination)] = budget
        entry["estimated_bytes"] += budget
        entry["free_bytes"] = min(entry["free_bytes"], usage.free)
    for entry in filesystems.values():
        # Fresh synchronous checkpoints: 20% extra, at least 32 GiB per filesystem.
        entry["headroom_bytes"] = max((entry["estimated_bytes"] + 4) // 5, 32 * 1024**3)
        entry["required_free_bytes"] = (
            entry["estimated_bytes"] + entry["headroom_bytes"]
        )
    result = {
        "estimate": estimate,
        "headroom_policy": "20%, minimum 32 GiB per filesystem",
        "filesystems": list(filesystems.values()),
    }
    insufficient = [
        entry
        for entry in filesystems.values()
        if entry["free_bytes"] < entry["required_free_bytes"]
    ]
    if insufficient:
        raise OSError(
            f"Insufficient disk space for selected captures/checkpoints: "
            f"{json.dumps(result, sort_keys=True)}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Original HF config/tokenizer directory",
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Original limo.jsonl or its directory"
    )
    parser.add_argument(
        "--shared-hf", type=Path, default=Path("/tmp/tlt_qwen35_shared_hf_v1")
    )
    parser.add_argument(
        "--output", required=True, help="Fresh actual Trainer output directory"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Separate fresh observation directory",
    )
    parser.add_argument(
        "--resume", type=Path, help="Own continuous run's complete checkpoint/step-50"
    )
    parser.add_argument(
        "--verify-batches-from",
        type=Path,
        help="Continuous run's observation directory; required for resume",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--router-compute-dtype", choices=("float32",))
    parser.add_argument(
        "--experts-implementation", choices=("native", "pre_down"), default="native"
    )
    parser.add_argument("--rotary-compute-dtype", choices=("float32",))
    parser.add_argument(
        "--gdn-numerical-policy",
        choices=("native", "split_qkv_fp32_norm"),
        default="native",
    )
    parser.add_argument("--gdn-allow-right-padding", action="store_true")
    parser.add_argument("--gdn-deterministic-reduction", action="store_true")
    parser.add_argument("--rotary-cache-dtype", choices=("float32",))
    parser.add_argument(
        "--linear-attention-backend", choices=("native", "npu_triton"), default="native"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Fixed AdamW LR for every update"
    )
    parser.add_argument("--snapshot-steps", type=int, nargs="+", default=[1, 50, 100])
    parser.add_argument("--logits-steps", type=int, nargs="+", default=[1, 50, 100])
    args = parser.parse_args()
    assert int(os.environ.get("WORLD_SIZE", "1")) == 8, (
        "Require 8 ranks; no single-rank model"
    )
    assert (args.resume is None) == (args.verify_batches_from is None)
    assert args.batch > 0 and args.batch % 8 == 0 and args.accumulation > 0
    assert args.sequence_length > 0 and math.isfinite(args.lr) and args.lr > 0
    for name in (
        "model",
        "data",
        "shared_hf",
        "artifacts",
        "resume",
        "verify_batches_from",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.data.is_dir():
        args.data = args.data / "limo.jsonl"
    args.snapshot_steps = sorted(set(args.snapshot_steps))
    args.logits_steps = sorted(set(args.logits_steps))
    assert all(1 <= step <= 100 for step in args.snapshot_steps + args.logits_steps)
    output = Path(args.output).resolve()
    assert not (
        output == args.artifacts
        or output in args.artifacts.parents
        or args.artifacts in output.parents
    )
    if args.resume:
        assert args.resume.name == "step-50" and (args.resume / ".metadata").is_file()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(minutes=60))
    trainer = None
    failure_path = None
    try:
        contract_list = [None]
        storage_list = [None]
        with synchronize_local_errors("Alignment source preflight"):
            if dist.get_rank() == 0:
                for path in (output, args.artifacts):
                    assert not path.exists() or (
                        path.is_dir() and not any(path.iterdir())
                    ), f"Stale/nonempty output: {path}"
                storage_list[0] = disk_preflight(args)
                contract_list[0] = source_contract(args)
        dist.broadcast_object_list(contract_list)
        dist.broadcast_object_list(storage_list)
        rank_dir = args.artifacts / f"rank-{dist.get_rank():05d}"
        failure_path = rank_dir / "failure.json"
        with synchronize_local_errors("Alignment rank environment agreement"):
            assert contract_list[0]["entry_sha256"] == ENTRY_STARTUP_SHA256
            assert contract_list[0]["versions"] == {
                name: importlib.metadata.version(name)
                for name in ("torch", "torch-npu", "transformers")
            }
        torch.set_default_dtype(torch.bfloat16)
        trainer = AlignmentTrainer(make_config(args), args, contract_list[0])
        with synchronize_local_errors("Alignment disk preflight recording"):
            write_json(trainer.rank_dir / "disk-preflight.json", storage_list[0])
        trainer.train()
        assert trainer.step == 100, f"Training ended early: {trainer.step}"
        with synchronize_local_errors("Alignment completion verification"):
            source_identity = verify_source_identity(contract_list[0])
            assert trainer.model.model.config._attn_implementation == "sdpa"
            assert (output / "checkpoint" / "step-100" / ".metadata").is_file()
            if not args.resume:
                assert (output / "checkpoint" / "step-50" / ".metadata").is_file()
            expected = range(51 if args.resume else 1, 101)
            assert all(
                (trainer.rank_dir / f"step-{step:05d}.json").is_file()
                for step in expected
            )
            write_json(
                trainer.rank_dir / "complete.json",
                {
                    "source_identity": source_identity,
                    "resolved_attention_backend": (
                        trainer.model.model.config._attn_implementation
                    ),
                    "status": "candidate_complete_reference_pending",
                    "step": 100,
                    "resume_step": 50 if args.resume else 0,
                    "entry_source": entry_source_state(),
                },
            )
    except BaseException as error:
        record_failure(failure_path, error)
        raise
    finally:
        # close()/process-group teardown can also fail after a constructor error.
        # Attempt both, while preserving the first active exception.
        primary_error = sys.exc_info()[1]
        cleanup_error = None
        operations = ([trainer.close] if trainer is not None else []) + [
            dist.destroy_process_group
        ]
        for operation in operations:
            try:
                operation()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if primary_error is None and cleanup_error is not None:
            record_failure(failure_path, cleanup_error)
            raise cleanup_error


if __name__ == "__main__":
    main()
