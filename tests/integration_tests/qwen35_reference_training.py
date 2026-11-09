"""Real TorchTitan v0.3 TrainerEx, full Qwen3.5 MoE, candidate CPU replay.

Use the existing reference environment (no dependency installation)::

    python -m tests.integration_tests.qwen35_reference_training --phase cpu-check
    python -m tests.integration_tests.qwen35_reference_training --phase prepare \
      --model /original/HF --shared-hf /scratch/shared-hf --native /scratch/native
    torchrun --standalone --nproc-per-node=8 -m \
      tests.integration_tests.qwen35_reference_training --phase train \
      --model /original/HF --shared-hf /scratch/shared-hf --native /scratch/native \
      --replay /scratch/candidate-observations --output /scratch/reference \
      --artifacts /scratch/reference-observations

Independent resume: add --resume /scratch/reference/checkpoint/step-50 and
--verify-from /scratch/reference-observations, with NEW output/artifact paths.
The same candidate replay and native initialization are required. Horizon=100.

cpu-check builds only meta models and tiny CPU optimizers; it does NOT build a
Trainer or call any NPU API. prepare writes model-only flat DCP on CPU and needs
roughly 70+ GB host RAM for the complete BF16 initialization. Neither phase is
training acceptance. Only train calls TrainerEx.train, without a custom loop.

Snapshots at 1/50/100 gather one native parameter at a time (all ranks), convert
one layer at a time on CPU, then save canonical HF5.13 dim-0 shards. This is
observation overhead, not a throughput benchmark. Logits are the original
outer forward, gradients are reduced/preclip, updates are FP32 differences of
BF16 post/pre-step weights. No world-size division is applied: reference FSDP
uses SUM and its loss already divides by global valid targets. Native gradient
clipping is retained, including its norm accumulation dtype; this entry does not
silently substitute candidate FP32 clipping or relax cross-framework tolerances.
The exact resume checks are within this reference only. Cross-framework
acceptance still requires the original comparator and its original tolerances.
Local IO/validation errors are coordinated before subsequent collectives;
failures inside TrainerEx construction or a collective remain runtime failures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import pickle
import random
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torchtitan
import torchtitan_npu  # noqa: F401 -- activate patches before Trainer imports
from safetensors import safe_open
from torch.distributed.tensor import DTensor, Replicate, Shard
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan_npu.config.configs import OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.torchtitan.components.checkpoint import (
    CheckpointManager,
)
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.qwen3_5 import model_registry
from torchtitan_npu.override.qwen3_5.parallelize import parallelize_qwen3_5_npu

PARAMETERS = 34_660_610_688
EXPERT_SUFFIXES = (".experts.gate_up_proj", ".experts.down_proj")
SNAPSHOTS = (1, 50, 100)


@contextmanager
def synchronize_local_errors(device):
    """Agree on local failures before entering the next world collective.

    Does not recover a failed collective or failures inside TrainerEx itself.
    """
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    if dist.is_initialized():
        flag = torch.tensor(int(error is not None), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if int(flag):
            if error is not None:
                raise error
            raise RuntimeError("Another reference rank failed local validation/IO")
    elif error is not None:
        raise error


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def save_tensor_file(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        torch.save(value, stream)


def tensor_info(value: torch.Tensor) -> dict:
    """Match candidate tensor descriptors exactly, including empty shard flag."""
    if value.layout != torch.strided:
        raise ValueError("Expected dense strided tensor")
    value = value.detach().cpu().contiguous()
    nonzero = False
    for chunk in value.reshape(-1).split(1024 * 1024):
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("Nonfinite observation")
        nonzero |= bool(torch.count_nonzero(chunk))
    digest = hashlib.sha256(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return {
        "sha256": digest.hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nonzero": nonzero,
        "bytes": value.numel() * value.element_size(),
        "empty_local_shard": not bool(value.numel()),
    }


def batch_hash(value: Any) -> str:
    def describe(item):
        if isinstance(item, torch.Tensor):
            return {"tensor": tensor_info(item)}
        if isinstance(item, dict):
            return {key: describe(item[key]) for key in sorted(item)}
        if isinstance(item, tuple | list):
            return {type(item).__name__: [describe(v) for v in item]}
        return {type(item).__name__: item}

    return hashlib.sha256(
        json.dumps(describe(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def boundary(state: dict, *, to_reference: bool) -> dict:
    """HF5.13 text layout <-> v0.3 adapter layout; never alter model maths."""
    result = {}
    for key, value in state.items():
        if to_reference and key.startswith("model."):
            assert not key.startswith("model.language_model."), "Already converted"
            key = "model.language_model." + key[6:]
        elif not to_reference and key.startswith("model.language_model."):
            key = "model." + key[len("model.language_model.") :]
        if key.endswith(EXPERT_SUFFIXES):
            value = value.transpose(-1, -2).contiguous()
        assert key not in result
        result[key] = value
    return result


def from_hf_group(adapter, source, shared_head=None):
    """Convert partial groups without the adapter synthesizing tied embeddings."""
    state = boundary(source, to_reference=True)
    injected = "lm_head.weight" not in state
    if injected:
        # Preserve the adapter's genuine shared-head boundary, even for layers.
        assert shared_head is not None and shared_head.ndim == 2
        assert shared_head.numel() > 0
        state["lm_head.weight"] = shared_head
    converted = adapter.from_hf(state)
    if injected:
        del converted["lm_head.weight"]
    return converted


def canonical_native_fqn(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "").replace("._orig_mod", "")


def shared_head_tensor(shared: Path, weight_map: dict) -> torch.Tensor:
    filename = weight_map["lm_head.weight"]
    assert Path(filename).name == filename
    with safe_open(shared / filename, framework="pt", device="cpu") as tensors:
        head = tensors.get_tensor("lm_head.weight")
    assert head.ndim == 2 and head.numel() and head.dtype == torch.bfloat16
    tensor_info(head)
    return head


def state_fingerprint(value: Any) -> str:
    """Value-based fingerprints, independent of tensor pickle/storage identity."""

    def describe(item):
        if isinstance(item, torch.Tensor):
            return {"tensor": tensor_info(item)}
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
    if category == "rng_state":
        state = {key: pickle.loads(value) for key, value in state.items()}
    return state_fingerprint(state)


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
        "coordinate": list(coordinate) if coordinate is not None else None,
        "placements": [str(p) for p in value.placements],
    }


def model_adam_manifest(model, optimizers) -> dict:
    names = {
        parameter: canonical_native_fqn(name)
        for name, parameter in model.named_parameters()
    }
    assert len(set(names.values())) == len(names), "Duplicate canonical FQN"
    result = {"model": {}, "adam": {}, "param_groups": []}
    owners = set()
    for parameter, name in names.items():
        local, layout = local_shard(parameter)
        result["model"][name] = {
            **layout,
            **tensor_info(local),
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
                assert parameter in names and parameter not in owners
                owners.add(parameter)
                state = optimizer.state[parameter]
                assert {"step", "exp_avg", "exp_avg_sq"} <= state.keys()
                entries = {}
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if key in ("exp_avg", "exp_avg_sq"):
                            assert value.shape == parameter.shape
                        local, layout = local_shard(value)
                        if key in ("exp_avg", "exp_avg_sq"):
                            assert layout == local_shard(parameter)[1]
                        entries[key] = {
                            **layout,
                            **tensor_info(local),
                        }
                    else:
                        entries[key] = {"value_sha256": state_fingerprint(value)}
                result["adam"][f"{index}:{names[parameter]}"] = entries
    assert owners == set(names), "Adam/model parameter coverage differs"
    return result


def verify_initial_slice(actual, expected, key: str) -> dict:
    assert actual.dtype == expected.dtype == torch.bfloat16, key
    assert actual.shape == expected.shape, key
    actual = actual.detach().cpu()
    assert torch.equal(actual, expected), f"Loaded initial shard differs: {key}"
    return tensor_info(actual)


def verify_initial_shards(model, adapter, shared: Path) -> dict:
    """Exact native dim0 local slices, at most one HF file plus shared head.

    Native FQNs/layouts intentionally differ from HF; no parameter gather or
    assumption that every rank owns a nonempty shared-gate shard is made.
    """
    index = json.loads((shared / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    head = shared_head_tensor(shared, weight_map)
    parameters = {
        canonical_native_fqn(name): parameter
        for name, parameter in model.named_parameters()
    }
    assert len(parameters) == len(list(model.named_parameters()))
    records = {}
    for filename in sorted(set(weight_map.values())):
        assert Path(filename).name == filename
        with safe_open(shared / filename, framework="pt", device="cpu") as tensors:
            source = {key: tensors.get_tensor(key) for key in tensors.keys()}
        assert all(weight_map[key] == filename for key in source)
        converted = from_hf_group(adapter, source, head)
        for name, expected in sorted(converted.items()):
            assert name in parameters and name not in records, name
            parameter = parameters[name]
            assert isinstance(parameter, DTensor), name
            actual, layout = local_shard(parameter)
            assert list(expected.shape) == layout["global_shape"], name
            start = layout["offset"]
            wanted = expected[start : start + actual.shape[0]]
            records[name] = {**layout, **verify_initial_slice(actual, wanted, name)}
            del actual, wanted
        del converted, source
    assert records.keys() == parameters.keys(), "Initial native FQN coverage differs"
    return records


def shared_contract(args) -> dict:
    index_path = args.shared_hf / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    provenance_path = args.shared_hf / "initialization_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    assert index["metadata"]["initialization_provenance_sha256"] == file_hash(
        provenance_path
    )
    assert provenance["config"]["original_sha256"] == file_hash(
        args.model / "config.json"
    )
    assert provenance["config"]["text_config_sha256"] == file_hash(
        args.shared_hf / "config.json"
    )
    assert len(index["weight_map"]) == 693
    assert index["metadata"]["total_size"] == PARAMETERS * 2
    assert set(index["weight_map"].values()) == set(provenance["shards"])
    for name, metadata in provenance["shards"].items():
        assert Path(name).name == name and name.endswith(".safetensors")
        assert file_hash(args.shared_hf / name) == metadata["sha256"]
    return {
        "shared_index_sha256": file_hash(index_path),
        "original_config_sha256": file_hash(args.model / "config.json"),
        "provenance_sha256": file_hash(provenance_path),
    }


def make_spec():
    spec = model_registry("35B-A3B")
    spec.parallelize_fn = parallelize_text
    # Native HF router has no auxiliary-free bias updates or auxiliary loss.
    for layer in spec.model.layers:
        layer.moe.load_balance_coeff = None
        if layer.attention is not None:
            layer.attention.inner_attention = ScaledDotProductAttention.Config()
    return spec


def text_inputs(module, args, kwargs):
    kwargs.setdefault("special_tokens", {"image_id": -1, "video_id": -1})
    return args, kwargs


def parallelize_text(model, **kwargs):
    model.vision_encoder = None  # still meta: never materialize unused vision
    model.register_forward_pre_hook(text_inputs, with_kwargs=True)
    return parallelize_qwen3_5_npu(model, **kwargs)


class ReplayTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    def __init__(self, config, *, tokenizer_path):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise RuntimeError("Replay must never tokenize")

    def decode(self, *args, **kwargs):
        raise RuntimeError("Replay must never decode")

    def get_vocab_size(self):
        return 248320


def replay_positions(batch, labels):
    """Preserve pad suffix; fixed GDN ignores it, RoPE+causal SDPA preserves prefix."""
    length = labels.numel()
    padding = batch.get("padding_mask")
    valid_length = length
    if padding is not None:
        assert padding.dtype == torch.bool and padding.shape in ((length,), (1, length))
        padding = padding.reshape(-1)
        assert not bool(padding[0]) and not bool((padding[:-1] & ~padding[1:]).any())
        assert bool((labels.reshape(-1)[padding] == -100).all())
        valid_length = int((~padding).sum())
    positions = batch.get("positions")
    if positions is None:
        positions = torch.arange(length).reshape(1, length)
    assert positions.device.type == "cpu" and positions.dtype in (
        torch.int32,
        torch.int64,
    )
    assert positions.shape in ((length,), (1, length)) and not bool(
        (positions < 0).any()
    )
    assert torch.equal(positions.reshape(-1)[:valid_length], torch.arange(valid_length))
    return positions.reshape(1, length)


class ReplayDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        replay_path: str

    def __init__(
        self,
        config,
        *,
        dp_world_size,
        dp_rank,
        tokenizer,
        seq_len,
        local_batch_size,
        snapshot_every_n_steps,
    ):
        assert (dp_world_size, seq_len, local_batch_size) == (8, 2048, 1)
        self.root = Path(config.replay_path) / f"rank-{dp_rank:05d}"
        self.rank = dp_rank
        self.cursor = 0
        self.records = []
        self.source_hash = file_hash(self.root / "run.json")

    def __iter__(self) -> Iterator:
        while self.cursor < 100:
            device = (
                torch.device("npu", torch.npu.current_device())
                if dist.is_initialized()
                else torch.device("cpu")
            )
            with synchronize_local_errors(device):
                step = self.cursor + 1
                path = self.root / "batches" / f"step-{step:05d}-batch-000.pt"
                saved = torch.load(path, map_location="cpu", weights_only=True)
                batch = saved["batch"]
                assert saved["sha256"] == batch_hash(batch), f"Corrupt replay: {path}"
                tokens, labels = batch["input"], batch["labels"]
                assert tokens.device.type == labels.device.type == "cpu"
                assert tokens.dtype == labels.dtype == torch.int64
                assert tokens.numel() == labels.numel() == 2048
                tokens, labels = tokens.reshape(1, 2048), labels.reshape(1, 2048)
                positions = replay_positions(batch, labels)
                position_evidence = {
                    "transformed": False,
                    "original": tensor_info(batch["positions"])
                    if batch.get("positions") is not None
                    else None,
                    "effective": tensor_info(positions),
                    "provided": batch.get("positions") is not None,
                }
                assert int(tokens.min()) >= 0 and int(tokens.max()) < 248320
                assert bool(
                    ((labels == -100) | ((labels >= 0) & (labels < 248320))).all()
                )
                valid = int((labels != -100).sum())
                assert valid > 0
                self.records.append(
                    {
                        "step": step,
                        "sha256": saved["sha256"],
                        "valid_targets": valid,
                        "position_metadata": position_evidence,
                    }
                )
                self.cursor += 1  # persisted cursor is NEXT batch, not last yield
            yield {"input": tokens, "positions": positions}, labels

    def state_dict(self):
        return {f"rank_{self.rank}": pickle.dumps((self.cursor, self.source_hash))}

    def load_state_dict(self, state):
        cursor, source_hash = pickle.loads(state[f"rank_{self.rank}"])
        assert source_hash == self.source_hash and 0 <= cursor <= 100
        self.cursor = cursor


class RankRNGState:
    """Rank-specific bytes prevent DCP replica deduplication of distinct RNGs.

    torch2.12 DTensor's tracker uses the device generator directly, rather than
    the older _rng_states dictionary. Explicit generators would need their own
    registration; this replay owns no random generator.
    """

    def __init__(self, rank, *, npu=True):
        self.key, self.npu = f"rank_{rank}", npu
        self.restored = False

    def state_dict(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
        }
        if self.npu:
            state["npu"] = torch.npu.get_rng_state().cpu()
        return {self.key: pickle.dumps(state)}

    def load_state_dict(self, state):
        saved = pickle.loads(state[self.key])
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
        torch.set_rng_state(saved["cpu"])
        if self.npu:
            torch.npu.set_rng_state(saved["npu"])
        self.restored = True


def rng_fingerprint(rng):
    state = pickle.loads(rng.state_dict()[rng.key])
    return batch_hash(
        {
            "python": repr(state["python"]),
            "numpy": repr(state["numpy"]),
            "cpu": state["cpu"],
            **({"npu": state["npu"]} if "npu" in state else {}),
        }
    )


def make_config(args):
    cfg = TrainerEx.Config(
        model_spec=make_spec(),
        hf_assets_path=str(args.model),
        dump_folder=str(args.output),
        tokenizer=ReplayTokenizer.Config(),
        dataloader=ReplayDataLoader.Config(replay_path=str(args.replay)),
        optimizer=OptimizerConfig(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": 3e-4,
                        "betas": (0.9, 0.999),
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                    },
                )
            ],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            decay_ratio=0.0,
            min_lr_factor=1.0,
        ),
        training=TrainingConfig(
            steps=100,
            seq_len=2048,
            local_batch_size=1,
            global_batch_size=8,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            max_norm=1.0,
            disable_cuda_graphs=True,
            enable_cpu_offload=False,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
            spmd_backend="partial_dtensor",
            fsdp_reshard_after_forward="always",
        ),
        activation_checkpoint=FullAC.Config(preserve_rng_state=True),
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=50,
            initial_load_path=str(args.resume or args.native),
            initial_load_model_only=args.resume is None,
            initial_load_in_hf=False,
            last_save_model_only=False,
            last_save_in_hf=False,
            async_mode="disabled",
            keep_latest_k=0,
        ),
        loss=CrossEntropyLoss.Config(global_vocab_size=248320),
        debug=DebugConfig(seed=42, deterministic=True),
    )
    cfg.metrics.log_freq = 1
    cfg.validator.enable = False
    cfg.override.imports = ["torchtitan_npu.override.qwen3_5.gated_delta.npu"]
    return cfg


def prepare(args):
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1
    contract = shared_contract(args)
    assert not args.native.exists(), "Refusing stale/partial native directory"
    spec = make_spec()
    with torch.device("meta"):
        model = spec.model.build()
    model.vision_encoder = None
    expected = model.state_dict()
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    adapter = Qwen35StateDictAdapter(spec.model, None)
    index = json.loads((args.shared_hf / "model.safetensors.index.json").read_text())
    head = shared_head_tensor(args.shared_hf, index["weight_map"])
    native = {}
    for filename in sorted(set(index["weight_map"].values())):
        with safe_open(
            args.shared_hf / filename, framework="pt", device="cpu"
        ) as shard:
            source = {key: shard.get_tensor(key) for key in shard.keys()}
        assert all(index["weight_map"][key] == filename for key in source)
        converted = from_hf_group(adapter, source, head)
        assert not (native.keys() & converted.keys())
        for key, value in converted.items():
            assert key in expected and value.shape == expected[key].shape, key
            assert value.dtype == torch.bfloat16
            tensor_info(value)
        # One shared shard contains complete layer/outer projection groups.
        restored = boundary(adapter.to_hf(converted), to_reference=False)
        assert restored.keys() == source.keys()
        for key in source:
            assert torch.equal(restored[key], source[key]), key
        native.update(converted)
        del source, converted, restored
    assert native.keys() == expected.keys()
    assert shared_contract(args) == contract, "Shared source changed during conversion"
    dcp.save(native, checkpoint_id=args.native, no_dist=True)
    metadata = dcp.FileSystemReader(args.native).read_metadata().state_dict_metadata
    assert metadata.keys() == native.keys(), "DCP must contain flat native model keys"
    for key, value in native.items():
        assert metadata[key].size == value.shape
        assert metadata[key].properties.dtype == value.dtype
    write_json(
        args.native / "reference_initial.json",
        {
            **contract,
            "format": "flat model-only native DCP",
            "parameters": PARAMETERS,
            "tensor_count": len(native),
            "dcp_files": {
                p.name: file_hash(p)
                for p in sorted(args.native.iterdir())
                if p.is_file()
            },
            "entry_sha256": file_hash(Path(__file__)),
        },
    )
    print("CPU prepare complete; no training performed")


class ReferenceTrainer(TrainerEx):
    """Observe real Trainer hooks; do not replace its loss/step/optimizer loop."""

    def __init__(self, config, options, contract):
        self.options, self.contract = options, contract
        super().__init__(config)
        with synchronize_local_errors(self.device):
            rank = dist.get_rank()
            self.rank_dir = options.artifacts / f"rank-{rank:05d}"
            self.rank_dir.mkdir(parents=True, exist_ok=False)
            assert len(self.model_parts) == self.gradient_accumulation_steps == 1
            assert (
                sum(p.numel() for p in self.model_parts[0].parameters()) == PARAMETERS
            )
            assert all(
                p.dtype == torch.bfloat16 for p in self.model_parts[0].parameters()
            )
            self.adapter = Qwen35StateDictAdapter(self.model_config, None)
            self.rng = RankRNGState(rank)
            self.checkpointer.states["rng_state"] = self.rng
            assert self.checkpointer.states["dataloader"] is self.dataloader
            write_json(
                self.rank_dir / "run.json",
                {
                    "contract": contract,
                    "output": str(options.output),
                    "resume": str(options.resume) if options.resume else None,
                    "status": "started_not_accepted",
                    "snapshots": SNAPSHOTS,
                    "observation": "layerwise gather then CPU HF dim0 shards",
                },
            )
        original_load = self.checkpointer.load

        def checked_load(*args, **kwargs):
            loaded = original_load(*args, **kwargs)
            with synchronize_local_errors(self.device):
                expected = 50 if options.resume else 0
                assert loaded and self.step == self.dataloader.cursor == expected
                assert self.lr_schedulers.state_dict()["last_epoch"] == expected
                assert self.rng.restored == bool(options.resume)
                self.validate_optimizer(expected)
                if options.resume:
                    saved = json.loads(
                        (
                            options.verify_from
                            / self.rank_dir.name
                            / "step-00050-checkpoint-state.json"
                        ).read_text()
                    )
                    checkpoint = options.resume / ".metadata"
                    assert saved["step"] == expected
                    assert saved["checkpoint"] == str(options.resume.resolve())
                    assert saved["metadata_sha256"] == file_hash(checkpoint)
                    restored = self.checkpoint_evidence()
                    assert all(
                        saved[key] == value for key, value in restored.items()
                    ), "Restored native model/Adam/RNG/loader/scheduler state differs"
                    write_json(self.rank_dir / "restored-state.json", restored)
                else:
                    assert all(not optimizer.state for optimizer in self.optimizers)
                    initial = verify_initial_shards(
                        self.model_parts[0], self.adapter, options.shared_hf
                    )
                    write_json(self.rank_dir / "initial-shards-exact.json", initial)
                write_json(
                    self.rank_dir / "loaded.json",
                    {
                        "step": self.step,
                        "cursor": self.dataloader.cursor,
                        "rng_restored": self.rng.restored,
                        "rng_sha256": rng_fingerprint(self.rng),
                        "initial_shards_exact": not bool(options.resume),
                        "model_adam_restored_exact": bool(options.resume),
                    },
                )
            return loaded

        self.checkpointer.load = checked_load
        original_save = self.checkpointer.save

        def checked_save(curr_step, last_step=False):
            certify = curr_step in (50, 100)
            with synchronize_local_errors(self.device):
                if certify:
                    assert self.checkpointer.async_mode.value == "disabled"
                    saved = self.checkpoint_evidence()
            result = original_save(curr_step, last_step)
            with synchronize_local_errors(self.device):
                if certify:
                    assert result, "Requested full checkpoint not saved"
                    assert saved == self.checkpoint_evidence(), (
                        "Native model/Adam/state changed across checkpoint save"
                    )
                    checkpoint = options.output / "checkpoint" / f"step-{curr_step}"
                    metadata = checkpoint / ".metadata"
                    assert metadata.is_file(), "Synchronous checkpoint not published"
                    write_json(
                        self.rank_dir / f"step-{curr_step:05d}-checkpoint-state.json",
                        {
                            "schema": "qwen35-native-checkpoint-evidence-v1",
                            "step": curr_step,
                            "checkpoint": str(checkpoint.resolve()),
                            "metadata_sha256": file_hash(metadata),
                            **saved,
                        },
                    )
            return result

        self.checkpointer.save = checked_save

    def checkpoint_evidence(self):
        """Local-only streaming hashes; no full_tensor or model-sized copies.

        Candidate-compatible model/adam/param_groups structure, with native
        reference FQNs. Cross-framework hashes are not expected to match;
        this certifies each framework's own exact checkpoint/resume boundary.
        """
        return {
            "cursor": self.dataloader.cursor,
            "rng_sha256": rng_fingerprint(self.rng),
            "model_adam": model_adam_manifest(self.model_parts[0], self.optimizers),
            "state_fingerprints": {
                key: checkpoint_state_fingerprint(
                    key, self.checkpointer.states[key].state_dict()
                )
                for key in ("lr_scheduler", "dataloader", "train_state", "rng_state")
            },
        }

    def validate_optimizer(self, expected):
        for optimizer in self.optimizers:
            assert isinstance(optimizer, torch.optim.AdamW)
            for group in optimizer.param_groups:
                assert group["lr"] == 3e-4 and tuple(group["betas"]) == (0.9, 0.999)
                assert group["eps"] == 1e-8 and group["weight_decay"] == 0
                assert not group["foreach"] and not group["fused"]
                if expected:
                    for parameter in group["params"]:
                        state = optimizer.state[parameter]
                        assert int(state["step"]) == expected
                        for name in ("exp_avg", "exp_avg_sq"):
                            value = state[name]
                            tensor_info(
                                value.to_local()
                                if isinstance(value, DTensor)
                                else value
                            )

    def dump_hf(self, kind):
        with synchronize_local_errors(self.device):
            folder = self.rank_dir / f"step-{self.step:05d}-{kind}"
            folder.mkdir(exist_ok=False)
            groups = defaultdict(dict)
            # state_dict supplies canonical names even under AC/FSDP wrappers.
            parameters = {id(p): p for p in self.model_parts[0].parameters()}
            for name, parameter in self.model_parts[0].named_parameters():
                canonical = name.replace("._checkpoint_wrapped_module", "")
                match = re.match(r"layers\.(\d+)\.", canonical)
                groups[match[1] if match else "outer"][canonical] = parameters[
                    id(parameter)
                ]
            index = {}
            rank = dist.get_rank()
            weight_map = json.loads(
                (self.options.shared_hf / "model.safetensors.index.json").read_text()
            )["weight_map"]
            head = shared_head_tensor(self.options.shared_hf, weight_map)
        for group in sorted(groups):
            with synchronize_local_errors(self.device):
                sources = {}
                for name, parameter in sorted(groups[group].items()):
                    value = parameter.grad if kind == "gradients-preclip" else parameter
                    assert value is not None, f"Missing {kind}: {name}"
                    assert value.shape == parameter.shape, name
                    sources[name] = value.detach()
            native = {}
            for name, value in sources.items():
                if isinstance(value, DTensor):
                    value = value.full_tensor()  # identical ordered collectives
                with synchronize_local_errors(self.device):
                    native[name] = value.cpu().contiguous().clone()
            with synchronize_local_errors(self.device):
                hf = boundary(self.adapter.to_hf(native), to_reference=False)
                assert hf, group
                back = from_hf_group(self.adapter, hf, head)
                assert back.keys() == native.keys(), "Adapter dropped native parameters"
                assert all(torch.equal(back[k], v) for k, v in native.items())
                del back
                shards, layouts = {}, {}
                for name, value in sorted(hf.items()):
                    size = (value.shape[0] + 7) // 8
                    offset = min(rank * size, value.shape[0])
                    local = value[offset : offset + size].contiguous().clone()
                    tensor_info(local)
                    shards[name] = local
                    layouts[name] = {
                        "global_shape": list(value.shape),
                        "shard_dim": 0,
                        "offset": offset,
                        **tensor_info(local),
                    }
                path = folder / f"{group}.pt"
                save_tensor_file(path, shards)
                if kind == "weights-after-update":
                    before = torch.load(
                        self.rank_dir
                        / f"step-{self.step:05d}-weights-before-update"
                        / path.name,
                        map_location="cpu",
                        weights_only=True,
                    )
                    update = {
                        key: shards[key].float() - before[key].float() for key in shards
                    }
                    for value in update.values():
                        tensor_info(value)
                    save_tensor_file(folder / f"{group}-delta.pt", update)
                    index[f"{group}-delta.pt"] = {
                        "sha256": file_hash(folder / f"{group}-delta.pt")
                    }
                index[path.name] = {"sha256": file_hash(path), "tensors": layouts}
                if self.options.verify_from:
                    baseline = (
                        self.options.verify_from / self.rank_dir.name / folder.name
                    )
                    for filename in (path.name,):
                        reference = torch.load(
                            baseline / filename, map_location="cpu", weights_only=True
                        )
                        assert reference.keys() == shards.keys()
                        assert all(
                            torch.equal(reference[k], v) for k, v in shards.items()
                        )
                    if kind == "weights-after-update":
                        reference = torch.load(
                            baseline / f"{group}-delta.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                        assert reference.keys() == update.keys()
                        assert all(
                            torch.equal(reference[k], v) for k, v in update.items()
                        )
                del native, hf, shards
        with synchronize_local_errors(self.device):
            write_json(
                folder / "index.json", {"format": "HF5.13 native text", "files": index}
            )

    def forward_backward_step(self, **kwargs):
        denominator = int(kwargs["global_valid_tokens"].detach().cpu())
        self.denominators.append(denominator)
        loss = super().forward_backward_step(**kwargs)
        self.losses.append(float(loss.detach().cpu()))
        return loss

    def train_step(self, data_iterator):
        self.losses, self.denominators = [], []
        self.dataloader.records = []
        logits = []
        norms = []
        original_clip = dist_utils.clip_grad_norm_

        def capture(module, inputs, output):
            with synchronize_local_errors(self.device):
                assert not logits and output.shape[-1] == 248320
                cpu = output.detach().cpu().contiguous()
                info = tensor_info(cpu)
                if self.step in SNAPSHOTS:
                    path = self.rank_dir / f"step-{self.step:05d}-logits-000.pt"
                    save_tensor_file(path, cpu)
                    info["file"] = path.name
                logits.append(info)

        def clip(*args, **kwargs):
            assert not norms, "Multiple gradient clipping calls"
            if self.step in SNAPSHOTS:
                self.dump_hf("gradients-preclip")
                self.dump_hf("weights-before-update")
            norm = original_clip(*args, **kwargs)
            norms.append(float(norm.detach().cpu()))
            return norm

        applied_lr = self.lr_schedulers.get_metrics()
        handle = self.model_parts[0].register_forward_hook(capture)
        dist_utils.clip_grad_norm_ = clip
        try:
            result = super().train_step(data_iterator)
        finally:
            dist_utils.clip_grad_norm_ = original_clip
            handle.remove()
        with synchronize_local_errors(self.device):
            assert (
                len(logits)
                == len(norms)
                == len(self.losses)
                == len(self.denominators)
                == 1
            )
            assert len(self.dataloader.records) == 1
            valid = torch.tensor(
                self.dataloader.records[0]["valid_targets"], device=self.device
            )
        dist.all_reduce(valid, op=dist.ReduceOp.SUM)
        with synchronize_local_errors(self.device):
            assert int(valid) == self.denominators[0] > 0
            total_loss = torch.tensor(
                self.losses[0], dtype=torch.float32, device=self.device
            )
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)  # ALREADY normalized; no /8
        with synchronize_local_errors(self.device):
            assert all(value == 3e-4 for value in applied_lr.values())
            assert math.isfinite(float(total_loss)) and math.isfinite(norms[0])
            assert self.lr_schedulers.state_dict()["last_epoch"] == self.step
        if self.step in SNAPSHOTS:
            self.dump_hf("weights-after-update")
        with synchronize_local_errors(self.device):
            record = {
                "step": self.step,
                "global_loss": float(total_loss),
                "local_normalized_loss": self.losses[0],
                "grad_norm_preclip": norms[0],
                "global_valid_targets": self.denominators[0],
                "applied_lr": applied_lr,
                "batches": self.dataloader.records,
                "logits": logits,
                "loader_cursor": self.dataloader.cursor,
            }
            if self.options.verify_from:
                baseline = json.loads(
                    (
                        self.options.verify_from
                        / self.rank_dir.name
                        / f"step-{self.step:05d}.json"
                    ).read_text()
                )
                assert record == baseline, "Reference independent resume differs"
            write_json(self.rank_dir / f"step-{self.step:05d}.json", record)
        return result


def train(args):
    assert int(os.environ.get("WORLD_SIZE", "1")) == 8
    assert not args.output.exists() and not args.artifacts.exists(), (
        "Require fresh directories"
    )
    contract = shared_contract(args)
    prepared = json.loads((args.native / "reference_initial.json").read_text())
    assert all(prepared[key] == value for key, value in contract.items())
    assert (args.native / ".metadata").is_file()
    assert prepared["entry_sha256"] == file_hash(Path(__file__))
    for name, digest in prepared["dcp_files"].items():
        assert Path(name).name == name
        assert file_hash(args.native / name) == digest
    metadata = dcp.FileSystemReader(args.native).read_metadata().state_dict_metadata
    assert len(metadata) == prepared["tensor_count"]
    assert sum(math.prod(value.size) for value in metadata.values()) == PARAMETERS
    assert all(value.properties.dtype == torch.bfloat16 for value in metadata.values())
    candidate_contract = None
    for rank in range(8):
        root = args.replay / f"rank-{rank:05d}"
        run = json.loads((root / "run.json").read_text())["contract"]
        assert run["shared_index_sha256"] == contract["shared_index_sha256"]
        assert (
            run["global_batch_size_per_accumulation"] == 8 and run["accumulation"] == 1
        )
        assert run["sequence_length"] == 2048 and run["horizon"] == 100
        assert run["world_size"] == 8 and run["geometry"] == "35B-A3B full 40 layers"
        assert run["max_num_documents"] == 1 and run["activation_checkpoint"] == "full"
        assert run["max_norm"] == 1.0 and run["compile"] is False
        assert run["deterministic"] is True
        for name, digest in run["tokenizer_hashes"].items():
            assert Path(name).name == name and file_hash(args.model / name) == digest
        assert run["optimizer"] == {
            "name": "AdamW",
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "implementation": "for-loop",
        }
        assert candidate_contract is None or candidate_contract == run
        candidate_contract = run
        assert (root / "complete.json").is_file(), "Candidate continuous run incomplete"
        for step in range(1, 101):
            assert (root / "batches" / f"step-{step:05d}-batch-000.pt").is_file()
    contract.update(
        candidate_contract=candidate_contract,
        entry_sha256=file_hash(Path(__file__)),
        trainer_sha256=file_hash(Path(TrainerEx.__init__.__code__.co_filename)),
        reference_sha256=file_hash(Path(torchtitan.__file__).parent / "trainer.py"),
    )
    if args.resume:
        assert args.resume.name == "step-50" and (args.resume / ".metadata").is_file()
        for rank in range(8):
            original = json.loads(
                (args.verify_from / f"rank-{rank:05d}" / "run.json").read_text()
            )
            assert original["contract"] == contract
            assert args.resume == Path(original["output"]) / "checkpoint" / "step-50"
    trainer = None
    try:
        trainer = ReferenceTrainer(make_config(args), args, contract)
        trainer.train()
        assert trainer.step == trainer.dataloader.cursor == 100
        for step in (100,) if args.resume else (50, 100):
            assert (args.output / "checkpoint" / f"step-{step}" / ".metadata").is_file()
        write_json(
            trainer.rank_dir / "complete.json",
            {
                "status": "reference_complete_comparison_pending",
                "step": 100,
                "resumed_from": 50 if args.resume else 0,
                "independent_resume_exact_scalars_logits_gradients_updates": bool(
                    args.resume
                ),
            },
        )
    finally:
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def cpu_check():
    """CPU API/meta/adapter/loader/state checks; never instantiate TrainerEx."""
    assert importlib.metadata.version("torchtitan") == "0.3.0"
    assert (
        "early_stop" in inspect.signature(torch.utils.checkpoint.checkpoint).parameters
    )
    spec = make_spec()
    with torch.device("meta"):
        model = spec.model.build()
    model.vision_encoder = None
    assert len(model.layers) == 40
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    native_meta = model.state_dict()
    adapter_meta = Qwen35StateDictAdapter(spec.model, None)
    hf_meta = boundary(adapter_meta.to_hf(native_meta), to_reference=False)
    back_meta = from_hf_group(adapter_meta, hf_meta)
    assert back_meta.keys() == native_meta.keys()
    assert all(back_meta[k].shape == v.shape for k, v in native_meta.items())
    assert len(hf_meta) == 693
    # Scalar shared gates have legitimate dim0 empty owners; never require
    # every rank's local shard to be nonzero/nonempty.
    gate = torch.ones(1, 2048, dtype=torch.bfloat16)
    for rank_id in range(8):
        shard = gate[min(rank_id, 1) : min(rank_id, 1) + 1]
        assert tensor_info(shard)["empty_local_shard"] == (rank_id != 0)
    FullAC.Config().build().apply(model)
    adapter = Qwen35StateDictAdapter(spec.model, None)
    witness = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.arange(60).reshape(2, 10, 3),
        "model.layers.0.mlp.experts.down_proj": torch.arange(30).reshape(2, 3, 5),
    }
    restored = boundary(
        adapter.to_hf(
            from_hf_group(adapter, witness, torch.ones(4, 3, dtype=torch.bfloat16))
        ),
        to_reference=False,
    )
    assert all(torch.equal(restored[key], value) for key, value in witness.items())
    rng = RankRNGState(0, npu=False)
    saved = rng.state_dict()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    rng.load_state_dict(saved)
    assert random.random() == expected[0] and float(np.random.rand()) == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    with tempfile.TemporaryDirectory(prefix="qwen35-reference-cpu-") as directory:
        root = Path(directory)
        rank = root / "rank-00000"
        (rank / "batches").mkdir(parents=True)
        write_json(rank / "run.json", {"synthetic": True})
        batch = {
            "input": torch.arange(2048).reshape(1, -1),
            "labels": torch.arange(2048).reshape(1, -1),
        }
        save_tensor_file(
            rank / "batches/step-00001-batch-000.pt",
            {"batch": batch, "sha256": batch_hash(batch)},
        )
        loader = ReplayDataLoader.Config(replay_path=str(root)).build(
            dp_world_size=8,
            dp_rank=0,
            tokenizer=None,
            seq_len=2048,
            local_batch_size=1,
            snapshot_every_n_steps=50,
        )
        _, labels = next(iter(loader))
        assert torch.equal(labels, batch["labels"])
        state = loader.state_dict()
        loader.cursor = 0
        loader.load_state_dict(state)
        assert loader.cursor == 1
        config = make_config(
            argparse.Namespace(
                model=root,
                output=root / "out",
                replay=root,
                native=root / "native",
                resume=None,
            )
        )
        assert config.training.dtype == "bfloat16"
        tiny = torch.nn.Linear(3, 2)
        optim = config.optimizer.build(model_parts=[tiny])
        scheduler = config.lr_scheduler.build(optimizers=optim, training_steps=100)
        for _ in range(100):
            optim.zero_grad()
            tiny(torch.ones(1, 3)).sum().backward()
            optim.step()
            scheduler.step()
            assert all(lr == 3e-4 for lr in scheduler.get_metrics().values())
        baseline = model_adam_manifest(tiny, optim)
        scheduler_before = state_fingerprint(scheduler.state_dict())
        payload = {
            "model": tiny,
            "optimizer": optim,
            "lr_scheduler": scheduler,
            "rng": rng,
            "loader": loader,
        }
        dcp.save(payload, checkpoint_id=root / "dcp", no_dist=True)
        metadata = dcp.FileSystemReader(root / "dcp").read_metadata()
        assert metadata.state_dict_metadata["model.weight"].size == tiny.weight.shape
        loader.cursor = 0
        with torch.no_grad():
            for parameter in tiny.parameters():
                parameter.add_(1)
            for optimizer in optim:
                for state in optimizer.state.values():
                    state["exp_avg"].add_(1)
                    state["exp_avg_sq"].add_(1)
        assert model_adam_manifest(tiny, optim) != baseline
        dcp.load(payload, checkpoint_id=root / "dcp", no_dist=True)
        assert loader.cursor == 1 and rng.restored
        assert model_adam_manifest(tiny, optim) == baseline
        assert state_fingerprint(scheduler.state_dict()) == scheduler_before
    print(
        json.dumps(
            {
                "status": "CPU_API_ONLY_NOT_TRAINING_ACCEPTANCE",
                "torch": torch.__version__,
                "torchtitan": torchtitan.__file__,
                "meta_parameters": PARAMETERS,
                "checks": [
                    "fullAC-meta",
                    "all-693-HF-layout-shapes",
                    "dynamic-empty-shard-ownership",
                    "HF5.13-adapter-roundtrip",
                    "RNG",
                    "replay-cursor",
                    "AdamW-constant-LR",
                    "flat-DCP-Stateful-CPU",
                    "model-Adam-scheduler-DCP-exact-restore",
                ],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("cpu-check", "prepare", "train"), required=True
    )
    for name in (
        "model",
        "shared-hf",
        "native",
        "replay",
        "output",
        "artifacts",
        "resume",
        "verify-from",
    ):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.phase == "cpu-check":
        cpu_check()
        return
    required = ["model", "shared_hf", "native"]
    if args.phase == "train":
        required += ["replay", "output", "artifacts"]
        assert bool(args.resume) == bool(args.verify_from)
    assert all(getattr(args, name) is not None for name in required), required
    sources = [args.model, args.shared_hf]
    if args.phase == "train":
        sources += [args.native, args.replay, args.verify_from, args.resume]
    outputs = [args.output, args.artifacts] if args.phase == "train" else [args.native]
    for output in outputs:
        for source in sources:
            if source is not None:
                assert source != output
                assert source not in output.parents and output not in source.parents
        for other in outputs:
            if other is not output:
                assert other != output
                assert other not in output.parents and output not in other.parents
    (prepare if args.phase == "prepare" else train)(args)


if __name__ == "__main__":
    main()
