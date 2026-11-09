"""CPU-only verifier contracts, without importing the NPU training entry."""

import ast
import hashlib
import json
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.components.checkpointer.core.random_state import RNGCheckpointState
from llmtuner.utils.module_names import canonical_fqn

ENTRY = (
    Path(__file__).resolve().parents[4]
    / "tests/integration_tests/qwen35_candidate_alignment.py"
)


def helpers():
    # Compile only the production verifier definitions: the integration entry
    # imports Trainer/recipe environments, which are not needed for these CPU
    # contracts. There are no replacements for tensor math, FSDP, or IO.
    tree = ast.parse(ENTRY.read_text())
    names = {
        "initial_hf_key",
        "tensor_info",
        "verify_initial_slice",
        "verify_runtime_precision",
        "state_fingerprint",
        "checkpoint_state_fingerprint",
        "model_adam_manifest",
        "observe_checkpoint_payloads",
        "local_shard",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == len(names)
    namespace = {
        "torch": torch,
        "hashlib": hashlib,
        "canonical_fqn": canonical_fqn,
        "FSDPModule": FSDPModule,
        "np": np,
        "json": json,
        "pickle": pickle,
        "Any": Any,
        "contextmanager": contextmanager,
        "DATALOADER": "dataloader",
        "DTensor": DTensor,
        "Replicate": Replicate,
        "Shard": Shard,
        "sys": sys,
        "synchronize_local_errors": synchronize_local_errors,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(ENTRY), "exec"), namespace
    )
    return namespace


def test_canonical_initial_key():
    convert = helpers()["initial_hf_key"]
    assert convert("model.model.layers.0._checkpoint_wrapped_module.weight") == (
        "model.layers.0.weight"
    )
    assert convert("model._orig_mod.lm_head.weight") == "lm_head.weight"
    with pytest.raises(AssertionError):
        convert("unexpected.weight")


def test_safetensors_exact_and_empty_dim_zero_slices(tmp_path):
    path = tmp_path / "tiny.safetensors"
    weight = torch.tensor([[0.125, -0.375]], dtype=torch.bfloat16)
    save_file({"gate": weight}, path)
    verify = helpers()["verify_initial_slice"]
    with safe_open(path, framework="pt", device="cpu") as tensors:
        view = tensors.get_slice("gate")
        for start in (0, 1):
            expected = view[start : start + 1] if start == 0 else view[start:start]
            actual = weight[start : start + 1].clone()
            result = verify(actual, expected, "gate")
            assert result["empty_local_shard"] == (start == 1)
        with pytest.raises(AssertionError, match="differs"):
            verify(weight + 1, view[:], "gate")
        with pytest.raises(AssertionError):
            verify(weight.float(), view[:], "gate")
        with pytest.raises(AssertionError, match="differs"):
            verify(weight[:, :1], view[:], "gate")


@pytest.mark.parametrize("wrong", [None, "cast", "reduce", "divide"])
def test_real_cpu_fsdp_runtime_precision(wrong):
    verify = helpers()["verify_runtime_precision"]
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        model = torch.nn.Linear(4, 4, bias=False).bfloat16()
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16 if wrong == "reduce" else torch.float32,
            cast_forward_inputs=wrong == "cast",
        )
        fully_shard(
            model,
            mesh=init_device_mesh("cpu", (dist.get_world_size(),)),
            mp_policy=policy,
        )
        model.set_gradient_divide_factor(2.0 if wrong == "divide" else 1.0)
        if wrong is None:
            result = verify(model)
            assert result["fsdp_states"] == 1
            assert result["gradient_divide_factor"] == 1.0
        else:
            with pytest.raises(AssertionError):
                verify(model)
    finally:
        if created:
            dist.destroy_process_group()


def test_adamw_bf16_moments_follow_parameter_storage():
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([parameter], foreach=False, fused=False)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == parameter.dtype
    assert int(state["step"]) == 1


def test_actual_adam_serialization_manifest_detects_moment_corruption(tmp_path):
    h = helpers()
    model = torch.nn.Linear(4, 3, bias=False).bfloat16()
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
    model(torch.ones(2, 4, dtype=torch.bfloat16)).float().square().sum().backward()
    optimizer.step()
    baseline = h["model_adam_manifest"](model, [optimizer])
    path = tmp_path / "step50.pt"
    torch.save({"model": model.state_dict(), "adam": optimizer.state_dict()}, path)
    restored = torch.nn.Linear(4, 3, bias=False).bfloat16()
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), foreach=False, fused=False
    )
    payload = torch.load(path, weights_only=True)
    restored.load_state_dict(payload["model"])
    restored_optimizer.load_state_dict(payload["adam"])
    assert h["model_adam_manifest"](restored, [restored_optimizer]) == baseline
    parameter = next(restored.parameters())
    restored_optimizer.state[parameter]["exp_avg"].add_(1)
    assert h["model_adam_manifest"](restored, [restored_optimizer]) != baseline


def test_observer_captures_actual_save_rng_boundary_and_restores_methods(tmp_path):
    h = helpers()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    rng = RNGCheckpointState(torch.device("cpu"))

    # Counter payloads stand in only for loader/train metadata; randomness is
    # the actual framework RNGCheckpointState and its real serialization.
    class Counter:
        def __init__(self, value):
            self.value = value

        def state_dict(self):
            return {"cursor": self.value}

    loader, train = Counter(50), Counter(50)
    states = {
        "lr_scheduler": scheduler,
        "dataloader": loader,
        "train_state": train,
        "rng_state": rng,
    }
    observed = {}
    original = rng.state_dict.__func__
    # Simulate later RNG-consuming work BEFORE the real state callbacks. The
    # observer must capture that boundary, rather than a pre-save approximation.
    pre_save_rng = h["checkpoint_state_fingerprint"]("rng_state", rng.state_dict())
    with h["observe_checkpoint_payloads"](states, observed):
        torch.rand(7)
        payload = {key: state.state_dict() for key, state in states.items()}
        for key, value in payload.items():
            assert observed[key] == h["checkpoint_state_fingerprint"](key, value)
        torch.save(payload, tmp_path / "metadata.pt")
    assert rng.state_dict.__func__ is original
    assert "state_dict" not in vars(rng)
    assert pre_save_rng != observed["rng_state"]
    torch.rand(9)
    saved = torch.load(tmp_path / "metadata.pt", weights_only=False)
    rng.load_state_dict(saved["rng_state"])
    assert (
        h["checkpoint_state_fingerprint"]("rng_state", rng.state_dict())
        == observed["rng_state"]
    )
    assert (
        h["checkpoint_state_fingerprint"]("rng_state", rng.state_dict())
        == observed["rng_state"]
    )
    with pytest.raises(RuntimeError, match="save failed"):
        with h["observe_checkpoint_payloads"](states, {}):
            raise RuntimeError("save failed")
    assert "state_dict" not in vars(loader) and "state_dict" not in vars(rng)


def observer_failure_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2
    )
    try:
        h = helpers()

        class State:
            def __init__(self, reject=False):
                self.reject = reject

            def state_dict(self):
                return {"cursor": 50}

            def __setattr__(self, name, value):
                if name == "state_dict" and self.reject:
                    raise ValueError("rank-local setup rejection")
                object.__setattr__(self, name, value)

        states = {
            key: State(key == "dataloader" and rank == 1)
            for key in ("lr_scheduler", "dataloader", "train_state", "rng_state")
        }
        with pytest.raises((ValueError, RuntimeError)):
            with h["observe_checkpoint_payloads"](states, {}):
                raise AssertionError(
                    "Must not enter save after rank-local setup failure"
                )
        assert all("state_dict" not in vars(state) for state in states.values())
        # A second collective succeeds only when both ranks exited observer
        # setup/cleanup with the same collective order.
        marker = torch.tensor(1)
        dist.all_reduce(marker)
        assert int(marker) == 2
    finally:
        dist.destroy_process_group()


def test_observer_setup_failure_is_coordinated_on_two_cpu_ranks(tmp_path):
    torch.multiprocessing.spawn(
        observer_failure_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2
    )


def test_cleanup_attempts_all_states_and_preserves_primary_error():
    h = helpers()

    class State:
        fail_cleanup = False

        def state_dict(self):
            return {"cursor": 50}

        def __delattr__(self, name):
            if name == "state_dict" and self.fail_cleanup:
                raise OSError("cleanup rejection")
            object.__delattr__(self, name)

    states = {
        key: State()
        for key in ("lr_scheduler", "dataloader", "train_state", "rng_state")
    }
    try:
        with pytest.raises(RuntimeError, match="primary save error"):
            with h["observe_checkpoint_payloads"](states, {}):
                states["rng_state"].fail_cleanup = True
                raise RuntimeError("primary save error")
        for key in ("lr_scheduler", "dataloader", "train_state"):
            assert "state_dict" not in vars(states[key])
    finally:
        states["rng_state"].fail_cleanup = False
        if "state_dict" in vars(states["rng_state"]):
            del states["rng_state"].state_dict
