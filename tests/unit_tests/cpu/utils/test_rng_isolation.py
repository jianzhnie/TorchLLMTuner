"""RNG isolation preserves primary failures and attempts remaining host restores."""

import pickle
import random

import numpy as np
import pytest
import torch

from llmtuner.utils.seeding import preserve_rng_state


@pytest.mark.parametrize("source", ["torch", "python", "numpy"])
@pytest.mark.parametrize("body_fails", [False, True])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
def test_rng_restore_attempts_all_sources_and_preserves_first_error(
    monkeypatch, source, body_fails, error_type
):
    states = {
        "torch": torch.get_rng_state().clone(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    setters = {
        "torch": (torch, "set_rng_state"),
        "python": (random, "setstate"),
        "numpy": (np.random, "set_state"),
    }
    originals = {
        name: getattr(module, method) for name, (module, method) in setters.items()
    }
    failure = error_type("injected RNG restore failure")
    original = ValueError("original isolated operation failed")
    calls = []
    try:
        with monkeypatch.context() as scoped:
            for name, (module, method) in setters.items():

                def restore(state, name=name):
                    calls.append(name)
                    if name == source:
                        raise failure
                    originals[name](state)

                scoped.setattr(module, method, restore)
            caught = None
            try:
                with preserve_rng_state(torch.device("cpu")):
                    random.random()
                    np.random.random()
                    torch.rand(4)
                    if body_fails:
                        raise original
            except BaseException as error:
                caught = error
            assert caught is (original if body_fails else failure)
            assert calls == ["torch", "python", "numpy"]
        if source != "torch":
            torch.testing.assert_close(
                torch.get_rng_state(), states["torch"], rtol=0, atol=0
            )
        if source != "python":
            assert random.getstate() == states["python"]
        if source != "numpy":
            assert pickle.dumps(np.random.get_state()) == pickle.dumps(states["numpy"])
    finally:
        for name, restore in originals.items():
            restore(states[name])


@pytest.mark.parametrize("source", ["python", "numpy", "torch", "deterministic"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
def test_assembly_rollback_attempts_remaining_restores_and_preserves_body_error(
    monkeypatch, source, error_type
):
    import os

    from llmtuner.utils.seeding import rollback_execution_settings_on_error

    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill_memory = torch.utils.deterministic.fill_uninitialized_memory
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    anomaly = (torch.is_anomaly_enabled(), torch.is_anomaly_check_nan_enabled())
    targets = {
        "python": (random, "setstate"),
        "numpy": (np.random, "set_state"),
        "torch": (torch, "set_rng_state"),
        "deterministic": (torch, "use_deterministic_algorithms"),
    }
    originals = {key: getattr(module, name) for key, (module, name) in targets.items()}
    calls = []
    original = ValueError("runtime assembly failed")
    try:
        with monkeypatch.context() as scoped:
            scoped.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            for key, (module, name) in targets.items():

                def restore(*args, key=key, **kwargs):
                    calls.append(key)
                    if key == source:
                        raise error_type("secondary rollback failed")
                    return originals[key](*args, **kwargs)

                scoped.setattr(module, name, restore)
            caught = None
            try:
                with rollback_execution_settings_on_error():
                    random.random()
                    np.random.random()
                    torch.rand(3)
                    # Invoke the real setter so only restoration is faulted.
                    originals["deterministic"](not deterministic)
                    torch.backends.cudnn.deterministic = not cudnn[0]
                    torch.backends.cudnn.benchmark = not cudnn[1]
                    torch.utils.deterministic.fill_uninitialized_memory = (
                        not fill_memory
                    )
                    torch.autograd.set_detect_anomaly(
                        not anomaly[0], check_nan=not anomaly[1]
                    )
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                    raise original
            except BaseException as error:
                caught = error
            assert caught is original
            assert calls == ["python", "numpy", "torch", "deterministic"]
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
            assert torch.backends.cudnn.deterministic == cudnn[0]
            assert torch.backends.cudnn.benchmark == cudnn[1]
            assert torch.utils.deterministic.fill_uninitialized_memory == fill_memory
            assert torch.is_anomaly_enabled() == anomaly[0]
            assert torch.is_anomaly_check_nan_enabled() == anomaly[1]
        if source != "python":
            assert random.getstate() == states[0]
        if source != "numpy":
            assert pickle.dumps(np.random.get_state()) == pickle.dumps(states[1])
        if source != "torch":
            assert torch.equal(torch.get_rng_state(), states[2])
        if source != "deterministic":
            assert torch.are_deterministic_algorithms_enabled() == deterministic
    finally:
        originals["python"](states[0])
        originals["numpy"](states[1])
        originals["torch"](states[2])
        originals["deterministic"](deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill_memory
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
        torch.autograd.set_detect_anomaly(anomaly[0], check_nan=anomaly[1])
