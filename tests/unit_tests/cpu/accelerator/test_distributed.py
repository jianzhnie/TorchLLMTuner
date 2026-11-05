import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace

import pytest

from llmtuner.accelerator import distributed


def test_default_group_uses_public_world_handle(monkeypatch):
    world = object()
    monkeypatch.setattr(distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(distributed.torch_dist, "group", SimpleNamespace(WORLD=world))

    assert distributed.get_default_group() is world


def test_default_group_is_none_before_initialization(monkeypatch):
    monkeypatch.setattr(distributed, "is_distributed", lambda: False)

    assert distributed.get_default_group() is None


@pytest.mark.parametrize("launcher", ["init_dist_pytorch", "init_dist_slurm"])
def test_invalid_init_backend_fails_before_device_or_launcher_access(
    monkeypatch, launcher
):
    monkeypatch.setattr(distributed, "device_type", "cuda")
    monkeypatch.setattr(
        distributed.torch.cuda,
        "set_device",
        lambda *args: pytest.fail("invalid backend must not initialize a device"),
    )
    with pytest.raises(ValueError, match="colossalai.*invalid"):
        getattr(distributed, launcher)("nccl", init_backend="invalid")


@pytest.mark.parametrize("launcher", ["unknown", "pytorch", "slurm"])
def test_public_launcher_rejects_invalid_options_before_start_method(
    monkeypatch, launcher
):
    monkeypatch.setattr(distributed, "device_type", "cuda")
    monkeypatch.setattr(
        distributed.mp,
        "get_start_method",
        lambda **kwargs: pytest.fail("invalid options must not inspect process state"),
    )
    with pytest.raises(ValueError):
        distributed.init_dist(launcher, init_backend="invalid")


@pytest.mark.parametrize(
    "backend,module,error_name",
    [
        ("hccl", "torch_npu", "NPU_IMPORT_ERROR"),
        ("cncl", "torch_mlu", "MLU_IMPORT_ERROR"),
        ("mccl", "torch_musa", "MUSA_IMPORT_ERROR"),
    ],
)
def test_communication_device_reports_missing_vendor_dependency(
    monkeypatch, backend, module, error_name
):
    cause = ModuleNotFoundError(f"missing {module}")
    monkeypatch.setattr(distributed, "get_backend", lambda group: backend)
    monkeypatch.setattr(distributed, module, None)
    monkeypatch.setattr(distributed, error_name, cause, raising=False)
    with pytest.raises(ImportError, match="communication requires") as failure:
        distributed.get_comm_device()
    assert failure.value.__cause__ is cause


def test_incompatible_smddp_does_not_block_default_distributed_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent(
                """
                import importlib.abc
                import sys

                class BrokenSageMaker(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "smdistributed":
                            raise ImportError("incompatible SageMaker extension")

                sys.meta_path.insert(0, BrokenSageMaker())
                from llmtuner.accelerator import distributed
                assert distributed.get_rank() == 0
                try:
                    distributed.init_dist_mpi("smddp")
                except ImportError as error:
                    assert "Amazon SageMaker DLC" in str(error)
                    assert "incompatible SageMaker" in str(error.__cause__)
                else:
                    raise AssertionError("unavailable backend was accepted")
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
