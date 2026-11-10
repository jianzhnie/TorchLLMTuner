"""KDA dependency validation runs before expensive model allocation."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from llmtuner.errors import EnvironmentUnsupportedError
from llmtuner.models.kimi import delta_attention
from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig


def kda_config():
    return KimiLinearConfig(
        num_hidden_layers=2,
        linear_attn_config={"kda_layers": [1], "full_attn_layers": [2]},
    )


def test_cpu_import_does_not_load_installed_npu_kernel_runtime():
    # Run in a fresh interpreter so an already imported Kimi module cannot make
    # this pass vacuously. Optional Ascend packages can raise AttributeError
    # rather than ImportError when CPU torch has no registered torch.npu.
    script = """
import builtins
from llmtuner.accelerator import device

device.is_device_type_available = lambda kind: False
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.startswith('triton_ascend_kernels'):
        raise AssertionError('CPU import reached an unavailable NPU runtime')
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from llmtuner.models.kimi import delta_attention
assert delta_attention.npu_chunk_kda is None
assert delta_attention.NPU_KDA_IMPORT_ERROR is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_npu_kda_reports_dependency_and_original_cause(monkeypatch):
    cause = ModuleNotFoundError("missing kernel runtime")
    monkeypatch.setattr(delta_attention, "npu_chunk_kda", None)
    monkeypatch.setattr(delta_attention, "NPU_KDA_IMPORT_ERROR", cause)
    with pytest.raises(
        EnvironmentUnsupportedError, match="triton_ascend_kernels"
    ) as exc:
        delta_attention.validate_kimi_kda_runtime(
            kda_config(), SimpleNamespace(type="npu")
        )
    assert exc.value.__cause__ is cause


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_cpu_and_meta_kda_do_not_require_accelerator_kernel(monkeypatch, device):
    monkeypatch.setattr(delta_attention, "npu_chunk_kda", None)
    monkeypatch.setattr(delta_attention, "accelerator_chunk_kda", None)
    delta_attention.validate_kimi_kda_runtime(kda_config(), torch.device(device))


def test_full_attention_only_and_other_models_do_not_require_kda(monkeypatch):
    monkeypatch.setattr(delta_attention, "accelerator_chunk_kda", None)
    config = KimiLinearConfig(num_hidden_layers=2, linear_attn_config=None)
    delta_attention.validate_kimi_kda_runtime(config, torch.device("cuda"))
    delta_attention.validate_kimi_kda_runtime(PretrainedConfig(), torch.device("cuda"))


def test_available_kda_kernel_allows_accelerator_configuration(monkeypatch):
    monkeypatch.setattr(delta_attention, "accelerator_chunk_kda", object())
    delta_attention.validate_kimi_kda_runtime(kda_config(), torch.device("cuda"))
