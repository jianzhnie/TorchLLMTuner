"""Published model launchers must resolve after example directory moves."""

import re
from importlib import import_module
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHERS = (
    "qwen3-dense/train_qwen3_8b_npu.sh",
    "qwen3-moe/train_qwen3_30b_npu.sh",
    "qwen3_5/train.sh",
    "kimi_k3/train.sh",
    "deepseek_v4_pro/train_pro.sh",
    "deepseek_v4_flash/train.sh",
)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_training_module_and_repository_root_exist(launcher):
    path = REPO_ROOT / "examples" / launcher
    source = path.read_text()
    modules = re.findall(r"-m\s+(examples\.[\w.]+)", source)
    assert len(modules) == 1, f"Expected one training module in {path}"
    module_file = REPO_ROOT.joinpath(*modules[0].split(".")).with_suffix(".py")
    assert module_file.is_file(), f"Missing training module: {module_file}"
    # Every published launcher is now exactly two levels below the repository.
    assert '${BASH_SOURCE[0]}")/../.."' in source


@pytest.mark.parametrize(
    "module",
    ["npu_checkpoint_backend_equivalence", "npu_profiler_equivalence"],
)
def test_auxiliary_npu_scripts_import_without_deleted_examples(module):
    script = import_module(f"tests.integration_tests.{module}")
    config = script.qwen3_config()
    assert config.model.hidden_size == 64
    assert config.model.num_hidden_layers == 2
    assert config.training.deterministic
    assert config.optimizer.learning_rate == 3e-4
