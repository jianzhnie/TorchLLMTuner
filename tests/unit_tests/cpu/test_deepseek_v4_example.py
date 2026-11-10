"""DeepSeek V4 examples expose existing FSDP policies without changing geometry."""

import pytest
from transformers import AutoConfig

from examples.deepseek_v4_pro.train import build_argument_parser, build_config
from llmtuner.errors import ConfigError
from llmtuner.models.hf.factory import build_model_config_for


def example_args(tmp_path, *options):
    AutoConfig.for_model("deepseek_v4").save_pretrained(tmp_path)
    return build_argument_parser().parse_args(
        [
            "--model",
            str(tmp_path),
            "--data",
            "unused.jsonl",
            "--output",
            "unused-output",
        ]
        + list(options)
    )


def test_cpu_offload_preserves_published_expert_geometry(tmp_path):
    args = example_args(
        tmp_path,
        "--cpu-offload",
        "--optimizer-state-residency",
        "device",
        "--layers",
        "4",
    )
    config = build_config(args)
    actual = build_model_config_for(config)
    assert config.parallel.fsdp_config.fsdp_cpu_offload
    assert config.optimizer.state_residency == "device"
    assert actual.moe_intermediate_size == 2048
    assert actual.n_routed_experts == 256
    assert actual.num_hidden_layers == 4


def test_cpu_offload_rejects_virtual_moments_before_training(tmp_path):
    with pytest.raises(ConfigError, match="virtual moments require NPU"):
        build_config(example_args(tmp_path, "--cpu-offload"))


def test_default_recipe_keeps_device_parameters_and_virtual_moments(tmp_path):
    config = build_config(example_args(tmp_path))
    assert not config.parallel.fsdp_config.fsdp_cpu_offload
    assert config.optimizer.state_residency == "npu_virtual"
