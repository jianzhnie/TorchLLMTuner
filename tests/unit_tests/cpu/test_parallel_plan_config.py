"""Nested parallel policies retain one source of truth through CLI and runtime."""

from dataclasses import asdict, fields, replace

import pytest

from llmtuner.config import (
    ContextParallelConfig,
    ExpertParallelConfig,
    FSDPConfig,
    ParallelConfig,
    PipelineParallelConfig,
    TensorParallelConfig,
)
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.trainer.cli import parse_config


def test_parallel_dataclass_owns_degrees_and_nested_plans():
    names = {item.name for item in fields(ParallelConfig)}
    assert {"tp_config", "fsdp_config", "ep_config", "cp_config", "pp_config"} <= names
    assert "tp_colwise_modules" not in names
    assert "fsdp_apply_modules" not in names
    config = ParallelConfig(
        tp_config=TensorParallelConfig(tp_colwise_modules=["layers.*"])
    )
    assert asdict(config)["tp_config"]["tp_colwise_modules"] == ["layers.*"]
    assert replace(config, tensor_parallel_size=2).tp_config is config.tp_config


@pytest.mark.parametrize("name", ["tp_colwise_modules", "fsdp_cpu_offload"])
def test_parallel_rejects_flattened_policy_keywords(name):
    with pytest.raises(TypeError, match=name):
        ParallelConfig(**{name: None})


def test_default_plans_and_mutable_fields_are_independent():
    first, second = ParallelConfig(), ParallelConfig()
    first.tp_config.tp_colwise_modules.append("first")
    first.fsdp_config.fsdp_apply_modules["layer"] = {}
    assert second.tp_config.tp_colwise_modules == []
    assert second.fsdp_config.fsdp_apply_modules == {}


@pytest.mark.parametrize("name", ["tp", "fsdp", "ep", "cp", "pp"])
def test_parallel_rejects_invalid_plan_type(name):
    with pytest.raises(ConfigError, match=f"{name}_config"):
        ParallelConfig(**{f"{name}_config": None})


def test_plans_validate_their_own_policies():
    with pytest.raises(ConfigError, match="conflicting styles"):
        TensorParallelConfig(tp_colwise_modules=["same"], tp_rowwise_modules=["same"])
    with pytest.raises(ConfigError, match="prefetch"):
        FSDPConfig(fsdp_forward_prefetch_count=-1)
    with pytest.raises(ConfigError, match="num_pp_microbatches"):
        PipelineParallelConfig(num_pp_microbatches=0)
    with pytest.raises(ConfigError, match="ep_torchao_pad_multiple"):
        ExpertParallelConfig(ep_torchao_pad_multiple=False)
    with pytest.raises(ConfigError, match="strategy"):
        ContextParallelConfig(context_parallel_strategy="unknown")


def test_nested_plans_keep_cross_degree_validation():
    with pytest.raises(UnsupportedCombinationError, match="sequence-parallel"):
        ParallelConfig(
            tensor_parallel_size=2,
            tp_config=TensorParallelConfig(enable_sequence_parallel=False),
        )
    with pytest.raises(UnsupportedCombinationError, match="KCP requires"):
        ParallelConfig(
            tensor_parallel_size=2,
            cp_config=ContextParallelConfig(context_parallel_strategy="kcp"),
        )
    with pytest.raises(UnsupportedCombinationError, match="has no effect"):
        ParallelConfig(ep_config=ExpertParallelConfig(ep_token_dispatcher="domino"))


def test_cli_grafts_plan_flags_before_cross_validation():
    config = parse_config(
        [
            "--tensor_parallel_size",
            "2",
            "--max_seq_len",
            "16",
            "--tp_colwise_modules",
            "layers.*.q_proj",
            "--fsdp_forward_prefetch_count",
            "2",
            "--expert_parallel_size",
            "2",
            "--ep_token_dispatcher",
            "domino",
            "--num_pp_microbatches",
            "2",
        ]
    )
    assert config.parallel.tp_config.tp_colwise_modules == ["layers.*.q_proj"]
    assert config.parallel.fsdp_config.fsdp_forward_prefetch_count == 2
    assert config.parallel.ep_config.ep_token_dispatcher == "domino"
    assert config.parallel.pp_config.num_pp_microbatches == 2
    with pytest.raises(UnsupportedCombinationError, match="sequence-parallel"):
        parse_config(
            ["--tensor_parallel_size", "2", "--enable_sequence_parallel", "false"]
        )


@pytest.mark.parametrize(
    "config_type",
    [
        TensorParallelConfig,
        FSDPConfig,
        ExpertParallelConfig,
        ContextParallelConfig,
        PipelineParallelConfig,
    ],
)
def test_parallel_policies_construct_independently(config_type):
    config_type()


def test_aggregate_revalidates_mutated_policy():
    plan = TensorParallelConfig()
    plan.tp_colwise_modules.append("same")
    plan.tp_rowwise_modules.append("same")
    with pytest.raises(ConfigError, match="conflicting styles"):
        ParallelConfig(tp_config=plan)


def test_cp_axis_constraints_remain_on_aggregate():
    # A strategy can be configured before its mesh degrees are known.
    plan = ContextParallelConfig(context_parallel_strategy="kcp")
    with pytest.raises(UnsupportedCombinationError, match="cannot be combined"):
        ParallelConfig(cp_config=plan, context_parallel_ulysses_size=2)
    with pytest.raises(ConfigError, match="requires strategy"):
        ParallelConfig(context_parallel_kcp_size=2)
    with pytest.raises(UnsupportedCombinationError, match="Ulysses CP axis"):
        ParallelConfig(
            context_parallel_ulysses_size=2,
            cp_config=ContextParallelConfig(context_parallel_load_balancer="headtail"),
        )
