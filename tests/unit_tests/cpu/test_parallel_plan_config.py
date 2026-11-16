"""Nested parallel policies retain one source of truth through CLI and runtime."""

import pytest

from llmtuner.config import (
    ContextParallelConfig,
    ExpertParallelConfig,
    FSDPConfig,
    ParallelConfig,
    TensorParallelConfig,
)
from llmtuner.errors import ConfigError, UnsupportedCombinationError


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


@pytest.mark.parametrize("name", ["tp", "fsdp", "ep", "cp"])
def test_parallel_rejects_invalid_plan_type(name):
    with pytest.raises(ConfigError, match=f"{name}_config"):
        ParallelConfig(**{f"{name}_config": None})


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


@pytest.mark.parametrize(
    "config_type",
    [
        TensorParallelConfig,
        FSDPConfig,
        ExpertParallelConfig,
        ContextParallelConfig,
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
