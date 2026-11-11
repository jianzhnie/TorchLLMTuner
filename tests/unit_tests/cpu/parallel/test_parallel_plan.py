"""The minimal immutable plan shared by resolution and execution."""

from dataclasses import FrozenInstanceError

import pytest

from llmtuner.parallel import ParallelPlan, PlanTarget


def test_plan_counts_resolved_targets_without_tensor_payloads() -> None:
    plan = ParallelPlan(
        stage="tp",
        targets=(
            PlanTarget("model.layers.0.mlp", "dense"),
            PlanTarget("model.layers.1.mlp", "moe"),
            PlanTarget("model.embed_tokens", "vocab"),
        ),
        degree=2,
        implementation="functional",
    )
    assert plan.count("dense") == 1
    assert plan.count("moe") == 1
    assert plan.count("vocab") == 1
    with pytest.raises(FrozenInstanceError):
        plan.degree = 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": ""},
        {"degree": 0},
        {"implementation": ""},
    ],
)
def test_plan_rejects_incomplete_execution_metadata(kwargs) -> None:
    values = {
        "stage": "tp",
        "targets": (),
        "degree": 2,
        "implementation": "functional",
        **kwargs,
    }
    with pytest.raises(ValueError):
        ParallelPlan(**values)


def test_plan_target_allows_the_root_module_path() -> None:
    target = PlanTarget("", "moe")
    assert target.path == ""
    assert target.kind == "moe"
