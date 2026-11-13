"""PP-local CP plans preserve globally declared target ownership and layout."""

from copy import deepcopy
from dataclasses import replace

import pytest
from torch import nn

from llmtuner.parallel.context_parallel.plan import (
    validate_pipeline_cp_plan,
)
from llmtuner.parallel.plan import ParallelPlan, PlanTarget


@pytest.mark.parametrize("change", [None, "missing", "unexpected", "layout"])
def test_pipeline_cp_validates_retained_targets(change):
    whole = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    part = deepcopy(whole)
    part[1] = nn.Identity()
    global_plan = ParallelPlan(
        "cp",
        tuple(
            PlanTarget(str(i), "attention", module, "kda_state")
            for i, module in enumerate(whole)
        ),
        2,
        "kcp",
    )
    target = replace(global_plan.targets[0], module=part[0])
    targets = (target,)
    if change == "missing":
        targets = ()
    elif change == "unexpected":
        targets += (PlanTarget("1", "attention", part[1], "kda_state"),)
    elif change == "layout":
        targets = (replace(target, layout="mla_ulysses"),)
    local_plan = replace(global_plan, targets=targets)
    if change is None:
        validate_pipeline_cp_plan(global_plan, local_plan, part)
    else:
        with pytest.raises(ValueError, match="Pipeline CP"):
            validate_pipeline_cp_plan(global_plan, local_plan, part)
