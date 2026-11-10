"""Step-boundary assertions must reject unsafe updates."""

import pytest
import torch

from llmtuner.trainer.runtime.gradients import check_step_condition


@pytest.mark.parametrize("value", [True, 1, 2.0])
def test_step_condition_accepts_true_scalar(value):
    check_step_condition(torch.tensor(value), "invalid step")


@pytest.mark.parametrize("value", [False, 0, 0.0])
def test_step_condition_rejects_false_scalar(value):
    with pytest.raises(RuntimeError, match="invalid step"):
        check_step_condition(torch.tensor(value), "invalid step")
