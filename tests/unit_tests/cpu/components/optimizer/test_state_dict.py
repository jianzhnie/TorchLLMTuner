"""Optimizer FQN serialization metadata validation, without named-param APIs."""

import pytest
import torch

from llmtuner.components.optimizer.state_dict import init_optim_state


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("fail_step", [False, True])
def test_optimizer_materialization_preserves_tensor_lr_and_gradients(
    optimizer_type, fail_step
):
    parameter = torch.nn.Parameter(torch.ones(2))
    gradient = torch.tensor([0.5, -0.25])
    parameter.grad = gradient
    learning_rate = torch.tensor(0.1, dtype=torch.float64)
    optimizer = optimizer_type([parameter], lr=learning_rate, foreach=False)
    seen = []

    def inspect_step(optimizer, args, kwargs):
        temporary_lr = optimizer.param_groups[0]["lr"]
        seen.append(temporary_lr)
        if fail_step:
            raise RuntimeError("injected optimizer failure")

    hook = optimizer.register_step_pre_hook(inspect_step)
    try:
        if fail_step:
            with pytest.raises(RuntimeError, match="injected optimizer failure"):
                init_optim_state(optimizer)
        else:
            init_optim_state(optimizer)
    finally:
        hook.remove()

    assert len(seen) == 1
    torch.testing.assert_close(seen[0], torch.zeros_like(learning_rate))
    assert optimizer.param_groups[0]["lr"] is learning_rate
    assert parameter.grad is gradient
    torch.testing.assert_close(parameter, torch.ones(2))
    if not fail_step:
        assert optimizer.state[parameter]["step"].item() == 0


@pytest.mark.parametrize("names", [[], ["first", "extra"]])
def test_optimizer_serialization_rejects_misaligned_parameter_names(names):
    from llmtuner.components.optimizer.state_dict import optim_state_dict_to_fqn_keys

    state = {"state": {}, "param_groups": [{"params": [0], "param_names": names}]}
    with pytest.raises(ValueError, match="zip"):
        optim_state_dict_to_fqn_keys(state)


def test_optimizer_restore_rejects_misaligned_parameter_names():
    from llmtuner.components.optimizer.state_dict import unflatten_optim_state_dict

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    optimizer.param_groups[0]["param_names"] = []
    with pytest.raises(ValueError, match="zip"):
        unflatten_optim_state_dict(optimizer, {})
