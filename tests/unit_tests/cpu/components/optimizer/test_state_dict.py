"""Optimizer FQN serialization metadata validation, without named-param APIs."""

import pytest
import torch


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
