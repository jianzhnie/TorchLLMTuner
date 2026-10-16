"""Optimizer FQN serialization metadata validation, without named-param APIs."""

import pytest
import torch

from llmtuner.components.optimizer.state_dict import init_optim_state


@pytest.mark.parametrize("reverse_order", [False, True])
def test_flatten_rejects_colliding_nested_state_keys(reverse_order):
    from llmtuner.components.optimizer.state_dict import get_flat_optim_state_dict

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD(
        [{"params": [parameter], "param_names": ["weight"]}], lr=0.1
    )
    entries = [("cache", {"value": 1}), ("cache.value", 2)]
    optimizer.state[parameter] = dict(reversed(entries) if reverse_order else entries)
    with pytest.raises(ValueError, match="keys collide when flattened"):
        get_flat_optim_state_dict(optimizer)


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("parent_value", [2, {}])
def test_restore_rejects_conflicting_nested_paths_without_mutating_state(
    reverse_order, parent_value
):
    from llmtuner.components.optimizer.state_dict import (
        get_flat_optim_state_dict,
        load_flat_optim_state_dict,
    )

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD(
        [{"params": [parameter], "param_names": ["weight"]}], lr=0.1
    )
    original = {"cache": {"leaf": 1}}
    optimizer.state[parameter] = original
    flat = get_flat_optim_state_dict(optimizer)
    del flat["state.weight.cache.leaf"]
    entries = [
        ("state.weight.cache.leaf", parent_value),
        ("state.weight.cache.leaf.child", 3),
    ]
    flat.update(reversed(entries) if reverse_order else entries)
    with pytest.raises(ValueError, match="Conflicting nested optimizer state paths"):
        load_flat_optim_state_dict(optimizer, flat_state=flat)
    assert optimizer.state[parameter] is original
    assert optimizer.state[parameter] == {"cache": {"leaf": 1}}
    if isinstance(parent_value, dict):
        assert parent_value == {}


@pytest.mark.parametrize("tensor_lr", [False, True])
def test_flat_restore_rejects_merging_groups_with_different_learning_rates(tensor_lr):
    from llmtuner.components.optimizer.state_dict import (
        get_flat_optim_state_dict,
        load_flat_optim_state_dict,
    )

    first = torch.nn.Parameter(torch.ones(2))
    second = torch.nn.Parameter(torch.ones(2))
    source = torch.optim.AdamW(
        [
            {
                "params": [first],
                "param_names": ["first"],
                "lr": torch.tensor(0.1) if tensor_lr else 0.1,
            },
            {
                "params": [second],
                "param_names": ["second"],
                "lr": torch.tensor(0.2) if tensor_lr else 0.2,
            },
        ]
    )
    target = torch.optim.AdamW(
        [{"params": [first, second], "param_names": ["first", "second"]}], lr=0.3
    )
    original_group = target.param_groups[0]
    with pytest.raises(ValueError, match="checkpoint group option 'lr' differs"):
        load_flat_optim_state_dict(target, flat_state=get_flat_optim_state_dict(source))
    assert target.param_groups[0] is original_group
    assert target.param_groups[0]["lr"] == 0.3
    assert not target.state


@pytest.mark.parametrize("persist", [False, True])
def test_flat_optimizer_state_preserves_empty_nested_dictionaries(tmp_path, persist):
    from llmtuner.components.optimizer.state_dict import (
        get_flat_optim_state_dict,
        load_flat_optim_state_dict,
    )

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD(
        [{"params": [parameter], "param_names": ["weight"]}], lr=0.1
    )
    optimizer.state[parameter] = {"cache": {}, "nested": {"empty": {}, "count": 3}}
    flat = get_flat_optim_state_dict(optimizer)
    assert flat["state.weight.cache"] == {}
    assert flat["state.weight.nested.empty"] == {}
    if persist:
        import torch.distributed.checkpoint as dcp

        dcp.save(flat, checkpoint_id=tmp_path)
        flat = {
            key: 0 if key == "state.weight.nested.count" else value
            for key, value in flat.items()
        }
        dcp.load(flat, checkpoint_id=tmp_path)
    load_flat_optim_state_dict(optimizer, flat_state=flat)
    assert optimizer.state[parameter] == {
        "cache": {},
        "nested": {"empty": {}, "count": 3},
    }


def test_flat_restore_preserves_empty_parameter_group_options():
    from llmtuner.components.optimizer.state_dict import (
        get_flat_optim_state_dict,
        load_flat_optim_state_dict,
    )

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter], "param_names": ["weight"], "lr": 0.1},
            {"params": [], "param_names": [], "lr": 0.2, "weight_decay": 0.3},
        ]
    )
    init_optim_state(optimizer)
    flat_state = get_flat_optim_state_dict(optimizer)
    empty_options = dict(optimizer.param_groups[1])
    load_flat_optim_state_dict(optimizer, flat_state=flat_state)
    assert optimizer.param_groups[1] == empty_options
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[1]["lr"] == 0.1
    assert torch.isfinite(parameter).all()


def test_materialization_restores_gradients_after_allocation_failure(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(2))
    gradient = torch.ones_like(parameter)
    parameter.grad = gradient
    optimizer = torch.optim.Adam([parameter], lr=0.1)

    def fail(tensor):
        raise RuntimeError("zero allocation failed")

    monkeypatch.setattr(torch, "zeros_like", fail)
    with pytest.raises(RuntimeError, match="zero allocation failed"):
        init_optim_state(optimizer)

    assert parameter.grad is gradient
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_materialization_restores_groups_after_tensor_lr_allocation_failure(
    monkeypatch,
):
    first = torch.nn.Parameter(torch.ones(2))
    second = torch.nn.Parameter(torch.ones(2))
    gradients = [torch.ones_like(first), torch.ones_like(second)]
    first.grad, second.grad = gradients
    tensor_lr = torch.tensor(0.2)
    optimizer = torch.optim.Adam(
        [{"params": [first], "lr": 0.1}, {"params": [second], "lr": tensor_lr}]
    )
    original_zeros_like = torch.zeros_like

    def allocate(tensor):
        if tensor is tensor_lr:
            raise RuntimeError("learning rate allocation failed")
        return original_zeros_like(tensor)

    monkeypatch.setattr(torch, "zeros_like", allocate)
    with pytest.raises(RuntimeError, match="learning rate allocation failed"):
        init_optim_state(optimizer)

    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[1]["lr"] is tensor_lr
    assert first.grad is gradients[0]
    assert second.grad is gradients[1]
    assert not optimizer.state


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
        unflatten_optim_state_dict(optimizer, flat_state={})


@pytest.mark.parametrize(
    "ids, names, message",
    [
        ([0, 1], ["weight", "weight"], "Duplicate optimizer parameter FQN"),
        ([0, 0], ["weight", "bias"], "Duplicate optimizer parameter id"),
    ],
)
def test_fqn_serialization_rejects_ambiguous_parameter_mapping(ids, names, message):
    from llmtuner.components.optimizer.state_dict import optim_state_dict_to_fqn_keys

    state = {
        "state": {0: {"step": 1}, 1: {"step": 2}},
        "param_groups": [{"params": ids, "param_names": names}],
    }
    with pytest.raises(ValueError, match=message):
        optim_state_dict_to_fqn_keys(state)


def test_incomplete_optimizer_checkpoint_is_rejected_without_state_changes():
    from llmtuner.components.optimizer.state_dict import (
        get_flat_optim_state_dict,
        load_flat_optim_state_dict,
    )

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], foreach=False)
    optimizer.param_groups[0]["param_names"] = ["weight"]
    init_optim_state(optimizer)
    checkpoint = get_flat_optim_state_dict(optimizer)
    del checkpoint["state.weight.exp_avg"]
    original_state = optimizer.state[parameter]
    original_moment = original_state["exp_avg"]
    with pytest.raises(KeyError, match="state.weight.exp_avg"):
        load_flat_optim_state_dict(optimizer, flat_state=checkpoint)
    assert optimizer.state[parameter] is original_state
    assert optimizer.state[parameter]["exp_avg"] is original_moment
    assert original_state["step"].item() == 0


def test_optimizer_container_rejects_checkpoint_keys_shared_between_optimizers():
    from llmtuner.components.optimizer.optimizer import OptimizersContainer

    container = object.__new__(OptimizersContainer)
    container.model_parts = []
    container.optimizers = []
    for _ in range(2):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
        optimizer.param_groups[0]["param_names"] = ["weight"]
        container.optimizers.append(optimizer)
    with pytest.raises(ValueError, match="overlap across inner optimizers"):
        container.state_dict()


def test_container_validates_all_checkpoint_states_before_loading_any():
    from llmtuner.components.optimizer.optimizer import OptimizersContainer

    container = object.__new__(OptimizersContainer)
    container.model_parts = []
    container.optimizers = []
    for name in ["first", "second"]:
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], lr=0.1)
        optimizer.param_groups[0]["param_names"] = [name]
        container.optimizers.append(optimizer)
    checkpoint = container.state_dict()
    checkpoint["param_groups.first.lr"] = 0.9
    del checkpoint["state.second.exp_avg"]
    first_state = next(iter(container.optimizers[0].state.values()))
    with pytest.raises(KeyError, match="state.second.exp_avg"):
        container.load_state_dict(checkpoint)
    assert container.optimizers[0].param_groups[0]["lr"] == 0.1
    assert next(iter(container.optimizers[0].state.values())) is first_state


def test_flat_optimizer_restore_keeps_parameter_names_for_the_next_save():
    from llmtuner.components.optimizer.state_dict import (
        get_flat_optim_state_dict,
        load_flat_optim_state_dict,
    )

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    optimizer.param_groups[0]["param_names"] = ["weight"]
    init_optim_state(optimizer)
    saved = get_flat_optim_state_dict(optimizer)
    load_flat_optim_state_dict(optimizer, flat_state=saved)
    assert optimizer.param_groups[0]["param_names"] == ["weight"]
    assert get_flat_optim_state_dict(optimizer).keys() == saved.keys()
