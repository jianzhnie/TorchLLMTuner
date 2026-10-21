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


@pytest.mark.parametrize("moment", ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"])
def test_container_rejects_wrong_moment_shape_before_loading_any_optimizer(moment):
    from llmtuner.components.optimizer.optimizer import OptimizersContainer

    container = object.__new__(OptimizersContainer)
    container.model_parts = []
    container.optimizers = []
    for name in ["first", "second"]:
        optimizer = torch.optim.AdamW(
            [{"params": [torch.nn.Parameter(torch.ones(2))], "param_names": [name]}],
            lr=0.1,
            amsgrad=True,
        )
        container.optimizers.append(optimizer)
    checkpoint = container.state_dict()
    checkpoint["param_groups.first.lr"] = 0.9
    checkpoint[f"state.second.{moment}"] = torch.zeros(3)
    original_states = [next(iter(opt.state.values())) for opt in container.optimizers]

    with pytest.raises(ValueError, match=f"state.second.{moment}.*shape"):
        container.load_state_dict(checkpoint)

    for optimizer, original_state in zip(
        container.optimizers, original_states, strict=True
    ):
        assert optimizer.param_groups[0]["lr"] == 0.1
        assert next(iter(optimizer.state.values())) is original_state


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize(
    "counter",
    [
        torch.tensor([1.0, 2.0]),
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
        torch.tensor(-1.0),
        torch.tensor(1.5),
        torch.tensor(True),
        True,
        "2",
    ],
)
def test_adam_restore_rejects_invalid_step_before_changing_state(
    optimizer_type, counter
):
    from llmtuner.components.optimizer.state_dict import (
        get_flat_optim_state_dict,
        load_flat_optim_state_dict,
    )

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = optimizer_type(
        [{"params": [parameter], "param_names": ["weight"]}],
        lr=0.1,
        foreach=False,
    )
    init_optim_state(optimizer)
    checkpoint = get_flat_optim_state_dict(optimizer)
    checkpoint["state.weight.step"] = counter
    checkpoint["param_groups.weight.lr"] = 0.9
    original_state = optimizer.state[parameter]
    with pytest.raises(ValueError, match="state.weight.step.*counter"):
        load_flat_optim_state_dict(optimizer, flat_state=checkpoint)
    assert optimizer.state[parameter] is original_state
    assert optimizer.param_groups[0]["lr"] == 0.1
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert original_state["step"].item() == 1
    assert torch.isfinite(parameter).all()


@pytest.mark.parametrize("failure", ["allocation", "post_hook"])
def test_failed_adam_materialization_discards_partial_state_and_can_retry(
    monkeypatch, failure
):
    parameter = torch.nn.Parameter(torch.ones(2))
    gradient = torch.ones_like(parameter)
    parameter.grad = gradient
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    before = parameter.detach().clone()
    original_zeros_like = torch.zeros_like
    calls = 0

    def allocate(value, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("partial state failed")
        return original_zeros_like(value, *args, **kwargs)

    def post_hook(*args, **kwargs):
        raise RuntimeError("partial state failed")

    handle = None
    if failure == "allocation":
        monkeypatch.setattr(torch, "zeros_like", allocate)
    else:
        handle = optimizer.register_step_post_hook(post_hook)
    with pytest.raises(RuntimeError, match="partial state failed"):
        init_optim_state(optimizer)
    assert not optimizer.state
    assert parameter.grad is gradient
    assert optimizer.param_groups[0]["lr"] == 0.1
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    monkeypatch.undo()
    if handle is not None:
        handle.remove()
    init_optim_state(optimizer)
    assert optimizer.state[parameter]["step"].item() == 0
    optimizer.step()
    assert optimizer.state[parameter]["step"].item() == 1


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("dtype,epsilon", [(torch.float32, 0.0), (torch.float16, 1e-8)])
def test_optimizer_materialization_avoids_zero_denominator_weight_corruption(
    optimizer_type, dtype, epsilon
):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=dtype))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = optimizer_type([parameter], lr=0.01, eps=epsilon, weight_decay=0)
    reference_optimizer = optimizer_type(
        [reference], lr=0.01, eps=epsilon, weight_decay=0
    )
    gradient = torch.ones_like(parameter)
    parameter.grad = gradient
    init_optim_state(optimizer)
    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    assert parameter.grad is gradient
    assert optimizer.param_groups[0]["eps"] == epsilon
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.state[parameter]["step"].item() == 0
    reference.grad = gradient.clone()
    optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    torch.testing.assert_close(
        optimizer.state[parameter], reference_optimizer.state[reference], rtol=0, atol=0
    )


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_optimizer_materialization_restores_epsilon_after_step_hook_failure(failure):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float16))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, eps=1e-8, weight_decay=0.3)
    gradient = torch.ones_like(parameter)
    parameter.grad = gradient
    before = parameter.detach().clone()
    original = failure("post allocation hook failure")

    def fail(*args, **kwargs):
        raise original

    handle = optimizer.register_step_post_hook(fail)
    try:
        with pytest.raises(failure) as caught:
            init_optim_state(optimizer)
        assert caught.value is original
        assert optimizer.param_groups[0]["weight_decay"] == 0.3
        assert optimizer.param_groups[0]["eps"] == 1e-8
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert parameter.grad is gradient
        assert parameter not in optimizer.state
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    finally:
        handle.remove()
    init_optim_state(optimizer)
    assert optimizer.state[parameter]["step"].item() == 0
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_optimizer_materialization_ignores_weight_decay_overflow(optimizer_type, dtype):
    parameter = torch.nn.Parameter(torch.tensor([torch.finfo(dtype).max], dtype=dtype))
    before = parameter.detach().clone()
    optimizer = optimizer_type([parameter], lr=0.01, weight_decay=10.0)
    init_optim_state(optimizer)
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert optimizer.param_groups[0]["weight_decay"] == 10.0
    assert optimizer.state[parameter]["step"].item() == 0
    assert torch.count_nonzero(optimizer.state[parameter]["exp_avg"]).item() == 0
    assert torch.count_nonzero(optimizer.state[parameter]["exp_avg_sq"]).item() == 0


@pytest.mark.parametrize(
    "saved_amsgrad,current_amsgrad", [(True, False), (False, True)]
)
def test_adam_restore_uses_saved_amsgrad_state_layout(saved_amsgrad, current_amsgrad):
    from copy import deepcopy

    from llmtuner.components.optimizer.state_dict import (
        get_flat_optim_state_dict,
        load_flat_optim_state_dict,
    )

    source = torch.nn.Parameter(torch.ones(2))
    target = torch.nn.Parameter(torch.ones(2))
    original = torch.optim.AdamW(
        [{"params": [source], "param_names": ["weight"]}], amsgrad=saved_amsgrad
    )
    restored = torch.optim.AdamW(
        [{"params": [target], "param_names": ["weight"]}], amsgrad=current_amsgrad
    )
    source.grad = torch.tensor([0.2, -0.3])
    original.step()
    target.data.copy_(source.detach())
    init_optim_state(restored)
    load_flat_optim_state_dict(
        restored, flat_state=deepcopy(get_flat_optim_state_dict(original))
    )
    assert restored.param_groups[0]["amsgrad"] is saved_amsgrad
    assert restored.state[target].keys() == original.state[source].keys()
    for parameter in (source, target):
        parameter.grad = torch.tensor([-0.1, 0.4])
    original.step()
    restored.step()
    torch.testing.assert_close(target, source, rtol=0, atol=0)
    torch.testing.assert_close(
        restored.state[target], original.state[source], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "state_name", ["step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"]
)
def test_adam_restore_rejects_nested_substitute_for_required_state(state_name):
    from llmtuner.components.optimizer.optimizer import OptimizersContainer

    container = object.__new__(OptimizersContainer)
    container.model_parts = []
    container.optimizers = [
        torch.optim.AdamW(
            [{"params": [torch.nn.Parameter(torch.ones(2))], "param_names": [name]}],
            lr=0.1,
            amsgrad=True,
        )
        for name in ("first", "second")
    ]
    checkpoint = container.state_dict()
    checkpoint["param_groups.first.lr"] = 0.9
    key = f"state.second.{state_name}"
    checkpoint[f"{key}.unexpected"] = checkpoint.pop(key)
    original_states = [next(iter(opt.state.values())) for opt in container.optimizers]

    with pytest.raises(KeyError, match=key):
        container.load_state_dict(checkpoint)

    for optimizer, original_state in zip(
        container.optimizers, original_states, strict=True
    ):
        assert optimizer.param_groups[0]["lr"] == 0.1
        assert next(iter(optimizer.state.values())) is original_state


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.AdamW])
def test_fqn_restore_matches_update_after_parameter_group_reordering(optimizer_type):
    from llmtuner.components.optimizer.state_dict import (
        get_flat_optim_state_dict,
        load_flat_optim_state_dict,
    )

    source = {
        name: torch.nn.Parameter(torch.tensor([value]))
        for name, value in (("left", 1.0), ("right", 2.0))
    }
    target = {
        name: torch.nn.Parameter(value.detach().clone())
        for name, value in source.items()
    }

    def build(parameters, names, rate):
        return optimizer_type(
            [
                {
                    "params": [parameters[name]],
                    "param_names": [name],
                    "lr": rate * (index + 1),
                }
                for index, name in enumerate(names)
            ],
            foreach=False,
        )

    original = build(source, ("left", "right"), 0.01)
    restored = build(target, ("right", "left"), 0.3)
    for index, parameter in enumerate(source.values()):
        parameter.grad = torch.full_like(parameter, index + 0.5)
    original.step()
    with torch.no_grad():
        for name in source:
            target[name].copy_(source[name])
    init_optim_state(restored)
    # A persisted checkpoint owns independent tensors. Native load_state_dict
    # may retain same-device tensor storage from an in-memory state dictionary.
    import copy

    load_flat_optim_state_dict(
        restored, flat_state=copy.deepcopy(get_flat_optim_state_dict(original))
    )
    torch.testing.assert_close(
        get_flat_optim_state_dict(restored),
        get_flat_optim_state_dict(original),
        rtol=0,
        atol=0,
    )
    for name in source:
        gradient = torch.full_like(source[name], -0.75 if name == "left" else 1.25)
        source[name].grad = gradient.clone()
        target[name].grad = gradient.clone()
    original.step()
    restored.step()
    for name in source:
        torch.testing.assert_close(target[name], source[name], rtol=0, atol=0)
    torch.testing.assert_close(
        get_flat_optim_state_dict(restored),
        get_flat_optim_state_dict(original),
        rtol=0,
        atol=0,
    )
