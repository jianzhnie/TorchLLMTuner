"""Optimizer migration preserves moments and the next real AdamW update."""

from copy import deepcopy

import pytest
import torch

from llmtuner.components.optimizer.state_dict import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from llmtuner.models.kimi.checkpoint import convert_kimi_optimizer_state

ROOT = "model.layers.0.block_sparse_moe."


def test_adamw_individual_to_grouped_state_preserves_next_update():
    torch.manual_seed(10)
    individuals = [torch.nn.Parameter(torch.randn(4, 8)) for _ in range(3)]
    native = torch.optim.AdamW(
        [(f"{ROOT}experts.{i}.w1.weight", p) for i, p in enumerate(individuals)],
        lr=0.01,
        amsgrad=True,
    )
    for p in individuals:
        p.grad = torch.randn_like(p)
    native.step()
    flat = get_flat_optim_state_dict(native)
    migrated = convert_kimi_optimizer_state(flat, num_experts=3, to_hf=False)
    restored = convert_kimi_optimizer_state(migrated, num_experts=3, to_hf=True)
    assert restored.keys() == flat.keys()
    for key in flat:
        if isinstance(flat[key], torch.Tensor):
            torch.testing.assert_close(restored[key], flat[key], atol=0, rtol=0)
        else:
            assert restored[key] == flat[key]
    parameter = torch.nn.Parameter(torch.stack([p.detach() for p in individuals]))
    grouped = torch.optim.AdamW(
        [(f"{ROOT}routed_experts.inner_experts.w1_EFD", parameter)],
        lr=0.01,
        amsgrad=True,
    )
    init_optim_state(grouped)
    load_flat_optim_state_dict(grouped, flat_state=migrated)
    gradients = [torch.randn_like(p) for p in individuals]
    for p, gradient in zip(individuals, gradients, strict=True):
        p.grad = gradient
    parameter.grad = torch.stack(gradients)
    native.step()
    grouped.step()
    torch.testing.assert_close(parameter, torch.stack(individuals), atol=0, rtol=0)


@pytest.mark.parametrize("field,value", [("step", torch.tensor(7.0)), ("lr", 0.02)])
def test_grouped_migration_rejects_incompatible_expert_history(field, value):
    category = "state" if field == "step" else "param_groups"
    state = {
        f"{category}.{ROOT}experts.{i}.w1.weight.{field}": (
            torch.tensor(1.0) if field == "step" else 0.01
        )
        for i in range(2)
    }
    state[f"{category}.{ROOT}experts.1.w1.weight.{field}"] = value
    snapshot = deepcopy(state)
    with pytest.raises(ValueError, match="disagree"):
        convert_kimi_optimizer_state(state, num_experts=2, to_hf=False)
    assert state.keys() == snapshot.keys()
    for key in state:
        assert state[key] == snapshot[key]


def test_ema_tensor_migration_roundtrips_without_changing_source():
    key = f"state.{ROOT}routed_experts.inner_experts.w2_EDF.ema_params"
    tensor = torch.randn(4, 8, 12)
    state = {key: tensor}
    split = convert_kimi_optimizer_state(state, num_experts=4, to_hf=True)
    restored = convert_kimi_optimizer_state(split, num_experts=4, to_hf=False)
    torch.testing.assert_close(restored[key], tensor, atol=0, rtol=0)
    assert state[key] is tensor


def test_migrated_step_counters_do_not_alias_source():
    keys = [f"state.{ROOT}experts.{i}.w1.weight.step" for i in range(2)]
    state = {key: torch.tensor(2.0) for key in keys}
    migrated = convert_kimi_optimizer_state(state, num_experts=2, to_hf=False)
    counter = migrated[f"state.{ROOT}routed_experts.inner_experts.w1_EFD.step"]
    counter.add_(1)
    assert all(value.item() == 2 for value in state.values())
    split = convert_kimi_optimizer_state(migrated, num_experts=2, to_hf=True)
    split[keys[0]].add_(1)
    assert split[keys[1]].item() == 3
    assert counter.item() == 3


def test_optimizer_migration_rejects_incomplete_expert_moments():
    with pytest.raises(ValueError, match="missing individual experts"):
        convert_kimi_optimizer_state(
            {f"state.{ROOT}experts.0.w1.weight.exp_avg": torch.randn(4, 8)},
            num_experts=2,
            to_hf=False,
        )


@pytest.mark.parametrize("category", ["state", "param_groups"])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_kimi_optimizer_rejects_duplicate_numeric_expert_indices(
    category, reverse_order
):
    prefix = f"{category}.model.layers.0.block_sparse_moe.experts."
    field = "exp_avg" if category == "state" else "lr"
    entries = [
        (prefix + f"0.w1.weight.{field}", torch.ones(2, 3)),
        (prefix + f"00.w1.weight.{field}", torch.zeros(2, 3)),
        (prefix + f"1.w1.weight.{field}", torch.ones(2, 3)),
    ]
    if reverse_order:
        entries.reverse()
    with pytest.raises(ValueError, match="duplicate expert index"):
        convert_kimi_optimizer_state(dict(entries), num_experts=2, to_hf=False)
