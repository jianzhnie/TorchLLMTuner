"""Round-trip tests for EP-aware expert-state serialization (checkpointing.py).

The collectives are simulated in-process: a fake EP group of world 2 whose
all_gather simply concatenates two predetermined rank slices. That is enough
to pin the contract that regressed once already -- the save direction gathers
local slices, the load direction must slice full tensors back (the shape gate
is direction-aware), scalars are left alone, and a degree/layout mismatch is
a loud error.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llmtuner.parallel.expert_parallel import checkpointing


def _model_with_experts(num_local: int = 2) -> nn.Module:
    torch.manual_seed(0)
    block = nn.Module()
    grouped = nn.Module()
    grouped.w1_EFD = nn.Parameter(torch.randn(num_local, 4, 3))
    grouped.w2_EDF = nn.Parameter(torch.randn(num_local, 3, 4))
    block.mlp = grouped
    return block


def _fake_group(monkeypatch, *, world: int, rank: int):
    """A fake EP group where rank r's local value is ``base + 100 * r``."""
    group = object()
    monkeypatch.setattr(checkpointing.distributed, "get_world_size", lambda g: world)
    monkeypatch.setattr(checkpointing.distributed, "get_rank", lambda g: rank)

    def fake_all_gather(out, value, group=None):
        out.copy_(torch.cat([value + 100 * r for r in range(world)], dim=0))

    monkeypatch.setattr(checkpointing.dist, "all_gather_into_tensor", fake_all_gather)
    return group


def test_save_gathers_and_load_slices_back(monkeypatch) -> None:
    model = _model_with_experts()
    grouped = model.mlp
    group = _fake_group(monkeypatch, world=2, rank=0)
    checkpointing.mark_experts_ep_sharded(grouped, group)

    shard_map = checkpointing.expert_shard_map([model])
    assert set(shard_map) == {"mlp.w1_EFD", "mlp.w2_EDF"}

    sd = model.state_dict()
    full = checkpointing.gather_expert_state(
        sd, shard_map=shard_map, model_parts=[model]
    )
    assert full["mlp.w1_EFD"].shape == (4, 4, 3)  # E = num_local * world
    torch.testing.assert_close(full["mlp.w1_EFD"][:2], grouped.w1_EFD.detach())
    torch.testing.assert_close(full["mlp.w1_EFD"][2:], grouped.w1_EFD.detach() + 100)

    # Load on rank 1 keeps the other half.
    _fake_group(monkeypatch, world=2, rank=1)
    loaded = checkpointing.load_expert_state(
        full, shard_map=shard_map, model_parts=[model]
    )
    torch.testing.assert_close(loaded["mlp.w1_EFD"], grouped.w1_EFD.detach() + 100)
    assert loaded["mlp.w1_EFD"].shape == grouped.w1_EFD.shape


def test_scalars_and_unrelated_state_are_untouched(monkeypatch) -> None:
    model = _model_with_experts()
    group = _fake_group(monkeypatch, world=2, rank=0)
    checkpointing.mark_experts_ep_sharded(model.mlp, group)
    shard_map = checkpointing.expert_shard_map([model])

    # An optimizer-style dict: expert-shaped state is transformed, the
    # scalar `step` under the same FQN prefix must not be.
    sd = {
        "state.mlp.w1_EFD.exp_avg": torch.zeros(2, 4, 3),
        "state.mlp.w1_EFD.step": torch.tensor(3.0),
    }
    out = checkpointing.gather_expert_state(
        sd, shard_map=shard_map, model_parts=[model]
    )
    assert out["state.mlp.w1_EFD.exp_avg"].shape == (4, 4, 3)
    assert out["state.mlp.w1_EFD.step"].item() == 3.0


def test_load_with_a_mismatched_expert_count_raises(monkeypatch) -> None:
    model = _model_with_experts()
    group = _fake_group(monkeypatch, world=2, rank=0)
    checkpointing.mark_experts_ep_sharded(model.mlp, group)
    shard_map = checkpointing.expert_shard_map([model])

    # 6 experts on disk, this rank holds 2 x ep=2 = 4 -> loud mismatch.
    bad = {"mlp.w1_EFD": torch.zeros(6, 4, 3)}
    try:
        checkpointing.load_expert_state(bad, shard_map=shard_map, model_parts=[model])
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "different expert layout" in str(exc)


@pytest.mark.parametrize("direction", ["save", "load"])
@pytest.mark.parametrize("shape", [(4, 4), (4, 5, 3)])
def test_expert_weight_shape_mismatch_is_rejected(monkeypatch, direction, shape):
    model = _model_with_experts()
    group = _fake_group(monkeypatch, world=2, rank=0)
    checkpointing.mark_experts_ep_sharded(model.mlp, group)
    transform = (
        checkpointing.gather_expert_state
        if direction == "save"
        else checkpointing.load_expert_state
    )
    with pytest.raises(ValueError, match="EP checkpoint weight.*incompatible shape"):
        transform(
            {"mlp.w1_EFD": torch.zeros(shape)},
            shard_map=checkpointing.expert_shard_map([model]),
            model_parts=[model],
        )


@pytest.mark.parametrize("direction", ["save", "load"])
def test_non_parameter_shaped_optimizer_state_is_untouched(monkeypatch, direction):
    model = _model_with_experts()
    group = _fake_group(monkeypatch, world=2, rank=0)
    checkpointing.mark_experts_ep_sharded(model.mlp, group)
    value = torch.ones(4)
    state = {"state.mlp.w1_EFD.statistics": value}
    transform = (
        checkpointing.gather_expert_state
        if direction == "save"
        else checkpointing.load_expert_state
    )
    result = transform(
        state,
        shard_map=checkpointing.expert_shard_map([model]),
        model_parts=[model],
    )
    assert result["state.mlp.w1_EFD.statistics"] is value


@pytest.mark.parametrize("optimizer_state", [False, True])
def test_activation_checkpoint_wrappers_preserve_expert_state_matching(
    monkeypatch, optimizer_state
):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    model = _model_with_experts()
    group = _fake_group(monkeypatch, world=2, rank=0)
    checkpointing.mark_experts_ep_sharded(model.mlp, group)
    model.mlp = checkpoint_wrapper(model.mlp)
    shard_map = checkpointing.expert_shard_map([model])
    assert set(shard_map) == {"mlp.w1_EFD", "mlp.w2_EDF"}
    assert set(checkpointing.index_checkpoint_parameters([model])) == set(shard_map)
    source = model.state_dict()
    if optimizer_state:
        source = {
            f"state.{key}.exp_avg": value.clone() for key, value in source.items()
        }
    saved = checkpointing.gather_expert_state(
        source, shard_map=shard_map, model_parts=[model]
    )
    for key, value in source.items():
        assert saved[key].shape[0] == 4
        torch.testing.assert_close(saved[key], torch.cat([value, value + 100]))
    _fake_group(monkeypatch, world=2, rank=1)
    restored = checkpointing.load_expert_state(
        saved, shard_map=shard_map, model_parts=[model]
    )
    for key, value in source.items():
        torch.testing.assert_close(restored[key], value + 100)


def test_live_parameter_index_rejects_colliding_model_part_names():
    parts = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
    with pytest.raises(ValueError, match="identifies different parameters"):
        checkpointing.index_checkpoint_parameters(parts)


def test_live_parameter_index_accepts_repeated_identity():
    model = torch.nn.Linear(2, 2)
    params = checkpointing.index_checkpoint_parameters([model, model])
    assert params["weight"] is model.weight


def test_expert_shard_map_rejects_conflicting_groups():
    parts = [_model_with_experts(), _model_with_experts()]
    for part in parts:
        checkpointing.mark_experts_ep_sharded(part.mlp, object())
    with pytest.raises(ValueError, match="different expert process groups"):
        checkpointing.expert_shard_map(parts)


def test_expert_shard_map_accepts_repeated_group_identity():
    model = _model_with_experts()
    group = object()
    checkpointing.mark_experts_ep_sharded(model.mlp, group)
    assert checkpointing.expert_shard_map([model, model])["mlp.w1_EFD"] is group
