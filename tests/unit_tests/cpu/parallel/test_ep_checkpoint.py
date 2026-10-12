"""Round-trip tests for EP-aware expert-state serialization (ckpt.py).

The collectives are simulated in-process: a fake EP group of world 2 whose
all_gather simply concatenates two predetermined rank slices. That is enough
to pin the contract that regressed once already -- the save direction gathers
local slices, the load direction must slice full tensors back (the shape gate
is direction-aware), scalars are left alone, and a degree/layout mismatch is
a loud error.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llmtuner.parallel.expert_parallel import ckpt


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
    monkeypatch.setattr(ckpt.dist_utils, "get_world_size", lambda g: world)
    monkeypatch.setattr(ckpt.dist_utils, "get_rank", lambda g: rank)

    def fake_all_gather(out, value, group=None):
        out.copy_(torch.cat([value + 100 * r for r in range(world)], dim=0))

    monkeypatch.setattr(ckpt.dist, "all_gather_into_tensor", fake_all_gather)
    return group


def test_save_gathers_and_load_slices_back(monkeypatch) -> None:
    model = _model_with_experts()
    grouped = model.mlp
    group = _fake_group(monkeypatch, world=2, rank=0)
    ckpt.mark_experts_ep_sharded(grouped, group)

    shard_map = ckpt.expert_shard_map([model])
    assert set(shard_map) == {"mlp.w1_EFD", "mlp.w2_EDF"}

    sd = model.state_dict()
    full = ckpt.gather_expert_state(sd, shard_map, [model])
    assert full["mlp.w1_EFD"].shape == (4, 4, 3)  # E = num_local * world
    torch.testing.assert_close(full["mlp.w1_EFD"][:2], grouped.w1_EFD.detach())
    torch.testing.assert_close(full["mlp.w1_EFD"][2:], grouped.w1_EFD.detach() + 100)

    # Load on rank 1 keeps the other half.
    _fake_group(monkeypatch, world=2, rank=1)
    loaded = ckpt.load_expert_state(full, shard_map, [model])
    torch.testing.assert_close(loaded["mlp.w1_EFD"], grouped.w1_EFD.detach() + 100)
    assert loaded["mlp.w1_EFD"].shape == grouped.w1_EFD.shape


def test_scalars_and_unrelated_state_are_untouched(monkeypatch) -> None:
    model = _model_with_experts()
    group = _fake_group(monkeypatch, world=2, rank=0)
    ckpt.mark_experts_ep_sharded(model.mlp, group)
    shard_map = ckpt.expert_shard_map([model])

    # An optimizer-style dict: expert-shaped state is transformed, the
    # scalar `step` under the same FQN prefix must not be.
    sd = {
        "state.mlp.w1_EFD.exp_avg": torch.zeros(2, 4, 3),
        "state.mlp.w1_EFD.step": torch.tensor(3.0),
    }
    out = ckpt.gather_expert_state(sd, shard_map, [model])
    assert out["state.mlp.w1_EFD.exp_avg"].shape == (4, 4, 3)
    assert out["state.mlp.w1_EFD.step"].item() == 3.0


def test_load_with_a_mismatched_expert_count_raises(monkeypatch) -> None:
    model = _model_with_experts()
    group = _fake_group(monkeypatch, world=2, rank=0)
    ckpt.mark_experts_ep_sharded(model.mlp, group)
    shard_map = ckpt.expert_shard_map([model])

    # 6 experts on disk, this rank holds 2 x ep=2 = 4 -> loud mismatch.
    bad = {"mlp.w1_EFD": torch.zeros(6, 4, 3)}
    try:
        ckpt.load_expert_state(bad, shard_map, [model])
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "different expert layout" in str(exc)
