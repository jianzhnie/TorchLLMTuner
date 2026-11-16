"""Generic distinct RNG coordinates and DTensor RNG initialization."""

import pytest
import torch

from llmtuner.utils.seeding import derive_distinct_seed


def _rng_snapshot(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.get_rng_state()


class TestDeriveDistinctSeed:
    def test_first_coordinate_varies_fastest(self):
        # Multi-dim generality: (r0, s0), (r1, s1) -> r0 + r1 * s0.
        assert derive_distinct_seed(100, [(2, 3), (1, 5)]) == 100 + 2 + 1 * 3

    def test_result_stays_in_torch_seed_range(self):
        base_seed = 2**64 - 2
        derived = derive_distinct_seed(base_seed, [(3, 4)])
        assert derived == (base_seed + 3) % 2**64
        torch.manual_seed(derived)  # must be accepted by torch

    def test_invalid_coordinates_are_rejected(self):
        for coordinates in [[(4, 4)], [(-1, 4)], [(0, 0)], [(1, True)]]:
            with pytest.raises(ValueError, match="seed coordinate"):
                derive_distinct_seed(1, coordinates)


def test_dtensor_rng_initialization_disables_global_sync_without_reseeding(monkeypatch):
    from types import SimpleNamespace

    from torch.distributed.tensor import _random

    from llmtuner.utils.seeding import initialize_dtensor_rng

    calls = []
    mesh = SimpleNamespace(device_type="cuda")
    tracker = object()
    monkeypatch.setattr(_random, "_rng_tracker", None)
    monkeypatch.setattr(_random, "is_rng_supported_mesh", lambda mesh: True)

    def create(mesh, *, run_state_sync):
        calls.append((mesh, run_state_sync))
        return tracker

    monkeypatch.setattr(_random, "OffsetBasedRNGTracker", create)
    before = torch.get_rng_state().clone()
    initialize_dtensor_rng(mesh)
    initialize_dtensor_rng(mesh)
    assert calls == [(mesh, False)]
    assert _random._rng_tracker is tracker
    assert torch.equal(torch.get_rng_state(), before)


def test_dtensor_rng_initialization_skips_cpu_and_missing_mesh(monkeypatch):
    from types import SimpleNamespace

    from torch.distributed.tensor import _random

    from llmtuner.utils.seeding import initialize_dtensor_rng

    def unexpected(*args, **kwargs):
        pytest.fail("CPU does not support the DTensor RNG tracker")

    monkeypatch.setattr(_random, "OffsetBasedRNGTracker", unexpected)
    initialize_dtensor_rng(None)
    initialize_dtensor_rng(SimpleNamespace(device_type="cpu"))
