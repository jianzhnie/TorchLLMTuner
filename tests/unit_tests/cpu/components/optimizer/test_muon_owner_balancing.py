"""Region ownership balances real Muon work deterministically."""

import torch

from llmtuner.components.optimizer.flex_shard._optimizer_reshard_schedule import (
    _RedistributionGroup,
    _TensorRegion,
)
from llmtuner.components.optimizer.flex_shard.dist_muon import (
    _assign_balanced_compute_regions,
    _ParameterComputeLayout,
    _RedistributionTransition,
)
from llmtuner.components.optimizer.flex_shard.optimizer_reshard import Owned


def _assign(names, *, cumulative=(900, 0), holders=(2, 7), ns_steps=(1,)):
    layouts = tuple(
        _ParameterComputeLayout(
            fqn=name,
            param=torch.empty(2, 2),
            group_index=0,
            storage_mesh_ranks=(2, 7),
            storage_layout_signature=(),
            local_storage_signature=(),
            compute_sharding=Owned(),
            storage_to_compute_transition=_RedistributionTransition(),
            resolved_compute_layout_signature=(),
            redistribution_storage_mesh_axis=None,
            compute_dtype=torch.float32,
        )
        for name in names
    )
    domain = ((2, 2), ((holders, _TensorRegion((0, 0), (2, 2))),))
    return _assign_balanced_compute_regions(
        layouts,
        (domain,) * len(layouts),
        group=_RedistributionGroup(
            process_group=None, participants=(2, 7), local_participant=2
        ),
        cumulative_loads=cumulative,
        ns_steps_by_group=ns_steps,
    )


def test_owner_balancing_preserves_empty_input_and_cumulative_load():
    assert _assign([]) == ((), (900, 0))


def test_owner_balancing_uses_stable_keys_independent_of_input_order():
    for names in (("b", "a", "c"), ("c", "a", "b")):
        assignments, cumulative = _assign(names)
        assert dict(
            zip(names, (regions[0][0] for regions in assignments), strict=True)
        ) == {
            "a": 7,
            "b": 2,
            "c": 7,
        }
        assert cumulative == (924, 48)
        assert all(
            regions[0][1] == _TensorRegion((0, 0), (2, 2)) for regions in assignments
        )


def test_current_bucket_load_takes_priority_over_storage_locality():
    assignments, cumulative = _assign(("a", "b"), holders=(2,))
    assert tuple(regions[0][0] for regions in assignments) == (2, 7)
    assert cumulative == (924, 24)


def test_storage_locality_takes_priority_over_earlier_bucket_load():
    assignments, cumulative = _assign(("a",), holders=(2,))
    assert assignments == (((2, _TensorRegion((0, 0), (2, 2))),),)
    assert cumulative == (924, 0)


def test_owner_balancing_cost_accounts_for_ns_steps():
    _, cumulative = _assign(("a", "b"), cumulative=(0, 0), ns_steps=(3,))
    assert cumulative == (72, 72)
