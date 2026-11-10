import pytest
import torch

from llmtuner.parallel.fully_shard_parallel.expert_reduction_layout import (
    SplitExpertReduceScatter,
    reorder_expert_region,
)


@pytest.mark.parametrize("experts", [0, 1, 3])
def test_split_layout_matches_independent_projection_concatenation(experts):
    gate = torch.arange(experts * 6).reshape(experts, 2, 3)
    up = gate + 100
    down = gate.transpose(1, 2).contiguous() + 200
    packed = torch.cat((gate, up), dim=1)
    source = torch.cat((packed.flatten(), down.flatten()))
    split = torch.empty_like(source)
    reorder_expert_region(
        source, split, experts=experts, width=2, hidden=3, inverse=False
    )
    assert torch.equal(split, torch.cat((gate.flatten(), down.flatten(), up.flatten())))
    restored = torch.empty_like(source)
    reorder_expert_region(
        split, restored, experts=experts, width=2, hidden=3, inverse=True
    )
    assert torch.equal(restored, source)


def test_communication_preserves_prefix_suffix_and_storage(monkeypatch):
    from contextlib import nullcontext

    from llmtuner.accelerator import collectives

    monkeypatch.setattr(
        collectives, "synchronize_local_errors", lambda *a, **kw: nullcontext()
    )

    class Group:
        def size(self):
            return 2

    class Communication:
        def __call__(self, output, inputs, group, op, async_op):
            output.copy_(inputs.view(2, -1).sum(0))

    source = torch.arange(80, dtype=torch.float32)
    original = source.clone()
    output = torch.empty(40)
    adapter = SplitExpertReduceScatter(
        Communication(),
        world_size=2,
        offset=2,
        experts=1,
        width=2,
        hidden=3,
        workspace_limit_bytes=1024,
    )
    adapter(output, source, Group(), "sum")
    assert torch.equal(output, original.view(2, -1).sum(0))
    assert torch.equal(source, original)
    with pytest.raises(ValueError, match="synchronous"):
        adapter(output, source, Group(), "sum", async_op=True)
    with pytest.raises(ValueError, match="distinct"):
        adapter(source[:40], source, Group(), "sum")


def test_installation_validates_every_unit_before_mutation(monkeypatch):
    from types import SimpleNamespace

    from torch import nn

    from llmtuner.parallel.fully_shard_parallel import expert_reduction_layout as layout
    from llmtuner.parallel.fully_shard_parallel import policies

    parameters = [
        nn.Parameter(torch.empty(8, 4, 3)),
        nn.Parameter(torch.empty(8, 3, 2)),
    ]
    mesh = SimpleNamespace(shard_mesh_size=2, replicate_mesh_dim=None)
    records = [
        SimpleNamespace(
            sharded_param=p,
            _orig_size=p.shape,
            padded_sharded_param_size=(p.shape[0] // 2, *p.shape[1:]),
            mesh_info=mesh,
            fsdp_placement=SimpleNamespace(dim=0),
        )
        for p in parameters
    ]
    group = SimpleNamespace(fsdp_params=records, _reduce_scatter_comm=object())
    installed = []

    class Module:
        def __init__(self, groups):
            self.state = SimpleNamespace(_fsdp_param_groups=groups)

        def _get_fsdp_state(self):
            return self.state

        def set_reduce_scatter_unused_params(self, value):
            assert value is True

        def set_custom_reduce_scatter(self, comm):
            installed.append(comm)

    good = Module([group])
    bad = Module([group, group])
    model = SimpleNamespace(
        named_parameters=lambda: zip(
            ("layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.experts.down_proj"),
            parameters,
            strict=True,
        )
    )
    monkeypatch.setattr(policies, "iter_fsdp_modules", lambda model: iter([good, bad]))
    with pytest.raises(ValueError, match="one FSDP mesh"):
        layout.install_split_expert_reduction(model, workspace_limit_bytes=1024)
    assert not installed
    monkeypatch.setattr(policies, "iter_fsdp_modules", lambda model: iter([good]))
    layout.install_split_expert_reduction(model, workspace_limit_bytes=1024)
    assert len(installed) == 1
    assert records[0].sharded_param is parameters[0]
    group._reduce_scatter_comm = installed[0]
    with pytest.raises(ValueError, match="already installed"):
        layout.install_split_expert_reduction(model, workspace_limit_bytes=1024)
    assert len(installed) == 1
    group._reduce_scatter_comm = object()
    mesh.replicate_mesh_dim = 0
    with pytest.raises(ValueError, match="HSDP"):
        layout.install_split_expert_reduction(model, workspace_limit_bytes=1024)
    assert len(installed) == 1


@pytest.mark.parametrize(
    "field,value", [("world_size", True), ("offset", 1.0), ("experts", -1)]
)
def test_geometry_requires_strict_valid_integers(field, value):
    values = dict(
        world_size=2, offset=0, experts=1, width=2, hidden=3, workspace_limit_bytes=1024
    )
    values[field] = value
    with pytest.raises(ValueError):
        SplitExpertReduceScatter(object(), **values)


def test_missing_split_workspace_budget_rejected():
    from llmtuner.config.parallel_config import FSDPConfig

    with pytest.raises(ValueError, match="budget"):
        FSDPConfig(fsdp_expert_reduce_layout="split")
    assert FSDPConfig().fsdp_expert_reduce_layout == "storage"


def test_split_entry_rejects_missing_or_single_rank_topology(monkeypatch):
    from types import SimpleNamespace

    from torch import nn

    from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
    from llmtuner.errors import UnsupportedCombinationError
    from llmtuner.parallel.fully_shard_parallel import fully_shard_parallel as entry

    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_expert_reduce_layout="split",
            fsdp_expert_reduce_workspace_limit_bytes=1024,
        )
    )
    model = nn.Linear(2, 2)
    with pytest.raises(UnsupportedCombinationError, match="topology"):
        entry.apply_fsdp(model, cfg, None)
    dims = SimpleNamespace(
        **{
            key: False
            for key in (
                "ep_enabled",
                "tp_enabled",
                "cp_enabled",
                "pp_enabled",
                "dp_replicate_enabled",
            )
        }
    )
    monkeypatch.setattr(
        entry, "resolve_fsdp_mesh", lambda dims: SimpleNamespace(size=lambda: 1)
    )
    monkeypatch.setattr(entry, "resolve_sparse_fsdp_mesh", lambda dims: None)
    with pytest.raises(UnsupportedCombinationError, match="more than one"):
        entry.apply_fsdp(model, cfg, dims)
