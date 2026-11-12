"""PP partitioning preserves model-owned block IDs and parameter namespaces."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.parallel.pipeline_parallel import partition
from llmtuner.parallel.pipeline_parallel.partition import split_model_into_stages


class Decoder(nn.Module):
    """Minimal explicit decoder with already noncontiguous global block IDs."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for block_id in ("3", "7"):
            self.layers.add_module(block_id, nn.Linear(4, 4))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_split_preserves_noncontiguous_block_ids_and_updates(tmp_path) -> None:
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(
            "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
        )
    try:
        if dist.get_world_size() != 1:
            pytest.skip("this local stage-chain test requires a single rank")
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("pp",))
        model = Decoder()
        _, parts = split_model_into_stages(
            model,
            mesh,
            "1F1B",
            torch.device("cpu"),
            [["layers.3"], ["layers.7"]],
        )
        assert list(parts[0].layers._modules) == ["3"]
        assert list(parts[1].layers._modules) == ["7"]
        original = dict(model.named_parameters())
        partitioned = dict(
            (name, parameter)
            for part in parts
            for name, parameter in part.named_parameters()
        )
        assert partitioned.keys() == original.keys()
        inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
        expected = model(inputs)
        actual = parts[1](parts[0](inputs))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for name, parameter in partitioned.items():
            assert parameter.grad.count_nonzero() > 0
            torch.testing.assert_close(
                parameter.grad, original[name].grad, rtol=0, atol=0
            )
        for parameters in (model.parameters(), partitioned.values()):
            torch.optim.SGD(parameters, lr=0.01, foreach=False).step()
        for name, parameter in partitioned.items():
            torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
    finally:
        if created:
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "targets",
    [
        [["layers.3"], ["layers.3"]],
        [["layers"], ["layers.7"]],
        [["layers.3", "layers.3"], ["layers.7"]],
        [["layers.3"], ["layers"]],
    ],
)
def test_overlapping_ownership_rejects_before_stage_construction(monkeypatch, targets):
    class Mesh:
        def get_local_rank(self):
            return 0

        def size(self):
            return 1

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid ownership reached stage construction")

    model = Decoder()
    # No parameters: ownership cannot depend solely on shared Parameter checks.
    model.layers._modules["3"] = nn.Identity()
    model.layers._modules["7"] = nn.Identity()
    monkeypatch.setattr(partition, "build_stage_from_modules", forbidden)
    rng = torch.get_rng_state().clone()
    blocks = dict(model.layers._modules)
    with pytest.raises(ValueError, match="overlap"):
        split_model_into_stages(model, Mesh(), "1F1B", torch.device("cpu"), targets)
    assert torch.equal(torch.get_rng_state(), rng)
    assert dict(model.layers._modules) == blocks


@pytest.mark.parametrize("alias", ["module", "buffer"])
def test_shared_resources_reject_before_copy_or_stage_creation(monkeypatch, alias):
    class Mesh:
        def get_local_rank(self):
            return 0

        def size(self):
            return 1

    def forbidden(*args, **kwargs):
        raise AssertionError("shared resource reached model copy")

    model = Decoder()
    first = nn.Module()
    first.register_buffer("history", torch.arange(4, dtype=torch.float32))
    second = nn.Module()
    second.register_buffer("history", first.history)
    model.layers._modules["3"] = first
    model.layers._modules["7"] = first if alias == "module" else second
    monkeypatch.setattr(partition.copy, "deepcopy", forbidden)
    before = first.history.clone()
    with pytest.raises(UnsupportedCombinationError, match=f"PP shared {alias}"):
        split_model_into_stages(
            model,
            Mesh(),
            "1F1B",
            torch.device("cpu"),
            [["layers.3"], ["layers.7"]],
        )
    torch.testing.assert_close(first.history, before, atol=0, rtol=0)
    assert model.layers._modules["3"] is first
    assert model.layers._modules["7"] is (first if alias == "module" else second)


def test_same_stage_module_and_buffer_aliases_remain_shared(monkeypatch):
    class Mesh:
        def get_local_rank(self):
            return 0

        def size(self):
            return 1

        def get_group(self):
            return None

    model = Decoder()
    shared = nn.Identity()
    shared.register_buffer("history", torch.arange(4, dtype=torch.float32))
    model.layers._modules["3"] = shared
    model.layers._modules["7"] = shared
    monkeypatch.setattr(partition, "ComposablePipelineStage", lambda *a, **kw: object())
    _, parts = split_model_into_stages(
        model, Mesh(), "1F1B", torch.device("cpu"), [["layers.3", "layers.7"]]
    )
    first = parts[0].layers._modules["3"]
    second = parts[0].layers._modules["7"]
    assert first is second
    assert first.history is second.history
    assert first is not shared
    assert first.history.data_ptr() != shared.history.data_ptr()


@pytest.mark.parametrize("state", ["parameter", "persistent", "duplicate"])
def test_runtime_replication_rejects_trainable_state_and_duplicate_targets(
    monkeypatch, state
):
    class Mesh:
        def get_local_rank(self):
            return 0

        def size(self):
            return 1

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid replica reached model copy")

    model = Decoder()
    model.runtime = nn.Module()
    if state == "parameter":
        model.runtime.register_parameter("weight", nn.Parameter(torch.ones(4)))
    else:
        model.runtime.register_buffer(
            "cache", torch.arange(4), persistent=state == "persistent"
        )
    targets = [["runtime"], ["runtime"]]
    if state == "duplicate":
        targets[0].append("runtime")
    monkeypatch.setattr(partition.copy, "deepcopy", forbidden)
    with pytest.raises(ValueError, match="persistent state|overlap"):
        split_model_into_stages(
            model,
            Mesh(),
            "1F1B",
            torch.device("cpu"),
            targets,
            replicated_module_names=("runtime",),
        )


def test_runtime_replication_creates_independent_nonpersistent_buffers(monkeypatch):
    class Mesh:
        def get_local_rank(self):
            return 0

        def size(self):
            return 1

        def get_group(self):
            return None

    model = Decoder()
    model.runtime = nn.Module()
    model.runtime.register_buffer("cache", torch.arange(4), persistent=False)
    monkeypatch.setattr(partition, "ComposablePipelineStage", lambda *a, **kw: object())
    _, parts = split_model_into_stages(
        model,
        Mesh(),
        "1F1B",
        torch.device("cpu"),
        [["layers.3", "runtime"], ["layers.7", "runtime"]],
        replicated_module_names=("runtime",),
    )
    for part in parts:
        assert "runtime.cache" not in part.state_dict()
        torch.testing.assert_close(part.runtime.cache, model.runtime.cache)
        assert part.runtime.cache.data_ptr() != model.runtime.cache.data_ptr()
    assert parts[0].runtime.cache.data_ptr() != parts[1].runtime.cache.data_ptr()
