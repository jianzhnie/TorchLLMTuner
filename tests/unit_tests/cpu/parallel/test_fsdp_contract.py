"""The FSDP contract: what the sharding layer reads off a model, and how.

``fsdp.py`` is written against torchtitan's ``Decoder`` shape -- a ``ModuleDict``
of layers plus four named modules on the top-level object. llmtuner trains HF
models, whose layout differs on both counts. These tests pin the two places that
reconcile them, since a break in either is invisible until either:

* the model silently fails to shard (an adapter that stops taking effect), or
* FSDP raises partway through a distributed run (a name that stopped resolving).

They run against the real ``HFTransformerModel`` over a tiny offline LLaMA --
now that there is only one wrapper, the contract is worth pinning on the object
FSDP actually receives rather than on a stand-in that could drift from it.

No process group is needed: the layer iterator is pure container logic, and the
accessors are plain attribute lookups on a tiny offline model.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("flex_attention", "spmd_types")


import pytest
import torch
import torch.nn as nn
from torch.nn import ModuleDict, ModuleList

import llmtuner.parallel.fully_shard_parallel.fully_shard_parallel as apply
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.models.hf.factory import build_model_config
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.fully_shard_parallel.fine_grained import (
    CustomFSDPStrategy,
    NativeFSDPStrategy,
    fully_shard_fine_grained,
    get_fsdp_strategy,
)
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    iter_transformer_layers,
)
from llmtuner.parallel.fully_shard_parallel.plan import set_fsdp_prefetch_windows

_VOCAB = 32


@pytest.mark.parametrize("implementation", ["custon", "", None])
def test_decoder_fsdp_rejects_unknown_implementation_before_model_access(
    monkeypatch, implementation
):
    monkeypatch.setattr(
        apply,
        "fully_shard",
        lambda *args, **kwargs: pytest.fail("invalid backend reached model sharding"),
    )
    with pytest.raises(ValueError, match="unknown FSDP implementation"):
        apply.apply_fsdp_to_decoder(
            None,
            None,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            fsdp_implementation=implementation,
        )


_HIDDEN = 8
_NUM_LAYERS = 3


def test_hsdp_cp_mesh_reuses_existing_groups(monkeypatch):
    from types import SimpleNamespace

    replicate_group = object()
    shard_group = object()
    selections = []

    class Submesh:
        mesh = torch.arange(4).reshape(2, 1, 2)
        device_type = "cpu"

        def __getitem__(self, axes):
            selections.append(axes)
            if axes == "dp_replicate":
                return SimpleNamespace(get_group=lambda: replicate_group)
            assert axes == ("dp_shard", "cp")

            def flatten(name):
                assert name == "dp_shard_cp"
                return SimpleNamespace(get_group=lambda: shard_group)

            return SimpleNamespace(_flatten=flatten)

    def optional_mesh(axes):
        assert axes == ["dp_replicate", "dp_shard", "cp"]
        return Submesh()

    captured = {}

    def from_group(groups, device_type, *, mesh, mesh_dim_names):
        captured.update(groups=groups, device_type=device_type, names=mesh_dim_names)
        torch.testing.assert_close(mesh, torch.arange(4).reshape(2, 2))
        return "reused_mesh"

    monkeypatch.setattr(apply.DeviceMesh, "from_group", staticmethod(from_group))
    topology = SimpleNamespace(
        cp_enabled=True,
        dp_replicate_enabled=True,
        get_optional_mesh=optional_mesh,
    )
    assert apply.resolve_fsdp_mesh(topology) == "reused_mesh"
    assert captured == {
        "groups": [replicate_group, shard_group],
        "device_type": "cpu",
        "names": ("dp_replicate", "dp_shard_cp"),
    }
    assert selections == [("dp_shard", "cp"), "dp_replicate"]


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("first_stage", [False, True])
def test_decoder_fsdp_excludes_pipeline_identity_boundaries(
    monkeypatch, tied, first_stage
):
    from types import SimpleNamespace

    model = nn.Module()
    model.enable_weight_tying = tied
    model.tok_embeddings = nn.Embedding(8, 4) if first_stage else nn.Identity()
    model.norm = nn.Identity() if first_stage else nn.LayerNorm(4)
    model.lm_head = nn.Identity() if first_stage else nn.Linear(4, 8)
    model.layers = nn.ModuleList([nn.Linear(4, 4)])
    calls = []
    monkeypatch.setattr(
        apply, "fully_shard", lambda modules, **kwargs: calls.append(modules)
    )
    monkeypatch.setattr(apply, "enable_fsdp_symm_mem", lambda *args: None)
    monkeypatch.setattr(apply, "disable_fsdp_gradient_division", lambda *args: None)
    apply.apply_fsdp_to_decoder(
        model,
        SimpleNamespace(mesh_dim_names=("dp_shard",)),
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        pp_enabled=True,
    )
    boundary = calls[0] if isinstance(calls[0], list) else [calls[0]]
    expected = [model.tok_embeddings] if first_stage else [model.norm, model.lm_head]
    assert boundary == expected
    assert calls[1:] == [model.layers[0], model]
    assert all(type(module) is not nn.Identity for module in boundary)


def _wrapper(*, tied: bool = False) -> HFTransformerModel:
    config = build_model_config(
        "llama",
        seq_len=32,
        arch_overrides={
            "vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "intermediate_size": 16,
            "num_hidden_layers": _NUM_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "tie_word_embeddings": tied,
        },
    )
    return HFTransformerModel(config)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("shared_layer", [False, True])
def test_chunked_boundaries_and_actual_aliases_are_owned_by_root(
    monkeypatch, tied, shared_layer
):
    from types import SimpleNamespace

    model = _wrapper(tied=tied)
    model.model.config.tie_word_embeddings = not tied
    if shared_layer:
        model.norm.weight = model.layers[0].input_layernorm.weight
    calls = []
    monkeypatch.setattr(
        apply, "fully_shard", lambda module, **options: calls.append((module, options))
    )
    monkeypatch.setattr(apply, "disable_fsdp_gradient_division", lambda *args: None)
    apply.apply_fsdp_to_decoder(
        model,
        SimpleNamespace(mesh_dim_names=("dp_shard",)),
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        root_owned_boundaries=True,
    )
    expected_layers = list(model.layers)[1:] if shared_layer else list(model.layers)
    assert [module for module, _ in calls] == [*expected_layers, model]
    assert calls[-1][1]["reshard_after_forward"] is False
    assert model.enable_weight_tying == tied


# -- container layout --------------------------------------------------------


def test_iter_transformer_layers_accepts_a_hf_module_list() -> None:
    layers = ModuleList([nn.Linear(4, 4) for _ in range(3)])

    assert [idx for idx, _ in iter_transformer_layers(layers)] == [0, 1, 2]


def test_iter_transformer_layers_accepts_a_torchtitan_module_dict() -> None:
    layers = ModuleDict({str(i): nn.Linear(4, 4) for i in range(3)})

    assert [idx for idx, _ in iter_transformer_layers(layers)] == ["0", "1", "2"]


def test_iter_transformer_layers_yields_blocks_in_order() -> None:
    layers = ModuleList([nn.Linear(4, 4) for _ in range(3)])
    as_dict = ModuleDict({str(i): m for i, m in enumerate(layers)})

    from_list = [block for _, block in iter_transformer_layers(layers)]
    from_dict = [block for _, block in iter_transformer_layers(as_dict)]

    assert all(a is b for a, b in zip(from_list, from_dict, strict=True))


# -- accessors ---------------------------------------------------------------


def test_decoder_parts_resolve_off_the_top_level_wrapper() -> None:
    """The FSDP layer reads these off the model it is handed, not off model.model."""
    wrapper = _wrapper()
    decoder = wrapper.model.model

    assert wrapper.tok_embeddings is decoder.embed_tokens
    assert wrapper.layers is decoder.layers
    assert wrapper.norm is decoder.norm
    assert wrapper.lm_head is wrapper.model.lm_head


def test_layers_are_a_container_torchtitan_can_iterate() -> None:
    wrapper = _wrapper()

    assert len(list(iter_transformer_layers(wrapper.layers))) == _NUM_LAYERS


def test_untied_head_reports_no_weight_tying() -> None:
    assert _wrapper(tied=False).enable_weight_tying is False


def test_tied_head_is_detected_by_parameter_identity() -> None:
    """FSDP2's own check: the two modules own one Parameter object."""
    wrapper = _wrapper(tied=True)

    assert wrapper.enable_weight_tying is True
    assert wrapper.tok_embeddings.weight is wrapper.lm_head.weight


def test_missing_head_reports_no_weight_tying() -> None:
    """A model whose head was stripped must not be grouped with the embedding."""
    wrapper = _wrapper(tied=True)
    wrapper.model.lm_head = None

    assert wrapper.enable_weight_tying is False


def test_tie_word_embeddings_flag_is_not_what_decides() -> None:
    """Intent and reality can disagree; identity is what FSDP acts on."""
    wrapper = _wrapper(tied=True)
    wrapper.model.config = type("C", (), {"tie_word_embeddings": False})()

    assert wrapper.enable_weight_tying is True


@pytest.mark.parametrize("tied", [False, True])
def test_accessors_do_not_add_duplicate_parameters(tied: bool) -> None:
    """The accessors must alias, not re-register, the underlying modules.

    Assigning ``self.x = module`` inside the wrapper would register a second
    copy, putting the same tensors in the state dict twice -- once under the
    HF name and once under the torchtitan name. These are properties, so the
    parameter count must equal the HF model's own.
    """
    wrapper = _wrapper(tied=tied)

    assert len(list(wrapper.parameters())) == len(list(wrapper.model.parameters()))


@pytest.mark.parametrize("reduce_dtype", ["float32", "bfloat16", "float64"])
def test_apply_fsdp_preserves_parameter_dtype_for_mixed_precision(
    monkeypatch: pytest.MonkeyPatch,
    reduce_dtype: str,
) -> None:
    """BF16 construction must not be silently widened by the FSDP policy."""

    class _Mesh:
        def size(self) -> int:
            return 2

    captured: dict[str, object] = {}
    monkeypatch.setattr(apply, "resolve_fsdp_mesh", lambda _dims: _Mesh())
    monkeypatch.setattr(apply, "resolve_sparse_fsdp_mesh", lambda _dims: None)
    monkeypatch.setattr(
        apply,
        "apply_fsdp_to_decoder",
        lambda _model, _mesh, **kwargs: captured.update(kwargs),
    )

    model = nn.Linear(4, 4, dtype=torch.bfloat16)
    parallel_dims = type(
        "ParallelTopologyStub",
        (),
        {"pp_enabled": False, "ep": 1},
    )()
    config = ParallelConfig()
    config.fsdp_config.fsdp_reduce_dtype = reduce_dtype

    apply.apply_fsdp(model, config, parallel_dims)

    assert captured["param_dtype"] is torch.bfloat16
    assert captured["reduce_dtype"] is getattr(torch, reduce_dtype)


def test_apply_fsdp_forwards_cpu_offload_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user-facing offload knobs must reach the FSDP2 policy builder."""

    class _Mesh:
        def size(self) -> int:
            return 2

    captured: dict[str, object] = {}
    monkeypatch.setattr(apply, "resolve_fsdp_mesh", lambda _dims: _Mesh())
    monkeypatch.setattr(apply, "resolve_sparse_fsdp_mesh", lambda _dims: None)
    monkeypatch.setattr(
        apply,
        "apply_fsdp_to_decoder",
        lambda _model, _mesh, **kwargs: captured.update(kwargs),
    )

    model = nn.Linear(4, 4)
    parallel_dims = type("ParallelTopologyStub", (), {"pp_enabled": False, "ep": 1})()
    config = ParallelConfig(
        fsdp_config=FSDPConfig(fsdp_cpu_offload=True, fsdp_cpu_offload_pin_memory=False)
    )

    apply.apply_fsdp(model, config, parallel_dims)

    assert captured["cpu_offload"] is True
    assert captured["cpu_offload_pin_memory"] is False


def test_fsdp_prefetch_supports_arbitrary_windows() -> None:
    class Unit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_targets: list[nn.Module] = []
            self.backward_targets: list[nn.Module] = []

        def set_modules_to_forward_prefetch(self, modules) -> None:
            self.forward_targets = list(modules)

        def set_modules_to_backward_prefetch(self, modules) -> None:
            self.backward_targets = list(modules)

    units = [Unit() for _ in range(5)]
    set_fsdp_prefetch_windows(units, forward_count=2, backward_count=3)

    assert units[0].forward_targets == units[1:3]
    assert units[2].forward_targets == units[3:5]
    assert units[4].backward_targets == [units[3], units[2], units[1]]

    # Reconfiguration must clear old windows, including a unit moved to the
    # boundary where it has no neighbour in that direction.
    set_fsdp_prefetch_windows(list(reversed(units)), forward_count=1, backward_count=1)
    assert units[0].forward_targets == []
    assert units[4].backward_targets == []
    set_fsdp_prefetch_windows(units, forward_count=0, backward_count=0)
    assert all(unit.forward_targets == [] for unit in units)
    assert all(unit.backward_targets == [] for unit in units)


@pytest.mark.parametrize("field", ["forward_count", "backward_count"])
@pytest.mark.parametrize("value", [True, 1.5, float("nan"), "2", None, -1])
def test_fsdp_prefetch_rejects_invalid_counts_before_mutation(field, value) -> None:
    class Unit(nn.Module):
        def set_modules_to_forward_prefetch(self, modules) -> None:
            pytest.fail("invalid counts must be rejected before configuring units")

        def set_modules_to_backward_prefetch(self, modules) -> None:
            pytest.fail("invalid counts must be rejected before configuring units")

    counts = {"forward_count": 1, "backward_count": 1, field: value}
    with pytest.raises(ValueError, match=field):
        set_fsdp_prefetch_windows([Unit(), Unit()], **counts)


def test_fsdp_prefetch_rejects_missing_interface_before_mutation():
    class Unit(nn.Module):
        def set_modules_to_forward_prefetch(self, modules):
            pytest.fail("invalid plan modified a preceding unit")

        def set_modules_to_backward_prefetch(self, modules):
            pytest.fail("invalid plan modified a preceding unit")

    with pytest.raises(TypeError, match="prefetch unit 1 lacks callable"):
        set_fsdp_prefetch_windows(
            [Unit(), nn.Identity()], forward_count=1, backward_count=1
        )


def test_fsdp_prefetch_rejects_duplicate_units_before_mutation():
    class Unit(nn.Module):
        def set_modules_to_forward_prefetch(self, modules):
            pytest.fail("duplicate units must fail before configuring windows")

        def set_modules_to_backward_prefetch(self, modules):
            pytest.fail("duplicate units must fail before configuring windows")

    unit = Unit()
    with pytest.raises(ValueError, match="distinct module objects"):
        set_fsdp_prefetch_windows([unit, unit], forward_count=1, backward_count=1)


def test_fsdp_strategy_selects_pyTorch_214_compatible_adapter() -> None:
    assert isinstance(get_fsdp_strategy("native"), NativeFSDPStrategy)
    assert isinstance(get_fsdp_strategy("custom"), CustomFSDPStrategy)


def test_native_strategy_does_not_load_the_custom_private_engine():
    import subprocess
    import sys
    from textwrap import dedent

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent(
                """
                import importlib.abc
                import sys

                import torch.nn as nn
                from torch.distributed.fsdp import fully_shard

                class MissingCustomEngine(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname.endswith('.fine_grained_state'):
                            raise ImportError('missing custom private API')

                sys.meta_path.insert(0, MissingCustomEngine())
                from llmtuner.errors import EnvironmentUnsupportedError
                from llmtuner.parallel.fully_shard_parallel.fine_grained import (
                    get_fsdp_strategy,
                )

                assert get_fsdp_strategy('native').get_fully_shard_fn() is fully_shard
                assert not any(x.endswith('.fine_grained_state') for x in sys.modules)
                custom = get_fsdp_strategy('custom').get_fully_shard_fn()
                try:
                    custom(nn.Linear(2, 2))
                except EnvironmentUnsupportedError as error:
                    assert 'PyTorch 2.14' in str(error)
                    assert "fsdp_implementation='native'" in str(error)
                    assert isinstance(error.__cause__, ImportError)
                else:
                    raise AssertionError('missing custom engine was ignored')
                """
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fine_grained_rejects_an_unrelated_hook_target() -> None:
    with pytest.raises(ValueError, match="hook_module must contain"):
        fully_shard_fine_grained(nn.Linear(2, 2), hook_module=nn.Identity())


def test_custom_contexts_share_streams_without_allocating_unused_streams(monkeypatch):
    require_env("fine_grained_fsdp")
    from types import SimpleNamespace
    from weakref import ref

    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext

    from llmtuner.parallel.fully_shard_parallel.fine_grained_state import (
        FineGrainedFSDPState,
    )

    owner = nn.Identity()
    states = [FineGrainedFSDPState() for _ in range(3)]
    for index, state in enumerate(states):
        state.hook_module = ref(owner)
        state._device = torch.device("cpu")
        state._fsdp_param_groups = [
            SimpleNamespace(reduce_scatter_max_input_buffers=index + 1)
        ]
    states[0]._state_ctx.all_states = states
    calls = []

    def init(context, device):
        calls.append(context)
        context.device_handle = object()
        for name in (
            "all_gather_copy_in_stream",
            "all_gather_stream",
            "reduce_scatter_stream",
            "all_reduce_stream",
        ):
            setattr(context, name, object())
        context.all_gather_state = None
        context.reduce_scatter_states = []
        context._last_post_reduce_events = {}
        context.post_forward_order = []

    monkeypatch.setattr(FSDPCommContext, "lazy_init", init)
    states[0]._init_shared_state()
    contexts = states[0].communication_contexts
    assert len(calls) == 1
    assert len(contexts) == 3
    for index, context in enumerate(contexts):
        assert context.all_gather_stream is contexts[0].all_gather_stream
        assert context.device_handle is contexts[0].device_handle
        assert context.post_forward_order is contexts[0].post_forward_order
        assert context.reduce_scatter_max_input_buffers == index + 1
    assert len({id(c.reduce_scatter_states) for c in contexts}) == 3
    assert len({id(c._last_post_reduce_events) for c in contexts}) == 3


@pytest.mark.parametrize("forward", [False, True])
def test_fine_grained_all_gather_waits_before_releasing_foreign_hook_buffers(
    monkeypatch, forward
):
    require_env("fine_grained_fsdp")
    from types import SimpleNamespace
    from weakref import ref

    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        AllGatherState,
        FSDPParamGroup,
    )

    from llmtuner.parallel.fully_shard_parallel.fine_grained_param_group import (
        FineGrainedFSDPParamGroup,
        HookAllGatherState,
    )

    owner, other_owner = nn.Identity(), nn.Identity()
    event, new_event = object(), object()
    payload = torch.ones(1)
    weak_payload = ref(payload)
    foreign = SimpleNamespace(
        all_gather_state=HookAllGatherState(
            SimpleNamespace(buffer=payload), event, ref(other_owner)
        )
    )
    del payload
    same_state = HookAllGatherState(object(), object(), ref(owner))
    same = SimpleNamespace(all_gather_state=same_state)
    local = SimpleNamespace(all_gather_state=None)
    group = object.__new__(FineGrainedFSDPParamGroup)
    group.hook_module = ref(owner)
    group.comm_ctx = local
    group.communication_contexts = [local, same, foreign]
    group._all_gather_result = object()
    group._training_state = (
        TrainingState.FORWARD if forward else TrainingState.PRE_BACKWARD
    )
    waits = []

    def wait(waited_event):
        assert weak_payload() is not None, "freed the buffer before its event wait"
        waits.append(waited_event)

    def native_wait(self):
        assert (foreign.all_gather_state is None) == forward
        assert same.all_gather_state is same_state
        self.comm_ctx.all_gather_state = AllGatherState(object(), new_event)

    monkeypatch.setattr(group, "_wait_all_gather_streams_on_event", wait)
    monkeypatch.setattr(FSDPParamGroup, "wait_for_unshard", native_wait)
    group.wait_for_unshard()
    assert waits == ([event] if forward else [])
    assert (weak_payload() is None) == forward
    assert local.all_gather_state.hook_module() is owner
    assert local.all_gather_state.event is new_event


@pytest.mark.parametrize("reduce_grads", [False, True])
def test_fine_grained_reduce_scatter_preserves_same_hook_and_tags_new_buffers(
    monkeypatch, reduce_grads
):
    require_env("fine_grained_fsdp")
    from types import SimpleNamespace
    from weakref import ref

    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPParamGroup,
        ReduceScatterState,
    )

    from llmtuner.parallel.fully_shard_parallel.fine_grained_param_group import (
        FineGrainedFSDPParamGroup,
        HookReduceScatterState,
    )

    owner, other_owner = nn.Identity(), nn.Identity()
    event, new_event = object(), object()
    payload = torch.ones(1)
    weak_payload = ref(payload)
    same_state = HookReduceScatterState(torch.ones(1), None, ref(owner))
    foreign = SimpleNamespace(
        reduce_scatter_states=[
            same_state,
            HookReduceScatterState(payload, event, ref(other_owner)),
            HookReduceScatterState(torch.ones(1), None, ref(other_owner)),
        ]
    )
    del payload
    local = SimpleNamespace(reduce_scatter_states=[])
    group = object.__new__(FineGrainedFSDPParamGroup)
    group.hook_module = ref(owner)
    group.comm_ctx = local
    group.communication_contexts = [local, foreign]
    group.reduce_grads = reduce_grads
    waits = []

    def wait(waited_event):
        assert weak_payload() is not None, "freed the buffer before its event wait"
        waits.append(waited_event)

    group.device_handle = SimpleNamespace(
        current_stream=lambda: SimpleNamespace(wait_event=wait)
    )

    def native_reduce(self):
        assert len(foreign.reduce_scatter_states) == (1 if reduce_grads else 3)
        if reduce_grads:
            self.comm_ctx.reduce_scatter_states.append(
                ReduceScatterState(torch.ones(1), new_event)
            )

    monkeypatch.setattr(FSDPParamGroup, "post_backward", native_reduce)
    group.post_backward()
    assert waits == ([event] if reduce_grads else [])
    assert foreign.reduce_scatter_states[0] is same_state
    assert (weak_payload() is None) == reduce_grads
    if reduce_grads:
        assert local.reduce_scatter_states[0].hook_module() is owner
        assert local.reduce_scatter_states[0].event is new_event
    else:
        assert local.reduce_scatter_states == []


@pytest.mark.parametrize("shared_weight", [False, True])
def test_custom_pattern_accepts_an_ancestor_hook_for_all_managed_units(
    monkeypatch, shared_weight
):
    from types import SimpleNamespace

    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    if shared_weight:
        model[1].weight = model[0].weight
    calls = []

    def shard(module, **options):
        calls.append((module, options))

    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda implementation: SimpleNamespace(get_fully_shard_fn=lambda: shard),
    )
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *args, **kwargs: None)
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_implementation="custom",
            fsdp_apply_modules={
                "0": {"hook_module": model},
                "1": {"hook_module": model},
            },
        )
    )
    plan.apply_fsdp_by_pattern(
        model,
        dp_mesh=object(),
        cfg=cfg,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    assert calls[-1][0] is model
    units = calls[:-1]
    assert len(units) == (1 if shared_weight else 2)
    assert all(options["hook_module"] is model for _, options in units)
    if shared_weight:
        assert units[0][0] == [model[0], model[1]]
    else:
        assert {module for module, _ in units} == {model[0], model[1]}


@pytest.mark.parametrize("implementation", ["native", "custom"])
@pytest.mark.parametrize("override", ["hook_module", "reshard_after_foward"])
def test_invalid_pattern_override_rejected_before_any_sharding(
    monkeypatch, implementation, override
):
    from llmtuner.config import ParallelConfig
    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_implementation=implementation,
            fsdp_apply_modules={"0": {}, "1": {override: nn.Identity()}},
        )
    )

    def unexpected(*args, **kwargs):
        pytest.fail("invalid override must be rejected before strategy execution")

    monkeypatch.setattr(plan, "get_fsdp_strategy", unexpected)
    with pytest.raises(ValueError, match=override):
        plan.apply_fsdp_by_pattern(
            model,
            dp_mesh=object(),
            cfg=cfg,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )


@pytest.mark.parametrize("implementation", ["native", "custom"])
@pytest.mark.parametrize("value", ["true", 0, -1, 3, 8])
def test_invalid_pattern_reshard_value_rejected_before_any_sharding(
    monkeypatch, implementation, value
):
    from types import SimpleNamespace

    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    calls = []
    mesh = SimpleNamespace(ndim=2, size=lambda dim: (2, 4)[dim])
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda _: SimpleNamespace(
            get_fully_shard_fn=lambda: lambda *a, **kw: calls.append(a)
        ),
    )
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *a, **kw: None)
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_implementation=implementation,
            fsdp_apply_modules={"0": {}, "1": {"reshard_after_forward": value}},
        )
    )
    with pytest.raises(ValueError, match="reshard_after_forward"):
        plan.apply_fsdp_by_pattern(
            model,
            dp_mesh=mesh,
            cfg=cfg,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )
    assert calls == []


@pytest.mark.parametrize("value", [None, True, False, 1, 2, 4])
def test_pattern_reshard_policy_uses_unit_mesh(monkeypatch, value):
    from types import SimpleNamespace

    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    calls = []
    unit_mesh = SimpleNamespace(ndim=2, size=lambda dim: (2, 4)[dim])
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda _: SimpleNamespace(
            get_fully_shard_fn=lambda: lambda module, **kw: calls.append(kw)
        ),
    )
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *a, **kw: None)
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={
                "0": {},
                "1": {"mesh": unit_mesh, "reshard_after_forward": value},
            }
        )
    )
    plan.apply_fsdp_by_pattern(
        model,
        dp_mesh=object(),
        cfg=cfg,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    assert len(calls) == 3
    assert calls[1]["mesh"] is unit_mesh
    assert calls[1]["reshard_after_forward"] is value


@pytest.mark.parametrize("implementation", ["native", "custom"])
def test_pattern_integer_reshard_with_spmd_rejected_before_sharding(
    monkeypatch, implementation
):
    from types import SimpleNamespace

    from llmtuner.parallel.fully_shard_parallel import plan

    calls = []
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda _: SimpleNamespace(
            get_fully_shard_fn=lambda: lambda *a, **kw: calls.append(a)
        ),
    )
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_implementation=implementation,
            fsdp_apply_modules={
                "0": {},
                "1": {
                    "reshard_after_forward": 1,
                    "dp_mesh_dims": SimpleNamespace(shard=("dp_shard",)),
                },
            },
        )
    )
    with pytest.raises(NotImplementedError, match="dp_mesh_dims"):
        plan.apply_fsdp_by_pattern(
            nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
            dp_mesh=object(),
            cfg=cfg,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )
    assert calls == []


@pytest.mark.parametrize("implementation", ["native", "custom"])
@pytest.mark.parametrize(
    "override,value",
    [
        ("mp_policy", object()),
        ("offload_policy", object()),
        ("shard_placement_fn", 1),
        ("ignored_params", {"weight"}),
    ],
)
def test_invalid_pattern_policy_rejected_before_sharding(
    monkeypatch, implementation, override, value
):
    from types import SimpleNamespace

    from llmtuner.parallel.fully_shard_parallel import plan

    calls = []
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda _: SimpleNamespace(
            get_fully_shard_fn=lambda: lambda *a, **kw: calls.append(a)
        ),
    )
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *a, **kw: None)
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_implementation=implementation,
            fsdp_apply_modules={"0": {}, "1": {override: value}},
        )
    )
    with pytest.raises(TypeError, match=override):
        plan.apply_fsdp_by_pattern(
            nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
            dp_mesh=object(),
            cfg=cfg,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )
    assert calls == []


@pytest.mark.parametrize("policy", ["mp_policy", "offload_policy"])
@pytest.mark.parametrize("implementation", ["native", "custom"])
def test_pattern_none_policy_follows_engine_contract(
    monkeypatch, policy, implementation
):
    from types import SimpleNamespace

    from llmtuner.parallel.fully_shard_parallel import plan

    calls = []
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda _: SimpleNamespace(
            get_fully_shard_fn=lambda: lambda module, **kw: calls.append(kw)
        ),
    )
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *a, **kw: None)
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_implementation=implementation, fsdp_apply_modules={"0": {policy: None}}
        )
    )

    def apply_pattern():
        plan.apply_fsdp_by_pattern(
            nn.Sequential(nn.Linear(2, 2)),
            dp_mesh=object(),
            cfg=cfg,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )

    if implementation == "native":
        with pytest.raises(TypeError, match=policy):
            apply_pattern()
        assert calls == []
    else:
        apply_pattern()
        assert calls[0][policy] is None


@pytest.mark.parametrize("container", [nn.ModuleList, nn.ModuleDict])
def test_pattern_container_rejected_before_any_sharding(monkeypatch, container):
    from llmtuner.config import ParallelConfig
    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(
        nn.Linear(2, 2),
        container([nn.Linear(2, 2)])
        if container is nn.ModuleList
        else container({"linear": nn.Linear(2, 2)}),
    )
    cfg = ParallelConfig(fsdp_config=FSDPConfig(fsdp_apply_modules={"*": {}}))

    def unexpected(*args, **kwargs):
        pytest.fail("container selection must fail before strategy execution")

    monkeypatch.setattr(plan, "get_fsdp_strategy", unexpected)
    with pytest.raises(ValueError, match="container without forward"):
        plan.apply_fsdp_by_pattern(
            model,
            dp_mesh=object(),
            cfg=cfg,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )


def test_fsdp_pattern_overrides_replace_common_policy(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.config import ParallelConfig
    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2))
    calls = []
    override_policy = apply.MixedPrecisionPolicy(param_dtype=torch.float32)
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *args, **kwargs: None)
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(fsdp_apply_modules={"0": {"mp_policy": override_policy}})
    )
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda implementation: SimpleNamespace(
            get_fully_shard_fn=lambda: (
                lambda module, **kwargs: calls.append((module, kwargs))
            )
        ),
    )
    monkeypatch.setattr(apply, "disable_fsdp_gradient_division", lambda model: None)
    plan.apply_fsdp_by_pattern(
        model,
        dp_mesh=object(),
        cfg=cfg,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    assert calls[0][0] is model[0]
    assert calls[0][1]["mp_policy"] is override_policy
    assert calls[1][0] is model
    assert calls[1][1]["mp_policy"] is not override_policy


def test_pipeline_patterns_rebind_copied_hooks_and_excluded_parameters():
    import copy

    from llmtuner.parallel.fully_shard_parallel.plan import (
        resolve_pipeline_fsdp_configs,
    )

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    first, second = copy.deepcopy(model), copy.deepcopy(model)
    first._modules.pop("1")
    second._modules.pop("0")
    mesh = object()
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_implementation="custom",
            fsdp_apply_modules={
                "0": {
                    "hook_module": model,
                    "mesh": mesh,
                    "ignored_params": {model[0].bias},
                }
            },
        )
    )
    configs = resolve_pipeline_fsdp_configs(model, [first, second], cfg)
    options = configs[0].fsdp_config.fsdp_apply_modules["0"]
    assert options["hook_module"] is first
    assert options["mesh"] is mesh
    assert options["ignored_params"] == {first[0].bias}
    assert configs[1].fsdp_config.fsdp_apply_modules == {"": {}}
    from llmtuner.parallel.fully_shard_parallel.plan import _resolve_pattern_units

    _, units, _ = _resolve_pattern_units(second, configs[1])
    assert units == {"": {}}
    assert cfg.fsdp_config.fsdp_apply_modules["0"]["hook_module"] is model


@pytest.mark.parametrize("recompute", [False, True])
def test_custom_fsdp_recompute_enters_backward_lifecycle(monkeypatch, recompute):
    require_env("fine_grained_fsdp")
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    from llmtuner.models.common.aux_loss import in_checkpoint_recompute
    from llmtuner.parallel.fully_shard_parallel.fine_grained_state import (
        FineGrainedFSDPState,
    )

    state = FineGrainedFSDPState()
    state._training_state = TrainingState.IDLE
    seen = []

    def pre_forward(self, module, args, kwargs):
        seen.append(self._training_state)
        return args, kwargs

    monkeypatch.setattr(FSDPState, "_pre_forward", pre_forward)
    token = in_checkpoint_recompute.set(recompute)
    try:
        assert state._pre_forward(nn.Identity(), (), {}) == ((), {})
    finally:
        in_checkpoint_recompute.reset(token)
    assert seen == [TrainingState.PRE_BACKWARD if recompute else TrainingState.IDLE]


def test_pipeline_patterns_reject_globally_unmatched_units():
    from llmtuner.parallel.fully_shard_parallel.plan import (
        resolve_pipeline_fsdp_configs,
    )

    cfg = ParallelConfig(fsdp_config=FSDPConfig(fsdp_apply_modules={"missing": {}}))
    with pytest.raises(ValueError, match="matched no module"):
        resolve_pipeline_fsdp_configs(nn.Sequential(nn.Linear(2, 2)), [], cfg)


@pytest.mark.parametrize("compiled", [False, True])
def test_fsdp_patterns_accept_logical_paths_after_checkpoint_and_compile(compiled):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    from llmtuner.parallel.fully_shard_parallel.plan import _resolve_pattern_units

    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)))
    first = model[0][0]
    ignored = model[0][1]
    model[0] = checkpoint_wrapper(model[0])
    block = model[0]
    if compiled:
        model = torch.compile(model, backend="eager")
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={"0": {}, "0.0": {}},
            fsdp_ignored_modules=["0.1"],
        )
    )
    modules, units, excluded = _resolve_pattern_units(model, cfg)
    assert {modules[name] for name in units} == {block, first}
    assert excluded == set(ignored.parameters())


@pytest.mark.parametrize("unit_ignored", [None, set()])
def test_unit_override_cannot_clear_global_fsdp_exclusions(monkeypatch, unit_ignored):
    from types import SimpleNamespace

    from llmtuner.config import ParallelConfig
    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    calls = []
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={"0": {"ignored_params": unit_ignored}},
            fsdp_ignored_modules=["1"],
        )
    )
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *a, **kw: None)
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda implementation: SimpleNamespace(
            get_fully_shard_fn=lambda: (
                lambda module, **kwargs: calls.append(kwargs["ignored_params"])
            )
        ),
    )
    monkeypatch.setattr(apply, "disable_fsdp_gradient_division", lambda model: None)
    plan.apply_fsdp_by_pattern(
        model,
        dp_mesh=object(),
        cfg=cfg,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    assert calls == [set(model[1].parameters()), set(model[1].parameters())]
    assert calls[0] is not calls[1]


@pytest.mark.parametrize(
    "pattern, message",
    [("missing", "matched no module"), ("{2-1}", "invalid module pattern range")],
)
def test_fsdp_ignored_patterns_reject_invalid_selections(pattern, message):
    from llmtuner.config.parallel_config import ParallelConfig
    from llmtuner.parallel.fully_shard_parallel.plan import _resolve_pattern_units

    model = nn.Sequential(nn.Linear(2, 2))
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={"0": {}}, fsdp_ignored_modules=[pattern]
        )
    )
    with pytest.raises(ValueError, match=message):
        _resolve_pattern_units(model, cfg)


def test_fsdp_apply_pattern_is_validated_when_every_module_is_ignored():
    from llmtuner.config.parallel_config import ParallelConfig
    from llmtuner.parallel.fully_shard_parallel.plan import _resolve_pattern_units

    model = nn.Sequential(nn.Linear(2, 2))
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={"{3-1}": {}}, fsdp_ignored_modules=["*"]
        )
    )
    with pytest.raises(ValueError, match="invalid module pattern range"):
        _resolve_pattern_units(model, cfg)


def test_ignored_fsdp_module_excludes_its_descendants():
    from llmtuner.config import ParallelConfig
    from llmtuner.parallel.fully_shard_parallel.plan import _resolve_pattern_units

    model = nn.ModuleDict(
        {
            "ignored": nn.Sequential(nn.Linear(2, 2)),
            "kept": nn.Sequential(nn.Linear(2, 2)),
            "ignored_extra": nn.Linear(2, 2),
        }
    )
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={"*": {}}, fsdp_ignored_modules=["ignored"]
        )
    )
    _, selected, ignored = _resolve_pattern_units(model, cfg)
    assert "ignored" not in selected
    assert "ignored.0" not in selected
    assert "kept.0" in selected
    assert "ignored_extra" in selected
    assert ignored == set(model["ignored"].parameters())


@pytest.mark.parametrize(
    "policy, pp, expected",
    [("never", 1, False), ("always", 2, True), ("default", 2, False)],
)
def test_fsdp_patterns_honor_runtime_policies(monkeypatch, policy, pp, expected):
    from types import SimpleNamespace

    from llmtuner.config import ParallelConfig
    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2))
    calls = []
    scopes = []
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *args, **kwargs: None)
    cfg = ParallelConfig(
        pipeline_parallel_size=pp,
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={"0": {}},
            fsdp_reshard_after_forward=policy,
            fsdp_symm_mem_scope="dense",
        ),
    )
    cfg.fsdp_config.enable_fsdp_symm_mem = True  # Exercise wiring without CUDA kernels.
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda _: SimpleNamespace(
            get_fully_shard_fn=lambda: lambda module, **kwargs: calls.append(kwargs)
        ),
    )
    monkeypatch.setattr(apply, "disable_fsdp_gradient_division", lambda _: None)
    monkeypatch.setattr(
        apply, "enable_fsdp_symm_mem", lambda model, scope: scopes.append(scope)
    )
    plan.apply_fsdp_by_pattern(
        model,
        dp_mesh=object(),
        cfg=cfg,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    assert all(call["reshard_after_forward"] is expected for call in calls)
    assert scopes == ["dense"]


def test_pattern_fsdp_with_ep_requires_sparse_mesh_before_mutation(monkeypatch):
    from llmtuner.config import ParallelConfig
    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2))
    cfg = ParallelConfig(
        expert_parallel_size=2, fsdp_config=FSDPConfig(fsdp_apply_modules={"0": {}})
    )
    calls = []
    monkeypatch.setattr(plan, "get_fsdp_strategy", lambda _: calls.append("shard"))
    with pytest.raises(ValueError, match="requires a sparse FSDP mesh"):
        plan.apply_fsdp_by_pattern(
            model,
            dp_mesh=object(),
            cfg=cfg,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    assert calls == []


def test_shared_pattern_units_reject_conflicting_policies_before_sharding(monkeypatch):
    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    model[1].weight = model[0].weight
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={
                "0": {"reshard_after_forward": True},
                "1": {"reshard_after_forward": False},
            }
        )
    )

    def unexpected(*args, **kwargs):
        pytest.fail("conflicting shared units mutated the model")

    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda _: type(
            "Strategy", (), {"get_fully_shard_fn": staticmethod(lambda: unexpected)}
        )(),
    )
    with pytest.raises(ValueError, match="share parameters.*conflicting"):
        plan.apply_fsdp_by_pattern(
            model,
            dp_mesh=object(),
            cfg=cfg,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )


def test_shared_pattern_dependency_cycle_merges_states(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.ModuleDict(
        {"a": nn.Sequential(nn.Linear(2, 2)), "b": nn.Sequential(nn.Linear(2, 2))}
    )
    # Parents own parameters used in the other branch's selected child.
    model["a"].register_parameter("shared", model["b"][0].weight)
    model["b"].register_parameter("shared", model["a"][0].weight)
    root = nn.Sequential(model)
    calls = []
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={name: {} for name in ("0.a", "0.a.0", "0.b", "0.b.0")}
        )
    )
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda _: SimpleNamespace(
            get_fully_shard_fn=lambda: lambda module, **kwargs: calls.append(module)
        ),
    )
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *args, **kwargs: None)
    monkeypatch.setattr(apply, "disable_fsdp_gradient_division", lambda model: None)
    plan.apply_fsdp_by_pattern(
        root,
        dp_mesh=object(),
        cfg=cfg,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    assert calls == [[model["a"], model["a"][0], model["b"], model["b"][0]], root]


@pytest.mark.parametrize("excluded", [None, set()])
def test_shared_pattern_units_compare_effective_exclusions(monkeypatch, excluded):
    from types import SimpleNamespace

    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    model[1].weight = model[0].weight
    calls = []
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={"0": {"ignored_params": excluded}, "1": {}}
        )
    )
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda _: SimpleNamespace(
            get_fully_shard_fn=lambda: lambda module, **kwargs: calls.append(module)
        ),
    )
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *args, **kwargs: None)
    monkeypatch.setattr(apply, "disable_fsdp_gradient_division", lambda model: None)
    plan.apply_fsdp_by_pattern(
        model,
        dp_mesh=object(),
        cfg=cfg,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    assert calls == [[model[0], model[1]], model]


def test_fsdp_ignored_pattern_resolves_shared_module_alias():
    from llmtuner.parallel.fully_shard_parallel.plan import _resolve_pattern_units

    shared = nn.Linear(2, 2)
    model = nn.ModuleDict({"first": shared, "alias": shared, "kept": nn.Linear(2, 2)})
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={"kept": {}}, fsdp_ignored_modules=["alias"]
        )
    )
    _, units, ignored = _resolve_pattern_units(model, cfg)
    assert units == {"kept": {}}
    assert ignored == set(shared.parameters())


def test_fsdp_apply_patterns_normalize_shared_module_aliases():
    from llmtuner.parallel.fully_shard_parallel.plan import _resolve_pattern_units

    shared = nn.Linear(2, 2)
    model = nn.Sequential(shared, shared)
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={"1": {"reshard_after_forward": False}, "*": {}}
        )
    )
    modules, units, ignored = _resolve_pattern_units(model, cfg)
    assert modules["0"] is shared
    assert units == {"0": {"reshard_after_forward": False}}
    assert ignored == set()


def test_fsdp_prefetch_windows_count_shared_states_once():
    from types import SimpleNamespace

    from torch.distributed.fsdp import FSDPModule

    class Unit(FSDPModule, nn.Linear):
        def __new__(cls, state):
            return object.__new__(cls)

        def __init__(self, state):
            nn.Linear.__init__(self, 2, 2)
            self.state = state

        def _get_fsdp_state(self):
            return self.state

    shared = SimpleNamespace()
    first = Unit(shared)
    middle = Unit(SimpleNamespace())
    last = Unit(shared)
    set_fsdp_prefetch_windows([first, middle, last], forward_count=1, backward_count=1)
    assert shared._states_to_forward_prefetch == [middle.state]
    assert shared._states_to_backward_prefetch == []
    assert middle.state._states_to_forward_prefetch == []
    assert middle.state._states_to_backward_prefetch == [shared]


@pytest.mark.parametrize("base_degree, unit_degree", [(2, 2), (1, 2), (2, 1)])
def test_pattern_fsdp_without_ep_shards_expert_features_when_degree_exceeds_experts(
    monkeypatch,
    base_degree,
    unit_degree,
):
    from types import SimpleNamespace

    from torch.distributed.tensor import Shard

    from llmtuner.models.common.moe.experts import GroupedExperts
    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(GroupedExperts(4, 8, 1, use_grouped_mm=False))
    calls = []
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda _: SimpleNamespace(
            get_fully_shard_fn=lambda: lambda module, **kwargs: calls.append(kwargs)
        ),
    )
    monkeypatch.setattr(plan, "set_fsdp_prefetch_windows", lambda *a, **kw: None)
    monkeypatch.setattr(apply, "disable_fsdp_gradient_division", lambda _: None)
    mesh = SimpleNamespace(size=lambda: base_degree, mesh_dim_names=("dp_shard",))
    unit_mesh = SimpleNamespace(size=lambda: unit_degree, mesh_dim_names=("dp_shard",))
    plan.apply_fsdp_by_pattern(
        model,
        dp_mesh=mesh,
        cfg=ParallelConfig(
            fsdp_config=FSDPConfig(fsdp_apply_modules={"0": {"mesh": unit_mesh}})
        ),
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    placement = calls[0]["shard_placement_fn"]
    for parameter in model[0].parameters():
        assert placement(parameter) == (Shard(1) if unit_degree > 1 else None)
    assert placement(nn.Parameter(torch.ones(4, 4))) is None


@pytest.mark.parametrize("layout", ["grouped", "managed", "nested"])
@pytest.mark.parametrize("scope", ["dense", "all"])
def test_symm_mem_selects_states_by_actual_moe_ownership(layout, scope):
    calls = []

    class Unit(nn.Module, apply.FSDPModule):
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __init__(self, state, *, moe=False):
            super().__init__()
            self.state = state
            self.moe_enabled = moe

        def _get_fsdp_state(self):
            return self.state

        def set_force_sum_reduction_for_comms(self, value):
            assert value is True

        def set_symm_mem_for_comm(self):
            calls.append(self.state)

    dense_state, expert_state = object(), object()
    if layout == "grouped":
        model = nn.ModuleList(
            [Unit(expert_state), Unit(expert_state, moe=True), Unit(dense_state)]
        )
    elif layout == "managed":
        model = nn.ModuleList([Unit(expert_state), Unit(dense_state)])
        model[0].child = nn.Identity()
        model[0].child.moe_enabled = True
    else:
        model = Unit(dense_state)
        model.child = Unit(expert_state, moe=True)

    apply.enable_fsdp_symm_mem(model, scope)
    expected = {dense_state} if scope == "dense" else {dense_state, expert_state}
    assert set(calls) == expected
    assert len(calls) == len(expected)


@pytest.mark.parametrize("compiled", [False, True])
def test_fsdp_references_follow_replacement_and_transparent_wrappers(compiled):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    from llmtuner.parallel.fully_shard_parallel.plan import FSDPPatternReferences

    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    original = model[0]
    policy = object()
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={
                "0": {
                    "ignored_params": {original.weight},
                    "hook_module": original,
                    "mesh": policy,
                }
            }
        )
    )
    references = FSDPPatternReferences(model, cfg)
    replacement = nn.Linear(4, 2, bias=False)
    model[0] = checkpoint_wrapper(replacement)
    wrapper = model[0]
    if compiled:
        model = torch.compile(model, backend="eager")
    rebound = references.bind(model).fsdp_config.fsdp_apply_modules["0"]
    assert rebound["ignored_params"] == {replacement.weight}
    assert rebound["hook_module"] is wrapper
    assert rebound["mesh"] is policy
    original_options = cfg.fsdp_config.fsdp_apply_modules["0"]
    assert original_options["ignored_params"] == {original.weight}
    assert original_options["hook_module"] is original


def test_fsdp_references_follow_all_parameter_aliases():
    from llmtuner.parallel.fully_shard_parallel.plan import FSDPPatternReferences

    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    model[1].weight = model[0].weight
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={"": {"ignored_params": {model[0].weight}}}
        )
    )
    references = FSDPPatternReferences(model, cfg)
    model[0] = nn.Linear(4, 2, bias=False)
    model[1] = nn.Linear(4, 2, bias=False)
    exclusions = references.bind(model).fsdp_config.fsdp_apply_modules[""][
        "ignored_params"
    ]
    assert exclusions == {model[0].weight, model[1].weight}


@pytest.mark.parametrize("override", ["hook_module", "ignored_params"])
def test_fsdp_references_reject_removed_objects_instead_of_dropping_them(override):
    from llmtuner.parallel.fully_shard_parallel.plan import FSDPPatternReferences

    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    value = model[0] if override == "hook_module" else {model[0].weight}
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(fsdp_apply_modules={"": {override: value}})
    )
    references = FSDPPatternReferences(model, cfg)
    model[0] = nn.Identity()
    if override == "hook_module":
        # A same-path replacement has a valid module mapping. Remove the path
        # completely to exercise a conversion that cannot retain the target.
        del model[0]
    with pytest.raises(ValueError, match=f"{override} lost its registered paths"):
        references.bind(model)


def test_fsdp_references_keep_objects_retained_under_new_paths():
    from llmtuner.parallel.fully_shard_parallel.plan import FSDPPatternReferences

    model = nn.Module()
    model.gate = nn.Linear(4, 4, bias=False)
    original = model.gate
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={
                "": {"ignored_params": {original.weight}, "hook_module": original}
            }
        )
    )
    references = FSDPPatternReferences(model, cfg)
    model.router = original
    del model.gate
    rebound = references.bind(model).fsdp_config.fsdp_apply_modules[""]
    assert rebound["ignored_params"] == {original.weight}
    assert rebound["hook_module"] is original
