"""Context-parallel attention: the single-rank guards and the kernel's setup.

The redistributions themselves need two ranks and live in
``tests/cp_equivalence.py`` and ``cp_wiring_equivalence.py``. What can be checked
without a mesh is where the module is *explicit*: the strategy dispatch, the
refusals that carry the whole safety argument, and the ulysses head-divisibility
check. The refusal matters more here than anywhere else in llmtuner -- every other
component degrades gracefully when a parallelism axis is off, but a CP
redistribution cannot, because skipping it does not skip work, it changes the
answer.

``apply_cp``'s argument validation runs before any collective, so most of the
kernel-facing checks below stand a size-1 "fake" process group up instead of a
real two-rank one; a genuinely multi-rank redistribution is the equivalence
harnesses' job.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("spmd_types")


import pytest
import torch

from llmtuner.config import ParallelConfig
from llmtuner.config.parallel_config import ContextParallelConfig
from llmtuner.parallel.context_parallel import (
    CPFlexKernel,
    apply_cp,
)


def test_apply_cp_is_a_no_op_when_cp_is_off() -> None:
    """CP=1 must hand the model back untouched rather than raise."""
    model = torch.nn.Linear(4, 4)
    cfg = ParallelConfig(context_parallel_kv_allgather_size=1)

    assert apply_cp(model, mesh=None, cfg=cfg) is model


def test_apply_cp_requires_the_flex_backend() -> None:
    """CP expresses the sharded mask as a BlockMask, which only flex consumes.

    A plain module has no HF config at all, so it fails the backend check --
    the same check that stops a CPU-run model (whose wrapper selected 'sdpa')
    from silently computing attention over its own shard only.
    """

    cfg = ParallelConfig(context_parallel_kv_allgather_size=2)

    with pytest.raises(RuntimeError, match="attention backend"):
        apply_cp(torch.nn.Linear(4, 4), mesh=None, cfg=cfg)


# -- the kernel's strategy dispatch -------------------------------------------


@pytest.fixture
def fake_cp_mesh():
    """A size-1 CP axis over the 'fake' backend -- enough to construct a kernel.

    ``CPFlexKernel.__init__`` captures the process group and the set of allowed
    strategies before any collective runs, so building one needs a mesh but not
    a second rank.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    store = FakeStore()
    dist.init_process_group("fake", store=store, rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("cp",))["cp"]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strategy", ["kv_allgather", "ulysses"])
def test_the_known_strategies_are_accepted(fake_cp_mesh, strategy: str) -> None:
    """Both strategies the package advertises must actually construct.

    Non-vacuity: this is the accepting half of the pair below, so a kernel that
    rejected everything cannot pass both.
    """
    kernel = CPFlexKernel(cp_mesh=fake_cp_mesh, strategy=strategy)

    assert kernel.strategy == strategy


def test_an_unknown_strategy_is_refused(fake_cp_mesh) -> None:
    """A typo'd strategy must fail at attach time, not silently pick one path.

    ``forward`` branches on the name, so an unrecognized value would fall
    through to the KV all-gather branch and train a *different* CP formulation
    than the one asked for, with the forward still producing plausible numbers.
    """
    with pytest.raises(NotImplementedError, match="not wired"):
        CPFlexKernel(cp_mesh=fake_cp_mesh, strategy="ring")


def test_the_default_strategy_is_kv_allgather(fake_cp_mesh) -> None:
    """The default has to match ``ParallelConfig``'s, or a caller that relies on
    both defaults gets whichever the other object happened to pick."""
    assert ParallelConfig().cp_config.context_parallel_strategy == "kv_allgather"
    assert CPFlexKernel(cp_mesh=fake_cp_mesh).strategy == "kv_allgather"


def test_dual_cp_axes_derive_total_degree() -> None:
    cfg = ParallelConfig(
        context_parallel_kv_allgather_size=2,
        context_parallel_ulysses_size=4,
    )

    assert cfg.cp_size == 8
    assert cfg.non_dp_sizes() == 8


def test_kcp_rejects_tensor_parallelism() -> None:
    with pytest.raises(Exception, match="KCP requires tensor_parallel_size=1"):
        ParallelConfig(
            context_parallel_kv_allgather_size=2,
            tensor_parallel_size=2,
            cp_config=ContextParallelConfig(context_parallel_strategy="kcp"),
        )


def test_kcp_has_an_independent_reachable_degree() -> None:
    cfg = ParallelConfig(
        context_parallel_kcp_size=4,
        cp_config=ContextParallelConfig(context_parallel_strategy="kcp"),
    )
    assert cfg.cp_size == 4
    assert cfg.derive_dp(8) == 2


def test_kcp_degree_requires_kcp_strategy() -> None:
    with pytest.raises(ValueError, match="requires strategy='kcp'"):
        ParallelConfig(context_parallel_kcp_size=2)


def test_kcp_reports_missing_model_before_optional_runtime(monkeypatch):
    from llmtuner.parallel.context_parallel.kcp import apply_kcp

    def runtime():
        pytest.fail("Model capability must be checked before kernel imports")

    monkeypatch.setattr("llmtuner.parallel.context_parallel.kcp._kcp_runtime", runtime)
    model = torch.nn.Linear(4, 4)
    original_forward = model.forward
    with pytest.raises(ValueError, match="no Kimi Delta Attention module"):
        apply_kcp(model, cp_mesh=None)
    assert model.forward == original_forward


def test_kcp_reapplication_replaces_wrapper_and_updates_group(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.parallel.context_parallel.kcp import apply_kcp

    class KDA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("q_conv1d", "k_conv1d", "v_conv1d", "A_log", "dt_bias"):
                setattr(self, name, None)

        def forward(self, x):
            return x

    calls = []

    def wrap(original, group):
        calls.append((original, group))

        def forward(self, x):
            return group

        return forward

    monkeypatch.setattr(
        "llmtuner.parallel.context_parallel.kcp._kcp_runtime", lambda: ()
    )
    monkeypatch.setattr("llmtuner.parallel.context_parallel.kcp.kcp_kda_forward", wrap)
    module = KDA()
    first_group, second_group = object(), object()
    assert apply_kcp(module, SimpleNamespace(get_group=lambda: first_group)) == 1
    assert module(None) is first_group
    assert apply_kcp(module, SimpleNamespace(get_group=lambda: second_group)) == 1
    assert module(None) is second_group
    assert calls == [(KDA.forward, first_group), (KDA.forward, second_group)]


def test_kcp_rejects_unit_convolution_before_loading_optional_runtime(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.parallel.context_parallel.kcp import kcp_kda_forward

    def runtime():
        pytest.fail("Invalid KCP convolution must fail before loading kernels")

    monkeypatch.setattr("llmtuner.parallel.context_parallel.kcp._kcp_runtime", runtime)
    module = SimpleNamespace(q_conv1d=torch.nn.Conv1d(4, 4, kernel_size=1))
    forward = kcp_kda_forward(lambda *args, **kwargs: None, group=None)
    with pytest.raises(ValueError, match="kernel size must be greater than 1"):
        forward(module, torch.zeros(1, 8, 4))


@pytest.mark.parametrize(
    "head_dim,lower_bound,message",
    [
        (257, None, "head dimension must be at most 256"),
        (64, -6, "lower bound must be in"),
        (64, 0, "lower bound must be in"),
        (64, float("nan"), "lower bound must be in"),
    ],
)
def test_kcp_rejects_invalid_kernel_options_before_import(
    monkeypatch, head_dim, lower_bound, message
):
    from types import SimpleNamespace

    from llmtuner.parallel.context_parallel.kcp import kcp_kda_forward

    def runtime():
        pytest.fail("Invalid KCP options must fail before loading kernels")

    monkeypatch.setattr("llmtuner.parallel.context_parallel.kcp._kcp_runtime", runtime)
    module = SimpleNamespace(
        q_conv1d=torch.nn.Conv1d(4, 4, kernel_size=3),
        head_k_dim=head_dim,
        gate_lower_bound=lower_bound,
    )
    forward = kcp_kda_forward(lambda *args, **kwargs: None, group=None)
    with pytest.raises(ValueError, match=message):
        forward(module, torch.zeros(1, 8, 4))


# -- ulysses head divisibility -------------------------------------------------
#
# The check lives behind mesh validation, so it needs a CP axis of size 2 --
# which a single process cannot build for real. The "fake" backend stands the
# mesh up without collectives, and the ulysses guards raise before any
# collective would run.


class _StubModel(torch.nn.Module):
    """Just enough of HFTransformerModel for apply_cp's validation path."""

    def __init__(self, num_attention_heads: int, num_key_value_heads: int) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.model = SimpleNamespace(
            config=SimpleNamespace(
                _attn_implementation="flex_torchtitan",
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
        )
        self.layers = torch.nn.ModuleList()
        self.cp_strategy = None

    def set_cp_mesh(self, cp_mesh, load_balancer=None, strategy="kv_allgather") -> None:
        self.cp_strategy = strategy


@pytest.fixture
def tp_cp_mesh():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    store = FakeStore()
    dist.init_process_group("fake", store=store, rank=0, world_size=4)
    try:
        yield init_device_mesh("cpu", (2, 2), mesh_dim_names=("tp", "cp"))
    finally:
        dist.destroy_process_group()


def _ulysses_cfg(tp: int) -> ParallelConfig:
    return ParallelConfig(
        tensor_parallel_size=tp,
        context_parallel_kv_allgather_size=2,
        cp_config=ContextParallelConfig(
            context_parallel_strategy="ulysses", context_parallel_load_balancer=None
        ),
    )


def test_ulysses_heads_must_divide_tp_times_cp(tp_cp_mesh) -> None:
    """TP shards heads first: 4 heads over tp=2 leaves 2 local, which cp=2
    divides -- but the same 4 heads over tp=1 must still pass, and 2 heads
    over tp=2 (1 local) must refuse even though 2 % cp == 0."""
    model = _StubModel(num_attention_heads=4, num_key_value_heads=4)
    apply_cp(model, mesh=tp_cp_mesh, cfg=_ulysses_cfg(tp=2))

    model = _StubModel(num_attention_heads=2, num_key_value_heads=2)
    with pytest.raises(ValueError, match=r"tp\*cp"):
        apply_cp(model, mesh=tp_cp_mesh, cfg=_ulysses_cfg(tp=2))


def test_ulysses_kv_heads_use_the_local_count_too(tp_cp_mesh) -> None:
    """num_key_value_heads falls under the same local-head rule."""
    model = _StubModel(num_attention_heads=8, num_key_value_heads=2)
    with pytest.raises(ValueError, match="num_key_value_heads"):
        apply_cp(model, mesh=tp_cp_mesh, cfg=_ulysses_cfg(tp=2))


def test_ulysses_replicated_attention_heads_are_not_divided_by_tp(tp_cp_mesh):
    from llmtuner.parallel.tensor_parallel.tensor_parallel import (
        ReplicatedAttentionSequenceBoundary,
    )

    class ReplicatedAttention(ReplicatedAttentionSequenceBoundary, torch.nn.Module):
        pass

    model = _StubModel(num_attention_heads=2, num_key_value_heads=2)
    layer = torch.nn.Module()
    layer.self_attn = ReplicatedAttention()
    model.layers.append(layer)
    apply_cp(model, mesh=tp_cp_mesh, cfg=_ulysses_cfg(tp=2))
    assert isinstance(layer.self_attn._titan_flex_kernel, CPFlexKernel)


def test_ulysses_packed_is_accepted_and_the_strategy_is_latched(tp_cp_mesh) -> None:
    """Ulysses x packed is a supported combination, and the wrapper must be
    told the strategy -- its mask handling branches on it (a packed document
    mask goes to the kernel full-length under ulysses, Q-sharded otherwise).

    Non-vacuity: the same attach under the default strategy must NOT latch
    'ulysses', so a stub that recorded a constant cannot pass both halves.
    """
    model = _StubModel(num_attention_heads=4, num_key_value_heads=4)
    model.model.config.attn_mask_type = "block_causal"
    apply_cp(model, mesh=tp_cp_mesh, cfg=_ulysses_cfg(tp=2))
    assert model.cp_strategy == "ulysses"

    model = _StubModel(num_attention_heads=4, num_key_value_heads=4)
    apply_cp(
        model,
        mesh=tp_cp_mesh,
        cfg=ParallelConfig(
            tensor_parallel_size=2, context_parallel_kv_allgather_size=2
        ),
    )
    assert model.cp_strategy == "kv_allgather"


@pytest.mark.parametrize("failure", ["missing_attention", "kernel_setup"])
def test_cp_missing_late_attention_preserves_earlier_kernel(monkeypatch, failure):
    from types import SimpleNamespace

    from llmtuner.models.hf.attention_backend import ATTENTION_BACKEND
    from llmtuner.parallel.context_parallel import context_parallel as cp

    old_kernel = object()
    attention = SimpleNamespace(_titan_flex_kernel=old_kernel)
    model = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation=ATTENTION_BACKEND)
        ),
        layers=[SimpleNamespace(self_attn=attention), SimpleNamespace()],
        set_cp_mesh=lambda *args, **kwargs: pytest.fail("invalid model changed mesh"),
    )

    class Mesh:
        mesh_dim_names = ("cp",)

        def __getitem__(self, key):
            return self

        def size(self):
            return 2

    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        if failure == "kernel_setup" and len(calls) == 2:
            raise RuntimeError("kernel setup failed")
        return object()

    if failure == "kernel_setup":
        model.layers[1].self_attn = SimpleNamespace()
    monkeypatch.setattr(cp, "CPFlexKernel", kernel)
    error = AttributeError if failure == "missing_attention" else RuntimeError
    message = "layer 1" if failure == "missing_attention" else "kernel setup failed"
    with pytest.raises(error, match=message):
        apply_cp(
            model, mesh=Mesh(), cfg=ParallelConfig(context_parallel_kv_allgather_size=2)
        )
    assert attention._titan_flex_kernel is old_kernel


@pytest.mark.parametrize("axis", ["cp_kv", "cp_ulysses"])
def test_dual_cp_rejects_wrong_axis_degree_before_kernel_setup(monkeypatch, axis):
    from types import SimpleNamespace

    from llmtuner.models.hf.attention_backend import ATTENTION_BACKEND
    from llmtuner.parallel.context_parallel import context_parallel as cp

    class Mesh:
        mesh_dim_names = ("cp",)

        def __init__(self, degree):
            self.degree = degree

        def __getitem__(self, key):
            return self

        def size(self):
            return self.degree

    model = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation=ATTENTION_BACKEND)
        )
    )
    topology = SimpleNamespace(
        get_optional_mesh=lambda name: Mesh(1 if name == axis else 2)
    )
    monkeypatch.setattr(cp, "_require_ulysses_compatible", lambda *a, **kw: None)
    with pytest.raises(ValueError, match=axis):
        apply_cp(
            model,
            mesh=Mesh(4),
            cfg=ParallelConfig(
                context_parallel_kv_allgather_size=2, context_parallel_ulysses_size=2
            ),
            parallel_dims=topology,
        )


@pytest.mark.parametrize("shape", [(4,), (1, 1, 4, 4), (2, 4), (1, 3)])
def test_kcp_rejects_nonlocal_mask_before_runtime(monkeypatch, shape):
    from llmtuner.parallel.context_parallel.kcp import kcp_kda_forward

    def runtime():
        pytest.fail("Invalid local mask must fail before loading kernels")

    monkeypatch.setattr("llmtuner.parallel.context_parallel.kcp._kcp_runtime", runtime)
    forward = kcp_kda_forward(lambda *args, **kwargs: None, group=None)
    with pytest.raises(ValueError, match="attention_mask must match"):
        forward(None, torch.ones(1, 4, 8), attention_mask=torch.ones(shape))


def test_kcp_rejects_empty_sequence_before_runtime(monkeypatch):
    from llmtuner.parallel.context_parallel.kcp import kcp_kda_forward

    def runtime():
        pytest.fail("Empty CP sequence must fail before loading kernels")

    monkeypatch.setattr("llmtuner.parallel.context_parallel.kcp._kcp_runtime", runtime)
    forward = kcp_kda_forward(lambda *args, **kwargs: None, group=None)
    with pytest.raises(ValueError, match="nonempty local sequence"):
        forward(None, torch.empty(1, 0, 8))


@pytest.mark.parametrize("q_lora_rank", [None, 4])
def test_kcp_checks_mla_heads_before_installation(monkeypatch, q_lora_rank):
    from llmtuner.parallel.context_parallel import kcp
    from tests.unit_tests.cpu.parallel.test_kimi_conversion import make_model

    model = make_model(latent=True, shared=1)
    from llmtuner.models.kimi.latent_attention import KimiLatentAttention

    config = model.layers[0].block_sparse_moe.config
    config.q_lora_rank = q_lora_rank
    config.qk_rope_head_dim = 2
    config.qk_nope_head_dim = 2
    config.kv_lora_rank = 4
    config.v_head_dim = 2
    config.mla_use_nope = True
    model.add_module("hybrid_attention", KimiLatentAttention(config, 0))

    from types import SimpleNamespace

    kda = torch.nn.Module()
    for name in ("q_conv1d", "k_conv1d", "v_conv1d", "A_log", "dt_bias"):
        setattr(kda, name, None)
    model.add_module("kda", kda)
    monkeypatch.setattr(kcp, "_kcp_runtime", lambda: ())
    group = SimpleNamespace(size=lambda: config.num_attention_heads + 1)
    with pytest.raises(ValueError, match="heads must be divisible"):
        kcp.apply_kcp(model, SimpleNamespace(get_group=lambda: group))
    assert not hasattr(model.hybrid_attention, "_kcp_attention")


def test_kcp_installs_mla_callable_without_changing_parameter_names(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
    from llmtuner.models.kimi.latent_attention import KimiLatentAttention
    from llmtuner.parallel.context_parallel import kcp

    config = KimiLinearConfig(
        hidden_size=12,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=3,
        qk_rope_head_dim=1,
        v_head_dim=3,
        mla_use_nope=True,
    )
    model = torch.nn.Module()
    model.mla = KimiLatentAttention(config, 0)
    model.kda = torch.nn.Module()
    for name in ("q_conv1d", "k_conv1d", "v_conv1d", "A_log", "dt_bias"):
        setattr(model.kda, name, None)
    before = dict(model.named_parameters())
    monkeypatch.setattr(kcp, "_kcp_runtime", lambda: ())
    first = SimpleNamespace(size=lambda: 2)
    assert kcp.apply_kcp(model, SimpleNamespace(get_group=lambda: first)) == 1
    original = model.mla._kcp_attention
    assert callable(original)
    second = SimpleNamespace(size=lambda: 2)
    assert kcp.apply_kcp(model, SimpleNamespace(get_group=lambda: second)) == 1
    assert model.mla._kcp_attention is not original
    after = dict(model.named_parameters())
    assert before.keys() == after.keys()
    assert all(after[name] is parameter for name, parameter in before.items())
