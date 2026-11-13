"""Model-owned CP setup, KCP constraints and removed-backend rejection."""

from __future__ import annotations

from tests.caps import require_env

require_env("spmd_types")


import pytest
import torch

from llmtuner.config import ParallelConfig
from llmtuner.config.parallel_config import ContextParallelConfig
from llmtuner.models.contracts import DecoderStructure
from llmtuner.parallel.context_parallel import apply_cp


def test_apply_cp_is_a_no_op_when_cp_is_off() -> None:
    """CP=1 must hand the model back untouched rather than raise."""
    model = torch.nn.Linear(4, 4)
    cfg = ParallelConfig(context_parallel_kv_allgather_size=1)

    assert apply_cp(model, mesh=None, cfg=cfg) is model


# -- the kernel's strategy dispatch -------------------------------------------


def test_the_default_strategy_is_kv_allgather() -> None:
    """``ParallelConfig``'s default; the kernel half lives below."""
    assert ParallelConfig().cp_config.context_parallel_strategy == "kv_allgather"


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
    """Explicit structure and input contract for CP setup-only tests."""

    decoder_structure = DecoderStructure(
        decoder="",
        embedding="embedding",
        blocks="layers",
        norm="norm",
        head="head",
        attention="self_attn",
    )

    def __init__(self, num_attention_heads: int, num_key_value_heads: int) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.config = SimpleNamespace(
            _attn_implementation="sdpa",
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        self.layers = torch.nn.ModuleList()
        self.cp_strategy = None

    def set_cp_mesh(self, cp_mesh, load_balancer=None, strategy="kv_allgather") -> None:
        self.cp_strategy = strategy


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
    from llmtuner.models.kimi.kimi_k3.latent_attention import KimiLatentAttention

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

    from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
    from llmtuner.models.kimi.kimi_k3.latent_attention import KimiLatentAttention
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


def test_cp_dispatches_from_model_backend_contract() -> None:
    class Model(_StubModel):
        context_parallel_backend = "compressor"

    cfg = ParallelConfig(context_parallel_kv_allgather_size=2)
    with pytest.raises(ValueError, match="HCA CP requires a cp mesh"):
        apply_cp(Model(4, 4), mesh=None, cfg=cfg)


@pytest.mark.parametrize("strategy", ["kv_allgather", "ulysses"])
def test_removed_standard_cp_fails_before_mutation(strategy):
    from llmtuner.parallel.context_parallel.context_parallel import resolve_cp_plan

    model = _StubModel(4, 4)
    cfg = ParallelConfig(
        context_parallel_kv_allgather_size=2,
        cp_config=ContextParallelConfig(context_parallel_strategy=strategy),
    )
    rng = torch.get_rng_state().clone()
    with pytest.raises(NotImplementedError, match="kernels have been removed"):
        resolve_cp_plan(model, cfg=cfg)
    with pytest.raises(NotImplementedError, match="kernels have been removed"):
        apply_cp(model, mesh=None, cfg=cfg)
    assert model.cp_strategy is None
    assert torch.equal(torch.get_rng_state(), rng)
