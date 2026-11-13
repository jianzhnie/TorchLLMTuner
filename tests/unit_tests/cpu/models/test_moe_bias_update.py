"""The auxiliary-loss-free expert bias: ``update_expert_bias`` and its hook.

Two things are pinned here, and they fail differently:

* the **update rule** -- sign-based, mean-centred, in place -- which is what
  keeps ``sum(expert_bias_E) == 0`` and makes the step size independent of how
  lopsided the load is. A rule that drifted from that would still "balance" the
  experts, just slowly and with the whole routed output shifting underneath it.
* the **hook wiring** -- that the counts are reduced over exactly the axes that
  shard a token stream, and over no others. Reducing over the expert dim or the
  pipeline dim, or skipping DP, produces a bias that is subtly wrong per rank
  rather than obviously broken.

The collectives are exercised with a size-1 gloo group, so the count is a no-op
arithmetically but the code path still runs. That is the same fixture shape
``test_topology.py`` uses.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from tests.caps import require_env

require_env("spmd_types")


import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Model
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

import llmtuner.models.common.moe.bias_update as bias_update
from llmtuner.models.common.moe.bias_update import (
    collect_routing_load_metrics,
    register_moe_balancing_hook,
    reset_routing_statistics,
    update_expert_bias,
    update_quantile_expert_bias,
)
from llmtuner.models.common.moe.experts import GroupedExperts, RoutedExperts
from llmtuner.models.common.moe.layer import MoE, collect_moe_layers
from llmtuner.models.common.moe.routing import (
    QuantileBalancedTopKRouter,
    TokenChoiceTopKRouter,
)
from llmtuner.models.contracts import DecoderStructure, MoEBlockContract
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel.model_replacement import swap_moe_blocks
from llmtuner.parallel.expert_parallel.token_dispatch import LocalTokenDispatcher
from llmtuner.trainer.runtime.training_observer import DispatchTimer

_NUM_EXPERTS = 4
_DIM = 8
_TOP_K = 2


def test_dispatch_observation_preserves_real_moe_forward_backward_and_update():
    torch.manual_seed(47)
    reference = _moe(coeff=None)
    reference.routed_experts.inner_experts.use_grouped_mm = False
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.1)
    candidate = deepcopy(reference)
    timer = DispatchTimer()
    candidate.routed_experts.dispatch_observer = timer
    left = torch.randn(7, _DIM, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    expected, actual = reference(left), candidate(right)
    assert expected.abs().sum() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
    for first, second in zip(
        reference.parameters(), candidate.parameters(), strict=True
    ):
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
    torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False).step()
    torch.optim.SGD(candidate.parameters(), lr=0.01, foreach=False).step()
    for first, second in zip(
        reference.parameters(), candidate.parameters(), strict=True
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert timer.consume_seconds() > 0
    assert timer.consume_seconds() == 0


def test_load_metrics_observe_nonuniform_counts_without_mutation():
    layer = _moe()
    part = _Holder([layer])
    counts = torch.tensor([0, 2, 6, 0], dtype=layer.tokens_per_expert_E.dtype)
    layer.tokens_per_expert_E.copy_(counts)
    result = collect_routing_load_metrics([part], None, torch.device("cpu"))
    assert result["moe/empty_expert_fraction"] == 0.5
    assert result["moe/max_to_mean_load"] == 3
    assert result["moe/load_cv"] == pytest.approx(1.2247449)
    torch.testing.assert_close(layer.tokens_per_expert_E, counts)
    layer.tokens_per_expert_E.copy_(torch.tensor([1, 0, 0, 0]))
    sparse = collect_routing_load_metrics([part], None, torch.device("cpu"))
    assert sparse["moe/max_to_mean_load"] == 4
    assert sparse["moe/empty_expert_fraction"] == 0.75
    layer.tokens_per_expert_E.zero_()
    assert collect_routing_load_metrics([part], None, torch.device("cpu")) == {}


@pytest.fixture(scope="module")
def single_rank_group(tmp_path_factory):
    """A size-1 gloo group: the collectives run, nothing moves."""
    created = not dist.is_initialized()
    if created:
        store = dist.FileStore(str(tmp_path_factory.mktemp("pg") / "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    yield
    if created:
        dist.destroy_process_group()


def _moe(*, coeff: float | None = 0.1, num_experts: int = _NUM_EXPERTS) -> MoE:
    return MoE(
        num_experts,
        RoutedExperts(
            GroupedExperts(_DIM, 8, num_experts),
            LocalTokenDispatcher(num_experts, _TOP_K),
        ),
        TokenChoiceTopKRouter(num_experts, _DIM, _TOP_K),
        load_balance_coeff=coeff,
    )


class _Holder(torch.nn.Module):
    """Stands in for a model part: ``collect_moe_layers`` walks ``.layers``."""

    def __init__(self, moes: list[MoE | None]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(moe) for moe in moes])


class _Layer(torch.nn.Module):
    """A decoder layer holding a MoE, or a dense one holding none.

    The block lives in the explicitly declared ``mlp`` slot, matching the
    contract the replacement and bias-update paths consume.
    """

    def __init__(self, moe: MoE | None) -> None:
        super().__init__()
        self.moe_block_name = "mlp" if moe is not None else None
        if moe is not None:
            self.mlp = moe


# -- the update rule ----------------------------------------------------------


def test_the_step_is_sign_based_with_magnitude_coeff() -> None:
    """Every expert moves by exactly ``+/- coeff`` -- never by how far off it is.

    That is what makes one runaway expert unable to dominate, and it is also
    why activation checkpointing's doubled count is harmless.
    """
    moe = _moe(coeff=0.1)
    moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    before = moe.expert_bias_E.clone()

    moe.update_expert_bias()

    delta = moe.expert_bias_E - before
    nonzero = delta[delta != 0]
    assert nonzero.numel() > 0
    assert torch.equal(nonzero.abs(), torch.full_like(nonzero, 0.1))


def test_under_loaded_experts_are_pushed_up_and_over_loaded_down() -> None:
    """The sign of the step is what steers load; getting it backwards diverges."""
    moe = _moe(coeff=0.1)
    # Expert 0 sees twice the mean, expert 1 sees nothing.
    moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    before = moe.expert_bias_E.clone()

    moe.update_expert_bias()

    delta = moe.expert_bias_E - before
    assert float(delta[0]) < 0, "the over-loaded expert must lose bias"
    assert float(delta[1]) > 0, "the under-loaded expert must gain bias"


def test_the_update_is_mean_centred() -> None:
    """``sum(expert_bias_E)`` stays 0, so the bias shifts choices, not the output.

    Without centring the bias would drift upward as a whole and act as a global
    additive term on every routing score.
    """
    moe = _moe(coeff=0.1)
    moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))

    moe.update_expert_bias()

    assert abs(float(moe.expert_bias_E.sum())) < 1e-9


def test_a_uniform_load_leaves_the_bias_alone() -> None:
    """Already-balanced counts mean every delta is 0 -- the non-vacuity check.

    Confirms the tests above are measuring the load and not simply adding a
    constant on every call.
    """
    moe = _moe(coeff=0.1)
    moe.tokens_per_expert_E.copy_(torch.full((_NUM_EXPERTS,), 7.0))

    moe.update_expert_bias()

    assert torch.equal(moe.expert_bias_E, torch.zeros(_NUM_EXPERTS))


def test_the_counter_is_drained_so_it_never_leaks_into_the_next_step() -> None:
    """The counter is per-step scratch; ``update_expert_bias`` is what zeroes it."""
    moe = _moe()
    moe.tokens_per_expert_E.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    moe.update_expert_bias()

    assert float(moe.tokens_per_expert_E.sum()) == 0.0


def test_a_disabled_coefficient_makes_the_update_a_no_op() -> None:
    """``load_balance_coeff=None`` means no bias buffer and nothing to update."""
    moe = _moe(coeff=None)
    moe.tokens_per_expert_E.copy_(torch.ones(_NUM_EXPERTS))

    moe.update_expert_bias()

    assert moe.expert_bias_E is None


# -- the hook -----------------------------------------------------------------


def test_the_hook_fires_once_per_optimizer_step() -> None:
    """One update per ``step()``, not one per microbatch.

    That granularity is the whole reason the bias lives outside the forward: it
    has to see a complete accumulation window.
    """
    moe = _moe()
    model = _Holder([moe])
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.zeros(1))], lr=0.1, foreach=False
    )
    register_moe_balancing_hook(optimizer, model_parts=[model], parallel_dims=None)

    assert float(moe.expert_bias_E.abs().sum()) == 0.0  # nothing yet: no counts
    moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    optimizer.step()

    assert float(moe.expert_bias_E.abs().sum()) > 0.0


def test_no_hook_is_registered_for_a_model_without_moe_layers() -> None:
    """A dense run must not pay a traversal or an empty collective per step."""
    model = _Holder([])
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.zeros(1))], lr=0.1, foreach=False
    )

    register_moe_balancing_hook(optimizer, model_parts=[model], parallel_dims=None)

    assert optimizer._optimizer_step_pre_hooks == {}


def test_no_hook_is_registered_when_every_coeff_is_none() -> None:
    """``load_balance_coeff=None`` everywhere means nothing consumes the counts.

    Registering anyway would cost one ``vstack`` + dp/cp all-reduce per step
    whose result every layer then ignores -- a pure-waste collective.
    """
    model = _Holder([_moe(coeff=None), _moe(coeff=None)])
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.zeros(1))], lr=0.1, foreach=False
    )

    register_moe_balancing_hook(optimizer, model_parts=[model], parallel_dims=None)

    assert optimizer._optimizer_step_pre_hooks == {}


def test_a_mixed_coeff_configuration_is_rejected() -> None:
    """Balancing only some layers silently would drift the rest, so fail fast.

    Torchtitan raises on the same configuration in
    ``_should_register_moe_balancing_hook``; the swap installs the coeff
    uniformly, so reaching this state means a hand-built model -- exactly the
    case a loud error should catch.
    """
    model = _Holder([_moe(coeff=0.1), _moe(coeff=None)])
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.zeros(1))], lr=0.1, foreach=False
    )

    with pytest.raises(ValueError, match="load_balance_coeff"):
        register_moe_balancing_hook(optimizer, model_parts=[model], parallel_dims=None)

    assert optimizer._optimizer_step_pre_hooks == {}


def test_a_dense_layer_among_sparse_ones_is_skipped() -> None:
    """Mixed sparse/dense models have layers with no MoE at all."""
    sparse = _moe()
    model = _Holder([sparse, None, None])

    assert collect_moe_layers(model) == [sparse]


@pytest.mark.parametrize("repeated_part", [False, True])
def test_shared_moe_block_is_balanced_once_per_optimizer_step(repeated_part):
    sparse = _moe(coeff=0.1)
    model = _Holder([sparse, sparse])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
    register_moe_balancing_hook(
        optimizer,
        model_parts=[model, model] if repeated_part else [model],
        parallel_dims=None,
    )
    sparse.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    before = sparse.expert_bias_E.clone()
    optimizer.step()
    torch.testing.assert_close(
        sparse.expert_bias_E - before, torch.tensor([-0.1, 0.1, 0.0, 0.0])
    )
    assert collect_moe_layers(model) == [sparse]


def test_wrapped_layers_are_unwrapped() -> None:
    """AC wraps each layer in CheckpointWrapper; the lookup must see through.

    Older torch versions do not forward attribute reads through the wrapper,
    so without the explicit unwrap ``collect_moe_layers`` returns [] and every
    downstream consumer (bias hooks, padding-mask staging, compile's MoE
    detection) silently no-ops.
    """
    sparse = _moe()

    class _FakeCheckpointWrapper(torch.nn.Module):
        """torch's CheckpointWrapper shape: inner module, no __getattr__."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self._checkpoint_wrapped_module = inner

    holder = torch.nn.Module()
    holder.layers = torch.nn.ModuleList([_FakeCheckpointWrapper(_Layer(sparse))])

    assert collect_moe_layers(holder) == [sparse]


@pytest.mark.parametrize("architecture", ["qwen3_moe", "deepseek_v3_reference"])
def test_hook_respects_real_swapped_model_routing_policy(architecture, monkeypatch):
    """End to end against the object graph the swap actually produces.

    The hand-built layers above encode an assumption about *where* the swap
    leaves the block; exercise an explicit Qwen3 and an external DeepSeek-V3
    reference decoder without restoring the removed HF wrapper. This catches
    the lookup reading ``layer.moe`` while the swap wrote
    ``layer.mlp`` -- a mismatch under which every other test in this file still
    passes, the register function quietly no-ops, and the bias never updates.
    """
    torch.manual_seed(0)
    if architecture == "qwen3_moe":
        config = Qwen3MoeConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            num_experts=8,
            num_experts_per_tok=2,
            mlp_only_layers=[0],
        )
        model = Qwen3MoeForCausalLM(config).float().eval()

    else:
        config = AutoConfig.for_model(
            "deepseek_v3",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            n_group=2,
            topk_group=1,
            first_k_dense_replace=1,
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,
            norm_topk_prob=True,
            max_position_embeddings=128,
        )
        model = DeepseekV3Model(config).float().eval()
        for index, layer in enumerate(model.layers):
            layer.moe_block_name = (
                "mlp" if index >= config.first_k_dense_replace else None
            )
            if layer.moe_block_name is not None:
                layer.mlp.moe_contract = MoEBlockContract(
                    router="gate",
                    num_experts=config.n_routed_experts,
                    top_k=config.num_experts_per_tok,
                    score_func="sigmoid",
                    route_norm=config.norm_topk_prob,
                    route_scale=config.routed_scaling_factor,
                    shared_expert="shared_experts",
                    num_expert_groups=config.n_group,
                    num_limited_groups=config.topk_group,
                )
    assert swap_moe_blocks(model) == 2

    layers = collect_moe_layers(model)
    assert len(layers) == 2, "the hook cannot see the swapped-in blocks"

    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.zeros(1))], lr=0.1, foreach=False
    )
    register_moe_balancing_hook(optimizer, model_parts=[model], parallel_dims=None)
    balanced = torch.zeros(8)
    counts = torch.tensor([10.0, 0.0, 5.0, 5.0, 3.0, 3.0, 2.0, 2.0])
    uses_bias = architecture == "deepseek_v3_reference"
    if not uses_bias:
        # Native Qwen3 softmax routing must not gain sigmoid-style bias updates.
        assert not optimizer._optimizer_step_pre_hooks
    for moe in layers:
        moe.tokens_per_expert_E.copy_(counts)
        if uses_bias:
            assert torch.equal(moe.expert_bias_E, balanced)
        else:
            assert moe.expert_bias_E is None and moe.load_balance_coeff is None

    optimizer.step()

    for moe in layers:
        if uses_bias:
            assert not torch.equal(moe.expert_bias_E, balanced), "the bias never moved"
            assert float(moe.tokens_per_expert_E.sum()) == 0.0, (
                "the counter was not drained"
            )
        else:
            torch.testing.assert_close(moe.tokens_per_expert_E, counts)


def test_the_collective_runs_over_a_real_process_group(single_rank_group) -> None:
    """The reduction path executes without a mesh -- e.g. before a mesh exists."""
    moe = _moe()
    moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    update_expert_bias([moe], parallel_dims=None)

    # A size-1 group leaves the counts alone, so the update is the same one the
    # direct test asserts.
    assert float(moe.expert_bias_E[1]) > 0


def test_bias_update_supports_different_expert_counts():
    layers = [_moe(num_experts=size) for size in (4, 8)]
    references = [_moe(num_experts=size) for size in (4, 8)]
    for layer, reference in zip(layers, references, strict=True):
        counts = torch.arange(layer.tokens_per_expert_E.numel(), dtype=torch.float32)
        layer.tokens_per_expert_E.copy_(counts)
        reference.tokens_per_expert_E.copy_(counts)
        reference.update_expert_bias()
    update_expert_bias(layers)
    for layer, reference in zip(layers, references, strict=True):
        torch.testing.assert_close(layer.expert_bias_E, reference.expert_bias_E)
        assert torch.count_nonzero(layer.tokens_per_expert_E) == 0


def test_every_layer_of_every_part_is_updated() -> None:
    """Under pipeline parallelism a part owns a subset; all of them must move."""
    a, b, c = _moe(), _moe(), _moe()
    parts = [_Holder([a, b]), _Holder([c])]
    for moe in (a, b, c):
        moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))

    update_expert_bias(
        [moe for part in parts for moe in collect_moe_layers(part)], parallel_dims=None
    )

    for moe in (a, b, c):
        assert float(moe.expert_bias_E[1]) > 0, "a layer was skipped"
        assert float(moe.tokens_per_expert_E.sum()) == 0.0, "a counter was not drained"


@pytest.mark.parametrize(
    "coeff", [float("nan"), float("inf"), -float("inf"), True, False, "0.1", 0, -1]
)
def test_moe_rejects_invalid_balance_coefficient(coeff):
    with pytest.raises(ValueError, match="load_balance_coeff must be finite"):
        _moe(coeff=coeff)


def test_shared_bias_uses_joint_counts_once_and_drains_both_layers():
    first, second = _moe(), _moe()
    second.expert_bias_E = first.expert_bias_E
    first.tokens_per_expert_E.copy_(torch.tensor([8.0, 0, 0, 0]))
    second.tokens_per_expert_E.copy_(torch.tensor([0.0, 3, 3, 3]))
    # Joint counts [8, 3, 3, 3]: one centered sign update.
    update_expert_bias([first, second])
    torch.testing.assert_close(
        first.expert_bias_E, torch.tensor([-0.15, 0.05, 0.05, 0.05])
    )
    assert torch.count_nonzero(first.tokens_per_expert_E) == 0
    assert torch.count_nonzero(second.tokens_per_expert_E) == 0


def test_shared_bias_rejects_conflicting_coefficients_before_mutation():
    first, second = _moe(coeff=0.1), _moe(coeff=0.2)
    second.expert_bias_E = first.expert_bias_E
    first.tokens_per_expert_E.fill_(2)
    with pytest.raises(ValueError, match="identical load_balance_coeff"):
        update_expert_bias([first, second])
    assert torch.count_nonzero(first.expert_bias_E) == 0
    torch.testing.assert_close(first.tokens_per_expert_E, torch.full((4,), 2.0))


def test_shared_counter_is_counted_once_with_independent_contribution():
    first, alias, independent = _moe(), _moe(), _moe()
    alias.expert_bias_E = independent.expert_bias_E = first.expert_bias_E
    alias.tokens_per_expert_E = first.tokens_per_expert_E
    first.tokens_per_expert_E.copy_(torch.tensor([8.0, 0, 0, 0]))
    independent.tokens_per_expert_E.copy_(torch.tensor([0.0, 12, 12, 12]))

    # [8, 12, 12, 12] must lower experts 1..3. Counting the shared buffer
    # twice gives [16, 12, 12, 12] and reverses every update direction.
    update_expert_bias([first, alias, independent])

    torch.testing.assert_close(
        first.expert_bias_E, torch.tensor([0.15, -0.05, -0.05, -0.05])
    )
    for layer in (first, alias, independent):
        assert not layer.tokens_per_expert_E.any()


def test_shared_counter_reduced_once_in_original_order(monkeypatch):
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([3.0, 4.0])
    sizes = []
    mesh = SimpleNamespace(get_group=lambda: None)
    topology = SimpleNamespace(
        ep_enabled=False,
        get_optional_mesh=lambda axis: mesh if axis == "dp" else None,
    )

    def reduce(tensor, **kwargs):
        sizes.append(tensor.numel())
        tensor.mul_(2)

    monkeypatch.setattr(bias_update, "all_reduce", reduce)
    result = bias_update._reduce_token_statistics([first, second, first], topology)

    assert sizes == [4]
    assert result[0] is result[2]
    torch.testing.assert_close(result[0], first * 2)
    torch.testing.assert_close(result[1], second * 2)
    torch.testing.assert_close(first, torch.tensor([1.0, 2.0]))


def test_shared_counter_updates_each_independent_bias_owner():
    first, second = _moe(), _moe()
    second.tokens_per_expert_E = first.tokens_per_expert_E
    first.tokens_per_expert_E.copy_(torch.tensor([8.0, 0, 0, 0]))

    update_expert_bias([first, second])

    expected = torch.tensor([-0.15, 0.05, 0.05, 0.05])
    torch.testing.assert_close(first.expert_bias_E, expected)
    torch.testing.assert_close(second.expert_bias_E, expected)
    assert not first.tokens_per_expert_E.any()


def test_shared_quantile_bias_uses_joint_histogram_and_original_bias():

    layers = [
        MoE(
            4,
            RoutedExperts(GroupedExperts(8, 8, 4), LocalTokenDispatcher(4, 2)),
            QuantileBalancedTopKRouter(4, 8, 2, num_bins=8),
            load_balance_coeff=None,
        )
        for _ in range(2)
    ]
    first, second = layers
    first.expert_bias_E.copy_(torch.tensor([-0.3, -0.1, 0.1, 0.3]))
    second.expert_bias_E = first.expert_bias_E
    h1 = first.router.quantile_balancer.required_bias_histogram_EB
    h2 = second.router.quantile_balancer.required_bias_histogram_EB
    h1.copy_(torch.arange(32).reshape(4, 8) % 7)
    h2.copy_(h1.flip(1) * 2)
    expected = first.router.quantile_balancer.estimate_expert_bias(
        h1 + h2, first.expert_bias_E.clone()
    )
    first.tokens_per_expert_E.fill_(3)
    second.tokens_per_expert_E.fill_(4)
    update_quantile_expert_bias(layers)
    torch.testing.assert_close(first.expert_bias_E, expected)
    assert not h1.any() and not h2.any()
    assert not first.tokens_per_expert_E.any()
    assert not second.tokens_per_expert_E.any()


def test_shared_histogram_updates_independent_quantile_bias_owners():

    layers = [
        MoE(
            4,
            RoutedExperts(GroupedExperts(8, 8, 4), LocalTokenDispatcher(4, 2)),
            QuantileBalancedTopKRouter(4, 8, 2, num_bins=8),
            load_balance_coeff=None,
        )
        for _ in range(2)
    ]
    histogram = layers[0].router.quantile_balancer.required_bias_histogram_EB
    histogram.copy_(torch.arange(32).reshape(4, 8) % 7)
    layers[1].router.quantile_balancer.required_bias_histogram_EB = histogram
    layers[0].expert_bias_E.copy_(torch.tensor([-0.3, -0.1, 0.1, 0.3]))
    layers[1].expert_bias_E.copy_(torch.tensor([0.2, -0.4, 0.3, -0.1]))
    expected = [
        layer.router.quantile_balancer.estimate_expert_bias(
            histogram.clone(), layer.expert_bias_E.clone()
        )
        for layer in layers
    ]
    assert histogram.any()
    for layer in layers:
        layer.tokens_per_expert_E.fill_(3)

    update_quantile_expert_bias(layers)

    for layer, bias in zip(layers, expected, strict=True):
        torch.testing.assert_close(layer.expert_bias_E, bias)
        assert not layer.tokens_per_expert_E.any()
    assert not histogram.any()


def test_reset_routing_statistics_clears_scratch_but_preserves_bias():
    sign_layer = _moe()
    quantile_layer = MoE(
        _NUM_EXPERTS,
        RoutedExperts(
            GroupedExperts(_DIM, 8, _NUM_EXPERTS),
            LocalTokenDispatcher(_NUM_EXPERTS, _TOP_K),
        ),
        QuantileBalancedTopKRouter(_NUM_EXPERTS, _DIM, _TOP_K, num_bins=8),
        load_balance_coeff=None,
    )
    sign_layer.expert_bias_E.copy_(torch.arange(_NUM_EXPERTS))
    quantile_layer.expert_bias_E.copy_(torch.arange(_NUM_EXPERTS) + 4)
    sign_layer.tokens_per_expert_E.fill_(3)
    quantile_layer.tokens_per_expert_E.fill_(5)
    histogram = quantile_layer.router.quantile_balancer.required_bias_histogram_EB
    histogram.fill_(7)
    sign_bias = sign_layer.expert_bias_E.clone()
    quantile_bias = quantile_layer.expert_bias_E.clone()
    part = _Holder([sign_layer, quantile_layer])

    reset_routing_statistics([part, part])

    assert not sign_layer.tokens_per_expert_E.any()
    assert not quantile_layer.tokens_per_expert_E.any()
    assert not histogram.any()
    torch.testing.assert_close(sign_layer.expert_bias_E, sign_bias)
    torch.testing.assert_close(quantile_layer.expert_bias_E, quantile_bias)


def test_declared_slots_control_native_moe_collection():
    sparse = _moe()
    layer = torch.nn.Module()
    layer.expert_block = sparse
    layer.mlp = _moe()
    layer.moe_block_name = "expert_block"
    holder = torch.nn.Module()
    holder.layers = torch.nn.ModuleList([layer, layer])
    assert collect_moe_layers(holder) == [sparse]
    layer.moe_block_name = None
    assert collect_moe_layers(holder) == []


def test_native_moe_collection_rejects_missing_slot_declaration():
    holder = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = _moe()
    holder.layers = torch.nn.ModuleList([layer])
    with pytest.raises(ValueError, match="moe_block_name"):
        collect_moe_layers(holder)


@pytest.mark.parametrize("compiled", [False, True])
def test_moe_collection_binds_declared_local_blocks_instead_of_stale_alias(compiled):
    sparse = _moe()
    holder = torch.nn.Module()
    holder.blocks = torch.nn.ModuleList([_Layer(sparse)])
    holder.decoder_structure = DecoderStructure(
        decoder="", embedding="embedding", blocks="blocks", norm="norm", head="head"
    )
    # A PP-local part need not own embedding/norm/head. Its convenience alias
    # may still refer to the former whole-model blocks and must not be used.
    object.__setattr__(holder, "layers", torch.nn.ModuleList([_Layer(_moe())]))
    model = torch.compile(holder, backend="eager") if compiled else holder
    assert collect_moe_layers(model) == [sparse]


def test_moe_collection_does_not_fall_back_from_malformed_structure():
    holder = _Holder([_moe()])
    holder.decoder_structure = "layers"
    with pytest.raises(TypeError, match="DecoderStructure"):
        collect_moe_layers(holder)
