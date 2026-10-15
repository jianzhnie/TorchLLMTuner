"""Quantile-balanced MoE routing: router, balancer, and hook.

Pinned here, mirroring the upstream semantics:

* the **cutoff observation** -- a training forward computes one biased
  Top-(K+1), routes on the first K ids, and records ``cutoff - score`` for
  every token/expert into the histogram. The routing *weight* is gathered
  from the unbiased scores, exactly as in the base router.
* the **quantile solve** -- ``estimate_expert_bias`` reads the
  ``top_k / num_experts`` quantile out of the histogram, interpolated within
  its crossing bin, and mean-centres the result so ``sum(expert_bias_E) ==
  0``. The worked example below is the one from the upstream GPU test, so a
  drift in the binning or the interpolation shows up as a wrong number, not
  a vague imbalance.
* the **hook wiring and mutual exclusion** -- the quantile hook no-ops
  without quantile routers, refuses a mixed model, fires once per optimizer
  step, and drains both the histogram and the token counter.

The collectives are exercised with a size-1 gloo group, as in
``test_moe_bias_update.py``: the reduction is a no-op arithmetically but the code
path still runs.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("spmd_types")


import pytest
import torch
import torch.distributed as dist

from llmtuner.models.common.moe.bias_update import (
    register_moe_balancing_hook,
    update_quantile_expert_bias,
)
from llmtuner.models.common.moe.experts import GroupedExperts, RoutedExperts
from llmtuner.models.common.moe.layer import MoE
from llmtuner.models.common.moe.routing import (
    QuantileBalancedTopKRouter,
    TokenChoiceTopKRouter,
)
from llmtuner.parallel.expert_parallel.token_dispatch import LocalTokenDispatcher

_NUM_EXPERTS = 4
_DIM = 8
_TOP_K = 2


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


def _router(**kwargs) -> QuantileBalancedTopKRouter:
    torch.manual_seed(0)
    return QuantileBalancedTopKRouter(_NUM_EXPERTS, _DIM, _TOP_K, **kwargs)


def _moe(**kwargs) -> MoE:
    return MoE(
        _NUM_EXPERTS,
        RoutedExperts(
            GroupedExperts(_DIM, 8, _NUM_EXPERTS),
            LocalTokenDispatcher(_NUM_EXPERTS, _TOP_K),
        ),
        _router(),
        load_balance_coeff=None,
        **kwargs,
    )


class _Holder(torch.nn.Module):
    """Stands in for a model part: ``collect_moe_layers`` walks ``.layers``."""

    def __init__(self, moes: list[MoE | None]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(moe) for moe in moes])


class _Layer(torch.nn.Module):
    """A decoder layer holding a MoE under ``mlp``, as the swap leaves it."""

    def __init__(self, moe: MoE | None) -> None:
        super().__init__()
        if moe is not None:
            self.mlp = moe


# -- construction invariants ---------------------------------------------------


def test_a_quantile_moe_carries_a_bias_buffer_with_no_coeff() -> None:
    """The quantile scheme routes on the bias, so the buffer must exist even
    though the sign-based update that would otherwise own it is off."""
    moe = _moe()

    assert moe.load_balance_coeff is None
    assert isinstance(moe.expert_bias_E, torch.Tensor)
    assert torch.equal(moe.expert_bias_E, torch.zeros(_NUM_EXPERTS))
    # Persistent: a resumed run keeps the balance it reached.
    assert "expert_bias_E" in moe.state_dict()


def test_a_coeff_and_a_quantile_router_are_rejected() -> None:
    """Two updates cannot both own expert_bias_E (upstream raises the same
    way for the combined configuration)."""
    with pytest.raises(ValueError, match="load_balance_coeff must be None"):
        MoE(
            _NUM_EXPERTS,
            RoutedExperts(
                GroupedExperts(_DIM, 8, _NUM_EXPERTS),
                LocalTokenDispatcher(_NUM_EXPERTS, _TOP_K),
            ),
            _router(),
            load_balance_coeff=0.1,
        )


def test_the_router_refuses_to_route_without_a_bias() -> None:
    router = _router()

    with pytest.raises(ValueError, match="requires an expert bias"):
        router(torch.randn(4, _DIM))


def test_the_histogram_is_not_checkpoint_state() -> None:
    router = _router()

    assert "quantile_balancer.required_bias_histogram_EB" not in router.state_dict()


def test_top_k_must_be_strictly_between_zero_and_num_experts() -> None:
    # top_k == num_experts leaves no (K+1)-th score to cut off at.
    with pytest.raises(ValueError, match="top_k"):
        QuantileBalancedTopKRouter(_NUM_EXPERTS, _DIM, _NUM_EXPERTS)


# -- the routing decision ------------------------------------------------------


def test_routing_uses_biased_choice_and_unbiased_weight() -> None:
    """The bias picks the experts; the weight is the bias-free score."""
    router = _router().eval()
    x_TD = torch.randn(6, _DIM)
    expert_bias_E = torch.tensor([10.0, 0.0, 0.0, 0.0])

    topk_scores_TK, topk_expert_ids_TK, _ = router(x_TD, expert_bias_E)

    assert torch.equal(topk_expert_ids_TK[:, 0], torch.zeros(6, dtype=torch.int64))
    with torch.no_grad():
        scores_TE = torch.sigmoid(router.gate(x_TD))
    torch.testing.assert_close(topk_scores_TK, scores_TE.gather(-1, topk_expert_ids_TK))


def test_a_training_forward_routes_like_the_base_router_but_observes() -> None:
    """Same ids as a plain biased top-k, one Top-(K+1), histogram filled."""
    torch.manual_seed(0)
    base = TokenChoiceTopKRouter(_NUM_EXPERTS, _DIM, _TOP_K)
    quantile = _router()
    with torch.no_grad():
        quantile.gate.weight.copy_(base.gate.weight)
    base.train()
    quantile.train()
    x_TD = torch.randn(8, _DIM)
    expert_bias_E = torch.tensor([0.3, -0.1, 0.0, 0.1])

    _, base_ids_TK, _ = base(x_TD, expert_bias_E)
    _, quantile_ids_TK, _ = quantile(x_TD, expert_bias_E)

    # sorted=True on the quantile path vs sorted=False on the base path: the
    # *sets* of chosen experts must agree per token.
    torch.testing.assert_close(
        quantile_ids_TK.sort(dim=-1).values, base_ids_TK.sort(dim=-1).values
    )
    histogram_EB = quantile.quantile_balancer.required_bias_histogram_EB
    # Every (token, expert) pair lands in exactly one bin.
    assert int(histogram_EB.sum()) == 8 * _NUM_EXPERTS


def test_eval_observes_nothing_and_matches_the_base_path_exactly() -> None:
    """No accumulation window is open in eval, so no histogram fill."""
    torch.manual_seed(0)
    base = TokenChoiceTopKRouter(_NUM_EXPERTS, _DIM, _TOP_K)
    quantile = _router()
    with torch.no_grad():
        quantile.gate.weight.copy_(base.gate.weight)
    base.eval()
    quantile.eval()
    x_TD = torch.randn(8, _DIM)
    expert_bias_E = torch.tensor([0.3, -0.1, 0.0, 0.1])

    _, base_ids_TK, _ = base(x_TD, expert_bias_E)
    _, quantile_ids_TK, _ = quantile(x_TD, expert_bias_E)

    torch.testing.assert_close(
        quantile_ids_TK.sort(dim=-1).values, base_ids_TK.sort(dim=-1).values
    )
    assert int(quantile.quantile_balancer.required_bias_histogram_EB.sum()) == 0


# -- the quantile solve ---------------------------------------------------------


def test_the_quantile_estimate_matches_the_worked_example() -> None:
    """The upstream worked example: E=4, K=2, 10 bins, zero bias.

    Every token has scores [0.92, 0.68, 0.31, 0.07], so the biased top-3
    cutoff is 0.31 and the required biases are [-0.61, -0.37, 0, 0.24].
    Over the [-1, 1] range with bin width 0.2 those land in bins 1, 3, 5, 6,
    the K/E quantile sits mid-bin, and mean-centring gives the bias below.
    """
    router = QuantileBalancedTopKRouter(_NUM_EXPERTS, _NUM_EXPERTS, _TOP_K, num_bins=10)
    router.train()
    with torch.no_grad():
        router.gate.weight.copy_(torch.eye(_NUM_EXPERTS))
    scores_TE = torch.tensor([0.92, 0.68, 0.31, 0.07]).expand(4, -1)
    expert_bias_E = torch.zeros(_NUM_EXPERTS)

    _, topk_expert_ids_TK, _ = router(torch.logit(scores_TE), expert_bias_E)

    assert torch.equal(topk_expert_ids_TK, torch.tensor([[0, 1]] * 4))
    histogram_EB = router.quantile_balancer.required_bias_histogram_EB
    assert histogram_EB.sum(dim=-1).tolist() == [4, 4, 4, 4]
    torch.testing.assert_close(
        router.quantile_balancer.estimate_expert_bias(histogram_EB, expert_bias_E),
        torch.tensor([-0.55, -0.15, 0.25, 0.45]),
    )


def test_the_estimate_is_mean_centred() -> None:
    """``sum(expert_bias_E) == 0``: the bias shifts choices, not the output."""
    router = _router(num_bins=32)
    router.train()
    x_TD = torch.randn(16, _DIM)
    expert_bias_E = torch.zeros(_NUM_EXPERTS)

    router(x_TD, expert_bias_E)
    next_bias_E = router.quantile_balancer.estimate_expert_bias(
        router.quantile_balancer.required_bias_histogram_EB, expert_bias_E
    )

    assert abs(float(next_bias_E.sum())) < 1e-6


def test_the_update_pushes_load_toward_balance() -> None:
    """A gate that always picks expert 0 must see its bias fall -- and the
    next forward, routed on the new bias, must spread tokens wider."""
    router = _router(num_bins=100)
    router.train()
    with torch.no_grad():
        # Expert 0 dominates every token's score.
        router.gate.weight.zero_()
        router.gate.weight[0].fill_(4.0)
    x_TD = torch.randn(32, _DIM)
    expert_bias_E = torch.zeros(_NUM_EXPERTS)

    router(x_TD, expert_bias_E)
    next_bias_E = router.quantile_balancer.estimate_expert_bias(
        router.quantile_balancer.required_bias_histogram_EB, expert_bias_E
    )

    assert float(next_bias_E[0]) < 0, "the over-loaded expert must lose bias"
    assert float(next_bias_E[1:].max()) > 0, "an idle expert must gain bias"

    _, _, routing_map_TE = router(x_TD, next_bias_E)
    counts_E = routing_map_TE.sum(dim=0)
    assert int(counts_E[0]) < 32, "the bias did not move any load off expert 0"
    assert int((counts_E[1:] > 0).sum()) > 0


# -- the hook -----------------------------------------------------------------


def test_balancing_entry_selects_sign_update_without_quantile_routers() -> None:
    model = _Holder(
        [
            MoE(
                _NUM_EXPERTS,
                RoutedExperts(
                    GroupedExperts(_DIM, 8, _NUM_EXPERTS),
                    LocalTokenDispatcher(_NUM_EXPERTS, _TOP_K),
                ),
                TokenChoiceTopKRouter(_NUM_EXPERTS, _DIM, _TOP_K),
                load_balance_coeff=0.1,
            )
        ]
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)

    register_moe_balancing_hook(optimizer, model_parts=[model], parallel_dims=None)

    assert len(optimizer._optimizer_step_pre_hooks) == 1
    moe = model.layers[0].mlp
    counts = torch.arange(_NUM_EXPERTS, dtype=moe.tokens_per_expert_E.dtype)
    moe.tokens_per_expert_E.copy_(counts)
    previous_bias = moe.expert_bias_E.clone()
    expected_bias = previous_bias + 0.1 * (counts.float().mean() - counts).sign()
    optimizer.step()
    torch.testing.assert_close(moe.expert_bias_E, expected_bias)
    assert moe.tokens_per_expert_E.count_nonzero() == 0


def test_a_mixed_model_is_rejected() -> None:
    """Quantile on some layers and sign-based on others is not a coherent
    scheme; fail fast rather than balance different layers differently."""
    sign_based = MoE(
        _NUM_EXPERTS,
        RoutedExperts(
            GroupedExperts(_DIM, 8, _NUM_EXPERTS),
            LocalTokenDispatcher(_NUM_EXPERTS, _TOP_K),
        ),
        TokenChoiceTopKRouter(_NUM_EXPERTS, _DIM, _TOP_K),
        load_balance_coeff=0.1,
    )
    model = _Holder([_moe(), sign_based])
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)

    with pytest.raises(ValueError, match="mutually exclusive"):
        register_moe_balancing_hook(optimizer, model_parts=[model], parallel_dims=None)

    assert optimizer._optimizer_step_pre_hooks == {}


def test_quantile_histogram_accumulation_exceeds_int32_without_wrapping():
    from llmtuner.models.common.moe.routing import QuantileBalancer

    balancer = QuantileBalancer(2, 1, 4)
    histogram = balancer.required_bias_histogram_EB
    assert histogram.dtype == torch.int64
    histogram[:, 2] = torch.iinfo(torch.int32).max
    balancer.observe(torch.full((1, 2), 0.5), torch.full((1, 1), 0.5), torch.zeros(2))
    assert histogram[:, 2].tolist() == [2147483648, 2147483648]
    assert histogram.sum().item() == 4294967296


def test_empty_quantile_observations_preserve_existing_bias_exactly():
    from llmtuner.models.common.moe.routing import QuantileBalancer

    balancer = QuantileBalancer(4, 2, 32)
    original = torch.tensor([-0.31, -0.12, 0.17, 0.26])
    current = original.clone()
    for _ in range(100):
        current = balancer.estimate_expert_bias(
            balancer.required_bias_histogram_EB, current
        )
    assert torch.equal(current, original)
    assert current.data_ptr() != original.data_ptr()


def test_the_hook_solves_the_bias_and_drains_the_scratch() -> None:
    """One step: the bias is overwritten by the quantile estimate, and both
    the histogram and the token counter are drained for the next window."""
    moe = MoE(
        _NUM_EXPERTS,
        RoutedExperts(
            GroupedExperts(_NUM_EXPERTS, 8, _NUM_EXPERTS),
            LocalTokenDispatcher(_NUM_EXPERTS, _TOP_K),
        ),
        QuantileBalancedTopKRouter(_NUM_EXPERTS, _NUM_EXPERTS, _TOP_K, num_bins=10),
        load_balance_coeff=None,
    )
    moe.train()
    model = _Holder([moe])
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    register_moe_balancing_hook(optimizer, model_parts=[model], parallel_dims=None)

    scores_TE = torch.tensor([0.92, 0.68, 0.31, 0.07]).expand(4, -1)
    with torch.no_grad():
        moe.router.gate.weight.copy_(torch.eye(_NUM_EXPERTS))
    moe.router(torch.logit(scores_TE), moe.expert_bias_E)
    moe.tokens_per_expert_E.copy_(torch.tensor([8.0, 8.0, 0.0, 0.0]))

    optimizer.step()

    torch.testing.assert_close(
        moe.expert_bias_E, torch.tensor([-0.55, -0.15, 0.25, 0.45])
    )
    assert int(moe.router.quantile_balancer.required_bias_histogram_EB.sum()) == 0
    assert float(moe.tokens_per_expert_E.sum()) == 0.0


def test_the_collective_runs_over_a_real_process_group(single_rank_group) -> None:
    """The reduction path executes without a mesh -- e.g. before one exists."""
    moe = _moe()
    moe.router.quantile_balancer.required_bias_histogram_EB.fill_(1)

    update_quantile_expert_bias([moe], parallel_dims=None)

    # A size-1 group leaves the histogram alone; the estimate runs off it.
    assert int(moe.router.quantile_balancer.required_bias_histogram_EB.sum()) == 0
    assert abs(float(moe.expert_bias_E.sum())) < 1e-6


def test_every_layer_of_every_part_is_updated() -> None:
    a, b, c = _moe(), _moe(), _moe()
    parts = [_Holder([a, b]), _Holder([c])]
    for moe in (a, b, c):
        moe.router.quantile_balancer.required_bias_histogram_EB.fill_(1)
        moe.tokens_per_expert_E.fill_(3.0)

    all_layers = [
        moe for part in parts for moe in part.modules() if isinstance(moe, MoE)
    ]
    update_quantile_expert_bias(all_layers, parallel_dims=None)

    for moe in (a, b, c):
        assert int(moe.router.quantile_balancer.required_bias_histogram_EB.sum()) == 0
        assert float(moe.tokens_per_expert_E.sum()) == 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_experts": 0}, "must be positive"),
        ({"dim": 0}, "must be positive"),
        ({"top_k": 0}, "top_k must"),
        ({"top_k": 9}, "top_k must"),
        ({"num_expert_groups": 0}, "num_expert_groups must"),
        ({"num_limited_groups": 1}, "requires num_expert_groups"),
        ({"num_expert_groups": 2, "num_limited_groups": 0}, "must be positive"),
        ({"num_expert_groups": 2, "num_limited_groups": 1, "top_k": 5}, "capacity"),
    ],
)
def test_router_rejects_invalid_expert_capacity(kwargs, match):
    from llmtuner.models.common.moe.routing import TokenChoiceTopKRouter

    with pytest.raises(ValueError, match=match):
        TokenChoiceTopKRouter(**({"num_experts": 8, "dim": 4} | kwargs))


@pytest.mark.parametrize("mode", ["full", "selective"])
@pytest.mark.parametrize("microbatches", [1, 3])
def test_quantile_bias_update_matches_with_checkpoint_recomputation(mode, microbatches):
    import copy

    from llmtuner.config.activation_checkpoint_config import SelectiveACConfig
    from llmtuner.memory.activation_checkpoint import wrap_full, wrap_selective

    torch.manual_seed(31)
    original = MoE(
        4,
        RoutedExperts(
            GroupedExperts(4, 8, 4, use_grouped_mm=False),
            LocalTokenDispatcher(4, 2),
        ),
        QuantileBalancedTopKRouter(4, 4, 2, num_bins=32),
        load_balance_coeff=None,
    )
    for parameter in original.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    replayed = copy.deepcopy(original)
    wrapped = (
        wrap_full(replayed)
        if mode == "full"
        else wrap_selective(replayed, SelectiveACConfig())
    )
    padding = torch.tensor([False, True, False, False, True, False, False])
    original.set_padding_mask(padding)
    replayed.set_padding_mask(padding)
    for _ in range(microbatches):
        inputs = torch.randn(7, 4)
        baseline = inputs.clone().requires_grad_()
        checkpointed = inputs.clone().requires_grad_()
        expected = original(baseline)
        actual = wrapped(checkpointed)
        torch.testing.assert_close(actual, expected)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(checkpointed.grad, baseline.grad)
    for left, right in zip(original.parameters(), replayed.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)
    before = original.router.quantile_balancer.required_bias_histogram_EB
    after = replayed.router.quantile_balancer.required_bias_histogram_EB
    assert int(before.sum()) == int((~padding).sum()) * 4 * microbatches
    assert int(after.sum()) > int(before.sum()), "Backward must replay routing"
    update_quantile_expert_bias([original], parallel_dims=None)
    update_quantile_expert_bias([replayed], parallel_dims=None)
    torch.testing.assert_close(replayed.expert_bias_E, original.expert_bias_E)
    assert not replayed.router.quantile_balancer.required_bias_histogram_EB.any()
    assert not replayed.tokens_per_expert_E.any()


def test_quantile_interpolation_preserves_fractional_large_token_counts():
    from llmtuner.models.common.moe.routing import QuantileBalancer

    histogram = torch.tensor(
        [[8388609, 1, 0, 8388609], [8388609, 0, 1, 8388608]], dtype=torch.int64
    )
    bias = torch.zeros(2)
    actual = QuantileBalancer(2, 1, 4).estimate_expert_bias(histogram, bias)
    assert actual.dtype == bias.dtype
    torch.testing.assert_close(actual, torch.tensor([0.125, -0.125]), rtol=0, atol=0)


def test_quantile_update_reduces_different_histogram_shapes(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.models.common.moe import bias_update

    layers = []
    expected = []
    for experts, bins in [(4, 16), (8, 32)]:
        layer = MoE(
            experts,
            RoutedExperts(
                GroupedExperts(4, 8, experts, use_grouped_mm=False),
                LocalTokenDispatcher(experts, 2),
            ),
            QuantileBalancedTopKRouter(experts, 4, 2, num_bins=bins),
            load_balance_coeff=None,
        )
        balancer = layer.router.quantile_balancer
        histogram = balancer.required_bias_histogram_EB
        histogram.copy_(torch.arange(histogram.numel()).reshape(histogram.shape) % 7)
        expected.append(
            balancer.estimate_expert_bias(histogram * 4, layer.expert_bias_E)
        )
        layer.tokens_per_expert_E.fill_(3)
        layers.append(layer)

    groups = {axis: object() for axis in ("dp", "cp", "tp")}
    calls = []

    def reduce(statistics, *, group):
        calls.append(group)
        statistics.mul_(2)

    topology = SimpleNamespace(
        ep_enabled=False,
        get_optional_mesh=lambda axis: SimpleNamespace(get_group=lambda: groups[axis]),
    )
    monkeypatch.setattr(bias_update, "all_reduce", reduce)
    update_quantile_expert_bias(layers, parallel_dims=topology)

    assert calls == [groups["dp"], groups["cp"]]
    for layer, reference in zip(layers, expected, strict=True):
        torch.testing.assert_close(layer.expert_bias_E, reference)
        assert not layer.router.quantile_balancer.required_bias_histogram_EB.any()
        assert not layer.tokens_per_expert_E.any()


def test_quantile_bias_update_with_no_layers_is_a_noop(monkeypatch):
    from llmtuner.models.common.moe import bias_update

    def unexpected_collective(*args, **kwargs):
        raise AssertionError("empty layer list must not start a collective")

    monkeypatch.setattr(bias_update, "all_reduce", unexpected_collective)
    update_quantile_expert_bias([], parallel_dims=None)
