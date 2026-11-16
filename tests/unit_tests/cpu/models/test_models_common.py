"""Shared MoE router projection, routing and grouped expert behavior."""

from __future__ import annotations

from tests.caps import require_env

require_env("spmd_types")


import pytest
import torch
import torch.nn as nn

from llmtuner.models.common.moe.experts import GroupedExperts
from llmtuner.models.common.linear import RouterGateLinear
from llmtuner.models.common.moe.routing import TokenChoiceTopKRouter


# -- RouterGateLinear --------------------------------------------------------


def test_router_gate_returns_fp32_from_a_bf16_input() -> None:
    torch.manual_seed(0)
    gate = RouterGateLinear(8, 4).to(torch.bfloat16)
    scores = gate(torch.randn(3, 8, dtype=torch.bfloat16))

    assert scores.dtype is torch.float32
    assert scores.shape == (3, 4)


def test_router_gate_returns_fp32_from_an_fp32_input() -> None:
    gate = RouterGateLinear(8, 4)
    scores = gate(torch.randn(3, 8))

    assert scores.dtype is torch.float32


def test_router_gate_gradients_are_fp32_in_both_directions() -> None:
    """The custom Function exists to pin the BACKWARD too, not just the forward.

    Without it autograd would derive a bf16 backward from a bf16 input, which is
    the path the router's top-k ordering is most sensitive to.
    """
    torch.manual_seed(0)
    gate = RouterGateLinear(8, 4).to(torch.bfloat16)
    x = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)

    gate(x).sum().backward()

    assert x.grad.dtype is torch.bfloat16
    assert gate.weight.grad is not None


def test_router_gate_matches_a_plain_fp32_linear() -> None:
    """The result is the same projection, computed in fp32."""
    torch.manual_seed(0)
    gate = RouterGateLinear(8, 4)
    plain = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        plain.weight.copy_(gate.weight)

    x = torch.randn(3, 8)
    torch.testing.assert_close(gate(x), plain(x.float()), rtol=0, atol=0)


def test_router_gate_float64_inputs_follow_explicit_fp32_projection_contract():
    gate = RouterGateLinear(3, 2).double()
    with torch.no_grad():
        gate.weight.copy_(
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64)
        )
    inputs = torch.tensor(
        [[1.0000000001, 2.0000000001, 3.0000000001]],
        dtype=torch.float64,
        requires_grad=True,
    )
    output = gate(inputs)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, inputs.float() @ gate.weight.float().T)
    output.sum().backward()
    expected_input_grad = torch.ones_like(output) @ gate.weight.detach().float()
    expected_weight_grad = torch.ones_like(output).T @ inputs.detach().float()
    assert inputs.grad.dtype == torch.float64
    assert gate.weight.grad.dtype == torch.float64
    torch.testing.assert_close(
        inputs.grad, expected_input_grad.double(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        gate.weight.grad, expected_weight_grad.double(), rtol=0, atol=0
    )


def test_router_gate_is_the_moe_router_projection() -> None:
    """One class, imported by the MoE router -- not a second copy of it."""
    from llmtuner.models.common.moe.routing import RouterGateLinear as FromRouter

    assert FromRouter is RouterGateLinear


# -- node-limited routing (DeepSeek-V3) ----------------------------------------

_E, _D, _K = 8, 16, 2


def _router(**kw) -> TokenChoiceTopKRouter:
    torch.manual_seed(0)
    return TokenChoiceTopKRouter(_E, _D, _K, **kw)


def test_group_limited_routing_confines_every_token_to_the_chosen_groups() -> None:
    """The whole point of the restriction: no token may reach an unchosen group.

    ``n_group=2`` splits the 8 experts into ``{0..3}`` and ``{4..7}``; with
    ``topk_group=1`` every token must draw its K experts from one of them. This
    is what bounds inter-node traffic when a group is a node, so a silent
    regression here would be a performance bug that still trains correctly --
    exactly the kind that never gets noticed.
    """
    router = _router(num_expert_groups=2, num_limited_groups=1)
    x = torch.randn(64, _D)

    ids = router._select_experts(torch.sigmoid(router.gate(x)))

    group_of = ids // (_E // 2)
    assert bool((group_of[:, :1] == group_of).all()), "a token crossed groups"


def test_without_grouping_tokens_are_free_to_cross_groups() -> None:
    """The non-vacuity check for the test above.

    Same weights, same tokens, grouping off: if the restriction were being
    applied unconditionally -- or if the grouping flags were ignored -- this
    would still show every token in one group, and the test above would prove
    nothing.
    """
    router = _router()
    x = torch.randn(64, _D)

    ids = router._select_experts(torch.sigmoid(router.gate(x)))

    group_of = ids // (_E // 2)
    assert not bool((group_of[:, :1] == group_of).all())


def test_the_group_score_is_the_sum_of_the_two_best_experts() -> None:
    """A group wins on its strongest *pair*, not its single best expert.

    DeepSeek-V3 Sec 2.1.1 scores a group by its top-2 sum, so a group with two
    solid experts beats one holding a single outlier. A ``max`` or ``mean`` rule
    would pick the other group here -- which is why this is pinned on the ids
    rather than on a property the two rules share.

    Groups are ``{0,1,2,3}`` and ``{4,5,6,7}``. Group 1 holds the highest single
    expert (0.90) but group 0 wins on the pair: its top two are 0.80 and 0.70,
    summing to 1.50 against group 1's 0.90 + 0.20. Under a ``max`` rule group 1
    would win and the top-2 ids would come from it.
    """
    router = _router(num_expert_groups=2, num_limited_groups=1)
    scores = torch.tensor([[0.80, 0.70, 0.00, 0.00, 0.90, 0.20, 0.00, 0.00]])

    ids = router._select_experts(scores)

    assert ids.tolist() == [[0, 1]], (
        "group 0 (top-2 sum 1.50) must beat group 1 (1.10), despite 0.90 being "
        "the single largest expert"
    )


def test_the_bias_shifts_which_experts_win_but_not_their_scores() -> None:
    """The load-balancing bias is a routing device, never a value.

    ``forward`` gathers the weight from the *unbiased* scores, so a bias strong
    enough to change the selection must leave the score carried by a given
    expert untouched. Otherwise the bias would quietly rescale the MoE output --
    it is added to choose experts, not to weight them.
    """
    router = _router()
    x = torch.randn(32, _D)
    unbiased = torch.sigmoid(router.gate(x))

    base_scores, base_ids, _ = router(x)
    bias = torch.zeros(_E)
    bias[5] = 10.0
    steered_scores, steered_ids, _ = router(x, bias)

    assert int((steered_ids == 5).sum()) > int((base_ids == 5).sum()), (
        "bias did nothing"
    )

    # Every returned score is that expert's unbiased score, wherever it landed.
    expected = unbiased.gather(1, steered_ids)
    torch.testing.assert_close(steered_scores, expected, rtol=0, atol=0)
    # ...and the same for the unsteered run, so the check above is not vacuous.
    torch.testing.assert_close(
        base_scores, unbiased.gather(1, base_ids), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_expert_groups": 2}, "num_limited_groups must be set"),
        (
            {"num_expert_groups": 3, "num_limited_groups": 1},
            "must be divisible by num_expert_groups",
        ),
        (
            {"num_expert_groups": 2, "num_limited_groups": 5},
            "cannot exceed num_expert_groups",
        ),
        (
            {"num_expert_groups": 8, "num_limited_groups": 1},
            "must be >= 2",
        ),
    ],
)
def test_a_malformed_group_config_is_rejected(kwargs: dict, message: str) -> None:
    """Every one of these would otherwise fail deep inside a topk with a shape error."""
    with pytest.raises(ValueError, match=message):
        TokenChoiceTopKRouter(_E, _D, _K, **kwargs)


# -- GroupedExperts: the two expert-GEMM paths -------------------------------


@pytest.mark.parametrize("use_grouped_mm", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_empty_expert_batch_preserves_input_and_weight_gradients(
    use_grouped_mm, dtype, monkeypatch
):
    def grouped_mm(*args, **kwargs):
        pytest.fail("Empty expert batches must not invoke a grouped GEMM kernel")

    monkeypatch.setattr(torch, "_grouped_mm", grouped_mm, raising=False)
    # Weights remain fp32, as with mixed precision training.
    experts = GroupedExperts(8, 12, 4, use_grouped_mm=use_grouped_mm)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    inputs = torch.empty(0, 8, dtype=dtype, requires_grad=True)
    output = experts(inputs, torch.zeros(4, dtype=torch.long))
    assert output.shape == (0, 8)
    assert output.dtype == dtype
    output.sum().backward()
    assert inputs.grad is not None
    assert inputs.grad.shape == inputs.shape
    for parameter in experts.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def _grouped(*, use_grouped_mm: bool | None, dtype=torch.float32, seed: int = 0):
    """A real (not ``torch.empty``) GroupedExperts, so comparisons mean something."""
    torch.manual_seed(seed)
    module = GroupedExperts(
        dim=16, hidden_dim=32, num_experts=4, use_grouped_mm=use_grouped_mm
    ).to(dtype)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, 0.03)
    return module


_COUNTS = torch.tensor([5, 6, 4, 7])


@pytest.mark.parametrize(
    "counts, error_type, message",
    [
        (torch.ones(3, dtype=torch.long), ValueError, "shape"),
        (torch.ones(2, 2, dtype=torch.long), ValueError, "shape"),
        (torch.ones(4), TypeError, "integer dtype"),
        (torch.ones(4, dtype=torch.bool), TypeError, "integer dtype"),
        (torch.tensor([-1, 2, 1, 2]), RuntimeError, "nonnegative"),
        (torch.tensor([1, 1, 1, 0]), RuntimeError, "sum"),
        (torch.tensor([2, 1, 1, 1]), RuntimeError, "sum"),
        (
            torch.tensor([2**62, 2**62, 2**62, 2**62 + 4]),
            RuntimeError,
            "exceed the routed token count",
        ),
    ],
)
def test_expert_segment_counts_fail_before_any_projection(
    counts, error_type, message, monkeypatch
):
    module = _grouped(use_grouped_mm=False)
    inputs = torch.randn(4, 16)
    original = inputs.clone()
    monkeypatch.setattr(
        module, "_grouped_mm", lambda *args: pytest.fail("invalid counts reached GEMM")
    )
    with pytest.raises(error_type, match=message):
        module(inputs, counts)
    torch.testing.assert_close(inputs, original)
    assert all(parameter.grad is None for parameter in module.parameters())


@pytest.mark.parametrize("shape", [(16,), (4, 15), (2, 2, 16)])
def test_expert_inputs_require_a_routed_token_matrix(shape):
    module = _grouped(use_grouped_mm=False)
    with pytest.raises(ValueError, match="Expert inputs must have shape"):
        module(torch.ones(shape), torch.ones(4, dtype=torch.long))


def test_grouped_experts_runs_every_expert_over_its_own_tokens() -> None:
    """The loop path segments the tokens, one expert each, no bleeding.

    Nothing else here would catch a segmentation bug: every expert gets tokens,
    the output shape is right either way, and a wrong segment boundary just
    makes the numbers quietly wrong. So the reference is built by hand, one
    expert at a time.
    """
    module = _grouped(use_grouped_mm=False)
    x = torch.randn(22, 16)

    # Each expert's slice of the tokens, as (start, end) pairs.
    bounds = [0, 5, 11, 15, 22]
    segments = list(zip(bounds[:-1], bounds[1:], strict=True))

    gate = torch.cat([x[s:o] @ module.w1_EFD[e].T for e, (s, o) in enumerate(segments)])
    up = torch.cat([x[s:o] @ module.w3_EFD[e].T for e, (s, o) in enumerate(segments)])
    expected = torch.cat(
        [
            module.activation_fn(gate[s:o], up[s:o]) @ module.w2_EDF[e].T
            for e, (s, o) in enumerate(segments)
        ]
    )
    assert torch.allclose(module(x, _COUNTS), expected, rtol=1e-5, atol=1e-6)


def test_the_grouped_mm_probe_follows_the_op(monkeypatch) -> None:
    """The probe decides the default, so it has to track the op's real behavior.

    Driven by monkeypatching rather than by asking this machine: a test that
    only compares the probe against the local op passes on any host where the
    op works even if the probe is hard-wired to ``True`` -- which is precisely
    the bug that would send every model down a branch that raises elsewhere.
    """
    from llmtuner.accelerator.capabilities import probe

    probe.cache_clear()

    def _works(*args, **kwargs):
        return torch.zeros(8, 8, dtype=torch.bfloat16)

    def _raises(*args, **kwargs):
        raise RuntimeError("strides should be multiple of 16 bytes")

    monkeypatch.setattr(torch, "_grouped_mm", _works, raising=False)
    assert _grouped(use_grouped_mm=None).use_grouped_mm is True

    monkeypatch.setattr(torch, "_grouped_mm", _raises, raising=False)
    probe.cache_clear()
    assert _grouped(use_grouped_mm=None).use_grouped_mm is False

    # An op that is simply absent counts as unavailable, not as an error.
    monkeypatch.delattr(torch, "_grouped_mm", raising=False)
    probe.cache_clear()
    assert _grouped(use_grouped_mm=None).use_grouped_mm is False
    probe.cache_clear()


def test_the_grouped_mm_probe_agrees_with_the_real_op_here() -> None:
    """On whatever host this runs, the probe must match the actual call."""
    from llmtuner.accelerator.capabilities import has

    reported = has("torch_grouped_mm")
    try:
        torch._grouped_mm(
            torch.zeros(8, 8, dtype=torch.bfloat16),
            torch.zeros(2, 8, 8, dtype=torch.bfloat16),
            offs=torch.tensor([4, 8], dtype=torch.int32),
        )
        actually_works = True
    except Exception:
        actually_works = False

    assert reported == actually_works
    # And the decision holds up end to end: a model built with no explicit flag
    # must survive the forward whichever branch the probe chose.
    module = _grouped(use_grouped_mm=None, dtype=torch.bfloat16)
    module(torch.randn(22, 16, dtype=torch.bfloat16), _COUNTS)


def test_the_fused_path_matches_the_loop_bit_for_bit_in_bf16() -> None:
    """Where the op is available, the two paths must agree exactly.

    They are the same arithmetic in the same dtype (bf16 in, bf16 out, no
    intermediate widening on either side), so "close" would be the wrong
    assertion and would hide a mis-segmented expert. Skipped where the op is
    unavailable.
    """
    from llmtuner.accelerator.capabilities import has

    if not has("torch_grouped_mm"):
        pytest.skip("torch._grouped_mm is unavailable on this build")

    loop = _grouped(use_grouped_mm=False, dtype=torch.bfloat16)
    fused = _grouped(use_grouped_mm=True, dtype=torch.bfloat16)
    fused.load_state_dict(loop.state_dict())
    x = torch.randn(22, 16, dtype=torch.bfloat16)

    assert torch.equal(loop(x, _COUNTS), fused(x, _COUNTS))


def test_the_fused_path_is_refused_for_a_wider_dtype() -> None:
    """The bf16-only kernel must not be handed fp32 or fp64 activations.

    ``torch._grouped_mm`` casts both operands to bf16. On an fp32 model that is
    a silent 8-mantissa-bit round trip -- measured at 1.4e-1 relative error --
    which is exactly what a loss-comparison gate cannot see.

    The comparison is explicitly loop-against-fused, both with the flag forced:
    on a build where the op is available, ``use_grouped_mm=None`` resolves to
    *fused*, so comparing a default against a forced fused model would compare
    the same path with itself and pass no matter what the gate did.
    """
    for dtype in (torch.float16, torch.float32, torch.float64):
        loop = _grouped(use_grouped_mm=False, dtype=dtype)
        forced = _grouped(use_grouped_mm=True, dtype=dtype)
        forced.load_state_dict(loop.state_dict())
        x = torch.randn(22, 16, dtype=dtype)

        # Same weights, same dtype: the loop is the reference and the fused
        # request must not reach the kernel, so these are bitwise equal.
        assert torch.equal(loop(x, _COUNTS), forced(x, _COUNTS))


def test_the_default_follows_the_probe() -> None:
    """``None`` means "decide here", and the decision is the probe's."""
    from llmtuner.accelerator.capabilities import has

    assert _grouped(use_grouped_mm=None).use_grouped_mm == has("torch_grouped_mm")
    # An explicit value still wins, which is what lets a test pin a path.
    assert _grouped(use_grouped_mm=False).use_grouped_mm is False
    assert _grouped(use_grouped_mm=True).use_grouped_mm is True


@pytest.mark.parametrize("num_bins", [0, -1])
def test_quantile_balancer_rejects_nonpositive_histogram_size(num_bins):
    from llmtuner.models.common.moe.routing import QuantileBalancer

    with pytest.raises(ValueError, match="num_bins must be positive"):
        QuantileBalancer(4, 1, num_bins)


@pytest.mark.parametrize("sizes", [(0, 8, 4), (8, 0, 4), (8, 8, 0)])
def test_grouped_experts_rejects_empty_geometry(sizes):
    with pytest.raises(ValueError, match="must be positive"):
        GroupedExperts(*sizes, use_grouped_mm=False)


def test_router_fp32_scores_and_gradients_are_preserved_under_autocast():
    torch.manual_seed(29)
    router = RouterGateLinear(4, 3)
    x = torch.randn(2, 4, requires_grad=True)
    expected = router(x)
    expected.sum().backward()
    expected_input_grad = x.grad.clone()
    expected_weight_grad = router.weight.grad.clone()
    router.zero_grad(set_to_none=True)
    x.grad = None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = router(x)
        actual.sum().backward()
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.equal(x.grad, expected_input_grad)
    assert torch.equal(router.weight.grad, expected_weight_grad)


@pytest.mark.parametrize("field", ["dim", "hidden_dim", "num_experts"])
@pytest.mark.parametrize("value", [True, False, 1.5, "2"])
def test_grouped_experts_rejects_noninteger_dimensions(field, value):
    options = {"dim": 4, "hidden_dim": 8, "num_experts": 2}
    options[field] = value
    with pytest.raises(ValueError, match=field + " must be positive integer"):
        GroupedExperts(**options)


@pytest.mark.parametrize("value", [1, "false", 0.0])
def test_grouped_experts_rejects_nonboolean_backend_override(value):
    with pytest.raises(TypeError, match="use_grouped_mm must be bool or None"):
        GroupedExperts(4, 8, 2, use_grouped_mm=value)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8])
def test_expert_small_integer_counts_allow_more_rows_than_dtype_range(dtype):
    module = _grouped(use_grouped_mm=False)
    inputs = torch.randn(300, 16)
    counts = torch.tensor([100, 100, 50, 50], dtype=dtype)
    actual = module(inputs, counts)
    expected = module(inputs, counts.to(torch.int64))
    assert actual.shape == (300, 16)
    assert actual.isfinite().all() and actual.count_nonzero() > 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
