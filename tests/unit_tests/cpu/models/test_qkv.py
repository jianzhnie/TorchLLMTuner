"""Fused QKV projection: layout, forward, and checkpoint interop.

The fused weight is not a plain concatenation of Q, K and V -- it is grouped by
KV head, so each group holds ``[q_heads..., k_head, v_head]``. Getting that
grouping wrong produces a correctly-shaped tensor that silently pairs each query
head with the wrong key/value head, which attention will happily train on. So the
tests below assert the *ordering*, not just the shapes.

The save/load hooks are the other half: they exist so an external checkpoint
written with separate ``wq``/``wk``/``wv`` tensors can be loaded at all. A
split/merge pair that is not a perfect round-trip would corrupt weights on every
resume, so that round-trip is asserted directly.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("spmd_types")


import pytest
import torch

from llmtuner.models.common.attention.qkv import QKVLinear, local_head_split

DIM = 16
HEAD_DIM = 4
N_HEADS = 8
N_KV_HEADS = 2
# n_kv_heads * (heads_per_kv + 2) * head_dim
FUSED_OUT = N_KV_HEADS * (N_HEADS // N_KV_HEADS + 2) * HEAD_DIM


def _qkv(*, bias: bool = False, seed: int = 0) -> QKVLinear:
    torch.manual_seed(seed)
    linear = torch.nn.Linear(DIM, FUSED_OUT, bias=bias)
    return QKVLinear(
        head_dim=HEAD_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, linear=linear
    )


def _tokens(seq: int = 6, seed: int = 1) -> torch.Tensor:
    return torch.randn(seq, DIM, generator=torch.Generator().manual_seed(seed))


# -- construction ------------------------------------------------------------


def test_empty_token_projection_preserves_head_layout_and_gradients():
    module = _qkv(bias=True)
    inputs = torch.empty(0, DIM, requires_grad=True)
    q, k, v = module(inputs)
    assert q.shape == (0, N_HEADS, HEAD_DIM)
    assert k.shape == v.shape == (0, N_KV_HEADS, HEAD_DIM)
    (q.sum() + k.sum() + v.sum()).backward()
    assert inputs.grad is not None and inputs.grad.shape == inputs.shape
    torch.testing.assert_close(
        module.wqkv.weight.grad, torch.zeros_like(module.wqkv.weight)
    )


def test_local_head_split_accepts_empty_token_axes():
    inputs = torch.empty(2, 0, N_HEADS * HEAD_DIM)
    assert local_head_split(inputs, HEAD_DIM).shape == (2, 0, N_HEADS, HEAD_DIM)


@pytest.mark.parametrize("leading_shape", [(), (5,), (2, 5), (2, 3, 5), (2, 0)])
def test_qkv_preserves_leading_axes_and_projection_gradients(leading_shape):
    module = _qkv(bias=True).double()
    inputs = torch.randn(*leading_shape, DIM, dtype=torch.float64, requires_grad=True)
    actual = module(inputs)
    # Independent per-projection GEMMs use the logical checkpoint weights.
    logical = module.state_dict(keep_vars=True)
    expected = tuple(
        torch.nn.functional.linear(
            inputs, logical[f"{name}.weight"], logical[f"{name}.bias"]
        ).unflatten(-1, (heads, HEAD_DIM))
        for name, heads in (("wq", N_HEADS), ("wk", N_KV_HEADS), ("wv", N_KV_HEADS))
    )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)
        assert value.is_contiguous()
    parameters = (inputs, module.wqkv.weight, module.wqkv.bias)
    actual_grads = torch.autograd.grad(
        sum(value.square().sum() for value in actual), parameters
    )
    expected_grads = torch.autograd.grad(
        sum(value.square().sum() for value in expected), parameters
    )
    for value, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(value, reference)


def test_r_dim_is_heads_per_kv_plus_two() -> None:
    """R bundles a KV group's Q heads with its one K and one V head."""
    qkv = _qkv()
    assert qkv.heads_per_kv == N_HEADS // N_KV_HEADS
    assert qkv.r_dim == qkv.heads_per_kv + 2


def test_fused_projection_is_sized_for_every_group() -> None:
    qkv = _qkv()
    assert qkv.wqkv.weight.shape == (FUSED_OUT, DIM)
    assert FUSED_OUT == N_KV_HEADS * qkv.r_dim * HEAD_DIM


def test_non_divisible_kv_heads_is_rejected() -> None:
    """A fractional heads-per-KV-group would silently truncate the reshape."""
    linear = torch.nn.Linear(DIM, FUSED_OUT)
    with pytest.raises(ValueError, match="divisible"):
        QKVLinear(head_dim=HEAD_DIM, n_heads=7, n_kv_heads=3, linear=linear)


# -- forward -----------------------------------------------------------------


def test_forward_returns_three_tensors_of_the_expected_shape() -> None:
    qkv = _qkv()
    x = _tokens()
    with torch.no_grad():
        q, k, v = qkv(x)
    assert q.shape == (x.shape[0], N_HEADS, HEAD_DIM)
    assert k.shape == (x.shape[0], N_KV_HEADS, HEAD_DIM)
    assert v.shape == (x.shape[0], N_KV_HEADS, HEAD_DIM)


def test_forward_extracts_the_grouped_layout() -> None:
    """Q/K/V must come from their own slots inside each KV group.

    Rebuilds the expected tensors by reshaping the fused weight the same way the
    module documents, then compares -- so a changed grouping order fails here
    rather than showing up as a subtly wrong loss.
    """
    qkv = _qkv()
    x = _tokens()
    hpk, r, hd = qkv.heads_per_kv, qkv.r_dim, HEAD_DIM

    fused = (x @ qkv.wqkv.weight.T).reshape(x.shape[0], N_KV_HEADS, r, hd)
    expected_q = fused[:, :, :hpk].reshape(x.shape[0], -1, hd)
    expected_k = fused[:, :, hpk].reshape(x.shape[0], -1, hd)
    expected_v = fused[:, :, hpk + 1].reshape(x.shape[0], -1, hd)

    with torch.no_grad():
        q, k, v = qkv(x)

    assert torch.equal(q, expected_q.contiguous())
    assert torch.equal(k, expected_k.contiguous())
    assert torch.equal(v, expected_v.contiguous())


def test_query_head_order_follows_the_kv_grouping() -> None:
    """Not a flattened Q block: group 0's heads come before group 1's.

    A plain concatenation of [all Q | all K | all V] would also produce a
    plausible Q tensor, so this pins the interleaved order specifically.
    """
    qkv = _qkv()
    x = _tokens()
    hpk, r, hd = qkv.heads_per_kv, qkv.r_dim, HEAD_DIM

    fused = (x @ qkv.wqkv.weight.T).reshape(x.shape[0], N_KV_HEADS, r, hd)
    with torch.no_grad():
        q, _, _ = qkv(x)

    # Head h of the output corresponds to KV group h // hpk, offset h % hpk.
    for h in range(N_HEADS):
        group, offset = divmod(h, hpk)
        assert torch.equal(q[:, h], fused[:, group, offset])


def test_forward_is_contiguous() -> None:
    """Attention and KV-cache kernels read raw memory, so views are not enough."""
    qkv = _qkv()
    with torch.no_grad():
        q, k, v = qkv(_tokens())
    for tensor in (q, k, v):
        assert tensor.is_contiguous()


def test_forward_accepts_a_bias() -> None:
    qkv = _qkv(bias=True)
    assert qkv.wqkv.bias is not None
    with torch.no_grad():
        q, k, v = qkv(_tokens())
    assert q.shape[0] == k.shape[0] == v.shape[0]


# -- checkpoint interop ------------------------------------------------------


def test_state_dict_exposes_separate_q_k_v_weights() -> None:
    """External checkpoints store wq/wk/wv; the fused form must present them."""
    qkv = _qkv(bias=True)
    state = dict(qkv.state_dict())

    assert "wqkv.weight" not in state
    for name in (
        "wq.weight",
        "wk.weight",
        "wv.weight",
        "wq.bias",
        "wk.bias",
        "wv.bias",
    ):
        assert name in state, f"missing {name}"


def test_saved_q_k_v_have_the_expected_shapes() -> None:
    qkv = _qkv()
    state = dict(qkv.state_dict())

    assert state["wq.weight"].shape == (N_HEADS * HEAD_DIM, DIM)
    assert state["wk.weight"].shape == (N_KV_HEADS * HEAD_DIM, DIM)
    assert state["wv.weight"].shape == (N_KV_HEADS * HEAD_DIM, DIM)


def test_saved_q_k_v_match_the_fused_grouping() -> None:
    """The split must invert the same grouping the forward reads."""
    qkv = _qkv()
    weight = qkv.wqkv.weight.detach().clone()
    hpk, r, hd = qkv.heads_per_kv, qkv.r_dim, HEAD_DIM

    grouped = weight.reshape(N_KV_HEADS, r, hd, DIM)
    state = dict(qkv.state_dict())

    assert torch.equal(state["wq.weight"], grouped[:, :hpk].reshape(-1, DIM))
    assert torch.equal(state["wk.weight"], grouped[:, hpk].reshape(-1, DIM))
    assert torch.equal(state["wv.weight"], grouped[:, hpk + 1].reshape(-1, DIM))


def test_split_then_merge_is_a_round_trip() -> None:
    """Load must reconstruct exactly what save produced, or resumes corrupt."""
    source = _qkv(bias=True)
    original = source.wqkv.weight.detach().clone()

    target = _qkv(bias=True, seed=99)
    assert not torch.equal(target.wqkv.weight, original)

    target.load_state_dict(source.state_dict())

    assert torch.equal(target.wqkv.weight, original)


def test_forward_is_unchanged_by_a_save_load_round_trip() -> None:
    """The end-to-end claim: same weights in, same Q/K/V out."""
    source = _qkv(seed=3)
    x = _tokens()
    with torch.no_grad():
        before = source(x)

    target = _qkv(seed=42)
    target.load_state_dict(source.state_dict())
    with torch.no_grad():
        after = target(x)

    for a, b in zip(before, after, strict=True):
        assert torch.equal(a, b)


def test_loading_without_all_three_keys_leaves_the_fused_weight_alone() -> None:
    """A partial state dict must not half-merge into a corrupt fused tensor."""
    qkv = _qkv()
    original = qkv.wqkv.weight.detach().clone()

    # Only wq present -- the merge hook requires all three.
    qkv.load_state_dict(
        {"wq.weight": torch.zeros(N_HEADS * HEAD_DIM, DIM)}, strict=False
    )

    assert torch.equal(qkv.wqkv.weight, original)


# -- local_head_split --------------------------------------------------------


def test_local_head_split_inserts_a_head_axis() -> None:
    t = torch.randn(6, N_HEADS * HEAD_DIM)
    out = local_head_split(t, HEAD_DIM)
    assert out.shape == (6, N_HEADS, HEAD_DIM)


def test_local_head_split_always_splits_the_last_axis() -> None:
    """It divides the final dim by ``head_dim``; the head axis is appended.

    An input whose last axis is already ``head_dim`` therefore gains a
    singleton head axis rather than passing through unchanged.
    """
    t = torch.randn(6, N_HEADS, HEAD_DIM)
    out = local_head_split(t, HEAD_DIM)
    assert out.shape == (6, N_HEADS, 1, HEAD_DIM)


def test_local_head_split_divides_the_last_axis_by_head_dim() -> None:
    t = torch.randn(6, 3 * HEAD_DIM)
    out = local_head_split(t, HEAD_DIM)
    assert out.shape == (6, 3, HEAD_DIM)


def test_local_head_split_is_a_view_not_a_copy() -> None:
    """It is documented as local, so it must not move data."""
    t = torch.randn(6, N_HEADS * HEAD_DIM)
    out = local_head_split(t, HEAD_DIM)
    assert out.data_ptr() == t.data_ptr()


def test_local_head_split_keeps_values_in_place() -> None:
    t = torch.arange(6 * N_HEADS * HEAD_DIM, dtype=torch.float32).reshape(
        6, N_HEADS * HEAD_DIM
    )
    out = local_head_split(t, HEAD_DIM)
    assert out.reshape(6, -1).equal(t)


@pytest.mark.parametrize("field", ["head_dim", "n_heads", "n_kv_heads"])
@pytest.mark.parametrize("value", [0, -1])
def test_qkv_rejects_non_positive_head_geometry(field, value):
    kwargs = {"head_dim": HEAD_DIM, "n_heads": N_HEADS, "n_kv_heads": N_KV_HEADS}
    kwargs[field] = value
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        QKVLinear(**kwargs, linear=torch.nn.Linear(DIM, FUSED_OUT))


@pytest.mark.parametrize("head_dim", [0, -1])
def test_local_head_split_rejects_non_positive_head_dim(head_dim):
    with pytest.raises(ValueError, match="head_dim must be positive"):
        local_head_split(torch.empty(2, 8), head_dim)


@pytest.mark.parametrize("param_name", ["weight", "bias"])
def test_checkpoint_rejects_fused_and_logical_qkv_keys(param_name):
    module = _qkv(bias=True)
    state = module.state_dict()
    parameter = getattr(module.wqkv, param_name)
    original = parameter.detach().clone()
    state[f"wqkv.{param_name}"] = original.clone()

    with pytest.raises(ValueError, match="both fused.*and logical"):
        module.load_state_dict(state)

    assert torch.equal(parameter, original)
    assert all(f"{name}.{param_name}" in state for name in ("wq", "wk", "wv"))


@pytest.mark.parametrize("name", ["head_dim", "n_heads", "n_kv_heads"])
@pytest.mark.parametrize("value", [True, False, 2.0, "2", None])
def test_qkv_sizes_require_integer_values(name, value):
    dimensions = dict(head_dim=HEAD_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS)
    dimensions[name] = value
    with pytest.raises(ValueError, match=f"{name} must be positive integer"):
        QKVLinear(**dimensions, linear=torch.nn.Linear(DIM, FUSED_OUT))


@pytest.mark.parametrize("value", [True, False, 2.0, "2", None])
def test_local_head_split_requires_integer_head_dim(value):
    with pytest.raises(ValueError, match="head_dim must be positive integer"):
        local_head_split(torch.empty(2, 8), value)


@pytest.mark.parametrize("failure", ["shape", "ambiguous"])
def test_qkv_load_hook_preserves_entries_if_bias_conversion_fails(failure):
    module = _qkv(bias=True)
    state = module.state_dict()
    if failure == "shape":
        state["wv.bias"] = torch.zeros(1)
    else:
        state["wqkv.bias"] = module.wqkv.bias.detach().clone()
    original = state.copy()

    with pytest.raises((RuntimeError, ValueError)):
        QKVLinear._merge_qkv_on_load(module, state, "")

    assert state.keys() == original.keys()
    assert all(state[key] is value for key, value in original.items())


def test_qkv_save_hook_preserves_entries_if_bias_split_fails():
    module = _qkv(bias=True)
    state = {
        "wqkv.weight": module.wqkv.weight.detach(),
        "wqkv.bias": torch.zeros(1),
    }
    original = state.copy()
    with pytest.raises(RuntimeError):
        QKVLinear._split_qkv_on_save(module, state, "", {})
    assert state.keys() == original.keys()
    assert all(state[key] is value for key, value in original.items())
