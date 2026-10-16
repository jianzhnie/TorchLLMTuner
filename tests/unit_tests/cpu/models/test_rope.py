"""Rotary position embeddings: shapes, conventions, and round-trips.

The vendored ``rope.py`` was verified bit-identical to torchtitan's original at
migration time (all five class x scaling combinations, both yarn truncate modes,
cache and forward output compared with ``torch.equal``). That check needs
torchtitan on the path and is not reproducible here, so what these tests cover
is the *behaviour* the embeddings must keep on their own -- the conventions that
would silently corrupt attention if a refactor inverted them.

The two formats differ in a way worth pinning down:

* ``ComplexRoPE`` rotates the pair ``(x[2i], x[2i+1])`` -- adjacent dims.
* ``CosSinRoPE`` rotates ``(x[:d/2], x[d/2:])`` -- split halves.

They are not interchangeable, and a model built for one will produce garbage
with the other. The tests below assert each format's own invariant rather than
comparing the two.
"""

from __future__ import annotations

import pytest
import torch

from llmtuner.config.rope_config import RoPEConfig
from llmtuner.models.common.rope import (
    ComplexRoPE,
    CosSinRoPE,
    RoPE,
    yarn_inv_freq,
)

DIM = 64
MAX_LEN = 128
TOKENS = 16
HEADS = 8


@pytest.mark.parametrize(
    "options,field",
    [
        ({"dim": 3}, "dim"),
        ({"dim": True}, "dim"),
        ({"max_context_length": 0}, "max_context_length"),
        ({"scaling": "unknown"}, "scaling"),
        ({"theta": 0}, "theta"),
        ({"theta": float("nan")}, "theta"),
        ({"theta": "10000"}, "theta"),
        ({"scaling": "llama", "scaling_factor": 0}, "scaling_factor"),
        ({"scaling": "llama", "low_freq_factor": 4}, "high_freq_factor"),
        (
            {"scaling": "llama", "original_max_position_embeddings": 0},
            "original_max_position_embeddings",
        ),
        ({"scaling": "yarn", "rope_factor": 0.5}, "rope_factor"),
        ({"scaling": "yarn", "rope_factor": "4"}, "rope_factor"),
        ({"scaling": "yarn", "rope_factor": True}, "rope_factor"),
        ({"scaling": "yarn", "rope_factor": 4, "theta": 1}, "theta"),
        ({"scaling": "yarn", "beta_slow": 0}, "beta_slow"),
        ({"scaling": "yarn", "rope_factor": 4, "beta_fast": 0.5}, "beta_fast"),
        ({"scaling": "yarn", "original_seq_len": 0}, "original_seq_len"),
        ({"scaling": "yarn", "truncate": "false"}, "truncate"),
    ],
)
def test_rope_config_rejects_invalid_dimensions_and_scaling(options, field):
    from llmtuner.errors import ConfigError

    values = {"dim": DIM, "max_context_length": MAX_LEN, **options}
    with pytest.raises(ConfigError, match=field):
        RoPEConfig(**values)


@pytest.mark.parametrize("rope_class", [ComplexRoPE, CosSinRoPE])
def test_plain_rope_ignores_unused_scaling_fields(rope_class):
    config = RoPEConfig(
        dim=DIM,
        max_context_length=MAX_LEN,
        scaling_factor=0,
        rope_factor=0,
        beta_fast=0,
    )
    expected = rope_class(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    torch.testing.assert_close(rope_class(config).cache, expected.cache)


def _qk(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(TOKENS, HEADS, DIM, generator=g),
        torch.randn(TOKENS, HEADS, DIM, generator=g),
    )


# -- cache construction ------------------------------------------------------


def test_complex_cache_holds_half_width_complex_values() -> None:
    rope = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    assert rope.cache.shape == (MAX_LEN, DIM // 2)
    assert rope.cache.is_complex()


def test_cossin_cache_holds_full_width_real_values() -> None:
    rope = CosSinRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    assert rope.cache.shape == (MAX_LEN, DIM * 2)
    assert not rope.cache.is_complex()


def test_cache_is_not_persisted_into_checkpoints() -> None:
    """The cache is derived from the config -- saving it would only go stale."""
    rope = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    assert "cache" not in rope.state_dict()


# -- forward contracts -------------------------------------------------------


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_forward_returns_query_and_key_with_input_shapes(cls: type[RoPE]) -> None:
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, k = _qk()
    with torch.no_grad():
        q_out, k_out = rope(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_forward_returns_only_query_when_key_is_none(cls: type[RoPE]) -> None:
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    with torch.no_grad():
        out = rope(q, None)
    assert isinstance(out, torch.Tensor)
    assert out.shape == q.shape


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_forward_preserves_dtype(cls: type[RoPE]) -> None:
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, k = _qk()
    with torch.no_grad():
        q_out, k_out = rope(q.bfloat16(), k.bfloat16())
    assert q_out.dtype is torch.bfloat16
    assert k_out.dtype is torch.bfloat16


# -- position handling -------------------------------------------------------


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_positions_select_the_matching_cache_row(cls: type[RoPE]) -> None:
    """Token t must be rotated by cache row ``positions[t]``, not row ``t``.

    Asserted on the gathered cache directly rather than on forward output: an
    output comparison cannot separate "used the wrong row" from "reordered the
    output the same way", which is exactly the bug worth catching.
    """
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    positions = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3])

    with torch.no_grad():
        gathered = rope._reshape_cache(q, positions)

    assert gathered.shape == (TOKENS, 1, rope.cache.shape[-1])
    for token in range(TOKENS):
        assert torch.equal(gathered[token, 0], rope.cache[positions[token]])


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_omitting_positions_uses_rows_in_order(cls: type[RoPE]) -> None:
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()

    with torch.no_grad():
        gathered = rope._reshape_cache(q, None)

    assert torch.equal(gathered[:, 0], rope.cache[:TOKENS])


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_position_past_the_cache_is_rejected(cls: type[RoPE]) -> None:
    """An out-of-range position indexes no row; catching it beats silent garbage."""
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    positions = torch.full((TOKENS,), MAX_LEN, dtype=torch.long)
    with pytest.raises(RuntimeError):
        # _assert_async defers the failure to the next kernel launch.
        with torch.no_grad():
            rope(q, None, positions)
        torch.cuda.synchronize() if torch.cuda.is_available() else None


# -- rotation alternatives ---------------------------------------------------


def test_token_zero_is_unrotated() -> None:
    """Position 0 is angle 0, so the rotation is the identity there."""
    rope = CosSinRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    with torch.no_grad():
        out = rope(q, None)
    torch.testing.assert_close(out[0], q[0], rtol=1e-5, atol=1e-6)


def test_complex_rope_inverse_undoes_the_forward_rotation() -> None:
    """Conjugating the cache is the inverse rotation -- the defining property."""
    rope = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, k = _qk()
    with torch.no_grad():
        q_rot, k_rot = rope(q, k)
        q_back, k_back = rope(q_rot, k_rot, inverse=True)
    torch.testing.assert_close(q_back, q, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k_back, k, rtol=1e-5, atol=1e-6)


def test_cossin_rope_rejects_inverse_rotation() -> None:
    """The split-half convention has no cached conjugate to reuse."""
    rope = CosSinRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    with pytest.raises(NotImplementedError):
        with torch.no_grad():
            rope(q, None, inverse=True)


def test_complex_rope_rotates_adjacent_dimension_pairs() -> None:
    """Pin the convention: (x[0], x[1]) rotates as one pair, not (x[0], x[d/2])."""
    rope = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q = torch.zeros(TOKENS, HEADS, DIM)
    q[1, 0, 0] = 1.0  # a single adjacent-pair component at position 1
    with torch.no_grad():
        out = rope(q, None)
    # Only dims 0 and 1 of that position may move.
    assert out[1, 0, 0] != 0.0
    assert out[1, 0, 1] != 0.0
    assert torch.equal(out[1, 0, 2:], torch.zeros(DIM - 2))


def test_cossin_rope_rotates_split_halves() -> None:
    """The other convention: x[0] pairs with x[d/2], not x[1]."""
    rope = CosSinRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q = torch.zeros(TOKENS, HEADS, DIM)
    q[1, 0, 0] = 1.0
    with torch.no_grad():
        out = rope(q, None)
    assert out[1, 0, 0] != 0.0
    assert out[1, 0, DIM // 2] != 0.0
    # Adjacent dim 1 stays untouched under this convention.
    assert out[1, 0, 1] == 0.0


# -- scaling modes -----------------------------------------------------------


def test_yarn_scaling_changes_the_cache() -> None:
    """A yarn config with rope_factor > 1 must rescale frequencies."""
    plain = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    yarn = ComplexRoPE(
        RoPEConfig(
            dim=DIM,
            max_context_length=MAX_LEN,
            scaling="yarn",
            rope_factor=4.0,
            original_seq_len=64,
        )
    )
    assert not torch.equal(plain.cache, yarn.cache)


def test_llama_scaling_changes_the_cache() -> None:
    plain = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    llama = ComplexRoPE(
        RoPEConfig(dim=DIM, max_context_length=MAX_LEN, scaling="llama")
    )
    assert not torch.equal(plain.cache, llama.cache)


def test_cossin_rope_rejects_llama_scaling() -> None:
    with pytest.raises(NotImplementedError):
        CosSinRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN, scaling="llama"))


def test_yarn_factor_of_one_is_a_no_op() -> None:
    """``scaling='yarn'`` only engages above 1.0, so 1.0 must match plain rope."""
    plain = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    yarn = ComplexRoPE(
        RoPEConfig(dim=DIM, max_context_length=MAX_LEN, scaling="yarn", rope_factor=1.0)
    )
    assert torch.equal(plain.cache, yarn.cache)


def test_yarn_truncate_controls_fractional_cutoffs() -> None:
    """truncate=True floors/ceils the beta cutoffs; False keeps them fractional."""
    options = dict(
        base=10000.0,
        rope_factor=4.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=64,
    )
    truncated = yarn_inv_freq(DIM, **options, truncate=True)
    fractional = yarn_inv_freq(DIM, **options, truncate=False)
    assert not torch.equal(truncated, fractional)


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
@pytest.mark.parametrize("compiled", [False, True])
def test_negative_position_does_not_select_cache_tail(cls, compiled):
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    if compiled:
        rope = torch.compile(rope, backend="eager", fullgraph=True)
    q, _ = _qk()
    positions = torch.zeros(TOKENS, dtype=torch.long)
    positions[0] = -1
    with pytest.raises(RuntimeError, match="position_ids"):
        rope(q, positions=positions)


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
@pytest.mark.parametrize("shape", [(1, TOKENS), (TOKENS, 1), (TOKENS - 1,), ()])
def test_rope_rejects_positions_outside_token_axis_contract(cls, shape):
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    with pytest.raises(ValueError, match="positions must have shape"):
        rope(q, positions=torch.zeros(shape, dtype=torch.long))


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_rope_reports_implicit_positions_exceeding_cache(cls):
    rope = cls(RoPEConfig(dim=DIM, max_context_length=TOKENS - 1))
    q, _ = _qk()
    with pytest.raises(ValueError, match="exceeds cache capacity"):
        rope(q)


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.float32])
def test_rope_rejects_mask_and_floating_position_dtypes(cls, dtype):
    rope = cls(RoPEConfig(dim=DIM, max_context_length=TOKENS))
    q, _ = _qk()
    with pytest.raises(TypeError, match="int32 or int64"):
        rope(q, positions=torch.ones(TOKENS, dtype=dtype))


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_rope_int32_positions_match_int64(cls):
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    positions = torch.arange(TOKENS)
    torch.testing.assert_close(
        rope(q, positions=positions.to(torch.int32)),
        rope(q, positions=positions),
        rtol=0,
        atol=0,
    )
