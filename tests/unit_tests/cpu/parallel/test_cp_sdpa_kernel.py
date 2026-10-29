"""CP's sdpa kernel: the mask rule and the paths that must not be taken.

The numerical equivalence lives in
``tests/integration_tests/cp_sdpa_equivalence.py`` (it needs ranks). What is
checked here needs no process group: the mask the kernel builds, the packed
refusal, and which CP shapes ``apply_cp`` accepts without flex.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from llmtuner.config.parallel_config import (
    ContextParallelConfig,
    ParallelConfig,
)
from llmtuner.parallel.context_parallel import context_parallel as cp_module
from llmtuner.parallel.context_parallel.sdpa_attention import (
    CPSdpaKernel,
    CPSdpaUlyssesKernel,
    DualCPSdpaKernel,
    causal_mask_for_local_queries,
    full_length_causal_mask,
)


def _fake_mesh(size: int, rank: int):
    return SimpleNamespace(
        size=lambda: size, get_local_rank=lambda: rank, get_group=lambda: object()
    )


class _OuterMesh:
    """The ``mesh`` argument ``apply_cp`` indexes by axis name."""

    def __init__(self, cp_mesh) -> None:
        self.mesh_dim_names = ("cp",)
        self._cp = cp_mesh

    def __getitem__(self, key):
        assert key == "cp"
        return self._cp


def test_the_mask_lets_each_query_see_its_own_block_and_earlier() -> None:
    """Rank 1 of 2 over 8 tokens: queries 0..3 of the block attend KV 0..4+i."""
    mask = causal_mask_for_local_queries(
        local_len=4, full_len=8, offset=4, device=torch.device("cpu")
    )

    assert mask.shape == (1, 1, 4, 8)
    assert mask.dtype == torch.bool
    # Query 0 (global 4) sees positions 0..4, and nothing after.
    assert mask[0, 0, 0].tolist() == [True] * 5 + [False] * 3
    # Query 3 (global 7, the last token) sees the whole sequence.
    assert mask[0, 0, 3].all()
    # The first block's queries never see the second block.
    first = causal_mask_for_local_queries(
        local_len=4, full_len=8, offset=0, device=torch.device("cpu")
    )
    assert not first[0, 0, :, 4:].any()


def test_packed_kv_requires_document_mask_before_collectives() -> None:
    kernel = CPSdpaKernel(cp_mesh=_fake_mesh(2, 0), packed=True)
    query = torch.zeros(1, 1, 2, 1)
    with pytest.raises(ValueError, match="full document mask"):
        kernel(query, query, query)


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(4, 4, dtype=torch.bool),
        torch.ones(1, 1, 4, 4),
        torch.ones(1, 1, 2, 4, dtype=torch.bool),
        object(),
    ],
)
def test_kv_rejects_invalid_document_mask_before_collectives(mask) -> None:
    kernel = CPSdpaKernel(cp_mesh=_fake_mesh(2, 0), packed=True)
    query = torch.zeros(1, 1, 2, 1)
    with pytest.raises(ValueError, match="mask"):
        kernel(query, query, query, block_mask=mask)


def test_construction_needs_no_process_group() -> None:
    """The rank list is resolved lazily, so a duck-typed mesh suffices."""
    kernel = CPSdpaKernel(cp_mesh=_fake_mesh(2, 1))

    assert kernel.cp_size == 2
    assert kernel.cp_rank == 1
    assert kernel._cp_ranks is None


@pytest.mark.parametrize("rank", [0, 1])
def test_packed_kv_slices_queries_without_crossing_documents(rank) -> None:
    kernel = CPSdpaKernel(cp_mesh=_fake_mesh(2, rank), packed=True)
    kernel._cp_ranks = (0, 1)
    generator = torch.Generator().manual_seed(73)
    query = torch.randn(1, 2, 8, 4, generator=generator)
    key = torch.randn(1, 2, 8, 4, generator=generator)
    value = torch.randn(1, 2, 8, 4, generator=generator)
    documents = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2])
    mask = (documents[:, None] == documents[None, :]).tril()[None, None]
    start = rank * 4
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    ).transpose(1, 2)[:, start : start + 4]
    causal = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=True
    ).transpose(1, 2)[:, start : start + 4]
    assert (expected - causal).abs().max() > 0.01
    with patch(
        "llmtuner.parallel.context_parallel.sdpa_attention._GatherSequence.apply",
        side_effect=[key, value],
    ):
        actual = kernel(
            query[:, :, start : start + 4],
            key[:, :, start : start + 4],
            value[:, :, start : start + 4],
            block_mask=mask,
        )
    torch.testing.assert_close(actual, expected)


class _FakeLayer:
    def __init__(self) -> None:
        self.self_attn = torch.nn.Module()


class _FakeModel(torch.nn.Module):
    def __init__(self, impl: str) -> None:
        super().__init__()
        self.layers = [_FakeLayer()]
        self.model = SimpleNamespace(
            config=SimpleNamespace(attn_mask_type="causal", _attn_implementation=impl)
        )


def _cp_config(*, strategy: str = "kv_allgather", cp: int = 2) -> ParallelConfig:
    return ParallelConfig(
        context_parallel_kv_allgather_size=cp,
        cp_config=ContextParallelConfig(context_parallel_strategy=strategy),
    )


def test_kv_allgather_on_a_non_flex_backend_attaches_the_sdpa_kernel() -> None:
    """This is what makes CP runnable without flex (the NPU case)."""
    model = _FakeModel("sdpa")
    recorded: dict = {}
    model.set_cp_mesh = lambda cp_mesh, **kwargs: recorded.update(kwargs)
    model._cp_mesh_kwargs = recorded

    cp_module.apply_cp(
        model, mesh=_OuterMesh(_fake_mesh(2, 0)), cfg=_cp_config(), parallel_dims=None
    )

    assert model.model.config._attn_implementation == "sdpa_cp"
    assert isinstance(model.layers[0].self_attn._titan_cp_sdpa_kernel, CPSdpaKernel)
    assert recorded["strategy"] == "kv_allgather"


def test_packed_kv_rejects_unvalidated_load_balancing() -> None:
    model = _FakeModel("sdpa")
    model.model.config.attn_mask_type = "block_causal"
    cfg = _cp_config()
    cfg.cp_config.context_parallel_load_balancer = "headtail"
    with pytest.raises(NotImplementedError, match="load_balancer=None"):
        cp_module.apply_cp(
            model,
            mesh=_OuterMesh(_fake_mesh(2, 0)),
            cfg=cfg,
            parallel_dims=None,
        )
    assert not hasattr(model.layers[0].self_attn, "_titan_cp_sdpa_kernel")
    assert model.model.config._attn_implementation == "sdpa"


def test_ulysses_gets_its_own_sdpa_kernel() -> None:
    """ulysses on sdpa: the swap needs no offset mask, only the causal one.

    The head swap leaves every rank holding the *full* sequence, so the mask is
    the unshifted causal one and no flex BlockMask is needed -- which is what
    makes this strategy expressible on this backend at all.
    """
    model = _FakeModel("sdpa")
    recorded: dict = {}
    model.set_cp_mesh = lambda cp_mesh, **kwargs: recorded.update(kwargs)

    cp_module.apply_cp(
        model,
        mesh=_OuterMesh(_fake_mesh(2, 0)),
        cfg=_cp_config(strategy="ulysses"),
        parallel_dims=None,
    )

    assert model.model.config._attn_implementation == "sdpa_cp"
    assert isinstance(
        model.layers[0].self_attn._titan_cp_sdpa_kernel, CPSdpaUlyssesKernel
    )
    assert model.model.config._cp_sdpa_ulysses is True


@pytest.mark.parametrize("packed", [False, True])
def test_dual_axes_attach_the_sdpa_kernel(packed) -> None:
    model = _FakeModel("sdpa")
    model.model.config.attn_mask_type = "block_causal" if packed else "causal"
    recorded = {}
    model.set_cp_mesh = lambda mesh, **kwargs: recorded.update(kwargs)
    cfg = _cp_config()
    cfg.context_parallel_kv_allgather_size = 2
    cfg.context_parallel_ulysses_size = 2
    kv_mesh = _fake_mesh(2, 1)
    ulysses_mesh = _fake_mesh(2, 0)
    topology = SimpleNamespace(
        get_optional_mesh=lambda name: kv_mesh if name == "cp_kv" else ulysses_mesh
    )
    cp_module.apply_cp(
        model,
        mesh=_OuterMesh(_fake_mesh(4, 2)),
        cfg=cfg,
        parallel_dims=topology,
    )
    kernel = model.layers[0].self_attn._titan_cp_sdpa_kernel
    assert isinstance(kernel, DualCPSdpaKernel)
    assert kernel.packed is packed
    assert kernel.cp_rank == 1
    assert model.model.config._cp_sdpa_dual
    assert recorded["mask_mesh"] is kv_mesh


def test_dual_sdpa_rejects_missing_topology() -> None:
    with pytest.raises(ValueError, match="require ParallelTopology"):
        cp_module.apply_cp(
            _FakeModel("sdpa"),
            mesh=_OuterMesh(_fake_mesh(4, 0)),
            cfg=ParallelConfig(
                context_parallel_kv_allgather_size=2,
                context_parallel_ulysses_size=2,
            ),
            parallel_dims=None,
        )


@pytest.mark.parametrize("invalid", ["load_balancer", "heads"])
def test_dual_sdpa_retains_ulysses_guards(invalid) -> None:
    model = _FakeModel("sdpa")
    model.model.config.num_attention_heads = 4
    model.model.config.num_key_value_heads = 3 if invalid == "heads" else 4
    cfg = ParallelConfig(
        context_parallel_kv_allgather_size=2,
        context_parallel_ulysses_size=2,
    )
    if invalid == "load_balancer":
        cfg.cp_config.context_parallel_load_balancer = "headtail"
    topology = SimpleNamespace(get_optional_mesh=lambda name: _fake_mesh(2, 0))
    message = "load_balancer=None" if invalid == "load_balancer" else "divisible"
    with pytest.raises(ValueError, match=message):
        cp_module.apply_cp(
            model,
            mesh=_OuterMesh(_fake_mesh(4, 0)),
            cfg=cfg,
            parallel_dims=topology,
        )
    assert model.model.config._attn_implementation == "sdpa"


# -- the ulysses swap's mask ---------------------------------------------------


def test_the_full_length_causal_mask_is_the_ordinary_lower_triangle() -> None:
    """No CP offset enters the ulysses mask: every rank holds the full sequence.

    That is the whole reason this strategy does not need flex: the kv_allgather
    kernel's mask has to be shifted by the rank's block start, this one does not.
    """
    mask = full_length_causal_mask(4, torch.device("cpu"))

    assert tuple(mask.shape) == (1, 1, 4, 4)
    assert mask.dtype == torch.bool
    for row in range(4):
        for col in range(4):
            assert bool(mask[0, 0, row, col]) is (col <= row)


def test_a_packed_document_mask_is_used_verbatim_by_the_ulysses_kernel() -> None:
    """The wrapper hands the unsharded document mask; the kernel adds nothing.

    The mask is already causal-and-same-document, so ANDing causality in again
    would be redundant -- and *replacing* it with the plain causal mask would
    silently let attention cross a document boundary.
    """
    kernel = CPSdpaUlyssesKernel(cp_mesh=_fake_mesh(1, 0), packed=True)
    document_mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    document_mask[0, 0, 1, 1] = True  # token 1 alone in its document
    seen: dict = {}

    def _fake_sdpa(q, k, v, *, attn_mask=None, **kwargs):
        seen["mask"] = attn_mask
        return q

    with patch("torch.nn.functional.scaled_dot_product_attention", _fake_sdpa):
        with (
            patch(
                "llmtuner.parallel.context_parallel.sdpa_attention.SeqToHead.apply",
                lambda x, group: x,
            ),
            patch(
                "llmtuner.parallel.context_parallel.sdpa_attention.HeadToSeq.apply",
                lambda x, group: x,
            ),
        ):
            kernel(
                torch.zeros(1, 1, 2, 1),
                torch.zeros(1, 1, 2, 1),
                torch.zeros(1, 1, 2, 1),
                block_mask=document_mask,
            )

    assert seen["mask"] is document_mask


def test_a_packed_ulysses_run_without_a_mask_is_refused() -> None:
    """No document mask for a packed corpus is unrecoverable: fail, don't guess."""
    kernel = CPSdpaUlyssesKernel(cp_mesh=_fake_mesh(2, 0), packed=True)

    with patch(
        "llmtuner.parallel.context_parallel.sdpa_attention.SeqToHead.apply",
        lambda x, group: x,
    ):
        with pytest.raises(ValueError, match="no document mask"):
            kernel(
                torch.zeros(1, 1, 2, 1),
                torch.zeros(1, 1, 2, 1),
                torch.zeros(1, 1, 2, 1),
            )


@pytest.mark.parametrize("strategy", ["kv", "ulysses"])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dropout": 0.1}, "dropout"),
        ({"dropout": -0.1}, "dropout"),
        ({"softcap": 30.0}, "softcap"),
        ({"s_aux": torch.ones(1)}, "s_aux"),
    ],
)
def test_kv_and_ulysses_reject_modifiers_before_collectives(
    strategy, kwargs, message
) -> None:
    mesh = _fake_mesh(2, 0)
    kernel_type = CPSdpaKernel if strategy == "kv" else CPSdpaUlyssesKernel
    kernel = kernel_type(cp_mesh=mesh)
    query = torch.zeros(1, 2, 2, 1)
    with (
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention.SeqToHead.apply"
        ) as seq_to_head,
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention._GatherSequence.apply"
        ) as gather,
    ):
        with pytest.raises(NotImplementedError, match=message):
            kernel(query, query, query, **kwargs)
    seq_to_head.assert_not_called()
    gather.assert_not_called()


@pytest.mark.parametrize("strategy", ["kv", "ulysses"])
@pytest.mark.parametrize("window", [0, -1, 1.5, True, "3"])
def test_kv_and_ulysses_reject_invalid_windows_before_collectives(
    strategy, window
) -> None:
    mesh = _fake_mesh(2, 0)
    kernel_type = CPSdpaKernel if strategy == "kv" else CPSdpaUlyssesKernel
    kernel = kernel_type(cp_mesh=mesh)
    query = torch.zeros(1, 2, 2, 1)
    with (
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention.SeqToHead.apply"
        ) as seq_to_head,
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention._GatherSequence.apply"
        ) as gather,
    ):
        with pytest.raises(ValueError, match="sliding_window.*positive integer"):
            kernel(query, query, query, sliding_window=window)
    seq_to_head.assert_not_called()
    gather.assert_not_called()


def _sliding_inputs(*, packed: bool):
    generator = torch.Generator().manual_seed(73)
    query, key, value = [
        torch.randn(1, 2, 8, 4, generator=generator, dtype=torch.float64)
        for _ in range(3)
    ]
    mask = torch.ones(8, 8, dtype=torch.bool).tril()
    if packed:
        documents = torch.repeat_interleave(torch.arange(2), torch.tensor([2, 6]))
        mask &= documents[:, None] == documents[None, :]
    return query, key, value, mask[None, None]


def _sliding_reference(query, key, value, mask, *, window: int):
    indices = torch.arange(query.shape[2])
    rows = indices[:, None]
    cols = indices[None, :]
    sliding = (cols <= rows) & (rows - cols < window)
    expected_mask = mask & sliding[None, None]
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=expected_mask
    ).transpose(1, 2)
    unwindowed = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    ).transpose(1, 2)
    assert not torch.allclose(expected, unwindowed)
    return expected


@pytest.mark.parametrize("packed", [False, True])
def test_kv_sliding_window_intersects_the_base_mask(packed) -> None:
    query, key, value, mask = _sliding_inputs(packed=packed)
    expected = _sliding_reference(query, key, value, mask, window=3)
    kernel = CPSdpaKernel(cp_mesh=_fake_mesh(2, 1), packed=packed)
    kernel._cp_ranks = (0, 1)
    with patch(
        "llmtuner.parallel.context_parallel.sdpa_attention._GatherSequence.apply",
        side_effect=[key, value],
    ):
        actual = kernel(
            query[:, :, 4:],
            key[:, :, 4:],
            value[:, :, 4:],
            block_mask=mask if packed else None,
            sliding_window=3,
        )
    torch.testing.assert_close(actual, expected[:, 4:])


@pytest.mark.parametrize("packed", [False, True])
def test_ulysses_sliding_window_uses_full_sequence_coordinates(packed) -> None:
    query, key, value, mask = _sliding_inputs(packed=packed)
    expected = _sliding_reference(query, key, value, mask, window=3)
    kernel = CPSdpaUlyssesKernel(cp_mesh=_fake_mesh(2, 0), packed=packed)
    with (
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention.SeqToHead.apply",
            side_effect=[query, key, value],
        ),
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention.HeadToSeq.apply",
            side_effect=lambda output, group: output[:, :, 4:],
        ),
    ):
        actual = kernel(
            query[:, :, 4:],
            key[:, :, 4:],
            value[:, :, 4:],
            block_mask=mask if packed else None,
            sliding_window=3,
        )
    torch.testing.assert_close(actual, expected[:, 4:])
