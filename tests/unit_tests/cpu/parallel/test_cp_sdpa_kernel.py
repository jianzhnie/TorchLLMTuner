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


def test_a_packed_corpus_is_refused_rather_than_masked_as_one_document() -> None:
    """sdpa takes a dense mask; a document structure has to be materialized."""
    with pytest.raises(NotImplementedError, match="packed corpus"):
        CPSdpaKernel(cp_mesh=_fake_mesh(2, 0), packed=True)


def test_construction_needs_no_process_group() -> None:
    """The rank list is resolved lazily, so a duck-typed mesh suffices."""
    kernel = CPSdpaKernel(cp_mesh=_fake_mesh(2, 1))

    assert kernel.cp_size == 2
    assert kernel.cp_rank == 1
    assert kernel._cp_ranks is None


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


def test_the_dual_axis_form_stays_on_flex() -> None:
    """Two composed masks have no sdpa spelling; one axis at a time does."""
    model = _FakeModel("sdpa")
    cfg = _cp_config()
    cfg.context_parallel_kv_allgather_size = 2
    cfg.context_parallel_ulysses_size = 2

    with pytest.raises(RuntimeError, match="only with a single CP axis"):
        cp_module.apply_cp(
            model,
            mesh=_OuterMesh(_fake_mesh(4, 0)),
            cfg=cfg,
            parallel_dims=None,
        )


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
    kernel = CPSdpaUlyssesKernel(cp_mesh=_fake_mesh(2, 0), packed=True)
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
