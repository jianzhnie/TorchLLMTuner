from types import SimpleNamespace

from tests.caps import require_env

require_env("flex_attention", "spmd_types")

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from llmtuner.parallel.context_parallel import input_sharding


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("balanced", [False, True])
def test_npu_cp_mask_preserves_global_causality(monkeypatch, rank, balanced):
    mask = create_block_mask(
        lambda batch, head, query, key: query >= key,
        1,
        1,
        256,
        256,
        device="cpu",
    )
    permutation = torch.arange(256)
    if balanced:
        permutation = torch.cat(
            [permutation[:64], permutation[192:], permutation[64:192]]
        )
    balancer = (
        SimpleNamespace(_generate_indices=lambda restore: permutation[None])
        if balanced
        else None
    )
    mesh = SimpleNamespace(
        device_type="npu", size=lambda: 2, get_local_rank=lambda: rank
    )
    monkeypatch.setattr(input_sharding, "require_torch_input_sharding", lambda: None)
    monkeypatch.setattr(input_sharding, "resolve_load_balancer", lambda *args: balancer)

    def build_mask(mask_mod, **kwargs):
        assert kwargs["Q_LEN"] == 128
        assert kwargs["KV_LEN"] == 256
        assert kwargs["device"] == "npu"
        return mask_mod(
            torch.tensor(0),
            torch.tensor(0),
            torch.arange(128)[:, None],
            torch.arange(256)[None, :],
        )

    monkeypatch.setattr(input_sharding, "create_attention_mask", build_mask)
    actual = input_sharding.shard_attention_mask_for_cp(mask, mesh)
    expected = permutation[rank * 128 : (rank + 1) * 128, None] >= permutation[None, :]
    torch.testing.assert_close(actual, expected)
