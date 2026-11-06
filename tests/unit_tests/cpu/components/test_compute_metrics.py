"""Global benchmark and rank-local logs must agree on compute utilization."""

import pytest

from llmtuner.components.metrics import compute_utilization, token_throughput


@pytest.mark.parametrize("dp,tp,cp,pp", [(8, 1, 1, 1), (2, 2, 1, 2), (1, 2, 2, 2)])
def test_global_and_per_device_utilization_agree(dp, tp, cp, pp):
    global_tokens = 8 * 4096 * 4
    seconds = 2.0
    world_size = dp * tp * cp * pp
    global_tps = token_throughput(global_tokens, seconds)
    rank_tps = token_throughput(
        global_tokens // dp, seconds, replication_degree=tp * cp * pp
    )
    options = dict(flops_per_token=27_913_617_408, peak_flops_per_device=376e12)
    total = compute_utilization(
        tokens_per_second=global_tps, world_size=world_size, **options
    )
    rank = compute_utilization(tokens_per_second=rank_tps, world_size=1, **options)
    assert total["tflops"] == pytest.approx(rank["tflops"] * world_size)
    assert total["utilization_percent"] == pytest.approx(rank["utilization_percent"])
