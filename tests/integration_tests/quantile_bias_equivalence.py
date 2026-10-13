"""Quantile histogram reduction across independent DP/CP axes.

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/quantile_bias_equivalence.py
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.models.common.moe.bias_update import update_quantile_expert_bias
from llmtuner.models.common.moe.routing import QuantileBalancer


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 4
        rank = dist.get_rank()
        for axes, ep_enabled in (
            (("dp", "cp"), False),
            (("dp", "tp"), False),
            (("dp", "tp"), True),
        ):
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=axes)
            topology = SimpleNamespace(
                ep_enabled=ep_enabled,
                get_optional_mesh=lambda name, mesh=mesh, axes=axes: (
                    mesh[name] if name in axes else None
                ),
            )
            balancer = QuantileBalancer(4, 2, 8)
            histogram = balancer.required_bias_histogram_EB
            histogram[:, rank * 2] = 800000000
            histogram[:, rank * 2 + 1] = torch.arange(4) + rank + 1
            # All experts have the same total after the axes are reduced except
            # their small per-bin offsets, so test count precision independently.
            pieces = [torch.empty_like(histogram) for _ in range(4)]
            dist.all_gather(pieces, histogram)
            selected = [
                value
                for index, value in enumerate(pieces)
                if "tp" not in axes or ep_enabled or index % 2 == rank % 2
            ]
            full = torch.stack(selected).sum(0)
            if axes == ("dp", "cp"):
                assert full.sum(-1).min() > torch.iinfo(torch.int32).max
            bias = torch.tensor([-0.3, -0.1, 0.1, 0.3])
            expected = balancer.estimate_expert_bias(full, bias)
            moe = SimpleNamespace(
                router=SimpleNamespace(quantile_balancer=balancer),
                expert_bias_E=bias.clone(),
                tokens_per_expert_E=torch.ones(4),
            )
            estimate = balancer.estimate_expert_bias
            observed = []

            def checked_estimate(
                histogram, bias, full=full, observed=observed, estimate=estimate
            ):
                torch.testing.assert_close(histogram, full, rtol=0, atol=0)
                observed.append(True)
                return estimate(histogram, bias)

            balancer.estimate_expert_bias = checked_estimate
            update_quantile_expert_bias([moe], topology)
            assert observed == [True]
            torch.testing.assert_close(moe.expert_bias_E, expected, rtol=0, atol=0)
            assert not histogram.any()
            assert not moe.tokens_per_expert_E.any()
            if rank == 0:
                print(
                    f"axes={axes}, EP={ep_enabled}: counts, bias and reset match",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
