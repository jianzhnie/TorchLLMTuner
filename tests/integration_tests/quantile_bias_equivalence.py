"""Sign-based counts and quantile histogram reduction across DP/CP/TP axes.

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/quantile_bias_equivalence.py
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.models.common.moe.bias_update import (
    register_moe_balancing_hook,
    update_expert_bias,
    update_quantile_expert_bias,
)
from llmtuner.models.common.moe.experts import GroupedExperts, RoutedExperts
from llmtuner.models.common.moe.layer import MoE
from llmtuner.models.common.moe.routing import (
    QuantileBalancedTopKRouter,
    TokenChoiceTopKRouter,
)
from llmtuner.parallel.expert_parallel.token_dispatch import LocalTokenDispatcher


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
            # Real MoE state: independently sum the selected rank counts and
            # apply the documented mean-centered sign rule.
            moe_sign = MoE(
                4,
                RoutedExperts(GroupedExperts(4, 8, 4), LocalTokenDispatcher(4, 2)),
                TokenChoiceTopKRouter(4, 4, 2),
                load_balance_coeff=0.05,
            )
            local_counts = torch.tensor([rank + 1, 9 - rank, rank * 3, 2.0])
            moe_sign.tokens_per_expert_E.copy_(local_counts)
            pieces_counts = [torch.empty_like(local_counts) for _ in range(4)]
            dist.all_gather(pieces_counts, local_counts)
            selected_counts = [
                counts
                for index, counts in enumerate(pieces_counts)
                if "tp" not in axes or ep_enabled or index % 2 == rank % 2
            ]
            totals = torch.stack(selected_counts).sum(0)
            delta = 0.05 * torch.sign(totals.mean() - totals)
            expected_bias = delta - delta.mean()
            part = torch.nn.Module()
            layer = torch.nn.Module()
            layer.mlp = moe_sign
            part.layers = torch.nn.ModuleList([layer])
            optimizer = OptimizersContainer(
                OptimizerConfig(implementation="for-loop"), model_parts=[part]
            )
            register_moe_balancing_hook(
                optimizer, model_parts=[part], parallel_dims=topology
            )
            optimizer.step()
            torch.testing.assert_close(
                moe_sign.expert_bias_E, expected_bias, rtol=0, atol=0
            )
            assert not moe_sign.tokens_per_expert_E.any()
            # A second update without a new forward must not move the bias.
            optimizer.step()
            torch.testing.assert_close(
                moe_sign.expert_bias_E, expected_bias, rtol=0, atol=0
            )

            shared_moe = MoE(
                4,
                RoutedExperts(GroupedExperts(4, 8, 4), LocalTokenDispatcher(4, 2)),
                TokenChoiceTopKRouter(4, 4, 2),
                load_balance_coeff=0.05,
            )
            shared_moe.expert_bias_E = moe_sign.expert_bias_E
            other_counts = local_counts.flip(0) * 2
            shared_moe.tokens_per_expert_E.copy_(other_counts)
            moe_sign.tokens_per_expert_E.copy_(local_counts)
            other_pieces = [torch.empty_like(other_counts) for _ in range(4)]
            dist.all_gather(other_pieces, other_counts)
            joint = totals + torch.stack(
                [
                    counts
                    for index, counts in enumerate(other_pieces)
                    if "tp" not in axes or ep_enabled or index % 2 == rank % 2
                ]
            ).sum(0)
            shared_delta = 0.05 * torch.sign(joint.mean() - joint)
            update_expert_bias([moe_sign, shared_moe], parallel_dims=topology)
            torch.testing.assert_close(
                moe_sign.expert_bias_E,
                expected_bias + shared_delta - shared_delta.mean(),
                rtol=0,
                atol=1e-8,
            )
            assert not moe_sign.tokens_per_expert_E.any()
            assert not shared_moe.tokens_per_expert_E.any()

            moe = MoE(
                4,
                RoutedExperts(GroupedExperts(4, 8, 4), LocalTokenDispatcher(4, 2)),
                QuantileBalancedTopKRouter(4, 4, 2, num_bins=8),
                load_balance_coeff=None,
            )
            balancer = moe.router.quantile_balancer
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
            moe.expert_bias_E.copy_(bias)
            moe.tokens_per_expert_E.fill_(1)
            estimate = balancer.estimate_expert_bias
            observed = []

            def checked_estimate(
                histogram, bias, full=full, observed=observed, estimate=estimate
            ):
                torch.testing.assert_close(histogram, full, rtol=0, atol=0)
                observed.append(True)
                return estimate(histogram, bias)

            balancer.estimate_expert_bias = checked_estimate
            layer_quantile = torch.nn.Module()
            layer_quantile.mlp = moe
            part_quantile = torch.nn.Module()
            part_quantile.layers = torch.nn.ModuleList([layer_quantile])
            optimizer_quantile = OptimizersContainer(
                OptimizerConfig(implementation="for-loop"), model_parts=[part_quantile]
            )
            register_moe_balancing_hook(
                optimizer_quantile, model_parts=[part_quantile], parallel_dims=topology
            )
            optimizer_quantile.step()
            assert observed == [True]
            torch.testing.assert_close(moe.expert_bias_E, expected, rtol=0, atol=0)
            assert not histogram.any()
            assert not moe.tokens_per_expert_E.any()
            # Shared bias owns the union of both routers' histograms. Use
            # independent gathered totals, with the same original bin frame.
            balancer.estimate_expert_bias = estimate
            histogram.copy_(pieces[rank])
            moe.expert_bias_E.copy_(bias)
            shared_quantile = MoE(
                4,
                RoutedExperts(GroupedExperts(4, 8, 4), LocalTokenDispatcher(4, 2)),
                QuantileBalancedTopKRouter(4, 4, 2, num_bins=8),
                load_balance_coeff=None,
            )
            shared_quantile.expert_bias_E = moe.expert_bias_E
            other_histogram = (
                shared_quantile.router.quantile_balancer.required_bias_histogram_EB
            )
            other_histogram.copy_(histogram.flip(1))
            shared_expected = estimate(full + full.flip(1), bias)
            update_quantile_expert_bias([moe, shared_quantile], parallel_dims=topology)
            torch.testing.assert_close(
                moe.expert_bias_E, shared_expected, rtol=0, atol=0
            )
            assert not histogram.any() and not other_histogram.any()
            if rank == 0:
                print(
                    f"axes={axes}, EP={ep_enabled}: counts, bias and reset match",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
