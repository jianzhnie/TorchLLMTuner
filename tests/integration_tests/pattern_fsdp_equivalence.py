"""Pattern FSDP must preserve tied weights and dense training numerics.

Run: PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/pattern_fsdp_equivalence.py

CPU/Gloo or CUDA/NCCL, float32, native/custom engines, shared pattern units.
Also checks tied decoder grouped-state prefetch targets and three SGD updates.
An expert unit overrides FSDP2 with HSDP (replicate=2, shard=1), retaining
the same reduction ranks while changing the parameter shard degree.
For CUDA append --device cuda to the torchrun command.
Append --prefetch-window 5 to exercise explicit windows beyond the unit count.
Compare three SGD updates against a dense model over both ranks' batches.
This does not exercise the custom multi-context engine.
"""

from __future__ import annotations

import argparse
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    apply_fsdp_to_decoder,
    enable_fsdp_symm_mem,
    force_sum_grad_reduction,
    iter_fsdp_modules,
)
from llmtuner.parallel.fully_shard_parallel.plan import apply_fsdp_by_pattern


class TiedDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Sequential(nn.Embedding(8, 4))
        self.block = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 8, bias=False)
        self.head.weight = self.embedding[0].weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(self.embedding(tokens)))


class DecoderLayout(TiedDecoder):
    """Exercise the decoder entry's grouped embedding/norm/head state."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([self.block])
        self.norm = nn.LayerNorm(4)
        self.enable_weight_tying = True

    @property
    def tok_embeddings(self) -> nn.Module:
        return self.embedding

    @property
    def lm_head(self) -> nn.Module:
        return self.head

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.block(self.embedding(tokens))))


class ExpertProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from llmtuner.models.common.moe.experts import GroupedExperts

        self.experts = GroupedExperts(4, 8, 1, use_grouped_mm=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.experts(
            hidden.reshape(-1, 4), torch.tensor([hidden.numel() // 4])
        ).reshape_as(hidden)


class SharedLayerDecoder(DecoderLayout):
    """Default decoder units share a trainable projection across two layers."""

    def __init__(self) -> None:
        super().__init__()
        second = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        second[0].weight = self.block[0].weight
        self.layers.append(second)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(self.norm(hidden))


class SharedMoEBlock(nn.Module):
    """A shared dense projection plus real single-expert grouped weights."""

    def __init__(self, projection: nn.Linear) -> None:
        super().__init__()
        from llmtuner.models.common.moe.experts import GroupedExperts

        self.projection = nn.Linear(4, 4)
        self.projection.weight = projection.weight
        self.moe_enabled = True
        self.moe = nn.Module()
        self.moe.router = nn.Module()
        self.moe.router.num_experts = 1
        self.moe.routed_experts = nn.Module()
        self.moe.routed_experts.inner_experts = GroupedExperts(
            4, 8, 1, use_grouped_mm=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        experts = self.moe.routed_experts.inner_experts
        counts = torch.tensor([hidden.numel() // 4], device=hidden.device)
        return self.projection(hidden) + experts(
            hidden.reshape(-1, 4), counts
        ).reshape_as(hidden)


class MoESharedDecoder(SharedLayerDecoder):
    """Dense and MoE decoder layers consume the same projection weight."""

    def __init__(self) -> None:
        super().__init__()
        self.layers[1] = SharedMoEBlock(self.block[0])


class BoundarySharedDecoder(DecoderLayout):
    """A decoder layer and final norm consume the same trainable bias."""

    def __init__(self) -> None:
        super().__init__()
        self.block[0].bias = self.norm.bias


class UntiedBoundarySharedDecoder(BoundarySharedDecoder):
    """Keep the embedding independent while sharing the layer/norm boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.enable_weight_tying = False
        self.head.weight = nn.Parameter(self.head.weight.detach().clone())


class RootSharedDecoder(DecoderLayout):
    """A root-owned projection also has a registered decoder-layer consumer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("external_weight", self.block[0].weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)
        hidden = self.block(hidden) + F.linear(hidden, self.external_weight)
        return self.head(self.norm(hidden))


class SingleExpertDecoder(TiedDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.block = ExpertProjection()


class CrossSharedDecoder(TiedDecoder):
    """Parent/child ownership in opposite branches forms a dependency cycle."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.right = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.left.register_parameter("shared", self.right[0].weight)
        self.right.register_parameter("shared", self.left[0].weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.block(self.embedding(tokens))
        hidden = self.left(hidden) + self.right(hidden)
        return self.head(hidden)


class ReusedDecoder(TiedDecoder):
    """One projection is called twice through different registered paths."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.alias = self.projection

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.block(self.embedding(tokens))
        hidden = self.alias(torch.tanh(self.projection(hidden)))
        return self.head(hidden)


def full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prefetch-window", type=int, default=0)
    args = parser.parse_args()
    if args.prefetch_window < 0:
        parser.error("--prefetch-window must be nonnegative")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("--device cuda requires an available CUDA device")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    try:
        assert dist.get_world_size() == 2
        mesh = init_device_mesh(device.type, (2,))
        replicated_mesh = init_device_mesh(
            device.type, (2, 1), mesh_dim_names=("dp_replicate", "dp_shard")
        )
        rank = dist.get_rank()
        for implementation in ("native", "custom"):
            for factory, patterns in (
                (DecoderLayout, None),
                (SharedLayerDecoder, None),
                (MoESharedDecoder, None),
                (BoundarySharedDecoder, None),
                (UntiedBoundarySharedDecoder, None),
                (RootSharedDecoder, None),
                (SingleExpertDecoder, {"embedding": {}, "head": {}, "block": {}}),
                (
                    SingleExpertDecoder,
                    {"embedding": {}, "head": {}, "block": {"mesh": replicated_mesh}},
                ),
                (TiedDecoder, {"embedding": {}, "head": {}, "block.0": {}}),
                (TiedDecoder, {"embedding": {}, "block": {}, "block.0": {}}),
                (TiedDecoder, {"embedding": {}, "embedding.0": {}, "head": {}}),
                (
                    CrossSharedDecoder,
                    {
                        name: {}
                        for name in (
                            "embedding",
                            "head",
                            "left",
                            "left.0",
                            "right",
                            "right.0",
                        )
                    },
                ),
                (ReusedDecoder, {"embedding": {}, "head": {}, "alias": {}}),
            ):
                torch.manual_seed(42)
                reference = factory().to(device)
                torch.manual_seed(42)
                model = factory().to(device)
                if patterns is None:
                    apply_fsdp_to_decoder(
                        model,
                        mesh,
                        param_dtype=torch.float32,
                        reduce_dtype=torch.float32,
                        pp_enabled=False,
                        fsdp_implementation=implementation,
                        forward_prefetch_count=args.prefetch_window,
                        backward_prefetch_count=args.prefetch_window,
                    )
                    shared = model.embedding._get_fsdp_state()
                    block = (
                        model._get_fsdp_state()
                        if isinstance(model, RootSharedDecoder)
                        else model.block._get_fsdp_state()
                    )
                    final = model.norm._get_fsdp_state()
                    assert model.head._get_fsdp_state() is final
                    if model.enable_weight_tying:
                        assert final is shared
                    if isinstance(model, BoundarySharedDecoder):
                        assert model.block[0].bias is model.norm.bias
                        assert block is final
                    if isinstance(model, SharedLayerDecoder):
                        second_projection = (
                            model.layers[1].projection
                            if isinstance(model, MoESharedDecoder)
                            else model.layers[1][0]
                        )
                        assert model.layers[0][0].weight is second_projection.weight
                        assert model.layers[1]._get_fsdp_state() is block
                        if isinstance(model, MoESharedDecoder):
                            experts = model.layers[1].moe.routed_experts.inner_experts
                            for parameter in experts.parameters():
                                assert parameter.placements[0].dim == 1
                                assert parameter.to_local().shape[0] == 1
                    if isinstance(model, RootSharedDecoder):
                        assert model.external_weight is model.block[0].weight
                        assert model._get_fsdp_state() is block
                    if args.prefetch_window and not isinstance(
                        model, BoundarySharedDecoder | RootSharedDecoder
                    ):
                        assert shared._states_to_forward_prefetch == [block]
                        assert shared._states_to_backward_prefetch == []
                        assert block._states_to_forward_prefetch == []
                        assert block._states_to_backward_prefetch == [shared]
                else:
                    cfg = ParallelConfig(
                        data_parallel_shard_size=2,
                        fsdp_config=FSDPConfig(
                            fsdp_implementation=implementation,
                            fsdp_apply_modules=patterns,
                            fsdp_forward_prefetch_count=args.prefetch_window,
                            fsdp_backward_prefetch_count=args.prefetch_window,
                        ),
                    )
                    apply_fsdp_by_pattern(
                        model,
                        dp_mesh=mesh,
                        cfg=cfg,
                        param_dtype=torch.float32,
                        reduce_dtype=torch.float32,
                    )
                if isinstance(model, MoESharedDecoder):
                    # Check scope against real native/custom states while replacing
                    # only the device-specific enable calls, not ownership discovery.
                    states = {
                        id(module._get_fsdp_state())
                        for module in iter_fsdp_modules(model)
                    }
                    sparse = id(model.layers[1]._get_fsdp_state())
                    for scope in ("dense", "all"):
                        selected = []
                        with (
                            patch.object(
                                FSDPModule,
                                "set_force_sum_reduction_for_comms",
                                lambda module, value: None,
                            ),
                            patch.object(
                                FSDPModule,
                                "set_symm_mem_for_comm",
                                lambda module, selected=selected: selected.append(
                                    id(module._get_fsdp_state())
                                ),
                            ),
                        ):
                            enable_fsdp_symm_mem(model, scope)
                        expected = states - {sparse} if scope == "dense" else states
                        assert set(selected) == expected
                        assert len(selected) == len(expected)
                # Match the public entry point's CPU/Gloo SUM policy.
                if device.type == "cpu":
                    force_sum_grad_reduction(model)
                tied = not isinstance(model, UntiedBoundarySharedDecoder)
                assert (model.embedding[0].weight is model.head.weight) == tied, (
                    "sharding changed embedding/head parameter identity"
                )
                assert isinstance(model.embedding[0].weight, DTensor)
                assert model.embedding[0].weight.to_local().shape == (4, 4)
                if isinstance(model, CrossSharedDecoder):
                    assert model.left.shared is model.right[0].weight
                    assert model.right.shared is model.left[0].weight
                    assert model.left.shared.to_local().shape == (2, 4)
                if isinstance(model, ReusedDecoder):
                    assert model.alias is model.projection
                    assert isinstance(model.alias.weight, DTensor)
                    assert model.alias.weight.to_local().shape == (2, 4)
                if isinstance(model, SingleExpertDecoder):
                    for parameter in model.block.experts.parameters():
                        assert parameter.to_local().shape[0] == 1
                        if patterns["block"].get("mesh") is replicated_mesh:
                            assert parameter.device_mesh.shape == (2, 1)
                            assert parameter.to_local().shape == parameter.shape
                        else:
                            assert parameter.placements[0].dim == 1
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
                for step in range(3):
                    generator = torch.Generator().manual_seed(100 + step)
                    tokens = torch.randint(8, (2, 6), generator=generator).to(device)
                    labels = torch.randint(8, (2, 6), generator=generator).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
                    reference_loss = F.cross_entropy(
                        reference(tokens).flatten(0, 1), labels.flatten()
                    )
                    reference_loss.backward()
                    local_loss = (
                        F.cross_entropy(
                            model(tokens[rank]).reshape(-1, 8),
                            labels[rank],
                            reduction="sum",
                        )
                        / labels.numel()
                    )
                    local_loss.backward()
                    total_loss = local_loss.detach().clone()
                    dist.all_reduce(total_loss)
                    torch.testing.assert_close(total_loss, reference_loss.detach())
                    reference_parameters = dict(reference.named_parameters())
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(
                            full(parameter.grad),
                            reference_parameters[name].grad,
                            atol=2e-6,
                            rtol=2e-5,
                        )
                    if isinstance(model, RootSharedDecoder):
                        assert model.external_weight is model.block[0].weight
                    optimizer.step()
                    reference_optimizer.step()
                    assert (model.embedding[0].weight is model.head.weight) == tied
                    if isinstance(model, BoundarySharedDecoder):
                        assert model.block[0].bias is model.norm.bias
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(
                            full(parameter),
                            reference_parameters[name],
                            atol=2e-6,
                            rtol=2e-5,
                        )
                if rank == 0:
                    layout = list(patterns) if patterns is not None else "decoder"
                    unit_mesh = (
                        "HSDP(2,1)"
                        if patterns is not None
                        and patterns.get("block", {}).get("mesh") is replicated_mesh
                        else "default"
                    )
                    print(
                        f"PASS {implementation} layout={layout} mesh={unit_mesh}",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
