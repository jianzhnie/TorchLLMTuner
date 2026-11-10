"""Independent VP mesh assembly, secondary FSDP and checkpoint conversion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from ..module_patterns import resolve_module_patterns
from .dispatcher import vocab_parallel_dispatch_forward

__all__ = [
    "RoutedVocabParallelEmbedding",
    "VocabParallelPlan",
    "gather_vocab_parallel_weight",
    "is_vfsdp_enabled",
    "load_vocab_parallel_weight",
    "vocab_fully_shard_modules",
    "vocab_parallelize_modules",
]


@dataclass(kw_only=True, slots=True)
class VocabParallelPlan:
    """Module selection and mixed-precision policy for independent VP.

    VP ranks hold different tokens, unlike TP ranks holding sequence shards.
    VFSDP ranks cover data replicas at one VP coordinate. Their group intersects
    the VP group only at the current rank. Singleton VFSDP leaves weights
    outside its management;
    callers must exclude them from an ancestor FSDP unit.
    """

    apply_modules: list[str]
    apply_vfsdp_modules: list[str] | None = None
    gradient_divide_factor: float | None = None
    output_dtype: torch.dtype | None = None
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    cpu_offload: bool = False
    cpu_offload_pin_memory: bool = True

    def __post_init__(self) -> None:
        if not self.apply_modules:
            raise ValueError("VocabParallelPlan.apply_modules must not be empty")
        if self.apply_vfsdp_modules is None:
            self.apply_vfsdp_modules = list(self.apply_modules)
        if self.gradient_divide_factor is None:
            self.gradient_divide_factor = float(
                dist.get_world_size() if dist.is_initialized() else 1
            )
        if self.gradient_divide_factor <= 0:
            raise ValueError("gradient_divide_factor must be positive")


class RoutedVocabParallelEmbedding(nn.Embedding):
    """Embedding whose DTensor weight and token dispatch share one VP mesh."""

    def __init__(
        self,
        source: nn.Embedding,
        *,
        vp_mesh: DeviceMesh,
        gradient_divide_factor: float,
    ) -> None:
        if vp_mesh.ndim != 1:
            raise ValueError("VP requires a one-dimensional device mesh")
        if source.max_norm is not None:
            raise ValueError("Vocab parallel does not support max_norm")
        if source.scale_grad_by_freq:
            raise ValueError("Vocab parallel does not support scale_grad_by_freq")
        if source.sparse:
            raise ValueError("Vocab parallel does not support sparse gradients")
        vp_size = vp_mesh.size()
        if source.num_embeddings % vp_size:
            raise ValueError(
                f"num_embeddings ({source.num_embeddings}) must be divisible by "
                f"VP size ({vp_size})"
            )
        super().__init__(
            source.num_embeddings,
            source.embedding_dim,
            padding_idx=source.padding_idx,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        vp_rank = vp_mesh.get_local_rank()
        weight = source.weight.detach()
        if weight.device.type != vp_mesh.device_type and not weight.is_meta:
            local_weight = weight.chunk(vp_size, dim=0)[vp_rank].to(vp_mesh.device_type)
            distributed_weight = DTensor.from_local(
                local_weight,
                vp_mesh,
                [Shard(0)],
                run_check=False,
                shape=source.weight.shape,
                stride=source.weight.stride(),
            )
        else:
            distributed_weight = distribute_tensor(weight, vp_mesh, [Shard(0)])
        self.weight = nn.Parameter(
            distributed_weight, requires_grad=source.weight.requires_grad
        )
        self.vp_mesh = vp_mesh
        self.vp_group = vp_mesh.get_group()
        self.vp_gradient_divide_factor = gradient_divide_factor
        self.vp_output_dtype: torch.dtype | None = None
        self.train(source.training)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return vocab_parallel_dispatch_forward(self, input_ids, self.vp_group)


def _selected_embeddings(
    model: nn.Module, patterns: list[str]
) -> list[tuple[str, nn.Embedding]]:
    matches = resolve_module_patterns(
        [name for name, _ in model.named_modules()],
        patterns=patterns,
        plan_name="Vocab parallel",
    )
    selected = []
    for name in sorted(matches):
        module = model.get_submodule(name)
        if not isinstance(module, nn.Embedding):
            raise TypeError(f"Vocab parallel module {name!r} must be nn.Embedding")
        selected.append((name, module))
    return selected


def vocab_parallelize_modules(
    model: nn.Module, vp_mesh: DeviceMesh, plan: VocabParallelPlan
) -> nn.Module:
    """Replace selected embeddings with owner-routed VP modules in place."""
    targets = _selected_embeddings(model, plan.apply_modules)
    parameter_counts: dict[int, int] = {}
    for _, parameter in model.named_parameters(remove_duplicate=False):
        parameter_counts[id(parameter)] = parameter_counts.get(id(parameter), 0) + 1
    for path, source in targets:
        if parameter_counts[id(source.weight)] != 1:
            raise ValueError(f"Vocab parallel over {path} cannot preserve tied weights")
    replacements = [
        (
            path,
            RoutedVocabParallelEmbedding(
                source,
                vp_mesh=vp_mesh,
                gradient_divide_factor=plan.gradient_divide_factor,
            ),
        )
        for path, source in targets
    ]
    for path, replacement in replacements:
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, replacement)
    return model


def is_vfsdp_enabled(vfsdp_mesh: DeviceMesh) -> bool:
    """Return whether secondary vocabulary FSDP spans multiple ranks."""
    return vfsdp_mesh.size() > 1


def _validate_vfsdp_mesh(vp_mesh: DeviceMesh, vfsdp_mesh: DeviceMesh) -> None:
    if vfsdp_mesh.ndim != 1:
        raise ValueError("VFSDP requires a one-dimensional device mesh")
    if vfsdp_mesh.device_type != vp_mesh.device_type:
        raise ValueError("VP and VFSDP meshes must use the same device type")
    if not is_vfsdp_enabled(vfsdp_mesh):
        return
    vp_ranks = set(dist.get_process_group_ranks(vp_mesh.get_group()))
    vfsdp_ranks = set(dist.get_process_group_ranks(vfsdp_mesh.get_group()))
    if vp_ranks & vfsdp_ranks != {dist.get_rank()}:
        raise ValueError(
            "VP and VFSDP groups must intersect only at the current rank: VP "
            "exchanges different tokens, while VFSDP shards the same VP-local "
            "table across data replicas"
        )


def vocab_fully_shard_modules(
    model: nn.Module,
    vfsdp_mesh: DeviceMesh,
    vp_plan: VocabParallelPlan,
) -> nn.Module:
    """Apply secondary FSDP, or configure the pure-VP communication policy."""
    targets = _selected_embeddings(model, vp_plan.apply_vfsdp_modules or [])
    if any(
        not isinstance(module, RoutedVocabParallelEmbedding) for _, module in targets
    ):
        raise ValueError("VFSDP targets must first be VP-parallelized")
    vp_mesh = targets[0][1].vp_mesh
    if any(module.vp_mesh is not vp_mesh for _, module in targets):
        raise ValueError("All VFSDP targets must share one VP mesh")
    _validate_vfsdp_mesh(vp_mesh, vfsdp_mesh)
    if not is_vfsdp_enabled(vfsdp_mesh):
        for _, embedding in targets:
            embedding.vp_gradient_divide_factor = vp_plan.gradient_divide_factor
            embedding.vp_output_dtype = vp_plan.output_dtype or vp_plan.param_dtype
        return model

    mp_policy = MixedPrecisionPolicy(
        param_dtype=vp_plan.param_dtype,
        reduce_dtype=vp_plan.reduce_dtype,
        cast_forward_inputs=False,
    )
    for _, embedding in targets:
        # FSDP owns gradient averaging here; scaling the return all-to-all too
        # would divide the same gradient twice.
        embedding.vp_gradient_divide_factor = 1.0
        options = {
            "mesh": vfsdp_mesh,
            "mp_policy": mp_policy,
            "shard_placement_fn": lambda _param: Shard(0),
        }
        if vp_plan.cpu_offload:
            options["offload_policy"] = CPUOffloadPolicy(
                pin_memory=vp_plan.cpu_offload_pin_memory
            )
        fully_shard(embedding, **options)
        if vp_plan.gradient_divide_factor != 1.0:
            embedding.set_gradient_divide_factor(vp_plan.gradient_divide_factor)
    return model


def gather_vocab_parallel_weight(
    embedding: RoutedVocabParallelEmbedding,
) -> torch.Tensor:
    """Reconstruct the global table without changing its live sharded storage."""
    local_shard = embedding.weight.full_tensor()
    if local_shard.shape[0] == embedding.num_embeddings:
        return local_shard
    pieces = [
        torch.empty_like(local_shard)
        for _ in range(dist.get_world_size(embedding.vp_group))
    ]
    dist.all_gather(pieces, local_shard.contiguous(), group=embedding.vp_group)
    return torch.cat(pieces, dim=0)


def load_vocab_parallel_weight(
    embedding: RoutedVocabParallelEmbedding, weight: torch.Tensor
) -> None:
    """Copy a full global table into existing VP/VFSDP parameter storage."""
    if tuple(weight.shape) != (embedding.num_embeddings, embedding.embedding_dim):
        expected = (embedding.num_embeddings, embedding.embedding_dim)
        raise ValueError(
            f"Expected global embedding shape {expected}, got {tuple(weight.shape)}"
        )
    parameter = embedding.weight
    local_shard = weight
    if parameter.shape != weight.shape:
        local_shard = weight.chunk(embedding.vp_mesh.size(), dim=0)[
            embedding.vp_mesh.get_local_rank()
        ]
    local_shard = local_shard.contiguous()
    value = distribute_tensor(local_shard, parameter.device_mesh, parameter.placements)
    with torch.no_grad():
        parameter.copy_(value)
