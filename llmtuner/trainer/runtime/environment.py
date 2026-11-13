"""Resolve process groups, topology, device and RNG before model construction.

The entry point owns process groups. This module returns explicit environment
values without allocating a model or writing attributes onto a Trainer.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from ...accelerator.device import (
    device_type,
    get_distributed_backend,
    get_env_dist_info,
)
from ...accelerator.distributed import (
    get_default_group,
    get_rank,
    get_world_size,
    init_dist_pytorch,
    is_distributed,
)
from ...parallel.topology import (
    ParallelTopology,
    build_device_mesh,
    resolve_parallel_topology,
)
from ...utils import seeding

if TYPE_CHECKING:
    from ...config.llmtuner_config import LLMTunerConfig


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    """Resolved rank and mesh dependencies, without model or service ownership.

    Process groups remain owned by the entry point. Constructing this result
    seeds execution settings but does not allocate a model or training service.
    """

    rank: int
    world_size: int
    local_rank: int
    parallel_dims: ParallelTopology | None
    device: torch.device
    mesh: DeviceMesh | None
    error_group: ProcessGroup | None


def build_runtime_environment(
    cfg: LLMTunerConfig,
    *,
    seed_everything: Callable[..., None],
    validate_validation: Callable[..., None],
) -> RuntimeEnvironment:
    """Resolve topology, validate evaluation and seed before model construction.

    Callbacks preserve the public Trainer customization points without giving
    this builder access to a Trainer or its mutable runtime attributes.
    """
    if cfg.deterministic:
        seeding.set_deterministic_algorithms()
    if not is_distributed() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist_pytorch(get_distributed_backend())
    error_group = get_default_group()
    rank, world_size, local_rank = get_env_dist_info()
    if is_distributed():
        # Programmatic callers may initialize the group without torchrun's
        # environment. Mesh degrees must describe the live group.
        rank = get_rank()
        world_size = get_world_size()

    # Resolve the degrees first: the PP seed offset below needs this rank's
    # stage coordinate, and degree resolution draws no random numbers, so
    # seeding after it leaves every non-PP run bit-identical.
    parallel_dims = resolve_parallel_topology(cfg, world_size=world_size)

    # Validation's infeasible combinations are rejected here, before the
    # model and dataloader exist: a ``steps=-1`` pass that cannot terminate
    # cleanly would otherwise hang on its collectives mid-run.
    if cfg.validation is not None:
        validate_validation(
            cfg.validation,
            dp_world_size=(
                1
                if parallel_dims is None
                else parallel_dims.dp_replicate * parallel_dims.dp_shard
            ),
            chunked_loss_num_chunks=cfg.training.chunked_loss_num_chunks,
            training_dataset=cfg.dataloader.dataset,
        )

    # Seed before model construction. PP stages use distinct seeds; ranks
    # sharing a stage retain identical initialization.
    seed = cfg.seed
    if parallel_dims is not None and parallel_dims.pp_enabled:
        pp_mesh = parallel_dims.get_optional_mesh("pp")
        seed = seeding.derive_distinct_seed(
            seed, [(pp_mesh.get_local_rank(), pp_mesh.size())]
        )
    seed_everything(
        seed, deterministic=cfg.deterministic, detect_anomaly=cfg.detect_anomaly
    )

    device = torch.device(
        f"{device_type}:{local_rank}" if device_type != "cpu" else "cpu"
    )

    # PP needs the dense mesh for this stage, which covers only its ranks.
    if parallel_dims is not None and parallel_dims.pp_enabled:
        mesh = parallel_dims.spmd_dense_mesh()
    else:
        mesh = build_device_mesh(parallel_dims)

    seeding.initialize_dtensor_rng(mesh)

    return RuntimeEnvironment(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        parallel_dims=parallel_dims,
        device=device,
        mesh=mesh,
        error_group=error_group,
    )
