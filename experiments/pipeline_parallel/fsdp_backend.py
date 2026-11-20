"""Pipeline-specific FSDP policy and assembly adapters.

Decoder wrapping uses production FSDP, including shared-parameter ownership and
expert placement. This module resolves PP resharding defaults and rejects root
boundaries that cannot be retained across pipeline stages.
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from experiments.pipeline_parallel.fsdp_policies import (
    get_fsdp_reshard_after_forward_policy,
)
from experiments.pipeline_parallel.topology import ParallelTopology
from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.accelerator.distributed import get_backend, is_distributed
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.parallel.compatibility import chunked_loss_fsdp
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    apply_fsdp_to_decoder as apply_shared_fsdp_to_decoder,
)
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    force_sum_grad_reduction,
    iter_transformer_layers,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from llmtuner.parallel.fully_shard_parallel.pattern import (
    apply_fsdp_by_pattern,
)
from llmtuner.parallel.fully_shard_parallel.policies import (
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    fsdp_shard_size,
    iter_fsdp_modules,
)
from llmtuner.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "apply_fsdp",
    "apply_fsdp_to_decoder",
    "disable_fsdp_gradient_division",
    "enable_fsdp_symm_mem",
    "fsdp_shard_size",
    "get_fsdp_reshard_after_forward_policy",
    "iter_fsdp_modules",
    "iter_transformer_layers",
    "resolve_fsdp_mesh",
    "resolve_sparse_fsdp_mesh",
]


def apply_fsdp_to_decoder(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    cpu_offload_pin_memory: bool = True,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    symm_mem_scope: str | None = None,
    forward_prefetch_count: int = 0,
    backward_prefetch_count: int = 0,
    fsdp_implementation: str = "native",
    root_owned_boundaries: bool = False,
    error_group: torch.distributed.ProcessGroup | None = None,
) -> None:
    """Apply the experimental PP policy through shared decoder FSDP wrapping."""
    with (
        synchronize_local_errors("Pipeline FSDP boundary validation", group=error_group)
        if error_group is not None
        else nullcontext()
    ):
        if pp_enabled and root_owned_boundaries:
            raise ValueError("root-owned boundaries require native decoder-only FSDP")
        reshard = get_fsdp_reshard_after_forward_policy(
            reshard_after_forward_policy, pp_enabled
        )
    # PP keeps parameters available between microbatches by default. Explicit
    # always/never policies retain their ordinary shared meaning.
    apply_shared_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cpu_offload=cpu_offload,
        cpu_offload_pin_memory=cpu_offload_pin_memory,
        reshard_after_forward_policy="always" if reshard else "never",
        ep_degree=ep_degree,
        edp_mesh=edp_mesh,
        symm_mem_scope=symm_mem_scope,
        forward_prefetch_count=forward_prefetch_count,
        backward_prefetch_count=backward_prefetch_count,
        fsdp_implementation=fsdp_implementation,
        root_owned_boundaries=root_owned_boundaries,
        error_group=error_group,
    )


def apply_fsdp(
    model: torch.nn.Module,
    cfg: ParallelConfig,
    parallel_dims: ParallelTopology | None = None,
    *,
    chunked_loss_num_chunks: int = 1,
) -> torch.nn.Module:
    """Fully-shard ``model`` (FSDP2). No-op when no DP/CP axis is active.

    Applied unconditionally whenever any of ``dp_replicate`` / ``dp_shard`` /
    ``cp`` is active, mirroring torchtitan (``llama3/parallelize.py``):

    * CP must enable FSDP even when ``dp_shard == 1``: every CP rank computes
      the loss over its own sequence shard, so its gradients are partial and
      only become global once reduced across the CP group. The FSDP submesh
      puts ``cp`` on the shard axis for exactly that reason -- parameters are
      sharded over the CP group too, the same semantic torchtitan uses.
    * ``dp_replicate`` alone (pure DDP) must still reduce gradients across
      the replica group: with a shard axis of size 1 the all-gather is a
      no-op and only the gradient all-reduce remains. Skipping FSDP here
      would leave replicas reading different data with gradients never
      reduced.

    The mesh handed to FSDP is the dedicated submesh from
    ``resolve_fsdp_mesh`` -- never the raw storage mesh, whose extra axes
    (``tp``, or more than two active axes) torch's default shape-based
    reading would mis-assign or reject.

    Delegates the wrapping to torchtitan's ``apply_fsdp_to_decoder``; this
    function only decides whether to shard at all, fixes the policy values,
    and applies the backend workaround.
    """
    split_expert_layout = cfg.fsdp_config.fsdp_expert_reduce_layout == "split"
    if parallel_dims is None:
        if split_expert_layout:
            raise UnsupportedCombinationError(
                "split expert reduction requires an actual sharding topology"
            )
        return model

    if split_expert_layout and (
        cfg.fsdp_config.fsdp_implementation != "native"
        or cfg.fsdp_config.fsdp_apply_modules
        or cfg.fsdp_config.enable_fsdp_symm_mem
        or parallel_dims.ep_enabled
        or parallel_dims.tp_enabled
        or parallel_dims.cp_enabled
        or parallel_dims.pp_enabled
        or parallel_dims.dp_replicate_enabled
    ):
        raise UnsupportedCombinationError(
            "split expert reduction requires native plain DP sharding"
        )

    storage_mesh = resolve_fsdp_mesh(parallel_dims)
    edp_mesh = resolve_sparse_fsdp_mesh(parallel_dims)
    # EP borrows ranks from the dense TP/CP/DP shard region. Dense DP can be
    # size 1 while the remaining expert-FSDP (efsdp) axis is still multi-rank.
    if storage_mesh.size() == 1 and (edp_mesh is None or edp_mesh.size() == 1):
        if split_expert_layout:
            raise UnsupportedCombinationError(
                "split expert reduction requires more than one shard rank"
            )
        return model
    if chunked_loss_num_chunks > 1:
        chunked_loss_fsdp(
            chunked_loss_num_chunks,
            fsdp_config=cfg.fsdp_config,
            hybrid_parallel=(
                parallel_dims.tp_enabled
                or parallel_dims.cp_enabled
                or parallel_dims.ep_enabled
                or parallel_dims.pp_enabled
            ),
        )

    # Preserve the dtype selected at model construction. Hard-coding float32
    # here silently makes FSDP all-gather BF16 parameters as FP32, doubling the
    # transient parameter and operator footprint in mixed-precision runs.
    # Model-declared separate parameters get a nested group; all remaining
    # parameters must share the decoder and boundary storage dtype.
    with synchronize_local_errors(
        "FSDP parameter dtype validation",
        group=parallel_dims.get_mesh("loss").get_group() if is_distributed() else None,
    ):
        separate_parameters = {
            parameter
            for module in model.modules()
            for name in getattr(module, "fsdp_separate_parameter_names", ())
            for parameter in (getattr(module, name),)
        }
        param_dtypes = {
            p.dtype for p in model.parameters() if p not in separate_parameters
        }
        if (
            any(p.dtype not in param_dtypes for p in separate_parameters)
            and cfg.fsdp_config.fsdp_apply_modules
        ):
            raise ValueError(
                "Model-declared separate dtype parameters require decoder FSDP "
                "wrapping; fsdp_apply_modules patterns are not supported"
            )
        if len(param_dtypes) > 1:
            raise ValueError(
                "FSDP requires a single parameter dtype, got "
                f"{sorted(map(str, param_dtypes))}. "
                "A model with deliberately fp32 parameters needs per-parameter "
                "dtype overrides, which llmtuner does not wire."
            )
        try:
            param_dtype = next(iter(param_dtypes))
        except StopIteration:
            param_dtype = torch.get_default_dtype()
        reduce_dtype = getattr(torch, cfg.fsdp_config.fsdp_reduce_dtype)

    if cfg.fsdp_config.fsdp_apply_modules:
        apply_fsdp_by_pattern(
            model,
            dp_mesh=storage_mesh,
            cfg=cfg,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            edp_mesh=edp_mesh,
            error_group=parallel_dims.get_mesh("loss").get_group(),
        )
        if is_distributed() and get_backend() != "nccl":
            force_sum_grad_reduction(model)
        return model

    apply_fsdp_to_decoder(
        model,
        storage_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=cfg.fsdp_config.fsdp_cpu_offload,
        cpu_offload_pin_memory=cfg.fsdp_config.fsdp_cpu_offload_pin_memory,
        reshard_after_forward_policy=cfg.fsdp_config.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        symm_mem_scope=(
            cfg.fsdp_config.fsdp_symm_mem_scope
            if cfg.fsdp_config.enable_fsdp_symm_mem
            else None
        ),
        forward_prefetch_count=cfg.fsdp_config.fsdp_forward_prefetch_count,
        backward_prefetch_count=cfg.fsdp_config.fsdp_backward_prefetch_count,
        fsdp_implementation=cfg.fsdp_config.fsdp_implementation,
        root_owned_boundaries=chunked_loss_num_chunks > 1,
        error_group=(
            parallel_dims.get_mesh("loss").get_group() if is_distributed() else None
        ),
    )
    # ``apply_fsdp_to_decoder`` already calls ``disable_fsdp_gradient_division``
    # and, when asked, ``enable_fsdp_symm_mem``; do not repeat them here.

    # gloo implements no PREMUL_SUM; NCCL does, and there it is the faster path.
    if is_distributed() and get_backend() != "nccl":
        force_sum_grad_reduction(model)

    if split_expert_layout:
        from llmtuner.parallel.fully_shard_parallel.expert_reduction_layout import (
            install_split_expert_reduction,
        )

        install_split_expert_reduction(
            model,
            workspace_limit_bytes=cfg.fsdp_config.fsdp_expert_reduce_workspace_limit_bytes,
        )

    return model
