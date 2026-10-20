"""FSDP2 wrapping: ``fully_shard`` applied with llmtuner's mesh conventions.

Dense parameters shard over the dp mesh (cp folded into the shard axis -- CP
gradients are partial and must be reduced, not replicated); EP expert weights
shard over the sparse (efsdp) mesh instead. Mixed precision, reshard policy
and CPU offload are read from the config; the function also decides the
per-module wrap policy (MoE blocks get the sparse treatment by their
``moe_enabled`` flag).
"""

from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard
from torch.nn import ModuleDict

from llmtuner.config.parallel_config import ParallelConfig

from ...accelerator import distributed
from ...accelerator.capabilities import has as capability
from ...utils.logging import get_logger
from ..topology import ParallelTopology
from .fine_grained import get_fsdp_strategy
from .plan import apply_fsdp_by_pattern, set_fsdp_prefetch_windows

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


def iter_transformer_layers(layers: nn.Module) -> Iterator[tuple[Any, nn.Module]]:
    """Yield ``(index, block)`` for the transformer block container.

    torchtitan's ``Decoder.layers`` is a ``ModuleDict`` keyed by index, so it
    iterates with ``.items()``; HF's ``PreTrainedModel`` stores a ``ModuleList``,
    which iterates by position. Both yield the same pairs.
    """
    if isinstance(layers, ModuleDict):
        return iter(layers.items())
    return iter(enumerate(layers))


def resolve_fsdp_mesh(parallel_dims: ParallelTopology) -> DeviceMesh:
    """Build the dense FSDP-only submesh.

    llmtuner's HF models hold plain tensors, so ``fully_shard`` cannot take an
    explicit ``DataParallelMeshDims``: torch requires every parameter to be a
    DTensor on the full SPMD mesh in that mode ("When dp_mesh_dims is
    provided, all parameters must be DTensors ... via distribute_module").
    Without mesh dims, torch reads the mesh by shape alone: 1-D means plain
    FSDP, 2-D means HSDP (dim 0 replicates, dim 1 shards), and anything
    higher-dimensional raises. Handing FSDP the raw multi-axis storage mesh
    therefore mis-assigns the axes whenever that mesh also carries ``tp`` or
    more than two active axes.

    Instead, rebuild a dedicated submesh over exactly the axes torchtitan
    declares in its ``DataParallelMeshDims``:

    * shard: ``dp_shard`` (force-kept-alive in the dense storage mesh even at
      size 1, so pure DDP gets a well-defined HSDP shard axis) plus ``cp``
      when CP is enabled, flattened into a single axis when both are active
      (torchtitan's flattened shard semantics);
    * replicate: ``dp_replicate`` when replication is enabled.

    The result is at most 2-D with the replicate axis first, so torch's
    default reading coincides with the intended one. A size-1 result means no
    DP/CP axis is active and FSDP is a no-op; the caller checks for that.
    """
    shard_axes = ["dp_shard"]
    if parallel_dims.cp_enabled:
        shard_axes.append("cp")
    replicate_axis = "dp_replicate" if parallel_dims.dp_replicate_enabled else None

    axes = ([replicate_axis] if replicate_axis else []) + shard_axes
    submesh = parallel_dims.get_optional_mesh(axes)
    assert submesh is not None  # dp_shard is always kept alive

    if replicate_axis is None and len(shard_axes) == 1:
        # 1-D: plain FSDP over the shard axis.
        return submesh
    if replicate_axis is None:
        # 2-D (dp_shard, cp): flatten into torchtitan's single shard axis.
        return submesh._flatten("dp_shard_cp")
    if len(shard_axes) == 1:
        # 2-D (dp_replicate, dp_shard): HSDP as torch reads it by default.
        return submesh
    # 3-D (dp_replicate, dp_shard, cp): DeviceMesh has no partial flatten, so
    # rebuild from the rank tensor. Row-major reshape keeps dp_replicate on
    # dim 0 and folds (dp_shard, cp) into a single shard dim 1.
    flat_ranks = submesh.mesh.reshape(submesh.mesh.size(0), -1)
    return DeviceMesh(
        submesh.device_type,
        flat_ranks,
        mesh_dim_names=(replicate_axis, "dp_shard_cp"),
    )


def resolve_sparse_fsdp_mesh(parallel_dims: ParallelTopology) -> DeviceMesh | None:
    """Sparse counterpart of ``resolve_fsdp_mesh`` for routed experts.

    Returns ``None`` when EP is disabled. Otherwise rebuilds the FSDP-only
    submesh over ``efsdp`` (shard) and, when enabled, ``dp_replicate``
    (replicate) -- the axes torchtitan declares as
    ``DataParallelMeshDims(shard="efsdp", replicate="dp_replicate")``. The
    raw sparse storage mesh also carries the ``ep`` axis, which torch's
    default 2-D reading would mistake for the shard axis, so it is excluded
    here the same way ``tp`` is excluded from the dense mesh.
    """
    if not parallel_dims.ep_enabled:
        return None
    axes = (["dp_replicate"] if parallel_dims.dp_replicate_enabled else []) + ["efsdp"]
    submesh = parallel_dims.get_optional_mesh(axes)
    assert submesh is not None  # efsdp is kept alive whenever ep > 1
    return submesh


def iter_fsdp_modules(model: nn.Module) -> Iterator[FSDPModule]:
    """Yield every ``FSDPModule`` under ``model`` (``ReplicateModule`` included)."""
    for module in model.modules():
        if isinstance(module, FSDPModule):
            yield module


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """
    Disable FSDP's automatic gradient division for all FSDP modules.

    Set gradient_divide_factor=1.0 to disable FSDP's automatic gradient division.
    We handle gradient scaling ourselves in the training loop with global token count.

    Note: This also works for ReplicateModule since it inherits from FSDPModule.

    Args:
        model: The model containing FSDP-wrapped or Replicate-wrapped modules
    """
    for module in iter_fsdp_modules(model):
        module.set_gradient_divide_factor(1.0)


def fsdp_shard_size(dp_mesh: DeviceMesh) -> int:
    """How many ranks FSDP shards a parameter's dim 0 over, on ``dp_mesh``.

    FSDP cuts a parameter's dim 0 only over its shard axes: ``dp_shard``,
    plus ``cp`` when CP is on (``resolve_fsdp_mesh`` folds the two into
    ``dp_shard_cp``). ``dp_replicate`` replicates, so it must not inflate the
    degree compared against ``num_experts`` -- under HSDP (dp_replicate > 1)
    the raw mesh size would over-count and mis-pick ``Shard(1)`` for the
    expert weights.
    """
    degree = dp_mesh.size()
    if "dp_replicate" in (dp_mesh.mesh_dim_names or ()):
        degree //= dp_mesh["dp_replicate"].size()
    return degree


def build_fsdp_placement_fn(
    expert_placements: dict[nn.Parameter, Shard],
    *,
    dp_mesh: DeviceMesh,
    edp_mesh: DeviceMesh,
) -> Callable[[nn.Parameter], Any]:
    """Keep EP-owned parameters on their sparse FSDP mesh inside any unit."""
    from torch.distributed.fsdp._fully_shard._fsdp_common import (
        FSDPMeshInfo,
        ShardPlacementResult,
    )
    from torch.distributed.fsdp._fully_shard._fsdp_init import _get_mesh_info

    edp_mesh_info = _get_mesh_info(edp_mesh, None)
    dp_mesh_info = _get_mesh_info(dp_mesh, None)
    assert isinstance(edp_mesh_info, FSDPMeshInfo)
    assert isinstance(dp_mesh_info, FSDPMeshInfo)

    def placement(param: nn.Parameter) -> ShardPlacementResult:
        expert_placement = expert_placements.get(param)
        if expert_placement is not None:
            return ShardPlacementResult(
                placement=expert_placement, mesh_info=edp_mesh_info
            )
        return ShardPlacementResult(placement=Shard(0), mesh_info=dp_mesh_info)

    return placement


def shard_experts_on_their_own_mesh(
    shard_fn: Callable[..., Any],
    experts: nn.Module,
    *,
    edp_mesh: DeviceMesh,
    fsdp_config: dict[str, Any],
    reshard_after_forward: bool | int,
) -> set[nn.Parameter]:
    """Shard a MoE experts child on the sparse mesh as its own FSDP unit.

    The nested form of what ``build_fsdp_placement_fn`` expresses on torch >=
    2.12, where one ``fully_shard`` takes a placement function that returns both
    a ``Shard`` and a mesh, so a single unit can own parameters on two meshes.
    torch 2.9's ``shard_placement_fn`` returns only the ``Shard`` and has no
    mesh slot, so the same split is expressed structurally instead: the experts
    child becomes its own FSDP unit on the sparse mesh, and the caller keeps
    the enclosing unit off those parameters via ``ignored_params``.

    Returns the child's parameters read *after* sharding: FSDP2 replaces the
    parameter objects it converts (they come back as DTensors), and
    ``ignored_params`` matches by identity, so the pre-shard objects would not
    match anything.
    """
    experts_config = dict(fsdp_config, mesh=edp_mesh)
    shard_fn(experts, **experts_config, reshard_after_forward=reshard_after_forward)
    return set(experts.parameters())


def enable_fsdp_symm_mem(model: nn.Module, scope: str | None = "all") -> None:
    """Enable symmetric-memory communication for the FSDP modules ``scope`` selects.

    ``None`` disables it. ``"all"`` covers every FSDP module; ``"dense"``
    skips states owning any ``moe_enabled`` subtree, including grouped dense/MoE
    consumers and root-managed MoE children. Settings apply once per state, so
    attention parameters sharing an MoE state are skipped with its experts.
    Symmetric memory is not always beneficial for expert communication.
    """
    if scope is None:
        return
    if scope not in ("all", "dense"):
        raise ValueError(
            f"enable_fsdp_symm_mem scope must be one of 'all', 'dense', None; "
            f"got {scope!r}"
        )
    states: dict[int, FSDPModule] = {}
    sparse_states: set[int] = set()
    pending = [(model, None, False)]
    visited = set()
    while pending:
        module, owner, in_moe = pending.pop()
        if isinstance(module, FSDPModule):
            owner = id(module._get_fsdp_state())
            states.setdefault(owner, module)
        in_moe = in_moe or getattr(module, "moe_enabled", False)
        key = (id(module), owner, in_moe)
        if key in visited:
            continue
        visited.add(key)
        if in_moe and owner is not None:
            sparse_states.add(owner)
        pending.extend((child, owner, in_moe) for child in module.children())
    # Grouped modules share communication settings. Decide the complete scope
    # before configuring anything, including MoE children managed by a root state.
    for owner, module in states.items():
        if scope == "dense" and owner in sparse_states:
            continue
        module.set_force_sum_reduction_for_comms(True)
        module.set_symm_mem_for_comm()


def get_fsdp_reshard_after_forward_policy(
    reshard_after_forward_policy: str, pp_enabled: bool
) -> bool:
    """Resolve fsdp_reshard_after_forward policy string to a boolean.

    Args:
        reshard_after_forward_policy: One of "always", "never", or "default".
        pp_enabled: Whether pipeline parallelism is enabled.

    Returns:
        Boolean indicating whether to reshard after forward.
    """
    match reshard_after_forward_policy:
        case "always":
            return True
        case "never":
            return False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            return not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )


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
) -> None:
    """Apply FSDP in place to a decoder-style transformer model.

    Shared by all dense and MoE decoders (llama3, qwen3, deepseek_v3,
    gpt_oss, qwen3_vl, ...). The MoE handling is a strict superset of the dense
    case: a dense model leaves ``ep_degree=1`` / ``edp_mesh=None`` and has no
    ``moe_enabled`` blocks, so every transformer block is sharded as a single
    FSDP unit and the expert-parallel prefetching below is skipped.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reductions.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to
            CPU. Defaults to False.
        cpu_offload_pin_memory (bool, optional): Whether CPU-offloaded buffers
            use pinned host memory. Defaults to True.
        reshard_after_forward_policy (str, optional): The policy to use for
            resharding after the forward pass. Defaults to "default". Other
            options: "never", "always".
            - "default" applies default resharding behavior, implementing
              "smart defaults" for known optimal scenarios.
            - "always" enables ``reshard_after_forward`` for all forward passes.
            - "never" disables ``reshard_after_forward`` for all forward passes.
        ep_degree (int, optional): Expert-parallel degree. Defaults to 1 (no EP),
            in which case the MoE-specific sharding and prefetching are no-ops.
        edp_mesh (DeviceMesh | None, optional): The FSDP mesh for routed experts
            when EP > 1. Required (non-None) iff ``ep_degree > 1``.
        symm_mem_scope (str | None): Symmetric-memory scope passed to
            ``enable_fsdp_symm_mem``: ``None`` disables it, ``"all"`` covers
            every FSDP module, ``"dense"`` skips MoE (sparse) blocks.

    Upstream additionally takes ``dp_mesh_dims``/``edp_mesh_dims``: the
    explicit declaration of which axes of a multi-dimensional SPMD mesh are
    data-parallel. llmtuner does not carry them. ``resolve_fsdp_mesh`` /
    ``resolve_sparse_fsdp_mesh`` hand FSDP a dedicated 1-D/2-D submesh, which
    torch's default shape-based reading resolves to exactly those axes; and
    since llmtuner's parameters are plain tensors rather than DTensors on the
    full SPMD mesh, the explicit form is not available here anyway (see
    ``resolve_fsdp_mesh``).
    """
    strategy = get_fsdp_strategy(fsdp_implementation)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    shard_fn = (
        fully_shard
        if fsdp_implementation == "native"
        else strategy.get_fully_shard_fn()
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy(
            pin_memory=cpu_offload_pin_memory
        )

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    # Resolve ownership before any sharding replaces Parameters. Include
    # the decoder boundary units so a layer/norm/head alias is not split into
    # independent sharded tensors. Existing tied embedding/head units stay intact.
    units: list[tuple[list[nn.Module], bool]] = []
    boundary = [
        module
        for module in (model.tok_embeddings, model.norm, model.lm_head)
        if module is not None
    ]
    final_reshard = reshard_after_forward_policy == "always"
    if model.enable_weight_tying:
        if boundary:
            units.append((boundary, final_reshard))
    else:
        if model.tok_embeddings is not None:
            units.append(([model.tok_embeddings], reshard_after_forward))
        if model.norm is not None and model.lm_head is not None:
            units.append(([model.norm, model.lm_head], final_reshard))
    boundary_count = len(units)
    units.extend(
        ([block], reshard_after_forward)
        for _, block in iter_transformer_layers(model.layers)
    )
    # Build one placement map before parameter replacement. A shared unit may
    # span dense and MoE layers, but every expert retains its own mesh/layout.
    expert_placements: dict[nn.Parameter, Shard] = {}
    expert_units: set[int] = set()
    for index, (modules, _) in enumerate(units):
        for module in modules:
            if not getattr(module, "moe_enabled", False):
                continue
            expert_units.add(index)
            experts = module.moe.routed_experts.inner_experts
            num_experts = module.moe.router.num_experts
            if ep_degree > 1:
                assert edp_mesh is not None
                shard_degree = edp_mesh["efsdp"].size() * ep_degree
            else:
                shard_degree = fsdp_shard_size(dp_mesh)
            placement = Shard(1 if shard_degree > num_experts else 0)
            for parameter in experts.parameters():
                previous = expert_placements.setdefault(parameter, placement)
                if previous != placement:
                    raise ValueError(
                        "Shared expert parameters have conflicting FSDP placements"
                    )
    if ep_degree > 1 and expert_placements:
        assert edp_mesh is not None
        # One fully_shard unit has to own parameters on two meshes here: the
        # experts on the sparse (efsdp) mesh, the rest on the dense one.
        # torch >= 2.12 expresses that with a placement function returning both
        # a Shard and a mesh; older torch's ``shard_placement_fn`` returns only
        # the Shard, so EP x FSDP would shard the experts over the dense mesh
        # -- mixing ranks of different EP coordinates, which no loss curve
        # reveals. Without the capability, ``shard_unit`` expresses the same
        # split structurally through ``shard_experts_on_their_own_mesh``.
        if capability("fsdp_per_param_mesh"):
            expert_placement_fn = build_fsdp_placement_fn(
                expert_placements, dp_mesh=dp_mesh, edp_mesh=edp_mesh
            )
        else:
            expert_placement_fn = None
    else:

        def expert_placement_fn(parameter: nn.Parameter) -> Shard | None:
            return expert_placements.get(parameter)

    # Root-owned registrations may alias a parameter inside a decoder unit.
    # Exclude every planned layer subtree when collecting the remaining owners;
    # recursively inspecting the root would incorrectly merge all units.
    nested_modules = {
        descendant
        for module in [
            *boundary,
            *[block for _, block in iter_transformer_layers(model.layers)],
        ]
        for descendant in module.modules()
    }
    root_parameters = {
        parameter
        for module in model.modules()
        if module not in nested_modules
        for parameter in module.parameters(recurse=False)
    }
    root_index = len(units)
    units.append(([model], False))
    groups = {index: {index} for index in range(len(units))}
    parameter_owners: dict[nn.Parameter, int] = {}
    for index, (modules, _) in enumerate(units):
        for module in modules:
            parameters = root_parameters if index == root_index else module.parameters()
            for parameter in parameters:
                previous = parameter_owners.setdefault(parameter, index)
                merged = groups[previous] | groups[index]
                for owner in merged:
                    groups[owner] = merged
    module_units = {
        module: index for index, (modules, _) in enumerate(units) for module in modules
    }
    sharded_units: set[int] = set()

    def shard_unit(index: int) -> None:
        if root_index in groups[index] and index != root_index:
            return  # Root and its shared consumers must be transformed last.
        if index in sharded_units:
            return
        owners = sorted(groups[index])
        modules = list(
            dict.fromkeys(module for owner in owners for module in units[owner][0])
        )
        # One state has one reshard policy. Retain any boundary unit's request
        # to keep parameters available through its last forward consumer.
        reshard = all(units[owner][1] for owner in owners)
        options = dict(fsdp_config)
        if expert_units.intersection(owners):
            if expert_placement_fn is not None:
                options["shard_placement_fn"] = expert_placement_fn
            else:
                # Nested-unit fallback (torch < 2.12): shard every experts
                # child on the sparse mesh as its own FSDP unit first, then
                # keep the group off the sharded parameters.
                assert edp_mesh is not None
                ignored: set[nn.Parameter] = set()
                expert_children = dict.fromkeys(
                    module.moe.routed_experts.inner_experts
                    for module in modules
                    if getattr(module, "moe_enabled", False)
                )
                for experts in expert_children:
                    ignored.update(
                        shard_experts_on_their_own_mesh(
                            shard_fn,
                            experts,
                            edp_mesh=edp_mesh,
                            fsdp_config=fsdp_config,
                            reshard_after_forward=reshard,
                        )
                    )
                options["ignored_params"] = ignored
        shard_fn(
            modules[0] if len(modules) == 1 else modules,
            **options,
            reshard_after_forward=reshard,
        )
        sharded_units.update(owners)

    for index in range(boundary_count):
        shard_unit(index)

    for _, transformer_block in iter_transformer_layers(model.layers):
        shard_unit(module_units[transformer_block])

    shard_unit(root_index)

    # None is a no-op; an unknown scope raises inside.
    enable_fsdp_symm_mem(model, symm_mem_scope)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)

    # HSDP when the data-parallel mesh carries a replicate axis, else pure FSDP.
    if "dp_replicate" in (dp_mesh.mesh_dim_names or ()):
        logger.info("Applied HSDP to the model")
    else:
        logger.info("Applied FSDP to the model")
    if cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    # EP keeps its historical one-unit explicit prefetch by default because
    # device/host synchronization in dispatch can disrupt implicit prefetch.
    forward_window = forward_prefetch_count or (1 if ep_degree > 1 else 0)
    backward_window = backward_prefetch_count or (1 if ep_degree > 1 else 0)
    if forward_window or backward_window:
        execution_order = [
            module
            for module in (
                model.tok_embeddings,
                *[block for _, block in iter_transformer_layers(model.layers)],
                model.norm,
                model.lm_head,
            )
            if isinstance(module, FSDPModule)
        ]
        set_fsdp_prefetch_windows(
            execution_order,
            forward_count=forward_window,
            backward_count=backward_window,
        )


def force_sum_grad_reduction(model: torch.nn.Module) -> None:
    """Use plain SUM for FSDP's gradient reduce comms.

    FSDP defaults to ``ReduceOp.PREMUL_SUM``, which pre-scales the local
    gradient so the collective needs no separate scaling kernel -- but only NCCL
    implements it, so every other backend raises. Forcing plain SUM gives up
    that optimization, which is why the caller only turns it on off-NCCL.
    """
    for module in iter_fsdp_modules(model):
        module.set_force_sum_reduction_for_comms(True)


def apply_fsdp(
    model: torch.nn.Module,
    cfg: ParallelConfig,
    parallel_dims: ParallelTopology | None = None,
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
    if parallel_dims is None:
        return model

    storage_mesh = resolve_fsdp_mesh(parallel_dims)
    edp_mesh = resolve_sparse_fsdp_mesh(parallel_dims)
    # EP borrows ranks from the dense TP/CP/DP shard region. Dense DP can be
    # size 1 while the remaining expert-FSDP (efsdp) axis is still multi-rank.
    if storage_mesh.size() == 1 and (edp_mesh is None or edp_mesh.size() == 1):
        return model

    # Preserve the dtype selected at model construction. Hard-coding float32
    # here silently makes FSDP all-gather BF16 parameters as FP32, doubling the
    # transient parameter and operator footprint in mixed-precision runs.
    # A single dtype is asserted rather than assumed: with mixed-dtype
    # parameters FSDP would all-gather the minority dtype AS the first
    # parameter's dtype, silently downcasting them (upstream has a
    # param_dtype_override_fn hook for this; llmtuner refuses instead).
    param_dtypes = {p.dtype for p in model.parameters()}
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
        )
        if distributed.is_distributed() and distributed.get_backend() != "nccl":
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
    )
    # ``apply_fsdp_to_decoder`` already calls ``disable_fsdp_gradient_division``
    # and, when asked, ``enable_fsdp_symm_mem``; do not repeat them here.

    # gloo implements no PREMUL_SUM; NCCL does, and there it is the faster path.
    if distributed.is_distributed() and distributed.get_backend() != "nccl":
        force_sum_grad_reduction(model)

    return model
