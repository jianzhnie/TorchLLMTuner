"""Activation checkpoint policies applied to decoder layers before FSDP.

Modes share the existing configuration and preserve module ownership:

* full: non-reentrant checkpointing with RNG preservation; retain NPU
  reduce-scatter outputs so replay cannot change low-precision MoE routing.
* selective: per-op save policy for expensive compute and collectives, with
  alternating matmul recomputation (see get_default_save_ops).
* memory_budget: set the compile partitioner's process-global activation
  budget; requires model compilation and the corresponding torch capability.
* region: annotate structural regions and checkpoint through optional
  torch_remat. Region names and matching live in remat_regions.py.

Checkpoint contexts preserve each outstanding forward's padding masks,
auxiliary-loss state and SPMD scope during backward replay. Full/selective
wrappers retain torch's early-stop behavior. The Dynamo LRU knob is changed
only when applying a policy, and only on torch builds that expose it.

Upstream provenance and implementation differences are recorded in
``docs/llmtuner_upstream_map.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from types import ModuleType

import torch
import torch._functorch.config
import torch.nn as nn
from torch._functorch.partitioners import get_default_op_list
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from llmtuner.config.activation_checkpoint_config import (
    VALID_AC_MODES,
    MemoryBudgetACConfig,
    RegionACConfig,
    SelectiveACConfig,
)

from ..accelerator.capabilities import has
from ..accelerator.spmd import capture_spmd_context
from ..errors import EnvironmentUnsupportedError
from ..models.common.aux_loss import AuxLoss, in_checkpoint_recompute
from ..models.common.moe.layer import MoE
from ..models.contracts import DecoderStructure
from ..utils.logging import get_logger
from .remat_regions import (
    region_names,
    region_policy,
    unmatched_region_patterns,
)

logger = get_logger(__name__)

try:
    import torch_remat
except ImportError as error:
    torch_remat = None
    REMAT_IMPORT_ERROR = error


__all__ = [
    "apply_ac",
    "full_policy",
    "require_torch_remat",
    "validate_ac_options",
    "wrap_full",
    "wrap_region",
]


def get_default_save_ops() -> set:
    """The ops whose activations ``"selective"`` saves rather than recomputes.

    Two sources, ported from upstream: torch's own list of compute-intensive
    ops (``get_default_op_list``), plus the explicit sets below. Each spec in
    those sets is either an op handle (always present) or a ``(root, dotted
    path)`` pair for an op that only exists in some builds -- resolved through
    ``getattr`` and skipped when it is not registered, so the same list runs
    against a CPU-only torch.

    The comm ops at the end are worth keeping pointed at even though llmtuner
    runs no DeepEP/HybridEP: they resolve-or-skip, so their absence is silent
    here while their presence (a future EP backend) is exactly when
    re-communication would be the thing to avoid.

    ``aten.topk`` is upstream's one deliberate omission. It saves topk because
    topk can be non-deterministic, so a recompute can pick different experts
    than the forward did. llmtuner cannot save it: HF's MoE routers normalize
    the values topk returns with an in-place divide (``router_top_value /=
    ...``), and torch's selective checkpoint raises "Tensor cached during
    selective activation checkpoint has been mutated" when a cached output is
    written afterwards -- on every MoE model, on any device. So a
    selective-AC run over HF MoE recomputes its topk and inherits the
    non-determinism: on a GPU whose topk kernel is not reproducible, the
    backward is then the gradient of a slightly different expert assignment.
    Expert assignment is a piecewise-constant partition of the scores, so a
    flip needs two experts to land on exactly equal scores; on CPU torches'
    topk is deterministic and the two runs agree bitwise (pinned by
    ``test_selective_ac_matches_uncheckpointed_bitwise_on_moe``).
    """
    # Outputs that are expensive to recompute (matmuls, attention, ...).
    compute_ops = [
        # SDPA variants
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_attention_math.default,
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        # Low-precision training always saves the absolute maximum used to
        # compute the quantization scaling factor.
        torch.ops.aten.max.default,
        # FlexInnerAttention (torch.ops.higher_order.flex_attention is the
        # same object).
        torch._higher_order_ops.flex_attention,
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.dtype,
        # Inductor-compiled code (only present when torch.compile is used).
        (torch._higher_order_ops, "inductor_compiled_code"),
        # torch_attn custom backend.
        (torch.ops, "torch_attn._varlen_attn.default"),
    ]

    # Communication ops: saving their outputs avoids re-communicating.
    comm_ops = [
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops._c10d_functional.all_to_all_single.default,
        # DeepEP (only present when it is installed).
        (torch.ops, "deepep.dispatch.default"),
        (torch.ops, "deepep.combine.default"),
        # HybridEP (only present when it is installed).
        (torch.ops, "hybridep.dispatch.default"),
        (torch.ops, "hybridep.combine.default"),
    ]

    def resolve(op_specs: list) -> set:
        # Upstream builds a dict here and then only ever uses its keys; a set
        # is the same value with the unused mapping dropped.
        ops = set()
        for spec in op_specs:
            if isinstance(spec, tuple):
                obj, path = spec
                try:
                    for part in path.split("."):
                        obj = getattr(obj, part)
                    ops.add(obj)
                except AttributeError:
                    pass
            else:
                ops.add(spec)
        return ops

    save_ops = {op.default for op in get_default_op_list().compute_intensive_ops}
    save_ops.update(resolve(compute_ops))
    save_ops.update(resolve(comm_ops))
    return save_ops


def mm_recompute_shapes(
    module: nn.Module, *, base_fqn: str | None, fqns: Sequence[str]
) -> set[tuple[int, int]]:
    """Collect the ``(in, out)`` weight shapes to force-recompute, by fqn.

    ``fqns`` are matched as substrings of each submodule's fully qualified
    name, exactly as upstream matches them -- so a pattern matches anywhere in
    the path, and the shape it yields applies to *any* matmul with that shape,
    not just the one whose module matched.
    """
    shapes: set[tuple[int, int]] = set()
    for module_fqn, submod in module.named_modules():
        fqn = f"{base_fqn}.{module_fqn}" if base_fqn else module_fqn
        if not any(f in fqn for f in fqns):
            continue
        if not isinstance(submod, nn.Linear):
            raise ValueError(
                "force_recompute_mm_shapes_by_fqns expected to match a "
                f"nn.Linear, but got: {submod}"
            )
        # ``shape[-1]`` / ``numel() // in_f`` instead of unpacking a 2D shape,
        # so a stacked weight (extra leading projection dim) resolves to the
        # same (in, out) pair its per-projection GEMMs run at.
        in_f = submod.weight.shape[-1]
        out_f = submod.weight.numel() // in_f
        shapes.add((in_f, out_f))
    return shapes


# Some backends (e.g. PrivateUse1) register aten.linear as a leaf op instead of
# decomposing it into aten.mm, so both spellings have to be handled.
_MM_OPS = (
    torch.ops.aten.mm.default,
    torch.ops.aten.mm.dtype,
    torch.ops.aten.linear.default,
)


def selective_policy(
    save_ops: set, mm_recompute_shapes: set[tuple[int, int]]
) -> Callable:
    """Build the per-op policy the selective context consults (upstream's
    ``_get_custom_policy``).

    The matmul-alternation counters live on the returned closure, so the
    caller must build one policy per ``context_fn`` invocation (i.e. per
    checkpoint frame), as upstream does. A policy shared across pending
    forwards keeps counting across them: with an odd number of matmuls per
    forward the second frame's parity flips, and its backward-time replay then
    disagrees with the decisions its forward saved by, silently replaying the
    wrong tensors.
    """
    meta = {"forward_mm_count": 0, "recompute_mm_count": 0}

    def wrapped_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
        # Always save CUDA -> CPU results rather than recomputing them (e.g. a
        # MoE D2H sync for all-to-all metadata).
        if (
            func == torch.ops.aten._to_copy.default
            and "cuda" in str(args[0].device)
            and "device" in kwargs
            and str(kwargs["device"]) == "cpu"
        ):
            return CheckpointPolicy.MUST_SAVE

        mode = "recompute" if ctx.is_recompute else "forward"
        mm_count_key = f"{mode}_mm_count"

        if func in _MM_OPS:
            weight_shape = args[1].shape
            # linear's weight is (out, in); normalize to mm's (in, out).
            if func == torch.ops.aten.linear.default:
                weight_shape = torch.Size((weight_shape[1], weight_shape[0]))
            if tuple(weight_shape) in mm_recompute_shapes:
                return CheckpointPolicy.PREFER_RECOMPUTE
            meta[mm_count_key] += 1

        # Save every compute/comm op in the set, except every second matmul.
        if func in save_ops:
            if func in _MM_OPS and meta[mm_count_key] % 2 == 0:
                return CheckpointPolicy.PREFER_RECOMPUTE
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return wrapped_policy


def wrap_selective(
    module: nn.Module, cfg: SelectiveACConfig, *, base_fqn: str | None = None
) -> nn.Module:
    """Wrap one block with the selective policy (upstream's ``_wrap_block``)."""
    save_ops = get_default_save_ops()
    mm_shapes = mm_recompute_shapes(
        module, base_fqn=base_fqn, fqns=cfg.force_recompute_mm_shapes_by_fqns
    )
    # One policy per context_fn invocation (per checkpoint frame), not per
    # block: the policy's matmul-alternation counters must restart for each
    # forward so its backward-time replay re-derives the same save decisions.
    wrapped = ptd_checkpoint_wrapper(
        module,
        context_fn=lambda: checkpoint_contexts(
            selective_policy(save_ops, mm_shapes), module
        ),
        preserve_rng_state=cfg.preserve_rng_state,
        determinism_check=cfg.determinism_check,
        early_stop=True,
        debug=cfg.debug,
    )
    # Match the wrapped block without recursively resetting mixed child modes.
    wrapped.training = module.training
    return wrapped


def full_policy(_ctx, op, *args, **_kwargs) -> CheckpointPolicy:
    """Recompute activations while preserving NPU reduction results.

    HCCL reduce-scatter can round differently on replay even with identical
    local inputs. Norms amplify that difference enough to change MoE routing,
    invalidating backward's token buckets. Save the reduction output without
    changing forward arithmetic, process-global determinism, or gradients.
    Other ops retain upstream's PREFER_RECOMPUTE policy and effect handling.
    """
    if (
        op == torch.ops._c10d_functional.reduce_scatter_tensor.default
        and args[0].device.type == "npu"
    ):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def checkpoint_contexts(policy: Callable, module: nn.Module | None = None):
    """Keep checkpoint's op policy while marking its backward-time replay."""
    forward_context, recompute_context = create_selective_checkpoint_contexts(policy)
    replay_spmd_context = capture_spmd_context()
    # HF stages masks outside decoder layers. Multiple outstanding forwards
    # must retain their own masks until their checkpoint backward executes.
    masks = (
        [
            (child, child._pending_padding_mask)
            for child in module.modules()
            if isinstance(child, MoE)
        ]
        if module is not None
        else []
    )

    @contextmanager
    def marked_recompute():
        token = in_checkpoint_recompute.set(True)
        current_masks = [(child, child._pending_padding_mask) for child, _ in masks]
        for child, mask in masks:
            child.set_padding_mask(mask)
        try:
            with replay_spmd_context():
                yield
        finally:
            for child, mask in current_masks:
                child.set_padding_mask(mask)
            in_checkpoint_recompute.reset(token)

    class RecomputeContext(type(recompute_context)):
        supports_higher_order_operators = True

        # Checkpoint may replay again for a retained graph. Generator context
        # managers are single-use, so create one for each entry.
        def __enter__(self):
            self.context = marked_recompute()
            self.context.__enter__()
            try:
                return super().__enter__()
            except BaseException as error:
                self.context.__exit__(type(error), error, error.__traceback__)
                raise

        def __exit__(self, exc_type, exc_value, traceback):
            try:
                return super().__exit__(exc_type, exc_value, traceback)
            finally:
                self.context.__exit__(exc_type, exc_value, traceback)

    return forward_context, RecomputeContext(
        policy,
        recompute_context.storage,
        recompute_context.allow_cache_entry_mutation,
    )


def wrap_full(module: nn.Module, *, preserve_rng_state: bool = True) -> nn.Module:
    """Wrap one block with the full policy (upstream's ``FullAC._wrap_block``).

    Upstream does not hand the wrapper a bare block: it passes ``full_policy``
    through a selective checkpoint context, so full AC runs on the same
    machinery as ``"selective"`` and torch preserves the outputs of ops with
    registered effects instead of recomputing them blindly.
    ``determinism_check`` and ``debug`` keep torch's defaults -- ``"default"``
    and ``False``, which are upstream's own defaults for this policy -- and
    llmtuner does not expose them for this mode (registered in the symbol
    guide). ``early_stop`` matches upstream #4836.
    """
    wrapped = ptd_checkpoint_wrapper(
        module,
        context_fn=lambda: checkpoint_contexts(full_policy, module),
        preserve_rng_state=preserve_rng_state,
        early_stop=True,
    )
    wrapped.training = module.training
    return wrapped


def disable_dynamo_lru_cache() -> None:
    """Select dynamo graphs in insertion order (upstream's SAC+PP workaround).

    With activation checkpointing and pipeline parallelism together, a second
    microbatch's forward recompiles with dynamic shapes enabled, so two valid
    compiled graphs exist for the same region. Dynamo's default latest-wins
    (LRU) selection can then hand back the one whose runtime wrapper expects
    an extra symint output, which SAC's cached inductor-HOP output does not
    carry, and the assertion fails; insertion order avoids it. See
    https://github.com/pytorch/pytorch/issues/166926.

    Upstream calls this at the top of every policy's ``apply``, not only on
    the PP path, and so does this module -- but only when a mode is actually
    being applied, and only where the knob exists. It is a process-global,
    private torch setting: this is the one place that touches it.
    """
    if not has("dynamo_lru_cache"):
        logger.info(
            "torch %s has no torch._C._dynamo.eval_frame._set_lru_cache; "
            "activation checkpointing runs without upstream's SAC + "
            "pipeline-parallel cache workaround.",
            torch.__version__,
        )
        return
    # pyrefly: ignore [missing-attribute]
    torch._C._dynamo.eval_frame._set_lru_cache(False)


def apply_memory_budget(cfg: MemoryBudgetACConfig) -> None:
    """Set the process-global activation-memory budget.

    This matches upstream's ``MemoryBudgetAC.apply``.

    The partitioner reads ``activation_memory_budget`` at compile time, so on
    a torch without that knob setting it would be a silent no-op -- refuse
    instead. Upstream never restores the global; it is a per-run setting.
    Upstream's ``apply`` sets the dynamo cache selection before touching the
    budget; the same order here, but after the check, so a config this torch
    cannot honour mutates nothing.
    """
    if not has("functorch_activation_memory_budget"):
        raise EnvironmentUnsupportedError(
            "mode='memory_budget' needs "
            "torch._functorch.config.activation_memory_budget, which this "
            f"torch ({torch.__version__}) does not have; the budget would be "
            "a global nothing reads."
        )
    disable_dynamo_lru_cache()
    torch._functorch.config.activation_memory_budget = cfg.memory_budget
    logger.info("Selected %s memory budget option", cfg.memory_budget)


def require_torch_remat() -> ModuleType:
    """Require the optional region-checkpoint runtime with installation guidance."""
    if torch_remat is None:
        raise ImportError(
            "activation_checkpoint_mode='region' needs the optional "
            "`torch_remat` package, which is not installed. It also requires "
            'torch >= 2.10. Install it with: pip install "torch_remat @ '
            'git+https://github.com/meta-pytorch/remat.git" -- or use '
            "activation_checkpoint_mode='full'/'selective'."
        ) from REMAT_IMPORT_ERROR
    return torch_remat


def wrap_region(
    block: nn.Module,
    cfg: RegionACConfig,
    *,
    base_fqn: str,
) -> nn.Module:
    """Wrap one block with the region policy (upstream's ``RegionAC``).

    Upstream mutates the block in place: its ``apply`` sets ``module.forward``
    to a ``torch_remat.checkpoint``-wrapped forward, after
    ``Module.configure_remat_regions`` has pushed the save patterns into the
    module tree that the model's own ``remat.region`` call sites read. HF models
    carry no such call sites, so this stands in for both halves at once: it
    annotates the block's ``nn.Linear``s as the regions -- see
    ``remat_regions`` for that vocabulary and why it is upstream's policy in
    HF's spelling -- and then checkpoints the block's forward.

    Two naming rules, both deliberate:

    * ``cfg.save_regions`` patterns are matched against region names *relative
      to the block* (``self_attn.q_proj``). That is upstream's rule, and it is
      what lets one policy cover every block.
    * the label handed to ``torch_remat`` is block-qualified
      (``layers.0.self_attn.q_proj``), so a trace or a memory report can tell
      two blocks' regions apart. torch_remat treats the name as an opaque label
      -- the matching happens here -- and block-qualified names are unique
      within a forward, which is the only property its uniqueness rule asks for.

    The package is reached through ``require_torch_remat``, so a test can stand
    the real thing in for a ``sys.modules`` entry -- the same way the torchao
    adapter is tested.
    """
    remat = require_torch_remat()
    regions = region_names(block)
    policy = region_policy(regions, cfg.save_regions, cfg.recompute_regions)
    for name, module in block.named_modules():
        if name in policy:
            module.forward = remat.region(
                module.forward, f"{base_fqn}.{name}", recompute=policy[name]
            )
        if isinstance(module, AuxLoss):
            module._metric_region_fn = remat.region(
                module._accumulate_and_inject,
                f"{base_fqn}.{name}.aux_loss",
                recompute=False,
            )
            module._metric_region_needs_tensor = getattr(
                remat, "recompute_needs_tensor", None
            )
    retained = sum(1 for recompute in policy.values() if not recompute)
    logger.info(
        "RegionAC on %s: %d regions, %d retained%s",
        base_fqn,
        len(regions),
        retained,
        f" ({', '.join(sorted(regions))})" if regions else "",
    )
    unmatched = unmatched_region_patterns(regions, cfg.save_regions)
    if unmatched:
        logger.warning(
            "RegionAC save_regions matched nothing in %s: %s (available: %s)",
            base_fqn,
            ", ".join(unmatched),
            ", ".join(sorted(regions)) or "none",
        )
    unmatched_recompute = unmatched_region_patterns(regions, cfg.recompute_regions)
    if unmatched_recompute:
        logger.warning(
            "RegionAC recompute_regions matched nothing in %s: %s (available: %s)",
            base_fqn,
            ", ".join(unmatched_recompute),
            ", ".join(sorted(regions)) or "none",
        )
    block.forward = remat.checkpoint(
        region_name=base_fqn,
        determinism_check=cfg.determinism_check,
        preserve_rng_state=cfg.preserve_rng_state,
    )(block.forward)
    return block


def validate_ac_options(
    mode: str,
    *,
    selective: SelectiveACConfig | None = None,
    memory_budget: MemoryBudgetACConfig | None = None,
    region: RegionACConfig | None = None,
    compile_enabled: bool = False,
) -> None:
    """Validate policy options without transforming models or changing global flags."""
    if mode not in VALID_AC_MODES:
        raise ValueError(
            f"Unknown activation checkpointing mode {mode!r}; expected one of "
            f"{VALID_AC_MODES}."
        )
    if mode == "selective" and selective is None:
        raise ValueError(
            "mode='selective' needs the SelectiveACConfig that carries its "
            "save set and rng/determinism settings; pass "
            "selective=cfg.training.selective_ac."
        )
    if mode == "region" and region is None:
        raise ValueError(
            "mode='region' needs the RegionACConfig that carries its save "
            "patterns; pass region=cfg.training.region_ac."
        )
    if mode == "memory_budget":
        if memory_budget is None:
            raise ValueError(
                "mode='memory_budget' needs the MemoryBudgetACConfig that "
                "carries the budget; pass "
                "memory_budget=cfg.training.memory_budget_ac."
            )
        if not compile_enabled:
            raise ValueError(
                "mode='memory_budget' requires compile: the budget is "
                "consumed by the compile partitioner, so without "
                "torch.compile it would silently do nothing."
            )


def apply_ac(
    model: nn.Module,
    mode: str = "none",
    *,
    selective: SelectiveACConfig | None = None,
    memory_budget: MemoryBudgetACConfig | None = None,
    region: RegionACConfig | None = None,
    compile_enabled: bool = False,
    preserve_rng_state: bool = True,
) -> nn.Module:
    """Wrap every decoder layer of ``model`` in a checkpoint wrapper.

    No-op when ``mode == "none"`` (the model is handed back untouched), so the
    caller needs no mode check of its own.

    ``mode == "full"`` checkpoints the whole layer: its forward activations are
    dropped and recomputed during backward. ``preserve_rng_state`` is this
    mode's knob; the default restores the RNG state for the recompute, which is
    what keeps a full-AC run bitwise-equal to an uncheckpointed one.

    ``mode == "selective"`` needs ``selective``, the ``SelectiveACConfig``
    carrying that policy's settings (it has its own ``preserve_rng_state``, so
    the two modes never share one).

    ``mode == "memory_budget"`` wraps nothing: it needs ``memory_budget`` (the
    ``MemoryBudgetACConfig``) and ``compile_enabled=True``, and sets the
    process-global budget the compile partitioner later reads.

    ``mode == "region"`` needs ``region`` (the ``RegionACConfig``) and the
    optional ``torch_remat`` package, required at apply time. It has no flat
    ``preserve_rng_state`` argument: torch_remat refuses ``True``, so the mode
    carries its own field, which can only stay ``False`` (see
    ``RegionACConfig``). Any other mode is a loud error.

    Apply after TP/EP/CP and before compile/FSDP (torchtitan's order in
    ``parallelize_llama``): the wrapper must enclose the TP-sharded layer, and
    FSDP has to wrap the checkpointed block so the recompute runs with
    all-gathered parameters instead of re-triggering the gather. For
    ``"memory_budget"`` the same call site is what guarantees the global is
    set before the compile it governs.
    """
    validate_ac_options(
        mode,
        selective=selective,
        memory_budget=memory_budget,
        region=region,
        compile_enabled=compile_enabled,
    )
    if mode == "none":
        return model
    if mode == "memory_budget":
        apply_memory_budget(memory_budget)
        return model

    structure = getattr(model, "decoder_structure", None)
    if structure is not None:
        if not isinstance(structure, DecoderStructure):
            raise TypeError("Activation checkpointing requires a DecoderStructure")
        layers = structure.bind_blocks(model)
    else:
        layers = getattr(model, "layers", None)
    if not isinstance(layers, nn.Module):
        raise TypeError(
            f"apply_ac expects an explicit model (with .layers as nn.Module); got "
            f"{type(model).__name__}."
        )

    if mode == "region":
        # Resolve the optional dependency before changing process-global state.
        require_torch_remat()
        # Region annotations modify the original blocks in place.
        disable_dynamo_lru_cache()

    # Build full/selective wrappers before replacing any layer. A bad FQN in
    # a later block must not leave earlier blocks checkpointed.
    blocks = tuple(layers.named_children())
    wrapped_blocks = []
    for layer_id, transformer_block in blocks:
        if mode == "selective":
            wrapped = wrap_selective(
                transformer_block, selective, base_fqn=f"layers.{layer_id}"
            )
        elif mode == "region":
            wrapped = wrap_region(
                transformer_block, region, base_fqn=f"layers.{layer_id}"
            )
        else:
            wrapped = wrap_full(
                transformer_block, preserve_rng_state=preserve_rng_state
            )
        wrapped_blocks.append((layer_id, wrapped))

    if mode != "region":
        disable_dynamo_lru_cache()
    for layer_id, wrapped in wrapped_blocks:
        layers.register_module(layer_id, wrapped)

    logger.info("Applied %s activation checkpointing to %d layers", mode, len(blocks))
    return model
