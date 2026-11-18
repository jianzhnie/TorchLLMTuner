"""The capability registry: one place that answers "does this build have X?".

Torch-version and environment probes (``hasattr`` on private config modules,
guarded imports) used to be scattered across a dozen call sites, each
re-answering the same question its own way. They live here now, one named
entry per capability, each documented with what it is, which torch version or
package introduces it, and who consumes it. Probes are cached: the answer
cannot change within a process.

Use ``has(name)`` at a guard point, keeping the site's own error message (the
message text is a tested contract); ``require(name, feature=...)`` is the
convenience form for new code, raising ``EnvironmentUnsupportedError`` with
the entry's unlock hint. An unknown name raises immediately -- a typo must
not silently read as "capability absent".

Not everything import-shaped belongs here; see docs/torchllmtuner_design.md
§3.2. Optional *packages* (renderers, torchao, torchvision) keep raising
plain ``ImportError`` at their own sites. Device discovery
(``accelerator/device.py``) is availability probing with silent-absent
semantics, a different question. Hard imports with no fallback (DTensor,
flex_attention) have nothing to probe.
"""

from __future__ import annotations

import inspect
from functools import cache

import torch

from llmtuner.accelerator.capabilities import (
    Capability,
    _fsdp_unshard_is_a_context_manager,
    dynamo_lru_cache_knob,
    grouped_mm_runs,
    importable,
    npu_ascendc_codegen,
)
from llmtuner.errors import EnvironmentUnsupportedError

try:
    import torch.distributed.pipelining.schedules as pipeline_schedules
except ImportError:
    pipeline_schedules = None

__all__ = ["CAPABILITIES", "has", "is_compiling", "require"]


def _pipelining_has_eval() -> bool:
    """Whether torch's pipeline schedules carry the microbatch eval driver.

    Attribute presence is not enough: torch 2.9 shipped ``eval(*args, target,
    losses, **kwargs)`` -- the microbatch keyword form (``arg_mbs`` /
    ``kwarg_mbs`` / ``target_mbs``) would be swallowed by ``**kwargs`` and
    misrouted into the model kwargs. Probe the signature instead. The training
    seam's ``_step_microbatches(..., return_outputs=...)`` arrived in the same
    window, so both are checked.
    """
    schedule = getattr(pipeline_schedules, "_PipelineSchedule", None)
    evaluate = getattr(schedule, "eval", None)
    if evaluate is None:
        return False
    eval_params = inspect.signature(evaluate).parameters
    if "arg_mbs" not in eval_params:
        return False
    step = getattr(schedule, "_step_microbatches", None)
    if step is None:
        return False
    return "return_outputs" in inspect.signature(step).parameters


CAPABILITIES: dict[str, Capability] = {
    # -- compile-time knobs (consumers: parallel/model_compile.py) -------------------
    "dynamo_capture_scalar_outputs": Capability(
        importable("torch._dynamo.config", "capture_scalar_outputs"),
        what="torch._dynamo.config.capture_scalar_outputs",
        since="torch 2.7 (dynamo config flag)",
        hint="Upgrade torch, or run the token-choice MoE without compile.",
        consumers="parallel/model_compile.py (token-choice MoE dispatch compile)",
    ),
    "inductor_micro_pipeline_tp": Capability(
        importable("torch._inductor.config", "_micro_pipeline_tp"),
        what="torch._inductor.config._micro_pipeline_tp",
        since="torch 2.8 (inductor micro-pipeline TP pass)",
        hint="Upgrade torch, or run without async TP.",
        consumers=(
            "parallel/model_compile.py (compile_config.enable_async_tensor_parallel)"
        ),
    ),
    "npu_ascendc_codegen": Capability(
        npu_ascendc_codegen,
        what="torch_npu._inductor.ascendc and the matching CANN AutoFuse runtime",
        since="torch_npu with the built-in AscendC Inductor backend",
        hint="Install the matching torch_npu/CANN runtime, or use aot_eager/eager.",
        consumers="parallel/model_compile.py (NPU inductor backend)",
    ),
    # -- symmetric memory (consumers: parallel/model_compile.py, tensor_parallel) ----
    "symm_mem": Capability(
        importable("torch.distributed._symmetric_memory", "enable_symm_mem_for_group"),
        what="torch.distributed._symmetric_memory.enable_symm_mem_for_group",
        since="torch 2.8 (symmetric-memory collectives, CUDA-only)",
        hint="Upgrade torch, or run without async TP / symm-mem collectives.",
        consumers="parallel/model_compile.py (async TP), "
        "tensor_parallel/tensor_parallel.py (fused symm-mem TP collectives), "
        "accelerator/linear_collectives.py",
    ),
    # -- activation checkpointing (consumer: memory/activation_checkpoint.py)
    "functorch_activation_memory_budget": Capability(
        importable("torch._functorch.config", "activation_memory_budget"),
        what="torch._functorch.config.activation_memory_budget",
        since="torch 2.6 (functorch partitioner budget knob)",
        hint="Upgrade torch, or use activation_checkpoint_mode='full'/'selective'.",
        consumers="memory/activation_checkpoint.py (mode='memory_budget')",
    ),
    # -- FSDP (consumers: fully_shard_parallel, trainer/runtime/assembly) ------
    "fsdp_unshard_context": Capability(
        _fsdp_unshard_is_a_context_manager,
        what="an FSDP unshard that stays open for the duration of a ``with`` "
        "block (``UnshardHandle.__enter__``); this probes API shape, not "
        "chunked-training correctness",
        since="no released torch through 2.12 (``UnshardHandle`` still has "
        "only ``wait()``); the probe, not a version number, decides",
        hint="Absent: chunked_loss_num_chunks > 1 is refused under FSDP; run "
        "with chunked_loss_num_chunks=1. A future context-manager API still "
        "requires validation of head ownership, gradients and restore.",
        consumers="trainer/runtime/assembly.py (chunked loss + FSDP guard)",
    ),
    "fsdp_per_param_mesh": Capability(
        importable(
            "torch.distributed.fsdp._fully_shard._fsdp_common",
            "ShardPlacementResult",
        ),
        what="torch.distributed.fsdp._fully_shard._fsdp_common."
        "ShardPlacementResult -- a placement function that also names a mesh, so "
        "ONE fully_shard unit can place parameters on two different meshes",
        since="torch 2.12 (2.9's shard_placement_fn returns Optional[Shard] and "
        "has no mesh slot)",
        hint="Absent: the EP > 1 decoder path shards the experts in their own "
        "nested FSDP unit on the sparse mesh instead "
        "(fully_shard_parallel.shard_experts_on_their_own_mesh); the "
        "pattern-selected path has no such fallback and refuses.",
        consumers="parallel/fully_shard_parallel/fully_shard_parallel.py "
        "(EP > 1 decoder path), plan.py (pattern-selected units)",
    ),
    "dynamo_lru_cache": Capability(
        dynamo_lru_cache_knob,
        what="torch._C._dynamo.eval_frame._set_lru_cache",
        since="private dynamo knob; absent from torch 2.2.2, present in the "
        "builds upstream targets",
        hint="Upgrade torch; without the knob, activation checkpointing runs "
        "without upstream's SAC + pipeline-parallel cache workaround.",
        consumers="memory/activation_checkpoint.py (disable_dynamo_lru_cache)",
    ),
    # -- pipeline eval driver (consumer: trainer/runtime/validation.py) --------
    "pipelining_microbatch_drivers": Capability(
        _pipelining_has_eval,
        what="torch.distributed.pipelining microbatch drivers "
        "(_PipelineSchedule.eval(arg_mbs=...) and "
        "_step_microbatches(return_outputs=...))",
        since="no released torch through 2.12 (2.9's and 2.12's eval both "
        "swallow the microbatch kwargs); the probe reads the signature",
        hint="Optional: when absent, PP falls back to the drivers in "
        "parallel/pipeline_parallel/microbatch_drivers.py, which reproduce the "
        "schedule setup torch's own step() performs.",
        consumers="trainer/runtime/{pipeline,validation}.py (chooses the native "
        "microbatch driver or the fallback)",
    ),
    # -- model kernels (consumer: models/common/moe/experts.py) ----------------
    "torch_grouped_mm": Capability(
        grouped_mm_runs,
        what="torch._grouped_mm",
        since="torch 2.7 (grouped GEMM, bf16; CPU-reachable but shape-constrained)",
        hint="Upgrade torch, or leave GroupedExperts on the looped-GEMM fallback.",
        consumers="models/common/moe/experts.py (GroupedExperts forward)",
    ),
}


@cache
def probe(name: str) -> bool:
    entry = CAPABILITIES.get(name)
    if entry is None:
        raise KeyError(
            f"unknown capability {name!r}; registered: {sorted(CAPABILITIES)}"
        )
    return bool(entry.probe())


def is_compiling() -> bool:
    """``torch.compiler.is_compiling()``, tolerated on builds that predate it.

    The name is public from torch 2.3. Below that there is no compile pass at
    all, so ``False`` is the honest answer rather than a crash -- which is what
    a caller asking "am I being traced?" wants to hear. Answered here rather
    than at each call site so the version difference has one home.
    """
    fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return bool(fn()) if fn is not None else False


def has(capability: str) -> bool:
    """Whether ``capability`` is present in this build (cached)."""
    return probe(capability)


def require(capability: str, *, feature: str) -> None:
    """Raise ``EnvironmentUnsupportedError`` unless ``has(capability)``.

    Guard points with an existing (tested) message should keep their own
    raise and use ``has``; this is the convenience form for new code.
    """
    if has(capability):
        return
    entry = CAPABILITIES[capability]  # has() already validated the name
    raise EnvironmentUnsupportedError(
        f"{feature} needs {entry.what}, which this torch "
        f"({torch.__version__}) does not carry. {entry.hint} "
        f"(capability {capability!r}, introduced {entry.since}.)"
    )
