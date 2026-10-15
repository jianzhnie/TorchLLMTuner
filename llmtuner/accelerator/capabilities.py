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

import importlib
from collections.abc import Callable
from functools import cache

import torch

from llmtuner.errors import EnvironmentUnsupportedError

from .device import get_current_device

__all__ = ["CAPABILITIES", "has", "is_compiling", "require"]


def hasattr_torch(module: str, attr: str) -> Callable[[], bool]:
    """Probe factory: ``hasattr(<torch submodule attr chain>, attr)``."""

    def probe() -> bool:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            return False
        return getattr(mod, attr, None) is not None

    return probe


def importable(module: str, attr: str | None = None) -> Callable[[], bool]:
    """Probe factory: the module imports (and optionally exposes ``attr``)."""

    def probe() -> bool:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            return False
        return attr is None or getattr(mod, attr, None) is not None

    return probe


def dynamo_lru_cache_knob() -> bool:
    """Whether dynamo exposes ``eval_frame._set_lru_cache``.

    The knob is private, so it is looked up on the import's own attribute
    chain (``import torch._dynamo``, then ``torch._C._dynamo.eval_frame``)
    rather than by importing the submodule by name -- which is how upstream
    reaches it, and the only spelling that can succeed. ``torch._C._dynamo``
    and ``eval_frame`` both exist in torch 2.2.2; the *function* on it does
    not, which is exactly the distinction this probe has to make.
    """
    try:
        importlib.import_module("torch._dynamo")
    except ImportError:
        return False
    eval_frame = getattr(getattr(torch._C, "_dynamo", None), "eval_frame", None)
    return getattr(eval_frame, "_set_lru_cache", None) is not None


def _pipelining_has_eval() -> bool:
    """Whether torch's pipeline schedules carry the microbatch eval driver.

    Attribute presence is not enough: torch 2.9 shipped ``eval(*args, target,
    losses, **kwargs)`` -- the microbatch keyword form (``arg_mbs`` /
    ``kwarg_mbs`` / ``target_mbs``) would be swallowed by ``**kwargs`` and
    misrouted into the model kwargs. Probe the signature instead. The training
    seam's ``_step_microbatches(..., return_outputs=...)`` arrived in the same
    window, so both are checked.
    """
    if not importable("torch.distributed.pipelining.schedules", "_PipelineSchedule")():
        return False
    import inspect

    import torch.distributed.pipelining.schedules as schedules

    evaluate = getattr(schedules._PipelineSchedule, "eval", None)
    if evaluate is None:
        return False
    eval_params = inspect.signature(evaluate).parameters
    if "arg_mbs" not in eval_params:
        return False
    step = getattr(schedules._PipelineSchedule, "_step_microbatches", None)
    if step is None:
        return False
    return "return_outputs" in inspect.signature(step).parameters


def grouped_mm_runs() -> bool:
    """Whether ``torch._grouped_mm`` can run here.

    Probed by doing it, rather than by checking the device or the torch
    version: the op is reachable on CPU as well as CUDA, and it imposes shape
    constraints of its own (strides must be 16-byte multiples, so the
    innermost dim has to be at least 8 bf16 elements). A version or device
    test would be wrong on both counts and would go stale silently.

    The probe is necessarily approximate -- a shape that satisfies the op
    need not be one a real layer uses. It uses the selected training device and
    transposed expert weights. It is deliberately shaped like the
    real call (``(T, K) @ (E, K, N)``, bf16, int32 offsets) so that it fails
    for the same reasons a real call would.
    """
    grouped_mm = getattr(torch, "_grouped_mm", None)
    if grouped_mm is None:
        return False
    try:
        device = get_current_device()
        grouped_mm(
            torch.zeros(8, 8, dtype=torch.bfloat16, device=device),
            torch.zeros(2, 8, 8, dtype=torch.bfloat16, device=device).transpose(-2, -1),
            offs=torch.tensor([4, 8], dtype=torch.int32, device=device),
        )
    except Exception:
        return False
    return True


class Capability:
    """One registry entry: probe, provenance, unlock hint, consumers."""

    def __init__(
        self,
        probe: Callable[[], bool],
        *,
        what: str,
        since: str,
        hint: str,
        consumers: str,
    ) -> None:
        self.probe = probe
        self.what = what
        self.since = since
        self.hint = hint
        self.consumers = consumers


CAPABILITIES: dict[str, Capability] = {
    # -- compile-time knobs (consumers: parallel/model_compile.py) -------------------
    "dynamo_capture_scalar_outputs": Capability(
        hasattr_torch("torch._dynamo.config", "capture_scalar_outputs"),
        what="torch._dynamo.config.capture_scalar_outputs",
        since="torch 2.7 (dynamo config flag)",
        hint="Upgrade torch, or run the token-choice MoE without compile.",
        consumers="parallel/model_compile.py (token-choice MoE dispatch compile)",
    ),
    "inductor_micro_pipeline_tp": Capability(
        hasattr_torch("torch._inductor.config", "_micro_pipeline_tp"),
        what="torch._inductor.config._micro_pipeline_tp",
        since="torch 2.8 (inductor micro-pipeline TP pass)",
        hint="Upgrade torch, or run without async TP.",
        consumers=(
            "parallel/model_compile.py (compile_config.enable_async_tensor_parallel)"
        ),
    ),
    "fx_regional_inductor": Capability(
        importable("torch.fx.passes.regional_inductor", "regional_inductor"),
        what="torch.fx.passes.regional_inductor (+ torch._dynamo.backends.common)",
        since="torch 2.10 (fx regional-inductor pass)",
        hint="Upgrade torch, use backend='inductor', or run without compile.",
        consumers="parallel/model_compile.py (aot_eager backend on flex models)",
    ),
    # -- symmetric memory (consumers: parallel/model_compile.py, tensor_parallel) ----
    "symm_mem": Capability(
        importable("torch.distributed._symmetric_memory", "enable_symm_mem_for_group"),
        what="torch.distributed._symmetric_memory.enable_symm_mem_for_group",
        since="torch 2.8 (symmetric-memory collectives, CUDA-only)",
        hint="Upgrade torch, or run without async TP / symm-mem collectives.",
        consumers="parallel/model_compile.py (async TP), "
        "tensor_parallel/tensor_parallel.py (fused symm-mem TP collectives), "
        "tensor_parallel/linear_ops.py",
    ),
    # -- activation checkpointing (consumer: memory/activation_checkpoint.py)
    "functorch_activation_memory_budget": Capability(
        hasattr_torch("torch._functorch.config", "activation_memory_budget"),
        what="torch._functorch.config.activation_memory_budget",
        since="torch 2.6 (functorch partitioner budget knob)",
        hint="Upgrade torch, or use activation_checkpoint_mode='full'/'selective'.",
        consumers="memory/activation_checkpoint.py (mode='memory_budget')",
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
        since="torch main/2.10+ (2.9's eval swallows the microbatch kwargs)",
        hint="Upgrade torch, or run with pipeline_parallel_size=1.",
        consumers="pipeline_parallel/pipeline_parallel.py (PP assembly), "
        "trainer/runtime/{pipeline,validation}.py (PP train/eval drivers)",
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
    fn = getattr(torch.compiler, "is_compiling", None)
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
