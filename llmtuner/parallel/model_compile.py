"""``torch.compile`` application: whole-model or per-block, plus its toggles.

This is the llmtuner counterpart of torchtitan's ``distributed/compile.py``.
That file does four independent things beyond the compile call itself, and
each lands here behind its own switch on ``CompileConfig`` (or its upstream
condition), so the default -- ``CompileConfig()`` -- is exactly the old
behavior: one whole-model ``torch.compile(model, backend="inductor")``.

* **Per-block compile** (``per_block=True``): compile each decoder layer
  separately with ``fullgraph=True``. The repeated block structure is then
  traced once and the graph reused, which is what makes compile time scale
  on deep models. Whole-model compile stays the default because it is the
  path every existing llmtuner run has taken.
* **Async TP** (``enable_async_tensor_parallel=True``): Inductor's
  micro-pipeline pass overlaps the TP collectives with the GEMMs inside
  compiled regions. It needs a TP mesh (``tensor_parallel_size > 1``),
  compile, symmetric memory registered for the TP group, and a torch that
  carries ``torch._inductor.config._micro_pipeline_tp`` -- any missing
  piece is a loud error at assembly time, never a silent skip.
* **capture_scalar_outputs**: token-choice MoE dispatch has data-dependent
  shapes (per-expert token counts), which dynamo cannot trace without
  ``torch._dynamo.config.capture_scalar_outputs``. Set -- with the same
  condition upstream documents -- when the compiled model part actually
  carries llmtuner MoE blocks; dense models keep the flag untouched.

Ordering: this runs after activation checkpointing and before FSDP, the
same slot the whole-model compile it replaces occupied.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from functools import wraps

import torch
import torch._dynamo.config as dynamo_config
import torch._functorch.config as functorch_config
import torch._inductor.config as inductor_config
import torch.nn as nn
from torch._dynamo.backends.registry import lookup_backend
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.config.compile_config import CompileConfig
from llmtuner.errors import EnvironmentUnsupportedError

from ..accelerator.capabilities import has
from ..accelerator.device import device_type
from ..accelerator.linear_collectives import enable_symm_mem
from ..models.common.moe.layer import collect_moe_layers
from ..models.contracts import DecoderStructure
from ..utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["apply_compile"]


@contextlib.contextmanager
def rollback_compile_settings_on_error() -> Iterator[None]:
    """Restore compile policy if later runtime assembly fails.

    Successful runtimes retain these settings for lazy tracing. Symmetric-memory
    group registration and concurrent policy changes are outside this scope.
    """
    capture = getattr(dynamo_config, "capture_scalar_outputs", None)
    async_tp = getattr(inductor_config, "_micro_pipeline_tp", None)
    memory_budget = getattr(functorch_config, "activation_memory_budget", None)
    try:
        yield
    except BaseException:
        if capture is not None:
            dynamo_config.capture_scalar_outputs = capture
        if async_tp is not None:
            inductor_config._micro_pipeline_tp = async_tp
        if memory_budget is not None:
            functorch_config.activation_memory_budget = memory_budget
        raise


def validate_compile_backend(backend: str) -> None:
    """Validate the NPU Codegen selected before importing the training stack."""
    if backend != "inductor" or device_type != "npu":
        return
    selected = os.environ.get("TORCHINDUCTOR_NPU_BACKEND")
    if selected != "ascendc":
        raise EnvironmentUnsupportedError(
            "NPU compile backend='inductor' requires "
            "TORCHINDUCTOR_NPU_BACKEND=ascendc. Set it in the launch shell "
            "before importing torch, or use backend='aot_eager'."
        )
    if not has("npu_ascendc_codegen"):
        raise EnvironmentUnsupportedError(
            "NPU compile backend='inductor' requires the built-in "
            "torch_npu._inductor.ascendc Codegen and matching CANN AutoFuse "
            "runtime. Install the pinned runtime, or use backend='aot_eager'."
        )
    logger.info("NPU Inductor Codegen: ascendc")


def validate_compile_environment(config: CompileConfig | None) -> None:
    """Resolve the selected backend before parallel transforms modify the model."""
    backend = "inductor" if config is None else config.backend
    validate_compile_backend(backend)
    lookup_backend(backend)


def _check_finite_output(output, name: str) -> None:
    if isinstance(output, torch.Tensor):
        if not torch.isfinite(output).all().item():
            raise FloatingPointError(f"Non-finite compiled output at {name}")
    elif isinstance(output, tuple | list):
        for value in output:
            _check_finite_output(value, name)
    elif isinstance(output, dict):
        for value in output.values():
            _check_finite_output(value, name)


def wrap_sdc_forward(module: nn.Module, name: str) -> nn.Module:
    target = (
        "_compiled_call_impl" if module._compiled_call_impl is not None else "forward"
    )
    original = getattr(module, target)

    @wraps(original)
    def checked(*args, **kwargs):
        output = original(*args, **kwargs)
        _check_finite_output(output, name)
        return output

    setattr(module, target, checked)
    return module


def apply_compile(
    model: nn.Module,
    *,
    compile_config: CompileConfig | None = None,
    tp_mesh: DeviceMesh | None = None,
) -> nn.Module:
    """Compile ``model`` according to ``compile_config``; returns the model.

    ``compile_config=None`` is the default run: whole-model inductor compile,
    identical to the plain ``torch.compile(model)`` this replaces. With
    ``per_block=True`` each decoder layer is compiled in place instead and
    the model is returned unwrapped -- either way the caller rebinds the
    result, so a wrapping transform cannot leave a stale module behind.

    ``tp_mesh`` is the tensor-parallel mesh (``parallel_dims``' ``tp``
    axis), consulted only when async TP is configured.
    """
    if compile_config is None:
        compile_config = CompileConfig()
    layers = None
    if compile_config.per_block:
        structure = getattr(model, "decoder_structure", None)
        if structure is not None:
            if not isinstance(structure, DecoderStructure):
                raise TypeError("Per-block compile requires a DecoderStructure")
            layers = structure.bind_blocks(model)
        else:
            layers = getattr(model, "layers", None)
    if compile_config.per_block and not isinstance(layers, nn.Module):
        raise TypeError("per_block compile requires a model with a .layers module")

    # Resolve registered/custom backends before changing module or global flags.
    validate_compile_environment(compile_config)
    has_moe = bool(collect_moe_layers(model))
    if has_moe:
        # Token-choice dispatch sizes its expert splits from the routing,
        # so the compiled graph has data-dependent shapes. Dense models
        # never touch the flag, keeping their trace bitwise unchanged.
        if not has("dynamo_capture_scalar_outputs"):
            raise EnvironmentUnsupportedError(
                "This torch has no torch._dynamo.config.capture_scalar_outputs, "
                "which compiling a token-choice MoE dispatch needs for its "
                "data-dependent shapes. Upgrade torch, or run this model "
                "without compile."
            )
    previous_async_tp = None
    if compile_config.enable_async_tensor_parallel and has(
        "inductor_micro_pipeline_tp"
    ):
        previous_async_tp = inductor_config._micro_pipeline_tp
    previous_capture = torch._dynamo.config.capture_scalar_outputs if has_moe else None
    compiled_blocks = []
    try:
        if compile_config.enable_async_tensor_parallel:
            maybe_enable_async_tp(compile_config, tp_mesh)
        backend = compile_config.backend
        if has_moe:
            torch._dynamo.config.capture_scalar_outputs = True
            logger.info("capture_scalar_outputs is enabled (token-choice MoE)")

        if compile_config.per_block:
            for layer_id, block in layers.named_children():
                # Module.compile, not torch.compile(block): it sets the compiled
                # call impl in place, so the layer keeps its identity for FSDP
                # and the state dict.
                compiled_blocks.append((block, block._compiled_call_impl))
                block.compile(backend=backend, fullgraph=True)
                if compile_config.enable_sdc:
                    wrap_sdc_forward(block, f"decoder layer {layer_id}")
                logger.info(f"Compiling decoder layer {layer_id} (fullgraph)")
            return model

        logger.info(f"Compiling the whole model (backend={backend!r})")
        compiled = torch.compile(model, backend=backend)
        if compile_config.enable_sdc:
            return wrap_sdc_forward(compiled, "model")
        return compiled
    except BaseException:
        if previous_async_tp is not None:
            inductor_config._micro_pipeline_tp = previous_async_tp
        for block, previous_impl in reversed(compiled_blocks):
            block._compiled_call_impl = previous_impl
        # Compilation setup may fail before a usable model is returned. Keep
        # that failed attempt from changing subsequent models' tracing policy.
        # Successful models still need this flag during lazy tracing.
        if has_moe:
            torch._dynamo.config.capture_scalar_outputs = previous_capture
        raise


def maybe_enable_async_tp(
    compile_config: CompileConfig, tp_mesh: DeviceMesh | None
) -> None:
    """Configure Inductor's async TP pass for the TP mesh.

    Every precondition is checked here and fails loudly: async TP asked for
    without a TP mesh (``tensor_parallel_size = 1`` or a single-process run),
    or a torch too old to carry the pass, are config errors, not situations
    to degrade through.
    """
    if not compile_config.enable_async_tensor_parallel:
        return
    if tp_mesh is None:
        raise ValueError(
            "compile_config.enable_async_tensor_parallel requires "
            "tensor_parallel_size > 1 with a real process group: async TP "
            "pipelines the TP collectives, and there is no TP group here."
        )

    if not has("inductor_micro_pipeline_tp"):
        raise EnvironmentUnsupportedError(
            "compile_config.enable_async_tensor_parallel needs "
            "torch._inductor.config._micro_pipeline_tp, which this torch "
            f"({torch.__version__}) does not carry. Upgrade torch, or run "
            "without async TP."
        )
    if not has("symm_mem"):
        raise EnvironmentUnsupportedError(
            "compile_config.enable_async_tensor_parallel needs "
            "torch.distributed._symmetric_memory.enable_symm_mem_for_group, "
            f"which this torch ({torch.__version__}) does not carry. Upgrade "
            "torch, or run without async TP."
        )
    if tp_mesh.ndim != 1 or tp_mesh.size() <= 1:
        raise ValueError("async TP requires a one-dimensional TP mesh with size > 1")
    enable_symm_mem(tp_mesh.get_group())

    inductor_config._micro_pipeline_tp = True
    logger.info("Async TP is enabled")
