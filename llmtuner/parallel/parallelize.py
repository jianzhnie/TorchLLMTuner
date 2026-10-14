"""One entry point that applies parallelism to a HuggingFace model.

Order matters, and it is the whole content of this file: TP / EP / CP are
declared first, activation checkpointing wraps each decoder layer next, then
``torch.compile``, and FSDP wraps last so its hooks sit outermost. The order
is not a comment -- it is the ``STAGES`` table in ``stages.py``, which both assembly
paths are driven by the same stage order.
Each ``apply_*`` is a no-op when its degree is 1 (or its mode off), so the
same call runs from a single device up to a full hybrid mesh.

PP is the exception to "one model in, one model out": with ``pp > 1`` the model
is cut into per-stage chunks first (``pipeline_parallel.apply_pp``), each chunk
then goes through TP / EP / CP / AC / compile / FSDP in the same order as the
unsplit path, and the caller gets back a ``PipelineParallelSetup`` (stages,
chunks, schedule) instead of a model.

On provenance: this is the *orchestration* half of torchtitan's
``parallelize_hf_transformers``. The other half was three things; the first
two llmtuner deliberately does not do, and dropping them is a decision, not an
oversight:

* **Untying ``tok_embeddings`` from ``lm_head``.** torchtitan un-ties them
  because its FSDP cannot shard a parameter shared by two FSDP groups. llmtuner
  instead detects the tie (``HFTransformerModel.enable_weight_tying``) and
  shards the embedding, norm and head as one unit -- so untying here would
  silently train an un-tied model that no longer matches its HF checkpoint.
  Models with ``tie_word_embeddings=False`` (llama, qwen) are unaffected either
  way.
* **Converting modules to a ``Module`` protocol.** llmtuner has none, and no
  on-the-fly sharding-config declarations either -- the TP plan lives as plain
  data in the model registry instead (see docs/torchllmtuner_design.md, SEAM 1).
* **Swapping in a native MoE.** No longer true: ``apply_ep`` swaps HF MoE
  blocks for the ``models/common`` MoE stack when ``ep > 1`` (see
  ``parallel/expert_parallel/model_replacement.py``). The conversion moves weights
  without re-initializing them, so the model retains HF's initialization.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.config.activation_checkpoint_config import (
    MemoryBudgetACConfig,
    RegionACConfig,
    SelectiveACConfig,
)
from llmtuner.config.compile_config import CompileConfig
from llmtuner.config.parallel_config import ParallelConfig

from ..memory.activation_checkpoint import apply_ac
from .context_parallel.context_parallel import apply_cp
from .expert_parallel.expert_parallel import apply_ep
from .fully_shard_parallel.fully_shard_parallel import apply_fsdp
from .model_compile import apply_compile
from .pipeline_parallel.pipeline_parallel import (
    PipelineParallelSetup,
    apply_pp,
    build_pipeline_schedule,
)
from .stages import STAGE_ORDER
from .tensor_parallel.tensor_parallel import apply_tp
from .topology import ParallelTopology

__all__ = ["parallelize_hf_transformers"]


def parallelize_hf_transformers(
    model: nn.Module,
    *,
    cfg: ParallelConfig,
    mesh: DeviceMesh | None,
    parallel_dims: ParallelTopology | None,
    device: torch.device | None = None,
    compile: bool = False,
    compile_config: CompileConfig | None = None,
    activation_checkpoint: str = "none",
    selective_ac: SelectiveACConfig | None = None,
    memory_budget_ac: MemoryBudgetACConfig | None = None,
    region_ac: RegionACConfig | None = None,
    global_batch_size: int | None = None,
) -> nn.Module | PipelineParallelSetup:
    """Apply every parallelism dimension the config asks for, in order.

    ``compile``, ``activation_checkpoint``, ``selective_ac`` and
    ``global_batch_size`` are training-side values, passed
    explicitly rather than read off a run-wide config: this layer's contract is
    ``ParallelConfig`` plus the handful of scalars the guards actually need.
    ``compile_config`` tunes the compile step (per-block, backend, async TP);
    ``None`` is the plain whole-model compile.
    ``global_batch_size`` is required only on the ``pp > 1`` path (microbatch
    validation). Per-token positions and masks of a packed corpus ride the
    schedule's per-microbatch kwargs, so no corpus restriction applies.
    ``selective_ac`` / ``memory_budget_ac`` / ``region_ac`` are read only when
    ``activation_checkpoint`` names their mode (``'selective'`` /
    ``'memory_budget'`` -- which also requires ``compile=True`` -- or
    ``'region'``, which needs the optional ``torch_remat`` at apply time).
    Those AC arguments are bound in the shared stage runner below,
    so the split path cannot checkpoint with a different policy than the
    unsplit one.

    Returns the (possibly wrapped) model -- or, with ``pp > 1``, a
    ``PipelineParallelSetup``: pipeline parallelism cuts the model into
    per-stage chunks, so there is no single module left to return. The two
    return shapes are how the caller learns which case it is in.
    """

    def apply_stages(
        part: nn.Module,
        *,
        dense_mesh: DeviceMesh | None,
    ) -> nn.Module:
        """Bind stage options once, then transform the model in declared order."""
        runners = {
            "tp": partial(apply_tp, mesh=dense_mesh, cfg=cfg),
            "ep": partial(apply_ep, cfg=cfg, ep_group=ep_group),
            "cp": partial(
                apply_cp, mesh=dense_mesh, cfg=cfg, parallel_dims=parallel_dims
            ),
            "ac": partial(
                apply_ac,
                mode=activation_checkpoint,
                selective=selective_ac,
                memory_budget=memory_budget_ac,
                region=region_ac,
                compile_enabled=compile,
            ),
            "compile": partial(
                apply_compile, compile_config=compile_config, tp_mesh=tp_mesh
            ),
            "fsdp": partial(apply_fsdp, cfg=cfg, parallel_dims=parallel_dims),
        }
        for name in STAGE_ORDER:
            if name != "compile" or compile:
                part = runners[name](part)
        return part

    # EP groups come from the sparse mesh in both assembly paths. Under PP
    # this slice contains only ranks in the current pipeline stage.
    ep_group = None
    if cfg.ep > 1:
        if parallel_dims is None:
            raise ValueError(
                f"ep={cfg.ep} needs a process group, but this run is "
                "single-process (parallel_dims is None). EP requires "
                "world_size > 1."
            )
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        if ep_mesh is None:
            raise ValueError(
                f"ep={cfg.ep} but parallel_dims has no multi-rank 'ep' axis."
            )
        ep_group = ep_mesh.get_group()
    tp_mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")

    if parallel_dims is not None and parallel_dims.pp_enabled:
        if global_batch_size is None:
            raise ValueError(
                "pp > 1 needs global_batch_size for microbatch validation; "
                "the trainer passes cfg.training.global_batch_size."
            )
        # apply_pp is PP-only: stage count, split, and the per-stage views.
        # The per-chunk application of the other dimensions lives here, so
        # this file is the single owner of the assembly order on both paths:
        # each stage's chunk goes through tp/ep/cp/ac/(compile)/fsdp in the same
        # relative order as the unsplit path below. The dense (dp, cp, tp)
        # view excludes the pp axis, so it covers exactly this stage's
        # coordinates -- the mesh the per-part apply_* functions would have
        # been handed had the model never been split.
        stages, model_parts, has_first_stage, has_last_stage = apply_pp(
            model,
            parallel_dims=parallel_dims,
            cfg=cfg,
            device=device if device is not None else next(model.parameters()).device,
            global_batch_size=global_batch_size,
        )
        dense_mesh = parallel_dims.spmd_dense_mesh()
        for i, part in enumerate(model_parts):
            model_parts[i] = apply_stages(part, dense_mesh=dense_mesh)
            # Rebind the stage's submodule in case a transform replaced the chunk.
            stages[i].submod = model_parts[i]
        return PipelineParallelSetup(
            # The schedule's loss runs on the last stage's logits, so it needs
            # the same vocab-parallel arguments the trainer's loss does. They
            # come from the un-split ``model`` and the TP axis, both still in
            # scope here; ``vocab_size`` is read defensively because this
            # function's contract is a plain ``nn.Module``.
            schedule=build_pipeline_schedule(
                stages,
                cfg=cfg,
                tp_group=None if tp_mesh is None else tp_mesh.get_group(),
                global_vocab_size=(
                    None if tp_mesh is None else getattr(model, "vocab_size", None)
                ),
            ),
            stages=stages,
            model_parts=model_parts,
            has_first_stage=has_first_stage,
            has_last_stage=has_last_stage,
        )

    return apply_stages(model, dense_mesh=mesh)
