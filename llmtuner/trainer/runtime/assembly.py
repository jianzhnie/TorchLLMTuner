"""Build the trainer's model, parallel layout, and training services.

Resolve the process group and topology, check validation feasibility, seed the
run, and select the dense mesh. Then build the HF model, apply parallel stages,
and create optimizer/scheduler/EMA services. Finally build the dataloader,
checkpointer, counters, and metrics. Model-dependent combination guards run
while building the model, before parallel transformations.

Parallel stage order is defined in ``parallel/stages.py``. Pipeline parallelism
splits the model first, then applies those stages to each local model part.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from ...accelerator.device import (
    device_type,
    get_distributed_backend,
    get_env_dist_info,
)
from ...accelerator.distributed import (
    get_rank,
    get_world_size,
    init_dist_pytorch,
    is_distributed,
)
from ...components.checkpointer.checkpoint_keys import DATALOADER, TRAIN_STATE
from ...components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from ...components.metrics import MetricsProcessor
from ...components.optimizer.ema import EMA
from ...components.optimizer.lr_scheduler import build_lr_scheduler
from ...components.optimizer.optimizer import OptimizersContainer
from ...models.common.aux_loss import register_aux_loss_zero_hook
from ...models.common.moe.bias_update import (
    register_moe_balancing_hook,
)
from ...models.hf.factory import (
    build_model_config_for,
    unwrap_text_config,
)
from ...models.hf.flops import flops_per_token
from ...models.hf.materialization import materialize_meta_model
from ...models.hf.model import HFTransformerModel
from ...models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from ...parallel import compatibility
from ...parallel.parallelize import parallelize_hf_transformers
from ...parallel.pipeline_parallel.pipeline_parallel import PipelineParallelSetup
from ...parallel.topology import build_device_mesh, resolve_parallel_topology
from ...utils import logging as logging_utils
from ...utils import seeding

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...config.llmtuner_config import LLMTunerConfig
    from ..trainer import Trainer


def _build_model(
    trainer: Trainer,
) -> tuple[HFTransformerModel, PretrainedConfig, bool]:
    """Build the HF model and resolve whether to load initial HF weights.

    Returns ``(model, hf_model_config, load_hf_weights)``; the model is on
    meta device when HF weights will be loaded (they are materialized after
    parallelism is applied, so the load sees the sharded layout).
    """
    cfg = trainer.cfg
    #
    # Chunked loss + PP is rejected up front: under PP the last stage's
    # loss is computed inside the schedule
    # (``pipeline_parallel/pipeline_parallel.py:scalar_loss_fn``), which receives logits
    # from the stage forward. Rewiring that seam for hidden states plus a
    # per-chunk backward is a PP-side change, so the combination loud-raises
    # here rather than training on a silently un-chunked (or wrong) loss.
    trainer._chunked_loss_num_chunks = cfg.training.chunked_loss_num_chunks
    if (
        trainer._chunked_loss_num_chunks > 1
        and trainer.parallel_dims is not None
        and trainer.parallel_dims.pp_enabled
    ):
        compatibility.chunked_loss_pp(
            trainer._chunked_loss_num_chunks, trainer.parallel_dims.pp
        )
    hf_model_config = build_model_config_for(cfg)
    # Programmatic Trainer callers need the same resolved geometry as the CLI.
    # In particular, synthetic token generation must use the real vocabulary,
    # including offline arch_overrides and local saved configurations.
    text_config = unwrap_text_config(hf_model_config)
    cfg.model.update_architecture(text_config)
    load_hf_weights = bool(
        cfg.checkpoint.enable
        and cfg.checkpoint.initial_load_in_hf
        and cfg.checkpoint.initial_load_path
    )
    # EP swap rewrites the expert layout, which the HF checkpoint does not
    # carry -- refuse loudly instead of loading garbage into the experts.
    if (
        load_hf_weights
        and trainer.parallel_dims is not None
        and trainer.parallel_dims.ep_enabled
    ):
        compatibility.ep_hf_initial_load(trainer.parallel_dims.ep)
    if load_hf_weights:
        with torch.device("meta"):
            model = HFTransformerModel(hf_model_config)
    else:
        model = HFTransformerModel(hf_model_config).to(trainer.device)
    return model, hf_model_config, load_hf_weights


def _apply_parallelism(
    trainer: Trainer, model: HFTransformerModel, *, load_hf_weights: bool
) -> None:
    """Apply configured parallel stages, then materialize HF-load targets."""
    cfg = trainer.cfg
    orchestration = parallelize_hf_transformers(
        model,
        cfg=cfg.parallel,
        mesh=trainer.mesh,
        parallel_dims=trainer.parallel_dims,
        device=trainer.device,
        compile=cfg.training.compile,
        compile_config=cfg.training.compile_config,
        activation_checkpoint=cfg.training.activation_checkpoint_mode,
        selective_ac=cfg.training.selective_ac,
        memory_budget_ac=cfg.training.memory_budget_ac,
        region_ac=cfg.training.region_ac,
        global_batch_size=cfg.training.global_batch_size,
    )
    if isinstance(orchestration, PipelineParallelSetup):
        # pp > 1: no single model survives the split -- this rank holds its
        # stages' chunks only, and the schedule drives them in
        # ``pp_forward_backward_body``.
        trainer.model = None
        trainer.model_parts = orchestration.model_parts
        trainer.pp_schedule = orchestration.schedule
        trainer.pp_has_first_stage = orchestration.has_first_stage
        trainer.pp_has_last_stage = orchestration.has_last_stage
        # Only the last stage computes CE. Other stages return a finite zero
        # for the step check and contribute nothing to the PP loss reduction.
        trainer._pp_loss_sentinel = torch.zeros((1,), device=trainer.device)
    else:
        trainer.model = orchestration
        trainer.model_parts = [orchestration]

    if load_hf_weights:
        for model_part in trainer.model_parts:
            materialize_meta_model(model_part, trainer.device)


def _build_optimizer_stack(trainer: Trainer) -> None:
    """Optimizer, lr schedule, EMA, and the MoE/aux-loss optimizer pre-hooks."""
    cfg = trainer.cfg
    trainer.optimizer = OptimizersContainer(
        cfg.optimizer, model_parts=trainer.model_parts
    )

    # The lr schedule. Built regardless of whether the knobs were touched:
    # the default is warmup_steps=0 with no decay, so the factor is a
    # constant 1.0 and step 1 runs at exactly ``cfg.lr``. That costs one
    # multiply per step and removes the branch that would otherwise decide
    # whether the lr is scheduled -- a branch whose two sides would have to
    # be kept numerically identical forever.
    #
    # Handed the *inner* optimizers, not the container: a LambdaLR reads
    # ``lr`` off its optimizer's param groups, and the container's own
    # groups carry none (they are the merged parameter view). This is why
    # the scheduler is a container too.
    trainer.lr_scheduler = build_lr_scheduler(
        cfg.lr_scheduler_config,
        optimizers=list(trainer.optimizer),
        training_steps=cfg.steps,
    )

    # The weight EMA, a sibling of the optimizer rather than part of it:
    # stepped explicitly in ``train_step`` after the real update, and
    # registered with the checkpointer under its own ``ema`` key. Built
    # only when configured -- None costs nothing.
    ema_config = cfg.training.ema
    trainer.ema = (
        EMA(
            model_parts=trainer.model_parts,
            decay=ema_config.decay,
            half_life_fraction=ema_config.half_life_fraction,
            start_step=ema_config.start_step,
            step_bias=ema_config.step_bias,
            update_every_n_steps=ema_config.update_every_n_steps,
            buffer_patterns=ema_config.buffer_patterns,
        )
        if ema_config is not None
        else None
    )

    # Aux losses (the MoE load-balance loss a swapped-in MoE carries)
    # accumulate per forward; this pre-hook rolls the per-instance sums
    # into the step registers at each optimizer step. Harmless when no
    # aux loss exists.
    #
    # Registered on the container, so it fires once per step() call --
    # not once per inner optimizer, which is what a loop over the inner
    # optimizers would give under pipeline parallelism.
    register_aux_loss_zero_hook(trainer.optimizer, model_parts=trainer.model_parts)
    # Select one balancing policy and register one container-level pre-hook.
    register_moe_balancing_hook(
        trainer.optimizer,
        model_parts=trainer.model_parts,
        parallel_dims=trainer.parallel_dims,
    )


def _build_checkpointer(trainer: Trainer, hf_model_config: PretrainedConfig) -> None:
    """Register model, optimizer, schedule, EMA, and resumable runtime state.

    The trainer carries step/token counters; the dataloader carries its cursor.
    Both must resume alongside weights. The scheduler restores its own progress
    and base learning rates; optimizer state alone cannot restore the schedule.
    """
    cfg = trainer.cfg
    if trainer.dataloader is None:
        raise RuntimeError(
            "Trainer dataloader must be initialized before building the checkpointer"
        )
    states: dict[str, Any] = {TRAIN_STATE: trainer, DATALOADER: trainer.dataloader}
    trainer.checkpointer = DistributedCheckpointManager(
        cfg.checkpoint,
        model_parts=trainer.model_parts,
        optimizer=trainer.optimizer,
        lr_scheduler=trainer.lr_scheduler,
        ema=trainer.ema,
        states=states,
        folder=cfg.dump_folder,
        sd_adapter=HFTransformerStateDictAdapter(
            hf_model_config, cfg.checkpoint.initial_load_path or cfg.hf_model
        ),
    )


def assemble_trainer_runtime(trainer: Trainer, cfg: LLMTunerConfig) -> None:
    trainer.cfg = cfg
    if not is_distributed() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist_pytorch(get_distributed_backend())
    trainer.rank, trainer.world_size, trainer.local_rank = get_env_dist_info()
    if is_distributed():
        # Programmatic callers may initialize the group without torchrun's
        # environment. Mesh degrees must describe the live group.
        trainer.rank = get_rank()
        trainer.world_size = get_world_size()

    # Resolve the degrees first: the PP seed offset below needs this rank's
    # stage coordinate, and degree resolution draws no random numbers, so
    # seeding after it leaves every non-PP run bit-identical.
    trainer.parallel_dims = resolve_parallel_topology(
        cfg, world_size=trainer.world_size
    )

    # Validation's infeasible combinations are rejected here, before the
    # model and dataloader exist: a ``steps=-1`` pass that cannot terminate
    # cleanly would otherwise hang on its collectives mid-run.
    if cfg.validation is not None:
        trainer.check_validation_feasibility(
            cfg.validation,
            dp_world_size=(
                1
                if trainer.parallel_dims is None
                else trainer.parallel_dims.dp_replicate * trainer.parallel_dims.dp_shard
            ),
            chunked_loss_num_chunks=cfg.training.chunked_loss_num_chunks,
            training_dataset=cfg.dataloader.dataset,
        )

    # Deterministic seeding BEFORE model build so ranks sharing an SPMD
    # group build identical initial weights -- the precondition for
    # bit-exact DP comparisons. Pipeline stages hold different layers, so
    # seeding every stage identically would correlate their initialization;
    # under PP each stage offsets the base seed by its stage rank (the
    # upstream distinct_seed_mesh_dims=["pp"] semantics), while ranks at
    # the same stage keep the base seed.
    seed = cfg.seed
    if trainer.parallel_dims is not None and trainer.parallel_dims.pp_enabled:
        pp_mesh = trainer.parallel_dims.get_optional_mesh("pp")
        seed = seeding.derive_distinct_seed(
            seed, [(pp_mesh.get_local_rank(), pp_mesh.size())]
        )
    trainer.seed_everything(
        seed, deterministic=cfg.deterministic, detect_anomaly=cfg.detect_anomaly
    )

    trainer.device = torch.device(
        f"{device_type}:{trainer.local_rank}" if device_type != "cpu" else "cpu"
    )

    # Select the dense mesh. ``parallel_dims``
    #    is the same resolved degrees the mesh was built from, kept so the
    #    trainer can ask "how many DP ranks?" without re-indexing the mesh.
    if trainer.parallel_dims is not None and trainer.parallel_dims.pp_enabled:
        # The dense (dp, cp, tp) mesh does not cover the world under PP,
        # so ``build_device_mesh``'s coverage backstop would reject it. The same
        # view over this rank's non-PP coordinates exists per stage and is
        # what the per-part apply_* functions index (parallelize
        # resolves it off parallel_dims itself); keep the attribute
        # consistent.
        trainer.mesh = trainer.parallel_dims.spmd_dense_mesh()
    else:
        trainer.mesh = build_device_mesh(trainer.parallel_dims)

    # Build the model before optimizer services capture its parameters.
    model, hf_model_config, load_hf_weights = _build_model(trainer)
    _apply_parallelism(trainer, model, load_hf_weights=load_hf_weights)
    _build_optimizer_stack(trainer)

    # Build the micro-batch source before the checkpointer, which
    #    serializes its read position alongside the model.
    trainer.dataloader = trainer.build_dataloader()

    # Counters the checkpoint carries. Kept as plain ints so a resumed run
    # can log "step 61 (resumed at 60)" without re-deriving them.
    trainer.step = 0
    trainer.ntokens_seen = 0

    _build_checkpointer(trainer, hf_model_config)

    # Metrics need the mesh (for the throughput
    #    divisor and the metrics rank) and the model config (for FLOPs per
    #    token). It replaces the plain per-step ``logger.info`` the loop used
    #    to emit: the same loss and grad_norm, plus throughput, MFU and
    #    memory, and the frequency is now one knob instead of two.
    #
    #    FLOPs use the already resolved HF geometry, so metrics describe the
    #    model built above without loading its configuration a second time.
    trainer.metrics = MetricsProcessor(
        cfg.metrics,
        parallel_dims=trainer.parallel_dims,
        dump_folder=cfg.dump_folder,
        pp_schedule=cfg.pipeline_parallel_schedule,
        num_flops_per_token=flops_per_token(
            unwrap_text_config(hf_model_config), seq_len=cfg.max_seq_len
        ),
        tag=cfg.metrics.tag,
    )
    # The trainer synchronizes PP loss metrics to every stage before logging.
    logging_utils.set_log_ranks(cfg.metrics.log_ranks)
