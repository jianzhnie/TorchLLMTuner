"""Trainer assembly: everything between "a config" and "a trainer".

``Trainer.__init__`` is a thin shell -- it delegates here. This module owns
the assembly order, which is a contract:

1. process group + rank resolution, then degree resolution
   (``build_parallel_dims``) -- the PP seed offset below needs this rank's
   stage coordinate;
2. combination guards -- the support matrix (``llmtuner/parallel/matrix.py``)
   fires here: validation feasibility, chunked-loss x PP, EP x HF-initial-load
   (EP x checkpoint itself is supported since 2026-10, see
   ``parallel/expert_parallel/ckpt.py``);
3. deterministic seeding (before any model build);
4. the mesh (``build_mesh`` -- see ``llmtuner/parallel/parallel_dims.py``);
5. the model, then parallelism in stage-table order
   (``llmtuner/parallel/stages.py``, driven by ``parallelize_hf_transformers``);
6. optimizer, lr schedule, EMA, and the MoE aux/balance pre-hooks;
7. dataloader, checkpointer, metrics.

Each step consumes only the previous steps' products. Nothing here is a
runtime hot path; it runs exactly once per run.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch

from .. import parallel
from ..accelerator.device import (
    device_module,
    device_type,
    get_distributed_backend,
    get_env_dist_info,
)
from ..accelerator.dist_utils import init_dist_pytorch, is_distributed
from ..components.checkpointer import DATALOADER, TRAIN_STATE, CheckpointManager
from ..components.metrics import MetricsProcessor
from ..components.optimizer import EMA, OptimizersContainer, build_lr_scheduler
from ..models.common.aux_loss import register_aux_loss_zero_hook
from ..models.common.moe.balancing import (
    register_moe_load_balancing_hook,
    register_moe_quantile_balancing_hook,
)
from ..models.hf.factory import (
    build_model_config_for,
    materialize_meta_model,
    num_flops_per_token,
)
from ..models.hf.model import HFTransformerModel
from ..models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from ..parallel import matrix
from ..parallel.parallel_dims import build_mesh, build_parallel_dims
from ..parallel.pipeline_parallel import PipelineParallelSetup
from ..utils import logger_utils
from ..utils.logger_utils import get_logger

logger = get_logger(__name__)


def _build_model(self, cfg):
    """Steps 2-3a: the HF model, its config, and the HF-initial-load decision.

    Returns ``(model, hf_model_config, load_hf_weights)``; the model is on
    meta device when HF weights will be loaded (they are materialized after
    parallelism is applied, so the load sees the sharded layout).
    """
    #
    # Chunked loss + PP is rejected up front: under PP the last stage's
    # loss is computed inside the schedule
    # (``pipeline_parallel/apply.py:scalar_loss_fn``), which receives logits
    # from the stage forward. Rewiring that seam for hidden states plus a
    # per-chunk backward is a PP-side change, so the combination loud-raises
    # here rather than training on a silently un-chunked (or wrong) loss.
    self._chunked_loss_num_chunks = cfg.training.chunked_loss_num_chunks
    if (
        self._chunked_loss_num_chunks > 1
        and self.parallel_dims is not None
        and self.parallel_dims.pp_enabled
    ):
        matrix.chunked_loss_pp(
            self._chunked_loss_num_chunks, self.parallel_dims.pp
        )
    hf_model_config = build_model_config_for(cfg)
    load_hf_weights = bool(
        cfg.checkpoint.enable
        and cfg.checkpoint.initial_load_in_hf
        and cfg.checkpoint.initial_load_path
    )
    # EP swap rewrites the expert layout, which the HF checkpoint does not
    # carry -- refuse loudly instead of loading garbage into the experts.
    if (
        load_hf_weights
        and self.parallel_dims is not None
        and self.parallel_dims.ep_enabled
    ):
        matrix.ep_hf_initial_load(self.parallel_dims.ep)
    if load_hf_weights:
        with torch.device("meta"):
            model = HFTransformerModel(hf_model_config)
    else:
        model = HFTransformerModel(hf_model_config).to(self.device)
    return model, hf_model_config, load_hf_weights


def _apply_parallelism(self, cfg, model, load_hf_weights: bool) -> None:
    """Step 3: apply every parallelism dimension, then materialize.

    Titan's order: tp/pp/cp/ep declared first, fsdp last (outer wraps
    inner). Each is a no-op when its degree is 1.
    """
    # 3. parallelism, in Titan's order: tp/pp/cp/ep declared first, fsdp last
    #    (outer wraps inner). Each is a no-op when its degree is 1. The
    #    parallel layer's contract is ParallelConfig plus explicit scalars,
    #    so the training-side values it needs are unpacked here.
    orchestration = parallel.parallelize_hf_transformers(
        model,
        cfg=cfg.parallel,
        mesh=self.mesh,
        parallel_dims=self.parallel_dims,
        device=self.device,
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
        self.model = None
        self.model_parts = orchestration.model_parts
        self.pp_schedule = orchestration.schedule
        self.pp_has_first_stage = orchestration.has_first_stage
        self.pp_has_last_stage = orchestration.has_last_stage
        # The loss exists only on the last stage; every other stage reports
        # this sentinel, which is finite (the finiteness check runs on every
        # rank) and never logged (the metrics rank is a last-stage rank).
        self._pp_loss_sentinel = torch.full((1,), -1.0, device=self.device)
    else:
        self.model = orchestration
        self.model_parts = [orchestration]

    if load_hf_weights:
        for model_part in self.model_parts:
            materialize_meta_model(model_part, self.device)


def _build_optimizer_stack(self, cfg) -> None:
    """Optimizer, lr schedule, EMA, and the MoE/aux-loss optimizer pre-hooks."""
    self.optimizer = OptimizersContainer(
        cfg.optimizer, model_parts=self.model_parts
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
    self.lr_scheduler = build_lr_scheduler(
        cfg.lr_scheduler_config,
        optimizers=list(self.optimizer),
        training_steps=cfg.steps,
    )

    # The weight EMA, a sibling of the optimizer rather than part of it:
    # stepped explicitly in ``train_step`` after the real update, and
    # registered with the checkpointer under its own ``ema`` key. Built
    # only when configured -- None costs nothing.
    ema_config = cfg.training.ema
    self.ema = (
        EMA(
            model_parts=self.model_parts,
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
    register_aux_loss_zero_hook(
        self.optimizer, self.model_parts, self.parallel_dims
    )
    # A second pre-hook on the same container, same granularity. No-op for
    # a model without MoE layers, which is every model except a swapped-in
    # one (the swap is what installs ``load_balance_coeff``).
    register_moe_load_balancing_hook(
        self.optimizer, self.model_parts, self.parallel_dims
    )
    # The quantile counterpart, registered alongside: the two schemes are
    # mutually exclusive per model, so exactly one of the two hooks ever
    # fires -- this one no-ops unless the swap installed quantile routers
    # (``moe_quantile_balancing``).
    register_moe_quantile_balancing_hook(
        self.optimizer, self.model_parts, self.parallel_dims
    )


def _build_checkpointer(self, cfg, hf_model_config) -> None:
    # 5. checkpointing, last because it needs the model and optimizer it is
    #    going to serialize, and because a checkpoint is meaningless until
    #    there is something shaped like a training state to save.
    #
    #    ``self`` rides along as TRAIN_STATE: the manager saves ``states``
    #    wholesale, and the step/token counters are not reachable from either
    #    the model or the optimizer, so a resumed run would otherwise restart
    #    its schedule from zero with weights that are already trained.
    #
    #    A loadable dataloader rides along too: resuming without its read
    #    position would resume the weights and restart the data, silently
    #    training a second pass over the beginning of the corpus.
    #
    #    The schedule rides along for one integer, ``last_epoch``, that
    #    nothing else in the checkpoint carries. The optimizer restores its
    #    ``base_lrs`` -- so the *current* lr comes back right -- but
    #    ``last_epoch`` is the scheduler's own counter, and a resumed run's
    #    fresh scheduler starts it at 0. Without it the curve restarts from
    #    the beginning on the step after a resume: silent whenever warmup
    #    and decay are both off (the lr is then constant and the mistake
    #    invisible), and wrong for the rest of the run once either is set.
    states: dict[str, Any] = {TRAIN_STATE: self}
    if self.dataloader is not None:
        states[DATALOADER] = self.dataloader
    self.checkpointer = CheckpointManager(
        cfg.checkpoint,
        model_parts=self.model_parts,
        optimizer=self.optimizer,
        lr_scheduler=self.lr_scheduler,
        ema=self.ema,
        states=states,
        folder=cfg.dump_folder,
        sd_adapter=HFTransformerStateDictAdapter(
            hf_model_config, cfg.checkpoint.initial_load_path or cfg.hf_model
        ),
    )


def build_trainer_state(self, cfg) -> None:
    self.cfg = cfg
    if (
        not is_distributed()
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    ):
        init_dist_pytorch(get_distributed_backend())
    self.rank, self.world_size, self.local_rank = get_env_dist_info()

    # Resolve the degrees first: the PP seed offset below needs this rank's
    # stage coordinate, and degree resolution draws no random numbers, so
    # seeding after it leaves every non-PP run bit-identical.
    self.parallel_dims = build_parallel_dims(cfg, self.world_size)

    # Validation's infeasible combinations are rejected here, before the
    # model and dataloader exist: a ``steps=-1`` pass that cannot terminate
    # cleanly would otherwise hang on its collectives mid-run.
    if cfg.validation is not None:
        self.check_validation_feasibility(
            cfg.validation,
            dp_world_size=(
                1
                if self.parallel_dims is None
                else self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard
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
    if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
        pp_mesh = self.parallel_dims.get_optional_mesh("pp")
        seed = derive_distinct_seed(
            seed, [(pp_mesh.get_local_rank(), pp_mesh.size())]
        )
    self.seed_everything(
        seed, deterministic=cfg.deterministic, detect_anomaly=cfg.detect_anomaly
    )

    self.device = torch.device(
        f"{device_type}:{self.local_rank}" if device_type != "cpu" else "cpu"
    )

    # 1. mesh (the process topology every dimension is built on). ``parallel_dims``
    #    is the same resolved degrees the mesh was built from, kept so the
    #    trainer can ask "how many DP ranks?" without re-indexing the mesh.
    if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
        # The dense (dp, cp, tp) mesh does not cover the world under PP,
        # so ``build_mesh``'s coverage backstop would reject it. The same
        # view over this rank's non-PP coordinates exists per stage and is
        # what the per-part apply_* functions index (parallelize
        # resolves it off parallel_dims itself); keep the attribute
        # consistent.
        self.mesh = self.parallel_dims.spmd_dense_mesh()
    else:
        self.mesh = build_mesh(self.parallel_dims)

    # 2. the model -- HF's own initialization, wrapped for this loop
    model, hf_model_config, load_hf_weights = _build_model(self, cfg)
    _apply_parallelism(self, cfg, model, load_hf_weights)
    _build_optimizer_stack(self, cfg)

    # 4. the micro-batch source. Built before the checkpointer, which
    #    serializes its read position alongside the model.
    self.dataloader = self.build_dataloader()

    _build_checkpointer(self, cfg, hf_model_config)

    # Counters the checkpoint carries. Kept as plain ints so a resumed run
    # can log "step 61 (resumed at 60)" without re-deriving them.
    self.step = 0
    self.ntokens_seen = 0

    # 6. metrics, last because it needs the mesh (for the throughput
    #    divisor and the metrics rank) and the model config (for FLOPs per
    #    token). It replaces the plain per-step ``logger.info`` the loop used
    #    to emit: the same loss and grad_norm, plus throughput, MFU and
    #    memory, and the frequency is now one knob instead of two.
    #
    #    ``num_flops_per_token`` is measured from the parameters, not from
    #    the config's sizes, so the number describes the model that actually
    #    exists -- including one whose sizes came from the Hub.
    self.metrics = MetricsProcessor(
        cfg.metrics,
        parallel_dims=self.parallel_dims,
        dump_folder=cfg.dump_folder,
        pp_schedule=cfg.pipeline_parallel_schedule,
        num_flops_per_token=num_flops_per_token(cfg),
        tag=cfg.metrics.tag,
    )
    # Under PP the loss exists on one rank and ``metrics.log_ranks`` decides
    # which ranks print, so a mismatched pair trains correctly and reports
    # nothing -- which reads exactly like a hang. Wire the console filter and
    # warn now, rather than leave the user to work it out at step 1.
    logger_utils.set_log_ranks(cfg.metrics.log_ranks)
    self.metrics.ensure_pp_loss_visible()


# -- seeding ------------------------------------------------------------------
def derive_distinct_seed(seed: int, distinct_coords: Iterable[tuple[int, int]]) -> int:
    """Offset ``seed`` by this rank's coordinates along distinct mesh dims.

    Each ``(local_rank, dim_size)`` pair contributes ``local_rank`` times the
    product of all previous dimensions' sizes -- row-major indexing over the
    distinct sub-mesh, so every coordinate tuple maps to a unique offset. The
    result is reduced mod 2**64 to stay inside ``torch.manual_seed``'s range.

    An empty ``distinct_coords`` (or all-zero local ranks, e.g. a size-1 dim)
    returns ``seed`` unchanged, so single-stage runs are bit-identical to
    seeding without derivation.
    """
    offset = 0
    cumulative_size = 1
    for local_rank, dim_size in distinct_coords:
        offset += local_rank * cumulative_size
        cumulative_size *= dim_size
    return (seed + offset) % 2**64


def seed_everything(
    seed: int, *, deterministic: bool, detect_anomaly: bool = False
) -> None:
    """Seed every generator this run draws from.

    torchtitan's ``set_determinism`` semantics; ``Trainer.seed_everything``
    delegates here.
    """
    torch.manual_seed(seed)
    # Hash randomization is not observable in this process (PYTHONHASHSEED is
    # read at interpreter start), but dataloader workers are spawned later and
    # do read it, so upstream sets it here for them. Same spelling.
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    if device_type != "cpu":
        device_module.manual_seed_all(seed)
    if deterministic:
        # torchtitan's ``set_determinism``, minus the parts that only exist
        # for its own stack (the DTensor mesh-aware RNG tracker, the
        # flex-attention kernels) and plus this one spelled out:
        # ``use_deterministic_algorithms(True)`` turns on
        # ``fill_uninitialized_memory``, whose fill kernel races with side
        # streams and is what made HF's RoPE init observe NaN, so upstream
        # turns it back off for the same reason.
        #
        # ``warn_only=False`` is llmtuner's fixed, stricter setting;
        # upstream reads it from ``debug.deterministic_warn_only``.
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.utils.deterministic.fill_uninitialized_memory = False
        # Deterministic cuBLAS needs a workspace split, not the default one.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if detect_anomaly:
        logger.warning(
            "Anomaly detection enabled. This incurs significant overhead and "
            "is for debugging only."
        )
        # ``check_nan=False``: the NaN/Inf gradient check calls
        # ``aten._is_any_true``, which has no DTensor sharding strategy and
        # would crash on sharded parameters. Stack-trace recording -- the
        # useful half -- stays on. Same setting as upstream's.
        torch.autograd.set_detect_anomaly(True, check_nan=False)
