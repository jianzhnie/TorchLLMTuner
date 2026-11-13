"""Build the trainer's model, parallel layout, and training services.

Resolve the process group and topology, check validation feasibility, seed the
run, and select the dense mesh. Then build the explicit model, apply parallel stages,
and create optimizer/scheduler/EMA services. Finally build the dataloader,
checkpointer, counters, and metrics. Model-dependent combination guards run
while building the model, before parallel transformations.

Parallel stage order is defined in ``parallel/stages.py``. Pipeline parallelism
splits the model first, then applies those stages to each local model part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule

from ...accelerator.collectives import synchronize_local_errors
from ...components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from ...components.checkpointer.backends.torch_checkpointing_backend import (
    TorchCheckpointingManager,
)
from ...components.checkpointer.core.checkpoint_keys import (
    DATALOADER,
    RNG_STATE,
    TRAIN_STATE,
)
from ...components.checkpointer.core.random_state import RNGCheckpointState
from ...components.checkpointer.model_state import ModelStateDictAdapter
from ...components.metrics import MetricsProcessor
from ...components.optimizer.ema import EMA
from ...components.optimizer.lr_scheduler import build_lr_scheduler
from ...components.optimizer.optimizer import OptimizersContainer
from ...components.optimizer.state_residency import validate_checkpoint_configuration
from ...errors import UnsupportedCombinationError
from ...memory.activation_swap import reset_activation_swap_metrics
from ...memory.materialization import materialize_meta_model
from ...models.common.aux_loss import register_aux_loss_metrics_hook
from ...models.common.moe.bias_update import (
    register_moe_balancing_hook,
)
from ...models.configuration import (
    build_model_config_for,
    resolve_model_class,
    unwrap_text_config,
)
from ...models.registry import resolve_state_format
from ...parallel import compatibility
from ...parallel.parallelize import parallelize_model
from ...parallel.pipeline_parallel.pipeline_parallel import PipelineParallelSetup
from ...utils import logging as logging_utils
from ...utils.flops import flops_estimation_note, flops_per_token
from .batch import build_training_dataloader
from .engine import LossSum, ParamContext, TrainingEngine
from .environment import RuntimeEnvironment, build_runtime_environment
from .training_observer import TrainingObserver, install_dispatch_timers

if TYPE_CHECKING:
    from torch.distributed.checkpoint.stateful import Stateful
    from transformers import PretrainedConfig

    from ...config.llmtuner_config import LLMTunerConfig
    from ...datasets.contracts import BaseDataLoader
    from ...parallel.topology import ParallelTopology
    from ..trainer import Trainer


@dataclass(frozen=True, slots=True)
class ModelAssembly:
    """Parallel model ownership and PP execution metadata for this rank."""

    model: torch.nn.Module | None
    model_parts: list[torch.nn.Module]
    pp_schedule: Any | None = None
    pp_has_first_stage: bool = False
    pp_has_last_stage: bool = False
    pp_loss_sentinel: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class OptimizerStack:
    """Optimizer services sharing the final parallel parameter layout."""

    optimizer: OptimizersContainer
    lr_scheduler: Any
    ema: EMA | None


@dataclass(frozen=True, slots=True)
class PreparedModel:
    """Model preparation output before optimizer or service construction.

    Counts describe unique logical parameters before distributed ownership
    changes. Configuration and local assembly are retained by reference.
    """

    assembly: ModelAssembly
    config: PretrainedConfig
    num_parameters: int
    num_trainable_parameters: int


def build_model(
    cfg: LLMTunerConfig,
    *,
    parallel_dims: ParallelTopology | None,
    device: torch.device,
) -> tuple[torch.nn.Module, PretrainedConfig, bool]:
    """Build the registered explicit model and resolve initial weight loading.

    Returns ``(model, model_config, load_initial_weights)``. The selected
    implementation owns its module tree; this function only resolves config,
    construction device and checkpoint policy.
    """
    # PP schedules consume logits; chunked loss requires a hidden-state path.
    chunked_loss_num_chunks = cfg.training.chunked_loss_num_chunks
    if (
        chunked_loss_num_chunks > 1
        and parallel_dims is not None
        and parallel_dims.pp_enabled
    ):
        compatibility.chunked_loss_pp(chunked_loss_num_chunks, parallel_dims.pp)
    if (
        chunked_loss_num_chunks > 1
        and parallel_dims is not None
        and parallel_dims.dp_shard_enabled
    ):
        compatibility.chunked_loss_fsdp(
            chunked_loss_num_chunks,
            fsdp_config=cfg.parallel.fsdp_config,
            hybrid_parallel=(
                parallel_dims.tp_enabled
                or parallel_dims.cp_enabled
                or parallel_dims.ep_enabled
                or parallel_dims.pp_enabled
            ),
            compile=cfg.training.compile,
        )
    model_config = build_model_config_for(cfg)
    # Programmatic Trainer callers need the same resolved geometry as the CLI.
    # In particular, synthetic token generation must use the real vocabulary,
    # including offline arch_overrides and local saved configurations.
    text_config = unwrap_text_config(model_config)
    cfg.model.update_architecture(text_config)
    load_initial_weights = bool(
        cfg.checkpoint.enable
        and cfg.checkpoint.initial_load_in_hf
        and cfg.checkpoint.initial_load_path
    )
    if load_initial_weights and parallel_dims is not None and parallel_dims.ep_enabled:
        # The registered format owns expert key/layout support. Assembly must
        # not infer it from architecture names or duplicate packing rules.
        state_format = resolve_state_format(
            text_config,
            expert_parallel=True,
            sharded_expert_parallel=cfg.checkpoint.sharded_expert_parallel,
        )
        if not state_format.supports_ep_initial_load(
            tp_degree=cfg.parallel.tp_size,
            sharded_tensor_parallel=cfg.checkpoint.sharded_tensor_parallel,
        ):
            compatibility.ep_hf_initial_load(parallel_dims.ep)
    if not load_initial_weights and cfg.model.initialization_device == "meta":
        if any(
            degree > 1
            for degree in (
                cfg.parallel.tp_size,
                cfg.parallel.pp_size,
                cfg.parallel.cp_size,
                cfg.parallel.ep_size,
            )
        ):
            raise UnsupportedCombinationError(
                "Random meta initialization supports local/DP/FSDP; "
                "TP/EP/PP/CP paths need an ownership-aware initializer."
            )
    model_class = resolve_model_class(text_config)
    if not load_initial_weights and cfg.model.initialization_device == "meta":
        for method in ("initialize_parameters", "materialization_buffers"):
            if not callable(getattr(model_class, method, None)):
                raise UnsupportedCombinationError(
                    f"Random meta initialization requires model-owned {method}()"
                )
    if load_initial_weights or cfg.model.initialization_device == "meta":
        with torch.device("meta"):
            model = model_class(text_config)
    else:
        model = model_class(text_config).to(device)
        initialize_buffers = getattr(model, "initialize_training_buffers", None)
        if callable(initialize_buffers):
            initialize_buffers()
    return model, model_config, load_initial_weights


def _fsdp_managed_parameters(model: torch.nn.Module) -> set[torch.nn.Parameter]:
    return {
        fsdp_param.sharded_param
        for module in model.modules()
        if isinstance(module, FSDPModule)
        for group in module._get_fsdp_state()._fsdp_param_groups
        for fsdp_param in group.fsdp_params
    }


def _model_materialization_buffers(
    model: torch.nn.Module, device: torch.device
) -> dict[tuple[torch.nn.Module, str], torch.Tensor]:
    """Ask the explicit model to rebuild non-checkpoint buffers.

    Buffer reconstruction is a model contract. Keeping this adapter here makes
    older explicit models fail at materialization with a useful error instead
    of allowing uninitialized runtime state into training.
    """
    builder = getattr(model, "materialization_buffers", None)
    if builder is None:
        return {}
    if not callable(builder):
        raise TypeError(
            f"{type(model).__name__}.materialization_buffers must be callable"
        )
    return builder(device)


def _apply_parallelism(
    model: torch.nn.Module,
    *,
    cfg: LLMTunerConfig,
    mesh: DeviceMesh | None,
    parallel_dims: ParallelTopology | None,
    device: torch.device,
    load_initial_weights: bool,
    error_group: ProcessGroup | None = None,
) -> ModelAssembly:
    """Apply configured stages and return this rank's explicit model ownership."""
    orchestration = parallelize_model(
        model,
        cfg=cfg.parallel,
        mesh=mesh,
        parallel_dims=parallel_dims,
        error_group=error_group,
        device=device,
        compile=cfg.training.compile,
        compile_config=cfg.training.compile_config,
        activation_checkpoint=cfg.training.activation_checkpoint_mode,
        activation_swap=cfg.training.activation_swap,
        activation_swap_sync_release=cfg.training.activation_swap_sync_release,
        activation_swap_max_host_bytes=cfg.training.activation_swap_max_host_bytes,
        selective_ac=cfg.training.selective_ac,
        memory_budget_ac=cfg.training.memory_budget_ac,
        region_ac=cfg.training.region_ac,
        global_batch_size=cfg.training.global_batch_size,
        chunked_loss_num_chunks=cfg.training.chunked_loss_num_chunks,
    )
    if isinstance(orchestration, PipelineParallelSetup):
        assembled = ModelAssembly(
            model=None,
            model_parts=orchestration.model_parts,
            pp_schedule=orchestration.schedule,
            pp_has_first_stage=orchestration.has_first_stage,
            pp_has_last_stage=orchestration.has_last_stage,
            # Non-last stages contribute a finite float32 zero to PP loss
            # reduction. Its scalar shape and dtype must match the last stage.
            pp_loss_sentinel=torch.zeros((), dtype=torch.float32, device=device),
        )
    else:
        assembled = ModelAssembly(model=orchestration, model_parts=[orchestration])

    if load_initial_weights or cfg.model.initialization_device == "meta":
        materialization_error: BaseException | None = None
        try:
            for model_part in assembled.model_parts:
                if cfg.parallel.fsdp_config.fsdp_cpu_offload:
                    materialize_meta_model(
                        model_part,
                        device,
                        parameter_device=torch.device("cpu"),
                        offloaded_parameters=_fsdp_managed_parameters(model_part),
                        runtime_buffers=_model_materialization_buffers(
                            model_part, device
                        ),
                    )
                else:
                    materialize_meta_model(
                        model_part,
                        device,
                        runtime_buffers=_model_materialization_buffers(
                            model_part, device
                        ),
                    )
        except BaseException as error:
            materialization_error = error
        # FSDP _apply may manage communication state. Finish materialization
        # outside the local-only boundary, then agree before optimizer setup.
        with synchronize_local_errors("Model materialization", group=error_group):
            if materialization_error is not None:
                raise materialization_error
        if not load_initial_weights:
            initialization_error: BaseException | None = None
            try:
                for model_part in assembled.model_parts:
                    initialize_model = getattr(
                        model_part, "initialize_parameters", None
                    )
                    if not callable(initialize_model):
                        raise TypeError(
                            f"{type(model_part).__name__} must implement "
                            "initialize_parameters() for meta initialization"
                        )
                    # DTensor initialization may communicate. Complete it
                    # outside the local-only boundary, just like materialization.
                    initialize_model()
            except BaseException as error:
                initialization_error = error
            with synchronize_local_errors(
                "Random model initialization", group=error_group
            ):
                if initialization_error is not None:
                    raise initialization_error
    return assembled


def prepare_parallel_model(
    cfg: LLMTunerConfig, *, environment: RuntimeEnvironment
) -> PreparedModel:
    """Construct, count and assemble a model without accessing a Trainer.

    Keep counting before TP/EP/PP/FSDP and preserve the materialization/load
    lifecycle in _apply_parallelism. No optimizer captures parameters until
    this builder has returned the final local layout.
    """
    with synchronize_local_errors("Model construction", group=environment.error_group):
        model, model_config, load_initial_weights = build_model(
            cfg, parallel_dims=environment.parallel_dims, device=environment.device
        )
        num_parameters = sum(parameter.numel() for parameter in model.parameters())
        num_trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    assembled = _apply_parallelism(
        model,
        cfg=cfg,
        mesh=environment.mesh,
        parallel_dims=environment.parallel_dims,
        device=environment.device,
        load_initial_weights=load_initial_weights,
        error_group=environment.error_group,
    )
    return PreparedModel(
        assembly=assembled,
        config=model_config,
        num_parameters=num_parameters,
        num_trainable_parameters=num_trainable_parameters,
    )


def create_training_engine(
    cfg: LLMTunerConfig,
    *,
    model_assembly: ModelAssembly,
    optimizer_stack: OptimizerStack,
    parallel_dims: ParallelTopology | None,
    device: torch.device,
    param_context: ParamContext,
    loss_sum: LossSum,
    error_group: ProcessGroup | None = None,
) -> TrainingEngine:
    """Bind execution dependencies without reading or mutating a Trainer.

    The assembly owns model parts and pipeline metadata; optimizer services
    refer to the final parameter layout. This factory consumes both results
    without reselecting model or parallel strategies. ``error_group`` covers
    all update participants, not only ranks sharing a DP or PP mesh axis.
    """
    return TrainingEngine(
        model_parts=model_assembly.model_parts,
        parallel_dims=parallel_dims,
        device=device,
        max_context_length=cfg.max_seq_len,
        chunked_loss_num_chunks=cfg.training.chunked_loss_num_chunks,
        param_context=param_context,
        loss_sum=loss_sum,
        optimizer=optimizer_stack.optimizer,
        lr_scheduler=optimizer_stack.lr_scheduler,
        ema=optimizer_stack.ema,
        pp_schedule=model_assembly.pp_schedule,
        pp_has_first_stage=model_assembly.pp_has_first_stage,
        pp_has_last_stage=model_assembly.pp_has_last_stage,
        pp_loss_sentinel=model_assembly.pp_loss_sentinel,
        pp_num_microbatches=cfg.parallel.pp_config.num_pp_microbatches,
        error_group=error_group,
    )


def build_optimizer_stack(
    cfg: LLMTunerConfig,
    *,
    model_parts: list[torch.nn.Module],
    parallel_dims: ParallelTopology | None,
    error_group: ProcessGroup | None = None,
) -> OptimizerStack:
    """Build optimizer services against the final parallel parameter layout.

    Pass the same error group to parameter planning and scheduler/EMA creation.
    This group coordinates failures; it does not change gradient ownership.
    """
    optimizer = OptimizersContainer(
        cfg.optimizer,
        sharded_expert_parallel=cfg.checkpoint.sharded_expert_parallel,
        sharded_tensor_parallel=cfg.checkpoint.sharded_tensor_parallel,
        model_parts=model_parts,
        initialization_group=error_group,
        pipeline_mesh=(
            parallel_dims.get_optional_mesh("pp") if parallel_dims is not None else None
        ),
    )

    with synchronize_local_errors("Optimizer services construction", group=error_group):
        lr_scheduler = build_lr_scheduler(
            cfg.lr_scheduler_config,
            optimizers=list(optimizer),
            training_steps=cfg.steps,
        )
        ema_config = cfg.training.ema
        ema = (
            EMA(
                model_parts=model_parts,
                sharded_expert_parallel=cfg.checkpoint.sharded_expert_parallel,
                sharded_tensor_parallel=cfg.checkpoint.sharded_tensor_parallel,
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
        # Container hooks run once per optimizer step, including multiple PP parts.
        register_aux_loss_metrics_hook(optimizer, model_parts=model_parts)
        register_moe_balancing_hook(
            optimizer, model_parts=model_parts, parallel_dims=parallel_dims
        )
    return OptimizerStack(optimizer=optimizer, lr_scheduler=lr_scheduler, ema=ema)


def build_checkpointer(
    cfg: LLMTunerConfig,
    model_config: PretrainedConfig,
    *,
    model_assembly: ModelAssembly,
    optimizer_stack: OptimizerStack,
    dataloader: BaseDataLoader,
    train_state: Stateful,
    device: torch.device,
    parallel_dims: ParallelTopology | None,
    error_group: ProcessGroup | None = None,
) -> DistributedCheckpointManager | TorchCheckpointingManager:
    """Build checkpoint orchestration around explicitly owned runtime state.

    ``error_group`` covers adapter preparation, format-conversion failures and
    external key coverage. Checkpoint manager IO keeps its own collective scope;
    this argument does not repartition persistence.
    """
    with synchronize_local_errors("Checkpoint services preparation", group=error_group):
        if cfg.checkpoint.enable and cfg.optimizer.state_residency == "npu_virtual":
            validate_checkpoint_configuration(
                cfg.checkpoint.backend, cfg.checkpoint.async_mode
            )
        if dataloader is None:
            raise RuntimeError(
                "Trainer dataloader must be initialized before "
                "building the checkpointer"
            )
        states: dict[str, Any] = {
            TRAIN_STATE: train_state,
            DATALOADER: dataloader,
            RNG_STATE: RNGCheckpointState(device, topology=parallel_dims),
        }
        sd_adapter = ModelStateDictAdapter(
            model_config,
            cfg.checkpoint.initial_load_path or cfg.hf_model,
            model_prefix="",
            expert_parallel=cfg.parallel.ep_size > 1,
            sharded_expert_parallel=cfg.checkpoint.sharded_expert_parallel,
            error_group=error_group,
        )
    manager_cls = (
        TorchCheckpointingManager
        if cfg.checkpoint.backend == "torch_checkpointing"
        else DistributedCheckpointManager
    )
    return manager_cls(
        cfg.checkpoint,
        model_parts=model_assembly.model_parts,
        optimizer=optimizer_stack.optimizer,
        lr_scheduler=optimizer_stack.lr_scheduler,
        ema=optimizer_stack.ema,
        states=states,
        folder=cfg.dump_folder,
        sd_adapter=sd_adapter,
    )


def build_metrics_processor(
    cfg: LLMTunerConfig,
    model_config: PretrainedConfig,
    *,
    parallel_dims: ParallelTopology | None,
    num_parameters: int,
    num_trainable_parameters: int,
    error_group: ProcessGroup | None = None,
) -> MetricsProcessor:
    """Build metrics with explicit construction-failure coordination.

    ``parallel_dims`` still defines metric reductions. ``error_group`` determines
    which peers must accept construction before the logger is published.
    """
    metrics = None
    try:
        with synchronize_local_errors("Metrics construction", group=error_group):
            text_config = unwrap_text_config(model_config)
            metrics = MetricsProcessor(
                cfg.metrics,
                parallel_dims=parallel_dims,
                dump_folder=cfg.dump_folder,
                num_parameters=num_parameters,
                num_trainable_parameters=num_trainable_parameters,
                pp_schedule=cfg.parallel.pp_config.pipeline_parallel_schedule,
                num_flops_per_token=flops_per_token(
                    text_config, seq_len=cfg.max_seq_len
                ),
            )
            note = flops_estimation_note(text_config, seq_len=cfg.max_seq_len)
            if note is not None:
                logging_utils.get_logger(__name__).info("%s", note)
    except BaseException:
        # The factory owns the logger until every rank accepts construction.
        # A peer failure leaves no object on Trainer for its close() to release.
        if metrics is not None:
            try:
                metrics.close()
            except BaseException:
                try:
                    logging_utils.get_logger(__name__).exception(
                        "Metrics cleanup failed after a construction error"
                    )
                except BaseException:
                    pass
        raise
    return metrics


def assemble_trainer_runtime(trainer: Trainer, cfg: LLMTunerConfig) -> None:
    trainer.cfg = cfg
    environment = build_runtime_environment(
        cfg,
        seed_everything=trainer.seed_everything,
        validate_validation=trainer.check_validation_feasibility,
    )
    trainer.rank = environment.rank
    trainer.world_size = environment.world_size
    trainer.local_rank = environment.local_rank
    trainer.parallel_dims = environment.parallel_dims
    trainer.device = environment.device
    trainer.mesh = environment.mesh
    error_group = environment.error_group
    trainer.error_group = error_group

    prepared_model = prepare_parallel_model(cfg, environment=environment)
    model_assembly = prepared_model.assembly
    trainer.model = model_assembly.model
    trainer.model_parts = model_assembly.model_parts
    optimizer_stack = build_optimizer_stack(
        cfg,
        model_parts=model_assembly.model_parts,
        parallel_dims=environment.parallel_dims,
        error_group=error_group,
    )
    trainer.optimizer = optimizer_stack.optimizer
    trainer.lr_scheduler = optimizer_stack.lr_scheduler
    trainer.ema = optimizer_stack.ema
    trainer.training_engine = create_training_engine(
        cfg,
        model_assembly=model_assembly,
        optimizer_stack=optimizer_stack,
        parallel_dims=environment.parallel_dims,
        device=environment.device,
        param_context=trainer.param_context,
        loss_sum=trainer.loss_sum,
        error_group=error_group,
    )

    # The checkpointer captures the loader cursor alongside model state.
    with synchronize_local_errors("Dataloader construction", group=error_group):
        trainer.dataloader = build_training_dataloader(
            cfg, parallel_dims=environment.parallel_dims
        )

    # Counters the checkpoint carries. Kept as plain ints so a resumed run
    # can log "step 61 (resumed at 60)" without re-deriving them.
    trainer.step = 0
    trainer.ntokens_seen = 0

    trainer.checkpointer = build_checkpointer(
        cfg,
        prepared_model.config,
        model_assembly=model_assembly,
        optimizer_stack=optimizer_stack,
        dataloader=trainer.dataloader,
        train_state=trainer.progress,
        device=environment.device,
        parallel_dims=environment.parallel_dims,
        error_group=error_group,
    )

    trainer.metrics = build_metrics_processor(
        cfg,
        prepared_model.config,
        parallel_dims=environment.parallel_dims,
        num_parameters=prepared_model.num_parameters,
        num_trainable_parameters=prepared_model.num_trainable_parameters,
        error_group=error_group,
    )
    trainer.training_observer = TrainingObserver(
        trainer.metrics, environment.device, group=error_group
    )
    trainer.dispatch_timers = []
    if cfg.metrics.measure_moe_dispatch:
        with synchronize_local_errors(
            "Dispatch observer installation", group=error_group
        ):
            trainer.dispatch_timers = install_dispatch_timers(
                model_assembly.model_parts
            )
        if not trainer.dispatch_timers:
            logging_utils.get_logger(__name__).warning(
                "Dispatch timing requested, but no native RoutedExperts boundary "
                "was found; dispatch timing is unavailable for this model."
            )
    if cfg.activation_swap:
        reset_activation_swap_metrics()
    # The trainer synchronizes PP loss metrics to every stage before logging.
    logging_utils.set_log_ranks(cfg.metrics.log_ranks)
    if environment.device.type == "npu":
        logging_utils.get_logger(__name__).info(
            "NPU step guards use explicit synchronous scalar reads; "
            "loss/gradient and valid-token checks remain enabled"
        )
