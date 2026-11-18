"""Build the trainer's model, parallel layout, and training services.

Builders consume explicit configuration, model ownership and runtime environment
and return their results. They never receive a Trainer or assign its attributes.
Trainer owns creation order and retains resources as soon as they are acquired.
Model-dependent combination guards run before parallel transformations.

Parallel stage order is defined in ``parallel/stages.py``. Pipeline parallelism
splits the model first, then applies those stages to each local model part.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from experiments.pipeline_parallel import compatibility
from experiments.pipeline_parallel.builders import (
    PipelineScheduleFactory,
    TrainingBuilders,
)
from experiments.pipeline_parallel.engine import (
    ParamContext,
    TrainingEngine,
)
from experiments.pipeline_parallel.execution import (
    ExecutionLayout,
    validate_local_pipeline_layouts,
)
from experiments.pipeline_parallel.metrics import MetricsProcessor
from experiments.pipeline_parallel.optimizer import OptimizersContainer
from experiments.pipeline_parallel.parallelize import (
    parallelize_model,
)
from experiments.pipeline_parallel.pipeline_parallel import (
    PipelineParallelSetup,
)
from experiments.pipeline_parallel.random_state import RNGCheckpointState
from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.checkpointer.backends.torch_checkpointing_backend import (
    TorchCheckpointingManager,
)
from llmtuner.components.checkpointer.core.base import BaseCheckpointManager
from llmtuner.components.checkpointer.core.checkpoint_keys import (
    DATALOADER,
    RNG_STATE,
    TRAIN_STATE,
)
from llmtuner.components.checkpointer.model_state import ModelStateDictAdapter
from llmtuner.components.loss_contract import LossFunctions
from llmtuner.components.optimizer.ema import EMA
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.state_residency import (
    validate_checkpoint_configuration,
)
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.memory.materialization import (
    materialize_meta_model,
    model_materialization_buffers,
)
from llmtuner.models.common.aux_loss import register_aux_loss_metrics_hook
from llmtuner.models.common.moe.bias_update import register_moe_balancing_hook
from llmtuner.models.configuration import (
    build_model_config_for,
    unwrap_text_config,
)
from llmtuner.models.configuration import (
    resolve_model_class as resolve_registered_model_class,
)
from llmtuner.models.flops import flops_estimation_note, flops_per_token
from llmtuner.models.registry import resolve_state_format
from llmtuner.parallel.assembly import ModelPartLayout, validate_model_part_layouts
from llmtuner.parallel.fully_shard_parallel.policies import fsdp_managed_parameters
from llmtuner.trainer.runtime.assembly import (
    build_loss_functions as build_loss_functions,
)
from llmtuner.utils import logging as logging_utils

if TYPE_CHECKING:
    from torch.distributed.checkpoint.stateful import Stateful
    from transformers import PretrainedConfig

    from experiments.pipeline_parallel.config import (
        LLMTunerConfig,
    )
    from experiments.pipeline_parallel.environment import RuntimeEnvironment
    from experiments.pipeline_parallel.topology import ParallelTopology
    from llmtuner.datasets.contracts import BaseDataLoader


def _resolve_factory(
    factory: Callable[..., Any] | None, default: Callable[..., Any]
) -> Callable[..., Any]:
    return default if factory is None else factory


@dataclass(frozen=True, slots=True)
class ModelAssembly:
    """Parallel model ownership and PP execution metadata for this rank."""

    model: torch.nn.Module | None
    model_parts: list[torch.nn.Module]
    layouts: tuple[ModelPartLayout, ...] = ()
    pp_schedule: Any | None = None
    pp_has_first_stage: bool = False
    pp_has_last_stage: bool = False
    pp_loss_sentinel: torch.Tensor | None = None

    @property
    def execution_layouts(self) -> tuple[ExecutionLayout, ...]:
        """Expose declared stage output and execution capabilities to consumers."""
        if self.layouts and len(self.layouts) != len(self.model_parts):
            raise ValueError("Execution layouts must match local model parts")
        execution = []
        for layout in self.layouts:
            if not isinstance(layout.execution, ExecutionLayout):
                raise TypeError(
                    "Every assembled model part must provide an ExecutionLayout"
                )
            execution.append(layout.execution)
        return tuple(execution)

    def validate_layouts(self) -> None:
        """Validate supplied layout metadata against the final local parts."""
        if self.layouts:
            validate_model_part_layouts(self.model_parts, self.layouts)
            execution = self.execution_layouts
            if self.pp_schedule is not None:
                validate_local_pipeline_layouts(
                    execution,
                    has_first_stage=self.pp_has_first_stage,
                    has_last_stage=self.pp_has_last_stage,
                )


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
    build_on_meta = load_initial_weights or cfg.model.initialization_device == "meta"
    initialize_from_meta = build_on_meta and not load_initial_weights
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
    if initialize_from_meta and any(
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
    if initialize_from_meta:
        for method in ("initialize_parameters", "materialization_buffers"):
            if not callable(getattr(model_class, method, None)):
                raise UnsupportedCombinationError(
                    f"Random meta initialization requires model-owned {method}()"
                )
    if build_on_meta:
        with torch.device("meta"):
            model = model_class(text_config)
    else:
        model = model_class(text_config).to(device)
        initialize_buffers = getattr(model, "initialize_training_buffers", None)
        if callable(initialize_buffers):
            initialize_buffers()
    return model, model_config, load_initial_weights


def _apply_parallelism(
    model: torch.nn.Module,
    *,
    cfg: LLMTunerConfig,
    mesh: DeviceMesh | None,
    parallel_dims: ParallelTopology | None,
    device: torch.device,
    load_initial_weights: bool,
    error_group: ProcessGroup | None = None,
    loss_functions: LossFunctions | None = None,
    pipeline_schedule_factory: PipelineScheduleFactory | None = None,
    on_model_acquired: Callable[[torch.nn.Module], None] | None = None,
) -> ModelAssembly:
    """Apply configured stages and return this rank's explicit model ownership."""
    parallel_assembly = parallelize_model(
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
        loss_sum=None if loss_functions is None else loss_functions.token_sum,
        pipeline_schedule_factory=pipeline_schedule_factory,
        on_model_acquired=on_model_acquired,
    )
    orchestration = parallel_assembly.execution
    if isinstance(orchestration, PipelineParallelSetup):
        assembled = ModelAssembly(
            model=None,
            model_parts=orchestration.model_parts,
            layouts=parallel_assembly.layouts,
            pp_schedule=orchestration.schedule,
            pp_has_first_stage=orchestration.has_first_stage,
            pp_has_last_stage=orchestration.has_last_stage,
            # Non-last stages contribute a finite float32 zero to PP loss
            # reduction. Its scalar shape and dtype must match the last stage.
            pp_loss_sentinel=torch.zeros((), dtype=torch.float32, device=device),
        )
    else:
        assembled = ModelAssembly(
            model=orchestration,
            model_parts=[orchestration],
            layouts=parallel_assembly.layouts,
        )

    if load_initial_weights or cfg.model.initialization_device == "meta":
        materialization_error: BaseException | None = None
        try:
            for model_part in assembled.model_parts:
                cpu_offload = cfg.parallel.fsdp_config.fsdp_cpu_offload
                materialize_meta_model(
                    model_part,
                    device,
                    parameter_device=torch.device("cpu") if cpu_offload else None,
                    offloaded_parameters=(
                        fsdp_managed_parameters(model_part) if cpu_offload else None
                    ),
                    runtime_buffers=model_materialization_buffers(model_part, device),
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
    for part, layout in zip(assembled.model_parts, assembled.layouts, strict=True):
        layout.refresh(part)
    return assembled


def prepare_parallel_model(
    cfg: LLMTunerConfig,
    *,
    environment: RuntimeEnvironment,
    loss_functions: LossFunctions | None = None,
    pipeline_schedule_factory: PipelineScheduleFactory | None = None,
    on_model_acquired: Callable[[torch.nn.Module], None] | None = None,
) -> PreparedModel:
    """Construct, count and assemble a model without accessing a Trainer.

    Keep counting before TP/EP/PP/FSDP and preserve the materialization/load
    lifecycle in _apply_parallelism. No optimizer captures parameters until
    this builder has returned the final local layout. ``on_model_acquired``
    transfers cleanup ownership before transformations or materialization can
    fail; it must only retain the module, without device work or collectives.
    """
    with synchronize_local_errors("Model construction", group=environment.error_group):
        model, model_config, load_initial_weights = build_model(
            cfg, parallel_dims=environment.parallel_dims, device=environment.device
        )
        if on_model_acquired is not None:
            on_model_acquired(model)
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
        loss_functions=loss_functions,
        pipeline_schedule_factory=pipeline_schedule_factory,
        on_model_acquired=on_model_acquired,
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
    loss_functions: LossFunctions,
    error_group: ProcessGroup | None = None,
) -> TrainingEngine:
    """Bind execution dependencies without reading or mutating a Trainer.

    The assembly owns model parts and pipeline metadata; optimizer services
    refer to the final parameter layout. This factory consumes both results
    without reselecting model or parallel strategies. ``error_group`` covers
    all update participants, not only ranks sharing a DP or PP mesh axis.
    """
    with synchronize_local_errors("Training engine construction", group=error_group):
        model_assembly.validate_layouts()
        return TrainingEngine(
            model_parts=model_assembly.model_parts,
            execution_layouts=model_assembly.execution_layouts,
            parallel_dims=parallel_dims,
            device=device,
            chunked_loss_num_chunks=cfg.training.chunked_loss_num_chunks,
            param_context=param_context,
            loss_sum=loss_functions.token_sum,
            chunked_loss_backward=loss_functions.chunked_backward,
            optimizer=optimizer_stack.optimizer,
            lr_scheduler=optimizer_stack.lr_scheduler,
            ema=optimizer_stack.ema,
            pp_schedule=model_assembly.pp_schedule,
            pp_has_first_stage=model_assembly.pp_has_first_stage,
            pp_has_last_stage=model_assembly.pp_has_last_stage,
            pp_loss_sentinel=model_assembly.pp_loss_sentinel,
            pp_num_microbatches=cfg.parallel.pp_config.num_pp_microbatches,
            error_group=error_group,
            max_norm=cfg.training.max_norm,
            grad_norm_precision=cfg.training.grad_norm_precision,
            expert_reduce_layout=cfg.parallel.fsdp_config.fsdp_expert_reduce_layout,
        )


def build_optimizer_stack(
    cfg: LLMTunerConfig,
    *,
    model_parts: list[torch.nn.Module],
    parallel_dims: ParallelTopology | None,
    error_group: ProcessGroup | None = None,
    layouts: tuple[ModelPartLayout, ...] = (),
    builders: TrainingBuilders | None = None,
    on_service_acquired: Callable[[str, Any], None],
) -> OptimizerStack:
    """Build optimizer services against the final parallel parameter layout.

    Pass the same error group to parameter planning and scheduler/EMA creation.
    This group coordinates failures; it does not change gradient ownership.
    on_service_acquired takes ownership of the built-in optimizer before inner
    construction, and of other services immediately after their factories return.
    Injected factories own partial construction until they return a service.
    It must only retain the object, without raising or performing device work.
    The caller retains failed releases for retry and closes services in dependency
    order, even when construction fails before this function returns.
    """
    with synchronize_local_errors("Optimizer layout validation", group=error_group):
        if not callable(on_service_acquired):
            raise TypeError("Optimizer services require a callable resource owner")
        if layouts:
            validate_model_part_layouts(model_parts, layouts)
    factories = builders or TrainingBuilders()
    optimizer_factory = _resolve_factory(factories.optimizer, OptimizersContainer)
    scheduler_factory = _resolve_factory(factories.scheduler, build_lr_scheduler)
    ema_factory = _resolve_factory(factories.ema, EMA)
    optimizer = optimizer_factory(
        cfg.optimizer,
        sharded_expert_parallel=cfg.checkpoint.sharded_expert_parallel,
        sharded_tensor_parallel=cfg.checkpoint.sharded_tensor_parallel,
        model_parts=model_parts,
        initialization_group=error_group,
        pipeline_mesh=(
            parallel_dims.get_optional_mesh("pp") if parallel_dims is not None else None
        ),
        runtime_layouts=layouts,
        **(
            {"on_acquired": lambda owner: on_service_acquired("optimizer", owner)}
            if factories.optimizer is None
            else {}
        ),
    )

    if factories.optimizer is not None:
        # Injected factories own partial construction; transfer their returned
        # service here. The built-in container transfers before construction.
        on_service_acquired("optimizer", optimizer)
    with synchronize_local_errors("Optimizer services construction", group=error_group):
        lr_scheduler = scheduler_factory(
            cfg.lr_scheduler_config,
            optimizers=list(optimizer),
            training_steps=cfg.steps,
        )
        on_service_acquired("lr_scheduler", lr_scheduler)
        ema_config = cfg.training.ema
        ema = None
        if ema_config is not None:
            ema = ema_factory(
                model_parts=model_parts,
                runtime_layouts=layouts,
                sharded_expert_parallel=cfg.checkpoint.sharded_expert_parallel,
                sharded_tensor_parallel=cfg.checkpoint.sharded_tensor_parallel,
                decay=ema_config.decay,
                half_life_fraction=ema_config.half_life_fraction,
                start_step=ema_config.start_step,
                step_bias=ema_config.step_bias,
                update_every_n_steps=ema_config.update_every_n_steps,
                buffer_patterns=ema_config.buffer_patterns,
            )
        on_service_acquired("ema", ema)
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
    error_group: ProcessGroup | None = None,
    builders: TrainingBuilders | None = None,
) -> BaseCheckpointManager:
    """Build checkpoint orchestration around explicitly owned runtime state.

    ``error_group`` covers adapter preparation, format-conversion failures and
    external key coverage. Checkpoint manager IO keeps its own collective scope;
    this argument does not repartition persistence.
    """
    with synchronize_local_errors("Checkpoint services preparation", group=error_group):
        model_assembly.validate_layouts()
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
            RNG_STATE: RNGCheckpointState(
                device,
                topology=(
                    model_assembly.layouts[0].topology
                    if model_assembly.layouts
                    else None
                ),
            ),
        }
        sd_adapter = ModelStateDictAdapter(
            model_config,
            cfg.checkpoint.initial_load_path or cfg.hf_model,
            expert_parallel=cfg.parallel.ep_size > 1,
            sharded_expert_parallel=cfg.checkpoint.sharded_expert_parallel,
            error_group=error_group,
            runtime_layouts=model_assembly.layouts,
        )
    manager_cls = (
        TorchCheckpointingManager
        if cfg.checkpoint.backend == "torch_checkpointing"
        else DistributedCheckpointManager
    )
    factories = builders or TrainingBuilders()
    manager_factory = _resolve_factory(factories.checkpoint, manager_cls)
    return manager_factory(
        cfg.checkpoint,
        model_parts=model_assembly.model_parts,
        runtime_layouts=model_assembly.layouts,
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
    builders: TrainingBuilders | None = None,
    on_service_acquired: Callable[[str, Any], None],
) -> MetricsProcessor:
    """Build metrics with explicit construction-failure coordination.

    ``parallel_dims`` still defines metric reductions. ``error_group`` determines
    which peers must accept construction. The caller takes ownership immediately
    after the factory returns, before peer coordination or logging can fail.
    The callback must retain the service without raising or performing device work.
    """
    factories = builders or TrainingBuilders()
    metrics_factory = _resolve_factory(factories.metrics, MetricsProcessor)
    with synchronize_local_errors("Metrics construction", group=error_group):
        if not callable(on_service_acquired):
            raise TypeError("Metrics construction requires a callable resource owner")
        text_config = unwrap_text_config(model_config)
        metrics = metrics_factory(
            cfg.metrics,
            parallel_dims=parallel_dims,
            dump_folder=cfg.dump_folder,
            num_parameters=num_parameters,
            num_trainable_parameters=num_trainable_parameters,
            pp_schedule=cfg.parallel.pp_config.pipeline_parallel_schedule,
            num_flops_per_token=flops_per_token(text_config, seq_len=cfg.max_seq_len),
        )
        on_service_acquired("metrics", metrics)
        note = flops_estimation_note(text_config, seq_len=cfg.max_seq_len)
        if note is not None:
            logging_utils.get_logger(__name__).info("%s", note)
    return metrics


def resolve_model_class(config):
    if config.model_type == "qwen3_moe":
        from experiments.pipeline_parallel.model import Qwen3MoeForCausalLM

        return Qwen3MoeForCausalLM
    return resolve_registered_model_class(config)
