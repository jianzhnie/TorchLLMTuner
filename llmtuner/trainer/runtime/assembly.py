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

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule

from ...accelerator.collectives import synchronize_local_errors
from ...accelerator.device import (
    device_type,
    get_distributed_backend,
    get_env_dist_info,
)
from ...accelerator.distributed import (
    get_default_group,
    get_rank,
    get_world_size,
    init_dist_pytorch,
    is_distributed,
)
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
from ...models.common.moe.experts import RoutedExperts
from ...models.configuration import (
    build_model_config_for,
    resolve_model_class,
    unwrap_text_config,
)
from ...parallel import compatibility
from ...parallel.parallelize import parallelize_model
from ...parallel.pipeline_parallel.pipeline_parallel import PipelineParallelSetup
from ...parallel.topology import build_device_mesh, resolve_parallel_topology
from ...utils import logging as logging_utils
from ...utils import seeding
from ...utils.flops import flops_per_token
from .engine import LossSum, ParamContext, TrainingEngine
from .training_observer import DispatchTimer, TrainingObserver

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...config.llmtuner_config import LLMTunerConfig
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
    # Kimi has an explicit full-expert HF layout adapter. Other EP families
    # still need their own checkpoint mapping before meta initialization.
    if (
        load_initial_weights
        and parallel_dims is not None
        and parallel_dims.ep_enabled
        and getattr(text_config, "model_type", None) != "kimi_linear"
        and not (
            getattr(text_config, "model_type", None) == "qwen3_moe"
            and cfg.checkpoint.sharded_expert_parallel
            and (cfg.parallel.tp_size == 1 or cfg.checkpoint.sharded_tensor_parallel)
        )
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


def _build_model(
    trainer: Trainer,
) -> tuple[torch.nn.Module, PretrainedConfig, bool]:
    """Compatibility bridge for callers using the original Trainer helper."""
    trainer._chunked_loss_num_chunks = trainer.cfg.training.chunked_loss_num_chunks
    return build_model(
        trainer.cfg,
        parallel_dims=trainer.parallel_dims,
        device=getattr(trainer, "device", torch.device("cpu")),
    )


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
) -> ModelAssembly:
    """Apply configured stages and return this rank's explicit model ownership."""
    orchestration = parallelize_model(
        model,
        cfg=cfg.parallel,
        mesh=mesh,
        parallel_dims=parallel_dims,
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
        with synchronize_local_errors("Model materialization"):
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
            with synchronize_local_errors("Random model initialization"):
                if initialization_error is not None:
                    raise initialization_error
    return assembled


def _install_model_assembly(trainer: Trainer, assembled: ModelAssembly) -> None:
    """Compatibility bridge for existing optimizer and checkpoint consumers."""
    trainer.model = assembled.model
    trainer.model_parts = assembled.model_parts
    trainer.pp_schedule = assembled.pp_schedule
    trainer.pp_has_first_stage = assembled.pp_has_first_stage
    trainer.pp_has_last_stage = assembled.pp_has_last_stage
    trainer._pp_loss_sentinel = assembled.pp_loss_sentinel


def create_training_engine(
    cfg: LLMTunerConfig,
    *,
    model_assembly: ModelAssembly,
    optimizer_stack: OptimizerStack,
    parallel_dims: ParallelTopology | None,
    device: torch.device,
    param_context: ParamContext,
    loss_sum: LossSum,
) -> TrainingEngine:
    """Bind execution dependencies without reading or mutating a Trainer.

    The assembly owns model parts and pipeline metadata; optimizer services
    refer to the final parameter layout. This factory consumes both results
    without reselecting model or parallel strategies.
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
    )


def build_training_engine(
    trainer: Trainer,
    assembled: ModelAssembly | None = None,
    optimizer_stack: OptimizerStack | None = None,
) -> TrainingEngine:
    """Bind execution to a model assembly, with a legacy construction fallback."""
    if assembled is None:
        model_parts = getattr(trainer, "model_parts", None)
        if model_parts is None:
            model_parts = [trainer.model]
        assembled = ModelAssembly(
            model=getattr(trainer, "model", None),
            model_parts=model_parts,
            pp_schedule=getattr(trainer, "pp_schedule", None),
            pp_has_first_stage=getattr(trainer, "pp_has_first_stage", False),
            pp_has_last_stage=getattr(trainer, "pp_has_last_stage", False),
            pp_loss_sentinel=getattr(trainer, "_pp_loss_sentinel", None),
        )
    cfg = getattr(trainer, "cfg", None)
    parallel_cfg = getattr(cfg, "parallel", None)
    pp_config = getattr(parallel_cfg, "pp_config", None)
    pp_num_microbatches = getattr(pp_config, "num_pp_microbatches", 1)
    max_context_length = getattr(cfg, "max_seq_len", 1)
    optimizer = (
        optimizer_stack.optimizer
        if optimizer_stack is not None
        else getattr(trainer, "optimizer", None)
    )
    lr_scheduler = (
        optimizer_stack.lr_scheduler
        if optimizer_stack is not None
        else getattr(trainer, "lr_scheduler", None)
    )
    ema = (
        optimizer_stack.ema
        if optimizer_stack is not None
        else getattr(trainer, "ema", None)
    )
    return TrainingEngine(
        model_parts=assembled.model_parts,
        parallel_dims=trainer.parallel_dims,
        device=getattr(trainer, "device", torch.device("cpu")),
        max_context_length=max_context_length,
        chunked_loss_num_chunks=getattr(trainer, "_chunked_loss_num_chunks", 1),
        param_context=trainer.param_context,
        loss_sum=trainer.loss_sum,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema=ema,
        pp_schedule=assembled.pp_schedule,
        pp_has_first_stage=assembled.pp_has_first_stage,
        pp_has_last_stage=assembled.pp_has_last_stage,
        pp_loss_sentinel=assembled.pp_loss_sentinel,
        pp_num_microbatches=pp_num_microbatches,
    )


def build_optimizer_stack(
    cfg: LLMTunerConfig,
    *,
    model_parts: list[torch.nn.Module],
    parallel_dims: ParallelTopology | None,
) -> OptimizerStack:
    """Build optimizer services against the final parallel parameter layout."""
    optimizer = OptimizersContainer(
        cfg.optimizer,
        sharded_expert_parallel=cfg.checkpoint.sharded_expert_parallel,
        sharded_tensor_parallel=cfg.checkpoint.sharded_tensor_parallel,
        model_parts=model_parts,
        initialization_group=get_default_group() if is_distributed() else None,
        pipeline_mesh=(
            parallel_dims.get_optional_mesh("pp") if parallel_dims is not None else None
        ),
    )

    with synchronize_local_errors("Optimizer services construction"):
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


def _build_optimizer_stack(
    trainer: Trainer, assembled: ModelAssembly | None = None
) -> OptimizerStack:
    """Compatibility bridge for existing optimizer construction integrations."""
    model_parts = trainer.model_parts if assembled is None else assembled.model_parts
    stack = build_optimizer_stack(
        trainer.cfg, model_parts=model_parts, parallel_dims=trainer.parallel_dims
    )
    trainer.optimizer = stack.optimizer
    trainer.lr_scheduler = stack.lr_scheduler
    trainer.ema = stack.ema
    return stack


def build_checkpointer(
    cfg: LLMTunerConfig,
    model_config: PretrainedConfig,
    *,
    model_assembly: ModelAssembly,
    optimizer_stack: OptimizerStack,
    dataloader: Any,
    train_state: Any,
    device: torch.device,
    parallel_dims: ParallelTopology | None,
) -> DistributedCheckpointManager | TorchCheckpointingManager:
    """Build checkpoint orchestration around explicitly owned runtime state."""
    with synchronize_local_errors("Checkpoint services preparation"):
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
            qwen3_expert_parallel=(
                getattr(model_config, "model_type", None) == "qwen3_moe"
                and cfg.parallel.ep_size > 1
                and cfg.checkpoint.sharded_expert_parallel
            ),
            kimi_expert_parallel=(
                getattr(model_config, "model_type", None) == "kimi_linear"
                and cfg.parallel.ep_size > 1
            ),
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


def _build_checkpointer(
    trainer: Trainer,
    model_config: PretrainedConfig,
    model_assembly: ModelAssembly | None = None,
    optimizer_stack: OptimizerStack | None = None,
) -> DistributedCheckpointManager | TorchCheckpointingManager:
    """Compatibility bridge retaining Trainer as the train-state owner."""
    if model_assembly is None:
        model_assembly = getattr(trainer, "model_assembly", None)
    if model_assembly is None:
        model = getattr(trainer, "model", None)
        model_assembly = ModelAssembly(
            model=model,
            model_parts=getattr(
                trainer, "model_parts", [] if model is None else [model]
            ),
        )
    if optimizer_stack is None:
        optimizer_stack = getattr(trainer, "optimizer_stack", None)
    if optimizer_stack is None:
        optimizer_stack = OptimizerStack(
            optimizer=getattr(trainer, "optimizer", None),
            lr_scheduler=getattr(trainer, "lr_scheduler", None),
            ema=getattr(trainer, "ema", None),
        )
    trainer.checkpointer = build_checkpointer(
        trainer.cfg,
        model_config,
        model_assembly=model_assembly,
        optimizer_stack=optimizer_stack,
        dataloader=trainer.dataloader,
        train_state=trainer,
        device=getattr(trainer, "device", torch.device("cpu")),
        parallel_dims=getattr(trainer, "parallel_dims", None),
    )
    return trainer.checkpointer


def build_metrics_processor(
    cfg: LLMTunerConfig,
    model_config: PretrainedConfig,
    *,
    parallel_dims: ParallelTopology | None,
    num_parameters: int,
    num_trainable_parameters: int,
) -> MetricsProcessor:
    """Build metrics from resolved model geometry and parallel topology."""
    with synchronize_local_errors("Metrics construction"):
        text_config = unwrap_text_config(model_config)
        metrics = MetricsProcessor(
            cfg.metrics,
            parallel_dims=parallel_dims,
            dump_folder=cfg.dump_folder,
            num_parameters=num_parameters,
            num_trainable_parameters=num_trainable_parameters,
            pp_schedule=cfg.parallel.pp_config.pipeline_parallel_schedule,
            num_flops_per_token=flops_per_token(text_config, seq_len=cfg.max_seq_len),
        )
        if text_config.model_type == "qwen3_5_moe_text":
            logging_utils.get_logger(__name__).info(
                "Qwen3.5 model FLOPs: recurrent-equivalent GDN, gated "
                "projections, active experts and depthwise convolution; "
                "full-square attention at sequence length %d. Excludes "
                "recomputation, padding overhead, optimizer and communication; "
                "this is model MFU, not measured kernel utilization.",
                cfg.max_seq_len,
            )
    return metrics


def assemble_trainer_runtime(trainer: Trainer, cfg: LLMTunerConfig) -> None:
    trainer.cfg = cfg
    if cfg.deterministic:
        seeding.set_deterministic_algorithms()
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

    # Seed before model construction. PP stages use distinct seeds; ranks
    # sharing a stage retain identical initialization.
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

    # PP needs the dense mesh for this stage, which covers only its ranks.
    if trainer.parallel_dims is not None and trainer.parallel_dims.pp_enabled:
        trainer.mesh = trainer.parallel_dims.spmd_dense_mesh()
    else:
        trainer.mesh = build_device_mesh(trainer.parallel_dims)

    seeding.initialize_dtensor_rng(trainer.mesh)

    # Compatibility field for direct dense/PP callers; the engine reads config.
    trainer._chunked_loss_num_chunks = cfg.training.chunked_loss_num_chunks
    # Build the model before optimizer services capture its parameters.
    with synchronize_local_errors("Model construction"):
        model, model_config, load_initial_weights = build_model(
            cfg, parallel_dims=trainer.parallel_dims, device=trainer.device
        )
        # Count unique logical parameters before TP/EP/PP/FSDP change ownership.
        num_parameters = sum(parameter.numel() for parameter in model.parameters())
        num_trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    model_assembly = _apply_parallelism(
        model,
        cfg=cfg,
        mesh=trainer.mesh,
        parallel_dims=trainer.parallel_dims,
        device=trainer.device,
        load_initial_weights=load_initial_weights,
    )
    _install_model_assembly(trainer, model_assembly)
    optimizer_stack = _build_optimizer_stack(trainer, model_assembly)
    trainer.model_assembly = model_assembly
    trainer.optimizer_stack = optimizer_stack
    trainer.training_engine = create_training_engine(
        cfg,
        model_assembly=model_assembly,
        optimizer_stack=optimizer_stack,
        parallel_dims=trainer.parallel_dims,
        device=trainer.device,
        param_context=trainer.param_context,
        loss_sum=trainer.loss_sum,
    )

    # The checkpointer captures the loader cursor alongside model state.
    with synchronize_local_errors("Dataloader construction"):
        # Preserve the public subclass hook. The default implementation uses
        # explicit data dependencies; the loader itself never receives Trainer.
        trainer.dataloader = trainer.build_dataloader()

    # Counters the checkpoint carries. Kept as plain ints so a resumed run
    # can log "step 61 (resumed at 60)" without re-deriving them.
    trainer.step = 0
    trainer.ntokens_seen = 0

    _build_checkpointer(trainer, model_config)

    trainer.metrics = build_metrics_processor(
        cfg,
        model_config,
        parallel_dims=trainer.parallel_dims,
        num_parameters=num_parameters,
        num_trainable_parameters=num_trainable_parameters,
    )
    trainer.training_observer = TrainingObserver(trainer.metrics, trainer.device)
    trainer.dispatch_timers = []
    if cfg.metrics.measure_moe_dispatch:
        for part in trainer.model_parts:
            for module in part.modules():
                # Dedicated fused forward implementations do not use this
                # dispatch boundary and must not report a vacuous zero timer.
                if type(module) is RoutedExperts:
                    timer = DispatchTimer()
                    module.dispatch_observer = timer
                    trainer.dispatch_timers.append(timer)
        if not trainer.dispatch_timers:
            logging_utils.get_logger(__name__).warning(
                "Dispatch timing requested, but no native RoutedExperts boundary "
                "was found; dispatch timing is unavailable for this model."
            )
    if cfg.activation_swap:
        reset_activation_swap_metrics()
    # The trainer synchronizes PP loss metrics to every stage before logging.
    logging_utils.set_log_ranks(cfg.metrics.log_ranks)
    if trainer.device.type == "npu":
        logging_utils.get_logger(__name__).info(
            "NPU step guards use explicit synchronous scalar reads; "
            "loss/gradient and valid-token checks remain enabled"
        )
