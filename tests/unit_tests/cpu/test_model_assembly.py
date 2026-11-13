from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmtuner.components.loss import cross_entropy_loss
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.parallel.pipeline_parallel.pipeline_parallel import PipelineParallelSetup
from llmtuner.trainer.runtime import assembly
from llmtuner.trainer.runtime import batch as batch_runtime
from llmtuner.trainer.trainer import Trainer


def apply_parallelism(model, cfg, monkeypatch, orchestration):
    monkeypatch.setattr(
        assembly, "parallelize_model", lambda *args, **kwargs: orchestration
    )
    return assembly._apply_parallelism(
        model,
        cfg=cfg,
        mesh=None,
        parallel_dims=None,
        device=torch.device("cpu"),
        load_initial_weights=False,
    )


def test_dense_parallel_assembly_returns_model_without_trainer(monkeypatch) -> None:
    model = nn.Linear(2, 2)

    result = apply_parallelism(model, LLMTunerConfig(), monkeypatch, model)

    assert result.model is model
    assert result.model_parts == [model]
    assert result.pp_schedule is None
    assert result.pp_loss_sentinel is None


def test_pipeline_parallel_assembly_returns_complete_execution_metadata(
    monkeypatch,
) -> None:
    model = nn.Linear(2, 2)
    schedule = object()
    setup = PipelineParallelSetup(
        schedule=schedule,
        stages=[],
        model_parts=[model],
        has_first_stage=True,
        has_last_stage=False,
    )

    result = apply_parallelism(model, LLMTunerConfig(), monkeypatch, setup)

    assert result.model is None
    assert result.model_parts == [model]
    assert result.pp_schedule is schedule
    assert result.pp_has_first_stage
    assert not result.pp_has_last_stage
    assert result.pp_loss_sentinel.shape == ()
    assert result.pp_loss_sentinel.dtype == torch.float32
    assert result.pp_loss_sentinel.device.type == "cpu"


def test_meta_assembly_materializes_before_model_initialization(monkeypatch) -> None:
    events = []

    class MetaModel(nn.Module):
        def initialize_parameters(self):
            events.append("initialize")

    model = MetaModel()
    cfg = LLMTunerConfig()
    cfg.model.initialization_device = "meta"
    monkeypatch.setattr(
        assembly,
        "materialize_meta_model",
        lambda *args, **kwargs: events.append("materialize"),
    )

    result = apply_parallelism(model, cfg, monkeypatch, model)

    assert result.model is model
    assert events == ["materialize", "initialize"]


def test_engine_consumes_model_assembly_without_legacy_pp_attributes() -> None:
    model = nn.Linear(2, 2)
    schedule = object()
    error_group = object()
    model_assembly = assembly.ModelAssembly(
        model=None,
        model_parts=[model],
        pp_schedule=schedule,
        pp_has_first_stage=True,
        pp_has_last_stage=True,
    )
    cfg = LLMTunerConfig()
    trainer = SimpleNamespace(
        cfg=cfg,
        parallel_dims=None,
        device=torch.device("cpu"),
        param_context=lambda: None,
        loss_sum=lambda logits, labels, **kwargs: logits.sum(),
    )

    engine = assembly.create_training_engine(
        cfg,
        model_assembly=model_assembly,
        optimizer_stack=assembly.OptimizerStack(None, None, None),
        parallel_dims=trainer.parallel_dims,
        device=trainer.device,
        param_context=trainer.param_context,
        loss_sum=trainer.loss_sum,
        error_group=error_group,
    )

    assert engine.model_parts == [model]
    assert engine.pp_schedule is schedule
    assert engine.pp_has_first_stage
    assert engine.pp_has_last_stage
    assert engine.error_group is error_group


def test_optimizer_stack_and_engine_build_without_trainer_attributes() -> None:
    model = nn.Linear(2, 2)
    cfg = LLMTunerConfig()

    stack = assembly.build_optimizer_stack(cfg, model_parts=[model], parallel_dims=None)
    trainer = SimpleNamespace(
        cfg=cfg,
        parallel_dims=None,
        device=torch.device("cpu"),
        param_context=lambda: None,
        loss_sum=lambda logits, labels, **kwargs: logits.sum(),
    )
    engine = assembly.create_training_engine(
        cfg,
        model_assembly=assembly.ModelAssembly(model=model, model_parts=[model]),
        optimizer_stack=stack,
        parallel_dims=trainer.parallel_dims,
        device=trainer.device,
        param_context=trainer.param_context,
        loss_sum=trainer.loss_sum,
    )

    assert stack.optimizer.model_parts == [model]
    assert len(stack.optimizer) == 1
    assert stack.lr_scheduler is not None
    assert stack.ema is None
    assert engine.optimizer is stack.optimizer
    assert engine.lr_scheduler is stack.lr_scheduler
    assert engine.ema is None


def test_checkpointer_consumes_explicit_runtime_owners(monkeypatch) -> None:
    captured = {}
    error_group = object()
    phases = []

    @contextmanager
    def coordinate(phase, *, group):
        phases.append((phase, group))
        yield

    monkeypatch.setattr(assembly, "synchronize_local_errors", coordinate)

    class Manager:
        def __init__(self, config, **kwargs):
            captured["config"] = config
            captured.update(kwargs)

    monkeypatch.setattr(assembly, "DistributedCheckpointManager", Manager)
    model = nn.Linear(2, 2)
    model_assembly = assembly.ModelAssembly(model=model, model_parts=[model])
    optimizer_stack = assembly.OptimizerStack(
        optimizer=object(), lr_scheduler=object(), ema=None
    )
    train_state = object()
    dataloader = object()
    cfg = LLMTunerConfig()

    manager = assembly.build_checkpointer(
        cfg,
        SimpleNamespace(model_type="qwen3_moe"),
        model_assembly=model_assembly,
        optimizer_stack=optimizer_stack,
        dataloader=dataloader,
        train_state=train_state,
        device=torch.device("cpu"),
        parallel_dims=None,
        error_group=error_group,
    )

    assert phases == [("Checkpoint services preparation", error_group)]
    assert isinstance(manager, Manager)
    assert captured["model_parts"] == [model]
    assert captured["optimizer"] is optimizer_stack.optimizer
    assert captured["lr_scheduler"] is optimizer_stack.lr_scheduler
    assert captured["ema"] is None
    assert captured["states"]["train_state"] is train_state
    assert captured["states"]["dataloader"] is dataloader


def test_engine_factory_uses_explicit_dependencies_without_changing_rng() -> None:
    model = nn.Linear(2, 2)
    cfg = LLMTunerConfig()
    cfg.training.chunked_loss_num_chunks = 3
    cfg.parallel.pp_config.num_pp_microbatches = 2
    schedule = object()
    sentinel = torch.zeros(())
    assembled = assembly.ModelAssembly(
        model=None,
        model_parts=[model],
        pp_schedule=schedule,
        pp_has_last_stage=True,
        pp_loss_sentinel=sentinel,
    )
    stack = assembly.OptimizerStack(object(), object(), None)
    rng = torch.get_rng_state().clone()
    weight = model.weight

    engine = assembly.create_training_engine(
        cfg,
        model_assembly=assembled,
        optimizer_stack=stack,
        parallel_dims=None,
        device=torch.device("cpu"),
        param_context=nullcontext,
        loss_sum=cross_entropy_loss,
    )

    assert engine.model_parts is assembled.model_parts
    assert engine.optimizer is stack.optimizer
    assert engine.lr_scheduler is stack.lr_scheduler
    assert engine.loss_sum is cross_entropy_loss
    assert engine.param_context is nullcontext
    assert engine.pp_schedule is schedule
    assert engine.pp_loss_sentinel is sentinel
    assert engine.pp_has_last_stage
    assert engine.pp_num_microbatches == 2
    assert engine.chunked_loss_num_chunks == 3
    assert engine.max_context_length == cfg.max_seq_len
    assert model.weight is weight
    assert torch.equal(torch.get_rng_state(), rng)


def test_registered_model_builds_without_trainer_or_process_group() -> None:
    cfg = LLMTunerConfig()
    cfg.model.arch_overrides = {
        "vocab_size": 19,
        "hidden_size": 16,
        "intermediate_size": 24,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
    }
    cfg.model.initialization_device = "cpu"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        expected_config = assembly.build_model_config_for(cfg)
        expected = assembly.resolve_model_class(expected_config)(expected_config)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(137)
        model, config, load_weights = assembly.build_model(
            cfg, parallel_dims=None, device=torch.device("cpu")
        )
        actual_rng = torch.get_rng_state().clone()

    assert not torch.distributed.is_initialized()
    assert not load_weights
    assert config.vocab_size == cfg.vocab_size == 19
    assert torch.equal(expected_rng, actual_rng)
    assert tuple(model.state_dict()) == tuple(expected.state_dict())
    assert sum(p.numel() for p in model.parameters()) > 0
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(
            parameter, expected.state_dict()[name], rtol=0, atol=0
        )
    inputs = torch.tensor([[1, 3, 8, 2]])
    output = model(inputs).logits
    assert output.shape == (1, 4, 19)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.abs().sum() > 0


def test_metrics_factory_uses_explicit_geometry_and_topology(monkeypatch) -> None:
    captured = {}

    class Metrics:
        def __init__(self, config, **kwargs):
            captured["config"] = config
            captured.update(kwargs)

    monkeypatch.setattr(assembly, "MetricsProcessor", Metrics)
    monkeypatch.setattr(assembly, "unwrap_text_config", lambda config: config)
    monkeypatch.setattr(
        assembly,
        "flops_per_token",
        lambda config, *, seq_len: 123 if seq_len == 8 else None,
    )
    cfg = LLMTunerConfig()
    cfg.training.max_seq_len = 8
    model_config = SimpleNamespace(model_type="qwen3_moe")

    metrics = assembly.build_metrics_processor(
        cfg,
        model_config,
        parallel_dims=None,
        num_parameters=101,
        num_trainable_parameters=89,
    )

    assert isinstance(metrics, Metrics)
    assert captured["config"] is cfg.metrics
    assert captured["parallel_dims"] is None
    assert captured["num_parameters"] == 101
    assert captured["num_trainable_parameters"] == 89
    assert captured["pp_schedule"] == cfg.parallel.pp_config.pipeline_parallel_schedule
    assert captured["num_flops_per_token"] == 123


def test_dataloader_factory_uses_explicit_rank_and_token_budget(monkeypatch) -> None:
    captured = {}
    loader = object()

    def build(cfg, **kwargs):
        captured.update(kwargs)
        return loader

    monkeypatch.setattr(batch_runtime, "build_dataset_dataloader", build)
    cfg = LLMTunerConfig()
    cfg.training.global_batch_size = 8
    cfg.training.max_seq_len = 16

    assert batch_runtime.build_training_dataloader(cfg, parallel_dims=None) is loader
    assert captured == {
        "dp_rank": 0,
        "dp_world_size": 1,
        "num_tokens_per_batch": 128,
    }

    mesh = SimpleNamespace(get_local_rank=lambda: 1, size=lambda: 2)
    axes = []
    topology = SimpleNamespace(
        pp_enabled=True,
        get_optional_mesh=lambda name, **kwargs: axes.append((name, kwargs)) or mesh,
    )
    assert batch_runtime.resolve_data_parallel_rank(topology) == (1, 2)
    assert axes == [("batch", {"include_singleton_axes": True})]
    with pytest.raises(ValueError, match="must be divisible"):
        batch_runtime.resolve_batch_size_per_rank(cfg, 3)


def test_production_assembly_uses_explicit_dataloader_factory(monkeypatch):
    calls = []
    original = assembly.build_training_dataloader

    def tracked(cfg, *, parallel_dims):
        calls.append((cfg, parallel_dims))
        return original(cfg, parallel_dims=parallel_dims)

    monkeypatch.setattr(assembly, "build_training_dataloader", tracked)
    cfg = LLMTunerConfig()
    cfg.model.arch_overrides.update(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    trainer = Trainer(cfg)
    try:
        assert calls == [(cfg, None)]
        batch = next(trainer.data_iterator())
        assert batch.input_ids.numel() > 0
        assert int(batch.input_ids.max()) < 19
    finally:
        trainer.close()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_metrics_factory_closes_unpublished_logger_after_peer_failure(
    monkeypatch, error_type, cleanup_fails
):
    error_group = object()
    failure = error_type("peer metrics construction failed")
    closed = []

    @contextmanager
    def coordinated_failure(phase, *, group=None):
        assert group is error_group
        assert phase == "Metrics construction"
        yield
        raise failure

    def close():
        closed.append(True)
        if cleanup_fails:
            raise ValueError("logger close failed")

    metrics = SimpleNamespace(close=close)
    monkeypatch.setattr(assembly, "synchronize_local_errors", coordinated_failure)
    monkeypatch.setattr(assembly, "MetricsProcessor", lambda *a, **kw: metrics)
    cfg = LLMTunerConfig()
    model_config = assembly.build_model_config_for(cfg)

    with pytest.raises(error_type) as caught:
        assembly.build_metrics_processor(
            cfg,
            model_config,
            parallel_dims=None,
            num_parameters=17,
            num_trainable_parameters=13,
            error_group=error_group,
        )

    assert caught.value is failure
    assert closed == [True]


def test_optimizer_factory_passes_explicit_coordination_group(monkeypatch):
    group = object()
    phases = []
    original = assembly.OptimizersContainer

    def optimizer_factory(*args, **kwargs):
        assert kwargs.pop("initialization_group") is group
        return original(*args, **kwargs)

    @contextmanager
    def coordinate(phase, *, group):
        phases.append((phase, group))
        yield

    monkeypatch.setattr(assembly, "OptimizersContainer", optimizer_factory)
    monkeypatch.setattr(assembly, "synchronize_local_errors", coordinate)
    stack = assembly.build_optimizer_stack(
        LLMTunerConfig(),
        model_parts=[nn.Linear(2, 2)],
        parallel_dims=None,
        error_group=group,
    )
    assert stack.optimizer is not None
    assert phases == [("Optimizer services construction", group)]


def test_model_preparation_counts_before_parameter_layout_changes(monkeypatch):
    cfg = LLMTunerConfig()
    environment = assembly.RuntimeEnvironment(
        rank=0,
        world_size=1,
        local_rank=0,
        parallel_dims=None,
        device=torch.device("cpu"),
        mesh=None,
        error_group=None,
    )
    model = nn.Module()
    model.trainable = nn.Parameter(torch.ones(3))
    model.alias = model.trainable
    model.frozen = nn.Parameter(torch.ones(5), requires_grad=False)
    model_config = object()
    events = []

    def construct(config, *, parallel_dims, device):
        assert config is cfg
        assert parallel_dims is environment.parallel_dims
        assert device is environment.device
        events.append("construct")
        return model, model_config, True

    def assemble(source, **kwargs):
        assert source is model
        assert kwargs["load_initial_weights"] is True
        assert kwargs["error_group"] is environment.error_group
        events.append("assemble")
        del source.trainable
        del source.alias
        return assembly.ModelAssembly(model=source, model_parts=[source])

    monkeypatch.setattr(assembly, "build_model", construct)
    monkeypatch.setattr(assembly, "_apply_parallelism", assemble)
    rng = torch.get_rng_state().clone()
    prepared = assembly.prepare_parallel_model(cfg, environment=environment)
    assert events == ["construct", "assemble"]
    assert prepared.config is model_config
    assert prepared.assembly.model is model
    assert prepared.num_parameters == 8
    assert prepared.num_trainable_parameters == 3
    assert sum(parameter.numel() for parameter in model.parameters()) == 5
    assert torch.equal(torch.get_rng_state(), rng)
