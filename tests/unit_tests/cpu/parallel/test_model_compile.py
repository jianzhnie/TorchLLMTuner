"""Compile defaults, async TP and capture_scalar_outputs.

These tests pin, on CPU with no process group:

* the default: whole-model compile, dense models never touch
  ``capture_scalar_outputs``, and ``CompileConfig()`` reproduces the old
  plain ``torch.compile(model)`` path;
* per-block compile wraps every decoder layer and not the model;
* the config-time loud errors: async TP without compile, async TP at tp=1;
* the assembly-time loud errors: async TP without a TP mesh or the required
  Inductor micro-pipeline capability;
* the MoE condition for ``capture_scalar_outputs`` (set when the model part
  carries token-choice MoE blocks, unset for a dense one), with the global
  restored afterwards.

Model-level tests construct explicit Qwen3 models without the removed HF
adapter. CPU aot_eager graphs also check AC composition, mixed train/eval modes,
parameter identity and frozen gradients against an eager reference.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from torch._dynamo import OptimizedModule
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as ReferenceQwen3MoeForCausalLM,
)

from llmtuner.config import (
    CompileConfig,
    LLMTunerConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel import model_compile as compile_mod
from llmtuner.parallel.model_compile import apply_compile
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config

try:
    ParallelConfig()
    _PARALLEL_CONFIG_ERROR = None
except Exception as e:  # noqa: BLE001 - env gate (torch 2.2 has no pipelining)
    _PARALLEL_CONFIG_ERROR = e

requires_parallel_config = pytest.mark.skipif(
    _PARALLEL_CONFIG_ERROR is not None,
    reason=f"this torch cannot build a ParallelConfig: {_PARALLEL_CONFIG_ERROR}",
)

_VOCAB = 32
_SEQ = 24
_NUM_LAYERS = 3


@pytest.fixture(autouse=True)
def cpu_compile_device():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compile_mod, "device_type", "cpu")
        yield


@pytest.mark.parametrize("selected", [None, "invalid"])
def test_npu_inductor_requires_launch_backend(monkeypatch, selected):
    from llmtuner.errors import EnvironmentUnsupportedError

    monkeypatch.setattr(compile_mod, "device_type", "npu")
    monkeypatch.delenv("TORCHINDUCTOR_NPU_BACKEND", raising=False)
    if selected is not None:
        monkeypatch.setenv("TORCHINDUCTOR_NPU_BACKEND", selected)
    with pytest.raises(EnvironmentUnsupportedError, match="before importing torch"):
        compile_mod.validate_compile_backend("inductor")


@pytest.mark.parametrize("available", [False, True])
def test_npu_inductor_checks_codegen_capability(monkeypatch, available):
    from llmtuner.errors import EnvironmentUnsupportedError

    monkeypatch.setattr(compile_mod, "device_type", "npu")
    monkeypatch.setenv("TORCHINDUCTOR_NPU_BACKEND", "ascendc")
    probes = []

    def probe(name):
        probes.append(name)
        return available

    monkeypatch.setattr(compile_mod, "has", probe)
    if available:
        compile_mod.validate_compile_backend("inductor")
    else:
        with pytest.raises(EnvironmentUnsupportedError, match="matching CANN"):
            compile_mod.validate_compile_backend("inductor")
    assert probes == ["npu_ascendc_codegen"]


def _model(seed: int = 0):
    torch.manual_seed(seed)
    config = model_config(
        num_hidden_layers=_NUM_LAYERS,
        mlp_only_layers=list(range(_NUM_LAYERS)),
        intermediate_size=32,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return Qwen3MoeForCausalLM(config)


@contextlib.contextmanager
def _preserve_dynamo_flag(name: str):
    old = getattr(torch._dynamo.config, name)
    try:
        yield
    finally:
        setattr(torch._dynamo.config, name, old)


# -- config-time validation ---------------------------------------------------


def test_compile_config_defaults_reproduce_whole_model_compile() -> None:
    cfg = CompileConfig()
    assert cfg.per_block is False
    assert cfg.backend == "inductor"
    assert cfg.enable_async_tensor_parallel is False


@pytest.mark.parametrize("backend", ["", "  ", None, True, 1])
def test_compile_config_rejects_empty_backend(backend) -> None:
    with pytest.raises(ValueError, match="backend"):
        CompileConfig(backend=backend)


@requires_parallel_config
def test_async_tp_requires_compile() -> None:
    with pytest.raises(ValueError, match="requires training.compile=True"):
        LLMTunerConfig(
            parallel=ParallelConfig(tensor_parallel_size=2),
            training=TrainingConfig(
                compile=False,
                compile_config=CompileConfig(enable_async_tensor_parallel=True),
            ),
        )


@requires_parallel_config
def test_async_tp_requires_tp() -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size > 1"):
        LLMTunerConfig(
            parallel=ParallelConfig(tensor_parallel_size=1),
            training=TrainingConfig(
                compile=True,
                compile_config=CompileConfig(enable_async_tensor_parallel=True),
            ),
        )


@requires_parallel_config
def test_async_tp_valid_combination_passes() -> None:
    LLMTunerConfig(
        parallel=ParallelConfig(tensor_parallel_size=2),
        training=TrainingConfig(
            compile=True,
            compile_config=CompileConfig(enable_async_tensor_parallel=True),
        ),
    )


# -- the compile step itself ----------------------------------------------------


def test_default_compile_is_whole_model_and_leaves_dynamo_flags() -> None:
    """compile_config defaults: whole-model compile, dense flags untouched."""
    with _preserve_dynamo_flag("capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = False
        model = apply_compile(_model())
        assert isinstance(model, OptimizedModule)
        assert not any(isinstance(layer, OptimizedModule) for layer in model.layers)
        # Dense model: the MoE-only flag is untouched.
        assert torch._dynamo.config.capture_scalar_outputs is False


def test_per_block_compile_wraps_each_layer_not_the_model() -> None:
    model = apply_compile(_model(), compile_config=CompileConfig(per_block=True))
    assert not isinstance(model, OptimizedModule)
    # Module.compile is in place: same objects, compiled call impl attached.
    assert all(layer._compiled_call_impl is not None for layer in model.layers)


def test_async_tp_without_tp_mesh_raises() -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size > 1"):
        apply_compile(
            _model(),
            compile_config=CompileConfig(enable_async_tensor_parallel=True),
            tp_mesh=None,
        )


def test_per_block_compile_rejects_missing_layers_before_global_changes(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid per-block model must fail before global setup")

    monkeypatch.setattr(compile_mod, "maybe_enable_async_tp", unexpected)
    with pytest.raises(TypeError, match=".layers module"):
        apply_compile(
            torch.nn.Linear(2, 2), compile_config=CompileConfig(per_block=True)
        )


@pytest.mark.parametrize("ndim,size", [(1, 1), (2, 4)])
def test_async_tp_rejects_invalid_mesh_before_registration(monkeypatch, ndim, size):
    from types import SimpleNamespace

    monkeypatch.setattr(compile_mod, "has", lambda capability: True)

    def unexpected():
        pytest.fail("invalid mesh must fail before process-group registration")

    mesh = SimpleNamespace(ndim=ndim, size=lambda: size, get_group=unexpected)
    with pytest.raises(ValueError, match="one-dimensional TP mesh with size > 1"):
        compile_mod.maybe_enable_async_tp(
            CompileConfig(enable_async_tensor_parallel=True), mesh
        )


def test_async_tp_on_unsupported_torch_loud_raises(monkeypatch) -> None:
    monkeypatch.setattr(compile_mod, "has", lambda capability: False)
    with pytest.raises(NotImplementedError, match="_micro_pipeline_tp"):
        apply_compile(
            _model(),
            compile_config=CompileConfig(enable_async_tensor_parallel=True),
            tp_mesh=object(),  # never reached: the capability check fires first
        )


def test_capture_scalar_outputs_set_only_for_token_choice_moe(monkeypatch) -> None:
    """The flag follows the MoE-dispatch condition, and only that condition.

    Stub the dispatch detector to isolate this global compile policy from
    expert conversion; real MoE training graphs are covered separately.
    """
    with _preserve_dynamo_flag("capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = False
        monkeypatch.setattr(compile_mod, "collect_moe_layers", lambda m: [object()])
        apply_compile(_model(), compile_config=CompileConfig())
        assert torch._dynamo.config.capture_scalar_outputs is True

    monkeypatch.undo()  # restore the real collect_moe_layers for the dense case

    with _preserve_dynamo_flag("capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = False
        apply_compile(_model(), compile_config=CompileConfig())
        assert torch._dynamo.config.capture_scalar_outputs is False


@pytest.mark.parametrize("per_block", [False, True])
def test_compiled_cpu_training_matches_eager_outputs_and_gradients(per_block):
    reference = _model()
    model = apply_compile(
        _model(), compile_config=CompileConfig(backend="aot_eager", per_block=per_block)
    )
    tokens = torch.arange(_SEQ) % _VOCAB
    expected = reference(tokens).logits
    actual = model(tokens).logits
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    reference_gradients = {
        name: parameter.grad for name, parameter in reference.named_parameters()
    }
    for name, parameter in model.named_parameters():
        canonical_name = name.removeprefix("_orig_mod.")
        torch.testing.assert_close(parameter.grad, reference_gradients[canonical_name])


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_async_tp_requires_inductor_backend(backend):
    from llmtuner.errors import ConfigError

    with pytest.raises(ConfigError, match="inductor"):
        CompileConfig(backend=backend, enable_async_tensor_parallel=True)


def test_missing_moe_compile_capability_fails_before_setup(monkeypatch):
    from llmtuner.errors import EnvironmentUnsupportedError

    model = torch.nn.Linear(4, 4)
    monkeypatch.setattr(compile_mod, "collect_moe_layers", lambda model: [object()])
    monkeypatch.setattr(compile_mod, "has", lambda name: False)
    with pytest.raises(EnvironmentUnsupportedError, match="capture_scalar_outputs"):
        apply_compile(model, compile_config=CompileConfig(backend="eager"))


def test_compiled_qwen_training_matches_native_hf_trajectory():
    """Use native HF forward as an oracle independent of the explicit model."""
    import copy

    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    torch.manual_seed(42)
    candidate = _model(seed=42)
    reference = ReferenceQwen3MoeForCausalLM(copy.deepcopy(candidate.config))
    reference.load_state_dict(candidate.state_dict(), strict=True)
    model = apply_compile(candidate, compile_config=CompileConfig(backend="aot_eager"))
    actual_optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0, foreach=False
    )
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=3e-4, weight_decay=0, foreach=False
    )
    generator = torch.Generator().manual_seed(1234)
    for _ in range(4):
        tokens = torch.randint(_VOCAB, (_SEQ,), generator=generator)
        targets = torch.randint(_VOCAB, (_SEQ,), generator=generator)
        actual_optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual = model(tokens).logits
        expected = reference(tokens.unsqueeze(0), use_cache=False).logits.squeeze(0)
        torch.testing.assert_close(actual, expected)
        actual_loss = torch.nn.functional.cross_entropy(actual, targets)
        expected_loss = torch.nn.functional.cross_entropy(expected, targets)
        torch.testing.assert_close(actual_loss, expected_loss)
        actual_loss.backward()
        expected_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        expected_parameters = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            torch.testing.assert_close(parameter.grad, expected_parameters[name].grad)
        actual_optimizer.step()
        reference_optimizer.step()
        for name, parameter in candidate.named_parameters():
            torch.testing.assert_close(parameter, expected_parameters[name])
    assert counters["stats"]["unique_graphs"] > 0


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_compile_setup_failure_restores_scalar_capture(
    monkeypatch, initial, error_type
):
    with _preserve_dynamo_flag("capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = initial
        monkeypatch.setattr(compile_mod, "collect_moe_layers", lambda model: [object()])

        def fail(*args, **kwargs):
            assert torch._dynamo.config.capture_scalar_outputs
            raise error_type("compile setup failed")

        monkeypatch.setattr(torch, "compile", fail)
        with pytest.raises(error_type, match="compile setup failed"):
            apply_compile(_model(), compile_config=CompileConfig())
        assert torch._dynamo.config.capture_scalar_outputs is initial


@pytest.mark.parametrize("failure_index", [0, 1, 2])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_block_compile_failure_restores_original_calls(
    monkeypatch, failure_index, error_type
):
    model = _model()
    blocks = list(model.layers)
    parameters = dict(model.named_parameters())
    state_keys = set(model.state_dict())

    # Include a previously compiled layer: rollback must restore its callable,
    # rather than indiscriminately clearing all compiled implementations.
    def original_impl(*args, **kwargs):
        return None

    blocks[0]._compiled_call_impl = original_impl
    previous = [block._compiled_call_impl for block in blocks]
    attempted = []

    def compile_block(block, **kwargs):
        attempted.append(block)
        block._compiled_call_impl = lambda *args, **kwargs: None
        if block is blocks[failure_index]:
            raise error_type("block setup failed")

    monkeypatch.setattr(type(blocks[0]), "compile", compile_block)
    with pytest.raises(error_type, match="block setup failed"):
        apply_compile(model, compile_config=CompileConfig(per_block=True))
    assert len(attempted) == failure_index + 1
    assert all(a is b for a, b in zip(model.layers, blocks, strict=True))
    assert all(
        block._compiled_call_impl is old
        for block, old in zip(blocks, previous, strict=True)
    )
    assert set(model.state_dict()) == state_keys
    assert all(
        dict(model.named_parameters())[key] is p for key, p in parameters.items()
    )


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("failure_stage", ["registration", "compile"])
def test_failed_async_compile_restores_inductor_pass(
    monkeypatch, initial, failure_stage
):
    import torch._inductor.config as inductor_config

    original = inductor_config._micro_pipeline_tp
    model = torch.nn.Linear(4, 4)

    def setup(*args):
        inductor_config._micro_pipeline_tp = True
        if failure_stage == "registration":
            raise RuntimeError("async setup failed")

    def fail(*args, **kwargs):
        assert inductor_config._micro_pipeline_tp is True
        raise RuntimeError("async compile failed")

    monkeypatch.setattr(compile_mod, "maybe_enable_async_tp", setup)
    monkeypatch.setattr(torch, "compile", fail)
    try:
        inductor_config._micro_pipeline_tp = initial
        with pytest.raises(RuntimeError, match="async"):
            apply_compile(
                model, compile_config=CompileConfig(enable_async_tensor_parallel=True)
            )
        assert inductor_config._micro_pipeline_tp is initial
    finally:
        inductor_config._micro_pipeline_tp = original


@pytest.mark.parametrize("per_block", [False, True])
@pytest.mark.parametrize("mode", ["full", "selective"])
@pytest.mark.parametrize("validation_fails", [False, True])
def test_compiled_ac_preserves_modes_identity_and_frozen_gradients(
    monkeypatch, per_block, mode, validation_fails
):
    import copy
    from types import SimpleNamespace

    from torch._dynamo.utils import counters

    from llmtuner.config.activation_checkpoint_config import SelectiveACConfig
    from llmtuner.memory.activation_checkpoint import apply_ac
    from llmtuner.utils.module_names import canonical_fqn

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                *[
                    torch.nn.Sequential(
                        torch.nn.Linear(4, 4),
                        torch.nn.Dropout(0.25),
                        torch.nn.Tanh(),
                        torch.nn.Linear(4, 4),
                    )
                    for _ in range(2)
                ]
            )

        def forward(self, inputs):
            return self.layers(inputs)

    torch.manual_seed(188)
    reference = Model().eval()
    reference.layers[0][1].train()
    reference.layers[0][0].weight.requires_grad_(False)
    original = copy.deepcopy(reference)
    parameters = dict(original.named_parameters())
    modes = {name: child.training for name, child in original.named_modules()}
    actual = apply_ac(
        original, mode, selective=SelectiveACConfig() if mode == "selective" else None
    )
    torch._dynamo.reset()
    counters.clear()
    try:
        actual = apply_compile(
            actual,
            compile_config=CompileConfig(backend="aot_eager", per_block=per_block),
        )
        assert actual.training is False
        actual_parameters = {
            canonical_fqn(name): parameter
            for name, parameter in actual.named_parameters()
        }
        assert actual_parameters.keys() == parameters.keys()
        assert all(
            actual_parameters[name] is value for name, value in parameters.items()
        )
        for name, child in actual.named_modules():
            canonical = canonical_fqn(name)
            if canonical in modes:
                assert child.training == modes[canonical]
        # Exercise the production mode/RNG guard around a real compiled eval
        # forward, then compare the subsequent training backward below.
        from llmtuner.config.validation_config import ValidationConfig
        from llmtuner.trainer.runtime import validation

        validation_reference = copy.deepcopy(reference).eval()
        validation_inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
        saved_modes = {module: module.training for module in actual.modules()}
        rng = torch.get_rng_state().clone()
        failure = ValueError("compiled validation failed")

        def run_validation_loop(trainer, config, step):
            assert not torch.is_grad_enabled()
            assert all(not module.training for module in actual.modules())
            torch.testing.assert_close(
                actual(validation_inputs),
                validation_reference(validation_inputs),
                rtol=0,
                atol=0,
            )
            torch.rand(7)
            if validation_fails:
                raise failure

        monkeypatch.setattr(validation, "run_validation_loop", run_validation_loop)
        trainer = SimpleNamespace(
            cfg=SimpleNamespace(validation=ValidationConfig(steps=1)),
            checkpointer=None,
            model_parts=[actual],
            device=torch.device("cpu"),
        )
        if validation_fails:
            with pytest.raises(ValueError) as caught:
                validation.validate(trainer, 1)
            assert caught.value is failure
        else:
            validation.validate(trainer, 1)
        assert {module: module.training for module in actual.modules()} == saved_modes
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        inputs = torch.randn(3, 4)
        expected_input = inputs.clone().requires_grad_()
        actual_input = inputs.clone().requires_grad_()
        torch.manual_seed(1880)
        expected = reference(expected_input)
        torch.manual_seed(1880)
        output = actual(actual_input)
        assert counters["stats"]["unique_graphs"] > 0
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        expected.square().sum().backward()
        output.square().sum().backward()
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=0, atol=0
        )
        for name, value in reference.named_parameters():
            parameter = actual_parameters[name]
            assert parameter.requires_grad is value.requires_grad
            if value.requires_grad:
                torch.testing.assert_close(parameter.grad, value.grad, rtol=0, atol=0)
            else:
                assert parameter.grad is None
    finally:
        torch._dynamo.reset()


def test_compile_sdc_accepts_finite_outputs_and_rejects_nan():
    module = torch.nn.Identity()
    compile_mod.wrap_sdc_forward(module, "test block")
    value = torch.ones(2, 3)
    assert module(value) is value

    module.forward = lambda value: value
    compile_mod.wrap_sdc_forward(module, "fault injection")
    with pytest.raises(FloatingPointError, match="fault injection"):
        module(torch.tensor([1.0, float("nan")]))
