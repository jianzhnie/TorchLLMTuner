"""The compile step: gating, defaults, and the four compile-side toggles.

``llmtuner/parallel/model_compile.py`` ports torchtitan's ``distributed/compile.py``:
per-block compile, async TP, regional_inductor and capture_scalar_outputs,
each behind its own switch so the default is the historical whole-model
``torch.compile(model)``.

These tests pin, on CPU with no process group:

* the default: whole-model compile, dense models never touch
  ``capture_scalar_outputs``, and ``CompileConfig()`` reproduces the old
  plain ``torch.compile(model)`` path;
* per-block compile wraps every decoder layer and not the model;
* the config-time loud errors: async TP without compile, async TP at tp=1;
* the assembly-time loud errors: async TP without a TP mesh, async TP on a
  torch without ``_micro_pipeline_tp``, a flex model on a non-inductor /
  non-aot_eager backend, and regional_inductor on a torch that lacks it;
* the MoE condition for ``capture_scalar_outputs`` (set when the model part
  carries token-choice MoE blocks, unset for a dense one), with the global
  restored afterwards.

The model-level tests need the flex/spmd import surface, which this
machine's torch (2.2.2) only provides under the stub preamble; without it
they skip rather than error, so the default-collected suite is unchanged.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from llmtuner.accelerator.capabilities import has
from llmtuner.config import (
    CompileConfig,
    LLMTunerConfig,
    ParallelConfig,
    TrainingConfig,
)

try:
    from torch._dynamo import OptimizedModule

    from llmtuner.models.hf.factory import build_model_config
    from llmtuner.models.hf.model import HFTransformerModel
    from llmtuner.parallel import model_compile as compile_mod
    from llmtuner.parallel.model_compile import apply_compile, maybe_regional_inductor

    _IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - env gate, see module docstring
    _IMPORT_ERROR = e

requires_runtime = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"this torch lacks the flex/spmd import surface: {_IMPORT_ERROR}",
)

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
_HIDDEN = 16
_SEQ = 24
_NUM_LAYERS = 3


def _model(seed: int = 0):
    torch.manual_seed(seed)
    config = build_model_config(
        "llama",
        seq_len=_SEQ,
        arch_overrides={
            "vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "intermediate_size": 32,
            "num_hidden_layers": _NUM_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    return HFTransformerModel(config)


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


def test_compile_config_rejects_empty_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        CompileConfig(backend="")


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


@requires_runtime
def test_default_compile_is_whole_model_and_leaves_dynamo_flags() -> None:
    """compile_config defaults: whole-model compile, dense flags untouched."""
    with _preserve_dynamo_flag("capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = False
        model = apply_compile(_model())
        assert isinstance(model, OptimizedModule)
        assert not any(isinstance(layer, OptimizedModule) for layer in model.layers)
        # Dense model: the MoE-only flag is untouched.
        assert torch._dynamo.config.capture_scalar_outputs is False


@requires_runtime
def test_per_block_compile_wraps_each_layer_not_the_model() -> None:
    model = apply_compile(_model(), compile_config=CompileConfig(per_block=True))
    assert not isinstance(model, OptimizedModule)
    # Module.compile is in place: same objects, compiled call impl attached.
    assert all(layer._compiled_call_impl is not None for layer in model.layers)


@requires_runtime
def test_async_tp_without_tp_mesh_raises() -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size > 1"):
        apply_compile(
            _model(),
            compile_config=CompileConfig(enable_async_tensor_parallel=True),
            tp_mesh=None,
        )


@requires_runtime
def test_per_block_compile_rejects_missing_layers_before_global_changes(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid per-block model must fail before global setup")

    monkeypatch.setattr(compile_mod, "maybe_enable_async_tp", unexpected)
    with pytest.raises(TypeError, match=".layers module"):
        apply_compile(
            torch.nn.Linear(2, 2), compile_config=CompileConfig(per_block=True)
        )


@requires_runtime
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


@requires_runtime
@pytest.mark.skipif(
    has("inductor_micro_pipeline_tp"),
    reason="this torch carries _micro_pipeline_tp; the loud-raise is unreachable",
)
def test_async_tp_on_unsupported_torch_loud_raises() -> None:
    with pytest.raises(NotImplementedError, match="_micro_pipeline_tp"):
        apply_compile(
            _model(),
            compile_config=CompileConfig(enable_async_tensor_parallel=True),
            tp_mesh=object(),  # never reached: the capability check fires first
        )


@requires_runtime
def test_regional_inductor_not_triggered_without_flex() -> None:
    """CPU/sdpa model + aot_eager: the backend passes through unwrapped."""
    backend = compile_mod.maybe_regional_inductor_backend(_model(), "aot_eager")
    assert backend == "aot_eager"


@requires_runtime
def test_flex_model_rejects_unknown_backend(monkeypatch) -> None:
    model = _model()
    monkeypatch.setattr(type(model), "uses_flex_attention", property(lambda self: True))
    with pytest.raises(ValueError, match="neither 'inductor' nor 'aot_eager'"):
        apply_compile(model, compile_config=CompileConfig(backend="eager"))


@requires_runtime
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch.fx.passes.regional_inductor")
    is not None,
    reason="this torch carries regional_inductor; the loud-raise is unreachable",
)
def test_regional_inductor_on_unsupported_torch_loud_raises(monkeypatch) -> None:
    model = _model()
    monkeypatch.setattr(type(model), "uses_flex_attention", property(lambda self: True))
    with pytest.raises(NotImplementedError, match="regional_inductor"):
        apply_compile(model, compile_config=CompileConfig(backend="aot_eager"))


@requires_runtime
def test_maybe_regional_inductor_is_null_when_disabled() -> None:
    with maybe_regional_inductor({}):
        pass  # must be a no-op context on the default path


@requires_runtime
def test_capture_scalar_outputs_set_only_for_token_choice_moe(monkeypatch) -> None:
    """The flag follows the MoE-dispatch condition, and only that condition.

    A real swapped MoE cannot be built in this environment (the installed
    transformers' Qwen3Moe layout predates the swap's probe), so the
    detection seam itself is stubbed; ``collect_moe_layers`` is pinned by the
    EP tests.
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


@requires_runtime
@pytest.mark.skipif(
    not has("fx_regional_inductor"), reason="regional inductor unavailable"
)
def test_regional_inductor_annotations_are_isolated_between_models(monkeypatch):
    import torch.fx.traceback as fx_traceback

    from llmtuner.models.hf import attention_backend

    regional_model, inductor_model, untouched_model = (_model() for _ in range(3))
    monkeypatch.setattr(
        type(regional_model), "uses_flex_attention", property(lambda self: True)
    )
    compile_mod.maybe_regional_inductor_backend(regional_model, "aot_eager")
    compile_mod.maybe_regional_inductor_backend(inductor_model, "inductor")
    annotations = []

    @contextlib.contextmanager
    def annotate(metadata):
        annotations.append(metadata)
        yield

    monkeypatch.setattr(fx_traceback, "annotate", annotate)
    monkeypatch.setattr(
        attention_backend,
        "flex_attention_forward",
        lambda *args, **kwargs: (args[1], None),
    )
    query = torch.zeros(1, 2, 4, 8)
    for model, expected_count in (
        (regional_model, 1),
        (inductor_model, 1),
        (untouched_model, 1),
        (regional_model, 2),
    ):
        output, _ = attention_backend.flex_attention_adapter(
            model.layers[0].self_attn, query, query, query, None
        )
        assert output is query
        assert len(annotations) == expected_count
    assert all("compile_with_inductor" in metadata for metadata in annotations)


@requires_runtime
def test_cp_flex_annotations_cover_attention_and_exclude_collectives(monkeypatch):
    from types import SimpleNamespace

    import transformers.integrations.flex_attention as hf_flex

    from llmtuner.parallel.context_parallel import attention

    active = False

    @contextlib.contextmanager
    def annotate(config, *, enabled):
        nonlocal active
        assert enabled
        active = True
        yield
        active = False

    query = SimpleNamespace(is_cuda=True)
    output = torch.randn(1, 4, 2, 8)

    def flex(module, q, k, v, mask, **kwargs):
        assert active
        return output, None

    monkeypatch.setattr(attention, "maybe_regional_inductor", annotate)
    monkeypatch.setattr(hf_flex, "flex_attention_forward", flex)
    actual = attention.run_flex(
        SimpleNamespace(_regional_inductor_enabled=True), query, query, query, None, {}
    )
    assert not active
    torch.testing.assert_close(actual, output.transpose(1, 2))


@requires_runtime
@pytest.mark.parametrize("per_block", [False, True])
def test_compiled_cpu_training_matches_eager_outputs_and_gradients(per_block):
    reference = _model()
    model = apply_compile(
        _model(), compile_config=CompileConfig(backend="aot_eager", per_block=per_block)
    )
    tokens = torch.arange(_SEQ) % _VOCAB
    expected = reference(tokens)
    actual = model(tokens)
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    reference_gradients = {
        name: parameter.grad for name, parameter in reference.named_parameters()
    }
    for name, parameter in model.named_parameters():
        canonical_name = name.removeprefix("_orig_mod.")
        torch.testing.assert_close(parameter.grad, reference_gradients[canonical_name])
