"""The FSDP contract: what the sharding layer reads off a model, and how.

``fsdp.py`` is written against torchtitan's ``Decoder`` shape -- a ``ModuleDict``
of layers plus four named modules on the top-level object. llmtuner trains HF
models, whose layout differs on both counts. These tests pin the two places that
reconcile them, since a break in either is invisible until either:

* the model silently fails to shard (an adapter that stops taking effect), or
* FSDP raises partway through a distributed run (a name that stopped resolving).

They run against the real ``HFTransformerModel`` over a tiny offline LLaMA --
now that there is only one wrapper, the contract is worth pinning on the object
FSDP actually receives rather than on a stand-in that could drift from it.

No process group is needed: the layer iterator is pure container logic, and the
accessors are plain attribute lookups on a tiny offline model.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("flex_attention", "spmd_types")


import pytest
import torch
import torch.nn as nn
from torch.nn import ModuleDict, ModuleList

import llmtuner.parallel.fully_shard_parallel.fully_shard_parallel as apply
from llmtuner.models.hf.factory import build_model_config
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.fully_shard_parallel.fine_grained import (
    CustomFSDPStrategy,
    NativeFSDPStrategy,
    fully_shard_fine_grained,
    get_fsdp_strategy,
)
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    iter_transformer_layers,
    set_fsdp_prefetch_windows,
)

_VOCAB = 32
_HIDDEN = 8
_NUM_LAYERS = 3


def _wrapper(*, tied: bool = False) -> HFTransformerModel:
    config = build_model_config(
        "llama",
        seq_len=32,
        arch_overrides={
            "vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "intermediate_size": 16,
            "num_hidden_layers": _NUM_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "tie_word_embeddings": tied,
        },
    )
    return HFTransformerModel(config)


# -- container layout --------------------------------------------------------


def test_iter_transformer_layers_accepts_a_hf_module_list() -> None:
    layers = ModuleList([nn.Linear(4, 4) for _ in range(3)])

    assert [idx for idx, _ in iter_transformer_layers(layers)] == [0, 1, 2]


def test_iter_transformer_layers_accepts_a_torchtitan_module_dict() -> None:
    layers = ModuleDict({str(i): nn.Linear(4, 4) for i in range(3)})

    assert [idx for idx, _ in iter_transformer_layers(layers)] == ["0", "1", "2"]


def test_iter_transformer_layers_yields_blocks_in_order() -> None:
    layers = ModuleList([nn.Linear(4, 4) for _ in range(3)])
    as_dict = ModuleDict({str(i): m for i, m in enumerate(layers)})

    from_list = [block for _, block in iter_transformer_layers(layers)]
    from_dict = [block for _, block in iter_transformer_layers(as_dict)]

    assert all(a is b for a, b in zip(from_list, from_dict, strict=True))


# -- accessors ---------------------------------------------------------------


def test_decoder_parts_resolve_off_the_top_level_wrapper() -> None:
    """The FSDP layer reads these off the model it is handed, not off model.model."""
    wrapper = _wrapper()
    decoder = wrapper.model.model

    assert wrapper.tok_embeddings is decoder.embed_tokens
    assert wrapper.layers is decoder.layers
    assert wrapper.norm is decoder.norm
    assert wrapper.lm_head is wrapper.model.lm_head


def test_layers_are_a_container_torchtitan_can_iterate() -> None:
    wrapper = _wrapper()

    assert len(list(iter_transformer_layers(wrapper.layers))) == _NUM_LAYERS


def test_untied_head_reports_no_weight_tying() -> None:
    assert _wrapper(tied=False).enable_weight_tying is False


def test_tied_head_is_detected_by_parameter_identity() -> None:
    """FSDP2's own check: the two modules own one Parameter object."""
    wrapper = _wrapper(tied=True)

    assert wrapper.enable_weight_tying is True
    assert wrapper.tok_embeddings.weight is wrapper.lm_head.weight


def test_missing_head_reports_no_weight_tying() -> None:
    """A model whose head was stripped must not be grouped with the embedding."""
    wrapper = _wrapper(tied=True)
    wrapper.model.lm_head = None

    assert wrapper.enable_weight_tying is False


def test_tie_word_embeddings_flag_is_not_what_decides() -> None:
    """Intent and reality can disagree; identity is what FSDP acts on."""
    wrapper = _wrapper(tied=True)
    wrapper.model.config = type("C", (), {"tie_word_embeddings": False})()

    assert wrapper.enable_weight_tying is True


@pytest.mark.parametrize("tied", [False, True])
def test_accessors_do_not_add_duplicate_parameters(tied: bool) -> None:
    """The accessors must alias, not re-register, the underlying modules.

    Assigning ``self.x = module`` inside the wrapper would register a second
    copy, putting the same tensors in the state dict twice -- once under the
    HF name and once under the torchtitan name. These are properties, so the
    parameter count must equal the HF model's own.
    """
    wrapper = _wrapper(tied=tied)

    assert len(list(wrapper.parameters())) == len(list(wrapper.model.parameters()))


def test_apply_fsdp_preserves_parameter_dtype_for_mixed_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BF16 construction must not be silently widened by the FSDP policy."""

    class _Mesh:
        def size(self) -> int:
            return 2

    captured: dict[str, object] = {}
    monkeypatch.setattr(apply, "resolve_fsdp_mesh", lambda _dims: _Mesh())
    monkeypatch.setattr(apply, "resolve_sparse_fsdp_mesh", lambda _dims: None)
    monkeypatch.setattr(
        apply,
        "apply_fsdp_to_decoder",
        lambda _model, _mesh, **kwargs: captured.update(kwargs),
    )

    model = nn.Linear(4, 4, dtype=torch.bfloat16)
    parallel_dims = type(
        "ParallelTopologyStub",
        (),
        {"pp_enabled": False, "ep": 1},
    )()
    config = type(
        "ParallelConfigStub",
        (),
        {
            "fsdp_reshard_after_forward": "default",
            "fsdp_cpu_offload": False,
            "fsdp_cpu_offload_pin_memory": True,
            "enable_fsdp_symm_mem": False,
        },
    )()

    apply.apply_fsdp(model, config, parallel_dims)

    assert captured["param_dtype"] is torch.bfloat16
    assert captured["reduce_dtype"] is torch.float32


def test_apply_fsdp_forwards_cpu_offload_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user-facing offload knobs must reach the FSDP2 policy builder."""

    class _Mesh:
        def size(self) -> int:
            return 2

    captured: dict[str, object] = {}
    monkeypatch.setattr(apply, "resolve_fsdp_mesh", lambda _dims: _Mesh())
    monkeypatch.setattr(apply, "resolve_sparse_fsdp_mesh", lambda _dims: None)
    monkeypatch.setattr(
        apply,
        "apply_fsdp_to_decoder",
        lambda _model, _mesh, **kwargs: captured.update(kwargs),
    )

    model = nn.Linear(4, 4)
    parallel_dims = type("ParallelTopologyStub", (), {"pp_enabled": False, "ep": 1})()
    config = type(
        "ParallelConfigStub",
        (),
        {
            "fsdp_reshard_after_forward": "default",
            "fsdp_cpu_offload": True,
            "fsdp_cpu_offload_pin_memory": False,
            "enable_fsdp_symm_mem": False,
        },
    )()

    apply.apply_fsdp(model, config, parallel_dims)

    assert captured["cpu_offload"] is True
    assert captured["cpu_offload_pin_memory"] is False


def test_fsdp_prefetch_supports_arbitrary_windows() -> None:
    class Unit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_targets: list[nn.Module] = []
            self.backward_targets: list[nn.Module] = []

        def set_modules_to_forward_prefetch(self, modules) -> None:
            self.forward_targets = list(modules)

        def set_modules_to_backward_prefetch(self, modules) -> None:
            self.backward_targets = list(modules)

    units = [Unit() for _ in range(5)]
    set_fsdp_prefetch_windows(units, forward_count=2, backward_count=3)

    assert units[0].forward_targets == units[1:3]
    assert units[2].forward_targets == units[3:5]
    assert units[4].backward_targets == [units[3], units[2], units[1]]


def test_fsdp_strategy_selects_pyTorch_214_compatible_adapter() -> None:
    assert isinstance(get_fsdp_strategy("native"), NativeFSDPStrategy)
    assert isinstance(get_fsdp_strategy("custom"), CustomFSDPStrategy)


def test_fine_grained_adapter_rejects_a_distinct_hook_target() -> None:
    with pytest.raises(ValueError, match="separate hook_module"):
        fully_shard_fine_grained(nn.Linear(2, 2), hook_module=nn.Identity())


def test_fsdp_pattern_overrides_replace_common_policy(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.config import ParallelConfig
    from llmtuner.parallel.fully_shard_parallel import plan

    model = nn.Sequential(nn.Linear(2, 2))
    calls = []
    override_policy = object()
    cfg = ParallelConfig(fsdp_apply_modules={"0": {"mp_policy": override_policy}})
    monkeypatch.setattr(
        plan,
        "get_fsdp_strategy",
        lambda implementation: SimpleNamespace(
            get_fully_shard_fn=lambda: (
                lambda module, **kwargs: calls.append((module, kwargs))
            )
        ),
    )
    monkeypatch.setattr(apply, "disable_fsdp_gradient_division", lambda model: None)
    plan.apply_fsdp_by_pattern(
        model,
        dp_mesh=object(),
        cfg=cfg,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    assert calls[0][0] is model[0]
    assert calls[0][1]["mp_policy"] is override_policy
    assert calls[1][0] is model
    assert calls[1][1]["mp_policy"] is not override_policy
