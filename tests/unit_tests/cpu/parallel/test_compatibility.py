"""The combination matrix: assembly/probe verdicts, types, messages, guards.

The matrix's scope is cross-layer combination verdicts (assembly and probe
phase). Config-phase combination checks live in the configs' own
``__post_init__`` and are covered by ``test_config.py`` -- there is no
config-vs-matrix agreement to test anymore because there is only one copy.
"""

import pytest

from llmtuner.parallel import compatibility


def test_chunked_fsdp_accepts_native_decoder_ownership():
    from llmtuner.config.parallel_config import FSDPConfig

    compatibility.chunked_loss_fsdp(4, fsdp_config=FSDPConfig())


@pytest.mark.parametrize(
    "options",
    [
        {"fsdp_implementation": "custom"},
        {"fsdp_apply_modules": {"layers.*": {}}},
        {"fsdp_cpu_offload": True},
    ],
)
def test_chunked_fsdp_rejects_unvalidated_ownership(options):
    from llmtuner.config.parallel_config import FSDPConfig
    from llmtuner.errors import UnsupportedCombinationError

    with pytest.raises(UnsupportedCombinationError, match="root-owned boundaries"):
        compatibility.chunked_loss_fsdp(4, fsdp_config=FSDPConfig(**options))


@pytest.mark.parametrize("option", ["hybrid_parallel", "compile"])
def test_chunked_fsdp_rejects_unvalidated_execution(option):
    from llmtuner.config.parallel_config import FSDPConfig
    from llmtuner.errors import UnsupportedCombinationError

    with pytest.raises(UnsupportedCombinationError, match="root-owned boundaries"):
        compatibility.chunked_loss_fsdp(4, fsdp_config=FSDPConfig(), **{option: True})


def test_every_row_has_a_verdict_reason_and_guard() -> None:
    for row in compatibility.ENTRIES:
        assert row.name == row.fn.__name__
        assert row.reason == row.fn.__doc__.strip()
        assert row.phase in ("assembly", "probe")
        assert issubclass(row.error, Exception)
        assert row.reason and row.guard


def test_every_guard_function_has_a_row() -> None:
    # The table and the functions cannot drift: every row references a real
    # function in this module, and every public guard function is tabled.
    import inspect

    public = {
        n
        for n, v in vars(compatibility).items()
        if inspect.isfunction(v)
        and v.__module__ == compatibility.__name__
        and not n.startswith("_")
    }
    assert {r.name for r in compatibility.ENTRIES} == public


def test_config_phase_is_out_of_scope() -> None:
    # Config-phase checks live in config/* __post_init__ (see test_config.py);
    # the matrix carries only the cross-layer verdicts.
    assert {r.phase for r in compatibility.ENTRIES} == {"assembly", "probe"}


def test_guards_point_at_functions_that_call_the_verdict() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).parents[4] / "llmtuner"
    for row in compatibility.ENTRIES:
        path, qualified_name = row.guard.split("::")
        node = ast.parse((root / path).read_text())
        for name in qualified_name.split("."):
            matches = [
                child
                for child in node.body
                if isinstance(
                    child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                )
                and child.name == name
            ]
            assert len(matches) == 1, row.guard
            node = matches[0]
        assert any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "compatibility"
            and child.func.attr == row.name
            for child in ast.walk(node)
        ), row.guard


# -- every row rejects with its entry's type and message ------------------------


def test_rows_reject_with_their_entry_type() -> None:
    class _Block:
        pass

    cases = [
        (compatibility.eager_flex_parallel, ("CP",), "mask/collective"),
        (compatibility.fused_qkv_tp, (), "tensor parallelism"),
        (compatibility.validation_once_requires_dp1, (4,), "data-parallel"),
        (
            compatibility.validation_once_requires_finite_corpus,
            (),
            "infinite synthetic",
        ),
        (compatibility.chunked_loss_pp, (2, 2), "chunked_loss_num_chunks=2"),
        (compatibility.ep_hf_initial_load, (2,), "initial_load_in_hf"),
        (compatibility.chunked_loss_validation, (2,), "chunked_loss_num_chunks=2"),
        (compatibility.chunked_loss_fsdp, (2,), "chunked_loss_num_chunks=2"),
        (compatibility.pp_weight_tying, (), "tied word embeddings"),
        (compatibility.dsa_pp, (), "DSA"),
        (compatibility.shared_expert_tp, ("layers.0.mlp", _Block()), "shared expert"),
        (compatibility.tp_moe_specs_without_block, (2, _Block()), "no HF MoE block"),
        (compatibility.tp_moe_non_tensor_output, (_Block(), _Block()), "not a bare"),
        (compatibility.quantile_requires_ep, (), "ep=1 never runs"),
        (compatibility.ptrr_load_balancer_backstop, (), "ptrr"),
        (compatibility.gpt_oss_layout, (_Block(),), "bias vectors"),
        (compatibility.group_limited_greedy, (), "single best expert"),
        (compatibility.router_bias, (_Block(),), "router bias"),
        (compatibility.quantile_requires_sigmoid, ("softmax", _Block()), "sigmoid"),
        (compatibility.quantile_no_group_limit, (_Block(),), "group-limited"),
        (compatibility.shared_expert_gate, (_Block(),), "shared_expert_gate"),
        (compatibility.shared_expert_tp_ep, (_Block(),), "tp x ep"),
        (compatibility.pp_packed_microbatch_split, (2,), "packed"),
    ]
    assert len(cases) == len(compatibility.ENTRIES)
    for fn, args, match in cases:
        entry = next(r for r in compatibility.ENTRIES if r.fn is fn)
        with pytest.raises(entry.error, match=match):
            fn(*args)


def test_validation_once_rows_are_config_errors() -> None:
    from llmtuner.errors import ConfigError

    with pytest.raises(ConfigError):
        compatibility.validation_once_requires_dp1(2)
    with pytest.raises(ConfigError):
        compatibility.validation_once_requires_finite_corpus()
    # ... and therefore still ValueError, for the legacy assertions.
    with pytest.raises(ValueError):
        compatibility.validation_once_requires_dp1(2)
