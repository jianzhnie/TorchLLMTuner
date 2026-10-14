import pytest

from llmtuner.parallel.module_patterns import (
    module_name_matches,
    resolve_module_patterns,
)


def test_invalid_pattern_is_checked_with_empty_module_inventory():
    with pytest.raises(ValueError, match="invalid module pattern range"):
        resolve_module_patterns([], patterns=["layers.{3-1}"], plan_name="FSDP")


@pytest.mark.parametrize(
    ("pattern", "name", "expected"),
    [
        ("model.layers.{*}.mlp", "model.layers.12.mlp", True),
        ("model.layers.{2-4}.*_proj", "model.layers.3.up_proj", True),
        ("model.layers.{2-4}.*_proj", "model.layers.5.up_proj", False),
        ("*.q_proj", "model.layers.0.self_attn.q_proj", True),
        ("layers.{*}", "layers.１２", False),
        ("layers.{*}", "layers.١٢", False),
        ("layers.{*}", "layers.12\n", False),
    ],
)
def test_extended_module_patterns(pattern: str, name: str, expected: bool) -> None:
    assert module_name_matches(name, pattern=pattern) is expected


def test_pattern_resolution_rejects_unmatched_pattern() -> None:
    with pytest.raises(ValueError, match="matched no module"):
        resolve_module_patterns(
            ["model.layers.0"], patterns=["missing.*"], plan_name="TP"
        )


@pytest.mark.parametrize("pattern", ["layers.{１-３}", "layers.{١-٣}"])
def test_numeric_ranges_require_ascii_digits(pattern):
    with pytest.raises(ValueError, match="invalid module pattern"):
        module_name_matches("layers.2", pattern=pattern)
