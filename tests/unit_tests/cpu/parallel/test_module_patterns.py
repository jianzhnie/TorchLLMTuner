import pytest

from llmtuner.parallel.module_patterns import (
    module_name_matches,
    resolve_module_patterns,
)


@pytest.mark.parametrize(
    ("pattern", "name", "expected"),
    [
        ("model.layers.{*}.mlp", "model.layers.12.mlp", True),
        ("model.layers.{2-4}.*_proj", "model.layers.3.up_proj", True),
        ("model.layers.{2-4}.*_proj", "model.layers.5.up_proj", False),
        ("*.q_proj", "model.layers.0.self_attn.q_proj", True),
    ],
)
def test_extended_module_patterns(pattern: str, name: str, expected: bool) -> None:
    assert module_name_matches(pattern, name) is expected


def test_pattern_resolution_rejects_unmatched_pattern() -> None:
    with pytest.raises(ValueError, match="matched no module"):
        resolve_module_patterns(["model.layers.0"], ["missing.*"], plan_name="TP")
