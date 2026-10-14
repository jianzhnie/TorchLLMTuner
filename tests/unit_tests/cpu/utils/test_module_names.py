"""Canonical names remain independent of model wrappers and checkpoint IO."""

import pytest

from llmtuner.utils.module_names import canonical_fqn


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.0.weight", "model.layers.0.weight"),
        ("_checkpoint_wrapped_module.model.weight", "model.weight"),
        (
            "layers.0._checkpoint_wrapped_module.attention.weight",
            "layers.0.attention.weight",
        ),
        (
            "_checkpoint_wrapped_module.layers.0._checkpoint_wrapped_module.weight",
            "layers.0.weight",
        ),
        (
            "layers._checkpoint_wrapped_module_extra.weight",
            "layers._checkpoint_wrapped_module_extra.weight",
        ),
    ],
)
def test_wrapper_removal_matches_complete_segments(name, expected):
    assert canonical_fqn(name) == expected


def test_custom_wrapper_segments_replace_the_default_policy():
    assert (
        canonical_fqn("outer.layers.inner.weight", wrapper_segments=("outer", "inner"))
        == "layers.weight"
    )
    name = "_checkpoint_wrapped_module.weight"
    assert canonical_fqn(name, wrapper_segments=()) == name
