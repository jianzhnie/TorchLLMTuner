"""Compatibility with serialized layouts created before the module move."""

import base64
import pickle

from llmtuner.components.optimizer.flex_shard.optimizer_reshard import (
    ComputeLayout,
    Owned,
)

# Produced with the real pre-move config/optimizer_layout.py, not a mock class.
LEGACY_LAYOUT = (
    "gASVlwAAAAAAAACMIGxsbXR1bmVyLmNvbmZpZy5vcHRpbWl6ZXJfbGF5b3V0lIwN"
    "Q29tcHV0ZUxheW91dJSTlCmBlF2UKGgAjBRfRnJvemVuQ29uZmlnTWFwcGluZ5ST"
    "lCmBlE59lIwGX2l0ZW1zlIwCZHCUaACMBU93bmVklJOUKYGUXZRihpSFlHOGlGJo"
    "BimBlE59lGgJKXOGlGJOZWIu"
)


def test_pre_move_layout_pickle_loads_with_immutable_mappings():
    restored = pickle.loads(base64.b64decode(LEGACY_LAYOUT))
    expected = ComputeLayout({"dp": Owned()})
    assert type(restored) is ComputeLayout
    assert restored == expected
    assert hash(restored) == hash(expected)
    assert dict(restored.shard_order_by_tensor_dim) == {}
