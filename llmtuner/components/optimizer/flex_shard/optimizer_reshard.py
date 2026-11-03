# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backward-compatible imports for FlexShard's original module path.

Declarative optimizer layout types live in :mod:`llmtuner.config.optimizer_layout`.
This module preserves their original import path and lazily exposes ``DistMuon``.
"""

from typing import TYPE_CHECKING

from llmtuner.config.optimizer_layout import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from llmtuner.utils.lazy_exports import resolve_export

if TYPE_CHECKING:
    from .dist_muon import DistMuon

__all__ = ["BlockShard", "BucketConfig", "ComputeLayout", "DistMuon", "Owned"]


def __getattr__(name: str):
    if name != "DistMuon":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = resolve_export(__package__, {"DistMuon": "dist_muon"}, name)
    globals()[name] = value
    return value
