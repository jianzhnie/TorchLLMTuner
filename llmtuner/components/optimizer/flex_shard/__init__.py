# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flexible storage-to-compute redistribution APIs."""

from llmtuner.utils.lazy_exports import resolve_export

from .optimizer_reshard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)

__all__ = [
    "DistMuon",
    "BlockShard",
    "BucketConfig",
    "ComputeLayout",
    "Owned",
]


def __getattr__(name: str):
    value = resolve_export(__name__, {"DistMuon": "dist_muon"}, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
