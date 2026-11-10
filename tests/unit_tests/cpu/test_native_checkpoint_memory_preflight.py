import zipfile

import pytest
import torch

from tests.integration_tests.native_checkpoint_memory_preflight import (
    GIB,
    preflight,
    record_bytes,
)


def test_real_torch_records_reject_budget_before_payload_decode(tmp_path):
    paths = [tmp_path / f"{index}.pt" for index in range(4)]
    for path in paths:
        torch.save({"weight": torch.arange(1024)}, path)
    pairs = [(paths[0], paths[1]), (paths[2], paths[3])]
    report = preflight(pairs, GIB)
    assert not report["admitted"] and report["payload_loads"] == 0
    assert report["estimated_bytes"] > 2 * GIB
    assert preflight(pairs, report["estimated_bytes"])["admitted"]


def test_compressed_storage_charged_uncompressed(tmp_path):
    path = tmp_path / "record.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("archive/data/0", bytes(1024 * 1024))
    assert record_bytes(path) >= 1024 * 1024 > path.stat().st_size


def test_unknown_serialization_fails_closed(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"x": torch.ones(1)}, path, _use_new_zipfile_serialization=False)
    with pytest.raises(ValueError, match="non-ZIP"):
        record_bytes(path)
