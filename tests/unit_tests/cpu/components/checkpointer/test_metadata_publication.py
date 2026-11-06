"""Failed local publication preserves readable generations and its root error."""

from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from llmtuner.components.checkpointer.storage import checkpoint_writer
from llmtuner.components.checkpointer.storage.checkpoint_writer import (
    GenerationFileSystemWriter,
)


@pytest.mark.parametrize("use_collectives", [False, True])
def test_generation_prefix_is_added_once_and_old_shards_remain(
    tmp_path, use_collectives
):
    writer = GenerationFileSystemWriter(tmp_path)
    generations = []
    for offset in (0, 3):
        metadata = dcp.save(
            {"weight": torch.tensor([2.0, 7.0]) + offset},
            storage_writer=writer,
            use_collectives=use_collectives,
        )
        filenames = {value.relative_path for value in metadata.storage_data.values()}
        assert filenames
        assert all(name.count(writer.save_id) == 1 for name in filenames)
        generations.append({name: (tmp_path / name).read_bytes() for name in filenames})
    assert generations[0].keys().isdisjoint(generations[1])
    for name, contents in generations[0].items():
        assert (tmp_path / name).read_bytes() == contents


@pytest.mark.parametrize("failure_phase", ["serialization", "fsync", "publication"])
@pytest.mark.parametrize("file_uri", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("reuse_writer", [False, True])
def test_failed_metadata_publication_cleans_temp_and_allows_retry(
    tmp_path, monkeypatch, failure_phase, file_uri, cleanup_fails, reuse_writer
):
    if file_uri:
        pytest.importorskip("fsspec")
    checkpoint = tmp_path.as_uri() if file_uri else tmp_path
    original = torch.tensor([2.0, 7.0])
    writer = GenerationFileSystemWriter(checkpoint)
    dcp.save({"weight": original}, storage_writer=writer)
    metadata = (tmp_path / ".metadata").read_bytes()

    def fail(*args, **kwargs):
        if failure_phase == "serialization":
            args[1].write(b"partial metadata")
        raise OSError("injected metadata failure")

    original_unlink = Path.unlink
    original_dump = checkpoint_writer.pickle.dump
    original_fsync = checkpoint_writer.os.fsync
    metadata_serialized = False

    def record_serialization(*args, **kwargs):
        nonlocal metadata_serialized
        result = original_dump(*args, **kwargs)
        metadata_serialized = True
        return result

    def fail_metadata_fsync(fd):
        # Shard writes must finish normally; fail only after metadata dump.
        if metadata_serialized:
            raise OSError("injected metadata failure")
        return original_fsync(fd)

    def fail_cleanup(path, *args, **kwargs):
        if path.name.startswith(".metadata.") and path.suffix == ".tmp":
            raise PermissionError("injected cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        if cleanup_fails:
            patch.setattr(Path, "unlink", fail_cleanup)
        if failure_phase == "serialization":
            patch.setattr(checkpoint_writer.pickle, "dump", fail)
        elif failure_phase == "fsync":
            patch.setattr(checkpoint_writer.pickle, "dump", record_serialization)
            patch.setattr(checkpoint_writer.os, "fsync", fail_metadata_fsync)
        else:
            patch.setattr(checkpoint_writer.os, "replace", fail)
        with pytest.raises(
            CheckpointException, match="injected metadata failure"
        ) as error:
            dcp.save(
                {"weight": original + 3},
                storage_writer=writer
                if reuse_writer
                else GenerationFileSystemWriter(checkpoint),
            )
        assert "injected cleanup failure" not in str(error.value)

    assert (tmp_path / ".metadata").read_bytes() == metadata
    assert bool(list(tmp_path.glob(".metadata.*.tmp"))) == cleanup_fails
    restored = {"weight": torch.zeros_like(original)}
    dcp.load(restored, checkpoint_id=checkpoint)
    torch.testing.assert_close(restored["weight"], original, rtol=0, atol=0)
    dcp.save(
        {"weight": original + 3},
        storage_writer=writer
        if reuse_writer
        else GenerationFileSystemWriter(checkpoint),
    )
    dcp.load(restored, checkpoint_id=checkpoint)
    torch.testing.assert_close(restored["weight"], original + 3, rtol=0, atol=0)
