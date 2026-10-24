import queue
from dataclasses import dataclass

import pytest

from llmtuner.components.checkpointer import filesystem


def test_uri_scheme_selects_remote_storage():
    assert filesystem.is_remote("gs://bucket/path")
    assert filesystem.is_remote("s3://bucket/path")
    assert not filesystem.is_remote("/abs/path")
    assert not filesystem.is_remote("relative/path")


def test_local_queries_do_not_resolve_remote_backend(tmp_path, monkeypatch):
    def unexpected_resolve(path):
        pytest.fail("local operations must not resolve a remote backend")

    monkeypatch.setattr(filesystem, "resolve", unexpected_resolve)
    model = tmp_path / "model.bin"
    model.write_bytes(b"checkpoint")
    assert filesystem.exists(model)
    assert filesystem.isfile(model)
    assert not filesystem.isdir(model)
    assert filesystem.isdir(tmp_path)
    assert not filesystem.exists(tmp_path / "missing")


def test_listdir_returns_names_not_paths(tmp_path):
    for name in ("b.txt", "a.txt"):
        (tmp_path / name).write_text("x")
    assert sorted(filesystem.listdir(tmp_path)) == ["a.txt", "b.txt"]


@dataclass(frozen=True)
class StoragePath:
    path: str

    def __fspath__(self) -> str:
        return self.path


def test_remote_pathlike_uses_filesystem_path_for_all_operations(tmp_path):
    from fsspec.core import url_to_fs

    uri = f"memory://{tmp_path.name}/checkpoint"
    fs, path = url_to_fs(uri)
    fs.makedirs(path)
    with fs.open(f"{path}/model.bin", "wb") as stream:
        stream.write(b"checkpoint")
    directory = StoragePath(uri)
    model = StoragePath(f"{uri}/model.bin")
    try:
        assert filesystem.is_remote(directory)
        assert filesystem.join(directory, "model.bin") == model.path
        assert filesystem.exists(directory)
        assert filesystem.isdir(directory)
        assert filesystem.isfile(model)
        assert filesystem.listdir(directory) == ["model.bin"]
        filesystem.rmtree(directory)
        assert not fs.exists(path)
        filesystem.rmtree(directory)
    finally:
        if fs.exists(path):
            fs.rm(path, recursive=True)


def test_local_pathlike_uses_filesystem_path(tmp_path):
    directory = StoragePath(str(tmp_path))
    model = tmp_path / "model.bin"
    model.write_bytes(b"checkpoint")
    assert not filesystem.is_remote(directory)
    assert filesystem.join(directory, "model.bin") == str(model)
    assert filesystem.isdir(directory)
    assert filesystem.isfile(StoragePath(str(model)))
    assert filesystem.listdir(directory) == ["model.bin"]


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_local_checkpoint_delete_reports_failures(monkeypatch, error):
    def fail(path):
        raise error("cannot delete checkpoint")

    monkeypatch.setattr(filesystem.shutil, "rmtree", fail)
    with pytest.raises(error, match="cannot delete checkpoint"):
        filesystem.rmtree("checkpoint")


def test_local_checkpoint_delete_ignores_missing_paths(tmp_path):
    filesystem.rmtree(tmp_path / "missing")


def test_purge_continues_after_local_delete_failure(monkeypatch, tmp_path):
    from llmtuner.components.checkpointer import base

    failed = tmp_path / "blocked"
    subsequent = tmp_path / "next"
    subsequent.mkdir()
    real_rmtree = filesystem.shutil.rmtree
    warnings = []

    def remove(path):
        if path == str(failed):
            raise PermissionError("checkpoint is locked")
        real_rmtree(path)

    monkeypatch.setattr(filesystem.shutil, "rmtree", remove)
    monkeypatch.setattr(base.logger, "warning", lambda *args: warnings.append(args))
    pending = queue.Queue()
    pending.put(str(failed))
    pending.put(str(subsequent))
    pending.put(None)

    base.purge_thread(pending, filesystem.rmtree)

    assert not subsequent.exists()
    assert len(warnings) == 1
    assert warnings[0][1] == str(failed)
    assert isinstance(warnings[0][2], PermissionError)
    assert pending.empty()


@pytest.mark.parametrize("phase", ["begin", "failure", "end", "shutdown"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
def test_purge_survives_logging_failure(tmp_path, monkeypatch, phase, error_type):
    import queue
    import threading

    from llmtuner.components.checkpointer import base

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    pending = queue.Queue()
    for path in (str(first), str(second), None):
        pending.put(path)
    deleted = []
    errors = []

    def remove(path):
        if phase == "failure" and path == str(first):
            raise OSError("transient delete failure")
        filesystem.rmtree(path)
        deleted.append(path)

    def report(message, *args, **kwargs):
        matches = {
            "begin": "Checkpointer is deleting",
            "failure": "Checkpointer failed to delete",
            "end": "Checkpointer deleted",
            "shutdown": "Destroying the purge thread",
        }
        if message.startswith(matches[phase]):
            raise error_type("purge logger failed")

    def run():
        try:
            base.purge_thread(pending, remove)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(base.logger, "info", report)
    monkeypatch.setattr(base.logger, "warning", report)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors == []
    assert not second.exists()
    assert first.exists() == (phase == "failure")
    assert str(second) in deleted
    assert pending.empty()
