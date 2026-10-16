"""Logger ownership, hierarchy and distributed console filtering."""

import io
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from llmtuner.utils import logging as module


@pytest.fixture
def logger_name():
    name = f"llmtuner_test_{uuid.uuid4().hex}"
    yield name
    for key in (name, f"{name}.child"):
        logger = logging.getLogger(key)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


def test_parent_and_child_emit_console_record_once(logger_name, capsys, caplog):
    module.get_logger(logger_name)
    logger = module.get_logger(f"{logger_name}.child")
    logger.warning("one console message")
    assert capsys.readouterr().out.count("one console message") == 1
    assert sum(r.message == "one console message" for r in caplog.records) == 1


def test_retrieval_updates_level_without_duplicate_handlers(logger_name):
    logger = module.get_logger(logger_name)
    handlers = logger.handlers[:]
    assert module.get_logger(logger_name, logging.DEBUG) is logger
    assert logger.level == logging.DEBUG
    assert logger.handlers == handlers
    # Handler inspection recovers when an application removes our handler.
    logger.removeHandler(handlers[0])
    module.get_logger(logger_name)
    assert len(logger.handlers) == 1
    handlers[0].close()


def test_existing_application_handler_is_preserved(logger_name, capsys):
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logging.getLogger(logger_name).addHandler(handler)
    logger = module.get_logger(logger_name)
    logger.warning("application message")
    assert output.getvalue().count("application message") == 1
    assert capsys.readouterr().out.count("application message") == 1
    assert handler in logger.handlers


def test_rank_filter_is_evaluated_at_emit_time(logger_name, monkeypatch, capsys):
    logger = module.get_logger(logger_name)
    monkeypatch.setattr(module.get_logging_state(), "ranks", frozenset({0}))
    monkeypatch.setattr(module, "get_distributed_rank", lambda: 2)
    logger.info("hidden")
    logger.error("rank failure")
    module.set_log_ranks([2])
    logger.info("selected rank")
    output = capsys.readouterr().out
    assert "hidden" not in output
    assert output.count("rank failure") == 1
    assert output.count("selected rank") == 1


@pytest.mark.parametrize("ranks", ["10", b"10", None, ["rank"], ["10"], [1.5]])
def test_log_rank_configuration_rejects_invalid_values(ranks):
    with pytest.raises(TypeError, match="iterable of integers"):
        module.set_log_ranks(ranks)


def test_log_rank_configuration_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        module.set_log_ranks([-1])


def test_concurrent_logger_initialization_installs_one_handler(logger_name):
    with ThreadPoolExecutor(max_workers=8) as pool:
        loggers = list(pool.map(module.get_logger, [logger_name] * 32))
    assert all(logger is loggers[0] for logger in loggers)
    assert len(loggers[0].handlers) == 1


def test_console_queries_rank_once(logger_name, monkeypatch, capsys):
    calls = []

    def rank():
        calls.append(None)
        return 0

    monkeypatch.setattr(module, "get_distributed_rank", rank)
    module.get_logger(logger_name).info("one rank query")
    assert len(calls) == 1
    assert "[Rank 0]" in capsys.readouterr().out


def test_formatter_does_not_mutate_application_record(monkeypatch):
    monkeypatch.setattr(module, "get_distributed_rank", lambda: 0)
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "%s", ("hi",), None)
    original = record.__dict__.copy()
    assert "hi" in module.RankFormatter("%(rank)s %(message)s").format(record)
    assert record.__dict__ == original


@pytest.mark.parametrize("rank, expected", [("3", 3), ("invalid", 0), ("-1", 0)])
def test_rank_fallback_needs_only_rank_environment(monkeypatch, rank, expected):
    from llmtuner.accelerator import distributed

    monkeypatch.setattr(distributed, "is_distributed", lambda: False)
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("WORLD_SIZE", "invalid")
    monkeypatch.setenv("LOCAL_RANK", "invalid")
    assert module.get_distributed_rank() == expected


def test_failed_rank_query_falls_back_without_recursive_logging(monkeypatch, capsys):
    from llmtuner.accelerator import distributed

    existing = logging.getLogger(module.__name__)
    old_level = existing.level
    old_handlers = existing.handlers[:]
    logger = module.get_logger(module.__name__, logging.DEBUG)
    monkeypatch.setattr(distributed, "is_distributed", lambda: True)

    def fail():
        raise RuntimeError("process group unavailable")

    monkeypatch.setattr(distributed, "get_rank", fail)
    monkeypatch.setenv("RANK", "0")
    try:
        logger.debug("fallback message")
        assert capsys.readouterr().out.count("fallback message") == 1
    finally:
        logger.setLevel(old_level)
        for handler in logger.handlers[:]:
            if handler not in old_handlers:
                logger.removeHandler(handler)
                handler.close()


def test_rank_query_reuses_accelerator_interface(monkeypatch):
    from llmtuner.accelerator import distributed

    monkeypatch.setattr(distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(distributed, "get_rank", lambda: 7)
    monkeypatch.setenv("RANK", "2")
    assert module.get_distributed_rank() == 7


def test_file_output_is_plain_utf8_and_default_mode_appends(
    logger_name, tmp_path, capsys
):
    path = tmp_path / "nested" / "train.log"
    path.parent.mkdir()
    path.write_text("previous run\n", encoding="utf-8")
    logger = module.get_logger(logger_name, log_file=path)
    logger.info("训练开始")
    contents = path.read_text(encoding="utf-8")
    assert contents.startswith("previous run\n")
    assert contents.count("训练开始") == 1
    assert "[Rank 0]" in contents
    assert "\x1b[" not in contents
    assert "训练开始" in capsys.readouterr().out


def test_file_output_can_be_added_after_logger_creation(logger_name, tmp_path):
    logger = module.get_logger(logger_name)
    path = tmp_path / "nested" / "train.log"
    assert module.get_logger(logger_name, log_file=path) is logger
    logger.info("new destination")
    assert "new destination" in path.read_text(encoding="utf-8")
    module.get_logger(logger_name, log_file=path.parent / "." / path.name)
    logger.info("no duplicate")
    assert path.read_text(encoding="utf-8").count("no duplicate") == 1
    assert len(logger.handlers) == 2


def test_overwrite_mode_truncates_once_and_shared_loggers_reuse_handler(
    logger_name, tmp_path
):
    path = tmp_path / "train.log"
    path.write_text("previous run\n", encoding="utf-8")
    parent = module.get_logger(logger_name, log_file=path, file_mode="w")
    parent.info("parent first")
    child = module.get_logger(f"{logger_name}.child", log_file=path, file_mode="w")
    child.info("child second")
    parent.info("parent third")
    contents = path.read_text(encoding="utf-8")
    assert "previous run" not in contents
    for message in ("parent first", "child second", "parent third"):
        assert contents.count(message) == 1


def test_file_output_is_rank_zero_even_when_console_selects_another_rank(
    logger_name, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "train.log"
    monkeypatch.setattr(module, "get_distributed_rank", lambda: 2)
    monkeypatch.setattr(module.get_logging_state(), "ranks", frozenset({2}))
    logger = module.get_logger(logger_name, log_file=path, file_mode="w")
    logger.info("rank two info")
    logger.error("rank two error")
    assert not path.exists()
    assert "rank two error" in capsys.readouterr().out
    monkeypatch.setattr(module, "get_distributed_rank", lambda: 0)
    logger.info("rank zero info")
    assert "rank zero info" in path.read_text(encoding="utf-8")


def test_changing_mode_does_not_truncate_existing_file(logger_name, tmp_path):
    path = tmp_path / "train.log"
    logger = module.get_logger(logger_name, log_file=path)
    logger.info("keep this")
    with pytest.raises(ValueError, match="another mode"):
        module.get_logger(logger_name, log_file=path, file_mode="w")
    assert "keep this" in path.read_text(encoding="utf-8")


def test_file_records_include_exception_traceback(logger_name, tmp_path):
    path = tmp_path / "train.log"
    logger = module.get_logger(logger_name, log_file=path)
    try:
        raise RuntimeError("training failure")
    except RuntimeError:
        logger.exception("step failed")
    contents = path.read_text(encoding="utf-8")
    assert "Traceback" in contents
    assert "RuntimeError: training failure" in contents
    assert "\x1b[" not in contents


def test_invalid_file_mode_is_rejected_before_creating_file(logger_name, tmp_path):
    path = tmp_path / "train.log"
    with pytest.raises(ValueError, match="file_mode"):
        module.get_logger(logger_name, log_file=path, file_mode="invalid")
    assert not path.exists()


def test_concurrent_file_configuration_writes_each_message_once(logger_name, tmp_path):
    path = tmp_path / "train.log"

    def write(i):
        logger = module.get_logger(logger_name, log_file=path, file_mode="w")
        logger.info("worker message %02d", i)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(16)))
    contents = path.read_text(encoding="utf-8")
    for i in range(16):
        assert contents.count(f"worker message {i:02d}") == 1


def test_closed_file_handler_can_be_reconfigured(logger_name, tmp_path):
    path = tmp_path / "train.log"
    logger = module.get_logger(logger_name, log_file=path)
    logger.info("before close")
    old_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    logger.removeHandler(old_handler)
    old_handler.close()
    module.get_logger(logger_name, log_file=path)
    logger.info("after close")
    contents = path.read_text(encoding="utf-8")
    assert contents.count("before close") == 1
    assert contents.count("after close") == 1


@pytest.mark.parametrize("mode", ["a", "w"])
def test_closed_attached_file_handler_does_not_hide_reconfigured_output(
    logger_name, tmp_path, mode
):
    path = tmp_path / "train.log"
    parent = module.get_logger(logger_name, log_file=path, file_mode=mode)
    child = module.get_logger(f"{logger_name}.child", log_file=path, file_mode=mode)
    child.info("before close")
    old = next(h for h in parent.handlers if isinstance(h, module.FileHandler))
    old.close()
    module.get_logger(logger_name, log_file=path, file_mode=mode)
    child.info("after close")
    contents = path.read_text(encoding="utf-8")
    assert contents.count("after close") == 1
    assert ("before close" in contents) == (mode == "a")
    assert old.stream is None
    assert old not in parent.handlers
    assert old in child.handlers
