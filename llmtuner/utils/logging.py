"""Colored distributed-aware logging with rank information.

llmtuner configures one stdout handler per module logger instead of touching the
root logger, so a module that wants its INFO lines on the console takes its
logger from ``get_logger`` here rather than from ``logging.getLogger``. The
rank decision is made by a filter *at emit time*, not when the logger is
created: engine modules are imported before ``init_dist``, when the real rank
is not knowable yet, so an import-time rank check would hand every rank a
rank-0 configuration.

Ancestor handlers owned by this module skip records already emitted to the same
destination. Optional file destinations use plain formatting and rank 0 only.
Application handlers still receive propagated records and retain their own
filtering policy.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass, field
from functools import lru_cache
from logging import Formatter, LogRecord
from operator import index
from pathlib import Path
from threading import RLock
from typing import ClassVar, Literal
from weakref import WeakKeyDictionary, WeakValueDictionary

from colorama import Fore, Style


@dataclass
class RecordState:
    rank: int
    destinations: set[str] = field(default_factory=set)


class LoggingState:
    """Own synchronization and weak caches without modifying log records."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.files: WeakValueDictionary[str, logging.FileHandler] = (
            WeakValueDictionary()
        )
        self.records: WeakKeyDictionary[LogRecord, RecordState] = WeakKeyDictionary()
        self.ranks = frozenset({0})
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self.reset_lock)

    def reset_lock(self) -> None:
        self.lock = RLock()

    def record(self, record: LogRecord) -> RecordState:
        with self.lock:
            state = self.records.get(record)
            if state is None:
                state = RecordState(get_distributed_rank())
                self.records[record] = state
            return state


@lru_cache(maxsize=1)
def get_logging_state() -> LoggingState:
    """Create shared configuration lazily; keep mutable state instance-owned."""
    return LoggingState()


class RankFormatter(Formatter):
    """Plain or colored rank-aware formatting without mutating log records."""

    COLORS: ClassVar[dict[str, str]] = {
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "DEBUG": Fore.LIGHTGREEN_EX,
    }

    def __init__(self, fmt: str | None = None, *, color: bool = False) -> None:
        super().__init__(
            fmt
            or (
                "%(asctime)s - [Rank %(rank)d] - "
                "%(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.color = color

    def format(self, record: LogRecord) -> str:
        rank = get_logging_state().record(record).rank
        record = copy(record)
        record.rank = rank
        message = super().format(record)
        if self.color:
            return self.COLORS.get(record.levelname, "") + message + Fore.RESET
        return message


class DestinationFilter(logging.Filter):
    """Apply rank policy and deduplicate emissions to a console or file."""

    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        self.path = path

    def filter(self, record: LogRecord) -> bool:
        config = get_logging_state()
        with config.lock:
            state = config.record(record)
            if self.path is None:
                if record.levelno < logging.ERROR and state.rank not in config.ranks:
                    return False
            elif state.rank != 0:
                return False
            destination = self.path or "console"
            if destination in state.destinations:
                return False
            state.destinations.add(destination)
            return True


class ConsoleHandler(logging.StreamHandler):
    """Console destination owned by this module."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)
        self.setFormatter(RankFormatter(color=True))
        self.addFilter(DestinationFilter())


class FileHandler(logging.FileHandler):
    """Plain UTF-8 file destination owned by this module."""

    def __init__(self, path: Path, mode: Literal["a", "w"]) -> None:
        super().__init__(path, mode=mode, encoding="utf-8", delay=True)
        self.setFormatter(RankFormatter())
        self.addFilter(DestinationFilter(str(path)))

    def filter(self, record: LogRecord) -> bool:
        # A shared handler may remain attached to another logger after close.
        # It must not reserve a destination or reopen an append stream.
        return not self._closed and super().filter(record)


def set_log_ranks(ranks: Iterable[int]) -> None:
    """Set which ranks print below-ERROR lines (default: rank 0 only).

    Normalize once at the configuration boundary so filters remain a cheap
    membership check. Reject strings and negative ranks early: accepting a
    string would otherwise interpret ``"10"`` as ranks 1 and 0.
    """
    if isinstance(ranks, str | bytes):
        raise TypeError("log ranks must be an iterable of integers")
    try:
        normalized = frozenset(index(rank) for rank in ranks)
    except (TypeError, ValueError) as exc:
        raise TypeError("log ranks must be an iterable of integers") from exc
    if any(rank < 0 for rank in normalized):
        raise ValueError("log ranks must be non-negative")
    get_logging_state().ranks = normalized


def get_logger(
    name: str,
    log_level: int = logging.INFO,
    *,
    log_file: str | Path | None = None,
    file_mode: Literal["a", "w"] = "a",
) -> logging.Logger:
    """Create or retrieve a module logger with a rank-aware stdout handler.

    Below ERROR, only ranks selected by set_log_ranks print (default: rank 0);
    ERROR and above print on every rank. ``DestinationFilter`` decides at emit time,
    so the logger can be created at module import time, before the process
    group exists. The handler is attached once per ``name``; repeat calls
    update the logger level without adding another handler. Records continue
    to propagate to application handlers; ancestor llmtuner handlers emit each
    record only once.

    ``log_file`` adds a UTF-8 file destination with no ANSI colors. Only rank 0
    writes the file, independently of ``set_log_ranks``. Parent directories
    are created automatically. Default mode is append; ``"w"`` truncates on
    the first emitted record. Repeated configuration of the same path reuses
    its handler; changing its mode is rejected rather than reopening the file.
    """
    if file_mode not in ("a", "w"):
        raise ValueError("file_mode must be either 'a' or 'w'")
    logger = logging.getLogger(name)
    # Serialize handler inspection and installation for concurrent callers.
    config = get_logging_state()
    with config.lock:
        logger.setLevel(log_level)
        for handler in logger.handlers[:]:
            if isinstance(handler, FileHandler) and handler._closed:
                logger.removeHandler(handler)
        # Do not remove handlers installed by an embedding application. Only
        # manage the handler owned by this module.
        if not any(isinstance(handler, ConsoleHandler) for handler in logger.handlers):
            logger.addHandler(ConsoleHandler())

        if log_file is not None:
            path = Path(log_file).expanduser().resolve()
            files = config.files
            handler = files.get(str(path))
            if handler is not None and not handler._closed:
                if handler.mode != file_mode:
                    raise ValueError("log file is already configured with another mode")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                handler = FileHandler(path, file_mode)
                # Shared paths must not reopen and truncate one another.
                files[str(path)] = handler
            logger.addHandler(handler)
    return logger


def get_distributed_rank() -> int:
    """Return the current distributed rank, falling back to the RANK env var or 0.

    Public because components outside the logging stack have to ask the same
    question -- the metrics processor's rank and the report it decorates are
    separate things, and it needs the first to decide the second.
    """
    try:
        from ..accelerator.distributed import get_rank, is_distributed

        # get_rank() returns 0 before initialization, so check first to retain
        # torchrun's environment rank during startup.
        if is_distributed():
            return get_rank()
    except Exception:
        # Logging here would query rank again through the formatter/filter and
        # recurse when distributed discovery fails with DEBUG logging enabled.
        pass

    # Only RANK matters here; malformed WORLD_SIZE/LOCAL_RANK must not prevent
    # logging a configuration error. Invalid RANK values safely fall back to 0.
    try:
        return max(0, int(os.environ.get("RANK", "0")))
    except ValueError:
        return 0
