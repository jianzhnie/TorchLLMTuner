"""Metric output sinks and their owned TensorBoard/WandB resources."""

from __future__ import annotations

import os
from typing import Any

from .logging import get_logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as error:
    SummaryWriter = None
    TENSORBOARD_IMPORT_ERROR = error

try:
    import wandb
except ImportError as error:
    wandb = None
    WANDB_IMPORT_ERROR = error

logger = get_logger(__name__)

__all__ = [
    "BaseLogger",
    "LoggerContainer",
    "TensorBoardLogger",
    "WandBLogger",
]


class BaseLogger:
    """A sink that takes metrics. Logging nothing is a valid implementation."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Writes scalars to a TensorBoard event directory."""

    def __init__(self, log_dir: str, tag: str | None = None):
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging requires the tensorboard package"
            ) from TENSORBOARD_IMPORT_ERROR

        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info("TensorBoard logging enabled. Logs will be saved at %s", log_dir)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            tag = key if self.tag is None else f"{self.tag}/{key}"
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Streams metrics to Weights & Biases.

    Experiment identity comes from the environment, matching torchtitan. That is
    deliberate for a launcher-driven tool: ``WANDB_RUN_ID`` and ``WANDB_RESUME_FROM``
    are how a requeued job rejoins its own run, and reading them from the
    environment means a launcher can set them without touching the config.
    Each logger creates and owns a separate run; closing it leaves existing
    runs alive. This requires WandB support for ``reinit="create_new"``.
    """

    def __init__(
        self,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        if wandb is None:
            raise ImportError(
                "WandB logging requires the wandb package"
            ) from WANDB_IMPORT_ERROR

        self.tag = tag

        os.makedirs(log_dir, exist_ok=True)
        self.run = wandb.init(
            reinit="create_new",
            entity=os.getenv("WANDB_TEAM"),
            project=os.getenv("WANDB_PROJECT", "llmtuner"),
            name=os.getenv("WANDB_RUN_NAME"),
            id=os.getenv("WANDB_RUN_ID"),
            notes=os.getenv("WANDB_RUN_NOTES"),
            tags=[
                tag.strip()
                for tag in os.getenv("WANDB_RUN_TAGS", "").split(",")
                if tag.strip()
            ]
            or None,
            group=os.getenv("WANDB_RUN_GROUP"),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE"),
            resume_from=os.getenv("WANDB_RESUME_FROM"),
            fork_from=os.getenv("WANDB_FORK_FROM"),
            dir=log_dir,
            config=config_dict,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        prefixed = {
            key if self.tag is None else f"{self.tag}/{key}": value
            for key, value in metrics.items()
        }
        # commit=True: without it wandb treats an explicit step as a partial
        # record and holds it open, so a metric logged once per step can be
        # merged into (or overwritten by) the next one.
        if self.run is None:
            raise RuntimeError("WandBLogger is closed")
        self.run.log(prefixed, step=step, commit=True)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


class LoggerContainer(BaseLogger):
    """Fans one ``log`` call out to every enabled sink."""

    def __init__(self) -> None:
        self._loggers: list[BaseLogger] = []

    def add_logger(self, logger_instance: BaseLogger) -> None:
        self._loggers.append(logger_instance)

    @property
    def number_of_loggers(self) -> int:
        return len(self._loggers)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log(metrics, step)

    def close(self) -> None:
        loggers, self._loggers = self._loggers, []
        first_error: BaseException | None = None
        for logger_instance in loggers:
            try:
                logger_instance.close()
            except BaseException as error:
                # Keep failed owners reachable so Trainer.close can retry;
                # successfully closed sinks must not be closed a second time.
                self._loggers.append(logger_instance)
                if first_error is None:
                    first_error = error
                else:
                    logger.exception("Additional metrics logger cleanup failure")
        if first_error is not None:
            raise first_error
