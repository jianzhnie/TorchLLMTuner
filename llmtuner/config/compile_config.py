"""Compile config (``training.compile=True``), grafted onto TrainingConfig."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmtuner.config.utils import require_bool
from llmtuner.errors import ConfigError


@dataclass(kw_only=True)
class CompileConfig:
    """Settings for ``training.compile=True`` -- how the model is compiled."""

    per_block: bool = field(
        default=False,
        metadata={
            "help": "Compile each decoder layer with fullgraph=True; False compiles "
            "the whole model."
        },
    )
    backend: str = field(
        default="inductor",
        metadata={"help": "torch.compile backend, for example inductor or aot_eager."},
    )
    enable_async_tensor_parallel: bool = field(
        default=False,
        metadata={
            "help": "Overlap compiled TP collectives and GEMMs; requires compile=True "
            "and TP > 1."
        },
    )
    enable_sdc: bool = field(
        default=False,
        metadata={
            "help": "Check compiled block outputs for non-finite values; adds "
            "synchronization and is intended for diagnostics."
        },
    )

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ConfigError("compile.backend must be a non-empty string.")
        require_bool(
            self,
            "per_block",
            "enable_async_tensor_parallel",
            "enable_sdc",
            group="compile",
        )
        if self.enable_async_tensor_parallel and self.backend != "inductor":
            raise ConfigError(
                "compile.enable_async_tensor_parallel requires backend='inductor': "
                "the TP micro-pipeline pass runs in Inductor's post-grad pipeline."
            )
