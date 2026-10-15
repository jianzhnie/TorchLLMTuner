"""Checkpointing config."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from llmtuner.components.checkpointer import filesystem
from llmtuner.components.checkpointer.checkpoint_keys import (
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
)
from llmtuner.config.field_checks_config import require_integer_at_least
from llmtuner.errors import ConfigError
from llmtuner.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(kw_only=True)
class CheckpointConfig:
    """What the checkpoint manager keeps, where, and when."""

    enable: bool = False
    """Whether to enable checkpointing."""

    folder: str = "checkpoint"
    """Checkpoint folder, relative to the trainer dump folder.

    Local native DCP rewrites use fresh shard generations and replace metadata
    only after writing completes. Old and failed generations remain until the
    step directory is purged, so repeated saves of one step consume extra disk.
    Local HF exports similarly publish an index referring to fresh generation
    shards. Native DCP supports local/file://, memory://, s3:// and gs:// paths;
    object storage must atomically replace the entire metadata object. An error
    after remote commit can leave the new generation published despite a failed
    acknowledgement. S3/GCS transports still require deployment validation.
    memory:// is limited to single-process synchronous or threaded saves.
    """

    interval: int = 500
    """Checkpointing interval in steps."""

    initial_load_path: str | None = None
    """Optional checkpoint path used when the output checkpoint folder is empty."""

    initial_load_model_only: bool = True
    """Whether an initial checkpoint restores only model state.

    Only consulted on the initial-load path, i.e. when ``initial_load_path``
    names a checkpoint; with no initial checkpoint there is nothing to load
    either way.
    """

    initial_load_in_hf: bool = False
    """Whether the initial checkpoint uses Hugging Face safetensors."""

    initial_load_in_hf_quantized: bool = False
    """Whether the initial Hugging Face checkpoint uses quantized keys."""

    last_save_model_only: bool = True
    """Whether the final checkpoint contains only model state.

    If a resumable checkpoint already exists at the same step, final weights
    are written to its ``model-export/`` child to preserve the training state.
    """

    last_save_in_hf: bool = False
    """Whether the final model-only checkpoint uses Hugging Face safetensors."""

    export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    """Model dtype used by a final model-only checkpoint."""

    keep_latest_k: int = 10
    """Recent artifacts to retain per category, or zero to retain all.

    Resumable training checkpoints and model-only exports have independent
    limits, so a final export cannot evict the last training recovery point.
    """

    purge_exempt: Callable[[int], bool] | None = None
    """Optional predicate that exempts checkpoint steps from purging.

    Programmatic-only, like ``optimizer.param_groups``: the parser can build a
    flag out of the field but cannot call the value, so set this from code.
    """

    load_step: int = -1
    """Load the checkpoint at the specified step. If -1, load the latest one."""

    exclude_from_loading: list[str] = field(default_factory=list)
    """Non-model state keys excluded from loading."""

    enable_first_step_checkpoint: bool = False
    """Whether to save immediately after the first training step."""

    create_seed_checkpoint: bool = False
    """Whether to initialize and save an unsharded seed checkpoint."""

    load_only: bool = False
    """Whether to permit loads while disabling all saves."""

    async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
    "DCP save mode: synchronous, threaded async, or pinned-memory async; ignored by "
    "torch_checkpointing."

    def __post_init__(self) -> None:
        if not self.folder.strip():
            raise ConfigError("checkpoint.folder must not be empty.")
        require_integer_at_least(self, "interval", group="checkpoint")
        if not isinstance(self.load_step, int) or isinstance(self.load_step, bool):
            raise ConfigError(
                f"checkpoint.load_step must be an integer >= -1, got {self.load_step!r}"
            )
        if self.load_step < -1:
            raise ConfigError(
                f"checkpoint.load_step must be >= -1, got {self.load_step}"
            )
        require_integer_at_least(self, "keep_latest_k", group="checkpoint", minimum=0)
        if self.keep_latest_k == 1:
            raise ConfigError(
                "checkpoint.keep_latest_k must not be 1: at least 2 replicas "
                "are kept, as the latest may still be in the process of being "
                "saved."
            )
        if MODEL in self.exclude_from_loading:
            raise ConfigError(
                f"checkpoint.exclude_from_loading must not contain {MODEL!r}."
            )
        if (
            OPTIMIZER in self.exclude_from_loading
            and LR_SCHEDULER not in self.exclude_from_loading
        ):
            raise ConfigError(
                f"checkpoint.exclude_from_loading: {LR_SCHEDULER} must be "
                f"excluded when {OPTIMIZER} is excluded."
            )

        if self.initial_load_path:
            self.initial_load_path = self.initial_load_path.strip()
            if not (
                self.initial_load_path.startswith("/")
                or filesystem.is_remote(self.initial_load_path)
            ):
                raise ConfigError(
                    "checkpoint.initial_load_path must be an absolute path or "
                    f"a remote URI (e.g. gs://...), got {self.initial_load_path}"
                )
        if self.initial_load_in_hf and not self.initial_load_model_only:
            raise ConfigError(
                "checkpoint.initial_load_in_hf requires initial_load_model_only."
            )
        if self.initial_load_in_hf_quantized and not (
            self.initial_load_in_hf and self.initial_load_path
        ):
            raise ConfigError(
                "checkpoint.initial_load_in_hf_quantized requires "
                "initial_load_in_hf and initial_load_path."
            )
        if self.last_save_in_hf and not self.last_save_model_only:
            raise ConfigError(
                "checkpoint.last_save_in_hf requires last_save_model_only=True."
            )

        async_lowered = self.async_mode.lower()
        if async_lowered not in ("disabled", "async", "async_with_pinned_mem"):
            raise ConfigError(
                "checkpoint.async_mode must be one of "
                f"disabled/async/async_with_pinned_mem, got {async_lowered!r}"
            )
        self.async_mode = async_lowered

        # Remote (fsspec) checkpoint IO supports only the native DCP format. HF
        # safetensors read/write to a remote URI is not implemented, so reject
        # the combination up front instead of failing deep inside DCP.
        if self.last_save_in_hf and filesystem.is_remote(self.folder):
            raise ConfigError(
                "checkpoint.last_save_in_hf is not supported with a remote "
                f"checkpoint.folder, got {self.folder}"
            )
        if (
            self.initial_load_in_hf
            and self.initial_load_path
            and filesystem.is_remote(self.initial_load_path)
        ):
            raise ConfigError(
                "checkpoint.initial_load_in_hf is not supported with a remote "
                f"initial_load_path, got {self.initial_load_path}"
            )

        if self.load_only and self.enable_first_step_checkpoint:
            logger.warning(
                "checkpoint.load_only is True; enable_first_step_checkpoint "
                "will be ignored."
            )
        # Note torchtitan's sibling warning for ``initial_load_model_only``
        # without an ``initial_load_path`` is deliberately not ported: llmtuner
        # builds a default Config on every run (``--help`` included), so that
        # warning would fire on runs that never load anything.
