"""The DCP checkpoint manager: sharded I/O, async saving, HF export.

Vendored from torchtitan's checkpoint component. The
``torch.distributed.checkpoint`` backend provides two distinct
things:

* **A sharded layout.** Under FSDP each rank holds one shard of every parameter
  and only needs to write that shard; DCP records the global layout and stitches
  the shards back together on load. llmtuner's old ``torch.save``-per-rank format
  could not express that -- it happened to work only because every run it was
  used for had one rank owning the whole tensor.
* **Async saving and lazy loading.** ``async_save`` stages to host memory and
  writes from a background thread, so the next training step does not wait on
  disk. ``load`` reads into the live state dict in place, so a resumed run does
  not materialize a second copy of the model.

Two departures from torchtitan, both subtractions:

* **No ``structured_logger``.** torchtitan wraps saves in ``sl.log_trace_span``
  and reports the async write's duration as a structured scalar from a
  ``save_future`` callback. The spans go; the duration stays, as a
  ``logger.info`` line from the same callback.

* **No ``torchtitan.tools.filesystem``.** The local equivalent is
  ``checkpointer/filesystem``, which has the same API for the paths this module
  uses. Deletion ignores missing paths and reports other failures, so a purge
  of an already-deleted directory is a no-op while permission or storage
  errors remain visible in the purge thread's logs.

``DistributedCheckpointAsyncMode.DISABLED`` is the default here, matching torchtitan.
"""

from __future__ import annotations

import enum
import os
import queue
import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)

from ...utils.garbage_collection import GarbageCollection
from . import filesystem

if TYPE_CHECKING:
    from ...config.checkpoint_config import CheckpointConfig

from ...utils.logging import get_logger
from .base import (
    BaseCheckpointManager,
    purge_thread,
)
from .checkpoint_keys import (
    DATALOADER,
    EMA,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    SAFETENSORS_INDEX,
)
from .storage import FilesystemCheckpointStorage

logger = get_logger(__name__)


class DataloaderLayoutLoadPlanner(DefaultLoadPlanner):
    """Allow pre-layout loader checkpoints while keeping all other keys strict."""

    def create_local_plan(self):
        assert self.metadata is not None
        saved_keys = self.metadata.state_dict_metadata
        prefix = f"{DATALOADER}.batch_layout."
        if (
            f"{DATALOADER}.version" in saved_keys or f"{DATALOADER}.steps" in saved_keys
        ) and not any(key.startswith(prefix) for key in saved_keys):
            layout_keys = [key for key in self.state_dict if key.startswith(prefix)]
            if layout_keys:
                logger.warning(
                    "Legacy dataloader checkpoint has no batch layout metadata; "
                    "its saved layout cannot be validated."
                )
                for key in layout_keys:
                    del self.state_dict[key]
        return super().create_local_plan()


class DistributedCheckpointAsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class DistributedCheckpointManager(BaseCheckpointManager):
    """Checkpoint manager backed by ``torch.distributed.checkpoint``.

    Note on Pipeline Parallelism and Virtual Stages:

    1. Even for simple PP schedules there is a separate optimizer per PP rank.
       Rank 0's optimizer has a ``param_group[0]`` referring to ``layers.0`` of
       the original model; rank 1's *also* has a ``param_group[0]``, since the
       index is positional, but referring to ``layers.1``. When saving, these
       collide and one of them is lost. Then on reload only one stage can restore
       its optimizer state and the others error.

       The solution is keying optimizer state by FQN instead of position. The
       ``OptimizersContainer`` does this unconditionally -- it has to, since one
       container already spans every model part, so there is no positional
       format for it to fall back to. torchtitan's container does the same.

    2. With complex PP schedules there are multiple model chunks per PP rank,
       which compounds (1) by also requiring us to reason about multiple local
       ``optim`` objects. ``ModelCheckpointState`` flattens the state dicts from each
       chunk into one before saving or loading, relying on the individual
       state dicts not to collide -- guaranteed for the model by correct
       pipeline splitting, and for the optimizer by the FQN keying in (1).

    3. LR schedulers index model state like optimizers do, so they are flattened
       the same way, under the assumption that all of them share a state dict.

    Args:
        config: how checkpointing is configured for this run.
        model_parts: the model chunks to checkpoint (one entry per PP stage).
        optimizer: the optimizer to checkpoint, as an ``OptimizersContainer``.
            Its state dict is flat and FQN-keyed, and it materializes a fresh
            optimizer's state before DCP plans the load.
        lr_scheduler: the lr schedule to checkpoint, as an
            ``LRSchedulersContainer``. Required: its ``last_epoch`` is the only
            thing that makes a resumed run continue the curve rather than
            restart it, and nothing else in the checkpoint carries it -- the
            optimizer restores group ``initial_lr`` values but not the step count.
            Registering
            it here also keeps it ordered after the optimizer, which it must be:
            scheduler restore reads those values and recomputes group ``lr``.
        ema: online EMA of model weights (``components.optimizer.EMA``), or
            None when the run has not configured one. Registered under the
            ``ema`` key; on a load that restores the model but not the EMA, the
            average is cold-started from the just-loaded weights.
        states: extra states to save beyond the model and optimizer.
        folder: absolute directory the checkpoints live in. Already joined with
            the run's dump folder by the caller.
        sd_adapter: converts model state dicts between the native layout and
            another format (HF safetensors). ``builder`` always passes
            ``HFTransformerStateDictAdapter``; passing none rejects the HF
            export paths at construction.
    """

    def __init__(
        self,
        config: CheckpointConfig,
        *,
        model_parts: list[nn.Module],
        optimizer: Any,
        lr_scheduler: Any,
        ema: Any | None = None,
        states: dict[str, Any],
        folder: str,
        sd_adapter: Any | None = None,
    ) -> None:
        super().__init__(
            config,
            model_parts=model_parts,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            states=states,
            folder=folder,
        )
        if not self.enable:
            return

        self._storage = FilesystemCheckpointStorage()

        self.sd_adapter = sd_adapter
        # llmtuner declares the HF safetensors options in its config but ships no
        # state-dict adapter to drive them, so every one of them is a capability
        # gap rather than a working option. Reject here -- at construction --
        # rather than where the option is first used, because that is minutes
        # into a run for a save. A loud rejection keeps the gap visible; a
        # silently ignored flag would train happily and write the wrong format.
        if self.sd_adapter is None:
            hf_options = [
                name
                for name, enabled in (
                    ("last_save_in_hf", self.last_save_in_hf),
                    ("initial_load_in_hf", self.initial_load_in_hf),
                    (
                        "initial_load_in_hf_quantized",
                        self.initial_load_in_hf_quantized,
                    ),
                )
                if enabled
            ]
            if hf_options:
                raise ValueError(
                    f"checkpoint.{', '.join(hf_options)} is not wired: the HF "
                    "safetensors paths need a state_dict adapter, and llmtuner "
                    "ships none. Use the native DCP format, or supply one by "
                    "passing sd_adapter to DistributedCheckpointManager."
                )

        # -- async and distributed infrastructure --
        try:
            self.async_mode = DistributedCheckpointAsyncMode(config.async_mode)
        except ValueError as e:
            raise ValueError(
                f"Unknown checkpoint async_mode {config.async_mode}"
            ) from e

        # A gloo group, not the training backend: the async save thread
        # communicates over this group while the main thread keeps using NCCL.
        # Reusing the training group would have the two contend for the same
        # collectives.
        self.pg: dist.ProcessGroup | None = None
        if dist.is_initialized() and self.async_mode in (
            DistributedCheckpointAsyncMode.ASYNC,
            DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM,
        ):
            self.pg = dist.new_group(backend="gloo")

        self.stager: DefaultStager | None = None
        self.staging_future: Future | None = None
        self.save_future: Future | None = None

        # -- retention policy (the values came from the base class) --
        self.purge_thread: threading.Thread | None = None
        if self.keep_latest_k > 0:
            self.purge_queue: queue.Queue[str | None] = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread,
                args=(self.purge_queue, self._storage.remove),
                daemon=True,
            )
            try:
                self.purge_thread.start()
            except BaseException:
                # The public close() skips incomplete construction, but this
                # point already owns an async process group.
                try:
                    self._close()
                except BaseException:
                    logger.exception("DCP cleanup failed during construction")
                raise

        logger.info(
            "Checkpointing active. Checkpoints will be loaded from and saved "
            f"to {self.folder}"
        )

        # Last, and load-bearing: everything above can raise (the HF-option
        # guard most notably), and ``__del__`` runs ``close`` on whatever was
        # left behind. Until this is set, the base class's public methods treat
        # the object as unusable rather than reading attributes that are absent.
        self._initialized = True

    def __del__(self) -> None:
        self.close()

    def _close(self) -> None:
        releases = []
        if (
            hasattr(self, "purge_thread")
            and self.purge_thread
            and self.purge_thread.is_alive()
        ):
            self.purge_queue.put(None)
            releases.append(self.purge_thread.join)

        if self.stager is not None:
            stager, self.stager = self.stager, None
            releases.append(stager.close)
        pg, self.pg = self.pg, None
        if pg is not None and dist.is_initialized():
            releases.append(lambda: dist.destroy_process_group(pg))
        first_error: BaseException | None = None
        for release in releases:
            try:
                release()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    logger.exception("Additional DCP resource cleanup failure")
        if first_error is not None:
            raise first_error

    def _save_checkpoint(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        *,
        async_mode: DistributedCheckpointAsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | AsyncSaveResponse | None:
        """Run the DCP save.

        Orchestrates the state dict transformation (to HuggingFace format, when
        asked), picks the storage writer, and dispatches on the requested
        synchronicity mode.

        Args:
            state_dict: the state dict to save.
            checkpoint_id: path identifying the checkpoint.
            async_mode: the saving/staging strategy.
            enable_garbage_collection: run a manual GC collect after the save.
            to_hf: use a ``HuggingFaceStorageWriter`` and adapt the state dict to
                safetensors and HF model definitions.

        Returns:
            ``None`` when saved synchronously; a ``Future`` for asynchronous
            disk I/O; or an ``AsyncSaveResponse`` for pinned-memory staging
            followed by disk I/O.
        """
        ret: Future | AsyncSaveResponse | None = None

        storage_writer: HuggingFaceStorageWriter | None = None
        fqn_to_index_mapping: dict[Any, int] | None = None

        if to_hf:
            if self.sd_adapter is None:
                raise ValueError("sd_adapter is required for to_hf=True")
            state_dict = self.sd_adapter.to_hf(state_dict)
            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping

            # If sharded, save into a subdirectory then consolidate up into
            # checkpoint_id.
            save_path = (
                os.path.join(checkpoint_id, "sharded")
                if fqn_to_index_mapping
                else checkpoint_id
            )
            storage_writer = HuggingFaceStorageWriter(
                path=save_path,
                save_distributed=True,
                fqn_to_index_mapping=fqn_to_index_mapping,
                enable_consolidation=not fqn_to_index_mapping,
            )
            # NOTE: with no fqn_to_index_mapping all FQNs go into a single
            # unified file, and the StorageWriter consolidates internally on one
            # rank. With a mapping the weights are spread across multiple files
            # (sharded), so the writer's internal consolidation is off and
            # consolidate_safetensors_files_on_every_rank does the merging.

        # For HF the storage_writer owns the path, so DCP gets no checkpoint_id.
        checkpoint_save_id = None if to_hf else checkpoint_id

        if async_mode == DistributedCheckpointAsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
            )
        elif async_mode == DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
            )

        if to_hf and fqn_to_index_mapping:
            consolidate_safetensors_files_on_every_rank(
                input_dir=os.path.join(checkpoint_id, "sharded"),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def _load_checkpoint(
        self,
        states: dict[str, Any],
        checkpoint_id: str,
        *,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        """Restore selected states through DCP or its HuggingFace reader.

        Handles both standard DCP sharded checkpoints and HuggingFace
        safetensors. Loading from HF routes through the adapter, which maps FQNs
        and handles format-specific sharding.

        Raises:
            ValueError: if ``from_hf`` is set without an ``sd_adapter``.
        """
        state_dict = self._flatten_model_state(states)

        if from_hf:
            if self.sd_adapter is None:
                raise ValueError(
                    "trying to load checkpoint in HF safetensors format, "
                    "but sd_adapter is not provided."
                )

            hf_state_dict = self.sd_adapter.to_hf(state_dict)
            hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
                checkpoint_id, from_quantized=from_quantized
            )

            dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

            state_dict = self.sd_adapter.from_hf(hf_state_dict)
            states[MODEL].load_state_dict(state_dict)
        else:
            # DCP's Stateful callback order is not an optimizer-before-scheduler
            # contract. Restore the scheduler only after optimizer groups load.
            scheduler = states.get(LR_SCHEDULER)
            if scheduler is not None:
                state_dict[LR_SCHEDULER] = scheduler.state_dict()
            dcp.load(
                state_dict,
                checkpoint_id=checkpoint_id,
                planner=DataloaderLayoutLoadPlanner(),
            )
            if scheduler is not None:
                scheduler.load_state_dict(state_dict[LR_SCHEDULER])

            # The model states were flattened into the top-level dict by
            # ``_flatten_model_state``, so DCP writes into those flat
            # entries rather than into the model. Pushing them back is a second
            # step, done explicitly here.
            if MODEL in states:
                states[MODEL].load_state_dict(state_dict)

        # Reseed EMA from the just-loaded weights if it wasn't itself restored
        # (excluded via exclude_from_loading, or a model_only load -- which is
        # also how EMA gets turned on mid-run against a checkpoint that
        # predates it). MODEL is never excludable, so its presence means model
        # weights were actually restored to reseed from.
        if MODEL in states and EMA in self.states and EMA not in states:
            self.states[EMA].load_state_dict({})

    def _save(self, curr_step: int, last_step: bool = False) -> bool:
        """Save the checkpoint for the current step.

        A save happens when any of these hold:
        1. It is the initial seed checkpoint (step 0).
        2. The current step matches the configured interval.
        3. ``last_step`` is set, which forces a save regardless of interval.

        Returns:
            Whether a checkpoint was written (or staged, in async modes).
        """
        if not self._should_save(curr_step, last_step):
            return False

        self.maybe_wait_for_saving()
        self._purge_stale_checkpoints(saving_step=curr_step)

        begin = time.monotonic()
        checkpoint_phase = (
            "saving"
            if self.async_mode == DistributedCheckpointAsyncMode.DISABLED
            else "staging"
        )
        logger.info(f"{checkpoint_phase.capitalize()} the checkpoint.")

        if last_step:
            self._save_last_step(curr_step)
            logger.info(
                f"Last step checkpoint completed in {time.monotonic() - begin:.2f}s"
            )
            return True

        checkpoint_id = self._create_checkpoint_id(curr_step)
        states = self._flatten_model_state()
        async_save_started_at: float | None = None

        if self.async_mode in (
            DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM,
            DistributedCheckpointAsyncMode.ASYNC,
        ):
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            if (
                self.async_mode == DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM
                and self.stager is None
            ):
                self.stager = DefaultStager(
                    StagingOptions(
                        use_pinned_memory=True,
                        use_shared_memory=True,
                        use_async_staging=True,
                        use_non_blocking_copy=True,
                    )
                )

            async_save_started_at = time.monotonic()
            result = self._save_checkpoint(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=self.async_mode,
            )
            if self.async_mode == DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM:
                # No GC needed on this path: the staging buffers are reused.
                if not isinstance(result, AsyncSaveResponse):
                    raise TypeError(
                        "ASYNC_WITH_PINNED_MEM save must return an "
                        f"AsyncSaveResponse, got {type(result).__name__}"
                    )
                self.staging_future = result.staging_completion
                self.save_future = result.upload_completion
            else:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                if not isinstance(result, Future):
                    raise TypeError(
                        f"ASYNC save must return a Future, got {type(result).__name__}"
                    )
                self.save_future = result

        else:
            self._save_checkpoint(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=DistributedCheckpointAsyncMode.DISABLED,
                enable_garbage_collection=True,
            )

        if async_save_started_at is not None:
            # The line below only measured staging: the background write's
            # duration is otherwise invisible, which is the number that says
            # whether async checkpointing is actually keeping up.
            assert self.save_future is not None

            def log_save_completion(future: Future) -> None:
                if future.cancelled():
                    logger.warning(
                        "Checkpoint write for step %d was cancelled.", curr_step
                    )
                    return
                error = future.exception()
                if error is not None:
                    logger.error(
                        "Checkpoint write for step %d failed: %s", curr_step, error
                    )
                    return
                logger.info(
                    "Finished writing the checkpoint for step %d in %.2f "
                    "seconds (async total).",
                    curr_step,
                    time.monotonic() - async_save_started_at,
                )

            self.save_future.add_done_callback(log_save_completion)

        logger.info(
            f"Finished {checkpoint_phase} the checkpoint in "
            f"{time.monotonic() - begin:.2f} seconds."
        )
        return True

    def _maybe_wait_for_staging(self) -> None:
        """Wait for staging to finish, when it is active.

        In ``ASYNC_WITH_PINNED_MEM`` mode the checkpoint data is first staged
        from device memory to pinned host memory. That staging is asynchronous
        and designed to overlap with the following training steps. This method
        ensures it has finished before the next checkpoint cycle begins or before
        training completes, avoiding memory contention on the pinned buffers.

        Raises:
            RuntimeError: if a staging future is set while the async mode is not
                ``ASYNC_WITH_PINNED_MEM``.
        """
        if self.staging_future is None:
            return

        if self.async_mode != DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM:
            raise RuntimeError(
                "self.staging_future is not None, "
                "but self.async_mode isn't ASYNC_WITH_PINNED_MEM."
            )

        staging_future = self.staging_future
        self.staging_future = None
        try:
            staging_future.result()
        finally:
            if not staging_future.done():
                self.staging_future = staging_future

    def _wait_for_saving(self) -> None:
        """Wait for any background save to complete.

        Blocking; ensures all checkpoint data has reached storage. The tracking
        future is cleared first so a completed failure is not retried by a later
        ``close()`` or ``__del__``. Interrupted waits retain an active write.

        Raises:
            RuntimeError: if a save future is set while the mode is ``DISABLED``.
        """
        if self.async_mode == DistributedCheckpointAsyncMode.DISABLED:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is DISABLED."
            )

        save_future = self.save_future
        assert save_future is not None
        self.save_future = None
        try:
            save_future.result()
        finally:
            if not save_future.done():
                self.save_future = save_future

    def _is_resumable_checkpoint(self, checkpoint_dir: str) -> bool:
        if not self._storage.isfile(filesystem.join(checkpoint_dir, ".metadata")):
            return False
        if self._parse_step(os.path.basename(checkpoint_dir)) == 0:
            # Seed checkpoints intentionally restore only model state.
            return True
        from torch.distributed.checkpoint._storage_utils import _storage_setup

        reader = _storage_setup(None, checkpoint_dir, reader=True)
        assert reader is not None
        keys = reader.read_metadata().state_dict_metadata
        return all(
            any(key == name or key.startswith(f"{name}.") for key in keys)
            for name in (OPTIMIZER, LR_SCHEDULER)
        )

    def _is_valid_checkpoint(self, checkpoint_dir: str) -> bool:
        return self._storage.isfile(filesystem.join(checkpoint_dir, ".metadata")) or (
            self._storage.isfile(filesystem.join(checkpoint_dir, SAFETENSORS_INDEX))
        )

    def _flatten_model_state(
        self, states: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Flatten model parameters into the top-level state dictionary.

        Merges the model's state dict into the top-level dict while leaving
        auxiliary states (optimizer, lr_scheduler) unflattened, giving the DCP
        writer one consistent format.

        Args:
            states: a custom dict to flatten. Defaults to ``None``, which
                uses the instance's own ``states``.
        """
        states = self.states if states is None else states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _save_last_step(self, curr_step: int) -> None:
        """Save the final checkpoint at the end of training.

        Handles the specific requirements of the final artifact: allowing model
        weights only (stripping optimizer state), converting to the export dtype,
        and optionally writing HF-compatible output.
        """
        # When last_save_model_only is False the full training state is written
        # with no dtype conversion, so the run can still be resumed. Otherwise
        # training is assumed complete and only the model is written, converted
        # if the current dtype differs from the export dtype.
        if self.last_save_in_hf:
            assert self.last_save_model_only, (
                "Only model can be saved when saving in HF safetensors format."
            )

        if self.last_save_model_only:
            states = self.states[MODEL].state_dict()

            states = {
                k: v.to(self.export_dtype)
                if isinstance(v, torch.Tensor)
                and v.is_floating_point()
                and v.dtype != self.export_dtype
                else v
                for k, v in states.items()
            }
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flatten_model_state()

        self._save_checkpoint(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=DistributedCheckpointAsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )


# Bound late so the module attribute ``DistributedCheckpointManager`` (used by the
# package export and tests) stays the class,
# while DATALOADER and LR_SCHEDULER remain importable from here for callers that
# build a fuller ``states`` dict -- the trainer puts one of each under those keys.
__all__ = [
    "DistributedCheckpointAsyncMode",
    "DistributedCheckpointManager",
    "DATALOADER",
    "EMA",
    "LR_SCHEDULER",
]
