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
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.stateful import Stateful

from ...accelerator.collectives import synchronize_local_errors
from ...errors import UnsupportedCombinationError
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
    RNG_STATE,
    SAFETENSORS_INDEX,
    TRAIN_STATE,
)
from .checkpoint_writer import GenerationFileSystemWriter
from .hf_storage import save_hf_checkpoint
from .load_planner import CheckpointLoadPlanner
from .storage import FilesystemCheckpointStorage

logger = get_logger(__name__)


class CoordinatedStager:
    """Agree on local staging success before any upload worker can save.

    The manager owns the underlying stager and the dedicated Gloo group. A
    callback may communicate from a staging thread, so this group must never
    be reused for training or the PROCESS executor's initialization broadcast.
    """

    should_synchronize_after_execute = False

    def __init__(self, stager: Any, group: dist.ProcessGroup | None) -> None:
        self.stager = stager
        self.group = group
        self.completion: Future[None] = Future()

    def stage(self, state_dict: dict[str, Any]) -> Future:
        try:
            staged = self.stager.stage(state_dict)
        except BaseException as error:
            staged = Future()
            staged.set_exception(error)
        if not isinstance(staged, Future):
            value = staged
            staged = Future()
            staged.set_result(value)
        coordinated: Future = Future()

        def finish(source: Future) -> None:
            try:
                with synchronize_local_errors("Checkpoint staging", group=self.group):
                    snapshot = source.result()
            except BaseException as error:
                self.completion.set_exception(error)
                if not isinstance(error, Exception):
                    # PyTorch 2.14 staging callbacks catch Exception only.
                    # Keep the original interrupt for the training thread,
                    # but finish the upload future without killing a worker.
                    upload_error = RuntimeError("Checkpoint staging interrupted")
                    upload_error.__cause__ = error
                    coordinated.set_exception(upload_error)
                else:
                    coordinated.set_exception(error)
            else:
                coordinated.set_result(snapshot)
                self.completion.set_result(None)

        staged.add_done_callback(finish)
        return coordinated

    def synchronize_staging(self) -> None:
        self.completion.result()

    def close(self) -> None:
        self.stager.close()


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
        self._validate_constructor_configuration(config, async_mode=config.async_mode)
        if not config.enable:
            super().__init__(
                config,
                model_parts=model_parts,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema=ema,
                states=states,
                folder=folder,
            )
            return

        # Only local preflight belongs here: model state construction below
        # may gather EP weights and must remain outside this boundary.
        with synchronize_local_errors("Checkpoint preflight"):
            effective_folder = filesystem.join(folder, config.folder)
            if config.last_save_in_hf and filesystem.is_remote(effective_folder):
                raise ValueError(
                    "checkpoint.last_save_in_hf is not supported with a remote "
                    f"effective checkpoint folder, got {effective_folder}"
                )
            self._storage = FilesystemCheckpointStorage()

            self.sd_adapter = sd_adapter
            # Runtime assembly supplies the HF wrapper adapter. Direct callers
            # must provide an adapter before enabling format-specific options.
            if self.sd_adapter is None:
                hf_options = [
                    name
                    for name, enabled in (
                        ("last_save_in_hf", config.last_save_in_hf),
                        ("initial_load_in_hf", config.initial_load_in_hf),
                        (
                            "initial_load_in_hf_quantized",
                            config.initial_load_in_hf_quantized,
                        ),
                    )
                    if enabled
                ]
                if hf_options:
                    raise ValueError(
                        f"checkpoint.{', '.join(hf_options)} is not wired: the HF "
                        "safetensors paths need a state_dict adapter. "
                        "Use the native DCP format, or supply one by "
                        "passing sd_adapter to DistributedCheckpointManager."
                    )

            # -- async and distributed infrastructure --
            try:
                self.async_mode = DistributedCheckpointAsyncMode(config.async_mode)
            except ValueError as e:
                raise ValueError(
                    f"Unknown checkpoint async_mode {config.async_mode}"
                ) from e

            storage_protocols = ()
            if filesystem.is_remote(effective_folder):
                storage_protocols = filesystem.resolve(effective_folder)[0].protocol
                if isinstance(storage_protocols, str):
                    storage_protocols = (storage_protocols,)
            if "memory" in storage_protocols and (
                self.async_mode == DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM
                or (dist.is_initialized() and dist.get_world_size() > 1)
            ):
                raise UnsupportedCombinationError(
                    "memory:// checkpoint storage is process-local and cannot be "
                    "shared by distributed ranks or the async_with_pinned_mem "
                    "save subprocess. Use a shared local/file://, s3:// or "
                    "gs:// folder."
                )

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

        # A gloo group, not the training backend: the async save thread
        # communicates over this group while the main thread keeps using NCCL.
        # Reusing the training group would have the two contend for the same
        # collectives.
        self.pg: dist.ProcessGroup | None = None
        self.staging_pg: dist.ProcessGroup | None = None
        self.stager: DefaultStager | None = None
        self.staging_future: Future | None = None
        self.save_future: Future | None = None

        # -- retention policy (the values came from the base class) --
        self.purge_thread: threading.Thread | None = None
        try:
            if dist.is_initialized() and self.async_mode in (
                DistributedCheckpointAsyncMode.ASYNC,
                DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM,
            ):
                self.pg = dist.new_group(backend="gloo")
            if (
                dist.is_initialized()
                and self.async_mode
                == DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM
            ):
                self.staging_pg = dist.new_group(backend="gloo")
            with synchronize_local_errors("Checkpoint services construction"):
                if self.keep_latest_k > 0:
                    self.purge_queue: queue.Queue[str | None] = queue.Queue()
                    self.purge_thread = threading.Thread(
                        target=purge_thread,
                        args=(self.purge_queue, self._storage.remove),
                        daemon=True,
                    )
                    self.purge_thread.start()
        except BaseException:
            # Release locally acquired groups and any started thread. Local
            # service errors have already been coordinated above.
            try:
                self._close()
            except BaseException:
                try:
                    logger.exception("DCP cleanup failed during construction")
                except BaseException:
                    pass  # Preserve the original construction failure.
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

            def close_stager() -> None:
                self.stager.close()
                self.stager = None

            releases.append(close_stager)
        first_error: BaseException | None = None
        for release in releases:
            try:
                release()
            except Exception as error:
                if self.stager is not None:
                    # A failed shutdown can leave staging threads alive even
                    # for ordinary exceptions. Retain their groups for retry.
                    raise
                if first_error is None:
                    first_error = error
                else:
                    try:
                        logger.exception("Additional DCP resource cleanup failure")
                    except BaseException:
                        pass  # Preserve the first resource cleanup failure.
        # A failed join/close may leave a worker alive. Keep its stager
        # and communication groups owned until a subsequent close succeeds.
        for name in ("staging_pg", "pg"):
            group = getattr(self, name, None)
            if group is None:
                continue
            if not dist.is_initialized():
                setattr(self, name, None)
                continue
            try:
                dist.destroy_process_group(group)
            except BaseException as error:
                # A failure before shutdown leaves a live owned group. A
                # failure after registry removal has already released it.
                # Registration alone does not promise backend usability after
                # a partial shutdown; retain ownership without claiming recovery.
                try:
                    dist.get_process_group_ranks(group)
                except KeyError:
                    setattr(self, name, None)
                if first_error is None:
                    first_error = error
                else:
                    try:
                        logger.exception("Additional DCP resource cleanup failure")
                    except BaseException:
                        pass  # Preserve the first resource cleanup failure.
            else:
                setattr(self, name, None)
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

        if not to_hf:
            # DCP invokes Stateful callbacks before its coordinated save. These
            # built-in states are local: materialize them together so a failed
            # callback cannot leave peers entering the writer protocol. Model,
            # optimizer, EMA and user callbacks may communicate and stay out.
            with synchronize_local_errors("Checkpoint local save preparation"):
                state_dict = dict(state_dict)
                for category in (LR_SCHEDULER, DATALOADER, TRAIN_STATE, RNG_STATE):
                    state = state_dict.get(category)
                    if isinstance(state, Stateful):
                        state_dict[category] = state.state_dict()

        if to_hf:
            if self.sd_adapter is None:
                raise ValueError("sd_adapter is required for to_hf=True")
            if async_mode != DistributedCheckpointAsyncMode.DISABLED:
                raise ValueError("HF exports require synchronous checkpoint saving")
            save_hf_checkpoint(
                self.sd_adapter.to_hf(state_dict),
                checkpoint_id,
                self.sd_adapter.fqn_to_index_mapping,
            )
            if enable_garbage_collection:
                with synchronize_local_errors("Checkpoint garbage collection"):
                    GarbageCollection.collect("GC collection invoked by checkpointer.")
            return None

        # Writer construction only resolves local storage/path policy. Agree
        # on success before any peer starts the DCP save protocol.
        with synchronize_local_errors("Checkpoint writer construction"):
            storage_writer = GenerationFileSystemWriter(checkpoint_id)

        if async_mode == DistributedCheckpointAsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_id,
                process_group=self.pg,
            )
        elif async_mode == DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM:
            stager = CoordinatedStager(self.stager, self.staging_pg)
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=stager,
            )
            if isinstance(stager, CoordinatedStager) and isinstance(
                ret, AsyncSaveResponse
            ):
                # PyTorch 2.14 marks an already-completed staging future as
                # successful without checking its exception. Keep our result.
                ret.staging_completion = stager.completion
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_id,
            )

        if enable_garbage_collection:
            with synchronize_local_errors("Checkpoint garbage collection"):
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
        if not from_hf:
            # Read only local compatibility records before DCP can write into
            # live model/optimizer tensors, including Kimi layout migrations.
            with synchronize_local_errors("Checkpoint local preflight preparation"):
                preflight_states = {
                    category: states[category]
                    for category in (RNG_STATE, DATALOADER, TRAIN_STATE, LR_SCHEDULER)
                    if category in states
                    and callable(getattr(states[category], "validate_state_dict", None))
                }
                preflight_destination = {
                    category: state.state_dict()
                    for category, state in preflight_states.items()
                }
            if preflight_states:
                preflight_planner = CheckpointLoadPlanner()
                dcp.load(
                    preflight_destination,
                    checkpoint_id=checkpoint_id,
                    planner=preflight_planner,
                )
                for category, state in preflight_states.items():
                    phase = "RNG" if category == RNG_STATE else category
                    with synchronize_local_errors(
                        f"Checkpoint {phase} preflight validation"
                    ):
                        if category != RNG_STATE or preflight_planner.has_rng_state:
                            state.validate_state_dict(preflight_destination[category])
            from .kimi_layout import load_changed_kimi_layout

            if load_changed_kimi_layout(
                checkpoint_id, states=states, adapter=self.sd_adapter
            ):
                with synchronize_local_errors("Checkpoint EMA cold start"):
                    if MODEL in states and EMA in self.states and EMA not in states:
                        self.states[EMA].reset_from_model()
                return
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
            rng_state = states.get(RNG_STATE)
            # DCP invokes Stateful callbacks after its own coordinated IO.
            # These local callbacks need a separate boundary before peers can
            # enter model/EMA communication. Keep collective-bearing states
            # (optimizer and EMA) on their existing restoration path.
            with synchronize_local_errors("Checkpoint local state preparation"):
                for category in (RNG_STATE, LR_SCHEDULER, DATALOADER, TRAIN_STATE):
                    if category in states:
                        state_dict[category] = states[category].state_dict()
            planner = CheckpointLoadPlanner()
            dcp.load(
                state_dict,
                checkpoint_id=checkpoint_id,
                planner=planner,
            )
            with synchronize_local_errors("Checkpoint scheduler restoration"):
                if scheduler is not None:
                    scheduler.load_state_dict(state_dict[LR_SCHEDULER])
            for category in (DATALOADER, TRAIN_STATE):
                with synchronize_local_errors(f"Checkpoint {category} restoration"):
                    if category in states:
                        states[category].load_state_dict(state_dict[category])

            # The model states were flattened into the top-level dict by
            # ``_flatten_model_state``, so DCP writes into those flat
            # entries rather than into the model. Pushing them back is a second
            # step, done explicitly here.
            with synchronize_local_errors("Checkpoint model validation"):
                if MODEL in states:
                    states[MODEL].validate_state_dict(state_dict)
            if MODEL in states:
                states[MODEL].load_state_dict(state_dict)

        # Reseed EMA from the just-loaded weights if it wasn't itself restored
        # (excluded via exclude_from_loading, or a model_only load -- which is
        # also how EMA gets turned on mid-run against a checkpoint that
        # predates it). MODEL is never excludable, so its presence means model
        # weights were actually restored to reseed from.
        with synchronize_local_errors("Checkpoint EMA cold start"):
            if MODEL in states and EMA in self.states and EMA not in states:
                self.states[EMA].reset_from_model()
        if not from_hf:
            # All ranks must reject invalid streams before any peer resumes.
            # These phases are local only; DCP and model collectives stay above.
            with synchronize_local_errors("Checkpoint RNG validation"):
                if rng_state is not None and planner.has_rng_state:
                    rng_state.validate_state_dict(state_dict[RNG_STATE])
            with synchronize_local_errors("Checkpoint RNG restoration"):
                if rng_state is not None and planner.has_rng_state:
                    # Restore last: model hooks and Stateful callbacks may draw RNG.
                    rng_state.load_state_dict(state_dict[RNG_STATE])

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

        self._wait_for_pending_save()
        with synchronize_local_errors("Checkpoint retention preflight"):
            self._purge_stale_checkpoints(saving_step=curr_step)

        begin = time.monotonic()
        checkpoint_phase = (
            "saving"
            if self.async_mode == DistributedCheckpointAsyncMode.DISABLED
            else "staging"
        )
        with synchronize_local_errors("Checkpoint save reporting"):
            logger.info(f"{checkpoint_phase.capitalize()} the checkpoint.")

        if last_step:
            self._save_last_step(curr_step)
            with synchronize_local_errors("Checkpoint completed retention"):
                self._purge_stale_checkpoints(saving_step=curr_step, completed=True)
            with synchronize_local_errors("Checkpoint save final reporting"):
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
            with synchronize_local_errors("Checkpoint garbage collection"):
                GarbageCollection.collect("GC collection invoked by checkpointer.")
            if self.async_mode == DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM:
                # All ranks enter even when a peer retained its stager after a
                # previous failed preparation. No save worker has started yet.
                with synchronize_local_errors("Checkpoint stager construction"):
                    if self.stager is None:
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
                if isinstance(result, AsyncSaveResponse):
                    self.staging_future = result.staging_completion
                    self.save_future = result.upload_completion
                elif isinstance(result, Future):
                    # Synchronous stagers have no staging future; PyTorch
                    # returns the active subprocess upload directly.
                    self.staging_future = None
                    self.save_future = result
                else:
                    raise TypeError(
                        "ASYNC_WITH_PINNED_MEM save must return a Future or "
                        f"AsyncSaveResponse, got {type(result).__name__}"
                    )
            else:
                if not isinstance(result, Future):
                    raise TypeError(
                        f"ASYNC save must return a Future, got {type(result).__name__}"
                    )
                # Own the live worker before any fallible post-launch cleanup.
                self.save_future = result
                self._pending_retention = {"saving_step": curr_step}
                with synchronize_local_errors("Checkpoint garbage collection"):
                    GarbageCollection.collect("GC collection invoked by checkpointer.")

        else:
            self._save_checkpoint(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=DistributedCheckpointAsyncMode.DISABLED,
                enable_garbage_collection=True,
            )

        if async_save_started_at is not None:
            self._pending_retention = {"saving_step": curr_step}
            # The line below only measured staging: the background write's
            # duration is otherwise invisible, which is the number that says
            # whether async checkpointing is actually keeping up.
            assert self.save_future is not None

            def log_save_completion(future: Future) -> None:
                try:
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
                except BaseException:
                    pass  # Reporting must not escape into the upload worker.

            self.save_future.add_done_callback(log_save_completion)

        if async_save_started_at is None:
            with synchronize_local_errors("Checkpoint completed retention"):
                self._purge_stale_checkpoints(saving_step=curr_step, completed=True)
        with synchronize_local_errors("Checkpoint save completion reporting"):
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
        try:
            keys = reader.read_metadata().state_dict_metadata
        except FileNotFoundError:
            # The retention worker may remove an already queued old directory
            # between the marker check and metadata read during another scan.
            return False
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
            model_state = states[MODEL].state_dict()
            collisions = sd.keys() & model_state.keys()
            if collisions:
                raise ValueError(
                    "Model checkpoint keys overlap training state categories: "
                    f"{sorted(collisions)}. Rename the model entries "
                    "or state categories."
                )
            sd.update(model_state)
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

            # state_dict above may communicate; conversion and reporting are
            # local and must succeed on every rank before writer construction.
            with synchronize_local_errors("Checkpoint final model preparation"):
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
            with synchronize_local_errors("Checkpoint final state reporting"):
                logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flatten_model_state()

        self._save_checkpoint(
            states,
            checkpoint_id=self._create_final_checkpoint_id(curr_step),
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
