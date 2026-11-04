"""Checkpointing: the manager contract, and the backends that implement it.

Ported from torchtitan's ``components/checkpointer/``. Four packages:

* ``core`` -- manager contracts, state keys and state adaptation. ``core.base``
  owns retention, step discovery, async draining and load selection without
  knowing how bytes are written.
* ``backends`` -- the DCP and optional torch-checkpointing managers. DCP shards
  under FSDP, saves asynchronously and loads in place. The optional backend
  defers its package imports and raises an install hint when constructed.
* ``storage`` -- filesystem paths, DCP readers/writers, planners and staging.
* ``hf`` -- Hugging Face storage and model-specific expert layouts.

``core.model_state`` adapts model parts to the checkpoint state interface.
``storage.storage`` defines the path-operation contract;
``storage.filesystem`` provides the filesystem adapter.
Model name normalization lives in ``utils/module_names.py``.

State keys. A checkpoint uses the names defined in ``core.checkpoint_keys``: ``model``,
``optimizer``, ``lr_scheduler``, ``dataloader``, ``train_state``, ``ema``.
llmtuner fills in the first two plus whatever the caller passes in ``states``:

* ``model`` -- a ``ModelCheckpointState`` over the model chunks.
* ``optimizer`` -- the ``OptimizersContainer`` itself, passed through. It is
  already a ``Stateful`` whose state dict is flat and FQN-keyed, which is what
  DCP needs to reshard a pipeline checkpoint, and it materializes a fresh
  optimizer's Adam moments before DCP plans a load.
* ``train_state`` -- the ``Trainer`` itself. torchtitan's ``Trainer`` is a
  ``Stateful`` exposing ``step`` and ``ntokens_seen``; llmtuner's trainer exposes
  the same two through ``state_dict``/``load_state_dict`` for the same reason:
  they are the counters a resumed run needs.
* ``lr_scheduler`` -- the ``LRSchedulersContainer`` from
  ``components/optimizer/lr_scheduler.py``, always registered. It holds one
  integer, ``last_epoch``, that nothing else in the checkpoint carries: the
  optimizer restores group ``initial_lr`` values and the saved ``lr``, but
  the step count is the scheduler's own, and a resumed run's fresh scheduler
  starts it at 0. Without it the curve restarts on the step after a resume --
  invisible while the lr is constant, wrong for the rest of the run once warmup
  or decay is set.
* ``dataloader`` -- the ``BaseDataLoader``, registered only when it is loadable.
  Both synthetic and Grain loaders support cursor restoration.
* ``ema`` -- the ``EMA`` pseudo-optimizer from
  ``components/optimizer/ema.py``, registered only when the run configures one
  (``training.ema_config``). Its state dict is the same flat, FQN-keyed layout
  as optimizer state. A load that restores the model but not this key -- an
  ``exclude_from_loading=["ema"]`` load, or any model-only load -- cold-starts
  the average from the just-loaded weights (see
  ``backends.distributed_checkpoint.DistributedCheckpointManager``).
"""

from importlib import import_module

from ...utils.lazy_exports import resolve_export

# Lazy (PEP 562): ``core.checkpoint_keys`` / ``storage.filesystem`` are lightweight
# are read by ``llmtuner/config``, which must stay importable without the
# torch.distributed surface ``core.base``/``backends.distributed_checkpoint`` pulls in
# (DTensor and friends are absent on older torch). Eager re-exports would make
# any submodule import pay
# for the backends.
_EXPORT_SOURCES = {
    "BaseCheckpointManager": "core.base",
    "CheckpointStorage": "storage.storage",
    "ModelCheckpointState": "core.model_state",
    # The state keys are resolved from their own dependency-free module rather
    # than through ``base``: the keys belong there, and a re-export chain was
    # one more place for the two to disagree.
    "DATALOADER": "core.checkpoint_keys",
    "EMA": "core.checkpoint_keys",
    "LR_SCHEDULER": "core.checkpoint_keys",
    "MODEL": "core.checkpoint_keys",
    "OPTIMIZER": "core.checkpoint_keys",
    "RNG_STATE": "core.checkpoint_keys",
    "SAFETENSORS_INDEX": "core.checkpoint_keys",
    "TRAIN_STATE": "core.checkpoint_keys",
    "DistributedCheckpointAsyncMode": "backends.distributed_checkpoint",
    "DistributedCheckpointManager": "backends.distributed_checkpoint",
    "filesystem": "storage.filesystem",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    """Resolve backend names on first touch (PEP 562 lazy re-export)."""
    if name == "filesystem":
        return import_module(f"{__name__}.storage.filesystem")
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
