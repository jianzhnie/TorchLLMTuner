"""Checkpoint managers and intentional lazy public exports.

core owns checkpoint policy and state adaptation; backends own byte IO and
backend resources; storage supplies paths, planners and staging; hf supplies
safetensors publication and expert storage. Model format rules remain owned by
models, and parallel state conversion remains owned by parallel.

Training state is registered under core.checkpoint_keys: model, optimizer,
lr_scheduler, dataloader, train_state, optional ema, rng_state and format metadata.
ModelCheckpointState wraps final local model ownership. Optimizer and EMA state
use FQN keys; scheduler and data state preserve progress across resume. Excluded
EMA cold-starts from loaded weights; RNG restores after other callbacks.

Exports stay lazy so configuration can import dependency-free state keys without
loading torch.distributed or optional checkpoint backends."""

from importlib import import_module

from ...utils.lazy_exports import resolve_export

# Configuration imports only dependency-free state keys. Keep package exports
# lazy so importing those keys does not load distributed or optional backends.
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
