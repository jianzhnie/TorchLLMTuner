"""Checkpoint format: the state-dict keys, the HF index file, FQN helpers.

Both ``components/checkpointer`` (which writes states under these keys) and
``llmtuner/config`` (which validates policies over them, such as
``exclude_from_loading``) need the same strings. The config layer reads this
submodule directly -- it is dependency-free, so that import never pulls the
checkpointer backends (``base``/``dcp`` and their torch.distributed surface)
into ``llmtuner.config``; the package ``__init__`` re-exports those lazily for
the same reason. ``models/hf/state_dict_adapter.py`` reads the same way for the
index name: it writes that file, the DCP checkpointer probes for it, and one
spelling means the two cannot drift apart.
"""

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"
EMA = "ema"

# Written by the HF safetensors export (``HFTransformerStateDictAdapter``) and
# probed by the DCP checkpointer's "is this a resumable checkpoint" test.
SAFETENSORS_INDEX = "model.safetensors.index.json"

__all__ = [
    "DATALOADER",
    "EMA",
    "LR_SCHEDULER",
    "MODEL",
    "OPTIMIZER",
    "SAFETENSORS_INDEX",
    "TRAIN_STATE",
    "canonical_fqn",
]


# The segment the activation-checkpoint wrapper inserts into named_parameters().
# It can appear at any level of an FQN and is not part of the canonical model
# contract. torch.compile is applied in place and adds no segment.
_WRAPPER_PREFIXES: tuple[str, ...] = ("_checkpoint_wrapped_module",)


def canonical_fqn(name: str, prefixes: tuple[str, ...] = _WRAPPER_PREFIXES) -> str:
    """Strip wrapper segments from a dotted FQN.

    A segment may appear at any level, e.g.
    ``layers.0._checkpoint_wrapped_module.attention.wq.weight`` ->
    ``layers.0.attention.wq.weight``.

    This is what lets an optimizer state keyed on parameter FQNs stay stable
    across a run that turns activation checkpointing on or off, which is the
    difference between a checkpoint that resumes and one that silently loads
    nothing for the wrapped layers.
    """
    return ".".join(p for p in name.split(".") if p not in prefixes)
