"""Own model lifetimes while declared FSDP communication work drains."""

from typing import Any

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from .communication import drain_fsdp_communications


class FSDPCommunicationResources:
    """Close state-owned communication resources without selecting an engine.

    States exposing drain_communications own their streams, events and retained
    buffers. Native states use the version adapter in communication.py. The
    adapter waits for existing work; it never invokes backward or iteration
    reset callbacks to manufacture a completed iteration during shutdown.
    """

    def __init__(self, model_parts: tuple[nn.Module, ...]) -> None:
        self.model_parts = model_parts
        # Retain identities until the whole drain completes; successful owners
        # may not support a second shutdown after their streams are released.
        self._drained_states: dict[int, Any] = {}

    def close(self) -> None:
        seen = set()
        native_contexts = []
        native_groups = []
        native_states = []
        first_error = None
        for part in self.model_parts:
            for module in part.modules():
                if not isinstance(module, FSDPModule):
                    continue
                try:
                    state = module._get_fsdp_state()
                    if id(state) in seen or id(state) in self._drained_states:
                        continue
                    seen.add(id(state))
                    drain = getattr(state, "drain_communications", None)
                    if drain is None:
                        context = state._comm_ctx
                        groups = getattr(state, "_fsdp_param_groups", None)
                        if groups is None:
                            # Older native FSDP owns at most one parameter group.
                            group = state._fsdp_param_group
                            groups = [] if group is None else [group]
                        # Resolve all metadata before adding this state to the
                        # native batch, so failed discovery stays retryable.
                        groups = tuple(groups)
                        native_groups.extend(groups)
                        native_contexts.append(context)
                        native_states.append(state)
                        continue
                    if not callable(drain):
                        raise TypeError("FSDP drain_communications must be callable")
                    drain()
                    self._drained_states[id(state)] = state
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        try:
            drain_fsdp_communications(native_contexts, native_groups)
            self._drained_states.update((id(state), state) for state in native_states)
        except BaseException as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            # Keep every model owner until all waits succeed, so dependent
            # optimizer/offload cleanup remains deferred and close can retry.
            raise first_error
        self.model_parts = ()
        self._drained_states.clear()
