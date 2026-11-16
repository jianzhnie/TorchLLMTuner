"""Drain Torch FSDP communication buffers without running backward callbacks.

The context/group fields follow Torch 2.9/2.12 FSDP and the custom multi-context
implementation. Keep this version adapter separate from Trainer and resource
ownership. Lazy contexts can lack streams before the first forward.
"""

from collections.abc import Iterable
from typing import Any

import torch


def drain_fsdp_communications(contexts: Iterable[Any], groups: Iterable[Any]) -> None:
    """Wait for all captured work before clearing buffers; preserve them on failure.

    Streams are borrowed, never destroyed. This is shutdown, not iteration reset:
    no gradients are reduced, parameters updated or new collectives submitted.
    """
    contexts = tuple({id(context): context for context in contexts}.values())
    groups = tuple({id(group): group for group in groups}.values())
    streams = {}
    for context in contexts:
        for name in (
            "all_gather_copy_in_stream",
            "all_gather_stream",
            "reduce_scatter_stream",
            "all_reduce_stream",
        ):
            stream = getattr(context, name, None)
            if stream is not None:
                streams[id(stream)] = stream
    waitables = dict(streams)
    gather_results = {}
    for context in contexts:
        # A construction failure may precede native context.lazy_init().
        retained = list(getattr(context, "reduce_scatter_states", ()))
        for name in ("all_gather_state", "reduce_scatter_state"):
            pending = getattr(context, name, None)
            if pending is not None:
                retained.append(pending)
        for pending in retained:
            if pending.event is not None:
                waitables[id(pending.event)] = pending.event
            gathered = getattr(pending, "all_gather_result", None)
            if gathered is not None:
                gather_results[id(gathered)] = gathered
        for event in getattr(context, "_last_post_reduce_events", {}).values():
            waitables[id(event)] = event
    for group in groups:
        # An explicit async unshard may not yet have transferred its
        # result to comm_ctx.all_gather_state. Preserve its Work and
        # event until every shutdown wait has completed successfully.
        gathered = getattr(group, "_all_gather_result", None)
        if gathered is not None:
            gather_results[id(gathered)] = gathered
        for name in (
            "_post_reduce_event",
            "_reshard_after_forward_event",
            "_all_reduce_hook_stream",
        ):
            waitable = getattr(group, name, None)
            if waitable is not None:
                waitables[id(waitable)] = waitable
        reduced = getattr(group, "_all_reduce_state", None)
        if reduced is not None and reduced.event is not None:
            waitables[id(reduced.event)] = reduced.event
        for param in group.fsdp_params:
            if param.grad_offload_event is not None:
                event = param.grad_offload_event
                waitables[id(event)] = event
    works = {}
    for gathered in gather_results.values():
        if gathered.all_gather_event is not None:
            event = gathered.all_gather_event
            waitables[id(event)] = event
        if gathered.all_gather_work is not None:
            work = gathered.all_gather_work
            works[id(work)] = work
    first_error = None
    for work in works.values():
        try:
            if work.wait() is False:
                raise RuntimeError("FSDP communication Work did not complete")
        except BaseException as error:
            if first_error is None:
                first_error = error
    for waitable in waitables.values():
        # CPU streams are synchronous placeholders with no synchronize method.
        # Distributed Work objects above still need their explicit wait.
        if isinstance(waitable, torch.cpu.Stream):
            continue
        try:
            waitable.synchronize()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error
    for group in groups:
        for name in (
            "_all_gather_result",
            "_all_reduce_state",
            "_post_reduce_event",
            "_reshard_after_forward_event",
        ):
            if hasattr(group, name):
                setattr(group, name, None)
        for param in group.fsdp_params:
            param.grad_offload_event = None
    for context in contexts:
        for name in ("all_gather_state", "reduce_scatter_state"):
            if hasattr(context, name):
                setattr(context, name, None)
        for name in (
            "reduce_scatter_states",
            "_last_post_reduce_events",
            "post_forward_order",
        ):
            container = getattr(context, name, None)
            if container is not None:
                container.clear()
