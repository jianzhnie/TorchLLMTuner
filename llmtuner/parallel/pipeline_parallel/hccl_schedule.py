"""Instance-local HCCL submission compatibility for two-rank 1F1B pipelines."""

from __future__ import annotations

from functools import lru_cache, update_wrapper
from inspect import signature
from types import FunctionType

import torch.distributed as dist
from torch.distributed.pipelining.schedules import Schedule1F1B

from llmtuner.errors import EnvironmentUnsupportedError, UnsupportedCombinationError


def ordered_p2p(ops: list[dist.P2POp], desc: str | None = None) -> list[dist.Work]:
    """Submit one peer's messages with matching HCCL send/receive order.

    Keep order within each direction and return all Work handles before waiting.
    The schedule still owns waits and tensor lifetimes. In particular, this must
    not send early from a stage's get_*_ops method.
    """
    if not ops:
        return []
    peers = {op.peer for op in ops}
    if (
        len(peers) != 1
        or any(op.group is not ops[0].group for op in ops)
        or any(op.op not in (dist.isend, dist.irecv) for op in ops)
    ):
        raise UnsupportedCombinationError(
            "Ordered HCCL 1F1B submission requires one peer, one group, "
            "and send/recv ops only"
        )
    peer = next(iter(peers))
    first = dist.isend if dist.get_rank() < peer else dist.irecv
    ordered = [op for op in ops if op.op is first]
    ordered.extend(op for op in ops if op.op is not first)
    return [
        op.op(
            op.tensor,
            group=op.group,
            tag=op.tag,
            **{"group_dst" if op.op is dist.isend else "group_src": op.group_peer},
        )
        for op in ordered
    ]


@lru_cache(maxsize=1)
def hccl_1f1b_schedule_class() -> type[Schedule1F1B]:
    """Bind upstream scheduling code to a private submission namespace.

    PyTorch's method calls a module-global helper, not an overridable method.
    Copy only its globals mapping: changing torch's module dictionary would also
    affect unrelated schedules and threads. Retain upstream code, closure and
    defaults rather than maintain a second copy of the scheduling algorithm.
    """
    method = Schedule1F1B._step_microbatches
    if (
        not isinstance(method, FunctionType)
        or not {"_batch_p2p", "_wait_batch_p2p"} <= set(method.__code__.co_names)
        or {"batch_isend_irecv", "_sorted_batch_p2p", "isend", "irecv"}
        & set(method.__code__.co_names)
        or tuple(signature(method).parameters)
        != ("self", "arg_mbs", "kwarg_mbs", "target_mbs", "losses", "return_outputs")
    ):
        raise EnvironmentUnsupportedError(
            "This PyTorch Schedule1F1B no longer exposes the expected P2P helper "
            "boundary; update the HCCL schedule compatibility adapter."
        )
    namespace = dict(method.__globals__)
    namespace["_batch_p2p"] = ordered_p2p
    bound = FunctionType(
        method.__code__,
        namespace,
        method.__name__,
        method.__defaults__,
        method.__closure__,
    )
    bound.__kwdefaults__ = method.__kwdefaults__
    update_wrapper(bound, method)

    class HCCLSchedule1F1B(Schedule1F1B):
        _step_microbatches = bound

    return HCCLSchedule1F1B
