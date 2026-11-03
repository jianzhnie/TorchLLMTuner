"""Coordinate rank-local staging before distributed checkpoint upload."""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

import torch.distributed as dist

from ...accelerator.collectives import synchronize_local_errors

__all__ = ["CoordinatedStager"]


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
                # Complete the training-side boundary before triggering upload
                # callbacks, which may block or perform their own initialization.
                self.completion.set_result(None)
                coordinated.set_result(snapshot)

        staged.add_done_callback(finish)
        return coordinated

    def synchronize_staging(self) -> None:
        self.completion.result()

    def close(self) -> None:
        self.stager.close()
