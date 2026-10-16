"""Schedule cyclic garbage collection at explicit training boundaries.

Adapted from torchtitan's ``tools/utils.py``. Disable automatic collection while
the helper is active and restore its previous state on close or setup failure.
The checkpointer also uses ``collect`` before allocating staging buffers.
"""

from __future__ import annotations

import gc
import time

from .logging import get_logger

logger = get_logger(__name__)


class GarbageCollection:
    """Periodic, manually scheduled cyclic collection.

    Args:
        gc_freq: collect every this many steps; must be positive (matching
            upstream, non-positive values are rejected).
        debug: collect every step and warn about tensor reference cycles.
    """

    def __init__(self, gc_freq: int = 1000, *, debug: bool = False) -> None:
        if not isinstance(gc_freq, int) or isinstance(gc_freq, bool) or gc_freq <= 0:
            raise ValueError(f"gc_freq must be a positive integer, got {gc_freq}")
        if not isinstance(debug, bool):
            raise TypeError(f"debug must be bool, got {debug!r}")
        self.gc_freq = gc_freq
        self.debug = debug
        self._was_enabled = gc.isenabled()
        self._closed = False
        gc.disable()
        try:
            self.collect("Initial GC collection")
            if debug:
                from torch.utils.viz._cycles import warn_tensor_cycles

                from ..accelerator.distributed import is_main_process

                if is_main_process():
                    warn_tensor_cycles()
        except BaseException:
            self.close()
            raise

    def run(self, step_count: int) -> bool:
        """Collect if this step should. Returns whether a collection ran."""
        if self._closed:
            raise RuntimeError("GarbageCollection is closed")
        if self.debug:
            self.collect(
                "Force GC to perform collection to obtain debug information",
                generation=2,
            )
            return True
        if step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Performing periodic GC collection")
            return True
        return False

    def close(self) -> None:
        """Restore the automatic collector's state before training began."""
        if self._closed:
            return
        self._closed = True
        if self._was_enabled:
            gc.enable()

    @staticmethod
    def collect(reason: str, *, generation: int = 1) -> None:
        begin = time.monotonic()
        gc.collect(generation)
        logger.info("[GC] %s took %.2f seconds", reason, time.monotonic() - begin)
