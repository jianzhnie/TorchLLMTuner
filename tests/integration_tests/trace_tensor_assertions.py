"""Diagnostic-only ATen assertion tracing for an integration module.

Run with torchrun ... -m tests.integration_tests.trace_tensor_assertions MODULE
[MODULE_ARGS]. This changes dispatch overhead: never use it for performance data.
All operators still execute; no assertions or device checks are suppressed.
"""

import runpy
import sys
import traceback

import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode


class AssertionTrace(TorchDispatchMode):
    """Report unique assertion call stacks on rank zero, then execute normally."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func).startswith("aten._assert_async") and (
            not dist.is_initialized() or dist.get_rank() == 0
        ):
            stack = "".join(traceback.format_stack(limit=20))
            if stack not in self.seen:
                self.seen.add(stack)
                print(f"ASSERTION TRACE {func}\n{stack}", flush=True)
        return func(*args, **(kwargs or {}))


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Provide an integration module and its optional arguments")
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    with AssertionTrace():
        runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
