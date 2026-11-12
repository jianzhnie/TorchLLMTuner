"""Optional accelerator KDA providers; CPU recurrence belongs to the model.

Selection follows the requested tensor device. Missing accelerator dependencies
are errors, never permission to transfer tensors or use the CPU reference.
"""

from collections.abc import Callable
from typing import Any

import torch

from ...accelerator.device import is_device_type_available
from ...errors import EnvironmentUnsupportedError

NPU_KDA_IMPORT_ERROR = None
ACCELERATOR_KDA_IMPORT_ERROR = None

npu_chunk_kda = None
if is_device_type_available("npu"):
    # Optional Ascend packages may access torch.npu during import; CPU model
    # construction must not load an unavailable device's runtime.
    try:
        from triton_ascend_kernels.attention.fla.kda import chunk_kda as npu_chunk_kda
    except ImportError as error:
        NPU_KDA_IMPORT_ERROR = error

try:
    from fla.ops.kda import chunk_kda as accelerator_chunk_kda
except ImportError as error:
    accelerator_chunk_kda = None
    ACCELERATOR_KDA_IMPORT_ERROR = error


def require_kda_kernel(device: torch.device) -> Callable[..., Any]:
    """Resolve the accelerator provider without allocating tensors or running it."""
    if device.type == "npu":
        kernel, error = npu_chunk_kda, NPU_KDA_IMPORT_ERROR
        dependency = "triton_ascend_kernels.attention.fla.kda"
    elif device.type == "cuda":
        kernel, error = accelerator_chunk_kda, ACCELERATOR_KDA_IMPORT_ERROR
        dependency = "fla.ops.kda"
    else:
        raise EnvironmentUnsupportedError(
            f"Accelerator KDA does not support device {device.type}"
        )
    if kernel is None:
        raise EnvironmentUnsupportedError(
            f"Kimi KDA training on {device.type} requires {dependency}. "
            "Install its compatible accelerator runtime before constructing the model."
        ) from error
    return kernel
