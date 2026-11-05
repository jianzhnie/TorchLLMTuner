"""Process-group bootstrap and rank/world queries (vendored, de-mmengined).

``init_dist_pytorch`` is the trainer's PG entry point; the slurm/mpi variants
serve standalone scripts. Rank queries use the live process group when it exists
and single-process defaults otherwise; device selection reads launcher variables.
"""

# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import functools
import os
import subprocess
from collections.abc import Callable
from typing import Literal

import torch
import torch.multiprocessing as mp
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup

from .device import device_type, get_distributed_backend, set_device
from .tensor_tree import get_tensor_tree_device as get_tensor_tree_device
from .tensor_tree import move_tensor_tree as move_tensor_tree

try:
    import torch_npu
except ImportError as error:
    torch_npu = None
    NPU_IMPORT_ERROR = error

try:
    import torch_mlu
except ImportError as error:
    torch_mlu = None
    MLU_IMPORT_ERROR = error

try:
    import torch_musa
except ImportError as error:
    torch_musa = None
    MUSA_IMPORT_ERROR = error


try:
    import deepspeed
except ImportError as error:
    deepspeed = None
    DEEPSPEED_IMPORT_ERROR = error

try:
    import colossalai
except ImportError as error:
    colossalai = None
    COLOSSALAI_IMPORT_ERROR = error


# Importing the extension registers the SageMaker process-group backend.
try:
    import smdistributed.dataparallel.torch.torch_smddp as torch_smddp
except ImportError as error:
    torch_smddp = None
    SMDDP_IMPORT_ERROR = error


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_default_group() -> ProcessGroup | None:
    """Return the initialized default process group, when one exists.

    ``group.WORLD`` is the public handle for the default process group.  Keep
    the uninitialized case explicit so callers can retain the backend-neutral
    ``None`` behavior without reaching into ``distributed_c10d`` internals.
    """

    if not is_distributed():
        return None
    return torch_dist.group.WORLD


def infer_launcher() -> Literal["pytorch", "slurm", "mpi", "none"]:
    if "WORLD_SIZE" in os.environ:
        return "pytorch"
    elif "SLURM_NTASKS" in os.environ:
        return "slurm"
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return "mpi"
    else:
        return "none"


def init_dist(launcher, backend="nccl", init_backend="torch", **kwargs) -> None:
    """Initialize distributed environment.

    Note:
        The llmtuner trainer calls ``init_dist_pytorch`` directly (it does
        not want this wrapper's ``mp.set_start_method('spawn')`` side
        effect); this multi-launcher entry is kept for standalone scripts.
        On vendor accelerators the backend is derived from the device
        layer (``device.get_distributed_backend``), and the ``backend``
        argument is honored only on the CUDA path.

    Args:
        launcher (str): Way to launcher multi processes. Supported launchers
            are 'pytorch', 'mpi' and 'slurm'.
        backend (str): Communication Backends. Supported backends are 'nccl',
            'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    if launcher not in ("pytorch", "mpi", "slurm"):
        raise ValueError(f"Invalid launcher type: {launcher}")
    if launcher != "mpi":
        validate_init_backend(init_backend)
    if device_type == "cpu" and backend == "nccl":
        # 'nccl' is the default spelling, not a choice: on a CPU-only box it
        # can only fail, so fall back to the derived backend (gloo).
        backend = get_distributed_backend()
    timeout = kwargs.get("timeout", None)
    if timeout is not None:
        # If a timeout (in seconds) is specified, it must be converted
        # to a timedelta object before forwarding the call to
        # the respective backend, because they expect a timedelta object.
        try:
            kwargs["timeout"] = datetime.timedelta(seconds=timeout)
        except TypeError as exception:
            raise TypeError(
                f"Timeout for distributed training must be provided as "
                f"timeout in seconds, but we've received the type "
                f"{type(timeout)}. Please specify the timeout like this: "
                f"dist_cfg=dict(backend='nccl', timeout=1800)"
            ) from exception
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    if launcher == "pytorch":
        init_dist_pytorch(backend, init_backend=init_backend, **kwargs)
    elif launcher == "mpi":
        init_dist_mpi(backend, **kwargs)
    elif launcher == "slurm":
        init_dist_slurm(backend, init_backend=init_backend, **kwargs)


def validate_init_backend(init_backend: str) -> None:
    """Reject invalid bootstrap choices before setting devices or launcher state."""
    if device_type != "cuda" and init_backend != "torch":
        raise ValueError(
            f"init_backend={init_backend!r} is not supported on {device_type}"
        )
    if init_backend not in ("torch", "deepspeed", "colossalai"):
        raise ValueError(
            'supported "init_backend" values are "torch", "deepspeed" '
            f'and "colossalai", but got {init_backend}'
        )


def init_dist_pytorch(backend, init_backend="torch", **kwargs) -> None:
    """Initialize distributed environment with PyTorch launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    validate_init_backend(init_backend)
    rank = int(os.environ["RANK"])
    # LOCAL_RANK is set by `torch.distributed.launch` since PyTorch 1.1
    local_rank = int(os.environ["LOCAL_RANK"])
    if device_type not in ("cpu", "cuda"):
        # Vendor accelerators (npu / mlu / musa / xpu): one generic path --
        # the backend string is owned by ``device.py``'s map rather than a
        # second hardcoded table here, and every vendor takes LOCAL_RANK
        # (the vendored musa branch used the global rank, which misplaces
        # ranks on multi-node runs).
        set_device(torch.device(device_type, local_rank))
        torch_dist.init_process_group(
            backend=get_distributed_backend(),
            rank=rank,
            world_size=int(os.environ["WORLD_SIZE"]),
            **kwargs,
        )
    elif device_type == "cpu":
        # gloo/mpi runs have no device to set.
        if init_backend == "torch":
            torch_dist.init_process_group(backend=backend, **kwargs)
        else:
            raise ValueError(f"init_backend={init_backend!r} is not supported on CPU")
    else:
        torch.cuda.set_device(local_rank)

        if init_backend == "torch":
            torch_dist.init_process_group(backend=backend, **kwargs)
        elif init_backend == "deepspeed":
            if deepspeed is None:
                raise ImportError(
                    "init_backend='deepspeed' requires the deepspeed package"
                ) from DEEPSPEED_IMPORT_ERROR

            deepspeed.init_distributed(dist_backend=backend, **kwargs)
        elif init_backend == "colossalai":
            if colossalai is None:
                raise ImportError(
                    "init_backend='colossalai' requires the colossalai package"
                ) from COLOSSALAI_IMPORT_ERROR

            colossalai.launch_from_torch(backend=backend, **kwargs)


def init_dist_mpi(backend, **kwargs) -> None:
    """Initialize distributed environment with MPI launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    if backend == "smddp" and torch_smddp is None:
        error_type = (
            ModuleNotFoundError
            if isinstance(SMDDP_IMPORT_ERROR, ModuleNotFoundError)
            else ImportError
        )
        raise error_type(
            "Please use an Amazon SageMaker DLC to access smdistributed: "
            "https://github.com/aws/deep-learning-containers/blob/master"
            "/available_images.md#sagemaker-framework-containers"
            "-sm-support-only"
        ) from SMDDP_IMPORT_ERROR
    local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    if device_type not in ("cpu", "cuda"):
        set_device(torch.device(device_type, local_rank))
        backend = get_distributed_backend()
    elif device_type == "cuda":
        torch.cuda.set_device(local_rank)
    if "MASTER_PORT" not in os.environ:
        # 29500 is torch.distributed default port
        os.environ["MASTER_PORT"] = "29500"
    if "MASTER_ADDR" not in os.environ:
        raise KeyError("The environment variable MASTER_ADDR is not set")
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch_dist.init_process_group(backend=backend, **kwargs)


def init_dist_slurm(backend, port=None, init_backend="torch", **kwargs) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    validate_init_backend(init_backend)
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    # Not sure when this environment variable could be None, so use a fallback
    local_rank_env = os.environ.get("SLURM_LOCALID", None)
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
    else:
        num_gpus = (
            1 if device_type == "cpu" else getattr(torch, device_type).device_count()
        )
        local_rank = proc_id % num_gpus
    if "MASTER_ADDR" not in os.environ:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]],
            check=True,
            capture_output=True,
            text=True,
        )
        hosts = result.stdout.splitlines()
        if not hosts or not hosts[0].strip():
            raise RuntimeError("scontrol returned no hostnames for SLURM_NODELIST")
        os.environ["MASTER_ADDR"] = hosts[0].strip()
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    else:
        os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(proc_id)

    # Slurm has now published the same environment as torchrun. Reuse its
    # CPU/CUDA PyTorch bootstrap; ColossalAI retains its Slurm-specific entry.
    if init_backend == "torch" and device_type in ("cpu", "cuda"):
        init_dist_pytorch(backend, **kwargs)
        return

    if device_type not in ("cpu", "cuda"):
        set_device(torch.device(device_type, local_rank))
        torch_dist.init_process_group(backend=get_distributed_backend(), **kwargs)
    elif device_type == "cpu":
        raise ValueError(f"init_backend={init_backend!r} is not supported on CPU")
    else:
        torch.cuda.set_device(local_rank)

        if init_backend == "deepspeed":
            if deepspeed is None:
                raise ImportError(
                    "init_backend='deepspeed' requires the deepspeed package"
                ) from DEEPSPEED_IMPORT_ERROR

            deepspeed.init_distributed(dist_backend=backend, **kwargs)
        elif init_backend == "colossalai":
            if colossalai is None:
                raise ImportError(
                    "init_backend='colossalai' requires the colossalai package"
                ) from COLOSSALAI_IMPORT_ERROR

            colossalai.launch_from_slurm(
                backend=backend,
                host=os.environ["MASTER_ADDR"],
                port=os.environ["MASTER_PORT"],
                **kwargs,
            )


def get_backend(group: ProcessGroup | None = None) -> str | None:
    """Return the backend of the given process group.

    Note:
        Calling ``get_backend`` in non-distributed environment will return
        None.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific
            group is specified, the calling process must be part of
            :attr:`group`. Defaults to None.

    Returns:
        str or None: Return the backend of the given process group as a lower
        case string if in distributed environment, otherwise None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_backend(group)
    else:
        return None


def get_world_size(group: ProcessGroup | None = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def get_rank(group: ProcessGroup | None = None) -> int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """

    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0


def get_dist_info(group: ProcessGroup | None = None) -> tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size


def is_main_process(group: ProcessGroup | None = None) -> bool:
    """Whether the current rank of the given process group is equal to 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        bool: Return True if the current rank of the given process group is
        equal to 0, otherwise False.
    """
    return get_rank(group) == 0


def master_only(func: Callable) -> Callable:
    """Decorate those methods which should be executed in master process.

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Return decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


def barrier(group: ProcessGroup | None = None) -> None:
    """Synchronize all processes from the given process group.

    This collective blocks processes until the whole group enters this
    function.

    Note:
        Calling ``barrier`` in non-distributed environment will do nothing.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        torch_dist.barrier(group)


def get_comm_device(group: ProcessGroup | None = None) -> torch.device:
    """Return the device for communication among groups.

    Args:
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        torch.device: The device of backend.
    """
    backend = get_backend(group)
    if backend == "hccl":
        if torch_npu is None:
            raise ImportError(
                "HCCL communication requires torch-npu"
            ) from NPU_IMPORT_ERROR

        return torch.device("npu", torch.npu.current_device())
    elif backend == torch_dist.Backend.NCCL:
        return torch.device("cuda", torch.cuda.current_device())
    elif backend == "cncl":
        if torch_mlu is None:
            raise ImportError(
                "CNCL communication requires torch-mlu"
            ) from MLU_IMPORT_ERROR

        return torch.device("mlu", torch.mlu.current_device())
    elif backend == "smddp":
        return torch.device("cuda", torch.cuda.current_device())
    elif backend == "mccl":
        if torch_musa is None:
            raise ImportError("MCCL communication requires torch-musa") from (
                MUSA_IMPORT_ERROR
            )

        return torch.device("musa", torch_musa.current_device())
    elif backend == "xccl":
        return torch.device("xpu", torch.xpu.current_device())
    else:
        # GLOO and MPI backends use cpu device by default
        return torch.device("cpu")
