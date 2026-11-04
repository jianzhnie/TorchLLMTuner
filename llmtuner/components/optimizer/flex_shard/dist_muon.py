# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DistMuon layout planning, construction, and distributed optimizer execution."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Literal, cast, overload

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate
from torch.optim import Optimizer

from llmtuner.config.optimizer_layout import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
)

from .muon_layout import (
    _ParameterComputeLayout,
    _resolve_storage_to_compute_transition,
)
from .muon_math import (
    _adjust_muon_learning_rate,
    _apply_muon_update,
    _compute_muon_direction,
    _matrix_batch_views_from_shape,
    _MatrixBatchView,
    _prepare_muon_input,
)
from .muon_redistribution import _resolve_muon_redistribution_plans
from .optimizer_reshard_runtime import _BucketedRedistributionRuntime
from .optimizer_reshard_schedule import (
    _bind_bucket_configs,
    _BucketSpec,
    _build_bucket_plans,
    _device_mesh_ranks,
    _dtensor_storage_region_for_participant,
    _LocalBucketPlan,
    _ParticipantPartition,
    _validate_bucket_plans_across_ranks,
)

__all__ = [
    "DistMuon",
]


def _normalize_param_groups(
    params: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy parameter groups and materialize their aligned values."""
    normalized_param_groups = []
    for param_group in params:
        normalized_group = dict(param_group)
        params_value = normalized_group.get("params", ())
        normalized_params = (
            (params_value,) if isinstance(params_value, Tensor) else tuple(params_value)
        )
        param_names_value = normalized_group.get("param_names")
        normalized_param_names = (
            () if param_names_value is None else tuple(param_names_value)
        )
        if param_names_value is None or len(normalized_params) != len(
            normalized_param_names
        ):
            raise ValueError("params and param_names must be aligned")
        normalized_group["params"] = normalized_params
        normalized_group["param_names"] = normalized_param_names
        normalized_param_groups.append(normalized_group)

    return normalized_param_groups


def _validate_compute_sharding_configuration(
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
) -> None:
    """Validate per-parameter compute sharding configuration."""
    for fqn, compute_layout in compute_sharding_by_fqn.items():
        if not isinstance(fqn, str):
            raise ValueError("compute_sharding_by_fqn keys must be strings")
        if type(compute_layout) is not ComputeLayout:
            raise ValueError("compute_sharding_by_fqn values must be ComputeLayout")


def _initialize_dist_muon(
    optimizer: DistMuon,
    *,
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    bucket_configs: Sequence[BucketConfig],
) -> None:
    """Initialize FlexShard for one newly constructed DistMuon.

    Every group must provide aligned ``params`` and ``param_names``. Every
    local parameter FQN must have one entry in ``compute_sharding_by_fqn``;
    extra compute-sharding entries for parameters on other pipeline stages are
    ignored. Parameter groups, compute layouts, and bucket configuration are
    frozen because optimizer state and collectives depend on them.
    """
    _validate_compute_sharding_configuration(compute_sharding_by_fqn)

    for param_group in optimizer.param_groups:
        group_params = tuple(param_group["params"])
        raw_param_names = param_group.get("param_names")
        param_names = () if raw_param_names is None else tuple(raw_param_names)
        if raw_param_names is None or len(group_params) != len(param_names):
            raise ValueError("params and param_names must be aligned")
        for _param, fqn in zip(group_params, param_names, strict=True):
            if fqn not in compute_sharding_by_fqn:
                raise ValueError(f"missing compute sharding for Muon parameter {fqn!r}")

    tensor_device = optimizer._validate_parameter_storage()
    compute_layouts = optimizer._build_parameter_compute_layouts(
        compute_sharding_by_fqn
    )
    optimizer._specs = _bind_bucket_configs(
        tuple(bucket_configs),
        compute_layouts,
        get_fqn=lambda layout: layout.fqn,
        get_storage_dtensor=lambda layout: layout.param,
        requires_redistribution=lambda layout: not layout.storage_is_compute_ready,
        get_redistribution_storage_mesh_axis=lambda layout: (
            layout.redistribution_storage_mesh_axis
        ),
    )
    optimizer._initialize_plan(compute_layouts)
    optimizer._validate_plan_across_ranks()
    optimizer._redistribution_runtime = _BucketedRedistributionRuntime[
        _ParameterComputeLayout
    ](tensor_device)
    optimizer._redistribution_runtime.reserve_buffers(
        optimizer._bucket_plans,
        local_tensor_spec=optimizer._local_tensor_spec,
    )
    optimizer.register_load_state_dict_pre_hook(_before_load_state_dict, prepend=True)
    optimizer.register_load_state_dict_post_hook(_after_load_state_dict, prepend=True)


class DistMuon(Optimizer):
    """Muon optimizer with FlexShard redistribution.

    Parameter groups, FQNs, storage layouts, compute layouts, and bucket plans
    are frozen after resharding is applied. Every configured parameter must
    have a layout-compatible DTensor gradient before each rank enters
    ``step()``.

    Matrix-batch compute views use batched BF16 kernels. They implement the
    same mathematical update as ``torch.optim.Muon`` running one matrix at a
    time, but bitwise equality across the two kernel schedules is not part of
    the contract.

    ``ns_device`` selects where Newton-Schulz runs. ``None`` keeps the storage
    device, where each parameter's local shard lives (the host under FSDP2 CPU
    offload). ``"accelerator"`` copies each compute view to the current
    accelerator, of any type, runs Newton-Schulz there and copies the result
    back.
    Momentum, parameters, gradients and redistribution buffers stay where they
    are, so this only changes anything when the parameters live on the host.
    """

    _specs: tuple[_BucketSpec, ...]
    _matrix_views_by_fqn: dict[str, tuple[_MatrixBatchView, ...]]
    _direction_row_scale_by_fqn: dict[str, Tensor]
    _redistribution_runtime: _BucketedRedistributionRuntime[_ParameterComputeLayout]
    _param_groups_frozen: bool

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        compute_sharding_by_fqn: Mapping[str, ComputeLayout],
        bucket_configs: Sequence[BucketConfig],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
        ns_device: Literal["accelerator"] | None = None,
    ) -> None:
        self._ns_device = self._resolve_ns_device(ns_device)
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        self._first_step_validated = False
        self._param_groups_frozen = False
        super().__init__(_normalize_param_groups(params), defaults)
        self.validate_param_groups(self.param_groups)
        self._param_groups_frozen = True
        _initialize_dist_muon(
            self,
            compute_sharding_by_fqn=compute_sharding_by_fqn,
            bucket_configs=bucket_configs,
        )

    @staticmethod
    def _resolve_ns_device(ns_device: str | None) -> torch.device | None:
        """Return the Newton-Schulz device, or ``None`` to keep the storage device."""
        if ns_device is None:
            return None
        if ns_device != "accelerator":
            raise ValueError(
                f"ns_device must be 'accelerator' or None, got {ns_device!r}"
            )
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        if accelerator is None:
            raise ValueError("ns_device='accelerator' requires an accelerator")
        return torch.device(accelerator.type, torch.accelerator.current_device_index())

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._preflight_step()
        self._redistribution_runtime.run(
            self._bucket_plans,
            local_tensor_spec=self._local_tensor_spec,
            prepare=self._prepare_local,
            compute=self._compute_update,
            finalize=self._apply_update,
        )
        return loss

    def initialize_state(self) -> None:
        """Create checkpoint destinations locally, without executing an update.

        Checkpoint callers coordinate allocation errors themselves. Running a
        synthetic step here would enter redistribution collectives that peers
        with existing state may skip.
        """
        for layout in self._parameter_compute_layouts:
            self._validate_momentum(layout)
        pending = [
            (layout.param, torch.zeros_like(layout.param))
            for layout in self._parameter_compute_layouts
            if "momentum_buffer" not in self.state.get(layout.param, {})
        ]
        # Publish only after all local allocations succeed, so retries retain
        # existing moments and do not observe partially initialized state.
        for parameter, momentum in pending:
            self.state[parameter]["momentum_buffer"] = momentum

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._param_groups_frozen:
            raise RuntimeError("DistMuon parameter groups are frozen")
        super().add_param_group(param_group)

    @staticmethod
    def validate_param_groups(groups: Iterable[Mapping[str, Any]]) -> None:
        """Validate live or restored options without state changes or collectives."""
        for group_index, group in enumerate(groups):
            ns_steps = group["ns_steps"]
            coefficients = group["ns_coefficients"]
            if (
                group.get("fused")
                or group.get("foreach")
                or any(
                    isinstance(group[name], bool)
                    or not isinstance(group[name], int | float)
                    or not math.isfinite(group[name])
                    or group[name] < 0
                    for name in ("lr", "weight_decay", "momentum", "eps")
                )
                or group["eps"] == 0
                or not isinstance(group["nesterov"], bool)
                or isinstance(ns_steps, bool)
                or not isinstance(ns_steps, int)
                or not 0 <= ns_steps < 100
                or not isinstance(coefficients, tuple | list)
                or len(coefficients) != 3
                or not all(
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in coefficients
                )
                or group["adjust_lr_fn"]
                not in (None, "original", "match_rms_adamw", "spectral_unclamped")
            ):
                raise ValueError(f"unsupported DistMuon group {group_index}")

    def _validate_parameter_storage(self) -> torch.device:
        local_devices = set()
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, DTensor):
                    raise TypeError("DistMuon requires DTensor parameters")
                local_device = param.to_local().device
                local_devices.add(local_device)
        if len(local_devices) != 1:
            raise ValueError("DistMuon requires one device per process")
        return local_devices.pop()

    def _build_parameter_compute_layouts(
        self,
        compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    ) -> tuple[_ParameterComputeLayout, ...]:
        parameters = []
        seen_names = set()
        seen_params = set()
        for group_index, group in enumerate(self.param_groups):
            params = group["params"]
            names = group["param_names"]
            for fqn, param in zip(names, params, strict=True):
                if fqn in seen_names or id(param) in seen_params:
                    raise ValueError(f"duplicate Muon parameter {fqn!r}")
                seen_names.add(fqn)
                seen_params.add(id(param))
                parameters.append((group_index, fqn, param))

        compute_layouts = []
        for group_index, fqn, param in parameters:
            if param.ndim < 2:
                raise ValueError(
                    f"DistMuon requires matrix parameters with at least 2 "
                    f"dimensions; {fqn!r} has "
                    f"shape {tuple(param.shape)}"
                )
            compute_layout = compute_sharding_by_fqn[fqn]
            mesh_names = param.device_mesh.mesh_dim_names
            assert mesh_names is not None
            for axis, name in enumerate(mesh_names):
                placement = param.placements[axis]
                if (
                    param.device_mesh.size(axis) > 1
                    and not isinstance(placement, Replicate)
                    and name not in compute_layout.shardings_by_mesh_axis
                ):
                    raise NotImplementedError(
                        f"DistMuon parameter {fqn!r} is sharded on mesh axis "
                        f"{name!r}, but its ComputeLayout omits that axis; "
                        "declare an explicit compute layout for every sharded axis"
                    )
            resolved_transition = _resolve_storage_to_compute_transition(
                fqn,
                param,
                compute_layout,
            )
            compute_dtype = compute_layout.compute_dtype
            if compute_dtype is None:
                compute_dtype = param.dtype
            if compute_dtype is not torch.bfloat16:
                raise NotImplementedError(
                    f"{fqn!r} would compute in {compute_dtype}: DistMuon runs "
                    "Newton-Schulz in torch.bfloat16, set compute_dtype="
                    "torch.bfloat16 on its ComputeLayout"
                )
            compute_layouts.append(
                _ParameterComputeLayout(
                    fqn=fqn,
                    param=param,
                    group_index=group_index,
                    storage_mesh_ranks=_device_mesh_ranks(param.device_mesh),
                    storage_layout_signature=_storage_layout_signature(param),
                    local_storage_signature=_local_storage_signature(param.to_local()),
                    compute_sharding=resolved_transition.compute_sharding,
                    storage_to_compute_transition=resolved_transition.storage_to_compute_transition,
                    resolved_compute_layout_signature=(
                        resolved_transition.resolved_compute_layout_signature
                    ),
                    redistribution_storage_mesh_axis=(
                        resolved_transition.redistribution_storage_mesh_axis
                    ),
                    compute_dtype=compute_dtype,
                )
            )
        return tuple(compute_layouts)

    def _initialize_plan(
        self,
        compute_layouts: Sequence[_ParameterComputeLayout],
    ) -> None:
        ns_steps_by_group = tuple(group["ns_steps"] for group in self.param_groups)
        result = _build_bucket_plans(
            compute_layouts,
            self._specs,
            get_fqn=lambda item: item.fqn,
            get_storage_dtensor=lambda item: item.param,
            requires_redistribution=lambda item: not item.storage_is_compute_ready,
            resolve_redistribution_plans=partial(
                _resolve_muon_redistribution_plans,
                ns_steps_by_group=ns_steps_by_group,
            ),
            get_compute_dtype=lambda item: item.compute_dtype,
        )

        def build_views(
            layout: _ParameterComputeLayout,
            compute_partition: _ParticipantPartition | None,
        ) -> tuple[_MatrixBatchView, ...]:
            param = layout.param
            compute_shape = (
                param.to_local().shape
                if compute_partition is None
                else torch.Size(compute_partition.tensor_shape)
            )
            compute_sharding = layout.compute_sharding
            if type(compute_sharding) is BlockShard:
                if compute_partition is None:
                    region = _dtensor_storage_region_for_participant(
                        param, param.device_mesh.get_rank()
                    )
                else:
                    (region,) = compute_partition.logical_regions
                assert region.shape == tuple(compute_shape) and region.offsets[1] == 0
                return _matrix_batch_views_from_shape(
                    compute_shape,
                    matrix_row_sizes=compute_sharding.block_sizes,
                    logical_row_start=region.offsets[0],
                )
            if not compute_shape.numel():
                return ()
            return (
                _MatrixBatchView(
                    shape=compute_shape,
                    strides=tuple(
                        math.prod(compute_shape[dim + 1 :])
                        for dim in range(len(compute_shape))
                    ),
                    offset=0,
                ),
            )

        matrix_views_by_fqn = {}
        direction_row_scale_by_fqn = {}
        for bucket in result.plans:
            if isinstance(bucket, _LocalBucketPlan):
                for item in bucket.items:
                    matrix_views_by_fqn[item.fqn] = build_views(item, None)
            else:
                for item in bucket.unredistributed_items:
                    matrix_views_by_fqn[item.fqn] = build_views(item, None)
                for item, plan in zip(
                    bucket.redistributed_items,
                    bucket.redistribution_plans,
                    strict=True,
                ):
                    matrix_views_by_fqn[item.fqn] = build_views(
                        item, plan.compute_partition(bucket.group.local_participant)
                    )
                    row_scale = _direction_row_scale(
                        item, adjust_lr_fn=self._group(item)["adjust_lr_fn"]
                    )
                    if row_scale is not None:
                        direction_row_scale_by_fqn[item.fqn] = row_scale
        self._bucket_plans = result.plans
        self._parameter_compute_layouts = result.ordered_items
        self._matrix_views_by_fqn = matrix_views_by_fqn
        self._direction_row_scale_by_fqn = direction_row_scale_by_fqn

    def _validate_plan_across_ranks(self) -> None:
        _validate_bucket_plans_across_ranks(
            self._bucket_plans,
            item_signature=self._plan_item_signature,
        )

    def _plan_item_signature(
        self, compute_layout: _ParameterComputeLayout
    ) -> tuple[Any, ...]:
        return (
            compute_layout.fqn,
            tuple(compute_layout.param.shape),
            tuple(compute_layout.param.stride()),
            str(compute_layout.param.dtype),
            compute_layout.param.to_local().device.type,
            compute_layout.storage_is_compute_ready,
            compute_layout.compute_sharding,
            compute_layout.resolved_compute_layout_signature,
            _device_mesh_ranks(compute_layout.param.device_mesh),
            tuple(map(str, compute_layout.param.placements)),
            self._group_signature(compute_layout),
        )

    def _group(self, compute_layout: _ParameterComputeLayout) -> dict[str, Any]:
        return self.param_groups[compute_layout.group_index]

    def _group_signature(
        self, compute_layout: _ParameterComputeLayout
    ) -> tuple[Any, ...]:
        group = self._group(compute_layout)
        return tuple(
            group[key]
            for key in (
                "lr",
                "weight_decay",
                "momentum",
                "nesterov",
                "ns_coefficients",
                "eps",
                "ns_steps",
                "adjust_lr_fn",
            )
        )

    def _preflight_step(self) -> None:
        """Fail the local worker before bucket communication on invalid input.

        TorchTitan's elastic launcher terminates peer workers after this error
        escapes. Do not add a validation collective to the optimizer hot path.
        """
        initialize_state = not self._first_step_validated
        missing_gradients = []
        changed_parameter_storage_fqn = None
        changed_gradient_storage_fqn = None
        gradients = [] if initialize_state else None
        for compute_layout in self._parameter_compute_layouts:
            if (
                changed_parameter_storage_fqn is None
                and compute_layout.storage_is_compute_ready
                and _local_storage_signature(compute_layout.param.to_local())
                != compute_layout.local_storage_signature
            ):
                changed_parameter_storage_fqn = compute_layout.fqn

            grad = compute_layout.param.grad
            if grad is None:
                missing_gradients.append(compute_layout.fqn)
            elif not isinstance(grad, DTensor) or not self._has_storage_layout(
                grad, compute_layout
            ):
                if changed_gradient_storage_fqn is None:
                    changed_gradient_storage_fqn = compute_layout.fqn
            elif gradients is not None:
                gradients.append((compute_layout, grad))

        if missing_gradients:
            raise RuntimeError(
                "DistMuon requires every configured gradient before "
                f"step(); missing gradients: {missing_gradients}"
            )
        if changed_parameter_storage_fqn is not None:
            raise RuntimeError(
                f"parameter local storage changed for "
                f"{changed_parameter_storage_fqn!r}; "
                "rebuild DistMuon"
            )
        if changed_gradient_storage_fqn is not None:
            raise RuntimeError(
                f"gradient storage layout changed for {changed_gradient_storage_fqn!r}"
            )

        if gradients is not None:
            for compute_layout, _grad in gradients:
                self._validate_momentum(compute_layout)

        # State creation happens only after every gradient and existing state
        # tensor has passed validation, so a deterministic input error cannot
        # partially update an earlier bucket.
        if gradients is not None:
            for compute_layout, grad in gradients:
                self._momentum(compute_layout, grad)
            self._first_step_validated = True

    @staticmethod
    def _has_storage_layout(
        tensor: DTensor, compute_layout: _ParameterComputeLayout
    ) -> bool:
        mesh_matches = (
            tensor.device_mesh is compute_layout.param.device_mesh
            or _device_mesh_ranks(tensor.device_mesh)
            == compute_layout.storage_mesh_ranks
        )
        return mesh_matches and (
            _storage_layout_signature(tensor) == compute_layout.storage_layout_signature
        )

    def _validate_momentum(self, compute_layout: _ParameterComputeLayout) -> None:
        momentum = self.state.get(compute_layout.param, {}).get("momentum_buffer")
        if momentum is None:
            return
        if not isinstance(momentum, DTensor) or not self._has_storage_layout(
            momentum, compute_layout
        ):
            raise RuntimeError(
                f"momentum storage layout changed for {compute_layout.fqn!r}"
            )

    def _momentum(
        self, compute_layout: _ParameterComputeLayout, grad: DTensor
    ) -> DTensor:
        state = self.state[compute_layout.param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(
                grad, memory_format=torch.preserve_format
            )
        return state["momentum_buffer"]

    def _local_gradient_and_momentum(
        self, compute_layout: _ParameterComputeLayout
    ) -> tuple[Tensor, Tensor, DTensor, dict[str, Any]]:
        grad = cast(DTensor, compute_layout.param.grad)
        momentum_state = cast(
            DTensor,
            self.state[compute_layout.param]["momentum_buffer"],
        )
        local_reference = compute_layout.param.to_local()
        if compute_layout.storage_is_compute_ready:
            local_reference = local_reference.detach()
        local_grad = grad.to_local().view_as(local_reference)
        local_momentum = momentum_state.to_local().view_as(local_reference)
        group = self._group(compute_layout)
        return local_grad, local_momentum, momentum_state, group

    def _prepare_local(
        self, compute_layout: _ParameterComputeLayout, out: Tensor
    ) -> None:
        grad, momentum, momentum_state, group = self._local_gradient_and_momentum(
            compute_layout
        )
        _prepare_muon_input(
            grad,
            momentum,
            momentum=group["momentum"],
            nesterov=group["nesterov"],
            out=out,
        )
        torch.autograd.graph.increment_version(momentum_state)

    def _compute_update(
        self, compute_layout: _ParameterComputeLayout, compute: Tensor
    ) -> None:
        if compute.numel() == 0:
            return
        group = self._group(compute_layout)
        # Stage onto the Newton-Schulz device only when the compute view lives
        # elsewhere, e.g. on the host under CPU offload.
        staged = (
            compute
            if self._ns_device is None or compute.device == self._ns_device
            else compute.to(self._ns_device, non_blocking=False)
        )
        _compute_muon_direction(
            staged,
            matrix_views=self._matrix_views_by_fqn[compute_layout.fqn],
            lr_reference_shape=(
                None
                if not compute_layout.storage_is_compute_ready
                else compute_layout.lr_reference_shape
            ),
            adjust_lr_fn=group["adjust_lr_fn"],
            ns_coefficients=group["ns_coefficients"],
            ns_steps=group["ns_steps"],
            eps=group["eps"],
        )
        if staged is not compute:
            compute.copy_(staged, non_blocking=False)

    def _apply_update(
        self, compute_layout: _ParameterComputeLayout, direction: Tensor
    ) -> None:
        group = self._group(compute_layout)
        local_param = compute_layout.param.to_local()
        if compute_layout.storage_is_compute_ready:
            local_param = local_param.detach()
        row_scale = self._direction_row_scale_by_fqn.get(compute_layout.fqn)
        if row_scale is not None:
            direction.mul_(row_scale)
        _apply_muon_update(
            local_param,
            direction,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            adjust_lr_fn=group["adjust_lr_fn"],
            compute_matrix_shape=compute_layout.lr_reference_shape,
        )
        torch.autograd.graph.increment_version(compute_layout.param)

    @staticmethod
    def _local_tensor_spec(
        compute_layout: _ParameterComputeLayout,
    ) -> tuple[torch.Size, torch.dtype, torch.device]:
        assert compute_layout.storage_is_compute_ready
        tensor = compute_layout.param.to_local().detach()
        return tensor.shape, tensor.dtype, tensor.device


def _block_row_spans(
    block_shard: BlockShard, row_start: int, row_end: int
) -> Iterable[tuple[int, int]]:
    """Yield ``(block_rows, span_rows)`` for the blocks overlapping a row range."""
    period = sum(block_shard.block_sizes)
    row = row_start
    while row < row_end:
        block_start = (row // period) * period
        for block_rows in block_shard.block_sizes:
            block_end = block_start + block_rows
            if row < block_end:
                span = min(block_end, row_end) - row
                yield block_rows, span
                row += span
                if row >= row_end:
                    break
            block_start = block_end


def _direction_row_scale(
    compute_layout: _ParameterComputeLayout, *, adjust_lr_fn: str | None
) -> Tensor | None:
    """Per-row factor the storage side applies to a redistributed direction."""
    compute_sharding = compute_layout.compute_sharding
    if type(compute_sharding) is not BlockShard:
        return None
    param = compute_layout.param
    reference_ratio = _adjust_muon_learning_rate(
        1.0, adjust_lr_fn, compute_layout.lr_reference_shape
    )
    columns = param.shape[-1]
    scale_by_block_rows = {
        block_rows: (
            _adjust_muon_learning_rate(1.0, adjust_lr_fn, (block_rows, columns))
            / reference_ratio
        )
        for block_rows in compute_sharding.block_sizes
    }
    if all(scale == 1.0 for scale in scale_by_block_rows.values()):
        return None
    region = _dtensor_storage_region_for_participant(
        param, param.device_mesh.get_rank()
    )
    row_start = region.offsets[0]
    scales = [
        scale_by_block_rows[block_rows]
        for block_rows, span in _block_row_spans(
            compute_sharding, row_start, row_start + region.shape[0]
        )
        for _ in range(span)
    ]
    local = param.to_local()
    return torch.tensor(scales, dtype=torch.float32, device=local.device).view(
        -1, *([1] * (local.ndim - 1))
    )


def _before_load_state_dict(optimizer: Optimizer, state_dict: dict[str, Any]) -> None:
    # Direct users bypass the container's coordinated validation boundary.
    # Reject malformed options before PyTorch replaces any state or groups.
    cast(DistMuon, optimizer).validate_param_groups(state_dict["param_groups"])


def _after_load_state_dict(optimizer: Optimizer) -> None:
    muon = cast(DistMuon, optimizer)
    # Optimizer.load_state_dict restores group values such as ns_steps after
    # construction, and those values affect compute planning and buffer sizes.
    muon.validate_param_groups(muon.param_groups)
    muon._initialize_plan(muon._parameter_compute_layouts)
    muon._redistribution_runtime.reserve_buffers(
        muon._bucket_plans,
        local_tensor_spec=muon._local_tensor_spec,
    )
    # A previously used optimizer must validate the restored momentum and
    # incoming gradients again before its first resumed update.
    muon._first_step_validated = False
    if not getattr(muon, "_defer_load_plan_validation", False):
        muon._validate_plan_across_ranks()


def _local_storage_signature(tensor: Tensor) -> tuple[Any, ...]:
    return (
        tensor.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _storage_layout_signature(tensor: DTensor) -> tuple[Any, ...]:
    local = tensor.to_local()
    # Empty shards address no elements, and autograd can choose different strides.
    local_strides = tuple(local.stride()) if local.numel() else ()
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.placements,
        tuple(local.shape),
        local_strides,
        local.dtype,
        local.device,
        local.is_contiguous(),
    )
