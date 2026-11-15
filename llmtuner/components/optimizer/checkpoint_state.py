"""Shared FQN checkpoint state for training optimizers and EMA averages."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from torch import nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from ...accelerator.collectives import synchronize_local_errors
from ...parallel.assembly import (
    ModelPartLayout,
    capture_model_part_layouts,
    validate_model_part_layouts,
)
from ...parallel.expert_parallel.checkpointing import (
    expert_shard_map,
)
from ..checkpointer.core.layout_conversion import (
    export_parallel_state,
    resolve_restore_layout,
    restore_parallel_state,
)
from .state_dict import collect_local_optimizer_state, unflatten_optim_state_dict


class OptimizerCheckpointState(Stateful):
    """Serialize owned state groups without exposing optimizer mutation methods."""

    model_parts: list[nn.Module]
    optimizers: Sequence[Optimizer]
    runtime_layouts: tuple[ModelPartLayout, ...]
    sharded_tensor_parallel: bool
    sharded_expert_parallel: bool
    _bound_layout_parameters: tuple[dict[str, nn.Parameter], ...] | None

    def _bind_runtime_layouts(
        self,
        layouts: tuple[ModelPartLayout, ...],
        *,
        validate_restore_groups: Callable[[Iterable[Mapping[str, Any]]], None]
        | None = None,
    ) -> None:
        """Bind final ownership within the caller's initialization boundary.

        Optional restore policy validation must be local and non-mutating. It
        runs for every saved group before any optimizer receives restored values.
        """
        self._validate_restore_groups = validate_restore_groups
        self.runtime_layouts = tuple(layouts) or capture_model_part_layouts(
            self.model_parts
        )
        self._bound_layout_parameters = None
        self._validate_runtime_layouts()
        self._bound_layout_parameters = tuple(
            dict(layout.parameter_bindings) for layout in self.runtime_layouts
        )

    def _validate_runtime_layouts(self) -> None:
        """Check the same final ownership before grouping, save, and restore."""
        bound = self._bound_layout_parameters
        if bound is not None and len(bound) != len(self.runtime_layouts):
            raise ValueError("Optimizer runtime layouts changed after construction")
        if len(self.runtime_layouts) != len(self.model_parts):
            raise ValueError("Optimizer layouts must match model parts")
        validate_model_part_layouts(self.model_parts, self.runtime_layouts)
        if bound is not None:
            for layout, parameters in zip(self.runtime_layouts, bound, strict=True):
                layout.validate_parameter_snapshot(parameters, consumer="Optimizer")

    def state_dict(self) -> dict[str, Any]:
        """A flat, FQN-keyed state dict covering every inner optimizer.

        Side effect: if an inner optimizer has not stepped yet,
        ``init_optim_state`` materializes its state first (a zero-gradient,
        zero-lr step) so DCP has tensors to read, and to write into on load. It
        leaves parameters alone, and is a no-op once state exists. All
        distributed ranks must participate: local serialization errors are
        coordinated before any TP/EP checkpoint gathering. State-dict and
        initialization hooks must not communicate inside this local phase.
        """
        with synchronize_local_errors("Optimizer checkpoint construction"):
            self._validate_runtime_layouts()
            result = collect_local_optimizer_state(self.optimizers)
        return export_parallel_state(
            result,
            model_parts=self.model_parts,
            expert_shards=(
                {}
                if self.sharded_tensor_parallel and self.sharded_expert_parallel
                else expert_shard_map(self.model_parts)
            ),
            sharded_tensor_parallel=self.sharded_tensor_parallel,
            sharded_expert_parallel=self.sharded_expert_parallel,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore checkpoint values after all ranks validate local state.

        All distributed ranks must participate. Validation failure prevents
        applying checkpoint values; lazy optimizer state may be initialized.
        Local load and optimizer plan-rebuild errors are coordinated across the
        world before any rank validates a rebuilt plan over its optimizer mesh.
        Other load hooks must be collective-free. Plan validation errors retain
        their mesh-local collective contract.
        Exceptions from load hooks or tensor copies are not atomic and require
        restoring the complete training checkpoint before continuing.

        Optimizers with collective restore validation must provide both
        load_local_state_dict(state) and validate_loaded_state(). The first
        must not communicate; the second runs only after all local loads pass.
        Mesh participants must agree on their validator ordering and groups.
        """
        with synchronize_local_errors("Optimizer restore layout validation"):
            self._validate_runtime_layouts()
            if not isinstance(state_dict, dict):
                raise TypeError("Optimizer checkpoint state must be a dictionary")
            restore_layout = resolve_restore_layout(
                state_dict,
                sharded_tensor_parallel=self.sharded_tensor_parallel,
                sharded_expert_parallel=self.sharded_expert_parallel,
            )
        # init_optim_state must run first: the unflattening step reads each
        # optimizer's live state to learn which state tensors to expect, so a
        # fresh optimizer would find nothing to write into.
        local_state = None
        local_state_validated = restore_layout.requires_local_state
        if local_state_validated:
            with synchronize_local_errors("Optimizer shard destination construction"):
                local_state = collect_local_optimizer_state(self.optimizers)
        state_dict = restore_parallel_state(
            state_dict,
            local_state=local_state,
            model_parts=self.model_parts,
            expert_shards=(
                {}
                if self.sharded_tensor_parallel and self.sharded_expert_parallel
                else expert_shard_map(self.model_parts)
            ),
            layout=restore_layout,
            coordinate_expert_errors=True,
        )
        # TP/EP layout communication above stays outside this local boundary.
        # Every rank must validate before peers apply state and enter the next
        # checkpoint component's collectives. State initialization hooks must
        # be local; communication inside hooks cannot be coordinated here.
        loaders: list[Callable[[dict[str, Any]], None]] = []
        mesh_validators: list[Callable[[], None]] = []
        with synchronize_local_errors("Optimizer checkpoint validation"):
            # Dense restore has not built local destinations yet. Sharded paths
            # already collected them before layout collectives; flattening them
            # again here only duplicates work over every optimizer moment.
            if not local_state_validated:
                collect_local_optimizer_state(self.optimizers)
            optimizer_states = [
                unflatten_optim_state_dict(optimizer, flat_state=state_dict)
                for optimizer in self.optimizers
            ]
            for optimizer, optimizer_state in zip(
                self.optimizers, optimizer_states, strict=True
            ):
                local_load = getattr(optimizer, "load_local_state_dict", None)
                mesh_validate = getattr(optimizer, "validate_loaded_state", None)
                if local_load is not None or mesh_validate is not None:
                    if not callable(local_load) or not callable(mesh_validate):
                        raise TypeError(
                            "Optimizer must provide callable load_local_state_dict "
                            "and validate_loaded_state together"
                        )
                    loaders.append(local_load)
                    mesh_validators.append(mesh_validate)
                else:
                    loaders.append(optimizer.load_state_dict)
                # Optional optimizer-owned preflight: collective-free and
                # non-mutating, before any optimizer receives checkpoint values.
                validate_groups = getattr(optimizer, "validate_param_groups", None)
                if validate_groups is not None:
                    if not callable(validate_groups):
                        raise TypeError(
                            "Optimizer validate_param_groups must be callable"
                        )
                    validate_groups(optimizer_state["param_groups"])
            if self._validate_restore_groups is not None:
                # Check every optimizer before any load hook applies values.
                # Keep policy failures inside the rank-coordinated boundary.
                for optimizer_state in optimizer_states:
                    self._validate_restore_groups(optimizer_state["param_groups"])
        # Local restore must finish everywhere before optimizer-owned mesh
        # validation. Ranks without validators still join the final boundary.
        # Ordinary hooks and local plan rebuilds must not communicate here.
        # Loading is not atomic on local failure.
        with synchronize_local_errors("Optimizer checkpoint application"):
            for load, optimizer_state in zip(loaders, optimizer_states, strict=True):
                load(optimizer_state)

        # PP stages without local experts skip these subgroup collectives.
        # Mesh participants must expose matching optimizer plans/order.
        plan_error: BaseException | None = None
        for validate_mesh in mesh_validators:
            try:
                validate_mesh()
            except BaseException as error:
                if plan_error is None:
                    plan_error = error
        # Subgroup validation may fail on only some pipeline/replica groups.
        # Rejoin the world before any rank advances to the next checkpoint
        # component. The failed load is not rolled back.
        with synchronize_local_errors("Optimizer mesh plan validation"):
            if plan_error is not None:
                raise plan_error
