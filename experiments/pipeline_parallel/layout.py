"""Pipeline-specific ownership metadata layered over generic local layouts."""

from dataclasses import dataclass

from experiments.pipeline_parallel.ownership import PipelineParameterOwnership
from llmtuner.parallel.assembly import ModelPartLayout as BaseModelPartLayout
from llmtuner.parallel.assembly import ModelPartLayoutSnapshot as BaseSnapshot


@dataclass(slots=True)
class ModelPartLayout(BaseModelPartLayout):
    pipeline_ownership: PipelineParameterOwnership | None = None

    def refresh(self, model) -> None:
        super(ModelPartLayout, self).refresh(model)
        if self.pipeline_ownership is not None:
            self.pipeline_ownership = self.pipeline_ownership.remap_parameters(
                self.replacements.parameter_targets
            )

    def validate(self, model) -> None:
        super(ModelPartLayout, self).validate(model)
        registered = set(self.parameter_bindings.values())
        if self.pipeline_ownership is not None and any(
            target not in registered
            for targets in self.pipeline_ownership.local_parameters.values()
            for target in targets
        ):
            raise ValueError("Pipeline parameter target is no longer registered")

    def snapshot(self):
        snapshot = super(ModelPartLayout, self).snapshot()
        return LayoutSnapshot(
            parameters=snapshot.parameters,
            topology=snapshot.topology,
            execution=snapshot.execution,
            pipeline_ownership=self.pipeline_ownership,
        )


@dataclass(frozen=True, slots=True, eq=False)
class LayoutSnapshot(BaseSnapshot):
    pipeline_ownership: PipelineParameterOwnership | None

    def validate(self, layout, *, consumer: str) -> None:
        super(LayoutSnapshot, self).validate(layout, consumer=consumer)
        if layout.pipeline_ownership is not self.pipeline_ownership:
            raise ValueError(
                f"{consumer} pipeline ownership changed; rebuild the consumer"
            )
