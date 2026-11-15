"""Resolve output and sequence contracts before transforming a local part."""

from dataclasses import dataclass
from typing import Literal

from torch import nn
from torch.distributed import ProcessGroup

from ..models.contracts import DecoderStructure
from ..models.execution import ModelExecutionCapabilities, PipelineStageIO
from .plan import ParallelPlan
from .stages import STAGE_ORDER
from .tensor_parallel.sharding import ShardingConfig


@dataclass(frozen=True, slots=True)
class ExecutionLayout:
    """External part boundary; internal attention gathers do not change it."""

    output: Literal["logits", "pipeline_hidden"]
    sequence_shard_axes: tuple[str, ...]
    vocabulary_sharded: bool
    vocabulary_size: int | None
    capabilities: ModelExecutionCapabilities
    chunked_loss: bool
    stages: tuple[str, ...]
    pipeline_io: PipelineStageIO | None = None

    def __post_init__(self) -> None:
        if self.output not in ("logits", "pipeline_hidden"):
            raise ValueError("Unknown model output layout")
        if not isinstance(self.capabilities, ModelExecutionCapabilities):
            raise TypeError("Execution layout requires model execution capabilities")
        if (
            type(self.vocabulary_sharded) is not bool
            or type(self.chunked_loss) is not bool
        ):
            raise TypeError("Execution layout flags must be boolean")
        if self.sequence_shard_axes not in ((), ("cp",), ("tp",), ("cp", "tp")):
            raise ValueError("Sequence shard axes must follow the CP/TP layout order")
        if self.vocabulary_size is not None and (
            type(self.vocabulary_size) is not int or self.vocabulary_size <= 0
        ):
            raise ValueError("Declared vocabulary size must be a positive integer")
        if self.vocabulary_sharded and (
            self.output != "logits"
            or "tp" not in self.sequence_shard_axes
            or self.vocabulary_size is None
        ):
            raise ValueError(
                "Vocabulary sharding requires TP logits and vocabulary size"
            )
        if self.pipeline_io is not None:
            if not isinstance(self.pipeline_io, PipelineStageIO):
                raise TypeError(
                    "Execution layout requires a valid pipeline I/O contract"
                )
            if self.pipeline_io.last != (self.output == "logits"):
                raise ValueError(
                    "Pipeline output layout disagrees with the last-stage role"
                )
        elif self.output == "pipeline_hidden":
            raise ValueError("Pipeline hidden output requires a stage I/O contract")
        if self.output == "logits" and not self.capabilities.logits:
            raise ValueError("Logits layout requires model-declared logits output")
        if self.chunked_loss and (
            self.pipeline_io is not None or not self.capabilities.hidden_states
        ):
            raise ValueError("Chunked loss requires hidden output without pipeline I/O")

    @property
    def input_kind(self) -> Literal["tokens", "pipeline_hidden"]:
        return (
            "pipeline_hidden"
            if self.pipeline_io is not None and not self.pipeline_io.first
            else "tokens"
        )

    def loss_kwargs(self, tp_group: ProcessGroup | None) -> dict[str, object]:
        """Bind the declared output vocabulary to its reduction group."""
        if self.output != "logits":
            raise ValueError("Pipeline hidden output cannot be consumed by LM loss")
        if not self.vocabulary_sharded:
            return {}
        if tp_group is None or self.vocabulary_size is None:
            raise ValueError("Vocabulary-sharded output requires a TP group and size")
        return {"tp_group": tp_group, "global_vocab_size": self.vocabulary_size}


def validate_execution_chain(layouts: tuple[ExecutionLayout, ...]) -> None:
    """Validate the complete chain in global stage order before construction.

    Every intermediate stage must emit the hidden representation consumed by
    the next stage, with matching width and dtype.  The check is deliberately
    metadata-only so it also covers custom pipeline partitions.
    """
    if not layouts:
        raise ValueError("Pipeline execution chain must contain at least one stage")
    if any(layout.pipeline_io is None for layout in layouts):
        raise ValueError("Pipeline execution layouts require stage I/O declarations")
    if layouts[0].input_kind != "tokens" or layouts[-1].output != "logits":
        raise ValueError("Pipeline chain must consume tokens and produce logits")
    for index, (left, right) in enumerate(zip(layouts, layouts[1:], strict=False)):
        if left.output != "pipeline_hidden" or right.input_kind != "pipeline_hidden":
            raise ValueError(
                f"Pipeline stages {index} and {index + 1} have incompatible "
                "execution boundary"
            )
        left_io, right_io = left.pipeline_io, right.pipeline_io
        if left_io.hidden_size != right_io.hidden_size:
            raise ValueError("Pipeline stages disagree on hidden activation width")
        if left_io.activation_dtype != right_io.activation_dtype:
            raise ValueError("Pipeline stages disagree on activation dtype")
        if left.sequence_shard_axes != right.sequence_shard_axes:
            raise ValueError("Pipeline stages disagree on sequence shard axes")


def resolve_execution_layout(
    model: nn.Module,
    *,
    tp_size: int,
    cp_size: int,
    pipeline: bool,
    tp_plan: ParallelPlan | None,
    chunked_loss: bool,
    compile_enabled: bool,
    activation_swap: bool,
) -> ExecutionLayout:
    """Check the supported stage chain from declarations before mutation.

    Layout-preserving CP/AC/swap/compile/FSDP retain the token boundary created
    by input preparation and TP. This describes existing execution, not arbitrary
    user-defined stage transformations.
    """
    capabilities = getattr(model, "execution_capabilities", None)
    if not isinstance(capabilities, ModelExecutionCapabilities):
        raise TypeError(
            "Model must declare execution_capabilities as ModelExecutionCapabilities"
        )
    structure = getattr(model, "decoder_structure", None)
    stage_io = getattr(model, "pipeline_stage_io", None)
    if stage_io is not None and not isinstance(stage_io, PipelineStageIO):
        raise TypeError("pipeline_stage_io must be a PipelineStageIO declaration")
    owns_output = stage_io.last if pipeline and stage_io is not None else True
    head_path = None
    if isinstance(structure, DecoderStructure):
        head_path = structure.head
        owns_output = not isinstance(model.get_submodule(head_path), nn.Identity)
    # The pre-partition model is also resolved with pipeline=True. Its stage
    # roles do not exist until partitioning; configured parts declare them.
    if pipeline and stage_io is not None:
        validate_pipeline_stage_io(model, first=stage_io.first, last=stage_io.last)
    vocabulary_sharded = False
    if tp_plan is not None and head_path is not None:
        for target in tp_plan.targets:
            if target.path == head_path and target.kind == "dense":
                if target.layout != "colwise":
                    raise ValueError(
                        "TP output head must preserve token shards and shard vocabulary"
                    )
                if not isinstance(target.spec, ShardingConfig):
                    raise TypeError("TP output projection requires a sharding contract")
                if target.spec.sequence_transform != "preserve":
                    raise ValueError(
                        "TP output projection changes token layout; loss labels remain "
                        "sequence-sharded and need an explicit output/target adapter"
                    )
                if tp_size > 1:
                    raise ValueError(
                        "Vocabulary-sharded loss requires identical token/label rows "
                        "across TP ranks; sequence-sharded inputs need an explicit "
                        "output/target adapter"
                    )
                vocabulary_sharded = True
    if chunked_loss and (pipeline or not capabilities.hidden_states):
        raise ValueError(
            "Chunked loss requires model-declared hidden output without PP"
        )
    if owns_output and not capabilities.logits:
        raise ValueError("Training and evaluation require model-declared logits output")
    if activation_swap and compile_enabled:
        raise ValueError("Activation swap cannot feed the compiled execution stage")
    if tp_size > 1 and tp_plan is None:
        raise ValueError("TP execution layout requires a resolved TP plan")
    vocabulary_size = getattr(model, "vocab_size", None)
    if vocabulary_sharded and (
        type(vocabulary_size) is not int or vocabulary_size < tp_size
    ):
        raise ValueError(
            "Vocabulary-sharded output requires a declared vocabulary size"
        )
    return ExecutionLayout(
        output="logits" if owns_output else "pipeline_hidden",
        sequence_shard_axes=tuple(
            axis for axis, size in (("cp", cp_size), ("tp", tp_size)) if size > 1
        ),
        vocabulary_sharded=vocabulary_sharded,
        vocabulary_size=vocabulary_size,
        capabilities=capabilities,
        chunked_loss=chunked_loss,
        stages=tuple(STAGE_ORDER),
        pipeline_io=stage_io,
    )


def validate_pipeline_stage_io(model: nn.Module, *, first: bool, last: bool) -> None:
    """Check configured roles against retained model registrations."""
    stage_io = getattr(model, "pipeline_stage_io", None)
    if not isinstance(stage_io, PipelineStageIO):
        raise TypeError("Declared pipeline models must provide PipelineStageIO")
    if (stage_io.first, stage_io.last) != (first, last):
        raise ValueError("Pipeline stage position disagrees with model I/O roles")
    structure = getattr(model, "decoder_structure", None)
    if structure is None:
        # Custom stages own their module roles through configure_pipeline_stage.
        return
    if not isinstance(structure, DecoderStructure):
        raise TypeError("Pipeline I/O validation requires DecoderStructure")
    for role, expected in (("embedding", first), ("norm", last), ("head", last)):
        module = model.get_submodule(getattr(structure, role))
        if (not isinstance(module, nn.Identity)) != expected:
            raise ValueError(f"Pipeline {role} ownership disagrees with stage I/O")
