"""Resolve output and sequence contracts before transforming a local part."""

from dataclasses import dataclass
from typing import Literal

from torch import nn
from torch.distributed import ProcessGroup

from ..models.contracts import DecoderStructure
from ..models.execution import ModelExecutionCapabilities
from .plan import ParallelPlan
from .tensor_parallel.sharding import ShardingConfig


@dataclass(frozen=True, slots=True)
class ExecutionLayout:
    """External part boundary; internal attention gathers do not change it."""

    output: Literal["logits"]
    sequence_shard_axes: tuple[str, ...]
    vocabulary_sharded: bool
    vocabulary_size: int | None
    capabilities: ModelExecutionCapabilities
    chunked_loss: bool

    def __post_init__(self) -> None:
        if self.output != "logits":
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
        if self.chunked_loss and not self.capabilities.hidden_states:
            raise ValueError("Chunked loss requires hidden output support")

    def validate_model(self, model: nn.Module) -> None:
        """Reject transforms that invalidate the resolved execution declaration."""
        if getattr(model, "execution_capabilities", None) != self.capabilities:
            raise ValueError("Transformed model execution capabilities changed")
        if self.vocabulary_size is not None and (
            getattr(model, "vocab_size", None) != self.vocabulary_size
        ):
            raise ValueError("Transformed model vocabulary size changed")

    def loss_kwargs(self, tp_group: ProcessGroup | None) -> dict[str, object]:
        """Bind the declared output vocabulary to its reduction group."""
        if self.output != "logits":
            raise ValueError("Pipeline hidden output cannot be consumed by LM loss")
        if not self.vocabulary_sharded:
            return {}
        if tp_group is None or self.vocabulary_size is None:
            raise ValueError("Vocabulary-sharded output requires a TP group and size")
        return {"tp_group": tp_group, "global_vocab_size": self.vocabulary_size}


def resolve_execution_layout(
    model: nn.Module,
    *,
    tp_size: int,
    cp_size: int,
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
    head_path = None
    if isinstance(structure, DecoderStructure):
        head_path = structure.head
        if isinstance(model.get_submodule(head_path), nn.Identity):
            raise ValueError(
                "Declared logits output requires an owned output projection"
            )
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
    if chunked_loss and not capabilities.hidden_states:
        raise ValueError("Chunked loss requires model-declared hidden output")
    if not capabilities.logits:
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
        output="logits",
        sequence_shard_axes=tuple(
            axis for axis, size in (("cp", cp_size), ("tp", tp_size)) if size > 1
        ),
        vocabulary_sharded=vocabulary_sharded,
        vocabulary_size=vocabulary_size,
        capabilities=capabilities,
        chunked_loss=chunked_loss,
    )
