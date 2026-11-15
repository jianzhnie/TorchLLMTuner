"""Model-owned execution capabilities, independent of parallel backends."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelExecutionCapabilities:
    """Supported output requests; hidden output uses skip_lm_head=True."""

    logits: bool = True
    hidden_states: bool = False

    def __post_init__(self) -> None:
        if type(self.logits) is not bool or type(self.hidden_states) is not bool:
            raise TypeError("Model execution capabilities must be boolean")


@dataclass(frozen=True, slots=True)
class PipelineStageIO:
    """Model-owned token/hidden boundary for a configured pipeline part.

    Token leading dimensions may be flat or batch/sequence. Intermediate
    activations preserve those dimensions and append hidden_size.
    """

    first: bool
    last: bool
    hidden_size: int

    def __post_init__(self) -> None:
        if type(self.first) is not bool or type(self.last) is not bool:
            raise TypeError("Pipeline boundary roles must be boolean")
        if type(self.hidden_size) is not int or self.hidden_size <= 0:
            raise ValueError("Pipeline hidden size must be a positive integer")
