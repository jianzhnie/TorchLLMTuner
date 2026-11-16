from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PipelineStageIO:
    """Model-owned token/hidden boundary for a configured pipeline part.

    Token leading dimensions may be flat or batch/sequence. Intermediate
    activations preserve those dimensions and append hidden_size.
    """

    first: bool
    last: bool
    hidden_size: int
    activation_dtype: torch.dtype

    def __post_init__(self) -> None:
        if type(self.first) is not bool or type(self.last) is not bool:
            raise TypeError("Pipeline boundary roles must be boolean")
        if type(self.hidden_size) is not int or self.hidden_size <= 0:
            raise ValueError("Pipeline hidden size must be a positive integer")
        if not isinstance(self.activation_dtype, torch.dtype):
            raise TypeError("Pipeline activation dtype must be torch.dtype")
        if not self.activation_dtype.is_floating_point:
            raise TypeError("Pipeline activation dtype must be floating point")

    def validate_input(self, inputs: torch.Tensor) -> None:
        """Validate token or hidden geometry before model-side computation."""
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Pipeline input must be a tensor")
        if self.first:
            if inputs.ndim not in (1, 2):
                raise ValueError("Pipeline token input must have one or two dimensions")
            if inputs.dtype not in (torch.int32, torch.int64):
                raise TypeError("Pipeline token input must contain integer token IDs")
        else:
            self._validate_hidden(inputs)

    def validate_output(
        self,
        outputs: torch.Tensor,
        *,
        token_shape: tuple[int, ...],
        output_width: int | None = None,
    ) -> None:
        """Check preserved token dimensions and the declared hidden boundary."""
        if not isinstance(outputs, torch.Tensor):
            raise TypeError("Pipeline output must be a tensor")
        if tuple(outputs.shape[:-1]) != token_shape or outputs.ndim < 2:
            raise ValueError("Pipeline output must preserve input token dimensions")
        if not outputs.is_floating_point():
            raise TypeError("Pipeline output must be floating point")
        if output_width is not None:
            if type(output_width) is not int or output_width <= 0:
                raise ValueError("Pipeline output width must be a positive integer")
            if outputs.shape[-1] != output_width:
                raise ValueError("Pipeline output disagrees with declared output width")
        if not self.last:
            self._validate_hidden(outputs)

    def _validate_hidden(self, hidden: torch.Tensor) -> None:
        if hidden.ndim not in (2, 3) or hidden.shape[-1] != self.hidden_size:
            raise ValueError("Pipeline hidden activation disagrees with declared width")
        if hidden.dtype != self.activation_dtype:
            raise TypeError("Pipeline hidden activation disagrees with declared dtype")
