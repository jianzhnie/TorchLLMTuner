"""Compile config (``training.compile=True``), grafted onto TrainingConfig."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmtuner.errors import ConfigError


@dataclass(kw_only=True)
class CompileConfig:
    """Settings for ``training.compile=True`` -- how the model is compiled.

    Ported from torchtitan's ``CompileConfig``, minus its ``components``
    list (llmtuner compiles the model only; the loss has no compile path).
    The defaults are exactly llmtuner's historical behavior -- one whole-model
    ``torch.compile(model, backend="inductor")`` -- so an existing run that
    never touches this config is bitwise unchanged. Each non-default knob is
    independent of the others.
    """

    per_block: bool = field(
        default=False,
        metadata={
            "help": "Compile each decoder layer separately (fullgraph=True) "
            "instead of the model as a whole: the repeated block structure "
            "is traced once and its graph reused, so compile time scales "
            "with one block rather than the depth. False (default) keeps "
            "the whole-model compile."
        },
    )
    backend: str = field(
        default="inductor",
        metadata={
            "help": "torch.compile backend. 'inductor' (default) or "
            "'aot_eager'; on a flex-attention model 'aot_eager' is wrapped "
            "in regional_inductor so the flex regions still lower to "
            "inductor. Any other backend on a flex model is an error, "
            "since flex has no non-inductor lowering."
        },
    )
    enable_async_tensor_parallel: bool = field(
        default=False,
        metadata={
            "help": "Pipeline tensor-parallel collectives with the GEMMs "
            "inside compiled regions (Inductor's micro-pipeline pass). "
            "Requires training.compile=True, tensor_parallel_size > 1, and "
            "a torch carrying torch._inductor.config._micro_pipeline_tp "
            "plus symmetric-memory registration; every missing piece is a "
            "loud error, never a silent skip."
        },
    )

    def __post_init__(self) -> None:
        if not self.backend:
            raise ConfigError("compile.backend cannot be empty.")
