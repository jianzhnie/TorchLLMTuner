"""Explicit structural declarations shared by model and training adapters.

Paths are relative to the causal model, not the Trainer wrapper. Binding is
metadata-only and must be repeated after structural transformations such as PP.
No modules are registered here and no parameter storage is copied.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class BoundDecoderStructure:
    """Ephemeral instance references; never attach as registered submodules."""

    declaration: DecoderStructure
    decoder: nn.Module
    embedding: nn.Module
    blocks: nn.ModuleList
    norm: nn.Module
    head: nn.Module


@dataclass(frozen=True, slots=True)
class MoEBlockContract:
    """Model-owned routing and fused-expert semantics for parallel execution."""

    router: str
    num_experts: int
    top_k: int
    route_norm: bool
    route_scale: float = 1.0
    score_func: str = "softmax"
    shared_expert: str | None = None
    expert_layout: str = "fused_gate_up"
    router_logits_in_output: bool = False
    num_expert_groups: int | None = None
    num_limited_groups: int | None = None

    def __post_init__(self) -> None:
        if not self.router or "." in self.router:
            raise ValueError("MoE router must be a direct child name")
        if self.shared_expert is not None and (
            not self.shared_expert or "." in self.shared_expert
        ):
            raise ValueError("MoE shared expert must be a direct child name")
        if self.num_experts < 1 or not 0 < self.top_k <= self.num_experts:
            raise ValueError("MoE expert count and top-k are inconsistent")
        if self.expert_layout != "fused_gate_up":
            raise ValueError(f"Unsupported MoE expert layout {self.expert_layout!r}")
        if self.score_func not in {"softmax", "sigmoid"}:
            raise ValueError(f"Unsupported MoE score function {self.score_func!r}")

    def bind(self, block: nn.Module) -> tuple[nn.Module, str, nn.Module | None]:
        """Resolve direct children without registering or inspecting the block."""
        router = block._modules.get(self.router)
        if not isinstance(router, nn.Module):
            raise ValueError(f"MoE router {self.router!r} must be a child module")
        shared = (
            None
            if self.shared_expert is None
            else block._modules.get(self.shared_expert)
        )
        return router, self.shared_expert or "shared_expert", shared


@dataclass(frozen=True, slots=True)
class DecoderStructure:
    """Model-owned paths; consumers must not guess alternative spellings."""

    decoder: str
    embedding: str
    blocks: str
    norm: str
    head: str
    attention: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed declarations before constructing or binding a model."""
        for role in ("decoder", "embedding", "blocks", "norm", "head"):
            path = getattr(self, role)
            if role == "decoder" and path == "":
                continue  # torch uses the empty path for the root module.
            if not isinstance(path, str) or any(not part for part in path.split(".")):
                raise ValueError(f"Decoder {role} must be a non-empty module path")
        for path in (self.embedding, self.blocks, self.norm):
            parent, _, name = path.rpartition(".")
            if parent != self.decoder or not name:
                raise ValueError(f"{path!r} must be a direct child of {self.decoder!r}")
        if self.attention is not None and (
            not isinstance(self.attention, str)
            or not self.attention
            or "." in self.attention
        ):
            raise ValueError("Decoder attention must be a direct child name or None")

    def bind(self, model: nn.Module) -> BoundDecoderStructure:
        decoder = model.get_submodule(self.decoder)
        embedding = model.get_submodule(self.embedding)
        blocks = model.get_submodule(self.blocks)
        norm = model.get_submodule(self.norm)
        head = model.get_submodule(self.head)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(f"Decoder blocks {self.blocks!r} must be a ModuleList")
        return BoundDecoderStructure(self, decoder, embedding, blocks, norm, head)

    def bind_attentions(self, model: nn.Module) -> tuple[nn.Module, ...]:
        """Bind only declared direct attention children, without model mutation.

        Read the current block collection so PP replacement cannot leave cached
        whole-model references. An absent declaration is unsupported, not a cue
        to search common attribute names.
        """
        if (
            not isinstance(self.attention, str)
            or not self.attention
            or "." in self.attention
        ):
            raise ValueError("CP requires a declared direct attention child name")
        blocks = model.get_submodule(self.blocks)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(f"Decoder blocks {self.blocks!r} must be a ModuleList")
        attentions = []
        for block_id, block in blocks._modules.items():
            attention = block._modules.get(self.attention)
            if not isinstance(attention, nn.Module):
                raise ValueError(
                    f"Block {self.blocks}.{block_id} requires declared attention "
                    f"child {self.attention!r}"
                )
            attentions.append(attention)
        return tuple(attentions)
