"""Explicit structural declarations shared by model and training adapters.

Paths are relative to the causal model, not the Trainer wrapper. Binding is
metadata-only and must be repeated after structural transformations such as PP.
No modules are registered here and no parameter storage is copied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import nn


def registered_submodule(model: nn.Module, path: str) -> nn.Module:
    """Bind a state-dict module path without following unregistered attributes."""
    module = model
    for name in path.split(".") if path else ():
        child = module._modules.get(name)
        if not isinstance(child, nn.Module):
            raise AttributeError(f"Declared module {path!r} is not registered")
        module = child
    return module


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
    activation: str = "silu"
    router_logits_in_output: bool = False
    num_expert_groups: int | None = None
    num_limited_groups: int | None = None
    shared_expert_gate: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.router, str) or not self.router or "." in self.router:
            raise ValueError("MoE router must be a direct child name")
        if self.shared_expert is not None and (
            not isinstance(self.shared_expert, str)
            or not self.shared_expert
            or "." in self.shared_expert
        ):
            raise ValueError("MoE shared expert must be a direct child name")
        if self.shared_expert_gate is not None and (
            self.shared_expert is None
            or not isinstance(self.shared_expert_gate, str)
            or not self.shared_expert_gate
            or "." in self.shared_expert_gate
        ):
            raise ValueError(
                "MoE shared expert gate requires a shared expert and direct child name"
            )
        if (
            type(self.num_experts) is not int
            or type(self.top_k) is not int
            or self.num_experts < 1
            or not 0 < self.top_k <= self.num_experts
        ):
            raise ValueError("MoE expert count and top-k are inconsistent")
        if type(self.router_logits_in_output) is not bool:
            raise ValueError("MoE router_logits_in_output must be boolean")
        if type(self.route_norm) is not bool:
            raise ValueError("MoE route_norm must be boolean")
        if (
            isinstance(self.route_scale, bool)
            or not isinstance(self.route_scale, (int, float))
            or not math.isfinite(self.route_scale)
            or self.route_scale <= 0
        ):
            raise ValueError("MoE route_scale must be finite and positive")
        groups, limited = self.num_expert_groups, self.num_limited_groups
        if (groups is None) != (limited is None):
            raise ValueError(
                "MoE expert groups and limited groups must be set together"
            )
        if groups is not None:
            if (
                type(groups) is not int
                or type(limited) is not int
                or groups < 1
                or not 0 < limited <= groups
                or self.num_experts % groups
            ):
                raise ValueError("MoE expert group geometry is inconsistent")
            if self.top_k > limited * (self.num_experts // groups):
                raise ValueError("MoE top-k exceeds selected expert group capacity")
        if not isinstance(self.activation, str) or not self.activation.strip():
            raise ValueError("MoE activation must be a non-empty name")
        if self.expert_layout != "fused_gate_up":
            raise ValueError(f"Unsupported MoE expert layout {self.expert_layout!r}")
        if not isinstance(self.score_func, str) or self.score_func not in {
            "softmax",
            "sigmoid",
        }:
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
        if self.shared_expert is not None and not isinstance(shared, nn.Module):
            raise ValueError(
                f"MoE shared expert {self.shared_expert!r} must be a child module"
            )
        if self.shared_expert_gate is not None and not isinstance(
            block._modules.get(self.shared_expert_gate), nn.Module
        ):
            raise ValueError(
                f"MoE shared expert gate {self.shared_expert_gate!r} "
                "must be a child module"
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
        decoder = registered_submodule(model, self.decoder)
        embedding = registered_submodule(model, self.embedding)
        blocks = self.bind_blocks(model)
        norm = registered_submodule(model, self.norm)
        head = registered_submodule(model, self.head)
        return BoundDecoderStructure(self, decoder, embedding, blocks, norm, head)

    def bind_blocks(self, model: nn.Module) -> nn.ModuleList:
        """Bind current local blocks without requiring PP-removed edge modules."""
        blocks = registered_submodule(model, self.blocks)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(f"Decoder blocks {self.blocks!r} must be a ModuleList")
        return blocks

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
        blocks = self.bind_blocks(model)
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
