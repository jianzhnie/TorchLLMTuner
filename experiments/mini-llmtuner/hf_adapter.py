"""HuggingFace causal-LM adapters for the experiment protocol."""

from dataclasses import dataclass

import torch
from torch import nn

from .assembly import AssemblyResult, assemble_model
from .config import ParallelConfig


@dataclass(frozen=True, slots=True)
class HFModelContract:
    architecture: str
    tp_plan: dict[str, str]
    targets: tuple[tuple[str, str, str], ...]
    vocab_size: int


def _linear_plan(model: nn.Module, architecture: str) -> HFModelContract:
    """Build a plan from actual modules, avoiding version-specific wildcards."""
    plan: dict[str, str] = {}
    targets = []
    for path, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = path.rsplit(".", 1)[-1]
            if leaf in {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}:
                plan[path] = "colwise"
            elif leaf in {"o_proj", "down_proj"}:
                plan[path] = "rowwise"
        if path.endswith("layers"):
            targets.append((path, "decoder", "replicated"))
    embed = "model.embed_tokens"
    try:
        model.get_submodule(embed)
    except AttributeError:
        embed = "model.model.embed_tokens"
    try:
        model.get_submodule(embed)
        plan[embed] = "embedding"
    except AttributeError:
        pass
    plan.setdefault("lm_head", "replicated")
    return HFModelContract(
        architecture=architecture,
        tp_plan=plan,
        targets=tuple(targets),
        vocab_size=model.config.vocab_size,
    )


class HFModelAdapter(nn.Module):
    """Adapt a HuggingFace causal LM to the prototype's model protocol."""

    def __init__(self, model: nn.Module, architecture: str) -> None:
        super().__init__()
        self.model = model
        contract = _linear_plan(model, architecture)
        self.contract = HFModelContract(
            architecture=contract.architecture,
            tp_plan={
                f"model.{path}": layout for path, layout in contract.tp_plan.items()
            },
            targets=tuple(
                (f"model.{path}", role, layout)
                for path, role, layout in contract.targets
            ),
            vocab_size=contract.vocab_size,
        )

    def validate_parallel(self, *, tp_degree: int, ep_degree: int) -> None:
        # Module-name inference alone cannot guarantee attention/DTensor semantics.
        if tp_degree != 1 or ep_degree != 1:
            raise NotImplementedError(
                "HF adapter TP/EP needs an architecture-specific plan"
            )

    @property
    def layers(self):
        layers = getattr(self.model, "model", self.model)
        return layers.layers

    def forward(
        self, input_ids: torch.Tensor, *, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, position_ids=position_ids, use_cache=False
        ).logits

    def preprocess_inputs(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return batch

    def parallelize(self, plan: ParallelConfig, *, world_size: int) -> AssemblyResult:
        return assemble_model(self, plan, world_size=world_size)


class LlamaModelAdapter(HFModelAdapter):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model, "llama")


class DeepSeekV3ModelAdapter(HFModelAdapter):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model, "deepseek_v3")
