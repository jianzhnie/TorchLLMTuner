"""Version record for native training checkpoints, read before live state."""

import json
from dataclasses import asdict
from typing import Any

import torch.distributed as dist
from torch import nn

from ....accelerator.collectives import synchronize_local_errors
from ....accelerator.distributed import get_world_size
from ....models.execution import ModelExecutionCapabilities


class CheckpointFormatState:
    """Immutable writer/reader contract; external model-only imports omit it."""

    VERSION = 1

    def __init__(self, *, model_contracts: str = "[]") -> None:
        self._model_contracts = model_contracts

    @classmethod
    def capture(cls, model_parts: list[nn.Module]) -> "CheckpointFormatState":
        """Capture declared logical capabilities, independent of PP ownership.

        All ranks contribute their local parts. Duplicate declarations disappear,
        so changing the number of pipeline parts does not itself change the
        contract. This does not establish topology or tensor-layout compatibility.
        Undeclared standalone models contribute no inferred capabilities.
        """
        with synchronize_local_errors("Checkpoint model contract capture"):
            local = set()
            for model in model_parts:
                capabilities = getattr(model, "execution_capabilities", None)
                configuration = getattr(model, "checkpoint_configuration", None)
                if capabilities is None and configuration is None:
                    continue
                if capabilities is not None and not isinstance(
                    capabilities, ModelExecutionCapabilities
                ):
                    raise TypeError("Checkpoint model requires explicit capabilities")
                vocabulary = getattr(model, "vocab_size", None)
                if vocabulary is not None and (
                    type(vocabulary) is not int or vocabulary <= 0
                ):
                    raise ValueError("Checkpoint vocabulary size must be positive")
                family = getattr(getattr(model, "config", None), "model_type", None)
                if family is not None and not isinstance(family, str):
                    raise TypeError("Checkpoint model_type must be a string")
                if configuration is not None and (
                    not isinstance(configuration, dict)
                    or any(not isinstance(key, str) for key in configuration)
                ):
                    raise TypeError(
                        "Model checkpoint_configuration must be a string-key dict"
                    )
                local.add(
                    json.dumps(
                        {
                            "family": family,
                            "configuration": configuration,
                            "capabilities": (
                                asdict(capabilities)
                                if capabilities is not None
                                else None
                            ),
                            "vocabulary_size": vocabulary,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
        declarations = sorted(local)
        if get_world_size() > 1:
            peers = [None] * get_world_size()
            dist.all_gather_object(peers, declarations)
            declarations = sorted({item for peer in peers for item in peer})
        return cls(model_contracts=json.dumps(declarations))

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": "llmtuner.native",
            "version": self.VERSION,
            "model_contracts": self._model_contracts,
        }

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("name") != "llmtuner.native":
            raise ValueError("Checkpoint does not declare the native training format")
        version = state_dict.get("version")
        if type(version) is not int or version != self.VERSION:
            raise ValueError(
                f"Unsupported native checkpoint format version: {version!r}"
            )

        if state_dict.get("model_contracts") != self._model_contracts:
            raise ValueError(
                "Checkpoint model execution contracts differ from this run"
            )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
