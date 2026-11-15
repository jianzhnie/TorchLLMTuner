"""Version record for native training checkpoints, read before live state."""

import json
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from torch import nn

from ....accelerator.collectives import synchronize_local_errors
from ....accelerator.distributed import get_world_size
from ....models.execution import ModelExecutionCapabilities
from ....parallel.topology import ParallelTopologySnapshot

if TYPE_CHECKING:
    from ....parallel.assembly import ModelPartLayout


class CheckpointFormatState:
    """Immutable writer/reader contract; external model-only imports omit it."""

    VERSION = 2

    def __init__(
        self,
        *,
        model_contracts: str = "[]",
        topology: ParallelTopologySnapshot | None = None,
    ) -> None:
        self._model_contracts = model_contracts
        self._topology = json.dumps(
            None if topology is None else asdict(topology), sort_keys=True
        )

    @classmethod
    def capture(
        cls,
        model_parts: list[nn.Module],
        *,
        runtime_layouts: tuple["ModelPartLayout", ...] = (),
    ) -> "CheckpointFormatState":
        """Capture declared logical capabilities, independent of PP ownership.

        All ranks contribute their local parts. Duplicate declarations disappear,
        so changing the number of pipeline parts does not itself change the
        contract. This does not establish topology or tensor-layout compatibility.
        Undeclared standalone models contribute no inferred capabilities.
        """
        with synchronize_local_errors("Checkpoint model contract capture"):
            topology = None
            if runtime_layouts:
                if len(runtime_layouts) != len(model_parts):
                    raise ValueError("Checkpoint layouts must cover every model part")
                topologies = {layout.topology for layout in runtime_layouts}
                if len(topologies) != 1:
                    raise ValueError(
                        "Checkpoint model parts declare different topologies"
                    )
                topology = topologies.pop()
                if topology is not None and not isinstance(
                    topology, ParallelTopologySnapshot
                ):
                    raise TypeError("Checkpoint requires a resolved topology snapshot")
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
            dist.all_gather_object(peers, (declarations, topology))
            if any(peer[1] != topology for peer in peers):
                raise ValueError("Checkpoint topology declarations differ across ranks")
            declarations = sorted({item for peer, _ in peers for item in peer})
        return cls(model_contracts=json.dumps(declarations), topology=topology)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": "llmtuner.native",
            "version": self.VERSION,
            "model_contracts": self._model_contracts,
            "topology": self._topology,
        }

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("name") != "llmtuner.native":
            raise ValueError("Checkpoint does not declare the native training format")
        version = state_dict.get("version")
        if type(version) is not int or version != self.VERSION:
            raise ValueError(
                f"Unsupported native checkpoint format version: {version!r}"
            )

        # Source provenance is validated independently of the destination.
        # RNG, cursor ownership and tensor-layout adapters own transition policy.
        try:
            topology = json.loads(state_dict["topology"])
            if topology is not None:
                expected = {item.name for item in fields(ParallelTopologySnapshot)}
                if not isinstance(topology, dict) or topology.keys() != expected:
                    raise ValueError("Incomplete topology snapshot")
                ParallelTopologySnapshot(**topology)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Checkpoint source topology record is invalid") from error

        if state_dict.get("model_contracts") != self._model_contracts:
            raise ValueError(
                "Checkpoint model execution contracts differ from this run"
            )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
