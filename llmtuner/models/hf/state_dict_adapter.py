"""Hugging Face safetensors mapping for :class:`HFTransformerModel`.

The wrapper stores a HF ``ForCausalLM`` below ``self.model``. Consequently its
state-dict FQNs have exactly one extra ``model.`` prefix compared with the
checkpoint produced by Transformers. No tensor conversion is required.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch.distributed as dist
from torch.distributed.checkpoint import HuggingFaceStorageReader

from ...components.checkpointer.checkpoint_keys import SAFETENSORS_INDEX


class HFTransformerStateDictAdapter:
    """Translate llmtuner wrapper FQNs to standard HF safetensors FQNs."""

    def __init__(self, model_config: Any, hf_assets_path: str | None) -> None:
        # Composite HF models store the trained decoder's weight ties here.
        self.model_config = getattr(model_config, "text_config", model_config)
        self.hf_assets_path = hf_assets_path
        self._hf_keys: set[str] | None = None
        self.fqn_to_index_mapping: dict[str, int] | None = None

        if hf_assets_path is None:
            return
        index_path = os.path.join(hf_assets_path, SAFETENSORS_INDEX)
        if not os.path.isfile(index_path):
            # Read only the header: single-file checkpoints omit tied heads
            # too, and need the same key coverage policy as indexed exports.
            tensor_path = os.path.join(hf_assets_path, "model.safetensors")
            if os.path.isfile(tensor_path):
                from safetensors import safe_open

                with safe_open(tensor_path, framework="pt") as tensors:
                    self._hf_keys = set(tensors.keys())
            return
        with open(index_path, encoding="utf-8") as stream:
            index = json.load(stream)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("HF checkpoint index requires a non-empty weight_map")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(filename, str)
            or not filename
            for key, filename in weight_map.items()
        ):
            raise ValueError(
                "HF weight_map keys and filenames must be non-empty strings"
            )
        self._hf_keys = set(weight_map)
        # HF filenames need not contain a shard number. Preserve their tensor
        # grouping while providing the writer's required contiguous 1..N IDs.
        shard_indices = {
            filename: shard_index
            for shard_index, filename in enumerate(sorted(set(weight_map.values())), 1)
        }
        self.fqn_to_index_mapping = {
            key: shard_indices[filename] for key, filename in weight_map.items()
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state = {}
        for key, value in state_dict.items():
            hf_key = key.removeprefix("model.")
            if hf_key in hf_state:
                raise ValueError(f"HF checkpoint key mapping collision: {hf_key!r}")
            hf_state[hf_key] = value
        if self._hf_keys is not None:
            requested = set(hf_state)
            if dist.is_available() and dist.is_initialized():
                # PP ranks own disjoint layer keys. Validate the whole model's
                # coverage while returning only this rank's tensor dictionary.
                keys_by_rank: list[set[str] | None] = [None] * dist.get_world_size()
                dist.all_gather_object(keys_by_rank, requested)
                requested = set().union(*keys_by_rank)
            missing = self._hf_keys - requested
            unexpected = requested - self._hf_keys
            # A tied lm_head is legitimately omitted from a HF checkpoint.
            if (
                getattr(self.model_config, "tie_word_embeddings", False)
                and "lm_head.weight" not in self._hf_keys
            ):
                unexpected.discard("lm_head.weight")
                hf_state.pop("lm_head.weight", None)
            if missing or unexpected:
                raise ValueError(
                    "HF checkpoint keys do not exactly cover the llmtuner model: "
                    f"missing={sorted(missing)[:8]}, "
                    f"unexpected={sorted(unexpected)[:8]}"
                )
        return hf_state

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = dict(hf_state_dict)
        if (
            "lm_head.weight" not in hf_state_dict
            and "model.embed_tokens.weight" in hf_state_dict
            and getattr(self.model_config, "tie_word_embeddings", False)
        ):
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]
        return {f"model.{key}": value for key, value in hf_state_dict.items()}

    def get_hf_storage_reader(
        self,
        path: str,
        *,
        from_quantized: bool = False,
        thread_count: int | None = None,
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            raise NotImplementedError("Quantized HF checkpoints are not supported")
        if thread_count is None:
            # Two workers overlap shard reads without heavy storage pressure.
            thread_count = 2
        return HuggingFaceStorageReader(path, thread_count=thread_count)
