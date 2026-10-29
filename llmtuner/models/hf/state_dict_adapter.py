"""Hugging Face safetensors mapping for :class:`HFTransformerModel`.

The wrapper stores a HF ``ForCausalLM`` below ``self.model``. Consequently its
state-dict FQNs have exactly one extra ``model.`` prefix compared with the
checkpoint produced by Transformers. Converted Kimi EP models additionally
map full grouped expert tensors back to individual HF expert projections.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import torch
import torch.distributed as dist

from ...accelerator.collectives import synchronize_local_errors
from ...components.checkpointer.checkpoint_keys import SAFETENSORS_INDEX
from ...components.checkpointer.hf_storage import IndexedHuggingFaceStorageReader


class HFTransformerStateDictAdapter:
    """Translate llmtuner wrapper FQNs to standard HF safetensors FQNs."""

    def __init__(
        self,
        model_config: Any,
        hf_assets_path: str | None,
        *,
        kimi_expert_parallel: bool = False,
        qwen3_expert_parallel: bool = False,
    ) -> None:
        # Composite HF models store the trained decoder's weight ties here.
        self.model_config = getattr(model_config, "text_config", model_config)
        self.kimi_expert_parallel = kimi_expert_parallel
        self.qwen3_expert_parallel = qwen3_expert_parallel
        if (
            qwen3_expert_parallel
            and getattr(self.model_config, "model_type", None) != "qwen3_moe"
        ):
            raise ValueError("qwen3_expert_parallel requires qwen3_moe model config")
        if (
            kimi_expert_parallel
            and getattr(self.model_config, "model_type", None) != "kimi_linear"
        ):
            raise ValueError("kimi_expert_parallel requires a kimi_linear model config")
        self.hf_assets_path = hf_assets_path
        self._hf_keys: set[str] | None = None
        self.fqn_to_index_mapping: dict[str, int] | None = None
        self.hf_export_fqn_to_index_mapping: dict[str, int] | None = None

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
                self._normalize_expert_keys()
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
        self.hf_export_fqn_to_index_mapping = dict(self.fqn_to_index_mapping)
        self._normalize_expert_keys()
        if not self.qwen3_expert_parallel:
            self.hf_export_fqn_to_index_mapping = self.fqn_to_index_mapping

    def _normalize_expert_keys(self) -> None:
        if (
            getattr(self.model_config, "model_type", None) == "qwen3_moe"
            and self._hf_keys is not None
        ):
            from ...components.checkpointer.hf_expert_storage import (
                grouped_expert_key,
                native_ep_expert_key,
            )

            grouped_keys = set()
            grouped_indices = {}
            for key in self._hf_keys:
                mapped = grouped_expert_key(key)
                target = mapped[0] if mapped is not None else key
                if self.qwen3_expert_parallel:
                    target = native_ep_expert_key(key)
                grouped_keys.add(target)
                if self.fqn_to_index_mapping is not None:
                    grouped_indices[target] = min(
                        grouped_indices.get(target, self.fqn_to_index_mapping[key]),
                        self.fqn_to_index_mapping[key],
                    )
            self._hf_keys = grouped_keys
            if self.fqn_to_index_mapping is not None:
                self.fqn_to_index_mapping = grouped_indices

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if self.kimi_expert_parallel:
            from ..kimi.checkpoint import convert_kimi_expert_state

            state_dict = convert_kimi_expert_state(
                state_dict,
                num_experts=self.model_config.num_experts,
                to_hf=True,
                coordinate_errors=True,
            )
        # Only dictionary mapping is local. Kimi tensor conversion above and
        # the global PP coverage gather below stay outside this boundary.
        with synchronize_local_errors("HF export key mapping"):
            hf_state = {}
            for key, value in state_dict.items():
                hf_key = key.removeprefix("model.")
                if self.qwen3_expert_parallel:
                    hf_key = hf_key.replace(".mlp.router.gate.", ".mlp.gate.")
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
        if getattr(self.model_config, "tie_word_embeddings", False):
            # All ranks with tied configuration participate, including PP parts
            # without either key. DTensor clone preserves placements locally.
            with synchronize_local_errors("HF export tied weights"):
                if (
                    "model.embed_tokens.weight" in hf_state
                    and "lm_head.weight" in hf_state
                ):
                    # Safetensors cannot serialize aliases within one shard.
                    # HF restores parameter sharing from its configuration.
                    hf_state["lm_head.weight"] = hf_state["lm_head.weight"].clone()
        return hf_state

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = dict(hf_state_dict)
        if (
            "lm_head.weight" not in hf_state_dict
            and "model.embed_tokens.weight" in hf_state_dict
            and getattr(self.model_config, "tie_word_embeddings", False)
        ):
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]
        state = {
            "model."
            + (
                key.replace(".mlp.gate.", ".mlp.router.gate.")
                if self.qwen3_expert_parallel
                else key
            ): value
            for key, value in hf_state_dict.items()
        }
        if self.kimi_expert_parallel:
            from ..kimi.checkpoint import convert_kimi_expert_state

            state = convert_kimi_expert_state(
                state,
                num_experts=self.model_config.num_experts,
                to_hf=False,
                coordinate_errors=True,
            )
        return state

    def to_hf_export(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Produce standard HF expert keys rather than virtual reader templates."""
        from ...components.checkpointer.chunked_tensor import ChunkedCheckpointTensor

        state = self.to_hf(state_dict)
        if not self.qwen3_expert_parallel:
            return state
        roles = {
            "w1_EFD": "gate_proj",
            "w3_EFD": "up_proj",
            "w2_EDF": "down_proj",
        }
        result = {}
        pending = []
        with synchronize_local_errors("Qwen3 HF export expert mapping"):
            for key, value in state.items():
                match = re.fullmatch(
                    r"(.+\.mlp\.)routed_experts\.inner_experts\.(w[123]_[A-Z]+)",
                    key,
                )
                if match is None:
                    targets = [key]
                else:
                    prefix, role = match.groups()
                    if role not in roles:
                        raise ValueError(f"Unknown Qwen3 expert projection: {key}")
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.ndim != 3
                        or value.shape[0] != self.model_config.num_experts
                    ):
                        raise ValueError(
                            f"Qwen3 HF export requires full expert stack: {key}"
                        )
                    targets = [
                        f"{prefix}experts.{index}.{roles[role]}.weight"
                        for index in range(self.model_config.num_experts)
                    ]
                for target in targets:
                    if target in result:
                        raise ValueError(f"Qwen3 HF export key collision: {target}")
                    result[target] = value if match is None else None
                if match is not None:
                    pending.append((targets, value))
        for targets, value in pending:
            if isinstance(value, ChunkedCheckpointTensor):
                for target in targets:
                    del result[target]
                for offset, local in value.chunks:
                    for index in range(local.shape[0]):
                        target = targets[offset[0] + index]
                        chunks = []
                        if target in result:
                            chunks.extend(result[target].chunks)
                        chunks.append((offset[1:], local[index]))
                        result[target] = ChunkedCheckpointTensor(
                            value.shape[1:], chunks
                        )
                continue
            for target, weight in zip(targets, value.unbind(0), strict=True):
                result[target] = weight
        return result

    def get_hf_storage_reader(
        self,
        path: str,
        *,
        from_quantized: bool = False,
        thread_count: int | None = None,
    ) -> IndexedHuggingFaceStorageReader:
        if from_quantized:
            raise NotImplementedError("Quantized HF checkpoints are not supported")
        if thread_count is None:
            # Two workers overlap shard reads without heavy storage pressure.
            thread_count = 2
        if getattr(self.model_config, "model_type", None) == "qwen3_moe":
            from ...components.checkpointer.hf_expert_storage import (
                GroupedExpertHuggingFaceStorageReader,
            )

            return GroupedExpertHuggingFaceStorageReader(
                path,
                num_experts=self.model_config.num_experts,
                thread_count=thread_count,
                native_ep_layout=self.qwen3_expert_parallel,
            )
        return IndexedHuggingFaceStorageReader(path, thread_count=thread_count)
