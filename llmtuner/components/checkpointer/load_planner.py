"""Plan strict DCP loads, including supported legacy checkpoint formats."""

from __future__ import annotations

import io

import torch
from torch.distributed.checkpoint._nested_dict import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.planner import ReadItem

from ...utils.logging import get_logger
from ...utils.module_names import canonical_fqn
from .checkpoint_keys import DATALOADER, OPTIMIZER, RNG_STATE

logger = get_logger(__name__)


class CheckpointLoadPlanner(DefaultLoadPlanner):
    """Load legacy layout/RNG metadata and compiled names with strict state keys."""

    def create_local_plan(self):
        assert self.metadata is not None
        saved_keys = self.metadata.state_dict_metadata
        # A flat planner otherwise shares this dictionary with its caller.
        self.state_dict = dict(self.state_dict)
        self.compiled_destinations: dict[str, str] = {}
        # Older synthetic loaders stored one shared cursor. Map that scalar to
        # this rank's destination; new rank-qualified checkpoints remain strict.
        steps_key = f"{DATALOADER}.steps"
        if steps_key in saved_keys and self.flatten_state_dict:
            for key in list(self.state_dict):
                if key.startswith(steps_key + "."):
                    self.state_dict[steps_key] = self.state_dict.pop(key)
                    self.mappings[steps_key] = self.mappings.pop(key)
                    logger.warning(
                        "Legacy synthetic checkpoint has one shared cursor; "
                        "rank-local positions cannot be recovered."
                    )
        saved_names: dict[str, str] = {}
        for saved_key in saved_keys:
            name = canonical_fqn(saved_key, wrapper_segments=("_orig_mod",))
            maximum_base = name.removesuffix(".max_exp_avg_sq") + ".exp_avg_sq"
            if name in self.state_dict or (
                name.startswith(f"{OPTIMIZER}.state.")
                and name.endswith(".max_exp_avg_sq")
                and maximum_base in self.state_dict
            ):
                if name in saved_names:
                    raise ValueError(
                        "Ambiguous compiled checkpoint keys: "
                        f"{saved_names[name]!r} and {saved_key!r} map to {name!r}"
                    )
                saved_names[name] = saved_key
        for name, saved_key in saved_names.items():
            if name == saved_key or name not in self.state_dict:
                continue
            # Keep live state canonical. Tensor destinations retain their identity;
            # byte records must write through the original canonical dictionary path.
            self.state_dict[saved_key] = self.state_dict.pop(name)
            self.compiled_destinations[saved_key] = name
            if self.flatten_state_dict:
                self.mappings[saved_key] = self.mappings.pop(name)
        # Adam's saved AMSGrad option determines whether a maximum moment
        # exists. DCP plans tensor reads before loading that option, so adapt
        # destinations using metadata instead of the fresh optimizer's default.
        suffix = ".max_exp_avg_sq"
        for key in list(self.state_dict):
            if key.startswith(f"{OPTIMIZER}.state.") and key.endswith(suffix):
                if key not in saved_keys:
                    del self.state_dict[key]
                    if self.flatten_state_dict:
                        path = self.mappings[key]
                        parent = self.original_state_dict
                        for segment in path[:-1]:
                            parent = parent[segment]
                        del parent[path[-1]]
                    else:
                        self.original_state_dict.pop(key, None)
        for key in saved_keys:
            if not key.startswith(f"{OPTIMIZER}.state.") or not key.endswith(suffix):
                continue
            if key in self.state_dict:
                continue
            base = key[: -len(suffix)] + ".exp_avg_sq"
            if base not in self.state_dict:
                continue  # Another pipeline stage or optimizer owns this state.
            destination = torch.empty_like(self.state_dict[base])
            self.state_dict[key] = destination
            if self.flatten_state_dict:
                path = self.mappings[base]
                target_path = (
                    *path[:-1],
                    str(path[-1])[: -len("exp_avg_sq")] + "max_exp_avg_sq",
                )
                self.mappings[key] = target_path
                set_element(self.original_state_dict, target_path, destination)
            else:
                self.original_state_dict[key] = destination
        rng_prefix = f"{RNG_STATE}."
        self.has_rng_state = any(key.startswith(rng_prefix) for key in saved_keys)
        if not self.has_rng_state:
            rng_keys = [key for key in self.state_dict if key.startswith(rng_prefix)]
            if rng_keys:
                logger.warning(
                    "Legacy checkpoint has no RNG state; random training "
                    "cannot resume with exact stream continuity."
                )
                for key in rng_keys:
                    del self.state_dict[key]
        prefix = f"{DATALOADER}.batch_layout."
        if (
            f"{DATALOADER}.version" in saved_keys or f"{DATALOADER}.steps" in saved_keys
        ) and not any(key.startswith(prefix) for key in saved_keys):
            layout_keys = [key for key in self.state_dict if key.startswith(prefix)]
            if layout_keys:
                logger.warning(
                    "Legacy dataloader checkpoint has no batch layout metadata; "
                    "its saved layout cannot be validated."
                )
                for key in layout_keys:
                    del self.state_dict[key]
        return super().create_local_plan()

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        if not self.flatten_state_dict:
            saved_key = read_item.dest_index.fqn
            name = self.compiled_destinations.get(saved_key, saved_key)
            self.original_state_dict[name] = torch.load(value, weights_only=False)
        else:
            super().load_bytes(read_item, value)
