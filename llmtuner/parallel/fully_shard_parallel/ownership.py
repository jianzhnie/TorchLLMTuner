"""Track FSDP override ownership across module and parameter replacements.

Only references and configuration metadata are retained. Replacement history is
borrowed from the local assembly layout; this consumer owns no separate mapping.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from torch import nn

from ...config.parallel_config import ParallelConfig
from ...errors import UnsupportedCombinationError
from ...utils.module_names import canonical_fqn
from ..module_patterns import compile_module_pattern
from ..replacements import ReplacementMap

__all__ = ["FSDPPatternReferences"]


def named_module_paths(model: nn.Module) -> dict[str, nn.Module]:
    """Index physical module paths and their transparent-wrapper aliases."""
    paths = dict(model.named_modules(remove_duplicate=False))
    for name, module in tuple(paths.items()):
        paths.setdefault(canonical_fqn(name), module)
    return paths


class FSDPPatternReferences:
    """Keep programmatic overrides bound across module/parameter replacement.

    Capture registered aliases before transforms, then resolve logical paths
    immediately before FSDP. Objects retained under renamed paths remain valid.
    Layout conversions that remove every alias require an explicit mapping;
    rejecting them prevents silently losing an exclusion or hook override.
    Explicit mappings take precedence over reused source paths. Every mapped
    target must still be registered, and wrapped module targets bind to their
    outer execution boundary.
    """

    def __init__(
        self, model: nn.Module, cfg: ParallelConfig, *, replacements: ReplacementMap
    ) -> None:
        self.cfg = cfg
        self.module_paths: dict[nn.Module, list[str]] = {}
        self.parameter_paths: dict[nn.Parameter, list[str]] = {}
        self.replacements = replacements
        for name, module in model.named_modules(remove_duplicate=False):
            self.module_paths.setdefault(module, []).append(canonical_fqn(name))
        for name, parameter in model.named_parameters(remove_duplicate=False):
            self.parameter_paths.setdefault(parameter, []).append(canonical_fqn(name))
        self.unit_modules = dict(model.named_modules())
        module_paths = named_module_paths(model)
        self.ignored_modules: dict[str, nn.Module | None] = {}
        ignored = set()
        for pattern in cfg.fsdp_config.fsdp_ignored_modules:
            compiled = compile_module_pattern(pattern)
            matches = {
                name: module
                for name, module in module_paths.items()
                if compiled.fullmatch(name)
            }
            self.ignored_modules.update(matches or {pattern: None})
            ignored.update(
                child for module in matches.values() for child in module.modules()
            )
        self.ignored_parameters = {
            parameter
            for module in ignored
            for parameter in module.parameters(recurse=False)
        }
        canonical_names = {module: name for name, module in self.unit_modules.items()}
        self.unit_options: dict[str, dict[str, Any]] = {}
        for pattern, options in cfg.fsdp_config.fsdp_apply_modules.items():
            compiled = compile_module_pattern(pattern)
            matches = [
                canonical_names[module]
                for name, module in module_paths.items()
                if (name or pattern == "")
                and module not in ignored
                and compiled.fullmatch(name)
            ]
            # Patterns may intentionally name the native EP layout, which
            # does not exist until conversion. Validate those at FSDP entry.
            for name in matches or [pattern]:
                self.unit_options.setdefault(name, {}).update(options)

    def record_replacements(
        self,
        modules: dict[nn.Module, nn.Module],
        parameters: dict[nn.Parameter, tuple[nn.Parameter, ...]],
    ) -> None:
        """Validate exclusions, then publish to the shared replacement map.

        Overrides still refer to the original objects. Keep their targets at
        the latest conversion, including when one parameter splits into several.
        """
        contributors: dict[nn.Parameter, set[nn.Parameter]] = {}
        for source, targets in parameters.items():
            for target in targets:
                contributors.setdefault(target, set()).add(source)
        globally_ignored = {
            parameter
            for module in self.ignored_modules.values()
            if module is not None
            for parameter in module.parameters()
        }
        for name, options in self.unit_options.items():
            exclusions = options.get("ignored_params") or set()
            if not isinstance(exclusions, set):
                continue  # Preserve the FSDP override type validation.
            excluded = globally_ignored | exclusions
            excluded = excluded | {
                target
                for source in excluded
                for target in self.replacements.parameter_targets.get(source, ())
            }
            for sources in contributors.values():
                if sources & excluded and not sources <= excluded:
                    raise UnsupportedCombinationError(
                        f"FSDP pattern {name!r}: cannot exclude part of a stacked "
                        "expert parameter; exclude all source projections in "
                        "the stack or use an unstacked expert layout"
                    )
        self.replacements.record_replacements(modules, parameters)

    def bind(self, model: nn.Module) -> ParallelConfig:
        modules = {}
        first_module_paths: dict[int, str] = {}
        for name, module in model.named_modules(remove_duplicate=False):
            # The outer transparent wrapper owns the logical boundary.
            path = canonical_fqn(name)
            modules.setdefault(path, module)
            # Include wrapped inner modules so explicit replacements resolve
            # to their outer execution boundary without losing identity.
            first_module_paths.setdefault(id(module), path)
        parameters = {
            canonical_fqn(name): parameter
            for name, parameter in model.named_parameters(remove_duplicate=False)
        }
        current_parameters = set(parameters.values())

        def mapped_module_path(source: nn.Module | None, context: str) -> str | None:
            if source not in self.replacements.module_targets:
                return None
            target = self.replacements.module_targets[source]
            path = first_module_paths.get(id(target))
            if path is None:
                raise ValueError(
                    f"{context}: module replacement target is no longer registered; "
                    "refresh the assembly mapping"
                )
            return path

        def bind_parameters(
            exclusions: set[nn.Parameter], context: str
        ) -> set[nn.Parameter]:
            rebound = set()
            for parameter in exclusions:
                if parameter in self.replacements.parameter_targets:
                    # An explicit conversion owns this identity. Its former
                    # paths may now belong to unrelated parameters; never union
                    # those registrations into a split/merged target set.
                    targets = set(self.replacements.parameter_targets[parameter])
                    if not targets <= current_parameters:
                        raise ValueError(
                            f"{context}: ignored_params replacement target is no "
                            "longer registered; refresh the assembly mapping"
                        )
                else:
                    targets = {
                        parameters[path]
                        for path in self.parameter_paths.get(parameter, ())
                        if path in parameters
                    }
                    if parameter in current_parameters:
                        targets.add(parameter)
                if not targets:
                    raise ValueError(
                        f"{context}: ignored_params lost its registered paths "
                        "during model transformation; configure exclusions "
                        "on the converted model"
                    )
                rebound.update(targets)
            return rebound

        overrides = {}
        for name, source in self.unit_options.items():
            pattern = canonical_fqn(name)
            mapped = mapped_module_path(
                self.unit_modules.get(name), f"FSDP pattern {pattern!r}"
            )
            if mapped is not None:
                pattern = mapped
            options = source.copy()
            hook = options.get("hook_module")
            if isinstance(hook, nn.Module):
                mapped = mapped_module_path(hook, f"FSDP hook for {pattern!r}")
                targets = [
                    modules[path]
                    for path in self.module_paths.get(hook, ())
                    if path in modules
                ]
                if mapped is not None:
                    options["hook_module"] = modules[mapped]
                elif targets:
                    options["hook_module"] = targets[0]
                elif id(hook) in first_module_paths:
                    options["hook_module"] = modules[first_module_paths[id(hook)]]
                else:
                    raise ValueError(
                        f"FSDP pattern {pattern!r}: hook_module lost its registered "
                        "paths during model transformation; configure the "
                        "override on the converted model"
                    )
            exclusions = options.get("ignored_params")
            if isinstance(exclusions, set):
                options["ignored_params"] = bind_parameters(
                    exclusions, f"FSDP pattern {pattern!r}"
                )
            if pattern in overrides and overrides[pattern] != options:
                raise ValueError(
                    f"FSDP overrides converge on {pattern!r} after model "
                    "transformation but have conflicting options"
                )
            overrides[pattern] = options
        ignored_names = []
        for name, source_module in self.ignored_modules.items():
            name = canonical_fqn(name)
            mapped = mapped_module_path(source_module, f"FSDP ignored module {name!r}")
            if mapped is not None:
                name = mapped
            elif name not in modules and source_module is not None:
                name = first_module_paths.get(id(source_module), name)
            ignored_names.append(name)
        # A source subtree can split across owners (Kimi gate weight and
        # correction bias). Preserve moved parameters without excluding the
        # target parameter's entire new parent subtree.
        module_exclusions: set[nn.Parameter] = set()
        for pattern in ignored_names:
            compiled = compile_module_pattern(pattern)
            module_exclusions.update(
                parameter
                for name, module in modules.items()
                if compiled.fullmatch(name)
                for parameter in module.parameters()
            )
        moved_exclusions = (
            bind_parameters(self.ignored_parameters, "FSDP ignored modules")
            - module_exclusions
        )
        if moved_exclusions:
            overrides.setdefault("", {})
            for options in overrides.values():
                exclusions = options.get("ignored_params")
                if exclusions is None or isinstance(exclusions, set):
                    options["ignored_params"] = (exclusions or set()) | moved_exclusions
        return replace(
            self.cfg,
            fsdp_config=replace(
                self.cfg.fsdp_config,
                fsdp_apply_modules=overrides,
                fsdp_ignored_modules=list(dict.fromkeys(ignored_names)),
            ),
        )
