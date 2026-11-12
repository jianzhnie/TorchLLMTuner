"""Experiment APIs; importing model math does not import training services."""

from importlib import import_module

_EXPORTS = {
    "AssemblyResult": "assembly",
    "assemble_model": "assembly",
    "ExperimentConfig": "config",
    "ParallelConfig": "config",
    "small_qwen3_moe_config": "config",
    "ModelContract": "model",
    "Qwen3MoeModel": "model",
    "ExecutionPlan": "resolver",
    "ResolvedTarget": "resolver",
    "resolve_plan": "resolver",
    "ExperimentTrainer": "trainer",
    "ModelProtocol": "protocol",
    "CheckpointAdapterProtocol": "protocol",
    "HFModelAdapter": "hf_adapter",
    "LlamaModelAdapter": "hf_adapter",
    "DeepSeekV3ModelAdapter": "hf_adapter",
    "load_hf_model": "hf_factory",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
