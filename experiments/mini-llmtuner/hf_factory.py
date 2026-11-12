"""HuggingFace model construction without coupling the trainer to Transformers."""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .hf_adapter import DeepSeekV3ModelAdapter, LlamaModelAdapter


def load_hf_model(
    model_name_or_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
):
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=config, torch_dtype=torch_dtype
    )
    if device is not None:
        model.to(device)
    if config.model_type == "llama":
        return LlamaModelAdapter(model)
    if config.model_type == "deepseek_v3":
        return DeepSeekV3ModelAdapter(model)
    raise ValueError(f"Unsupported HuggingFace model type: {config.model_type}")
