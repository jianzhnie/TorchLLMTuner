# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import torch


def get_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype string: '{dtype_str}'. Supported: {list(mapping.keys())}")
    return mapping[dtype_str]