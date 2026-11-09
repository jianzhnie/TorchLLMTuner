"""Compatibility entry point for the relocated Qwen3-30b recipe."""

from importlib import import_module

_recipe = import_module("examples.qwen3-moe.train_qwen3_30b_npu")
qwen3_30b_npu_config = _recipe.qwen3_30b_npu_config
main = _recipe.main

if __name__ == "__main__":
    main()
