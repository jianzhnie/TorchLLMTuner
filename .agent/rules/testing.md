---
name: testing
description: 测试约定——CPU 单元测试与 torchrun 集成测试的分工
paths:
  - tests/**
  - "test_*.py"
---

# 测试

## 测试类型

- **CPU 单元测试**（`tests/unit_tests/cpu/`）：纯逻辑，无需分布式环境或
  加速器。默认套件：`pytest -q --disable-warnings`。
- **分布式集成/等价性脚本**（`tests/integration_tests/`）：验证集合通信与
  device mesh 行为，**用 `torchrun` 显式运行**（如
  `torchrun --nproc_per_node=N tests/integration_tests/<script>.py`，
  或经 `run_all.py`）。不被普通 pytest 套件收集。

## 命名

- 测试文件：`test_<behavior>.py`；测试函数：`test_<expected_behavior>`。
- 集成脚本用描述性命名，如 `fsdp_mesh_equivalence.py`、`cp_tp_equivalence.py`。
- 共享辅助模块不带 `test_` 前缀，避免被 pytest 收集。

## 硬件门控

- 依赖硬件或可选依赖的测试用 `tests.caps.require_env` 门控——缺硬件必须是
  跳过而非失败。模块级导入依赖用 `require_env(...)`；单个测试用
  `skip_without(...)`。
- 每个测试自包含；测试间不共享可变状态。

## 流程规则

- 纯逻辑加 CPU 单测；集合通信或 mesh 行为加 `torchrun` 脚本。
- 提交改动前：跑完整单测套件（`pytest -q --disable-warnings`）和
  `pre-commit run --all-files`。
- 修改 `llmtuner/parallel/` 时，检查 `tests/integration_tests/` 是否存在
  对应等价性脚本，有硬件时用 `torchrun` 运行。
- 验证全部模块可编译：`python -m compileall -q llmtuner`。

## 修复优先于规避

测试失败时修根因。不悄悄加 `@skipif`、`try/except` 或放宽断言——确需跳过
（硬件门控）时在提交说明中明确写出。
