---
name: add-unit-test
description: >
  在 tests/unit_tests/cpu/ 下新增或扩展 CPU 单元测试。适用于为纯逻辑写
  测试、提升覆盖率、mock 分布式/硬件行为。不适用于 torchrun 集成脚本
  （见 rules/testing.md 与 hf-model-precision-validation 技能）。
---

# 编写单元测试

`tests/unit_tests/cpu/` 的流程指南。硬约束在 `.agent/rules/testing.md`，
本技能是 how-to。

## 适用与不适用

- **适用**：为纯 CPU 逻辑写测试、提升覆盖率、mock 分布式/硬件行为。
- **不适用**：集合通信或 mesh 行为 → `tests/integration_tests/` 的
  `torchrun` 等价性脚本（编写约定见 `hf-model-precision-validation` 技能的
  `references/equivalence-scripts.md`）；并行配置的数值验收 →
  `hf-model-precision-validation` 技能。

## 硬约束（简版）

- 运行器：**pytest**（`pytest -q --disable-warnings`）。风格为 plain
  pytest 函数——现有套件（如 `test_parallel_plan_config.py`）是
  `def test_...` 加 `pytest.mark.parametrize`；除非相邻模块已在用，不引入
  `unittest.TestCase`。
- 命名：文件 `test_<behavior>.py`，函数 `test_<expected_behavior>`。
- 仅 CPU：不用真实 NPU/GPU；分布式通信用 mock 或 fake（monkeypatch 集合
  通信，或仅在代码路径要求已初始化进程组时构造单 rank、world-size-1
  环境）。
- `tests.caps.require_env` 用于**导入**需要可选能力的模块（dtensor、
  grain、transformers_qwen3 等）。纯 CPU 逻辑测试应在任何环境干净导入、
  无需门控；确需导入此类模块时，在模块顶部用 `require_env(...)`，或对单个
  测试用 `skip_without(...)`。

## 流程

1. 读被测模块与同目录（`tests/unit_tests/cpu/<layer>/`）最近的既有测试，
   对齐 fixture 风格。
2. 合适时复用 `tests/data_fixtures.py` 中的 fixture。
3. 按 Arrange–Act–Assert 写测试：一个测试一个行为，期望值显式写出；错误
   路径用 `pytest.raises(..., match=...)`（配置测试是范式）。
4. 配置改动：扩展 `tests/unit_tests/cpu/test_config.py` /
   `test_parallel_plan_config.py` 风格的校验测试（合法值通过，非法值抛
   `ConfigError`/`UnsupportedCombinationError`）。
5. 收尾前跑 `pytest -q --disable-warnings`（全套件必须保持全绿）和
   `ruff check tests`。

## 范围外

- 分布式 `torchrun` 脚本本身的新增 → 遵循
  `hf-model-precision-validation/references/equivalence-scripts.md` 的约定
