# /test — 运行单元测试套件（pytest）

运行仓库的 CPU 单元测试套件并修复失败。

## 用法

```bash
/test
```

## 行为

1. 在仓库根目录运行 `pytest -q --disable-warnings`。
2. 若有失败：定位失败测试与被测代码，诊断根因并修复，重跑直至全绿。
3. 不得靠跳过或删除失败测试让套件变绿——修根因。

## 说明

- 适用：CPU 单元测试（`tests/unit_tests/`）。不适用：分布式等价性验证
  （`tests/integration_tests/` 的 `torchrun` 脚本）——它们需显式单独运行，
  可能要硬件（用 `tests.caps.require_env` 门控）。
- 测试约定见 `.agent/rules/testing.md`。
