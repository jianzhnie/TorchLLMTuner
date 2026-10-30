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

- `tests/integration_tests/` 的分布式等价性脚本**不在**本次运行范围；
  它们用 `torchrun` 显式执行，可能需要硬件（用 `tests.caps.require_env`
  门控）。
- 测试约定见 `.agent/rules/testing.md`。
