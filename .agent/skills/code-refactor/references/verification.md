# 验证与完成判据

每个模块重构完成后必做，全绿才能进入下一模块。

## 1. 全量门禁

```bash
python -m compileall -q llmtuner        # 全部模块可编译
ruff check llmtuner tests               # 风格与常见错误
ruff format --check llmtuner tests      # 格式
pytest -q --disable-warnings            # CPU 单测全绿（全量，不是只跑改动模块）
python -m llmtuner --steps 20           # 短训练冒烟，行为不变
```

## 2. 针对该模块的全面测试

- 跑该模块对应的**全部**单测（如 `pytest tests/unit_tests/cpu/ -k <模块相关关键词>`），
  一个都不能跳过；重构导致既有测试失效时，先判断是重构破坏了行为（修代码）
  还是测试断言依赖了被合并的旧结构（更新测试并说明理由）。
- 若触及公共入口或跨层调用，额外跑关联模块的单测，确认没有隐式依赖被破坏。
- 分布式行为若被触及，运行 `tests/integration_tests/` 中对应的 torchrun 等价性
  脚本（用 `tests.caps.require_env` 判断环境是否支持；不支持则明确说明未验证
  的部分）。
- 重构前后各跑一次 `python -m llmtuner --steps 20`，对比 loss 输出一致，作为
  行为不变的实证。

## 3. 完成判据

某模块只有同时满足以下条件才算完成：

- 五条门禁命令全绿
- 该模块相关单测零跳过零失败
- 行为冒烟 loss 一致
- 产出该批次的变更清单与删除清单

任何一项不满足，就地修复，不得带病进入下一模块。
