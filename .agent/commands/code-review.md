# /code-review — 代码审查

对照仓库规则审查当前改动。先加载 `.agent/rules/code-style.md`，其违规视为
阻断性问题。

## 用法

```bash
/code-review            # 审查未提交改动
/code-review branch     # 审查当前分支对 main 的 diff
```

## 行为

审查 diff（`git diff HEAD` 或分支 diff），覆盖：

- **风格合规**——`.agent/rules/code-style.md`（ruff E/F/I/UP/B、88 列、
  命名、叶子模块导入、`__init__.py` 导出）
- **测试**——`.agent/rules/testing.md`（CPU 单测 vs torchrun 集成测试、
  `tests.caps.require_env` 门控、命名）
- **分布式正确性**——涉及 `llmtuner/parallel/` 时加载
  `.agent/rules/distributed.md`（FSDP 显存生命周期、集合通信顺序、公共入口）
- **代码质量**——死代码、复杂度、错误处理

## 输出格式

```markdown
## 审查摘要
**文件**: …
**问题**: N 个（阻断 / 建议）
### 阻断问题 …（文件:行号 + 具体修法）
### 建议 …
### 良好之处 …
```

发现违反 `code-style.md` 时，明确指出违反的规则并给出修正后的代码。
