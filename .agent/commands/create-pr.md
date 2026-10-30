# /create-pr — 创建 Pull Request

用 `gh` 创建 PR，遵循仓库的 PR 规范。

## 用法

```bash
/create-pr                 # 为当前分支创建 PR
/create-pr --draft         # 创建草稿 PR
```

## 流程

1. 检查前置条件：分支已推送 origin（必要时
   `git push -u origin <branch>`），提交遵循 Conventional Commit 风格。
2. 运行验证并记录结果：
   - `pytest -q --disable-warnings`
   - `ruff check llmtuner tests`
   - `ruff format --check llmtuner tests`
3. 按 `AGENTS.md` 的 Commit & Pull Request Guidelines 生成 PR 正文：

```markdown
## Problem
<解决什么问题>

## Changes
<结构性或行为变更摘要>

## Validation
- `pytest -q --disable-warnings` — <结果>
- `ruff check llmtuner tests` — <结果>
- `ruff format --check llmtuner tests` — <结果>

Closes #<issue>   <!-- 如适用 -->
```

4. 创建 PR：`gh pr create --title "<type>(<scope>): <subject>" --body-file <file>`。
5. 运行时行为变更附日志；关联相关 issue。仅视觉输出才附截图。
