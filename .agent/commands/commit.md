# /commit — Lint、暂存、提交

按仓库的 Conventional Commit 风格提交改动。

## 用法

```bash
/commit                              # 自动生成提交消息
/commit -m "feat(parallel): add X"   # 显式指定消息
```

## 流程

1. 加载 `.agent/rules/code-style.md`——不提交仍违反规则的代码。
2. 运行 `ruff check llmtuner tests` 与 `ruff format --check llmtuner tests`；
   提交前修复或报告失败。
3. 查看 `git status` 与 `git diff`，只暂存预期改动。
4. 生成或确认 Conventional Commit 消息（见 `/gen-commit-msg`）：
   `<type>(<scope>): <subject>`，如
   `refactor(parallel): reorganize topology modules`、
   `feat(tokenizer): add multimodal tokens`。
   允许的 type：`feat`、`fix`、`refactor`、`docs`、`test`、`chore`。
5. 提交。用户未要求时不 push。

## 说明

- 主题行：祈使句，结尾无句号，约 80 字符。
- 不含 AI/IDE 署名尾注（`Co-authored-by:`、`Made-with:`）。
