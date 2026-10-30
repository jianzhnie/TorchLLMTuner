# /gen-commit-msg — 生成提交消息

根据已暂存改动生成 Conventional Commit 消息，不执行提交。

## 用法

```bash
/gen-commit-msg                  # 根据暂存改动自动生成
/gen-commit-msg --scope parallel # 强制指定 scope
```

## 流程

### 第 1 步：分析改动

```bash
git diff --cached --name-only
git diff --cached --stat
git diff --cached
git log --oneline -5
```

### 第 2 步：确定 type

| Type | 适用场景 |
| ---- | -------- |
| `feat` | 新功能或新能力 |
| `fix` | bug 修复 |
| `refactor` | 非功能/非修复的代码改动 |
| `docs` | 仅文档 |
| `test` | 新增或修复测试 |
| `chore` | 构建、依赖、配置改动 |

### 第 3 步：推断 scope

| 改动路径 | Scope |
| -------- | ----- |
| `llmtuner/parallel/` | `parallel` |
| `llmtuner/parallel/fully_shard_parallel/` | `fsdp` |
| `llmtuner/parallel/tensor_parallel/` | `tp` |
| `llmtuner/parallel/pipeline_parallel/` | `pp` |
| `llmtuner/parallel/context_parallel/` | `cp` |
| `llmtuner/parallel/expert_parallel/` | `ep` |
| `llmtuner/config/` | `config` |
| `llmtuner/datasets/` | `datasets` |
| `llmtuner/models/` | `models` |
| `llmtuner/memory/` | `memory` |
| `llmtuner/trainer/` | `trainer` |
| `llmtuner/components/` | `components` |
| `tests/` | `test` |
| `docs/` | `docs` |
| `examples/` | `examples` |
| `.agent/` | `ci` |
| 跨多个领域 | 省略 scope 或用更宽泛的词 |

### 第 4 步：生成消息

```
<type>(<scope>): <subject>

<body>
```

规则：

- 主题行：祈使句，约 80 字符，结尾无句号
- 正文解释"为什么"而非"做了什么"
- 不含 AI/IDE 署名尾注

### 第 5 步：预览与确认

展示预览并请用户确认，然后给出可直接使用的 `git commit -m` 命令。

## 另见

- `/commit`——lint + 提交流程
- `AGENTS.md`——Commit & Pull Request Guidelines
