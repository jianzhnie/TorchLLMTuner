---
name: code-refactor
description: >
  系统性重构优化 llmtuner/ 代码结构：重组文件、简化函数与类、去冗余、
  删除无用抽象与浅包装，行为不变前提下提升清晰度与可读性。
  适用于代码结构优化、重构、去冗余任务；不适用于功能开发、
  改变公共 API 语义或数值行为的改动（那些走正常开发流程）。
---

# 代码重构技能

对 `llmtuner/` 做系统性重构。本文件是流程骨架与硬约束，细节按需加载。

## 开工前必读

1. `AGENTS.md` 与 `.agent/rules/code-style.md`、`.agent/rules/distributed.md` —— 硬约束，
   违反 code-style 属于阻断性问题。
2. `.agent/skills/dev-guide/SKILL.md` —— 本仓库开发准则，尤其"重构与命名"一节。
3. 目标模块现状：先通读再动手，禁止凭猜测重构。

## 硬约束（不复述，见引用）

- 职责边界、重构原则、命名与归位规则：
  [references/principles-and-boundaries.md](references/principles-and-boundaries.md)
- 验证门禁、模块级全面测试与完成判据：
  [references/verification.md](references/verification.md)

## 工作流程

```text
1. 通读目标模块 → 找出冗余、浅包装、错位归属、命名模糊点
2. 按模块分批重构（config → datasets → models → parallel → memory → components → trainer）
3. 每批完成后执行 references/verification.md 的全部验证
4. 全绿且产出变更/删除清单 → 进入下一模块；否则就地修复
```

## 每批产出

- 变更清单：文件/函数/类的移动、合并、删除及理由
- 删除清单：附无人调用的证据（全局搜索结果）

## 存疑则停

无法确定是否冗余的代码（可能是公共 API、可能被 examples/ 或外部使用），列出并说明，不擅自删除。

## 禁止事项

- 不改变公共 API 的签名与语义（除非单列并经确认）
- 不新增依赖
- 不新增测试文件/文档文件，除非任务明确要求
- 不做与重构无关的功能开发
- 不改动 `llmtuner/`、`tests/`、`examples/`、`docs/`、`AGENTS.md` 之外的文件
