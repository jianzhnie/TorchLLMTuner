# .agent 目录索引

本目录存放 TorchLLMTuner 的 agent 配置：全局钩子、斜杠命令、分层规则、
agent 定义与可复用技能。agents 采用 thin proxy 模式 —— agent 是薄壳，
规则集中维护在 `rules/`，流程细节沉淀在 `skills/`。

## 顶层

| 文件 | 说明 |
|---|---|
| `settings.json` | 全局钩子配置：对 `Write\|Edit\|MultiEdit` 注册两个 PostToolUse 钩子 |
| `README.md` | 本索引文件 |

## hooks/ — PostToolUse 钩子脚本

| 文件 | 说明 |
|---|---|
| `hooks/enforce-code-style.sh` | 编辑 `.py` 后自动执行 `ruff check --fix` + `ruff format`；ruff 不在 PATH 时静默跳过 |
| `hooks/check-related-files.sh` | 按被编辑路径输出中文提醒（parallel→集成测试、config→examples、models→memory、datasets→单测），不阻断 |

## commands/ — 斜杠命令

| 文件 | 说明 |
|---|---|
| `commands/test.md` | 运行 `pytest -q --disable-warnings` 并定位修复失败 |
| `commands/commit.md` | 先跑 ruff，再按 Conventional Commit 风格提交 |
| `commands/gen-commit-msg.md` | 根据 staged diff 生成 Conventional Commit 消息（含本仓库 scope 映射） |
| `commands/create-pr.md` | 用 `gh` 创建 PR，正文含问题说明、变更摘要、验证命令与结果 |
| `commands/code-review.md` | 对当前改动做代码审查，加载 `rules/` 检查 |

## rules/ — 分层规则（agent 按需加载的硬约束）

| 文件 | 说明 |
|---|---|
| `rules/code-style.md` | Python 3.10、88 列、ruff E/F/I/UP/B、命名规范、叶子模块导入、`__init__.py` 公共 API |
| `rules/testing.md` | CPU 单测 vs torchrun 集成测试分工、`test_<behavior>` 命名、`tests.caps.require_env` 门控 |
| `rules/distributed.md` | FSDP/TP/PP/CP/EP 各维度注意事项（内存生命周期、公共入口、集合通信正确性） |

## agents/ — agent 定义（thin proxy，引用 rules/ 与 skills/）

| 文件 | 说明 |
|---|---|
| `agents/code-reviewer.md` | 完整代码审查：加载全部三个 rules，只读，不修改代码 |
| `agents/simple-code-reviewer.md` | 快速预检：仅风格与明显错误，供轻量场景使用 |
| `agents/code-verifier.md` | 验证改动可运行（pytest/ruff/compileall），只报告不修改 |
| `agents/planner.md` | 实现规划，只读，输出分步计划 |
| `agents/fsdp-dev-expert.md` | FSDP 开发专家，含真实文件导航与对应集成测试脚本 |
| `agents/tensor-dev-expert.md` | TP 开发专家（含 sequence-parallel 约束） |
| `agents/pipeline-dev-expert.md` | PP 开发专家（microbatch/bubble 相关） |
| `agents/ep-dev-expert.md` | EP 开发专家（调度器选项与已注册缺口） |

## scripts/ — 维护脚本

| 文件 | 说明 |
|---|---|
| `scripts/check_agents_catalog.py` | 校验 agents frontmatter（name/description/tools）、名称一致性、引用的 rules/skills 路径存在；`python3 .agent/scripts/check_agents_catalog.py` 直接运行 |

## skills/ — 可复用技能

| 文件 | 说明 |
|---|---|
| `skills/dev-guide/SKILL.md` | 仓库开发指南（本仓库原有） |
| `skills/karpathy-guidelines/SKILL.md` | Karpathy 风格开发准则（本仓库原有） |
| `skills/add-unit-test/SKILL.md` | `tests/unit_tests/cpu/` 单测编写 how-to：pytest 风格、命名、mock 分布式通信 |
| `skills/parallel-strategy-analyzer/SKILL.md` | 并行策略推荐入口：三阶段流程，输出 FSDP/TP/PP/CP/EP 组合与配置 |
| `skills/parallel-strategy-analyzer/references/io-contract.md` | 策略分析的输入表与输出契约 |
| `skills/parallel-strategy-analyzer/references/memory-estimation.md` | 显存估算公式（模型状态、激活、五种 AC 模式、CPU offload） |
| `skills/parallel-strategy-analyzer/workflows/01-collect-inputs.md` | 阶段一：模型/硬件/负载信息收集与可整除性预检 |
| `skills/parallel-strategy-analyzer/workflows/02-baseline-and-search.md` | 阶段二：基线与决策树（TP/CP 轴/EP 调度器/HSDP 约束） |
| `skills/parallel-strategy-analyzer/workflows/03-cost-and-config.md` | 阶段三：通信成本表、bubble 公式、`ParallelConfig` 字段片段 |
| `skills/hf-model-precision-validation/SKILL.md` | 精度验证入口：四阶段流程（基线→单策略→组合→重计算），loss + FP32 梯度范数双对齐 |
| `skills/hf-model-precision-validation/references/equivalence-scripts.md` | `tests/integration_tests/` 等价性脚本按关注点的映射表与编写约定 |
| `skills/hf-model-precision-validation/references/diagnosis.md` | 数值发散诊断顺序（输入同一性→状态同一性→首个发散点→集合语义→精度路径） |
| `skills/hf-model-integration/SKILL.md` | HF 模型接入入口：`llmtuner/models/hf/` 7 步流程与完成门槛 |
| `skills/hf-model-integration/references/integration-workflow.md` | 接入细节：ModelConfig 对齐、声明式并行挂接、AC 兼容、examples/tests 交付 |
