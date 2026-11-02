# 任务：指导 TorchLLMTuner 有选择地借鉴 FSDPTurbo

你正在改进 TorchLLMTuner。请阅读 FSDPTurbo 的实现，将其中经过验证的显存优化和工程方法，以符合 TorchLLMTuner 架构的方式移植或适配。
目标不是复制仓库，而是在保持现有训练语义的前提下，提高性能、显存效率和
真实模型可用性。

## 1. 仓库与约束

- 工作仓库：`/home/jianzhnie/llmtuner/llm/TorchLLMTuner`
- 参考仓库：`/home/jianzhnie/llmtuner/llm/FSDPTurbo`
- 对比说明：`docs/fsdpturbo_comparison.md`
- 遵守工作仓库的 `AGENTS.md`、`.agent/skills/dev-guide/SKILL.md` 和
  `.agent/karpathy-guidelines/SKILL.md`；阅读参考仓库自己的开发约束。
- 先记录两边版本和已有改动。只修改当前能力所需文件，不覆盖或回退已有工作，
  未经授权不切分支、不提交、不推送。
- 不直接复制 FSDPTurbo 的 Trainer、根配置或模型 wrapper，不引入对
  `fsdp_turbo` 包的运行时依赖。参考其算法、算子边界和验证方法，按
  TorchLLMTuner 的职责分层实现。
- 不为“架构一致”复制 registry、patch 或配置体系；只有存在多个真实消费者时
  才抽象公共接口。保留 TorchLLMTuner 的配置流、`STAGE_ORDER`、组合矩阵和
  HF wrapper 契约。
- 不将 README 中的功能声明当作实现或验证证据；必须追踪真实调用方、配置、
  算子、前后向和测试。

## 2. 运行环境

| 项目 | 预期配置，运行前以实测为准 |
| --- | --- |
| 容器 | `hyper-parallel-env` |
| 硬件 / 软件 | 8×Ascend910、CANN 9.1.0、torch / torch-npu 2.12 |
| 模型 | `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B` |
| 数据 | `/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl` |
| 训练入口 | `examples/train_qwen3_30b_npu.sh` → `train_qwen3_30b_npu.py` |

节点以 `available_nodes.txt` 为候选。运行前从宿主机检查训练进程和 8 卡占用，
核对容器、挂载和版本，不凭镜像名或容器内 `npu-smi` 推断。使用：

```bash
docker exec hyper-parallel-env bash -lc '...'
```

依赖只安装到隔离环境，不修改容器全局 site-packages。离线包由宿主机准备到
`/home/jianzhnie/llmtuner/wheels`。保留失败日志，不抢占其他任务的设备。

## 3. 总体策略

坚持“先基线、后单项、再组合”：

1. 用 TorchLLMTuner 当前真实入口建立可复现基线，固定代码版本、初始权重、
   数据顺序、拓扑、精度、有效 batch、seed、步数和预热步数。
2. 一次只选择一个能力完成“源码分析 → 最小实现 → 数值验证 → 性能测量 →
   文档”的闭环，不同时移植量化、swap 和多个 kernel。
3. 先证明默认路径不变，再证明新路径正确，最后讨论性能。没有收益的适配不合并。
4. 新能力必须显式启用；不支持模型、设备、精度或并行组合时，在配置或装配期
   给出带替代方案的异常，不静默回退。
5. CPU、mock、算子 smoke 和小模型只证明各自范围，不能替代目标模型、目标
   NPU 和真实 Trainer 验收。

## 4. 借鉴优先级

### P0：真实模型性能与显存基线

先确认瓶颈，不凭推测优化。记录端到端 step time、预热后分布、吞吐、峰值
显存、实际 backend 和 CPU/AiCPU fallback。分别保存冷启动、稳态和恢复数据，
不用低精度日志时间差代替直接计时。

### P1：activation / optimizer 显存优化

完成 P0 基线后，选择 activation swap、optimizer swap 或 offload 中的一项。
重点阅读 `fsdp_turbo/memory/` 和 optimizer factory。实现时必须覆盖：

- forward/backward 生命周期、stream 同步和 host buffer 所有权；
- 梯度累积、FSDP/TP/EP、AC 和 compile 的交互；
- optimizer 惰性状态、分片布局、EMA、异步 checkpoint 和失败恢复；
- 显存收益与传输开销，不能把 host 拷贝时间隐藏在指标之外。

### P2：工程能力

根据实际需要补充真实模型 system test、性能 baseline 和结构化配置入口。
优先复用现有配置与测试机制，不复制 FSDPTurbo 的 YAML 体系或自由 patch。

## 5. 每个能力的实施要求

开始前用简短计划明确：

- 参考文件及对应 TorchLLMTuner 接缝；
- 要解决的问题和预期收益；
- 默认路径是否变化；
- 数值参考、容差、性能指标和明确不覆盖的范围。

实现时遵守：

- 最小修改根因，不顺手重构；公共 API 保持兼容。
- 数学语义、参数/ buffer dtype、shape、layout、路由顺序和归约边界必须有依据。
- 参考实现存在设备专用假设时，逐条说明保留、适配或拒绝的理由。
- 纯逻辑补 CPU 单测；collective/mesh 补集成测试；硬件用例用
  `tests.caps.require_env` 门控。
- 测试必须 non-vacuous：包含能区分新旧实现的输入，必要时证明关闭优化后
  断言会失败。只有 import、编译或有限值检查不算数值验证。
- 行为改变后同步更新受影响的配置说明、设计文档和示例。

## 6. 验收标准

每个能力至少完成：

1. **默认回归**：功能关闭时，相关既有测试和默认训练语义不变。
2. **局部正确性**：与当前生产路径或高精度独立参考比较输出及输入/参数梯度；
   运行前固定逐项容差，不事后放宽。
3. **端到端正确性**：通过真实 Trainer 比较逐步 loss、grad_norm 和适用的
   关键权重，不以手写循环替代最终验收。
4. **状态恢复**：独立进程恢复后，模型、optimizer、scheduler、dataloader、
   RNG、EMA 和训练进度与连续运行满足预定一致标准。
5. **目标组合**：逐项验证声明支持的 FSDP/TP/PP/CP/EP、AC、compile 和精度；
   未覆盖组合明确保留为限制。
6. **性能收益**：同一环境、同一拓扑和相同工作负载下，报告预热后 step time、
   吞吐和峰值显存的分布及测量方法。性能回退或收益不明确时不得宣称优化成功。

质量门禁：

```bash
pytest -q --disable-warnings
ruff check llmtuner tests
ruff format --check llmtuner tests
python -m compileall -q llmtuner
git diff HEAD --check
pre-commit run --all-files
```

先跑受影响测试，再跑完整门禁。无关失败单独列出，不顺手修改；检查
pre-commit 自动产生的变化。

## 7. 交付格式

每次只交付一个完整能力，并报告：

- 问题、参考实现和适配设计；
- 修改文件及默认行为是否变化；
- 实际执行的命令、环境、运行 ID 和日志路径；
- 数值、恢复、性能和门禁结果；
- 未验证范围、失败项和下一步建议。

没有真实 NPU 证据时，只能交付阶段性实现，明确写出“目标 NPU 未验收”。
不得以代码移植完成、CPU 通过或单次成功运行宣称整体对齐、全组合支持或
性能提升。现在先执行 P0，分析 profile 后选择唯一的 P1 候选并完成闭环。
