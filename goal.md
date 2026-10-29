# TorchLLMTuner Ascend NPU 适配任务

## 1. 目标与约束

修复 TorchLLMTuner 的 Ascend NPU 训练路径，完成 P0–P4 验收及与
`torchtitan-npu` 的数值对齐，交付代码、测试和可复现证据。
本文件定义任务目标；当前进度以 `docs/npu_ascend_status.md` 为准。

- 工作仓库：`/home/jianzhnie/llmtuner/llm/TorchLLMTuner`；训练基线：同级
  `torchtitan-npu`；语义参考：同级 `torchtitan`。
- 遵守 `AGENTS.md` 和 `.agent/skills/dev-guide/SKILL.md`，记录仓库版本和已有
  改动；仅修改相关内容，不覆盖已有改动，未经授权不切分支、不提交。
- 来源映射仅维护在 `docs/llmtuner_upstream_map.md`，不新增来源标注注释。

## 2. 环境

| 项目 | 预期配置 |
| --- | --- |
| 镜像 / 容器 | `hyper-parallel-npu:cann9.1.0-torch29` / `hyper-parallel-env` |
| 设备与软件 | 8×Ascend910、CANN 9.1.0、torch 2.12、torch-npu 2.12 |
| 模型 | `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B` |
| 数据 | `/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl` |
| 训练入口 | `examples/train_qwen3_30b_npu.sh` → 同目录 `train_qwen3_30b_npu.py` |

候选节点以`available_nodes.txt`为准。运行前从宿主机确认目标节点8卡空闲，并核对
容器、设备、挂载及版本；不得仅凭容器内`npu-smi`判断占用。用
`docker exec hyper-parallel-env bash -lc '...'`执行，按实测环境判断。容器离线装依赖，
兼容包由宿主机下载至`/home/jianzhnie/llmtuner/wheels`。

## 3. 验收范围（P0 → P4）

先建立有限矩阵，覆盖全部必测项，无需穷举组合。运行前固定模型、数据、拓扑、
精度、步数、参考路径、断言及阈值，将“各支持路径”展开为具体配置。
标明目标模型验收或辅助验证。缺失标准及范围变更须经用户确认，
不得事后选阈值、绕过保护或删减必测项。

通过须同时满足：退出码为 0、数值有限、目标路径生效、预定断言及指标达标。
记录实际后端和 CPU/AiCPU 回退，禁止静默降级，不要求纯 AiCore。
CPU、mock、仅导入、临时 monkeypatch、小模型及单配置结果不得外推。
辅助验证不替代目标 NPU 验收；诊断性修复须纳入正式实现并复验。

### P0：基础训练

- 8 卡 FSDP 运行指定 Qwen3-30B 入口；预先固定 `LLMTUNER_STEPS`、序列长度、
  有效 batch、精度、seed 及预热步数。
- 以下四组分别运行，独立保存日志和 checkpoint；均使用 8 卡，
  `dp_replicate=1`、`PP=1`、`CP=1`：

  | 方案 | dp_shard（dense） | TP | EP |
  | --- | --- | --- | --- |
  | 1 | 8 | 1 | 1 |
  | 2 | 4 | 2 | 1 |
  | 3 | 4 | 2 | 4 |
  | 4 | 8 | 1 | 8 |

  EP 复用 DP 设备，不能将 `dp_shard × TP × EP` 作为卡数；方案 3 的专家
  FSDP 分片度为 2。以实际 dense / sparse mesh 和 rank 分组核验拓扑。
  四组保持相同训练设置；两步运行仅证明短程训练与保存，不能替代恢复、
  性能或跨布局数值验收。复用配置及代码版本仍适用的已有证据，避免重复运行。
- 报告 loss 趋势及异常（不要求单调下降），直接计量端到端 step 耗时并报告
  预热后分布，不用低精度日志时间差替代。
- 对比连续训练与独立进程恢复：保持初始状态、数据顺序、总步数及调度一致，
  检查模型、optimizer、step、随机状态、数据游标及启用的 scheduler / EMA，
  按预定的精确一致或容差标准验收。

### P1：单并行族

核实 `tests/integration_tests/` 的实际参数，在 NPU 上按预定参考路径比较
输出、loss、梯度等适用指标；跨框架对齐见 P4。

| 并行族 | 必测范围 |
| --- | --- |
| FSDP2 / HSDP | 分片 / 复制 mesh、checkpoint |
| TP | 张量切分、grad norm |
| PP | 1F1B / Interleaved1F1B、真实语料 |
| CP | KV all-gather / Ulysses / KCP |
| EP | AllToAll / TorchAO padded dispatcher |

### P2：并行组合

对照 README、`llmtuner/parallel/compatibility.py` 及实现，列明 DP 复制/分片、
并行度、mesh、world size 和 rank 分组，不凭并行度乘积推断拓扑。
至少覆盖：DP×P1 各支持路径、TP×FSDP、CP×TP、EP×FSDP、TP×EP、TP×EP×CP、
PP×EP、PP×CP、PP×AC、PP×validation。

声明与实现冲突记为缺口，缺测试则补测。不支持配置须在校验或装配时拒绝；
拒绝测试不代表功能通过或缺口关闭。

### P3：训练特性

逐项验证 activation checkpointing（AC：full / selective）、模型 compile、EMA、
validation、chunked / vocab-parallel loss、确定性模式、metrics / profiler、
DCP / torch_checkpointing 两种 checkpoint 后端、数据 packing、多模态管线。
compile 须证明编译图实际执行，profiler 须采集 NPU kernel，两种 checkpoint
后端均须验证恢复。多模态管线须明确模态、预处理、packing、加载和恢复路径；
视频及多模态模型训练须单独列明是否纳入矩阵，不以管线通过代表训练通过。

### P4：跨框架数值对齐

- 对齐初始权重、输入及顺序、随机状态、精度、优化器及超参、loss 归约、
  有效 batch、并行布局和步数；相同 seed 不代表相同初始状态。
- 按局部前反向 → 整网单步 → 多步训练比较 logits、loss、grad norm 和关键
  权重。多步验收须用双方真实训练入口，独立算子或手写循环仅作诊断。
- 核实基线版本、依赖及 `docs/developer_guides/numerical_alignment.md`，复用
  `scripts/numerical_alignment/utils/metrics.py`。以下阈值若与基线冲突，
  须经用户确认。

  | 指标对象 | 阈值（全部满足） |
  | --- | --- |
  | logits | `mean_ae ≤ 0.02`、`mean_re ≤ 0.02`、`cos ≥ 0.9999`；`max_re` 仅诊断 |
  | 每步标量 loss / grad_norm | `ae ≤ 0.02`、`re ≤ 0.005` |

- logits 逐时点、逐张量验收，loss / grad norm 逐步验收；误差定义和零值处理
  沿用基线，不以汇总值掩盖失败。关键权重须另定名称、时点和阈值，
  不套用 logits 阈值或默认逐位一致；标准未定不得宣称 P4 通过。
- 按来源映射 A（移植）/ B（适配）/ C（独有）/ D（缺口）分析差异；
  C/D 不自动豁免，不可比项须说明原因及替代验证。

## 4. 执行与记录

按 P0 → P4 优先推进，仅在 `docs/npu_ascend_status.md` 维护进度及矩阵。
续作时先读该文件，核对代码、日志和设备占用，不重复已排除的诊断。
前置失败或标准未定仅阻塞依赖项，继续独立用例。

1. **盘点**：核对声明、实现、测试及依赖，确定矩阵和验收标准。
2. **修复**：保留复现，最小修改根因；纯逻辑补 CPU 测试，collective / mesh
   补集成测试，硬件用例用 `tests.caps.require_env` 门控。
3. **验证**：先跑受影响测试和目标 NPU 用例，再跑以下门禁；无关失败单列，
   不顺手修复，并检查 pre-commit 自动修改。
4. **记录**：每配置一行，诊断另列；证据关联运行 ID、版本和适用范围，
   不拼接不同运行的观测作为同一次通过证据。修改后重验受影响配置。

```bash
pytest -q --disable-warnings
ruff check llmtuner tests
ruff format --check llmtuner tests
python -m compileall -q llmtuner
git diff --check
pre-commit run --all-files
```

统一记录版本和环境，每行关联运行 ID 及完整命令（含环境变量）：

`测试项 | 配置 | 基线/阈值 | 实际结果 | 状态 | 命令 | 退出码 | 日志 | 原因/解锁条件`

## 5. 阻塞与交付

- 状态仅用 `未跑 / 通过 / 失败 / blocked`。不适用项单列依据，不计为通过；
  必测项不能自行改列为不适用。
- 环境、依赖或标准缺失记 blocked，附证据及解锁条件；断言不满足记失败，
  附复现。间歇性失败未解决，不因单次成功改记通过。
- 同一问题连续 3 轮修复复验未解决，记录已排除假设并转向独立用例；有新证据
  再续。无独立工作则报告所需输入或资源，不重复空跑或改写既有失败状态。
- 交付代码、测试、矩阵、复现命令及剩余缺口。日志按运行保存至
  `outputs/qwen3-30b-npu/logs/`，保留失败记录。
- 完成条件：必测范围无遗漏、矩阵全部适用项通过、缺口关闭、质量门禁通过
  （无关门禁失败仅可由用户明确豁免）。否则交付阶段性结果及剩余失败与阻塞。

参考：`README.md`、`docs/torchllmtuner_design.md`。
