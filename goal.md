# TorchLLMTuner Ascend NPU 适配与验收

修复 Ascend NPU 训练路径，完成 P0–P4 验收及与 `torchtitan-npu` 的数值对齐，
交付代码、测试和可复现证据。本文件定义要求；进度与验收矩阵统一维护在
`docs/npu_ascend_status.md`。

## 1. 工作边界与环境

- 工作目录：`/home/jianzhnie/llmtuner/llm/TorchLLMTuner`。
  训练基线为同级 `torchtitan-npu`，语义参考为同级 `torchtitan`。
- 遵守 `AGENTS.md`、`.agent/skills/dev-guide/SKILL.md` 和
  `.agent/karpathy-guidelines/SKILL.md`；保留已有改动，未经授权不切分支或提交。
- 来源映射统一更新至 `docs/llmtuner_upstream_map.md`。

| 项目 | 预期配置（运行前核实实际值） |
| --- | --- |
| 镜像 / 容器 | `hyper-parallel-npu:cann9.1.0-torch29` / `hyper-parallel-env` |
| 硬件 / 软件 | 8×Ascend910、CANN 9.1.0、torch / torch-npu 2.12 |
| 模型 | `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B` |
| 数据 | `/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl` |
| 入口 | `examples/train_qwen3_30b_npu.sh` → 同目录 `train_qwen3_30b_npu.py` |

从 `available_nodes.txt` 选择节点，在宿主机确认 8 卡空闲，并核对容器、挂载、
设备及依赖版本；不能仅凭容器内 `npu-smi` 判断占用。使用
`docker exec hyper-parallel-env bash -lc '...'` 执行。容器离线安装依赖，
所需包由宿主机下载至 `/home/jianzhnie/llmtuner/wheels`。

## 2. 统一验收规则

运行前建立有限矩阵，覆盖下列必测项，无需穷举组合。每项固定模型、数据、
拓扑、精度、步数、参考实现和断言阈值，并标明“目标模型验收”或“辅助验证”。
缺失标准、阈值冲突和范围变更须经用户确认，不得事后放宽标准或删减必测项。

通过须同时满足：退出码为 0、数值有限、目标路径实际执行、预定断言达标。
记录实际后端及 CPU/AiCPU 回退，不要求纯 AiCore，但禁止静默降级。
CPU、mock、仅导入、小模型、临时 monkeypatch 或单配置结果不能替代目标 NPU
验收；诊断修复须进入正式实现并复验。已有证据仅在代码、配置和环境仍适用时复用。

### P0：目标模型训练、性能与恢复

以下四组分别运行，独立保存日志和 checkpoint；均为 8 卡，
`dp_replicate=1`、`PP=1`、`CP=1`。

| 方案 | dense dp_shard | TP | EP |
| --- | --- | --- | --- |
| 1 | 8 | 1 | 1 |
| 2 | 4 | 2 | 1 |
| 3 | 4 | 2 | 4 |
| 4 | 8 | 1 | 8 |

EP 复用 DP 设备，卡数不能按 `dp_shard × TP × EP` 计算；方案 3 的专家 FSDP
分片度为 2。以实际 dense / sparse mesh 和 rank 分组核验拓扑。

- 四组保持相同模型、数据顺序、有效 batch、序列长度、精度、seed、训练步数
  （`LLMTUNER_STEPS`）和预热设置；记录必要的布局相关差异。
- 报告 loss 趋势及异常，不要求单调下降。直接测量预热后端到端 step 时间分布，
  不用低精度日志时间差代替。两步运行不能代表恢复、性能或跨布局数值验收。
- 比较连续训练与独立进程恢复：固定初始状态、数据顺序、总步数及调度，验证
  模型、optimizer、step、随机状态、数据游标及启用的 scheduler / EMA，
  按预定的精确一致或容差标准验收。

### P1：单并行族

核实 `tests/integration_tests/` 的真实参数，在 NPU 上对照参考路径验证输出、
loss、梯度等适用指标；跨框架对齐另按 P4 验收。

| 并行族 | 必测范围 |
| --- | --- |
| FSDP2 / HSDP | 分片 / 复制 mesh、checkpoint |
| TP | 张量切分、grad norm |
| PP | 1F1B / Interleaved1F1B、真实语料 |
| CP | KV all-gather / Ulysses / KCP |
| EP | AllToAll / TorchAO padded dispatcher |

### P2：并行组合

对照 README、`llmtuner/parallel/compatibility.py` 及实现，将 DP 与 P1 各支持
路径的组合展开为具体配置；至少覆盖 TP×FSDP、CP×TP、EP×FSDP、TP×EP、
TP×EP×CP、PP×EP、PP×CP、PP×AC、PP×validation。
每项记录 DP 复制/分片、并行度、mesh、world size 和 rank 分组。
声明与实现冲突须列为缺口，缺测试则补测；不支持的配置须明确拒绝，
拒绝测试通过不能算作该功能通过或缺口关闭。

### P3：训练特性

逐项验证 AC（full / selective）、模型 compile、EMA、validation、
chunked / vocab-parallel loss、确定性、metrics / profiler、
DCP / torch_checkpointing 两种 checkpoint 后端、数据 packing 和多模态管线。
compile 须证明编译图执行，profiler 须采集 NPU kernel，两种 checkpoint 后端
均须验证恢复。多模态须明确模态、预处理、packing、加载和恢复路径，
另行列明视频及多模态模型训练是否纳入范围，不能用管线通过代替训练通过。

### P4：跨框架数值对齐

对齐初始权重、输入及顺序、随机状态、精度、optimizer 及超参、loss 归约、
有效 batch、并行布局和步数；相同 seed 不等于相同初始状态。
按“局部前反向 → 整网单步 → 多步训练”比较 logits、loss、grad norm 和关键权重。
多步验收使用双方真实训练入口，独立算子和手写循环仅作诊断。

核实基线版本与依赖，遵循 `../torchtitan-npu/docs/developer_guides/numerical_alignment.md`，
复用 `../torchtitan-npu/scripts/numerical_alignment/utils/metrics.py` 的误差定义及零值处理。

| 指标 | 阈值（全部满足） |
| --- | --- |
| logits | `mean_ae ≤ 0.02`、`mean_re ≤ 0.02`、`cos ≥ 0.9999`；`max_re` 仅诊断 |
| 每步标量 loss / grad_norm | `ae ≤ 0.02`、`re ≤ 0.005` |

logits 按时点、张量验收，loss / grad norm 逐步验收，不用汇总值掩盖失败。
关键权重须预先确定名称、时点和阈值，不套用 logits 标准或默认逐位一致。
按来源映射 A（移植）/ B（适配）/ C（独有）/ D（缺口）分析差异；
C/D 不自动豁免，不可比项须说明原因与替代验证。标准未定不得宣称 P4 通过。

## 3. 执行、记录与交付

1. 先读进度文档，核对版本、已有改动、证据和设备占用，按 P0 → P4 推进。
   前置失败仅阻塞依赖项，继续独立工作，不重复已排除的诊断。
2. 保留复现并最小化修复：纯逻辑补 CPU 测试，collective / mesh 补集成测试；
   适用的硬件测试通过 `tests.caps.require_env` 门控。
3. 运行受影响测试、目标 NPU 用例及以下门禁，检查自动修改；无关失败单列。

```bash
pytest -q --disable-warnings
ruff check llmtuner tests
ruff format --check llmtuner tests
python -m compileall -q llmtuner
git diff --check
pre-commit run --all-files
```

在进度文档中每配置一行，诊断另列。记录代码版本、环境、运行 ID、完整命令
（含环境变量）及证据适用范围；修改后重验受影响项，不拼接不同运行作为一次通过。

`测试项 | 配置 | 基线/阈值 | 实际结果 | 状态 | 命令 | 退出码 | 日志 | 原因/解锁条件`

- 状态：`未跑 / 通过 / 失败 / blocked`。断言失败记失败；环境、依赖或标准缺失
  记 blocked，并附证据及解锁条件。不适用项单列依据，不计为通过；必测项不可自行豁免。
- 间歇性失败不因单次成功而关闭。同一问题连续 3 轮修复复验未解决，记录已排除
  假设并转向独立项；有新证据再续，无独立项则报告所需输入或资源。
- 日志按运行保存至 `outputs/qwen3-30b-npu/logs/`，保留失败记录。
- 交付代码、测试、矩阵、复现命令和剩余缺口。全部适用必测项、缺口与质量门禁
  均通过后才算完成；否则明确为阶段性结果。无关门禁失败仅可由用户明确豁免。
