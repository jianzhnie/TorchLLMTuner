# TorchLLMTuner 简介

TorchLLMTuner 是基于 PyTorch `DeviceMesh` 的混合并行训练框架，Python 包名为
`llmtuner`。它直接适配 Hugging Face Causal LM，将 FSDP2、TP、PP、CP、EP
组织在一套可校验的训练流程中。

项目以 TorchTitan 的并行语义为参考，重点提供统一 Trainer、明确装配顺序和
可靠的 checkpoint，而不是复制上游全部功能。

## 1. 适用场景

- 在单机或多机环境训练 Hugging Face 文本模型；
- 组合 FSDP2、TP、PP、CP、EP，并检查其数值和状态正确性；
- 在 CUDA 或昇腾 NPU 上复用统一控制流；
- 使用 CPU 完成单进程运行和纯逻辑测试。

无法保证数学语义或运行安全的组合会在配置或装配阶段明确拒绝，不静默降级。

## 2. 核心能力

### 2.1 混合并行

| 并行方式 | 当前实现 |
|---|---|
| FSDP2 / HSDP | composable `fully_shard`，支持 DP shard、DP replicate、混合精度、reshard、CPU offload 和预取 |
| TP | 由 HF TP plan 驱动，使用序列并行语义的通信线性层 |
| PP | 基于 `torch.distributed.pipelining`，支持 1F1B、Interleaved1F1B |
| CP | 支持 KV AllGather、Ulysses、KCP 及适用的 packed/varlen 输入 |
| EP | 转换支持的 HF MoE block，默认 AllToAll，另有 TorchAO、MC2、Domino 专用路径 |

Dense world size 为：

```text
WORLD_SIZE = dp_replicate × dp_shard × cp × tp × pp
```

EP 复用 `dp_shard × cp × tp` 稀疏区域，不额外乘入 world size：

```text
(dp_shard × cp × tp) % ep == 0
efsdp = (dp_shard × cp × tp) / ep
```

`data_parallel_shard_size=-1` 时自动推导 DP shard，实际 ownership 以运行时
mesh 和 rank 分组为准。

### 2.2 训练与显存管理

- activation checkpoint：full、selective、memory-budget、region；
- 可选 NPU 激活交换，以及 FSDP CPU offload 和双向预取；
- Adam/AdamW 的 `npu_virtual` moment 驻留，支持同步及异步原生 DCP；
- DistMuon/FlexShard：独立配置参数计算布局并复用 FQN 状态恢复；
- chunked loss、梯度累积、裁剪、学习率调度、EMA 和 validation；
- DCP、异步保存、HF 导入导出及完整训练状态恢复；
- 控制台、TensorBoard、W&B、profiler、显存快照和编译计数器。

这些能力均有模型、拓扑和环境约束，不能按功能名称推断全部组合已验收。

### 2.3 模型与数据

- 通过 `AutoConfig` 或本地自定义 config 构造 `ForCausalLM`；
- 支持自定义架构注册、HF 状态字典和并行后参数布局适配；
- 使用 Grain 构建数据流水线，支持 random、JSONL、HF dataset 和 packing；
- 提供文本、多模态数据处理、chat template 和可选多轮 renderer。

当前 `HFTransformerModel` 是文本 decoder 适配器，不消费图片、视频或 MRoPE
输入。数据管线支持多模态不等于模型已支持视觉训练。

## 3. 总体架构

```text
CLI / Python
    ↓
LLMTunerConfig
    ↓
Trainer
    ↓
HFTransformerModel
    ↓
parallelize_hf_transformers
    ↓
TP → EP → CP → Activation Checkpoint
    → Activation Swap（可选）→ Compile（可选）→ FSDP
    ↓
训练 / validation / checkpoint / metrics
```

装配顺序由 `llmtuner/parallel/stages.py` 唯一声明。启用 PP 时，先切分当前
rank 的 stage，再对每个 model part 应用同一顺序，最后构建 schedule。

## 4. 训练流程

1. 解析并校验配置，初始化设备、进程组和 RNG；
2. 构建 mesh、HF wrapper 和模型；
3. 按 stage 表应用并行变换；
4. 创建 optimizer、scheduler、EMA、dataloader、checkpoint 和指标组件；
5. 准备 batch，执行前向与反向，归一化 loss，裁剪梯度并更新参数；
6. 按配置运行 validation、周期保存和最终保存；
7. 关闭训练资源，并由 CLI 销毁进程组。

PP 由 schedule 驱动 microbatch；非 PP 直接调用模型。两条路径共用 batch、
loss 归一化、优化器、指标和 checkpoint 语义。

## 5. 配置体系

配置按职责拆分在 `llmtuner/config/`：

| 配置 | 职责 |
|---|---|
| `ModelConfig` | 模型来源、形状、计算 dtype、架构覆盖项 |
| `ParallelConfig` | DP/FSDP、TP、PP、CP、EP 和 dispatcher |
| `OptimizerConfig` | AdamW、DistMuon、参数组、计算布局和驻留策略 |
| `TrainingConfig` | batch、步数、梯度累积、AC、激活交换、compile |
| `DataloaderConfig` | 数据源、tokenizer、packing 和媒体字段 |
| `CheckpointConfig` | 保存、恢复、异步模式和 HF 导入导出 |
| `MetricsConfig` / `ProfilerConfig` | 日志、TensorBoard、W&B 和 profiling |
| `LLMTunerConfig` | 聚合配置并执行跨组校验 |

CLI 使用扁平参数，复杂嵌套字段通过 Python 传入：

```bash
python -m llmtuner --help
# 或安装后
llmtuner-train --help
```

## 6. 安装与快速开始

环境要求为 Python 3.10+、PyTorch 2.12+、Transformers 5.11.x；精确依赖见
`pyproject.toml`。

```bash
git clone https://github.com/jianzhnie/TorchLLMTuner.git
cd TorchLLMTuner
pip install -e .
```

开发环境：

```bash
pip install -e '.[dev]'
pre-commit install
```

### 单进程训练

默认使用随机 token 和小型离线模型，无需下载权重：

```bash
python -m llmtuner --steps 20
```

### 两卡 FSDP2

```bash
torchrun --nproc_per_node=2 -m llmtuner \
  --data_parallel_shard_size 2 \
  --global_batch_size 8 \
  --steps 20
```

### 两卡 TP

```bash
torchrun --nproc_per_node=2 -m llmtuner \
  --tensor_parallel_size 2 \
  --max_seq_len 128 \
  --steps 20
```

### 八卡 FSDP × TP × PP

```bash
torchrun --nproc_per_node=8 -m llmtuner \
  --data_parallel_shard_size 2 \
  --tensor_parallel_size 2 \
  --pipeline_parallel_size 2 \
  --num_pp_microbatches 2 \
  --global_batch_size 8 \
  --max_seq_len 128
```

### Checkpoint

```bash
python -m llmtuner \
  --steps 20 \
  --enable \
  --interval 10 \
  --dump_folder ./outputs
```

最终 model-only 或 HF 导出物不包含完整训练状态，不能用于精确续训。

### 真实数据与示例

```bash
python -m llmtuner \
  --dataset local_jsonl \
  --dataset_path ./corpus.jsonl \
  --tokenizer_path ./tokenizer \
  --steps 20

python -m examples.train_qwen3
python -m examples.train_deepseek_v3
```

昇腾示例见 `examples/train_qwen3_8b_npu.sh` 和
`examples/train_qwen3_30b_npu.sh`。

## 7. 常见边界

- TP>1 只能使用序列并行路径；`enable_sequence_parallel=False` 不受支持。
- `max_seq_len` 必须能被 `cp × tp` 整除。
- chunked loss 不能与 PP 或 validation 同时启用。
- PP 不支持跨首尾 stage 的 tied word embedding。
- 一维 packed token 流不能在 PP 中切分为多个 microbatch。
- EP 依赖可等价表达的模型布局；未知结构、bias 或路由规则会被拒绝。
- DeepEP/HybridEP 是登记的能力缺口，默认 dispatcher 为 AllToAll。
- `memory_budget` AC 需要 compile，`region` AC 需要可选 `torch_remat`。
- 激活交换当前与 compile 互斥；`npu_virtual` 仍限 Adam/AdamW。
- DistMuon 的计算布局必须在构造时提供，恢复时也需匹配。

完整裁决以配置校验、`parallel/compatibility.py` 和实际错误信息为准。

## 8. 代码结构与开发

```text
llmtuner/
├── accelerator/       # 设备、进程组、collective、能力探测
├── components/        # optimizer、EMA、checkpoint、metrics、FlexShard
├── config/            # 分组配置与聚合配置
├── datasets/          # 数据图、packing、文本与多模态处理
├── memory/            # AC、区域重计算、激活交换
├── models/            # HF wrapper、通用构件和模型专用适配
├── parallel/          # 五类并行、拓扑、stage 和组合裁决
├── trainer/           # 训练与验证编排
└── utils/             # 日志、RNG、GC 和监控工具
```

常用检查：

```bash
pytest -q --disable-warnings
ruff check llmtuner tests
ruff format --check llmtuner tests
pre-commit run --all-files
```

`tests/integration_tests/` 包含需要 `torchrun` 和相应设备的脚本，默认 pytest
不会收集。运行前须核对各自 docstring 中的 rank、参数和环境要求。CPU、mock、
小模型和单步通过不能外推为其他设备或大规模组合通过。

## 9. 代码规模与简洁性

简洁性主要体现在统一配置流、固定装配顺序和直接 HF 接入，不等于每个模块
代码都更少。当前工作树的典型数据如下：

| 模块 | TorchLLMTuner | 对应外部实现 | 解读 |
|---|---:|---:|---|
| Activation checkpoint | 717 行 | TorchTitan `activation_remat.py` 210 行 | 本项目保留四种 AC 模式和 NPU 重放策略，实现更大 |
| Pipeline parallel | 1033 行 | TorchTitan `pipeline_parallel.py` 913 行 | 总量接近，职责拆分不同 |
| Checkpoint | 4675 行 | TorchTitan checkpointer 1990 行 | 本项目额外覆盖 HF/Kimi 布局、异步发布和失败恢复 |
| 装配顺序表 | 80 行 | FSDPTurbo 构造入口 18 行 | 本项目集中声明跨阶段契约，不能只按入口行数比较系统复杂度 |

详细口径、有效代码行数和方法级对比见：

- 架构与运行时契约：`docs/torchllmtuner_design.md`
- TorchTitan 系列对比：`docs/torchtitan_comparison.md`
- FSDPTurbo 对比：`docs/fsdpturbo_comparison.md`
- Ascend 910C 实施与验证：`docs/ascend_910c_adoption_guide.md`
