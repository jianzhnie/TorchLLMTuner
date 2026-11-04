# FSDPTurbo 与 TorchLLMTuner 对比

## 1. 结论

两个项目都支持 Hugging Face 模型的分布式训练，但抽象层级不同：

- **FSDPTurbo 是模型变换与设备加速库**：接收用户已构建的模型，注入并行、
  量化、重计算、activation/optimizer swap 和 FSDP，强项是昇腾算子与低精度。
- **TorchLLMTuner 是完整训练框架**：统一管理模型构建、五类并行、数据、
  validation、指标、checkpoint 和资源生命周期，强项是控制面与正确性。

两者可以互补：FSDPTurbo 提供设备执行能力，TorchLLMTuner 提供训练控制面。
整合时必须定义模型、状态和错误边界，不能只拼接目录或配置。

## 2. 总体定位

| 维度 | FSDPTurbo | TorchLLMTuner |
|---|---|---|
| 核心入口 | `FSDPTurbo(config, model)` | `Trainer(LLMTunerConfig)` |
| 项目类型 | 模型变换与训练加速库 | 混合并行训练框架 |
| 模型来源 | 用户创建的 `nn.Module` | 框架构造 HF Causal LM |
| 主要输出 | 已应用并行、量化和显存变换的模型 | 完整训练任务及其状态 |
| 硬件方向 | 昇腾 NPU 优先，保留 CUDA/Triton | CPU、CUDA、NPU 统一控制流 |
| 失败策略 | 配置、mesh 和模块匹配检查 | 配置、拓扑、组合矩阵和模型 probe 分层拒绝 |

## 3. 装配方式

### FSDPTurbo

```text
用户模型
  → Quantization → TP → CP → Module Patch → EP/EFSDP
  → Recompute / Activation Swap / Chunk Batch → FSDP
  → 用户训练循环或 BaseTrainer
```

其优势是以模型变换为中心，便于嵌入已有脚本。模块名 pattern 和框架级 patch
灵活，但数学语义更依赖接入者。

### TorchLLMTuner

```text
CLI / Config → Trainer → HF wrapper
  → TP → EP → CP → Activation Checkpoint
  → Activation Swap（可选）→ Compile（可选）→ FSDP
  → 训练 / validation / checkpoint / metrics
```

`llmtuner/parallel/stages.py` 唯一声明装配顺序。PP 先切分 stage，再对每个
model part 应用同一顺序。Trainer 负责从初始化到资源释放的完整生命周期。

## 4. 并行能力

| 能力 | FSDPTurbo | TorchLLMTuner |
|---|---|---|
| FSDP2 | 原生与 custom FSDP，支持 pattern、prefetch、CPU offload | composable FSDP2/HSDP，支持 reshard、混合精度、CPU offload 和 EP sparse mesh |
| DP replicate | 独立 DP 与 FSDP 轴组成 HSDP | `data_parallel_replicate_size` 与 `data_parallel_shard_size` |
| TP | 用户配置 colwise、rowwise、sequence pattern | HF TP plan 驱动，并校验模型与 MoE 布局 |
| PP | 无核心 pipeline runtime | 1F1B、Interleaved1F1B |
| CP | Ulysses、KV AllGather、KCP | KV AllGather、Ulysses、KCP，含适用的双轴路径 |
| EP | eager、fused、MC2、Domino | 默认 AllToAll，另有 TorchAO、MC2、Domino 专用路径 |
| EFSDP | 显式配置 | 从 sparse region 推导 |
| Compile | 无统一装配阶段 | 全模型或逐 block、regional compile、async TP |
| 组合校验 | 分散在配置和变换入口 | 配置、topology、compatibility 和模型 probe 分层执行 |

并行名称相同不表示模型和组合语义相同。并行度字段不能机械迁移。

## 5. 显存优化与量化

| 能力 | FSDPTurbo | TorchLLMTuner |
|---|---|---|
| 激活重计算 | 按模块 pattern | full、selective、memory-budget、region |
| Chunk batch | 按 batch 维切分指定模块输入 | 无等价实现；梯度累积和 PP microbatch 不是替代 |
| Chunked loss | 主要由专用模型路径调用 | Trainer 内置，受组合约束 |
| Activation swap | 已集成 | 已实现 NPU saved-tensor 激活交换，与 AC 生命周期对齐 |
| Optimizer swap | AdamWSwap | 以 `npu_virtual` Adam moment 驻留为主；AdamWSwap 未实现 |
| FSDP CPU offload | 已集成 | 已集成，并支持 pin-memory 和双向预取 |
| DistMuon/FlexShard | 非核心主线 | 已接入参数计算布局、FQN 状态和 NPU 验证 |
| MXFP8 / W8A16 | 可选集成 | 未实现通用量化训练栈 |
| 低精度 AllGather | 可选集成 | 未实现 |

FSDPTurbo 的量化工具更丰富；TorchLLMTuner 更侧重通用 AC、并行语义、完整
状态恢复和优化器布局。

## 6. 模型接入

### FSDPTurbo

```python
model = create_your_model()
model = FSDPTurbo(config, model)
```

用户负责模型构建，pattern 和 patch 选择转换目标。适合非标准模型和已有系统，
但错误 pattern 或不适用布局可能到运行时才暴露。

### TorchLLMTuner

```python
cfg = LLMTunerConfig(...)
trainer = Trainer(cfg)
trainer.train()
```

框架通过 `AutoConfig` 或本地 config 构造模型。`HFTransformerModel` 提供统一
forward 和部件视图，TP 使用 HF `_tp_plan`，EP probe 检查 MoE 布局。无法等价
转换的结构会明确失败，但高度定制 forward 的适配成本更高。

## 7. 训练、数据与 checkpoint

| 维度 | FSDPTurbo | TorchLLMTuner |
|---|---|---|
| Trainer | BaseTrainer 或用户训练循环 | 统一 Trainer 与 runtime 模块 |
| 数据 | 内置文本与多模态 dataloader | Grain、packing、文本/多模态和 renderer |
| Validation | 无等价通用核心子系统 | 内置 Trainer、数据、归约和指标 |
| EMA | 无通用在线 EMA | 已接入 optimizer step 和 checkpoint |
| 指标 | TrainingMonitor、profiler | 控制台、TensorBoard、W&B、profiler、显存快照 |
| 主要后端 | DCP | DCP，另可选 `torch_checkpointing` |
| 训练状态 | 模型和 optimizer 等基础状态 | 模型、optimizer、scheduler、dataloader、trainer、RNG、EMA |
| 异步保存 | 无统一等价入口 | disabled、async、async_with_pinned_mem |
| HF 导入导出 | 主要由用户流程负责 | 内置初始加载与最终导出 |
| EP 变更恢复 | 提供分片状态基础 | 使用逻辑完整专家张量并按当前 EP 重新分片 |

可选后端存在不等于所有设备和布局均已验证恢复。

## 8. 对应实现的代码量

以下统计基于 2026-11-04 的本地工作树，每格为“物理行 / 有效代码行”。
有效代码行排除空行、纯注释和 docstring。目录包含全部 `.py` 文件；不同项目的
功能边界并不完全一致。

| 模块或方法 | TorchLLMTuner | FSDPTurbo | 解读 |
|---|---:|---:|---|
| Activation checkpoint / recompute | 717 / 384 | `memory/recompute/recompute.py`：404 / 212 | 本项目支持更多 AC 模式和 NPU 重放约束 |
| Activation swap | 289 / 236 | `swap_activation/`：465 / 234 | 有效代码行接近；FSDPTurbo 另有 `swap/` 通用层 1109 / 658 行 |
| Optimizer 驻留 | `state_residency.py`：186 / 144 | `swap_optimizer.py`：395 / 285 | Virtual 与分批 AdamWSwap 机制不同，不按行数判断开销 |
| MC2 dispatch + ops | 483 / 409 | `dispatcher_mc2.py` + NPU `grouped_matmul_mc2.py`：270 / 226 | 本项目包含更多布局校验、空分组回退和指标 |
| Domino dispatch | 390 / 312 | `domino_dispatcher.py`：494 / 334 | 本项目物理行少约 21%，有效代码行仅少约 7% |
| 并行装配入口 | `parallelize_hf_transformers`：193 / 166 | `FSDPTurbo.__init__`：18 / 17 | 前者内联组合校验，后者将工作分派给更多方法，入口行数不代表总复杂度 |

这些数据说明：TorchLLMTuner 的部分单文件适配更紧凑，但完整的组合校验、
回退和生命周期管理会增加代码。比较简洁性时还应检查重复逻辑和调用层级，
不能将外部依赖或委托给其他模块的代码视为“零成本”。

## 9. 测试与主要取舍

| 项目 | 测试重点 |
|---|---|
| FSDPTurbo | CP、EP dispatcher、量化、NPU 算子、真实模型 system tests 和性能基线 |
| TorchLLMTuner | 配置、模型构件、checkpoint、并行契约的 CPU 单测，以及多进程和真实 NPU 等价验证 |

FSDPTurbo 的主要优势是昇腾专用优化、量化和模型覆盖；主要代价是没有 PP、
组合信息分散，且模型语义更依赖 pattern 使用者。

TorchLLMTuner 的主要优势是五类并行统一编排、清晰的生命周期和状态验证；
主要代价是缺少通用量化栈、设备融合算子库较小，CLI 不支持 YAML/JSON，定制
模型依赖 HF wrapper。

## 10. 选型与迁移建议

- 需要昇腾量化、低精度通信或成熟设备算子：优先 FSDPTurbo。
- 需要 PP、五类并行组合、统一 validation 和完整状态恢复：优先
  TorchLLMTuner。
- 建设长期系统时，可保留 TorchLLMTuner 的控制面，有选择地接入 FSDPTurbo
  执行能力，但应分别验证数值、状态和性能。

从 FSDPTurbo 迁移时，先建立原生精度基线，再重新设计 mesh 和并行组合，最后
逐项启用量化或 swap。迁移 PP 时没有直接落点，需要取消 PP 或引入独立运行时。
迁移 checkpoint 应使用双方都能解释的 HF 权重，不复制物理 shard。
