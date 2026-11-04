# TorchLLMTuner、TorchTitan 与 TorchTitan-NPU 对比

## 1. 结论

三个项目不是同一层级的替代品：

- **TorchTitan** 是完整的 PyTorch 原生训练平台，覆盖预训练、SFT、RL，
  提供原生模型、五类并行、低精度和成熟 recipes。
- **TorchTitan-NPU** 是 TorchTitan 的 Ascend 插件，复用上游 Trainer、配置树
  和并行基础设施，通过 patch、extension、override 和 NPU 算子完成适配。
- **TorchLLMTuner** 是独立的精简实现，以 HF config 和具体 `ForCausalLM` 为
  默认模型来源，用聚合 dataclass、普通 `apply_*` 函数和固定 stage 表组织训练。

选型重点是训练场景与维护成本，而不是功能名称数量。

## 2. 依赖关系

```text
TorchTitan
├── 原生模型与训练平台
├── experiments/transformers_modeling_backend（实验 HF 后端）
└── TorchTitan-NPU
    ├── 复用上游 Trainer / Config / parallel runtime
    ├── patches + extensions
    ├── 显式 override
    └── NPU models + ops + 性能扩展

TorchLLMTuner
├── 独立 Trainer / Config / parallel runtime
├── 默认直接包装 HF 模型
└── 参考 TorchTitan 语义，但不是其插件
```

TorchTitan-NPU 与 TorchTitan 是插件与宿主关系，升级时必须核对固定版本。
TorchLLMTuner 与 TorchTitan 是独立实现与语义参考关系，上游新能力不会自动
进入本项目。

## 3. 核心架构

| 维度 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| 核心入口 | `Trainer(LLMTunerConfig)` | Python recipe 返回 `Trainer.Config` | 导入插件后调用上游 `torchtitan.train.main` |
| 配置机制 | 分组 dataclass；扁平 CLI + Python 复杂字段 | `Configurable` 配置树与 Python recipes | 继承上游并增加 NPU schema 和 override |
| 模型协议 | 普通 HF module + wrapper 五部件视图 | 原生模型协议；另有实验 HF backend | 继承上游并添加 NPU 模型与适配 |
| 并行装配 | 固定 stage 表 | 模型声明 sharding，运行时组合 | 上游装配 + NPU 特化 |
| 生命周期 | 独立 Trainer 全部管理 | Trainer 与 TrainingEngine 分工 | 上游生命周期 + NPU 扩展 |

TorchLLMTuner 将配置流收敛为
`CLI / Python → LLMTunerConfig → 顶层装配 → 显式参数`，减少框架基类，但放弃
上游通用配置树和递归 override 机制。

## 4. 并行与计算能力

| 能力 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| FSDP2 / HSDP | dense/sparse mesh、CPU offload | 原生 FSDP，另有 DDP、FlexShard 等 | 继承并扩展 NPU offload |
| TP | HF TP plan、序列并行线性层 | 原生声明式 sharding、async TP | 继承并按模型和算子适配 |
| PP | 1F1B、Interleaved1F1B | 更多 pipelining schedules | 继承上游，取决于 recipe |
| CP | KV AllGather、Ulysses、KCP | KV AllGather、Ulysses | 继承并增加模型专用长序列适配 |
| EP | AllToAll、TorchAO、MC2/Domino | 原生 EP、DeepEP/HybridEP、Dist-MoE | 继承并适配 NPU dispatch 和路由 |
| Compile | 全模型/逐 block、regional、async TP | compile、CUDA graph、GraphTrainer | NPU Inductor、AutoFuse、GraphTrainer |
| 组合校验 | 配置、topology、compatibility、probe | 配置树、模型/parallelism 契约 | 上游校验 + NPU 配置与模型约束 |

三者使用相同的并行名称，不代表同一模型支持相同组合。实验 HF backend 与
TorchTitan 原生模型路径也必须分别评估。

## 5. 模型与训练场景

| 维度 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| 默认模型来源 | HF config 和具体 `ForCausalLM` | 仓内原生模型；实验 HF backend | 上游原生模型 + NPU 专用模型 |
| 模型覆盖 | 取决于 wrapper、TP plan、MoE 布局和 adapter | Llama、Qwen、DeepSeek、Kimi 等 | 重点为 DeepSeek、Qwen 和上游已适配模型 |
| 文本预训练/SFT | 统一训练入口与数据管线 | 完整 recipes | 继承并提供 NPU recipes |
| 多模态 | 有数据处理；当前 wrapper 不消费媒体/MRoPE | 原生支持多种视觉、音频和生成模型 | 提供特定图文/视频适配 |
| LoRA / MTP / Muon | DistMuon/FlexShard 已接入；无通用 LoRA/MTP | 提供对应训练能力 | 提供特定模型扩展 |
| RL | 未实现 | TitanRL | 按上游兼容性另行评估 |

TorchLLMTuner 的优势是 HF 接入路径短，而不是模型数量或训练场景更广。

## 6. 显存、精度与性能优化

| 能力 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| Activation checkpoint | full、selective、memory-budget、region | full、selective、regional remat | 继承并扩展 |
| Activation swap | NPU saved-tensor 激活交换 | 依具体路径 | NPU 专用交换与 offload |
| Optimizer state | `npu_virtual` Adam moments | 依具体路径 | Virtual/Swap/NovaSwap 等专用扩展 |
| DistMuon/FlexShard | 已接入参数计算布局与恢复 | 原生 FlexShard | 继承并扩展 Muon/offload |
| CPU offload | FSDP 参数/梯度与双向预取 | FSDP offload、BF16 optimizer states | NPU pinned/staging 和 optimizer offload |
| 低精度训练 | 无通用量化栈 | MXFP8、NVFP4 等可选路径 | 多种低精度格式，部分限定 Ascend 950 |
| 图优化 | 通用 compile 与 async TP | compile、CUDA graph、GraphTrainer | NPU AutoFuse、图编译和 SDC |

TorchTitan-NPU 的优势来自上游训练栈与 NPU 专用执行能力。没有同环境测量时，
不能据此推断 TorchLLMTuner 的性能优劣。

## 7. 数据、观测与 checkpoint

| 维度 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| 数据 | Grain、packing、random、JSONL/HF dataset、多模态处理 | Grain、mixing、chat/SFT、多模态 | 继承并增加模型专用适配 |
| Validation / EMA | 内置 validation 与在线 EMA | 上游通用能力 | 继承并验证特定 NPU 路径 |
| 指标与调试 | loss、TFLOPs、MFU、TensorBoard、W&B、profiler、编译计数器 | 完整 metrics、Flight Recorder、SDC replay | NPU profiler、SDC 和数值对齐工具 |
| 主要后端 | DCP，可选 `torch_checkpointing` | 上游 checkpoint 体系 | 继承并扩展 NPU staging/offload |
| 训练状态 | 模型、optimizer、scheduler、dataloader、trainer、RNG、EMA | 完整训练状态与 HF 转换工具 | 继承并按模型和扩展适配 |
| 异步保存 | disabled、async、pinned-memory async | 上游异步 checkpoint | 继承并扩展 NPU staging |
| 跨 EP 布局 | 逻辑完整专家表示，按当前 EP 重新分片 | 依上游模型与 adapter | 依上游和 NPU adapter |

三者的 FQN、parallel layout 和 metadata 不同，不能假设 DCP 目录可直接互换。

## 8. 依赖与验证口径

| 项目 | 关键依赖与验证重点 |
|---|---|
| TorchLLMTuner | Python 3.10+、Torch 2.12+、Transformers 5.11.x；CPU 单测、分布式等价脚本、真实 NPU 验证 |
| TorchTitan | Python 3.11+，推荐近期 PyTorch；CPU/GPU 单测、模型 CI、收敛和性能验证 |
| TorchTitan-NPU | Python 3.12+、固定 Torch/Torch-NPU/CANN；NPU smoke、模型 integration、loss compare 和数值对齐 |

比较结果时必须区分原生模型与 HF 模型、CPU 与真实设备、单并行族与组合拓扑、
数值对齐与吞吐性能。

## 9. 对应实现的代码量

统计时间为 2026-11-04，使用本地工作树。每格为“物理行 / 有效代码行”；
有效代码行排除空行、纯注释和 docstring。外部文件取职责最接近的实现，功能
范围差异单列说明，不能将行数直接解释为性能或质量排名。

| 模块或方法 | TorchLLMTuner | TorchTitan / TorchTitan-NPU | 主要范围差异 |
|---|---:|---:|---|
| AC | 717 / 384 | TorchTitan `activation_remat.py`：210 / 124 | 本项目支持四种 AC 模式及 NPU 重放约束 |
| PP | 1033 / 594 | TorchTitan `pipeline_parallel.py`：913 / 617 | 本项目物理行多 13%，有效代码行少约 4%，只是拆分方式不同 |
| CP | 2383 / 1652 | TorchTitan `context_parallel.py`：394 / 292 | 本项目另含 KCP、压缩注意力和多种 mask/输入布局 |
| Checkpoint | 4675 / 3352 | TorchTitan checkpointer：1990 / 1311 | 本项目增加 HF/Kimi、远程发布和失败恢复路径 |
| Trainer | 2631 / 1726 | `trainer.py` + `training_engine.py`：1460 / 1185 | 本项目包含 runtime 装配、validation 和 PP 驱动 |
| `Trainer.train` | 150 / 83 | TorchTitan `Trainer.train`：62 / 42 | 本项目单方法更集中，但不能据此推断整个训练栈更简单 |
| DistMuon/FlexShard | 4377 / 3716 | TorchTitan FlexShard：4533 / 3786 | 两者规模接近，本项目物理行少约 3% |
| Virtual Optimizer | 186 / 144 | NPU `cpu_offload_adamw.py`：312 / 252 | 驻留和更新机制不同，只比较适配层规模 |
| Compile SDC 两个函数 | 24 / 22 | NPU `compile/sdc_checksum.py`：81 / 46 | 本项目仅做图外非有限值检查，上游 checksum 语义更完整 |

总体上，TorchLLMTuner 的简洁性来自统一入口和依赖方向，而非全模块行数更少。
TorchTitan 的通用模型协议和训练场景更完整；TorchTitan-NPU 的设备能力还依赖
上游实现，不能只统计插件文件就得出完整系统代码量。

## 10. 各自代价

### TorchLLMTuner

优势是 HF 接入直接、五类并行统一 stage、控制面精简。代价是模型与训练场景
较少，缺少通用 RL/VLM、量化生态和成熟设备 recipes，需要持续维护上游语义映射。

### TorchTitan

优势是模型与训练场景最广，原生并行、compile、低精度、RL、调试和 checkpoint
完整。代价是 `Configurable` 和模型协议增加适配成本，近期特性依赖新 PyTorch。

### TorchTitan-NPU

优势是最大化复用 TorchTitan recipes，并深入 Ascend 算子、图编译、长序列和
显存优化。代价是强依赖固定上游与 NPU 软件栈，部分能力限定特定模型或
Ascend 950，导入顺序和符号绑定复杂。

## 11. 选型建议

- 需要上游最新技术、广泛模型、成熟 recipes 或 RL：优先 TorchTitan。
- 希望保留 TorchTitan 训练体系并使用 Ascend 专用能力：优先 TorchTitan-NPU。
- 主要训练 HF Causal LM，重视精简装配和并行数值研究：优先 TorchLLMTuner。

长期系统应选择一条主干，不要同时维护三套配置、模型和 checkpoint 语义。
迁移时通过共同 HF 权重或正式转换器，固定初始权重、数据顺序和精度，再逐步
比较 loss、grad_norm 和完整恢复。
