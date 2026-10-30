# TorchLLMTuner、TorchTitan 与 TorchTitan-NPU 对比

## 1. 对比范围与口径

本文比较以下本地仓库的源码、配置入口、模型接入和测试结构：

- TorchLLMTuner：`/home/jianzhnie/llmtuner/llm/TorchLLMTuner`
- TorchTitan：`/home/jianzhnie/llmtuner/llm/torchtitan`
- TorchTitan-NPU：`/home/jianzhnie/llmtuner/llm/torchtitan-npu`

| 仓库 | 当前提交 | 定位 |
|---|---|---|
| TorchLLMTuner | `eb21191` | 独立的 HF 混合并行训练框架 |
| TorchTitan | `0f9fd2bd6` | PyTorch 原生的大规模生成式模型训练平台 |
| TorchTitan-NPU | `0a10bfc` | 基于固定 TorchTitan 版本的 Ascend NPU 插件 |

TorchLLMTuner 工作区可能包含未提交修改；另外两个仓库检查时无未提交改动。
本文是静态源码比较，未重新执行三套训练或性能基准。

能力说明采用以下口径：

- **原生集成**：已进入对应仓库的主要训练路径；
- **可选/实验**：需要专用依赖、模型、设备或实验入口；
- **继承**：由上游运行时提供，不代表插件仓重新实现；
- **缺口**：没有对应实现，或当前配置明确拒绝。

功能存在不等于所有模型、拓扑、精度和设备组合均已验收。

## 2. 结论摘要

三个项目不是同一层级的替代品：

- **TorchTitan 是完整上游平台**：覆盖预训练、SFT、RL，提供原生模型、
  五类并行、低精度训练、丰富的性能优化和真实模型 recipes。核心抽象是
  `Configurable`、`Trainer.Config` 和模型协议。
- **TorchTitan-NPU 是上游插件**：复用 TorchTitan 的 Trainer、配置树、模型
  协议和并行基础设施，通过 patch、extension、显式 override、模型模块和
  NPU 算子完成 Ascend 适配。它不是独立训练栈。
- **TorchLLMTuner 是独立精简实现**：以 HF `AutoModelForCausalLM` 为默认模型
  来源，移除 TorchTitan 的配置物化和模型基类协议，用聚合 dataclass、普通
  `apply_*` 函数和固定 stage 表组织训练。范围更集中，但模型、RL、低精度与
  设备优化覆盖不及前两者。

选择建议：

| 需求 | 更适合的项目 |
|---|---|
| 跟踪 PyTorch 原生训练技术、丰富模型与预训练到 RL 全流程 | TorchTitan |
| 在 Ascend NPU 上复用 TorchTitan，并使用 CANN、图编译和显存优化 | TorchTitan-NPU |
| 直接训练 HF Causal LM，研究精简的五类并行装配 | TorchLLMTuner |
| 需要成熟 recipes、Muon、MTP、LoRA、VLM 或 RL | TorchTitan；NPU 场景再评估 TorchTitan-NPU |
| 需要 Ascend 专用算子、Virtual/Swap Optimizer 或低精度训练 | TorchTitan-NPU |
| 需要扁平 CLI、较少框架基类和显式模块边界 | TorchLLMTuner |

## 3. 三者的依赖与继承关系

```text
TorchTitan
├── 原生模型与训练平台
├── experiments/transformers_modeling_backend（实验性 HF 后端）
└── TorchTitan-NPU
    ├── 复用上游 Trainer / TrainingEngine / Config / parallel runtime
    ├── patches + extensions（按包导入顺序加载）
    ├── override（由 override.imports 显式选择）
    └── NPU models + ops + 性能扩展

TorchLLMTuner
├── 独立 Trainer / Config / parallel runtime
├── 默认直接包装 HF 模型
└── 参考 TorchTitan 语义，但不作为其插件运行
```

关键区别：

1. TorchTitan-NPU 与 TorchTitan 是**插件和宿主**关系；升级上游时必须核对
   固定版本、patch 目标和 override 接口。
2. TorchLLMTuner 与 TorchTitan 是**独立实现和语义参考**关系；不能把上游
   新能力自动视为 TorchLLMTuner 已具备。
3. TorchTitan 的 HF modeling backend 位于 `experiments/`，不能与原生模型
   路径或 TorchLLMTuner 的默认 HF 路径混为一谈。

## 4. 核心架构

| 维度 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| 核心入口 | `Trainer(LLMTunerConfig)` | Python recipe 返回 `Trainer.Config`，再由 Trainer/TrainingEngine 执行 | `torchtitan_npu.train` 导入插件后调用上游 `torchtitan.train.main` |
| 配置机制 | 分组 dataclass 聚合；CLI 扁平标量 + Python 复杂字段 | `Configurable` 配置树、嵌套 `Config.build()`、Python recipes | 继承上游；增加 NPU schema、转换和 override |
| 模型协议 | 普通 HF `nn.Module` + `HFTransformerModel` 五部件视图 | 原生 `Module`、`BaseModel`、sharding/state-dict 协议；另有实验 HF backend | 继承上游协议，并添加 NPU 模型和适配 |
| 并行装配 | `parallel/stages.py` 声明固定 stage 表 | 模型声明 sharding，TrainingEngine/parallelism 模块组合 | 主要继承上游，模型和 override 提供 NPU 特化 |
| 扩展方式 | 普通函数、adapter、组合矩阵 | config override、模型组件、experiments | patch、extension、override、models、ops 分层 |
| 生命周期 | 独立 Trainer 全部管理 | Trainer 管数据/验证/报告，TrainingEngine 管执行/优化/checkpoint | 上游生命周期 + NPU 扩展 |

### 4.1 配置与模型抽象的取舍

TorchTitan 的 `Configurable` 让组件配置可以递归物化，并通过 `override`
替换配置树中的组件；代价是组件通常同时维护配置类和实现类，模型还须满足
初始化、sharding 和 remat 等协议。

TorchLLMTuner 将配置流收敛为：

```text
CLI / Python → LLMTunerConfig → 顶层装配 → 显式参数
```

模型只需满足 HF wrapper 的 forward、部件视图和 TP plan 契约。这样减少了
框架基类，但也放弃了上游通用配置树和组件递归 override 机制。

TorchTitan-NPU 保留上游抽象，因而能复用其训练语义和扩展点；相应地，NPU
适配必须严格处理导入顺序、配置转换和被替换符号的既有绑定。

## 5. 并行与计算能力

| 能力 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| FSDP2 / HSDP | 原生集成；dense/sparse mesh、细粒度参数处理和 CPU offload | 原生集成；另有 DDP、FlexShard 等路径 | 继承并扩展上游；含 NPU offload 和模型专用策略 |
| TP | HF TP plan；sequence-parallel 线性层；支持适用的 MoE-under-TP | 原生模型声明式 sharding、async TP | 继承，按模型和 NPU 算子适配 |
| PP | 已接入 1F1B、Interleaved1F1B | 1F1B、Interleaved1F1B、GPipe 及其他 pipelining schedules | 继承上游，实际范围取决于模型 recipe |
| CP | KV AllGather、Ulysses、KCP；适用的双轴 KV×Ulysses | KV AllGather、Ulysses | 继承上游，并有模型专用的长序列/attention 适配 |
| EP | HF MoE 转换；AllToAll、可选 TorchAO、MC2/Domino 专用路径 | 原生 EP、DeepEP/HybridEP、Dist-MoE | 继承并适配 DeepEP、模型路由和 NPU dispatch |
| Compile | 全模型/逐 block、regional inductor、async TP | regional compile、CUDA graphs、GraphTrainer 实验 | 继承并扩展 NPU Inductor、AutoFuse、GraphTrainer |
| 组合校验 | 配置、topology、`compatibility.py`、模型 probe | 配置树校验、模型/parallelism 契约和 override | 上游校验 + NPU 配置、patch 和模型约束 |

注意：

- 三者都出现五类并行名称，不代表同一模型上支持相同组合。
- TorchTitan 原生模型路径与实验 HF backend 的覆盖不同；后者文档明确列出
  MoE × PP 等未完成边界。
- TorchTitan-NPU 的并行能力必须按固定上游版本和具体 NPU recipe 判断。
- TorchLLMTuner 的 KCP、MC2/Domino 等专用实现不能外推为通用模型支持。

## 6. 模型与训练场景

| 维度 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| 默认模型来源 | HF `AutoModelForCausalLM` | 仓库内原生模型定义；HF backend 属实验 | 上游原生模型 + NPU 专用模型 |
| 模型覆盖 | 取决于 HF wrapper、TP plan、MoE 布局和专用 adapter | Llama、Qwen、DeepSeek、gpt-oss、Kimi、Muse、FLUX 等 | 重点包括 DeepSeek-V3.2/V4/V4.1、Qwen3.5 及上游已适配模型 |
| 文本预训练/SFT | 支持统一训练入口和数据管线 | 完整 recipes 和模型训练路径 | 继承并提供 NPU recipes |
| 多模态 | 有数据处理；当前 HF wrapper 是文本 decoder，不消费媒体/MRoPE | 原生支持多种文本、视觉、音频和生成模型，按模型而异 | 提供特定图文/视频模型适配，不能外推到全部上游模型 |
| LoRA / MTP / Muon | 无与上游同等的完整通用能力 | 已提供对应训练能力 | 提供特定模型 LoRA、MTP 和 Muon 扩展 |
| RL | 缺口 | TitanRL，覆盖训练与生成协同 | 本文未发现独立通用 RL 栈；按上游兼容性另行评估 |

TorchLLMTuner 的优势是 HF 接入路径短，而不是模型数量和训练场景更广。
如果模型需要视觉、RL 或成熟模型专用优化，应优先评估上游及其 NPU 插件。

## 7. 内存、精度与性能优化

| 能力 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| Activation checkpointing | full、selective、memory_budget、region | full、selective、regional remat | 继承并扩展 |
| Chunked / vocab-parallel loss | 集成，受 PP/validation 等组合约束 | 有对应训练技术与模型路径 | 继承 |
| CPU offload | FSDP 参数/梯度等策略 | FSDP CPU offload、BF16 optimizer states 等 | 上游能力 + NPU pinned/staging、optimizer offload |
| Activation/optimizer swap | 无通用实现 | 依具体路径，不是统一主卖点 | Virtual Optimizer、Swap Optimizer、NovaSwap 等专用扩展 |
| 低精度训练 | 无通用 MXFP8/NVFP4 训练栈 | MXFP8、NVFP4 等可选路径 | MXFP4/MXFP8、Block FP8、HiFloat8 等；部分限定 Ascend 950 |
| 设备算子 | 主要复用 PyTorch/DTensor/Inductor，少量专用适配 | PyTorch 原生及 CUDA/Flash/量化生态 | AscendC、Triton、TileLang、PyPTO、CANN 等 |
| 图优化 | 通用 compile 与 async TP | compile、CUDA graph、GraphTrainer | NPU AutoFuse、图编译和 SDC 等扩展 |

TorchTitan-NPU 的性能优势来自上游训练栈与 NPU 专用执行能力的组合。
TorchLLMTuner 当前更侧重可读性、通用装配和正确性，不应在缺少同环境测量时
推断其性能优劣。

## 8. 数据、Validation 与可观测性

| 维度 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| 数据框架 | Grain、文本 packing、random、JSONL/HF dataset、多模态处理和 renderer | Grain、packing、mixing、checkpointable state、chat/SFT 和多模态 | 继承上游，增加模型专用 chat/VLM 数据适配 |
| Validation | Trainer 内置，支持非 PP 与 PP schedule | Trainer/Validator 通用路径 | 继承并有 NPU extension |
| EMA | 在线 EMA、参数和适用 buffer、checkpoint 恢复 | 上游 EMA | 继承并验证特定 NPU 路径 |
| 指标 | loss、吞吐、TFLOPs、TensorBoard、W&B | 完整 metrics、TFLOPs/MFU、TensorBoard、W&B | 继承并适配 NPU 指标 |
| 调试 | profiler、memory snapshot、确定性模式 | profiler、memory snapshot、Flight Recorder、SDC replay、loss compare | NPU profiler、内存工具、SDC 和数值对齐脚本 |

## 9. Checkpoint 与状态兼容

| 维度 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| 主要后端 | DCP；可选 `torch_checkpointing` | DCP / `torch_checkpointing` 上游体系 | 继承上游，含 NPU 扩展 |
| 训练状态 | 模型、optimizer、scheduler、dataloader、trainer、RNG、EMA | 完整训练状态与 HF 转换工具 | 继承，按模型和扩展适配 |
| 异步保存 | disabled、threaded async、pinned-memory async | 上游异步 checkpoint | 继承并扩展 NPU staging/offload |
| HF 导入导出 | 内置 adapter；EP/TP 初始加载受布局约束 | HF safetensors 与 DCP 转换工具 | 继承，并有量化/模型专用 storage adapter |
| 跨 EP 布局 | 逻辑完整专家表示，按当前 EP 重新分片 | 依上游模型和 checkpoint adapter | 依上游及 NPU 模型 adapter |

三者的 FQN、模型定义、parallel layout 和 checkpoint metadata 不相同。
不能假设 DCP 目录可以直接互换；迁移应通过共同可解释的 HF 权重或显式
转换器，并验证模型、optimizer 和训练进度中需要保留的部分。

## 10. 配置与使用方式

### TorchLLMTuner

```bash
python -m llmtuner --steps 20
```

- 标量通过扁平 CLI，复杂字段通过 Python；
- 默认从 HF 架构或本地模型构造；
- 适合快速运行和阅读完整训练路径。

### TorchTitan

```bash
MODULE=torchtitan_recipes.models.llama3 CONFIG=llama3_8b ./run_train.sh
```

- recipe 是返回完整 `Trainer.Config` 的 Python 函数；
- CLI 选择 `--module` / `--config`，不提供通用的 `--section.option` 配置；
- 配置与模型 sharding 紧密关联，改并行度时可能需要重建 model config；
- 适合可复制、可编程和可维护的训练配方。

### TorchTitan-NPU

```python
from torchtitan.train import main
import torchtitan_npu

main()
```

- 训练入口先导入插件，再进入上游 main；
- 自动导入顺序为 `patches → config → extensions → ops`；
- 具体 override 由 `override.imports` 显式选择；
- 使用上游 recipe 机制及 NPU 配置/模型示例，不能套用 TorchLLMTuner 的扁平参数。

## 11. 依赖与软件栈

| 项目 | Python | 关键依赖口径 |
|---|---|---|
| TorchLLMTuner | `>=3.10` | `torch>=2.12`、`transformers>=5.11,<5.12`、`spmd_types==0.2.5`、Grain |
| TorchTitan | `>=3.11` | 推荐近期 PyTorch nightly；`spmd_types==0.3.1`、Grain、torch_remat、torch_checkpointing 等 |
| TorchTitan-NPU | `>=3.12` | 固定 `torch==2.14.0`、对应 torch-npu、`torchtitan==0.3.0`、CANN/Triton Ascend 等 |

依赖差异本身就是迁移风险：不能只比较 Python API，还要核对 PyTorch、
Transformers、SPMD、checkpoint 和设备 runtime 的版本组合。

## 12. 测试与验证策略

| 项目 | 测试特点 |
|---|---|
| TorchLLMTuner | CPU pytest + 独立多进程等价性脚本；强调分片与全量参考、状态恢复和环境能力门控 |
| TorchTitan | CPU/GPU unit tests、集成 recipes、H100/B200/ROCm/RL CI、真实模型收敛和性能验证 |
| TorchTitan-NPU | CPU UT、NPU smoke/ST、模型 integration tests、loss compare 和数值对齐工具；强调固定上游与真实 NPU 证据 |

比较测试结果时必须区分：

- 原生模型与 HF 模型；
- 上游默认路径与 experiments；
- 插件静态测试、CPU 测试与真实 NPU 训练；
- 单并行族、组合拓扑和完整训练恢复；
- loss/grad_norm 数值对齐与吞吐/MFU 性能结论。

本文没有重新运行这些测试，因此不给出三方当前通过率或性能排名。

## 13. 目录映射

| 职责 | TorchLLMTuner | TorchTitan | TorchTitan-NPU |
|---|---|---|---|
| 训练入口 | `llmtuner/trainer/cli.py`、`trainer.py` | `torchtitan/train.py`、`trainer.py`、`training_engine.py` | `torchtitan_npu/train.py` + 上游入口 |
| 配置 | `llmtuner/config/` | `torchtitan/config/`、`torchtitan_recipes/` | `torchtitan_npu/config/` + 上游 recipes |
| 模型 | `models/hf/`、`common/`、`kimi/` | `torchtitan/models/` | `torchtitan_npu/models/` + 上游模型 |
| 并行 | `llmtuner/parallel/` | `torchtitan/distributed/` | 上游 distributed + extensions/override/models |
| 设备算子 | `accelerator/` 与少量模型专用实现 | PyTorch/CUDA 生态和模型组件 | `torchtitan_npu/ops/` |
| 数据 | `llmtuner/datasets/` | `torchtitan/components/data/` | 上游 + NPU 模型/override dataloader |
| Checkpoint | `components/checkpointer/` | `components/checkpointer/` | 上游 + NPU extensions/tools |
| 实验扩展 | 无同规模独立体系 | `torchtitan/experiments/`、`torchtitan/rl/` | `experiments/`、NPU extensions |

## 14. 各自优势与主要代价

### 14.1 TorchLLMTuner

优势：

- HF 模型接入直接，配置与函数调用链较短；
- 五类并行使用统一 stage 表和明确组合裁决；
- 训练、validation、EMA、checkpoint 和等价测试集中在较小代码面；
- 适合研究、二次开发和定位 HF 并行数值问题。

主要代价：

- 模型与训练场景覆盖较小，无通用 RL/VLM 训练栈；
- 缺少上游成熟的低精度、Muon、MTP、LoRA 和性能生态；
- NPU 专用算子和大规模 recipes 不如 TorchTitan-NPU；
- 独立于上游意味着需要持续维护语义映射和版本适配。

### 14.2 TorchTitan

优势：

- PyTorch 原生、模型与训练场景最广；
- 并行、compile、低精度、RL、调试和 checkpoint 能力完整；
- Python recipes 与组件 override 适合可复现研究；
- 真实模型 CI、收敛和性能工作流更成熟。

主要代价：

- `Configurable` 和模型协议增加理解与适配成本；
- 核心模型、实验 HF backend 和可选依赖的覆盖需要分别判断；
- 近期特性依赖新 PyTorch，环境升级和版本匹配要求较高。

### 14.3 TorchTitan-NPU

优势：

- 最大化复用 TorchTitan 训练语义与 recipes；
- Ascend 算子、图编译、长序列、低精度和显存优化更深入；
- 模型专用 NPU 集成和真实系统测试更有针对性；
- patch/extension/override 分层使多数适配无需修改上游 checkout。

主要代价：

- 强依赖固定上游和 NPU 软件栈，升级需要系统复验；
- 自动 patch 与显式 override 的导入顺序和符号绑定较复杂；
- 某些能力限定特定模型或 Ascend 950，不能泛化到所有 NPU；
- 插件中的模型/扩展迁移债务需要持续收敛。

## 15. 迁移与选型建议

### 从 TorchLLMTuner 迁移到 TorchTitan

1. 先决定使用原生模型还是实验 HF backend，两者不能按同一覆盖承诺处理；
2. 将扁平配置改写为完整 Python recipe 和 `Trainer.Config`；
3. 将 HF wrapper/apply 函数映射为模型 sharding、初始化和 TrainingEngine 契约；
4. 用 HF 权重或正式转换工具迁移 checkpoint，不复制物理 shard；
5. 固定初始权重、数据顺序和精度，逐步比较 loss、grad_norm 和关键状态。

### 从 TorchTitan 迁移到 TorchTitan-NPU

1. 使用插件要求的上游版本，而不是直接假设最新 main 兼容；
2. 从对应 NPU 模型 recipe 开始，明确启用的 extensions 和 overrides；
3. 先关闭可选量化和图优化建立 BF16/FP32 基线，再逐项启用；
4. 核对设备、CANN、torch-npu、TorchAO 和 kernel 支持矩阵；
5. 验证完整 checkpoint 恢复，而非只比较前向或短程 loss。

### 从 TorchTitan 迁移到 TorchLLMTuner

1. 确认目标模型能够由 HF wrapper 表达，并核对 TP/EP 布局；
2. 将 recipe 拆成 model/parallel/optimizer/training 配置；
3. 移除对原生模型协议、RL 或上游专用优化的隐式依赖；
4. 按 `STAGE_ORDER` 重建并行装配，逐项检查不支持组合；
5. 通过共同 HF 权重迁移，并重新验证数值和完整恢复。

## 16. 最终建议

- **优先 TorchTitan**：需要上游最新训练技术、广泛模型、成熟 recipes 或 RL。
- **优先 TorchTitan-NPU**：目标明确是 Ascend，并希望保留 TorchTitan 的模型
  和训练体系，同时使用 NPU 专用性能能力。
- **优先 TorchLLMTuner**：主要训练 HF Causal LM，重视精简、显式装配和
  并行数值研究，能够自行承担模型覆盖与设备优化验证。

如果建设长期统一系统，较合理的方式是明确选择一条主干：要么以
TorchTitan/TorchTitan-NPU 为训练和设备底座，要么以 TorchLLMTuner 为精简
控制面并有选择地移植能力。不要同时维护三套配置、模型和 checkpoint 语义，
再用名称相同的并行度推断它们等价。

## 17. 主要源码索引

TorchLLMTuner：

- `llmtuner/trainer/trainer.py`
- `llmtuner/config/llmtuner_config.py`
- `llmtuner/models/hf/model.py`
- `llmtuner/parallel/`

TorchTitan：

- `torchtitan/train.py`
- `torchtitan/trainer.py`
- `torchtitan/training_engine.py`
- `torchtitan/config/`
- `torchtitan/models/`
- `torchtitan/distributed/`
- `torchtitan/experiments/transformers_modeling_backend/`

TorchTitan-NPU：

- `torchtitan_npu/train.py`
- `torchtitan_npu/config/`
- `torchtitan_npu/patches/`
- `torchtitan_npu/extensions/`
- `torchtitan_npu/override/`
- `torchtitan_npu/models/`
- `torchtitan_npu/ops/`
- `docs/architecture.md`
