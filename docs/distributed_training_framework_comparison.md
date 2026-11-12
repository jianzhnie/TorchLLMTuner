# FSDPTurbo、TorchTitan 与 HyperParallel：大模型训练框架如何实现解耦

大模型训练框架通常同时面对三类变化：模型结构不断演进，并行策略从单一 FSDP 扩展到 TP、PP、CP、EP 等组合，训练过程还需要持续加入混合精度、重计算、checkpoint 和性能分析。如果这些逻辑全部写进模型的 `forward()` 或训练脚本，代码很快会变得难以复用和验证。

本文阅读了以下三个源码仓库的当前实现，关注一个核心问题：**模型定义、分布式并行和通用训练循环分别由谁负责，它们通过什么接口连接起来？**

- FSDPTurbo：`/Users/robin/work_dir/FSDPTurbo`
- TorchTitan：`/Users/robin/work_dir/torchtitan`
- HyperParallel：`/Users/robin/work_dir/hyper-parallel`

## 先看结论：三种不同的解耦取向

三者都在训练开始前完成并行初始化，训练循环本身不直接编写 collective 通信。但它们选择的边界不同：

| 框架 | 模型侧抽象 | 并行侧抽象 | 训练侧抽象 | 主要特点 |
|---|---|---|---|---|
| FSDPTurbo | HuggingFace 或自定义 `nn.Module` | 外部 `FSDPTurbo` wrapper，按 plan 依次注入能力 | `BaseTrainer` 模板方法 | 接入普通 HF 模型成本低，并行逻辑集中；复杂模型依赖模块命名和 patch 约定 |
| TorchTitan | `BaseModel` + 嵌套 `Config` 协议 | 模型的 `parallelize()`、`_parallelize()`、`_apply_fsdp()` | `Trainer` + `TrainingEngine` | 协议最清晰，模型必须显式声明输入、分片、流水线和权重适配 |
| HyperParallel | HF AutoModel + 模型族 adapter + YAML recipe | DeviceMesh/DTensor、并行 API 和声明式 plan | `BaseTrainer` stages + `TextTrainer`/`VLMTrainer` | 把模型接入、硬件优化和训练组件拆开，面向 NPU 超节点扩展 |

可以把三者的核心调用链概括为：

```text
FSDPTurbo:   ModelFactory -> 原生模型 -> FSDPTurbo(wrapper + plans) -> BaseTrainer.train
TorchTitan:  Model.Config.build -> BaseModel.parallelize(context) -> TrainingEngine -> Trainer
HyperParallel: YAML/AutoModel -> Adapter + DTensor plans -> BaseTrainer stages -> Text/VLMTrainer
```

## FSDPTurbo：用外部 Wrapper 统一改造模型

### 代码分层

FSDPTurbo 的入口相对直接：模型工厂负责“创建模型”，`FSDPTurbo` 负责“把模型变成分布式模型”，`BaseTrainer` 负责“运行训练”。

1. **模型创建**：`fsdp_turbo/training/factories/model_factory.py` 中的 `ModelFactory` 支持自定义创建函数、注册表和 HuggingFace `AutoModelForCausalLM.from_pretrained()`。模型实现集中在 `fsdp_turbo/models/`。
2. **并行改造**：`fsdp_turbo/fsdp_turbo.py` 中的 `FSDPTurbo` 继承 `torch.nn.Module`，接收 `config` 和原始模型。构造过程会初始化 parallel state，然后按顺序应用量化、TP、CP、模块 patch、EP、VP、重计算、activation swap、chunk batch 和 FSDP/HSDP。
3. **训练执行**：`fsdp_turbo/training/trainer.py` 中的 `BaseTrainer.setup()` 通过 `build_tokenizer()`、`build_model()`、`build_optimizer()`、`build_scheduler()` 和 `build_dataloader()` 组装组件；`train()` 负责 epoch、梯度累积、监控、checkpoint 和 profiler，并把模型计算委托给 `prepare_batch()`、`train_step()` 等扩展点。

### 它是怎样解耦的？

FSDPTurbo 把并行逻辑放在模型外部。训练器拿到的不是原始模型，而是已经经过并行和内存优化处理的模型，因此 epoch loop 不需要知道 TP 的切分方式，也不需要直接调用 CP 或 EP 通信。

并行能力采用相互独立的注入函数，例如 `tensor_parallel_modules()`、`expert_parallelize_modules()` 和 `fully_shard_parallel_modules()`。是否启用、使用哪个 mesh、如何匹配模块，都由配置中的 plan 决定。对非标准存储，模型可以提供 `pre_fsdp_hook`，在 FSDP 包装前完成准备。

这种方式的优点是**侵入性低、适合快速接入大量现成模型**。局限也很明确：并行实现需要依赖模块路径、模块类型和 patch 约定；当模型结构特殊时，问题往往出现在注入顺序或匹配规则中，而不是模型类本身。

## TorchTitan：用模型协议连接模型与训练引擎

### 代码分层

TorchTitan 把“模型可被训练”定义成一组明确协议，而不是只要求它是一个 `nn.Module`。

1. **模型协议**：`torchtitan/protocols/model.py` 的 `BaseModel` 继承配置化 `Module`。每个模型拥有嵌套的 `Config`，并需要实现 `preprocess_inputs()`、`get_nparams_and_flops()` 和 `_apply_fsdp()`。模型还可以声明流水线首尾模块、state-dict adapter 以及 PP 支持情况。
2. **并行生命周期**：`BaseModel.parallelize()` 负责激活 SPMD 上下文、调用模型的 `_parallelize()`、应用 activation checkpoint，最后调用 `_apply_fsdp()`。模型配置通过 `set_sharding_()` 把并行信息传递给各个子模块。
3. **训练执行**：`torchtitan/trainer.py` 中的 `Trainer` 负责数据、验证和指标策略；`torchtitan/training_engine.py` 中的 `TrainingEngine` 负责 microbatch、非 PP/PP 的 forward-backward、优化器更新、CUDA Graph、checkpoint、profiling 和 replay。

### 它是怎样解耦的？

TorchTitan 的关键是把训练引擎依赖收敛到协议上。引擎不需要了解某个模型是 Llama、Qwen 还是 Kimi，只需要调用统一的输入预处理、forward/backward 和优化接口。模型则显式承担三类与自身结构相关的责任：将 dataloader batch 转换成模型输入和 labels；为子模块设置 sharding 和并行布局；描述 PP 切分边界以及 checkpoint 权重转换方式。

因此，模型替换通常不需要修改 `TrainingEngine`，并行组合也不会散落在通用训练循环中。相比 FSDPTurbo，这种方式的边界更清楚、静态约束更强；代价是接入新模型时需要实现完整协议，模型代码本身也必须理解一部分并行语义。

## HyperParallel：用 Adapter、Plan 和组件体系连接 HF 与硬件运行时

### 代码分层

HyperParallel 的目标不仅是支持多种并行，还要把 HuggingFace 模型适配到昇腾 NPU 和超节点运行时，因此它的分层比前两者更宽。

1. **模型接入**：`hyper_parallel/models/api.py` 提供不依赖 trainer 的 `from_pretrained()` facade；`models/<family>/` 保存模型族 adapter、注册逻辑、生成的 NPU modeling 文件和训练 recipe。模型可以复用 HuggingFace 的结构和权重格式。
2. **分布式核心**：`core/dtensor/` 提供 DeviceMesh、DTensor、placement 和 redistribution；`core/tensor_parallel/`、`core/fully_shard/`、`core/pipeline_parallel/` 以及 CP/EP 模块提供并行实现。Adapter 或 plan 描述模块替换、布局和通信规则，运行时再把这些规则注入模型。
3. **训练组件**：`hyper_parallel/models/README.zh.md` 描述了由 YAML 和 dotted CLI override 驱动的 `TrainerConfig`。`TextTrainer` 和 `VLMTrainer` 组合 `BaseTrainer` 的 setup、model、loss、data、optimizer、scheduler 和 callback stages。`scripts/train_lm.py` 只负责解析配置、启动 codegen sidecar、创建 trainer 并调用 `train()`。

### 它是怎样解耦的？

HyperParallel 把并行、重计算、offload、量化和算子替换视为系统能力，通过 adapter 和配置声明后注入模型。损失函数、优化器、checkpoint、量化模块和高性能算子则分别位于 `components/`，由配置中的 build target 组合。

这样，训练循环不需要绑定某个模型，也不需要绑定某种并行方式；模型只需满足 HF 结构和 adapter 契约。与此同时，DTensor layout、拓扑感知通信、NPU 多核调度和内存池化被留在 core/runtime 层，模型代码可以保持相对稳定。

它的优势是**扩展面最宽**，能够把模型生态、并行策略和硬件优化统一起来。相应的代价是接入链路较长：新模型除了要能加载，还要满足模块路径、布局、精度和 checkpoint 转换等契约，必要时还会涉及生成代码。

## 三者放在一起看：解耦边界在哪里？

| 问题 | FSDPTurbo | TorchTitan | HyperParallel |
|---|---|---|---|
| 谁创建模型？ | `ModelFactory` | `Config.build()` | HF facade / 模型注册器 |
| 谁决定并行？ | 外部 `FSDPTurboConfig` 和各类 plan | 模型 config + `ParallelismConfig` | YAML recipe + AdapterSpec + DeviceMesh/DTensor |
| 谁处理模型输入？ | Trainer 的 `prepare_batch()`，必要时由模型配合 | `BaseModel.preprocess_inputs()` | dataset/get_batch/data transform 组件与 adapter |
| 谁负责 forward/backward？ | `BaseTrainer.train_step()` | `TrainingEngine.forward_backward()` | `BaseTrainer` 的训练 runtime |
| 模型需要了解多少并行？ | 较少，但依赖模块约定和 hook | 较多，需要实现并行协议 | 适配器承担大部分并行接入，模型需满足布局契约 |
| 最适合的场景 | 快速改造 HF 模型、集中式并行封装 | 需要强协议和可组合并行的研究型训练 | NPU/超节点、硬件协同和大规模模型接入 |

三种方式没有绝对的优劣，实质上是三种工程取舍：外部包装把模型改动降到最低，但更多依赖运行时匹配规则；模型协议让边界最清晰，但把并行知识显式带入模型实现；适配器与声明式 plan 在两者之间增加一层，换取 HF 兼容性和硬件扩展能力。

## 对 TorchLLMTuner 的设计建议

结合这三个项目，可以采用“协议 + plan + 组件”的组合方式：

1. 保留一个只负责生命周期的 `BaseTrainer`，把初始化、数据迭代、梯度累积、日志和 checkpoint 与具体并行实现隔离。
2. 定义轻量 `ModelProtocol`，至少统一 `preprocess_inputs()`、`forward()`、`parallelize()` 和 checkpoint adapter；简单 HF 模型可以通过默认实现接入。
3. 为 TP、FSDP、CP、EP、PP、重计算和量化建立声明式 plan，在训练开始前一次性应用，训练 step 中不出现并行分支。
4. 将 loss、optimizer、scheduler、checkpoint、metrics 和 profiler 设计为可替换组件，使文本、VLM 和 MoE 共用同一训练引擎。
5. 对复杂模型保留显式 hook，用于声明模块边界、参数布局和流水线切分；不要把这些信息隐藏在大量字符串 patch 中。
6. 把并行后的模型状态、optimizer 状态和原始 HF 权重之间的转换独立出来，避免 checkpoint 逻辑反向污染模型定义。

## 参考源码

- FSDPTurbo：`fsdp_turbo/fsdp_turbo.py`、`fsdp_turbo/training/trainer.py`、`fsdp_turbo/training/factories/model_factory.py`
- TorchTitan：`torchtitan/protocols/model.py`、`torchtitan/trainer.py`、`torchtitan/training_engine.py`、`docs/composability.md`
- HyperParallel：`README.zh.md`、`hyper_parallel/models/README.zh.md`、`hyper_parallel/models/api.py`、`hyper_parallel/trainer/base.py`、`hyper_parallel/trainer/text_trainer.py`、`hyper_parallel/core/dtensor/`
