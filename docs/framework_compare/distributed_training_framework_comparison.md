# FSDPTurbo、TorchTitan 与 HyperParallel：模型与分布式训练如何解耦

大模型训练框架需要同时应对模型结构、并行策略和训练流程的变化。接入一个新的 MoE 模型，往往不仅要增加 `forward()`，还要明确专家参数如何打包、路由分数在哪里相乘、参数替换后如何恢复 optimizer 状态。如果这些知识散落在训练器和并行实现中，新增模型就会牵动整个系统。

因此，架构设计的关键在于：**模型知识由谁表达，并行层如何消费这些知识，训练循环通过什么接口完成计算与恢复。** 本文围绕这三个问题，比较 FSDPTurbo、TorchTitan 和 HyperParallel，并讨论对 TorchLLMTuner 的启示。

## 阅读导航

- [一、调研范围与评价方法](#一调研范围与评价方法)
- [二、总体结论](#二总体结论)
- [三、FSDPTurbo：外部包装与固定装配](#三fsdpturbo外部包装与固定装配)
- [四、TorchTitan：模型协议与训练引擎](#四torchtitan模型协议与训练引擎)
- [五、HyperParallel：适配器、计划与执行器](#五hyperparallel适配器计划与执行器)
- [六、横向比较](#六横向比较)
- [七、对 TorchLLMTuner 的建议](#七对-torchllmtuner-的建议)
- [八、源码索引与延伸阅读](#八源码索引与延伸阅读)

## 一、调研范围与评价方法

本文依据本机以下仓库的源码，以及已有的三份专项分析：

| 框架 | 源码位置 | 重点关注 |
|---|---|---|
| FSDPTurbo | `/Users/robin/work_dir/FSDPTurbo` | 外部包装器、并行计划、固定变换顺序 |
| TorchTitan | `/Users/robin/work_dir/torchtitan` | 模型协议、分片声明、Trainer 与 TrainingEngine |
| HyperParallel | `/Users/robin/work_dir/hyper-parallel` | 模型适配器、ShardingPlan、planner/builder、训练组件 |

本文分析的是设计方式，不是性能排名，也不构成目标硬件上的数值或吞吐验收。源码路径随版本变化；例如，旧分析中的 TorchTitan `models/base/`，在当前所读源码中对应的核心协议位于 `protocols/`。历史文件数和代码行数不作为比较依据。

### 解耦需要看四个层面

| 层面 | 需要回答的问题 | 常见风险 |
|---|---|---|
| 代码依赖 | 通用并行代码是否直接导入具体模型？ | 循环依赖、加载成本和修改范围扩大 |
| 结构约定 | 是否依赖固定模块路径、属性名和参数形状？ | 漏匹配、误匹配、版本升级失效 |
| 数学语义 | router、专家布局、mask、状态递推由谁定义？ | 张量形状正确，但数值结果错误 |
| 生命周期 | 初始化、并行变换、重计算、编译与恢复如何排序？ | 参数引用失效、重复通信、恢复后行为变化 |

少量 import 不等于完全解耦。合理的目标是让依赖通过明确契约表达，并把模型特有语义留在能够解释它的位置。

文中 TP、EP、CP、PP 分别指张量并行、专家并行、上下文并行和流水线并行；FSDP/HSDP 指全分片/混合分片数据并行。FQN 指模块或参数的完整限定名称。

## 二、总体结论

三个框架都把并行装配与训练迭代做了分层，但模型知识的归属不同。

| 框架 | 主要设计方式 | 模型知识主要放在哪里 | 训练侧的主要抽象 |
|---|---|---|---|
| FSDPTurbo | 配置选择目标，外部包装器按固定顺序变换模型 | 模型约定、配置、patch 和专用并行实现 | `BaseTrainer` 及其构建/训练扩展点 |
| TorchTitan | 模型声明初始化和分片语义，通用引擎消费协议 | `Module`、`BaseModel`、模型组件及 sharding 配置 | `Trainer` + `TrainingEngine` |
| HyperParallel | 适配器声明能力，planner 生成计划，builder 执行变换 | 模型适配器、计划规则及部分通用推导逻辑 | `BaseTrainer` 阶段 + `TextTrainer`/`VLMTrainer` |

这三种方式是不同的工程取舍，不构成先后演进或优劣排序。外部包装缩小模型改动，模型协议明确语义归属，适配器与计划则为外部模型兼容提供中间层。

还需要区分两个时点：**训练前完成并行装配，训练中执行并行通信。** all-to-all、梯度归约和 PP 调度不会因为装配提前完成而消失；解耦的意义是让训练器调用执行接口，而不是在业务循环中编写每种并行的细节。

## 三、FSDPTurbo：外部包装与固定装配

### 3.1 组件如何连接

模型创建由 `ModelFactory` 处理，支持自定义函数、注册构造器和 HuggingFace `from_pretrained()`。`FSDPTurbo(config, model)` 接收已经创建的模型，应用并行与内存策略。`BaseTrainer.setup()` 通过 `build_*()` 方法组装模型、优化器、数据和监控组件，随后进入训练循环。

```text
BaseTrainer.setup()
  ├─ build_tokenizer()
  ├─ build_model()：创建模型并接入并行包装
  ├─ build_optimizer() / build_scheduler()
  └─ build_dataloader() / checkpoint / monitor
       ↓
BaseTrainer.train()
  └─ prepare_batch() → train_step() → 更新与记录
```

这里的箭头表示职责关系，不要求所有模型都使用相同构造函数。

### 3.2 并行层如何改造模型

`FSDPTurbo` 构造入口按以下顺序组织能力：

```text
parallel state → 量化 → TP → TP 元信息 → CP → 模块 patch
  → EP → VP → pre-FSDP hook → 重计算 → 激活交换 → Chunk Batch → FSDP
```

TP 通过模块匹配构造 `ColwiseParallel`、`RowwiseParallel`、`SequenceParallel`，再调用 PyTorch `parallelize_module()`。EP 为选中的专家容器建立分片和 dispatcher。FSDP 根据计划选择包装单元、忽略项、策略及预取关系；部分专家分片在 EP 阶段完成，不能把末尾 FSDP 理解为所有参数唯一的包装时点。

模型可通过 `pre_fsdp_hook` 为非标准存储做准备。native/custom FSDP 差异通过策略接口隔离，mesh、group 和 rank 则通过 `ParallelState` 统一访问。

### 3.3 收益与边界

这种设计的优势是装配路径直接：配置决定目标，包装器决定顺序，并行函数执行变换。对满足既有结构约定的模型，可以减少对模型主体的修改。

但复杂度仍然存在于模块命名、专家布局、配置 pattern、patch 和安装顺序中。全局并行状态方便调用，也增加了测试隔离和多拓扑共存的难度。借鉴时应保留策略重叠检测、共享参数检查和 rank 失败协调，不能把较短入口等同于更小的正确性责任。

## 四、TorchTitan：模型协议与训练引擎

### 4.1 模型显式声明训练所需的语义

当前源码的 `protocols/module.py` 定义配置化 `Module`，承载参数初始化、buffer 和 sharding 等约定；`protocols/model.py` 的 `BaseModel` 在此基础上提供输入预处理、并行生命周期和流水线接口。

几个重要接口是：

- `Config.build()`：根据配置创建模块或模型。
- `Config.set_sharding_()`：设置模型及子组件的分片配置。
- `preprocess_inputs()`：把数据转换为模型输入、labels 和额外参数，可包含 CP 输入处理。
- `parallelize()`：组织模型分片、重计算和 FSDP 生命周期。
- `pipeline()`：结合模型首尾模块声明和配置，调用通用 PP 切分与调度实现。
- `state_dict_adapter_cls`：指定外部权重格式的转换适配器。

这些接口允许默认实现、继承和模型级覆盖，并不意味着每个模型都必须重新实现所有能力。

### 4.2 Trainer 与 TrainingEngine 分工

`Trainer` 负责数据、验证和指标策略，创建并持有 `TrainingEngine`。引擎负责模型执行、优化、checkpoint、性能分析和重放等通用机制。

```text
Trainer 初始化
  → 准备模型配置和 sharding 声明
  → 创建 TrainingEngine，由引擎组织模型构建与并行化
  → 构建数据、验证和指标组件

Trainer.train_step()
  → TrainingEngine.forward_backward()
  → TrainingEngine.optim_step()
```

`BaseModel.parallelize()` 的默认流程包含 local compile 绑定、SPMD 上下文、`_parallelize()`、激活重计算和 `_apply_fsdp()`。模型可以覆盖该流程，因此它是默认生命周期，不是所有模型的不可变调用链。

### 4.3 为什么 distributed 目录不能代表全部并行实现

| 能力 | 主要职责位置 |
|---|---|
| 初始化与分片声明 | `protocols/`、模型配置和模型组件 |
| TP | 模型 Linear、Attention、sharding 声明和相关执行机制 |
| EP | 模型 MoE 组件及通用专家通信实现 |
| CP | 输入切分工具与 attention 侧的布局、通信和数学处理 |
| FSDP / PP / mesh | `distributed/` 通用基础设施与模型边界声明 |
| Checkpoint | 训练组件和 state-dict adapter |

TorchTitan 把较多并行数学放在模型及公共模型组件中，因此不能只统计 `distributed/` 的文件数。其主要收益是模型结构、初始化与数学约束靠近所有者；代价是模型需要适配协议，迁移外部 HF 模型时仍要处理结构和状态格式差异。

## 五、HyperParallel：适配器、计划与执行器

### 5.1 区分底层原语、模型装配和训练应用

HyperParallel 的几个目录承担不同层级的职责：

| 层级 | 位置 | 主要职责 |
|---|---|---|
| 分布式原语 | `core/` | DeviceMesh、DTensor、布局重分布、TP/FSDP/PP 等底层能力 |
| 模型装配 | `distributed/` | 计划推导、参数分片、CP/EP 集成、forward 包装及 FSDP 适配 |
| 模型接入 | `models/` | 模型族适配器、注册、构造、状态映射和配方 |
| 训练服务 | `components/`、`data/` | loss、优化器、checkpoint、量化与数据管线 |
| 训练应用 | `trainer/`、`scripts/` | 配置解析、组件构建、训练循环和生命周期 |

模型 API 的 `from_pretrained()` 门面不依赖 Trainer，支持程序化接入。训练应用则由 YAML 与命令行覆盖生成 `TrainerConfig`，再创建 `TextTrainer` 或 `VLMTrainer`。底层 API 和训练应用是不同入口，不应混成一条强制调用链。

### 5.2 从能力声明到模型变换

```text
模型 registration/provider
  → ModelAdapterSpec：声明模型能力与专用规则
  → ShardingPlanner：推导并合并覆盖规则
  → ShardingPlan：记录参数布局、通信边界和计算注入
  → builder/apply：执行模型变换
```

`ModelAdapterSpec` 可声明模块替换、CP/EP provider、FSDP 单元、重计算、loss、checkpoint 和物化策略。registry 延迟发现模型族，再按需加载 provider，降低通用层对具体模型实现的直接依赖。

`ShardingPlan` 是解析与执行之间的中间表示，包含参数 placement、输入输出 layout、通信边界、特殊参数处理和 tied weights。planner 可结合通用命名规则与用户覆盖生成计划；特殊区域可通过 `derive=False` 等配置关闭自动推导，改为显式声明。

builder 的主要阶段为配置规范化、参数分片、特殊参数处理、forward 包装和 tied weights 处理。生产与 validate 路径可共享局部计算及通信逻辑，但对 DTensor 检查与局部张量执行采用不同处理方式。

### 5.3 适配器减少了什么耦合

适配器减少的是通用层直接导入具体模型、集中维护模型发现和加载全部重型依赖的需求。它也让专用 CP、EP、FSDP 规则有了明确归属。

但结构和数学耦合仍需单独评估：

| 区域 | 仍需关注的模型知识 |
|---|---|
| Planner | FQN、projection 名称、attention/MLP/MoE 角色推断 |
| EP routing 与旧 recipes | 模型族别名、router 与 shared expert 组合 |
| KDA CP | Kimi projection API、状态递推、卷积和输出形式 |
| FSDP adapter | 自动 decoder/expert 发现与模型配置识别 |
| Forward 包装 | mask/cache、参数和返回结构、重计算及编译顺序 |

因此，更准确的评价是：**直接依赖方向较清晰，结构与数学契约仍然重要，部分模型专用实现仍有迁移空间。** 其值得借鉴的部分是能力声明、延迟注册和计划中间表示；使用这些机制时，应避免让自动推导成为数值正确性的唯一依据。

## 六、横向比较

### 6.1 三个边界分别放在哪里

| 问题 | FSDPTurbo | TorchTitan | HyperParallel |
|---|---|---|---|
| 模型从哪里来？ | HF 或自定义工厂 | 模型 Config 与协议体系 | HF 兼容入口及模型族适配器 |
| 并行知识由谁表达？ | 配置、模型约定及专用实现 | 模型及子组件显式声明 | 适配器规则与 planner 推导 |
| 谁执行模型变换？ | 外部包装器和各并行函数 | 模型并行生命周期与通用基础设施 | builder 根据 ShardingPlan 执行 |
| 谁隔离训练流程？ | BaseTrainer 扩展点和训练服务 | Trainer 与 TrainingEngine | Trainer 阶段和配置化组件 |
| 主要维护成本 | pattern、patch、顺序和隐式约定 | 协议实现与模型迁移 | adapter、推导规则和运行时包装契约 |

### 6.2 四个容易混淆的判断

**入口短，不代表系统复杂度低。** FSDPTurbo 的部分复杂度在配置和 dispatcher，TorchTitan 的部分复杂度在模型组件，HyperParallel 的部分复杂度在适配器与 builder。应沿完整调用链比较。

**声明式配置，不等于完整执行计划。** 并行度和 pattern 只是请求；可检查的执行计划还需要明确目标、布局、ownership、冲突及前后置条件。

**训练循环统一，不等于执行路径完全相同。** PP 需要 schedule，CP 需要 attention 通信，MoE 需要 dispatch。可以统一引擎接口，但不能省略这些不同的执行语义。

**可加载模型，不等于支持所有并行组合。** HF 权重能够加载，只证明构造和格式路径可用；TP、EP、CP、PP、FSDP 及其组合仍需分别验证几何、状态和数值结果。

## 七、对 TorchLLMTuner 的建议

TorchLLMTuner 既有实现与迁移路径还涉及 HF 适配、参数替换、NPU/HCCL、PP ownership 和 checkpoint 恢复。借鉴上述框架时，应逐步收敛职责，而不是按目录规模删减逻辑。具体状态与实施批次统一维护在[架构设计及重构路线图](llmtuner_refactoring_roadmap.md)。

### 7.1 建立五个稳定边界

```text
模型：定义结构、数学、初始化和状态契约
  ↓
计划解析：绑定目标，检查布局、ownership、能力与冲突
  ↓
并行执行：按已验证计划分片、通信和包装
  ↓
训练引擎：执行前后向、优化和并行 schedule
  ↓
Trainer：组织数据、累积、日志、保存恢复和资源关闭
```

这是职责关系，不要求创建五套管理类。模型契约可从少量属性、方法或 Protocol 开始，已有叶子模块能承担职责时就不再新增转发层。

### 7.2 六项可落地的改进

1. **生命周期与并行算法分离。** BaseTrainer 负责组件初始化、数据迭代、梯度累积、记录和恢复；TP、EP、CP 与 PP 的执行细节交给引擎和并行后端。
2. **模型提供轻量协议。** 统一输入预处理、forward、并行能力及状态适配入口。简单模型可用默认适配器，复杂模型显式声明专家数学、参数轴和 PP 边界。
3. **先解析计划，再修改模型。** 在变换前校验目标、布局、共享参数和能力组合；训练前完成装配，训练中由固定执行路径调用通信和 schedule。重计算与量化也需声明生命周期依赖。
4. **训练服务支持替换。** loss、optimizer、scheduler、checkpoint、metrics 和 profiler 通过明确接口组合；文本、VLM 和 MoE 在兼容的执行接口下复用训练引擎。
5. **模型差异保留在模型侧。** router score、packed halves、KDA 状态递推和 pipeline I/O 不依赖通用层猜测。模块替换返回旧到新映射，执行前检查过期绑定。
6. **格式转换与恢复编排分离。** 模型 state adapter 处理 HF 外部键、打包和别名；checkpoint 服务管理模型、optimizer、scheduler、RNG、数据游标及保存加载过程。跨拓扑恢复单独定义支持范围。

### 7.3 用验证结果约束支持声明

结构迁移应对照初始化、logits、loss、梯度、参数更新、共享身份和 RNG。分布式验收覆盖适用拓扑、空专家、非对称输入、失败传播和独立恢复；重计算、编译、swap 与 PP 的组合需要单独验证。

模型装配失败不提供天然事务回滚；资源接口应明确 owner、等待、释放和异常传播。性能比较则固定模型、数据、精度和拓扑，记录最慢 rank、稳态吞吐、显存与 Host 峰值，并用 trace 确认实际执行路径。

## 八、源码索引与延伸阅读

以下路径相对各框架仓库根目录，用于定位本文讨论的实现。

| 框架 | 入口与关键实现 |
|---|---|
| FSDPTurbo | `fsdp_turbo/fsdp_turbo.py`；`fsdp_turbo/fsdp_turbo_config.py`；`fsdp_turbo/training/trainer.py`；`fsdp_turbo/training/factories/model_factory.py` |
| FSDPTurbo 并行 | `fsdp_turbo/distributed/parallel_state.py`；`fsdp_turbo/distributed/tensor_parallel/tensor_parallel.py`；`fsdp_turbo/distributed/expert_parallel/expert_parallel.py`；`fsdp_turbo/distributed/fully_shard_parallel/fully_shard_parallel.py` |
| TorchTitan | `torchtitan/protocols/model.py`；`torchtitan/protocols/module.py`；`torchtitan/protocols/sharding.py`；`torchtitan/trainer.py`；`torchtitan/training_engine.py` |
| TorchTitan 并行 | `torchtitan/distributed/fsdp.py`；`torchtitan/distributed/parallelism_context.py`；`torchtitan/config/parallelism.py` |
| HyperParallel | `hyper_parallel/models/api.py`；`hyper_parallel/models/adapter_spec.py`；`hyper_parallel/models/registry.py`；`hyper_parallel/trainer/text_trainer.py` |
| HyperParallel 装配 | `hyper_parallel/distributed/plan.py`；`hyper_parallel/distributed/apply.py`；`hyper_parallel/distributed/_builder/planner.py`；`hyper_parallel/distributed/_builder/fsdp_adapter.py` |
| HyperParallel 专用语义 | `hyper_parallel/distributed/expert_parallel/routing.py`；`hyper_parallel/distributed/context_parallel/kimi_delta_attention.py` |

专项分析与实施方案：

- [FSDPTurbo 框架对比](framework_compare/fsdpturbo_comparison.md)
- [TorchTitan 框架对比](framework_compare/torchtitan_comparison.md)
- [TorchLLMTuner 架构设计及重构路线图](llmtuner_refactoring_roadmap.md)
