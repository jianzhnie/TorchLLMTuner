# TorchLLMTuner 设计文档

TorchLLMTuner 是直接基于 Hugging Face `transformers` 的大模型并行训练框架。
Python 包名为 `llmtuner`，支持 FSDP2 / HSDP、TP、PP、CP 和 EP。

本文说明长期有效的架构、模块职责与运行时契约，不记录逐次修复或单次实验结果。
TorchTitan 来源与 A/B/C/D 分类见 [上游映射](llmtuner_upstream_map.md)；
具体运行的验证范围以对应测试、日志和验收记录为准。

## 1. 设计目标

框架移除了 TorchTitan 的 `Configurable` 配置物化协议和 `Module` 模型协议，
但保留并行训练所需的显式边界，而不是简单删除复杂度。

- **单向配置流**：`CLI / Python -> LLMTunerConfig -> 顶层装配 -> 显式参数`。
  不遍历配置树，不递归调用 `build()`，也不在运行时隐式改写配置。
- **普通函数组合并行变换**：每个 `apply_*` 的依赖都写在函数签名中；
  底层模块不导入 Trainer，也不读取全局运行配置。
- **复用 HF 模型协议**：模型不要求继承框架基类；无法识别的结构必须明确报错。
- **显式能力边界**：不支持的配置、缺失的环境能力和不兼容的模型布局必须
  fail-fast，不能静默降级或改变数学语义。
- **结构变化不改变数值语义**：仅改变装配或实现的修改，必须由逐位一致或
  容差明确的等价性测试保护。

`torch.distributed` 提供 DeviceMesh、ProcessGroup、FSDP2 和流水线调度；
`spmd_types` 提供 SPMD 类型系统；`transformers` 提供模型构造和 `_tp_plan`。
这些依赖不属于需要重新实现的模型抽象。

## 2. 总体架构

```text
CLI / Python
     |
     v
LLMTunerConfig（config/）
     |
     v
Trainer（trainer/trainer.py）
     |  构建 mesh、HF wrapper、模型和训练服务
     v
parallelize_hf_transformers（parallel/parallelize.py）
     |
     |  TP -> EP -> CP -> AC -> Compile（可选）-> FSDP
     v
HFTransformerModel（models/hf/model.py）
     |
     |  线程局部分布式上下文
     v
accelerator/spmd.py <-> models/common/*
```

主要装配方向是 `trainer -> parallel -> models`。以下依赖是有意保留的底层原语，
不构成对 Trainer 的反向依赖：

- `models/common/tp_projections.py` 使用 TP collective + GEMM 原语。
- `models/common/embedding.py` 与 loss 共用词表分片边界函数。
- HF wrapper 调用 CP 输入分片、编译标注和 mesh 拓扑接口。
- EP/CP 装配代码可以引用 `models/common` 中的模型构件。

跨层依赖必须停留在小型数学原语或显式装配接口，不能形成隐式装配。
消费层的 rank 查询和普通归约经 `accelerator/distributed` 门面调用；
TP/CP fused kernel、FSDP、SPMD 上下文和 checkpoint 进程组等引擎代码可直接使用
`torch.distributed`。

### 2.1 目录职责

```text
llmtuner/
  __main__.py                   python -m llmtuner，转交 trainer/cli.py
  config/                       分组配置、聚合配置和 CLI 标量视图
  trainer/
    cli.py                      参数解析与进程组生命周期
    trainer.py                  训练、验证、优化器更新和状态恢复编排
    runtime/                    装配、batch、PP、validation 和归约辅助逻辑
  models/
    hf/                         HF 构造、包装、物化与 state-dict 适配
    common/                     通用 attention、MoE、投影和嵌入构件
    kimi/                       Kimi Linear 专用模型与布局适配
  parallel/
    topology.py                 mesh 与 rank ownership
    stages.py                   装配顺序的唯一声明
    parallelize.py              统一装配入口
    compatibility.py            跨层组合裁决
    model_compile.py            编译变换
    tensor_parallel/            TP 入口、linear kernel 和 checkpointing
    fully_shard_parallel/       FSDP2/HSDP 入口、plan 和细粒度参数处理
    pipeline_parallel/          PP 切分、schedule 和 microbatch 驱动
    context_parallel/           CP attention、输入分片和压缩路径
    expert_parallel/            EP 模型替换、dispatch、通信和 checkpointing
  memory/                       activation checkpointing 与 remat 区域
  accelerator/                  device、distributed、collectives、SPMD 和能力探测
  components/                   loss、optimizer/EMA、checkpoint、metrics 和 profiler
  datasets/                     数据源、组合、loader、packing、文本和多模态管线
  utils/                        日志、seed、GC、监控和通用工具
```

CPU 单测位于 `tests/unit_tests/cpu/`；分布式集成脚本位于
`tests/integration_tests/`。集成脚本可能是 `torchrun` 程序，也可能是自行启动
子进程的 Python 驱动，运行前须核对各自 docstring 中的命令和参数。

## 3. 横切契约

### 3.1 异常分类

框架异常均继承 `LLMTunerError`，同时保留相应的内建异常类型以兼容既有调用方。

| 异常 | 同时继承 | 用途 |
|---|---|---|
| `ConfigError` | `ValueError` | 字段值或配置本身无效，需要修改配置 |
| `UnsupportedCombinationError` | `NotImplementedError` | 单项合法，但当前组合不受支持 |
| `EnvironmentUnsupportedError` | `NotImplementedError` | 当前环境缺少所需版本或能力，并给出解锁条件 |

可选第三方包缺失仍使用 `ImportError`；模块内部未知枚举值或未实现的抽象方法
仍可使用原生 `NotImplementedError`。错误信息应说明原因、影响范围和可行的
修复方式，而不是只报告失败位置。

### 3.2 环境能力探测

`accelerator/capabilities.py` 集中探测 PyTorch 版本相关的私有配置项和可选能力：

- `has(name)` 查询并缓存结果；未知名称立即抛出 `KeyError`。
- `require(name, feature=...)` 在缺失时抛出带解锁条件的
  `EnvironmentUnsupportedError`。
- 调用点保留面向具体功能的错误文案；可选包导入和纯设备发现不强制走注册表。
- 无降级路径的硬依赖不需要伪装成可选能力。

### 3.3 配置校验与组合裁决

职责按决策所需信息划分：

1. 字段值域及只依赖配置的组合校验，放在各配置的 `__post_init__` 中。
2. 需要模型、拓扑或运行时信息的组合校验，集中在
   `parallel/compatibility.py`，于 assembly 阶段执行。
3. 需要实际 HF MoE 布局的校验，在 probe 阶段执行。

`compatibility.py` 的每个普通函数负责裁决和错误文案，`ENTRIES` 记录函数、
阶段、异常类型和守卫位置。触发条件保留在调用点，避免矩阵与实现重复维护条件。
它不是通用规则引擎，也不替代字段校验或环境能力探测。

### 3.4 公开 API

| 层级 | 入口 | 约定 |
|---|---|---|
| 稳定用户接口 | `llmtuner.LLMTunerConfig`、`llmtuner.Trainer`、`llmtuner.config.*`、`python -m llmtuner` | 示例和常规训练使用 |
| 集成接口 | `llmtuner.parallel.parallelize_hf_transformers`、`llmtuner.errors`、装配 stage、组合矩阵、能力注册表 | 扩展作者使用，稳定性弱于用户接口 |
| 内部实现 | 未列出的模块和私有符号 | 不承诺兼容，测试可直接使用 |

`llmtuner.trainer` 中的配置再导出仅用于兼容旧调用方；新代码从
`llmtuner.config` 导入。实现代码从定义符号的叶子模块导入，避免引入不必要的
引擎依赖。

## 4. 三个核心接缝

### 4.1 配置入口

`LLMTunerConfig` 组合四个顶层 dataclass：

```text
ModelConfig + ParallelConfig + OptimizerConfig + TrainingConfig
```

其余数据、checkpoint、scheduler、metrics 等配置归属于相应顶层配置。
CLI 使用 `HfArgumentParser` 解析标量视图；字典、嵌套 dataclass 列表和 callable
等无法由 CLI 表达的字段通过 Python 传入。`from_groups()` 创建新配置对象，
不修改调用方传入的配置实例。

Trainer 装配时解析 HF 架构并同步模型尺寸；CLI 解析本身不读取 HF 配置。
需要预览时可显式调用 `cfg.auto_fill_model()`。

只有顶层装配读取聚合配置。并行层接收 `ParallelConfig` 和明确的标量参数，
底层模型构件不接收完整运行配置。

### 4.2 HF wrapper

`HFTransformerModel` 是唯一的 HF 包装层，直接调用解析出的 `ForCausalLM` 类，
不通过 monkey patch 替换 HF 初始化。对并行层公开三个契约：

1. **前向接口**：接收一维 token 流以及显式 `positions`、`attention_masks`，
   返回 logits。wrapper 负责 batch 维、RoPE 位置和 BlockMask 适配。
2. **部件视图**：`named_children()` 呈现
   `tok_embeddings / layers / norm / lm_head / rotary_emb`。
   这只是遍历视图，不重注册模块，也不改变 state-dict key。
3. **TP plan**：`tp_plan` 将 HF `_tp_plan` 重写为 wrapper 内的实际模块路径。
   plan 是纯数据，因此不需要额外的模型 registry。

HF checkpoint 键转换由 state-dict adapter 负责，不能隐藏在 `named_children()` 中。

### 4.3 并行变换与分布式上下文

每个并行族由同名包提供公共入口。装配顺序由 `parallel/stages.py` 的
`STAGES` / `STAGE_ORDER` 唯一声明：

```text
TP -> EP -> CP -> AC -> Compile（可选）-> FSDP
```

- AC 包住已经完成 TP/EP/CP 改造的层。
- Compile 位于 AC 之后、FSDP 之前；PP 下每个 model part 使用相同顺序。
- FSDP 最后执行，使其 hooks 位于外层。
- PP 先切分 stage，再逐 part 装配，最后创建 schedule 并返回
  `PipelineParallelSetup`。
- 即使 dense DP 分片度为 1，只要 CP、复制轴或专家分片需要 FSDP，就不能跳过
  FSDP 装配。

模型构件通过 `accelerator/spmd.py` 的线程局部上下文读取 dense/sparse mesh。
`spmd_context(parallel_dims)` 管理作用域，`spmd_mesh_group()` 和
`spmd_mesh_size()` 提供查询；单例轴分别返回 `None` 和 1。
该上下文是模型内部唯一的 ambient 分布式状态入口。

## 5. 并行与模型设计

### 5.1 Mesh 与拓扑

`parallel/topology.py` 统一构建和校验 `ParallelTopology` 及 DeviceMesh。
world mesh 保留 PP 外轴，并派生数据加载、dense storage、dense forward/backward、
sparse EP、batch 和 loss 等视图。PP stage 从同一拓扑解析自己的子视图，
不能将 PP 简化为“不在 mesh 中”。

运行时模型代码通过 SPMD 上下文访问 mesh，避免重复传递或自行解释 rank 分组。
EP 复用已有 DP 设备轴；并行度乘积不能替代对实际 mesh 和 rank ownership 的检查。

### 5.2 TP

TP 由 HF `_tp_plan` 驱动。`ColumnParallelLinear` 和 `RowParallelLinear`
提供 sequence-parallel 形式的 collective + GEMM 边界；plan 缺省时读取
`model.tp_plan`。无法安全处理的 bias 或布局必须明确拒绝。

支持的 MoE-under-TP 路径沿专家 hidden 维切分 routed experts，router 保持复制，
并通过块边界 all-gather / reduce-scatter 对齐序列分片。标准 shared expert
可在相同边界内切分；非标准布局和不受支持的 shared-expert × TP × EP 组合会被拒绝。
复制参数的梯度归约必须排除已经按 TP 或 EP 切分的参数。

`colwise_gather_output` 当前保留 lm_head 复制；loss 已能接收 vocab-parallel
参数，但这不表示所有 gather-output 路径均已实现。

### 5.3 CP

CP 支持 KV all-gather 和 Ulysses token/head 交换，并支持相应 SDPA/Flex
实现路径。输入分片同步处理 `input_ids`、labels 和 positions，attention mask
按所用 kernel 的语义分片或保留全局文档结构。

主要约束：

- 序列长度必须满足 CP/TP 联合分片的整除要求；headtail 还需满足额外整除条件。
- Ulysses 要求 attention head 数满足切分要求，且不与 load balancer 组合。
- packed 文档边界是全局元数据，不能从已切分 positions 中猜测。
- 双轴 KV × Ulysses 的查询偏移按 KV local rank 计算，不能直接使用总 CP rank。
- 完整 SDPA 文档 mask 占用 `O(T²)` 空间，不代表稀疏 attention kernel 的性能。

### 5.4 EP

EP 通过结构探测将可表达的 HF MoE 块替换为原生 router、grouped experts 和
dispatcher。EP=1 使用本地 dispatch；EP>1 使用本地专家切片和 EP 通信组。
AllToAll 是默认后端，TorchAO padded dispatcher 是可选后端；DeepEP/HybridEP
仍按环境或缺失实现明确拒绝。

辅助负载均衡 loss 通过梯度注入参与训练，不改变主 loss 数值。专家与 dense
参数的梯度范数分别归约，再使用同一个全局裁剪系数。

EP checkpoint 以完整专家张量为存储表示：保存前汇集专家切片，加载时按当前
EP 布局重新分片，从而支持改变 EP 度数。保存前校验各 EP 子组的状态键、shape、
dtype、layout 和张量类型；该协议不等同于对任意通信故障或任意模型布局的保证。

### 5.5 PP

PP 先按层归属切分 stage，再为每个 model part 执行标准装配。
1F1B 和 Interleaved1F1B 使用 PyTorch pipelining schedule；schedule 的
`scale_grads=False`，loss 由 Trainer 统一归一化。

首 stage 接收输入，末 stage 接收 labels 并产生 loss；其他 stage 返回设备上的
零 loss 哨兵。多 model-part optimizer 由同一个 `OptimizersContainer` 管理，
optimizer state 使用参数 FQN，避免 stage 内位置索引冲突。

microbatch 切分要求非标量张量能按第 0 维对齐。多 microbatch 暂不支持一维
packed corpus，也不支持按媒体 span 或坐标轴重新分配非行布局张量。
tied embeddings、DSA × PP、chunked loss × PP 等不兼容组合明确拒绝。

### 5.6 FSDP2 / HSDP

FSDP 最后包装模型，dense 与专家参数使用各自的 mesh 和所有权规则。
`fsdp_reduce_dtype` 独立控制梯度通信精度，不改变参数存储 dtype；默认 FP32。
选择 BF16 或 FP64 时必须分别验证后端支持、数值和性能。

`fsdp_apply_modules` 的 `ignored_params`、`hook_module` 和
`fsdp_ignored_modules` 使用对象引用。装配在 TP/EP 等变换前记录逻辑路径，
变换后按参数或模块身份重新绑定，避免保留失效引用。无法保持原意的部分专家
排除布局必须在安装新 block 前拒绝。

被排除参数的最终 FSDP 归属以运行时为准。Trainer 在梯度累积后、TP 归约和
裁剪前同步普通 Tensor 梯度，未使用参数保持 `grad=None`。

## 6. 训练、数据与模型选项

### 6.1 单步数据流

```text
DataLoader -> Batch
           -> HFTransformerModel.preprocess_inputs
           -> 统一 batch / mask，先 CP 后 TP 切分序列
           -> spmd_context
           -> model forward（TP / CP / EP 在对应边界通信）
           -> loss sum 与全局有效 token 归一化
           -> backward
           -> TP/EP/PP 感知的梯度归约、norm 和 clipping
           -> optimizer / scheduler / EMA
```

PP 下由 schedule 驱动 microbatch forward/backward。validation 使用 eval 模式和
`no_grad`，复用对应的单模型或 PP 驱动，不更新参数，也不推进训练 token 计数。
有限数据、零有效 token 以及 DP>1 时的 `validation.steps=-1` 有各自明确约束。

### 6.2 数据与 mask

非 random 语料经过 packing，attention 默认不得跨文档边界。
`attn_mask_type` 由数据侧推导：random 语料或单文档场景使用 causal，其余使用
block_causal；不提供可能与实际数据矛盾的独立开关。

重复 map 数据源在 repeat 前探测初始本地分片是否含有效样本，避免空分片造成
无限等待。该探测最多处理到首个有效样本，保存并恢复 host RNG 和 loader 游标，
但意味着该路径不再完全惰性；它不保证后续随机 epoch 或 streaming 源永不耗尽。

packed checkpoint 恢复需要独立游标快照和恢复失败处理。回滚失败时关闭 iterator，
日志和清理不得覆盖原始恢复异常。

### 6.3 Qwen3 数值与计算选项

这些选项是显式的构造期策略，不是全局 HF 类替换，也不支持对已分片或已编译
模型热切换。

- `qkv_projection=fused`：按 KV-group 顺序拼接 Q/K/V 权重并执行一次 GEMM。
  不新增参数、不改 state-dict key，但每次 forward 有拼接开销；尚不支持 TP。
  融合会改变反向归约边界，不承诺与 separate 逐位一致。
- `attention_backend=eager_flex`：为 dense Qwen3 使用独立 eager Flex 接口，
  不是融合 kernel。当前拒绝 TP、CP 和 compile，并受 dropout、bias、KV cache
  等输入约束。
- `rotary_init_device=target`：在目标训练设备重建 dense Qwen3 default RoPE
  频率。它改变频率构造的舍入轨迹；直接操作 wrapper 的调用方须显式调用
  对应初始化函数，普通 `.to()` 不会自动执行。
- `norm_backend`：根据模型支持选择 native、torch、npu、aten 或 joint。
  `aten` 和 `joint` 只承诺一阶梯度及受支持的浮点输入；joint 会增加残差副本，
  不代表性能优化。各后端可能改变舍入轨迹，不能假定训练轨迹逐位延续。
- `rotary_compute_dtype`、`rotary_cache_dtype` 和 `compute_dtype` 分别控制
  RoPE 计算、RoPE 缓存和 lm_head 计算精度，不能混淆为参数存储精度。

## 7. Checkpoint、EMA 与失败边界

### 7.1 状态格式

- DCP 是主要 checkpoint 后端；`torch_checkpointing` 为可选后端，依赖兼容的
  上游 API，不能用替身测试证明真实多 rank 恢复。
- optimizer state 使用扁平 FQN 键，例如 `state.<fqn>.exp_avg`，
  不使用 stage 内位置索引。
- 可恢复训练必须保存模型、optimizer、scheduler、dataloader、RNG、EMA 和
  trainer 进度中已启用的部分。最终 model-only 保存只是导出物，不能续训；
  需要续训时设置 `last_save_model_only=False`。
- HF safetensors 导出不等同于完整训练 checkpoint；模型、generation 和
  tokenizer 配置需要按导出能力另行提供。tied head 导出可能需要独立临时副本，
  但不改变训练模型的参数共享。

### 7.2 EMA

EMA 固定跟踪构造时的 FQN、shape、dtype、device 和张量身份；DTensor 还需匹配
mesh、placements 和本地 shape。逐步更新在匹配的本地存储上执行，不隐式重分片。

分布式 `state_dict()` / `load_state_dict()` 必须所有 rank 同序调用。
`reset_from_model()` 用于不含 collective 的本地重置。浮点或复数 buffer 可通过
`buffer_patterns` 独立跟踪；空跟踪 rank 仍参与容器的协调协议。

### 7.3 本地错误协调

`synchronize_local_errors` 用于“各 rank 先执行本地操作，再共同决定是否继续”的
边界，例如模型、dataloader、optimizer、metrics、checkpoint metadata 和本地
Stateful 序列化。其原则是：

1. 可能通信的 `state_dict()`、模型转换和保存 I/O 放在本地协调区间之外。
2. 本地阶段全部成功后才进入下一组 collective，避免 peer 在不一致状态中继续。
3. 等待后台 future 在本地进行，完成后再协调结果；不能在析构路径无条件通信。
4. 部分更新、部分加载、第三方回调副作用和已发布 checkpoint 不保证事务回滚；
   失败后从完整 checkpoint 恢复。

该机制不处理建组失败、进程退出、peer 挂起或 collective 内部故障，也不能把
整个并行变换包装后宣称具备故障恢复能力。

### 7.4 异步保存与资源所有权

异步 staging、上传、retention 和 GC 均由明确的所有者管理。异步异常传播到
staging/save 等待点；完成报告失败不代表 checkpoint 未发布，不能据此删除恢复点。
stager 或 worker 关闭失败时保留可重试所需的所有权，避免活跃线程访问已释放的
通信资源。Profiler 和内存历史也遵循“清理成功后才能放弃重试能力”的原则。

## 8. 验证策略

- `tests/unit_tests/cpu/` 验证纯逻辑、局部模块和故障分支。
- `tests/integration_tests/` 验证真实 collective、mesh、Trainer 和恢复路径；
  `run_all.py` 只负责发现和启动，不能替代对脚本参数及组合矩阵的核对。
- 硬件依赖通过 `tests.caps.require_env` 做模块级能力 skip，`-rs` 输出缺失原因。
  skip 是环境报告，不是功能通过。
- 等价性测试应覆盖输出、loss、梯度、状态键和恢复中适用的指标，并包含
  non-vacuity 检查，防止空输入或错误参考让测试恒通过。
- CPU、mock、小模型和单拓扑结果只证明其实际覆盖范围，不能外推到 NPU、
  其他并行组合或跨框架数值对齐。

常用门禁：

```bash
pytest -q --disable-warnings
ruff check llmtuner tests
ruff format --check llmtuner tests
python -m compileall -q llmtuner
git diff --check
pre-commit run --all-files
```

## 9. 已知边界

以下不是完整支持矩阵；具体裁决以 `parallel/compatibility.py` 和配置校验为准。

- chunked loss × PP、chunked loss × validation、PP × tied embeddings、
  DSA × PP 和 V 风格 PP schedule 等路径未支持或未验证。
- fused QKV × TP、eager Flex × TP/CP/compile 明确拒绝。
- DeepEP/HybridEP、部分 TorchAO dispatch、symmetric-memory fused TP 等依赖特定
  软件或设备能力，不能由导入成功推断训练可用。
- EP 只支持可等价表达的模型布局；转置专家、per-expert bias、非标准 shared
  expert 和无法表示的路由规则必须拒绝。
- EP 初始 HF 权重加载需要相应 sharded adapter；Qwen3MoE、Kimi Linear 和其他
  布局的支持条件不同。
- routed experts 的纯 TP 是本框架有意保留的扩展，不因上游弃用而自动视为缺口。
- “已接线”“单测通过”“分布式原语等价”和“真实模型端到端验收”是不同结论，
  文档不得互相替代。

## 附录：optimizer checkpoint 兼容性

旧 positional optimizer 布局按 stage 内参数位置编号，PP 下不同 stage 会产生
同名键并互相覆盖。当前布局统一使用扁平 FQN：

| 内容 | 当前键示例 |
|---|---|
| Adam 一阶矩 | `state.<fqn>.exp_avg` |
| Adam 二阶矩 | `state.<fqn>.exp_avg_sq` |
| step | `state.<fqn>.step` |
| 参数组 | `param_groups.<fqn>.lr` |

旧 positional checkpoint 不能直接加载当前构建，且不提供自动转换。加载应明确
失败，不能静默冷启动 optimizer。该变更只涉及 optimizer 子树，不改变模型权重、
trainer 计数器或 dataloader cursor 的格式。
