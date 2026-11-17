# TorchLLMTuner 设计文档

TorchLLMTuner 是使用本地显式模型实现的混合并行训练框架，复用
Hugging Face `transformers` 的配置、输出协议和部分叶子算子。
Python 包名为 `llmtuner`。本文描述稳定的模块职责与运行时契约，不记录逐次
修复或单次实验结果。

## 1. 设计原则

- **单向配置流**：`CLI / Python → LLMTunerConfig → 顶层装配 → 显式参数`。
  不递归物化配置，也不在运行时隐式改写。
- **普通函数组合并行变换**：每个 `apply_*` 显式声明依赖，底层模块不导入
  Trainer，也不读取全局配置。
- **复用 HF 模型协议**：模型无需继承框架基类；无法识别的结构必须明确报错。
- **显式能力边界**：不支持的配置、环境能力缺失和不兼容布局必须 fail-fast。
- **数值语义可验证**：结构或实现变化必须由逐位一致或容差明确的测试保护。

`torch.distributed` 提供 DeviceMesh、ProcessGroup、FSDP2 和流水线调度；
`spmd_types` 提供 SPMD 类型系统；`transformers` 提供配置、输出协议和参考算子。
这些依赖无需重新抽象。

## 2. 总体架构

```text
CLI / Python
    ↓
LLMTunerConfig
    ↓
Trainer
    ↓
显式模型构造与并行装配
    ↓
TP → EP → CP → AC → Swap（可选）→ Compile（可选）→ FSDP
    ↓
本地模型分段 + ModelPartLayout
    ↓
TrainingEngine → 模型执行与训练组件
```

主要依赖方向是 `trainer → parallel → models`。`models/common` 中的小型数学
原语可被并行层复用，但不能形成隐式装配或反向依赖 Trainer。

### 目录职责

```text
llmtuner/
├── config/          # 分组配置、聚合配置和 CLI 标量视图
├── trainer/         # CLI、训练、验证和 runtime 装配
├── models/          # 模型注册、结构契约、各模型族与共享数学构件
├── parallel/        # 五类并行、拓扑、stage 和组合裁决
├── memory/          # AC、区域重计算、激活交换
├── accelerator/     # 设备、分布式、collectives、能力探测
├── components/      # optimizer、FlexShard、EMA、checkpoint、指标
├── datasets/        # 数据组合、loader、packing、文本与多模态
└── utils/           # 日志、seed、GC 和通用工具
```

CPU 单测位于 `tests/unit_tests/cpu/`；分布式集成脚本位于
`tests/integration_tests/`。运行集成脚本前须核对各自 docstring 中的命令、
rank 和参数。

## 3. 横切契约

### 3.1 异常分类

| 异常                          | 同时继承              | 用途                                   |
| ----------------------------- | --------------------- | -------------------------------------- |
| `ConfigError`                 | `ValueError`          | 字段值或配置无效                       |
| `UnsupportedCombinationError` | `NotImplementedError` | 单项合法，但组合不受支持               |
| `EnvironmentUnsupportedError` | `NotImplementedError` | 环境缺少所需版本或能力，并给出解锁条件 |

可选第三方包缺失仍使用 `ImportError`。错误信息应说明影响范围和可行修复方式。

### 3.2 环境能力探测

`accelerator/capabilities.py` 集中探测版本相关的私有配置和可选能力：

- `has(name)` 查询并缓存结果，未知名称抛出 `KeyError`；
- `require(name, feature=...)` 在缺失时给出解锁条件；
- 可选包导入和纯设备发现不强制走注册表；
- 无降级路径的硬依赖不伪装成可选能力。

### 3.3 配置与组合裁决

1. 字段值域和仅依赖配置的组合由各配置的 `__post_init__` 校验。
2. 需要模型、拓扑或跨层信息的组合集中在 `parallel/compatibility.py`。
3. 组件生命周期约束由组件边界负责，例如 optimizer state 与 checkpoint 的
   backend/async mode。
4. 需要真实 HF MoE 布局的校验在 probe 阶段执行。

`compatibility.py` 不是通用规则引擎，也不替代字段校验和能力探测。

### 3.4 公开接口

| 层级     | 入口                                                                   | 约定                       |
| -------- | ---------------------------------------------------------------------- | -------------------------- |
| 用户接口 | `llmtuner.LLMTunerConfig`、`Trainer`、`config.*`、`python -m llmtuner` | 常规训练使用               |
| 集成接口 | `parallelize_model`、stage、组合矩阵、能力注册表                       | 扩展作者使用，稳定性较弱   |
| 内部实现 | 未列出的模块与私有符号                                                 | 不承诺兼容，测试可直接使用 |

实现代码从定义符号的叶子模块导入，避免引入不必要的引擎依赖。

## 4. 三个核心接缝

### 4.1 配置入口

`LLMTunerConfig` 组合：

```text
ModelConfig + ParallelConfig + OptimizerConfig + TrainingConfig
```

其余数据、checkpoint、scheduler、metrics 等配置归属于相应顶层配置。CLI 只
表达标量；字典、嵌套列表和 callable 通过 Python 传入。`from_groups()` 创建
新配置对象，不修改传入实例。

只有顶层装配读取聚合配置。并行层接收 `ParallelConfig` 和明确参数，底层模型
构件不接收完整运行配置。

### 4.2 模型拥有结构与执行语义

`models/registry.py` 保存配置工厂和模型类的惰性加载入口；
`models/configuration.py::resolve_model_class()` 只选择已注册的本地实现，
不自动回退到任意 HF 模型。注册不要求模型继承某个框架基类。

模型直接提供以下声明，通用并行层不猜测候选属性、类名或 forward 源码：

- `DecoderStructure` 定义注册模块路径，供 PP、AC、TP、CP 和 FSDP 绑定。
  `layers` 等便利属性不替代真实 state-dict 路径。
- 每层的 `moe_block_name` 标明稀疏槽位；`None` 明确声明 dense 层。
  `MoEBlockContract` 或模型自有转换接口定义路由与专家数学。
- `ModelExecutionCapabilities` 声明执行能力；`PipelineStageIO` 定义流水线边界。
- `tp_config` 给出模型原生 TP 计划，`prepare_inputs()` 负责输入语义。
  通用文本准备复用 `models/common/input_preparation.py`。

HF 权重键映射由模型状态格式和 checkpoint adapter 负责。模型便利属性不得
重复注册模块或参数；PP 和参数替换后，消费者重新绑定当前实例的声明。

### 4.3 并行变换与分布式上下文

`parallel/stages.py` 唯一声明：

```text
TP → EP → CP → AC → Swap（可选）→ Compile（可选）→ FSDP
```

- AC 包住已经完成 TP/EP/CP 改造的层。
- 激活交换位于 AC 外层，保留 checkpoint 重放生命周期，当前与 compile 互斥；
  full-AC 下不启用交换，D2H 与后续计算重叠，H2D 等待对应 D2H；
  `activation_swap_sync_release=True` 时改为在释放前等待 D2H。
- Compile 位于 AC 之后、FSDP 之前。
- PP 先切分 stage，再逐 part 装配，最后创建 schedule。
- FSDP 最后执行，使其 hooks 位于外层。
- 即使 dense DP 分片度为 1，只要 CP、复制轴或专家分片需要 FSDP，就不能跳过。

模型构件通过 `accelerator/spmd.py` 的线程局部上下文读取 dense/sparse mesh。
这是模型内部唯一的 ambient 分布式状态入口。

## 5. 并行与模型设计

### 5.1 Mesh 与拓扑

`parallel/topology.py` 统一构建 `ParallelTopology` 和 DeviceMesh。world mesh
保留 PP 外轴，并派生数据加载、dense storage、dense forward/backward、
sparse EP、batch 和 loss 视图。PP stage 从同一拓扑解析子视图，不能将 PP
简化为“不在 mesh 中”。

EP 复用已有 DP 设备轴。并行度乘积不能替代实际 mesh 和 rank ownership 检查。

### 5.2 TP

TP 由模型的 `tp_config` 和显式配置覆盖驱动。`ColumnParallelLinear` 和
`RowParallelLinear` 提供 sequence-parallel 的 collective + GEMM 边界。
解析阶段绑定具体模块和参数，执行阶段消费计划并拒绝失效绑定；无法安全处理
的 bias 或布局必须拒绝。

支持的 MoE-under-TP 路径沿专家 hidden 维切分 routed experts，router 保持
复制，并通过块边界 all-gather / reduce-scatter 对齐序列分片。非标准 shared
expert × TP × EP 组合会被拒绝。

### 5.3 CP

CP 的公共入口只接受已接入的模型自有 KCP 或 Compressor 后端；通用
standard-attention CP 内核已移除。拓扑和输入分片代码保留 KV、Ulysses 等轴的
描述能力，不代表所有后端组合均可执行。

模型必须提供 `set_cp_mesh()` 及对应后端的验证、前向接口。CP 预检先拒绝
不支持的后端，再绑定具体 attention 目标；PP 切分后重新检查本地目标归属。
输入分片同步处理 tokens、labels、positions 和 padding 元数据，不能从已切分
positions 中猜测全局文档边界。

### 5.4 EP

EP 按模型声明的稀疏槽位和数学契约，将支持的专家块替换为原生 router、
grouped experts 和 dispatcher。EP=1 使用本地 dispatch；EP>1 使用本地专家切片和 EP 通信组。
AllToAll 是默认后端，TorchAO padded dispatcher 是可选后端，MC2/Domino 有
专用实现。DeepEP/HybridEP 仍明确拒绝。

布局检查不能只检查首个专家。非 SiLU 激活或带 bias 的专家投影必须拒绝。
辅助负载均衡 loss 通过梯度注入参与训练，不改变主 loss 数值；专家与 dense
参数的梯度范数分别归约，再使用同一全局裁剪系数。

EP checkpoint 使用逻辑完整专家张量：保存前汇集切片，加载时按当前 EP 布局
重新分片，从而支持改变 EP 度数。保存前校验状态键、shape、dtype、layout 和
张量类型。

### 5.5 PP

PP 先按层归属切分 stage，再为每个 model part 执行标准装配。1F1B 和
Interleaved1F1B 使用 PyTorch schedule，`scale_grads=False`，loss 由 Trainer
统一归一化。

首 stage 接收输入，末 stage 接收 labels 并产生 loss，其他 stage 返回零 loss
哨兵。多 model-part optimizer 由同一个 `OptimizersContainer` 管理，optimizer
state 使用参数 FQN，避免 stage 内位置索引冲突。

microbatch 要求非标量张量按第 0 维对齐。多 microbatch 暂不支持一维 packed
corpus，也不支持按媒体 span 或坐标轴重新分配非行布局张量。

### 5.6 FSDP2 / HSDP

FSDP 最后包装模型，dense 与专家参数使用各自的 mesh 和所有权规则。
`fsdp_reduce_dtype` 独立控制梯度通信精度，默认 FP32；选择 BF16 或 FP64 时
必须单独验证。

`fsdp_apply_modules` 与 `fsdp_ignored_modules` 使用模块路径模式；单元选项的
`ignored_params`、`hook_module` 可含对象引用。装配在 TP/EP 前记录归属和
替换映射，变换后重新绑定，避免重用已失效的路径或对象。

CPU offload 与 pin-memory、forward/backward prefetch 分开配置。optimizer
state 的 NPU 常驻与 CPU 常驻是不同策略，不能共用含义不清的开关。

## 6. 训练、数据与模型选项

### 6.1 单步数据流

```text
DataLoader → Batch
  → TrainingEngine.preprocess → 模型 prepare_inputs
  → 统一 batch / mask，先 CP 后 TP 切分
  → spmd_context
  → model forward（TP / CP / EP 在对应边界通信）
  → loss sum 与全局有效 token 归一化
  → backward
  → TP/EP/PP 感知的梯度归约、norm 和 clipping
  → optimizer / scheduler / EMA
```

PP 由 schedule 驱动 microbatch。validation 使用 eval 和 `no_grad`，复用对应的
单模型或 PP 驱动，不更新参数，也不推进训练 token 计数。

运行时职责集中在以下位置，避免新增同义入口：

| 模块                           | 职责                                                        |
| ------------------------------ | ----------------------------------------------------------- |
| `trainer/trainer.py`           | 恢复、训练循环、验证时机、保存与资源关闭                    |
| `trainer/runtime/assembly.py`  | 构造模型和训练服务，连接显式依赖                            |
| `trainer/runtime/engine.py`    | 前后向、累积、梯度同步、裁剪及 optimizer/scheduler/EMA 更新 |
| `trainer/runtime/batch.py`     | 统一训练/验证 loader 构造及 batch 适配                      |
| `trainer/runtime/results.py`   | 进度计数与步骤结果类型                                      |
| `trainer/runtime/gradients.py` | 梯度校验、归约及参考范数布局                                |

训练与验证均使用 `build_rank_dataloader()`，由 `repeat` 和可选 dataset 区分。
`TrainingBuilders` 只注入组件工厂，组件不读取或改写完整 Trainer。
裁剪策略在引擎装配时绑定；Trainer 通过 `finish_step()` 提交更新，诊断 hook
仍观察同步后、裁剪前的梯度。迁移与对照方式见
[训练执行边界迁移记录](refactoring_runtime_execution.md)。

### 6.2 数据与 mask

非 random 语料经过 packing，attention 默认不得跨文档边界。`attn_mask_type`
由数据侧推导：random 或单文档使用 causal，其余使用 block_causal，不提供可能
与数据矛盾的独立开关。

重复 map 数据源在 repeat 前探测初始本地分片是否含有效样本，避免空分片造成
无限等待。该探测最多处理到首个有效样本，保存并恢复 host RNG 和 loader 游标，
但不再完全惰性，也不保证后续随机 epoch 或 streaming 源永不耗尽。

packed checkpoint 恢复需要独立游标快照和恢复失败处理。回滚失败时关闭
iterator，清理不得覆盖原始恢复异常。

### 6.3 Qwen3 数值与计算选项

这些选项是构造期策略，不支持对已分片或已编译模型热切换：

- `qkv_projection=fused`：拼接 Q/K/V 权重并执行一次 GEMM，不改变参数或
  state-dict key，但有拼接开销，尚不支持 TP。
- `attention_backend=eager_flex`：dense Qwen3 的独立 eager Flex 接口，拒绝
  TP、CP 和 compile，并受 dropout、bias、KV cache 等约束。
- `rotary_init_device=target`：在目标设备重建 dense Qwen3 default RoPE 频率，
  改变频率构造的舍入轨迹。
- `norm_backend`：按模型支持选择 native、torch、npu、aten 或 joint。joint
  增加残差副本，不代表性能优化。安装替代后端前检查实例自定义 norm forward，
  不覆盖未知实现；重新应用策略时只撤销框架拥有的适配，保留用户 hooks。
- `rotary_compute_dtype`、`rotary_cache_dtype`、`compute_dtype` 分别控制
  RoPE 计算、缓存和 lm_head 精度，不能混同于参数存储精度。

### 6.4 算力与 SDC 可观测性

`utils.monitoring.get_peak_flops` 将 `Ascend910_9392` 和显式 `910C` 映射为每
NPU 376 TFLOPS dense BF16 峰值。MFU 使用模型 FLOPs/token 与每设备 TPS 计算；
未知设备或模型几何时不报告，不用零值冒充测量。benchmark 在 Trainer 关闭前
保存几何与峰值元数据，避免生命周期结束后的空引用。

`compile.enable_sdc=True` 在 `_compiled_call_impl` 图外检查非有限输出，避免
Dynamo 追踪 `.item()`。该诊断只拦截 NaN/Inf，不是整图逐位校验；开启后会同步，
默认关闭。EP8 动态 expert shape 另有独立 Dynamo guard 限制。

## 7. Optimizer 与计算布局

`OptimizersContainer` 可按 model part 和 optimizer name 组合参数组，所有状态
使用扁平 FQN。

Adam/AdamW 的 `optimizer.state_residency="npu_virtual"` 将 moments 放在
Host-backed NPU 存储。原生 DCP 支持同步、threaded async 和 pinned-memory async；
非 DCP、AMSGrad、fused/capturable 和 FSDP CPU-offload 等组合明确拒绝。加载先
重新绑定 virtual destinations，再拷贝恢复数据。失败加载不保证数值回滚，必须
完成有效加载后才能继续训练。

DistMuon/FlexShard 将参数存储布局与 optimizer 计算布局解耦：

- `compute_sharding_by_fqn` 描述每个参数的 `ComputeLayout`；
- `bucket_configs` 描述打包重分片与通信顺序；
- 布局和 bucket 在构造时固定，恢复时必须使用匹配配置；
- `ns_device="accelerator"` 在 NPU 执行 Newton-Schulz，Host 存储与 NPU
  计算之间的重分片通信缓冲区按 HCCL 设备分配；
- `npu_virtual` 仅限 Adam/AdamW，DistMuon 会明确拒绝；
- dense 矩阵和 MoE 批量矩阵需分别验证更新、状态键和恢复。

checkpoint 的本地状态准备通过 `init_optim_state` 分派：Adam/AdamW 保留
零学习率创建状态及复位的路径；DistMuon 使用 `initialize_state()` 仅分配
本地 momentum，不执行 step 或重分片通信。已有 momentum 保持原对象，
新状态在所有本地分配成功后发布，调用方再协调错误并进入布局 collective。
checkpoint 初始化不改变 DistMuon 的首步梯度校验状态。

DistMuon 的参数组选项校验由 `validate_param_groups()` 统一定义，构造、
容器恢复预检和独立 optimizer 的 load 前置/后置 hook 复用同一规则。
直接调用 `DistMuon.load_state_dict()` 时，前置 hook 先拒绝非法组选项，
避免后置校验失败时学习率或状态已被替换；该本地 hook 不承担跨 rank 协调。
容器在 `Optimizer checkpoint validation` 的 rank 错误协调边界内检查所有
待恢复参数组，再允许任何 optimizer 应用状态，避免非法选项先修改本地状态、
再让其他 rank 进入计划 collective。这不承诺后续分配或拷贝失败时数值回滚。

布局配置类型定义在 `config/optimizer_layout.py`；`flex_shard` 包和旧
`optimizer_reshard.py` 路径保留类型导出。运行时实现位于
`components/optimizer/flex_shard/`，按职责拆分：

| 模块                            | 职责                                     |
| ------------------------------- | ---------------------------------------- |
| `dist_muon.py`                  | optimizer 接口、构造与更新编排           |
| `muon_math.py`                  | momentum、Newton-Schulz 与参数更新数学   |
| `muon_layout.py`                | 存储到计算布局的解析与约束               |
| `muon_redistribution.py`        | 根据 Muon 矩阵边界生成重分布计划         |
| `optimizer_reshard_schedule.py` | tensor-region 路由与 collective 调度规划 |
| `optimizer_reshard_runtime.py`  | 打包缓冲区、异步通信与计算执行           |

普通模块名不意味着内部计划类型成为公开 API；辅助符号的私有边界仍以其职责为准。

## 8. Checkpoint、EMA 与失败边界

### 8.1 状态格式

DCP 是主要后端，`torch_checkpointing` 为可选后端。可恢复训练必须保存模型、
optimizer、scheduler、dataloader、RNG、EMA 和 trainer 进度中已启用的部分。
最终 model-only 或 HF 导出物只是导出结果，不能精确续训。

### 8.2 EMA

EMA 固定跟踪构造时的 FQN、shape、dtype、device 和张量身份；DTensor 还需匹配
mesh、placements 和本地 shape。逐步更新在匹配的本地存储上执行，不隐式重分片。
分布式 `state_dict()` / `load_state_dict()` 必须所有 rank 同序调用。

EMA 与训练优化器共用 `optimizer/checkpoint_state.py::OptimizerCheckpointState`
的参数归属校验和 FQN 状态序列化。EMA 不继承训练优化器的更新接口；Trainer
显式调用 `step(current_step)`。

### 8.3 本地错误协调

`synchronize_local_errors` 用于“各 rank 先执行本地操作，再共同决定是否继续”：

1. 可能通信的 `state_dict()`、模型转换和保存 I/O 放在协调区间之外。
2. 本地阶段全部成功后才进入下一组 collective。
3. 后台 future 在本地等待，完成后再协调结果，不能在析构路径无条件通信。
4. 部分更新、部分加载、第三方回调和已发布 checkpoint 不保证事务回滚，
   失败后从完整 checkpoint 恢复。

该机制不能处理建组失败、进程退出、peer 挂起或 collective 内部故障。

### 8.4 异步保存与资源所有权

异步 staging、上传、retention 和 GC 均由明确所有者管理。异步异常传播到
staging/save 等待点；完成报告失败不代表 checkpoint 未发布，不能据此删除
恢复点。

`checkpointer/storage/staging.py` 负责本地 staging 结果的跨 rank 协调。成功后先发布
训练侧完成信号，再触发上传回调。manager 继续持有底层 stager 和通信组，直到
上传结束。PROCESS staging 在上一轮完成后释放实例，下一轮重新创建；构造
失败时保留尚未使用的实例。

`Trainer.close()` 按依赖顺序释放资源。检查点 drain 失败时保留 checkpointer
和它仍可能读取的 dataloader、优化器等依赖；独立的日志、验证和 GC 资源仍尝试
关闭。成功释放的资源不重复关闭，失败资源由原所有者保留以便重试。

## 9. 验证策略

- CPU 单测验证纯逻辑、局部模块和故障分支。
- 集成脚本验证真实 collective、mesh、Trainer 和恢复路径。
- 硬件依赖通过 `tests.caps.require_env` 做模块级 skip，`-rs` 查看原因。
- 等价性测试覆盖输出、loss、梯度、状态键和恢复，并包含 non-vacuity 检查。
- CPU、mock、小模型和单拓扑结果不能外推到其他设备或组合。
- skip、启动成功、单步可运行不能作为大规模性能或稳定性结论。


常用门禁：

```bash
pytest -q --disable-warnings
ruff check llmtuner tests
ruff format --check llmtuner tests
python -m compileall -q llmtuner
git diff --check
pre-commit run --all-files
```

## 10. 模块边界与简洁性

以公共入口、依赖方向、重复实现和失败语义评估复杂度，不用历史文件行数推断
性能或正确性。训练编排保留在 Trainer，数学和输入语义归模型，参数变换归并行层，
状态序列化归组件。合并共享逻辑时，需同时迁移调用方和验证脚本；删除内部模块
后不保留无人使用的兼容转发层。

## 11. 已知边界

- chunked loss × PP、chunked loss × validation、PP × tied embeddings、
  DSA × PP 和 V 风格 schedule 未支持或未验证。
- fused QKV × TP、eager Flex × TP/CP/compile 明确拒绝。
- DeepEP/HybridEP、部分 TorchAO dispatch 和 symmetric-memory fused TP 依赖
  特定软件或设备能力。
- EP 只支持可等价表达的布局；转置专家、per-expert bias、非标准 shared expert
  和无法表示的路由必须拒绝。
- “已接线”“单测通过”“分布式原语等价”和“真实模型端到端验收”是不同结论，
  文档不得互相替代。

## 附录：optimizer checkpoint 键

旧 positional optimizer 布局按 stage 内参数位置编号，PP 下会产生同名键。
当前统一使用扁平 FQN：

| 内容        | 当前键示例               |
| ----------- | ------------------------ |
| Adam 一阶矩 | `state.<fqn>.exp_avg`    |
| Adam 二阶矩 | `state.<fqn>.exp_avg_sq` |
| step        | `state.<fqn>.step`       |
| 参数组      | `param_groups.<fqn>.lr`  |

旧 positional checkpoint 不提供自动转换，加载应明确失败，不能静默冷启动
optimizer。该布局不改变模型权重、trainer 计数器或 dataloader cursor。
