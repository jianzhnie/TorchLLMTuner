# TorchLLMTuner 代码优化建议：审查结论与执行计划

本文记录针对当前 `llmtuner/` 实现的代码审查结果和后续优化顺序。静态代码
只能确认潜在开销与职责边界；性能收益须由目标设备上的基准和 trace 证明。
重构先保留现有数值、拓扑、checkpoint 和恢复语义，再逐项验证。

## 1. 审查结论

| 主题 | 当前实现 | 结论 |
| --- | --- | --- |
| TP/EP checkpoint 布局 | `components/checkpointer/model_state.py` 与 `components/optimizer/optimizer.py` 的编排分支有重复；实际 TP/EP gather、shard 和 restore 已复用 `parallel/tensor_parallel/checkpointing.py` 与 `parallel/expert_parallel/checkpointing.py`。模型和优化器的目标状态构造、错误协调及加载生命周期并不完全相同。 | 原建议把问题概括为“两套布局转换”并要求新增通用 manager，表述过度。先保留共享转换；只有能消除真实分支漂移且覆盖完整恢复矩阵时，才抽取窄接口。不要让统一包装掩盖不同生命周期。 |
| 长函数 | `swap_hf_moe_blocks`、`apply_fsdp_to_decoder`、`clip_grad_norm_`、`Trainer.train_step`、`apply_fsdp_by_pattern` 均存在，且各自包含多个职责阶段。 | 按状态变更边界逐个拆分，先固定行为和失败路径。函数长度本身不足以证明拆分收益；尤其训练步骤与分布式 collective 不应机械拆 helper。 |
| 并行装配 | `parallelize_hf_transformers` 单独接收编译、AC、global batch 和 chunked-loss 选项；内部 `apply_stages` 同时用于 dense 与 PP 本地模型分段。 | `ParallelExecutionOptions` 可减少签名参数，但需要改调用方和测试，尚无证据说明接口复杂度已造成错误。所谓 dense/PP stage runner 分叉目前不存在；不另建重复 stage context。 |
| 兼容接口 | `llmtuner/parallel/compatibility.py` 已集中处理多项上游和版本差异，部分设备 API 适配也留在所属模块。 | 不再平行新增通用 `accelerator/compat/`。仅当多个业务模块重复处理同一个第三方私有 API 时，才迁入有明确所有者的兼容函数。 |
| 错误语义 | 项目已有 `ConfigError`、`UnsupportedCombinationError` 和 `EnvironmentUnsupportedError`，并在配置与装配边界使用。 | 方向正确，但这是维护要求，不是当前缺少的一项新架构工作。新改动沿用现有错误类型，逐处补齐可执行诊断即可。 |
| 测试体系 | 已有 CPU 契约测试、拓扑测试、checkpoint 等价集成脚本及 NPU 专项脚本。 | 优先补足本次改动对应的回归用例；只有多份脚本确实重复同一生命周期时才抽公共 harness。性能测试与正确性测试分开。 |

## 2. 高风险结构优化

### 2.1 Checkpoint 布局转换

模型和优化器保存路径都调用共享的 TP/EP layout 函数，joint TP+EP 分片也已有
专用实现。恢复仍需分别建立模型参数和 optimizer state 的本地目标；EP-only、
TP chunk、joint chunk、旧 full-tensor checkpoint、空分片及错误协调的先后顺序
不能任意合并。

本轮已按该矩阵核对并运行现有 CPU/Gloo 集成脚本，并新增 TP2×DP2 保存到
TP2×DP1 恢复用例。结果如下：

| 覆盖项 | 测试与结果 |
| --- | --- |
| TP 完整张量、chunked save/restore、legacy full tensor、optimizer 冷恢复后下一步 | `tp_dp_checkpoint_layout.py` 通过；覆盖 packed/unpacked、三个 DP 分片维度、不均匀分片及空 DP shard。 |
| EP 完整张量与 chunked save/restore、optimizer/EMA 恢复 | `ep_dp_checkpoint_layout.py` 通过；覆盖三个 DP 分片维度。 |
| joint TP+EP 和 EP-only 的模型、optimizer、EMA 冷恢复及下一步 | `tp_ep_checkpoint_layout.py` 4-rank 通过。 |
| 空 ownership | `empty_expert_checkpoint_layout.py` 及 TP empty-DP 参数通过。 |
| 不同并行度恢复 | 新增 `tp_dp_checkpoint_reshard.py` 通过：TP2×DP2 checkpoint 在独立 TP2×DP1 作业中恢复，optimizer 冷启动后继续更新与连续参考完全相同。TP 或 EP 度数变化尚未覆盖。 |
| rank 本地验证失败的协调 | TP/EP 专项 2-rank Gloo 测试通过。joint TP+EP 的 4-rank 测试让每个 rank 轮流单独提供错误 chunk，所有 owner 都收到失败，目标状态保持不变。 |

实际命令均在 PyTorch 2.12 CPU 环境执行，超时退出码为 0：

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_dp_checkpoint_layout.py
LLMTUNER_TEST_DP_SHARD_DIM=1 PYTHONPATH=. torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_dp_checkpoint_layout.py
LLMTUNER_TEST_DP_SHARD_DIM=2 PYTHONPATH=. torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_dp_checkpoint_layout.py
LLMTUNER_TEST_UNEVEN_DP=1 PYTHONPATH=. torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_dp_checkpoint_layout.py
LLMTUNER_TEST_EMPTY_DP=1 PYTHONPATH=. torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_dp_checkpoint_layout.py
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 tests/integration_tests/ep_dp_checkpoint_layout.py
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 tests/integration_tests/empty_expert_checkpoint_layout.py
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_ep_checkpoint_layout.py
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 tests/integration_tests/tp_ep_dp_checkpoint_layout.py
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/integration_tests/tp_checkpoint_coordination.py
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/integration_tests/ep_checkpoint_coordination.py
PYTHONPATH=. python tests/integration_tests/tp_dp_checkpoint_reshard.py
```

审查 joint TP+EP 的单 rank 失败用例时发现，错误报告只经过 TP group；尚未收到
错误的 rank 若已完成 EP group 报告，可能成功返回并继续使用未恢复状态。现在
`parallel/tensor_parallel/checkpointing.py` 将 TP 与 EP ownership groups 纳入有序
错误传播，并用有界多轮报告把错误传到所有相连 owner。4-rank joint 测试验证每个 rank
轮流失败时无目标张量被部分修改。

这些结果覆盖了当前 CPU/Gloo 下的布局与恢复边界，但不代表 NPU/collective
验收，也未验证 TP/EP 度数变化。现有共享 layout 转换保持不变；在确认更广泛的
拓扑恢复需求前，不引入包揽保存、恢复和 EMA 的通用
`CheckpointLayoutManager`。任何后续抽取仍须保持连续训练与独立恢复在预定精度下
等价，并证明失败路径不会死锁。

### 2.2 长函数拆分

候选入口及建议边界：

- `parallel/expert_parallel/model_replacement.py::swap_hf_moe_blocks`：架构检查、
  转换计划、参数映射、替换和后置校验；保持不同 HF 模型族的拒绝条件。
- `parallel/fully_shard_parallel/fully_shard_parallel.py::apply_fsdp_to_decoder`：
  boundary 规划、ownership 分组、policy 生成、包装与后置 hook；用 FSDP 契约及
  集成测试固定调用顺序。
- `accelerator/collectives.py::clip_grad_norm_`：局部范数、mesh 归约、非有限值
  处理和缩放可以有内部边界；所有 rank 必须保持一致的 collective 顺序。
- `trainer/trainer.py::Trainer.train_step`：可按 batch、前后向、梯度更新和指标
  阶段评估；先证明异常处理、gradient accumulation、validation 和 checkpoint
  协调语义不会改变。
- `parallel/fully_shard_parallel/plan.py::apply_fsdp_by_pattern`：区分 pattern
  解析、引用绑定和 FSDP 应用；先检查现有纯逻辑 helper 是否已覆盖这些职责。

每次只拆一个入口，新增测试验证输入输出、异常和状态；不为缩短行数添加一两行
的转发 helper。

### 2.3 并行装配选项

`parallel/parallelize.py::parallelize_hf_transformers` 目前把运行时选项作为关键字
参数传入。若引入不可变的执行选项对象，应同时更新唯一生产装配调用、测试和公开
接口文档，并证明对象能减少真实的透传差异。现有 `ParallelConfig` 继续只描述拓扑
和并行策略。dense 与 PP 已共用 `apply_stages`，此处没有重复实现需要合并。

## 3. 性能候选项与约束

任何候选项都先记录模型、序列长度、batch、拓扑、精度、预热步数和计时方法；至少
报告正确性、端到端 step 时间分布、显存峰值和目标 kernel/通信 trace。无 trace 时
只记录为待验证假设，不宣称提速。

### 3.1 Qwen3 fused QKV

位置：`models/hf/qkv_projection.py::fused_qkv_projection`。

确认每次 fused forward 都 reshape 并拼接 Q/K/V 权重与 bias，再调用一次 linear；
该路径是显式选择的 Qwen3 选项，不代表所有模型都承担此开销。激活重计算会重复
执行拼接。先比较 separate 与 fused 的目标设备端到端结果；训练态不能跨 backward
复用带旧 autograd graph 的 packed tensor。永久 fused 参数需要同时解决 optimizer
更新、参数身份、TP 变换和 state-dict 双向适配，不能仅替换 forward。

### 3.2 Eager Flex mask

位置：`models/hf/attention_backend.py::eager_flex_attention_adapter`。

确认每次调用都从 HF mask 构造 `BlockMask`；无 mask 时还会生成 causal dense mask，
非 bool mask 会做设备侧合法性检查。输入 mask 可能包含 batch/document 信息，因此
shape 和 device 不能单独作为缓存键。只有 mask 内容有稳定身份/版本且生命周期明确时
才缓存；优先评估从 position/document metadata 直接构造 block mask 是否能避免 dense
中间量，并检查 eager Flex 实际是否为目标生产后端。

### 3.3 CP sliding-window mask

位置：`parallel/context_parallel/sdpa_attention.py::_apply_sliding_window`。

causal mask 已按 kernel 实例缓存；启用 `sliding_window` 时，当前实现仍会每次生成
坐标和二维 bool 窗口 mask，再与 causal 或 packed document mask 相交。缓存二维 mask
可能减少重复构造，但每层额外保留 O(T²) 显存。先测 NPU 时间和峰值显存；无 packed
mask 的情况可评估缓存最终 causal-window mask，动态 document mask 则须继续逐批求交。

### 3.4 CPU-offload 梯度范数

位置：`accelerator/collectives.py::_offloaded_gradient_norm` 与 `clip_grad_norm_`。

确认每个 offloaded DTensor 的本地范数按其 Shard placement 归约；裁剪阶段按 mesh
分组，但每组此前都会把同一个全局 clip coefficient 搬到 CPU。已将 offloaded
gradient 汇总后只计算/搬运一次系数，再分别缩放各本地梯度。此改动减少重复标量
搬运，不改变范数归约通信；还需在真实 CPU-offload NPU 运行中测量收益。

正确性已由 `tests/integration_tests/fsdp_cpu_offload_clipping.py` 的 4-rank NPU
用例验证：1、2、∞ 范数与 foreach 开关的六种组合均通过。日志：
`outputs/code_optimization_cpu_offload_clipping_20261026.log`。该用例没有计时，
因此性能收益仍待基准确认。

若后续考虑合并多个梯度的 norm collective，必须按相同 mesh、axis、dtype 和 norm
定义组合 p 次幂，并以 TP/EP/PP、空分片、非有限值和 CPU-offload 集成结果验证。

### 3.5 EP dispatch 主机同步

位置：`parallel/expert_parallel/torchao_dispatch.py::_permute`。

`sum().item()` 将 padded token 数变成 Python 整数，用于裁掉未使用容量；它会同步设备，
但当前输出 shape 依赖该动态值。先用 trace 量化延迟与通信重叠影响，再评估固定容量、
算子侧裁剪或异步 shape 管理；不可在未验证 grouped GEMM padding 契约前删掉裁剪。

### 3.6 重复 map 数据空样本预检

位置：`datasets/composition.py::apply_iteration_policy`。

仅 `repeat=True` 的 map dataset 会在 DP 切片后扫描当前 rank 的整个 shard，以在进入
无限 repeat 前拒绝空 shard；不是每次迭代扫描，也不是无条件扫描全局数据集。优化前
先量启动时间并确认 Grain 的 lazy transform / RNG 语义。若复用处理计数，计数必须与
shuffle、DP 切分和 processor 的确定性语义一致。

## 4. 第三方兼容与文档

- 继续将同一依赖 API 的版本差异放在已有 `parallel/compatibility.py` 或对应能力模块；
  不建立无人维护的通用兼容层。
- 保留解释 ownership、collective 顺序、设备/dtype/layout 约束和非直观行为的代码
  注释。历史故障和上游差异移至文档时，代码应保留足以解释当前约束的链接或摘要。
- checkpoint 布局图只有在统一定义 global coordinates、rank ownership、packed half
  和空分片后再添加，不能用概念图替代代码契约。
- 更新现有文档索引前，先确认目标文件存在且未包含未提交的用户修改。

## 5. 执行顺序

1. 为一个有明确热路径依据的小改动补回归测试；本轮已合并 CPU-offload clip
   coefficient 的单次主机搬运。全量 CPU 测试 4344 passed、13 skipped；4-rank
   NPU 六种范数组合、全仓 Ruff、格式和 compileall 检查通过。真实 NPU 性能数据
   尚未采集。
2. 对 QKV、Flex mask、CP sliding-window、EP dispatch 和 dataset preflight 分别采集
   基线。一次只优化有可重复收益且没有显存/正确性回退的候选项。
3. Checkpoint 布局矩阵已完成 CPU/Gloo 审查，并修复 joint TP/EP 单 rank 失败未传到
   所有相连 ownership group 的问题。暂时没有支持新增布局抽象的证据；若继续调整
   checkpoint 编排，先补 TP/EP 度数变化和目标设备恢复覆盖，再评估窄 helper，不预设
   新 manager 是目标。
4. 按职责边界逐个拆分长函数、评估执行选项对象；每项单独验证并记录实际收益。
5. 只对真实重复的第三方 API 适配和测试生命周期抽公共接口。

每项执行记录写明代码版本、配置、命令、退出码、结果范围和未验证条件；纯 CPU
测试不得代表 NPU/collective 验收。性能改善只有在目标设备复测后才标为通过。
本轮 NPU 日志为 `outputs/code_optimization_cpu_offload_clipping_20261026.log`。
`pre-commit` 未安装在当前主机或容器环境，本轮以其 Ruff、格式、空白和编译检查
分别验证。
