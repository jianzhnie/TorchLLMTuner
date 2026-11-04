# Ascend 910C：实现差异与能力吸收路线

## 1. 定位与证据边界

TorchLLMTuner 应保留统一 Trainer、HF 模型适配、并行装配和 checkpoint 控制面，
优先吸收设备内核、显存策略和运行可靠性设计。判断是否引入某项能力的依据是
当前瓶颈、适配成本与可复现收益，而非参考项目是否有同名开关。

本文依据 2026-11-04 本地源码审查；各工作树可能包含未提交修改：

| 项目 | 审查时 HEAD | 在比较中的角色 |
| --- | --- | --- |
| TorchLLMTuner | `ec7f78d` | 当前实现与接入位置 |
| TorchTitan | `0f9fd2bd6` | 通用训练、DTensor 布局、区域重计算与恢复契约 |
| TorchTitan-NPU | `0a10bfc` | TorchTitan 的 NPU 适配、融合、offload 与可靠性扩展 |
| FSDPTurbo | `1dae0be` | FSDP 训练中的交换、模块适配与 MoE 通信优化 |

源码存在只证明已实现；测试脚本存在不代表已验收。上游文档描述的能力，也不
自动适用于本项目当前软件栈、目标模型或并行组合。本文不是全仓正确性通过证明。

目标环境为 `hyper-parallel-env`、8×910C、CANN 9.1.0、torch/torch-npu 2.12，
模型为 `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B`。
此前设备实测标识为 8 个 `Ascend910_9392`，仅凭该字符串不能确认具体 910C SKU；
版本、拓扑和设备映射须在每次运行前重新记录。升级编译软件栈应使用独立环境，
避免同时改变数值基线和执行后端。

## 2. 总体实现差异

| 维度 | TorchLLMTuner 当前实现 | 参考实现的区别 | 应吸收的内容 |
| --- | --- | --- | --- |
| 模型与训练控制 | `models/hf/` 适配 HF，`trainer/` 编排，`parallel/` 统一装配 | TorchTitan 以模型/recipe 与通用训练组件组织；TorchTitan-NPU 在其上使用 extension、override 和 patch；FSDPTurbo 使用模块计划及模型适配 | 明确模型能力和转换契约，减少按模型名称散落的分支；不引入第二套 Trainer |
| 并行与重计算 | TP→EP→CP→AC→swap→compile→FSDP；PP 先切分再复用阶段 | TorchTitan 的模型并行计划和重计算策略与原生模型结构绑定；NPU override 和 FSDPTurbo 的模块替换不能直接套到 HF 参数树 | 复用算法与设备原语，重新绑定 FQN、参数身份、共享权重及 collective 顺序 |
| 设备执行 | native/eager 与显式选择的 NPU kernel；编译前检查后端能力 | TorchTitan-NPU 提供 AscendC Codegen、融合组件与独立 GraphTrainer 路径；FSDPTurbo 提供 NPU 算子和 dispatcher | 小边界的设备 backend、可观测 fallback、明确 shape/dtype/layout 限制 |
| 显存管理 | AC、activation swap、FSDP CPU offload、virtual Adam moments、DistMuon 布局 | 参考项目对交换粒度、资源池、预取和 optimizer 计算布局有不同实现 | 按参数、梯度、moments、激活、workspace 分项选策略，避免一个 offload 开关包揽全部含义 |
| 保存与恢复 | `checkpointer/core/`、`backends/`、`storage/`、`hf/` 分工；TP/EP layout 共享 | TorchTitan 提供通用 manager/backend；NPU 有 swap-storage 保存适配；FSDPTurbo 有其训练状态与恢复流程 | 保留本项目 FQN、ownership、失败协调和原子发布；设备存储策略必须加入恢复矩阵 |

源码入口：[并行装配](../llmtuner/parallel/parallelize.py)、
[阶段顺序](../llmtuner/parallel/stages.py)、
[TorchTitan HF 并行实验](../../torchtitan/torchtitan/experiments/transformers_modeling_backend/parallelize.py)、
[TorchTitan-NPU 架构](../../torchtitan-npu/docs/architecture.md)、
[FSDPTurbo 项目指南](../../FSDPTurbo/docs/zh/project_guide.md)。

## 3. 编译与融合：先补执行证据

### 3.1 AutoFuse 与 GraphTrainer 是不同路径

本项目的 [model_compile.py](../llmtuner/parallel/model_compile.py) 支持模型/块编译，
并在 NPU Inductor 路径检查 `TORCHINDUCTOR_NPU_BACKEND=ascendc` 和 Codegen 能力。
`aot_eager` 用于 Dynamo/AOTAutograd 正反向兼容验证，不证明生成了 AscendC 融合
kernel。仅设置环境变量、导入成功或训练完成，也不能证明融合生效。

[TorchTitan-NPU compile 指南](../../torchtitan-npu/docs/feature_guides/torch_compile.md)
复用 PyTorch Inductor 和 torch-npu 的 AscendC Codegen；GraphTrainer 则涉及整步图、
通信调度和重计算 pass。将后者整体搬入当前 Trainer 会改变状态与调度边界，
不应作为普通 compile backend 切换。

优先改进：

1. 为固定软件栈建立 eager→`aot_eager`→Inductor 的分级验收，版本要求由实际
   Codegen/CANN ABI 决定，不把某个上游主线版本写成本项目永远的最低要求。
2. 记录 graph break、重编译、lowering、生成 kernel、fallback 和缓存命中。
3. 对 attention/MoE 等热点逐块编译，明确 AC 与 FSDP 的顺序及动态 token 约束。
4. 编译开销、缓存复用和稳态收益分开计时；有正确性回退时不以吞吐抵消。

### 3.2 融合算子：已有路径优先完善

本项目已有 RMSNorm、RoPE、SwiGLU、grouped GEMM、token dispatch 等接入点：
[HF 算子适配](../llmtuner/models/hf/)、
[grouped-MM 对齐算子](../llmtuner/accelerator/grouped_mm.py)、
[专家实现](../llmtuner/models/common/moe/experts.py)。
TorchTitan-NPU 使用组件 override/custom op，FSDPTurbo 提供 NPU 算子与 dispatcher。
数学语义相同不代表权重打包、score 乘法位置或反向归约相同。

吸收重点为统一接入契约，而不是再增加一层通用包装：

- 给出 dtype、shape、stride、对齐、空输入、并行 ownership 与 fallback 条件；
- custom op 补齐 schema、Fake/Meta、Autograd、alias/mutation，验证动态空专家；
- 检查输出、输入梯度、参数梯度、router score 梯度及更新后的权重；
- profiler 证明目标 kernel 确实执行，避免测试实际只跑 fallback。

参考：[融合算子指南](../../torchtitan-npu/docs/feature_guides/fused_ops.md)。

## 4. 显存优化：区分驻留、交换与计算布局

### 4.1 Virtual Optimizer 与 AdamWSwap

| 项目 | 机制 | 关键差异与风险 |
| --- | --- | --- |
| 本项目 `npu_virtual` | step pre-hook 为 Adam/AdamW moments 分配 Host-backed、NPU 可访问 storage | 处理 DTensor 本地 shard、空 shard、FP32 step counter、冷/重复加载和 destination 重绑；不是通用 optimizer offload |
| TorchTitan-NPU Virtual Optimizer | override optimizer 配置，使用 `empty_with_swapped_memory` | 同样只接管 Adam moments；其指南的专用 writer 重点覆盖同步 native DCP，不能据此推出异步组合全部可用 |
| FSDPTurbo AdamWSwap | CPU pinned state 按显存余量分批 H2D 更新，再 D2H | 有 batch 容量、参数预取和 clip 后梯度保留策略；需要管理额外 PCIe 流量及峰值 workspace |

本项目 [state_residency.py](../llmtuner/components/optimizer/state_residency.py)
当前配置允许 DCP 的 disabled、async、async-with-pinned-memory；拒绝非 DCP、
AMSGrad、capturable、differentiable、fused 等不支持路径。配置允许不代表每种
拓扑都已验收。DCP pin-memory 请求在当前 PyTorch/NPU 环境可能被忽略，
PROCESS 上传通过不能作为 pinned staging 已生效的证明。

优化顺序：先量 moments 是否为长期 HBM 主因；优先完善现有 virtual 策略的
checkpoint 与 Host 资源上限，再按需求考虑分批 AdamWSwap。两种策略应分别配置，
不能将 Host-backed 地址访问与显式分批拷贝写成同一种机制。

参考：[Virtual Optimizer](../../torchtitan-npu/docs/feature_guides/virtual_optimizer.md)、
[AdamWSwap](../../FSDPTurbo/docs/zh/features/swap_optimizer.md)。

### 4.2 Activation swap 与区域重计算

本项目 [activation_swap.py](../llmtuner/memory/activation_swap.py) 在 decoder
saved-tensor hook 上做 D2H/H2D；保护输入、参数、buffer 和 storage 别名，使用
stream 等待与引用释放，支持重复 unpack 和丢弃图清理。当前 compile 与 swap
互斥；full AC 下 swap 跳过，不能把二者同时开启解释为双重显存收益。

FSDPTurbo 使用 `TensorSwapManager` 和模块匹配计划；TorchTitan 的
[REMAT](../../torchtitan/torchtitan/distributed/REMAT.md) 则侧重区域重计算。
交换减少驻留但消耗 Host/PCIe；重计算减少保存但增加计算，二者应按层成本选择。

可吸收方向：

1. 按层/模块选择交换对象，设置最小 tensor 字节数、Host 预算和在途副本上限，
   优先避免小 tensor 产生大量 pinned 分配和 stream 开销。
2. 若 profiling 证明分配成本显著，再引入有 owner、wait、close 的 buffer pool；
   不复用仍被异步拷贝或 retained graph 引用的内存。
3. 用每层重算成本与搬运字节选择 AC 或 swap；Selective/region AC 组合先解决
   saved-tensor hook 重放契约，不能直接沿用其他框架的“可组合”说明。
4. 保留别名、重复 backward、异常清理、梯度累积和 checkpoint 续训测试。

参考：[FSDPTurbo 激活交换](../../FSDPTurbo/docs/zh/features/swap_activation.md)。

### 4.3 FSDP CPU offload

本项目使用 PyTorch FSDP `CPUOffloadPolicy`、显式前后向预取窗口，Host 梯度范数
与裁剪保留 CPU 路径。它与 virtual moments 是不同策略：前者还改变参数和梯度
的驻留位置，影响范围更大。

TorchTitan-NPU 的 [CPU-offload 扩展](../../torchtitan-npu/docs/feature_guides/cpu_offload.md)
和 FSDPTurbo 的 clip/optimizer 协同值得吸收：让梯度搬运、裁剪、optimizer 更新
共用显式驻留状态，减少重复 H2D/D2H。不能仅把预取窗口调大就视为优化。

验收必须同时记录 Host RSS/pinned 内存、PCIe 字节与耗时、FSDP 通信 workspace，
并检查 TP/EP/PP、gradient accumulation、非有限梯度及恢复后第一步。

## 5. MoE：从同名 dispatcher 转向实际执行路径

本项目已有 AllToAll、MC2、Domino；MC2 针对空分组做协调回退，受 EP 规模、
权重布局和设备算子契约约束。入口为
[mc2_dispatch.py](../llmtuner/parallel/expert_parallel/mc2_dispatch.py)、
[mc2_ops.py](../llmtuner/parallel/expert_parallel/mc2_ops.py)、
[domino_dispatch.py](../llmtuner/parallel/expert_parallel/domino_dispatch.py)。

FSDPTurbo 的 MC2/Domino 将部分通信与专家计算融合；TorchTitan-NPU 还有图级
通信 overlap。当前 eager dispatcher 与整图调度在启动顺序、动态 shape、反向
依赖和通信资源上不同，不能直接迁移性能数字。

优先改进：

- 按真实专家数、top-k、token 分布和 EP 度数记录 fused/fallback 比例与原因；
- 分解 permute、计数同步、AllToAll、GMM、combine 和 backward，观察最慢 rank；
- 比较不均衡路由、零接收、空专家及 AC 重放，检查 TP replicated 梯度是否重复归约；
- 只有 trace 确认存在 overlap 空间后，才调整预取、容量或流水；
- router bias/统计、专家 ownership 和变 EP 恢复纳入同一正确性矩阵。

参考：[FSDPTurbo MC2](../../FSDPTurbo/fsdp_turbo/distributed/expert_parallel/dispatcher_mc2.py)、
[Domino](../../FSDPTurbo/fsdp_turbo/distributed/expert_parallel/domino_dispatcher.py)、
[TorchTitan-NPU EP overlap](../../torchtitan-npu/docs/feature_guides/graph_trainer_ep_overlap.md)。

## 6. DistMuon/FlexShard：已引入，重点转向峰值与恢复

本项目已有 [flex_shard/](../llmtuner/components/optimizer/flex_shard/)，并非缺少一个
Muon 类。其核心与 [TorchTitan FlexShard](../../torchtitan/torchtitan/distributed/flex_shard/README.md)
一致：参数持久存储布局与临时 optimizer 计算布局分离，通过 FQN、compute layout
和 bucket 组织重分片、Newton-Schulz 计算及结果回传。

本地 HF 参数树、EP ownership、参数组及 DCP 恢复与 TorchTitan recipe 的绑定方式
不同。`npu_virtual` 的 Adam moments 策略不能直接套到 DistMuon；也不能把
`ns_device="accelerator"` 的 CPU-offload 运行解释为 optimizer 全程在 CPU。

后续重点：

1. 逐阶段量重分片 buffer、dtype 转换、Newton-Schulz workspace 和释放时点；
   区分长期保留与瞬时峰值，再调整 bucket 或流水槽位。
2. 显式约定 TP/EP 下矩阵边界、分片轴和空 ownership，不能按本地 shape 猜测。
3. 加载时重建并校验 compute layout；测试 moments、scheduler、EMA 及下一步更新。
4. AdamW 与 Muon 更新规则不同，应分别建立参考和收敛标准，不要求二者 loss
   轨迹逐位一致，也不能用短程有限值检查替代收敛验收。

## 7. 可靠性：非有限检查不等于 SDC checksum

| 能力 | 本项目当前边界 | TorchTitan-NPU 对应设计 | 改进方向 |
| --- | --- | --- | --- |
| Eager SDC | 启动环境选择原生 torch-npu 检查，受运行时版本约束 | 原生 gradient/HCCL 检查及 step 状态 | 记录是否实际安装 hook、何时提交检测状态，失败 step 不提交 |
| Compile 检测 | `wrap_sdc_forward` 在编译调用外递归检查输出非有限值 | post-grad pass 在受支持 matmul 后插入 checksum custom op，并与原生 gate 配合 | 将现有功能明确称为非有限输出检查；完整 checksum 另设能力与测试 |
| Checkpoint | 原子发布、异步异常传播、rank-local 失败协调、DCP/HF layout | 通用 checkpoint 加 NPU 存储适配；extension 还提供 hash manifest 等选项 | 区分文件完整性、数值有效性和训练状态一致性，分别验收 |

关键差异：**有限但错误的结果不会被 `isfinite` 检出；文件 hash 正确也不证明
训练数值正确。** NaN 注入通过不能作为完整 SDC 覆盖证明。

完整 checksum 若引入，应有版本门禁、Fake/Autograd 契约、编译缓存键、故障注入
和开销测量，避免在 Trainer 中增加无条件逐 tensor 主机同步。
Anticipatory Routing 还涉及专家 ID 重放、数据游标、RNG、AC 和恢复，保持独立立项。

参考：[本地编译检查](../llmtuner/parallel/model_compile.py)、
[上游 checksum pass](../../torchtitan-npu/torchtitan_npu/extensions/compile/sdc_checksum.py)、
[SDC 指南](../../torchtitan-npu/docs/feature_guides/sdc.md)、
[Anticipatory Routing](../../torchtitan-npu/docs/feature_guides/anticipatory_routing.md)。

## 8. 历史实验如何影响优化选择

以下只摘录能定位的已有报告，未在本文重写期间重跑，不作为当前 HEAD 的验收。

| 历史实验 | 报告中的结果 | 可以得到的结论 |
| --- | --- | --- |
| Qwen3-30B，8 rank，100 step，warmup 10 | device 平均 7.9128s、峰值 allocated 28.44 GiB；virtual 平均 9.1782s、14.22 GiB | 该配方下 virtual 以 step 时间换 HBM；不是吞吐优化，也不是与外部框架对齐证明 |
| 30B EP8 MC2，12 step，warmup 3 | 平均 6.9228s，p50 6.9660s，p90 7.6505s | 必须结合 fused/fallback 计数解释；单凭 dispatcher 配置名不能评价融合 kernel |

原始报告：
[device](../outputs/qwen3-30b-eager-100step-20261102/benchmark.json)、
[virtual](../outputs/qwen3-30b-virtual-100step-20261102/benchmark.json)、
[MC2](../outputs/qwen3-30b-ep8-mc2-steady2048-v4-20261103/benchmark.json)。
这些报告记录 `git_dirty=true`，应补源码差异归档才能精确重现；`outputs/` 是本地
生成物，链接可能随清理失效，不应提交生成日志。前两份报告的基线 commit 为
`168ecff3`，MC2 为 `98fb94ce`，均不是本文审查 HEAD。

报告中的计时是同步后的最慢 rank 完整 `train_step`，不含外层 checkpoint、验证、
GC、启动等耗时；不能称为包含全部运维成本的长训端到端吞吐。loss/grad_norm
轨迹相同也不能单独证明所有模型权重、moments 与 RNG 精确一致。

本轮重写未重新审计旧文中的其余跨节点、OOM、恢复和微算子性能数字，因此不再
将它们写成无条件“已通过”总表；后续应归入带配置、命令、退出码和原始证据的报告。

## 9. 实施优先级与验收

| 优先级 | 差距/瓶颈 | 建议动作与接入位置 | 完成条件 |
| --- | --- | --- | --- |
| P0 | 各能力的支持与证据混写 | recipe 固定软件栈/拓扑，报告关联 commit、diff、配置与退出码 | 四个目标布局分别有当前版本数值与恢复报告，失败/skip 不计通过 |
| P0 | 编译可用与真实融合未分级 | `parallel/model_compile.py`、`accelerator/capabilities.py` 加强运行可观测性 | eager/aot/Inductor 分别验收；实际 kernel、重编译、数值及资源证据完整 |
| P0 | 恢复矩阵与 Host-backed 状态复杂 | 复用现有 checkpointer/optimizer 接口，补旧格式、空 shard、异步失败及变并行度 | 独立进程恢复下一步一致，故障时不发生不匹配 collective；明确 pin-memory 是否生效 |
| P1 | swap/CPU-offload 数据搬运与 Host 峰值 | `memory/`、FSDP 与 optimizer 驻留边界按 trace 调优 | Host/HBM 有界，同配置数值通过；重复稳态测量证明收益或明确代价 |
| P1 | MC2 名称与实际融合路径脱节 | EP dispatcher 原因计数、真实路由和通信/计算 trace | fused 与 fallback 均非空验证；真实 30B 有独立性能结论 |
| P1 | DistMuon 瞬时峰值与布局成本 | `flex_shard/` 分段计量和 bucket/生命周期优化 | 峰值来源明确，更新/恢复正确，长程收敛与性能分别验收 |
| P1 | 非有限检查覆盖弱于 checksum | 保留现有检查，按需求加入独立 checksum backend | 有限错误/非有限错误分别注入，记录覆盖范围与运行开销 |
| P2 | 更大范围的执行体系变化 | GraphTrainer、Anticipatory Routing、新低精度能力独立设计 | 先证明目标硬件/模型需要且支持，再考虑迁移 |

四个目标布局为 `dp_shard=8`、`dp_shard=4/TP=2`、`TP=2/EP=4`、
`dp_shard=8/EP=8`。EP 与 dense DP/TP 的拓扑关系应以本项目 mesh 构造为准，
不能把全部配置值直接相乘推算 world size。

不优先直接移植：Ascend 950 专用低精度路径、CUDA Graph、H100/NVLink Async TP
性能结论、第二套配置/Trainer，以及缺少目标模型需求的广泛功能扩展。
FSDPTurbo Chunk Batch 也不是 gradient accumulation、PP microbatch 或 chunked loss
的替代；其当前文档明确不支持与 CP 组合。

## 10. 统一验证与复现入口

| 验证层级 | 必须比较 |
| --- | --- |
| 算子/模块 | 输出、输入/参数/score 梯度、空输入、非对称数据、实际 kernel 与 fallback |
| 整网与并行组合 | 同初始权重/数据/RNG，loss、logits、grad_norm、参数更新，最慢 rank 与 ownership |
| 长程与恢复 | 连续 100 step 对独立进程 50→100；模型、moments、scheduler、RNG、数据游标、实际启用的 EMA |
| 多节点与故障 | 默认任务队列、rank-local 失败、staging/writer 异常、资源释放与健康重试 |
| 性能 | 相同配置、预热后重复测量、p50/p90、峰值 allocated/reserved HBM、Host RSS/pinned、通信与拷贝 trace |

优先复用现有入口，不另建 benchmark 栈：

- [30B recipe](../examples/train_qwen3_30b_npu.py) 与
  [benchmark](../examples/npu_training_benchmark.py)：真实配置、耗时与资源；
- [EP profile](../tests/integration_tests/npu_ep_profile.py)、
  [MC2 等价](../tests/integration_tests/ep_mc2_equivalence.py)：通信与实际分支；
- [activation swap](../tests/integration_tests/npu_activation_swap.py)、
  [DistMuon](../tests/integration_tests/npu_dist_muon_30b.py)：状态和资源生命周期；
- [TP/EP checkpoint](../tests/integration_tests/tp_ep_checkpoint_layout.py)、
  [异步生命周期](../tests/integration_tests/async_checkpoint_lifecycle_equivalence.py)：恢复与释放；
- [compile SDC](../tests/integration_tests/npu_compile_sdc.py)：现有非有限检测边界。

910C 算力利用率按约定的每设备 dense BF16 峰值 **376 TFLOPS** 报告：

`MFU(%) = 100 × 训练 FLOPs/token × 全局 tokens/s ÷ (参与设备数 × 376e12)`。

MoE 应使用实际激活专家及 attention 等计算的 FLOPs 估算，不能直接代入 30B
总参数量。明确 padded/有效 tokens、前反向 FLOPs 定义、设备计数单位及 AC 是否
计入；按设备数归一化后不要再除一次 TP/CP。该值是模型估算 MFU，不能等同于
`npu-smi` 的 AICore 活跃率。若 dtype/硬件峰值不同，应使用相应峰值并另列口径。

结构迁移在相同执行策略下要求恢复与更新语义不变；更换 kernel、归约或 dtype 时
预先声明容差。Profiler 采集与无 profiler 性能对照分开运行；小模型、mock、skip、
启动成功和短程有限值均不能替代目标 30B、分布式或长程验收。
