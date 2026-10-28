# FSDPTurbo Distributed 与 TorchLLMTuner Parallel 实现审查

## Custom FSDP 多运行时隔离补充（2026-10-15）

`fine_grained_runtime_isolation.py` 在双 rank CPU/Gloo 验证两个 live 模型：
不同 seed、reshard 与参数组布局；交错 forward，交替 backward 顺序；一方
forward 失败并 reset 时，另一方未完成的 forward/backward 保持正确。
普通与非重入重计算两种配置均通过独立 dense 输出、梯度和三次 momentum SGD
比较；state_ctx/通信 context 身份不共享，收尾状态清空。
该脚本现还为两个模型分别构造生产 EMA，fixed decay=0.5，三次 SGD 后
完整 DTensor 平均权重均与独立 dense lerp 参考一致，普通/重计算配置退出 0。
该结果不等于多完整 Trainer 的全局 RNG/编译开关隔离，不覆盖线程或设备 streams，
也不涵盖动态 placement/mesh 变更。

## Custom FSDP 混合精度边界补充（2026-10-15）

`fine_grained_mixed_precision.py` 在真实双 rank CPU/Gloo 通过 32 配置：
参数 bf16/fp16、归约 fp32/bf16、输出 fp32/bf16、reshard 与输入梯度开/关。
每配置比较三次 SGD 的输出、输入/参数梯度和更新权重，并检查 FP32 master。
参考为原生 FSDP，双方采用相同 Block forward/cast 边界；默认数值容差保持。

hook 迁移也会迁移 MixedPrecisionPolicy 的输入/输出转换边界。原生逐 Linear
cast 与 custom Block cast 会使中间 tanh 使用不同 dtype，实测梯度不一致；
不能仅凭相同 policy 字段宣称不同 hook 边界逐位等价。该 CPU 证据不覆盖
AC、compile、DCP 或目标设备重叠执行，原失败和定位过程见逐文件台账。

### 混合精度 AC/compile 的当前边界

相同 32 配置下，full AC、aot_eager、full AC+aot_eager、Inductor 各自与原生
FSDP eager 参考的三次更新比较通过。full AC+Inductor 在首配置梯度比较失败，
最大绝对误差 0.00439453125，保持原断言，尚未关闭。

`fine_grained_mixed_precision.py --local-probe` 无需分布式即可重现 BF16
双 Linear+tanh 的第一层梯度差异：不使用 AC 时 Inductor 梯度逐位一致，使用
生产 wrap_full 后误差出现；输出及末层梯度一致。差异已定位到 Inductor 的低精度转换省略：编译梯度与显式 FP32 tanh 中间值
参考逐位一致。`--local-probe` 对该独立参考执行精确断言，同时报告与 eager 差异。
设置 `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` 后，同一真实两 rank
full AC+Inductor 的 32 配置全部通过原 eager 比较；未放宽容差。
默认融合的原失败保留为数值策略差异，不将其描述为 FSDP 通信缺陷。
该设置须在导入/编译前生效，未测性能，也不保证所有算子或设备逐位等价。

## 当前生产归一化与编译状态（2026-10-15）

以下为当前工作树的补充结论；后文诊断批次中的“未接入生产”和“待验证”
保留为历史状态。验证范围以下表为准，不将局部通过推广到所有 compile/FSDP 组合。

| 路径 | 当前实现 | 当前证据与限制 |
| --- | --- | --- |
| 默认 native Qwen3/Qwen3 MoE RMSNorm | 构造期 residual norm 输入 view，保持参数身份与键 | AOT/eager 恢复及已记录的 native/custom FSDP、EP、TP 组合通过；Inductor 融合仍可能改变归约舍入，跨模式逐位恢复不保证 |
| 显式 `model.norm_backend=aten` | 生产前后向 custom op 边界，配置经 factory 到实际 Qwen norm | CPU 一阶、fp16/bf16/fp32；36 项独立 HF 输出/梯度逐位回归及 opcheck；不支持目标设备或高阶梯度 |
| aten + custom FSDP + full AC + dynamic EMA | PP1、q_proj 细粒度分片，生产 norm 绑定断言 | 两 rank CPU/Gloo Inductor 连续训练、同模式恢复、Inductor→eager 严格完整状态恢复通过；保留原断言 |
| 生产 aten DTensor | leading shard 权重梯度 Partial；hidden shard 重分布到复制 | 独立 HF 参考；2-rank eager/Inductor、2×2 mesh Inductor 前后向通过；float32 默认容差，断言实际编译；未覆盖完整混合精度并行训练 |
| 同步编译装配失败 | 恢复 scalar capture、regional 标志、逐层原 compiled callable、async TP pass | 故障注入回归通过；不回滚延迟 tracing 缓存，不注销 PyTorch 对称内存注册；成功模型共存的全局配置隔离仍开放 |

长期复核入口：`tests/unit_tests/cpu/models/test_normalization.py`、
`tests/unit_tests/cpu/models/test_hf_wrapper.py`、
`tests/unit_tests/cpu/parallel/test_model_compile.py`、
`tests/integration_tests/aten_norm_dtensor_equivalence.py` 和
`tests/integration_tests/pp_checkpoint_equivalence.py`。
命令、具体配置和结果见 `docs/file_review.md`。aten 保留归约边界会损失融合性能，
因此保持显式 opt-in；诊断基准的耗时不能当作生产端到端性能指标。


## 1. 审查目标与基线

本文逐项对比：

- `/Users/robin/work_dir/FSDPTurbo/fsdp_turbo/distributed`
- `/Users/robin/work_dir/TorchLLMTuner/llmtuner/parallel`

审查目标包括：

1. 判断 TorchLLMTuner 的并行实现是否在自身模型、Trainer 和 mesh 契约下正确；
2. 检查 FSDP、TP、PP、CP、EP 及其组合是否完整接线；
3. 找出 FSDPTurbo 中可以安全迁移的能力；
4. 区分通用能力、设备专用优化和模型专用 patch，避免直接复制不兼容实现；
5. 记录本轮已经完成的迁移和后续工作。

源码基线：

| 项目 | 版本 | 审查提交 |
|---|---:|---|
| FSDPTurbo | `0.1.0` | `8dfec48` |
| TorchLLMTuner | `0.1.0` | `e6ed6d5`，加本文记录的未提交迁移 |

## 2. 审查结论

TorchLLMTuner 的核心并行路径在当前约束内是完整且内部一致的：

- dense mesh 满足 `dp_replicate × dp_shard × cp × tp × pp = WORLD_SIZE`；
- sparse mesh 将 `dp_shard × cp × tp` 重新分解为 `efsdp × ep`；
- PP 先切分模型，其余变换对每个 stage chunk 按统一 stage 表应用；
- TP、EP、CP、activation checkpoint、compile、FSDP 的相对顺序由单一表声明；
- Trainer 对 CP/TP token shard、PP loss、全局 valid-token denominator 和 checkpoint 状态有统一处理；
- 不安全的模型布局和组合由配置、mesh、matrix 和 probe 分层拒绝。

本轮没有发现需要推翻现有 mesh 代数或装配顺序的错误。发现并处理的工程缺口是：

1. **FSDP CPU offload 未暴露**：底层函数已有参数，但 `apply_fsdp` 固定传入 `False`；本轮已迁移 FSDPTurbo/TorchTitan 的配置能力。
2. **FSDP 兼容入口残留**：`fully_shard_parallel/apply.py` 只做再导出，与“同名文件为核心入口”的目录约定冲突；本轮已删除。
3. **旧路径元数据**：组合矩阵和测试曾指向已删除的 TP/PP/EP/FSDP 文件；前一轮已修复并由真实文件检查覆盖。

当前已接线 KCP、Domino 双切片语义及 PyTorch 2.14 custom FSDP。custom 新增状态子类、ancestor hook relocation 和多个独立 buffer/event 上下文，共享通信 stream 保持 collective 顺序；CPU/Gloo 多参数组、重计算及异常 reset 数值验证通过。自定义参数组已补齐跨 hook 的 event 等待和提前释放；完整迁移仍需目标设备及剩余生命周期验证。Domino 已加入 CUDA dispatch/combine 的通信 stream/event 调度，目标设备数值与实际重叠尚未验证；NPU 和完整反向重叠仍待迁移；KCP 需要目标设备验证。MC2 已接入可选 torch_npu 融合 autograd 与 MoE adapter，缺包/算子时明确报错，真实 NPU 数值及 TP 适配仍待完成；低精度通信栈仍未迁移。

## 3. 模块映射

| 领域 | FSDPTurbo | TorchLLMTuner | 结论 |
|---|---|---|---|
| 进程组与 mesh | `parallel_state.py` | `topology.py`、`accelerator/distributed.py` | 两者模型不同，不能直接替换 |
| 通信原语 | `dist_ops.py` | `accelerator/collectives.py`、TP/CP kernel 内部原语 | TorchLLMTuner 分层更细 |
| FSDP | `fully_shard_parallel/` | `fully_shard_parallel/fully_shard_parallel.py` | 核心能力相当，选层策略不同 |
| Custom FSDP | `fine_grained_fully_shard/` | `fully_shard_parallel/fine_grained.py`、`fine_grained_state.py` | 自定义 hook/context 生命周期，复用 2.14 参数组；完整上游优化及设备验收仍待完成 |
| TP | `tensor_parallel/tensor_parallel.py` | `tensor_parallel/tensor_parallel.py`、`linear.py` | TorchLLMTuner 对 HF/序列并行做了更深适配 |
| CP | `context_parallel/` | `context_parallel/` | 策略有交集，模型接入方式不同 |
| EP | `expert_parallel/` | `expert_parallel/`、`models/common/moe/` | 专家表示和 dispatcher 不同 |
| PP | 无 | `pipeline_parallel/` | TorchLLMTuner 独有 |
| 编排 | `FSDPTurbo.__init__` | `parallelize.py`、`stages.py` | TorchLLMTuner 使用显式 stage contract |
| 组合裁决 | 分布在 config/apply | `compatibility.py`、config、probe | TorchLLMTuner 更集中 |

## 4. Mesh 与并行状态

### 4.1 FSDPTurbo

`ParallelState` 分别创建三套 mesh：

```text
(dp, fsdp, tp)
(edp, efsdp, ep)
(data, kvallgather, ulysses-or-kcp, ctp)
```

特点：

- 每套 mesh 的首轴由剩余 world size 推导；
- CP 可由 KV AllGather 与 Ulysses 两个轴组合，KCP 是另一互斥路径；
- dense 参数使用 `dp_fsdp`，专家使用 `edp_efsdp`；
- 全局 singleton 提供动态生成的 `get_*_group` 等方法。

该设计适合将多种 CP/NPU 优化作为正交 patch 注入，但多个 mesh 对同一 rank 空间采用不同分解，组合正确性依赖调用者理解每套 group 的语义。Singleton 也使同一进程中用不同配置重建训练任务变得困难。

### 4.2 TorchLLMTuner

`ParallelTopology` 从一个 world mesh 派生所有 view：

```text
dense:  pp × dp_replicate × dp_shard × cp × tp
sparse: pp × dp_replicate × efsdp × ep
efsdp = dp_shard × cp × tp / ep
```

另外缓存 dataloader、loss、storage 和 forward/backward 所需子网格。所有 view 来自同一个 world mesh，避免重复构建后缓存引用不同 process group。

审查结果：

- `WORLD_SIZE`、自动推导 `dp_shard` 和 EP 整除约束正确；
- PP 轴保留在 dense/sparse 世界分解中，每个 stage 的 EP group 不跨 PP stage；
- loss mesh 包含 DP、CP 和 TP，符合 TorchLLMTuner 的端到端序列分片语义；
- size-1 `dp_shard` 被保留，使纯 DDP 仍可通过 FSDP2 的 HSDP mesh 完成梯度归约；
- 不需要迁移 FSDPTurbo 的全局 singleton 或动态方法生成。

## 5. FSDP 与 HSDP

### 5.1 共同能力

两个项目都使用 composable FSDP2，并具备：

- MixedPrecisionPolicy；
- FSDP/HSDP mesh；
- decoder layer 与根模块的嵌套包装；
- CPU offload 底层支持；
- forward/backward prefetch；
- EP 专家参数的独立 mesh/placement；
- 非 NCCL/HCCL 场景下的梯度归约适配。

### 5.2 主要差异

| 维度 | FSDPTurbo | TorchLLMTuner |
|---|---|---|
| 选层 | 用户提供模块通配符和逐模块 plan | 默认 HF decoder 分层；可用 `fsdp_apply_modules` 选择 pattern 单元及逐单元覆盖 |
| 忽略参数 | `ignored_modules`/`ignored_params` | `fsdp_ignored_modules` 排除完整子树；融合 EP 转换后重绑原模块路径与参数引用，EP 参数使用 placement callback；CPU 参数归属验证不代表自动同步排除参数梯度 |
| 实现后端 | native 或 custom fine-grained FSDP | native 或适配 PyTorch 2.14 的 custom 参数组/state/hook；设备验证仍有缺口 |
| dtype | 配置显式提供 param/reduce/output dtype | 参数 dtype 来自模型构建，reduce 固定 FP32；混合参数 dtype 明确拒绝 |
| prefetch | 可配置前后预取层数 | 支持显式窗口并按共享 state 去重；decoder EP 在 count=0 时默认相邻单元 |
| 专家 placement | EFSDP 模块列表与 `Shard(0/1)` | 根据 `efsdp × ep` 和全局专家数选择 placement |
| gradient scaling | 专家模块设置 divide factor | 关闭 FSDP 自动平均，由 Trainer 的全局 token normalization 定义缩放 |

TorchLLMTuner 的实现与其 Trainer 契约一致。直接迁移 FSDPTurbo 的 `gradient_divide_factor` 会重复缩放梯度；pattern 适配通过共享参数所有权合并 FSDP state、冲突检查和 EP placement 保留 HF wrapper 的参数关系；这不代表任意模型的执行顺序均已验证。

### 5.3 本轮迁移：CPU offload

新增配置：

```python
ParallelConfig(
    fsdp_config=FSDPConfig(
        fsdp_cpu_offload=True,
        fsdp_cpu_offload_pin_memory=True,
    ),
)
```

配置现在贯通：

```text
ParallelConfig
→ apply_fsdp
→ apply_fsdp_to_decoder
→ CPUOffloadPolicy(pin_memory=...)
```

默认值保持关闭，避免改变既有训练行为。测试覆盖两个字段从用户配置传递到 FSDP policy builder。

### 5.4 适配与验证缺口

- **custom fine-grained FSDP**：已有生产 param group/state/hook 路径及多 rank CPU/Gloo 等价证据；NPU/CUDA 生命周期、通信重叠及性能仍需目标设备验证。
- **任意窗口 prefetch**：已有 `fsdp_forward_prefetch_count` / `fsdp_backward_prefetch_count` 和共享 state 去重。pattern 路径按模块注册顺序配置，不能推断任意 forward 的实际执行顺序；目标设备峰值内存、重叠与收益待验证。空 backward targets 保留 PyTorch 默认预取，不代表关闭预取。
- **output dtype 和逐参数 dtype override**：TorchLLMTuner 当前拒绝 mixed parameter dtype，避免 FSDP 静默转换；需要明确模型需求后再扩展。

## 6. Tensor Parallel

### 6.1 FSDPTurbo

FSDPTurbo 将用户 pattern 转成 PyTorch `ColwiseParallel`、`RowwiseParallel` 和 `SequenceParallel` plan，再调用 `parallelize_module`。实现简洁，适合标准 Linear 布局；正确性依赖配置 pattern 完整覆盖模型边界。

### 6.2 TorchLLMTuner

TorchLLMTuner 的 TP 是端到端序列并行实现：

- Trainer 先把 token/label 切成 `T / tp`；
- Column projection 内部 all-gather sequence；
- Row projection 输出 reduce-scatter 回 sequence shard；
- attention q/k/v 在 attention 边界 gather，避免 HF reshape 使用错误的局部 sequence shape；
- embedding、LM head 和 vocab-parallel loss 共用相同词表 shard 边界；
- MoE expert feature 轴、shared expert 和 TP/EP 边界有单独处理；
- head 数、权重 shape、bias 和未识别 MoE layout 会在修改权重前拒绝。

审查结果：

- TP 输入分片和 fused collective 边界相互匹配；
- TP loss 归约包含 sequence shard，不会把本地 token loss 当作全局 loss；
- TP 与 CP 的顺序是先 CP shard，再在 CP shard 内做 TP sequence shard；
- `max_seq_len % (cp × tp) == 0` 的根配置约束正确；
- bias projection 当前明确拒绝，属于覆盖范围限制，不是静默错误。
- dense TP 不支持共享 projection 权重：同一模块注册到多个路径、或不同
  Linear 共用同一 weight 时，包装前按全部参数路径拒绝。共享 projection 应
  保持 replicated；已有拒绝测试确认模块和参数身份不变。TP selector 默认
  使用去重模块路径，不能认为 FSDP 的 alias pattern 支持自动适用于 TP。
- pattern FSDP 已按共享参数所有者合并状态，支持忽略/选中模块别名并去重；
  这是 FSDP 路径的实现，不改变上述 TP 限制。

FSDPTurbo 的 pattern plan 不应直接迁移，因为 TorchLLMTuner 需要额外处理 attention reshape、vocab loss 和 MoE boundary。后续可以借鉴的是受约束的用户 plan override，并在 realization 前复用现有 shape/probe 校验。

## 7. Context Parallel

### 7.1 策略对比

| 策略 | FSDPTurbo | TorchLLMTuner |
|---|---|---|
| KV AllGather | 支持，function patch | 支持，`CPFlexKernel` |
| Ulysses | 支持，function patch | 支持，sequence/head all-to-all |
| KCP | 支持 KDA/KCP 专用状态和 patch | 已接线，模型专用路径，待目标硬件数值验证 |
| CP 组合轴 | KV AllGather 可与 Ulysses 组合 | 已实现双轴，`cp = cp_kv × cp_ulysses`；CPU 前向及通信梯度已验证 |
| Padding | helper 可 pad 到 group size | 配置要求序列能被 `cp × tp` 整除 |
| Load balance | 无同名统一配置 | `headtail`；`ptrr` 明确拒绝 |
| Packed/varlen | 模型 patch 路径 | full BlockMask + Q shard，Ulysses 保持 full mask |

### 7.2 正确性审查

TorchLLMTuner 的 CP 分为两半：

1. Trainer/model wrapper 对 input ids、labels、positions、padding mask 和 BlockMask 使用同一 mesh 与 load balancer 分片；
2. attention kernel 对局部 Q 与重分布后的 K/V 执行注意力。

关键约束已显式处理：

- KV AllGather 只对 Q 轴切 BlockMask，KV 轴保持全长；
- headtail 要求 `seq_len % (2 × cp) == 0`；
- Ulysses 要求 global heads 能被 `tp × cp` 整除；
- Ulysses 禁止 load balancer，避免 token 顺序被重排后训练另一个模型；
- CPU/不具备 FlexAttention 的环境在 apply 阶段失败；
- padding mask 按与 token 相同的切分传播。

这些约束比自动 padding 更保守，但避免 padding token、causal mask、packed document 边界和 loss denominator 之间出现隐式差异。

### 7.3 验证范围与适配差异

- KCP 依赖 KDA 状态与模型 patch，不是通用 attention strategy；已接线，仍需目标硬件验证；
- KV AllGather × Ulysses 使用独立子网，总 CP 度数取两轴乘积，代入既有世界规模公式；
- `cp_dual_axis_equivalence.py` 在 PyTorch 2.14.0、四 rank CPU/Gloo 上通过：普通与 packed 前向最大误差均为 0。前向使用 eager flex；通信梯度检查以 dense SDPA 替代 attention，对生产通信原语的梯度采用 `1e-9` 容差，不能据此宣称 flex backward 或设备融合内核已验证；
- FSDPTurbo 的 Transformers loss patch 不适用于 TorchLLMTuner，因为后者在 Trainer 中统一计算 summed loss 和全局 token denominator。

## 8. Expert Parallel

### 8.1 FSDPTurbo

FSDPTurbo 通过用户指定模块 pattern 应用 EP，并支持：

- eager dispatcher；
- fused dispatcher；
- MC2；
- Domino；
- EFSDP；
- fixed router；
- 量化 dispatcher 与低精度 AllGather 路径。

部分能力依赖 NPU grouped matmul、HCCL 或模型实现按照 FSDPTurbo 约定暴露专家结构。

### 8.2 TorchLLMTuner

TorchLLMTuner 先 probe HF MoE block，再转换为公共 MoE 实现：

- 识别 router、fused/separate expert weights、top-k、route scale 和 shared expert；
- 将专家权重逐项搬入 `GroupedExperts`；
- EP rank 只持有本地专家；
- All-to-All dispatcher 负责 token dispatch/combine；
- 可选 TorchAO padded dispatcher；
- Domino 双切片 dispatch/compute/combine，允许空 token rank 参与两次交换；
- debug-only fixed router，按 FSDPTurbo 的 `(token + slot) % E` 选择、固定 `1/K` 权重；
  CPU/Gloo EP2 的 alltoall/Domino × full AC × 不均匀/空source/稀疏/全padding
  16 场景已用原 HF 权重的独立逐专家算式核对输出、全部梯度和一步 SGD；
  公开配置、aux loss 与单次统计均覆盖，见 `docs/file_review.md` 的
  FIXED-ROUTER-UPSTREAM-01（该记录修正了 FIXED-ROUTER-EP-01 的旧语义）。
  该证据不覆盖容量丢弃、设备内核或其他并行轴。
- aux-loss 在训练 forward 注入梯度；optimizer pre-hook 汇总指标，并执行
  sign-based 或 quantile bias 更新；
- checkpoint adapter gather/load 逻辑专家张量，使恢复不绑定物理 EP shard；
- EP>1 时 TP 将专家处理交给 EP，专家按数量维分片并消费本地序列；
  EP=1 的特征维 TP 则使用序列 gather/reduce-scatter 边界，两者不能混为同一布局。

审查结果：

- EP group 来自 sparse mesh，并在 PP 下限定在本 stage；
- swap 在 FSDP 之前执行，FSDP 看到的是最终专家参数；
- HF initial checkpoint 与 swapped expert layout 的不安全组合被明确拒绝；
- GPT-OSS bias、未知 group scoring、shared expert gate 等未覆盖布局会拒绝转换；
- `deepep`/`hybridep` 是登记缺口，配置不会静默回退。

### 8.3 未迁移项

- MC2 已接入 Ascend fused dispatch/GEMM/combine 及反向；需要 torch_npu 三算子和 HCCL，NPU 数值待验证，MC2 × TP 尚未适配；
- Domino 已有 CUDA dispatch/combine 的 stream/event 调度及存储生命周期管理；CUDA 数值与实际 overlap 待验证，NPU 及完整反向重叠待迁移；
- 低精度 dispatcher 依赖当前尚未移植的量化栈；
- FSDPTurbo dispatcher 不能直接作用于 TorchLLMTuner 的 `GroupedExperts`，接口和权重布局不同。

## 9. Pipeline Parallel

FSDPTurbo 当前没有对应实现。TorchLLMTuner 使用 `torch.distributed.pipelining`：

- 自动或显式生成每个 model part 的 FQN；
- 支持物理 stage 和 virtual stage；
- 使用 PyTorch 注册 schedule，包括 1F1B 和兼容的 interleaved schedule；
- first-stage modules 与 decoder layer 一起切分；
- packed positions 和 mask 作为 microbatch kwargs 传递；
- validation 通过 schedule eval 路径执行；
- 每个 stage chunk 继续应用 TP、EP、CP、AC、compile 和 FSDP。

已知边界均有 guard：tied embedding 跨首尾 stage、DSA dtype seam、packed flat stream 的 token 中间 microbatch 切割，以及 chunked loss。

## 10. 混合并行装配顺序

TorchLLMTuner 的顺序为：

```text
PP split
  └─ TP → EP → CP → Activation Checkpoint → Compile → FSDP
```

顺序理由：

1. TP 先建立 projection 与 sequence collective 边界；
2. EP 消费 TP 后的 dense/MoE 布局并拥有 routed experts；
3. CP 安装 attention kernel；
4. AC 包裹完成并行变换的 layer；
5. compile 捕获最终计算图；
6. FSDP 最后包装，使参数通信 hook 位于最外层。

该顺序与 TorchTitan 一致，也符合 TorchLLMTuner 的 HF swap 模型。FSDPTurbo 的 Quantization → TP → CP → Patch → EP → Memory → FSDP 顺序包含其量化和 patch 特性，不能原样替换 TorchLLMTuner stage 表。

## 11. 能力完整度矩阵

| 能力 | 状态 | 说明 |
|---|---|---|
| FSDP2 | 已实现，待目标设备验证 | dense、MoE、HSDP、reshard、mixed precision |
| FSDP CPU offload | 本轮补齐 | 已暴露 enable 与 pin-memory 配置 |
| FSDP custom backend | 部分迁移，CPU 数值通过；完整验收未完成 | 自定义状态/参数组、ancestor hooks、多通信上下文及跨 hook 缓冲释放；复用 2.14 collectives；目标设备及剩余生命周期待验证 |
| TP dense | 已实现，待目标设备验证 | attention/MLP/embed/head/vocab loss |
| TP MoE | 有条件实现 | 已支持布局完整；未知布局拒绝 |
| PP | 已实现，待目标设备验证 | stage、schedule、microbatch、validation |
| CP KV AllGather | 已实现，待目标设备验证 | FlexAttention 与 BlockMask 契约内 |
| CP Ulysses | 已实现，待目标设备验证 | head 整除和顺序约束内 |
| CP KV AllGather × Ulysses | 已实现，CPU 通信数值通过；待生产 attention 后端验证 | KV2×Ulysses2 的 rank、mask 索引和通信梯度有独立参考；不等于完整模型或 GPU flex 验证 |
| CP KCP | 部分实现；纯 KDA 待设备验证 | 已接纯 KDA 实例 patch，本地 Kimi Linear 已注册并有 CPU 训练证据；Kimi Linear MLA 已接实例级 Ulysses callable，CPU/Gloo CP2 的独立全序列 eager 输出/输入梯度/参数梯度参考 8 组通过；KDA 目前拒绝 padding（本地语义为移除 token，不能以清零替代）；完整 hybrid KDA+MLA 训练、Kimi K3 及独立 compressor 路径仍未验收（KCP-HYBRID-01）；需 fla 与 Ascend kernels，目标设备仍未验证 |
| EP All-to-All | 已实现，待目标设备验证 | 已支持 HF MoE 布局内 |
| EP TorchAO | 可选 | 需要 TorchAO |
| EP Domino | CPU 数值通过 | 双切片提前发起 token exchange，再消费并计算；前向/梯度、重计算、空 rank 通过；两片 count 合并；CUDA dispatch/combine stream/event 调度已加入但未经目标设备验证；NPU/完整反向重叠仍缺失 |
| EP MC2 | 部分实现，NPU 验证待运行 | torch_npu 三融合算子、HCCL、FP16/BF16；CPU 模拟梯度通过；TP 候选已接通但全模型梯度验收失败（MC2-TP-PUBLIC-01）；独立专家 forward hook 绕过已修复，真实 CPU FSDP 生命周期通过（融合算子模拟）；NPU 仍待验，见 MC2-EXPERT-HOOK-02 |
| TP × FSDP | 已实现，待目标设备验证 | 有多进程等价性测试 |
| CP × TP | 已实现，待目标设备验证 | 联合 sequence 整除和 loss mesh |
| EP × FSDP | 已实现，待目标设备验证 | sparse mesh 与 expert placement |
| TP × EP | 已接线，SGD限定CPU参考通过；多轴AdamW验收开放 | EP>1按专家数量分片；EP1特征分片为另一布局，fixed router的CPU DP2×TP2完整恢复已通过，设备及其他轴组合仍需验收 |
| PP × TP/CP/EP/FSDP | 已接线 | 各 stage 使用相同 stage order；设备组合测试按环境运行 |
| TP × EP × CP | 已接线 | mesh 与输入/loss 路径完整；依赖支持的模型和设备 |

入口接线、实现语义和验证范围分别判断。下列当前工作区 CPU/Gloo 证据
补充上表，不能推断为目标设备、其他模型或所有度数组合已通过。

| 检查 | 当前数值证据 | 范围与限制 |
| --- | --- | --- |
| 保存subprocess × FSDP恢复 | `fsdp_checkpoint_equivalence.py --fsspec --process-save-cpu`，2rank CPU/Gloo，native/custom × shard/replicate源目标8配置；rank0在保存子进程内写shard后失败 | 真实PROCESS executor、writer序列化、共享CPU staging、异常回传和fresh runtime续训；同拓扑精确一致，跨拓扑/独立dense参考1e-5。关闭pinning/async staging/nonblocking copy，不证明设备pinned传输、云端或并发manager正确；PyTorch daemon按主进程生命周期复用 |
| 前向变更 × 异步快照恢复 | `async_snapshot_equivalence.py`，独立2rank full/resume，CPU/Gloo，custom FSDP2、PP1、动态EMA | 延迟真实DefaultStager读取、共享CPU存储与缓存复用、PROCESS上传；持久forward buffer及续训全状态精确一致。禁用pinning/nonblocking copy，不证明设备staging或并发load安全 |
| EP × efsdp × AC/compile × 异步快照恢复 | `async_snapshot_equivalence.py`，独立2/4rank full/resume，Qwen3-MoE、EP2、custom FSDP、full AC、aot_eager、动态EMA、PROCESS | 2rank为efsdp1，4rank为efsdp2；实际MoE/DTensor mesh、非空compile图、延迟shared CPU staging和缓存复用均断言；更新前恢复状态与最终模型/AdamW/scheduler/EMA/trainer逐位一致，loss误差0。只证明相同编译模式CPU/Gloo恢复，不证明跨eager/compile、Inductor、设备pinning或NCCL/HCCL |
| PP/TP/EP低精度梯度范数 | `parallel_grad_norm_equivalence.py`，2rank CPU/Gloo、FP16局部梯度、FP64独立参考 | 三种轴分别归约得到norm=500并正确裁剪，PP空梯度stage及PP/TP/EP无参数参与者归约norm=400；仅原语数值证据，不证明完整并行模型布局或CUDA/NPU路径 |
| 本地数据身份 × PP恢复 | `pp_checkpoint_equivalence.py full/resume WORKDIR --packed`，2rank CPU/Gloo、PP2、native FSDP、动态EMA | 各rank资产目录不同但内容相同，独立进程恢复全状态精确一致、loss差异0；CPU attention为dense SDPA测试替代，不覆盖远端recipe身份或CUDA flex内核 |
| fsspec DCP同step失败恢复 | `fsdp_checkpoint_equivalence.py --fsspec`，2rank CPU/Gloo，native/custom × shard/replicate源目标8场景；memory/file CPU同步/线程故障与重试回归 | file://真实适配、rank0写后失败、fresh runtime恢复及后续训练；同拓扑精确一致，跨拓扑/独立dense参考1e-5。S3/GCS整对象发布已接入但未实测；memory不能跨进程，不支持其多rank/pinned-subprocess组合 |
| HF同step失败恢复 | `hf_checkpoint_equivalence.py`，2rank CPU/Gloo，native/custom × 单/双输出分片 × 合并/发布失败8场景 | 索引generation发布、各rank收到错误、fresh DTensor模型恢复旧权重与成功重试均精确一致；只覆盖local HF，非remote或设备性能证据 |
| custom FSDP 生命周期 | `fine_grained_fsdp_equivalence.py`，2 rank，16 配置；每个 parent hook 下两个 unit、两 parent、grad/no-grad 输入、reshard、重计算、单/多参数 mesh，2 microbatch × 3 momentum SGD 更新及前向/反向失败后 reset；step2 DCP 保存到新建运行时，step3 恢复继续 | 独立 dense 输出、输入/参数梯度、更新参数及 momentum；恢复与连续训练逐位一致；真实 context/缓冲归属与清理断言；仅 CPU/Gloo float32，不证明设备 overlap、混合精度或 Trainer checkpoint 装配 |
| custom FSDP checkpoint manager | `rng_checkpoint_equivalence.py`，2 rank，同步/异步；root hook 下3 context、rank独立随机流和dropout，新运行时恢复两步 | loss、模型、AdamW、scheduler逐位一致；真实manager与staging/close；不覆盖Trainer构造、拓扑变化或目标设备 |
| custom relocated FSDP低精度DCP恢复 | `fine_grained_checkpoint_equivalence.py`；2 rank CPU/Gloo，BF16/FP16 compute、FP32 master/reduction × reshard开关 × full AC开关，8配置 | 同步生产DCP保存step2，不同seed新建模型/AdamW/scheduler/EMA；恢复后更新前及step3输出、完整梯度、更新后状态逐位一致，退出0；未覆盖compile、异步transport、拓扑变化、数据/RNG、目标设备或完整Trainer |
| custom FSDP Trainer恢复 | `pp_checkpoint_equivalence.py`，PP1/FSDP2、custom、dynamic EMA；独立 full/resume 进程，2+2步 | loss、模型、AdamW、scheduler、trainer计数及EMA精确一致；默认module hooks、CPU/Gloo；relocated hooks与目标设备待验证 |
| PP2 × FSDP2 Trainer恢复 | `pp_checkpoint_equivalence.py`，4 rank，native/custom，1F1B、4 microbatch、dynamic EMA；独立 full/resume 进程，2+2步 | loss、模型各 rank shard、AdamW、scheduler、trainer计数及EMA精确一致；CPU/Gloo float32；实际 custom state accessor 修复反向收尾；其他 schedule、relocated hooks、混合精度及目标设备待验证 |
| PP2 × custom FSDP2跨AMSGrad恢复 | `pp_checkpoint_equivalence.py`，4 rank、1F1B、4 microbatch、dynamic EMA；源/目标AMSGrad 1→0与0→1，独立full/resume进程2+2步 | 两方向各rank更新前完整状态及最终模型/optimizer/scheduler/trainer/EMA精确一致，后续loss差0；CPU/Gloo float32，不外推EP、compile、其他schedule或目标设备；命令与日志见file_review对应批次 |
| EP2 × custom FSDP4跨AMSGrad恢复 | `pp_checkpoint_equivalence.py`，4 rank、PP1、qwen3_moe eager experts、dynamic EMA；AMSGrad 1→0与0→1，独立full/resume进程2+2步 | 两方向各rank更新前完整状态、最终模型/optimizer/scheduler/trainer/EMA精确一致，后续loss差0；CPU/Gloo float32，未覆盖compile、TP、共享专家或目标设备；证据见file_review对应批次 |
| 多本地 stage × custom FSDP2 | 4 rank PP2，Interleaved1F1B/ZBVZeroBubble，6层、4 microbatch；`pp_equivalence.py` 四次 AdamW 独立 dense 参考，`pp_checkpoint_equivalence.py` full/resume | loss最大误差分别4.768e-7/9.537e-7，梯度范数、裁剪梯度和更新参数容差1e-5；dynamic EMA恢复逐位一致；仅CPU/Gloo float32，其他调度、AC及设备待验证 |
| PP2 × CP2 Trainer训练 | `pp_cp_equivalence.py`，4 rank，KV-CP、seq256；custom的1F1B累积1/2、Interleaved1F1B累积1及native的1F1B累积1，四次AdamW | 独立dense参考loss最大差异4.768e-7，梯度范数/裁剪梯度/更新参数容差1e-5；真实CP collective/autograd及mask predicate，CPU SDPA替换attention；不验证生产flex sparse索引/内核、设备、packed/headtail或恢复 |
| PP2 × CP2 headtail | CPU SDPA替代的独立重排参考通过 | custom FSDP、1F1B四步；loss/梯度检查通过，未重排参考首层V投影参数差异1.239e-5；独立KV/mask列重排参考FP32通过，参数差异7.400e-6，double诊断4.221e-7；支持归约舍入/AdamW放大解释，保留未重排失败及设备缺口 |
| packed PP2 × CP2训练/验证 | `pp_real_corpus_equivalence.py`，4rank、KV-CP、custom FSDP、1F1B、seq256、三步AdamW、finite validation | 独立dense参考最终参数最大差异1.071e-6，训练/验证loss容差1e-5；真实文档mask predicate及CP collective，CPU SDPA替代；全padding microbatch通过，不覆盖生产flex内核、设备、headtail/双CP或恢复 |
| packed PP2 × CP2 DCP恢复 | `pp_checkpoint_equivalence.py --packed`，4rank、custom FSDP、KV-CP、1F1B、seq256、dynamic EMA，独立full/resume进程 | step2恢复后两步loss差异0；模型各rank shard、AdamW、scheduler、trainer counters、EMA逐位一致；Grain自动恢复；CPU SDPA替代，不证明生产flex、设备或其他组合 |
| PP共享参数所有权 | 同stage身份/前向/梯度/SGD回归通过；跨stage明确拒绝 | 分割前按实际Parameter身份全局检查，避免deepcopy静默解除共享；跨stage梯度同步未实现，storage view与共享buffer不在此检查范围 |
| 默认FSDP dense层共享 | `pattern_fsdp_equivalence.py`新增默认decoder两层共享，native/custom两rank通过 | grouped state与Parameter身份保持，独立dense loss/梯度/三次SGD更新；不覆盖MoE混合或层与embedding/norm交叉共享、storage views、设备 |
| 默认FSDP层/norm边界共享 | `pattern_fsdp_equivalence.py`，tied/untied边界，native/custom两rank CPU/Gloo，prefetch窗口2 | 共享bias和state保持；独立dense loss/梯度/三次SGD更新等价；该bias fixture未验证DCP、MoE/root共享、storage views或设备 |
| 默认FSDP root/dense共享 | `pattern_fsdp_equivalence.py`，root external weight实际参与前向，native/custom两rank CPU/Gloo | root管理共享层，Parameter身份及独立dense loss/梯度/三次SGD更新保持；未验证新root布局DCP、MoE共享、storage views或设备 |
| 默认FSDP dense/MoE共享 | `pattern_fsdp_equivalence.py`，native/custom两rank CPU/Gloo，单专家Shard(1) | 共享投影及state保持，独立dense loss/全部梯度/三次SGD更新等价；26总场景通过；未验证新混合布局DCP或设备 |
| 默认FSDP EP共享MoE层 | `ep_fsdp_equivalence.py --shared-projection --fsdp-implementation native/custom`，四rank EP2/FSDP4 | 两MoE层共享Q Parameter/state；独立dense Q/router/专家梯度与loss通过1e-4容差；expert efsdp2与dense mesh4保持；未比较更新/DCP或设备 |
| 共享FSDP state symm_mem范围 | 六CPU布局/范围回归；`pattern_fsdp_equivalence.py`真实native/custom混合state拦截设备enable调用 | dense排除整个MoE共享state，all每state一次；嵌套独立state边界正确；仅验证策略，未运行设备symm_mem通信/性能 |
| 默认FSDP EP共享MoE层恢复 | `pp_checkpoint_equivalence.py`，MoE1/EP2/PP1/FSDP4、shared Q、native/custom、dynamic EMA | 四rank CPU/Gloo step2恢复至step4，loss差异0、模型/AdamW/scheduler/trainer/EMA逐位一致；共享身份/state及专家mesh2/router mesh4保持；未验证共享专家、root专家别名或设备 |
| HF外部专家别名布局限制 | root fused参数/整个HF block别名两CPU回归；`ep_fsdp_equivalence.py --external-expert-alias`四rank EP2 | 布局不变复用source保持支持；fused split/EP slice与未适配外部消费者在installation前明确拒绝，模型身份保持；guard仅修复静默共享破坏，外部布局迁移未实现 |
| EP转换后root专家别名 | `ep_fsdp_equivalence.py --converted-root-expert`；`pp_checkpoint_equivalence.py` root expert模式，native/custom四rank EP2/FSDP4 | sparse placement/梯度等价；修复别名EP索引后fresh DCP恢复loss差异0，模型/AdamW/scheduler/trainer/EMA逐位一致；旧异常布局checkpoint、外部完整专家消费者、storage views/设备不在保证内 |
| 默认FSDP root共享恢复 | `pp_checkpoint_equivalence.py`，PP1/FSDP2、root Q projection alias、native/custom、dynamic EMA | 两rank CPU/Gloo新进程step2恢复至step4，模型/AdamW/scheduler/trainer/EMA逐位一致、loss差异0、共享身份保持；不外推MoE/storage views或设备 |
| 默认FSDP层/norm共享恢复 | `pp_checkpoint_equivalence.py`，PP1/FSDP2、shared RMSNorm weight、native/custom、dynamic EMA | 两rank CPU/Gloo新进程step2恢复至step4，loss及模型/AdamW/scheduler/trainer/EMA逐位一致，共享Parameter/state保持；不外推其他共享布局或设备 |
| 默认FSDP共享dense层恢复 | `pp_checkpoint_equivalence.py`，PP1/FSDP2、shared Q projection、native/custom、dynamic EMA | 新进程Trainer step2恢复至step4，loss、模型shard、AdamW、scheduler、trainer counters、EMA精确一致；共享Parameter/state保持；CPU/Gloo，不覆盖设备或其他共享布局 |
| 双 CP 拓扑与 mask | `dual_cp_topology.py`，4 rank，KV2×Ulysses2；独立 rank 公式与 subgroup SUM，真实 causal/packed BlockMask 及模型 mask 构建入口 | 验证 512-token mask 的全局 Q 偏移和文档边界；不执行完整模型 attention 或 backward |
| 双 CP 通信前向与反向 | `dual_cp_attention_equivalence.py`，4 rank，float64；noncausal/causal/packed × Q4/KV4、Q4/KV2 或 Q6/KV2，共 9 配置；独立全序列 SDPA 比较输出及 Q/K/V 梯度，容差 1e-10 | 保留真实 DualCPFlexKernel collective/autograd，仅将 `run_flex` 替换为 SDPA；不验证生产 flex BlockMask、目标设备 GQA、TP/PP 组合或完整 HF forward |
| PP2 × EP2 | `pp_ep_equivalence.py`，4 rank，1F1B、4 microbatch、4 次 AdamW；默认/full/selective 比较 loss、裁剪梯度和更新参数 | HF Qwen3 MoE；aux 系数 0.3 的 full/selective 也通过；无 padding、其他 schedule 或设备证据 |
| EP2 × FSDP4 | `ep_fsdp_equivalence.py` native/custom 的默认/pattern 比较 loss、router 和专家梯度 | eval 模式；expert mesh rank [0,2]/[1,3]、dense rank [0,1,2,3]；custom 另断言真实状态与两个参数组；不覆盖辅助 loss、参数更新、relocated hooks 或 bias 更新 |
| root MoE TP2 全参数梯度与更新 | `tp_moe_equivalence.py` 默认 FP32 router 与 `--router-fp64`，两 rank CPU/Gloo、FP64 专家与 shared MLP；首次/重复 apply_tp，生产 Trainer 复制梯度归约、一次 SGD | 独立手动分片图比较输入及全部参数梯度和更新；FP64 router 额外比较 dense router 梯度；FP32 partial/dense 的舍入差异保留；非完整 HF/FSDP/AC/设备验证 |
| PP 验证日志失败传播 | `pp_real_corpus_equivalence.py`，PP2/native 两 rank 与 PP2×KV-CP2/custom 四 rank、1F1B、packed corpus | 正常有限验证对照完整模型；末阶段 sink 失败传给所有 stage，module 模式/CPU RNG 精确恢复，后续 barrier 成功；CPU SDPA 替代 attention，不证明设备或其他 schedule |
| MoE balancing hook | `quantile_bias_equivalence.py`，4 rank，DP×CP、DP×TP 有/无 EP；真实 optimizer hook 更新 bias 与清理统计 | 独立 rank 求和参考；包含大整数 histogram；未执行 Trainer 装配、router forward 或参数更新 |
| Quantile完整Trainer恢复（批次117） | `trainer_ema_process_resume.py corpus async dp2 ac validate moe ep2 quantile`，native/custom FSDP，真实DeepSeek-V3 sigmoid router、EP2、full AC、动态EMA；连续/初始/恢复三个独立作业，七类状态及本地布局精确 | CPU/Gloo；bias非零且持久化、histogram非持久且清零、验证隔离；不证明compile、其他轴、目标设备或拓扑变化 |
| Quantile编译恢复（批次118） | 上述命令追加compile，native/custom各三个独立作业；生产aot_eager图数断言与完整状态比较通过 | CPU/Gloo；不证明eager/compile算术等价、Inductor、其他轴或目标设备 |
| MoE AC 与编译 | CPU 单测 full/selective，多 outstanding forward、不同/无 mask；Dynamo eager 比较输出和梯度 | 回放保存对应 mask、抑制重复统计，异常清理；未验证 Inductor、region 或设备并行编译 |

执行集成脚本时设置 `PYTHONPATH=.`，用文档所列 rank 数启动。当前 macOS
CPU 验证使用 `GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1`，属于 eager
证据。详细命令与日志见 `docs/file_review.md`；上述历史结果需在相关代码
变化后重新验证。完整 custom FSDP、MC2 融合和 Domino 的完整 accelerator pipeline 仍有
实现或验证缺口；CPU 验证不能替代目标设备需求。

双 CP 通信数值的可复现命令（2026-10-15 当前工作区复跑，9 配置均通过；见 `DUAL-CP-EVIDENCE-02`）：

```bash
PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 \
python -m torch.distributed.run --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29847 \
  tests/integration_tests/dual_cp_attention_equivalence.py
```


## 12. 本轮代码变更

1. 新增 `ParallelConfig.fsdp_config.fsdp_cpu_offload`，默认 `False`；
2. 新增 `ParallelConfig.fsdp_config.fsdp_cpu_offload_pin_memory`，默认 `True`；
3. 将配置传入 `CPUOffloadPolicy(pin_memory=...)`；
4. 增加 FSDP 配置传递测试；
5. 删除 `fully_shard_parallel/apply.py` 兼容转发文件；
6. 保持 `fully_shard_parallel/fully_shard_parallel.py` 为唯一核心入口；
7. 更新相关文档和目录映射。

## 13. 后续迁移优先级

### P0：正确性与可验证性

- 为 FSDP CPU offload 增加 CUDA/NPU 多卡 forward/backward 与 checkpoint 测试；
- 为 PP × TP × EP × CP 建立目标设备的小模型组合矩阵；
- 为每个可选 dispatcher 建立相同输入、路由结果和梯度的等价性基线。

### P1：通用性能能力

- 验证已有 FSDP prefetch 窗口的目标设备内存、重叠及性能，并检查 pattern 注册顺序与实际执行顺序的差异；
- 增加受约束的 TP plan override，只允许覆盖现有校验可证明安全的 Linear；
- 抽象 EP dispatcher protocol，为设备后端保留接入点。

### P2：设备专用迁移

- 将 FSDPTurbo NPU collective/GMM 通过独立 backend registry 接入；
- 用 HCCL/NPU CI 验证已接线 MC2 adapter，并继续 TP/混合并行适配；
- KCP 作为新的 CP strategy 与 KDA 模型能力一起迁移；
- 量化栈落地后再迁移低精度 AllGather 和量化 dispatcher。

## 14. 验证命令

CPU 契约测试：

```bash
pytest -q tests/unit_tests/cpu/parallel tests/unit_tests/cpu/test_config.py
```

完整单元测试：

```bash
pytest -q
```

代码检查：

```bash
pre-commit run --all-files
```

多进程数值测试位于 `tests/integration_tests/`，应按照脚本要求使用对应数量的 `torchrun` rank 和目标设备执行。
# Migration status (2026-10-12)

The first migration slice is implemented in the working tree:

- configurable forward and backward FSDP prefetch windows, with the previous
  EP one-unit behavior retained when both values are zero;
- extended module patterns (`*`, `?`, `{*}`, `{start-end}`) shared by TP and
  FSDP, including unmatched/conflicting-plan validation;
- pattern-selected native FSDP units, ignored modules, per-unit overrides and
  explicit root wrapping;
- pattern TP overrides layered over the model's HF plan;
- the FSDPTurbo fixed router flag: expert `(token + slot) % E` with constant
  `1/K` scores, bypassing learned scores, normalization and route scaling.
  This is separate from torchtitan `_debug_force_load_balance`, which keeps
  learned scores and uses `(token * K + slot) % E`.
- independent `context_parallel_kv_allgather_size` and
  `context_parallel_ulysses_size` axes. The composite kernel gathers K/V over
  the first group, exchanges Q/K/V from sequence to heads over the second,
  attends `S / kv` queries against full-sequence K/V, and applies the inverse
  exchange to its output. Both axis fields can be combined; their product determines the total CP degree.
  Each axis can also be used alone with the other degree set to one.

Domino is integrated at the dispatcher/module boundary: routing, dispatch,
expert compute and combine execute in two independently tracked token slices,
with checkpoint-compatible `GroupedExperts` weights. CUDA slice 1 dispatch
and both combine exchanges use a dedicated communication stream with
producer/consumer synchronization and allocator stream ownership. Expert
computation uses the caller's stream, with a final event joining both outputs. Target-device numerical and profiler overlap verification are pending;
NPU adaptation and the complete upstream backward overlap pipeline remain
unimplemented. MC2 now has a lazily loaded torch_npu fused autograd adapter and a routed-expert
execution path preserving the existing three-parameter checkpoint layout.
Missing packages/operators cause capability errors; there is no ordinary
all-to-all fallback. CPU emulation covers adapter algebra, while real NPU/HCCL
numerics remain unverified. TP2/EP2 configuration and model assembly are
reachable, but the CPU-emulated fused path still fails the existing FP16
parameter-gradient acceptance check; see `MC2-TP-CURRENT-STATE-RECHECK-02`.
Configuration acceptance and adapter wiring do not establish numerical support.

KCP is wired as a separate CP strategy for structurally recognized Kimi Delta
Attention modules. It installs the FSDPTurbo-compatible KDA forward, builds the
global sequence context, runs CP-aware causal convolutions and `chunk_kda`, and
enforces batch-size/cache/packed-input and TP incompatibility constraints. Its
`fla` and `triton_ascend_kernels` imports remain lazy, with an actionable error
when an Ascend KCP runtime is unavailable. The current KDA path requires
equal, nonempty local sequences and rejects padding: zeroing padded inputs
would change convolution/recurrent state. Compact padding/state exchange remains
unfinished (`KCP-PADDING-STATE-01`); rejection is a boundary, not completed migration.

Custom fine-grained FSDP now uses its own composable state contract and
`FineGrainedFSDPState`, with PyTorch 2.14 initialization helpers, parameter groups
and collectives. Hooks may attach to an ancestor containing every managed
module. The same strategy must wrap all units and the root. States sharing a
hook target receive different buffer/event contexts; communication streams and
backward prefetch order remain shared to preserve collective order, as upstream.
Forward completion, backward finalization and failed-iteration reset clean every
context. Custom parameter groups wait on retained buffers from other hooks before
releasing them, while preserving same-hook retention and native local queue
limits. Native 2.14 collectives and post-reduce event handling remain intact.
Target-device overlap and performance are not established.

### PyTorch 2.14 custom FSDP 生命周期迁移核对

以下结论来自安装环境源码（Python 3.13 / PyTorch 2.14.0）及本地 FSDPTurbo
`distributed/fine_grained_fully_shard/{fully_shard,state,param_group,utils}.py`，
下表为源码契约；运行时证据范围另见能力矩阵中的 custom FSDP 生命周期条目。

| 生命周期 | FSDPTurbo 原实现 | PyTorch 2.14 接口与迁移要求 |
|---|---|---|
| 状态初始化 | `init(modules, device, mp_policy, hook_module, comm_ctx_index)` | 原生 `init` 新增 `auto_reshard_after_forward`；自定义初始化必须保留 root 自动 reshard 语义及 relocated hook 注册 |
| 参数组 | 单个 `_fsdp_param_group` | `_fsdp_param_groups` 列表；`_init_param_group` 按 shard/replicate process group 分组，所有生命周期都须遍历全部组 |
| 上下文建立 | 按每个 hook target 的 unit 序号分配 context | 共享 `_state_ctx`，为每个组设置 `_param_group_index` 和 `_num_param_groups`；每个 context 单独计算 input buffer cap |
| forward 入口 | 沿用原生 root setup | 原生只同步当前 context 的 all-gather stream；多 context 必须全部等待 optimizer event 或当前 compute stream |
| forward 收尾 | 释放所有 context 的 all-gather state | 保留 output cast、重计算及 grouped forward 收尾；使用各 context 自己的 stream 等待其 event 后释放结果 |
| backward 收尾 | 单个 `reduce_scatter_state` | 改为 `reduce_scatter_states` 列表；全部 context 等待并释放全部保留 buffer，清除各自的执行顺序 |
| 失败恢复 | 旧实现未覆盖当前接口 | `_reset_iter_state` 还处理 `_last_post_reduce_events`、group reset 和 partial gradient；必须扩展到全部 context |

FSDPTurbo 的 `utils.py::copy_fsdp_comm_ctx` 共享四个通信 stream，也直接共享
`post_forward_order` 列表；多 context 不等于多套独立 stream。迁移时应分别定义
stream、event、buffer 和执行顺序的所有权，不能通过浅拷贝整个 context 沿用旧行为。
`param_group.py` 还依赖旧版 collective 返回布局，单靠过滤函数参数不能证明
PyTorch 2.14 生命周期兼容。

完整迁移的验证至少需要：同一 parent hook 下多 unit、不同 parent hook、
多参数 mesh/group、无梯度输入、梯度累积、重计算、混合精度、prefetch、
异常后 reset、checkpoint 恢复，以及连续多步 loss/梯度/参数等价。
现有 `fsdp_mesh_equivalence.py ... custom` 未配置 relocated hooks，不能单独
证明多上下文语义。新增 `fine_grained_fsdp_equivalence.py` 覆盖上述 CPU 子集：

```bash
PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 \
python -m torch.distributed.run --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29851 \
  tests/integration_tests/fine_grained_fsdp_equivalence.py
```

脚本可加 `--device cuda` 验证 CUDA/NCCL；目标设备、混合精度、编译及
多Trainer组合仍待验证；Trainer relocated-hook恢复现由
`pp_checkpoint_equivalence.py`的`LLMTUNER_TEST_RELOCATED_HOOKS=1`覆盖：
PP1/FSDP2、四投影共享attention祖先hook、四通信context、dynamic EMA，
独立full/resume进程的2+2步loss/model/AdamW/scheduler/trainer/EMA逐位一致。
四rank PP2×FSDP2/custom、1F1B、4 microbatch、dynamic EMA使用同一选项
验证分割后的hook对象归属及四context；独立full/resume的loss和完整状态逐位
一致。全模型patterns先解析再绑定本地parts，未选中子单元的part仍shard root。
full AC同一PP2配置亦通过独立full/resume逐位一致；custom状态按显式checkpoint
replay标记进入PRE_BACKWARD，避免多microbatch的IDLE状态误走forward hooks。
1F1B/selective AC以及Interleaved1F1B/full AC（每rank两个本地stage、六层）
也通过相同独立full/resume逐位一致检查，后者逐part验证hook/context归属。
仅CPU/Gloo float32，不证明目标设备、region AC、compile或全部PP schedule组合。
whole-model compile另有PP1/FSDP2/custom relocated-hook/dynamic EMA的
`aot_eager`独立full/resume证据：2+2步loss和完整状态逐位一致；resume额外
确认OptimizedModule及Dynamo unique_graphs非零。该证据不覆盖Inductor、
compile×AC、PP compiled调度或目标设备。
两 outstanding forward 逆序反向已由
同脚本 --outstanding-forwards 的16配置验证；同rank同步注入的
backward异常reset已验证并修复梯度残留。
relocated-hook DCP 恢复已在16配置脚本中验证；CPU event 单测只证明等待与
释放顺序，不能推断真实设备上的重叠收益。

MC2 has an integrated fused runtime adapter. Completion still requires real
Ascend kernel forward/backward, empty-token and recomputation evidence; CPU
emulation and configuration acceptance do not establish hardware correctness.
The target script is `tests/integration_tests/ep_mc2_equivalence.py`.

### PP 优化器参数规则补充

Trainer 装配向 OptimizersContainer 显式传入 PP mesh。统一配置中的 regex 可仅
命中某个 stage；是否无效按整个 PP 组的实际参数分配检查，本地参数仍必须唯一
归属。直接构造不传 mesh 时仅检查本地 parts；传入 mesh 要求所有 stage 同时
调用并使用同序规则。

CPU/Gloo 两 rank 的 stage 专用不同学习率与独立 AdamW 更新精确一致；
`LLMTUNER_TEST_OPTIMIZER_GROUPS=1` 的 `pp_equivalence.py` 验证真实 PP2 Trainer
四次更新，embedding/head/catch-all 规则的 loss、梯度与参数匹配独立 dense
参考。此证据不覆盖设备后端、rank 间配置不一致或构造前异常协调。

### Compiled checkpoint 旧名称兼容

DCP load planner 支持旧 checkpoint 中完整 `_orig_mod` 包装段的名称迁移，
覆盖模型、optimizer tensor/参数组选项和 EMA，保留严格缺失键检查；多个保存键
映射到同一当前键时拒绝加载。实际两 rank CPU/Gloo、custom FSDP、重定位 hook、
`aot_eager` 和 dynamic EMA 的旧 compiled checkpoint 恢复后，全部最终状态精确
比较通过。旧测试参考文件仅做键名规范化，未更改 tensor 数值。

新保存点快照确认 compiled → eager 加载后、首次更新前的状态精确一致，但后续
更新仍出现最终数值差异；跨模式最终等价尚未通过。该证据不覆盖 Inductor 或目标
设备编译。

### Custom FSDP × full AC × compile 独立参考

生产 AC recompute context 现继承 native SAC dispatch mode，以满足 PyTorch compile
的 context 类型契约，同时保留重计算标记和 MoE mask 恢复。
`fine_grained_fsdp_equivalence.py --compile-backend aot_eager` 在两 rank CPU/Gloo
上通过 16 种输入梯度、reshard、full AC、多个参数组组合；每种配置在两 rank
独立确认生成编译图。输出、输入与参数梯度、momentum、更新参数与 eager dense
参考比较，DCP 新运行时恢复保持精确比较。测试中的私有状态表观察仅在 eager
执行，反向后的生命周期检查仍保留。此证据不覆盖 Inductor、目标设备或 HF
跨 eager/compile 恢复后的精确数值一致性。

同脚本追加 `--outstanding-forwards` 的 16 配置也全部通过：两次 forward 后逆序
backward，覆盖 full AC 与新建运行时恢复。全量 CPU suite 为 3180 passed、2 skipped。

真实 HF Trainer 两 rank CPU/Gloo 也通过 full AC + custom FSDP + aot_eager
+ dynamic EMA 的连续训练/恢复精确比较，分别覆盖 Qwen3 与 Qwen3MoE + EP2，
均使用重定位 attention hooks，且保存点快照和最终状态精确一致、loss 差异为 0。
MoE/EP 这项为同实现恢复证据，不替代独立 dense 数值参考。

### PP2 × EP2 × AC × compile 独立数值参考

`pp_ep_equivalence.py --compile-backend aot_eager --aux-coefficient 0.3`
在四 rank CPU/Gloo 上分别配合 `--activation-checkpoint full` 和 `selective`
通过。每 rank 生成编译图；四步 loss、clip 后梯度及 AdamW 更新参数与独立
未编译、无 AC、单进程 EP1 参考一致，保持既有 `1e-4` 容差。初始参考模型
从真实 PP/EP shard 重建；实际 stage/专家分片断言保持。该结果扩展计算正确性
证据，设备融合内核、Inductor 和跨模式精确恢复仍需独立验收。

Qwen3 另有四步 CPU/aot_eager 回归，以原生 HF CausalLM.forward 为独立参考，
比较 logits、loss、clip 后梯度和 AdamW 更新参数，采用 torch.testing 既有
float32 默认容差且确认生成编译图。该结果补充计算证据；跨 eager/compile
checkpoint 的最终 bitwise 比较仍保留原断言和失败记录，不能据此宣称已通过。

### 跨执行模式首次更新诊断

`compile_trajectory_diagnostics.py` 在真实两 rank custom FSDP、重定位 hooks、
动态 EMA 上记录 optimizer.step 前后状态。aot_eager 连续训练与同模式恢复在
step3/4 的参数和 clip 后梯度逐位一致；eager 恢复 step3 更新前参数一致，但
12 个梯度键已不同，最大绝对误差约 1.49e-8，随后参数产生差异。跨模式最终
严格比较仍失败，不能据此宣布等价或放宽容差。下一步区分 clip 前梯度、前向
及反向/归约分歧；当前仅为 CPU/Gloo 定位证据。

后续 step3 诊断确认：两个 rank 的更新前参数和前向 logits 逐位相同，裁剪前
已有 12 个梯度键不同（最大约 1.86e-8），全局梯度范数逐位相同。因此首个
观察差异在反向/归约路径，裁剪不是来源；AOT backward 与归约仍需继续隔离。
严格跨模式恢复仍未通过。

进一步在原生 `foreach_reduce` 入口捕获局部完整梯度：首次更新25个参数的调用
顺序一致，但第0层和embedding的12个梯度键在归约前已不同，最大约1.86e-8；
第1层、最终norm/lm_head一致。compiled恢复控制逐位一致，下一步定位层间
backward与分支梯度累加；当前证据已将首次差异置于reduce-scatter之前。

默认 module hooks 对照（两 rank CPU/Gloo、dynamic EMA、aot_eager）：native 与
custom FSDP 的 compiled full、compiled resume、eager resume 均通过原始最终
逐位恢复检查，loss误差0。首次更新logits、归约输入、clip前梯度及更新后参数
也逐位一致。目前失败收敛到已测的重定位 attention hooks 配置，不能推广为
所有 compile/FSDP 组合；该配置的图边界和分支梯度累加仍待定位。

后续同分组反证：保持 Q-only pattern，仅将 hook owner 从 attention 切回 Q
projection 自身，compiled full/同模式恢复通过，但 eager 恢复仍失败；首次更新
logits一致，归约前同样12键差异（最大约1.86e-8）。因此上述“收敛到重定位”仅为
当时观察，现已被反证，不能归因于 hook 重定位本身。当前需定位 projection
细粒度分组与 AOT 图边界的交互；默认 decoder 单元通过的证据仍有效。

同 Q-only pattern 的原生 PyTorch FSDP 也复现上述跨模式严格失败：compiled
控制通过，eager在归约前出现相同12键梯度差异。复现不依赖custom state或
重定位hook，当前应继续隔离细粒度原生FSDP与AOT图分割/梯度累加交互。

### 无FSDP图边界反证

无FSDP的Q projection显式图边界重放：

- 诊断脚本新增测试专用 `LLMTUNER_TEST_REPLAY_Q_BREAK=1`，仅对Q projection的forward使用 `torch.compiler.disable`，其余模型保持aot_eager；未修改生产实现或严格恢复断言。
- 使用真实step2完整checkpoint及step3捕获输入，rank0/rank1分别独立重放，无通信、无FSDP。两个进程均退出0，logits逐位一致，梯度在float32默认容差内，但同样12键不逐位一致；embedding最大绝对差分别1.862645149230957e-8与1.30385160446167e-8，与真实归约前案例一致。无显式边界的既有重放两rank均逐位一致。
- 复现不需要FSDP：显式AOT图边界已足以触发同样键集合和误差幅度。尚未逐元素核对差值张量，也未证明具体梯度累加顺序，不能将幅度一致等同于根因完全确定。
- 复现命令：`LLMTUNER_TEST_REPLAY_Q_BREAK=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --replay-backward <capture-root> <rank>`；capture-root须含真实checkpoint/step-2和trace-resume-aot_eager/step-3-rank-<rank>.pt。两rank原始日志为 `/tmp/llmtuner-q-break-replay-{0,1}.log`。
- 下一行动：保存重放梯度并逐元素比较真实FSDP归约输入，随后导出分段AOT backward图核对共享输入分支累加；跨模式严格恢复仍失败，目标设备验证仍开放。

### 图边界重放与归约输入逐位对应

无FSDP重放与真实归约输入的逐元素核对：

- 扩展 `compile_trajectory_diagnostics.py --replay-backward` 的Q图边界诊断，分别读取eager/aot_eager真实step3的 `gradients_before_reduce`，校验参数键集合并比较全部完整局部梯度；不使用clip后或归约后状态作替代。
- rank0/rank1均退出0，两种模式各25个参数全部逐位一致，四个 `replay_vs_fsdp_gradient_differences` 列表均为空。日志 `/tmp/llmtuner-q-break-compare-{0,1}.log`。因此已测配置中的12键跨模式分歧完整复现于无FSDP显式Q图边界路径，不只是最大误差碰巧相同。
- 这将首个差异定位到分段编译反向执行，未证明具体算子或共享输入累加顺序。跨模式严格恢复仍未通过，不能认定FSDP所有行为均正确，也不能放宽恢复断言。
- 本批仅修改诊断测试与记录；相关pre-commit通过。下一行动：导出该无FSDP最小重放的AOT backward分段，核对Q/K/V共享输入梯度的累加顺序，构造独立的小规模多分支参考。

### 分段反向图与简化模型边界

AOT backward图导出及简化分支反证：

- 用 `TORCH_LOGS=aot_graphs` 配合既有Q图边界rank0重放命令导出真实分段图，进程退出0，保留 `/tmp/llmtuner-q-break-aot-graphs.log`。对应attention backward图中，V输入梯度view_12与K输入梯度view_15经 `aten.add.Tensor` 形成add_9；Q projection在禁用编译的边界外，Q输入梯度不在该图内累加。这确认了分段梯度汇合结构，未直接观察eager的运行时累加次序。
- 独立临时三分支实验使用固定seed7、输入128×64、三组64→128线性投影，Q设显式边界，其余aot_eager。对 `q*k+v` 及带Q/K RMSNorm的变体分别运行，均退出0，前向及输入梯度逐位一致；独立计算三条输入梯度后，eager/compiled均逐位对应 `(v+k)+q`。简化模型未复现，不能将真实问题直接归因于三分支加法不结合性。
- 原真实Qwen重放的完整梯度对应证据仍有效，跨模式严格恢复仍失败。下一行动：保留真实attention的SDPA、Q/K norm及残差依赖，隔离分支汇合；避免继续用已证实不复现的简单逐元素三分支扩大随机规模。本批无生产或测试代码修改，仅新增诊断证据记录。

### 首次残差梯度汇合的重结合

反向残差汇合的首次重结合差异：

- 诊断重放新增测试专用 `LLMTUNER_TEST_REPLAY_ADDS=<path>`，仅在backward中用TorchDispatchMode捕获形状1×128×64的add/add_输入，不包裹forward编译。rank0重放退出0，全部25参数仍逐位对应真实FSDP归约输入，观察器未改变已测结果。命令沿用Q_BREAK重放并增加上述变量，日志 `/tmp/llmtuner-backward-adds.log`，捕获 `/tmp/llmtuner-backward-adds.pt`。
- 两模式前6次加法的输入逐位一致。首个不同汇合处，eager记录6/7的输入可写为 `(r+a)+b`，AOT记录6/7为 `r+(a+b)`；r、a、b分别跨模式逐位相同，两个中间结果也分别逐位等于对应独立重算。两种重结合结果的最大绝对差9.313225746154785e-10，与捕获的汇合结果差异一致。这是残差与分段分支的三路汇合，不能仅用无残差Q/K/V小模型解释。
- 此证据确认已测rank0首次差异来自浮点加法重结合；尚未据此修改数值契约或放宽跨模式最终断言。下一行动：将残差路径纳入独立小模型复现，并核对rank1同一汇合；评估能保持生产编译机制的确定性累加实现及其代价。跨模式严格恢复仍失败，设备验证仍开放。

### 独立prenorm残差复现

rank1残差重结合确认与独立小模型：

- rank1启用REPLAY_ADDS完成真实输入重放退出0，25参数仍逐位对应FSDP归约输入。前6次加法输入相同，首次三路汇合的r/a/b及两个中间结果分别逐位对应；eager `(r+a)+b` 与AOT `r+(a+b)` 最大绝对差4.656612873077393e-10。rank0/rank1均确认该结构，不再仅为rank0观察。
- 新增诊断入口 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --residual-backward`。固定seed7，128×64输入、三组64→64线性、共享输入RMSNorm、Q/K RMSNorm、Q显式图边界和残差；完全不依赖HF/FSDP/checkpoint。退出0，前向逐位相同，输入梯度最大绝对差4.76837158203125e-7；输出仅为诊断，没有宣称等价通过。去掉共享输入prenorm的对照仍逐位相同。
- 日志 `/tmp/llmtuner-backward-adds-rank1.log`、`/tmp/llmtuner-residual-minimal.log`。相关pre-commit通过。下一行动：在独立小模型导出prenorm backward，检查残差与RMSNorm内部两个梯度分量的汇合，评估固定该边界累加的编译兼容方案。生产与原严格恢复断言未变，问题未关闭。

### RMSNorm反向分量的独立机制断言

独立小模型确认RMSNorm反向两分量与残差重结合：

- 导出 `--residual-backward` 的AOT图，prenorm backward返回 `add(mul_2,mul_6)`：直接分量与方差分量在图内先相加，再与图外残差汇合。日志 `/tmp/llmtuner-residual-minimal-graphs.log`，退出0。
- 小模型新增反向add输入捕获及机制断言：两模式前4次add输入逐位相同；最后三份输入分别为残差g、prenorm直接梯度、方差梯度，跨模式均逐位相同。断言eager输入梯度逐位等于 `(residual+direct)+variance`，compiled输入梯度逐位等于 `residual+(direct+variance)`。诊断运行退出0，保持最大绝对差4.76837158203125e-7，日志 `/tmp/llmtuner-residual-merge-assertions.log`。
- 这解释最小复现及真实追踪的首个三路汇合，未证明通过更改本项目FSDP即可消除编译器重分组。下一行动：检查上游RMSNorm实现/编译边界契约与PyTorch已有保持backward边界的机制；先在独立模型验证候选，再决定生产适配。禁止通过关闭compile或放宽严格恢复断言将问题标为关闭。

### 归一化输入副本候选

可微分输入副本候选的独立验证：

- 核对本地torchtitan `torchtitan/models/common/norm.py::GatedRMSNorm.forward` 使用F.rms_norm和local_compile；并非HF Qwen的手写RMSNorm，不能直接假设其反向边界相同。PyTorch strict_mode的autograd为未实现，不能用作可训练边界方案。
- 独立复现新增 `LLMTUNER_TEST_CLONE_PRENORM=1`：仅在共享输入RMSNorm前执行x.clone，保留Q图边界与aot_eager。仓库诊断命令退出0，六次add输入全部逐位一致，前向及输入梯度逐位一致，maxabs=0；日志 `/tmp/llmtuner-clone-prenorm-repo.log`。原无clone机制断言保留，候选仅在测试。
- clone在eager中建立梯度汇合边界，使norm直接/方差梯度先合并后传给原输入；增加一次张量复制且改变原eager舍入路径。未验证真实HF、FSDP、AC、Inductor、DTensor及设备内存代价，不能宣称生产修复或所有模式确定性。
- 下一行动：在真实Qwen无FSDP重放两种模式均给残差前norm输入建立此边界，比较全部参数梯度；保持已有原实现追踪作为对照，不将新实现梯度与旧路径混为恢复通过证据。

### 真实Qwen单步候选验证

真实Qwen重放验证归一化输入副本候选：

- `--replay-backward` 支持CLONE_PRENORM，在两侧每层input_layernorm/post_attention_layernorm注册测试专用输入clone hook；在deepcopy与compile前完成，不改生产HF实现。候选模式明确跳过与旧实现归约梯度的同路径对应检查，仍比较候选eager/AOT的全部梯度，并新增逐位断言。
- 使用同一真实step2 checkpoint和step3捕获输入，rank0/rank1均退出0，logits逐位相同、25个参数梯度全部逐位相同，差异列表均为空。命令为既有Q_BREAK重放加 `LLMTUNER_TEST_CLONE_PRENORM=1`；日志 `/tmp/llmtuner-clone-qwen-{0,1}.log`。原无clone的失败及逐位归约对应证据保留。
- 该结果是单步无分布式候选验证，不是原实现修复或最终恢复通过。尚需真实FSDP新轨迹的full/compiled resume/eager resume、AC及设备验证；内存复制代价与适用模型范围未确定。下一行动：将相同测试变换在assembly前施加于真实Trainer，运行Q-only细粒度FSDP的全新轨迹与严格恢复。

### 真实两rank候选严格恢复通过

真实Trainer细粒度FSDP候选的严格恢复验证：

- 诊断assembly配置在parallelism前给每层两处残差前norm添加输入clone hook，受CLONE_PRENORM测试开关控制；生产代码未改。新建轨迹目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-clone-trajectory-0ikepmob`，不覆盖原失败。
- 实际两rank CPU/Gloo、Q-only pattern/custom FSDP、attention hook owner、dynamic EMA。四步compiled(aot_eager) full、同模式resume及eager resume分别独立torchrun，均退出0。原pp_checkpoint_equivalence的恢复前完整状态和最终模型/optimizer/EMA/scheduler/trainer严格断言均保留；两rank pre-update state failures=0，恢复loss最大绝对差0。日志 `/tmp/llmtuner-clone-trajectory-{full,control,eager}.log`。
- 运行方式：各阶段 `LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_TRACE_PROJECTIONS=q_proj LLMTUNER_TEST_CLONE_PRENORM=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=<free-port> tests/integration_tests/compile_trajectory_diagnostics.py <full|resume> <fresh-root>`；full及compiled控制设置COMPILE_BACKEND=aot_eager，eager不设置。
- 候选闭合了该配置的严格恢复实验，未闭合生产缺陷：尚未选择正式HF适配机制、界定适用norm/模型、验证AC/native FSDP/Inductor与目标设备，也未量化复制代价。下一行动：审查HF适配入口并设计有明确范围的梯度汇合边界，避免无条件复制所有模型norm或依赖测试hook。

### 共享存储view边界候选

无额外数据复制的view边界候选：

- 生产接入审查定位到HF构造适配层：模型行为变换应在HF构建时对eager/compile一致施加，不能仅在apply_compile改变已保存数值路径。当前不做无范围的全模型norm hook替换。
- 在独立prenorm残差复现中，以x.view_as(x)替代x.clone，同样前向与输入梯度逐位一致，maxabs0；view共享存储，避免clone的数据副本，但会增加autograd节点，编译/DTensor契约仍需实测。
- 真实Qwen诊断新增测试专用VIEW_PRENORM选项，rank0真实checkpoint/输入重放退出0，logits与全部参数梯度逐位一致，日志 `/tmp/llmtuner-view-qwen-0.log`。该结果未覆盖rank1和真实FSDP，不能沿用clone候选的恢复通过结论。
- 下一行动：验证view候选rank1与两rank新轨迹严格恢复，并检查TP非连续输入和AC；若有效，优先采用不复制数据且限定已知纯norm的模型适配。生产代码未修改。

### view候选与full AC严格恢复

view候选真实FSDP与full AC严格恢复：

- rank1无FSDP真实输入重放退出0，logits/全部25参数梯度逐位一致，日志 `/tmp/llmtuner-view-qwen-1.log`。普通非连续转置张量view_as另验证共享storage、保持stride和平方和反向；此检查不代表DTensor/TP证据。
- 诊断assembly支持VIEW_PRENORM，在并行变换前为两处残差前norm建立共享存储view输入。两rank CPU/Gloo、Q-only custom FSDP/attention hooks/dynamic EMA，分别无AC和full AC新建轨迹，各执行compiled full、compiled resume、eager resume，六个torchrun均退出0。原始恢复前与最终完整状态逐位断言保留，loss差异0。
- 无AC目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-view-trajectory-e5d2j71p`，full AC目录同父路径 `llmtuner-view-ac-trajectory-ph8ad6ag`；日志 `/tmp/llmtuner-view-trajectory-{full,control,eager}.log` 与 `/tmp/llmtuner-view-ac-trajectory-{full,control,eager}.log`。沿用clone批次命令，将CLONE_PRENORM替换为VIEW_PRENORM，full AC再加 `LLMTUNER_TEST_AC=full`。
- 候选避免clone数据副本且覆盖重计算路径，但仍仅为测试适配；生产接入、native FSDP、TP/DTensor、Inductor/目标设备均未完成。下一行动：用可保持模块身份/FQN的HF归一化适配实现共享输入view边界，限定已知纯RMSNorm类并验证导出、参数身份和独立HF数值；不得仅按属性名称替换任意norm。

### 生产Qwen3适配与严格恢复

生产Qwen3残差归一化梯度汇合适配：

- `models/hf/model_setup.py::stabilize_residual_norm_gradients` 在HF构造阶段对精确Qwen3RMSNorm/Qwen3MoeRMSNorm类型的input_layernorm/post_attention_layernorm添加顶层可deepcopy、kwargs兼容的view输入pre-hook。只适配已知纯norm，保持原模块/Parameter身份、权重与FQN，重复调用不重复注册；HFTransformerModel在模型构造后调用。两模式一致应用，不依赖compile开关，不修改HF类或全局注册。
- 适配改变原eager浮点汇合顺序；旧checkpoint仍严格加载相同键，但旧实现训练轨迹不能宣称逐位延续。共享存储view避免clone副本，不承诺Inductor消除view后的全部设备确定性。
- 新增两类CPU回归：参数身份/键保持、幂等、原生HF前向及梯度参考、非连续输入/kwargs和deepcopy。全量pytest最终3187 passed、2 skipped，退出0，日志 `/tmp/llmtuner-production-norm-all-units.log`。
- 无候选VIEW/CLONE开关的真实两rank CPU/Gloo、custom Q-only FSDP/attention hooks/full AC/dynamic EMA，新轨迹compiled full、compiled resume、eager resume均退出0，完整状态严格断言通过、loss差异0。目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-production-norm-trajectory-8xmjq08y`，日志 `/tmp/llmtuner-production-norm-trajectory-{full,control,eager}.log`；运行方式同view/full AC批次但移除VIEW_PRENORM。
- 全部适用tracked/untracked文件pre-commit通过，明确排除受保护examples/native_alltoall.py；compileall通过。未新增生产文件，逐文件清单路径不变，本批重新审查model.py/model_setup.py的构造调用链。下一行动：生产native FSDP对照及TP/DTensor view布局检查，再扩展MoE；Inductor与目标设备仍待验证，整体目标未完成。

### 生产适配的并行组合证据

生产norm边界的native、MoE/EP与TP×FSDP复核：

- 两rank CPU/Gloo、full AC/dynamic EMA/Q-only细粒度分组：native FSDP使用projection自身hooks，custom MoE使用Qwen3MoE+EP2及attention hooks。两组各compiled full、compiled resume、eager resume均退出0，原始恢复前及最终完整状态逐位断言保持，loss差异0；未启用VIEW/CLONE测试变换。
- native目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-production-norm-native-mm_189j0`；MoE目录同父路径 `llmtuner-production-norm-moe-p08bl3d1`。日志 `/tmp/llmtuner-production-norm-{native,moe}-{full,control,eager}.log`。native沿用生产轨迹命令设FSDP_IMPLEMENTATION=native、TRACE_LOCAL_HOOKS=1；MoE设MOE=1、EP_SIZE=2，保留custom。
- 四rank `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=<free-port> tests/integration_tests/tp_fsdp_equivalence.py` 退出0，真实Qwen3 Trainer TP2×FSDP2四步与单进程参考比较，loss最大差4.768e-7、参数最大差7.000e-6，既有容差不变，failed ranks=0。日志 `/tmp/llmtuner-production-norm-tp-fsdp.log`；覆盖实际DTensor分片训练，不代表跨模式TP恢复或所有view layout。
- 本批未修改生产源码，仅补充生产适配的并行证据。下一行动：Inductor真实编译数值与恢复契约，以及其他仍开放的设备/生命周期待办；CPU/Gloo结论不推广到CUDA/NPU内核，整体目标仍未完成。

### Inductor同模式恢复及跨模式差异

CPU Inductor真实轨迹及独立原生HF数值对照：

- 两rank CPU/Gloo、custom Q-only FSDP/full AC/dynamic EMA，backend=inductor；compiled full及compiled resume均退出0，eager resume最终严格检查退出1，保留失败。恢复前完整状态无差异，step3参数相同；但logits已有rank0=1.7881393432617188e-7、rank1=1.4901161193847656e-7差异，25个归约前梯度键不同，更新后16键不同。这与此前仅反向重结合的aot_eager案例不同，生产view适配不承诺消除Inductor前向融合差异。
- 目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-production-norm-inductor-8ayuu02z`；日志 `/tmp/llmtuner-production-norm-inductor-{full,control,eager}.log`。运行方式同生产full AC轨迹，将COMPILE_BACKEND从aot_eager设为inductor；不启用候选变换。未放宽严格断言。
- `--isolated-backward` 支持环境选择backend，原生HF参考删除本项目norm hook，仅使用其原始CausalLM forward。补充直接compiled-vs-native logits及所有参数梯度比较，采用原有torch.testing float32默认容差，确认unique_graphs>0；`LLMTUNER_TEST_COMPILE_BACKEND=inductor TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --isolated-backward` 退出0，日志 `/tmp/llmtuner-production-inductor-direct-native.log`。梯度和logits非逐位一致，该计算正确性证据不替代跨模式严格恢复。
- 本批仅改诊断测试和文档，相关pre-commit通过。下一行动：核对Inductor浮点融合/重关联选项与实际生成代码，判断可否在保留编译机制下建立约定的跨模式数值契约；继续保留严格失败为开放项，不能将默认容差计算通过当作逐位恢复通过。

### Inductor归约算法差异

Inductor前向归约的独立复现：

- 本机PyTorch2.14已为use_fast_math=False、cpp.enable_unsafe_math_opt_flag=False、cpp.enable_floating_point_contract_flag=off；cpp_builder确认相应编译选项，不能通过重复关闭这些开关解释或消除当前差异。
- 新增 `--inductor-norm`：seed42、128×64输入，返回pow2/mean/rsqrt/norm，确认编译图且保持float32默认容差。实际退出0；pow2逐位一致，默认mean最大差2.384185791015625e-7，cpp.simdlen=1标量路径仍差3.5762786865234375e-7；最终输出差分别4.76837158203125e-7、7.152557373046875e-7。生成代码确认向量和标量路径不同归约，首次差异在mean，不足以判断rsqrt独立误差。
- 运行 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-norm`；日志 `/tmp/llmtuner-inductor-norm-repo.log`，生成代码 `/tmp/llmtuner-inductor-norm-probe.log`。不依赖HF/FSDP/checkpoint，关闭SIMD未恢复逐位一致；相关pre-commit通过。
- 下一行动：核对ATen与生成归约累加树及保持归约算子边界的可训练编译机制；不继续重复已关闭fast-math选项。生产未改，Inductor跨模式严格恢复失败仍开放，该实验未证明整模型全部差异来源。

### ATen归约边界的前后向隔离

可训练ATen mean边界的隔离实验：

- 新增 `--inductor-mean-boundary` 诊断，用测试命名空间custom_op仅保留ATen mean末维归约，提供fake及expand/末维长度的autograd公式，其余pow/rsqrt/乘法继续Inductor。无生产注册或全局lowering修改。
- seed42、128×64实际运行退出0，unique_graphs=1；平方、mean、rsqrt及最终norm四份前向结果全部逐位一致，新增严格断言。输入梯度仍非逐位一致，最大绝对差9.5367431640625e-7，保持float32默认容差断言并通过。归约边界在该前向实验有效，但反向存在独立融合/运算顺序差异，不能将保留mean推广成完整训练修复。
- 命令 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-mean-boundary`；日志 `/tmp/llmtuner-mean-boundary-repo.log`。相关pre-commit通过。ATen完整SumKernel实现未包含在本地安装头文件中，不声称已核对其完整累加树。
- 下一行动：用同一输入及上游梯度隔离norm backward公式，区分custom mean backward与其外部融合；在完整训练证据前不接入生产opaque算子。Inductor跨模式严格恢复失败保留，整体目标未完成。

### norm反向归约的独立边界实验

固定上游梯度的norm backward归约隔离：

- 新增 `--inductor-norm-backward`，seed42/128×64固定输入、随机上游g及同一eager计算的scale，逐段返回direct、sum(g*x)、scale^3、variance、mean_grad、square_grad、indirect和最终梯度。手工eager公式与原生autograd逐位一致（无loss反向），确认参考。
- Inductor首次在sum(g*x)出现差异2.1457672119140625e-6，direct/scale^3/square_grad逐位一致，最终梯度差2.384185791015625e-7。测试命名空间custom_op仅保留该sum为ATen边界，其余公式仍编译，八项返回结果全部逐位一致。custom sum是固定张量公式的前向隔离探针，未注册高阶梯度，也不用于训练生产。
- 命令 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-norm-backward`；实际退出0，原float32默认容差全部通过，日志 `/tmp/llmtuner-norm-backward-boundary.log`。
- 已将前后向归约差异分别隔离；不代表整模型所有算子均可逐位一致。下一行动：评估完整norm前后向opaque边界的可训练实现及独立梯度契约，然后决定是否适合生产，不能通过全面回退eager或放宽恢复断言关闭问题。生产未改，Inductor跨模式严格恢复仍失败。

### 可训练opaque norm独立契约

可训练opaque RMSNorm的一阶独立契约：

- 新增 `--inductor-opaque-norm` 测试专用custom_op前后向，前向保留原ATen pow/mean/rsqrt/weight顺序，backward使用已独立核对的公式并作为opaque边界，提供fake和一阶autograd注册。外层保持Inductor fullgraph编译，没有修改生产norm或注册生产算子。
- seed42/128×64、随机权重和固定随机上游梯度：输出、输入梯度、权重梯度全部逐位一致（误差0）；2×4 float64有限差分gradcheck通过。实际仓库诊断命令退出0，日志 `/tmp/llmtuner-opaque-norm-repo.log`，命令 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`。
- 限制：仅一阶普通CPU张量；不支持高阶autograd声明，未验证混合精度、DTensor、AC或整模型所有融合算子。保留ATen执行有性能代价，尚无生产采用决定。
- 下一行动：只在诊断中将真实Qwen纯RMSNorm替换成等式边界，先比较整模型前向以判断还有无其他差异源；若仍不同，不把norm单点修复当成跨模式恢复解决方案。Inductor严格跨模式失败仍开放，生产本批未改。

### opaque norm真实Qwen前后向

真实Qwen的opaque norm与独立HF参考：

- 新增 `--inductor-opaque-qwen`，仅给诊断实例的精确Qwen3RMSNorm绑定opaque forward，原生HF参考在变换前deepcopy并删除本项目view hook，避免参考共用候选算子；生产与全局HF类未修改。
- 首次真实attention反向暴露非连续Q/K norm的fake stride错误，原始日志 `/tmp/llmtuner-opaque-qwen-native-reference.log` 保留退出1。测试算子反向显式contiguous并使fake相同，未放宽metadata断言；修正后退出0，日志 `/tmp/llmtuner-opaque-qwen-contiguous.log`。
- `LLMTUNER_TEST_COMPILE_BACKEND=inductor TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-qwen` 实际编译图确认，候选eager/Inductor logits和全部参数梯度逐位一致。两者分别及直接compiled对原生HF logits/梯度均通过既有float32默认容差；原生参考梯度非逐位相同，原因包含norm梯度边界与残差汇合差异，不能宣称原HF轨迹逐位保持。
- 该证据覆盖随机单batch的完整Qwen前后向，不是多步FSDP严格恢复；dtype断言限定float32，DTensor/混合精度/高阶及性能未实现。下一行动：opcheck包含非连续输入的fake/alias/autograd契约，并将候选限制在测试中运行真实Inductor恢复，不凭单batch通过接入生产。原生产跨模式严格失败仍开放。

### opaque候选真实Inductor恢复通过

opaque norm算子契约与真实Inductor严格恢复：

- `LLMTUNER_TEST_NORM_OPCHECK=1 ... --inductor-opaque-norm` 对连续及转置非连续3D输入，前向/反向算子各opcheck的schema、autograd注册、faketensor、aot_dispatch_dynamic全部SUCCESS；float64 gradcheck及float32逐位比较仍通过，退出0，日志 `/tmp/llmtuner-opaque-opcheck.log`。不意味着反向算子支持高阶梯度。
- 新增测试专用OPAQUE_NORM装配路径（实际变量 `LLMTUNER_TEST_OPAQUE_NORM=1`），在parallelism前为诊断实例绑定候选norm，保持生产模块与全局HF类不改。两rank CPU/Gloo、Q-only custom FSDP/attention hooks/full AC/dynamic EMA，Inductor full、Inductor resume、eager resume均退出0，原完整状态逐位断言通过、两rank pre-update failures=0、loss差异0。
- 目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-opaque-norm-inductor-lcz0zkym`；日志 `/tmp/llmtuner-opaque-norm-inductor-{full,control,eager}.log`。沿用生产Inductor轨迹命令加 `LLMTUNER_TEST_OPAQUE_NORM=1`，各阶段一致应用；原无opaque的跨模式失败记录保留。
- 此结果闭合该测试候选的多步恢复，不代表生产路径已修复。候选仅float32/普通CPU、一阶；保留ATen边界带来性能与编译融合取舍，混合精度/DTensor/device仍未验收。下一行动：核对能否以明确配置与设备能力契约提供可维护的生产归一化机制，先处理混合dtype及任意前导维/冻结权重的autograd语义，避免将测试实现直接复制到生产。

### norm候选维度与冻结参数契约

候选norm一维权重梯度缺陷与冻结契约：

- 审查发现测试候选对一维输入使用sum(tuple(range(ndim-1)))，空tuple在PyTorch表示全维归约，会错误返回标量权重梯度。已改为仅ndim>1归约前导维，一维直接保留逐元素g*norm；生产未使用该候选，不影响已有生产状态。
- 新增 `LLMTUNER_TEST_NORM_SHAPES=1`，覆盖(4,)、(0,4)、(3,4)、(2,3,4)，各输入/权重都训练、仅输入训练、仅权重训练，共12配置。前向与原生公式、所需输入/权重梯度逐位一致；冻结项grad保持None。原Inductor逐位验证及float64 gradcheck仍通过，实际退出0，日志 `/tmp/llmtuner-opaque-shapes.log`。
- 命令 `LLMTUNER_TEST_NORM_SHAPES=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`；相关pre-commit通过。下一行动：混合dtype的HF精确cast/梯度cast契约；候选未达生产适用条件，Inductor生产跨模式严格失败仍开放。

### norm候选混合dtype契约

opaque候选低精度cast与权重dtype契约：

- 测试候选对fp16/bf16输入采用FP32归一化，结果回转输入dtype后乘权重；backward先按乘法输入dtype回转，再转FP32计算norm梯度，最后回转输入梯度，权重梯度归约后转权重dtype。fake输出dtype按输入/权重promote_types，不再总是假设输入dtype。
- `LLMTUNER_TEST_NORM_DTYPES=1` 覆盖fp16/bf16/fp32输入×三种权重dtype，共9组合，与独立原生Qwen3RMSNorm比较输出dtype/逐位前向/逐位输入及权重梯度，全部通过，退出0。日志 `/tmp/llmtuner-opaque-dtypes.log`。这9项为普通CPU eager契约，不是低精度Inductor或设备证据。
- 修改后重新运行SHAPES=1及OPCHECK=1，12个维度/冻结配置、连续非连续算子检查、原float32 Inductor逐位对照及float64数学公式gradcheck仍通过，退出0，日志 `/tmp/llmtuner-opaque-contracts-after-dtype.log`。float64 gradcheck用于数学公式，不声明HF强制FP32的float64语义一致。
- 相关pre-commit通过；生产未改。下一行动：真实混合dtype Inductor输出/fake验证及设备契约评估，然后明确候选性能与支持范围再决定生产接入；原生产Inductor跨模式严格失败仍开放。

### norm候选实际混合dtype编译

九种dtype组合的实际Inductor契约：

- 在既有DTYPES矩阵增加 `LLMTUNER_TEST_NORM_COMPILE_DTYPES=1`，每组独立reset Dynamo计数并以Inductor/fullgraph调用候选，反向后断言unique_graphs>0，避免只验证eager。
- fp16/bf16/fp32输入×三种权重dtype共9组合均退出前完成逐位输出/输入梯度/权重梯度与原生Qwen3RMSNorm比较，输出dtype一致；整次进程退出0，原独立float32逐位与float64 gradcheck仍通过。日志 `/tmp/llmtuner-opaque-compiled-dtypes.log`。
- 命令 `LLMTUNER_TEST_NORM_DTYPES=1 LLMTUNER_TEST_NORM_COMPILE_DTYPES=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`。这是普通CPU张量编译证据，不覆盖CUDA/NPU或FSDP mixed precision。生产未改。
- 下一行动：明确DTensor的自定义算子分发/placement契约与性能取舍，再决定生产可维护实现；不能将CPU dtype矩阵推广为全设备机制完成。生产Inductor跨模式严格失败仍开放。

### 候选DTensor分发缺口

候选自定义norm的真实DTensor分发缺口：

- 新增 `LLMTUNER_TEST_NORM_DTENSOR=1` 两rank诊断，输入4×8采用Shard(0)，权重8采用Replicate，真实CPU/Gloo调用候选norm。两个rank均报NotImplementedError：Operator llmtuner_diagnostic.norm.default does not have a sharding strategy registered。用all_gather_object确认一致错误后正常清理，探针退出0；此退出0表示预期缺口被确认，不是能力通过。
- 命令 `LLMTUNER_TEST_NORM_DTENSOR=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=<free-port> tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`；日志 `/tmp/llmtuner-norm-dtensor-probe.log`。
- 核对torch.distributed.tensor.experimental.register_sharding接口可为自定义算子定义output/input placement策略。前导维分片可局部归一化，hidden维分片不能直接局部执行；反向复制权重的梯度需声明Partial(sum)，不能误标Replicate导致丢失归约。尚未注册策略或宣称实现。
- 下一行动：在测试命名空间定义前导维分片及复制策略，并以独立dense输入/权重梯度核对真实两rank反向和partial归约；生产本批未改，Inductor生产跨模式问题与设备验证仍开放。

### 候选DTensor布局数值验证

候选DTensor前后向布局策略及hidden重分布：

- 测试命名空间注册前向Replicate与前导维Shard策略，权重保持Replicate；反向input gradient同分片，前导维分片的weight gradient标Partial(sum)。没有hidden维局部策略，允许DTensor按复制策略重分布后执行。
- 真实两rank CPU/Gloo分别测试复制、Shard(0)序列、Shard(1)hidden输入；输出和两类梯度full_tensor与独立dense公式通过原float32默认容差。序列分片断言权重梯度Partial，hidden分片断言输出Replicate，确认未错误局部归一化。全部rank完成并退出0，日志 `/tmp/llmtuner-norm-dtensor-hidden.log`；原无策略失败日志保留。
- 命令沿用NORM_DTENSOR=1探针，当前语义变为布局数值验证。注册仅覆盖一维mesh测试，未验证多轴、编译DTensor或设备，不声明hidden高效分布式归约；hidden路径需要实际all-gather。
- 下一行动：核对实际TP norm的本地/DTensor输入契约与多轴策略，再评估生产机制；当前候选仍测试限定，生产Inductor严格跨模式失败仍开放。

### norm候选二维mesh验证

四rank二维mesh候选布局验证：

- NORM_DTENSOR探针新增 `LLMTUNER_TEST_NORM_MESH_2D=1`，真实四rank CPU/Gloo构造2×2 mesh，覆盖(R,R)、(S0,R)、(S0,S0)。完整输出及输入/权重梯度与独立dense参考通过float32默认容差，逐轴权重梯度Replicate/Partial(sum)断言通过，进程退出0。
- 命令 `LLMTUNER_TEST_NORM_DTENSOR=1 LLMTUNER_TEST_NORM_MESH_2D=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=<free-port> tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`；日志 `/tmp/llmtuner-norm-dtensor-2d.log`。full_tensor物化S0/S0时PyTorch警告顺序all-gather成本，保留警告，不将数值验证当通信性能优化。
- 核对tensor_parallel.py的apply_tp明确拒绝额外SequenceParallel变换：norm消费已有sequence shard。候选DTensor策略不应额外复制该归约职责；实际生产自定义TP路径仍需独立候选集成，不能仅凭模块说明宣称运行输入类型已验证。
- 下一行动：编译DTensor候选前后向并检查Partial输出，随后整理生产采用的性能/生命周期边界；本批仅诊断与文档，生产Inductor跨模式严格失败仍开放。

### 候选Inductor DTensor验证

候选Inductor DTensor前后向与二维mesh：

- NORM_DTENSOR增加 `LLMTUNER_TEST_NORM_COMPILE_DTENSOR=1`，各layout独立reset计数，以Inductor/fullgraph执行，backward后断言unique_graphs>0；其余dense参考/placements/Partial断言保持。
- 两rank一维mesh复制/S0/S1及四rank二维mesh(R,R)/(S0,R)/(S0,S0)全部完成，输出和两类梯度通过既有float32默认容差；权重Partial(sum)及hidden重分布断言通过，两进程组均退出0。日志 `/tmp/llmtuner-norm-compiled-dtensor.log`、`/tmp/llmtuner-norm-compiled-dtensor-2d.log`；PyTorch opaque值弃用及顺序all-gather警告保留。
- 运行沿用NORM_DTENSOR探针，添加COMPILE_DTENSOR=1及PCH=0；二维再加MESH_2D=1并使用四rank。没有更改生产算子/注册，CPU编译证据不能推广目标设备或完整TP Trainer。
- 下一行动：整理可维护生产归一化实现的明确配置与注册生命周期，评估是否保留ATen边界及其性能代价；候选CPU单/多轴契约已具证据，避免重复相同矩阵。生产Inductor跨模式严格失败与硬件项仍开放。

### norm边界CPU性能取舍

候选opaque norm生产采用的CPU性能代价：

- 增加NORM_BENCHMARK=1诊断，以torch.utils.benchmark单线程、每配置3次预热、blocked_autorange至少0.3秒，固定上游梯度计算forward+backward并清空grad，排除编译。实际退出0，日志 `/tmp/llmtuner-norm-benchmark.log`。
- 输入128×64：原生eager45.08µs、融合Inductor36.64µs、opaque Inductor84.74µs；512×1024：分别1474.71/732.76/1903.81µs。opaque约融合路径2.31/2.60倍。仅本机CPU局部算子结果，未测峰值内存或完整Trainer吞吐，不推广GPU/NPU。
- 命令 `LLMTUNER_TEST_NORM_BENCHMARK=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`；修正benchmark闭包显式绑定循环变量，避免后续复用时捕获漂移。
- 生产设计决定：不得无条件用opaque替换当前已验证view适配；需要显式的模型数值边界选项，eager与compile同选项运行，记录不兼容轨迹的限制。下一行动：在模型配置定义保留ATen norm边界的明确选项和校验，再做可维护生产实现/注册生命周期；性能证据不作为放弃严格恢复的理由。整体目标未完成。


### TP checkpoint修复当前状态（2026-10-15）

新增生产TP全局布局适配；当前HEAD 500f166加未提交修复，TP2及TP2×custom
FSDP2真实Trainer正常保存恢复逐位通过，模型/optimizer/EMA均使用全局tensor；
metadata gate_proj shape为(128,64)。packed MoE及gate/up绑定DCP与后续Adam
更新也通过。本地规划错误协调的两rank CPU/Gloo八配置及健康重试已通过；
故障模型不含进程死亡、组注册损坏及collective内部失败。实现状态仍为部分修复：
TP×EP中间布局shape缺陷已由四rank CPU适配器精确往返修复验证；
真实Trainer world4/TP2/EP2/custom FSDP、float32/dynamic EMA的独立
DCP full/resume也通过，更新前及最终完整状态逐位一致、后续loss差0
（TP-EP-CKPT-01）。跨compile新增实测失败：eager保存→aot_eager恢复
更新前完整状态通过，但首次编译SDPA的序列128与mask64形状冲突
（TP-COMPILE-ATTN-01，已修复）；后续专家动态guard与resolver图边界
导致的零梯度也已修复。生产aot_eager恢复重新运行完整状态逐位一致、
后续loss差0（TP-COMPILE-NUMERIC-01）。compiled保存→eager恢复亦逐位
通过；custom保存→native compiled恢复通过。上述仅CPU/aot_eager配置，
Inductor、目标设备和拓扑变化仍待验收。
详细命令、结果和下一动作见file_review续接批次。下节为原失败证据。

### P0：TP checkpoint 布局缺失（2026-10-15，原始失败证据）

真实Trainer PP1/TP2、CPU/Gloo float32、dynamic EMA的独立full/resume恢复失败。
更新前模型、优化器与EMA状态已不同，后续loss偏离；独立读取DCP确认不同rank
同FQN本地切片被当作复制tensor去重，仅保存一个rank值及本地shape。
现有TP数值训练证据不证明保存恢复支持；plain TP及TP外层布局均需适配，
不能外推已有EP/custom FSDP恢复结果。复现、原始状态、日志、根因和修复验收
见file_review的P0批次。原始失败已由上述正常CPU路径修复证据覆盖，组合与设备范围仍未完整验收。

### TP 同一 shared MLP gate/up 绑定（2026-10-15）

生产 TP 现支持同一 shared MLP 的 gate/up 共用 Parameter 或 Linear：按原参数
身份只切片一次，保留绑定、冻结状态及 feature metadata。真实两rank CPU/Gloo
绑定用例（含FP64 router）输出、输入/参数梯度、重复应用及SGD更新通过原容差。
外部别名、gate/down不同轴、跨block专家共享仍拒绝；其别名保留迁移及设备、
FSDP/EP/compile/checkpoint组合验证尚未完成。证据见file_review对应批次。

### TP 跨 MoE 参数别名边界（2026-10-15）

`apply_tp`在变换前拒绝待切片专家参数的多个注册路径，包括跨block共用
experts模块、共用packed权重和外部别名，避免重复切片或静默解除共享。
三类CPU回归确认失败不改变参数身份、数值或block类；保留共享关系的TP
实现仍未完成，拒绝配置不构成该能力已迁移。普通两rank MoE等价复跑通过，
不能外推共享专家布局通过。


### Custom FSDP 跨 AMSGrad 恢复补证（2026-10-15）

`fine_grained_checkpoint_equivalence.py`现覆盖32配置：BF16/FP16、reshard、
full AC、源/目标AMSGrad开关。两rank CPU/Gloo真实DCP、relocated hooks、
FP32 master/reduction、AdamW/scheduler/EMA；新建运行时恢复后更新前、
下一步输出/完整梯度及更新后状态逐位一致，退出0。
不包含PP/EP、compile、异步transport、拓扑变化、数据/RNG或目标设备。


### Kimi dense KDA 的 Trainer 恢复补充（2026-10-15）

| 配置 | 实现状态 | 验证层级 | 执行结果 | 证据 |
| --- | --- | --- | --- | --- |
| Kimi dense KDA，FSDP custom2，TP/PP/CP/EP1，FP32，AC/compile关闭，EMA关闭 | 部分（完整Kimi迁移未完成） | CPU/Gloo数值 | 通过 | pp_checkpoint_equivalence.py，KIMI=1；恢复前后完整状态、后续loss精确一致 |
| 同上，EMA=dynamic | 部分（完整Kimi迁移未完成） | CPU/Gloo数值 | 通过 | 相同脚本，动态EMA状态与最终状态精确一致 |

运行命令、版本、环境、loss和证据目录见 `file_review.md` 的
“Kimi 真实 Trainer/custom FSDP 完整恢复”。此处不外推 MoE、hybrid KCP、
其他并行组合、编译路径或目标设备正确性。


Kimi 本地 MoE 补充（2026-10-15）：`pp_checkpoint_equivalence.py` 设置
`LLMTUNER_TEST_KIMI=1 LLMTUNER_TEST_KIMI_MOE=1`，custom FSDP2、
TP/PP/CP/EP1、CPU/Gloo/FP32、AC/compile关闭、动态EMA，E4/top2、
group2/retained1、shared1，两阶段完整恢复严格通过、后续loss差0。
实现仍为部分（EP转换等未完成），验证层级为CPU数值，执行结果通过。
详细配置、版本和日志摘要见 `file_review.md` 对应批次；不能外推EP能力。

### Kimi attention residual 参数归属修复（2026-10-15）

| 配置 | 实现状态 | 验证层级 | 执行结果 | 证据 |
| --- | --- | --- | --- | --- |
| dense Kimi 两层KDA，attn_res_block_size1，custom FSDP2，TP/PP/CP/EP1，FP32，AC/compile关闭，dynamic EMA | 部分（Kimi整体迁移未完成） | CPU/Gloo数值 | 通过 | ATTNRES-FSDP-01；KIMI=1、ATTN_RES=1、EMA=dynamic 的真实 Trainer 独立恢复；两rank更新前状态零失败，后续loss差0，完整最终状态严格一致 |

HF子模块遍历现在包含注册父模块，FSDP可发现额外输出残差参数；PP分割
保留承载标准部件的结构父模块。上述用例不包含PP历史跨阶段传递或EP转换，
也不关闭独立生命周期参考的多seed梯度误差待定位项。
来源、失败与修复记录见 `file_review.md` 的 ATTNRES-FSDP-01。

Kimi attention residual CPU 生命周期参考补充：ATTNRES-NUMERIC-01 已用
seed0..31、block_size1/2 验证固定上游FP32聚合顺序的独立history列表参考，
64例logits逐位一致、全部参数梯度原2e-6容差通过；逐token数学公式另有
16例输入/参数梯度1e-6验证。原逐候选生命周期参考的7例失败保留在台账，
此次关闭的是限定配置的参考待定位项；硬件、其他dtype与并行组合仍未验收。

### Kimi EP 生产路径补充（2026-10-15）

| 配置 | 实现状态 | 验证层级 | 执行结果 | 证据 |
| --- | --- | --- | --- | --- |
| Kimi E4/top2/group2/shared1，EP2，其余轴1，FP32，无FSDP/AC/compile/EMA | 部分 | CPU/Gloo数值 | 通过 | kimi_ep_equivalence.py；普通/latent × SiLU/Situ × 不均匀/空rank共8场景；独立完整专家输出与梯度对照 |
| Kimi两层KDA+MoE，custom FSDP2/EP2，其余轴1，FP32，dynamic EMA，AC/compile关闭 | 部分 | CPU/Gloo数值 | 通过 | pp_checkpoint_equivalence.py；KIMI=1、KIMI_MOE=1、EP_SIZE=2，真实连续/独立恢复严格一致 |
| 同上，latent width32+RMSNorm | 部分 | CPU/Gloo数值 | 通过 | 再加KIMI_LATENT=1；完整恢复严格一致、后续loss差0 |

生产转换位于 `parallel/expert_parallel/kimi_conversion.py`，复用native
alltoall；模型宽度路由，latent空间专家计算，combine后norm/up，shared保留
原输入。correction bias保留原Parameter身份且不自动sign更新。
Native checkpoint采用转换后grouped FQN；原HF逐专家格式的双向映射、跨EP
开关恢复、Kimi TP/其他dispatcher/quantile适配及设备验证仍开放。
来源、数值容差、失败和完整命令见 `file_review.md` 的 KIMI-EP-01。

### Kimi EP 的 HF 权重映射（2026-10-15）

KIMI-HF-STATE-01 已实现 full grouped ↔ HF individual expert 权重、router、
correction bias与latent键映射，显式由adapter的kimi_expert_parallel开关选择。
EP2/CPU/Gloo/FP32、其余轴1的8个通信场景新增完整权重gather→HF→独立
模型slice验证，键、权重及恢复后输出逐位一致；无FSDP/AC/compile/EMA。
实现仍部分：Trainer公开HF初始化保持拒绝，待真实存储加载验收；native跨EP
开关的optimizer/EMA迁移、FSDP下HF存储及设备验证未关闭。

### Kimi Trainer 初始 HF 加载接通（2026-10-15）

KIMI-HF-INIT-01 已解除 Kimi 专属的 EP 初始 HF 加载限制，其他 fused MoE
家族仍需自己的映射。真实磁盘safetensors → meta → custom FSDP2/EP2 →
加载/切片 → 前向 → 一次优化器更新在CPU/Gloo/FP32通过；latent32+norm、
shared1、group2/top2，其余轴1，AC/compile/EMA关闭。全权重逐位一致，
源模型前向对照1e-6。补修MoE非持久token计数器meta重建，未知buffer仍拒绝。
命令与失败证据见file_review.md；不外推其他组合或native跨格式训练状态恢复。

同场景FSDP实现改native后同样通过全权重、前向及一次更新；日志与命令见
KIMI-HF-INIT-01最终验收补充。native/custom均为CPU数值证据。

### Kimi FSDP/EP 实际 HF 导出存储（2026-10-15）

KIMI-HF-EXPORT-01：native/custom FSDP2+EP2、其他轴1，CPU/Gloo/FP32，
latent32+norm/shared1/group2/top2，AC/compile关闭。真实Trainer训练后调用
公开save(last_step=True)，读取发布index与safetensors，全部权重逐位一致；
原始Kimi严格加载后前向1e-6。完整训练checkpoint与同step model-export
分开存放，运行状态未改变且能继续更新；native场景包含dynamic EMA。
详情和适用范围见file_review.md。该证据关闭限定配置的HF导出存储缺口，
保留native跨EP开关训练状态迁移及目标设备缺口。

KIMI-EP-OPT-01 已增加flat optimizer/EMA完整张量的逐专家↔grouped转换，
CPU真实AdamW/AMSGrad下一次更新逐位一致；step/groups不一致明确拒绝，
scalar counter不跨runtime共享。生产DCP模板与load planner尚未接入，
实现状态仍部分；Trainer跨EP开关恢复未验收。该转换采用HF专家FQN，
不代表HF safetensors保存optimizer。证据见file_review.md。

KIMI-EP-DCP-01 接入native DCP源专家布局识别和读取模板，逐专家↔grouped
转换后依次恢复模型、optimizer、其他状态、RNG。单进程CPU真实DCP文件的
两方向AdamW/AMSGrad下一次更新及EMA精确状态验证通过；多rank Trainer跨EP
开关仍未验收。跨布局每rank分配完整CPU权重/状态，同布局保留sharded路径；
此实现不提供低峰值主机内存保证。详见file_review.md。

KIMI-EP-CROSS-01：CPU/Gloo/FP32 custom FSDP2，Kimi latent/shared专家，
EP1保存→EP2独立Trainer恢复在显式exclude_from_loading=[rng_state]时通过。
更新前model/optimizer/EMA/trainer/scheduler完整状态精确一致，后续两步loss
1e-6通过；AC/compile关闭、dynamic EMA、其余轴1。默认精确RNG跨拓扑
恢复仍拒绝，且校验提前到状态应用前。反向和目标设备仍待验收。

KIMI-EP-CROSS-02：EP2→EP1反向恢复同样在两rank CPU/Gloo、FP32、custom
FSDP2、Kimi latent/shared、dynamic EMA、其余轴1、AC/compile关闭时通过。
显式排除rng_state；更新前完整模型/optimizer/EMA/trainer/scheduler精确一致，
后续两步loss1e-6通过。独立源model快照按两rank专家轴拼接，该参考显式
限制world_size=2，不外推多轴rank顺序。详见file_review.md。


### HCA CP × custom FSDP 的新增实测行

| 配置 | 实现/验证 | 结果与边界 |
| --- | --- | --- |
| DeepSeek V4 全HCA，Ulysses CP2 × dp_shard2，TP/PP/EP1，custom FSDP，FP32 CPU/Gloo，动态EMA，AC/compile关 | 部分compressor能力已实现；CPU真实Trainer训练及恢复 | 四进程full/resume通过，恢复前状态及最终状态通过，后续loss差0；脚本 `hca_trainer_checkpoint.py`，命令/版本见file_review的COMPRESSOR-CP-FSDP-01。未证明CSA、其他组合或目标设备 |


| 新增配置 | 实现状态/验证层级 | 结果 |
| --- | --- | --- |
| DeepSeek V4 HCA+CSA两层，CP2/DP1/TP/PP/EP1，CPU/Gloo FP32，动态EMA，AC/compile关 | 该子集已实现；CPU真实wrapper独立完整HF参考及Trainer恢复 | logits/梯度/更新通过；full/resume更新前完整状态与最终状态通过，后续loss差0；见COMPRESSOR-CP-MIXED-01。未证明sliding/padding/CSA×FSDP/目标设备 |

### 三类 compressor 层 × custom FSDP 的新增实测行

| 配置 | 实现与验证 | 结果与边界 |
| --- | --- | --- |
| DeepSeek V4 sliding/HCA/CSA三层，CP2×dp_shard2/custom FSDP，TP/PP/EP1，CPU/Gloo FP32，动态EMA，AC/compile关 | 该子集生产入口已实现；CPU真实Trainer训练及独立进程恢复 | full/resume均exit0；四rank更新前状态failures0，后续loss差0、最终训练状态通过。命令/版本见file_review的COMPRESSOR-CP-THREE-FSDP-01；未独立验RNG payload，不证明padding/其他轴/目标设备 |

### Domino 标准局部专家故障协议

| 配置与机制 | 实现状态 | 验证证据与剩余 |
| --- | --- | --- |
| EP2，标准GroupedExperts/内置激活、普通Tensor且无hook/compile，局部专家slice0或slice1异常 | 部分实现：dispatch前统一资格、计算后错误协调、已发起通信收尾 | DOMINO-LOCAL-EXPERT-01记录CPU/Gloo双方异常后barrier与同实例前后向重试；正常独立参考仍覆盖。CUDA仅stream模拟；含专家FSDP hook、跨组生命周期及目标设备/性能未验收，DOMINO-FAILURE-01整体开放 |

### Domino 错误域与外层 FSDP 恢复

| 配置 | 实现与验证 | 结果与边界 |
| --- | --- | --- |
| dp_shard4、EP2/efsdp2、TP/PP/CP1；外层native/custom FSDP，Domino标准局部专家 | 生产装配用stage内loss group传播准备/资格/计算错误；CPU/Gloo FP32 | 两种FSDP均四rank异常退出、barrier、显式root reset、同实例独立参考loss/router/专家梯度通过，tol1e-4；见DOMINO-ERROR-DOMAIN-01。不证明直接专家hook、完整Trainer、其他轴或CUDA故障恢复 |

### Domino 直接专家 FSDP hook 的计算异常

| 配置 | 实现与验证 | 结果与边界 |
| --- | --- | --- |
| dp_shard4/EP2/efsdp2，pattern直接选inner_experts，native或custom FSDP，TP/PP/CP1 | 部分实现：已知单模块hook外保留通信生命周期，计算体记录原异常，posthook后完整域协调 | CPU/Gloo FP32两实现均reset前专家state IDLE，四rank异常退出/barrier/root reset，同实例独立HF参考loss及梯度通过；见DOMINO-DIRECT-FSDP-01。grouped/relocated/compile/训练统计/目标设备仍未验收 |

### Domino grouped/relocated FSDP 错误生命周期

| 配置 | 实现与验证 | 结果与边界 |
| --- | --- | --- |
| dp_shard4/EP2/efsdp2，共享两层专家，pattern合并state，native/custom | grouped MultiHandle及state callback核对后启用计算异常记录 | CPU/Gloo FP32四rank异常传播、已运行/未运行集合、root reset、同实例独立HF参考共享梯度通过；见DOMINO-GROUPED-RELOCATED-01 |
| 同拓扑，独立或共享专家state，custom hook_module迁至根 | 多context未完成forward显式root reset | 两配置均真实context数>=2，reset清通信记录、同实例数值重试通过；训练统计/完整Trainer/compile/其他轴/目标设备仍开放，DOMINO-FAILURE-01未整体验收 |

### Domino Trainer 更新提交后故障恢复

| 配置 | 验证 | 边界 |
| --- | --- | --- |
| CPU/Gloo FP32，dp_shard4/EP2/efsdp2，Domino/custom FSDP，TP/PP/CP1，动态EMA，AC/compile关 | full与注入更新后故障的resume均exit0；新runtime更新前状态含CPU RNG零差异，后续loss差0及最终state通过；见DOMINO-TRAINER-FAILED-COMMIT-01 | 恢复成功checkpoint，非原地通信恢复；未验设备、quantile/其他组合或完整train()异常循环 |


### Fixed router：无 gate 主损失梯度与 FSDP

FIXED-ROUTER-ENTRY-01 记录公开转换API在模型修改前复用配置校验，拒绝fixed
与quantile冲突。CPU/Gloo EP2 的alltoall/Domino×AC×四种token/padding场景
现扩展aux系数0/0.03共32场景，验证无aux时gate.grad=None、SGD保持gate不变。
另四rank DP shard4/EP2/efsdp2、native/custom FSDP各一次真实backward，
独立HF专家算式的loss/专家梯度1e-4检查通过，router无梯度且mesh归属正确。
该历史证据不覆盖TP复制梯度、设备、FSDP多步或恢复；EP1/TP1已启用本地固定路由。
批次108补齐EP1/TP2完整Trainer的CPU恢复证据并移除原公开配置限制；目标设备仍待验收。


Fixed router EP1 的完整 Trainer 恢复证据见 FIXED-ROUTER-TRAINER-RESUME-01：
两rank CPU/Gloo、DP shard2、native/custom FSDP、dynamic EMA，分别四步连续
训练与第2步后新进程恢复；更新前模型/AdamW/scheduler/trainer/EMA逐位匹配，
后两步loss差异0，最终状态检查通过。此证据不替代RNG/data独立状态快照，
也不覆盖EP1/TP>1、设备或编译组合。

### 恢复源一致性检查后的 PP2 回归

| 配置 | 验证结果 | 边界 |
| --- | --- | --- |
| CPU/Gloo FP32，Qwen3，PP2/1F1B/4微批，TP/CP/EP1，无 AC/compile/EMA | 新恢复源与状态类别比较开启；真实 Trainer 连续4步及 step2 新进程恢复均通过，更新前状态零失败，后续 loss 差0，最终训练状态一致 | 见 `docs/file_review.md` 的 CHECKPOINT-LOAD-CONSENSUS-PP-01；不代表目标设备、远程存储或其他组合通过 |

### Domino 本地计算中断传播补充

`DOMINO-EXPERT-INTERRUPT-01`：CPU/Gloo、EP2、TP/PP/CP1、FP32，标准通信无关
GroupedExperts 计算体，两 slice 和两失败 rank 分别注入 ValueError、KeyboardInterrupt、
SystemExit，12 项错误协调及同 runtime 后续 forward/backward 通过。原异常对象保留，
已启动传输执行清理；清理自身中断的继续等待由 CPU 单测覆盖。CUDA 流仅有模拟测试，
本项不证明设备通信故障恢复、hook 失败或其他并行组合；详见 `docs/file_review.md`。

### PP 与 TP/EP 范数归约组合证据

`GRAD-NORM-PP-COMBINATION-REVIEW-04`：四进程 CPU/Gloo、float64 显式梯度，
PP2×TP2 和 PP2×EP2 的复制 dense 参数加物理分片，在 L1/L2/L∞ 下符合独立
完整向量范数与裁剪参考（rtol1e-14/atol0）。单分片 NaN 在所有 rank 的裁剪前
拒绝；原两进程低精度、DTensor、空 stage 场景亦通过。脚本为
`tests/integration_tests/parallel_grad_norm_equivalence.py`，分别以 nproc4/2 运行。
此证据覆盖范数算法，梯度未由模型反向生成，不能代替真实模型并行组合验证。

### MC2×TP 当前状态复测

在 `a488490e07704af32644d3ea444100c8a2498f67` 上复跑
`tests/integration_tests/mc2_tp_ep_equivalence.py`（两进程 CPU/Gloo、FP16、TP2/EP2、
MC2 算子替身）仍 exit1：rank0 五个、rank1 七个参数梯度不满足原 atol5e-4/rtol1e-2。
`MC2-TP-PUBLIC-01` 保持开放。算子代数单测9 passed及上游调用一致均不能关闭此项；
Ascend 实际融合内核、完整更新和数值契约仍缺验证。详见台账
`MC2-TP-CURRENT-STATE-RECHECK-02`，日志 `/tmp/llmtuner-mc2-tp-current-review.log`。

### HF custom FSDP × full AC × prefetch 数值补证

`tests/integration_tests/hf_fsdp_ac_prefetch_equivalence.py` 在两 rank CPU/Gloo、
FP32、三层 Qwen3、custom shard2、前后向窗口 2 下，对照未并行原始 HF
模型，三步 momentum SGD 的输出、全部参数梯度及更新参数通过
`atol=1e-6, rtol=1e-5`。TP/PP/CP/EP 均为 1，无 compile。
这补充独立数值参考，不证明设备异步重叠或其他并行组合；完整命令和日志见
`file_review.md` 的 `HF-CUSTOM-AC-PREFETCH-DENSE-ORACLE-REVIEW-05`。

### MC2 EP2 连续更新与空输入补证

`MC2-DISTRIBUTED-MULTISTEP-REVIEW-05`：CPU/Gloo、FP16、EP2，真实all-to-all、
MC2算子模拟，同一实例三步momentum SGD；不均匀输入、单rank无输入、单rank
无接收和全部rank无token均通过原容差下的输出、梯度、参数及动量比较。
参考为本地全量GroupedExperts；无TP/PP/CP/FSDP/AC/compile/EMA，
不关闭MC2×TP的FP16失败或Ascend设备验证缺口。脚本为
`tests/integration_tests/mc2_distributed_adapter.py`。

### KCP MLA 通信几何预检

`KCP-MLA-GEOMETRY-01`：MLA分支在mask及Q/K/V通信前核对各CP rank的
batch、heads、sequence、QK维度及V维度。五类几何不一致均在payload交换前拒绝，
CPU/Gloo CP2原16组MLA输出与梯度等价检查继续通过。不增加不等长分片支持，
KDA padding/compacted state exchange及Ascend设备验证仍开放。

### TP2 × KV-CP2 完整状态恢复

`TRAINER-CP-TP-SEQUENCE-MESH-01`修复日志归约把二维CP×TP mesh直接用于get_group的错误。
四rank CPU/Gloo FP32、Qwen3、custom FSDP、packed JSONL、累积2、动态EMA，
连续四步与step2新进程恢复通过更新前和最终完整状态比较，后两步loss差0。
真实TP/CP/FSDP/DCP，attention使用显式CPU SDPA替身，无AC/compile。
不替代独立dense模型数值参考或目标设备flex验证。

### TP2 × KV-CP2 × full AC 恢复补证

`TP-CP-FULL-AC-RESTORE-REVIEW-05`：四rank CPU/Gloo、FP32、custom FSDP、
packed JSONL、累积2、动态EMA及full AC，step2新进程恢复前完整状态逐位一致，
后两步loss差0，最终状态通过。attention为显式SDPA替身，无compile，
不代表设备flex内核或完整并行矩阵验收。运行中发现并修复当前工作区mesh字段
重命名未同步调用方的问题，详见TRAINER-REDUCTION-FIELDS-01。

### TP2 × KV-CP2 × selective AC 恢复补证

`TP-CP-SELECTIVE-AC-RESTORE-REVIEW-05`：四rank CPU/Gloo、FP32、custom FSDP、
packed JSONL、累积2、动态EMA及逐op selective AC，step2新进程恢复前完整状态
一致，后两步loss差0，最终状态通过。attention使用显式SDPA替身，无compile。
此SAC为本仓库保留的历史逐op策略，当前torchtitan SelectiveAC已采用RegionAC预设，
两者语义差异仍按upstream_map记录，不视为当前上游策略或目标设备验收。


### 当前工作区三轴前向复验（TP-EP-CP-DECLARATION-REVIEW-06）

四 CPU/Gloo rank 的 `tests/integration_tests/tp_ep_cp_equivalence.py`
于本轮执行成功：TP2 × KVCP2 × EP2，world=4（EP 复用已有轴），
Qwen3MoE、seq=256、8 experts、FP64。实际检查 TP 投影替换、每 rank
专家数量、CP/TP 拼接 logits 和 summed loss，对照独立未分片 HF 参考，
保持原始 1e-6 容差。日志：`/tmp/llmtuner-tp-ep-cp-current-06.log`。

此行只补充前向数值证据：脚本使用 no_grad，不覆盖 backward、optimizer
更新、FSDP/PP、目标设备及完整恢复，相关矩阵缺口仍开放。


### TP2 × KVCP2 × EP2 的 CE 训练参考（REVIEW-06）

新增 `tests/integration_tests/tp_ep_cp_training_equivalence.py`，四 CPU/Gloo
rank，经生产并行装配分别运行 native/custom FSDP。显式 train 模式，FP64
Qwen3MoE、两层、8 experts、seq256、三步 SGD(momentum=0.9)。对照独立
未分片 HF 模型，每 rank 27 参数的梯度、更新后权重及 momentum，以及
logits/全局 CE loss 全部通过，atol/rtol=1e-6。

本场景两侧 router_aux_loss_coef=0，attention 为已有显式 dense-mask SDPA
替代；真实 CP 通信/autograd、TP、EP alltoall 与 FSDP 生效。此结果补充
CE 训练证据，不能关闭 aux loss、production flex backward、目标设备、
Trainer 编排及完整恢复验收。日志：
`/tmp/llmtuner-tp-ep-cp-training-{custom,native}-ce-06.log`。


### 三轴辅助损失更新补验（TP-EP-CP-AUX-TRAINING-REFERENCE-06）

上述训练脚本现支持 `custom aux` / `native aux`，router_aux_loss_coef=0.01。
独立 HF 参考从 router logits/专家选择计算每层全流均衡损失并直接加入 CE，
生产端按 Trainer 契约设置分母、进入 SPMD context，实际触发 CP/TP 统计
归约及辅助梯度注入。每层指标、27参数梯度、三步SGD更新和momentum均通过，
保持atol/rtol=1e-6；参考辅助梯度贡献明确大于1e-6。

四CPU/Gloo rank、TP2/KVCP2/EP2、native/custom FSDP均exit0；日志
`/tmp/llmtuner-tp-ep-cp-aux-{custom,native}-final-06.log`。此项补充该具体
配置的带aux训练证据，仍不能证明设备flex、累积/packing/padding、AC/compile、
Trainer完整编排及恢复矩阵，其他缺口保持开放。


### 三轴 padding 与双分母（TP-EP-CP-PADDING-DENOMINATOR-REFERENCE-06）

训练脚本追加 `aux padding` 可复现全局184个非padding路由token、168个
CE有效标签，其中cp1/tp1的本地64token全部padding。masked prompt仍参与
路由均衡、padding排除；两类分母独立设置。四CPU/Gloo rank、TP2/KVCP2/EP2、
native/custom FSDP、FP64、aux=0.01的三步梯度/更新/momentum及每层aux指标
对照独立HF公式全部通过，保持atol/rtol1e-6。日志
`/tmp/llmtuner-tp-ep-cp-padding-{custom,native}-06.log`。

此为真实通信下的padding统计证据，不是物理空输入rank、packer、累积、
Trainer完整编排或设备flex验证；未关闭其他并行矩阵与恢复缺口。


### 三轴 AC/aux/padding 更新（REVIEW-07）

`tp_ep_cp_training_equivalence.py <custom|native> aux padding <full|selective>`
经生产装配启用AC，TP2/KVCP2/EP2、四CPU/Gloo rank、aux0.01、FP64，
四种引擎/AC配置均三步通过全部27参数梯度/权重/momentum和辅助指标检查，
保持atol/rtol1e-6。selective另断言两层CheckpointWrapper实际存在。
backward后指标与原forward相同，未重复累加。参考为无AC独立HF模型。
日志 `/tmp/llmtuner-triaxis-ac-{custom,native}-07.log` 与
`/tmp/llmtuner-triaxis-sac-{custom,native}-final-07.log`。
仍使用CPU SDPA attention替代，未验证设备flex、累积、compile或恢复。


### 三轴 AC 与梯度累积（TP-EP-CP-ACCUMULATION-AC-REFERENCE-07）

三轴训练脚本支持在AC参数后追加累积次数，如
`custom aux padding full 2`。native/custom FSDP × full/selective AC四场景
各三步通过：两个不同microbatch累计后只更新一次，路由/CE step分母为
368/336，27参数梯度、更新及momentum与独立HF参考一致（atol/rtol1e-6）。
每层body实际每step执行4次，辅助指标按step累计且重计算不重复写入。
日志 `/tmp/llmtuner-triaxis-accum-{full,selective}-07.log` 与
`/tmp/llmtuner-triaxis-accum-native-{full,selective}-07.log`。
每批有效数相同；不覆盖不等长加权、尾窗口、Trainer编排、设备flex或恢复。


### 三轴不等有效数累计（TP-EP-CP-UNEQUAL-TOKEN-ACCUMULATION-07）

脚本末尾 `full 2 uneven`（custom）和 `selective 2 uneven`（native）
各三步通过。两批有效路由/CE分别184/168与152/128，step总分母336/296；
独立HF参考按有效数分别加权，27参数梯度/权重/momentum及辅助指标保持
atol/rtol1e-6通过。四CPU/Gloo rank、TP2/KVCP2/EP2、FP64、aux0.01，
日志 `/tmp/llmtuner-triaxis-uneven-{custom,native}-07.log`。
仍使用SDPA替代attention，不覆盖Trainer计数、尾窗口耗尽、compile与恢复。


### 真实 Trainer step 的 TP/EP/CP 累积验证

`trainer_tp_ep_cp_equivalence.py` 在四 CPU/Gloo rank 上使用生产并行装配和
`Trainer.train_step`，TP2/KVCP2/EP2、native/custom FSDP、full AC、aux=0，
每 step 两批有效标签数168/128。两个实现均连续三步对照独立完整HF summed
CE参考，27参数梯度、更新和loss以atol/rtol1e-6通过，消耗token指标一致。
证据详见台账 `TRAINER-TP-EP-CP-UNEQUAL-REFERENCE-10`。

运行时为手动装配，scheduler/checkpointer无操作；使用显式CPU attention
替代，不覆盖完整构造、数据loader、aux、compile、恢复及目标设备内核。

后续机制检查（`TRAINER-TRIAXIS-MOMENTUM-REPLAY-11`）确认两种FSDP实现
每step每层body实际执行4次，且27参数的SGD动量状态与HF参考均以1e-6通过。
该检查增加AC重计算和优化器隐藏状态证据，沿用上述环境与覆盖边界。


`TRAINER-TRIAXIS-AUX-DENOMINATOR-12` 进一步覆盖真实Trainer的aux分母与
metrics pre-hook：CE总数296、路由总数336，系数0.01，native/custom各三步。
独立HF路由公式验证aux均值，全部27参数梯度/更新/动量以1e-6通过，full AC
每层body执行4次且未重复aux指标。覆盖边界沿用手动运行时及CPU替代限制。


三轴Trainer后续使用生产WSD scheduler（warmup2/total4/linear decay0.5），
独立参考手工设置lr=[0.01,0.02,0.02]；native/custom+aux各三步均通过，
同时验证日志lr、下步lr及scheduler进度。见
`TRAINER-TRIAXIS-SCHEDULER-ORDER-13`。optimizer仍为直接SGD，EMA关闭，
checkpointer无操作；完整构造与恢复仍未由本测试覆盖。


`TRAINER-TRIAXIS-EMA-POSTUPDATE-14` 启用生产固定decay0.75 EMA，native/
custom三轴Trainer+aux各三步，27个EMA参数状态与独立HF更新后平均公式以
1e-6通过。此处EMA已启用；buffer、动态/间隔策略和完整恢复仍未覆盖。


生产AdamW容器三轴验证目前失败，详见开放项
`TRAINER-TRIAXIS-ADAMW-NUMERICS-15`：native/custom均未通过原1e-6权重
等价断言，custom定位到首步专家w1_EFD更新后差异3.0e-6至6.3e-6，
该参数梯度先通过原比较。原因未定，不得沿用SGD结果宣称AdamW组合已验收。


AdamW首步诊断（`TRAINER-TRIAXIS-ADAMW-FORMULA-16`）表明native/custom
同一失败索引的更新与实际梯度独立公式仅差1.39e-17；实际/参考梯度约1e-11
的差异经eps1e-8归一化可完整解释3.0e-6至6.3e-6权重差。梯度差的上游
来源与后续步尚未定位，原AdamW验收保持失败，不扩展为组合通过。


AdamW因素对照（`TRAINER-TRIAXIS-ADAMW-FACTOR-CONTROLS-17`）中，关闭AC
保持原aux专家差异不变；关闭aux后full/none均在首步k_proj单元素失败，
误差2.44e-6。AC不是当前首步差异的必要条件，aux不是整体失败的唯一来源。
三组控制均未验收通过，未修改原容差或以控制替代原组合要求。


Q/K梯度精度归因（`TRAINER-TRIAXIS-LOCAL-WGRAD-ROUNDING-25`）：native/
custom在noaux/none AC/显式FP32 gate参考下，两层Q/K梯度均与独立局部
`dy.T@x→FP32→CP sum→FP64→累积`公式零差。不能用完整梯度最后舍入
替代该契约。原完整HF AdamW权重断言仍失败，专家、aux/full及后续步验收开放。

### 当前训练与恢复证据补充（批次 38–50）

以下是限定场景的实测结果。历史诊断按发生顺序保留，本表不将局部通过升级为
完整组合或目标设备验收；各命令、日志和失败记录见 `file_review.md` 对应 ID。

| 场景 | 路径与配置 | 当前结果 | 未覆盖边界 |
| --- | --- | --- | --- |
| PP2 × KV-CP2，1F1B | 四 CPU/Gloo rank，native/custom FSDP，累积2、4 microbatch、4步 AdamW | 两实现通过原1e-5容差；最大梯度误差5.576e-8，参数4.389e-6（49） | DP shard=1；无TP/EP、EMA、恢复、compile |
| PP2 × KV-CP2，Interleaved1F1B | custom FSDP，每rank2个part、6层，其余同上 | 最大梯度误差4.284e-8，参数5.452e-6；通过（50） | 同上；不外推至native多part场景 |
| PP2 × KV-CP2，ZBVZeroBubble | custom FSDP，V布局首尾stage同rank，每rank2个part | 最大梯度误差8.900e-8，参数9.930e-6；通过但接近1e-5阈值（50） | 同上；未证明其他dtype或模型精度 |
| DDP/native/custom FSDP随机恢复 | 两CPU rank，sync/async，真实AdamW、scheduler、EMA与dropout | 保存边界及后续两步状态精确；保留rank局部失败协调回归（38–39） | 同进程新运行时，无Trainer/loader、拓扑变化 |
| 完整Trainer跨进程恢复 | 单CPU/eager，Qwen3，synthetic或本地JSONL/Grain，动态间隔EMA | 三独立进程；更新前七类状态、最终状态及独立EMA公式精确（40–42、48） | 无多rank/多轴、compile、设备内核；语料为生成fixture |
| 完整Trainer从异步源恢复 | 本地JSONL/Grain，interval1；初始运行至3步，明确恢复异步第2步 | 首次更新前及后续至4步完整状态精确，实际future存在（44） | 最终保存仍同步；无pinned-memory、崩溃耐久性验证 |

PP×CP三行均使用显式CPU dense-mask SDPA替代attention计算；CP序列分片、
KV通信/autograd、PP调度与Trainer装配仍为生产实现。这些结果不能证明生产
flex内核正确性或通信重叠性能。

完整Trainer恢复的七类状态为模型、optimizer、scheduler、EMA、loader、trainer
和解码后的RNG；检查恢复后的首次更新前边界，避免只比较最终权重掩盖漏载。
EMA另外用独立半衰期及lerp公式核对，避免两条轨迹同时未更新却误判一致。

#### AdamW三轴诊断的当前边界

前述15–25为历史诊断。后续参考修正为生产FP32 router、FP32 CE、共享softmax
及独立aux公式；Q/K和专家的首步局部梯度重放与生产结果可精确一致，专家输入
身份检查拒绝重复行歧义（详见台账26–36）。这解释了部分数值契约差异，尚未
建立完整多步独立参考：aux/full AC的完整HF权重比较仍在专家 `w1_EFD` 上以
原1e-6容差失败，最大差异约2.0208e-6。局部重放使用生产边界顺序/计数，不能
替代完整独立参考；TP2×KV-CP2×EP2的生产AdamW验收继续保持开放。


### 三轴独立 FP32 归约参考（批次115）

三轴数值补充（批次115）：`triaxis_rounded_reference.py` 使用独立dense参考，
将QKV与专家权重梯度按CP token半区计算并执行FP32求和，QKV还按TP特征半区
计算。CPU/Gloo TP2×KVCP2×EP2、native/custom FSDP、full AC、aux.01、AdamW
三步训练在原1e-6容差下通过，包括梯度、权重、optimizer、scheduler和EMA。
该参考不读取生产梯度；默认HF参考仍保留失败，不能混称两种算术契约。
此证据限定固定拓扑和CPU替代attention，不证明位级等价、目标设备或完整恢复；
问题15仍待参考契约审定及其余范围验收，详见逐文件台账。
批次116另通过native无aux/无AC、native有aux/无AC，以及未启用激活诊断hook
的native有aux/full AC、custom无aux/full AC；均为三步原容差比较。参考入口
明确拒绝非连续TP2/KVCP2/EP2配置，并要求显式选择FP32 router参考。

### Fixed router 特征 TP：实现进展与公开边界（批次 59–75）

批次59–75期间 `ParallelConfig` 拒绝 fixed_router=True、EP=1、TP>1；
批次108通过公开Trainer的CPU DP2×TP2完整流程后移除该限制。
下表前六行仍是低层机制证据，不能单独证明完整训练；未列出的轴为1，
前六行环境为CPU/Gloo、FP64。目标设备及未运行组合仍待独立验收。

| 路径 | 已取得证据 | 保留边界 |
| --- | --- | --- |
| HF→原生 MoE 特征 TP 转换 | 保留序列 gather/SUM reduce-scatter、padding、专家特征分片及 checkpoint 名称；两 rank TP2，288种 eager 场景对照独立 FFN 公式 | 显式构造共享 MLP，非真实家族完整训练 |
| 辅助损失 | 按 gathered token 副本数缩放梯度，metric保持原值；aux0/.03、混合/全padding、AC验证 | 非多轴SPMD统计验收 |
| 生产 compile 入口 | whole-model/per-block aot_eager各12场景；实际生成图，数值、更新、状态验证通过 | 非Inductor/目标设备；早期裸块测试遗漏MoE发现契约，失败记录保留 |
| 新运行时 DCP 恢复 | AdamW、scheduler、EMA、RNG，AC×compile×sync/async八场景；更新前与后续两步逐位一致 | 同OS进程新运行时，无Trainer/loader状态、新OS进程或拓扑变化 |
| Native FSDP×TP | 四rank DP2×TP2；上述八场景恢复通过 | decoder局部分片，不证明完整Trainer计划或独立dense缩放等价 |
| Custom FSDP×TP | 双hook上下文，以及pattern入口配置前后向prefetch窗口2，八场景恢复通过；EMA逐参数独立公式 | 不证明任意窗口、真实通信重叠或性能 |
| 完整 Trainer（批次108） | 四rank DP2×TP2、EP1、Qwen3 fixed MoE aux.03、真实Grain packing、async、每step验证；native eager及native/custom full AC+aot_eager，连续/初始/恢复三个独立OS作业，七类状态与local shard精确，EMA独立公式 | CPU；不证明完整模型独立dense梯度参考、PP/CP、拓扑变化、Inductor或设备性能 |

测试入口为 `tests/integration_tests/fixed_router_tp_equivalence.py`（默认eager，
可加 `--compile` 或 `--fullgraph`）及 `fixed_router_tp_checkpoint.py`
（默认两rank TP2，可用四rank `--fsdp`，custom再加 `--custom-fsdp`）。
命令、失败与最终日志详见 `file_review.md` 对应批次；恢复一致性不替代独立训练
数值参考。EMA混合Tensor/DTensor更新按类型和布局分组，避免foreach类型混用。

批次108已补齐真实Trainer、模型家族、数据管线、完整状态和独立进程恢复的
上述CPU证据；声明设备与dtype仍需验收。三轴AdamW开放项、
MC2设备数值、KCP及Domino设备/性能缺口不因本节局部通过而关闭。

### 完整 Trainer 的分布式新进程恢复补充（批次 84–85）

`trainer_ema_process_resume.py corpus [async] dp2 [custom]` 使用三个独立
两rank作业，分别连续训练、生成checkpoint和恢复续训。CPU/Gloo、Qwen3、
本地JSONL/Grain packing；原生与自定义FSDP各sync/async两场景通过。
每rank七类状态在加载边界及续训后逐位一致，动态EMA有独立公式断言，
实际DTensor分片必达；custom额外验证FineGrainedFSDPState。
日志与命令见 `docs/file_review.md` 批次84–85。

该证据补齐单轴DP2的完整Trainer新进程恢复，不覆盖TP/CP/EP/PP组合、
compile、设备、拓扑变化、任意hook迁移或prefetch窗口。前文单rank历史
证据的范围保持原样，不能将低层多轴恢复与此项拼接为完整多轴验收。

批次86追加 `compile` 选项：native/custom FSDP各在CPU aot_eager完整模型
编译及异步保存下，通过三个独立DP2作业恢复；每个worker强制编译图数大于零，
七类状态及动态EMA公式精确。仅证明各自编译运行的连续/恢复一致，不证明
eager/compile算术等价、Inductor或设备内核。日志见审查台账批次86。

批次87补充上述native/custom DP2编译异步恢复的本地布局比较：模型、optimizer、
EMA每个DTensor的local tensor、global shape/stride、placements和mesh均在
加载边界及续训后精确一致；完整值比较同时保留。该快照只用于测试，训练
checkpoint格式不变，也不构成拓扑变化恢复的证据。

批次88追加 `ac`，native/custom FSDP DP2均通过full AC＋aot_eager＋异步
完整Trainer新进程恢复；实际两个CheckpointWrapper及编译图必达，七类状态与
本地布局精确。范围仍为CPU同拓扑DP2；不宣称AC性能、其他策略或多轴已验收。

批次90追加 `validate`：每个训练step后执行两批真实验证，在native/custom
FSDP DP2＋full AC＋aot_eager＋async组合下，各三个独立作业完整恢复通过。
每次验证前后直接比较七类训练状态及本地布局，均未改变；不是仅比较两个
可能同样错误的恢复结果。范围为dense Qwen3/CPU默认RNG流，不扩展到MoE、
PP、设备stream、私有generator或并发Trainer隔离。

批次91以 `moe` 选项覆盖Qwen3 MoE（4专家/top2、aux0.03、fixed router EP1）。
native/custom FSDP DP2＋AC＋aot_eager＋异步保存＋每step验证的完整新进程恢复
通过；每次验证不改变七类训练状态、本地布局或两个非持久token计数。仅EP1，
不能推广EP>1或quantile router。测试launcher显式使用loopback rendezvous，
规避本机FQDN被解析至代理网段的环境问题。

批次92追加 `ep2`：生产EP2与DP shard2在两rank拓扑重叠（efsdp1），
不是四rank独立DP×EP。native/custom FSDP、full AC、aot_eager、异步保存、
每step验证的完整Trainer新进程恢复通过；实际专家分片及EP通信组必达。
恢复一致性不能替代独立dense算术参考，也不证明efsdp>1或其他并行轴。

批次93使用 `dp4 ep2` 补充efsdp2：三个独立四rank完整Trainer作业，
native/custom FSDP各在AC＋aot_eager＋async＋每step验证下恢复精确。
专家实际DTensor mesh2及本地半分片必达；dense DP shard4、EP2、efsdp2，
TP/CP/PP均1。该恢复证据不替代独立dense梯度和更新等价验证。

批次94修正验证快照的普通Tensor别名：此前只有DTensor路径clone，普通Tensor
原地修改可能漏报。现在所有张量独立复制，并在每个worker实际修改/恢复活参数
验证快照检测能力；单rank及custom EP2×efsdp2编译完整恢复复跑通过。90–93的
普通Tensor验证隔离声明应结合本项限制理解，不能仅凭旧快照认为完全已验证。

### EMA 本地更新后的恢复复验（批次242）

CPU/Gloo、custom FSDP、MoE fixed router/aux0.03、full AC、aot_eager、
动态 EMA、每步验证和异步中间 DCP 下，以下组合重新通过三个独立作业的
加载边界完整状态与连续训练终态精确对照：

| rank 数 | dense DP shard | TP | EP | expert FSDP | EMA 通信计数 |
| --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 1 | 2 | 批次245复跑每次调用实测 0 |
| 4 | 4 | 1 | 2 | 2 | 每次调用实测 0 |

EP 场景新增 `CommDebugMode` 包围实际 `EMA.step()`，并检查逐步调用序列；
统计不包含 Trainer 外层错误协调及 checkpoint 通信。完整状态、布局、动态
EMA 公式和验证隔离断言均保留。命令、日志和产物见审查台账242批。
批次245在新增验证 RNG 进入错误协调后，复跑上述 DP2×TP2 全组合，加载
边界与续训终态仍精确一致，并补齐 TP 场景每次 EMA 更新通信计数为 0。
批次249在当前 RNG/Grain 游标修复后，以真实本地 JSONL、tokenizer 和 Grain
concat packing 再跑 DP2×TP2 的 custom/full AC/aot_eager/EMA/验证/async DCP
组合。四 rank 在新进程加载边界及续训终态的七类状态和本地布局均零容差一致，
实际 EMA 每次更新通信计数为 0；命令与产物见审查台账249批。
此证据不覆盖 TP2+EP2 同时启用、CP/PP、CUDA/NPU、拓扑变化或独立 dense
训练算术等价；不能把两个场景合并为三轴恢复证明。

### FirstFit×PP完整加载边界（批次253）

`trainer_ema_process_resume.py dp4 pp2 custom compile ac validate async first_fit`
在CPU/Gloo四rank、DP shard2×PP2、Interleaved1F1B四stage/四microbatches下通过。
六层dense Qwen3、custom FSDP、full AC、aot_eager、动态EMA、每步验证与真实FirstFit
共同启用；四rank恢复step2后首次更新前的七类状态及本地布局与保存点精确一致，
step4完整终态与连续训练零容差一致。实际EMA每次更新通信0，独立动态公式通过。

本项通过Trainer生产构造及train循环，不是252批实际CLI含seed导入配置的同次
边界证明。TP/CP/EP1，满chunk语料；不覆盖目标设备、其他schedule或拓扑变化。
命令、产物和默认模式回归见台账253批。

### CLI的FirstFit×PP×seed联合恢复（批次252）

当前代码在CPU/Gloo四rank、DP shard2×PP2、Interleaved1F1B四stage/四microbatches、
custom FSDP、full AC、async DCP、eager下，真实FirstFit语料结合step0模型导入通过。
独立四rank边界检查确认seed43初始权重被未分片seed替换，非模型状态与CPU RNG不变；
连续3步与先2步后新进程续训至3步的完整checkpoint零容差一致。

该组合把143批seed×PP和216批FirstFit×DP的独立证据连接到同一实际CLI运行。
step0是模型迁移；step2加载后首次更新前的完整状态未被本脚本比较。TP/CP/EP1，
无EMA/validation/compile；不覆盖目标设备、余数packing或完整状态拓扑迁移。
命令、端口及日志见台账252批。

### FirstFit与quantile完整恢复补验（批次250）

CPU/Gloo四rank、dense DP shard4/EP2/expert FSDP2、custom FSDP、DeepSeek-V3
sigmoid quantile、真实Grain FirstFit、full AC、aot_eager、动态EMA、每步验证及
async DCP组合通过连续/初始/恢复三个独立作业。加载step2后更新前完整状态与
保存点精确一致，续训step4的七类状态及本地DTensor布局与连续训练零容差一致。
实际expert bias非零、histogram不持久且更新后清空、验证隔离与EMA独立公式均通过，
每次实际EMA更新通信计数为0。命令和产物见台账250批。

TP/PP/CP均为1；FirstFit使用满chunk语料。此结果不覆盖目标设备、任意padding/
余数窗口、独立dense训练算术等价、通信故障原地恢复或拓扑变化。

### 非有限值更新保护的补充验证（批次132–133）

`tests/integration_tests/trainer_nonfinite_update.py` 通过真实 Trainer、Qwen3
模型、Adam、scheduler 和 EMA 检查单 rank 故障后的状态不变量。以下均为
CPU/Gloo、full AC、aot_eager，native/custom FSDP 分别运行通过：

| 配置 | ranks | 模型与故障目标 | 每种 FSDP 的故障场景 |
| --- | --- | --- | --- |
| DP shard2，TP/CP/PP/EP1 | 2 | dense；返回 loss NaN 或本地梯度 Inf | 4 |
| DP shard2，TP2，CP/PP/EP1 | 4 | fixed-router MoE；loss NaN 或 routed-expert 梯度 Inf | 8 |
| DP shard2，EP2，TP/CP/PP1 | 2 | fixed-router MoE；loss NaN 或 routed-expert 梯度 Inf | 4 |

最后一行 EP 与 dense DP 共享两 rank，专家 FSDP 度数为1，不是四 rank 的独立
DP×EP。每个 rank 都分别作为故障源；失败前已有非零 Adam moments 和 EMA
历史。故障后模型、optimizer、scheduler、EMA 及本地分片精确不变，所有 rank
都报非有限值，后续有限值步骤可更新权重与推进调度器。

这是更新状态保护的证据，不是独立 dense 数值等价或完整恢复证明；不覆盖数据、
RNG 和 step 回滚。CPU 断言后可继续运行不能推广到 CUDA 异步设备断言。CP、PP、
设备内核、Inductor、其他路由器和 efsdp>1 的此类故障测试仍需单独验证。

批次134补充 DeepSeek V4 HCA 的真实 Compressor Ulysses CP：DP shard2×CP2、
四 rank、TP/PP/EP1，native/custom FSDP 均通过同类故障保护回归。本组为 eager、
无 AC，未替换 attention kernel。每种实现覆盖所有四 rank 的 loss NaN 和 HF
原生专家梯度 Inf，共16种故障及16次有限值后续更新；所有更新状态及本地分片
保持原精确断言。本项不覆盖普通 HF Flex CP、KV AllGather、KCP、CSA/sliding
混合层，亦不补足 CP 下 AC/compile、设备和完整恢复的其他验收要求。

批次135补充 DP shard2×PP2 的真实 1F1B 调度（4 microbatches，四 rank）：
native/custom FSDP 均在 eager、无 AC 下通过全部 rank 的 loss 标记和本地梯度
非有限值测试，共16种故障及16次后续有限值更新。非末级 loss 测试明确针对
有限性标记不变量：生产路径通常返回固定 sentinel，不代表复现了正常前向
自行生成 NaN sentinel。旧逻辑只将该标记传到同一 PP 链，其他 DP 副本仍可
更新；现在将 local loss 和 grad norm 合成后做一次 world MIN，保证统一拒绝。
完整 CPU 回归及 custom FSDP 的 DP2×TP2、DP2×HCA CP2 故障回归亦通过。

批次136在有限性归约改动后，复跑 DP shard2×PP2 的 Interleaved1F1B 完整恢复：
四 rank、6层 dense Qwen3、每 rank 两个本地 parts、4 microbatches、full AC、
aot_eager、async DCP、dynamic EMA。native/custom 各以独立 full/resume 作业
运行通过，四 rank 的加载边界快照均存在且更新前比较无差异；最终参数、Adam、
scheduler、EMA、trainer counters、RNG 和 synthetic loader 状态精确相等，续训
loss 与连续训练逐位一致。证据包含实际编译图和各 local part 的编译检查。
本项通过 Trainer 构造与 train_step 驱动既有恢复 harness，不等同于 CLI 主循环、
真实语料、设备内核、Inductor 或其他 PP schedule 已一并验证。

批次137扩展真实 CLI 恢复脚本 `cli_checkpoint_equivalence.py`：`pp2 async ac`
使用 torch.distributed.run --module llmtuner 启动连续、提前结束、恢复三个独立
四 rank 作业，DP shard2×PP2、Interleaved1F1B、4 microbatches、full AC、async
DCP、eager。native/custom 均通过最终完整 checkpoint 精确比较，含全部四 rank
解码后的 RNG、模型、Adam、scheduler、loader 和非零 trainer token 计数。
该测试不替换 Trainer，也不通过 hook 调用 train_step；CLI main 实际调用 train，
并负责进程组退出。EMA/validation/compile backend 仍是程序化配置，没有为本项
测试添加 CLI 功能。默认单进程回归也通过；本项不提供 CLI 更新前加载边界快照，
需结合136的独立边界检查理解，不能据此单独宣称完整恢复要求均已满足。

### 补充验收：tied HF 权重导出（第 184–185 批）

`tests/integration_tests/hf_tied_export.py` 使用生产 TP/FSDP 入口，将共享
embedding/head 的微型 Llama 权重导出后交给标准 AutoModel 加载。

| 环境 | TP | DP shard | FSDP | 文件数 | 结果 |
| --- | --- | --- | --- | --- | --- |
| CPU/Gloo，FP32，2 ranks | 1 | 2 | native/custom | 1/2 | 权重、共享身份、logits 精确匹配 |
| CPU/Gloo，FP32，4 ranks | 2 | 2 | native/custom | 1/2 | 同上；TP 切片、FSDP 本地存储减半已断言 |

所有场景 PP/CP/EP=1，无 AC/compile/EMA，仅验证权重导出和标准 HF 加载。
配置由调用方另供；不代表完整训练恢复、TP 训练数值或目标设备验收。


### 最近 optimizer/EMA 修改后的恢复复验（批次258–260）

以下均为当前工作区、CPU/Gloo、custom FSDP、full AC、aot_eager、每step验证、
async中间checkpoint及真实Grain数据（258–259为first_fit，260为concat）；
连续、初始、恢复为独立OS作业。
七类状态与本地DTensor布局在step2加载边界和step4终态精确一致，EMA另有独立公式。

| 组合 | 实际机制 | 证据 |
| --- | --- | --- |
| 四rank DP2×PP2 | dense Qwen3，Interleaved1F1B，每rank两个parts | 第258批，`/tmp/llmtuner-pp-resume-258.log`，退出0 |
| 四rank dense DP4、EP2/efsdp2 | DeepSeek-V3 sigmoid quantile，非零expert bias持久化、临时histogram清空及验证隔离 | 第259批，`/tmp/llmtuner-ep-resume-259.log`，退出0 |
| 四rank DP2×TP2、EP1 | Qwen3 fixed MoE aux.03，实际专家特征分片；本项使用concat/Grain而非first_fit | 第260批，`/tmp/llmtuner-tp-resume-260.log`，退出0 |

这是254–256批optimizer/EMA修改后的限定配置回归，不证明独立dense梯度等价、
空EMA或全冻结PP stage、拓扑变化、其他轴组合、目标设备dispatcher及性能。
完整命令和产物见`docs/file_review.md`；历史证据按原版本和范围保留。


### Kimi HF实际Trainer复验（第279批）

当前转换与错误协调实现经CPU/Gloo实际HF初始化、训练和导出验证：

| dense FSDP | EP / expert FSDP | AC / compile | 结果 |
| --- | --- | --- | --- |
| custom dp_shard2 | 2 / 1 | none / eager | 原HF加载、导出逐key精确，logits 1e-6，继续更新通过 |
| custom dp_shard4 | 2 / 2 | full / aot_eager | 同上；实际编译包装与非零图断言通过 |

两场景均FP32、latent Kimi MoE、alltoall，TP/PP/CP为1。导出保持完整训练
checkpoint文件哈希与运行时状态。命令、日志及产物见file_review第279批；此处
验证HF权重初始化与导出，不代表新进程全状态恢复、设备内核或性能验收。


### FSDP排除参数的训练归约（第315批）

CPU/Gloo、PyTorch2.14；以下各项PP/CP/EP均为1。

| 组合与实现 | 实际入口与参考 | 验证范围 |
| --- | --- | --- |
| DP2，native/custom | 真实Qwen3 Trainer；独立未分片模型与显式shifted CE | 三步AdamW、累积2、裁剪、全参数梯度/更新1e-6及embedding副本一致 |
| DP2，native/custom | 并行API与Trainer归约；FP64 dense参考 | 三步AdamW、单rank使用/全组未使用参数、范数与更新1e-12 |
| TP2×DP2，native/custom，eager/aot_eager | 并行API与Trainer归约；FP64 dense参考 | 三步SGD、TP分片排除参数、梯度和更新1e-12；不是完整Trainer循环 |

脚本为trainer_fsdp_excluded_gradients.py、fsdp_excluded_gradient_equivalence.py、
fsdp_pattern_references.py，位于tests/integration_tests/。真实Trainer参考沿用
实际当前学习率，不独立证明scheduler。EP/CP组合、稀疏梯度、被排除但仍为
DTensor的参数及完整恢复仍需独立验收；此表不扩展设备或性能支持声明。


### EP排除参数训练对照与CP边界（第316批）

四rank CPU/Gloo、custom dense DP4 / EP2 / expert FSDP2，真实Qwen3MoE
alltoall。ep_fsdp_pattern_references.py --train，以及附加--ignore-modules，
分别排除原gate/up投影和整个专家子树；三步输出、全参数梯度、范数、裁剪及
SGD更新对照独立原始HF模型均通过1e-6。直接调用Trainer归约方法，aux loss=0，
PP/TP/CP1、无AC/compile/EMA/恢复。禁用归约的诊断对照在首步w1梯度失败，
rank0 max abs0.0029611242935061455、shape(2,8,16)，原容差1e-6。

另行尝试真实Trainer DP2×CP2，KV/native及Ulysses/custom均未取得训练证据：
默认CPU SDPA被CP拒绝；显式真实Flex则因上游不支持CPU backward退出1。
失败日志与诊断脚本见file_review第316批，需支持Flex backward的设备重验；
不把EP2/efsdp2或CPU CP前向结果扩展为该组合训练通过。


### 完整Trainer EP排除与恢复（第317批）

custom dense DP4、EP2/efsdp2，Qwen3MoE alltoall，原gate/up投影排除、
累积2、裁剪0.5、aux0.03。真实Trainer三步AdamW对照原HF独立CE及显式辅助
目标，eager和full AC+aot_eager均以1e-6通过全参数梯度和更新。
同轴完整新进程DCP恢复加dynamic EMA、RNG和数据游标边界比较通过，
四rank更新前failures0，恢复两步loss差0。脚本及完整命令见file_review第317批。
这是固定拓扑的限定排除路径；不证明任意pattern、EP×TP/CP、目标设备或性能。


### HSDP排除参数与双专家副本轴（第318批）

CPU/Gloo四rank dense replicate2/shard2：native/custom FSDP三步FP64独立
AdamW参考通过1e-12，覆盖单rank使用与全组未使用排除参数；custom完整Trainer
full AC+aot_eager、累积2、裁剪0.5的三步全参数梯度/AdamW参考通过1e-6。
八rank replicate2/efsdp2/EP2、custom FSDP、真实Qwen3MoE alltoall，排除
原gate/up时三步独立HF输出、梯度、范数、裁剪与SGD更新通过1e-6；直接调用
Trainer归约，aux=0，无AC/compile/恢复。各轴大小与二维DTensor mesh有显式
断言。具体命令及最终日志见file_review第318批，不能据此扩展为完整八rank
Trainer或HSDP恢复通过。
