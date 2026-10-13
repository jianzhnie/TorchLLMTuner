# FSDPTurbo Distributed 与 TorchLLMTuner Parallel 实现审查

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

FSDPTurbo 的 KCP、MC2、Domino、自定义 fine-grained FSDP 和低精度通信没有直接迁移。它们依赖特定模型 patch、NPU kernel、HCCL 行为或另一套专家布局，需要独立设计和设备数值测试。

## 3. 模块映射

| 领域 | FSDPTurbo | TorchLLMTuner | 结论 |
|---|---|---|---|
| 进程组与 mesh | `parallel_state.py` | `topology.py`、`accelerator/distributed.py` | 两者模型不同，不能直接替换 |
| 通信原语 | `dist_ops.py` | `accelerator/collectives.py`、TP/CP kernel 内部原语 | TorchLLMTuner 分层更细 |
| FSDP | `fully_shard_parallel/` | `fully_shard_parallel/fully_shard_parallel.py` | 核心能力相当，选层策略不同 |
| Custom FSDP | `fine_grained_fully_shard/` | 无 | 设备和维护成本较高，暂不迁移 |
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
| 选层 | 用户提供模块通配符和逐模块 plan | 根据 HF wrapper 的 decoder 结构固定分层 |
| 忽略参数 | `ignored_modules`/`ignored_params` | EP 分支使用参数身份和 placement callback |
| 实现后端 | native 或 custom fine-grained FSDP | PyTorch native composable FSDP2 |
| dtype | 配置显式提供 param/reduce/output dtype | 参数 dtype 来自模型构建，reduce 固定 FP32；混合参数 dtype 明确拒绝 |
| prefetch | 可配置前后预取层数 | EP 时使用 TorchTitan 相邻层显式 prefetch 链 |
| 专家 placement | EFSDP 模块列表与 `Shard(0/1)` | 根据 `efsdp × ep` 和全局专家数选择 placement |
| gradient scaling | 专家模块设置 divide factor | 关闭 FSDP 自动平均，由 Trainer 的全局 token normalization 定义缩放 |

TorchLLMTuner 的实现与其 Trainer 契约一致。直接迁移 FSDPTurbo 的 `gradient_divide_factor` 会重复缩放梯度；直接迁移任意模块 pattern 也会破坏 HF wrapper 对 tied embeddings、LM head 和 EP 参数身份的处理。

### 5.3 本轮迁移：CPU offload

新增配置：

```python
ParallelConfig(
    fsdp_cpu_offload=True,
    fsdp_cpu_offload_pin_memory=True,
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

### 5.4 未迁移项

- **custom fine-grained FSDP**：FSDPTurbo 的实现包含 param group/state/strategy，并与其 hook 和 NPU 路径耦合；需要独立多卡正确性测试。
- **任意层数 prefetch**：TorchLLMTuner 已按 TorchTitan 对 EP 设置相邻层 prefetch。增加可配置窗口会改变峰值内存和通信时序，应先建立基准再实现。
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

FSDPTurbo 的 pattern plan 不应直接迁移，因为 TorchLLMTuner 需要额外处理 attention reshape、vocab loss 和 MoE boundary。后续可以借鉴的是受约束的用户 plan override，并在 realization 前复用现有 shape/probe 校验。

## 7. Context Parallel

### 7.1 策略对比

| 策略 | FSDPTurbo | TorchLLMTuner |
|---|---|---|
| KV AllGather | 支持，function patch | 支持，`CPFlexKernel` |
| Ulysses | 支持，function patch | 支持，sequence/head all-to-all |
| KCP | 支持 KDA/KCP 专用状态和 patch | 未实现 |
| CP 组合轴 | KV AllGather 可与 Ulysses 组合 | 单一策略二选一 |
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

### 7.3 未迁移项

- KCP 依赖 KDA 状态与模型 patch，不是通用 attention strategy；
- KV AllGather × Ulysses 双轴组合会改变当前单一 CP 轴和世界规模公式；
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
- aux-loss 和 quantile balancing 通过 optimizer pre-hook 更新；
- checkpoint adapter gather/load 逻辑专家张量，使恢复不绑定物理 EP shard；
- TP × EP 的 expert feature shard 和 sequence boundary 有独立实现。

审查结果：

- EP group 来自 sparse mesh，并在 PP 下限定在本 stage；
- swap 在 FSDP 之前执行，FSDP 看到的是最终专家参数；
- HF initial checkpoint 与 swapped expert layout 的不安全组合被明确拒绝；
- GPT-OSS bias、未知 group scoring、shared expert gate 等未覆盖布局会拒绝转换；
- `deepep`/`hybridep` 是登记缺口，配置不会静默回退。

### 8.3 未迁移项

- MC2 与 Domino 依赖通信/计算融合和特定设备后端；
- fixed router 属训练算法变化，需要配置、checkpoint 和指标共同设计；
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
| FSDP2 | 完整 | dense、MoE、HSDP、reshard、mixed precision |
| FSDP CPU offload | 本轮补齐 | 已暴露 enable 与 pin-memory 配置 |
| FSDP custom backend | 未实现 | FSDPTurbo 专有路径，需单独验证 |
| TP dense | 完整 | attention/MLP/embed/head/vocab loss |
| TP MoE | 有条件完整 | 已支持布局完整；未知布局拒绝 |
| PP | 完整 | stage、schedule、microbatch、validation |
| CP KV AllGather | 完整 | FlexAttention 与 BlockMask 契约内 |
| CP Ulysses | 完整 | head 整除和顺序约束内 |
| CP KCP | 未实现 | 模型与 kernel 专用能力 |
| EP All-to-All | 完整 | 已支持 HF MoE 布局内 |
| EP TorchAO | 可选 | 需要 TorchAO |
| EP MC2/Domino | 未实现 | 设备专用融合路径 |
| TP × FSDP | 完整 | 有多进程等价性测试 |
| CP × TP | 完整 | 联合 sequence 整除和 loss mesh |
| EP × FSDP | 完整 | sparse mesh 与 expert placement |
| TP × EP | 完整 | expert feature shard 与 boundary |
| PP × TP/CP/EP/FSDP | 已接线 | 各 stage 使用相同 stage order；设备组合测试按环境运行 |
| TP × EP × CP | 已接线 | mesh 与输入/loss 路径完整；依赖支持的模型和设备 |

这里的“完整”指代码入口、配置、运行时和错误处理闭环；多机大模型性能仍需针对目标设备单独验收。

## 12. 本轮代码变更

1. 新增 `ParallelConfig.fsdp_cpu_offload`，默认 `False`；
2. 新增 `ParallelConfig.fsdp_cpu_offload_pin_memory`，默认 `True`；
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

- 在基准证明有收益后，为 FSDP prefetch 增加有限、明确的窗口配置；
- 增加受约束的 TP plan override，只允许覆盖现有校验可证明安全的 Linear；
- 抽象 EP dispatcher protocol，为设备后端保留接入点。

### P2：设备专用迁移

- 将 FSDPTurbo NPU collective/GMM 通过独立 backend registry 接入；
- 在有 HCCL/NPU CI 后迁移 MC2；
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
- the FSDPTurbo fixed router flag, wired to llmtuner's existing deterministic
  round-robin router implementation.
- independent `context_parallel_kv_allgather_size` and
  `context_parallel_ulysses_size` axes. The composite kernel gathers K/V over
  the first group, exchanges Q/K/V from sequence to heads over the second,
  attends `S / kv` queries against full-sequence K/V, and applies the inverse
  exchange to its output. The legacy `context_parallel_kv_allgather_size` remains available
  for single-axis configurations and cannot be mixed with the new fields.

Domino is integrated at the dispatcher/module boundary: routing, dispatch,
expert compute and combine execute in two independently tracked token slices,
with checkpoint-compatible `GroupedExperts` weights. Accelerator stream overlap
is still pending device validation. MC2 is registered with an early capability
error because its defining fused all-to-all/grouped-matmul Ascend operators are
not present in this runtime; it never silently degrades to ordinary all-to-all.

KCP is wired as a separate CP strategy for structurally recognized Kimi Delta
Attention modules. It installs the FSDPTurbo-compatible KDA forward, builds the
global sequence context, runs CP-aware causal convolutions and `chunk_kda`, and
enforces batch-size/cache/packed-input and TP incompatibility constraints. Its
`fla` and `triton_ascend_kernels` imports remain lazy, with an actionable error
when an Ascend KCP runtime is unavailable.

Custom fine-grained FSDP is available through a PyTorch 2.14 compatibility
adapter. It keeps independent pattern-selected FSDP units, ignored parameters,
per-unit policy overrides, arbitrary prefetch windows and the FSDPTurbo strategy
entry point while delegating FSDP state and collectives to PyTorch 2.14. A plan
that asks to attach a unit to a different `hook_module` is rejected because that
private hook redirection contract no longer exists in PyTorch 2.14. The legacy
multi-communication-context optimization is therefore not claimed by this
adapter.

MC2's fused runtime still requires its runtime engine. It must not be marked
complete from configuration fields or copied source alone; their remaining
integration boundary is described in the capability matrix below.
