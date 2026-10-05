# llmtuner → torchtitan 对应关系表

[llmtuner](../llmtuner) 拿掉了 TorchTitan 的 `Configurable` 与 `Module` 两个抽象层，换来一个
明显更短的框架：123 个 Python 模块（99 个实现模块）、约 29.7k 行，覆盖 TP / FSDP2 /
CP / EP / PP 五条并行路径的装配、训练循环、checkpoint 与等价性测试。本文是这些模块与
torchtitan 之间对应关系的**唯一权威**。

## 怎么用这张表

**先查表，再动手。** 不要在 llmtuner 源码里加 `# upstream: <path> @ <sha>` 之类的来源
标注——那种标注试过又被退了，因为 sha 会腐烂而表不会。

**分类不是装饰，是操作指令。** 把 A 类的高保真同步规则套到 B 类文件上会毁掉设计；套到
C 类上会把项目**故意删掉**的抽象又拽回来。

**比结构用 AST，不要比 diff 行数。** `components/checkpointer/filesystem.py` 的 diff 有几十行，代码差异
是 **0**——全是改写措辞。做法是剥掉 docstring、`ast.unparse`、再
`difflib.SequenceMatcher`。下表 `ratio` 列就是这么来的。

目录级速查（细节以下方分类表为准）：

| llmtuner | torchtitan |
| --- | --- |
| `parallel/**` | `distributed/` |
| `components/metrics.py` | `observability/metrics.py` |
| `models/common/scatter_add.py` | `ops/scatter_add.py` |
| `parallel/pipeline_parallel/pipeline.py` | `experiments/transformers_modeling_backend/pipeline.py` |
| `datasets/text/processors.py` | `hf_datasets/text_datasets.py` |
| `components/checkpointer/filesystem.py` | `tools/filesystem.py` |

## 图例

- **A 移植（vendored）**——从 torchtitan 移植。上游改了，llmtuner **应该**逐项核对。
- **B 适配（adapted）**——同一个想法、不同的形状。上游改了，**读意图、不要抄形状**。
- **C llmtuner 独有**——上游没有对应物（或同名不同源）。不要"对齐"它。
- **D 缺口**——上游有，llmtuner **真的没有**。补还是不补是决策，不是疏漏。

分类按文件的**主要维护策略**划分；一个文件只能有一个主分类。文件内部若混合了移植与
适配逻辑，在"改写点"列中另行说明，避免同一文件同时收到互相冲突的操作指令。

`ratio` 是 AST 相似度的历史快照，**历史行是"全量最佳匹配"**（重现脚本的第 1 列），
仅对 A/B 类有意义。**1.000 不代表逐字相同**——剥掉 docstring 后 `ast.unparse` 归一化了
空白和引号；要判断"真逐字"，看 ratio 为 1.000 且人工确认过的那两个。少数行的"改写点"
列会额外给出**同名比较**值——两者差得远时，最佳匹配多半是噪音，以同名值为准
（见"重现这张表"末尾的教训）。

2026-09-21 复核后从 A1 移入 A2 的十行使用当前工作树的**指定对应文件直接比较**值，不是
全量最佳匹配；这些值用于解释为何维护策略已经变化，不与历史排名混用。

## A1 —— 高保真移植（改动需逐位验证）

只有 `components/checkpointer/utils.py` 和 `components/checkpointer/filesystem.py` 在该快照中经人工确认
属于"去 docstring 后结构等价"；其余行即使 ratio 很高也不是逐字复制。

| llmtuner | torchtitan | ratio |
| --- | --- | --- |
| `components/checkpointer/utils.py` | `components/checkpointer/utils.py` | 1.000 |
| `components/checkpointer/filesystem.py` | `tools/filesystem.py` | 1.000 |
| `components/optimizer/utils.py` | `components/optimizer/utils.py` | 0.996 |
| `datasets/multimodal/image.py` | `hf_datasets/multimodal/utils/image.py` | 0.977 |
| `datasets/multimodal/text_utils.py` | `hf_datasets/multimodal/utils/text.py` | 0.971 |
| `datasets/multimodal/video.py` | `hf_datasets/multimodal/utils/video.py` | 0.967 |
| `components/tokenizer.py` | `components/tokenizer.py` | 0.907 |

## A2 —— 移植但已改写（比例中等，需逐处核对）

| llmtuner | torchtitan | ratio | 改写点 |
| --- | --- | --- | --- |
| `components/checkpointer/__init__.py` | `components/checkpointer/__init__.py` | 0.757 | |
| `components/checkpointer/base.py` | `components/checkpointer/base.py` | 0.717 | |
| `components/checkpointer/dcp.py` | `components/checkpointer/dcp.py` | 0.735 | 本地/remote storage、HF export 与生命周期已重塑 |
| `components/checkpointer/torch_checkpointing.py` | `components/checkpointer/torch_checkpointing.py` | 0.265 | |
| `components/loss.py` | `components/loss.py` | 0.297 | 去 loss 类层次，保留自由函数与 vocab-parallel 数学 |
| `components/metrics.py` | `observability/metrics.py` | 0.381 | |
| `components/optimizer/lr_scheduler.py` | `components/optimizer/lr_scheduler.py` | 0.373 | 去 `Configurable` |
| `components/optimizer/optimizer.py` | `components/optimizer/optimizer.py` | 0.240 | 容器化改写；`OptimizerWrapper` 已删 |
| `components/profiler.py` | `observability/profiler.py` | 0.413 | |
| `datasets/collators.py` | `components/data/collators.py` | 0.145 | |
| `datasets/dataset.py` | `components/data/dataset.py` | 0.267 | 去 `Configurable`；三个节点类去 `Config` 后缀，构建走自由函数 `build_dataset` |
| `datasets/loader.py` | `components/data/loader.py` | 0.590 | 去 `Configurable`；`GrainDataLoader` 直接收参数，无 config 类 |
| `datasets/multimodal/collator.py` | `hf_datasets/multimodal/mm_collator.py` | 0.777 | 增加 MRoPE grid/run/长度校验，当前已是契约适配 |
| `datasets/multimodal/datasets.py` | `hf_datasets/multimodal/mm_datasets.py` | 0.310 | 去 `Configurable`；packing 改自由函数 `build_mm_sample_packing` |
| `datasets/packing.py` | `components/data/packing.py` | 0.065 | 自由函数外还增加文档容量、padding mask、长文档切分和可恢复 remainder，按语义维护 |
| `datasets/sources.py` | `components/data/sources.py` | 0.700 | |
| `datasets/text/processors.py` | `hf_datasets/text_datasets.py` | 0.767 | 路径与 processor 构造契约已适配 |
| `datasets/types.py` | `components/data/types.py` | 0.506 | 去 Configurable 后重塑 build context 与 iteration policy |
| `models/common/aux_loss.py` | `models/common/aux_loss.py` | 0.682 | |
| `models/common/async_linear.py`（2026-09-26 文件名对齐上游，原 dist_gemm.py） | `models/common/async_linear.py` | 0.527 | **parity 保留**（2026-10-04 用户决定）：DistGEMM* 模块层依赖 QKV/FFN  vendored 部件，唯一接线方式是替换 HF 自带部件（违反五部件契约），无生产消费者；不删，作为上游对照参考 |
| `models/common/feed_forward.py` | `models/common/feed_forward.py` | 0.560 | **曾写完又被退**，不要在没有明确指令时重新引入；parity 保留（2026-10-04，HF 自带 FFN，无生产消费者） |
| `models/common/linear.py` | `models/common/linear.py` | 0.620 | `PartialBiasRowwiseLinear` 仅测试引用（上游已删除该类），parity 保留（2026-10-04） |
| `models/common/attention/masks.py` | `models/common/attention.py` | 0.380 | 拆出了 mask 部分 |
| `models/common/moe/`（`block`/`router`/`experts`/`dispatcher`/`load_balance`/`balancing` 六文件） | `models/common/moe.py` + `models/common/token_dispatcher.py` | 0.155 | 2026-09-28 十七次增量拆成子包 |
| `models/common/multimodal.py` | `models/common/multimodal.py` | 0.888 | 保留算法来源，但加入同步规避与更严格的 span/run 校验；parity 保留（2026-10-04，VLM 融合在 HF 复合模型内部完成，无生产消费者） |
| ~~`models/common/param_init.py`~~ | — | — | **2026-09-25 移除**：torchtitan parity 的 vendored 死代码（llmtuner 走 HF 模型自带 `_init_weights`，全仓零引用） |
| `models/common/attention/qkv.py` | `models/common/attention.py` | 0.242 | parity 保留（2026-10-04，HF 自带 attention 投影，无生产消费者） |
| `models/common/rope.py` | `models/common/rope.py` | 0.616 | 上游持续重构后结构已分叉；同步公式与边界修复，不同步 Module/缓存形状；parity 保留（2026-10-04，wrapper 的 rotary_emb 是 HF 自带的） |
| `models/common/scatter_add.py` | `ops/scatter_add.py` | 0.711 | |
| `models/common/moe/dispatcher.py` | `models/common/token_dispatcher.py` | 0.441 | 2026-09-25 起含 `TorchAOTokenDispatcher` 可选导入适配层（torchao `permute_and_pad` 委托，未装 loud-raise）；DeepEP/HybridEP 保持登记缺口，见 D 表 |
| `parallel/activation_checkpoint.py` | `distributed/activation_checkpoint.py` | 0.374 | **FullAC + SelectiveAC + MemoryBudgetAC 已移植**（后者按上游语义设 `torch._functorch.config.activation_memory_budget`，需 compile，torch 无该 knob 时 loud-raise）；RegionAC 已接入（2026-09-29 二十五次增量：`region_ac` + `parallel/remat_regions.py`，声明通道以 HF block 的 `nn.Linear` FQN 结构等价替代上游 `Module.configure_remat_regions`，受限项只有上游自带的 torch_remat 需 torch ≥ 2.10，apply 期 loud-raise） |
| `parallel/fully_shard/fsdp.py` | `distributed/fsdp.py` | 0.815 | 多轴 mesh 重建、HF decoder 与 MoE placement 是 llmtuner 适配（2026-09-28 逐项复核，见审计“八次增量”） |
| `parallel/parallel_dims.py` | `distributed/parallel_dims.py` | 0.772 | llmtuner 扩展 world/loss/sparse mesh 视图，不能按旧 A1 结构覆盖；`build_parallel_dims` / `build_mesh` 自 `accelerator/mesh.py` 并入，上游无单一对应物（mesh 逻辑散在 `distributed/parallel_dims.py` 与 `trainer.py`） |
| `parallel/pipeline_parallel/pipeline.py` | `experiments/transformers_modeling_backend/pipeline.py` | 0.686 | `None` → `nn.Identity`；每 stage 追加 `rotary_emb`；stage 内 layer 保留原始索引（不重新编号），避免多 stage state-dict FQN 冲突 |
| `parallel/tensor_parallel/linear.py` | `models/common/async_linear.py`（原 `distributed/linear.py` → e72fd863d 搬入 dist_gemm.py → 9e159aed7 改名，数学不变） | 0.511 | 保留 fused/fallback 数学意图，但运行时上下文和 autograd 形状已适配 llmtuner |
| `accelerator/collectives.py` | `distributed/utils.py`（vendored `set_pg_timeouts` 与 EP 感知 `clip_grad_norm_` 两个符号） | 部分 | 2026-09-24 从 `parallel/` 迁入 `accelerator/`；EP 裁剪按物理本地 expert 参数适配（免 DTensor "ep" 轴断言）；同日复核后由 C 改标 A2 |

## B —— 适配层（读意图，不要抄形状）

这几个是 **torchtitan 每个模型一个文件** 的那种东西的**替代品**。照搬它们的形状会破坏
分片契约。

| llmtuner | 替代掉的上游 | ratio |
| --- | --- | --- |
| `models/hf/model.py`（2026-09-28 十八次增量随上游命名；+ `models/hf/factory.py` 构建侧、`models/hf/flops.py` 算术侧） | `experiments/transformers_modeling_backend/model.py` 的包装层；上游另有 `models/*/model.py` 各一份 | 0.059 |
| `models/hf/state_dict_adapter.py` | `experiments/transformers_modeling_backend/state_dict_adapter.py`；llmtuner 更强：读 safetensors index 做 missing/unexpected 严格校验；上游的 `hf_to_titan_moe_state_dict` 转换对因 llmtuner EP swap 直接搬运 HF 权重（无第二 key 布局）而不需要 | — |
| `parallel/parallelize.py`（2026-09-26 文件名对齐上游，原 parallelize_hf.py） | `experiments/transformers_modeling_backend/parallelize.py` + 各 `models/*/parallelize.py` | 0.089 |
| `parallel/tensor_parallel/tp.py`（+ `apply.py` 入口） | 各模型 TP plan；上游的 TP 声明层已随 DTensor 后端迁到 `protocols/sharding.py` + 各模型 `*_sharding.py`，旧的 `distributed/tensor_parallel.py` 于 `7e7f271e0` 删除。llmtuner 是**手写 plan realizer**，对应上游的声明式 `_sharding_config` 面（逐项对应见本文「TP/SP 对齐结论」） | 0.056 |
| `parallel/expert_parallel/apply.py` + `swap.py` | `experiments/.../moe_replacement.py` + 各模型 EP parallelize；llmtuner 搬运 HF 权重而非重新初始化 | 0.036–0.146 |
| `parallel/fully_shard/apply.py` | 各 `models/*/parallelize.py` 的 FSDP driver；HF 五部件适配 | 0.155 |
| `parallel/pipeline_parallel/apply.py` | `distributed/pipeline_parallel.py`；llmtuner 直接消费 HF stage 部件 | 0.130 |
| `trainer/trainer.py` | `trainer.py`，基本重写 | 0.065 |
| `config/`（顶层配置包） | `config/configs.py` | 0.189 |
| `trainer/train.py` | `train.py` | 0.186 |
| `models/common/moe/experts.py` | `models/common/grouped_experts.py` + `models/gpt_oss/moe.py` | 0.119 |
| `utils/gc.py` | `tools/utils.py` 的 GC helper，去 structured logger | 0.211 |

**注意 mesh 构建**：原 `accelerator/mesh.py` 已并入 `parallel/parallel_dims.py`，上游没有
单一对应物——mesh 逻辑散在 `distributed/parallel_dims.py` 和 `trainer.py` 里。整文件按
A2 分类（见上表），mesh 构建这一段记在该行的"改写点"里，不另占一个分类行。

## C —— llmtuner 独有（不要对齐上游）

| llmtuner | 说明 |
| --- | --- |
| `accelerator/spmd_context.py` | `spmd_types` pip 包的**独立活跃适配层**，由 trainer 和 `models/common/*` 使用；2026-09-24 从 `utils/` 迁入 |
| `parallel/context_parallel/apply.py` | 0.058；CP 的编排层，上游无对应文件 |
| `parallel/context_parallel/cp_kernel.py` | 0.051；llmtuner 独有的 CP flex kernel |
| `parallel/context_parallel/input_shard.py` | 0.078 |
| `utils/logger_utils.py` | 0.070，上游无对应；2026-09-24 起全仓模块 logger 统一经 `get_logger`（handler 挂模块 logger，rank 过滤在发射时判定，修掉了"import 时 rank 未知"的旧缺陷） |
| `accelerator/monitoring.py` | 与 `tools/utils.py` 0.107，独立实现（含 `get_peak_flops`）；2026-09-24 从 `utils/` 迁入 |
| `components/checkpointer/checkpoint_keys.py` | 上游无 |
| `accelerator/device.py` | 上游无（0.382 是噪音，命中实验目录）；2026-09-24 从 `utils/` 迁入 `accelerator/` |
| `models/common/activation.py` | 与上游同名但不同源；公式由 llmtuner 自持，不能按 A 类覆盖 |
| `models/common/embedding.py` | 与上游同名但不同源；包含 llmtuner 的 vocab-shard 契约；Embedding 类 parity 保留（2026-10-04，HF 自带 tok_embeddings；vocab-shard 公式由 components/loss.py 自持） |
| `datasets/random_data.py` | 合成语料，上游无 |
| `datasets/build.py` | 工厂；上游把 `build()` 放在 config 上 |
| `accelerator/dist.py` + `accelerator/dist_utils.py` | 2026-09-24 加入：vendored 自 OpenMMLab `mmengine.dist`（**不是 torchtitan 来源**），已去 mmengine 化，设备谓词与后端表统一由同包的 `accelerator/device.py` 提供；不进 trainer 装配路径。2026-09-28 复核（十次增量）：只按「能力缺口」对照上游 `distributed/utils.py`，结论是**无缺口**——上游的 `dist_sum`/`dist_max`/`dist_mean`/`dist_sum_tensor` 在 llmtuner 侧是 `all_reduce`（调用点 clone + in-place），`set_pg_timeouts`/`clip_grad_norm_` 已迁入 `accelerator/collectives.py`，仅有的 `init_distributed`/fake 后端差异已单列于 D 表 |
| `trainer/seed.py` | 2026-09-24 加入：上游 `distributed/utils.py::set_determinism` 的 distinct-seed 派生公式的纯函数提取（仅该项，非全文件移植）；DTensor RNG tracker 不移植。2026-09-28（十次增量）补齐该函数剩余的两个可移植件：`PYTHONHASHSEED = str(seed % 2**32)`（为后续 spawn 的 dataloader worker 而设）与 `detect_anomaly`（`torch.autograd.set_detect_anomaly(True, check_nan=False)` + 上游同文告警），落点为 `Trainer._seed_everything` 与 `TrainingConfig.detect_anomaly`；不移植的仍是只服务上游自有栈的两件（DTensor mesh-aware RNG tracker、flex-attention 确定性内核调优） |

**已清理悬空链**：`parallel/sharding.py` 与 `parallel/spmd_shims.py` 没有运行时消费者，
已在 2026-09-21 一并删除。`accelerator/spmd_context.py` 是独立活代码，不在删除组内。TP 的
活跃实现继续是 `parallel/tensor_parallel/tp.py` 的 plan 引擎。

**2026-09-23 死代码清理**（第二轮排查后删除，均为全仓零调用者，含 tests/examples）：
`parallel/context_parallel/primitives.py`（曾列为 B 类，等价实现早已并入
`cp_kernel.py`；上游 `models/common/cp_attention.py` 的语义对照因此直接落到
`cp_kernel.py`）、`models/common/flex_kernel.py`（`HFFlexKernel` 从未实例化，其
`_sharding_config` 兼容字段随之消失）、`models/common/nn_modules.py`（仅被
`__init__.py` 再导出的 nn 别名）、`parallel_dims.py` 的 7 个未用 API
（`unfold_dp_axis*`、`get_dense_tp_mesh`、`resolve_mesh`、`get_activated_mesh`、
`world_mesh`、`fsdp_enabled`、`seq_len_divisor`）、`apply_fsdp_to_multimodal_encoder`
（上游 2026-09 由 `apply_fsdp_to_vision_encoder` 改名；服务于 `models/common/multimodal.py`
的 vision tower，llmtuner 的 `models/common/multimodal.py` 只有 span/gather 融合算子、
没有编码器模块，故无消费者）、`ParallelConfig.backend` 字段（backend 由设备类型推导，不再有环境变量覆盖）
及若干零散项。明细见审计记录。

## D —— 真正缺失

| 上游 | 影响 |
| --- | --- |
| `distributed/compile.py` | **已移植**（2026-09-24，批 8）：逐 block compile、async TP `_micro_pipeline_tp`、`regional_inductor`、`capture_scalar_outputs` 四件全部落 `llmtuner/parallel/compile.py` + `CompileConfig`，见下"已从 D 移除" |
| `models/common/moe_sharding.py` | **部分移除**（2026-09-25）。其载荷 MoE-under-TP 已在 `parallel/tensor_parallel/tp.py` 落 B 类适配：HF plan 的 `packed_colwise`/`packed_rowwise`/`moe_tp_experts` 规格不再 raise，专家权重沿 F 维原地切分、router Replicate、块边界 AG/RS 对偶 collective；**tp×ep 同日起按上游语义放行**（TP 只切 dense、EP 独占 routed 专家沿 E 切、router Replicate，`apply_tp` 在 ep>1 时把块留给 swap，专家梯度排除由 `tp_sharded_param_ids` 统一判定；shared-expert×tp 保持 loud-raise；tp×ep×cp 自 2026-10-02 起放行（上游 release 套件实测 MoE FSDP+TP+EP+CP，对齐通过；真多卡数值等价测试 tests/integration_tests/tp_ep_cp_equivalence.py 待 torch≥2.12 多卡复跑）。 同日 pp×ep / pp×cp 解锁（上游 native 路径按 model part 装配、sparse mesh 带 pp 轴；dense CP+PP 在 release CI）：`pp_cp_ep` 裁决删除，等价脚本 pp_ep_equivalence.py / pp_cp_equivalence.py 待复跑。声明层+装配层就位并有 CPU 单测，但真多卡前后向等价性**环境未覆盖**（本机 torch 2.2.2 的 gloo 可用，缺的是模型/并行层所需的新 API，见"版本与漂移"），待 torch≥2.12 多卡复跑后方可视为完整移除。见下"已从 D 移除（部分）" |
| `components/optimizer/ema.py`（2026-09 新增，515 行） | **已移植**（2026-09-24，`llmtuner/components/optimizer/ema.py`）：在线 EMA 模型平均，config/trainer/checkpointer 三侧接线完成，见下"已从 D 移除" |
| `components/optimizer/optimizer.py` 的 `implementation="fused_opt_states_bf16"`（+ `_register_bf16_optimizer_state_hook`） | **登记缺口**（2026-09-28，components 批次 optimizer 走查）：上游第四种实现模式用 Adam 的 step pre-hook 预建 bf16 `exp_avg`/`exp_avg_sq`（fused CUDA 核据此走 fp32 参数+bf16 状态的混合精度路径，省一半优化器状态显存），再用 `register_load_state_dict_post_hook` 在 DCP 载入后把被 torch 转回参数 dtype 的状态重新降为 bf16。llmtuner 的 `implementation` 只声明 `fused`/`foreach`/`for-loop`（配置期即 Literal 拒绝，装配期 `_build_impl_kwargs` 再兜一道 `ValueError`）。不移植的理由是**不可验证**：它的全部价值来自那个 CUDA fused 核，本机无 CUDA 也无从复现上游的显存收益；同时它改写 checkpoint 里 Adam 状态的 dtype，属续训兼容敏感面，盲写风险高于收益。解锁条件：CUDA 目标设备 + 确有优化器状态显存诉求；届时实现要点即上面两条 hook（上游 `components/optimizer/optimizer.py:339` 起） |
| `components/optimizer/optimizer.py` 的 `optimizer_factory_kwargs_by_name` | **登记缺口（无消费者）**（2026-09-28，同上）：上游用它把「实例级对象」——per-parameter compute metadata、通信 bucket 规格——按 optimizer 名传进工厂，而它现在的两个消费者（`DistMuon`、Float8 系优化器）都在 llmtuner 裁剪面内（前者属已在 C 类登记为范围外的 `distributed/flex_shard/`）。按本仓"有调用者再补"的口径不预置字段；补 `DistMuon` 时一并加 |
| `components/optimizer/optimizer.py` 的 `init_cache_state_dict` | **故意删除，不是缺口**（2026-09-29，二十二次增量）：上游该方法在基类是 `pass` no-op，服务 TorchFT 容器（其子类覆写）与 TorchFT 训练循环的无条件调用；llmtuner 无 TorchFT（D 表已登记裁剪），移植物里那份 no-op **全仓零调用者**，属无效抽象，已删。若将来接入 TorchFT，补回一个 `pass` 方法即可 |

| Ulysses CP × varlen/packed（baff3c681） | **已移植**（2026-09-25，批 5）：`apply_cp` 不再 fail-fast，wrapper 全长透传文档 mask、kernel 按 mask Q 长度分派，见下"已从 D 移除" |
| 多轮对话 SFT 的 renderer 路径（4a0d8dab3） | **已适配为可选路径**（2026-09-25，§9.1 第 12 项）：不引入硬依赖、不复制 Configurable 外形。`datasets/text/renderer.py` 为可选导入适配层（`build_chat_renderer` + `RendererTokenizerWrapper`），`ChatProcessor(renderer=...)` 走多-turn renderer 分支，`--chat_renderer`/`--messages_field` 接线 `local_jsonl_sft`；未装 `renderers` 时启用 loud-raise（ImportError 带安装指引），默认关闭逐位不变。真实库数值**未验证**（本机无 renderers，单测以 fake 模块覆盖接口与 mask 移位语义）；解锁条件：pyproject 加 optional extra `renderers==0.1.11` 后装包复跑 |
| `models/common/moe/dispatcher.py` 的 DeepEP/HybridEP 两个 dispatcher | 登记缺口（2026-09-25，§9.1 第 13 项）：CUDA-only（`deep_ep`/`hybridep` 内核 + GB200/NVLink72 假设）且 dispatch/combine 经上游 `distributed/deepep/` wrappers（1155 行）驱动，可选导入无法忠实表达契约，故不 vendor；`ParallelConfig.ep_token_dispatcher="deepep"/"hybridep"` 配置期 NotImplementedError（含解锁条件），swap 入口防御性同语义。解锁条件：vendor 上游 wrappers + pyproject 加 CUDA-only optional extra + CUDA 目标设备复跑数值。`AllToAllTokenDispatcher` 满足同一 dispatch/combine 契约 |
| `models/common/moe/dispatcher.py` 的 `TorchAOTokenDispatcher` | **已适配为可选导入适配层**（2026-09-25，§9.1 第 13 项）：torchao 不进 pyproject、不复制上游 Config 嵌套。`TorchAOTokenDispatcher(num_experts, top_k, pad_multiple)` 继承 `AllToAllTokenDispatcher`，仅 `_permute`/`_unpermute` 改委托 torchao `permute_and_pad`（expert-major 重排 + 每组 pad 到 `pad_multiple`，EP=1 本地 padded permute 路径一并移植），构造期 lazy import，未装 torchao loud-raise ImportError（带 `pip install torchao` 指引）；`ParallelConfig.ep_token_dispatcher="torchao"` + `ep_torchao_pad_multiple`（默认 16=FP8）接线 `apply_ep` → swap，默认 `alltoall` 逐位不变。数值**环境未覆盖**（本机无 torchao/CUDA，单测以 sys.modules fake 覆盖 sentinel-row padding 契约与 EP=1 combine 等价性）；解锁条件：CUDA 目标设备装 torchao 复跑 |
| DSA（DeepSeek sparse attention）的稠密 additive mask 路径 | **已移植（2026-09-28，十九次增量）**：`models/common/attention/masks.py::build_dense_attention_mask` + `models/hf/model.py::get_attention_masks` 的 DSA 分支，与上游 `_build_dense_attention_mask` 逐行等价（causal / block_causal 两种 `attn_mask_type`）；flex 照旧运行并按 mask 类型当 `score_mask`（HF 集成分支）。**未覆盖**：真 DSA 模型端到端（transformers 的 DSA 家族需 torch≥2.4 才能建模型）与 CP×DSA / PP×DSA（两者显式拒绝；PP 拒绝于 2026-10-07 补登：非首 stage 的 embedding 已置 Identity，稠密 mask 建不出） |
| 单进程模拟多卡的 debug 后端（`comm.backend` 的 `fake` / `real_pp_fake_spmd`，2026-09 新增的 `DistributedTopology`） | 登记缺口（2026-09-28，parallel_dims 走查）：上游用 torch 的 `backend="fake"` 建一个"逻辑世界"，可在单进程内模拟任意 world_size 的 mesh（`real_pp_fake_spmd` 再叠一个真实 PP 组，供 PP 边通信）；llmtuner 只有 `world_size == 1 → parallel_dims is None` 与真多卡两条路，单机并行验证走 gloo + torchrun 集成测试。解锁条件：torch 提供 `backend="fake"`（本机 2.2.2 无此 backend）+ 决定给 `accelerator/dist_utils.py` 加一条 debug 后端；届时 mesh 构造无需改动（`build_mesh` 已是 `world_size` 驱动）。2026-09-28 十九次增量曾按该方向实现（`init_distributed` + `NGPU`/`FAKE_PP_RANK`），**按用户要求整条撤下**：fake 后端与 `DistributedTopology` 都不引入，保留本条为登记缺口；同轮的另一项（DSA 稠密 mask）不受影响 |
| vocab-sharded `lm_head` + 端到端 vocab-parallel loss | **D 类，两步走，第一步已完成（2026-09-27）**。第二步（模型侧）未实现：上游 HF 路径把 `lm_head` 的 weight/bias 沿 vocab 维 `S(0)` 切、输入从 sequence-parallel gather 回全长、输出 `S(-1)`（vocab 分片），core `cross_entropy_loss` 检测到分片后走 vocab-parallel CE（`hf_sharding.py` 的 `lm_head` 段）。llmtuner 仍把 HF plan 的 `colwise_gather_output` 解析为 None、`lm_head` 保持复制（`tensor_parallel/tp.py::resolve_plan`）。**第一步（loss 侧接线，已完成）**：`Trainer._loss_vocab_kwargs()` + `HFTransformerModel.vocab_size` 把 `tp_group`/`global_vocab_size` 送到四个调用点（`Trainer._loss_sum`、`chunked_lm_head_cross_entropy`、PP `_scalar_loss_fn`、Validator），`components/loss.py` 按形状分派，因此复制 head 下逐位不变；第二步（vocab-shard realizer + head 已分片但 loss 未被告知时 loud-raise）在多卡环境复跑后再做。

**已从 D 移除（部分）**（2026-09-25）：`models/common/moe_sharding.py`——上游该文件是
声明层：`ShardingConfig` 声明 router 参数 TP Replicate、routed 专家权重仅在 EP 开时
沿专家维 E 取 placement（DP_REPLICATE/EFSDP 为 R,EP 为 S(0)），由上游 Module 协议
的 parallelize 引擎消费。llmtuner 按 B 类语义适配，不复制其 Config 协议：MoE-under-TP
的声明改由 HF tp_plan 的 `packed_colwise`/`packed_rowwise`/`moe_tp_experts` 规格承载
（`resolve_plan` 解析为 None），执行落在 `parallel/tensor_parallel/tp.py` 的结构路
径——`shard_experts_for_tp`（`down_proj (E,D,F)` 切 dim 2，`gate_up_proj (E,2F,D)`
gate/up 两半各自切 dim 1,router 不动）+ `TPMoeSequenceBoundary`（`__class__` swap
安装块边界 sequence all-gather / reduce-scatter，与 dense TP 同一对偶契约，序列维
-2)。梯度语义：边界 collective 的注册反向互为对偶；router 权重 Replicate，梯度由
`_allreduce_replicated_tp_grads` 求和；被切专家参数经块上 `tp_sharded_param_ids`
从该归约排除。state_dict FQN 不变、tp=1 逐位不变。**tp×ep（同日第二段）**：按上游
语义放行——TP 只切 dense,routed 专家由 EP 独占沿专家维 E 切，router Replicate;
`apply_tp` 在 `cfg.ep > 1` 时跳过 MoE 块扫描/分片/边界安装（块留给 `apply_ep`
swap,swap 后的原生 MoE 直接消费/产出 T/tp 序列分片，即上游 ep+sp 的
sequence-parallel 布局，无边界 collective);trainer 的排除判定抽为模块级
`tp_sharded_param_ids`（三类：dense TP realizer、MoE-under-TP 的 F 分片、EP 的
`GroupedExperts` E 切片；EP 专家梯度按 rank 完备，跨 TP 求和会混不同专家的梯度）。
组合矩阵终态（config 期校验在各 config `__post_init__`，跨层裁决单一来源 `parallel/matrix.py`；2026-09-26 收窄）：tp>1×ep>1（cp=1）放行；tp>1×ep>1×cp>1 在
`ParallelConfig.__post_init__` fail-fast（未验证）;shared-expert 块 ×tp:gate/up/down 布局放行（2026-10-02，`shard_shared_expert_for_tp` F 维分片，边界内无 collective；非标准布局如 Qwen2Moe 门控仍 loud-raise)；tp×ep×shared 在 swap `convert_block` 处 loud-raise。2026-10-06 修复：dense realizer 路径曾把 HF plan 声明的 `shared_experts.*_proj` 再包一层 ColumnParallelLinear/RowParallelLinear，造成 F/tp² 双重切分（静默错值）；现 MoE 块内部一律排除出 dense targets;plan 声明 MoE 规格但
探针找不到块（ep=1）loud-raise;GPT-OSS 布局 loud-raise。aux loss / padding-mask
LB / quantile hook 的归约轴此前已按 ep_enabled 含 tp 书写，放行后不重复计数、无需
改动。测试
`tests/unit_tests/cpu/parallel/test_tp_moe.py`：规格解析、分片重建、单进程
partial-sum 等价（reduce-scatter 求和的算术内容，无进程组）、FQN 稳定、幂等、
ep>1 时 apply_tp 原样放行 MoE 块、shared-expert×tp（gate/up/down）放行、tp×ep×shared 拒绝、梯度排除规则、组合
矩阵各格。**未覆盖**：真多卡 forward/backward 等价（本机 torch 2.2.2 无
DTensor/spmd 执行栈，gloo 下功能 collective 未验证）——待 torch≥2.12 多卡复跑。

**已从 D 移除**（2026-09-25 批 5 移植）：Ulysses CP × varlen/packed（上游
baff3c681）——上游形态是把 Ulysses 的 token↔head resharding 提为
`UlyssesCPInnerAttention` 共享层，`UlyssesCPVarlenInnerAttention` 借 MRO 把
`super().forward` 派发到 `VarlenInnerAttention`；varlen 元数据（cu_seqlens）不随输入
分片（`cp_shard` 把 `attention_masks` 摘出再原样放回），因为 all-to-all 后每个 rank
都持有全长 token 流。llmtuner 按 B 类语义适配、不复制类层次：packed 语料的"varlen
元数据"在 HF/flex 集成里是烘进 BlockMask 的文档结构，因此
`hf/model.py` 的 `preprocess_inputs` 在 `ulysses` 策略下把全长文档 mask **不 Q 分片**透传
（`set_cp_mesh` 新增 `strategy` 闩锁），`CPFlexKernel._forward_ulysses` 按 mask 的 Q
长度 == 全长序列 分派：全长即用传入 mask，否则照旧重建全长 causal mask。决策全部
config/shape 驱动、rank 对称。`apply_cp` 对 ulysses×`block_causal` 的 fail-fast 移除，
ulysses×load-balancer 拒绝与 heads÷(tp×cp) 校验不变；不启用 varlen 的稠密路径逐位不
变（kernel 重建的 causal mask 与 wrapper 同源同参）。测试：
`tests/integration_tests/cp_ulysses_varlen_equivalence.py`（2-rank gloo：seam 全长
mask 契约、分片 logits/loss == 单卡稠密 block-causal 参考、SDPA 内层下前后向
collective 对偶与 varlen mask 梯度等价、non-vacuity 反证）——本机 torch 2.2.2 无
flex 模块，**环境未覆盖，待 torch≥2.12 + 目标设备复跑**；单测补
`test_ulysses_packed_is_accepted_and_the_strategy_is_latched`。上游 GPT-OSS 的
Ulysses 拒绝（per-head sinks 只走 TP 分片）不适用：llmtuner 尚无 GPT-OSS 支持。

**已从 D 移除**（2026-09-24 批 8 移植）：`distributed/compile.py`——四件互相独立的
能力全部落 `llmtuner/parallel/compile.py::apply_compile`，由
`config/training.py::CompileConfig`（`training.compile_config`，默认全关）驱动，
装配顺序不变（AC 之后、FSDP 之前；PP 下每 chunk 由
`parallel/parallelize.py::parallelize_hf_transformers` 过同一函数）：

* **逐 block compile**（`per_block=True`）：每个 decoder layer `block.compile(
  backend=..., fullgraph=True)`（`Module.compile` 就地，state-dict 键与 FSDP 包装
  不变）；默认 False 保持整体 `torch.compile(model, backend="inductor")`，与旧
  路径逐位一致。
* **async TP**（`enable_async_tensor_parallel=True`）：设
  `torch._inductor.config._micro_pipeline_tp` 并为 TP group 注册 symmetric
  memory（按 group 名去重，PP 每 chunk 重入安全）。配置校验期（
  `LLMTunerConfig.__post_init__`）拒绝 无 compile / tp=1 两种组合；装配期对
  无 TP mesh、torch 无 `_micro_pipeline_tp`、无 `enable_symm_mem_for_group` 三种
  情形 loud-raise，不静默跳过。
* **regional_inductor**：flex 只有 inductor lowering，故非 inductor backend 下
  flex 模型必须 scoop。`backend="aot_eager"` 且模型走 flex（wrapper 新 property
  `uses_flex_attention`）时用 `torch.fx.passes.regional_inductor` 包
  `aot_autograd`；annotation 落 `hf/model.py` 的 `flex_attention_hf` 的
  `maybe_regional_inductor({})`（默认 nullcontext，inductor/eager 路径零开销）。
  flex 模型配其他非 inductor backend → `ValueError`；torch 无 regional_inductor
  → `NotImplementedError`；sdpa 模型 backend 原样透传。inductor_configs 传空
  （llmtuner 走 HF 的 flex 集成，不带上游 FlexInnerAttention 的 autotune 配置）。
* **capture_scalar_outputs**：按上游条件在编译的 model part 含 token-choice MoE
  block（`iter_moe_layers` 非空，即 EP swap 后的 llmtuner MoE 栈）时设
  `torch._dynamo.config.capture_scalar_outputs=True`；dense 模型不动该全局量
  （逐位不变）；torch 无此 knob 时 loud-raise。

未移植（登记）：上游同文件的 `skip_fwd_side_effects_in_bwd_under_checkpoint`
（AC+compile 的 side-effect 重放分歧开关，llmtuner 未遇到其失败场景，需要时按上游
注释补）与 `FakeTensorMode.__init__` 的 `torch.compiler.disable` monkeypatch
（修上游 pytorch#178887，等上游修复即废弃的临时措施）。`CompileConfig.components`
未移植（llmtuner 只编译 model，loss 无 compile 通路）。

**已从 D 移除**（2026-09-24 批 4 移植）：`pipeline_with_first_stage_modules`——
多模态 first-stage 模块并入 stage 0，落为 `apply_pp` 的可选关键字参数
`first_stage_module_fqns: Sequence[str] | None`（默认 None，默认时切分与
state-dict 键逐位不变）+ `parallel/pipeline_parallel/apply.py::
prepend_first_stage_modules`（仅作用于自动生成的切分，把存在的 FQN 按序前插
stage 0；已被切分占有的 FQN 与重复 FQN loud-raise，缺失模块跳过；显式
`module_fqns_per_model_part` 给定时忽略并告警，与上游委托语义一致）。配套改动
`split_model_into_stages`：wrapper `named_children()` 不呈现的额外顶层模块
（注册在 decoder 旁的多模态编码器等）在非属主 stage 上一律置 `nn.Identity`
——上游"pruned on other stages"语义；装五部件的容器（wrapper 内层 HF 模型）
通过"包含已呈现部件"判定跳过，绝不置空。不变量：stage FQN 稳定（并入模块保持
原名顶层子模块，optimizer/checkpoint 键不跨 stage 冲突）、五部件契约与
`named_children()` 语义不变。当前无真实消费者（多模态 vision encoder 路径未
接线），属"能力就位 + 契约测试"；多 stage 真跑待目标设备（torch≥2.12）复跑。

**已从 D 移除**（2026-09-24 批 4 移植）：validation 循环——上游
`components/validate.py::Validator` 落 `trainer/validate.py`（2026-09-26 文件名对齐上游，原 validation.py）
（`Trainer.validate`/`should_validate`/`check_validation_feasibility` 的
薄委托背后）+
`config/training.py::ValidationConfig`（`training.validation_config`，默认
None 关闭，关闭时训练循环逐位不变；programmatic-only，同 `ema_config`）。
语义对齐：eval 模式 + `no_grad`、结束恢复 train；loss 按全局有效 token 数
归一化，token 计数走 dp mesh、loss 和走 dp×cp×tp loss mesh（与训练 loss 同一
对 mesh、同一归一化）；每次 pass 新建并关闭临时 dataloader（`repeat=False`
对应 `steps=-1`），不进 checkpoint、不动 `ntokens_seen`；训练循环内调用点在
checkpoint save 之后、profiler.step 之前，与上游同序。两条上游 bug fix 一并
移植：零 batch / 零有效 token 报 `ValueError`（上游 6c2dadbb3），dp>1 拒绝
`steps=-1`（上游 90b25912f，在 trainer 构造期、真实 dp degree 已知后检查）；
另对 random 无限语料的 `steps=-1` 同样 fail-fast。PP × validation 2026-10-02 起支持（schedule eval 驱动，见 symbol guide）；此前未支持：
llmtuner 的 PP loss 计算内嵌在 schedule 的训练步里，无 `pp_schedule.eval`
对应的 eval 通路，构造期 `NotImplementedError`（loud-raise，不静默跳过）。

**已从 D 移除**（2026-09-24 批 2 移植）：quantile-balanced MoE routing——
`QuantileBalancedTopKRouter` + `QuantileBalancer` + `register_moe_quantile_balancing_hook`
（biased top-(K+1) cutoff、1000-bin 直方图、分位数 mean-centred 覆写 bias），与
sign-based bias 互斥（同层 raise、跨层 hook raise），经 `ParallelConfig.
moe_quantile_balancing` 启用；MoE padding-mask 负载均衡——`MoE.set_padding_mask`
一次性暂存通道（HF layer 签名穿不了 mask），mask 只过滤负载均衡统计
（tokens_per_expert、aux loss f/p、quantile 直方图），不动 routing 执行，无 mask
逐位不变；CP/TP 由 `shard_padding_mask_for_cp/tp` 与 token 流同序切分。

**已从 D 移除**（2026-09-24 批 3a 移植）：在线 EMA——`llmtuner/components/optimizer/ema.py`
（515 行上游 `components/optimizer/ema.py` 的语义移植）：`EMA` 复用
`OptimizersContainer` 的 flat FQN state-dict 契约，`decay = 2**(-1/(half_life_fraction*num_updates))`
动态计划或固定 decay，firing count 由 trainer step 推导（resume 不重置 decay），
`step_bias` 支持阶段重编号，`start_step`/`update_every_n_steps` 门控，可选
`buffer_patterns` 浮点 buffer 跟踪（整型 buffer 拒绝）；checkpointer 增加 `ema`
state 键与 `_find_load_step(max_step=)`，llmtuner/config 完成三侧接线。上游
DTensor unwrap/rewrap 与 CUDA `torch._foreach_lerp_` 专项未移植（llmtuner 的 FSDP2
张量本身就是 DTensor，容器 state dict 直接交给 DCP）。

**已从 D 移除**（2026-09-24 批 1 移植）：`CastLinear`——lm_head compute-dtype
变换，落 `models/common/cast_linear.py`（`nn.Linear` 子类，state-dict FQN 不变），
经 `ModelConfig.compute_dtype` 启用，默认关闭；router `_debug_force_load_balance`
——落 `TokenChoiceTopKRouter` 同名构造参数，round-robin 语义与上游逐字一致；
PP per-stage seed——`trainer/seed.py` 的 `derive_distinct_seed`（上游
`distinct_seed_mesh_dims=["pp"]` 同公式），trainer 在 `pp_enabled` 时按 stage rank
偏移，pp=1 逐位不变；DTensor RNG tracker 不移植（初始化走 materialize 路径）。

**故意删除，不是缺口**（不要"补回来"）：`components/quantization/`、
`structured_logger/`、`protocols/`、`configurable.py`。

**已从 D 移除**：`distributed/activation_checkpoint.py` 的 `SelectiveAC`——于
2026-09-21 移植（见 A2）；`MemoryBudgetAC`——于 2026-09-24 移植：它没有策略代码，
只是设一个 `torch._functorch.config.activation_memory_budget` 全局量让 compile
partitioner 做取舍，落为 `training.activation_checkpoint_mode='memory_budget'` +
`MemoryBudgetACConfig`（budget ∈ [0,1]，同上游校验），按上游 trainer 校验在
compile 关闭时 fail-fast；torch 无该 knob（本机 2.2.2 即如此）时 loud-raise 而非
静默设一个没人读的全局量；上游的 `visualize_memory_budget_pareto`（往 dump folder
倒 SVG）未移植，llmtuner 的 AC 路径没有 dump folder 概念。该文件仍有一处未移植，是
依赖而非删减：`RegionAC` 需要 `torch_remat`——**已于 2026-09-29 二十五次增量接入**
（`mode='region'` + `RegionACConfig` + `parallel/remat_regions.py`），受限条件只剩上游
那一个：`torch_remat` 要求 torch ≥ 2.10，本机 2.2.2 无法 import，故 apply 期
loud-raise ImportError 并把 torch 版本与安装命令一并写进文案；上游的
`Module.configure_remat_regions` 声明通道以"HF block 的 `nn.Linear` FQN"结构等价替代，
理由与取舍见 `remat_regions.py` 与 `wrap_region` 的 docstring。原先同列的
`disable_dynamo_lru_cache`（SAC+PP 的重编译 workaround）已于 2026-09-27 随 pp×AC 对齐
一并移植，见下方六次增量。

**已从 D 移除**：`tools/validate.py`——上一版既写了它、又写"上游也没有这个路径，已从
表里移除"，自相矛盾。核实：上游 `torchtitan/tools/validate.py` **确实不存在**，这一行
没有意义，删掉。

## E —— 包面（`__init__.py` 与入口）

重组出口，不是移植内容：这些文件定义 llmtuner 的**公开 API 面**，上游对应物是同名
`__init__.py`（若存在）。改上游的导出列表时才需要看这里。全表 20 个（19 个
`__init__.py` + `__main__.py`），行数为 2026-09-27 实测。

| llmtuner | 行数 | torchtitan |
| --- | --- | --- |
| `__init__.py` | 35 | `__init__.py`（根面：`LLMTunerConfig` + `Trainer`） |
| `__main__.py` | 6 | `train.py` 的入口对等物 |
| `accelerator/__init__.py` | 73 | 上游无对应（PEP 562 懒加载包面） |
| `components/__init__.py` | 0 | 空文件 |
| `components/checkpointer/__init__.py` | 87 | `components/checkpointer/__init__.py` |
| `components/optimizer/__init__.py` | 39 | `components/optimizer/__init__.py` |
| `config/__init__.py` | 81 | 上游 config 包面（llmtuner 全量再导出 16 个配置类） |
| `datasets/__init__.py` | 64 | `components/data/__init__.py` |
| `datasets/multimodal/__init__.py` | 16 | `hf_datasets/multimodal/`（刻意不导入子模块） |
| `datasets/text/__init__.py` | 10 | `hf_datasets/`（刻意不导入子模块） |
| `models/__init__.py` | 0 | 空文件 |
| `models/common/__init__.py` | 53 | `models/common/__init__.py` |
| `parallel/__init__.py` | 49 | `distributed/__init__.py` |
| `parallel/context_parallel/__init__.py` | 27 | `distributed/context_parallel/__init__.py` |
| `parallel/expert_parallel/__init__.py` | 20 | 上游无对应（见 C 类） |
| `parallel/fully_shard/__init__.py` | 5 | 上游无对应子包（只再导出 `apply_fsdp`） |
| `parallel/pipeline_parallel/__init__.py` | 12 | 上游无对应 |
| `parallel/tensor_parallel/__init__.py` | 23 | 上游 `distributed/tensor_parallel.py` 已删除，后继是 `protocols/sharding.py` + 各模型 `*_sharding.py` 的声明面 |
| `trainer/__init__.py` | 27 | 上游无对应（配置再导出为兼容别名） |
| `utils/__init__.py` | 0 | 空文件 |

`__init__.py` 的 ratio 平均偏低（0.3 上下）是正常的——它们导出的是各自的公开面，不是
从上游抄结构。表里的数值是"文件行数"，不是 ratio。

**空文件**（0 行）：`components/__init__.py`、`models/__init__.py`、`utils/__init__.py`。

**两个子包**：`datasets/` 下按语料分 `text/` 和 `multimodal/`，其余 8 个模块平铺在
`datasets/` 根下。上游的 `hf_datasets/` 是 `components/data/` 的兄弟目录，llmtuner 曾用
`datasets/hf/` 镜像它（4 层，三个 0 字节的 `__init__.py`），先被溶解成单层，再于重组
时按语料分成两个子包。

划分依据是**内容**而非上游路径：`datasets/` 根下的模块与上游 `components/data/` 一一
对应，且都被两边共用（`loader.py` 默认 `TextCollator`、`multimodal/mm_collator.py`
复用 `collators.py` 的 `Collator`/`TrainerBatch`），所以不进任何一边；只有 `text.py`
和 5 个 `mm_*.py` 是语料专属。两个子包的 `__init__.py` 都**刻意不导入子模块**——惰性
导入契约靠这一点维持。

**上游路径对应关系因此不再一一成立**（`hf_datasets/multimodal/utils/image.py` 在
llmtuner 侧是 `datasets/multimodal/image.py`），本表的 llmtuner 列是唯一权威。

## TP/SP 对齐结论（2026-09-27）

本轮把 llmtuner 的 TP / sequence-parallel 与 TorchTitan `f35966713` 逐项核对。结论：
**数学与通信角色逐条等价，差异集中在"声明形态"、两处已修缺口与一处已登记的 D 类缺口**。

| 面 | llmtuner | 上游 | 判定 |
| --- | --- | --- | --- |
| 声明形态 | `parallel/tensor_parallel/tp.py` 的 `ShardingConfig` + `resolve_plan`（消费 HF `tp_plan` 的规格字符串） | `protocols/sharding.py::ShardingConfig` 挂在模块 `_sharding_config` 上，由 `Module._parallelize` 分发 | **等价（形态不同）**：两者都是"权重切分 + 激活布局"的声明，llmtuner 的词汇表来自 HF |
| colwise | `ColumnParallelLinear`：weight 切 dim 0，融合 all-gather 入、输出 feature-sharded | `hf_sharding.py:60 _hf_colwise_config`：weight/bias `S(0)`、out `S(-1)` | **等价** |
| rowwise | `RowParallelLinear`：weight 切 dim 1，融合 reduce-scatter 出（回到序列分片） | `hf_sharding.py:71 _hf_rowwise_config`：weight `S(1)`、bias `R`、out_src `P`、out_dst → SP placement | **等价**（上游 `P`→SP 的重分布正是 llmtuner 融合 RS 的位置） |
| 注意力边界 | `GatherSequenceFirst` + `ColwiseLinearNoGather`：父模块持有 gather，q/k/v 退化为 plain feature-sharded GEMM | `decoder_sharding.py:218 set_gqa_attention_sharding` + `_attach_flex_kernel`（SP 输入在注意力内部 gather 回 Replicate） | **等价** |
| 序列并行语义 | TP 即 SP：batch 先按 CP、再按 TP 切；`parallelism.enable_sequence_parallel=false` 直接 config-raise | `sp_enabled = tp_enabled and enable_sequence_parallel`（`parallel_dims.py:550`） | **有意分歧**：llmtuner 没有"激活全复制"的退化路径 |
| 序列切分顺序 | 先 CP（`models/hf/model.py:591`）后 TP（`:623`），TP 切在 CP 分片内 | `hf_sharding.py:52 _hf_sequence_parallel_placement()` = `PartitionSpec(DP, (CP, TP), None)` | **等价**：CP 外、TP 内的联合切分 |
| norm 权重 | q/k norm 保持复制（HF 4.57 起 plan 已不声明它们），梯度由 `Trainer._allreduce_replicated_tp_grads`（`trainer/trainer.py:470`）汇总 | `decoder_sharding.py:177 norm_config`：SP 时权重 `R`，"BWD AR 交给 FSDP" | **等价**（同 D14：上游归 FSDP、llmtuner 归 trainer，数值一致） |
| token 计数 / loss mesh | `trainer/batch.py:221` 计 `labels.numel() // (cp*tp)`；loss mesh 含 tp（`parallel/parallel_dims.py:220`） | loss mesh 只含 dp×cp（`parallel_dims.py:260`） | **耦合差异**：上游把 tp 的归约放进 vocab-parallel CE，llmtuner 的 head 是复制的、必须跨 tp 求和。两侧各自自洽，随 lm_head 缺口一同处理 |
| lm_head 与 loss | HF 的 `colwise_gather_output` 解析为 None → head 保持复制（全 vocab）+ 普通 CE；loss 侧参数已接线（按形状分派，复制下 no-op） | head `S(0)`/`S(-1)` vocab 分片 + core `cross_entropy_loss` 检测分片走 vocab-parallel CE | **D 类缺口，两步走的第二步未做**：loss 侧接线 2026-09-27 完成，head 真分片与"未接线即 loud-raise"待做，见上"D —— 真正缺失"表 |
| 注意力头整除 | 本轮之前只有 ulysses CP 路径检查 `% (tp*cp)` | `config/validation.py:150 head_shard_degree`：解析期即校验 `heads % (tp*cp)` | **本轮修复**：新增 `parallel/head_sharding.py`；`apply_tp` 查 `% tp`、ulysses CP 查 `% (tp*cp)`，合起来即上游那一次检查 |
| 未实现的 HF 规格 | `colwise_rep` / `rowwise_rep` / `local_*` / `gather` / `replicate` / `sequence_parallel` loud-raise | 这些是 DTensor 时代的 replicated-activation 布局，上游由 SPMD 声明承担 | **有意拒绝**：llmtuner 的 GEMM 是 SP 对偶 collective，没有全复制激活路径；本轮把报错改成指名 + 说明理由 |

**验证边界**：TP/SP 的数值等价需要多卡与 torch≥2.12（symm-mem、`spmd_types`、
`torch.distributed.pipelining`），本机（torch 2.2.2、CPU）不可达，可执行的只有声明层/
装配期单测与 2-rank gloo 等价性（design §8 第 10 项）。因此上表的"等价"是**代码级核对**
结论；多卡上的数值等价仍是待办，不能据本表声称已验证。

## 版本与漂移

- 本文最近一次人工审计工作树：llmtuner `11002c1`（+本轮改动），TorchTitan `b64103072`；
  详细验证记录见
  `llmtuner_torchtitan_alignment_audit_2026-09-23.md`（不在当前工作区）。
- 2026-09-26 增量审计：基线推进至 TorchTitan `9e159aed7`（审计时上游 HEAD 附近），
  llmtuner 工作树 HEAD `8c0ac4c`（+本轮文档改动）。`b64103072..9e159aed7` 间三个
  提交的处理结论：

  | 提交 | 结论 |
  |---|---|
  | `9e159aed7` TP projection 后端重构（#4704） | **语义已对齐，无代码动作**。通信角色不变量在 llmtuner 已成立：column 拥有 input collective（`ColumnParallelLinear` 融合 all-gather）、row 拥有 output collective（`RowParallelLinear` 融合 reduce-scatter）；共享输入多投影在父模块一次性 gather（`GatherSequenceFirst` + `ColwiseLinearNoGather`，同上游"父模块持有、子投影为 plain Linear"语义）。`_linear()` seam 服务 LoRA/量化（llmtuner 裁剪面，不移植）；`PartialBiasRowwiseLinear` 上游删除并并入 `RowParallelLinear`，llmtuner 同名类的 bias I→P 语义本就一致，保留（仅测试使用）。AsyncTensorParallelTransform 重写是上游 Module-registry 面的模块替换实现，llmtuner async TP 走 inductor `_micro_pipeline_tp` + symm-mem，机制不受影响；"转换后（LoRA/量化）投影不支持 async TP"的约束在 llmtuner 无对应面（两者均裁剪），不登记守卫。上游 `dist_gemm.py` 改名 `async_linear.py`，本文映射随之更新。 |
  | `847f98a6f` RegionAC AllToAll remat regions（#4837） | **RegionAC 侧本条已不再受阻**（2026-09-29 二十五次增量接入 RegionAC）；仍缺的是 **DeepEP** 那一半（CUDA deep_ep 核 + 上游 `distributed/deepep/` wrappers，D 表登记）。上游把 TokenDispatcher 变成 Module 是 remat region 的载体，llmtuner 的 dispatcher 不是 Module，故这条 region 声明在 llmtuner 无对应物——但可独立移植的语义（dispatch/combine 恒 SAVE）经核对**已在 llmtuner 成立**：selective AC 的 save set 含 `_c10d_functional.all_to_all_single`（`activation_checkpoint.py` 的 `comm_ops`），即 llmtuner AllToAllTokenDispatcher 用的原语；RegionAC 路径下的差别登记于此：region 词表只含 `nn.Linear`，该 collective 不在任何 region 内，因而随 block 整体重算（上游是把它声明成恒 SAVE 的 region）；要抹平需要给 dispatcher 一个 remat region 通道，属 D 表 DeepEP 那一半的解锁范围。 |
  | `090c0c931` graph_trainer none AC MemoryPolicy（#4476） | **实验目录，不适用**。`experiments/graph_trainer/` 无 llmtuner 对应面；等义语义 llmtuner 已有（`activation_checkpoint_mode='none'`）。 |
  上一轮审计（llmtuner `528dc9d` × TorchTitan `c6e416bbd`）引用的
  `llmtuner_torchtitan_alignment_audit_2026-09-21.md` 不在当前工作区。
- 2026-09-27 增量审计：基线推进至 TorchTitan `c8a3e7666`（审计时 HEAD），llmtuner 工作树
  `b6bbb33`（本轮改动落为 `142bb6d`）。`9e159aed7..c8a3e7666` 共 20 个提交、118 个文件，与本仓相关
  的只有 6 处（其余集中在 `rl/`、`quantization/`、`experiments/graph_trainer/`、
  `overrides/fused_mla.py`、`distributed/flex_shard/`、`config/transform/`，属范围外）。
  逐项结论：

  | 上游文件/符号 | 上游意图 | 分类 | llmtuner 处理 |
  |---|---|---|---|
  | `distributed/activation_checkpoint.py::FullAC` / `SelectiveAC` 的 `early_stop` `False`→`True`（#4836） | 性能：recompute 产出全部所需张量后即停，不再重放区域剩余算子，省 1-4% step time。上游 8×H100 实测数值不变（多数 exact，最大 loss 差 2.8e-6）、峰值 reserved 内存漂移 0；旧的 `False` 是上游 #1580 的 llama4 内存泄漏 workaround，已失效 | **A** | **已同步**：`parallel/activation_checkpoint.py` 两处改为 `early_stop=True`，并把 docstring 里"非默认旋钮"的理由改写为上游 #4836 的结论 |
  | `models/common/linear.py` 新增 `GroupedLinear`（`num_linears` 投影轴，w13 存 `[E,2,F,D]`）、`models/common/moe.py` 用它重写 `GroupedExperts`、`models/common/moe_sharding.py` 把单一 `inner_experts` 配置拆成 `w13`/`w2` 两份、`distributed/fsdp.py` 专家放置改为 `_linear_param_shard_placements(include_unstacked_grouped=True)`（即 `Shard(weight.ndim-2)`） | 新能力面：把融合的 w13 投影做成带 projection 轴的 `GroupedLinear`（载体是 blockwise 量化与 LoRA），FSDP 相应沿矩阵行而非默认 dim 0 切 | **B** | **无需动作（表示等价）**：llmtuner 的专家是 packed 3-D（`gate_up_proj (E,2F,D)`、`down_proj (E,D,F)`），`Shard(ndim-2)` 与 llmtuner 的 `Shard(1)` 切的是同一段——`[E,2F,D]` 的第 i 个 chunk 与 `[E,2,F,D]` 沿 F 的第 i 个 chunk 重合，`down_proj` 两侧同为 `Shard(1)`；dense 侧 llmtuner 的 fused QKV 是 2-D `[r*H,D]`，默认 `Shard(0)` 即切输出维，正确。`num_linears` 轴服务量化/LoRA，两者都在 llmtuner 裁剪面内 |
  | `models/common/token_dispatcher.py`、`distributed/deepep/deepep.py` | 注释改名 `GroupedExperts`→`RoutedExperts` | C | 无动作（纯注释，无语义） |
  | `experiments/transformers_modeling_backend/moe_replacement.py`、`state_dict_adapter.py` | Module-registry / state-dict adapter 面的 MoE 构建适配 | C | llmtuner 无该面（swap 直拷 HF 权重），不移植 |
  | `models/common/multimodal.py`、`distributed/flex_shard/`、`quantization/`、`rl/`、`experiments/graph_trainer/`、`overrides/fused_mla.py`、`config/transform/*` | 范围外 | C | 不适用 |

  本轮环境与验证（与 design doc §7 同轮记录）：Python 3.11.5（conda base，`python -V`）
  / torch 2.2.2 / CPU gloo。
  该 torch 缺的是**一组**新 API，不是单个包：`spmd_types==0.2.5` 装了但 import 失败
  （缺 `torch.distributed._local_tensor`）、`torch.distributed.tensor` 无公开
  `DTensor`、无 `torch.distributed._composable.fsdp`、无 `torch.nn.attention`
  （flex_attention）、无 `torch.distributed.pipelining`、无 `torch.OutOfMemoryError`、
  无 CUDA。实测：25 个 integration 脚本 **2 passed / 23 failed**，23 个失败全部是上述
  API 缺失（10 × `DTensor` ImportError、7 × `torch.nn.attention`、4 ×
  `spmd_types`、2 × `_composable.fsdp`），没有一个失败来自本仓逻辑
  ——唯一通过的两个是 `reduce_equivalence`（只依赖 gloo）与
  `vocab_parallel_loss_equivalence`（只依赖 `components/loss.py`）。
  CPU 单测 143 passed / 9 failed / 59 skipped，9 个 failed 全是
  `torch.OutOfMemoryError` 与 `torch.distributed.pipelining` 缺失。因此本轮
  **没有**复跑任何等价性测试，可运行的只有静态门禁与不依赖上述面的 CPU 单测。
  要复跑 ②（vocab-sharded head）与 ③（PP）的等价性，需要 torch≥2.12 + 多卡；
  ② 的 first half（loss 调用点 vocab-aware）不需要多卡，但需要 trainer 可导入，
  即同样受这批 API 阻断。
  同轮另修一处 llmtuner 内部缺陷：`62df262` 把 `parallel/parallelize_hf.py` 改名为
  `parallel/parallelize.py` 时漏改两个 test 侧 import
  （`tests/unit_tests/cpu/parallel/test_activation_checkpoint.py`、
  `tests/integration_tests/ep_fsdp_equivalence.py`）。该缺陷此前一直被环境门禁跳过掩盖，
  在 torch≥2.12 的环境里会是 `ModuleNotFoundError`；已于 `142bb6d` 修复。
- 2026-09-27 二次增量（同日，走查驱动）：基线再前进到 TorchTitan `f35966713`
  （`c8a3e7666` 之后 14 个提交、177 个文件，`git -C <torchtitan> log --oneline c8a3e7666..f35966713`）。
  与本仓相关的面逐项核对如下（其余在 `rl/`、`experiments/`、`quantization/`、
  `models/qwen3_5|6|8`、`torchtitan_recipes/tests`，范围外）：

  | 上游改动 | 分类 | llmtuner 处理 |
  |---|---|---|
  | `trainer.py`、`train.py`、`components/loss.py` **未变** | —— | 无动作；`llmtuner_trainer_walkthrough.md`（已退役，2026-10-08） 引用的行号据此复核通过 |
  | `training_engine.py`（42 行）、`components/validate.py`（25 行）：PP 编排、`max_num_documents` 的传入点 | **A/B（PP、校验面）** | 走查已覆盖：校验器的 PP 分支是 llmtuner 的 loud-raise 缺口（D18）、`max_num_documents` 的裁剪理由经复核成立（D19）；PP 损失函数的双驱动接线缺陷已修（D17，`make_schedule_loss_fn`），见 `llmtuner_trainer_walkthrough.md`（已退役，2026-10-08） §11.3 |
  | `distributed/{parallel_dims,pipeline_parallel,utils,fsdp,activation_checkpoint}.py`，`distributed/context_parallel/` 包重整为 `context_parallel.py` | **B（并行面，走查进行中）** | 属 `parallel/` 批次（`llmtuner_trainer_walkthrough.md`（已退役，2026-10-08） §12 第 2 项），已开走查：`activation_checkpoint.py` 于 2026-09-27 对齐（pp×AC 放行 + `_disable_dynamo_lru_cache` 移植，见六次增量）；`parallel_dims.py` / `fsdp.py` / `context_parallel.py` / `utils.py` 仍待走，`stages.py` 的 stage 表与 `matrix.py` 守卫同批复核（stage 顺序与上游 `parallelize_hf_transformers` 的 sharding→AC→compile→FSDP 一致；上游 PP 的 `head_shard_degree` 式解析期校验在本表「TP/SP 对齐结论」已登记）。本轮只确认本表引用的 `torchtitan/distributed/parallel_dims.py:260` 与 `set_determinism`（`torchtitan/distributed/utils.py:118`）语义未变 |
  | `config/{parallelism,validation,configs}.py`：`max_num_documents` 的 CUDA graph 校验（`torchtitan/config/validation.py:49`） | C | llmtuner 无图通路（D10），不适用 |

  本轮验证（与 `llmtuner_trainer_walkthrough.md`（已退役，2026-10-08） 同轮）：CPU 单测
  **143 passed / 59 skipped / 9 failed**，9 个失败与前一轮同因（7 个
  `torch.OutOfMemoryError`、2 个 `torch.distributed.pipelining` 缺失），都在未触碰的文件；
  新增 1 个受 `pipelining` 门禁的用例
  （`tests/unit_tests/cpu/parallel/test_pipeline.py:413`），本机跳过、torch≥2.12 环境执行。
  文档 `文件:行号` 引用经机械校验（文件存在 + 行号在范围内）全部命中，脚本见
  `llmtuner_torchtitan_alignment_workflow.md`（已并入本文，2026-10-08） §7.1。
- 2026-09-27 三次增量（TP/SP 对齐走查）：逐项判定见上「TP/SP 对齐结论」。本轮落地：
  1. **注意力头整除守卫**（对齐上游 `config/validation.py:150 head_shard_degree`）：新增
     `llmtuner/parallel/head_sharding.py`；`apply_tp` 校验 `heads % tp`，ulysses CP 校验
     `heads % (tp*cp)`（原来的内联检查改为调用同一函数，拒绝语义与消息要点不变）。
     此前纯 TP（cp=1）没有任何守卫：`shard_weight` 只看特征维，而 8 个 KV 头 × head_dim 128
     = 1024 特征恰好能被 `tp=16` 整除，错误要等 HF 的 head reshape 才暴露。测试：
     `tests/unit_tests/cpu/parallel/test_head_sharding.py`（7 例，**无**环境门禁，本机可跑）、
     `test_tp.py` 新增 2 例（守卫在触碰 mesh 之前触发；整除时不被误伤）、
     `test_cp.py` 原有 3 例继续覆盖 `tp*cp` 路径。
  2. **plan 读取统一**：新增 `tp.py::model_tp_plan`（`tp_plan` 属性优先、回退 `_tp_plan`），
     `resolve_plan` 与 `apply_tp` 的 MoE 规格探测共用它。`HFTransformerModel.tp_plan` 在内层
     `_tp_plan` 为空时回退到内层 `tp_plan` 属性——HF 只用属性暴露 plan 的模型不再被读成
     "无 plan"（那样 `apply_tp` 会静默一个投影都不切）。测试：`test_hf_wrapper.py` 新增 1 例
     （属性回退 + 属性优先的取舍各钉一次）。
  3. **HF `*_rep` 类规格的报错改写**：`colwise_rep` / `rowwise_rep` / `local_*` / `gather` /
     `replicate` / `sequence_parallel` 仍然 loud-raise（llmtuner 没有 replicated-activation
     TP 路径，静默改成 `colwise` 会让相邻算子拿到它不预期的布局），但报错现在指名规格、列出
     可接受词汇并给出理由。**影响面**：Apertus / GLM-4V / Phi-4-multimodal / Llama-4 /
     FlexOlmo 的 HF `base_model_tp_plan` 带这些规格，这些家族要么改写 plan、要么不开 TP。
     测试：`test_tp.py` 6 例 parametrize + 1 例 typo 仍拒绝。
  4. **文档修订**：三处"上游 `distributed/tensor_parallel.py` 已删除、无后继"改为指向真正的
     后继（`protocols/sharding.py` + `models/common/decoder_sharding.py` +
     `experiments/transformers_modeling_backend/hf_sharding.py`），并新增本文「TP/SP 对齐结论」。
  同轮顺带修掉一个会误报的用例：`tests/unit_tests/cpu/parallel/test_tp.py` 的 Qwen3 用例断言
  transformers 4.57 已不再声明的 `replicated_with_grad_allreduce`（在 torch≥2.12 环境会失败），
  改为把该规格注入真实 plan 来钉住分支，恢复版本无关性。
  本轮验证：`test_head_sharding.py` 7 passed（本机）；`test_tp.py` / `test_cp.py` /
  `test_hf_wrapper.py` 在本机被 `spmd_types` 门禁跳过，改以注入最小 torch/spmd 垫片后脱门禁运行，
  三处分别为 20 passed / 3 passed（其余因 fake PG 与 DeviceMesh 的 2.2 差异 errored）/
  1 passed，新增用例全绿。
- 2026-09-27 四次增量（删除 batch-invariant 特性面）：删除 `llmtuner/utils/batch_invariant.py`
  及全部消费点。依据是上游语义：torchtitan 的开关由 `debug.batch_invariant` 驱动
  `distributed/utils.py::set_batch_invariance`（注册 ATen override + 设 NCCL 环境），但
  `trainer.py` 对 SFT 直接 `raise ValueError("Batch-invariant mode is not needed in
  supervised learning.")`——该特性只服务 `rl/`。llmtuner 是 SFT trainer，且
  `set_batch_invariant_mode` 全仓无生产调用者，原模块自述"开关可设但未安装 kernel"的
  TODO 即其唯一存在理由。删除面：`batch_invariant.py`（整文件）；`hf_wrapper.py` /
  `cp_kernel.py` 的 `separate_full_blocks=not is_in_batch_invariant_mode()` 改为常量
  `True`（即此前的默认值；torch 2.10 无该旋钮时 `models/common/attention/masks.py` 仍会 pop 掉）；
  `components/loss.py` 的 `_GatherVocabShards` / `_gather_vocab_shards` 与
  `compute_logprobs` 的 gather 分支（vocab-parallel 的 `reduction="none"` 路径保留）；
  `test_cp.py` 的 `test_full_length_mask_tracks_batch_invariant_mode`；
  `vocab_parallel_loss_equivalence.py` 的 gather 值/梯度等价块；
  `cp_ulysses_equivalence.py` 的同名调用点；以及本文、符号指南、设计文档与
  `llmtuner/README.md` 的登记行。
- 2026-09-27 五次增量（lm_head 缺口第一步 + config 走查）：
  1. **vocab-parallel loss 四处接线**（D 类 `lm_head` 缺口的第一步，纯 no-op）：
     `Trainer.loss_vocab_kwargs()`（原 `_loss_vocab_kwargs`；TP mesh + 模型自身 HF config 的 `vocab_size`，
     新增 `HFTransformerModel.vocab_size` 属性）驱动 `Trainer.loss_sum`（原 `_loss_sum`）、
     `chunked_lm_head_cross_entropy`、PP 的 `scalar_loss_fn`（2026-09-29 去私有化，原名 `_scalar_loss_fn`）与 Validator 路径；
     选择仍按形状（`components/loss.py`），所以 lm_head 复制的今天每条路径都走
     普通 CE，逐位不变。测试：`test_chunked_loss.py` 增 1 例（分片参数下值与三份
     梯度不变）、`test_trainer.py` 增 2 例（缺 TP 轴/缺词表 → 空 kwargs）、
     `test_pipeline.py` 增 1 例（schedule loss 同样收参数且仍为 no-op）、
     `test_hf_wrapper.py` 增 1 例（属性取内层 config，缺字段返回 None）。
  2. **CP `context_parallel_load_balancer` 默认对齐上游**：`"headtail"` → `None`
     （上游 `config/parallelism.py` 默认 None，headtail 由 recipe 显式打开，其
     `test_config_manager.py` 亦断言默认 None）。均衡分片改变每个 rank 参与的
     token 集合，不该由默认替用户决定；`test_config.py` 的默认断言与 ulysses
     配对用例同步改写（拒绝仍是"组合"拒绝）。
  3. **非 CLI 字段不再出现在 `--help`**：新增 `config/cli.py`——`arch_overrides`
     （dict）、`param_groups`（嵌套 dataclass 列表）、`purge_exempt`（callable）
     三个字段，HfArgumentParser 会为它们建 flag 但拒绝一切取值（实测
     `invalid dict value` / `invalid ParamGroupConfig value` / `invalid Callable
     value`），既不能用又占帮助面。`init=False` 是 HfArgumentParser 唯一的跳过
     钩子，于是给解析器一份生成视图（子类 + `init=False` + 沿用基类默认值），
     `isinstance`/`__post_init__`/默认值全部不变；`PARSER_GROUPS` 一并搬到该模块，
     组集合与顺序自此单一来源。测试：`test_config.py` 增 2 例（三个 flag 不在
     option 行、视图解析后仍是真组且默认值在位）——实测 flag 数 127 → 124。
  4. **`test_config.py` 门禁收窄**：`require_env('dtensor')` → `require_env('pipelining')`
     （真正的硬依赖是 `ParallelConfig.__post_init__` 里的 `get_schedule_class` 导入，
     不是 DTensor）；checkpointer 常量改为用例内局部导入 + `skip_without('dcp')`
     （`tests/caps.py` 新增的用例级守卫）；两处 `llmtuner.trainer.LLMTunerConfig`
     改为 `llmtuner.config.LLMTunerConfig`（配置测试不该为此拖入引擎层）。
     `symm_mem` 用例改为能力判定（CC≥9 设备上 skip），不再假设开发机是 CPU。
  5. **复核后不改的两项**：`pipeline_parallel_schedule_csv`——上游同样在装配期
     校验（`distributed/pipeline_parallel.py:360`，先查文件存在、再 `_load_csv`），
     移进 `__post_init__` 反而偏离，保持现状；`fsdp_symm_mem_scope`——上游用
     `scope=None` 表达"关"且 `tyro.conf.Suppress` 不进 CLI，llmtuner 用
     `enable_fsdp_symm_mem=False` + scope 两字段，**默认语义等价**，见符号指南该行。
  6. 顺带发现（第六轮已修）：`train.py` docstring 称"YAML/JSON 文件可位置传入"，
     实测 `HfArgumentParser.parse_args_into_dataclasses(['x.json'])` 直接报
     "Some specified arguments are not used by the HfArgumentParser"（该版本
     transformers 只把 `parse_json_file`/`parse_yaml_file` 做成"整体替换 argv"的
     方法，没有 argv 路径入口），docstring 与实现不符；现改为如实说明"只有旗标，
     没有配置文件通道"，并写明上游用的是 `--module`/`--config` 选择配置函数。
  本轮验证：`test_config.py` 在垫片环境下 38 passed / 1 failed（唯一失败是垫片
  `get_schedule_class` 不抛 ValueError 造成的既有假阴性，HEAD 同）；四处新增用例
  在本机分别随 `test_chunked_loss.py`（12 passed）与脱门禁的
  `test_trainer.py`/`test_pipeline.py`/`test_hf_wrapper.py` 通过（+2/+1/+1）；
  全量 CPU 套件 151 passed / 59 skipped / 9 failed（失败集与基线一致）。
- 2026-09-27 六次增量（pp × AC 对齐，`parallel/` 批次第一项）：
  1. **AC 进入 PP 路径**。上游 `experiments/.../parallelize.py` 的 PP 分支把
     `ac_config` 交给每个 model part 自己的 `parallelize` 调用（
     `pipeline_hf_transformers` 里 `m.parallelize(..., ac_config=ac_config, ...)`），
     而它的 pipeline recipe 全部登记 `SelectiveAC.Config()`；llmtuner 此前在
     `parallelize_hf_transformers` 入口对 `pp > 1` 直接拒绝 AC
     （`matrix.pp_activation_checkpoint`，文案自述"尚未接线"），等于拒绝上游的默认
     组合。现在 `stages.py` 的 `ac` 行 `on_pp=True`，PP 的 per-chunk runner 表加上
     `ac`，两条路径共用同一个 `apply_ac` 闭包（AC 参数不可能在两边漂移）；
     `matrix.pp_activation_checkpoint` 与其表行一并删除。逐 chunk 应用是安全的：
     `split_model_into_stages` 保留 `layers` 容器（层号沿用原索引），`apply_ac`
     折的正是本 chunk 持有的那些层，与上游 `model.get_submodule("layers")` 同构。
  2. **`disable_dynamo_lru_cache` 移植**（上一步解锁的前置条件）。上游在每个
     policy 的 `apply()` 开头调用它，修的是 AC+PP+Flex 下"第二个 microbatch
     以动态 shape 重编译 → 同一区域存在两张合法图 → dynamo 默认 latest-wins
     可能选到期望多一个 symint 的那张，而 SAC 缓存的 inductor-HOP 输出没有它"
     （pytorch/pytorch#166926）。llmtuner 现按上游位置调用（包装模式在折层前、
     `memory_budget` 在设全局量前），但走 `has("dynamo_lru_cache")` 能力门
     （新增注册项）：本机 torch 2.2.2 的 `torch._C._dynamo.eval_frame` 存在而无
     `_set_lru_cache`，此时记一条 info 后继续，不因缺 workaround 拒绝整轮训练。
  3. 测试：`test_assembly_stages.py` 的 `PP_STAGE_ORDER` 断言改为
     `("tp","ac","compile","fsdp")`；`test_matrix.py` 删掉该行用例（表与模块函数
     一一对应的断言自动跟随）；`test_activation_checkpoint.py` 增 4 例——PP 分支
     逐 chunk 折层且 `stage.submod` 重绑、`mode='none'` 在 PP 上仍是 no-op、
     workaround 在 knob 存在时置 `False` / 缺失时不动它。
  4. 文档：`llmtuner/README.md` 的"被明确拒绝的组合"表把 AC 从 `pp > 1` 行移除；
     符号指南 AC 行与能力表、设计文档能力表同步。
  5. 顺带修正 `train.py` docstring 里"YAML/JSON 可位置传入"的不实说法（见五次增量
     第 6 条）：现在是"只有旗标"，并说明上游的配置入口是 `--module`/`--config`
     选配置函数，不是配置文件。
  本轮验证：`test_matrix.py` + `test_assembly_stages.py` 13 passed、
  `test_capabilities.py` 8 passed、全量 CPU 套件 151 passed / 59 skipped / 9 failed
  （失败集与基线一致，本机）。新增的 4 例 AC 用例在本机**不可执行**：`test_activation_checkpoint.py`
  经 `parallelize_hf_transformers` 链式 import `parallel/fully_shard/fsdp.py`，后者要
  `torch.distributed._composable.fsdp.FSDPModule` 与 `torch.distributed.fsdp.CPUOffloadPolicy`
  （torch≥2.4/2.6），而本机 2.2.2 连 AC 模块自身都进不去（模块级
  `torch.ops.aten.mm.dtype`）。故改用"打桩 `apply_ac` + 假 ParallelDims"的编排级检查：
  `apply_ac` 对 PP chunk 恰好调用一次、调用顺序为 `pp → ac → compile → fsdp → schedule`、
  `PP_STAGE_ORDER` 为 `("tp","ac","compile","fsdp")`、`submod` 已重绑（本机实测通过，
  脚本不留仓）；4 例用例本身为代码审阅，需 torch≥2.5 的 CI 执行。真值等价
  （AC×PP 的 loss/grad）仍需 torch≥2.12 + 多卡，与 2-rank gloo 的 `pp_equivalence.py`
  同属未闭合项。
- 2026-09-28 七次增量（`parallel_dims.py` 复核，`parallel/` 批次第二项）：
  逐项对上游 `distributed/parallel_dims.py`。结论：**结构等价，三处差异全部有意**
  ——(a) `loss` mesh 含 `tp`（上游是 `dp*cp`；llmtuner 的 TP 端到端切序列，每个 rank
  的 loss 和只覆盖 `T/tp`，见 "TP/SP 对齐结论" 与 `lm_head` 行）；(b) 上游 `_validate`
  用 `assert`，llmtuner 用带消息的 `ValueError`（`-O` 下 assert 会消失）；(c) 上游
  `DistributedTopology` / `enable_sequence_parallel` / `_real_pp_group_for_fake_spmd`
  与 SPMD-only 的 `unfold_dp_axis*` / `get_activated_mesh` / `resolve_mesh` /
  `get_dense_tp_mesh` 在 llmtuner 无对应（前者是 debug 后端与 SP 开关，后者只服务
  spmd_types，已计入 2026-09-23 死代码清理）。本轮把其中两件从"文档说明"变成"可执行
  契约"：
  1. `_validate_meshes` 的尺寸表提取为 `_expected_mesh_sizes()`：不再只有建过 mesh
     才可达，因此 `loss == batch * cp * tp`（唯一含 tp 的轴）可以在没有进程组的机器上
     钉住——这正是最容易被后来者"顺手改成上游那样"的一行。`test_parallel_dims.py` 新增
     1 例断言该式与非 tp 轴不变。
  2. 三处把 loss mesh 说成"dp+cp"的旧注释改为事实（`models/common/aux_loss.py`、
     `models/common/moe/block.py`、`test_parallel_dims.py` 的用例 docstring）；代码侧
     `trainer`/`validate` 的门本来就是 `dp_cp_enabled or tp_enabled`，与表一致。
  3. 新增 D 类登记：上游的单进程 fake-SPMD debug 后端（`backend="fake"` /
     `real_pp_fake_spmd`，torch 2.2 无此 backend）。
  本轮验证：`test_parallel_dims.py` 脱门禁后 9 passed（新增用例在列），唯一 error 是
  需要真 gloo 进程组的用例被沙箱 `Cannot resolve 127.0.0.1` 挡住；全量 CPU 套件
  151 passed / 59 skipped / 9 failed（失败集与基线一致）；ruff 干净。
- 2026-09-28 八次增量（`fully_shard/fsdp.py` 复核，`parallel/` 批次第三项）：
  逐项对上游 `distributed/fsdp.py`（含 `experiments/transformers_modeling_backend/` 与
  `models/common/decoder.py` 的调用面），并读 `resolve_fsdp_mesh`／
  `resolve_sparse_fsdp_mesh`／`disable_fsdp_gradient_division`／
  `enable_fsdp_symm_mem`／`get_fsdp_reshard_after_forward_policy`／prefetch 段的对应实现。
  结论：**装配逻辑等价，差异四类全部有意**——
  (a) **mesh 表示**：上游把「多轴 storage mesh + `DataParallelMeshDims(shard=dp_shard[,cp],
  replicate=dp_replicate)`」交给 `fully_shard`（它的参数是 SPMD DTensor，必须显式声明轴），
  llmtuner 的参数是普通 tensor，改为 `resolve_fsdp_mesh` 重建 1-D/2-D 专用子网，torch 按形状
  默认读法得到同一组轴；专家侧同理（`(dp_replicate?, efsdp)` 对
  `DataParallelMeshDims(shard="efsdp", replicate="dp_replicate")`）。
  (b) **专家计数来源**：上游读 `moe.num_experts`（SPMD DTensor 的逻辑总数），llmtuner 读
  `moe.router.num_experts`（EP swap 后 `inner_experts` 只剩本地切片 total/ep）；两者比较的
  仍是同一个 total，阈值语义不变。
  (c) `linear_param_shard_placements`（上游把 stacked/grouped 权重按 `Shard(ndim-2)` 切矩阵行）
  在 llmtuner 无对应物：llmtuner 只有 packed 3-D 专家权重（`w1`/`w3` `(E,F,D)`、`w2`
  `(E,D,F)`）与 2-D `nn.Linear`，前者的 `Shard(1)` 就是输出维、后者的默认 `Shard(0)` 就是
  输出维；且 llmtuner 没有任何 `num_linears` 式融合 Linear（上游该 helper 只对这类权重生效）。
  (d) `apply_fsdp_to_multimodal_encoder`（vision tower 整体分片）无消费者：llmtuner 的
  `models/common/multimodal.py` 只有 span/gather 融合算子，没有编码器模块。
  本轮四处动作：
  1. **命名统一（degree/size）**：`apply_fsdp_to_decoder` 的参数保持 `ep_size`，**不跟**
     上游改名成 `ep_degree`——llmtuner 的配置面统一拼 `*_size`（`expert_parallel_size` /
     `data_parallel_shard_size`，见 `config/parallel.py` 的字段说明），`ep_size` 与之同词根。
     同一次把 `_fsdp_shard_degree` 改名 `fsdp_shard_size`、
     `require_heads_divisible_by(degree=, divisor=)` 改为 `(size=, axis=)`（两个调用点
     `apply_tp` / ulysses CP 同步）；其余 prose 里的 "degree" 只作概念词保留。
  2. **私有面收敛**：把 4 个其实被跨模块生产代码使用的 `_` 符号转正 ——
     `iter_moe_layers`（`moe/balancing.py` / `hf/model.py` / `compile.py` 三处）、
     `iter_fsdp_modules`（`fully_shard/apply.py`）、`resolve_top_k` /
     `resolve_score_func`（`expert_parallel/convert.py`），并给 `fully_shard/fsdp.py` 与
     `expert_parallel/probe.py` 补 `__all__` 明确公共面（fsdp.py 的表面 = 上游 `__all__`
     减去 llmtuner 有意不带的两个入口）。只被单测引用的 `get_default_save_ops` /
     `yarn_inv_freq`（原 `_get_default_save_ops`/`_yarn_inv_freq`）于二十次增量改为公开：
     本仓规则是模块级 helper 默认公开，`_` 只留给框架协议名与同名校验核。
  3. docstring/注释校正：删掉 `apply_fsdp_to_decoder` 里「上游 `dp_mesh_dims` 入口从未接线」
     这句含混说法，改为写明 mesh 表示差异的成因（上游参数是 DTensor，llmtuner 不是）；MoE
     分支的 NOTE 补一句上游的 stacked-Linear 覆盖与 llmtuner 无需覆盖的理由；`Shard(1)`
     覆盖旁注明它就是上游 `Shard(ndim-2)` 切的那一段。
  4. 新增可执行契约
     `test_efsdp_placement.py::test_packed_expert_weights_shard_their_output_dim_at_index_one`：
     钉住三个 packed 专家权重的形状与「输出维在第 1 轴」。此前这行等价性只写在文档里，改布局
     （挪动专家轴或 up/down 投影的特征轴）会让 flat `Shard(1)` 静默切错段。
  本轮验证：`test_efsdp_placement.py` 脱门禁后 **9 passed**（含新增用例；其中 3 个需要 size-1
  gloo 进程组的用例在沙箱外执行）、`test_fsdp_contract.py` 脱门禁 **12 passed**。两者都依赖
  一个临时 harness（torch 2.2 既没有 FSDP2 surface 也不支持 1-D 命名 mesh 切片），shim 只
  模拟 FSDP2 的模块接口与 `edp_mesh["efsdp"]` 的取值，不实现分片本身，故真机分片行为仍需
  torch≥2.12 + 多卡复跑（见 symbol guide §12）。全量 CPU 套件 151 passed / 59 skipped /
  9 failed（失败集与基线一致）。
- 2026-09-28 九次增量（`context_parallel/` 输入分片复核，`parallel/` 批次第四项）：
  逐项对上游 `distributed/context_parallel.py`（上游此前的 `context_parallel/` 包已重整为
  单文件；kernel 侧在 `models/common/cp_attention.py`）。结论：**语义等价，差异三类，全部
  登记**——
  (a) **表示**：上游 `shard_tensors(input_dict, input_shardings, permutation)` 走 `SpmdType`
  声明 + `spmd.shard(R→S(seq_dim))`；llmtuner 没有声明面，改为
  `shard_batch_for_cp/tp`、`shard_padding_mask_for_cp/tp`、`shard_attention_mask_for_cp`
  五个显式入口，置换与切分一并委托 torch 私有的 `_context_parallel_shard`（它接
  `load_balancer`，permute+shard 在同一步内完成）。两端对「先置换、再按 seq 轴切」的语义
  一致，mask 只切 Q 轴、KV 保持全长也一致。
  (b) **负载均衡器形态**：上游是 `ContextParallelLoadBalancer` 抽象类 + `HeadTail` /
  `PTRR` 两个实现（构造期吃 `seq_len` 与 `attention_metadata`，`generate_permutation()`
  产出全局置换）；llmtuner 用字符串 `context_parallel_load_balancer`（`"headtail"` /
  `None`）现场构造 torch 的 `_HeadTailLoadBalancer`，`seq_len % (2*cp)` 的校验两边相同。
  `"ptrr"` 仍是登记缺口（配置期 + 装配期两道 loud-raise）：它要从 BlockMask 推置换，而
  llmtuner 切 batch 时还没有 mask（mask 由 wrapper 在 forward 内按 positions 构建）。
  (c) **校验位置**：上游 `shard_tensors` 每次调用查 `shape[seq_dim] % cp == 0`、以及所有
  CP 张量同 seq len/同 device；llmtuner 把 `max_seq_len % cp` 放在配置解析期
  （`config/root.py::__post_init__`），运行期交给 torch 的 helper。后者更早、更强（trainer
  路径的 T 恒为 `max_seq_len`），但没有覆盖「调用方传入与 `max_seq_len` 不同的 T」这种绕过
  配置的用法——登记为已知差异，本轮不补冗余运行期检查（本机 torch 2.2 无 CP helper，该分支
  无法验证，且会与 `require_torch_cp` 的调用顺序纠缠）。
  本轮只做复核与登记、无代码改动（同日的命名/私有面调整为八次增量的一部分）：
  `cp_kernel.py` ↔ 上游 `models/common/cp_attention.py` 的逐项对照、
  `accelerator/dist_utils.py` ↔ 上游 `distributed/utils.py` 仍待走。
- 2026-09-28 十次增量（`cp_kernel.py` 与 `accelerator/dist_utils.py` 复核，`parallel/`
  批次第五项，收尾）：两项复核 + 一处补移植，结论全登记——
  (a) **`context_parallel/cp_kernel.py` ↔ 上游 `models/common/cp_attention.py`：语义等价。**
  kv_allgather 走的是**同一个** torch 自定义算子 `flex_cp_allgather`（上游 HF 后端的
  `_wrap_flex_kernel_cp` 用的就是它，`(b, heads, seq, dim)` 的 dim 2、进程组名捕获、
  backward 由算子自带 reduce-scatter），Q 保持 token 分片、mask 为 Q 分片/KV 全长；
  ulysses 的 token↔head all-to-all 与上游 `UlyssesCPInnerAttention` 是同一置换（scatter
  head、gather seq，反向互为转置），mask 分派（packed 全长 vs 单文档重建）与九次增量
  记录的 wrapper 语义一致。**一处显式差异**：上游**原生**模型的
  `KVAllGatherCPFlexInnerAttention.Config` 暴露 `reduce_dtype: float32|bfloat16`
  （`backward_options={"op_dtype": ...}`，默认 fp32），llmtuner 委托 torch 算子因而没有该
  旋钮——但上游 **HF 后端**（llmtuner 真正对应的那一支）同样没有，故属「与 HF 后端对齐、
  与原生模型路径不同」，不是什么待补参数；本文附录中
  「dtype 可配（默认 fp32）」的旧措辞已在本次改正。
  (b) **`accelerator/dist_utils.py` ↔ 上游 `distributed/utils.py`：无能力缺口。**
  两者血缘不同（llmtuner 那支 vendored 自 OpenMMLab `mmengine.dist`，上游等价物在
  `accelerator/collectives.py` 与调用点的 `accelerator/dist.all_reduce`），故本次按能力
  而非形状对照，逐项终态：`dist_sum`/`dist_max`/`dist_mean`/`dist_sum_tensor` → llmtuner
  用 `all_reduce` 在调用点（trainer/validator 的 loss·token 归约；`components/metrics.py`
  完全不碰 `torch.distributed`），不重建命名薄封装（`collectives.py` 模块 docstring 同步
  改写）；`set_pg_timeouts`/`clip_grad_norm_` → 已移植且已复核（EP 裁剪按物理本地 expert
  参数分组并跨 EP 归约，免去上游「每个参数都是带 `"ep"` 轴的 DTensor」断言；dense-only
  路径与上游逐行同构）；`init_distributed` + `DistributedTopology`（`fake` /
  `real_pp_fake_spmd`）→ 唯一的真缺口，已挂 D 表（本机 torch 2.2.2 无 `backend="fake"`，
  不可验证）。`batch_invariant`/`bf16x9` 属四次增量已删特性面，不重复。
  (c) **补移植（唯一代码改动）**：上游 `set_determinism` 中两件可移植件落到
  `Trainer.seed_everything`（原 `_seed_everything`，2026-09-29 去私有化）—— `PYTHONHASHSEED = str(seed % 2**32)`（本进程读不到，但
  之后 spawn 的 dataloader worker 会读，故必须在此设置，与上游同拼写）与
  `detect_anomaly`（`TrainingConfig.detect_anomaly`，默认 `False` 逐位不变；
  `torch.autograd.set_detect_anomaly(True, check_nan=False)` + 上游同文告警，
  `check_nan=False` 的理由同上游：NaN/Inf 梯度检查走 `aten._is_any_true`，无 DTensor
  sharding 策略，分片参数上会崩，而记录栈这一半保留）。`collectives.py` 的 `__all__`
  故意不再导出确定性符号——该模块的 docstring 声明「每个符号都源自上游」，而
  `set_determinism` 在 llmtuner 无同名对应物，落点就是 seed 助手。
  测试：`tests/unit_tests/cpu/test_trainer.py` 新增两条（`PYTHONHASHSEED` 导出、
  `detect_anomaly` 的 `check_nan=False` 契约），取消 gate 后 33 passed（基线 31），
  失败集不变。
  本轮后 `parallel/` 批次的复核面（`stages.py`/`matrix.py`/`activation_checkpoint.py`/
  `parallel_dims.py`/`fully_shard/fsdp.py`/`context_parallel/`/`cp_kernel.py`/
  `accelerator/dist_utils.py`）全部走完，余下只有需要多卡的 TP/SP 布局根治（D14）与
  D18 的 PP×校验实现。
- 2026-09-28 十一次增量（`components/` 批次开篇：checkpointer + profiler 走查）：
  (a) **checkpointer（`components/checkpointer/`）逐文件对照，语义已对齐，改动只一处死代码。**
  对照方式：AST 归一化（剥 docstring/注释后 `ast.unparse`）逐文件 diff，再人工核对语义。
  `base.py`：策略方法（`_parse_step`/`_find_load_step`/`_purge_stale_checkpoints`/
  `_states_to_load`/`_create_checkpoint_id`/`ModelWrapper`/`purge_thread`/`shares_storage`（当时名 `_shares_storage`））
  与上游逐行同构；`dcp.py`：`_save`/`_load_checkpoint`/`dcp_save`/`_save_last_step`/
  `_flattened_model_states_sd` 与上游逐行同构（含 async 三段模式、`exclude_from_loading`
  与 staged 目录保留）；`torch_checkpointing.py`：策略与 `_save_last_step` 的 HF 导出
  路径（`sharded/` 子目录 + `pre_finalize_callback` 里的 barrier+consolidate）同构；
  `filesystem.py` 与上游 `tools/filesystem.py` 1:1（ratio 1.000），`utils.py::canonical_fqn`
  同构。**登记的有意差异（均为适配而非缺口）**：
  (1) `BaseCheckpointManager` 新增 `_initialized` 门（上游无）：上游靠各类自己的
  `hasattr(self, "staging_future"/"_manager")` + `getattr(self, "save_future", None)` 容错
  部分构造；llmtuner 的 `dcp.CheckpointManager.__init__` 会在 HF 选项上 raise，为此用一个
  显式标志统一挡住 `load`/`save`/`close`/`maybe_wait_for_*`，语义等价、形状不同；
  (2) 新增 `enable`（配置驱动）：上游的 manager 只在配置了 checkpointer 时才构造，llmtuner
  总是构造一个 `CheckpointConfig`，所以 `__init__`、`_should_save`、`_should_prewarm`
  三处按 `enable` 短路；
  (3) dcp 的 4 处 `assert` 改显式 `raise`（ValueError/TypeError）——同上游语义，但 `-O`
  下不会消失；HF 选项在无 `sd_adapter` 时报的错误更具体（llmtuner 不随包发布 adapter，
  三个 HF 选项都是登记缺口而非可用开关）；
  (4) `base._should_purge` 的 `dist.get_rank() == 0` → `dist_utils.is_main_process()`
  （非分布式下前者会 raise）；
  (5) 配置校验全量搬到 `config/checkpoint.py::__post_init__`（13 条逐条对齐），额外把
  上游 `dcp.Config.async_mode` 与 `training_engine` 的 `create_seed_checkpoint` 收进同一
  dataclass；上游那条 `initial_load_model_only` 无 `initial_load_path` 的告警**故意不移植**
  （llmtuner 每次运行都会构造默认 config，含 `--help`，该告警会无端触发）；
  (6) 命名：`_FilesystemCheckpointStorage`/`_async_save_config` 去私有化为
  `FilesystemCheckpointStorage`/`async_save_config`（测试需要这两个 seam），符合本仓
  "非必需不加 `_`" 的取向；`EXPORT_DTYPE_MAP` 与 `models/common/cast_linear.py` 的
  `TORCH_DTYPE_MAP` 是同一张 3 键表的两次书写（上游只有 `config/__init__.py` 一份），
  登记为已知重复，不为此把 components 反向依赖 models；
  (7) 上游 `dcp.py` 的 `SaveDone` 是**零引用死类**，不移植。
  **该模块本次唯一的代码改动**：`TorchCheckpointingManager` 的 `staging_future` 是死成员
  （`__init__` 置 None、`_close` 读一次，全仓无任何赋值点；上游该类根本没有这个成员，
  它在 dcp 里才有对应物），连同 `_close` 里那段空转一起删除。
  `torch_checkpointing` 后端仍未装、未跑（可达性说明见该文件 docstring），故只做静态复核。
  (b) **profiler（`components/profiler.py` ↔ `observability/profiler.py`）：对齐，并修一处
  设备面缺口。** 策略、目录布局、OOM 处理（`caused_by_oom` 防环 + 隐式链）、
  `MemoryProfiler` 的频率/命名/协议 4 全部同构。**发现的缺口**：上游的 activity 列表是
  "CUDA 可用加 CUDA、否则 XPU 可用加 XPU"，llmtuner 只加了 CUDA 分支，且 docstring 写了
  "设备只有 cuda/cpu 两种"——但 `accelerator/device.py` 的 `DEVICE_PRIORITY` 是
  npu/cuda/musa/mlu/xpu 五值，`xpu` 可达，于是 XPU 运行的 trace 会退化成 CPU-only。已按
  resolved device 补 `xpu` 分支（其余设备仍 CPU-only，与上游一致），并改正该 docstring；
  新增用例 `test_the_trace_activity_follows_the_resolved_device`（cpu/cuda/xpu 三态钉住）。
  登记的有意裁剪：`leaf_folder`（只服务上游 torchft 的 per-replica 子目录，llmtuner 无副本
  概念）、CUDA-graph annotations（随 D10 无图路径）、`structured_logger` span、`active()`
  builder（llmtuner 构造处已知道 step 与 folder）；memory history 改经
  `accelerator/monitoring` 的 device 探针（上游非 CUDA 分支调 `torch.memory`，不存在）。
  (c) 测试/文档：`test_profiler.py` +1 例；本文附录 §7.2/§7.3
  与 `llmtuner_trainer_walkthrough.md`（已退役，2026-10-08） §12 同步。`components/` 批次余下 metrics 与 optimizer
  两个模块。
- 2026-09-28 十二次增量（`components/` 批次收尾：metrics + optimizer 走查）：同样以 AST 归一化
  逐文件对照，结论如下。
  (a) **metrics（`components/metrics.py` ↔ `observability/metrics.py`）：逐项对齐，无能力缺口。**
  训练与校验两条日志的 key 集合、`tps`/`tflops` 公式、`time_metrics`/`memory` 派生、
  `should_log` 的 `step == 1 or step % log_freq == 0` 判据全部一致。差异四类，全部为已登记
  裁剪或适配：(1) 上游的自由函数 `compute_training_performance_metrics`（供 trainer 与实验
  共用）没有移植，llmtuner 把它内联进 `MetricsProcessor._derive` 的 `_Derived` dataclass；
  语义相同，形状不同；(2) MFU 抑制条件：上游看 `has_quantization`，llmtuner 看
  `gpu_peak_flops == 0 or num_flops_per_token == 0`（无量化路径，故用"设备未知就不报"等价替代）；
  (3) `_build_metric_logger` 去掉 `ft_enable`/`ft_replica_id`（无副本概念）、wandb 失败策略
  改为只吞 ImportError，rank 判定改用 `get_distributed_rank()`（单进程安全）；
  (4) `DeviceMemoryMonitor` 改吃 `device_type`、CPU 分支返回全零 `DeviceMemStats`、
  `_to_pct` 加除零守卫、`build_device_memory_monitor()` 在 CPU 上不打印容量；
  `ensure_pp_loss_visible` 增加 `not pp_enabled` 提前返回（上游该函数无此守卫，仅靠唯一调用点
  门控，语义等价），`_get_metrics_rank` 去私有化为 `get_metrics_rank`。模块 docstring 声称
  "从不调用 `torch.distributed`"经复核成立。
  **本次唯一代码改动**：删除 `MetricsProcessor.set_num_flops_per_token`——全仓零调用者，且其
  docstring 的理由（"metrics processor 先于模型构造、当时拿不到该数"）与 llmtuner 的实现相反
  （构造函数本来就收 `num_flops_per_token`）；上游也没有这个方法（它由 trainer 直接赋值属性）。
  (b) **optimizer（`components/optimizer/`）：三条登记差异（两条为本次新登记），其余对齐。**
  `optimizer.py` 的构造算法（逐 model part × 逐 optimizer 名的分组、pattern 首个命中即认领、
  `_build_impl_kwargs`、`_log_optimizer`、`step`、flat FQN state dict）与上游同构。
  **新登记**：上游第四种 `implementation="fused_opt_states_bf16"`（Adam step pre-hook 预建
  bf16 状态 + load post-hook 复原 dtype，CUDA fused 混合精度核）与
  `optimizer_factory_kwargs_by_name` 未移植，两条均入 D 表（前者不可验证：价值全在 CUDA
  fused 核且改写 checkpoint 状态 dtype；后者无消费者）。**补登记**：`_resolve_optimizer_factory`
  的第三个工厂 `DistMuon` 位于 `distributed/flex_shard/`，该目录早已作为范围外 C 类登记，
  此处把两者显式连上；`default_adamw`（上游 `lr=8e-4`/betas (0.9,0.95)/wd 0.1 的便捷构造）
  未移植，其上游调用者只有 torchft llama3 recipe 与 RL 示例（均在裁剪面内），训练路径由
  config 的 catch-all 默认组覆盖（betas 用 torch 默认，config 已声明为 NUMERIC 分叉）。
  **形状差异**：MoE 负载均衡/quantile hook 的注册点从容器挪到 `trainer/builder.py`（上游是
  各模型自己的代码注册），hook 本体在 `models/common/moe/balancing.py`；与上游"容器只提供
  `Optimizer.__init__` 的 hook 机制"一致。**优于上游的两处**：`_validate_params` 会点名未被
  认领的可训练参数并检出重复认领（上游只是一条 `assert expected == actual`）；`step` 的
  closure 断言改 `ValueError`。
  `utils.py`（ratio 0.996）：只剩 `isinstance(x, A | B)` 与 `zip(..., strict=False)` 之类
  现代化改写，无能力差异。`lr_scheduler.py`：WSD 公式逐行核对一致（同样的 0-based `+1`
  修正、`stable_steps = total + 1 - warmup - decay`、三种 decay 形状与 `min_lr_factor` 缩放）；
  差异为 `wsd_factor`（当时名 `_wsd_factor`）提到模块级 + `build_lr_scheduler` 自由函数取代上游嵌套闭包与
  `Config.build`，以及已登记的 `decay_ratio` 默认 0.0（上游 None）、`total_steps < 训练步数`
  拒绝、`load_state_dict({})` 空字典 no-op、`LRSchedulersContainer(total_steps=...)` 断言 seam。
  `ema.py` 属 2026-09-24 批 3a 已移植项，本轮无改动。
  (c) **走查中发现并修掉一个真 bug（本轮唯一行为改动）：`implementation="fused"` 在 CPU 上必崩。**
  取证：`torch.optim.AdamW([p], fused=True)` 在 CPU 上**构造期**即 `RuntimeError`
  （`requires ... supported devices: ['cuda','xpu','privateuseone']`），但 llmtuner（与上游同形）
  把 `fused` 放进每个 param group 而非构造参数，torch 只校验构造参数，于是 CPU 上构造通过、
  第一次 `step()` 才炸：`NotImplementedError: Could not run 'aten::_fused_adamw_' with arguments
  from the 'CPU' backend`。而 `implementation` 的默认值正是 `fused`，所以**文档里的 quickstart
  （单进程 `--steps 20`）在 CPU 上会在首步 optimizer.step 崩溃**；旧 config help 却写着
  "falls back to the for-loop kernel elsewhere"，测试 `test_the_implementation_flag_reaches_the_
  inner_optimizer` 还把 `group["fused"] is True` 钉住（因为它只构造、不 step，看不见崩溃）。
  上游不会遇到：torchtitan 是 GPU-only。**修法（保持上游形状、落实本仓自己的承诺）**：
  `_build_impl_kwargs` 先问 torch 该设备有没有 fused 核（`torch.optim.adam.
  _get_fused_kernels_supported_devices()`，取不到时回退 `{"cuda","xpu"}`），没有就把
  `fused` 降级为 for-loop 并记一行 info；`fused` 是"内核偏好"而非硬要求（它是默认值），
  且在 CPU 上 for-loop 与 foreach 实测逐位一致（5 步 AdamW，`torch.equal` 全真）。CUDA/XPU 上
  行为与上游逐字相同。测试：改写 `test_the_implementation_flag_reaches_the_inner_optimizer`
  为按设备判定，新增降级用例与**回归用例**（默认 config 在 CPU 上真的 `step()` 一次）。
  (d) 验证：全量 CPU 套件 152 passed / 59 skipped / 9 failed（失败集同基线）；optimizer
  子套件 43 passed（2 个 `test_optimizer_config` 失败仍是缺 `torch.distributed.pipelining`）。
- 2026-09-28 十三次增量（`components/` 全树收口：`loss.py` + `tokenizer.py` 走查）：这两个文件
  不在 `walkthrough` §12 第 3 项的子清单里（loss 由 vocab-loss 批次覆盖过半、tokenizer 属
  A1），因此是回答"components 是否完全对齐"时才补走的。结论：
  (a) **`components/tokenizer.py` ↔ 上游同名文件：对齐。** AST 归一化后 11 处差异全是
  `Configurable`/嵌套 `Config` 的移除（`BaseTokenizer`/`HuggingFaceTokenizer`/
  `MultiModalTokenizer` 改成显式构造参数）、`open(..., 'r')` 之类现代化，以及上游 PR-1540 的
  `assets/tokenizer` 弃用提示（llmtuner 无该历史路径）。`apply_chat_template` 注入 bos/eos
  kwargs 与 `add_generation_prompt=True` 是已登记的上游 backend tokenizer 同源改动。
  (b) **`components/loss.py` ↔ 上游同名文件：数值路径逐行对齐，结构差异四条（三条本轮新登记）。**
  逐行核对的是最要紧的那段：vocab-parallel `forward` 的三个 TP all-reduce（max → sumexp →
  gather）、shard 边界公式（`chunk_size = ceil(V/tp)` + 双侧 `min(V, ...)`）、`out_of_range`
  掩码、`backward` 的融合导数（`grad_update = out_of_range - 1` 与
  `(grad_input + exp(log_probs)) * grad_output`）、`vocab_parallel_entropy`（当时名 `_vocab_parallel_entropy`）的 `isneginf`
  守卫与 stack 后单次 all-reduce、`compute_logprobs` 的两条分支、`mse_loss` 的
  `float().detach()` 与 sum 归约——与上游逐行一致（含 `reduction="none"` 的 `[T]` 语义）。
  **新登记差异**：(1) 上游 `cross_entropy_loss` 暴露 `reduction: sum|none`，llmtuner 固定
  `"sum"`（`"none"` 只有 `LossParallelCrossEntropy.apply` 与 `compute_logprobs` 用），无消费者；
  (2) 上游 `ChunkedLossWrapper` 支持**多输出** tuple pred/labels（dMTP 类模型）并返回
  `(loss, metrics)`（`_combine_chunk_metrics` 逐 chunk 合并），llmtuner 的
  `chunked_lm_head_cross_entropy` 只接单个 `(T, H)`、只返回求和 loss；(3) 上游用预分配缓冲的
  `GradAccumulator`（就地拷贝），llmtuner 用 list + `torch.cat`（多一次 `T*H` 拷贝）。
  另登记：上游的 `spmd_typecheck` 静态断言不移植（llmtuner 无 spmd 曲面），
  `_LossParallelCrossEntropy`/`_VocabParallelEntropy` 去私有化；`compute_logprobs`/`mse_loss`
  在 llmtuner **无生产调用者**（上游分别只服务 `rl/` 与 flux 的 `MSELoss`，均在裁剪面内），
  仓内唯一引用是 `tests/integration_tests/vocab_parallel_loss_equivalence.py`，按"移植曲面"保留；
  chunk 循环期间不改 lm_head 的 FSDP reshard/grad-sync 早前已登记为性能差异，此处不重复。
  (c) 结论：`llmtuner/components/` 六个单元（checkpointer、metrics、profiler、optimizer、
  loss、tokenizer）已全部走查完毕；未对齐项只剩逐条登记的差异与 D 表条目
  （`fused_opt_states_bf16`、`optimizer_factory_kwargs_by_name`、范围外的 `DistMuon`，以及
  未安装/未跑过的 `torch_checkpointing` 后端）。本轮无代码改动，套件数字不变
  （152 passed / 59 skipped / 9 failed）。
- 2026-09-28 十四次增量（`walkthrough` §12 第 4 项开篇：`datasets/` 核心走查）：`components/data/`
  与 `llmtuner/datasets/` 的六个核心文件逐一做了 AST 归一化对照 + 逐行读数值相关段，结论是
  **忠实移植**，差异全部是已登记的"去 Configurable/去 microbatch 类型"那一类。
  (a) `packing.py`（3 hunk / 28 行）：只有两处 `Config.build()` → `build_*_packing()` 自由函数，
  grain 图本身逐字相同——`length_struct`/`padding_struct`（`labels` 填 `IGNORE_INDEX`、`padding_mask`
  填 `True`）、`meta_features=("labels","positions")`、`seed`/`shuffle_bins`/`num_packing_bins`/
  `max_sequences_per_bin`、`DocumentAwareConcatThenSplitIterDataset/Iterator`（2026-09-29 去私有化，原名带 `_` 前缀；含 remainder
  `get_state/set_state`）、`next_document_chunk_end`/`packing_output_is_full`（两者均 2026-09-29 去私有化，原名带 `_`）全部无差异。
  (b) `collators.py`（4/21）：`TextCollator` 的载荷逐字相同（zeros + `torch.cat` + 超长 raise +
  `positions[num_tokens:].remainder_(max_context_length)` + `num_valid_tokens=(labels !=
  IGNORE_INDEX).sum()`）。差异只有类型面：上游 `TrainingMicrobatch`/`TokenizedTrainingMicrobatch`
  两个 dataclass → llmtuner 的 `TrainerBatch: TypeAlias = dict[str, Any]`，字段名 `input` 不变；
  `num_rows_per_microbatch()` → `num_rows_per_batch()`。**一处需要登记的设备面差异**：
  `HAS_PIN_MEMORY` 上游取 `torch.accelerator.is_available()`（"机器上有加速器"），llmtuner 取
  `should_use_pin_memory()`（"**本进程解析出的**设备是加速器"）——在"有 GPU 但 `--use_cpu`"的
  机器上前者为 True、后者为 False；pin 内存只对加速器拷贝有意义，故 llmtuner 的判定更贴合实际，
  且 CPU-only 机器上不会去 pin。
  (c) `dataset.py`（6/93）与 `sources.py`（6/63）：同样只有 Configurable → dataclass + 自由函数的
  形状变化。`dataset.py` 的 DP 分片数学被提成 `shard_for_dp`（`divmod` + `min(rank, remainder)`
  的错位切分逐字相同，`DatasetConcat` 内联的那份也改调它，公式一致）、子策略
  （`shuffle=False, repeat=False, dp_rank=0, dp_world_size=1`）、mix 的 `seed + index` 派生、
  `MapDataset.mix`/`IterDataset.mix` 分派全部一致；`sources.py` 的索引 JSONL 解析、
  `split_dataset_by_node`、streaming 不支持精确 resume 的拒绝、`load_dataset` 的一等字段与
  kwargs 冲突校验（提成 `reject_duplicated_hf_fields`，当时名 `_reject_duplicated_hf_fields`）全部一致。
  (d) `types.py`（3/27）与 `loader.py`（3/23）：`DatasetBuildContext`/`DatasetIterationPolicy`
  新增 `__post_init__` 校验（已登记）；`TrainingMicrobatch` 类层次删除符合 microbatch 分叉；
  llmtuner 另加 `Batch` dataclass（`input_ids`/`labels`，合成路径用；放这里是为了避免
  models 层反向 import 数据源，模块 docstring 已说明）。loader 的
  `dataset.batch(collator.num_rows_per_batch(), drop_remainder=repeat, batch_fn=collator)` 与
  `ThreadPrefetchIterDataset` 逐字相同，差异只是"数据集图由调用方建好再传进来"（C 类单入口设计）。
  (e) `datasets/text/processors.py` ↔ 上游 `hf_datasets/text_datasets.py`：SFT 信号的三条关键规则
  **逐字一致**——超长丢弃（`len(tokens) - 1 > max_context_length`，按 `-1` 而非全长）、prompt 段
  labels 置 `IGNORE_INDEX`（`labels[:max(prompt_len - 1, 0)]`）、renderer 路径用
  `~loss_mask[1:]` 掩码；`DATASETS` 注册表三项（c4/c4_test/c4_validation）同源。llmtuner 新增
  `make_local_jsonl{,_sft,_multiturn}` 三个工厂（C 类，服务 CLI 的本地语料）与 `renderer` 参数
  直通。**交叉验证**：llmtuner 的 tokenizer 抄的是上游 `experiments/transformers_modeling_backend/
  tokenizer.py` 的 `apply_chat_template`（默认 `add_generation_prompt=True`），而上游 SFT 数据路径
  用的是基础 `components/tokenizer.py` 的 tokenizer（HF 默认 False）——所以 llmtuner 的
  `ChatProcessor` 必须显式传 `add_generation_prompt=False` 才与上游渲染一致，代码里正是这么写的，
  该写法与它依赖的那个默认值同源。
  (f) 环境与验证边界：`tests/unit_tests/cpu/datasets/` 的 3 个文件（含 57 例的
  `test_data_pipeline.py`）整体被 `require_env('grain')` 门控，本机未装 grain 故**全部 skip**；
  pyproject 钉 `grain==0.2.18`，而本机 PyPI 镜像只有 `0.2.3`（`pip install --dry-run` 甚至误报
  可装，`pip download` 直接失败），版本不符，故**不在本机装**——装错版本会让这批用例的结论失去
  意义。因此本轮的 `datasets/` 结论是**代码级对照**（AST 归一化 + 逐行读），不是运行期证据；
  运行期等价性仍需在装了 grain 0.2.18 的环境复跑这 57 例。
  (g) `datasets/` 余下未走：`random_data.py` 与 `build.py`（都是已登记的 C 类）、
  `text/renderer.py`（已登记的可选导入适配层）、`multimodal/*`（下一轮）。`models/common/*`
  的数值等价性（§12 第 4 项的后半）尚未开始。
- 2026-09-28 十五次增量（`datasets/` 重构与优化）：
  (a) **去重三处，全部行为不变**（用归一化 AST diff 逐条确认改动范围）：
  (1) `lambda sample: sample is not None` 在 `text/text.py`（6 处）与 `multimodal/mm_datasets.py`
  （3 处）共 9 份，提成 `dataset.py::is_not_none` —— 每个 processor 都用返回 `None` 表示"丢弃这行"，
  过滤谓词本就只有一个；具名后九处引用读作同一契约，且谓词可 pickle。
  (2) `if isinstance(graph, grain.MapDataset): graph = graph.to_iter_dataset(read_options=...)`
  在 6 处重复（`packing.py` ×3、`loader.py`、`multimodal/mm_datasets.py`、`dataset.py` 的 mix 子节点），
  提成 `dataset.py::as_iter_dataset(graph, context=...)`；`loader.py` 那处原用局部 `read_options`，
  与它构造 context 时传入的是同一个对象，故语义一致。
  (3) `packing.py` 的 4 键 `length_struct` 字面量出现两次（concat-then-split 与 first-fit），
  提成模块内 `row_lengths(context)`（当时名 `_row_lengths`）；`text/text.py` 的三个 `make_local_jsonl*` 工厂各自重复
  `SingleDataset(source=IndexedJsonlSource(patterns=(path,)), ..., post_filters=...)`，提成
  `local_jsonl_recipe(path=, processor=)`（当时名 `_local_jsonl_recipe`）。全部是同值替换，无行为变化。
  (b) **一次"差一点改错"的记录（有价值，留档）**：对照上游时发现 `process_cc12_wd_sample`
  缺了上游的 `if image is None: texts=[text]` 兜底，看起来是"缺图行会崩"的 bug——实测确认
  `insert_vision_placeholders([None, "text"], [])` 会 `TypeError: sequence item 0: expected str
  instance, NoneType found`。但进一步读仓内用例发现**这是本仓有意为之**：
  `test_insert_vision_placeholders_rejects_a_slot_with_no_token_count` 与
  `test_process_cc12_wd_sample_raises_when_the_image_field_is_absent` 把"缺图 = 样本畸形 → 响亮失败"
  钉成了契约（上游则是静默降级为纯文本行）。故**不改行为**，只在 `process_cc12_wd_sample` 处
  留注释说明该分叉并指向用例；差异登记于 symbol guide §6。
  (c) 有意不动的重复：`process_cc12_wd_sample`/`process_obelics_sample`（当时名 `_process_obelics_sample`）那份九参数显式转发表
  与上游逐字相同（`_process_mm_sample` 的上游调用方也这么写），属"上游形状"，不为了 DRY 而偏离。
  (d) 验证：本机**无法运行** `tests/unit_tests/cpu/datasets/`——pyproject 钉 `grain==0.2.18`，
  而该版本在 PyPI 上只有 macOS **arm64** wheel（`macosx_11_0_arm64`），没有 x86_64，本机是 Intel
  mac 故装不上；能装的最新版 `0.2.3` 又缺 `ConcatThenSplitIterDataset`（默认 packing 用的就是它），
  拿它跑不能说明问题，故不装。`tests/caps.py` 的 `grain` 探针探测的是 `grain.python`（会连带 import
  jax），所以本机这批用例继续 skip。**本轮的验证手段**：逐文件归一化 AST diff（证明"只改了预期的
  那几处"）、`ruff check`（含 F821 未定义名检查）、以及把被修/被查函数的实际调用跑通——
  `process_cc12_wd_sample` 在纯文本行上的行为用 sys.modules 里的 grain 桩真跑过（确认兜底会去
  `insert_vision_placeholders` 的那条 TypeError 路径，也确认显式 raise 是当前行为）。
  运行期等价性仍需在装对 grain 的机器（Linux/arm64 mac）复跑这 57+ 例。
- 2026-09-28 十六次增量（`walkthrough` §12 第 4 项的 `models/common/*` 走查 + 一处真缺口修复）：
  (a) **逐文件 AST 归一化复核**（`models/common/` 全部同名文件；上游独有文件另计）：
  `moe.py`(17 hunks)、`aux_loss.py`(12)、`rope.py`(12)、`async_linear.py`(12)、`activation.py`(9)、
  `linear.py`(6)、`embedding.py`(6)、`feed_forward.py`(6)、`multimodal.py`(3)、
  `__init__.py`(2)、`token_dispatcher.py`(28)。逐处定性后**无新增代码缺口**，结论与登记见
  symbol guide §4；其中三组要点：
  (1) `moe.py`：结构性差异（`RoutedExperts` 只持 `GroupedExperts`+dispatcher；`tokens_per_expert_E`
  归 MoE；`expert_bias_E` 由 MoE 按 quantile 路由注册）与两处**有意分叉**——上游 MoE-under-TP 的
  三个 `_maybe_*_across_tp` 方法在 llmtuner 由块边界 AG/RS 对偶 + tp×ep 的 T/tp 分片直通替代；
  AC 下的 token 重复计数不去重（上游 `// 2` 是给 expert-usage 指标用的，llmtuner 的
  `sign(mean − x)` 对任何正的均匀缩放不变，也不记录该指标）。归约轴口径：
  上游 = `loss` mesh(dp×cp) + EP 时 dense-TP；llmtuner 显式枚举 `dp`+`cp`(+EP 时 `tp`)，
  同一 rank 集合（不能复用 llmtuner 的 `loss` mesh——它含 tp，见 `parallel_dims` 文档）。
  `MicrobatchWiseLoadBalanceLoss` 的 `('cp','tp' if EP else 'cp')` 与上游 `('cp','tp')` 在
  可达域内等价（上游在 tp>1 无 EP 时于 `MoE.forward` 断言拒绝）；丢掉的
  `spmd_local_context('dp')` 是类型检查上下文，llmtuner 必须在 ambient dense mesh 上查 tp group。
  (2) `aux_loss.py` / `rope.py` / `async_linear.py` / `activation.py` / `linear.py` /
  `embedding.py` / `feed_forward.py` / `multimodal.py`：去 Config/Module 协议与 spmd 注解的同型差异；
  `linear.py` 保留 `RouterGateLinear`/`PartialBiasRowwiseLinear`（后者仅测试使用），
  `Linear`/`ColumnParallelLinear`/`RowParallelLinear`/`GroupedLinear` 分别由 `nn.Linear`+TP realizer、
  `GroupedExperts`、`cast_linear.py` 承接；`activation.py` 只留 `SwiGLU`（`Sigmoid`/`Softmax`/
  `SqrtSoftplus` 由 router 的字符串打点替代，`SiTUGLU` 只在 Kimi 面）。
  (3) `token_dispatcher.py` 索引数学逐行核对：`_local_reorder`/`_permute`/`_unpermute`/`combine`
  与上游逐字一致（含 rank-major→expert-major 的 `input_starts[seg_ids] + arange −
  output_starts[seg_ids]`），`all_to_all_single`+`materialize` 对应上游编译分支的
  `all_to_all_single`/`wait_tensor`。**同轮修正一处文档错误**：`BaseEPTokenDispatcher.num_experts`
  被写成"每 rank 专家数"，实为全局数。
  (b) **上游独有三文件定性**：`param_init.py`（已删死代码）、`lora.py`（裁剪面：LoRA 由 HF/peft 提供）、
  `config_utils.py`（**新增登记**：上游 config 工厂层，llmtuner 无 config tree，其判定分别落在
  `expert_parallel/probe.py` / `parallel/matrix.py` / `models/hf/factory.py` /
  `models/hf/model.py` 的 `flex_supported`，逐函数映射见 symbol guide §10）。
  同轮定性上游实验目录里三个此前未登记的文件：`module_conversion.py`（把 HF 模块 `__class__`
  换成 `Module` 协议子类，好让 Module registry 的 `parallelize()` 生效）—— llmtuner 无
  `Module` 协议层，故无对应物，同类 `__class__` swap 技术用于
  `GatherSequenceFirst`/`TPMoeSequenceBoundary`/TP realizer；`config_registry.py`（实验用家族
  config 注册表）—— llmtuner 用 HF `AutoConfig` + `config/` 门面 + `hf/factory.py`；`__init__.py`
  的模型注册表 —— HF auto mapping（`resolve_model_class`）。
  (c) **代码改动：`num_flops_per_token` 重写为结构感知**（本轮唯一实现变更；原"统一近似"对
  MoE 系统性偏低——Qwen3-MoE 约 2×、DeepSeek-V3 约 1.8×，MLA 的 `head_dim` 还是 rope 切片）。
  新增纯函数 `flops_per_token(arch, *, seq_len)` 与上游同名 `quadratic_attention_flops_per_token`：
  MoE 层 = router + top_k 路由专家（active ratio）+ 全部 shared 专家；MLA 用 `q_lora`/`kv_lora`
  与 `qk_head_dim`/`v_head_dim`；`layer_types` 支持 full/sliding/chunked；稠密与 MoE 混栈按
  `first_k_dense_replace`/`mlp_only_layers`/`decoder_sparse_step`/`moe_layer_freq` 分层。
  **拒绝猜**：几何不可解析时返回 0（缺尺寸、MoE 宽度或层划分不明、`layer_types` 短于层数、
  参数面不可推导的层型如 `linear_attention`），延续原契约；未移植 `delta_rule_flops_per_token`。
  验证：DeepSeek-V3 真实 config 反推 active params = 3.64e10（发布值 37B，差值即未计入的
  norms/biases —— 一个既错 MLA 项又错专家权重的公式不可能落到这个数）；稠密路径与旧公式逐位
  相同；20 个新用例在 `tests/unit_tests/cpu/models/test_flops.py`（**不依赖 pipelining/wandb/flex
  门禁**，真实 HF config 驱动）全绿。
  (d) 顺带修掉两个**从未运行过**的坏用例：原 `test_metrics.py` 的 FLOPs 用例 monkeypatch
  `hf/model.py` 里从未导出的 `build_model_config_for` 并调 `num_flops_per_token`（两者都在
  `hf/factory.py`），而包装模块
  从不导出这两个名字；该模块又被 `require_env('wandb','pipelining','flex_attention')` 整体
  skip，故一直未暴露。FLOPs 用例已迁至新模块，`test_metrics.py` 删去该段与三个失去用途的 import。
  (e) 验证：`tests/unit_tests` = 172 passed / 59 skipped / 9 failed（失败集与上轮完全一致：
  7 例 profiler-OOM + 2 例 optimizer_config 缺 `torch.distributed.pipelining`）；`ruff check`
  通过。**未覆盖**：真多卡 EP/CP/PP 数值等价、CUDA/XPU fused kernel、`torch_checkpointing`
  后端、renderer 真库路径（环境不可达，见 §12 验证边界）。
- 2026-09-28 十七次增量（`models/` 目录与文件组织重构；**只搬家，不改行为**）：
  (a) **新布局**：`models/` 分成两半——`hf/`（HF 适配层，B 类）与 `common/`（模型词汇表）。`common/`
  里两个「文件家族」升级为子包，其余仍是一文件一节点：
  ```
  models/
    __init__.py            两半的导览（不 re-export）
    hf/                    __init__.py + factory.py + model.py + flops.py
                           + state_dict_adapter.py
    common/
      __init__.py          发现面索引（re-export，仍是 `from .models.common import MoE` 的入口）
      activation.py  async_linear.py  aux_loss.py  cast_linear.py  embedding.py
      feed_forward.py  linear.py  multimodal.py  rope.py  scatter_add.py
      attention/           __init__.py（纯文档，见下）+ qkv.py + masks.py
      moe/                 __init__.py（纯文档）+ block.py + router.py + experts.py
                           + dispatcher.py + load_balance.py + balancing.py
  ```
  旧→新逐条映射（代码、文档、用例同轮改完）：
  | 旧 | 新 |
  |---|---|
  | `models/hf_wrapper.py` | `models/hf/wrapper.py`（十八次增量再改名 `models/hf/model.py`） |
  | `models/hf_factory.py` | `models/hf/factory.py` |
  | `models/hf_state_dict_adapter.py` | `models/hf/state_dict_adapter.py` |
  | `models/common/moe.py`（MoE 本体） | `models/common/moe/block.py` |
  | `models/common/moe.py`（`RoutedExperts`） | `models/common/moe/experts.py` |
  | `models/common/grouped_experts.py` | `models/common/moe/experts.py` |
  | `models/common/routers.py` | `models/common/moe/router.py` |
  | `models/common/token_dispatcher.py` | `models/common/moe/dispatcher.py` |
  | `models/common/moe.py`（`MicrobatchWiseLoadBalanceLoss`） | `models/common/moe/load_balance.py` |
  | `models/common/balancing.py` | `models/common/moe/balancing.py` |
  | `models/common/qkv.py` | `models/common/attention/qkv.py` |
  | `models/common/masks.py` | `models/common/attention/masks.py` |
  (b) **三条组织约定**（写进各包 docstring）：① 子包索引只做导航，不 re-export——导入一个节点不得
  拖进它的兄弟（`dispatcher` 需要 EP collectives、`load_balance` 需要 SPMD mesh 上下文；
  `attention/__init__.py` 更必须保持惰性，因为 `masks.py` 在模块层 import
  `torch.nn.attention.flex_attention`，否则 `models.common` 在 CPU/无 flex 的 torch 上直接不可导入）；
  ② 发现面索引只有 `models/common/__init__.py` 一个（平名字可从那里取）；
  ③ `moe/block.py` 里的 `__getattr__` 懒导出 hack 删除（拆包后循环依赖消失，改为调用方直接
  import 叶子）。
  (c) **机械等价性证明**：写脚本把 `git show HEAD:<旧路径>` 的每一个顶层 `def`/`class` 去 docstring
  归一化后与它在新区里的位置逐一定比——**57 个顶层定义、0 个函数体变化**；唯二的「缺失」是
  `RoutedExperts`（已证 byte-identical，落在 `moe/experts.py`）与被删除的 `__getattr__`。
  行为不变的其余证据：`ruff check`（含 F821/I001）通过；用 `spmd_types`/flex 垫片把 26 个
  models 模块 + `parallel/*`/`components/metrics` 逐个 import，28/30 成功（两个失败是环境既有限制：
  `trainer.trainer` 缺 grain.experimental、`trainer.builder` 触发 transformers 5.9 在 torch 2.2.2
  下的内部 `NameError`，与本轮无关）；`tests/unit_tests` = 172 passed / 59 skipped / 9 failed
  （失败集与上一轮完全一致）。
  (d) **有意不动**：类名/函数名一个没改（它们与上游符号一一对应，是映射表的锚点）；测试模块
  文件名保持原样（`test_hf_wrapper.py` 等仍测同一模块，只是模块路径变了）；`models/hf/model.py`pper.py`
  的 import 段行数不变，故文档里 `wrapper.py:591/639` 这类行号引用仍指向原语义。

- 2026-09-28 十八次增量（`models/hf/` 收尾：按职责拆算术、按上游改名）：
  (a) **`factory.py` 拆出 `flops.py`**：前者 581 行里混着四件事（HF config 构造 / 类解析 /
  meta materialize / FLOPs 计算），后者的纯算术（`flops_per_token` +
  `quadratic_attention_flops_per_token` + 六个私有 helper，322 行）没有构建语义，独立成节点后
  `factory.py` 回到 259 行、职责单一。入口 `num_flops_per_token(cfg)` 仍留在 `factory.py`——它
  需要先解析出 config，再委托 `flops.flops_per_token`；`flops.py` 只依赖
  `transformers.configuration_utils`，不依赖 torch。
  (b) **`hf/wrapper.py` → `hf/model.py`**：上游对应文件就叫
  `experiments/transformers_modeling_backend/model.py`，改名后映射 1:1、类名不变
  （仍是 `HFTransformerModel`）。纯 rename，行号不变，故文档里 `model.py:591/639` 的引用语义不动。
  (c) 验证：以十七次增量提交为基线做「按定义」比对——`factory.py`+`flops.py` 16 个顶层定义、
  `wrapper.py`→`model.py` 7 个顶层定义，**0 个函数体变化**（纯搬家）；`ruff check` 通过；
  垫片环境下 9/9 目标模块可导入（含新 `flops.py`，且同一 Qwen3-MoE config 的 FLOPs 与拆分前
  逐位相同：1151827968）；`tests/unit_tests` 172 passed / 59 skipped / 9 failed（失败集不变）。
  `models/` 模块数 26→27，全仓 122→123 个 `.py`（99 实现模块）。

- 2026-09-28 十九次增量（D 表该批的 DSA 稠密 mask 落地 + 三项可读性/健壮性改进）：
  (a) **DSA 稠密 additive mask（移植）**：`models/common/attention/masks.py::build_dense_attention_mask`
  （纯张量：`[1,1,T,T]`，0 允许 / -inf 屏蔽，`block_causal` 与 flex 的 causal+same-document 同义），
  `models/hf/model.py::get_attention_masks` 对 `uses_dsa(config)`（`index_topk`）走该分支——上游
  `_build_dense_attention_mask` 的逐行等价物；`__init__` 里原来的构造期 `NotImplementedError` 换成
  `self.uses_dsa = uses_dsa(config)`。flex 路径不变：HF 的 flex 集成按 mask 类型分支，稠密张量当
  `score_mask`（上游注释同义）。**同轮显式拒绝 CP × DSA**（`_get_cp_attention_masks`：
  CP 的 mask 通道只接受 BlockMask，稠密张量要手工 Q 切分 + 过 load balancer，未验证不给近似）。
  验证：新用例把稠密 mask 与 `get_causal_mask_mod` / `get_document_mask_mod` 在同一索引网格上比对
  （`causal & same_document` 逐位相等），另有 dtype/shape/取值集合断言。
  (b) 附带改进（同轮，均为可读性/可验证性）：① `models/common/__init__.py` 改成**惰性索引**
  （PEP 562 + `_EXPORTS` 表 + TYPE_CHECKING 静态视图），兑现它自己文档里"命名一个节点不得导入它"
  的承诺——此前 `import llmtuner.models.common.rope` 会连带 MoE 栈与 `DTensor`；② `attention/masks.py`
  的 flex import 惰性化（`flex_ops()` 缓存），于是该模块在无 flex 的 torch 上可导入，DSA 稠密 mask
  也因此能在本机跑用例；③ `capabilities.is_compiling()` 收掉 `torch.compiler.is_compiling` 在
  torch 2.2 缺失的问题（rope 的 bounds check 调用点），避免纯 torch 模块在新老版本上二选一。
  **测试口径变化**：`test_masks.py` / `test_rope.py` / `test_multimodal.py` 三处 `require_env('spmd_types')`
  门禁被证实是惰性的（它们只依赖纯 torch 代码）故移除，新增 `test_models_index.py`（惰性索引），
  `test_masks.py` 增稠密 mask 段。顺带修掉一处此前被门禁掩盖的失效 import
  （`test_models_common.py` 仍在 `from llmtuner.models.common import grouped_experts`）。
  全量：**258 passed / 56 skipped / 9 failed**（失败集仍是那 9 项环境问题：7 profiler-OOM +
  2 optimizer_config 缺 pipelining）；相比本轮之前 172 passed，多出的用例全部是此前被门禁
  跳过、现在真跑起来的。
  (c) **同轮试做、随后整条撤下的部分（留档）**：单进程 fake 后端（`init_distributed` +
  `NGPU`/`FAKE_PP_RANK` + `DistributedTopology` 类型与 `parallelism.comm_backend` 配置字段）曾实现并在
  本机验证通过（`NGPU=8` → 单进程逻辑世界 8、DeviceMesh 可建；`FAKE_PP_RANK=2`×pp=4 → 物化逻辑 rank 4），
  但**按用户要求撤下**：不引入 `DistributedTopology`、不引入 `comm_backend`，`accelerator/dist_utils.py` /
  `config/parallel.py` / `trainer/builder.py` 回到原状，该能力继续按 D 表登记为缺口（上面那两行）。
  撤下的原因值得记下：它是本批唯一需要新增配置面与类型面的东西，而那两项的收益（单机模拟多卡）在
  本仓的验证边界里仍属"未覆盖"，不如保持接口不变。

- 2026-09-29 二十次增量（`_` 私有面：全仓审计 + 两批去私有化）：
  (a) **新规则**（写进 symbol guide 第五条横切约定）：模块级 helper / 数据类**默认公开**；
  前导 `_` 只保留三种——① 框架协议要求的名字（`scatter_add.py` 的 `_backward`/`_setup_context`
  是 `torch.library.custom_op` + autograd 的契约）；② 与"公开包装"配对的低层核心
  （`accelerator/dist.py` 的 `_broadcast_object_list`/`_all_gather_object`/`_gather_object`：
  同名的公开版返回 list，私有版收 out-list，去前缀会重名）；③ 类内 protected 方法
  （子类/同包协作者的 template-method 钩子）。类内 `self._x` 属性属于封装，不在本条范围。
  (b) **模块级：95 → 5 个前导 `_`**。审计脚本枚举全部模块级 `def`/`class` 后，按"跨模块使用
  → 必须公开"先处理了 6 个（`packing/{conversions,iterators}` 的 5 个 + `parallel/
  pipeline_parallel/apply.py::scalar_loss_fn`，见下一条十六次/二十次记录），随后按新规则把
  其余 90 个一并公开（含 32 个上游同名 helper：`_yarn_inv_freq`→`yarn_inv_freq`、
  `_maybe_check_max_pos`→`maybe_check_max_pos`、`_RouterGateLinearFunction`→
  `RouterGateLinearFunction`、`_get_default_save_ops`→`get_default_save_ops`、
  `_maybe_enable_async_tp`→`maybe_enable_async_tp`、`_get_pipeline_metadata`→
  `get_pipeline_metadata`、`_EMAParamOptimizer`→`EMAParamOptimizer`、
  `_BackendCheckpointStorage`→`BackendCheckpointStorage` 等），并顺带把两处"去前缀会读歪"
  的改成更准确的名字：`hf/flops.py::_MoE`→`MoEGeometry`（避免与模型类的 `MoE` 混淆）、
  `checkpointer/torch_checkpointing.py::_Backend`→`BackendConfig`（避免与 torch 的
  `Backend` 枚举混淆）、`components/metrics.py::_Derived`→`DerivedMetrics`。
  同时 `scatter_add.py` 中 `@register_fake` 的匿名 `def _` 命名为
  `deterministic_scatter_add_fake`。**上游同名 helper 的拼写分叉是有意的**：symbol guide §10
  的映射行仍以"llmtuner 公开名 ↔ 上游 `_name`"记录（`docs/` 里历史条目保留旧名，读法见本条）。
  (c) **方法级：109 → 103 个前导 `_`**。13 个被"别的模块"调用的私有方法里，6 个 llmtuner 自有的
  公开（`Trainer.seed_everything`/`example_model`/`loss_vocab_kwargs`/`loss_sum`/`param_context`、
  `FeedForward.split_gate_up`），5 个上游同名 protected 钩子按规则③保留
  （`checkpointer/base.py` 的 `_should_save`/`_purge_stale_checkpoints`/`_create_checkpoint_id`、
  `optimizer.py` 的 `_post_init`/`_validate_params`）。
  (d) **两处踩坑留档**：① 集成测试里各自定义的*本地参考实现* `_loss_sum` 一度被脚本误改
  （会造成 `loss_sum = loss_sum(...)` 自遮蔽），已回退——测试内的私有模型不在审计范围；
  ② 批量改名会顺带打中*第三方私有路径*：`from torch.distributed._functional_collectives import ...`
  被改成 `functional_collectives`（3 个文件 + 1 个测试模块导入失败），已逐条还原为
  `_functional_collectives`。教训：改名脚本必须排除 `模块.名字` 形式。
  (e) 验证：脚本复查"跨模块使用却仍带 `_` 的模块级符号" = 0（仅剩 `capabilities.py` 里对
  `disable_dynamo_lru_cache` 的字符串引用与另一个模块的同名 `resolve` 两处误报）；
  `ruff check`（F821 覆盖全部改名后引用）、`git diff --check`、`compileall` 通过；
  垫片环境下 44/49 目标模块可导入（5 个失败全是本机 `grain` 缺 `experimental` 的既有环境问题）；
  `tests/unit_tests` = 248 passed / 56 skipped / 9 failed（失败集不变）。

- 2026-09-29 二十一次增量（全仓去冗余/去重复复查）：
  (a) **脚本先扫一遍**，四类检查都跑过，结论是"没有大块重复"：
  AST 归一化后**没有两个函数体相同**（只有 1 语句的 `__dir__`/`log` 之类）；同名函数跨模块的
  那些是**有意的分派**（`trainer/trainer.py` 的方法一行委托给 `batch.py`/`validate.py`/
  `pp_steps.py` 的自由函数，供测试 monkeypatch；`checkpointer/base.py` 声明协议、
  `dcp.py`/`torch_checkpointing.py` 各自实现）；常量字面量只有 `components/checkpointer/`
  的 state-key `__all__` 与 `checkpoint_keys.py` 重复（见 (c)）。
  (b) **五份复制粘贴的懒索引实现 → 一份**：`llmtuner/__init__.py`、`accelerator/__init__.py`、
  `parallel/__init__.py`、`components/checkpointer/__init__.py`、`models/common/__init__.py`
  各自重复 `_EXPORT_SOURCES.get(...)` + `AttributeError` 文案 + `importlib.import_module` 相对导入 +
  `__dir__` 排序。抽成 `utils/lazy_exports.py`（`export_names` / `resolve_export`），各索引只留
  自己的"名字 → 子模块"表（这张表才是可发现的东西）与两行 `__getattr__`/`__dir__`。
  `llmtuner/__init__.py` 的 `Trainer` 特例也改用同一张表。验证：五个索引逐一解析成功
  （`accelerator.get_dist_info`/`parallel.apply_tp`/`models.common.SwiGLU`/`checkpointer.canonical_fqn`），
  每个索引对未知名字仍抛 `AttributeError`（不静默返回 `None`）。
  (c) **删死代码（登记 7 个，实际落地 5 个——见二十四次增量 (c) 的更正）**
  （`accelerator/device.py` 里无任何消费者的 mmengine 面）：
  `is_cuda_available`、`is_mlu_available`、`is_musa_available`、`is_mps_available`、
  `is_dipu_available`、`get_max_cuda_memory`、`get_max_musa_memory`（连带不再需要的
  `importlib.util` 导入与那段"逐厂商谓词"注释），模块 docstring 改写为"哪些留下了、为什么，
  其余按本仓对 vendored-but-unused 代码的一贯做法删掉"。保留 `is_npu_available`（`dist.py`
  在用）与 `is_npu_support_full_precision`（后者在二十四次增量中按"有调用者再补"删除）。
  (d) **两个 checkpoint 后端的 `__init__` 去重 20 行 ×2**：`dcp.CheckpointManager` 与
  `TorchCheckpointingManager` 逐字重复的"策略装配"（`self.states` 三键排序 + `load_only`/
  `exclude_from_loading`/`initial_load_*`/`last_save_*`/`export_dtype`/`keep_latest_k`/
  `purge_exempt` 共 12 个字段）提到新增的 `BaseCheckpointManager.__init__`；两个后端现在
  `super().__init__(...)` 后只做自己的存储部分（并各自 `if not self.enable: return`）。
  `EXPORT_DTYPE_MAP` 随之从 `dcp.py` 挪到 `base.py`——此前 `torch_checkpointing.py` 反向
  `from .dcp import EXPORT_DTYPE_MAP`，一个后端依赖另一个后端的常量。验证方式（本机 checkpointer
  测试全被 `dp/dtensor` 门控跳过）：用 AST 比对各后端构造路径上"被赋值的 `self.属性` 集合"，
  `dcp` 22/22、`torch_checkpointing` 24/24，无缺无多；`torch_checkpointing` 里仍在用的
  `OPTIMIZER` 等导入保留。
  (e) **单一事实来源**：`components/checkpointer/base.py` 的 `__all__` 改为
  `list(checkpoint_keys.__all__)`（state-key 名字此前在 `base.py` 与 `checkpoint_keys.py`
  各写一遍），re-export 用 `X as X` 形式让 linter 认账。
  (f) **两处同文件内的重复块提到了有名字的 helper**：① `datasets/dataset.py` 的
  "shuffle → `shard_for_dp` → repeat" 三件套在 `build_map_dataset` 与 `build_concat` 里各写一遍
  （连注释都重复），提成 `apply_iteration_policy(dataset, policy)`（顺序即契约，注释随之搬进
  helper）；② `datasets/multimodal/image.py` 的 `resize_to_pixel_budget` 把 `smart_resize` 的
  整段预算算术（aspect 上限检查 + 取整 + 两个 beta 分支）又抄了一遍，改为委托 `smart_resize`
  （自己只留"先把短边放大到 factor"这步与返回形状的 padding 槽）。
  验证：① 用 AST 比对，新旧调用点的那三段语句**逐字相同**；② 把新旧两个 `resize_to_pixel_budget`
  各自从源码里 AST 取出、在 36 组参数（含放大路径、上行/下行 beta 分支与报错路径）上跑，
  结果逐一相同。
  (g) **空 `__init__.py` 补文档**：`utils/`（列出 gc / logger_utils / lazy_exports 与"为什么是叶子层"）
  与 `components/`（三个模块 + 一个子包的分工，以及 checkpointer 索引为什么懒）。
  (h) 验证：`ruff check`、`git diff --check` 通过；`tests/unit_tests` = 248 passed / 56 skipped /
  9 failed（失败集不变）；重跑"零引用符号"扫描只剩 `deterministic_scatter_add_fake`
  ——那是 `@register_fake` 注册钩子，不是死代码。

- 2026-09-29 二十二次增量（去冗余/去重复复查第二遍 + 删两处无效抽象；含一次**按用户要求回退**）：
  (a) **复制粘贴的校验文案 → 两个有名字的校验器**。`max_num_documents must be positive`、
  `num_packing_bins must be positive`、`dp_world_size must be positive` 三条文案此前各写在
  2–3 处（dataclass 的 `__post_init__` 与同值的普通参数入口各一份），
  `cannot resume after changing the effective data-parallel degree` 写 2 处。新增
  `datasets/types.py::require_positive(name, value)` 与 `datasets/loader.py::require_same_dp_degree(...)`，
  由 `DatasetBuildContext`/`DatasetIterationPolicy`/`GrainDataLoader`/`RandomTokenDataLoader`/
  `packing/build.py`/`multimodal/datasets.py` 共用。验证：把新旧 `__post_init__` 与 dp 检查各自
  AST 取出后跑行为网格（三个 int 字段 + `None` 的 64 组、dp degree 9 组），异常文案与
  正常路径**零差异**。
  (b) **HF 索引文件名单一来源**：`model.safetensors.index.json` 此前在
  `dcp.py::_is_valid_checkpoint`（探测）与 `models/hf/state_dict_adapter.py`（写出/读取）
  各拼一遍，新增 `checkpoint_keys.SAFETENSORS_INDEX` 承载（该模块本就是依赖自由的叶子，
  `models/hf` 导入它不引入 checkpointer 后端面），两处改为引用。
  (c) **两个 checkpoint 后端 12 个共享字段里再省 2 个**：`folder`（`filesystem.join(folder, config.folder)`）
  与 `interval` 在 `dcp`/`torch_checkpointing` 里逐字重复赋值，提到
  `BaseCheckpointManager.__init__`（`folder` 参数）。28 处构造路径上的 `self.属性`集合
  仍与改动前一致（dcp 22、torch_checkpointing 24，`folder`/`interval` 改由基类贡献）；
  checkpointer 单测在本机被 `dcp/dtensor` 门控跳过，故以该 AST 等价性为证。
  同轮把 `components/checkpointer/__init__.py` 的 state-key 映射改为直接从
  `checkpoint_keys` 解析（二十一次 (e) 的 `base.__all__ = list(checkpoint_keys.__all__)`
  随之收尾：`base` 现在只 `__all__` 自己定义的四个名字，不再 import 它用不到的
  `DATALOADER`/`TRAIN_STATE`；对外 `llmtuner.components.checkpointer.<KEY>` 不变）。
  (d) **删两处无效抽象**（本轮的"避免无效的抽象"）：
  ① `components/optimizer/optimizer.py::init_cache_state_dict` —— 只有 `pass` 的 no-op，
  全仓（含测试）**零调用者**，docstring 自述的存在理由是"上游子类会覆写、上游训练循环会
  无条件调用"，而 TorchFT 整个在 llmtuner 裁剪面内（D 表已登记）；已删除，并在 D 表登记
  为有意删除项。② `models/common/rope.py::RoPE` —— 三个钩子（`_precompute_cache`/
  `_reshape_cache`/`apply_rotary_emb`）体是 `raise NotImplementedError` 却未声明抽象，
  基类看着可实例化、实际构造即崩。改为 `ABC` + `@abstractmethod`（与仓内其余 7 个基类
  同一写法）：`RoPE` 实例化现在抛 `TypeError`，`ComplexRoPE`/`CosSinRoPE` 正常构造、
  缓存形状不变。
  (e) **回退：共享日志常量层删除**（用户判定为多余抽象）。二十二次曾把四条
  "两个后端共用的日志文案"提到 `base.py`（`CHECKPOINTING_ACTIVE_LOG` 等），
  **已按用户要求整条撤回**，字面量回到各自后端调用点（与上游逐字同形）；
  `base.py` 的 `__all__` 只剩四个类/契约名。
  (f) 对照扫描（无动作，登记理由）：AST 归一化后**无相同函数体**、无 ≥0.75 相似的近重复函数；
  跨模块相同的 3–4 语句窗口只剩两处——两个后端 `__init__` 的 purge 线程装配与
  `_wait_for_saving`（上游即按后端各写一份，且 `_wait_for_saving` 在 `base` 里是
  显式 `@abstractmethod` 契约），保持不回并；≥25 字符的跨模块字符串重复只剩索引
  re-export 名、能力注册表键与 trainer 委托方法的 docstring，均有意为之；
  `components/checkpointer/utils.py`（仅 `canonical_fqn`）**不并**——它是 `config`/`optimizer`
  依赖的依赖自由叶子，并进 optimizer 会把 checkpointer 后端面反向拖进那条导入链。
  (g) 验证：`ruff check`（含 `tests`）、`git diff --check`、`python -m compileall -q llmtuner`
  全通过；`tests/unit_tests` = 248 passed / 56 skipped / 9 failed（失败集不变，仍是
  profiler 的 7 个 OOM 用例 + optimizer_config 的 2 个缺 `torch.distributed.pipelining`）。

- 2026-09-29 二十三次增量（去冗余/去重复复查第三遍：把上一轮没扫的维度补齐，并修掉两处）：
  (a) **先补扫描覆盖**（这一轮新做的检查项）：跨模块 5–10 语句窗口（标识符/数字归一化后）、
  同一条函数体内的重复语句组、同形状错误/日志文案、跨模块**字面量值集合**（tuple/set/dict/
  `frozenset(...)`）、配置 dataclass 字段的零引用扫描、`except ImportError` 各站点、
  ≥60 行函数与 ≥15 方法类清单。前三项与"字段零引用""死名"三条**结果为空**——即：
  没有可合并的大块重复，没有死的配置字段，没有无人引用的模块级名字。
  (b) **值集合重复 → 修一处**：跨模块同值集合共 5 对，其中 4 对是**有意**的（下方 (e) 登记），
  1 对是真重复：`parallel/activation_checkpoint.py` 的 `VALID_AC_MODES` 与
  `config/training.py::TrainingConfig.__post_init__` 里手写的四个模式各写一遍。
  现由 **config 侧持有**（`config/training.py::VALID_AC_MODES`，紧挨它约束的字段）：
  该文件本来就 import `torch.distributed.pipelining` 做同类校验，方向不变；`config/__init__.py`
  导出；`parallel/activation_checkpoint.py` 改为 `from llmtuner.config import VALID_AC_MODES`
  并保留 `__all__` 里的再导出，故 `from llmtuner.parallel.activation_checkpoint import
  VALID_AC_MODES`（测试在用）与 `apply_ac` 自身的成员校验都不变。验证：两处取到的是
  **同一个对象**；四个合法模式全部接受、`'bogus'` 被拒（消息含字段名与集合）、
  `'region'` 当时仍走 `EnvironmentUnsupportedError` 分支（**二十五次增量已改为正常模式**，
  见该条目）；`test_activation_checkpoint.py` 那条
  `test_valid_modes_are_the_configs_accepted_set` 的断言名至此名副其实。
  (c) **四处同形状的 `>= 1` 守卫 → 一个循环**：`TrainingConfig.__post_init__` 里
  `global_batch_size`/`max_seq_len`/`steps`/`gradient_accumulation_steps` 四段
  `if self.X < 1: raise ConfigError(f"X must be >= 1, got {self.X}")` 逐字相同，且
  `test_config.py` 本就按 `f"{field} must be >= 1"` 参数化这四个名字，`config/parallel.py`
  与 `parallel/parallel_dims.py` 也早就是"一组名字 + 一个循环"的写法；现统一为该写法
  （`chunked_loss_num_chunks` 的文案自带括号说明，单独保留）。验证：8 组（4 字段 ×
  {0, -1}）的异常文案与改动前**逐字相同**（直接构造 `TrainingConfig` 比对，因为
  `test_config.py` 整模块被 `pipelining` 门控在本机跳过）。
  (d) 验证：`ruff check`（含 `tests`，两处 import 排序由 `ruff check --fix` 归位）、
  `git diff --check`、`python -m compileall -q llmtuner` 全通过；`tests/unit_tests` =
  248 passed / 56 skipped / 9 failed（失败集不变）。`parallel/activation_checkpoint.py`
  在本机 torch 2.2.2 下 import 即失败（上游 `torch._functorch.partitioners.get_default_op_list`
  不存在），故该文件的改动以 AST/文本等价 + 上述 `VALID_AC_MODES` 对象同一性为证。
  (e) **登记为有意保留的重复（本轮明确不动）**：
  ① `('alltoall','torchao','deepep','hybridep')`（`config/parallel.py` 的本地校验元组 vs
  `models/common/moe/dispatcher.py::EP_DISPATCHER_BACKENDS`）——两个方向的收敛都别扭：
  config→models 会让配置解析拉进 models 包，models→config 会让每个模型导入拉进 config 包
  （且 `parallel/activation_checkpoint.py` 已 config→parallel，存在环的隐患）。保持两份，
  与 (c) 同形的 `swap.py` 成员校验一起以测试钉住（`test_ep_token_dispatcher.py`）。
  ② `('fused','foreach','for-loop')`（`config/optimizer.py` 的 `Literal[...]` 类型 vs
  `components/optimizer/optimizer.py` 的运行时 `ValueError`）——`Literal` 的参数必须是字面量，
  抽成元组也无法复用同一处，收益仅剩消息文案。
  ③ `EXPORT_DTYPE_MAP` vs `cast_linear.TORCH_DTYPE_MAP`——已在二十一次增量登记：一张是
  checkpointer 能导出的 dtype、一张是 lm_head 能计算的 dtype，**重叠是巧合**，代码与文档均已注明。
  ④ `(0.5, 0.5, 0.5)` 的 `image_mean`/`image_std` 默认值出现在
  `datasets/multimodal/{image,video,datasets}.py` 三处签名默认——三者都是叶子模块，
  提取常量需要一个共同宿主（`image.py`）并让另两个反向 import 它，为纯默认值新增模块间耦合
  不划算。登记在案，改 Qwen-VL 归一化时三处同步改。

- 2026-09-29 二十四次增量（专查"无效抽象 + 过度拆分的小函数"，并**补上二十一次漏删的死面**）：
  (a) **小函数粒度扫描**（四条口径同时判定：行数、仓库内调用点、是否在任一 `__all__`/索引里
  属公开面、是否以**值**形式被传递）：模块级"≤8 行 + 1 个调用点 + 非公开"只剩
  `scatter_add.py::deterministic_scatter_add_fake`（`@register_fake` 注册钩子，不是拆分产物）；
  放宽到 9–14 行仍只剩 `accelerator/device.py::get_max_cuda_memory`（见 (c)）。
  方法侧："≤4 行 + 1 个调用点 + 不覆写基类 + 无装饰器"只剩
  `MetricsProcessor.add_data_loading_time`（3 行记录器，production 1 处 + 测试 3 处）与
  `Trainer.should_continue_training`（`while self.should_continue_training():` 的具名循环条件）。
  结论：**没有需要内联的过度拆分**——库内既没有 1 行转发的包装，也没有只为单一调用点存在的
  短助手。
  (b) **类级抽象扫描**（方法数 / 自有代码行数 / 被继承次数 / 被实例化次数 / 是否 ABC）：
  报出的都是应有的形态——枚举、异常类、dataclass/NamedTuple（"tiny" 是定义使然）、
  `ActivationFn`/`BaseTokenizer`/`BaseEPTokenDispatcher`（ABC + 共享实现 + 单子类，均为 A2 移植）。
  两个 `instantiations=0` 的**假阳性**已核实：`tp.py::GatherSequenceFirst` 与
  `tp.py::TPMoeSequenceBoundary` 是 `__class__` swap 安装的 mixin（`apply.py:154,196` 用类对象
  赋值），从不被调用。
  (c) **补删二十一次漏掉的死面**（本轮唯一实质删除）。二十一次增量登记"删掉无消费者的
  mmengine 面（7 个名字）"，但 `d065928` 实际只删了 5 个（`is_mlu_available`/
  `is_musa_available`/`is_mps_available`/`is_dipu_available`/`get_max_musa_memory`）：
  `is_cuda_available` 与 `get_max_cuda_memory` 仍在 `accelerator/device.py` 里，零调用者，
  且与该模块 docstring 自述的"这些没有调用者、不为自身保留"**直接矛盾**（文档比代码更"干净"）。
  本轮删除这两个，并连带删除同样零消费者的 `is_npu_support_full_precision`（它的唯一效果是
  为一次 `torch_npu.npu.utils.get_soc_version()` 探测而 import 该私有模块）及其专用的
  `from torch_npu.npu import utils as _npu_utils` 包装；模块 docstring 随之改写，
  并注明 `is_npu_available` 保留的原因（`accelerator/dist.py` 的 `broadcast_object_list`
  分支在用）。判断口径即本仓既有的"有调用者再补"：NPU 全精度探测在 llmtuner 内既无调用者
  也无第二处概念引用，将来要接时补回约 10 行即可（从 `torch_npu` 拿 `get_soc_version()`）。
  验证：`rg` 全仓（含 tests/docs）再无这四个名字的代码引用；模块 import 正常，
  `is_npu_available`/`is_device_type_available`/`should_use_pin_memory`/`set_device` 等
  其余面不变；`tests/unit_tests/cpu/accelerator` 12 passed / 1 skipped。
  (d) **复核后判定保留的"移出体外"结构**：`trainer/trainer.py` 有 11 个方法是一行委托
  （`return batch.dp_rank_world_size(self, ...)` 等，bodies 在 `batch.py`/`pp_steps.py`/
  `validate.py`）。本轮核实其理由成立且**不是**可无痛消除的抽象：测试调用的是**方法**
  （`trainer.batch_size_per_rank(2)`、`Trainer.pp_forward_backward_body(...)`、
  `trainer.data_iterator()`），不是那些自由函数；`trainer.py` 的类 docstring 与属性注解
  明确写了"`__new__` 构造的 Trainer 是测试驱动纯 helper 的方式"，`__init__` 委托
  `builder.build_trainer_state` 也有"装配顺序即契约"的说明；全仓**没有**任何
  `monkeypatch.setattr(batch_mod, ...)`。即：拆分的作用是把 1111 行的类留在"状态 + 方法面"
  这一层，正文按关注点分文件，属有意结构而非过度拆分；拆掉它反而要重写这些测试。
  (e) 验证：`ruff check`（含 `tests`）、`git diff --check`、`python -m compileall -q llmtuner`
  全通过；`tests/unit_tests` = 248 passed / 56 skipped / 9 failed（失败集不变）。

- 2026-09-29 二十五次增量（**接入 RegionAC**，AC 四种 policy 至此全部落地）：
  (a) **上游结构**（`torchtitan/distributed/activation_checkpoint.py` + 本轮 clone 的
  `meta-pytorch/remat@d302699b`）：`RegionAC.Config.save_regions` 是一组"相对 transformer
  block"的 glob；`Module.configure_remat_regions` 把 pattern 下推到模块树的
  `_remat_save_patterns`，模型代码在 `remat.region(fn, name, recompute=...)` 的调用点把
  局部名解析成 `attention.qkv` 这类相对名；`RegionAC.apply` 逐 block
  `configure_remat_regions(...)` 再用 `remat.checkpoint(region_name=f"layers.{i}",
  determinism_check=..., preserve_rng_state=False)(module.forward)` 包住 forward。
  `preserve_rng_state=True` 与 `debug=True` 都被上游 config 拒绝。
  (b) **llmtuner 的等价物**：HF 模型既没有 `remat.region` 调用点也没有 `Module` 协议，
  所以声明通道改为**结构等价**——`parallel/remat_regions.py` 把"block 内每个 `nn.Linear`
  的 block 相对 FQN"当作 region 名（`self_attn.q_proj`/`mlp.down_proj`/MoE 的
  `router.gate`），这正是上游 `save_regions` 示例所指的那些投影；`wrap_region` 逐 Linear 装
  `remat.region`、再对 block 装 `remat.checkpoint`。三条取舍写在该文件与 `wrap_region`
  的 docstring 里：① Linear 的 `forward(input) -> Tensor` 才是 `remat.region` 要求的
  扁张量签名（任意 HF 子模块可能收发嵌套结构），② 这些 region 互为兄弟、不嵌套，
  因此撞不上 torch_remat 的"save 里套 recompute"错误与"同一 phase 内名字唯一"约束，
  ③ 词表有意不含 packed 专家权重（`GroupedExperts` 不是 `Linear`）、attention 内部
  softmax（HF 无对应子模块）与 norm/激活（重算很便宜，正是该模式的意义）。pattern 仍按
  **块相对名** fnmatch（上游规则，一份策略覆盖所有 block），交给 torch_remat 的 label 则
  是**块限定名**（`layers.0.self_attn.q_proj`，便于 trace/显存报告区分 block；remat 只把名字
  当标签，匹配在 llmtuner 这侧）。
  (c) **接线**：`VALID_AC_MODES` 增加 `"region"`（第五个取值）；新增
  `RegionACConfig(save_regions, determinism_check="default", preserve_rng_state=False)`，
  `preserve_rng_state=True` 在 config 期即 `ConfigError`（上游与 torch_remat 都拒，文案指向
  `RecomputeStateHook`）；上游的 `debug` 字段**不暴露**（torch_remat 的 checkpoint 没有该
  旋钮，上游只是因为基类带了这个字段）；`TrainingConfig.region_ac` 字段 +
  `parallelize_hf_transformers(region_ac=...)` + `builder` 透传；原先 config 与 `apply_ac`
  两处的 `NotImplementedError` 全部删除。`torch_remat` 走**懒 import**
  （`require_torch_remat`，与 checkpointer 的 `require_torch_checkpointing` 同法），因此
  config 期与本模块 import 期都不需要该包，缺包时 apply 期抛 ImportError 并同时给出
  **torch ≥ 2.10** 与安装命令（本机 torch 2.2.2 装不上：`torch_remat` 的依赖即
  `torch>=2.10.0`，已实测 pip 报错）。
  (d) **验证**：本机 torch 2.2.2 连 `activation_checkpoint.py` 都 import 不了（缺
  `torch._functorch.partitioners.get_default_op_list`、`torch.utils.checkpoint.CheckpointPolicy`、
  `torch.ops.aten.mm.dtype` 三处新 API），所以数值与真包行为**未验证**，如实登记。
  可验证的部分全部钉住：`remat_regions` + `RegionACConfig` + `VALID_AC_MODES` 的 13 条用例
  在本机**实跑通过**（`tests/unit_tests/cpu/parallel/test_remat_regions.py`，非门控）；
  `wrap_region`/`require_torch_remat` 用 AST 取出后在带 fake `torch_remat` 的命名空间里实跑，
  观察到的标注与 checkpoint 调用与设计一致（7 个 region、`self_attn.*`+`mlp.down_proj`
  共 5 个 retained、`checkpoint(region_name='layers.0', determinism_check='default',
  preserve_rng_state=False)`、block 返回且 forward 已替换、forward/backward 可跑通、
  未命中 pattern 记 warning、空 pattern 全 recompute、缺包 ImportError 含 torch ≥ 2.10）；
  AC 测试模块新增 3 条 region 用例（fake 包 + 缺包守卫 + HF 词表断言，本机被 module 级
  env 门控跳过）。解锁条件：torch ≥ 2.10 机器上装 `torch_remat` 后复跑这些用例并
  与未 checkpoint 的 forward/backward 对数值。
  (e) 验证：`ruff check`（含 `tests`）、`git diff --check`、`compileall` 全通过；
  `tests/unit_tests` = 261 passed / 56 skipped / 9 failed（比上一轮 +13，全部是本轮新增用例；
  失败集不变）。

- 2026-09-29 二十六次增量（`full` AC 与上游 `FullAC` 逐字对齐）：
  (a) **查出的差异**：上游 `FullAC._wrap_block` **不是裸 wrapper**，它把恒
  `PREFER_RECOMPUTE` 的 `_full_ac_policy` 经 `create_selective_checkpoint_contexts`
  作为 `context_fn` 传给 `ptd_checkpoint_wrapper`；llmtuner 此前传的是 torch 默认的
  `noop_context_fn`（纯非重入路径）。语义差别在"输出不可重算/带注册副作用"的算子上：
  上游那条路径让 torch 仍落 SAVE（上游注释："Recompute pure operations while PyTorch
  preserves registered effects"），裸 wrapper 则一律重算。该差异此前**没有登记**，
  本轮修掉。torch 侧依据：非重入实现里 `context_fn` 是叠加层，重算由
  `_checkpoint_hook` 驱动（`torch/utils/checkpoint.py` 的
  `_checkpoint_without_reentrant_generator`），所以两条路径同源、不是两种机制。
  (b) **对齐方式**：新增 `full_policy`（上游 `_full_ac_policy` 的同义实现）与
  `wrap_full`（上游 `FullAC._wrap_block` 的同形实现，含 `context_fn`/`preserve_rng_state`/
  `early_stop=True`），`apply_ac` 的 full 分支改调 `wrap_full`；`full` 的
  `determinism_check`/`debug` 仍不暴露，但已核实 torch 的非重入默认值就是
  `determinism_check="default"`、`debug=False`（`torch/utils/checkpoint.py` 的
  `_DEFAULT_DETERMINISM_MODE`），与上游 config 的默认相同——即行为一致、只是不可配，
  原登记项继续成立。
  (c) 验证：本机缺 `CheckpointPolicy`（torch 2.2.2），该模块与相关用例仍被 env 门控
  跳过，故数值未在本机复跑（既有 `test_full_ac_matches_uncheckpointed_bitwise`
  与"内层 forward 两次"的重算证明即那台机器上的守卫）。可验证部分全部实跑：
  AST 取出 `full_policy`/`wrap_full` 后在带 stub 的命名空间执行，确认 wrapper 收到
  `context_fn`（其构建时把 `full_policy` 交给 selective 上下文）、
  `preserve_rng_state`（默认 True）与 `early_stop=True`；新增两条用例
  （`test_full_policy_prefers_recompute_for_every_op`、
  `test_full_ac_runs_on_the_selective_context`）钉住策略与接线。
  `ruff check`（含 `tests`）、`git diff --check`、`compileall` 全通过；
  `tests/unit_tests` = 261 passed / 56 skipped / 9 failed（失败集不变）。

- 检查后续漂移：`git -C <torchtitan> log f35966713..HEAD -- torchtitan/`。
- 2026-09-23 映射修订：上游 `distributed/linear.py` 已删除、内容迁入
  `models/common/dist_gemm.py`（改名 `AsyncAllGatherLinear`/`AsyncLinearReduceScatter`，
  数学不变；上游 9e159aed7 再把该文件改名 `async_linear.py`），此后上游
  async_linear.py 同时对应 llmtuner 的 `parallel/tensor_parallel/linear.py`
  （autograd 原语）与 `models/common/async_linear.py`（模块层，llmtuner 2026-09-26
  同步改名），一对二；
  `distributed/tensor_parallel.py` 已随 DTensor 后端整体删除（`7e7f271e0`）；继任者
  不是另一个同名文件，而是把 TP 变成声明的三处：`protocols/sharding.py`
  （`ShardingConfig`、`in/out_src/dst_shardings`、`local_spmd`）、
  `models/common/decoder_sharding.py`（core 模型的 `norm_config` /
  `token_id_placement` / `set_gqa_attention_sharding`）与
  `experiments/transformers_modeling_backend/hf_sharding.py`（HF 模型的
  `set_hf_sharding_configs` 及其 `_hf_colwise_config` / `_hf_rowwise_config` /
  `_hf_sequence_parallel_placement`）；算子侧是 `models/common/async_linear.py`。
  逐项对应见本文「TP/SP 对齐结论」。
- 早前基线：llmtuner `528dc9d`，TorchTitan `c6e416bbd`。
- 表中的 ratio 除 A2 中明确标为 2026-09-21 复核的十行外，来自早期结构快照
  （llmtuner `f5be809` 附近），只用于解释来源，**不是当前工作树的实时相似度**。源码
  变化后应运行下方脚本重算，不能据旧 ratio 判定漂移。
- 2026-09-22 设备验证补充：Qwen3-8B 已按 TorchTitan 的 meta 构建 → FSDP → `to_empty`
  → checkpoint load 顺序完成 8 卡 HCCL、4096 序列的真实训练，并完成完整 DCP
  save→resume。训练示例显式使用 `last_save_model_only=False`；上游默认的 model-only
  最终 checkpoint 只适合作为导出物，不能作为续训状态。
- 同日并行复核修正了 EP 不应计入 world-size 乘积的 config helper，以及 Torch 2.10
  functional-collective 的 TP fallback API。2-rank FSDP/TP/CP/EP-grad-norm 等价性
  通过；PP 1F1B 的多步轨迹仍有约 `8.5e-3` 最大偏差，保持未通过状态。

## 重现这张表

保存为 `recompute_map.py`，把两个根目录指向你自己的 checkout，然后
`python recompute_map.py | sort -t$'\t' -k1 -rn`。它是产出本表的**实际脚本**（不是
伪码），两分钟内跑完。

```python
"""llmtuner -> torchtitan 结构相似度。输出: ratio \t ratio2 \t llmtuner路径 \t 上游路径 \t 代码长度"""
import ast, difflib
from pathlib import Path

HP = Path("<你的>/TorchLLMTuner/llmtuner")
TT = Path("<你的>/torchtitan/torchtitan")   # 必须限定在 torchtitan/ 下!

def strip_src(p):
    t = ast.parse(p.read_text())
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef, ast.Module)):
            if (n.body and isinstance(n.body[0], ast.Expr)
                    and isinstance(n.body[0].value, ast.Constant)
                    and isinstance(n.body[0].value.value, str)):
                n.body.pop(0)
            if not n.body:
                n.body.append(ast.Pass())
    return ast.unparse(ast.fix_missing_locations(t))

up = {p.relative_to(TT).as_posix(): strip_src(p) for p in TT.rglob("*.py")}

for p in sorted(HP.rglob("*.py")):
    a = strip_src(p)
    if len(a) < 40:            # 空 __init__.py 等,跳过(见下方教训)
        continue
    best = second = 0.0
    best_name = second_name = ""
    for name, b in up.items():
        # 上界剪枝: ratio <= 2*min/(sum)。比不过当前最优就不必构造 matcher。
        if 2 * min(len(a), len(b)) / (len(a) + len(b)) <= best:
            continue
        r = difflib.SequenceMatcher(None, a, b).ratio()   # autojunk 保持默认 True
        if r > best:
            second, second_name = best, best_name
            best, best_name = r, name
        elif r > second:
            second, second_name = r, name
    print(f"{best:.3f}\t{second:.3f}\t{p.relative_to(HP).as_posix()}\t{best_name}\t{len(a)}")
```

使用要点（均为实测教训）：

- **必须先把 torchtitan 的候选集限定在 `torchtitan/torchtitan/` 下**，否则会匹配到
  `experiments/rl/` 之类的噪音。
- **性能**：不要用 `autojunk=False`，91 × 444 会跑十几分钟。默认 `autojunk=True` 加
  那条上界剪枝就能在两分钟内跑完。
- **不要用 `diff` 行数判断改动量**；`quick_ratio()` 也不能用来筛候选——它是上界不是
  估计，会漏掉真正的对应物（实测漏过 `components/loss.py`）。
- **第 2 列是次优匹配**，它比最优值还重要：两者接近说明这个文件的归属有歧义，需要
  人工判；两者差距大，最优才可信。


---

# 附录：符号级对应（原 `llmtuner_torchtitan_symbol_guide.md`，2026-10-08 并入）

文件级分类以本表正文为权威；本附录在其上增加符号级导航。冲突时先修正文件级表，再更新附录。

### 1. 结论与使用规则

当前审计结论：核心训练链路 TP、FSDP2、CP、EP、PP、checkpoint、数据装配和训练循环均有
明确来源或明确的 llmtuner 独立设计。已发现的差异大多是有意移除 `Configurable`、
TorchTitan `Module` 和声明式 `_sharding_config` 后产生的形状变化，而不是算法漂移。

正确性状态采用四档；它描述的是对应行所列契约，不代表该文件在所有设备和并行组合下都已
验证：

- **通过**：所列逻辑与上游等价，且有单测、等价性测试或实际容器验证。
- **通过（适配）**：实现形状不同，但契约和数学语义已验证。
- **受限**：实现正确，但只覆盖 llmtuner 明确支持的组合；表中会写出边界。
- **悬空/决策项**：当前无调用者或缺少完整执行引擎，不应假装已经支持。

操作规则：

1. A 类先比较同名符号，再比较所在文件；可以同步 bug fix，但要保留 llmtuner 的参数入口。
2. B 类只同步不变量、错误检查和数学意图，不复制 TorchTitan 的类层次。
3. C 类没有可同步对象；只用 llmtuner 测试和调用图判断。
4. D 类不是"漏了一个函数"，通常是组合能力或依赖缺失，需要单独设计。
5. 上游新增 `Config`、`build()`、`Module.Config`、`sharding_config` 时，不可机械迁入。

验证上游基线为 2026-09-22 的 TorchTitan `c6e416bbd`；2026-09-23 已审计至
`b64103072`（记录见
`llmtuner_torchtitan_alignment_audit_2026-09-23.md`（不在当前工作区））。
2026-09-27 增量审计至 TorchTitan `c8a3e7666`（`9e159aed7..c8a3e7666`，20 个提交）：
与本仓相关的只有 activation checkpoint 的 `early_stop` 同步（A 类，已改）与一组
`GroupedLinear` 表示重构（B 类，表示等价、无需动作）；逐项结论见
[`llmtuner_upstream_map.md`](./llmtuner_upstream_map.md) 的版本与漂移章节。
最新 `vllm-ascend-env` 容器已实际完成 8 卡 HCCL Qwen3-8B、4096 序列、真实 HF 权重和
真实 SFT 数据的 FSDP2+Full AC 训练，并完成完整 DCP checkpoint 的 save→resume：从
step 1 恢复 optimizer、scheduler、dataloader 和 train state 后完成并保存 step 2。
镜像使用 Torch/torch-npu 2.10；这些是明确路径的设备证据，不代表全套组合均已验证。
2026-09-21 的设备验证记录文件不在当前工作区，相关结论以本文件 §12 与
2026-09-23 审计记录为准。

### 2. 顶层装配与训练

| llmtuner 重要符号 | TorchTitan 对应实现 | 主要差异 | 结论/维护动作 |
|---|---|---|---|
| `trainer.train.parse_config`, `main` | `torchtitan/train.py` | llmtuner 直接构造一个集中式 dataclass 配置；上游构造 Configurable 树；group 嫁接表在 `LLMTunerConfig.from_groups`（2026-09-26 起，自 train.py 内联迁入） | 通过（适配）；同步启动顺序和全局运行时设置，不同步配置树 |
| `llmtuner.config.LLMTunerConfig` 及各子 config | `config/configs.py` 与各组件嵌套 `Config` | llmtuner 的 SEAM 0：全部字段集中；上游字段分散在组件 | 通过（适配）；新增功能必须先落到这里 |
| `LLMTunerConfig.auto_fill_model` | 上游模型 registry/config build | llmtuner 用 HF `AutoConfig` 填充；上游选原生模型 config | 通过；本地模型与 Hub 配置分别测试 |
| `parallel.parallel_dims.build_parallel_dims`, `build_mesh` | `distributed/parallel_dims.py` + `trainer.py` | 上游无单一对应函数；llmtuner 把解析与 mesh 构造分开 | 通过（适配） |
| `accelerator.dist_utils.init_dist_pytorch` | `train.py` 的 PG 初始化段 | 2026-09-24 起为 trainer 的 PG 引导（原 `mesh.init_distributed` 已并入）：trainer 直接调它而非 `init_dist` 门面，避开后者的 `mp.set_start_method('spawn')` 副作用；厂商加速器 backend 由设备层推导，CUDA 路径才消费 `backend` 实参 | 通过；后端特有行为需实际设备验证 |
| `Trainer.__init__` | `torchtitan/trainer.py::Trainer.__init__` | llmtuner 直接接收 HF wrapper、容器和自由函数；没有 Configurable build | 通过（适配） |
| `Trainer.batch_generator` | `Trainer.next_batch`/post-dataloading 路径 | llmtuner 把 dataloader exhausted、CP/TP shard 和设备搬运集中处理 | 通过（适配） |
| `Trainer.forward_backward_step`, `train_step` | `Trainer.train_step` 及 PP/non-PP 分支 | llmtuner 显式支持梯度累积、chunk loss、PP loss；非日志 step 不保留 loss graph | 通过；有 graph 释放回归测试 |
| `ntokens_seen` 计数 | 上游 `training_engine.py` 的计数段 | 每 rank 只计 `labels.numel() // (cp*tp)` 的本地份额，`train_step` 在 dp×cp×tp loss mesh 上求和还原语料总量；2026-09-23 修复（此前 CP/TP>1 时虚高 cp×tp 倍，对应上游 ec953b360 的口径修正） | 通过（适配）；绝对值尚无多 rank 端到端断言 |
| `Trainer._allreduce_replicated_tp_grads` | 上游 SPMD/TP placement 自动归约 | llmtuner 手写 TP plan，复制参数必须显式 SUM | 通过（适配）；新增 TP module 类型时必须更新识别集合 |
| `Trainer.state_dict`, `load_state_dict` | 上游 trainer state Stateful | llmtuner 只保存训练步等最小状态 | 通过 |
| `Trainer.train`, `close` | 上游同名方法 | 生命周期更短；仍保证 profiler/checkpointer/logger drain | 通过 |
| `Trainer.validate`, `should_validate`, `check_validation_feasibility` | `components/validate.py::Validator`（含上游 6c2dadbb3 零 batch/零有效 token 报错、90b25912f dp>1 拒绝 `steps=-1`） | 2026-09-24 移植：`training.validation_config`（`ValidationConfig`，freq/steps/dataset，默认 None 关闭且逐位不变）；eval 模式 + `no_grad`，结束后恢复 train；loss 按全局有效 token 归一化，token 走 dp mesh、loss 走 dp×cp×tp loss mesh，与训练同语义；`steps=-1` 对 random 无限语料亦拒绝；PP 组合 2026-10-02 起支持：`trainer/validate.py::validate_body_pp` 驱动 schedule 的 eval 通路（与上游 `pp_schedule.eval` 同 seam），microbatch 管线与训练体一致 | 通过（适配）；多 rank 归约语义与 PP 组合待目标设备验证 |

### 3. Hugging Face 模型适配层

这是 B 类核心，不能按上游每模型一个 `model.py` 的形状重写。

| llmtuner 重要符号 | TorchTitan 对应实现 | 主要差异 | 结论/维护动作 |
|---|---|---|---|
| `build_model_config`, `build_model_config_for` | `experiments/transformers_modeling_backend/model.py` config 构造 | llmtuner 同时支持离线 architecture、Hub id、本地 checkpoint | 通过（适配） |
| `unwrap_text_config` | 上游 VLM text config 选择 | llmtuner 把组合模型收敛成统一文本 decoder 契约 | 通过 |
| `resolve_model_class` | 上游模型 registry | llmtuner 使用 HF auto mapping，不维护模型注册表 | 通过（适配） |
| `HFTransformerModel.__init__` | transformers backend wrapper + 各原生 Decoder | 暴露 `tok_embeddings/layers/norm/lm_head/rotary_emb` 五部件；不复制参数注册 | 通过（适配） |
| GQA 构造校验 | `models/common/attention.py::GQAttention.Config.__post_init__` | llmtuner 在 wrapper 边界校验 head 正数和 `Q heads % KV heads == 0` | 通过；Transformers 5.14 本身会漏掉后一项 |
| `uses_dsa` + `attention/masks.build_dense_attention_mask` | 上游 `_uses_dsa` + `HFTransformerModel._build_dense_attention_mask` | **2026-09-28（十九次增量）稠密路径已移植**：`get_attention_masks` 对 `index_topk`（DSA 特征）模型返回 `[1,1,T,T]` 的 0/-inf additive mask（`block_causal` 与 flex modifier 同语义，有等价比对用例），flex 仍跑并把稠密 mask 当 `score_mask`（HF 的 flex 集成按 mask 类型分支）；CP × DSA 显式 `NotImplementedError`（CP 的 mask 通道只切 BlockMask，稠密张量要手工 Q 切分，未验证不给近似） | 通过（适配） |
| `experts_implementation` 旋钮 | 上游 `TitanMoeModelConfig.experts_implementation` + wrapper 应用 | 2026-09-24 起 `ModelConfig.experts_implementation`（默认 `native`）经 config 门面传到 HF config，wrapper 校验"可设置或 raise"（上游同语义），非法值先 raise；EP>1 无意义（swap 整块替换） | 通过 |
| `named_children` | 上游 `Decoder` 的自然子树 | HF CausalLM 多套一层 `model`，llmtuner 只改遍历视图，不改 state_dict FQN | 通过；FSDP/TP/PP 合约测试覆盖 |
| `tp_plan` | HF `_tp_plan` + 上游 sharding config | llmtuner 重写路径前缀供手写 plan 引擎消费 | 通过（适配） |
| `preprocess_inputs` | 上游 post-dataloading process | 合并 batch、构造 mask、先 CP 后 TP 切序列 | 通过；CP×TP 等价测试覆盖 |
| `get_attention_masks`, `_apply_attention` | `models/common/attention.py` 及 transformers backend | llmtuner 对 packed corpus 构造 BlockMask，对 CPU SDPA 明确拒绝错误语义 | 通过（适配） |
| `forward` | transformers backend wrapper forward | 首 stage 接 token，后续 PP stage 接 hidden states；统一 logits 输出 | 通过；PP stage chaining 测试覆盖 |
| `num_flops_per_token` / `flops_per_token` / `quadratic_attention_flops_per_token` | 各模型 FLOPs 估算（`models/utils.py` 的 attention helper + HF backend `get_nparams_and_flops`）+ observability | llmtuner 从 HF config 推导（不做参数遍历——EP/TP/FSDP 之后每 rank 只有分片），算术在 `models/hf/flops.py`、入口 `num_flops_per_token` 在 `models/hf/factory.py`；**2026-09-28 重写为结构感知**：MoE 层 = router + top_k 路由专家（active ratio）+ 全部 shared 专家，MLA 用 `q_lora`/`kv_lora` 与 `qk_head_dim`/`v_head_dim`（不是 MLA config 里那个等于 rope 切片的 `head_dim`），`layer_types` 支持 full/sliding/chunked，稠密与 MoE 混栈按 `first_k_dense_replace`/`mlp_only_layers`/`decoder_sparse_step`/`moe_layer_freq` 分层 | 通过（适配）。几何不可解析时返回 **0**（缺尺寸、MoE 宽度/层划分不明、`layer_types` 短于层数、`linear_attention` 等参数面不可推导的层型），延续"0 优于猜"契约；未移植：上游 `delta_rule_flops_per_token`（Qwen3-Next 线性注意力）。验证：DeepSeek-V3 真实 config 反推 active params = 3.64e10（发布值 37B，差值即未计入的 norms/biases）；稠密路径与旧公式逐位相同 |

### 4. models/common

#### 4.1 数学算子与初始化

| llmtuner 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `activation.ActivationFn`, `SwiGLU` | `models/common/activation.py` 的对应激活 | 同名但历史来源不完全相同；公式有单测，**通过**，不要按 AST 强行替换 |
| `feed_forward.compute_ffn_hidden_dim` | 同名函数 | 去 Config，舍入公式一致，**通过** |
| `FeedForward.forward`, `SigmoidGatedFeedForward.forward` | 同名类 | llmtuner 接受现成 `nn.Module` 投影；上游由嵌套 Config 构建，**通过（适配）** |
| `param_init.skip_param_init`, `depth_scaled_std` | ~~`models/common/param_init.py`~~ | **2026-09-25 已删除**：parity vendored 死代码，llmtuner 用 HF 自带 `_init_weights`，无消费者 |
| `Embedding.forward` | 上游同名文件仅供概念比较 | llmtuner 是 C 类独立实现并支持 vocab shard bounds；不是同名移植。2026-09-23 移植上游 #4637 同源修复：vocab-parallel 分支把全局 `padding_idx` 映射为本地坐标，只有持有该行的 shard 传入，修复越界崩溃与他 shard 行梯度被静默抑制，**通过** |
| `scatter_add.deterministic_scatter_add` 及 autograd hooks | `ops/scatter_add.py` | 路径不同，算法来源明确；前后向测试覆盖，**通过** |
| `moe.experts.GroupedExperts.forward` | `models/common/grouped_experts.py` 与 `models/gpt_oss/moe.py` | llmtuner 统一 grouped-mm/fallback，并承载 HF 权重形状，**通过（适配）** |
| `cast_linear.CastLinear`, `to_cast_linear` | `models/common/linear.py::CastLinear`（150c4f73a 配套） | 前向 input/weight/bias 转 `compute_dtype` 后 `F.linear`，参数保原 dtype（autograd 回 cast）；`nn.Linear` 子类 + 同 `Parameter` 重绑定，state-dict FQN 与 tying 不变；经 `ModelConfig.compute_dtype` 启用，默认关闭逐位回归，**通过（适配）** |

#### 4.2 Attention、RoPE 与 mask

| llmtuner 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `attention.qkv.local_head_split` | `models/common/attention.py::local_head_split` | 去 SPMD 注解，reshape 语义一致，**通过** |
| `attention.qkv.QKVLinear` | `FusedQKVLinear`/QKV 部分 | llmtuner 注入 plain linear 并用 state_dict hook 拆合 HF Q/K/V，**通过（适配）** |
| `QKVLinear._split_qkv_on_save/_merge_qkv_on_load` | 上游 fused QKV state hooks | llmtuner 额外兼容 DTensor gather 与原始 FQN，round-trip 测试覆盖，**通过**。上游 1e4b1f686 把 QKV 转换移入 HF adapters；llmtuner 不跟随——checkpoint 以 HF `wq/wk/wv` 名义存取是本地契约 |
| `RoPEConfig`, `RoPE`, `ComplexRoPE`, `CosSinRoPE` | `models/common/rope.py` | 去 Module/Config 协议，缓存为普通 buffer。2026-09-29 二十二次增量：`RoPE` 基类的三个钩子（`_precompute_cache`/`_reshape_cache`/`apply_rotary_emb`）体是 `raise NotImplementedError`（基类构造即崩）却未声明抽象，改为 `ABC` + `@abstractmethod`（与仓内其余基类同写法），实例化改抛 `TypeError`，两个子类行为与缓存形状不变，**通过** |
| `yarn_inv_freq` | 上游 `_yarn_inv_freq` | 已包含 YaRN `low==0/low==high` 和显式 factor 启用修复，**通过** |
| `maybe_check_max_pos` | 上游 `_maybe_check_max_pos` | async assert，compile 时跳过，**通过**。上游 7e7f271e0 已删除 DTensor positions 包装；llmtuner 本无此路径 |
| mask modifier 系列 | `models/common/attention.py` 对应 mask helpers | llmtuner 拆成 `attention/masks.py`；公式一致，**通过** |
| `create_varlen_metadata_for_document` | 上游同名 helper | llmtuner 支持固定容量和动态路径，**通过** |
| `create_attention_mask` | `flex_attention.create_block_mask` 调用点 | llmtuner 缓存 compile，并兼容 Torch 2.10 缺少 `separate_full_blocks`，**通过（适配）** |

#### 4.3 MoE 与路由

| llmtuner 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `PartialBiasRowwiseLinear` | 上游 9e159aed7 已删除：bias 的 I→P 转换并入新 `RowParallelLinear`；llmtuner 同名类语义本就一致，保留（仅测试使用），**通过** |
| `RouterGateLinear`, `RouterGateLinearFunction` | `models/common/linear.py` 的 `RouterGateLinearFunction`（原 `_RouterGateLinearFunction`） | 前向 FP32 输出、后向 FP32 GEMM；CUDA bf16 使用 `out_dtype`，其他设备安全提升，**通过** |
| `TokenChoiceTopKRouter.forward` | `models/common/routers.py` router + 上游 `models/deepseek_v3/moe.py` 的 group-limited `_select_experts` | llmtuner 参数化而非 Config 构建，保留 softmax/sigmoid、group limit、route norm，**通过（适配）**。上游 e07084202 抽出可覆写 hooks，llmtuner 以 `_select_experts` 为覆写 seam，数学一致。2026-09-24 起 `_debug_force_load_balance` 调试开关已移植（构造参数，round-robin `(t*K+k)%E`，gating 值仍取真实 score，bias/group 限制均绕过——与上游逐字一致）。2026-09-28 十六次增量：`_select_experts_within_groups` 与上游 `deepseek_v3/moe.py` 逐行等价（组分为组内 top-2 之和、`topk` 选组、`scatter` 掩码、越组 `-inf`、`flatten` 后 `topk`），差别只在校验前移到 `__init__`（上游在 forward 里 raise）；llmtuner 把该路由并入共享 router 的依据是 HF 把 `n_group`/`topk_group` 放在同一条 config 路径上 |
| `RoutedExperts.forward`, `MoE.forward` | 上游同名逻辑 | llmtuner 专家权重是 EP swap 后的本地切片，不是上游 SPMD DTensor，**通过（适配）**。2026-09-24 起 `MoE.set_padding_mask` 一次性暂存通道（上游 d34a13fdf 同源）：mask（True=padding）只过滤负载均衡统计（`tokens_per_expert_E`、aux loss f/p、quantile 直方图），routing 决策/dispatch/expert compute 始终跑完整 token 流，无 mask 逐位不变；CP/TP 由 `shard_padding_mask_for_cp/tp` 与 token 流同序切分。**2026-09-28 复核（十六次增量）**：`RoutedExperts` 只持 `GroupedExperts` + dispatcher（上游持 `w13`/`w2` 两个 `GroupedLinear` + 激活 + `output_postprocess`，后者上游也只有用例引用）；`tokens_per_expert_E` 从 router 移到 MoE（上游 hook 读 `moe.router.tokens_per_expert_E`，llmtuner 读 `moe.tokens_per_expert_E`，同 `persistent=False`）；MoE-under-TP 的三个 `_maybe_*_across_tp` 方法由块边界 AG/RS 对偶（`TPMoeSequenceBoundary`）与 tp×ep 直接消费 T/tp 分片替代，故本类不设；`expert_bias_E` 由 MoE 按 quantile 路由注册（上游由 `KimiLatentMoE` 子类删后重注册）；AC 下的重复计数不去重——上游对 NO_REENTRANT 做 `// 2` 是因为它还供 expert-usage 指标，而 llmtuner 的 `update_expert_bias` 用 `sign(mean − x)`，任何正的均匀缩放不改变更新方向且不记录该指标 |
| `QuantileBalancedTopKRouter`, `QuantileBalancer`, `register_moe_quantile_balancing_hook` | 上游 f8bb599a7 同名实现 | 训练时 biased top-(K+1)：前 K dispatch、第 K+1 个 biased 分为 cutoff；1000-bin int32 直方图（non-persistent）按 token 分片轴 all-reduce 后取 `top_k/num_experts` 分位数（bin 内插值），mean-centred 覆写 `expert_bias_E`；与 sign-based bias 互斥（同层构造 raise、跨层 hook raise、全 quantile 时 LB hook 自动不注册）；`ParallelConfig.moe_quantile_balancing` 启用，**通过（适配）** |
| `MoE.update_expert_bias` | 上游 expert bias 更新 | 在 optimizer step hook 执行；跨 PP part 汇总，**通过**。2026-09-24 起注册严格性与上游对齐：所有 MoE层 `load_balance_coeff` 混合配置（部分为 None）即 `ValueError`（上游 `_should_register_moe_balancing_hook` 同源），coeff 全 None 时不注册 hook（免每步无谓 collective） |
| `MicrobatchWiseLoadBalanceLoss` | 上游 load-balance loss | llmtuner 用 autograd carrier 注入并按有效 token 归一，**通过（适配）** |
| `aux_loss.AuxLoss.inject/collect_aux_loss_metrics` | `models/common/aux_loss.py` | 去全局 Module registry，使用显式寄存器与 step denominator；与上游逐符号一致（`reduce_mesh="dp"` ↔ llmtuner `"batch"` 是 mesh 命名适配），**通过** |
| `LocalTokenDispatcher` | `models/common/token_dispatcher.py` | 本地排序、dispatch/combine 与上游同意图，**通过** |
| `AllToAllTokenDispatcher` | 上游 EP dispatcher | llmtuner 直接操作本地专家切片和 PG，非 MinimalAsyncEP（上游已删除该实验），**通过（适配）** |
| `LocalTokenDispatcher`/`AllToAllTokenDispatcher` 的索引数学（2026-09-28 十六次增量逐行核对） | 上游同名方法 | `_local_reorder`（stable argsort，索引再除 `top_k`）、`_permute`（rank-major → expert-major：`input_starts[seg_ids] + arange − output_starts[seg_ids]`）、`_unpermute`（`new_empty` + index put）、`combine`（fp32 乘 score + `deterministic_scatter_add`）与上游逐字一致；`all_to_all_single` + `_materialize` 对应上游编译分支的 `all_to_all_single`/`wait_tensor`（非编译分支的 `spmd.all_to_all` 只是同一 collective 的 SPMD 包装）。**同轮修一处文档错误**：`BaseEPTokenDispatcher` 的 docstring 原写 `num_experts` 是"每 rank 专家数"，实为全局数（`convert.py` 传全局 E，上游 DeepEP/HybridEP 亦用 `self.num_experts // ep_group.size()`） |
| `TorchAOTokenDispatcher` | 上游同名 dispatcher | 可选导入适配层：`_permute`/`_unpermute` 委托 torchao `permute_and_pad`（expert-major 重排 + token 组 pad 到 `pad_multiple`，EP=1 本地 padded permute 路径一并移植）；构造期 lazy import，未装 torchao loud-raise ImportError 带安装指引；`ParallelConfig.ep_token_dispatcher="torchao"` + `ep_torchao_pad_multiple` 接线，**通过（适配）**，数值**环境未覆盖**（无 torchao/CUDA），待 CUDA 目标设备复跑 |
| `DeepEPTokenDispatcher` / `HybridEPTokenDispatcher` | 上游同名 dispatcher | **登记缺口（loud-raise）**：CUDA-only（`deep_ep`/`hybridep` 内核）且 dispatch/combine 需上游 `distributed/deepep/` wrappers（1155 行，未 vendor），可选导入无法忠实表达契约；`ParallelConfig.ep_token_dispatcher` 选到即 NotImplementedError（含解锁条件：vendor wrappers + CUDA optional extra + CUDA 设备复跑），swap 入口防御性同语义；`AllToAllTokenDispatcher` 满足同一 dispatch/combine 契约 |
| `async_linear` 三个模块（2026-09-26 文件名对齐上游，原 dist_gemm.py） | `models/common/async_linear.py` | 保留 fused collective+GEMM，mesh 从 llmtuner context 获取。上游 e72fd863d 重构为组合式 `Async*Linear`、9e159aed7 再改为继承新 `ColumnParallelLinear`/`RowParallelLinear` 通信角色，llmtuner 保持子类式委托 `parallel/tensor_parallel/linear.py`，数学等价，**受限**：需 TP/CUDA 能力 |

#### 4.4 多模态

| llmtuner 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `get_vision_positions` | `models/common/multimodal.py` | 已采用一次性 `.tolist()`，避免逐 item CUDA sync；严格检查 run/token 数，**通过** |
| `scatter_vision_embeds` | 同名函数 | 原地 span fusion，消费数不一致即失败，**通过** |
| `build_vision_bank_indices`, `gather_vision_embeds` | 同名函数 | 去 `spmd.local()` 类型注解，tensor 语义一致，**通过** |

### 5. 并行层

#### 5.1 Mesh 与公共 collectives

| llmtuner 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `ParallelDims.from_config/_validate` | `distributed/parallel_dims.py` | llmtuner 使用稳定 `ValueError`，支持 `dp_shard=-1` 精确推导并校验 EP 整除，**通过** |
| `build_mesh` 与 mesh accessor | 同文件的 mesh 构造/flatten | llmtuner 额外建立 loss mesh，因 TP 端到端切序列，**通过（适配）** |
| `get_all_one_dimensional_meshes` | 同名上游方法 | 已排除 fake-backed axes，**通过** |
| 分布式初始化（`accelerator/dist_utils.py`） | `distributed/utils.py` 的 `init_distributed` + `DistributedTopology`（`comm.backend` 的 `fake` / `real_pp_fake_spmd` 两种逻辑世界） | **未移植（登记）**：上游可在单进程内用 torch 的 `backend="fake"`（+ 真实 PP 组）模拟整个多卡拓扑，llmtuner 只有"`world_size == 1` → `parallel_dims is None`"与真多卡两条路，单机并行验证走 gloo + torchrun 集成测试。解锁条件：torch 提供 `backend="fake"`（本机 2.2.2 无）+ 决定给初始化加一条 debug 后端；见 `torchllmtuner_design.md` §8 的验证边界 |
| `collectives.set_pg_timeouts` | 上游 trainer/comm timeout | llmtuner 独立实现，**通过（适配）** |
| 归约调用（train_step 的 loss/token 归约） | 上游 scattered reductions | 2026-09-24 起收敛为 `accelerator.dist.all_reduce` 在调用点直接使用（clone + in-place collective），原 `dist_sum`/`dist_max`/`dist_sum_tensor` 薄封装已删除；`reduce_equivalence.py` 验证 all_reduce 语义与 clone 调用惯例（trainer 的内联 clone 由 review 保证），**通过** |
| `clip_grad_norm_` | 上游 distributed grad clipping | llmtuner 额外按本地 expert/dense 参数分组并跨 EP 归约，支持 DP/TP/PP/EP，**通过（适配）**。2026-09-28 十次增量复核：dense-only 路径与上游逐行同构（含 DTensor 先 `full_tensor()` 再跨 PP 归约的 `p` 次幂技巧），EP 分支免去上游「每个参数都必须是带 `"ep"` 轴的 DTensor」断言；上游的 `dist_sum`/`dist_max`/`dist_mean` 薄封装**不重建**——llmtuner 对应物是 `accelerator/dist.all_reduce` 在调用点（trainer/validator）使用，`components/metrics.py` 不做任何 `torch.distributed` 调用 |
| 种子与确定性（`Trainer.seed_everything`（原 `_seed_everything`）、`trainer/seed.py`） | `distributed/utils.py::set_determinism` | 2026-09-24 起先移植四项确定性开关（`use_deterministic_algorithms`、`cudnn.deterministic/benchmark`、`fill_uninitialized_memory=False`、`CUBLAS_WORKSPACE_CONFIG`）；2026-09-28 十次增量补齐剩余可移植件：`PYTHONHASHSEED = str(seed % 2**32)`（为之后 spawn 的 dataloader worker）与 `TrainingConfig.detect_anomaly`（`set_detect_anomaly(True, check_nan=False)` + 上游同文告警，`check_nan=False` 因 NaN/Inf 检查走 `aten._is_any_true` 无 DTensor 策略）；PP 的 distinct-seed 派生（`trainer/seed.py`）对应上游同函数公式。不移植两件：DTensor mesh-aware RNG tracker（上游用于分片参数初始化，llmtuner 走 HF 自身初始化）与 `warn_only` 开关（llmtuner 固定 `False`，更严）。**通过（适配）** |

#### 5.2 TP

| llmtuner 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `AllGatherLinear`, `LinearReduceScatter` | `models/common/async_linear.py` 的 `AsyncAllGatherLinear`/`AsyncLinearReduceScatter`（原 `distributed/linear.py` → dist_gemm.py → async_linear.py，数学不变） | fused symmetric-memory autograd 实现；提供 functional collective fallback，**通过** |
| `all_gather_linear`, `linear_reduce_scatter` | 同上非融合语义 | CPU/gloo fallback，前后向是 collective 对偶，**通过** |
| `ColumnParallelLinear`, `RowParallelLinear`（2026-09-26 改名对齐上游，原 `ColwiseLinear`/`RowwiseLinear`） | 上游 `models/common/linear.py` 同名类（9e159aed7 起拥有各自 collective） | llmtuner 替换 HF `nn.Linear`，不使用 ParallelStyle；plan 规格字符串 `colwise`/`rowwise` 与 factory 不变，**通过（适配）** |
| `ColwiseLinearNoGather` | 无对应物（上游为父模块一次性 gather + plain Linear 子投影） | llmtuner 特有 realizer：输出保留 sequence shard；保持原名（llmtuner 特有，非改名对象），**通过** |
| `resolve_plan`, `match` | HF `_tp_plan` + 上游 sharding registry | 支持 colwise/rowwise/replicated；`colwise_gather_output` 当前保守保持 lm_head 复制，**通过（适配）** |
| `apply_tp` | transformers backend parallelize + 各模型 parallelize | 手写 pattern plan；plan 里的 MoE 规格（`packed_colwise`/`packed_rowwise`/`moe_tp_experts`）解析为 None 并走结构路径（`shard_experts_for_tp` 沿 F 维原地切分专家权重、`TPMoeSequenceBoundary` 加块边界 AG/RS 对偶），不再 raise；tp×ep 放行（TP 只切 dense、EP 独占 routed 专家，ep>1 时 `apply_tp` 把 MoE 块留给 swap），shared-expert×tp（gate/up/down 布局）自 2026-10-02 起放行（feature 切分无 collective、输出 partial 随边界 RS 归约；等价脚本 tests/integration_tests/shared_expert_tp_equivalence.py 待复跑），未知布局仍 loud-raise；tp×ep×cp 自 2026-10-02 起按上游语义放行（**受限：真多卡前后向等价性环境未覆盖**，等价脚本 tests/integration_tests/tp_ep_cp_equivalence.py 待 torch≥2.12 复跑）（见 §9.1） |

#### 5.3 FSDP2

| llmtuner 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `resolve_fsdp_mesh`, `resolve_sparse_fsdp_mesh` | `distributed/fsdp.py` mesh dims | llmtuner 把多轴 mesh 重建为 FSDP 可理解的 1D/2D mesh，**通过（适配）** |
| `apply_fsdp_to_decoder` | 同名上游函数 | 支持 HF ModuleList、MoE expert placement、prefetch。2026-09-23 移植上游 4b5023b80 同源修复：专家分片度经 `fsdp_shard_size` 只计 shard 轴，HSDP 下不再误选 `Shard(1)`，**通过（适配）**。2026-09-28 逐项复核（八次增量）：装配顺序/reshard 策略/权重绑定/prefetch 与上游逐段一致；参数名**不跟**上游的 `ep_degree`，保持 `ep_size`（llmtuner 的配置面统一拼 ``*_size``，见 `config/parallel.py`；`ep_size` 也与 `expert_parallel_size` 同词根）；专家分片度的辅助函数改名为 `fsdp_shard_size`（原 `_fsdp_shard_degree`，同一次统一）；专家计数读 `moe.router.num_experts`（本地切片的 `inner_experts.num_experts` 会把它变成 `efsdp*ep**2`）；不带上游的 `dp_mesh_dims`/`edp_mesh_dims`——llmtuner 传专用 1-D/2-D 子网，torch 按形状读出的轴与上游显式声明一致，且普通 tensor 参数上这两个入口本就不可用，**通过（适配）** |
| （无）`linear_param_shard_placements`、`apply_fsdp_to_multimodal_encoder` | 上游 `distributed/fsdp.py` 的另两个导出 | 前者按 `Shard(ndim-2)` 切 stacked/grouped 线性权重：llmtuner 没有 `num_linears` 式融合 Linear，2-D `nn.Linear` 默认 `Shard(0)`、packed 3-D 专家权重 `Shard(1)` 已切在同一个输出维；后者服务 vision tower，llmtuner 的 `models/common/multimodal.py` 只有融合算子、无编码器模块（无消费者）。**有意未移植**，布局等价性由 `test_efsdp_placement.py::test_packed_expert_weights_shard_their_output_dim_at_index_one` 钉住 |
| `enable_fsdp_symm_mem` | 同名上游函数 | 2026-09-23 起支持 `scope="all"/"dense"/None`（上游 65e495dda），非法 scope 抛 ValueError；经 `fsdp_symm_mem_scope` config 字段（默认 "all"）对用户开放，**通过（适配）**。2026-09-27 复核：上游把"关"表达为 `fsdp_symm_mem_scope=None`（单字段，且 `tyro.conf.Suppress` 不进 CLI），llmtuner 拆成 `enable_fsdp_symm_mem=False` + `fsdp_symm_mem_scope="all"` 两字段并允许 CLI 传入——**默认语义等价**（两个默认都是关；scope 默认值在 enable=False 时不可达），差异只在形态与可见性，故不改 |
| `disable_fsdp_gradient_division` | 同名上游 helper | global valid-token loss 自行缩放，故禁用 FSDP 平均，**通过** |
| `apply_fsdp` | 各模型 `parallelize.py` 的 driver | 固定 dtype 策略，非 NCCL 强制 SUM；兼容 Torch 2.10 类型缺失，**通过（适配）**。配置面收窄登记：`cpu_offload` 未接线（`fully_shard/apply.py` 恒 `False`），param/reduce dtype 固定（模型 dtype / fp32），所有 DP 轴为 1 时不装 MixedPrecisionPolicy（数值等价） |

#### 5.4 CP、EP、PP 与 AC

| llmtuner 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `apply_cp` | `distributed/context_parallel.py`（上游此前的 `context_parallel/` 包已重整为单文件）+ 模型 parallelize | 给 HF attention 注入 kernel；校验 backend、mesh 和 Ulysses heads；2026-09-25 起 ulysses×packed 不再 fail-fast（经 `set_cp_mesh(strategy=...)` 闩锁策略），**通过（适配）**。2026-09-28 复核：上游该文件现在是「输入分片 + 负载均衡器」的声明面（`shard_tensors` / `get_cp_input_seq_len` / `ContextParallelLoadBalancer`），kernel 在 `models/common/cp_attention.py`；llmtuner 的对应物是 `context_parallel/input_shard.py` + `cp_kernel.py`，逐项结论见审计「九次增量」 |
| `shard_batch_for_cp/tp`、`shard_padding_mask_for_cp/tp`、`shard_attention_mask_for_cp` | `distributed/context_parallel.py::shard_tensors`（上游走 `SpmdType` 声明 + `spmd.shard(R→S(seq_dim))`） | llmtuner 无声明面，改为逐类张量的显式入口，permute+切分委托 torch 私有 `_context_parallel_shard`；默认连续切分与 `headtail` 均衡切分的语义与上游一致（`seq_len % (2*cp)` 两边同样校验），mask 只切 Q 轴、KV 保持全长。三处差异：负载均衡器由上游的抽象类 + `generate_permutation()` 换成字符串现场构造 torch 的 `_HeadTailLoadBalancer`；`ptrr` 仍是登记缺口（loud-raise：它要在切 batch 时读 BlockMask，而 mask 在 wrapper 的 forward 内才构建）；上游 `shard_tensors` 每次调用的 `shape[seq_dim] % cp` 与「所有 CP 张量同 seq len/同 device」两条检查，llmtuner 改在配置解析期查 `max_seq_len % cp`（更早更强，但不覆盖绕过配置传入别的 T 的用法）。**通过（适配，两处已知差异见审计）** |
| `CPFlexKernel`（KV all-gather / Ulysses 两条路径） | `models/common/cp_attention.py` 同名意图 | 剥掉上游 CPInnerAttention/FlexInnerAttention 类层，redistribution 与 kernel 合在 `context_parallel/cp_kernel.py`。2026-09-28 十次增量复核：KV all-gather 用的是**同一个** torch 算子 `flex_cp_allgather`（上游 HF 后端的 `_wrap_flex_kernel_cp` 也用它；backward 的 reduce-scatter 由算子自带，故 llmtuner 没有 `reduce_dtype` 旋钮——上游只有**原生**模型的 `KVAllGatherCPFlexInnerAttention.Config` 有该字段，HF 后端同样没有）；Ulysses 为 seq↔head all-to-all，与上游 `UlyssesCPInnerAttention` 同一置换。2026-09-23 起独立的 `primitives.py` 已删除（零调用者的重复实现）；2026-09-24 起 ulysses 的 `_full_length_causal_mask` 复用 `masks.create_attention_mask`（与 wrapper 同 builder、同参数），本地 inspect 兼容副本已删；2026-09-25 起 ulysses 支持 packed/varlen——wrapper 全长透传文档 mask，kernel 按 mask Q 长度分派（上游 `UlyssesCPVarlenInnerAttention` 语义，varlen 元数据不随 token 分片） | **通过（适配）** |
| `swap_hf_moe_blocks` | transformers backend `moe_replacement.py` | 上游重新初始化，llmtuner 搬运 HF 权重；不是共享实现，等价性测试覆盖，**通过（适配）** |
| `apply_ep` | 上游模型 EP parallelize | 先 swap 再建立 dispatcher/组，**通过（适配）** |
| `generate_llm_fqn_per_model_part` | transformers backend `pipeline.py` | 加权切层公式一致，**通过** |
| `split_model_into_stages` | 同文件 stage split | 删除模块用 `Identity`，每 stage 保留 rotary，兼容 Torch 2.10 `PipelineStage`，**通过（适配）** |
| `apply_pp`, `build_pipeline_schedule` | `distributed/pipeline_parallel.py` | llmtuner 直接消费 HF 五部件契约；2026-10-02 起 pp×ep / pp×cp 放行（pp_cp_ep 裁决删除），**通过（适配）** |
| `apply_pp(first_stage_module_fqns=...)`, `prepend_first_stage_modules` | 同文件 `pipeline_with_first_stage_modules` | 额外顶层模块并入 stage 0：仅作用自动切分，存在的 FQN 按序前插，已占有/重复 FQN raise、缺失跳过，显式 `module_fqns_per_model_part` 给定时忽略并告警（同上游委托语义）；`split_model_into_stages` 配套把 wrapper `named_children()` 不呈现的额外顶层模块在非属主 stage 置 `Identity`（上游 "pruned on other stages" 语义），装五部件的容器经"包含已呈现部件"判定跳过。stage FQN 稳定、默认 None 逐位不变，**通过（适配）** |
| `apply_ac`, selective helpers, `apply_memory_budget`, `disable_dynamo_lru_cache` | `distributed/activation_checkpoint.py` | FullAC/SelectiveAC 已移植，**通过**；FullAC 的对齐同时含策略这一层（2026-09-29 二十六次增量）：上游 `FullAC._wrap_block` 不是裸 wrapper，而是把恒 `PREFER_RECOMPUTE` 的 `_full_ac_policy` 经 `create_selective_checkpoint_contexts` 传入，让 torch 对「输出不可重算/带注册副作用」的算子仍落 SAVE；llmtuner 已按同形补上 `full_policy` + `wrap_full`（`determinism_check`/`debug` 仍走 torch 默认值——实测 torch 默认即 `default`/`False`，与上游 config 默认相同，故行为一致，只是不可配）；两处 `early_stop` 已于 2026-09-27 跟随上游 #4836 同步为 `True`（此前为上游 #1580 的 `False` workaround）。MemoryBudgetAC 已移植为 `mode='memory_budget'` + `MemoryBudgetACConfig`（设 `torch._functorch.config.activation_memory_budget`，需 compile，torch 无 knob 时 loud-raise），见 §9.1；RegionAC 已接入（2026-09-29 二十五次增量：`region_ac` + `parallel/remat_regions.py`，以 HF block 的 `nn.Linear` FQN 作 region 名，替代上游 `Module.configure_remat_regions` 声明通道；`preserve_rng_state=True` 配置期即拒，torch_remat 需 torch ≥ 2.10，apply 期 loud-raise）。`_disable_dynamo_lru_cache` 亦已移植（同上，上游在每个 policy 的 `apply` 开头调用），并经 `has("dynamo_lru_cache")` 能力门：torch 2.2.2 有 `torch._C._dynamo.eval_frame` 而无 `_set_lru_cache`，此时记 info 后继续。AC 也跑在 PP 路径上（2026-09-27：`stages.py` 的 `ac` 行 `on_pp=True`，逐 chunk 折层，与上游把 `ac_config` 交给每个 model part 的 `parallelize` 同构）。FullAC 的 `determinism_check`/`debug` 旋钮未暴露（固定默认值），登记于此 |
| `apply_compile`, `maybe_enable_async_tp`, `maybe_regional_inductor_backend`, `maybe_regional_inductor` | `distributed/compile.py` 同名函数 | 四件全移植为 `parallel/compile.py` + `CompileConfig`（`training.compile_config`，默认全关 = 旧整体 compile 逐位不变）：逐 block compile 用 `Module.compile` 就地（`per_block=True`）；async TP 设 `_micro_pipeline_tp` + symm-mem 注册（按 group 名去重），配置期拒无 compile/tp=1，装配期对无 mesh/旧 torch loud-raise；regional_inductor 仅 `aot_eager`×flex 触发（wrapper `uses_flex_attention` 判定，annotation 在 `flex_attention_hf`，inductor_configs 传空），flex×其他 backend `ValueError`、torch 无该模块 `NotImplementedError`；`capture_scalar_outputs` 按上游条件（`iter_moe_layers` 非空）设置，dense 不动。上游的 `skip_fwd_side_effects_in_bwd_under_checkpoint` 与 FakeTensorMode monkeypatch 未移植（登记于 upstream map），**通过（适配）** |

`VALID_AC_MODES` 自 2026-09-29（二十三次增量）起声明在 `config/training.py`，紧挨它约束的
`TrainingConfig.activation_checkpoint_mode`；`parallel/activation_checkpoint.py` 改为从这里
import 并保留 `__all__` 再导出（此前两侧各写一份同一个四元组，配置校验与 `apply_ac` 的成员
检查可能各自漂移）。同轮的 `config/training.py`：`global_batch_size` / `max_seq_len` /
`steps` / `gradient_accumulation_steps` 四处逐字相同的 `>= 1` 守卫并为同一循环，与
`config/parallel.py`、`parallel/parallel_dims.py` 的既有写法一致（异常文案逐字不变）。

### 6. 数据系统

| llmtuner 重要符号组 | TorchTitan 对应实现 | 差异与正确性 |
|---|---|---|
| `DatasetBuildContext`, `DatasetIterationPolicy` | `components/data/types.py` | 去 Configurable，参数校验已补齐（`__post_init__` 三条/三条，上游无）。上游 ec953b360 把 `num_tokens_per_batch` 改名 `num_tokens_per_microbatch`；llmtuner 保持旧名且内部自洽，属故意分叉。2026-09-28 十四次增量：AST 归一化仅 3 hunk，除上述与 `Batch` dataclass（合成路径容器，放这里以免 models 反向 import 数据源）外无差异，**通过** |
| `TextSequence`, `SampleProcessor`, `SingleDataset` | `components/data/dataset.py` | 类去 `Config` 后缀；构建走自由函数。2026-09-28 十四次增量：AST 归一化 6 hunk 全是 Configurable→dataclass+自由函数；DP 分片数学提成 `shard_for_dp` 后与 `DatasetConcat` 内联版公式逐字一致（`divmod` + `min(rank, remainder)` 错位切分）、mix 的 `seed + index` 派生、子策略 `shuffle=False, repeat=False, dp=0/1` 均一致，**通过（适配）** |
| `WeightedDataset`, `DatasetMix`, `DatasetConcat` | 同文件 config nodes | 数据组合语义保留，**通过** |
| `build_dataset` 与 `_build_*` | 上游各 config `.build()` | llmtuner 工厂替代对象构建协议，**通过（适配）** |
| source 类与 `build_source` | `components/data/sources.py` | 同上；HF streaming cursor 显式 Stateful。2026-09-28 十四次增量：AST 归一化 6 hunk 全是 Configurable→dataclass+`_*DataSource`+`build_source` 分派；索引 JSONL 解析、`split_dataset_by_node`、streaming 不支持精确 resume 的拒绝、`load_dataset` 一等字段与 kwargs 冲突校验（提成 `_reject_duplicated_hf_fields`）全部逐字一致，**通过** |
| `GrainDataLoader` | `components/data/loader.py` | 直接收参数，无 loader Config，state round-trip 保留。2026-09-28 十四次增量：`dataset.batch(collator.num_rows_per_batch(), drop_remainder=repeat, batch_fn=collator)` 与 `ThreadPrefetchIterDataset(prefetch_buffer_size=num_prefetch_batches)` 逐字相同；差异只是数据集图由调用方建好再传入（`DataloaderConfig` 单入口设计的 C 类工厂），**通过** |
| `TextCollator` | `components/data/collators.py` | packed labels/positions 与 valid-token 计数契约。上游 d398a8fb9/ec953b360 已把 `batch` 改名 `microbatch` 并引入 `TrainingMicrobatch` 类型；llmtuner 保持 dict 版 `TrainerBatch`（labels 与 num_valid_tokens 已内含），语义等价，属故意分叉。2026-09-28 十四次增量：载荷逐字一致（zeros+cat、超长 raise、`positions[num_tokens:].remainder_()`、`num_valid_tokens=(labels != IGNORE_INDEX).sum()`）；**一处设备面差异登记**：`HAS_PIN_MEMORY` 上游用 `torch.accelerator.is_available()`（有加速器），llmtuner 用 `should_use_pin_memory()`（本进程解析出的设备是加速器），"有 GPU 但 `--use_cpu`"时前者 True、后者 False，后者更贴合 pin 内存的实际用途，**通过** |
| packing build 函数和 iterators | `components/data/packing.py` | registry 选择移到 `DataloaderConfig.packing`，算法保留；document-aware iterator、padding_mask 与可恢复 remainder 均已含上游 f23d7dfe2 载荷。2026-09-28 十四次增量：AST 归一化仅 3 hunk（两处 `Config.build()` → `build_*_packing()` 自由函数），grain 图逐字相同（`length_struct`/`padding_struct` 的 `IGNORE_INDEX`/`True` 填充、`meta_features`、`seed`/`shuffle_bins`/`num_packing_bins`/`max_sequences_per_bin`，以及 `_DocumentAwareConcatThenSplitIterator` 的 `get_state`/`set_state`），**通过** |
| `TextProcessor`, `ChatProcessor` | `hf_datasets/text_datasets.py` | llmtuner 路径重组，处理语义一致；SFT prompt/response token 边界校验与上游 8108e201a 逐字一致。2026-09-28 十四次增量逐行核对三条 SFT 信号规则：超长丢弃判据（`len(tokens) - 1 > max_context_length`）、prompt 段 labels 置 `IGNORE_INDEX`（`labels[:max(prompt_len - 1, 0)]`）、renderer 路径 `~loss_mask[1:]` 掩码，全部与上游一致；`DATASETS` 注册表三项同源，llmtuner 另加 `make_local_jsonl{,_sft,_multiturn}` 工厂（C 类）。**交叉验证**：llmtuner 的 tokenizer 抄的是上游 `experiments/transformers_modeling_backend/tokenizer.py`（`add_generation_prompt` 默认 True），而上游 SFT 数据路径用基础 tokenizer（HF 默认 False），所以 `ChatProcessor` 显式传 `add_generation_prompt=False` 才是与上游渲染一致的那一半，代码正是如此，**通过** |
| `MultiModalCollator` | `hf_datasets/multimodal/mm_collator.py` | 已增加 MRoPE grid/run/长度校验，**通过** |
| image/text/video helpers | `hf_datasets/multimodal/utils/*` | A1 移植，路径扁平化；单测覆盖，**通过** |
| `MultiModalProcessor` 与 packing helpers | `hf_datasets/multimodal/mm_datasets.py` | 去 Configurable，packing 为自由函数。2026-09-28 十五次增量：归一化 AST diff 逐条核对，唯一语义差异是**有意分叉**——上游 `_process_cc12_wd_sample` 对缺 `jpg` 的行兜底成纯文本行（`texts=[text]`），llmtuner 不兜底，让 `insert_vision_placeholders` 的 `None` 槽在 `"".join` 处响亮失败；理由是 cc12m-wds 缺图属样本畸形而非解码失败，仓内两例
`test_insert_vision_placeholders_rejects_a_slot_with_no_token_count` /
`test_process_cc12_wd_sample_raises_when_the_image_field_is_absent` 把该契约钉住。九参数显式转发表与上游逐字相同（上游调用方同形），不为了 DRY 偏离。其余为 `_mm_sample` → `process_mm_sample` 等去私有化与 Config→显式参数，**通过（适配）** |
| `RandomTokenSource/RandomTokenDataLoader` | 无对应 | C 类合成数据；DP rank/world 校验已覆盖，**通过** |
| `datasets.build.build_dataloader` | 上游 config `.build()` 调度仅供概念比较 | C 类工厂，是 llmtuner 单入口设计，**通过（适配）** |

### 7. Components

#### 7.1 Loss、optimizer、scheduler

| llmtuner 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `cross_entropy_loss`, `LossParallelCrossEntropy` | `components/loss.py` | 以 logits shape 选择 vocab-parallel（上游按 spmd tp size 选）；非法 label async 拒绝。2026-09-28 十三次增量逐行复核：forward 的三个 TP all-reduce（max/sumexp/gather）、shard 边界公式、`_shard_local_labels` 的映射与 `backward` 的融合导数（`out_of_range - 1` 那一步）与上游逐行一致。差异登记：上游 `cross_entropy_loss` 有 `reduction: sum|none` 参数，llmtuner 固定 `"sum"`（`"none"` 只在 `LossParallelCrossEntropy.apply`/`compute_logprobs` 里显式用），无消费者故不补；上游的 `spmd_typecheck` 静态断言不移植（llmtuner 无 spmd 曲面）；类名去私有化。**通过（适配）** |
| `vocab_shard_bounds`, `next_token_targets` | 上游公式散在 loss/训练器 | llmtuner 提取成共享 helper，**通过（适配）**。`vocab_shard_bounds` 的 `chunk_size=ceil(V/tp)`、`min(V, ...)` 双侧夹取与上游 forward 内的内联公式逐行同构 |
| `chunked_lm_head_cross_entropy` | 上游 `ChunkedLossWrapper` | 自行 backward 以控制 logits 峰值，**通过**。允许不整除的短尾 chunk（sum 归约下数值等价）。**2026-09-28 十三次增量补登记三条结构差异**：(1) 上游 wrapper 支持**多输出**（tuple pred/labels，服务 dMTP 一类多输出模型），llmtuner 只接单个 `(T, H)`；llmtuner 无此类模型，故无消费者；(2) 上游 `__call__` 返回 `(loss, metrics)` 并有 `_combine_chunk_metrics` 逐 chunk 指标合并，llmtuner 只返回求和 loss；(3) 上游用预分配缓冲的 `GradAccumulator`（就地拷贝），llmtuner 用 list + `torch.cat`（多一次 `T*H` 拷贝）。另有性能差异登记：不合并 lm_head 的 FSDP reshard/grad-sync（上游在 chunk 循环期间禁用），chunked×FSDP 下每 chunk 多一次 all-gather/reduce-scatter，数值等价 |
| `compute_logprobs`, `mse_loss` | 上游对应 loss | 直接自由函数，无 BaseLoss。2026-09-23 起分片路径的 `return_entropy` 真正生效：entropy 经 `vocab_parallel_entropy` 免 gather 计算（上游 a3d59d316 同源），**通过**。严格性差异登记：`tp_group` 已给但 `global_vocab_size=None` 时静默走全词表路径（上游 raise），当前无调用者触发。2026-09-28 十三次增量补登记：两者在 llmtuner **均无生产调用者**（上游的 `compute_logprobs` 只服务 `rl/`，`mse_loss` 只被 flux 的 `MSELoss` 选到，两者都在裁剪面内），仓内唯一引用是 `tests/integration_tests/vocab_parallel_loss_equivalence.py`，故按"移植曲面"保留而非删除 |
| `OptimizersContainer` | `components/optimizer/optimizer.py` | 删除 OptimizerWrapper；多 PP part 容器直接实现 Optimizer/Stateful surface。2026-09-28 十二次增量复核：构造算法（分组/首个 pattern 命中/`_build_impl_kwargs`/`step`/flat FQN state dict）与上游同构，三条登记差异——上游第四种 `implementation="fused_opt_states_bf16"`（bf16 Adam 状态 + load post-hook 复原 dtype，价值全在 CUDA fused 核）与 `optimizer_factory_kwargs_by_name`（无消费者）均已入 D 表；`DistMuon` 工厂属早已登记为范围外的 `distributed/flex_shard/`；`default_adamw` 便捷构造（上游 `lr=8e-4`/betas (0.9,0.95)/wd 0.1）未移植，其调用者只有 torchft recipe 与 RL 示例。MoE 负载均衡/quantile hook 的注册点从容器挪到 `trainer/builder.py`（上游由各模型代码注册），hook 本体在 `models/common/balancing.py`。优于上游：`_validate_params` 点名未被认领的参数并检出重复认领（上游仅一条 assert），`step` 的 closure 断言改 `ValueError`。**同轮修掉一个 CPU 真 bug**：llmtuner/上游都把 `fused` 放进 param group（支持组级覆盖），而 torch 只在**构造参数**上校验设备，于是 CPU 上默认 `implementation="fused"` 构造通过、首步 `optimizer.step()` 崩在 `aten::_fused_adamw_`；上游 GPU-only 不会遇到。现按设备解析：torch 没有该设备的 fused 核时降级为 for-loop 并记 info（CUDA/XPU 与上游逐字相同；CPU 上 for-loop 与 foreach 实测逐位一致），**通过（适配）** |
| `init_optim_state` | `components/optimizer/utils.py` | 已支持部分参数已有 Adam state，并保持首次真实 step=1，**通过** |
| flat state dict helpers | 同文件 | FQN flat format，支持 nested state，**通过** |
| `wsd_factor`, `LRSchedulersContainer`, `build_lr_scheduler` | `components/optimizer/lr_scheduler.py` | 去 Configurable，数学与 state 语义保留，**通过**。默认值分叉登记：上游 `decay_ratio=None`（默认）表示 warmup 后贯穿余程 decay；llmtuner 无 None，默认 `0.0` 表示永不 decay（config docstring 声明为有意设计）。另新增 `total_steps < training_steps` 拒绝（上游会跑出负 lr）。2026-09-28 十二次增量复核：WSD 公式逐行一致（同样的 0-based `+1` 修正、`stable_steps = total + 1 - warmup - decay`、三种 decay 形状与 `min_lr_factor` 缩放），差异仅形状（`_wsd_factor` 提到模块级、`build_lr_scheduler` 自由函数取代嵌套闭包与 `Config.build`）与两处小新增：`load_state_dict({})` 空字典 no-op（上游会 KeyError）、`LRSchedulersContainer(total_steps=...)` 作为曲线长度断言 seam（上游无此属性） |

#### 7.2 Checkpoint

| llmtuner 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `ModelWrapper` | `components/checkpointer/base.py` | 合并 PP parts，缓存稳定 storage 供 async staging，**通过** |
| `CheckpointStorage` | 上游 backend storage seam | llmtuner Protocol，不依赖 Configurable，**通过（适配）** |
| `BaseCheckpointManager` 生命周期方法 | 同名基类 | load/save/close、异步 drain、retention 集中在基类。2026-09-23 起 resume 优先于 initial_load_* 时记 info 日志（上游 810e62786）。2026-09-28 十一次增量复核：策略方法与上游逐行同构，两处适配登记——新增 `_initialized` 门（上游靠各类自己的 `hasattr`/`getattr` 容错部分构造，见 `dcp.CheckpointManager.__init__` 的 HF 选项拒绝）与 `enable` 短路（上游的 manager 只在配置了 checkpointer 时才构造）；`_should_purge` 的 rank 判定换 `dist_utils.is_main_process()`（非分布式下 `dist.get_rank()` 会 raise），**通过（适配）** |
| `_parse_step/_find_load_step/_purge_stale_checkpoints` | 同名策略 | exact `step-N`、清理 staged/abandoned、保留豁免，**通过** |
| `dcp.CheckpointManager` | `components/checkpointer/dcp.py` | 本地/remote DCP、HF export guard。2026-09-23 起异步写总时长经 `save_future` done-callback 记 info 日志（上游 d9ca9e55a，以 info 行替代 structured scalar）。2026-09-28 十一次增量复核：`_save`/`_load_checkpoint`/`dcp_save`/`_save_last_step`/`_flattened_model_states_sd` 与上游逐行同构，4 处 `assert` 改显式 `raise`、HF 选项的拒绝信息更具体（llmtuner 不随包发布 `sd_adapter`）；配置校验 13 条逐条搬到 `config/checkpoint.py`（另收上游 `dcp.Config.async_mode` 与 `training_engine.create_seed_checkpoint`），`initial_load_model_only` 无 `initial_load_path` 的上游告警故意不移植；上游 `SaveDone` 是零引用死类、不移植；`EXPORT_DTYPE_MAP` 2026-09-29（二十一次增量）从 `dcp.py` 移到 `base.py`（两个后端都要它，原先 `torch_checkpointing` 反向 import `dcp`），它与 `models/common/cast_linear.TORCH_DTYPE_MAP` 仍是两张同表（一张写 checker 能导出的 dtype、一张写 lm_head 能计算的 dtype，重叠是巧合），登记为已知重复，**通过（适配）** |
| `TorchCheckpointingManager` | 同名 backend | optional dependency 延迟导入，保存统一经过 backend。2026-09-28 十一次增量：删除死成员 `staging_future`（只被 `__init__` 置 None、`_close` 读一次，全仓无赋值点；上游该类无此成员），HF 导出路径（`sharded/` + barrier + consolidate）与上游同构；后端未安装故仍属静态复核，**通过（适配）** |
| `FilesystemCheckpointStorage`、`async_save_config` | 上游 `_FilesystemCheckpointStorage`、`_async_save_config` | 去私有化：测试需要这两个 seam（storage 契约/本地 IO、异步配置分支），符合"非必需不加 `_`"的取向，**通过（适配）** |
| `canonical_fqn` | `components/checkpointer/utils.py` | A1，移除 checkpoint wrapper segment，**通过** |

#### 7.3 Observability、profiler、tokenizer

| llmtuner 符号组 | TorchTitan 对应实现 | 差异与正确性 |
|---|---|---|
| `DeviceMemoryMonitor` | `observability/metrics.py` | 后端中立设备 API，**通过**。2026-09-28 十二次增量：吃 `device_type` 而非 device 串，CPU 分支返回全零 `DeviceMemStats`，`_to_pct` 加除零守卫，`build_device_memory_monitor()` 在 CPU 上不打印容量（上游会在无设备名时仍打一行） |
| logger 类与 `LoggerContainer` | 同文件 | optional TensorBoard/W&B 延迟导入，**通过**；镜像需安装对应包。2026-09-23 起 `WandBLogger.log` 带 `commit=True`（上游 e0e35fe5a），防显式 step 被合并 |
| `MetricsProcessor` | 同名上游类 | 去 Configurable；按真实 step window 算吞吐/MFU，log frequency 构造时校验，**通过（适配）**。2026-09-28 十二次增量复核：训练/校验两条日志的 key 集合与 `tps`/`tflops`/`time_metrics`/`memory` 公式与上游一致；差异为形状——上游自由函数 `compute_training_performance_metrics` 的载荷内联进 `_derive`/`_Derived`，MFU 抑制条件由 `has_quantization` 换成 `gpu_peak_flops == 0 or num_flops_per_token == 0`（无量化路径的等价替代，见模块 docstring），`_build_metric_logger` 去掉 `ft_*` 并只吞 ImportError；`should_log` 不再有写 `step_last_log` 的副作用（移到 `log`/`log_validation`，且容忍先 log 后 should_log）。删除零调用者的 `set_num_flops_per_token`（上游无此方法）。模块全程不碰 `torch.distributed`（复核成立） |
| `get_metrics_rank`, `ensure_pp_loss_visible` | 上游 `_get_metrics_rank` / PP warning | llmtuner 明确 PP schedule 可见性：rank 查询去私有化为 `get_metrics_rank`，`ensure_pp_loss_visible` 增加 `not pp_enabled` 提前返回（上游无该守卫，只靠唯一调用点门控，语义等价），**通过** |
| `Profiler`, `MemoryProfiler` | `observability/profiler.py` | 去 Configurable，schedule 与 OOM 处理保留；`caused_by_oom` 与上游 773e16e75 语义等价（含防环与隐式链）。2026-09-28 十一次增量：修一处设备面缺口——上游 activity 列表是「CUDA 可用加 CUDA，否则 XPU 可用加 XPU」，llmtuner 原先只加 CUDA 且 docstring 误称设备只有 cuda/cpu，而 `accelerator/device.py` 的 `DEVICE_PRIORITY` 含 xpu（可达），XPU 运行的 trace 会退化成 CPU-only；现按 resolved device 补 `xpu` 分支（其余设备仍 CPU-only，与上游一致），用例 `test_the_trace_activity_follows_the_resolved_device` 钉住 cpu/cuda/xpu 三态。裁剪登记：`leaf_folder`（只服务上游 torchft 的 per-replica 子目录）、CUDA-graph annotations（随 D10 无图路径）、`structured_logger` span、`active()` builder；memory history 经 `accelerator/monitoring` 的 device 探针（上游非 CUDA 分支调不存在的 `torch.memory`），**通过（适配）** |
| `BaseTokenizer`, `HuggingFaceTokenizer` | `components/tokenizer.py` | A1；encode 强制 `add_special_tokens=False` 后自行处理 BOS/EOS。2026-09-23 起 `apply_chat_template` 接受 `Sequence[Mapping]`（上游 4a0d8dab3 多轮 SFT 配套），**通过**。2026-09-24 起 `apply_chat_template` 自动注入 `bos_token`/`eos_token` kwargs 与默认 `add_generation_prompt=True`（上游 backend tokenizer 同源）；SFT 全量渲染在 `datasets/text/processors.py` 显式传 `add_generation_prompt=False` |
| `MultiModalTokenizer` | 同文件多模态 tokenizer | 组合 text/vision token 契约，**通过** |

### 8. Utils 与 C 类模块

| llmtuner 符号组 | TorchTitan 对应 | 结论 |
|---|---|---|
| `utils.filesystem.*` | `tools/filesystem.py` | A1，去 docstring 后 AST 等价，**通过** |
| `accelerator.spmd_context.*` | 意图接近 `distributed/spmd_types.py`，实际基于 pip `spmd_types` | C 类活代码；不要替换成上游 module protocol |
| `accelerator.device.*` | 无可靠同源 | C 类，统一 NPU/CUDA/MLU/MUSA/CPU 设备信息与 backend 选择，含 mmengine 风格厂商谓词 |
| `accelerator.monitoring.*` | 部分意图见 `tools/utils.py` | C 类，包含 peak FLOPS（含 MI350X）和 memory snapshot |
| `utils.gc.GarbageCollection` | `tools/utils.py` GC helper | 去 structured logger，**通过（适配）** |
| `utils.logger_utils.*` | 无单一对应 | C 类日志格式与 rank helper；2026-09-24 起全仓模块 logger 统一经 `get_logger`（发射时 rank 过滤），文件输出参数随零调用删除 |
| `utils.checkpoint_keys` | 无文件对应 | C 类，打断 config→checkpointer 导入环 |

### 9. 缺口、悬空链与禁止误判项

#### 9.1 真缺口

- `distributed/compile.py`：**已移植**（2026-09-24 批 8，`parallel/compile.py::apply_compile`
  + `CompileConfig`）。逐 block compile（`per_block`）、async TP（`_micro_pipeline_tp`
  + symm-mem，配置期/装配期双层 loud-raise）、regional_inductor（`aot_eager`×flex
  才 scoop，annotation 在 `flex_attention_hf`）、`capture_scalar_outputs`（含
  token-choice MoE block 时设置，dense 不动）四件各自独立开关，默认全关即旧整体
  compile 逐位不变。未移植登记：`skip_fwd_side_effects_in_bwd_under_checkpoint`、
  FakeTensorMode monkeypatch、`components` 列表。见 §5 符号行与 upstream map。
- `models/common/moe_sharding.py`：**部分移除**（2026-09-25）。TP×MoE 组合能力的
  声明层与装配层已就位（B 类适配，见 upstream map"已从 D 移除（部分）"）：
  `apply_tp` 接受 `moe_tp_experts` 等规格并结构性地实现 MoE-under-TP——专家权重
  F 维原地切分、router Replicate、块边界 AG/RS;**tp×ep 同日起按上游语义放行**
  （TP 只切 dense、EP 独占 routed 专家、router Replicate;`apply_tp` 在 ep>1 时把
  MoE 块留给 swap，专家梯度排除由 `tp_sharded_param_ids` 统一判定）;
  shared-expert×tp（gate/up/down 布局）同日放行（未知布局保持 loud-raise）；tp×ep×shared 保持拒绝。真多卡前后向等价性环境未覆盖，待 torch≥2.12
  复跑。符号对应：上游
  `expert_param_placement_sparse`（EP 轴 S(0) 声明）→ llmtuner EP swap 的 per-rank
  experts 切片（`parallel/expert_parallel/convert.py::convert_block`)；上游
  `dense_param_placement(tp=R)` 的 router Replicate 声明 → llmtuner router 不切 +
  `_allreduce_replicated_tp_grads` 求和；上游
  `_moe_sharding_config` 的块边界 in/out 声明（ep=1 时 Replicate）→
  `tensor_parallel/tp.py::TPMoeSequenceBoundary`（入口 `apply_tp` 在 `tensor_parallel/apply.py`）;ep>1 时的 sequence-parallel 布局
  → swap 后 MoE 直接消费/产出 T/tp 分片（无边界 collective);HF 侧
  `packed_colwise`/`moe_tp_experts` 规格 → `shard_experts_for_tp`。
- RegionAC：**已接入（2026-09-29 二十五次增量）**，`mode='region'` + `RegionACConfig`
  + `parallel/remat_regions.py`。上游的 `Module.configure_remat_regions` 协议在 llmtuner
  无对应物，改为结构等价：HF block 里的每个 `nn.Linear` 就是一个 region（名字取该
  block 内相对 FQN，pattern 语义与上游同为 fnmatch 相对名），再由
  `wrap_region` 用 `torch_remat.checkpoint` 包 block.forward。唯一受限项是上游自带的：
  `torch_remat` 要求 torch ≥ 2.10（本机 2.2.2 import 即失败），故 apply 期
  `ImportError`，文案同时给出 torch 版本要求与安装命令；`preserve_rng_state=True` 在
  config 期即拒（torch_remat 也拒，指向 `RecomputeStateHook`）。**数值未验证**：
  解锁条件 = torch ≥ 2.10 机器上安装 torch_remat 后复跑。MemoryBudgetAC 已于
  2026-09-24 移植（`mode='memory_budget'`，需 compile，语义同上游）。
- 2026-09-23 审计新增登记（上游 `c6e416bbd..b64103072` 引入）：
  - `components/optimizer/ema.py`：在线 EMA 模型平均（1b9eef3bd，515 行）——**已移植**
    （批 3a，`llmtuner/components/optimizer/ema.py` + config/trainer/checkpointer
    三侧接线，checkpoint `ema` 键，见 §4 与 upstream map"已从 D 移除"）。
  - Quantile-balanced MoE routing（f8bb599a7）：**已移植**（批 2，
    `QuantileBalancedTopKRouter` + `QuantileBalancer` + quantile hook；与
    sign-based bias 互斥，`ParallelConfig.moe_quantile_balancing` 启用，见 §4.3）。
  - MoE padding-mask 负载均衡（d34a13fdf）：**已移植**（批 2，
    `MoE.set_padding_mask` 通道；mask 只过滤统计不动执行，无 mask 逐位不变，
    见 §4.3）。
  - `CastLinear`（lm_head compute-dtype 变换，150c4f73a 配套）。
  - Ulysses CP × varlen/packed（baff3c681）：**已移植**（2026-09-25，批 5）。B 类
    适配：不复制 `UlyssesCPVarlenInnerAttention` 类层次；wrapper 在 ulysses 策略下
    全长透传文档 mask（`set_cp_mesh(strategy=...)`），kernel 按 mask Q 长度分派；
    `apply_cp` 对该组合的 fail-fast 移除。2-rank 等价性测试
    `cp_ulysses_varlen_equivalence.py` 已写，本机 torch 2.2.2 无 flex，环境未覆盖
    待复跑。详见 upstream map"已从 D 移除"。
  - 多轮对话 SFT 的 renderer 路径（4a0d8dab3）：**已适配为可选路径**
    （2026-09-25，第 12 项）。B 类语义适配：不复制 Configurable 外形、不新增
    硬依赖。上游 `components/renderer.py::RenderersLibraryConfig.build` →
    llmtuner `datasets/text/renderer.py::build_chat_renderer`（renderer 名以 CLI
    字符串传入，lazy `importlib` 探测；`auto`/`default` 两个 renderer 同样
    loud-refuse）；上游 `RendererTokenizerWrapper` → llmtuner 同名类（逐字，
    适配 `HuggingFaceTokenizer`）；上游 `ChatProcessor.Config.renderer` /
    `_tokenize_with_renderer` → llmtuner `ChatProcessor(renderer=...)` /
    `_tokenize_with_renderer`（`build_training_sample(..., ensure_final_stop=True)`、
    mask 随 label 移位、超长丢弃、文本限定，语义逐字）；renderer 路径与
    chat-template 路径互斥（构造期二选一），renderer 在时不再要求 eos_id。
    接线：`DataloaderConfig.chat_renderer`/`messages_field`（仅
    `dataset=local_jsonl_sft`）→ `datasets/build.py` →
    `make_local_jsonl_sft_multiturn`。未装 `renderers` 时启用 ImportError
    带 `pip install renderers==0.1.11` 指引；默认关闭逐位不变。真实库数值
    未验证（本机无 renderers，单测用 sys.modules fake 模块覆盖），解锁条件：
    pyproject 加 optional extra 后复跑。
  - validation 循环：**已移植**（批 4，`Trainer.validate`/`should_validate` +
    `ValidationConfig`，上游 `components/validate.py::Validator` 对应物；上游
    6c2dadbb3 的零 batch/零有效 token 报错与 90b25912f 的 dp>1 拒绝
    `steps=-1` 两条校验一并移植，另对 random 无限语料的 `steps=-1` 同样
    fail-fast；PP × validation 已于 2026-10-02 解锁（schedule eval 驱动），
    见 §2 trainer 表）。
- 2026-09-24 复核新增登记：
  - `token_dispatcher.py` 的 TorchAO/DeepEP/HybridEP 三个 dispatcher：**已对齐**
    （2026-09-25，§9.1 第 13 项）。TorchAO 落为可选导入适配层
    `TorchAOTokenDispatcher`（`_permute`/`_unpermute` 委托 torchao
    `permute_and_pad`，构造期 lazy import，未装 loud-raise ImportError 带
    `pip install torchao` 指引），DeepEP/HybridEP 保持登记缺口（CUDA-only +
    上游 `distributed/deepep/` wrappers 未 vendor），配置期
    NotImplementedError 带解锁条件；`ParallelConfig.ep_token_dispatcher` /
    `ep_torchao_pad_multiple` 接线，默认 `alltoall` 逐位不变。torchao 数值
    **环境未覆盖**（本机无 torchao/CUDA，单测以 fake 模块覆盖 sentinel-row
    padding 契约），解锁条件：CUDA 目标设备装 torchao 复跑。
  - router `_debug_force_load_balance`：**已移植**（批 1，`TokenChoiceTopKRouter`
    同名构造参数，round-robin 语义逐字一致，见 §4.3）。
  - `CastLinear`：**已移植**（批 1，`models/common/cast_linear.py` +
    `ModelConfig.compute_dtype`，state-dict FQN 不变，默认关闭，见 §4.1）。
  - PP per-stage seed：**已移植**（批 1，`trainer/seed.py::derive_distinct_seed` +
    trainer 接线；DTensor RNG tracker 不移植）。
  - `pipeline_with_first_stage_modules`：**已移植**（批 4，`apply_pp` 的
    `first_stage_module_fqns` 参数 + `prepend_first_stage_modules`；
    `split_model_into_stages` 配套置空非属主 stage 上的额外顶层模块，stage
    FQN 稳定，默认 None 逐位不变；当前无消费者，见 §5.4）。
  - transformers_modeling_backend 复核（同目录全量盘点，结论：其余功能均有
    等价支持或已登记裁剪）曾登记三项，**均已于 2026-09-24 对齐**：
    - DSA 模型：2026-09-28 起走稠密 additive mask（此前是构造期 fail-fast）；CP×DSA 仍拒绝（稠密 additive mask
      路径仍不实现，但静默错误语义已消除，见 §3）。
    - `experts_implementation` 旋钮：已移植（`ModelConfig` 字段 + wrapper
      应用，"可设置或 raise"上游同语义，见 §3）。
    - chat template 的 `bos_token`/`eos_token`/`add_generation_prompt` 自动
      注入：已移植到 `HuggingFaceTokenizer.apply_chat_template`（见 §7.3）。

#### 9.2 有意删除

`Configurable`、TorchTitan `Module`、`protocols/`、`structured_logger/`、quantization
组件均是设计裁剪，不应为了"对应完整"重新加入。

2026-09-29 二十二次增量再删一处：`OptimizersContainer.init_cache_state_dict` —— 上游基类是
`pass` no-op，只服务 TorchFT 容器（子类覆写）与 TorchFT 训练循环的无条件调用；llmtuner
无 TorchFT，移植物里那份 no-op 全仓零调用者，属无效抽象。将来接入 TorchFT 时补回一个
`pass` 方法即可（已同步登记在 `llmtuner_upstream_map.md` 的 D 表）。

#### 9.3 已清理的悬空链

`parallel/sharding.py` → `parallel/spmd_shims.py` 因没有上层消费者已整体删除；
2026-09-23 第二轮排查又删除了三个零调用者文件（`context_parallel/primitives.py`、
`models/common/flex_kernel.py`（连同悬空的 `HFFlexKernel._sharding_config`）、
`models/common/nn_modules.py`）与 `parallel_dims.py` 的 7 个未用 API。`accelerator/spmd_context.py`
是独立活代码。

### 10. 全模块符号索引

下表是快速查找入口，覆盖当前 99 个非 `__init__.py` / `__main__.py` 实现模块（截至
2026-09-28：`llmtuner/` 下 123 个 `.py`）。列出的为顶层类/函数和重要公共方法；私有
helper 在前文涉及关键算法时单列。成组条目（`config/`、`trainer/trainer.py` 等）在行内
一并列出同组子模块。"同文件"指本文前述路径变换后的 TorchTitan 文件。

| llmtuner 模块 | 重要符号 | 对应类别 |
|---|---|---|
| `errors.py` | `LLMTunerError` 与 `ConfigError` / `UnsupportedCombinationError` / `EnvironmentUnsupportedError` 三类 fail-fast 层级 | C，llmtuner 独有 |
| `components/checkpointer/base.py` | `ModelWrapper`, `CheckpointStorage`, `BaseCheckpointManager`, `purge_thread` | A2，同文件 |
| `components/checkpointer/dcp.py` | `CheckpointManager`, `AsyncMode` | A2，同文件 |
| `components/checkpointer/torch_checkpointing.py` | `TorchCheckpointingManager` 与 backend config helpers | A2，同文件 |
| `components/checkpointer/utils.py` | `canonical_fqn` | A1，同文件 |
| `components/loss.py` | CE、vocab CE、chunked CE、logprobs、MSE | A2，同文件 |
| `components/metrics.py` | monitor、logger、`MetricsProcessor` | A2，`observability/metrics.py` |
| `components/optimizer/lr_scheduler.py` | WSD factor、scheduler container/build | A2，同文件 |
| `components/optimizer/optimizer.py` | `OptimizersContainer` | A2，同文件 |
| `components/optimizer/ema.py` | 在线 EMA 平均，复用 flat FQN state-dict 契约 | A2，同文件（上游 1b9eef3bd） |
| `components/optimizer/utils.py` | optimizer state 初始化与 flat/FQN 转换 | A1，同文件 |
| `components/profiler.py` | `Profiler`, `MemoryProfiler` | A2，`observability/profiler.py` |
| `components/tokenizer.py` | tokenizer 三类 | A1，同文件 |
| `datasets/build.py` | `build_dataloader` | C，config build 替代品 |
| `datasets/collators.py` | `Collator`, `TextCollator` | A2，`components/data/collators.py` |
| `datasets/dataset.py` | dataset nodes 与 build 工厂 | A2，`components/data/dataset.py` |
| `datasets/loader.py` | `BaseDataLoader`, `GrainDataLoader` | A2，`components/data/loader.py` |
| `datasets/multimodal/collator.py` | `MultiModalCollator` | A2，`hf_datasets/multimodal/mm_collator.py` |
| `datasets/multimodal/datasets.py` | processor 与 sample packing | A2，`hf_datasets/multimodal/mm_datasets.py` |
| `datasets/multimodal/image.py` | decode/resize/patch helpers | A1，`hf_datasets/multimodal/utils/image.py` |
| `datasets/multimodal/text_utils.py` | padding 与 placeholder helpers | A1，`hf_datasets/multimodal/utils/text.py` |
| `datasets/multimodal/video.py` | video load/process | A1，`hf_datasets/multimodal/utils/video.py` |
| `datasets/packing.py` | 两种 packing 与 Stateful iterator | A2，`components/data/packing.py` |
| `datasets/random_data.py` | synthetic source/loader | C |
| `datasets/sources.py` | JSONL/HF sources 与 cursor | A2，`components/data/sources.py` |
| `datasets/text/processors.py` | text/chat processors | A2，`hf_datasets/text_datasets.py` |
| `datasets/text/renderer.py` | 可选 renderer 适配层（`build_chat_renderer` / `RendererTokenizerWrapper`） | B，`components/renderer.py`（未装 `renderers` 时 ImportError） |
| `datasets/types.py` | build context/iteration policy | A2，`components/data/types.py` |
| `models/common/activation.py` | activation wrappers | C，同名不同源 |
| `models/common/aux_loss.py` | aux-loss carrier/registry/hooks | A2，同文件 |
| `models/common/async_linear.py` | TP-overlap projections/FFN | A2，同路径同名（上游 9e159aed7 改名，llmtuner 2026-09-26 跟随） |
| `models/common/cast_linear.py` | `CastLinear` / `to_cast_linear`（lm_head compute-dtype 变换） | A2，`models/common/linear.py`（150c4f73a） |
| `models/common/embedding.py` | vocab-aware embedding | C，同名不同源 |
| `models/common/feed_forward.py` | FFN helpers/classes | A2，同文件 |
| `models/common/moe/experts.py` | `GroupedExperts`, `RoutedExperts` | B，common + gpt_oss MoE |
| `models/common/linear.py` | router/partial-bias linear | A2，同文件 |
| `models/common/attention/masks.py` | mask mods、varlen metadata | A2，`attention.py` 拆分 |
| `models/common/moe/`（`block.py` MoE 本体 + `router.py` + `experts.py` + `dispatcher.py` + `load_balance.py` + `balancing.py`） | router、experts、MoE、balance loss、bias 更新钩子 | A2，上游单文件 `moe.py` + `token_dispatcher.py`；2026-09-28 拆为子包（十七次增量） |
| `models/common/multimodal.py` | vision/text fusion helpers | A2，同文件 |
| ~~`models/common/param_init.py`~~ | init context/std helper | 已于 2026-09-25 删除（死代码） |
| `models/common/attention/qkv.py` | fused QKV 与 state hooks | A2，`attention.py` 拆分 |
| ~~`models/common/config_utils.py`~~ | 上游 config 工厂：`make_*_config`、`fused_*_param_init`、`get_attention_config`、`make_token_dispatcher_config` | 无对应物（2026-09-28 定性）：llmtuner 没有 config tree，`Module.Config` 那层整体不存在，其*判定*分别落在 `expert_parallel/probe.py`（top_k/score_func/route_norm/route_scale/expert groups）、`parallel/matrix.py`（组合裁决与 loud-raise）、`models/hf/factory.py`（HF config 构建）、`models/hf/model.py` 的 `_flex_supported`（attention backend 选择） |
| ~~`models/common/lora.py`~~ | `get_lora_linear`/`get_lora_grouped_linear` | 无对应物（裁剪面）：LoRA 由 HF/peft 提供，llmtuner 不持有上游 `_linear()` seam 与量化轴，故不移植 |
| `models/common/rope.py` | RoPE 全家族 | A2，同文件 |
| `models/common/scatter_add.py` | deterministic scatter-add autograd | A2，`ops/scatter_add.py` |
| `models/common/moe/dispatcher.py` | local/all-to-all dispatchers + TorchAO 可选导入适配层；DeepEP/HybridEP 登记缺口（config 期 loud-raise） | A2，同文件 |
| `models/hf/model.py` + `models/hf/factory.py` + `models/hf/flops.py` | wrapper/forward、config 构建/类解析/meta materialize、FLOPs | B，transformers backend model（`model.py` 与上游 `.../model.py` 同名） |
| ~~`experiments/transformers_modeling_backend/` 的 `module_conversion.py` / `config_registry.py` / `__init__.py`~~ | HF 模块的 `Module` 协议转换 / 实验用家族 config 注册表 / 模型注册表 | 无对应物（2026-09-28 定性）：llmtuner 没有 `Module` 协议这一层（同类 `__class__` swap 技术用于 `GatherSequenceFirst`/`TPMoeSequenceBoundary`/TP realizer），config 由 HF `AutoConfig` + `config/` 门面构建，模型类由 HF auto mapping 解析（`resolve_model_class`） |
| `models/hf/state_dict_adapter.py` | HF↔llmtuner state-dict 键转换与 safetensors index 严格校验 | B，`experiments/.../state_dict_adapter.py` |
| `parallel/activation_checkpoint.py` | full/selective AC | A2，distributed AC |
| `parallel/compile.py` | `apply_compile`（逐 block compile / async TP / regional_inductor / capture_scalar_outputs） | A2，`distributed/compile.py` |
| `accelerator/collectives.py` | reductions、timeouts、grad norm | A2（部分），`distributed/utils.py` 的两个 vendored 符号 |
| `parallel/context_parallel/apply.py` | `apply_cp` | C，独立 HF 编排层 |
| `parallel/context_parallel/cp_kernel.py` | `CPFlexKernel` 与 seq/head autograd | C，CP flex attention 组合实现 |
| `parallel/context_parallel/input_shard.py` | CP/TP batch 和 mask sharding | C，独立输入分片层 |
| `accelerator/dist.py` | object collectives、all_reduce/gather、collect_results | C，vendored 自 OpenMMLab `mmengine.dist`（非 torchtitan 来源），已去 mmengine 化 |
| `accelerator/dist_utils.py` | init_dist 多 launcher（后端字符串由 `device.py` 单源驱动）、rank/group 查询、`cast_data_device` | C，同上 |
| `parallel/expert_parallel/apply.py` | `apply_ep` | B，模型 EP parallelize |
| `parallel/expert_parallel/swap.py`（编排）+ `probe.py`（探测）+ `convert.py`（转换） | HF MoE 探测、权重搬运与 swap | B，transformers backend `moe_replacement.py` |
| `parallel/fully_shard/fsdp.py` | FSDP engine、mesh 与 placement | A2，`distributed/fsdp.py`（2026-09-28 逐项复核，八次增量） |
| `parallel/fully_shard/apply.py` | `apply_fsdp` HF driver | B，各模型 parallelize |
| `parallel/parallel_dims.py`（`build_parallel_dims` / `build_mesh` 2026-09-25 自 `accelerator/mesh.py` 并入） | `ParallelDims` 与 mesh accessors、dims/mesh/distributed init | A2，`distributed/parallel_dims.py`；mesh 构建段上游无单一对应物 |
| `parallel/head_sharding.py`（2026-09-27） | attention 头数整除守卫：`apply_tp` 的 `% tp` 与 ulysses CP 的 `% (tp*cp)` 共用一个实现 | B，`config/validation.py::validate_context_parallel` 的 `head_shard_degree`（上游在解析期校验，llmtuner 在装配期，因为头数只存在于模型 config 里） |
| `parallel/context_parallel/apply.py` 读的 `context_parallel_load_balancer` 默认值（2026-09-27） | CP 输入分片是否做 headtail 均衡 | B，`config/parallelism.py` 的 `context_parallel_load_balancer`（上游默认 `None`=连续分片，headtail 由 recipe 显式打开；上游自测 `test_config_manager.py` 亦断言默认 `None`）→ llmtuner 同日把默认从 `"headtail"` 改为 `None` 对齐，均衡分片改为显式选择（它改变每个 rank 参与 attention 的 token 集合，不该由默认替用户决定） |
| `parallel/parallelize.py`（2026-09-26 文件名对齐上游，原 parallelize_hf.py） | 五种并行的总装配 | B，transformers backend parallelize |
| `parallel/stages.py` | `STAGES` / `STAGE_ORDER` / `PP_STAGE_ORDER`：装配顺序契约的单一来源 | C，上游无对应物 |
| `parallel/pipeline_parallel/pipeline.py` | FQN split 与 stage 构造 | A2，transformers backend pipeline |
| `parallel/pipeline_parallel/apply.py` | metadata、apply、schedule build | B，`distributed/pipeline_parallel.py` |
| `parallel/tensor_parallel/linear.py` | fused/fallback collective GEMM | A2，`models/common/async_linear.py`（原 `distributed/linear.py`，上游经 dist_gemm.py 搬迁改名） |
| `parallel/tensor_parallel/tp.py` + `apply.py` | HF plan realizer 与 `apply_tp` 入口 | B，各模型 TP plan（上游 `distributed/tensor_parallel.py` 于 `7e7f271e0` 删除，后继为 `protocols/sharding.py` + `hf_sharding.py` / `decoder_sharding.py` 的声明面） |
| `config/`（`model/parallel/optimizer/checkpoint/data/training/root.py` + `cli.py`） | 全部配置 dataclass，逐组 `__post_init__` 校验；`cli.py` 是解析面的视图（`PARSER_GROUPS` + `cli_groups`），把 CLI 载不动的字段摘出 `--help` | B，`config/configs.py` + 嵌套 Config |
| `trainer/train.py` | parse/main | B，根 `train.py` |
| `trainer/trainer.py` + `builder.py`（装配段）/ `validate.py` / `pp_steps.py` / `batch.py` / `seed.py` | 完整训练生命周期 | B，根 `trainer.py` + `training_engine.py` |
| `components/checkpointer/checkpoint_keys.py` | checkpoint state key 常量 | C |
| `accelerator/device.py` | 设备发现、backend 选择、pin-memory 判定、NPU 谓词；**2026-09-29 二十一次增量**登记删掉无消费者的 mmengine 面（`is_cuda_available`/`is_mlu_available`/`is_musa_available`/`is_mps_available`/`is_dipu_available`/`get_max_cuda_memory`/`get_max_musa_memory`），但**实际只删掉 5 个**（`is_mlu_available`/`is_musa_available`/`is_mps_available`/`is_dipu_available`/`get_max_musa_memory`），`is_cuda_available`/`get_max_cuda_memory` 漏删；**二十四次增量**补齐这两个，并再删同样零消费者的 `is_npu_support_full_precision` 及其专用的 `torch_npu.npu.utils` 导入（`is_npu_available` 保留——`accelerator/dist.py` 在用） | C |
| `components/checkpointer/filesystem.py` | path/storage helpers | A1，`tools/filesystem.py` |
| `utils/gc.py` | `GarbageCollection` | B，`tools/utils.py` |
| `utils/logger_utils.py` | `get_logger`（彩色 formatter + 发射时 rank 过滤）、`get_distributed_rank` | C |
| `utils/lazy_exports.py` | `export_names` / `resolve_export`：五个包索引共用的 PEP 562 懒加载实现（2026-09-29 二十一次增量，此前是五份复制） | C |
| `accelerator/monitoring.py` | device/memory/FLOPS helpers | C；部分意图可参考 `tools/utils.py` |
| `accelerator/spmd_context.py` | SPMD mesh 上下文 | C，pip `spmd_types` 适配 |

五条横切约定，适用于上表所有模块：

- **公开面**：稳定面 = `llmtuner.LLMTunerConfig` / `llmtuner.Trainer` /
  `llmtuner.config.*` / CLI；集成面与内部分级见 design doc §3.4。
- **组合判定**：分工两处——config 期可判的组合校验在各 config 的 `__post_init__`
  （与其余字段校验同处）；跨层组合裁决（assembly/probe 期）的单一来源是
  `parallel/matrix.py`（每个组合一个普通函数 + 底部 `ENTRIES` 扁平表一行；守卫点
  触发、函数给出判定与文案）。
- **能力探测**：torch 版本/环境探测集中在 `accelerator/capabilities.py`
  （`has`/`require`，缺失经 `EnvironmentUnsupportedError` 报解锁指引）。
- **错误处理**：fail-fast 类型层级在 `llmtuner/errors.py`——`ConfigError`（配置非法，
  兼 `ValueError`）、`UnsupportedCombinationError`（组合拒绝，兼
  `NotImplementedError`）、`EnvironmentUnsupportedError`（依赖缺失、文案带解锁条件，
  兼 `NotImplementedError`）；可选包缺失保持 `ImportError`。
- **命名与私有面**：模块级 helper / 数据类**默认公开**（本仓是学习/参考实现，读者应当能直接
  import 任何一层；模块级私有名一旦被跨模块或测试引用就是在说谎）。前导 `_` 只保留三种：
  ① 框架协议要求的名字（`scatter_add.py` 的 `_backward`/`_setup_context` 是
  `torch.library.custom_op` + autograd 的契约）；② 与"公开包装"配对的低层核心——
  `accelerator/dist.py` 的 `_broadcast_object_list`/`_all_gather_object`/`_gather_object`
  同名公开版返回 list、私有版收 out-list，去前缀会重名；③ 类内 protected 方法（子类或同包
  协作者用的 template-method 钩子，如 `checkpointer/base.py` 的 `_should_save` 族与
  `optimizer.py::_post_init`/`_validate_params`）。类内 `self._x` 属性属于封装，不在本条范围。
  2026-09-29（二十次增量）按此做完整个 `llmtuner/`：模块级前导 `_` 从 95 个降到 **5 个**
  （3 个同名校验核 + 2 个协议名），方法级从 109 降到 103（6 个 llmtuner 自有的跨模块方法公开，
  5 个上游同名钩子保留）。**与上游同名的 helper 现在名字不同**（llmtuner 无前缀、上游带 `_`），
  §10 的映射行按"llmtuner 公开名 ↔ 上游名"读；`docs/` 里二十次增量之前的历史条目保留旧名。

### 11. 上游同步检查清单

每次同步 TorchTitan 时按以下顺序执行：

1. `git -C <torchtitan> log <baseline>..HEAD -- torchtitan/`，先按本手册路径缩小范围。
2. A 类剥离 docstring 后比较 AST；不要用 diff 行数或 `quick_ratio()`。
3. B 类写出上游修复的不变量，例如"非法 label 必须在 collective 前失败"，再在 llmtuner
   的 seam 上实现，不复制其 Config/Module 外形。
4. C 类只检查调用者与测试，不根据同名或低 ratio 猜来源。
5. 对每个变更至少运行相关 CPU 单测、`ruff check`、`compileall`、`git diff --check`。
6. 并行语义变更必须增加多进程等价性测试；NPU/CUDA 专属 kernel 需要实际设备验证。
7. 更新文件级映射和本文的结论、限制及验证数字；不要把环境不支持写成代码已支持。

### 12. 已知验证边界

- 最新 `vllm-ascend` 镜像的 Torch 2.10 缺少新版 FSDP per-parameter mesh result，且
  Transformers 5.14 超出项目声明范围；EP×FSDP 等能力不能在该镜像完整验证。
- 镜像默认缺少部分项目依赖；临时补齐首批所需依赖后有 169 项通过，5 项因容器没有正确
  映射 Ascend driver/device、在 torch_npu 初始化阶段失败。未运行集合没有通过结论。
- CPU 单测不能证明 symmetric-memory、HCCL/NCCL、真实多卡 overlap 的性能与死锁安全；
  它们必须由 integration/equivalence 脚本和目标设备补足。
- AST 相似度只用于找候选，不是正确性证明。本文的"通过"来自不变量审计和测试证据。
- 本机 torch 2.2 既没有 FSDP2 面（`torch.distributed.fsdp.fully_shard`、
  `_composable.fsdp.FSDPModule`、`_fully_shard` 私有包）也不支持 1-D 命名 mesh 的
  轴切片（上游同样写的 `edp_mesh["efsdp"]`）。`test_efsdp_placement.py` 与
  `test_fsdp_contract.py` 因此要靠一个临时 harness（`/tmp`，不入仓）才能跑：shim 只
  提供 FSDP2 的模块接口（`set_modules_to_*` 等）与命名轴的取值，**不做真实分片**，
  所以这两份文件证明的是放置决策与装配顺序，不是真机分片行为——后者仍需 torch≥2.12
  + 多卡复跑。
- TorchTitan 的 `last_save_model_only=True` 默认值会生成不可续训的最终导出物；训练示例
  必须显式设为 `False`。不要把带 `.metadata` 的 model-only DCP 误判为完整训练
  checkpoint。
- 2026-09-22 的 Torch 2.10 复核发现 PP 1F1B 尚未通过轨迹等价性：首步 loss 一致，后续
  optimizer step 后偏离，4 step 最大约 `8.5e-3`。FSDP、TP、CP 和 EP grad norm 的对应
  2-rank 等价性通过；PP 在根因修复前不能归入"通过"。
- 2026-09-27 在 macOS/Intel 开发机上复核：torch 2.2.2 低于项目要求（>=2.12），PyPI 无
  ≥2.3 的 macOS x86_64 wheel（conda-forge 有 osx-64 构建，尚未装入）。该机 pytest 的
  `_readline_workaround` 会在 import `readline` 时 segfault，须加 `-p no:capture`
  （`python -m pytest -p no:capture tests/unit_tests -q`，细节见 design doc §7）。
  实测 CPU 单测 143 passed / 9 failed / 59 skipped，9 个 failed 全部来自
  `torch.OutOfMemoryError` 与 `torch.distributed.pipelining` 缺失；25 个 integration
  脚本 2 passed / 23 failed，失败全部来自这批 torch 新 API 缺失（`DTensor`、
  `spmd_types`、`torch.distributed._composable.fsdp`、`torch.nn.attention`）。
  **因此本轮没有任何等价性结论**，需在 torch>=2.12 + 多卡环境重跑受影响套件。
- 2026-09-28 复核同一台 macOS/Intel 机的运行细节：(1) 本轮 CPU 套件为 152 passed /
  59 skipped / 9 failed（失败集与基线一致，仍是 profiler-OOM ×7 + `torch.distributed.
  pipelining` 缺失 ×2）；(2) **偶发 SIGABRT**：本轮全量运行 11 次中 2 次在 `import torch`
  阶段中止，日志首行为 `OMP: Error #179: Function Can't open SHM failed` —— 沙箱下 Intel
  OpenMP 拿不到共享内存，属环境抖动，与代码改动无关（重跑即恢复）；(3)
  `tests/unit_tests/cpu/components/test_metrics.py` 被 `require_env('wandb',
  'pipelining', 'flex_attention')` 整体门控，本机永远 skip；临时脱门禁后 69 passed /
  5 failed，5 个失败**均属该文件既有漂移、与本轮改动无关**：2 个是自带 fake wandb 的
  `log()` 没有 `commit` 参数（`WandBLogger.log(..., commit=True)` 是 2026-09-23 的改动，
  因该文件常被跳过而没同步到 fake）、1 个 wandb 缺失路径的断言与实现不符、2 个引用已不存在
  的 `hf/model.py` 里从未导出的 `build_model_config_for`（2026-09-28 十七/十八次增量已随
  FLOPs 用例迁移一并删除该断言）。登记待修，不在本轮范围内。
- 2026-09-28 同轮的 optimizer 面：`tests/unit_tests/cpu/components/optimizer/
  test_optimizer_container.py` 整体被 `torch_param_names` 门控（该 cap 探测的是
  optimizer `state_dict()` 的 param group 是否带 `param_names`，torch 2.2.2 无），
  本机同样永远 skip；脱门禁后 18 passed / 1 failed，唯一失败即门控原因本身
  （`test_state_round_trips_into_a_fresh_container` 需要 `param_names`）。`gpu-only` 的
  部分（真 CUDA fused 核、XPU）仍需目标设备验证。
- 2026-09-23 审计同样在 macOS 开发机执行：当轮 `spmd_types`/`grain` 未装、32 个测试文件
  收集即失败，改动只有静态门禁、shim 级验证与 vocab-loss 2-rank gloo 等价性覆盖。详见
  `llmtuner_torchtitan_alignment_audit_2026-09-23.md`（不在当前工作区）。

---

# 附录：对齐工作流（原 `llmtuner_torchtitan_alignment_workflow.md`，2026-10-08 并入）

本文指导 agent 持续对齐 `llmtuner` 与 TorchTitan，同时保持 llmtuner 的设计边界和运行
正确性。它是执行流程，不取代三份事实文档：

- 本文正文（文件级映射）：文件来源与 A/B/C/D 分类的唯一权威。
- 本文附录（符号级对应）：函数、类和方法的对应关系。
- [`torchllmtuner_design.md`](./torchllmtuner_design.md)：架构契约、装配顺序和支持边界。

若三份文档与源码冲突，以当前源码及可复现测试为准；确认事实后先修正文档，再继续迁移。

文中 `<llmtuner>` / `<torchtitan>` 指两个仓库的检出根目录（本仓库即 `<llmtuner>` 的
上一级；torchtitan 检出位置依机器而定）。

### 1. 目标与非目标

#### 1.1 目标

1. 识别 TorchTitan 在上次审计基线之后的有效变化。
2. 按 A/B/C/D 分类将其移植、适配、忽略或登记为能力缺口。
3. 用单测、多进程等价性测试和目标设备测试证明数学语义与运行契约正确。
4. 同步维护文件映射、符号导航、设计边界和验证记录。

#### 1.2 非目标

- 不追求目录结构、类层次或 diff 行数与 TorchTitan 一致。
- 不重新引入 `Configurable`、TorchTitan `Module`、配置树或模型 registry。
- 不把有意裁剪的 quantization、structured logger、protocol 等内容当成遗漏。
- 不用静默退化掩盖未支持的并行组合。
- 不在源码中添加会腐烂的固定 SHA `# upstream:` 注释。

### 2. 不可破坏的 llmtuner 契约

每轮对齐前必须确认并保持：

1. 配置沿 `CLI -> LLMTunerConfig -> 顶层显式参数` 单向传递。
2. 并行装配由普通函数完成，非 PP 路径顺序保持（单一来源
   `llmtuner/parallel/stages.py` 的 `STAGES` 表，测试钉死）：

   ```text
   apply_tp -> apply_ep -> apply_cp -> apply_ac -> compile -> apply_fsdp
   ```

3. HF wrapper 继续暴露 `tok_embeddings / layers / norm / lm_head / rotary_emb`
   五部件契约；`named_children()` 不负责改写 state-dict FQN。
4. 模型和并行底层不得读取 trainer 的全局运行配置。
5. TP、CP、EP、PP 和 FSDP 的未支持组合必须 fail fast。
6. `accelerator/spmd_context.py` 是活跃运行路径；旧的悬空链
   `parallel/spmd_shims.py -> parallel/sharding.py` 已于 2026-09-21 删除，不要以
   任何形式复活或与之混淆。

### 3. 执行前准备

#### 3.1 工作区保护

1. 查看 `git status --short`，记录用户已有修改和未跟踪文件。
2. 不回退、不覆盖、不格式化与本轮任务无关的修改。
3. 将任务拆成可以独立验证的小批次；不要同时重写数学逻辑、装配顺序和测试基线。
4. 编辑源码或文档时使用小范围 patch。

#### 3.2 加载环境

```bash
cd <llmtuner>
source ./set_env.sh   # 存在时
```

目标设备测试使用项目指定的最新 `vllm-ascend-env` 容器。当前 agent 无法进入该容器时，
继续完成静态检查和 CPU 可运行测试，并明确把设备验证列为未完成，不能声称已经通过。

#### 3.3 记录审计基线

```bash
git -C <llmtuner> rev-parse HEAD
git -C <torchtitan> rev-parse HEAD
git -C <torchtitan> log <上次-torchtitan-基线>..HEAD -- torchtitan/
```

把双方提交、日期、Python/PyTorch/Transformers 版本写入本轮验证记录。固定 SHA 只进入
审计记录和文档的版本章节，不进入源码注释。

### 4. 建立变更清单

对 TorchTitan 的每个变更文件执行：

1. 在 `llmtuner_upstream_map.md` 查找文件级对应关系。
2. 在 symbol guide 中定位受影响的函数、类及 llmtuner 调用者。
3. 用 `rg` 检查 llmtuner 当前调用图、测试和公开导出。
4. 记录上游改动真正维护的不变量，而不是先复制实现。
5. 为每项变更建立记录：

| 字段 | 内容 |
|---|---|
| TorchTitan 文件/符号 | 上游路径和符号 |
| llmtuner 文件/符号 | 本地路径和符号 |
| 主分类 | A / B / C / D |
| 上游意图 | bug fix、边界检查、性能或新能力 |
| 必须保持的不变量 | shape、dtype、FQN、collective、梯度或生命周期语义 |
| 计划动作 | 移植、适配、无需动作或单独立项 |
| 验证方式 | 单测、等价性、容器或目标设备测试 |

不要仅凭同名文件或最高 AST ratio 判断来源。AST 相似度用于找候选，不是正确性证明。

### 5. 按分类执行

#### 5.1 A 类：高保真移植

适用于算法和结构主要来自 TorchTitan 的文件。执行顺序：

1. 比较同名符号，再比较文件整体。
2. 剥离 docstring 后用 AST 结构比较辅助定位差异。
3. 逐项核对签名、shape、dtype、异常、collective、autograd 和 state-dict 行为。
4. 同步有效 bug fix，但移除上游 `Configurable`、`Module` 等 llmtuner 不采用的外形。
5. 运行该模块单测及对应等价性测试。

推荐批次：

1. filesystem、初始化、RoPE、mask、packing 等纯函数。
2. loss、optimizer、scheduler、checkpoint。
3. QKV、MoE、dispatcher、TP linear 和 FSDP engine。

#### 5.2 B 类：语义适配

适用于相同意图但实现形状不同的文件。只迁移不变量、错误检查和数学语义，不复制上游
类层次。重点模块：

- `models/hf/model.py`
- `parallel/parallelize.py`
- `parallel/tensor_parallel/tp.py`
- `parallel/expert_parallel/*`
- `parallel/context_parallel/cp_kernel.py`（CP redistribution 与 kernel；原 `primitives.py` 已于 2026-09-23 删除）
- `parallel/fully_shard/apply.py`
- `parallel/pipeline_parallel/apply.py`
- `trainer/*` 与 `parallel/parallel_dims.py` 的 build 入口

实现前必须写明对应不变量，例如：

- TP 权重布局和前后向 collective 必须互为对偶。
- CP 分片输出和梯度必须等价于全序列参考。
- EP swap 必须精确保留 HF 权重、router 语义和本地 expert 范围。
- PP stage FQN 必须稳定，optimizer/checkpoint 键不能跨 stage 冲突。
- loss 必须按全局有效 token 数归一化。

#### 5.3 C 类：llmtuner 独有

C 类没有可机械同步的上游实现，只能依据 llmtuner 的调用图、设计契约和测试判断是否修改。

- `accelerator/spmd_context.py` 是活代码。
- CP 编排、flex kernel、random dataset、build factory 和工具模块不能因低相似度或
  同名文件被强行覆盖。

#### 5.4 D 类：真正缺口

D 类必须独立设计和验收，不能伪装成单文件同步。当前清单以
本文正文（文件级映射） D 类表为准。截至 2026-09-27 的
剩余项只有：

- RegionAC：**已接入**（2026-09-29），声明通道改为结构等价（HF block 的 `nn.Linear`
  FQN 即 region 名），唯一受限项是上游自带的 `torch_remat` 需 torch ≥ 2.10；apply 期
loud-raise 并给出安装命令，数值待 torch ≥ 2.10 机器复跑。
- DeepEP / HybridEP dispatcher：CUDA-only，且上游 `distributed/deepep/` wrappers
  未 vendor。
- TP×MoE 的**多卡等价性证据**：声明层与装配层已就位（`shard_experts_for_tp` +
  `TPMoeSequenceBoundary`），缺的是 torch≥2.12 多卡复跑，不是实现。

已从 D 类移除的能力（`distributed/compile.py` 四件、MemoryBudgetAC、在线 EMA、
quantile routing、Ulysses×varlen、validation 循环、`pipeline_with_first_stage_modules`
等）逐项记录在同表的"已从 D 移除"段落里。

对 D 类先写设计提案，说明依赖、契约、组合矩阵、失败模式和测试计划。未经明确授权，
不要在普通上游同步任务中扩大范围实现这些能力。

### 6. 推荐实施批次

按风险由低到高执行，每批独立验证：

1. **纯函数与工具**：filesystem、初始化、RoPE、mask、packing。
2. **components**：loss、optimizer、scheduler、checkpoint、metrics、profiler。
3. **datasets**：source、loader、packing、文本和多模态处理。
4. **模型数学层**：QKV、MoE、dispatcher、grouped experts、aux loss。
5. **并行原语**：TP linear、CP redistribution、EP all-to-all、FSDP mesh。
6. **总装配**：HF wrapper、trainer、PP、checkpoint resume 和混合拓扑。

一批失败时停止扩展下一批，先定位是代码错误、测试假设、依赖缺失还是环境 API 不兼容。

### 7. 正确性验证流水线

#### 7.1 静态门禁

```bash
ruff check llmtuner tests
python -m compileall -q llmtuner
git diff --check
python check_doc_refs.py docs/*.md          # 见下
```

文档引用校验（六份文档全部用 `文件:行号` 定位，行号会随源码改动漂移，必须机械复核）。
保存为 `check_doc_refs.py` 并在仓库根目录运行：`python check_doc_refs.py docs/*.md`。

```python
import pathlib, re, sys

UPSTREAM = pathlib.Path("<torchtitan checkout>")   # 与文档 §0 记录的基线一致
ROOT = {"torchtitan/": UPSTREAM, "llmtuner/": pathlib.Path("."),
        "tests/": pathlib.Path("."), "docs/": pathlib.Path(".")}
REF = re.compile(r"`([A-Za-z0-9_./\-]+\.(?:py|md|toml|yaml|json|sh)):(\d+)(?:-(\d+))?`")

total = bad = 0
for doc in sys.argv[1:]:
    for m in REF.finditer(pathlib.Path(doc).read_text()):
        total += 1
        path, start, end = m.group(1), int(m.group(2)), m.group(3)
        root = next((r for p, r in ROOT.items() if path.startswith(p)), None)
        src = root / path if root is not None else None
        lines = len(src.read_text().splitlines()) if src and src.exists() else 0
        if not lines:
            print(f"{doc}: {m.group(0)} -> 文件不存在"); bad += 1
        elif not 1 <= start <= int(end or start) <= lines:
            print(f"{doc}: {m.group(0)} -> 越界（文件 {lines} 行）"); bad += 1
print(f"引用 {total} 条，问题 {bad} 条")
```

局限：只认完整的 `` `路径:行号` ``。文档里的续引（``（`:207`）``、`` `:244-260` ``）
脚本看不到，改动附近代码后要人工回看；反过来，脚本通过**不等于**符号对得上——
上游文件变化过（见 upstream map 的"版本与漂移"）时必须抽查符号内容。

同时检查：

- 非入口实现模块仍被 symbol guide 覆盖。
- upstream map 中一个文件只有一个主要分类。
- 没有新增 `Configurable`、TorchTitan `Module` 或固定 SHA 来源注释。
- 底层模块没有新增对 trainer/run config 的反向依赖。
- 未支持组合仍有明确异常或拒绝路径。

#### 7.2 CPU 单测

先运行受影响模块，再裸跑完整套件：`python -m pytest tests/unit_tests -q`
（无 --ignore；环境门禁在测试模块自身的 `require_env(...)` 声明里，见
design doc §7）。记录 `passed / failed / skipped`；`[env] missing: ...` 的
skip 理由（`pytest -rs`）即 optional dependency 或 PyTorch API 不匹配的
未运行项清单。

macOS 开发机例外：pytest 的 `_readline_workaround` 会在 import `readline` 时
segfault（与 torch 无关），加 `-p no:capture` 即可正常收集
（`python -m pytest -p no:capture tests/unit_tests -q`）；原因与影响见 design doc §7。

"可运行集合通过"不能写成"全套测试通过"。失败项不得在没有证据时归因于环境。

#### 7.3 多进程等价性

数学或分布式语义变化至少覆盖：

| 领域 | 必须验证的内容 |
|---|---|
| TP | forward/backward、replicated gradient、vocab loss、TP×FSDP |
| CP | KV all-gather、Ulysses、packed、CP×TP、梯度归约 |
| EP | HF 权重搬运、all-to-all、aux loss、拒绝的 grad clip、EP×FSDP |
| PP | 1F1B、Interleaved1F1B、full/resume checkpoint |
| Loss | vocab-parallel、chunked loss、全局 token 归一化和跨 rank 可见性 |

等价性测试必须满足：

1. 分片执行与单卡/全量参考在明确容差内一致。
2. 同时检查输出、loss、关键参数梯度和必要的 state-dict/FQN。
3. 包含 non-vacuity 断言，避免全零输入或全零梯度造成虚假通过。
4. collective 前的输入校验必须在所有 rank 对称执行，避免局部异常导致其他 rank 挂死。

#### 7.4 目标设备与容器验证

在最新 `vllm-ascend-env` 中验证：

- HCCL/NCCL collective 和多卡进程生命周期。
- NPU/CUDA 专属 fused kernel 与 fallback 的选择。
- symmetric-memory 路径。
- 多卡 overlap、超时和死锁安全。
- checkpoint 保存、退出、恢复后的 loss 轨迹。

若容器中的 PyTorch 缺少所需私有 API，记录为"环境未覆盖"，保留 fail-fast，不可为了
让测试变绿而绕过 placement、DTensor 或 collective 语义检查。

### 8. 结果记录模板

每个批次完成后追加一份记录（长期记录单独成文，命名
`llmtuner_torchtitan_alignment_audit_<日期>.md`，并在 upstream map 的版本章节登记）：

```markdown
### <批次名称>

- llmtuner 基线：`<sha>`
- TorchTitan 基线：`<sha>`
- 环境：Python / PyTorch / Transformers / backend
- 涉及分类：A / B / C / D

#### 变更

| llmtuner 符号 | TorchTitan 符号 | 上游意图 | 本地处理 |
|---|---|---|---|

#### 验证

| 命令 | passed | failed | skipped/deselected | 备注 |
|---|---:|---:|---:|---|

#### 未验证或受限

- <设备、依赖、组合或 API 边界>

#### 文档更新

- [ ] upstream map
- [ ] symbol guide
- [ ] design document
```

### 9. 完成门禁

只有同时满足以下条件，才能宣告一批对齐完成：

1. 上游变化已经逐文件、逐关键符号审计。
2. A/B/C/D 主分类明确且无冲突。
3. llmtuner 设计契约和装配顺序没有被破坏。
4. 新增或改变的能力具有对应单测或等价性测试。
5. 未支持组合保持 loud-raise。
6. CPU、容器和目标设备结果被分别记录，没有扩大结论。
7. 静态门禁通过，相关测试无未解释失败。
8. 三份权威文档已按事实同步更新。
9. 工作区中用户原有的无关修改保持不变。

若任一项未满足，状态写为"部分完成"或"受限"，列出剩余动作，不得用"已完全对齐"
代替具体证据。

### 10. Agent 最终交付格式

最终回复应简洁包含：

1. 已对齐的文件和核心语义。
2. 修复或保留的关键不变量。
3. 实际执行的验证命令和结果。
4. skipped/deselected/环境不兼容项。
5. 尚未支持的组合和 loud-raise 状态。
6. 更新后的文档链接。

不要仅报告"测试通过"或"已与上游一致"；必须说明通过了哪些测试、在哪个环境中通过，
以及哪些能力仍未验证。
