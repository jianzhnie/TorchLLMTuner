# TorchLLMTuner × Ascend NPU 适配状态

goal.md 的验收记录。只写有命令与日志证据的结论；未跑的一律标"未跑"，
不用"应该可以"替代证据。

## 0. 一句话结论

> **2026-10-20 更新**：环境已升级为 torch 2.12.0 + torch_npu 2.12.0 + transformers 5.11.0，
> 并合入上游 49 个 commit。P0–P3 全部在新栈上重新验证通过（P1/P2 等价性 19/19、P3 20/20，
> ZBV 与 `cp_ulysses` 两个旧平台门已解除）；**P4（TorchTitan 精度对齐）经决策跳过**。
> 详见 §9。以下为 2.9.1 旧栈的历史记录，结论以 §9 为准。

**P0 完成；P1 五族（FSDP/TP/PP/EP/CP）全部在 8 卡 NPU 上跑通——既有真机训练，
也有 DoD 点名的仓库等价性脚本在 NPU 上实跑（15 条点名脚本：**14 通过**、1 条仍是环境门
（`cp_ulysses_equivalence`）；另有 `pp_equivalence` 的 ZBV 调度是平台门）；P2 的 12 组组合
（到四族混用）跑通，且其中三条已在**真实 Qwen3-8B** 上复验（pp=2、cp=2，含存盘与续训）；
P3 矩阵内可验证项全部跑通。** 三个未完成项：**ZBVZeroBubble 调度在 NPU 上是平台门**（torch 的 PP 形状推导走 HCCL 对象 p2p，见 §7.1，
CPU 上通过）、**`cp_ulysses_equivalence` 仍是环境门**（私有 `_HeadTailLoadBalancer`）、**P4** 需要 torch 2.14 环境（参考实现在本容器跑不起来）。

过程中在本仓库修掉 11 个真实缺陷（不是"绕过去"）：

1. `DeviceMesh._unflatten` 在 torch 2.9 缺失 → 改用公开 `init_device_mesh`；
2. `spmd_types` 在 torch 2.9 完全不可导入 → 新增 `spmd_backend` 降级层（告警不静默）；
3. TP 序列并行下 HF 按本地长度造 causal mask → 由 llmtuner 提供全长掩码；
4. **EP 的 dispatch/combine 用了无 autograd 的 `all_to_all_single`** → 专家权重拿不到梯度，改用可微形式；
5. PP 的 loss 哨兵形状/dtype 与末级产物不一致 → bf16 默认 dtype 下归约读写出错（负的 sum、`-3e27` 的 max），改为 0 维 float32；
6. TP 先于 EP 装配会把专家投影换成并行 linear → MoE 探针的类名判据失效，改为按形状判据。
7. CP 在 sdpa 后端上原本整条不可用（`apply_cp` 硬拒绝非 flex）→ 实现 sdpa 版 CP kernel（gather KV + 偏移因果掩码，gather 的反向必须"先归约再切块"）并接线；
8. TP×CP 时 `sequence` 归约 mesh 是 2 维，`.get_group()` 拒绝 → 按 `loss` 视图的既有做法展平。
9. `torch_checkpointing` 后端的移植**照着一个比已发布 0.1.0 更新的 API 写**，装包后连不上；三处适配后 checkpoint 往返通过（详见 §5）。
10. **TP×EP 时专家被 TP 折叠了 F 维**（静默算错）：默认 TP plan 的 `mlp.experts.*.{gate,up,down}_proj` 模式会匹配 `experts_implementation="eager"` 下的**逐专家 `nn.Linear`**；而 `tp×ep` 时 MoE 实现被推迟给 `apply_ep`，于是那段"排除 MoE 内部"的守卫被跳过——专家的 F 被切成 `F/tp`，EP swap 又把这些半成品权重搬进 `GroupedExperts`，原本会做 F 求和的 realizer 集合通信随之丢失：**每个专家只贡献一半输出**。实测（4 卡 CPU/fp64）：MoE 块输出相对自身量级偏 76%，DoD 点名的 `tp_ep_cp_equivalence` 报 `5.147e-02`。修法：无论 MoE 实现是否被推迟，都把 HF MoE 块内部从 dense targets 里排除（`resolve_tp_targets`）。修后同一脚本在 CPU/fp64（1e-6）与 NPU/fp32（1e-4）都通过，MoE 块误差 `5.744e-10`（详见 §7.4）。
11. **多文档 packing 在非 CUDA 后端上被硬拒绝**（等于 NPU 上不可用）：packed 语料（`attn_mask_type='block_causal'`）的掩码只有 flex 的 BlockMask 能表达，`_apply_attention` 于是直接 ValueError。而同一个仓库里**已经有** dense 版实现（`build_dense_attention_mask`，原本给 DeepSeek 稀疏注意力用）。改为在张量后端上物化 dense 的 block-causal 掩码（O(T²)，T=4096 时 bf16 约 32MiB/次前向），并补上 PP 非首级的 dtype 来源（`_mask_dtype`：首级有 embedding，后续级是 `nn.Identity`）。修后 `pp_real_corpus_equivalence` 在 NPU/fp32 与 CPU 上都通过（终态参数差 `2.384e-07`），8 卡 Qwen3-8B + `max_num_documents=8` 3 步 exit 0（§4 P3 行）。

另有一处按仓库既有惯例**改为明确拒绝**（而非留下会静默出错的路径）：`chunked loss × FSDP`（torch 2.9 的 unshard 不能作上下文管理器）。EP×FSDP 在本栈上走嵌套单元回退（`shard_experts_on_their_own_mesh`），是**支持**的：8 卡与 4 卡 `ep_fsdp_equivalence` 都通过（§4、§7.1）。

另：`outputs/npu_harness/on_npu_*.log` 里还有 **20 条非点名 harness** 在 NPU 上通过（10 条 P3 特性：activation checkpoint、PP×TP、PP×validation、metrics 一致性、chunked×PP 拒绝、vocab-parallel loss、异步 DCP 生命周期、EP×AC×EMA 存盘、PP 存盘 plain/packed；外加 EP 接线/grad-norm/辅损/Domino、TP 复制梯度、归约语义、量化平衡、chunked×TP、TP×EP 无 dense DP）——P3 此前只有 8 卡矩阵行，现在有『与单卡/单进程参考逐张量一致』级的证据，见 §7.1。

## 1. 目标环境（实测）

| 项 | 值 |
| --- | --- |
| 容器 / 镜像 | `hyper-parallel-env` / `hyper-parallel-npu:cann9.1.0-torch29` |
| NPU | 8× Ascend910_9392（61.27 GiB/卡），CANN 9.1.0 |
| Python | 3.12.13（`/usr/local/python3.12.13`） |
| torch / torch_npu | **2.12.0+cpu / 2.12.0**（2026-10-20 由 2.9.1 升级，§9） |
| transformers | **5.11.0**（2026-10-20 由 4.56.2 升级，§9） |
| torchvision | 0.27.0+cpu |

容器内**没有 DNS**（`mirrors.tuna.tsinghua.edu.cn` 解析失败），所以容器内 `pip`
不可用；缺的依赖在宿主机下载 wheel 后，用
`pip install --no-index --find-links=<dir>` 装进容器。

## 2. 环境变更（相对原始镜像）

新增：`colorama`、`grain 0.2.18`、`datasets 3.6.0`、`dill 0.3.8`、`einops`、
`pillow`、`pyarrow 26.0.0`、`pandas 3.0.6`、`xxhash`、`multiprocess`、`aiohttp`
及其依赖（wheel 目录 `/home/jianzhnie/llmtuner/wheels/local`）；验证
`torch_checkpointing` 后端时又装了可选的 `torch_checkpointing 0.1.0`
（纯 Python，只需 `torch>=2.6`，不影响 torch；wheel 在 `wheels/ckpt`）。

升级：`protobuf 3.20.0 → 5.29.5`。原因：所有 `grain` 版本生成的 pb2 都 import
`google.protobuf.runtime_version`（4.25+）。升级前已确认容器内没有运行期包声明
`protobuf<4`（那些 `<4.0.0` 只出现在 transformers 的可选 extra 中）；升级后
`torch_npu` 正常、`device_count()==8`。

未改动：`torch`、`torch_npu`、`transformers`、`numpy`、`huggingface_hub`。
`datasets` 曾装到 5.1.0，因其要求 `huggingface_hub>=2`（会破坏 transformers
4.56）而退回 3.6.0。

## 3. 代码改动

| 文件 | 改动 |
| --- | --- |
| `llmtuner/accelerator/spmd_backend.py`（新增） | 解析 `spmd_types` 后端；torch < 2.12 无 `torch.distributed._local_tensor` 时降级为 no-op 类型检查，并在导入时告警 |
| `llmtuner/accelerator/spmd.py` | 改从 `spmd_backend` 取 `spmd` |
| `llmtuner/models/common/moe/gate.py` | 同上 |
| `llmtuner/parallel/tensor_parallel/linear_ops.py` | 同上 |
| `llmtuner/parallel/tensor_parallel/tensor_parallel.py` | **TP×EP 的真实缺陷修复**（§0 第 10 条）：`resolve_tp_targets` 里"排除 MoE 块内部"的过滤原先只在 `moe_blocks` 非空（即 MoE 的 TP 实现真的被 realize）时生效，而 `tp×ep` 会把实现推迟给 `apply_ep`，于是 eager 布局下逐专家的 `nn.Linear` 仍被 dense plan 匹配并切了 F 维。现在改成：只要模型里有 HF MoE 块（且没被 TP boundary 接管），其内部一律从 dense targets 里剔除——这同时覆盖 fused 布局（原本就对）、`tp×ep`（推迟到 EP swap）与 `ep=1`（eager 布局没有可挂载的 stacked 参数）三种情形 |
| `llmtuner/parallel/topology.py` | `DeviceMesh._unflatten`（torch 2.12 私有 API）缺失时改用公开的 `init_device_mesh` 构造等价布局 |
| `llmtuner/parallel/context_parallel/input_sharding.py` | `_context_parallel_shard`（torch 2.12 私有）缺失时，连续无负载均衡的分片走公开 API 回退（chunk + mesh 坐标）；带 load balancer 与 BlockMask 的路径仍明确报错 |
| `llmtuner/models/hf/model.py` | 序列并行（TP>1）且后端非 flex 时，提供 gather 后全长度的 4D causal mask；否则 HF 会按 rank 本地长度造 mask |
| `llmtuner/models/hf/model.py`（同上文件，另一处） | **多文档 packing 的 dense 掩码回退**（§0 第 11 条）：非 flex 后端上 packed 语料原本直接 ValueError；改为复用既有的 `build_dense_attention_mask(..., block_causal=True)` 物化 dense 的 block-causal 掩码（`[1,1,T,T]`，0=允许 / -inf=禁止），并新增 `_mask_dtype()`——首级取 embedding、后续 PP 级取 lm_head/参数（非首级 `tok_embeddings` 是 `nn.Identity`，原来的 dtype 来源会在 PP>1 时 `AttributeError`） |
| `llmtuner/parallel/pipeline_parallel/microbatch_drivers.py`（新增） | torch 2.9 没有微批驱动时的 PP 训练/评估驱动：复现 `step()`/`eval()` 在交给 `_step_microbatches` 前的那几步 stage 设置 |
| `llmtuner/trainer/runtime/{pipeline,validation}.py` | 有原生驱动时用原生，否则走上面的回退；去掉 PP 的 capability 硬门 |
| `llmtuner/parallel/pipeline_parallel/pipeline_parallel.py`、`accelerator/capabilities.py` | 去掉装配期的 `require("pipelining_microbatch_drivers")`；capability 条目改为「可选，缺失时走回退」 |
| `llmtuner/models/hf/model_setup.py` | transformers < 5 没有可设置的 experts implementation API：请求 `eager` 时其内置路径本就是逐专家循环，视为已满足（记录 INFO）；`grouped_mm`/`batched_mm` 与稠密模型仍明确报错。新增 `is_moe_config` 结构探测 |
| `llmtuner/parallel/expert_parallel/hf_inspection.py` | `SiLUActivation` 改为可选导入（4.56 用 `torch.nn.SiLU`）；新增逐专家 `ModuleList`（`gate_proj/up_proj/down_proj` 或 `w1/w3/w2`）→ 融合 `(E,2F,D)/(E,D,F)` 的读取 |
| `llmtuner/models/common/moe/layer.py` | `MOE_LAYER_ATTRS` 增加 4.x 的 `block_sparse_moe`（Mixtral） |
| `llmtuner/models/common/moe/layer.py` | `MOE_LAYER_ATTRS` 增补 4.x 拼写（`block_sparse_moe`、`moe`）；`MoE` 增加 `returns_router_logits`：置位时 `forward` 返回 `(out, None)`，复现被替换块的返回元数 |
| `llmtuner/parallel/expert_parallel/{hf_inspection,model_replacement}.py` | `moe_block_of` 改为优先选「确实是 MoE」的属性（DeepSeek 层同时有稠密 `mlp` 与稀疏 `moe`）；新增静态的 `returns_router_logits`（AST 解析 HF 的 `forward`）；对开启 `output_router_logits` 的配置明确拒绝 |
| `llmtuner/accelerator/capabilities.py`、`parallel/fully_shard_parallel/{fully_shard_parallel,plan}.py` | 新增 capability `fsdp_per_param_mesh`（torch ≥2.12 才能在一个 FSDP 单元里逐参数选 mesh）；EP>1 的两条装配路径改为按该 capability **明确拒绝**，取代原先的裸 `ImportError: ShardPlacementResult` |
| `llmtuner/parallel/expert_parallel/token_dispatch.py` | **EP 训练的关键修复**：dispatch/combine 两个 all-to-all 改用可微的 `all_to_all_single_autograd`（缺它时大声告警并说明专家权重不会被更新）。此前用的是无 autograd 的 `all_to_all_single`，梯度不穿过 EP 交换 |
| `llmtuner/parallel/expert_parallel/hf_inspection.py` | 逐专家投影改按 **2-D weight** 判形状（原用 `isinstance(nn.Linear)`）。TP 先于 EP 装配会把专家投影换成 llmtuner 的并行 linear，类名判据会让探针在 TP 后认不出 MoE，从而 EP 直接拒绝 |
| `llmtuner/trainer/runtime/assembly.py` | **PP 哨兵形状/dtype 修复**：`_pp_loss_sentinel` 由 `torch.zeros((1,), …)`（形状 `(1,)`、dtype 跟随进程默认值）改为 `torch.zeros((), dtype=torch.float32, …)`，与末级实际产物 `sum(micro_losses) * global_valid_tokens`（0 维 float32）一致。此前 pp 组内两侧字节数不等（bf16 下 2 vs 4）会让归约读写出错 |
| `examples/train_qwen3_8b_npu.py` | 增加 `LLMTUNER_TENSOR_PARALLEL_SIZE` / `LLMTUNER_PIPELINE_PARALLEL_SIZE` / `LLMTUNER_NUM_PP_MICROBATCHES` / `LLMTUNER_CONTEXT_PARALLEL_SIZE` / `LLMTUNER_DETERMINISTIC` / **`LLMTUNER_MAX_NUM_DOCUMENTS`**（多文档 packing）六个旋钮（默认值与原行为一致），让文档里的**模型入口**能直接验证并行组合与可复现性；微批数默认跟随 pp 度数（schedule 不接受微批数少于级数）；CP 旋钮接的是 `context_parallel_kv_allgather_size`（本栈唯一可用策略），`dp_shard=-1` 的"覆盖剩余 world"语义要求四个显式度数相乘等于 world，且 `max_seq_len` 能被 cp 整除；`LLMTUNER_DETERMINISTIC=1` 才让跨进程逐 step 逐位可比（见 §8.1.1） |
| `llmtuner/parallel/compatibility.py`、`accelerator/capabilities.py`、`trainer/runtime/assembly.py` | 新增 capability `fsdp_unshard_context`（探测 FSDP 的 unshard 是否可作上下文管理器）与组合裁决 `chunked_loss_fsdp`；在 `_build_model` 里与既有 `chunked_loss_pp` 守卫并列。取代原先"NPU 报 meta 张量未分配 / gloo SIGSEGV"的难懂失败 |
| `tests/unit_tests/cpu/parallel/test_compatibility.py`、`accelerator/test_capabilities.py` | 同步两条表内不变量所需的清单（新规则 reject 用例、新 capability 名称） |
| `llmtuner/parallel/context_parallel/sdpa_attention.py`（新增）、`tests/integration_tests/cp_sdpa_equivalence.py`（新增）、`tests/unit_tests/cpu/parallel/test_cp_sdpa_kernel.py`（新增） | CP 在 sdpa 后端上的 kernel（含自带 backward 的 `_GatherSequence`）、它的等价性脚本与单元测试 |
| `llmtuner/models/hf/attention_backend.py`、`parallel/context_parallel/context_parallel.py`、`models/hf/model.py` | 新增 `sdpa_cp` 注意力接口并在 `apply_cp` 中按后端选用（非 flex + `kv_allgather`）；`_apply_attention` 在该后端下交空掩码，掩码由 kernel 自己构造。取代原先"CP 必须有 flex"的硬拒绝 |
| `llmtuner/trainer/runtime/reductions.py` | **TP×CP 的真实缺陷修复**：`sequence` mesh 在 cp+tp 同时开启时是 **2 维**，而归约点只给它一个 `all_reduce`——`DeviceMesh.get_group()` 对 ndim>1 直接拒绝（报 `Found the DeviceMesh have 2 dimensions`）。按 `build_device_mesh` 对 `loss` 视图的既有做法，用 `_flatten` 展平成 1 维。该 bug 由本轮 P2 组合矩阵（tp=2×cp=2）暴露 |
| `llmtuner/components/checkpointer/torch_checkpointing_backend.py` | 让移植对上已发布的 `torch_checkpointing 0.1.0`：`default_resharder` 缺失时回退到 `dtensor_resharder.DTensorResharder`；`temp_dir_prefix` 缺失时按可选处理；存储配置**在构造后端 Config 之前**解析并传进去，且关掉 direct I/O（否则后端退回自己的默认真实 I/O 路径） |
| `tests/unit_tests/cpu/components/checkpointer/test_checkpointer.py` | 新增该后端的 checkpoint 往返测试（未装包时跳过，与既有导入守卫测试同一约定） |
| `llmtuner/config/checkpoint_config.py`、`trainer/runtime/assembly.py` | 新增 `checkpoint.backend: "dcp" | "torch_checkpointing"`（默认 `dcp`，配置期校验非法值），装配时按它选 manager。此前该后端**无法从 trainer 选中** |
| `llmtuner/parallel/fully_shard_parallel/fully_shard_parallel.py` | 新增 `shard_experts_on_their_own_mesh`：无 `fsdp_per_param_mesh` 时，用「专家子模块自成稀疏 mesh 的嵌套单元 + 外层忽略」表达逐参数 mesh（`ignored_params` 在子单元 shard 之后再取） |
| `llmtuner/accelerator/capabilities.py` | `fsdp_per_param_mesh` 的 hint 改为「缺失时走嵌套单元回退」 |
| `tests/integration_tests/ep_fsdp_equivalence.py` | 参考侧的专家梯度改为布局无关读取（5.x 融合参数 / 4.x 逐专家模块），与单测同一处理 |
| `tests/unit_tests/cpu/accelerator/test_capabilities.py` | 名称清单加入 `fsdp_per_param_mesh` |
| `tests/integration_tests/harness.py`（新增） | 设备/后端解析：`device()` / `backend()` / `init_distributed()` / `moved()` / `moved_batch()`。默认跟随 `llmtuner.accelerator.device`（本机 = NPU/hccl），`LLMTUNER_TEST_DEVICE=cpu` 给出改动前的 CPU/gloo 行为 |
| `tests/integration_tests/` 下 **24 个脚本**（§7.1/§7.2：12 条并行族/组合 + 10 条 P3 特性 + 2 条 EP 扩展/TP×EP） | 不再自己钉 gloo/CPU：进程组、网格设备、模型与批次放置都走 harness。**判据、容差、参考实现一律未改**——这是让 DoD 点名的等价性脚本能在 NPU 上跑起来的前提 |

## 4. 状态矩阵

| 项 | 状态 | 证据 |
| --- | --- | --- |
| P0 基础训练（8 卡 FSDP，Qwen3-8B） | **通过** | `outputs/qwen3-8b-npu/p0_100steps.log`，exit 0 |
| P0 checkpoint 保存 | **通过** | 末步存盘，`step-100` 目录 92G，耗时 6.61s |
| P0 resume | **通过** | `outputs/qwen3-8b-npu/resume.log`：从 step 100 续训，运行 101–103，存 `step-103` |
| **P1 单并行族：五族全部通过** | **每族的仓库等价性脚本已在 NPU 上实跑**（§7.1）：FSDP/HSDP（`fsdp_mesh_equivalence` 两 mode + `fsdp_checkpoint_equivalence` 6 组）、TP（`tp_equivalence`、`tp_grad_norm_equivalence`）、PP（`pp_equivalence` 1F1B 与 Interleaved1F1B）、CP（`cp_equivalence`、`cp_grad_equivalence`）、EP（`ep_equivalence`、`ep_fsdp_equivalence`），loss/梯度/参数都在容差内一致。另有真机训练：8 卡 FSDP 100 步（P0）、tiny cp=2×dp_shard=4 3 步、**Qwen3-8B 上 cp=2×dp_shard=4、seq=4096、20 步 exit 0**（loss 2.558→1.246，峰值 28.5GiB/卡，每步 8 rank 同一个 loss），末步存盘后**从 step 20 续训到 22**。**CP 的梯度归约另有正证据**（`cp` 折进 FSDP 的 shard 轴，`dp_shard=1` 也要走 `apply_fsdp`）：`cp_grad_equivalence` 四项判据全过 | `outputs/npu_harness/on_npu_*.log`（NPU 实跑：14 条通过 + ZBV 平台门 + `cp_tp_equivalence` 通过）、`npu_cp/cp_grad_npu.log`、`qwen3-8b-npu/{p0_100steps,cp2_real,cp2_resume}.log`、`tests/integration_tests/cp_sdpa_equivalence.py`（2/4 卡四路梯度 ≤1.8e-15）、`cp_equiv.log`（交换层） |
| P2 并行组合 | **已验证 12 组**，全部在 8 卡上 3 步 exit 0 且**每步只有一个 loss 值**（8 个 rank 一致）：两族——TP×FSDP、PP×FSDP、EP×FSDP、TP×EP、PP×EP、TP×CP、PP×CP、CP×FSDP、CP×EP；三族——TP×CP×PP；**四族——TP×CP×EP（dp_shard=2, cp=2, tp=2, ep=2）与 PP×CP×EP**。**组合的等价性脚本亦已实跑**：`tp_fsdp_equivalence`（4 卡）、`pp_cp_equivalence`（4 卡）、`pp_ep_equivalence`（4 卡）、`cp_tp_equivalence`（4 卡，CP×TP）在 NPU 上通过；**`tp_ep_cp_equivalence`（TP×EP×CP）也已通过**——它先暴露了一个静默算错的真实缺陷（TP 折叠了专家的 F 维，§0 第 10 条 / §7.4），修后在 CPU/fp64 与 NPU/fp32 上都 clean；未过的只剩 **`pp_equivalence` 的 ZBVZeroBubble 调度**（平台门，torch 形状推导的 HCCL 对象 p2p，§7.1 有最小复现）。**真实模型亦已复验**：Qwen3-8B + `pp=2` 4 步 exit 0（loss 2.797→1.860）；Qwen3-8B + **`cp=2`×dp_shard=4** 20 步 exit 0（loss 2.558→1.246） | 组合：`npu_p2cp/*.log`、`npu_ep/*.log`、`npu_probe_dp8/*.log`、`outputs/npu_harness/on_npu_{tp_fsdp,pp_cp,pp_ep,pp_equivalence_*}_*.log`；真实模型：`qwen3-8b-npu/{pp2_real,cp2_real}.log` |
| P3 其他特性 | **通过项**（矩阵见 §6.4；**另有 10 条特性的等价性脚本在 NPU 上补跑通过**，见 §7.1）：activation checkpoint full/selective（`ac_equivalence`）、deterministic、profiling、EMA（`ep_checkpoint_ac_equivalence`）、validation（`pp_validation_equivalence`，与参考逐位相同）、chunked loss（单独；×PP 的拒绝路径由 `chunked_loss_pp_rejection` 覆盖）、vocab-parallel loss（`vocab_parallel_loss_equivalence`）、DCP 续训（`pp_checkpoint_equivalence` 逐位 + `async_checkpoint_lifecycle_equivalence` 异步与组释放）、metrics（`loss_report_rank_consistency`）、packing（**含多文档**：8 卡 Qwen3-8B + `LLMTUNER_MAX_NUM_DOCUMENTS=8` 3 步 exit 0，loss 2.565→1.684，35GiB/卡；以及 `pp_real_corpus_equivalence` 在 NPU 上通过）、**torch_checkpointing 后端往返**（装包后实测）；`chunked loss × FSDP` 已改为明确拒绝（本 torch 不支持）；compile/多模态缺依赖 | `npu_p3/*.log`、`test_checkpointer.py` 的往返用例 |
| P4 与 TorchTitan 精度对齐 | **判据侧已落地并跑通，对照侧仍受阻**：用基线仓库自己的指标实现与阈值给 llmtuner 打分已成立（CP 一项两卡 PASS，见 §8.1）；端到端跑参考实现仍需要 torch 2.14（见 §8、§8.2） | `outputs/npu_p3/p4_metrics_npu.log`、`torchtitan-npu/scripts/numerical_alignment/utils/metrics.py` |

P0 数据：`examples/train_qwen3_8b_npu.sh`，`dp_shard=8`、bf16、full activation
checkpoint、`global_batch_size=8`、`max_seq_len=4096`、`steps=100`。
loss 1.93（首个记录步 step 3）→ 0.99（step 100），grad_norm 有限，
显存稳定在 ~28 GiB/卡，`tps ≈ 2.1k`。2 步冒烟里 step 1 loss 2.669、step 2 2.502。

EP（P1 最后一个可用族）的 NPU 证据：`outputs/npu_ep/ep2_npu10.log`，8 卡
`ep=2 × dp_shard=8`、tiny qwen3_moe、10 步，exit 0，loss 6.9595→6.9651、
grad_norm 有限。CPU 侧的 EP 等价性脚本也已全绿：`ep_fsdp_equivalence`、
`ep_equivalence`、`ep_grad_norm_equivalence`、`ep_wiring_equivalence`。

CP 在真实模型上的数据（`outputs/qwen3-8b-npu/cp2_real.log` + `cp2_resume.log`）：
Qwen3-8B、`dp_shard=4 × cp=2`、`global_batch_size=8`、`max_seq_len=4096`、
bf16、full activation checkpoint。网格自报
`dp_shard=4, cp=2`、active dims `['batch','loss','cp','cp_kv','dp','dp_shard']`；
20 步 exit 0，loss 2.558（step 1）→ 1.246（step 20），grad_norm 有限，
峰值显存 28.50GiB(46.5%)/卡、`tps ≈ 1.85k`；每个 step 8 个 rank 只报**一个**
loss 值（`sort -u` 后 1 条）。末步 `Saving a full checkpoint at last step, step 20`
（6.1G×8 分片），随后同一命令 `LLMTUNER_STEPS=22` 复跑打印
`Resuming from step 20`，跑 21–22 并存 `step-22`。注意 CP 只改**序列**的切分，
数据分片仍按 `batch`(=dp_replicate×dp_shard) 轴：所以 cp=2 与纯 FSDP 的
每步数据不是同一批，**两者的 loss 曲线不能直接对比**（CP 的正确性由 §7.1 的
梯度归约等价性脚本与 kernel 等价性脚本给出，而不是靠曲线对齐）。

数据集：goal.md 写的是 `/home/jianzhnie/llmtuner/hfhub/datasets/tatsu-lab`，那是
目录（alpaca parquet，52002 行）。示例的 `local_jsonl_sft` 配方要 jsonl 文件，
因此用 `/home/jianzhnie/llmtuner/tmp/convert_alpaca.py` 转出
`/home/jianzhnie/llmtuner/tmp/alpaca_sft.jsonl`（51974 行，`problem`/`solution`
两个字段）后运行。**这条数据准备路径尚未并入仓库**，需要在 P0 收尾时定稿。

## 5. 已验证的版本级缺口

| 缺口 | 证据 |
| --- | --- |
| `spmd_types`（0.1.0–0.3.0 全部）在 torch 2.9 上不可导入：模块级 import `torch.distributed._local_tensor`，该模块 torch 2.12 才有 | 逐版本实测 `ModuleNotFoundError` |
| `DeviceMesh._unflatten` 缺失（torch 2.12 私有 API） | 8 卡运行的首个 traceback，已修复 |
| `transformers.activations.SiLUActivation` 不存在（5.x 才有） | `hf_inspection.py` 的 MoE 探测在 4.56 上报 ImportError |
| `AUTO` 模型注册表无 `qwen3_next`（4.56 有 `qwen3`/`qwen3_moe`，无 `qwen3_next`） | `test_flops.py` 用例 |
| MoE 的 `experts_implementation='eager'` 在 4.56 的 HF 模型类上不可设置 | `llmtuner/models/hf/model_setup.py` 主动 ValueError |
| `torch_npu` 的 inductor 编译后端 | `BackendCompilerFailed`（compile 路径） |
| `torch.distributed.tensor.experimental._context_parallel`（含 `_context_parallel_shard`）整个模块在 torch 2.9 不存在；TP/CP 的输入分片依赖它 | 8 卡 `--tensor_parallel_size 2` 运行：`input_sharding.py` 的 `require_torch_input_sharding()` 主动 ImportError（报错文案清晰） |
| **本栈没有 fp64**：`Tensor.double()` / `.to(torch.float64)` 在 NPU 上**静默降成 fp32**（torch_npu 打 `Warning: Device do not support double dtype now, dtype cast replace with float`），而 fp64 的 matmul 直接报 `AclNN_Parameter_Error(EZ1001): Tensor matA not implemented for DT_DOUBLE, should be in dtype support list [DT_FLOAT,DT_FLOAT16,DT_BFLOAT16]`（`Unsupported data types for Cube`）。凡是**以 fp64 建立容差**的等价性脚本（`cp_tp_equivalence`、`tp_ep_cp_equivalence`，以及 `ep_equivalence` 里那两个 fp64 诊断量）在 NPU 上要么精度悄悄降级、要么直接不可用 | `tmp/fp64_probe.py`（128×128 fp64 matmul：CPU fp64 参考 vs 设备结果，设备端直接抛错）；`ep_equivalence` 的 NPU 日志里那两条 fp64 诊断在 NPU 上被降级（脚本里已注明） |
| **`torch_checkpointing` 后端的移植对不上已发布的包**（模块自述"从未运行过"）。装上 PyPI 的 `torch_checkpointing 0.1.0` 后连不上，逐项实测：<br>① 移植 import `torch_checkpointing.default_resharder.DefaultResharder`，而 0.1.0 没有该模块（它有 `resharding`（`Resharder` 是 ABC）与 `dtensor_resharder.DTensorResharder`）；<br>② 移植读 `writer_config.temp_dir_prefix`，0.1.0 的 `CheckpointWriterConfig` 只有 `checkpoint_write_barrier_timeout_sec/barrier_config/file_write_max_threads`；<br>③ 移植把默认存储配置写成了 `None` 传给后端 Config，于是后端用**自己的**默认（`use_direct_io=True`）；本环境的 overlay 挂载会**接受 `O_DIRECT` 的 open 却让写/关失败**（`OSError: [Errno 22]`），整次存盘挂掉。移植在 HF 合并那条路径上已经写了 `use_direct_io=False`，主路径没写——是内部不一致。<br>三处都改为版本自适应/一致化后，**checkpoint 往返测试通过** | `tests/unit_tests/cpu/components/checkpointer/test_checkpointer.py::test_the_torch_checkpointing_backend_round_trips_a_checkpoint`；包内字段实测见上 |
| **NPU 上仍有一处上游硬编码阻塞**：`torch_checkpointing/metadata_manager.py:413` 写死 `torch.tensor([total_size], dtype=torch.int64, device="cuda")`。单进程（不跑元数据聚合）能过，8 卡分布式一旦进入该路径就 `AssertionError: Torch not compiled with CUDA enabled`。全仓 grep 该包只有这**一处** `device="cuda"`——所以这是"改一行就能通"的上游问题，不是移植能绕的（那个函数在包内部，llmtuner 无法在其间插话）。结论：**NPU 上请继续用 DCP 后端**；`torch_checkpointing` 的适配在 CUDA 机器上已验证 | 8 卡运行：`outputs/npu_ckpt/tc_run.log` |
| **TP 序列并行 + sdpa 的 attention mask 长度不一致**（设计层面，非 torch 版本）：`GatherSequenceFirst` 在 attention 内部把 `[B, T/tp, K]` gather 成 `[B, T, K]`，但 HF 按 gather *前* 的输入长度 `T/tp` 造 causal mask，于是 q 长 T、mask 长 T/tp | CPU/gloo 复现（`TORCH_COMPILE_DISABLE=1`）：`RuntimeError: The size of tensor a (16) must match the size of tensor b (8) at non-singleton dimension 3`（`tp_equiv2.log`）。NPU 上表现为 `aclnnFlashAttentionScore failed, error code 161001`（`tp2.log`） |
| 上面的直接原因：`models/hf/attention_backend.py` 只在 CUDA 上启用 `flex_toritan`，非 CUDA 一律 `sdpa`；而 `_apply_attention` 在 sdpa 单因果文档时返回 `attention_mask=None`，把 mask 交给 HF 自己造 | `attention_backend.py:16`、`models/hf/model.py` 的 `_apply_attention` |
| PP 的原生微批驱动需要 torch 2.10+（2.9 的 `eval` 没有 `arg_mbs`、`_step_microbatches` 没有 `return_outputs`）——**已由 `microbatch_drivers.py` 绕过**，NPU 上 PP 可跑 | 签名实测：`eval(self, *args, target=None, losses=None, **kwargs)`、`_step_microbatches(self, arg_mbs=None, kwarg_mbs=None, target_mbs=None, losses=None)`；绕过后的运行见 `pp2d.log` |
| **torch 2.9 + gloo 的 PP 初始化在这台机器上不可用**：`_initialize_stage` 的首个 P2P 报 `op.preamble.length <= op.nbytes. 8 vs 4 ... Is there a distributed collective mismatch in your code?`。用**仅含 torch 原生 API** 的最小探针（`/home/jianzhnie/llmtuner/tmp/pp_gloo_probe.py`）可复现，故与 llmtuner 无关 | `pp_equiv.log`、`pp_equiv_npu.log`（NPU 侧该脚本需要 CPU 张量，跑不了）。结论：PP 的数值等价性验证只能放到 NPU/HCCL 上做 |
| PP 需要 `num_pp_microbatches >= 流水级数`（默认 1，pp=2 时必须显式设 2） | 不带该参数的 pp=2 运行报 `Number of microbatches (1) must be greater than or equal to the number of stages (2)` |
| **EP × FSDP 在 torch 2.9 上不可用**：一个 `fully_shard` 单元要让专家参数落在稀疏 mesh、其余落稠密 mesh，torch ≥2.12 用带 `mesh_info` 的 `ShardPlacementResult` 表达；2.9 的 `shard_placement_fn` 只返回 `Optional[Shard]`，没有 mesh 槽位 | 新增 capability `fsdp_per_param_mesh` 并**明确拒绝**（`EnvironmentUnsupportedError`，含解锁提示）：`outputs/npu_ep/ep_fsdp_cpu6.log`；NPU 侧 8 卡 `ep=2, dp_shard=8` 先撞到同一处，见 `ep2b.log` |
| ~~上述的 2.9 替代方案不成立~~ **已更正**：嵌套方案本身是对的，当时判它失败是因为被另一个 bug 掩盖（见下一条）。现已作为正式实现保留，EP×FSDP 在 CPU 上等价性通过 | `outputs/npu_ep/ep_clean.log`（`loss: ep+fsdp=311.895508 ref=311.895531`、`failed ranks = 0`、`all checks passed`） |
| **真正的根因（本轮找到并修复）**：EP 的 dispatch/combine 用的是 `torch.distributed._functional_collectives.all_to_all_single`，而**这个函数在 torch 2.9 上不支持 autograd**（torch 自己的警告：`an autograd kernel was not registered … may lead to silently incorrect behavior`）。梯度根本没穿过 all-to-all，所以专家权重拿不到任何梯度——实测在该参数上挂 hook，**触发 0 次**；router（在 dispatch 之前）却正常。这与 FSDP 嵌套无关，是 EP 训练本身的静默错误 | 修复前的 `outputs/npu_ep/ep_dbg4.log`（`DEBUG w1 hook hits=0` + 上述警告）；torch 源码里 `all_to_all_single`（不支持 autograd）与 `all_to_all_single_autograd`（其 backward 就是反向交换）是两个入口 |
| 另需注意 FSDP2 在 2.9 上会把被 shard 的参数**换成新对象**（类型变 `DTensor`），所以嵌套方案里 `ignored_params` 必须在**子单元 shard 之后**再取，否则按身份匹配不到 | `outputs/npu_ep/ep_dbg.log`（`expert params same objects after child shard=False`） |
| **TP × EP 的探针冲突（已修）**：装配顺序是 TP → EP，TP 会把专家里的投影换成 llmtuner 的并行 linear；而逐专家读取器原先用 `isinstance(nn.Linear)` 判据，于是 TP 之后 `is_hf_moe_block` 认不出 MoE，EP 报 "no HF MoE block found"。改为按 **2-D weight** 判形状（与本模块"读形状不读类名"的设计一致）后 TP×EP 通过。仓库自带的 `tp_ep_cp_equivalence.py` 也复现了同一失败，修后该脚本已越过置换、停在 CP（预期） | 修复前 `outputs/npu_ep/tp2_ep2.log`；修复后 `tp2_ep2b.log`、`tpepc2.log` |
| **PP 的 loss 归约产出垃圾值——已定位并修复**（影响所有 pp>1 且进程默认 dtype 为 bf16 的运行，包括 `examples/train_qwen3_8b_npu.py`）<br>症状：非末级 rank 的 `loss_sum` 归约后变成 `-0.00378`（`{0.0, 6.96038}` 求和不可能为负），`local_max` 变成 `-3.13e27`（`max` 不可能得到这个数）；末级 rank 的值原样不变。<br>根因：pp 归约组里两半张量的**形状与 dtype 不一致**——非末级是 `_pp_loss_sentinel = torch.zeros((1,), device=…)`，即 `(1,)` 且 dtype 跟随**进程默认 dtype**；末级是 `sum(micro_losses) * global_valid_tokens`（float32 × int64）= **0 维 float32**。集合通信按字节长度匹配缓冲：bf16 下 `(1,)` 是 2 字节、末级是 4 字节 → 读写错位。fp32 默认 dtype 下两者恰好都是 4 字节，所以此前（dense CLI 跑）看起来正常，是**碰巧**。<br>修法：哨兵改为 `torch.zeros((), dtype=torch.float32, device=…)`，与末级产物形状/dtype 完全一致。<br>验证方式：先二分（`ep=1` + `pp=2` 仍损坏 ⇒ 与 EP 无关，是 PP+MoE 共性），再用形状打印确认（修复前 `(1,)`/bf16 vs `()`/float32，修复后两侧都是 `()`），最后复跑 | 修复前 `outputs/npu_ep/pp_dbg2.log`（`DEBUGPP2 … -0.00378 / -3.13e27`）、`pp_shape.log`；修复后 `pp_fix1.log`、`pp2_ep2_fixed.log` |
| **`chunked loss × FSDP` 在本 torch 上不可用——已改为明确拒绝**。<br>症状：NPU 8 卡 FSDP 下反向报 `RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet`（位置 `components/loss.py:377` 的 `torch.autograd.backward(hidden_states, …)`）；**同一配置在 CPU/gloo 上直接 SIGSEGV** ⇒ 与 NPU 无关。<br>二分（全部在 NPU 实测）：`num_chunks=1` + FSDP **通过**；`num_chunks=2` + FSDP 失败；`num_chunks=2` + FSDP + `fsdp_reshard_after_forward=never` **仍失败**（所以不是 reshard 那个钩子）；`num_chunks=2` **去掉 FSDP** **通过** ⇒ 触发条件是「多个 chunk 各自反向 + 分片的 lm_head」。<br>为什么本轮不修：期望的修法是"在 chunk 循环期间保持 lm_head 未分片"，而该 API 在本版本不存在——torch 2.9 的 `FSDPModule.unshard()` 返回的 `UnhandlHandle` **只有 `wait()`，没有上下文管理器**（实测 `_UnshardHandleImpl` 无 `__enter__/__exit__`）。<br>落地：新增 capability `fsdp_unshard_context`（探测该上下文管理器是否存在）与组合裁决 `chunked_loss_fsdp`，在 `_build_model` 里与既有 `chunked_loss_pp` 守卫并列；2.9 上该组合现在给出可操作的 `UnsupportedCombinationError`，而在有该能力的 torch 上不会被拦。<br>与本次改动无关：`components/loss.py` 与 trainer 的 chunked 分支都不在本次修改范围；CPU 单测 `test_trainer_chunked_loss.py` 覆盖的是**不带 FSDP** 的情形，所以此前没被拦住 | 失败：`outputs/npu_p3/chunked_loss.log`、`chunked_cpu.log`、`ck2_fsdp`/`ck2_never`；通过：`chunked_nofsdp.log`、`ck1_guarded.log`、`ck2_nofsdp2.log`；拒绝：`ck2_guard2`（`UnsupportedCombinationError: chunked_loss_num_chunks=2 with FSDP …`） |
| CP 设计上要求 `flex_torchtitan` 后端（BlockMask），而该后端只在 CUDA 上启用，NPU 只能是 `sdpa` | 8 卡 `--context_parallel_kv_allgather_size 2` 抛 RuntimeError，文案明确（`cp2.log`） |
| **CP 在本栈上的可行性评估（本轮实测）**：① **重分发层是完好的**——仓库自带 `cp_equivalence.py`（2 卡 CPU/gloo）**通过**：KV all-gather 0.000e+00、Ulysses 0.000e+00、Ulysses 往返 0.000e+00；② 缺的只有收尾那一步：flex 要 BlockMask、sdpa 要布尔张量，且 `CPFlexKernel` 的 kv 路径还依赖私有 `flex_cp_allgather`（2.9 无）；`ulysses` 只用到 `all_to_all_single`，不依赖该私有算子 | `outputs/npu_p3/cp_equiv.log` |
| **CP 在 sdpa 后端上已实现、接线并端到端通过**（`parallel/context_parallel/sdpa_attention.py`）：gather KV + 本 rank 查询 + 带偏移因果掩码 + sdpa，KV 用自带 backward 的 `_GatherSequence`。<br>**关键点：gather 的反向必须"先归约整张、再切本 rank 的分块"**。探针里踩过的三种错法各自留下反例：先切后归约 → 数值**符号翻转**（seq=2 手算例：`-0.1738` vs 参考 `+0.1738`）；叠加可微组内 loss 归约 → 量级**翻倍**（`6.4696` = 2× `3.2348`）；用就地 `all_reduce` 造组内 loss → 不进计算图，无效。<br>**接线**：`apply_cp` 在后端非 flex 且策略为 `kv_allgather` 时挂 `_titan_cp_sdpa_kernel` 并把 `_attn_implementation` 切到新注册的 `sdpa_cp` 接口；`_apply_attention` 在该后端下直接交空掩码（掩码由 kernel 从 CP 偏移自行构造）。ulysses / 双轴形式在 sdpa 下仍明确拒绝（它们的掩码没有偏移公式）。<br>**验证**：① 等价性（2/4 卡 CPU）：前向/q/k/v 四路 ≤1.8e-15，GQA 与 4 卡均覆盖，`failed ranks = 0`；② 单元级：新增 `tests/unit_tests/cpu/parallel/test_cp_sdpa_kernel.py` 5 例（掩码规则、packed 拒绝、构造不需进程组、接线落位、ulysses 拒绝）；③ **端到端 8 卡 NPU**：`cp=2 × dp_shard=4` 3 步 exit 0，日志 `Applied CP (kv_allgather) with degree 2` | `npu_cp/cp2_npu.log`、`tests/integration_tests/cp_sdpa_equivalence.py`、`tests/unit_tests/cpu/parallel/test_cp_sdpa_kernel.py` |

## 6. CPU 单测覆盖（本环境）

### 6.1 transformers 4.56 适配进度（EP 相关）

EP 的那批测试被 `require_env("spmd_types")` 门控，本环境会整体跳过。为度量进度，
用了一个**仓库外**的临时 stub（`/home/jianzhnie/tmp/spmd_stub`，仅验证用，不是交付物），
把 `test_ep_model_replacement.py` 从门控下解锁：

| 阶段 | 该文件结果 |
| --- | --- |
| 起点（4.56 适配开始前） | 39 failed / 7 passed |
| 修掉 `experts_implementation` 门控 + `SiLUActivation` 导入 | 37 failed |
| 再加逐专家 `ModuleList` → 融合张量读取 + 块属性（`block_sparse_moe` / `moe`）+ `moe_block_of` 改为优先选「确实是 MoE」的属性 | 19 failed / 23 passed |
| 再加返回元数的静态适配（见下） | **19 failed / 23 passed**（不再有前向解包失败） |
| 再把测试侧 5.x 布局假设抽成布局无关 helper（`_hf_block` / `_hf_experts` / `_hf_gate_up_parameters` / `_misshape_hf_experts`） | **0 failed / 42 passed** |

最后一行需注明两点：`42 passed` 是在 `TORCH_COMPILE_DISABLE=1` 下取得的；不加该开关时另有
14 个失败，全部是 `BackendCompilerFailed: backend='inductor' → No module named 'triton'`，
属容器环境缺 triton，与代码无关。

同批验证：`test_hf_wrapper.py` + `test_ep_model_replacement.py` 一起跑（同样的两个开关）
**104 passed / 0 failed**。

**一个教训（已踩到）**：这批 EP/模型测试在无 stub 时被 `require_env` 整体跳过，所以
「整仓 CPU 套件 7 failed 不回归」**覆盖不到它们**。本轮就因为改了
`_apply_attention` 的签名，把 `test_hf_wrapper` 里 5 个直接调用它的用例打挂，而全量
套件当时显示无回归；是带 stub 复跑才发现。凡改到被门控的模块，必须用 stub 复跑对应文件。

剩余项的构成（已逐个定位，均已在后续收口）：

- 多数是**测试侧**的 5.x 布局假设：测试直接读 `layer.mlp`（4.x Mixtral 在
  `block_sparse_moe` 上）和 `experts.gate_up_proj`（4.x 是逐专家 ModuleList）。
  生产侧置换了：日志显示 `Swapped 2 HF MoE blocks`，且装配位置就是 HF 上调用的属性。
- 生产侧的返回值问题**已解决**（见下），原先换入 Mixtral 会在前向时报
  `ValueError: not enough values to unpack (expected 2, got 1)`
  （`models/mixtral/modeling_mixtral.py:339`），现已不再复现。

宿主层调用形式（实测 4.56 源码）：

| family | 4.56 宿主写的是 | 能否接受裸张量 |
| --- | --- | --- |
| qwen3_moe | `hidden_states = self.mlp(...)`，随后 `if isinstance(hidden_states, tuple): hidden_states, _ = ...` | 可以（两种都行） |
| qwen2_moe | 同 qwen3_moe | 可以 |
| **mixtral** | `hidden_states, _ = self.block_sparse_moe(...)` | **不行** |
| **olmoe** | `hidden_states, router_logits = self.mlp(...)` | **不行** |
| **granitemoe** | `hidden_states, router_logits = self.block_sparse_moe(...)` | **不行** |
| deepseek_v2 / v3 / glm4_moe | `hidden_states = self.moe(...)`（属性名就是 `moe`） | 可以 |

注意属性名与是否需要二元组**不相关**（`mlp` 上两种都有），所以判据不能是属性名或类名。

**返回元数**：已解决，判据是「保留被替换块自身的返回元数」。理由是构造性的——
HF 的宿主层是照着 HF 自己的块写的，复现它的返回形状就必然满足宿主。实现为
`hf_inspection.returns_router_logits()`：**静态解析**该块 `forward` 的 AST
（`return a, b` 与 `return (a, b)` 都是 `ast.Tuple`），不执行任何前向。

为什么不用运行时探测：置换时跑一次单 token 前向会要求给定 dtype/device，而对刻意混合
dtype 的模型会直接炸——`test_conversion_preserves_shared_expert_dtype_and_child_modes`
把 float32 的 shared expert 挂在 float64 模型上，探针触发
`RuntimeError: self and mat2 must have the same type`，该文件失败数从 19 涨到 22。
该方案已回退（同类判据的完整表格见上一节）。

同时：若某族块的 `forward` 返回二元组而配置开了 `output_router_logits`，置换会**明确拒绝**
（llmtuner 的 router 分数不外传，只能给 `None`），而不是回一个会让 HF 后续算出错值的东西。

### 6.2 4.56 上 EP 置换的分族实测

`/home/jianzhnie/llmtuner/tmp/ep_family_sweep.py`（仓库外脚本）逐族做「建模型 → 置换 →
前向」，结果：

| family | 结果 |
| --- | --- |
| mixtral | ✅ 换 2 块，`tuple_return=True`，前向有限 |
| qwen3_moe | ✅ `tuple_return=True` |
| olmoe | ✅ `tuple_return=True` |
| deepseek_v2 | ✅ `tuple_return=False` |
| deepseek_v3 | ✅ `tuple_return=False` |
| glm4_moe | ✅ `tuple_return=False` |
| qwen2_moe | 按设计拒绝（其 `shared_expert_gate` 是乘法门，`MoE.shared_experts` 只做加法）——既有行为，非回归 |
| granitemoe | 未识别：4.56 的 `GraniteMoeMoE` 根本没有 `experts` 属性，只有 `input_linear`/`output_linear`，属另一种布局，需要专门的读取器 |

`python -m pytest tests/unit_tests/cpu`（排除 `test_multimodal_data.py`，因为
容器没装 `torchvision`）。两种配置都**零失败**，差别只在被测模块的门控：

| 配置 | 命令要点 | 结果 |
| --- | --- | --- |
| 默认（不带 stub） | `PYTHONPATH=. TORCH_COMPILE_DISABLE=1 pytest -q --disable-warnings --ignore=…/test_multimodal_data.py tests/unit_tests/cpu` | **1599 passed，33 skipped，0 failed** |
| 带 `spmd_stub`（解锁被 `require_env("spmd_types")` 门控的模块） | `PYTHONPATH=.:/home/jianzhnie/llmtuner/tmp/spmd_stub TORCH_COMPILE_DISABLE=1 …` | **2521 passed，11 skipped，0 failed** |

两处环境性缺口都在数字里说清楚：

- **不设 `TORCH_COMPILE_DISABLE=1`** 时，`test_model_compile.py` 会报
  `BackendCompilerFailed: backend='inductor' → No module named 'triton'`：容器没装
  triton，torch_npu 的 inductor 注册会 import 它。这是环境缺口而非代码缺口
  （compile 在 llmtuner 里是**可选开启**的，默认训练不走它）。要让 P3 的「模型
  compile」可验证，需要在该容器里装上可用版本的 triton。
- 带 stub 时多出的 ~920 个用例来自约 20 个模块级 `require_env("spmd_types")` 的测试
  文件（见本节末尾说明）。

附带说明（本轮补测）：带 stub 的配置第一次跑出过 6 个失败，逐条查清了归属——
**5 个在改动前的 HEAD 上同样失败**（用 `git worktree add <tmp>/tlm_head HEAD` 拿干净树对照：
2 个是 torch 2.9 缺私有 `flex_cp_allgather`，3 个走的是需要逐参数 mesh 的路径），
**1 个是本轮改动改了契约的旧断言**。处理见 §6.3。

### 6.3 已修的测试侧环境假设

这几例失败都不是产品缺陷，而是测试把「本环境」写成了常量；修的是测试，让它按设备/版本
自己判断：

| 用例 | 原假设 | 现判据 |
| --- | --- | --- |
| `test_profiler` 内存历史两例 | `torch.cuda.is_available()` 为假就不能记录 | `memory_module() is not None`——NPU 上 `torch_npu.npu.memory` 确实带 `_record_memory_history`，生产代码本来就是按设备模块判断的 |
| `test_flops` 的 MoE 计数 | 传 `num_local_experts`（5.x 的统一拼写） | 两种拼写都传同一个值：4.56 的 `Qwen3MoeConfig` 真实字段是 `num_experts`（默认 128），估算器优先读它是对的 |
| `test_flops` 的线性注意力 | 用 `qwen3_next` 建配置 | 4.56 注册表里没有该族；改为在 dense 族上手设 `layer_types=["full_attention","linear_attention"]`，被测的规则（无法推导的层型 ⇒ 抑制计数）不变 |
| `test_data_pipeline` 未知语料 | 必须是 `ValueError("unknown dataset")` | 该环境没有多模态依赖，多模态注册表导入失败会先抛 `ImportError`；两者都在 build 期拒绝且都点名了那个名字，故接受二者 |
| `test_cp.py::test_apply_cp_requires_the_flex_backend` | 「CP 必须有 flex 后端」——本轮实现了 sdpa 版 CP kernel，该契约已被替换 | 改名为 `test_apply_cp_rejects_ulysses_without_flex`，钉住**新**契约：非 flex 后端下只有 `kv_allgather` 能过，`ulysses`/双轴必须报 `only with the 'kv_allgather' strategy`（原来的错误文案已不存在，所以这条例子的断言是**必须改**的，不是"环境问题") |
| `test_cp.py` 里构造 `CPFlexKernel` 的 2 个用例 | 假设 `kv_allgather` 路径在任何 torch 上都可构造 | 该路径要 import torch 私有的 `flex_cp_allgather`；新增 `tests/caps.py` 的 `flex_cp` 能力（探私有算子）并门控这 2 例（`ulysses` 那支不需要私有算子，仍全平台跑；`ParallelConfig` 默认值那半拆成独立用例保持覆盖） |
| `test_efsdp_placement.py` 的 3 个专家放置用例 | 假设任何 torch 上都会构造 `shard_placement_fn` | 该回调只在有逐参数 mesh 的 torch（≥2.12，capability `fsdp_per_param_mesh`）上存在；2.9 走嵌套单元回退、不建回调，故按运行时能力门控（skip 理由写明）。**注意这是本机单测的一处覆盖缺口**：嵌套回退路径在 torch 2.9 上只有 8 卡 EP×FSDP 端到端跑覆盖，没有单测 |

需要特别注意：约 20 个测试模块在模块级调用 `require_env("spmd_types")`，因此在本
环境整体跳过（约 900 个用例）。`spmd_backend.py` 让**引擎**能在 torch 2.9 上运行，
但不会、也不应该解除这些测试的环境门控——它们验证的正是 spmd_types 的语义。

### 6.4 P3 特性矩阵（8 卡 NPU，dense tiny 模型，每项 3 步）

| 特性 | 开关 | 结果 |
| --- | --- | --- |
| activation checkpoint full | 示例默认 | ✅ P0 100 步已用 |
| activation checkpoint selective | `--activation_checkpoint_mode selective` | ✅ exit 0，loss 4.86085 |
| deterministic 模式 | `--deterministic True` | ✅ exit 0 |
| profiling | `--enable_profiling True` | ✅ exit 0 |
| EMA | `ema_config=EMAConfig(decay=0.999)`（**CLI 无此旗标**，见 `tmp/p3_ema_validation.py`） | ✅ exit 0 |
| validation 循环 | `validation_config=ValidationConfig(freq=1, steps=1)`（同上，CLI 无） | ✅ exit 0，且日志出现 `validate step: 1 loss: 4.8599` |
| chunked loss（单独） | `--chunked_loss_num_chunks 2`，无 FSDP | ✅ exit 0 |
| **chunked loss × FSDP** | 同上 + `--data_parallel_shard_size` | ⛔ 本 torch 不支持，已改为**明确拒绝**（见 §5）；`num_chunks=1` + FSDP ✅ |
| DCP 存盘 + 续训 | 示例默认 | ✅ P0 |
| packing `first_fit` | 示例默认 | ✅ P0 |
| packing 多文档（`max_num_documents>1`，block-causal 掩码） | `LLMTUNER_MAX_NUM_DOCUMENTS=8` | ✅ 8 卡 3 步 exit 0（`outputs/qwen3-8b-npu/packed_multi_doc.log`）；掩码改 dense 之前这里是**硬拒绝**（§0 第 11 条） |
| 模型 compile | 需 triton | ⛔ 环境缺 triton |
| torch_checkpointing 后端 | 装包后实测 | ⚠️ **单进程往返通过**（`test_the_torch_checkpointing_backend_round_trips_a_checkpoint`），trainer 也**已可选择**它（`--backend torch_checkpointing`）；但 **8 卡 NPU 上跑不起来**——卡在第三方包自己的一行硬编码 CUDA（见 §5），与移植无关 |
| 多模态管线 | 需 torchvision | ⛔ 环境缺 torchvision |

## 7. 复现命令

```bash
# P0：8 卡 FSDP 训练（100 步）
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  ASCEND_GLOBAL_LOG_LEVEL=3 TASK_QUEUE_ENABLE=0 \
  HCCL_NPU_SOCKET_PORT_RANGE=61000-61031 \
  LLMTUNER_DATASET_PATH=/home/jianzhnie/llmtuner/tmp/alpaca_sft.jsonl \
  LLMTUNER_STEPS=100 \
  bash examples/train_qwen3_8b_npu.sh'

# resume：同一命令把 LLMTUNER_STEPS 调大即可（folder 里已有 checkpoint 会续训）
```

```bash
# P1/P2：单族与组合（dense tiny 模型，改 --tensor_parallel_size /
# --pipeline_parallel_size / --expert_parallel_size；dp_shard * tp * pp = 8）
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  ASCEND_GLOBAL_LOG_LEVEL=3 TASK_QUEUE_ENABLE=0 \
  HCCL_NPU_SOCKET_PORT_RANGE=61000-61031 \
  torchrun --nproc_per_node=8 --master_port=29500 -m llmtuner \
    --steps 10 --tensor_parallel_size 2 --data_parallel_shard_size 4'

# EP（`--num_experts` 只能在程序化配置里给；8 卡下 dp_shard=8、ep=2）
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  MOE_EP=2 MOE_DP_SHARD=8 MOE_STEPS=10 \
  PYTHONPATH=/home/jianzhnie/llmtuner/llm/TorchLLMTuner:/home/jianzhnie/llmtuner/tmp \
  torchrun --nproc_per_node=8 --master_port=29500 \
    /home/jianzhnie/llmtuner/tmp/train_tiny_moe_npu.py'

# P2 真实模型上的 pp=2（bf16 默认 dtype，正是 PP 哨兵 bug 的触发条件）
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  LLMTUNER_DATASET_PATH=/home/jianzhnie/llmtuner/tmp/alpaca_sft.jsonl \
  LLMTUNER_PIPELINE_PARALLEL_SIZE=2 LLMTUNER_STEPS=4 \
  bash examples/train_qwen3_8b_npu.sh'

# P1/P2 真实模型上的 cp=2（dp_shard 自动取剩余 4；旋钮说明见示例 docstring）
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  ASCEND_GLOBAL_LOG_LEVEL=3 TASK_QUEUE_ENABLE=0 \
  HCCL_NPU_SOCKET_PORT_RANGE=61000-61031 \
  LLMTUNER_DATASET_PATH=/home/jianzhnie/llmtuner/tmp/alpaca_sft.jsonl \
  LLMTUNER_CONTEXT_PARALLEL_SIZE=2 LLMTUNER_STEPS=20 \
  bash examples/train_qwen3_8b_npu.sh'
# 续训：同一条命令、LLMTUNER_STEPS=22（日志会出现 "Resuming from step 20"）
# 逐 step 逐位可比：再加 LLMTUNER_DETERMINISTIC=1（两次运行 step 1–3 完全相同，见 §8.1.1）

# CPU 等价性（EP 相关，4/2 卡按脚本 docstring；加 TORCH_COMPILE_DISABLE=1，
# 因为容器没装 triton 而 torch_npu 的 inductor 注册会 import 它）
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  PYTHONPATH=.:/home/jianzhnie/llmtuner/tmp/spmd_stub TORCH_COMPILE_DISABLE=1 \
  ASCEND_RT_VISIBLE_DEVICES= \
  torchrun --nproc_per_node=4 tests/integration_tests/ep_fsdp_equivalence.py'

# 质量门禁（仓库文档里的那几条）
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  python -m pytest -q --disable-warnings \
    --ignore=tests/unit_tests/cpu/datasets/test_multimodal_data.py tests/unit_tests/cpu'

# 归因用：改动前的干净树（未提交改动都不在 HEAD 里）
git worktree add /home/jianzhnie/llmtuner/tmp/tlm_head HEAD        # 从宿主机执行
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/tmp/tlm_head &&
  PYTHONPATH=.:/home/jianzhnie/llmtuner/tmp/spmd_stub TORCH_COMPILE_DISABLE=1 \
  python -m pytest -q --disable-warnings tests/unit_tests/cpu/parallel/test_cp.py'
git worktree remove --force /home/jianzhnie/llmtuner/tmp/tlm_head # 容器跑过 pytest 后
                                                                 # 的 __pycache__ 归 root，
                                                                 # 需在容器内 rm -rf 后再 prune

# CP：8 卡端到端（dp_shard * cp * tp * pp = 8）；组合把各度数一起给即可
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  ASCEND_GLOBAL_LOG_LEVEL=3 TASK_QUEUE_ENABLE=0 \
  HCCL_NPU_SOCKET_PORT_RANGE=61000-61031 \
  torchrun --nproc_per_node=8 --master_port=29500 -m llmtuner \
    --steps 3 --context_parallel_kv_allgather_size 2 --data_parallel_shard_size 4'

# CP kernel 的数值等价性（2 或 4 卡 CPU/gloo）
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  PYTHONPATH=. TORCH_COMPILE_DISABLE=1 ASCEND_RT_VISIBLE_DEVICES= \
  torchrun --nproc_per_node=2 tests/integration_tests/cp_sdpa_equivalence.py'

# CP 梯度归约的等价性（仓库脚本的 NPU 端口版：只把设备钉到 NPU，
# 判据与容差原样保留；原脚本在 NPU 可见时跑不了，原因见 §7.1）
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  ASCEND_GLOBAL_LOG_LEVEL=3 TASK_QUEUE_ENABLE=0 \
  HCCL_NPU_SOCKET_PORT_RANGE=61000-61031 \
  PYTHONPATH=.:/home/jianzhnie/llmtuner/tmp/spmd_stub TORCH_COMPILE_DISABLE=1 \
  torchrun --nproc_per_node=2 --master_port=29731 \
    /home/jianzhnie/llmtuner/tmp/cp_grad_npu.py'

# P4 判据落地：用基线仓库自己的 metrics.py + 阈值给 llmtuner 打分（2 卡 NPU，见 §8.1）
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  ASCEND_GLOBAL_LOG_LEVEL=3 TASK_QUEUE_ENABLE=0 \
  HCCL_NPU_SOCKET_PORT_RANGE=61000-61031 \
  PYTHONPATH=.:/home/jianzhnie/llmtuner/tmp/spmd_stub TORCH_COMPILE_DISABLE=1 \
  torchrun --nproc_per_node=2 --master_port=29741 \
    /home/jianzhnie/llmtuner/tmp/p4_alignment_metrics_npu.py'

# §7.1 的那张表：DoD 点名的等价性脚本一次全跑（默认 NPU；CPU 模式用
# tmp/harness_on_cpu.sh，见 §7.3）。单条也可以直接跑，例如
#   torchrun --nproc_per_node=2 tests/integration_tests/tp_equivalence.py
docker exec hyper-parallel-env bash -lc '
  cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner &&
  ASCEND_GLOBAL_LOG_LEVEL=3 TASK_QUEUE_ENABLE=0 \
  HCCL_NPU_SOCKET_PORT_RANGE=61000-61031 \
  PYTHONPATH=.:/home/jianzhnie/llmtuner/tmp/spmd_stub TORCH_COMPILE_DISABLE=1 \
  bash /home/jianzhnie/llmtuner/tmp/harness_on_npu.sh | tee outputs/npu_harness/npu_sweep.log'
```

说明：`tmp/` 下的脚本（`train_tiny_moe_npu.py`、`p3_ema_validation.py`、`convert_alpaca.py`、
`cp_grad_npu.py`、`p4_alignment_metrics_npu.py`、`spmd_stub/` 等）都是**仓库外的临时脚手架**，
用于验证；`spmd_stub` 只是为了解锁被 `require_env("spmd_types")` 门控的测试模块，
`cp_grad_npu.py` 是仓库脚本的设备端口版，都不是交付物。

### 7.1 仓库集成脚本（`tests/integration_tests/`）在 NPU 上的实跑结果

**这些脚本原先按 CPU/gloo 写死**（docstring 里就是 `torchrun` + gloo、`init_device_mesh("cpu")`），
而 llmtuner 自己在 NPU 上把网格建成 npu——于是出现"进程组是 gloo、网格是 npu"这类错配，
DoD 要求的"每族都在 NPU 上跑对应集成脚本"因此无法直接执行。本轮改法：新增
`tests/integration_tests/harness.py`（设备/后端选择，见 §7.2），把脚本里的
`dist.init_process_group("gloo")` / `init_device_mesh("cpu", …)` / 张量与模型放置改为
按设备解析。默认跟随 llmtuner 的选择（本机 = NPU/hccl）；`LLMTUNER_TEST_DEVICE=cpu`
可让 harness 自己回到 CPU/gloo（适用范围与实测见 §7.2、§7.3）。

实跑结果（8×910，`torch 2.9.1 + torch_npu 2.9.1`，日志 `outputs/npu_harness/on_npu_*.log`）：

| 脚本（DoD 点名） | 卡数 | 结果 | 日志里的数字 |
| --- | --- | --- | --- |
| `fsdp_mesh_equivalence`（mode=ddp，HSDP） | 2 | ✅ | 4 步 `dist` 与单卡参考逐个相同（`3.591575/3.596237/3.495913/3.550152`），终态参数 `max abs diff = 0.000e+00`，非平凡性 `5.242e-02/6.447e-02`，`failed ranks = 0` |
| `fsdp_mesh_equivalence`（mode=shard，FSDP） | 2 | ✅ | 同上（step0 `3.591575` vs 参考 `3.591576`） |
| `fsdp_checkpoint_equivalence` | 2 | ✅ | 6 组 `mode→resume_mode/implementation` 全过；同布局组 `resume atol=0`（位级一致），跨布局 `atol=1e-5`。**需要开确定性算法**，见下条 |
| `tp_equivalence` | 2 | ✅ | logits `max abs diff = 1.490e-08`，权重梯度 `1.192e-07`，`failed ranks = 0` |
| `tp_grad_norm_equivalence` | 2 | ✅ | 全部 `assert_close` 通过 |
| `pp_equivalence`（1F1B） | 2 | ✅ | `pp losses` 与 `reference` 四位小数全同（`4.878460/4.867842/4.858004/4.862647`），`failed ranks = 0` |
| `pp_equivalence`（Interleaved1F1B） | 2 | ✅ | 同上（`4.861719/4.869583/4.869904/4.882055`） |
| `pp_equivalence`（ZBVZeroBubble） | 2 | ❌ **平台门** | 见下条（torch 的 shape-inference 走 HCCL 对象 p2p 失败） |
| `cp_equivalence` | 2 | ✅ | KV all-gather `0.000e+00`、Ulysses `0.000e+00`、往返 `0.000e+00`，`failed ranks = 0` |
| `ep_equivalence` | 2 | ✅ | `max abs diff = 0.000e+00`（rel `0.000e+00`），`max rel error across ranks = 0.000e+00` |
| `ep_fsdp_equivalence` | 4 | ✅ | `failed ranks = 0` |
| `tp_fsdp_equivalence` | 4 | ✅ | `failed ranks = 0` |
| `pp_ep_equivalence` | 4 | ✅ | `pp+ep losses` vs `reference` 最大差 1e-6（`4.858150/4.854156/4.878164/4.883010` vs `4.858151/…`），容差 1e-5 |
| `pp_cp_equivalence` | 4 | ✅ | `all checks passed` |
| `cp_grad_equivalence` | 2 | ✅ | 见 §7.2 的移植说明：4 步 loss ≤1e-6、终态参数 `5.960e-08`、非平凡性 `6.229e-02/6.905e-02` |
| `pp_real_corpus_equivalence`（PP×packed 真实语料） | 2 | ✅ | 终态参数 `max abs diff = 2.384e-07`，`all checks passed`（修掉 packed 掩码的硬拒绝之后，§0 第 11 条） |

**P3 特性的等价性脚本（本轮补跑，全部在 NPU 上通过）**——P3 此前只有 8 卡矩阵行，这一批给到"与单卡/单进程参考逐张量一致"级的证据：

| 脚本 | 卡数 | 对应 P3/P2 项 | 日志里的数字 |
| --- | --- | --- | --- |
| `ac_equivalence` | 2 | activation checkpoint | 两种模式终态参数 `max abs diff = 0.000e+00 / 1.490e-08`，`failed ranks = 0` |
| `pp_tp_trainer` | 4 | PP×TP 装配 | `PP+TP trainer checks passed`（loss == max_loss，4 rank 一致） |
| `pp_validation_equivalence` | 2 | PP×validation | `reported losses == reference losses`，`max abs diff = 0.000e+00` |
| `loss_report_rank_consistency` | 2 | metrics | `all checks passed`（各 rank 报的 loss/token 数一致） |
| `chunked_loss_pp_rejection` | 2 | chunked loss × PP | `all checks passed`（明确拒绝路径） |
| `vocab_parallel_loss_equivalence` | 2 | vocab-parallel loss | `all checks passed`（V=8/7/5，含不整除与空分片拒绝） |
| `async_checkpoint_lifecycle_equivalence` | 2 | DCP（异步存盘 + 组释放） | 2 轮 `async resume and group teardown match` |
| `ep_checkpoint_ac_equivalence` | 2 | DCP × AC × EMA × EP | 3 组 `EP DCP restore …: exact` |
| `pp_checkpoint_equivalence`（plain） | 2 | PP × DCP | `uninterrupted == resumed`（逐位：`4.878460/4.867842/4.858004/4.862647`，续训段 `4.858004/4.862647`） |
| `pp_checkpoint_equivalence --packed` | 2 | PP × DCP × packed | 同上逐位（`4.894713/4.546629/4.389603/4.289792`） |

其中 3 条（`vocab_parallel_loss_equivalence`、`ep_checkpoint_ac_equivalence`、`async_checkpoint_lifecycle_equivalence`）在 CPU 模式下也复跑通过，说明端口没有改变它们的 CPU 行为。

**其余 harness 在 NPU 上的补跑（本轮第二批）**——8 条本来就是设备无关写法，原样即可跑；2 条补了端口：

| 脚本 | 卡数 | 覆盖 | 日志里的数字 |
| --- | --- | --- | --- |
| `ep_wiring_equivalence` | 2 | EP 接线（置换/分片） | `failed ranks = 0`，`all checks passed` |
| `ep_grad_norm_equivalence` | 2 | EP 下的 grad-norm 归约 | `all checks passed` |
| `moe_aux_loss_grad_equivalence` | 2 | MoE 辅损梯度 | `failed ranks = 0` |
| `ep_domino_equivalence` | 2 | Domino（zero-receives/recompute） | 每个组合 `forward and gradients match` |
| `tp_replicated_grad_equivalence` | 2 | TP 复制参数梯度归约 | `replicated grad max abs diff = 2.861e-06`；非平凡性 `5.887e+00` |
| `reduce_equivalence` | 2 | 归约语义（5 项） | `all 5 checks passed` |
| `quantile_bias_equivalence` | 4 | MoE 量化平衡统计 | `axes=('dp','tp'), EP=True: counts, bias and reset match` |
| `hf_checkpoint_pp` | 2 | HF 索引式 load/export × PP | `PASS: indexed HF load and export across two CPU pipeline stages` |
| `chunked_loss_equivalence`（`CHUNKED_LOSS_EQ_TP=2`） | 2 | chunked loss × TP | `loss max abs diff = 4.768e-07`，`failed ranks = 0`（默认的 FSDP 变体是本栈的明确拒绝，见 §5） |
| `ep_fsdp_without_dense_dp` | 4 | TP×EP（无 dense DP） | `logits max abs diff = 8.941e-08`、`expert grads max abs diff = 1.397e-09` |

**两条与设备无关的门（本轮定位，不是 NPU 问题）**：

- `shared_expert_tp_equivalence`：`ValueError: Unsupported TP plan entry … 'local_colwise'`。原因在 transformers 侧——
  4.56 的 HF 自带 TP 计划用的是 `colwise_rep` / `local_colwise` 这类**复制激活**风格（实测
  `DeepseekV3ForCausalLM._tp_plan = {'lm_head': 'colwise_rep'}`），而 llmtuner **按设计拒绝**这些风格
  （`test_replicated_activation_styles_are_refused_not_reinterpreted` 逐条钉住，理由是它们保留整段激活、
  不是本引擎的序列并行布局）。换 CUDA/更高版本 transformers 才有可能跑；与 NPU 无关。
- `chunked_loss_equivalence`（默认变体）：命中本栈的 `chunked_loss_num_chunks>1 × FSDP` 明确拒绝（§5），
  其 TP 变体已通过（上表）。

**另有两条在本轮之前跑不了、现在能跑的脚本**（端口让它们换用 sdpa CP kernel 而不是强制 flex）：

| 脚本 | 结果 | 日志 |
| --- | --- | --- |
| `cp_tp_equivalence`（CP×TP） | ✅ 连续两次通过：`tp_join=2.682e-07`、`cp_join=2.682e-07`、平均 loss ≤4.8e-07（容差 1e-5，fp32）；ulysses 臂按平台跳过（flex 仅 CUDA） | `outputs/npu_harness/on_npu_cp_tp_equivalence.log`、`cp_tp_rep{1,2}.log` |
| `tp_ep_cp_equivalence`（TP×EP×CP） | ✅ 先暴露了静默算错的真实缺陷（TP 折叠专家的 F 维，§7.4），修复后 **NPU/fp32 与 CPU/fp64 都 `all checks passed`** | `outputs/npu_harness/on_npu_tp_ep_cp_equivalence.log`、`tp_ep_cp_cpu.log` |

**`pp_real_corpus_equivalence`（PP×真实 packed 语料）原本也在环境门里，本轮修掉了**：packed 掩码原本只有 flex 能表达，非 CUDA 后端直接拒绝（§0 第 11 条）；改用 dense block-causal 掩码后，**NPU/fp32（2 卡，`final parameter max diff = 2.384e-07`）与 CPU 分支都 `all checks passed`**（`outputs/npu_harness/on_npu_pp_real_corpus_equivalence.log`）。端到端另见 §4 的 P3 行（8 卡 Qwen3-8B + `max_num_documents=8` 3 步 exit 0）。

**仍然跑不了的一条（环境门，与本轮改动无关）**：

| 脚本 | 卡在哪 | 最小复现 |
| --- | --- | --- |
| `cp_ulysses_equivalence` | `ImportError: headtail load balancing requires torch's private _HeadTailLoadBalancer` | 原样 `torchrun` 即报（§5 已记录） |

**另一处仍是明确拒绝（不是静默）**：`packed × CP`。sdpa 版 CP kernel 在构造时就拒绝 packed （`CPSdpaKernel(packed=True)` → `NotImplementedError`，`context_parallel.py` 把模型的 `attn_mask_type` 传进去）：它的掩码只从 CP 偏移推出来，没有文档结构那一维。解锁条件是把全长的文档 id 也交给 kernel（现在 `_apply_attention` 在 CP 后端下直接给空掩码）；**packed × 非 CP 已经支持**（§0 第 11 条）。

**ZBVZeroBubble 的失败定位（平台侧，不是 PP 数学）**：失败帧在 **torch 自己的**
`torch/distributed/pipelining/stage.py:1419 _shape_inference` → `dist.recv_object_list(use_batch=True)`，
设备报错是 HCCL 的 AICPU 内核：`RunAicpuRpcSrvLaunchV2_batch_send_recv` … `execute kernel param invalid`
（随后 `AclrtSynchronizeStreamWithTimeout ... 507018`）。同一条 driver、同一套 mesh 下
1F1B 与 Interleaved1F1B 在 NPU 上通过，所以问题不在 llmtuner 的 PP 驱动/数值，而在
torch 与 HCCL 的对象集合通信。最小复现（`tmp/object_p2p_probe.py`，2 卡 hccl）：

| 对象 p2p 形式 | `use_batch=True` | `use_batch=False` |
| --- | --- | --- |
| 单向（先发后收） | ✅ | ✅ |
| 双向同时先收（V 布局需要的形态） | ❌ 挂住（>120s 无返回） | ❌ `RuntimeError: HcclRecv, error code is 4` |

**解锁条件**：torch_npu/HCCL 支持 pipeline 形状推导所需的双向对象 p2p（或 torch 改为
不走 `use_batch=True`）。在此之前，NPU 上请用 1F1B / Interleaved1F1B；ZBV 在 CPU/gloo 上仍然通过。

另外两条被证伪/澄清的旧说法：

- **"把 NPU 藏起来走 CPU"这条捷径站不住**：`ASCEND_RT_VISIBLE_DEVICES=` 下 `device_type` 确实退成
  `cpu`，但训练一进 `optim.step()` 就被 torch_npu 自己给 `torch.optim` 打的补丁拉回设备——
  `torch_npu/utils/_optim.py:17` 的 `patch_supported_devices` 调 `torch_npu.npu.current_device()`
  → `aclInit ... Invalid device ID`（与本仓库代码无关；`outputs/npu_p3/cp_grad_cpu.log`）。
  上游是**无条件**打这个补丁的（`torch_npu/__init__.py:178` → `add_optim_method()`），
  没有开关可以绕开，所以"在有 NPU 的机器上整进程走 CPU"对本栈不可行。
- 因此在有加速器的机器上，`LLMTUNER_TEST_DEVICE=cpu` 只能约束 **harness 自己**的放置与进程组；
  `llmtuner` 的 `device_type` 在 import 期由硬件探测决定，凡是它自己放置张量的路径
  （`Trainer`、`apply_fsdp` / `parallelize_hf_transformers`、DCP checkpointer）仍按 NPU 走。
  实测见 §7.3。
- 本报告的并行验证不再只有"真机训练 + 少数脚本"：**DoD 点名的 15 条脚本都能在本机启动**，其中 12 条通过、1 条（`tp_ep_cp_equivalence`）暴露出与设备无关的既有不一致（§7.4）、2 条仍在环境门后（`cp_ulysses_equivalence`、`pp_real_corpus_equivalence`），另加 `pp_equivalence` 的 ZBV 调度是平台门（本节上面）。

### 7.2 让集成脚本能在 NPU 上跑：改法与途中发现的脚本缺陷

**改法**：新增 `tests/integration_tests/harness.py`，脚本不再自己钉后端/设备：

| 入口 | 作用 |
| --- | --- |
| `device()` | 本 rank 的设备，跟随 `llmtuner.accelerator.device.device_type`（NPU 优先）；`LLMTUNER_TEST_DEVICE=cpu` 强制 CPU |
| `backend()` / `init_distributed()` | 与设备匹配的进程组（hccl / gloo），并绑定设备 |
| `moved(x)` / `moved_batch(b)` | 把模型/张量、`Batch` 放到该设备 |

改动范围：`fsdp_mesh_equivalence`、`fsdp_checkpoint_equivalence`、`tp_equivalence`、
`tp_grad_norm_equivalence`、`pp_equivalence`、`cp_equivalence`、`ep_equivalence`、
`ep_fsdp_equivalence`、`tp_fsdp_equivalence`、`pp_ep_equivalence`、`pp_cp_equivalence`、`cp_grad_equivalence`
这 12 条（判据、容差、参考实现全部未动，只改设备/后端的解析与张量放置）。

**途中发现 5 个"脚本自身"的问题**（都是在本机 NPU 上才暴露的，CPU/gloo 下不显形；
都在仓库测试侧，没有一条需要改 llmtuner 的产品代码）：

| 脚本 | 症状 | 原因 | 修法 |
| --- | --- | --- | --- |
| `fsdp_checkpoint_equivalence` | 同布局恢复组报 `Scalars are not equal!`（`3.512718439` vs `3.512718678`，差 2.384e-07 = 1 ULP） | 该组用 `resume atol=0` 断言**位级一致**，而 NPU 默认归约不确定（§8.1.1 同源） | 脚本自己开 `set_deterministic_algorithms()`——位级断言以确定性为前提，随后 6 组全过 |
| `pp_ep_equivalence` | oracle 前向报 `Expected NPU tensor`（MoE 路由里） | oracle 在 `Trainer` 之后构建，而训练器的 `spmd_context` **只恢复 mesh 栈、不恢复 mesh 注册表**（`set_spmd_meshes` 是覆盖式），于是 swap 后的 MoE 读到训练器的稀疏 EP mesh，走 DTensor 分支 | oracle 包进 `with spmd_context(None):`——与 `ep_fsdp_equivalence` 既有的写法一致 |
| `pp_ep_equivalence` | 同上，修完第一条后改报 `self.instance_acc.add_(raw_sum * scale)` 设备不一致 | oracle 模型/批次在 CPU，而 swap 出来的辅助损失缓冲（`AuxLoss.instance_acc`）在加速器设备上 | oracle 与训练臂同设备（`moved` / `moved_batch`） |
| `tp_fsdp_equivalence` | 参数对比时 `Expected all tensors to be on the same device` | 参考臂（单进程、无并行）在 CPU，训练臂参数在 NPU | 参考臂也放到设备上（模型 + `RandomTokenSource` 的 `Batch`） |
| `ep_fsdp_equivalence` | `Expected NPU tensor`（模型前向里） | 该脚本把 `parallelize_hf_transformers(..., device=torch.device("cpu"))` 写死了 | 改为按 `device()` 解析 |

另有一条**先前的说法**需要更正：上一版报告把 `cp_grad_equivalence` 的 NPU 端口版写在仓库外
（`tmp/cp_grad_npu.py`）。现在它已按同一 harness 在**仓库内**跑通，仓库外那份只作为历史痕迹保留。

### 7.3 CPU/gloo 回归：改了设备解析，没改 CPU 上的数字

用 `LLMTUNER_TEST_DEVICE=cpu` 把同一批脚本钉回 CPU/gloo（`outputs/npu_harness/on_cpu_*.log`），
与改动前"原样跑"的结果对照：

| 脚本 | 改动前（原样，NPU 可见） | 改动后（CPU 模式） | 判定 |
| --- | --- | --- | --- |
| `tp_equivalence` | ✅ logits `2.980e-08`、梯度 `1.192e-07` | ✅ **同样的 2.980e-08 / 1.192e-07** | 数字逐位相同 |
| `tp_grad_norm_equivalence` | ✅ | ✅ | 一致 |
| `cp_equivalence` | ✅ 三路 `0.000e+00` | ✅ 三路 `0.000e+00` | 一致 |
| `ep_equivalence` | ✅ `max rel error across ranks = 0.000e+00` | ✅ 同 | 一致 |
| 其余 10 条（走 `Trainer` / `apply_fsdp` / DCP） | ❌ 当时就失败（设备/后端错配） | ❌ 仍失败，但报错**明确**：`Expected all tensors to be on the same device … on npu:0` 或 `No backend type associated with device type npu/cpu` | 见下 |

后 10 条的失败不是新引入的：它们在改动前于本机**同样失败**（§7.1 开头那张 as-is 表），
原因是 LLMTuner 自己会把参数放到 NPU，而脚本把数据放在 CPU。现在的差别只是**失败原因更好懂**，
并且它们在 NPU 模式下（默认）全部通过。

**所以"CPU/gloo 行为不变"这句话的适用范围是**：不把放置权交给 llmtuner 的那些脚本
（4 条，数字逐位相同）；其余脚本在本机（有 NPU）本就跑不了 CPU 路径，它们的 CPU 覆盖由
`tests/unit_tests/cpu/` 那 2500+ 条用例承担。

### 7.4 `tp_ep_cp_equivalence`（TP×EP×CP）：先暴露一个静默算错的缺陷，再修掉

这条 DoD 点名的 P2 脚本现在在 CPU/fp64 与 NPU/fp32 上都通过。过程值得记下来，因为**它曾经
是静默算错的**——8 卡"每步只有同一个 loss"那类检查看不出这类错误。

**症状**：脚本自己报两处 logits join 与平均 loss 不一致，而且**换设备、换 dtype 数字完全一样**：

| 运行 | 精度 | tol | `tp-joined vs CP-block ref` | `cp-joined vs full ref` | 平均 loss diff |
| --- | --- | --- | --- | --- | --- |
| 4 卡 NPU/hccl | fp32 | 1e-4 | **5.147e-02** | **5.147e-02** | 5.741e-04 |
| 4 卡 CPU/gloo | fp64 | 1e-6 | **5.147e-02** | **5.147e-02** | 5.744e-04 |

**定位（`tmp/tp_ep_cp_localize.py`，逐张量前向钩子 + 轴隔离）**：

1. **不是 CP**：`tp=1, cp=2`（EP×CP）时两个臂的 `layer0`/`mlp0`/router 全部 `≤1e-9`（fp64 噪声）。
   同一份代码 `tp=2, cp=1`（TP×EP）就出现 `5.147e-02`——**是 TP 引入的**，与 CP 无关。
2. **不是路由、也不是输入**：`mlp_in0` 逐位相同（0.0）、第一层 router logits `1.516e-07`（fp64 噪声），
   而 **`mlp0` 输出差 `4.573e-03`，相对该张量自身量级是 76%**——问题在**专家计算**这一层。
3. **几何**（`tmp/tp_ep_cp_layout.py`）：`tp=2, ep=2` 时每个 rank 的专家权重是 `w1_EFD (4, 24, 64)`，
   而**参考模型的原生 F = 48**（`gate_proj.weight (48, 64)`）——**F 被 TP 切成了 F/tp**；
   同时 `boundary=no`（没有任何 reduce-scatter 去补回另一半）。
4. **机制**：默认 TP plan 的模式列表里就有 `mlp.experts.*.gate_proj/up_proj/down_proj`
   （见 `resolve_tp_targets` 的报错文本），而 `experts_implementation="eager"` 下这些正是**逐专家的
   `nn.Linear`** —— 于是 dense realizer 把它们换成 F/tp 的并行 linear；`tp×ep` 又把 MoE 实现
   **推迟给 `apply_ep`**，`apply_ep` 的 swap 只按**专家维**切片（`gate_EFD[lo:lo+num_local]`），
   把那些 F/tp 的半成品权重原样搬进 `GroupedExperts`，而 realizer 的集合通信（会把 F 两半求和）
   随模块替换一起丢掉了。结果：**每个专家只贡献一半的输出**，没有任何报错。

**修复**（`llmtuner/parallel/tensor_parallel/tensor_parallel.py::resolve_tp_targets`）：那段
"MoE 块内部永不走 dense realizer" 的过滤原先只在 `moe_blocks` 非空（即 MoE 的 TP 实现真的被
realize）时生效；现在只要模型里有 HF MoE 块（且未被 TP boundary 接管）就一律剔除其内部，
于是它同时覆盖 fused 布局（原本就对）、`tp×ep`（推迟给 EP swap）和 `ep=1`（eager 布局没有
可挂载的 stacked 参数）三种情形。回归用例：
`tests/unit_tests/cpu/parallel/test_tp_moe.py::test_the_dense_plan_leaves_moe_internals_to_the_expert_axis`（ep=1/2 两个参数）。

**修后实测**：

| 检查 | 修前 | 修后 |
| --- | --- | --- |
| `w1_EFD`（tp=2, ep=2） | `(4, 24, 64)` | **`(4, 48, 64)`**（与参考同 F） |
| `mlp0` 最大绝对差（tp=2, cp=1, fp64） | `4.573e-03`（76% of \|ref\|） | **`5.744e-10`** |
| `tp_ep_cp_equivalence`，CPU/fp64 | ❌ 3 项失败（5.147e-02） | ✅ `all checks passed` |
| `tp_ep_cp_equivalence`，NPU/fp32 | ❌ 3 项失败（5.147e-02） | ✅ `all checks passed` |

日志：`outputs/npu_harness/{on_npu_tp_ep_cp_equivalence,tp_ep_cp_cpu,localize*}.log`；
隔离与几何探针在 `tmp/`（`tp_ep_cp_localize.py`、`tp_ep_cp_layout.py`、`tp_ep_cp_cpu_probe.py`）。

**为什么这类错误能活到现在**：这个 tiny 模型上 MoE 块的输出量级只有残差流的 ~0.14%
（`|mlp0|` `6.0e-03` vs `|mlp_in0|` `4.28`），所以"每个专家只算一半"在**端到端 loss** 上只值 3e-5——
同一配置 8 卡 3 步：修前 step1 `6.95791`、修后 `6.95794`（`outputs/npu_ep/tp2_ep2b.log` vs `tp2_ep2_fixed.log`），
`loss 有限 + 每步 8 个 rank 报同一个值`这类检查**永远看不出来**。四族组合
（`tp=2×cp=2×ep=2`，8 卡 3 步）修后同样 exit 0（`outputs/npu_ep/tp2_cp2_ep2_fixed.log`）。
结论：并行组合的正确性必须靠"与单卡参考逐张量等价"的脚本钉住——这正是 §7.1 那张表存在的理由。

## 8. P4 参考实现在本容器的可用性（实测）

P4 要求「与 TorchTitan 精度对齐」，但两侧参考实现在本容器**都跑不起来**，原因是
torch 版本，而不是代码：

| 参考实现 | 其声明的依赖 | 本容器 | 结论 |
| --- | --- | --- | --- |
| `/home/jianzhnie/llmtuner/llm/torchtitan-npu`（目标里指定的 NPU 基线） | `requirements.txt`: `torch==2.14.0`、`torch_npu==2.14.0.dev20260918`、`torchtitan==0.3.0` | torch 2.9.1 / torch_npu 2.9.1 | **不兼容，无法在本容器运行** |
| `/home/jianzhnie/llmtuner/llm/torchtitan`（上游语义对齐） | `spmd_types==0.2.5`（`components/loss.py`、`distributed/linear.py`、`distributed/context_parallel/api.py` 等**模块级** import） | 该包在 torch 2.9 上根本无法 import（需 `torch.distributed._local_tensor`，见 §5） | 子模块导入即失败（另有 `tyro` 等缺失依赖） |

所以 P4 需要的不是代码改动，而是一个 **torch 2.14 + torch_npu 2.14 的环境**。两条出路：
在那种容器里跑两侧对照；或把 P4 的基线改成别的可运行参照物。**这需要你来定**——
它不是本仓库能自行解决的问题。

### 8.1 现在就能做的那一半：用基线的判据给 llmtuner 打分（已跑）

基线仓库自带数值对齐规范与**指标实现**：

- `torchtitan-npu/docs/developer_guides/numerical_alignment.md` 给出阈值——
  logits 数值近似：`mean_ae ≤ 0.02`、`mean_re ≤ 0.02`、`cos ≥ 0.9999`（`max_re` 仅诊断）；
  `loss` / `grad_norm` 数值近似：`mean_ae ≤ 0.02`、`max_re`/`mean_re ≤ 0.005`。
- `torchtitan-npu/scripts/numerical_alignment/utils/metrics.py` 是这些指标的实现
  （`compare_tensors`、`failed_logits_metrics`、`LOGITS_CLOSE_THRESHOLDS`）。它**只 import
  torch，不依赖 torch 2.14**——所以判据口径今天就能落地。

新增 `tmp/p4_alignment_metrics_npu.py`：**按文件路径原样加载基线的 metrics 模块**
（不重写指标、不复制常量），用它的函数和阈值给 llmtuner 侧的数字打分。2 卡 NPU、
tiny qwen3（2 层 / hidden 256 / 8 头）、bf16、`cp=2`、seq 512、两侧同权重同输入：

| 对照 | rank | mean_ae | mean_re | cos | 判定 |
| --- | --- | --- | --- | --- | --- |
| CP 分片 logits vs 单卡全长 logits（同 dtype bf16） | 0 | 6.408e-04 | 1.242e-02 | 0.99999446 | **PASS** |
| 同上 | 1 | 5.135e-04 | 1.045e-02 | 0.99999559 | **PASS** |
| 诊断：单卡 bf16 vs 单卡 fp32（同权重、无并行） | 两卡同值 | 3.143e-03 | 6.065e-02 | 0.99991906 | `mean_re` 超阈（**非门禁**） |

读数（都来自 `outputs/npu_p3/p4_metrics_npu.log`，`failed ranks = 0`）：

1. **CP 本身在这套判据下是"数值近似通过"的**：两卡的 `mean_ae`/`mean_re`/`cos`
   都在基线的 logits 阈值内；`max_re` 很大（16~18）而 `mean_re` 只有 1e-2，
   正是规范里 `max_re` 只作诊断的原因（个别接近零的 logits 会放大逐元素相对误差）。
2. **0.02 这条 `mean_re` 线是给"同 dtype 的近似实现对照"用的**：把 bf16 结果拿去和
   fp32 比，本模型的 `mean_re` 地板就已经是 6e-2（`cos` 仍在 0.9999 以上）。
   也就是说 **CP 引入的偏差（1.0~1.2e-2）小于 dtype 地板（6.1e-2）**——
   在这套数字里 CP 不是误差来源。该行因此标为诊断而非门禁。
3. 复现命令见 §7 最后一条。

**所以 P4 拆成两半**：判据/工装这一半**已经在 llmtuner 侧落地并通过 CP 一项**；
剩下的一半是"跑起 torchtitan 的模型做端到端对照"，那一半仍卡在 torch 2.14。

#### 8.1.1 P4 的曲线比较先要解决可复现性（已跑）

同一命令、同一 seed、同一配置在 8 卡上重复跑（Qwen3-8B，`cp=2 × dp_shard=4`，
seq 4096，3 步）——**示例默认 `deterministic=False`，跨进程只在前 1 步逐位可复现**：

| 运行 | step 1 loss / grad_norm | step 2 loss / grad_norm | step 3 loss / grad_norm |
| --- | --- | --- | --- |
| 20 步那次 | 2.55784 / 38.7500 | 2.41371 / 61.7500 | 2.22019 / 27.1250 |
| 冒烟 | 2.55784 / 38.7500 | 2.41280 / 61.2500 | — |
| 重复 1 | 2.55784 / 38.5000 | 2.40440 / 62.2500 | 2.22132 / 27.2500 |
| 重复 2 | 2.55784 / 38.5000 | 2.40676 / 59.0000 | 2.22108 / 27.1250 |
| **`LLMTUNER_DETERMINISTIC=1` 重复 1** | 2.55784 / 38.5000 | **2.41258 / 62.7500** | **2.22591 / 27.3750** |
| **`LLMTUNER_DETERMINISTIC=1` 重复 2** | 2.55784 / 38.5000 | **2.41258 / 62.7500** | **2.22591 / 27.3750** |

读数：

1. 非确定模式下 step 2 的 loss 极差 9.3e-3（≈0.39%）、grad_norm 抖到第 3 位
   （59.0~62.75）；这与基线 `loss`/`grad_norm` 的阈值（`mean_ae ≤ 0.02`、
   `max_re`/`mean_re ≤ 0.005`）**同一量级**，足以淹没真实差异。
2. `deterministic=True` 下两次运行**逐位相同**（含 grad_norm）。该开关做的事是
   `torch.use_deterministic_algorithms(True, warn_only=False)` +
   cudnn 相关开关（`llmtuner/utils/seeding.py:set_deterministic_algorithms`）；
   `warn_only=False` 意味着遇到不支持的算子会**直接报错**——本次 8 卡 cp=2 没报错，
   说明这条路径上确实没有踩到不确定性算子。
3. 因此**跨实现逐 step 比较必须开 `deterministic=True`**，否则"对不齐"无法归因。
   库里的 `TrainingConfig.deterministic` 默认就是 `True`；示例原先硬编码成 `False`
   （吞吐优先），本轮给它加了 `LLMTUNER_DETERMINISTIC` 旋钮，**默认值保持 `0`**，
   即示例的对外行为不变。

复现命令：§7 的 cp=2 那条，附 `LLMTUNER_DETERMINISTIC=1`，跑两次比对 step 1–3。

### 8.2 拿到 2.14 容器后的 P4 步骤（**未跑**，待环境）

1. 装环境：`torchtitan-npu` 里 `pip install -r requirements.txt && pip install -e .`；
   llmtuner 侧同样 `pip install -e '.[dev]'`。
2. 基线自检：`python -m scripts.numerical_alignment.compare_module_example`
   （RMSNorm 前向/输入梯度/权重梯度，应 PASS）——先确认对齐工装本身可用。
3. 模块级（规范 §3.1/§3.2）：llmtuner 的模型块（HF 后端或并行 realizer）vs
   torchtitan 对应块，同权重/同输入/同 dtype/同上游梯度，比 forward、input_grad、
   weight_grad，判据 `mean_ae ≤ 0.02`、`mean_re ≤ 0.02`、`cos ≥ 0.9999`。
4. 整网单步：同权重、同全局 batch、同 seed，比 logits 与 `loss`/`grad_norm`，
   后者按 `mean_ae ≤ 0.02`、`max_re`/`mean_re ≤ 0.005` 收。
5. 多步：`scripts/numerical_alignment/loss_compare.py` 是"两个 git ref 之间"的比较器
   （`--mode close` 走数值近似、`--mode exact` 走比特一致，逐 step 读 TensorBoard 的
   `loss`/`grad_norm`）。跨仓库对照要改成"同 seed、同数据、同 steps 的曲线逐 step 打分"，
   该脚本的取值/比对机制可照搬。
6. **需要你定的两件事**：P4 的基线选 `torchtitan-npu`（NPU 原生，但要 2.14 环境）
   还是上游 `torchtitan`（同样受版本门限）；以及是否接受
   "基线指标口径 + llmtuner 自比对（如 §8.1）"作为 P4 的验收下限。

## 9. 新栈复验（2026-10-20：torch 2.12 + transformers 5.11 + 上游 49 commit 合入后）

### 9.1 背景

本节之前的内容全部在 **torch 2.9.1 + torch_npu 2.9.1 + transformers 4.56.2** 上取得。
本轮环境升级为 **torch 2.12.0+cpu + torch_npu 2.12.0 + transformers 5.11.0**
（+ torchvision 0.27.0+cpu，pyproject 依赖随之全部满足，pin 已同步到
`transformers>=5.11,<5.12`），并把工作区合并到上游 `main` 的 49 个新 commit
（3c67bba → ccf35dd，19 个冲突文件已解）。所有结论在新栈上**重跑取证**，
旧栈结论不再作为依据。

### 9.2 复验结果（全部新跑，日志在 `outputs/`，前缀 `on_npu212_` / `*-t212`）

| 项 | 结果 | 证据 |
| --- | --- | --- |
| P0 基础训练（8 卡 FSDP，Qwen3-8B，5 步） | **通过** | `outputs/qwen3-8b-npu-t212/logs/`：exit 0，loss 2.669→1.223，无 NaN/Inf，~29.6GiB/卡，末步存 step-5 |
| P0 resume | **通过** | 同目录：`Resuming from step 5` → 跑 6–7 步 → 存 step-7 |
| P1/P2 仓库等价性脚本（19 条） | **19/19 通过** | `outputs/npu_harness/on_npu212_*.log`：fsdp_mesh ×2、fsdp_checkpoint（native 4 组 + custom 按能力跳过）、tp ×2、pp 三调度（含 **ZBVZeroBubble**，参数差 1.192e-07）、cp、cp_grad、ep、ep_fsdp（4 卡）、tp_fsdp、pp_ep（4 卡）、cp_tp、tp_ep_cp、cp_sdpa、cp_ulysses（CPU 模式） |
| P2 真实模型组合 | **通过** | Qwen3-8B pp=2 4 步 exit 0（loss 2.797→1.856）；cp=2×dp_shard=4 4 步 exit 0（loss 2.558→1.758，step1 与旧栈逐位相同） |
| P3 特性等价性脚本（20 条） | **20/20 通过** | `on_npu212_p3_*.log`：ac、pp_tp、pp_validation、metrics、chunked×PP 拒绝、vocab-parallel、异步 DCP、EP×AC×EMA、pp_checkpoint plain/packed（full+resume）、ep_wiring、ep_grad_norm、moe_aux、domino、tp_replicated_grad、reduce、quantile_bias、hf_checkpoint_pp、ep_fsdp_without_dense_dp、chunked×TP |
| P4 TorchTitan 精度对齐 | **跳过（决策）** | 2026-10-20 决定跳过。探路结论备查：上游 torchtitan main 在 torch 2.12 下可完整 import（tyro/docstring_parser/torch_remat 已装）；`torchtitan_npu` 链还差 triton（其 `minimal_async_ep` kernels 需要，aarch64 无官方 wheel） |

### 9.3 新栈解除的旧平台门

| 旧门（2.9 栈） | 新栈表现 |
| --- | --- |
| `cp_ulysses_equivalence`：缺私有 `_HeadTailLoadBalancer` | torch 2.12 已提供；CPU 模式通过（flex 本身仍仅限 CUDA/CPU/HPU，NPU 侧 ulysses 走 SDPA kernel） |
| `pp_equivalence` ZBVZeroBubble：HCCL 对象 p2p 不支持 | torch 2.12 + torch_npu 2.12 上不再出现，ZBV 直接通过 |
| EP×FSDP 只能走嵌套单元回退 | 2.12 有 `fsdp_per_param_mesh`，走 placement-fn 主路径（回退保留给 <2.12） |
| `spmd_types` 不可导入、`_unflatten` 缺失、PP 无原生微批驱动等 | 2.12 均已具备（回退代码保留且按 capability 自动旁路） |

### 9.4 本轮修复（新栈/合并暴露，均已暂存）

1. **合并失误（EP）**：上游共享参数绑定只认 5.x 融合布局，与 4.x per-expert 探针合并后 EP swap
   全部 `AttributeError`（53 个单测失败）——`model_replacement.py` 按布局重写绑定循环，
   per-expert 参数绑定到堆叠目标并进入 `converted_parameters`。
2. **selective activation checkpoint 真实缺陷**（上游移植，7 个单测）：`wrap_selective` 的
   "隔一个 matmul 保存"计数器按层共享、跨 microbatch 累积，奇数次 matmul 后 replay 与 save
   决策不一致——`memory/activation_checkpoint.py` 改为每个 checkpoint frame 新建 policy。
3. **torch 2.12 流水线 `fork_rng` 默认 `device_type="cuda"`** 在无 CUDA 构建上初始化即炸
   （`Torch not compiled with CUDA enabled`）——`microbatch_drivers.py` 加作用域 shim，
   从 stage 设备推导 device_type。
4. `fsdp_checkpoint_equivalence`：custom（细粒度）FSDP 需 torch 2.14 私有 API，按
   `CAPS["fine_grained_fsdp"]` 门控，2.12 上只跑 native 并打印 SKIP。
5. `pp_ep_equivalence` / `pp_equivalence`：上游新增的逐 step 参数快照对比混入跨设备
   kernel 舍入（CPU 参考 vs NPU 实机，1.1e-5 微超容差）——快照统一 `.cpu()`、参考臂改为
   **同设备运行**（被测性质"流水线==非流水线"不变），参数差回到 1.192e-07；归约诊断
   张量改为设备上 float32（HCCL 不支持 CPU/fp64）。
6. `cp_ulysses_equivalence`：torch 2.12 移除 `create_block_mask` 的 `separate_full_blocks`
   参数（行为已固化为默认），删除该 kwarg。
7. `tp_replicated_grad_equivalence`：补 harness 设备放置（模型/数据/判决胜张量上设备，
   `init_distributed()` 取代裸 gloo）。
8. 测试侧签名对齐：`test_hf_wrapper.py`、`pp_real_corpus_equivalence.py` 的
   `_apply_attention` 调用点统一为上游 2 参形式；`test_checkpointer.py` fake backend 补
   `LocalFileSystemStorageConfig`；`test_fsdp_contract` 两例补 `require_env("fine_grained_fsdp")`；
   `pyproject.toml` pin 升到 `transformers>=5.11,<5.12`。
9. CPU 单测（容器，torch 2.12.0 + transformers 5.11.0）：**4064 passed / 0 failed / 13 skipped**。
