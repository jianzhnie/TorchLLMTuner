# TorchLLMTuner

> **Hybrid-parallel training for HuggingFace models over a Torch DeviceMesh** —
> a compact, readable distributed-training framework (FSDP2 / TP / PP / CP / EP),
> semantically aligned with TorchTitan.

`llmtuner` 用 `transformers` 的模型本体（`AutoModelForCausalLM`，离线可用
`AutoConfig.for_model` 构造小模型），并行化的硬活集中在 `parallel/`：声明式
TP、专家并行的 HF MoE 置换、CP 内核、PP 调度、FSDP2 包装，全部按一张显式的
装配顺序表（`parallel/stages.py`）组合，不支持的组合在配置期或装配期
loud-raise（组合裁决集中在 `parallel/compatibility.py`），绝不静默退化。

> GitHub 仓库名为 `TorchLLMTuner`，Python 包名为 `llmtuner`。

## 特性

- **FSDP2 / HSDP**：`fully_shard` 包装在最后；混合精度、reshard 策略、CPU offload；
  EP 开启时专家与 dense 参数走分离的稀疏/稠密网格。
- **TP**：序列并行构造（HF tp_plan 驱动，`ColumnParallelLinear` / `RowParallelLinear`
  自带通信边界）；MoE-under-TP（专家 F 维分片 + 块边界 AG/RS）；可选逐 block
  compile、async TP、regional inductor（`parallel/model_compile.py`）。
- **PP**：1F1B / Interleaved1F1B；first-stage modules 并入；validation 走 schedule
  的 eval 通路；打包真实语料的 positions 随 microbatch 穿管。
- **CP**：`kv_allgather` 与 `ulysses` 双策略，headtail 负载均衡；
  Ulysses 支持 varlen/packed。单轴 KV 的 SDPA 路径支持连续分片下的
  packed 文档掩码，packed 与负载均衡组合仍明确拒绝；单轴及双轴
  KV×Ulysses 的具体验证范围见 `docs/npu_ascend_status.md`。
- **EP**：HF MoE 块置换为原生栈（权重搬运逐位精确）；AllToAll dispatcher（可选
  TorchAO padded-permute 后端）；aux-loss-free bias 与 quantile 负载均衡；
  EP 感知的 checkpoint（存全量专家张量，resume 不受 EP 度数约束）。
- **组合**：DP×一切、TP×FSDP、CP×TP、EP×FSDP、TP×EP、TP×EP×CP、PP×EP、PP×CP、
  PP×AC、PP×validation 等均已接线；不支持的组合由组合矩阵逐项拒绝并写明解锁条件。
- **训练侧**：分组 `LLMTunerConfig`（CLI 扁平旗标 + YAML/JSON）、在线 EMA、
  validation 循环、chunked/vocab-parallel loss、确定性模式、metrics/profiler、
  DCP 与 torch_checkpointing 双 checkpoint 后端。
- **数据**：Grain 管线（packing、多模态、chat template / 可选多轮 renderer）、
  合成 random 语料开箱即跑。

## 安装

```bash
git clone https://github.com/jianzhnie/TorchLLMTuner.git
cd TorchLLMTuner
pip install -e .
```

核心依赖：`torch>=2.12`、`transformers>=5.9,<5.10`、`spmd_types`、`grain`，
以及数据层的一组常用包（见 `pyproject.toml` 注释）。可选：`renderers`（多轮
chat renderer）、`torchao`（EP padded dispatcher）——未装时启用对应功能会
得到带安装指引的 `ImportError`。

## 快速开始

```bash
# 单设备（无需 GPU/torchrun）
python -m llmtuner --steps 20

# 数据并行，2 进程
torchrun --nproc_per_node=2 -m llmtuner --data_parallel_shard_size 2
```

## 示例（`examples/`）

自包含脚本，离线构造小模型。**必须从仓库根目录用 `-m` 跑**（直接
`python examples/x.py` 会把 `examples/` 放进 `sys.path`，`import llmtuner`
会失败）：

```bash
python -m examples.train_qwen3           # Qwen3 + GQA
python -m examples.train_deepseek_v3     # DeepSeek-V3：路由 MoE + MLA（arch_overrides 演示）
```

DeepSeek-V3 例子里 `n_routed_experts` / `q_lora_rank` 这类字段走
`ModelConfig.arch_overrides` 传——不传会拿到 `for_model("deepseek_v3")` 的默认
671B 形状。NPU 多卡 FSDP 见 `examples/train_qwen3_8b_npu.sh`。

## 代码地图

| 位置 | 职责 |
|---|---|
| `llmtuner/config/` | 分组配置（Model/Parallel/Optimizer/Checkpoint/Data/Training + `LLMTunerConfig` 聚合根），每组自校验 |
| `llmtuner/parallel/` | 并行五族（`tensor_parallel` / `expert_parallel` / `context_parallel` / `pipeline_parallel` / `fully_shard_parallel`）+ `stages.py`（装配顺序）、`compatibility.py`（组合裁决）、`model_compile.py`、`topology.py` |
| `llmtuner/memory/` | activation checkpoint 与其他内存优化 |
| `llmtuner/models/` | `hf/`（HF wrapper 五部件契约、config 工厂、state-dict 适配）+ `common/`（MoE 栈、loss 件、上游对照实现） |
| `llmtuner/components/` | loss / optimizer+EMA / lr_scheduler / checkpointer / metrics / profiler / tokenizer |
| `llmtuner/datasets/` | Grain 数据层（sources、loader、packing、text、multimodal、random 语料） |
| `llmtuner/accelerator/` | 设备/进程组/集合通信抽象 + `capabilities.py`（环境能力探测） |
| `llmtuner/trainer/` | `trainer.py`（训练与验证编排）、`runtime/`（运行时装配与校验）及 CLI 入口 |
| `llmtuner/errors.py` | 三类异常：`ConfigError` / `UnsupportedCombinationError` / `EnvironmentUnsupportedError` |

架构契约与组合边界详见 `docs/torchllmtuner_design.md`；与 TorchTitan 的逐项
对应（文件级 + 符号级 + 对齐工作流）见 `docs/llmtuner_upstream_map.md`。

## 验证

```bash
# 单元测试（任何机器可跑；环境不满足的模块自动 skip 并给出原因）
python -m pytest tests/unit_tests -q -rs

# 多进程等价性（torchrun 脚本，需 torch>=2.12 分布式栈）
python tests/integration_tests/run_all.py --list
python tests/integration_tests/run_all.py vocab_parallel_loss_equivalence
```

单测用能力标记（`tests/caps.py`）按环境自动 skip，`-rs` 输出即环境覆盖报告；
等价性脚本验证分片执行与单卡参考在明确容差内一致（输出、loss、梯度、
state-dict FQN），全部 non-vacuous。

## 已知限制

- 多进程训练需要 CUDA/NCCL 或 NPU/HCCL；CPU+gloo 只能跑单进程与声明层测试。
- TP 融合算子走 symmetric memory（CUDA）；CPU 上只验证声明层与权重切分。
- DeepEP/HybridEP dispatcher 是登记的缺口（CUDA-only，配置期 loud-raise）。
- 部分组合仍未支持（chunked loss×PP、PP×tied embeddings 等），
  全部 loud-raise 且文案带解锁条件——完整矩阵见 `docs/torchllmtuner_design.md`。
- EP 的 HF 初始加载按模型布局校验：Qwen3MoE 须启用
  `checkpoint.sharded_expert_parallel`，TP>1 时还须启用
  `checkpoint.sharded_tensor_parallel`；Kimi Linear 使用独立布局适配。
  其他 EP 模型仍明确拒绝；实际验证范围见 `docs/npu_ascend_status.md`。
- chunked loss×原生默认布局 FSDP 的 NPU FP32 路径已验证绑定/非绑定权重；
  embedding、norm、head 及其别名由 root 持有，直到 decoder backward 才归约。
  此路径不支持自定义 FSDP patterns、CPU offload、TP/CP/EP/PP 或 compile；
  保留完整边界权重会增加驻留内存，配置与验证范围见 `docs/npu_ascend_status.md`。

## License

Apache-2.0（见 `LICENSE`）。
