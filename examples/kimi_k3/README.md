# Kimi K3 随机初始化 NPU 训练

使用 `llmtuner/models/kimi` 的 `KimiLinearForCausalLM`，从官方 K3
`config.json` 的 `text_config` 构造模型，不执行官方建模 remote code，也不加载
预训练权重。tokenizer 显式使用本地官方 `TikTokenTokenizer`，需要 `tiktoken`；
每次聊天渲染都会核对训练管线编码与官方分段编码一致。

默认保留 93 层及全部专家、隐藏维度和注意力配置，采用 BF16、FSDP、full AC、
Host-backed optimizer 状态及同步 DCP。完整模型约 2.78T 参数，已记录 8 卡
完整结构 OOM。当前复测中，四层纯 FSDP 在单层 all-gather 额外申请 55.48GiB
时 OOM，四层 EP8 在反向 reduce-scatter 额外申请 13.78GiB 时 OOM。
旧两层 EP8 独立恢复虽退出0，但第52步起数值偏离，恢复验收失败；原因是
同步 DCP 对 virtual moment 局部视图读写零值。修复后重新生成检查点，连续100步
与独立50→100恢复均退出0，最终236个张量和758项序列化状态零容差数值一致。
该结果仅覆盖两层 EP8、BF16、full AC、同步 DCP 与 `npu_virtual` 配置；
旧检查点丢失的 moments 无法由加载修复重建，旧失败记录保留。
跨框架对齐、完整结构及其他组合仍未完成。`--layers` 只减少层数并裁剪 KDA/MLA
层调度，不缩小其他结构；两层不覆盖后续 MLA。

KCP 尚未在910C通过：当前 Ascend KDA 的 CP forward/backward 路径限定 A5
(arch3.5)，不能把普通 KDA 训练通过外推为 KCP 支持，也不能删掉硬件 guard。
恢复比较日志：`outputs/qwen3-30b-npu/logs/kimi_virtual_fixed_resume_exact_20261109.log`。

## 环境与依赖

```bash
export PYTHON=/tmp/tlt_npu_env_20261109/bin/python
export MODEL_PATH=/home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K3
export DATASET_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl
export OUTPUT_PATH=/tmp/kimi-k3-run
```

隔离环境需要 torch-npu、tiktoken、Triton-Ascend 和
`triton_ascend_kernels.attention.fla.kda`。KDA 源码按 FSDPTurbo 的 Atlas A2/A3
说明选择 A2/A3 适用实现。历史运行使用 commit `66be562e4e9cec7f03c8088de399dad1d08bdc38`；
当前复测固定本地源码 commit `7acb9c40a6ca827e885d902b7eaee5d45457b230`，
以显式 `PYTHONPATH` 加载，配合离线 Triton-Ascend 3.2.1。
该源码的旧依赖声明不能作为当前软件栈已兼容的证据，须以实际运行验证。
缺失依赖会在 Trainer 分配模型参数前明确报错，不会回退 CPU recurrence。

## 启动

```bash
export PYTHONPATH=/home/jianzhnie/llmtuner/wheels/kcp_probe/triton-ascend-kernels/src
bash examples/kimi_k3/train.sh --layers 2 --ep 8 --steps 100 \
  --sequence-length 2048 --batch 8 --checkpoint-interval 50
```

拓扑参数为 `--dp-shard`、`--dp-replicate`、`--tp`、`--ep`、`--pp`、`--cp`；
多节点使用 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。`--ac` 配置
重计算，`--compile-backend none|aot_eager|inductor` 配置编译，`--load-step`
恢复 checkpoint。选项暴露不等于组合已经验收，现有能力检查会拒绝不支持的组合。
运行证据与失败记录见 [NPU验收记录](../../docs/npu_acceptance_20261109.md)。
