# Kimi K3 随机初始化 NPU 训练

使用 `llmtuner/models/kimi` 的 `KimiLinearForCausalLM`，从官方 K3
`config.json` 的 `text_config` 构造模型，不执行官方建模 remote code，也不加载
预训练权重。tokenizer 显式使用本地官方 `TikTokenTokenizer`，需要 `tiktoken`；
每次聊天渲染都会核对训练管线编码与官方分段编码一致。

默认保留 93 层及全部专家、隐藏维度和注意力配置，采用 BF16、FSDP、full AC、
Host-backed optimizer 状态及同步 DCP。完整模型约 2.78T 参数；已记录的 8 卡完整运行 OOM。2026-11-09 本地实现复测中，
四层纯 FSDP 在单层 unshard 额外申请 56.31GiB 时 OOM，四层 EP8 也在物化/分片
阶段 OOM，两层 EP8 默认 DP 推导还在反向 reduce-scatter 额外申请 13.78GiB 时
OOM。短程端到端验收使用官方第一层（dense + KDA）的纯 FSDP；两层 EP8 只完成
前向，MoE 反向尚未通过。`--layers` 只减少层数并裁剪 KDA/MLA 层调度，不缩小
其他结构。

## 环境与依赖

```bash
export PYTHON=/tmp/tlt_npu_env_20261109/bin/python
export MODEL_PATH=/home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K3
export DATASET_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl
export OUTPUT_PATH=/tmp/kimi-k3-run
```

隔离环境需要 torch-npu、tiktoken、Triton-Ascend 和
`triton_ascend_kernels.attention.fla.kda`。KDA 源码按 FSDPTurbo 的 Atlas A2/A3
说明使用 `fengrui886/triton-ascend-kernels` 的 `kda_a3` 分支；本次来源固定为
`66be562e4e9cec7f03c8088de399dad1d08bdc38`。缺失依赖会明确报错，不会回退
CPU recurrence。

## 启动

```bash
bash examples/kimi_k3/train.sh --layers 1 --steps 100 \
  --sequence-length 2048 --batch 8 --checkpoint-interval 50
```

拓扑参数为 `--dp-shard`、`--dp-replicate`、`--tp`、`--ep`、`--pp`、`--cp`；
多节点使用 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。`--ac` 配置
重计算，`--compile-backend none|aot_eager|inductor` 配置编译，`--load-step`
恢复 checkpoint。选项暴露不等于组合已经验收，现有能力检查会拒绝不支持的组合。
