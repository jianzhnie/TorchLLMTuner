# DeepSeek V4 Pro NPU 训练

使用 Transformers 原生 `DeepseekV4ForCausalLM` 和本目录官方 tokenizer，随机初始化
全部参数，不加载预训练权重。默认保留61层及全部专家、Hyper-Connections、CSA/HCA
和hash路由结构。hash router的随机top-k表使用与TorchTitan-NPU相同的生成约定。

```bash
export PYTHON=/tmp/tlt_npu_env_20261109/bin/python
export MODEL_PATH=/home/jianzhnie/llmtuner/hfhub/models/deepseek-ai/DeepSeek-V4-Pro
export DATASET_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl
export OUTPUT_PATH=/tmp/deepseek-v4-pro
bash examples/deepseek_v4_pro/train_pro.sh --steps 100 --batch 8 --sequence-length 2048
```

`--layers N`仅在完整模型使用全部可用卡仍OOM后减层，并同步截断attention/MLP
调度，不改变其他结构。DP/TP/EP/PP/CP、AC、compile、optimizer状态驻留和checkpoint
参数见`examples/deepseek_v4_pro/train.py`；可配置不代表组合已验收。随机meta初始化
目前仅开放local/DP/FSDP，其他拓扑须通过独立初始化与数值验收后开放。

2026-11-09实测：完整61层反向额外申请2.50GiB时OOM，已分配55.36GiB。
随后30层、BF16、FSDP8、full AC、LIMO、sequence2048、batch8的一步前后向、
更新和同步DCP保存退出0。loss13.23101、grad norm27.8250、参数量43.324198B、
HBM reserved58.34GiB，OOM0、allocator retry2。日志为容器
`/tmp/deepseek_v4_pro_30layer_20261109.log`。验收命令应将上述默认命令改为
`--layers 30 --steps 1 --checkpoint-interval 1`。一步结果不是跨框架或稳态性能
验收；FLOPs/MFU因通用估计器未覆盖V4而未报告。
