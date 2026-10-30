# HF 模型接入流程

新 HF 模型接入 TorchLLMTuner 的逐步指南。

## 第 1 步 — 权威来源

- 锁定 `transformers` 版本与确切的模型类（如 `Qwen3MoeForCausalLM`）。
  先查 `tests/caps.py` 是否已有探针（`transformers_qwen3`、
  `transformers_gpt_oss`）；新家族需新增探针。
- 确认已安装的 `AutoModelForCausalLM` 能构造发布的架构——`AutoConfig`
  能加载本身不证明任何事。
- 冒烟/测试用离线小配置（`AutoConfig.for_model("<arch>", vocab_size=...,
  num_hidden_layers=2, ...)`，见 `examples/train_qwen3.py`）；接入过程绝不
  要求下载权重。

## 第 2 步 — 配置对齐

- 把 HF 字段映射到 `llmtuner/config/model_config.py` 的 `ModelConfig`：
  `model_name_or_path`、`vocab_size`、`hidden_size`、`intermediate_size`、
  `num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`，MoE
  模型另加 MoE 字段。
- 经 `LLMTunerConfig`（`llmtuner/config/llmtuner_config.py`）组装完整配置。
- 新配置校验补 CPU 单测（`tests/unit_tests/cpu/test_config.py` 风格：
  合法值通过，非法值抛 `ConfigError`）。

## 第 3 步 — 模型构造

- 先复用 `llmtuner/models/common/` 组件（`feed_forward.py`、`rope.py`、
  `embedding.py`、`moe/`、`attention/`）与 `models/hf/` 辅助
  （`normalization.py`、`qkv_projection.py`、`residual_normalization.py`、
  `attention_backend.py`），再考虑新写模块。
- HF 检查点的权重名映射走 `models/hf/state_dict_adapter.py` 的既有模式——
  验证 保存 → 加载 往返与 HF 布局一致。
- 构造出的模型必须在**任何并行或替换生效前**能直接跑前向/反向。

## 第 4 步 — 并行适配

- 经家族公共入口（`llmtuner/parallel/<family>/<family>.py`）与声明式
  pattern 字段挂接：TP 用 `tp_config.tp_colwise_modules` /
  `tp_rowwise_modules`；FSDP 计划用 `fsdp_config.fsdp_apply_modules` /
  `fsdp_ignored_modules`。
- MoE 模型：EP 经 `expert_parallel_size` + `ep_config.ep_token_dispatcher`
  （默认 `alltoall`；`mc2` 为 NPU 专用）；专家块的模型替换在
  `llmtuner/parallel/expert_parallel/model_replacement.py`。
- 绝不在通用并行/配置代码中插模型家族 `if` 分支——应扩展声明式契约。

## 第 5 步 — 显存特性兼容性

- 跑通计划支持的每个 `training.activation_checkpoint_mode`（`full`/
  `selective` 包裹每个 decoder 层；`region` 从 block 的 `nn.Linear` 派生
  region——确认 block 暴露了它们；`memory_budget` 要求
  `training.compile=True`）。
- 确认与 `llmtuner/memory/` 的兼容性（models→memory 的提醒由
  `.agent/hooks/check-related-files.sh` 承担）。

## 第 6 步 — 示例与测试

- `examples/train_<model>.py`：离线小配置 + `Trainer(...).train()`，仿照
  `train_qwen3.py`；NPU 多卡启动器仿照 `train_qwen3_30b_npu.py` / `.sh`。
- CPU 测试放 `tests/unit_tests/cpu/models/`；并行等价性放
  `tests/integration_tests/`，用 `tests.caps.require_env` 门控。
- 移交前的验证命令：`pytest -q --disable-warnings`、
  `ruff check llmtuner tests`、`python -m compileall -q llmtuner`，有硬件时
  加跑相关 `torchrun` 等价性脚本。
