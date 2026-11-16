# 实验流水线并行

PP 从生产 `llmtuner/parallel/pipeline_parallel` 迁到此目录。生产 Trainer、
装配器、执行引擎及 CLI 不再安装流水线 schedule；生产 `LLMTunerConfig`
不再包含 PP 配置字段。按要求删除 PP 专用测试，不在实验目录另存测试副本。

## 代码与入口

- `pipeline_parallel.py`、`partition.py`、`ownership.py`：stage 切分、归属及原生 PyTorch schedule。
- `microbatch_drivers.py`、`runtime.py`：microbatch 驱动与训练/验证执行。
- `plan.py`、`fsdp_plan.py`、`execution.py`、`layout.py`：PP 局部计划、边界及 ownership 元数据。
- `parallelize.py`、`assembly.py`、`engine.py`、`trainer.py`：独立实验装配及生命周期入口。
- `parallel_config.py`、`config.py`、`cli.py`：实验配置与命令行。
- `batch.py`、`loader_builder.py`、`collators.py`、`reductions.py`、`gradients.py`、
  `metrics.py`、`step_metrics.py`：实验 PP 的数据、归约和上报连接。

程序调用使用实验叶子模块：

```python
from experiment.pipeline_parallel.config import LLMTunerConfig
from experiment.pipeline_parallel.parallel_config import (
    ParallelConfig,
    PipelineParallelConfig,
)
from experiment.pipeline_parallel.trainer import Trainer
```

命令行入口为 `python -m experiment.pipeline_parallel.cli`，分布式启动使用
`torchrun --nproc_per_node=N -m experiment.pipeline_parallel.cli ...`。
配置中的 `--pipeline_parallel_size`、`--pipeline_parallel_schedule` 和
`--num_pp_microbatches` 由此入口解析；生产 CLI 不再提供 schedule 配置。
运行目录应为仓库根目录，并设置 `PYTHONPATH=.`。

实验实现复用生产模型契约、TP/EP/CP 变换与 checkpoint 服务；PP 专用拓扑、
模型阶段行为、集合通信及优化器协调保存在此目录。
生产代码不导入 `experiment`。实验布局继承生产布局契约以复用状态适配器。
此目录不在 setuptools 的 `llmtuner*` 发布包中。

`hccl_schedule.py` 保持删除，实验 PP 使用 PyTorch 原生 schedule。

## 本次迁移验证

迁移后在 `hyper-parallel-env` 中完成双进程 CPU/Gloo 的独立 write/resume，
两次进程分别完成训练与恢复后的更新及生命周期检查：

- `/tmp/llmtuner-migrated-pp-final-write.log`
- `/tmp/llmtuner-migrated-pp-final-resume.log`

这些验证使用小模型与合成数据，不能替代迁移后的 NPU 或长程验收。
PP 测试源文件已按要求删除；上述日志是删除前对迁移实现的验证记录。

进一步移出生产 PP 拓扑和 RNG 元数据后，双进程 CPU/Gloo 小模型再次完成
两次训练更新和关闭检查，日志为 `/tmp/llmtuner-experiment-smoke2.log`。
随后在两个独立 torchrun 进程中保存第 1 步 checkpoint、恢复 RNG/优化器/
scheduler 并完成第 2 步更新，日志为
`/tmp/llmtuner-experiment-recovery-write.log` 和
`/tmp/llmtuner-experiment-recovery-resume2.log`。
这些检查不包含 NPU 或长程验收，也不构成与连续训练的数值等价证明。
