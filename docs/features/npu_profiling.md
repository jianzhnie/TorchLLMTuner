# NPU profiler 采集等级

`ProfilerConfig.npu_profiler_level` 可显式选择 torch-npu 的采集等级：

```python
from llmtuner.config.observability_config import ProfilerConfig

profiler_config = ProfilerConfig(
    enable_profiling=True,
    npu_profiler_level="level1",
    profile_freq=4,
    profiler_warmup=1,
    profiler_active=1,
    profiler_repeat=1,
)
```

将它传给现有 `TrainingConfig.profiler_config`。CLI 对应
`--npu_profiler_level level1`；合法值为 `level0`、`level1`、`level2`。
默认 `None` 不传 experimental config，保留 torch-npu 默认行为；
其他设备忽略该选项，关闭 profiling 时不会初始化 profiler。

Level0 可用于较低开销的调用/设备时间线，但不足以保证 HCCL 的传输类型、
size 和等待细节齐全。需要通信归因时使用 Level1，并检查实际产物；
具体字段取决于 CANN、设备和算子。Level2 开销更高，当前尚无本项目的真实设备
验收结果，不将配置可选择等同于验证通过。

默认输出位于运行目录的 `profiling/traces/`：每个采集窗口有 rank trace，
`raw/` 保留 torch-npu 原始数据及采集 metadata。分析时先核对
`profiler_info_<rank>.json` 的 `_profiler_level`，再检查通信数据字段是否有效。
size=-1、transport=N/A 等占位值不能用于带宽或等待归因。

真实双rank Ascend910_9392、torch/torch-npu 2.12、CANN 9.1.0 的小型
all-reduce 验证已确认 Level1 生效：两个rank均产出非空trace，HCCL数据包含
有效LOCAL/SDMA、size和duration。它只验证采集接入，不证明30B通信根因、
跨节点采集或训练性能。正式吞吐测量应关闭 profiling，编译与预热另行记录。

实际30B DP8的Level1采集已完成两版本各14步，八rank均退出0。
注意：`HCCLTaskSingleDevice` 可能只包含主流的Notify_Record/Notify_Wait，
size为0，不代表原始数据缺少传输。此环境的payload记录位于`kfc_info.db`；
需要运行官方完整离线分析，才能得到包含KFC执行信息的通信报告：

```python
import torch_npu

torch_npu.profiler.profiler.analyse(
    "/path/to/raw/<rank>_ascend_pt",
    max_process_number=4,
    export_type="text",
)
```

检查该rank目录下`ASCEND_PROFILER_OUTPUT/communication.json`是否实际生成，
并核对算子名称、起止时间和有效payload；函数返回不等同于解析完整。
报告的step标签可能与框架iteration目录相差一，不能只按标签匹配。
本项目已将实际30B rank0的50次reduce-scatter逐项匹配到active trace，
44次包含正payload。不同链路分类可能重叠，子任务时间也可能重叠，
不能跨类别直接相加；主流Notify_Wait不能直接解释成纯网络等待。

## 宿主机频率与调度观测

性能对照应同时记录逐卡频率、温度/功率和主机调度，避免将不同运行阶段的
`npu-smi info` 单次截图当作环境一致的证据。AICore usage 不等于训练 MFU。

当前主机已验证 `/usr/local/dcmi/libdcmi.so` 的只读频率查询可用。
必须以同机 `/usr/local/dcmi/dcmi_interface_api.h` 为 ABI 依据：

```c
int dcmi_init(void);
int dcmi_get_device_frequency(int card_id, int device_id,
                             enum dcmi_freq_type input_type,
                             unsigned int *frequency);
```

本机枚举为 AICore current=7、AICore max=9、HBM=6；其他环境先核实头文件，
不要直接复制枚举或猜测结构体。先检查 `dcmi_init` 返回码，每次查询记录
card/chip、单调时间、返回码和数值；失败记为缺失，不写成0。
`npu-smi info -m` 显示本机4个card各2个计算chip，另有MCU，不能将MCU
计作训练设备。通过只读查询采样，不调用调频/功耗设置接口。

温度/功率可用 `npu-smi info -t sensors -i <card> -c <chip>` 和
`npu-smi info -t power -i <card> -c <chip>`；先测查询开销，再选择有界
采样频率。当前主机 `pidstat`、`vmstat` 和 `/proc/schedstat` 可读，
`/proc/pressure/cpu` 与 `/sys/class/devfreq` 不存在，采集器需明确缺失字段。
容器PID与宿主机PID应核对映射，再记录训练进程CPU用量、上下文切换和
可用的调度等待计数；其他进程的瞬时高用量不能直接证明训练受到干扰。

2026-11-06空闲主机接口探测：八chip三类频率查询全部返回0，读数分别为
800/1850/1600；这是空闲读数，不能外推训练频率、降频原因或性能。
原始记录为忽略目录中的
`outputs/optimizer_current_20261105/dcmi_frequency_probe_20261106.json`。
下一次配对基准应从预热前持续记录至测量结束，关联最慢rank的step窗口，
并单独评估采样开销；当前仅验证了查询能力，尚未完成连续训练遥测。

已在忽略目录准备诊断采样器
`outputs/optimizer_current_20261105/sample_host_npu.py`（仅适用于上面核实的
本机ABI/4×2映射）。每轮记录单调/墙钟时间、DCMI查询返回码与原始频率/
温度/功率、`/proc/stat`、loadavg、meminfo及可选训练进程各线程schedstat。
`--match`仅匹配Python进程参数，使用唯一训练脚本名；原始命令行不写入日志。
输出用排他创建避免覆盖，逐轮flush；SIGTERM/SIGINT结束采样，不终止训练。

```bash
python outputs/optimizer_current_20261105/sample_host_npu.py \
  --output outputs/optimizer_current_20261105/host_sampling_unique.jsonl \
  --samples 300 --interval 5 --match train_qwen3_30b_npu.py
```

先核实节点映射和库，不在不兼容主机直接运行此诊断脚本。本机3轮空闲采样
每轮约56–58ms（8chip×5项查询全部返回0），实际间隔约1000ms；启用进程
扫描并记录一个短暂Python进程时单轮约126ms，成功记录各线程schedstat，
SIGTERM退出0且JSONL完整。原始功率值例如1678，未在脚本中猜测换算单位。
这些只是接口/生命周期探测，不证明采样在训练中零扰动；正式基准需比较
有无采样的影响，并保留同一采样配置。不要将每轮最大耗时当作端到端开销。
