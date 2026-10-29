# Ascend NPU 验证状态

## 2026-10-26：可用节点上的 gather 整模型 full/resume 配对

`available_nodes.txt`列出的节点为10.1.0.37、10.1.0.69、10.1.0.90。
启动前逐项检查了10.1.0.69和10.1.0.90的容器、torch/torch_npu版本及
npu-smi；选择10.1.0.69执行，使用NPU0/1，启动时两卡无运行进程。
10.1.0.37在检查时已有进程，因此未使用。节点和容器均为
`hyper-parallel-env`，实际版本为torch 2.12.0+cpu、torch_npu 2.12.0，
NPU可用。

在同一节点、同一新工作目录
`outputs/pp2-whole-gather-pair-20261026`中先后执行gather诊断包装器的
`full`和`resume`。该用例是PP2两卡辅助模型，不是30B目标训练；它用于
验证整模型诊断的训练、checkpoint和独立进程恢复链。包装器只在进程内
替换匹配的三处负偏移加载，连续加载后用`tl.gather`重排，并保留原始
有限性和精度断言；未修改生产实现或已安装依赖。full和resume均明确
返回`TORCHRUN_EXIT=0`，full完成4步并保存step-2 checkpoint。

| 配置 | 结果 | 状态 | 日志 |
| --- | --- | --- | --- |
| PP2 / gather诊断 / full | 4步；loss `4.878459930419922 → 4.867842197418213 → 4.858004093170166 → 4.86264705657959`；checkpoint metadata存在 | 通过 | `outputs/qwen3-30b-npu/logs/pp2_whole_gather_pair_full_20261026.log` |
| PP2 / gather诊断 / 独立resume | 两rank更新前模型、optimizer、scheduler和trainer状态失败数均为0；恢复loss `4.858004093170166、4.86264705657959`；与连续训练尾段最大差 `0.000e+00` | 通过 | `outputs/qwen3-30b-npu/logs/pp2_whole_gather_pair_resume_20261026.log` |

该基础配对未启用EMA、随机状态或loader状态快照，不作为这些状态的
恢复验收。完整状态配对另在10.1.0.90、同一容器和NPU0/1运行，使用
独立工作目录`outputs/pp2-whole-gather-state-20261026`；环境增加
`LLMTUNER_TEST_EMA=dynamic`、`LLMTUNER_TEST_RNG_STATE=1`及
`LLMTUNER_TEST_LOADER_STATE=1`，仍启用Inductor及原精度检查。
full与resume均明确返回`TORCHRUN_EXIT=0`。full完成4步并保存step-2
checkpoint；独立resume在更新前两rank比较失败数均为0，EMA、RNG、
loader、模型、optimizer、scheduler和trainer状态全部与连续运行对应
快照一致，恢复loss尾段最大差`0.000e+00`，原断言打印`all checks passed`。

完整状态配对日志：
`outputs/qwen3-30b-npu/logs/pp2_whole_gather_state_full_20261026.log`及
`outputs/qwen3-30b-npu/logs/pp2_whole_gather_state_resume_20261026.log`。
复现时设置与基础配对相同的PYTHONPATH、NPU0/1、Inductor backend、
`INDUCTOR_ASCEND_CHECK_ACCURACY=1`及独立TORCHINDUCTOR_CACHE_DIR，
再附加上述三个状态变量；分别执行：

```bash
timeout 900 torchrun --standalone --nproc_per_node=2 \
  outputs/npu_inductor_gather_probe.py full \
  outputs/pp2-whole-gather-state-20261026
timeout 900 torchrun --standalone --nproc_per_node=2 \
  outputs/npu_inductor_gather_probe.py resume \
  outputs/pp2-whole-gather-state-20261026
```

full的调试产物确认rank1后缀11反向kernel实际生成三处连续
`tl.load`及`tl.gather`（`tmp3_raw`、`tmp4_raw`、`tmp29_raw`）；resume
复用同一编译缓存。两份完整配对日志均无`CHECK ACCURACY FAILED`。
这些结果关闭该辅助诊断的训练和恢复缺口，但不扩展到生产compile、
30B目标模型、四种八卡拓扑或纯AiCore；此前生产compile失败状态保持，
仍需将根因修复正式化并按目标范围复验。

本轮同一合法设备环境下的CPU质量检查通过：
`outputs/qwen3-30b-npu/logs/cpu_suite_valid_env_20261026.log`记录
4302 passed、13 skipped、298 warnings，exit 0、192.80s。使用临时
`PYTHONPYCACHEPREFIX`执行`python -m compileall -q llmtuner`通过；
`ruff check llmtuner tests`、`ruff format --check llmtuner tests`和
`git diff --check`均通过。此前将`ASCEND_RT_VISIBLE_DEVICES`设为空的
CPU运行因CANN报Invalid device ID而产生的210个失败不计入代码结果，
保留其日志但不作为有效质量门禁。

## 2026-10-26：Inductor gather修复接入点审查及P0四拓扑两步训练

只读检查`CompileConfig`、`parallel/model_compile.py`、装配顺序和现有NPU
注意力适配。生产编译入口只将配置中的backend交给Dynamo注册表及
`torch.compile`；`regional_inductor`只包装flex区域。`masks.py`中NPU选用
eager `create_block_mask`的适配解决的是另一项BlockMask构造编译问题，
与反向kernel的负偏移索引无关。仓库没有`NPUIndexTritonKernel`专用适配入口。

2026-10-26整模型诊断在进程内包装
`torch_npu._inductor.codegen.triton.NPUIndexTritonKernel.codegen_kernel`，
对严格匹配的三处负偏移`tl.load`改为连续加载后`tl.gather`；数值断言及
独立进程恢复已通过，但属于诊断包装，不是生产修复。修复应落在有版本及
回归测试保障的torch-npu/Triton-Ascend codegen适配；不能仅新增CompileConfig
开关，也不能把针对生成源码的运行时monkeypatch并入生产路径。正式改动前须
固定依赖版本、核实上游源码和API，并以原始FX输入覆盖掩码、尾块、布局及
反向数值；随后复验整模型训练与恢复。当前没有生产代码或依赖改动，原P3
Inductor失败状态保持。

本轮在当前工作树分别完成四组Qwen3-30B两步训练。启动前检查节点及设备占用，
前两组使用10.1.0.18，后两组使用10.1.0.90。当前`available_nodes.txt`列出
10.1.0.116、10.1.0.12、10.1.0.14、10.1.0.179、10.1.0.18、10.1.0.29、
10.1.0.42、10.1.0.69、10.1.0.71和10.1.0.90。运行容器为
`hyper-parallel-env`；记录的torch版本为2.12.0+cpu、torch_npu为2.12.0。

四组均使用8卡、`dp_replicate=1`、`PP=1`、`CP=1`、全局batch 8、序列长度
4096、BF16、full activation checkpointing、seed 42、确定性训练和2步训练，
每步保存checkpoint。模型为本地Qwen3-30B-A3B，数据为LIMO。日志中的mesh
初始化参数及活动轴与请求拓扑一致；EP=4组还创建了`efsdp`轴。逐rank的
process-group成员另由2026-10-24拓扑探针验证，详见下方同名记录；探针比较了
`get_process_group_ranks()`与各rank的`DeviceMesh.mesh`。该探针和本次训练是
两项独立证据，不合并为同一次运行。`topology.py`与`parallel_config.py`在该探针后
未修改，故相同四组配置的rank分组记录仍适用。各配置的step 1/2 loss和grad norm
如下；日志未发现Traceback、ERROR或非有限值错误。

| 配置（DP shard / TP / EP） | Step 1 loss / grad norm | Step 2 loss / grad norm | 状态 | 日志 |
| --- | --- | --- | --- | --- |
| 8 / 1 / 1 | 0.53537 / 3.3731 | 0.43012 / 2.3510 | 通过（限两步训练与保存） | `outputs/qwen3-30b-npu/logs/p0_matrix_dp8_tp1_ep1_20261026.log` |
| 4 / 2 / 1 | 0.63489 / 4.2312 | 0.38214 / 2.8902 | 通过（限两步训练与保存） | `outputs/qwen3-30b-npu/logs/p0_matrix_dp4_tp2_ep1_20261026.log` |
| 4 / 2 / 4 | 0.63443 / 4.2626 | 0.38290 / 2.8798 | 通过（限两步训练与保存） | `outputs/qwen3-30b-npu/logs/p0_matrix_dp4_tp2_ep4_20261026.log` |
| 8 / 1 / 8 | 0.53600 / 3.4291 | 0.42917 / 2.3705 | 通过（限两步训练与保存） | `outputs/qwen3-30b-npu/logs/p0_matrix_dp8_tp1_ep8_20261026.log` |

checkpoint位于`outputs/qwen3-30b-npu/p0_matrix_20261026/<配置>/checkpoint/`；
每个step目录均有`.metadata`和8个rank分片。四个`torchrun`子任务退出码均为0。
日志显示ArgSort因int32/int64输入回退到AiCPU；不能据此声称全程纯AiCore。

训练完成后，两个外层调度进程因执行期间脚本被修改而报告shell语法错误；训练
子任务本身已成功完成，日志和checkpoint已逐项核验。当前
`outputs/run_p0_four_topologies_20261026.sh`通过`bash -n`。复跑单组可执行：

```bash
bash outputs/run_p0_four_topologies_20261026.sh dp8_tp1_ep1
bash outputs/run_p0_four_topologies_20261026.sh dp4_tp2_ep1
bash outputs/run_p0_four_topologies_20261026.sh dp4_tp2_ep4
bash outputs/run_p0_four_topologies_20261026.sh dp8_tp1_ep8
```

这些结果只通过P0的两步训练与checkpoint写入检查，不代表完整P0通过。尚未完成
独立进程恢复对照、预热后的step耗时分布和跨拓扑数值对齐；也不应用两步loss
变化推断收敛或性能优劣。gather生产修复仍未完成，相关P3状态保持失败。

### P0：dp_shard=8 / TP=1 / EP=8 独立进程恢复

按`available_nodes.txt`选用10.1.0.14；启动前宿主`npu-smi`未报告运行进程，
容器实测8个NPU可见，torch 2.12.0+cpu、torch_npu 2.12.0。原连续训练在
10.1.0.90完成。恢复进程在10.1.0.14运行，配置保持8卡、dp_replicate=1、
dp_shard=8、TP=1、EP=8、PP/CP=1、BF16、full AC、seed42、确定性、
batch8、seq4096；从原step-1 checkpoint的只读符号链接加载，目标总步数仍为2，
只执行step2，恢复结果写入独立目录。原step-1/2产物未覆盖。

恢复日志确认`Resuming from step 1`，加载耗时25.92秒，step2 loss/grad norm
为`0.42917 / 2.3705`；与连续训练step2日志一致。8个rank均保存step-2，
进程退出码0。使用`compare_30b_checkpoint_chunks.py`对连续训练和恢复后的
step-2逐物理chunk精确比较：21473项全部一致，failures为空；包括model 4632、
optimizer 16791、dataloader 39、rng_state 8、lr_scheduler 1、train_state 2。
RNG载荷在比较前解码。审计退出码0。日志中的`_assert_async`回退到CPU，整数
ArgSort回退到AiCPU；不声称纯AiCore。

| 项目 | 配置/结果 | 证据 |
| --- | --- | --- |
| 连续训练 | step1→2；step2 loss 0.42917、grad norm 2.3705；10.1.0.90 | `outputs/qwen3-30b-npu/logs/p0_matrix_dp8_tp1_ep8_20261026.log` |
| 独立恢复 | step1恢复后执行step2；exit 0；step-2 checkpoint有metadata及8个rank分片 | `outputs/qwen3-30b-npu/p0_matrix_20261026/dp8_tp1_ep8_resume_20261026/logs/train_20261026_024557.log` |
| 完整状态比较 | 21473 chunks精确一致，0 failures；exit 0 | `outputs/p0_matrix_dp8_tp1_ep8_resume_state_compare_20261026.log` |

复现时为恢复建立新dump目录，并将原训练`checkpoint/step-1`链接到新目录的
`checkpoint/step-1`；不得让恢复进程写入原训练checkpoint目录。训练使用：

```bash
docker exec \
  -e PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime \
  -e LLMTUNER_QWEN3_30B_PATH=/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B \
  -e LLMTUNER_DATASET_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl \
  -e LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE=1 \
  -e LLMTUNER_DATA_PARALLEL_SHARD_SIZE=8 \
  -e LLMTUNER_TENSOR_PARALLEL_SIZE=1 \
  -e LLMTUNER_EXPERT_PARALLEL_SIZE=8 \
  -e LLMTUNER_PIPELINE_PARALLEL_SIZE=1 \
  -e LLMTUNER_CONTEXT_PARALLEL_SIZE=1 \
  -e LLMTUNER_STEPS=2 -e LLMTUNER_GLOBAL_BATCH_SIZE=8 \
  -e LLMTUNER_MAX_SEQ_LEN=4096 -e LLMTUNER_DETERMINISTIC=1 \
  -e LLMTUNER_SAVE_CHECKPOINT=1 -e LLMTUNER_CHECKPOINT_INTERVAL=1 \
  -e LLMTUNER_CHECKPOINT_LOAD_STEP=1 -e LLMTUNER_ENABLE_TENSORBOARD=1 \
  -e LLMTUNER_FSDP_CPU_OFFLOAD=0 \
  -e LLMTUNER_SHARDED_TENSOR_CHECKPOINT=0 \
  -e LLMTUNER_SHARDED_EXPERT_CHECKPOINT=1 \
  -e LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/p0_matrix_20261026/dp8_tp1_ep8_resume_20261026 \
  -e MASTER_PORT=29685 hyper-parallel-env bash -lc \
  'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && timeout 1800 bash examples/train_qwen3_30b_npu.sh'

docker exec hyper-parallel-env bash -lc \
  'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/compare_30b_checkpoint_chunks.py outputs/qwen3-30b-npu/p0_matrix_20261026/dp8_tp1_ep8/checkpoint/step-2 outputs/qwen3-30b-npu/p0_matrix_20261026/dp8_tp1_ep8_resume_20261026/checkpoint/step-2' \
  > outputs/p0_matrix_dp8_tp1_ep8_resume_state_compare_20261026.log 2>&1
```

此项关闭P0中EP=8配置的独立进程恢复子项，仅适用于该拓扑和本次训练设置；其余
三种拓扑的恢复配对、预热后step耗时分布和P4跨布局数值验收仍未完成。


## 2026-10-24：gather通过版本编译产物核验

真实FX三次四输出精确通过的gather版本，ttadapter核验为0个
memref.load、11个memref.copy，未走此前失败改写的标量加载路径。
编译元数据确认num_warps=1、num_stages=1、inject_barrier_all=true，
编译hash为113c7639f0be4447385287775728601b563a661e10064b30a3cc41bbc5a32034。
ttadapter SHA256为a04b081864946c6ef86ee3b681a5a2494b9eb636517788989b7a328c0dc00546。
IR及compiler-audit.json保存于同次UUID捕获目录，后续可核对二进制
编译身份和转换路径。文本中的gather出现次数包含位置标记，不能
视为实际gather算子数量。此核验不扩展单快照独立kernel结果的范围；
整模型和生产路径尚未复验，原compile验收失败保持。


## 2026-10-24：连续加载后gather重排通过真实FX独立对照

本轮重读原始目标；上一轮为实质进展。node92启动前设备空闲。
三维最小诊断增加LLMTUNER_FX11_GATHER=1：连续加载完整128维，
随后寄存器内gather将前64维映射到后半维，保留原掩码语义。
session61853退出0，加载、乘积及最终输出maxAE均0。日志
outputs/qwen3-30b-npu/logs/fx11_mask_3d_gather_20261024.log。

真实FX独立重放仅替换三处负偏移加载为连续加载后gather，
固定原tiling4/3/4/128、warps1、stages1、内部barrier，保留原
assert_close及equal_nan=False。session44351退出0，三次重放
四个输出均有限且maxAE全部0。12份捕获及其repeat/output/variant
元数据已独立CPU审计，结果全部一致，原精度断言没有放宽。
捕获目录outputs/fx11-captures/original-gather-65a9bc425b764eb1b94599c6597a119c，
audit.json记录逐输出结果。日志
outputs/qwen3-30b-npu/logs/fx11_gather_replay_20261024.log。

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 LLMTUNER_FX11_GATHER=1 TORCHINDUCTOR_CACHE_DIR=/tmp/llmtuner-fx11-gather-replay-20261024 timeout 180 python outputs/fx11_replay_probe.py > outputs/qwen3-30b-npu/logs/fx11_gather_replay_20261024.log 2>&1
```

该结果仅覆盖一个保存快照的独立真实FX kernel，不代表整模型或
生产compile支持。未修改生产实现或已安装依赖。下一步将该
连续加载重排应用于保留原断言的整模型诊断，核实其余捕获及
kernel，再决定正式适配方式；原compile验收失败保持。


## 2026-10-24：真实FX后续捕获按运行及输出隔离

诊断入口fx11_replay_probe.py现为每次运行创建独立UUID目录，
按repeat及output序号保存张量，并嵌入变体、重复次数、输出序号
及graph hash，避免不同输出或运行覆盖。保留原kernel、输入及
assert_close；本次未重新运行硬件用例，因此不构成新的数值证据。
原py_compile因outputs已有__pycache__权限失败，改用不写缓存的
AST语法检查；后续硬件运行还须核实生成函数的捕获路径和元数据。
原compile失败保持，未修改生产实现。

补充静态验证：在node92仅执行诊断入口的源码改写赋值，不导入
归档或触发编译，生成后的真实归档AST有效；run函数首项为
capture_output的global声明，equal_nan=False及失败重抛均保留。
入口AST、两个诊断脚本Ruff及git diff --check通过。该检查不证明
捕获已在硬件上执行，也不新增数值通过证据。


## 2026-10-24：取模IR归档及捕获来源限制

原始与取模ttadapter归档至outputs/fx11-modulo-ir-audit-20261024/，
manifest记录路径及SHA256。原始367行、0个标量load、13个copy、
2个循环；取模483行、3个标量load、10个copy、10个循环。取模
循环从tensor.empty开始，后续仍有掩码select，不能仅凭初始化
形式认定NaN根因。需检查后续lowering的实际语义。

旧共享捕获fx11-first-output-20261024.pt经CPU审计含128个NaN，
全部位于前半维，参考有限；报告
outputs/fx11-legacy-capture-finite-audit-20261024.json。该文件曾被
多个实验覆盖且未嵌入run元数据，仅与最后取模运行相符，不能独立
绑定具体运行或与此前原始捕获混用。诊断入口现按索引及编译选项
变体分别命名后续捕获，避免不同变体互相覆盖；未重新运行失败用例。
Ruff及diff检查通过。生产实现未修改，compile失败保持。


## 2026-10-24：取模索引仅最小诊断通过，真实FX产生NaN

三维诊断增加LLMTUNER_FX11_MODULO_INDEX=1，在原后半维掩码下
以r%64等价替代r-64。session87088退出0：加载、乘积、最终输出
maxAE均0，数据正确位于后半维。日志
outputs/qwen3-30b-npu/logs/fx11_mask_3d_modulo_20261024.log，缓存
/tmp/llmtuner-fx11-mask3d-modulo-20261024。

真实FX归档三处负偏移使用同样改写，原tiling、barrier及阈值不变。
session62003退出1：首次首个输出实际非有限、参考有限，565/32768
元素不匹配，出现NaN；首个断言即停止，不构成三次完整比较。
日志outputs/qwen3-30b-npu/logs/fx11_modulo_replay_20261024.log，
缓存/tmp/llmtuner-fx11-modulo-replay-20261024。复现沿用对应脚本
既有命令，增加LLMTUNER_FX11_MODULO_INDEX=1及上述独立缓存/日志。

maximum与取模两种等价改写均仅在最小诊断通过，真实kernel分别
UB越界及NaN，不能提升为生产修复。后续应核查真实kernel的
加载转换及融合资源布局，保留原始有限但数值错误的对照，
避免重复已失败改写。生产实现未修改，compile失败及完整目标保持。


## 2026-10-24：非负索引真实kernel转换为嵌套标量加载

对比原重放与safe-index重放缓存：原ttadapter为367行、13个
memref.alloc；改写后为486行、10个memref.alloc。改写后的三处
负偏移加载转换成嵌套scf.for、逐元素memref.load及tensor更新，
原实现则使用memref.copy子视图。两者required_ub_bits均为0、
shared均为1，不能据此得到有效UB需求或证明越界原因。
这一变化说明等价索引改写改变了实际lowering路径；最小用例通过
不能外推到包含归约和多个输出的真实kernel。尚无生产修复。
后续需保留真实FX比较，检查标量循环的UB布局或采用保持向量加载
的等价索引表达；不原样重复已发生UB越界的运行。


## 2026-10-24：非负索引修复最小加载错位，但真实FX触发UB越界

三维最小用例增加LLMTUNER_FX11_SAFE_INDEX=1，将掩码外的
r-64改为maximum(r-64,0)，保留原掩码。session65852退出0：
加载、乘积及最终输出maxAE均0，mismatch_count为0，数据正确
出现在后半维。日志outputs/qwen3-30b-npu/logs/fx11_mask_3d_safe_20261024.log，
缓存/tmp/llmtuner-fx11-mask3d-safe-20261024。该结果仅为最小诊断。

随后在真实FX归档的三处负偏移加载中应用同样的等价非负索引，
固定原tiling、保留barrier及原assert_close。session96076退出1，
首次执行触发VEC读写UB越界，ACL同步错误507035，尚未获得数值
比较结果。不能将最小用例成功提升为真实kernel修复。日志
outputs/qwen3-30b-npu/logs/fx11_safe_replay_20261024.log，缓存
/tmp/llmtuner-fx11-safe-replay-20261024。进程退出后npu-smi确认
设备Health为OK且无运行进程，未重启设备或重复执行失败命令。

复现命令沿用各自脚本的既有命令，增加LLMTUNER_FX11_SAFE_INDEX=1，
分别使用上述缓存及日志。生产实现、已安装依赖及原归档未修改。
后续应检查等价改写后真实kernel的UB布局和资源需求，再决定修复
路径；原compile失败和完整P0–P4验收范围保持。


## 2026-10-24：关闭select analysis未修复三维加载错位

三维最小诊断增加LLMTUNER_FX11_NO_SELECT_ANALYSIS=1，保持输入、
布局、barrier及精确断言。session94816退出1，加载数据仍出现在
前半维，后半维全零，最终16384/32768元素不匹配、maxAE32704。
缓存确认enable_select_analysis=false、inject_barrier_all=true，
编译hash为cb9f839ff6508928724ff91f0f64b4949b82e21744c22d9dbba1cefe10ceb88b。
此选项未改善当前最小复现，不重复相同实验。

复现沿用三维诊断命令，增加LLMTUNER_FX11_NO_SELECT_ANALYSIS=1，
缓存/tmp/llmtuner-fx11-mask3d-no-select-20261024，日志
outputs/qwen3-30b-npu/logs/fx11_mask_3d_no_select_20261024.log。
报告及保存张量由本次运行更新，先前结果仍在各自日志中。
下一步应检查三维目的子视图偏移的后续lowering，或用等价非负索引
改写做真实FX对照，不能以最小用例替代正式验收。生产实现未修改，
原compile失败及完整P0–P4目标保持。


## 2026-10-24：三维masked-load数据精确错置至前半维

在三维分阶段诊断中保存加载、乘积及最终张量，session84955退出1。
加载非零元素16384个，前半维最大幅值32704，后半维全部为零；
CPU审计进一步确认加载前半维与应加载到后半维的数据逐元素一致，
maxAE0，最终select输出全部为零。这比此前仅凭maxAE判断更具体：
该最小用例的加载数据被精确放到了错误的半维，随后后半维select
消除了贡献。尚未证明具体出错pass，也不能直接外推其他加载布局。

张量outputs/fx11-mask-3d-stages-20261024.pt，审计
outputs/fx11-mask-3d-shift-audit-20261024.json；日志
outputs/qwen3-30b-npu/logs/fx11_mask_3d_capture_20261024.log。
命令沿用三维stages诊断及缓存，仅日志改为capture日志。
下一步检查三维memref.copy子视图的目的偏移转换，并以该最小
复现和真实FX双重验证候选修复。未修改生产代码，compile失败保持。


## 2026-10-24：三维最小用例在加载阶段已出现错误

在三维最小诊断中增加加载值和乘积输出，保持相同输入、布局和barrier。
session57843退出1：load_max_ae、product_max_ae和最终maxAE均32704，
最终16384/32768元素不匹配，输出有限。中间输出改变了kernel，
但仍复现原错误，表明该诊断中加载阶段已不正确；不能继续将负号
或select本身视为最早已证实的故障点。尚未保存加载张量逐元素分布，
不能仅凭maxAE断言全部加载值为零。

日志outputs/qwen3-30b-npu/logs/fx11_mask_3d_stages_20261024.log，
报告outputs/fx11-mask-3d-results-20261024.json已由本次结果更新，
上一版结果仍保留在原日志。复现沿用三维最小命令，缓存改为
/tmp/llmtuner-fx11-mask3d-stages-20261024，日志改为上述stages日志。
下一步对比三维加载的广播维度与动态偏移转换，保留真实FX复现。
未修改生产实现，完整目标及compile失败状态保持。


## 2026-10-24：三维广播负偏移最小用例复现错误

上一轮属于进展；本轮重读原始目标并确认node92无NPU作业后，新增
outputs/fx11_mask_3d_probe.py。保留4×3×128块、head步长8192、
sequence步长128、sin广播、负偏移masked load及重复select，
使用可精确表示的整数输入，直接比较负号分支。启用内部barrier。
session66570退出1：输出有限，但16384/32768元素不匹配，
maxAE32704，最大相对误差1。相比此前一维动态用例通过，
三维结构已足以触发错误，无需完整模型或融合归约。尚未分别
观测此三维用例的加载、乘法和select中间结果，不能指定出错pass。

复现（node92容器仓库目录）：

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/llmtuner-fx11-mask3d-20261024 timeout 180 python outputs/fx11_mask_3d_probe.py > outputs/qwen3-30b-npu/logs/fx11_mask_3d_20261024.log 2>&1
```

结果outputs/fx11-mask-3d-results-20261024.json。该精确比较仅用于
最小诊断，不替换真实FX或正式阈值。下一步对三维加载及广播逐阶段
输出，并比较head维为1与4的触发边界；原compile失败保持。


## 2026-10-24：负偏移masked-load最小用例的触发边界

新增诊断outputs/fx11_mask_minimal_probe.py，分别输出掩码加载、乘法、
负号和select结果，以整数可精确表示的输入逐元素比较，仅用于定位，
不替代真实FX验收。首次静态单行版本session10161退出1，在
ConvertTritonIRToLinalgIR报expected offsets to be non-negative, got -64，
未执行kernel。日志outputs/qwen3-30b-npu/logs/fx11_mask_minimal_20261024.log。

随后加入program_id动态行偏移，执行两行；session19572退出0，
四个中间结果全部有限且maxAE均0，精确assert_close通过。
报告outputs/fx11-mask-minimal-results-20261024.json，日志
outputs/qwen3-30b-npu/logs/fx11_mask_minimal_dynamic_20261024.log。
命令（node92容器仓库目录，启动前设备已确认空闲）：

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/llmtuner-fx11-minimal-dynamic-20261024 timeout 180 python outputs/fx11_mask_minimal_probe.py > outputs/qwen3-30b-npu/logs/fx11_mask_minimal_dynamic_20261024.log 2>&1
```

动态最小用例未复现原融合图的负号贡献缺失，不能将该问题归结为所有
负偏移masked-load均错误。后续需保留原三维广播、重复select及融合
归约结构进一步缩小复现。未修改生产实现，原compile失败保持。


## 2026-10-24：禁用紧耦合缓冲区复用未改善后缀11

独立复现增加LLMTUNER_FX11_NO_BUFFER_REUSE=1，仅设置
disable_tightly_coupled_buffer_reuse=True，原tiling、输入、barrier及
assert_close不变。session43337退出1，首次输出仍431/32768不匹配，
maxAE为5.360797877074219e-05，最大误差位置(0,0,5,64)。
缓存确认该选项及inject_barrier_all均true，编译hash为
627b62b5308265985cb4e40923be6f065a0878fb10c1dd889e3279ea43b56269。
该选项无法修复当前快照，不能据此排除其他缓冲区或编译阶段的问题。

复现沿用独立重放命令，增加LLMTUNER_FX11_NO_BUFFER_REUSE=1，
缓存/tmp/llmtuner-fx11-no-buffer-reuse-20261024，日志
outputs/qwen3-30b-npu/logs/fx11_no_buffer_reuse_20261024.log。
本轮取得独立复现、负号贡献缺失和两项选项对照证据，未修改生产实现。
后续应转向最小masked-load/select复现，避免继续盲目切换选项。
原compile失败及完整P0–P4目标保持。


## 2026-10-24：后缀11转换前后IR已归档

从固定原tiling的独立重放缓存提取ttir和ttadapter，保存至
outputs/fx11-ir-audit-20261024/，manifest.json记录原路径及SHA256。
ttir包含负偏移masked load、负号arith.subf及后半维select；
ttadapter仍包含对应负号及tensor.insert_slice，不能仅据IR中出现
这些操作推断实际执行正确，也不能声称已证明该阶段删除了负号分支。
ttir SHA256为79938f66c45602f13188519f4fa24532e3ce7ecdace53638a877a441850fa8e5，
ttadapter为2fe96ca6bb090445fb3a0086f9bbf3b77d91deb995e34c6a451b897429126de0。
本轮无生产修改；独立复现、负号贡献缺失及掩码fallback无效的证据保持。
下一步构造负偏移masked load与select的最小对照，并检查实际编译产物，
不能用新的手写参考替代既有真实FX验收。


## 2026-10-24：掩码fallback转换未修复后缀11负号分支缺失

检查独立复现的ttadapter，中间表示仍含负号arith.subf、动态掩码加载
及tensor.insert_slice，尚不能确定实际贡献在哪个阶段丢失。诊断脚本
增加可选LLMTUNER_FX11_MASK_FALLBACK=1，仅切换
enable_mask_fallback_conversion，保持原tiling、barrier和assert_close。
session46550退出1，首个输出仍431/32768不匹配、maxAE
5.360797877074219e-05，与原复现完全相同。缓存元数据确认
enable_mask_fallback_conversion=true、inject_barrier_all=true，
编译hash为8f921b0cfa09fc2b516a04515e681992799f1a6acf0368cbfe37b739c6185212。
故该选项对照没有修复当前快照，不再重复相同实验。

复现沿用独立重放命令，增加LLMTUNER_FX11_MASK_FALLBACK=1，
缓存/tmp/llmtuner-fx11-mask-fallback-20261024，日志
outputs/qwen3-30b-npu/logs/fx11_mask_fallback_20261024.log。
未修改生产实现或已安装依赖。后续需建立更小的负号掩码加载复现，
核实转换前后语义及同步；原compile精度失败保持。


## 2026-10-24：后缀11误差精确对应RoPE负号分支缺失

在独立重放原断言前保存首个输出及真实FX参考，session9889退出1，
保留原阈值。CPU索引审计得到431个超阈值元素全部位于最后维64–127，
前半维maxAE为0。不是仅依据超阈值计数：进一步用同一保存输入直接
计算cos项，实际输出后半维与cos项逐元素完全一致（maxAE0）。
应加入的负号分支为-(输入前半维×sin前半维)，其最大幅值
5.360797877074219e-05，与实际最大误差完全一致。错误符号假设
也不能解释结果；当前证据精确支持该保存快照的负号分支贡献缺失。
尚未证明缺失发生在哪一编译阶段，不能据此直接修改生产实现。

证据：outputs/fx11-first-output-20261024.pt、
outputs/fx11-error-index-audit-20261024.json、
outputs/fx11-negative-branch-audit-20261024.json；运行日志
outputs/qwen3-30b-npu/logs/fx11_replay_capture_20261024.log。
复现命令沿用下一节独立重放命令，日志改为上述capture日志。
下一步在独立用例中检查负号分支的load/select及后端转换，
以真实FX和原断言复验，不放宽阈值。原compile失败保持。


## 2026-10-24：后缀11固定原tiling独立重放复现训练误差

新增诊断outputs/fx11_replay_probe.py，保留归档文件，仅在进程内映射
设备1到0并以CPU加载自有保存输入。固定原训练单kernel缓存jo中的
X1BLOCK4、X1BLOCK_SUB3、X0BLOCK_SUB4、num_warps1、num_stages1，
R2BLOCK_SUB为kernel内常量128；继续启用inject_barrier_all。
归档原assert_close阈值不变，将捕获后打印改为重新抛出，避免退出0掩盖失败。

session44275退出1，首次重放首个输出实际与参考均有限，maxAE为
5.360797877074219e-05，431/32768元素不匹配，最大误差位置
(0,0,5,64)，最大相对误差210.6627197265625。该误差与full日志
最后一次后缀11输出0比较完全相同。首个断言失败即停止，因此不能声称
完成三次或四个输出的完整重放。此证据建立独立kernel复现，未定位根因。

命令（node92容器仓库目录）：

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/tmp/llmtuner-fx11-replay-20261024 timeout 180 python outputs/fx11_replay_probe.py > outputs/qwen3-30b-npu/logs/fx11_replay_20261024.log 2>&1
```

原compile失败保持。下一步在这个独立复现中检查逐元素错误的索引分布、
tiling边界与加载掩码，保留原阈值及失败证据；不重复整模型编译。


## 2026-10-24：后缀11真实FX参考审计排除同量级eager误差

上一轮为进展：全barrier诊断完成训练及独立恢复，但kernel精度失败仍在。
本轮重读原始目标，定位真实FX归档
/tmp/llmtuner-whole-barrier-20261024/traced_fx_graph_cache/
cca5hw7wddefsacllmkvxl6zgwwecyeytg2d66bbjwcrrz4ctjfi/。
该图包含RoPE反向乘加及两个末维归约，返回四个输出。仅通过AST提取
GraphModule并加载保存输入，不导入归档或触发编译。八个输入和CPU输出
全部有限，输出最大幅值约4.03e-5至8.32e-5，已有kernel误差与信号同量级，
不能未经验证归为普通舍入误差。审计保存outputs/fx11-cpu-input-audit-20261024.json。

node92 npu-smi确认无作业后，单卡可见设备0执行同一真实FX的NPU eager，
session52883退出0。三次重放四个输出均有限；两个逐元素输出相对CPU
maxAE均0，两个归约输出分别7.275957614183426e-12和
6.366462912410498e-12。报告outputs/fx11-eager-reference-audit-20261024.json。
这排除当前保存快照的eager参考具有此前同量级误差，不能证明编译kernel正确。
未修改生产实现或归档，原compile失败保持。

源码核对还确认后端disable_fma只在force_simt_only分支传递给编译器，
而目标缓存元数据为SIMD/aiv、force_simt_only=false。enable_fp_fusion虽为true，
当前Python后端仅见选项定义，尚无证据说明切换会影响此路径；因此未运行
无依据的关闭FMA实验。下一步应固定原launcher配置，独立重放后缀11，
逐输出比较真实FX，并核验生成代码的掩码、布局及同步边界。


## 2026-10-24：全kernel内部barrier整模型诊断完成训练及恢复

诊断入口 outputs/npu_inductor_barrier_probe.py 在进程内包装 triton.compile，
为全部kernel设置 inject_barrier_all=True，再执行既有 fx_api_probe 诊断链。
保留原训练有限性及checkpoint断言，未修改生产实现或已安装依赖。
node92启动前npu-smi确认无作业。此用例为两卡PP辅助模型，不能外推30B。

full session97536退出0：完成4步并保存step-2 checkpoint，loss依次为
4.878459930419922、4.868197441101074、4.857847213745117、
4.862201690673828，未触发此前step3的非有限保护。
resume session15868退出0：独立进程恢复2步，原检查打印all checks passed，
恢复loss与连续训练尾段max abs diff为0。仅说明该诊断路径训练和恢复完成。

full日志仍有68条CHECK ACCURACY FAILED，全部来自
triton_unk_fused_add_matmul_mul_neg_slice_slice_backward_sum_transpose_unsqueeze_view_11，
最大绝对误差8.984571468317881e-05，没有NaN误差报告；resume亦有该kernel
不匹配。后端比较失败不会导致进程非零退出，且部分kernel缺少FX映射，
故退出0及checkpoint一致不代表compile数值验收通过。CPU/AiCPU回退仍存在。
原compile失败保持；下一步分析后缀11真实FX及生成代码，不能把全barrier
诊断直接提升为正式支持或放宽比较阈值。

复现：node92容器hyper-parallel-env，仓库目录，环境如下；两阶段各
timeout900，torchrun --standalone --nproc_per_node=2
outputs/npu_inductor_barrier_probe.py {full|resume} outputs/pp2-whole-barrier-20261024。

```bash
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
ASCEND_RT_VISIBLE_DEVICES=0,1
TORCHINDUCTOR_NPU_BACKEND=default
LLMTUNER_TEST_COMPILE_BACKEND=inductor
TORCH_COMPILE_DEBUG=1
TORCHINDUCTOR_CACHE_DIR=/tmp/llmtuner-whole-barrier-20261024
INDUCTOR_ASCEND_CHECK_ACCURACY=1
```

TORCH_COMPILE_DEBUG_DIR分别为outputs/pp2-whole-barrier-debug-20261024及
outputs/pp2-whole-barrier-resume-debug-20261024。日志分别为
outputs/qwen3-30b-npu/logs/pp2_whole_barrier_20261024.log及
outputs/qwen3-30b-npu/logs/pp2_whole_barrier_resume_20261024.log。


## 2026-10-24：同步阶段IR导出未取得有效IR

八份捕获的24次barrier真实FX比较已完成，见下一节。随后诊断脚本
增加LLMTUNER_FX13_DEBUG_IR=1，仅为目标kernel设置debug=True。
session7551终止exit0，三次输出及参考有限、maxAE均0。命令沿用
barrier单快照命令，缓存改为/tmp/llmtuner-fx13-barrier-ir-20261024，
增加LLMTUNER_FX13_DEBUG_IR=1、TRITON_KERNEL_DUMP=1，日志
outputs/qwen3-30b-npu/logs/fx13_barrier_ir_20261024.log。

实际编译命令包含enable-hivm-inject-barrier-all-sync=True及
bishengir-print-ir-after=hivm-graph-sync-solver，但导出的
kernel.npuir.mlir仅375字节，内容为bisheng参数废弃及target triple
警告，没有有效IR。因此不能据此比较同步指令或声称定位缺失barrier。
原始文本保存outputs/fx13-barrier-sync-ir-20261024.mlir，仅作导出
失败证据。后续需核实该编译器实际pass名称及IR打印选项；不要重复
当前无效导出命令。脚本Ruff及diff检查通过，未修改生产代码或依赖，
原compile失败保持。


## 2026-10-24：内部barrier真实FX对照扩展至八份捕获

诊断脚本outputs/fx13_barrier_probe.py新增可选
LLMTUNER_FX13_CAPTURE_MATRIX=1，加载既有八份rank0捕获，
每份重复三次，使用后缀13真实FX图及参数映射计算NPU参考，
固定原tiling4/3/4/128、num_warps1、num_stages1并注入内部barrier。
结果写入后显式断言全部输出有限且maxAE为0；该精确比较仅为诊断，
不新增或替换正式验收阈值。脚本Ruff通过。

node92启动前npu-smi确认无作业；session72083终止exit0。
八份捕获共24次的实际及参考输出全部有限，maxAE全部0。
结果outputs/fx13-barrier-matrix-20261024.json，日志
outputs/qwen3-30b-npu/logs/fx13_barrier_matrix_20261024.log。

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/tmp/llmtuner-fx13-barrier-20261024 LLMTUNER_FX13_CAPTURE_MATRIX=1 timeout 180 python outputs/fx13_barrier_probe.py > outputs/qwen3-30b-npu/logs/fx13_barrier_matrix_20261024.log 2>&1
```

该结果将内部barrier对照由一个快照扩展到八个已有捕获；尚未定位
具体同步缺陷，未验证整模型或其他反向融合kernel，不能关闭compile
失败。下一步比较生成IR同步边界，并在保留全部原断言的整模型
诊断中核实影响。未修改生产实现或已安装依赖。


## 2026-10-24：内部barrier注入消除后缀13当前快照误差

先核对历史记录，避免重复输入同步及multibuffer关闭实验。本轮
检查Ascend Triton compiler.py，确认inject_barrier_all编译选项映射
至enable-hivm-inject-barrier-all-sync。新增诊断脚本
outputs/fx13_barrier_probe.py：固定原训练tiling4/3/4/128及
num_warps1、num_stages1，仅在目标kernel的triton.compile调用中
加入inject_barrier_all=True，保留真实FX参考及归档比较断言。
未修改生产实现、归档或已安装依赖；脚本Ruff通过。

node92启动前npu-smi确认无作业；session6037终止exit0。
三次输出及参考均有限，maxAE均0，未打印归档assert_close失败。
命令（容器仓库目录）：

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/tmp/llmtuner-fx13-barrier-20261024 timeout 180 python outputs/fx13_barrier_probe.py > outputs/qwen3-30b-npu/logs/fx13_barrier_20261024.log 2>&1
```

结果outputs/fx13-barrier-results-20261024.json。编译缓存元数据
确认inject_barrier_all=true、num_warps1、num_stages1；编译hash
6432d3e8f0608c815bf4c84825436c854f0bf610416c19b3d4b96915b3c070e9，
npubin SHA256为df2128b42432b7622da5e7b7ca9dae98c7e18022e94e182a62a0986b6e5654ec。
该结果支持内部同步相关假设，但仅覆盖同一归档快照的三次独立重放，
未证明具体缺失barrier位置，未复验其他捕获或整模型，也不是正式修复。
下一步应比较启用前后生成IR的同步边界并扩大既有捕获验证，原compile
失败及其他验收缺口保持。


## 2026-10-24：强制原训练tiling复现后缀13误差

重读目标后，node92的npu-smi确认无作业。在诊断进程内覆盖
NPUCachingAutotuner._precompile_worker，仅对后缀view_13、尚未
产生compile_results的实例设置唯一triton.Config：Y0BLOCK4、
Y0BLOCK_SUB3、X2BLOCK_SUB4、X3BLOCK_SUB128、split_axis[0]、
split_blocks[4]、num_warps1、num_stages1，随后调用原预编译实现。
使用同一真实FX归档及data.pth，设备映射和CPU加载沿用前述重放。
每次run重新加载保存输入，连续运行三次，并显式打印输出有限性、
maxAE及实际launcher。未修改归档、生产实现或已安装依赖。

session64415已终止exit0，表示诊断收集完成，不表示数值通过：
归档run捕获并打印assert_close失败。三次实际及参考输出均有限，
maxAE依次为4.1288654756499454e-5、1.2699345350265503、
0.9995847940444946；不匹配元素分别56、1191、1040/32768。
实际launcher打印确认上述唯一tiling及num_warps1、num_stages1，
故该次错误不再能用自动调优选择不同配置解释。日志：
outputs/qwen3-30b-npu/logs/fx13_forced_original_config_20261024.log。

环境为PYTHONPATH=.:outputs/baseline-runtime、
ASCEND_RT_VISIBLE_DEVICES=0、TORCHINDUCTOR_CACHE_DIR=
/tmp/llmtuner-fx-squeeze-20261024，timeout180。图映射仍为
cq2i7wsw6itjqkjzhwavimdy5dr5qslvekznqai36ogxokcobz25。
相同保存输入的不同重放结果提示原tiling执行不稳定，但本次未逐次
核验输入传输、存储及实际二进制身份，不据此确定后端内部根因。
原compile失败保持；下一步应保留这一真实FX对照复现，检查原tiling
生成二进制或后端同步边界，不重复既有手写公式诊断。


## 2026-10-24：后缀13训练与重放tiling差异已定位

读取当前缓存及后端选择实现，确认设备1独立kernel文件
/tmp/llmtuner-fx-squeeze-20261024/qg/cqghgbkbyzpbtogbbyea56rlzu4wczvybs3v3432apcrwomqursy.py
对应best_config为Y0BLOCK4、Y0BLOCK_SUB3、X2BLOCK_SUB4、
X3BLOCK_SUB128，num_warps1、num_stages1、time_taken_ms0。设备0
重放文件sj/csjyshe3xg7djnfeibpuf3ij7lpcqpbdlcttwgvieiajbpjmsfce.py
对应配置为Y0BLOCK1、Y0BLOCK_SUB1、X2BLOCK_SUB4、X3BLOCK_SUB64，
num_warps1、num_stages1、time_taken_ms207。两者图哈希相同。

源码证据：既有split_probe诊断将确定性模式下的所有kernel固定为
首个配置；后端原生_should_skip_autotune_for_determinism仅跳过
reduction/persistent reduction，pointwise仍benchmark自动选配置。
上轮归档重放未应用该诊断补丁，因此成功结果与训练采用不同tiling，
不能作为原配置修复证据。下一步应明确强制原配置并记录实际launcher，
而非继续默认自动调优重放。未修改依赖或生产代码，compile失败保持。

缓存搜索原始结果保存outputs/fx13-cache-config-audit-20261024.json；
其中p6整图文件含多个kernel，其目录best_config属于另一reduction
kernel，不能绑定到后缀13。qg和sj已通过单kernel定义及设备元数据
精确核实，以上结论只使用这两项，不使用p6的误匹配。


## 2026-10-24：后缀13归档独立重放与真实FX参考一致

启动前node92的npu-smi确认无作业。直接运行后缀13真实FX归档，
首次session7498 exit1：归档设备元数据index1在单卡可见环境无效；
进程内映射index0后session18782 exit1：归档torch.load仍尝试原设备1。
两次均未获得kernel输出比较，失败日志分别为日志目录下
fx13_archived_kernel_20261024.log和
fx13_archived_kernel_devicefix_20261024.log。

随后仅在进程内将归档加载改为map_location=cpu、weights_only=False，
保留原归档文件及data.pth，session52880 exit0，日志
outputs/qwen3-30b-npu/logs/fx13_archived_kernel_loadfix_20261024.log。
显式增加实际/参考有限性及maxAE打印后，session82608 exit0，输出
EXPLICIT_AUDIT True True 0.0，证明本次独立重放输出与FX参考一致。
补充审计曾因临时字符串换行SyntaxError在执行前退出，修正后才获得上述结果。

复现环境：容器仓库目录，PYTHONPATH=.:outputs/baseline-runtime，
ASCEND_RT_VISIBLE_DEVICES=0，TORCHINDUCTOR_CACHE_DIR=
/tmp/llmtuner-fx-squeeze-20261024，timeout180；归档位于该缓存
traced_fx_graph_cache/cq2i7wsw6itjqkjzhwavimdy5dr5qslvekznqai36ogxokcobz25
同名.py。进程内仅替换index=1为index=0及上述torch.load参数，
执行归档run，使用其真实FX参考及原assert_close。

该运行未记录并核对与原训练相同的实际tiling/launcher选择，且设备
映射及进程环境不同，因此不能证明原训练kernel已修复或排除其错误。
下一步应比较两次实际launch配置与运行上下文，避免把独立重放成功
外推为原整模型通过。未修改生产代码或依赖，原compile失败保持。


## 2026-10-24：后缀13真实FX图NPU eager参考有限且与CPU一致

node92启动前npu-smi确认无作业，使用ASCEND_RT_VISIBLE_DEVICES=0，
通过AST仅提取后缀13映射的GraphModule及输入映射，不导入归档模块
或触发Inductor编译。加载同一data.pth，按call_args_mapping和
mismatch_indices_shapes恢复五个输入，分别执行CPU及NPU eager图。
session30228终止exit0；三次NPU eager输出均有限、NaN计数0，
对CPU参考maxAE均为0。结果为
outputs/fx-qnorm-npu-reference-audit-20261024.json。

映射cq2i7wsw6itjqkjzhwavimdy5dr5qslvekznqai36ogxokcobz25的
同一保存输入已证明CPU和NPU eager参考一致；下一步应对该快照的
真实编译kernel输出进行对应检查。该证据仍不能与原训练日志末次
NaN调用拼接，也不证明所有输入或整模型编译正确。未修改生产代码
或依赖，原compile失败保持；保留CANN padding分配警告。


## 2026-10-24：真实FX归档CPU参考审计区分输入NaN传播

对上轮真实traced_fx_graph_cache进行CPU审计，仅通过AST提取
GraphModule及参数映射，未导入归档模块、启动NPU编译或修改依赖。
按call_args_mapping和mismatch_indices_shapes重建五个FX输入，
执行保存的真实FX forward；审计exit0，结果保存于
outputs/fx-qnorm-cpu-reference-audit-20261024.json。

后缀13映射cq2i7wsw6itjqkjzhwavimdy5dr5qslvekznqai36ogxokcobz25：
五个保存输入均有限，CPU参考输出有限、NaN计数0。后缀10映射
cqr4vez7jygbpudzshyjnu4resv2xmpthwaydx3yeypp5shgvjq6：
保存的梯度输入已有512个NaN，归约输入已有4个NaN，CPU参考输出
有512个NaN。因此后缀10的归档证明上游非有限值已进入该kernel，
不能将日志NaN直接解释为该kernel从有限输入生成NaN。

另核实后端compare_outputs使用equal_nan=True，check_accuracy_triton
不传播比较布尔结果为训练失败；仍须依赖Trainer原有有限性保护。
该CPU审计不证明原NPU参考输出有限，也不确定后缀13日志末次
NaN属于哪一次保存快照，禁止拼接为同一次运行调用的因果结论。
下一步优先检查后缀13对应执行与上游输出；原compile失败保持。


## 2026-10-24：真实FX参考比较执行，反向kernel误差及step3非有限复现

诊断脚本outputs/npu_inductor_fx_api_probe.py新增squeeze闭包适配：
对照当前PyTorch原生实现，将evaluate_static_shape替换为guard_int，
将单例维度判断替换为guard_or_false(sympy.Eq(s, 1))；保留torch-npu
的view捕获及注册包装，仅修改诊断进程，未修改依赖或生产实现。Ruff通过。

node92启动前npu-smi确认无作业。session36703已终止exit1；两个rank
均在step3触发Trainer原有Loss or gradient norm is not finite保护，
停止于optimizer更新前。日志为outputs/qwen3-30b-npu/logs/pp2_fx_squeeze_20261024.log。

完整命令沿用下一节，缓存改为/tmp/llmtuner-fx-squeeze-20261024，
debug目录outputs/pp2-fx-squeeze-debug-20261024，训练目录
outputs/pp2-fx-squeeze-20261024，日志改为上述路径，其余参数不变。

本次生成真实traced_graph_hash及traced_graph_dir，目录为
/tmp/llmtuner-fx-squeeze-20261024/traced_fx_graph_cache；日志实际出现
CHECK ACCURACY FAILED，结合已核实的check_accuracy_triton调用链，
证明部分kernel已执行FX参考与真实launcher比较，不再仅有环境变量证据。
反向融合kernel后缀11的四个输出报告误差；后缀13最大有限绝对误差
至少0.037398386746644974，末段后缀13及10比较出现NaN，随后step3
训练保护终止。后端诊断容差rtol1.3e-6、atol1e-5不替代正式验收标准。

部分前向kernel仍报告无法重建FX图或无有效dump_path，故不能宣称
所有kernel均被比较；也未确定NaN首先发生在编译输出还是参考输出。
下一步可读取本次真实FX映射与对应kernel缩小错误边界，无需重复
旧占位映射重放。该运行包含诊断monkeypatch，不新增功能通过项，
原compile失败保持。


## 2026-10-24：FX concat诊断适配后推进至squeeze旧API失败

对照当前PyTorch原生ir.ConcatKernel.create，确认非拼接维度使用
check_equals_and_simplify检查相等并返回简化表达式。新增忽略目录诊断
outputs/npu_inductor_fx_api_probe.py，仅在进程内将torch-npu concat捕获
的guard_equals调用替换为该接口，随后执行既有dual_codegen诊断；
不修改已安装依赖或生产实现，保留原训练断言。脚本Ruff通过。

启动前node92的npu-smi确认无作业；session87355已终止exit1。完整命令
沿用下一节FX精度检查命令，将入口替换为outputs/npu_inductor_fx_api_probe.py，
缓存改为/tmp/llmtuner-fx-api-20261024，debug目录改为
outputs/pp2-fx-api-debug-20261024，训练目录改为
outputs/pp2-fx-api-20261024，日志改为
outputs/qwen3-30b-npu/logs/pp2_fx_api_20261024.log；其余环境及参数不变。

本次越过concat旧API失败，随后在aten.squeeze.dim lowering退出：
torch-npu lowering_fx.py第1077行调用不存在的evaluate_static_shape。
源码核实当前PyTorch原生squeeze使用guard_int处理dim，并用
guard_or_false(sympy.Eq(s, 1))处理单例维度；torch-npu捕获版本同时
保留旧evaluate_expr写法，不能仅补一个静态shape别名就认定语义一致。
尚未到达Q norm反向或证明FX参考调用执行；原compile数值失败保持，
该进展仅为诊断入口兼容性证据，不新增验收通过项。


## 2026-10-24：FX精度检查诊断遇到concat捕获API不兼容

上一轮仅汇总四组已有训练结果，未推进剩余验收。本轮重读原始目标及
开发指南后，node92启动前npu-smi确认八卡无运行进程，使用NPU0/1
启动独立缓存的FX精度检查诊断。session50392已终止，exit1。
日志：outputs/qwen3-30b-npu/logs/pp2_fx_accuracy_20261024.log。

容器仓库目录完整运行命令：

```bash
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_NPU_BACKEND=default LLMTUNER_TEST_COMPILE_BACKEND=inductor TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR=/tmp/llmtuner-fx-accuracy-20261024 TORCH_COMPILE_DEBUG_DIR=outputs/pp2-fx-accuracy-debug-20261024 INDUCTOR_ASCEND_CHECK_ACCURACY=1 timeout 900 torchrun --standalone --nproc_per_node=2 outputs/npu_inductor_dual_codegen_probe.py full outputs/pp2-fx-accuracy-20261024 > outputs/qwen3-30b-npu/logs/pp2_fx_accuracy_20261024.log 2>&1
```

实际在前向mask构造的aten.cat.default lowering退出：已安装torch-npu
codegen/ir_fx.py第550行调用V.graph.sizevars.guard_equals，但当前
PyTorch SizeVarAllocator无此属性。尚未到达目标Q norm反向kernel，
不能认定FX参考比较已执行，也不作为原NaN问题的复现或修复。
下一步需对照当前PyTorch符号约束API，核实该捕获适配的等价实现，
再决定诊断修复；本轮未修改生产实现或已安装依赖。


## 2026-10-24：Q norm后端FX精度检查入口核实

只读检查当前安装的npu_compare.py与npu_triton_heuristics.py：
check_accuracy_triton依赖get_triton_fx_graph_call加载实际映射目录；
映射不存在时返回None，不执行FX参考比较。debug_kernel_in_run仅
保存kernel前后参数，不提供kernel内部中间结果，与已有捕获范围重复。
针对归档rank0目标Q norm kernel进行AST精确提取，确认
traced_graph_hash=TRACED_GRAPH_HASH、traced_graph_dir=TRACED_GRAPH_DIR，
均为占位符，不能直接用该旧产物调用后端FX参考检查。
核查结果保存于outputs/qnorm-fx-debug-availability-20261024.json，
本地审计exit0；未启动硬件任务或修改依赖。
当前依赖配置表明INDUCTOR_ASCEND_DUMP_FX_GRAPH或
INDUCTOR_ASCEND_CHECK_ACCURACY会启用FX捕获；后续若重新编译诊断，
必须先验证实际映射文件生成及参考调用生效，不能仅凭环境变量认定通过。
本轮未获取内部错误边界，原compile失败及其他验收缺口保持。

## 2026-10-24：固定存储重放的输入完整性核验

诊断脚本新增 LLMTUNER_REPLAY_CHECK_INPUTS=1，仅允许独立输出模式，
每次 kernel 执行后逐项将五个输入与捕获原值精确比较。启动前 node92
设备空闲；session18077 终止 exit0，表示诊断收集完成。沿用固定存储
完整命令，额外设置 LLMTUNER_REPLAY_CHECK_INPUTS=1；日志为
outputs/qwen3-30b-npu/logs/qnorm_replay_check_inputs_20261024.log。
SUB3/4 各24次的全部五个输入均保持原值，输出均有限；最大 AE 分别
0.26084139943122864 与 0.0012420632410794497。两组各16次相邻
输出比较均不精确。因此本次固定存储对照中，输入被改写不能解释
输出变化；该观测仍不能确定后端内部的首个错误边界。
结果为日志目录下
qnorm_isolated_replay_sync_input_deterministic_fixed_storage_check_inputs_matched_options_separate_output_20261023.json。
诊断脚本 Ruff 通过；未修改生产实现、依赖或验收阈值，原 compile
失败保持。该分支已有多轮对照，不继续重复相同输入/选项试跑；
后续需取得编译后执行或后端中间结果的新证据再推进。

## 2026-10-24：Q norm固定输入输出存储对照

诊断脚本新增可选LLMTUNER_REPLAY_FIXED_STORAGE=1，仅允许独立输出模式；
每份输入每种SUB复用五个输入及输出地址，每次重置输出为NaN，记录
地址及相邻输出精确比较。默认行为不变，不改生产实现或依赖。
启动前node92无NPU作业；session71939终止exit0，表示收集完成。
完整环境沿用上一节确定性对照命令，额外设置
LLMTUNER_REPLAY_FIXED_STORAGE=1，日志改为
outputs/qwen3-30b-npu/logs/qnorm_replay_fixed_storage_20261024.log。

离线审计确认每份捕获每种SUB的三次输入地址和输出地址均不变。
SUB3/4各24次输出均有限，最大AE分别0.25527143478393555与
0.00031085326918400824；每组16次相邻输出比较全部不精确。
因此重新分配不能充分解释重复结果变化；尚未在该对照逐次检查
kernel执行后的全部输入内容，不据此断言输入绝对未被改写。
结果位于日志目录
qnorm_isolated_replay_sync_input_deterministic_fixed_storage_matched_options_separate_output_20261023.json。
诊断脚本Ruff通过，原整模型compile失败保持，不新增验收通过项。

## 2026-10-24：Q norm重放确定性设置对照

读取八份捕获的实际配置，均为Y0BLOCK4、Y0BLOCK_SUB3、X2SUB4、
X3SUB128、split_axis=[0]、split_blocks=[4]。当前依赖
npu_triton_heuristics.py::_precompile_config仅将kernel形参中的配置
加入编译常量，split元数据不在形参中；GridNpu按ceil(64/4)生成16，
与直接重放grid16一致，未发现该处launch差异。

独立重放此前未开启原训练确定性设置，故诊断脚本增加可选
LLMTUNER_REPLAY_DETERMINISTIC=1，调用正式set_deterministic_algorithms，
结果JSON记录实际确定性及fill_uninitialized_memory。默认行为不变。
启动前node92设备空闲；首次session48659 exit1，因精简PYTHONPATH
缺colorama在导入阶段停止，未执行kernel。补上已有运行时目录后
session99651 exit0；未安装新依赖，保留首次日志。

```bash
PYTHONPATH=.:outputs/baseline-runtime:outputs/qwen3-30b-runtime:/home/jianzhnie/llmtuner/wheels/tf511 ASCEND_RT_VISIBLE_DEVICES=0 LLMTUNER_REPLAY_SYNC_INPUT=1 LLMTUNER_REPLAY_MATCH_OPTIONS=1 LLMTUNER_REPLAY_SEPARATE_OUTPUT=1 LLMTUNER_REPLAY_DETERMINISTIC=1 timeout 240 python outputs/replay_qnorm_captured.py > outputs/qwen3-30b-npu/logs/qnorm_replay_deterministic_retry1_20261024.log 2>&1
```

实际deterministic_algorithms=True、fill_uninitialized_memory=False。
SUB3/4各24次均有限，最大AE分别0.3772181272506714与
0.0010554728796705604；分别8/4份输入的重复误差变化。该对照未
消除异常，不能作为生产修复。结果保存于日志目录
qnorm_isolated_replay_sync_input_deterministic_matched_options_separate_output_20261023.json。
脚本Ruff通过；未修改生产代码、依赖或整模型验收标准。

## 2026-10-24：Q norm重放输入搬运核验

node92容器启动前npu-smi确认无作业；使用NPU0逐项读取八份捕获的
五个输入，共40项。CPU捕获均有限、连续；CPU→NPU→CPU往返值
逐位一致，所有非单例维度stride均保持一致。审计exit0，逐项shape、
CPU/NPU stride及NPU格式保存至
outputs/qnorm-input-transfer-audit-20261024.json。
因此本次检查不支持输入搬运破坏值或有效stride的假设；捕获时CPU clone
之前的原始设备格式未由该检查证明，也不排除kernel执行问题。
保留CANN分配padding警告，未改生产实现或依赖，未重跑整模型训练。
原compile非有限与重复执行不稳定失败仍未修复。

## 2026-10-24：Q norm masked load零填充重放

先核对历史后发现，上一轮 store 地址枚举重复了已有
qnorm_address_audit_20261023.json，不能视为新根因进展。
本轮转向尚未验证的 masked load 无效 lane 假设：忽略目录诊断脚本
outputs/replay_qnorm_captured.py 新增 LLMTUNER_REPLAY_ZERO_MASKED_LOADS=1，
通过 AST 为原五个 tl.load 显式添加 other=0.0，默认行为不变。
不改归档、生产实现或已安装依赖。

node92 hyper-parallel-env 启动前 npu-smi 无作业；session71950
终止 exit0，仅表示收集完成。完整命令在容器仓库根目录执行：

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 LLMTUNER_REPLAY_SYNC_INPUT=1 LLMTUNER_REPLAY_MATCH_OPTIONS=1 LLMTUNER_REPLAY_SEPARATE_OUTPUT=1 LLMTUNER_REPLAY_ZERO_MASKED_LOADS=1 timeout 240 python outputs/replay_qnorm_captured.py > outputs/qwen3-30b-npu/logs/qnorm_replay_zero_masked_20261024.log 2>&1
```

八份捕获、SUB3/4 各三次：每组 24 次均有限、无精确案例；
SUB3 最大 AE=3.164468700643441e27，八份捕获重复误差均变化；
SUB4 最大 AE=0.0008884543785825372，七份重复误差变化。
显式零填充未消除异常，不能作为修复或关闭 compile 失败。
结果文件仍沿用脚本历史日期后缀：
outputs/qwen3-30b-npu/logs/qnorm_isolated_replay_sync_input_zero_masked_matched_options_separate_output_20261023.json。
诊断脚本 Ruff 通过；原整模型断言及阈值未修改。

## 2026-10-24：Inductor Q norm逻辑写入地址覆盖审计

重读原始目标后核对当前依赖 reduction 实现与归档 kernel，未修改依赖。
针对 Y0BLOCK=4、SUB=3 尾循环可能越过 program 边界的假设，确认
归档源码对 y0/y0_0 均使用 min(Y0BLOCK+y0_offset,y0_numel) mask。
CPU 枚举实际 grid16、y64/x2=4/x3=128 的有效 store 地址：
SUB3 和 SUB4 各覆盖 32,768 个地址，每地址恰好写一次，无缺失、
重复写或跨 program 地址重叠。审计 exit0，结果保存至
outputs/qnorm-store-address-coverage-20261024.json。
这排除给定源码与参数下的逻辑 store 地址重叠，不证明编译后执行、
局部 buffer 复用、输入读取或 transpose lowering 正确；原非有限与
重复执行不稳定问题仍未修复。未运行 NPU 训练，不新增 compile 通过结论。

## 2026-10-24：DP复制与EP覆盖边界核对

按当前 topology.py 的实际校验，EP 必须整除 dp_shard × CP × TP，
dp_replicate 不参与该区域。因此 TP=CP=1、dp_shard=1、EP>1 的
纯复制 DP×EP 配置不合法，不把它列为缺少硬件测试或功能通过。
合法的复制轴覆盖应使用已有 HSDP×EP，或明确带 TP/CP 的配置，
分别核验对应布局；不能将 HSDP 结果称为纯复制 DP×EP。
已核对 ep_fsdp_equivalence.py：其 --dp-replicate=2 在 world4/8
分别构造 dense shard2/4，实际属于 HSDP；ep_fsdp_without_dense_dp.py
的 replicate 模式同时包含 TP4，也不属于上述非法配置。
本轮仅澄清有限矩阵边界，不新增通过项、不缩减合法必测范围。
metrics 精确比较仍因归约顺序标准待确认而保持失败，未改参考或阈值。

## 2026-10-24：PP重计算断言修改后质量复验

全量CPU session22591终止exit0：4302 passed、13 skipped、
298 warnings、169.63s；日志
outputs/qwen3-30b-npu/logs/pp_recompute_full_cpu_20261024.log。
node92容器使用测试venv、visible device0及独立pycache目录
/tmp/llmtuner-pp-recompute-quality-20261024，标准tf511/30B/baseline
PYTHONPATH。全仓Ruff、format458文件、容器compileall及diff检查通过。

完整pre-commit隔离快照/tmp/llmtuner-current-quality-u080k9qf，
session41203终止exit1；已核对完整diff，仅hf_inspection.py两处
tuple-isinstance改union及karpathy SKILL末尾换行。未回写无关文件，
未豁免门禁失败。两个新增PP重计算NPU参考运行均已终止exit0，
分别为full/1F1B session8903与selective/Interleaved session70528；
其余AC/调度组合不由本次结果外推，整体P0–P4仍未完成。

## 2026-10-24：selective AC与Interleaved流水实际重计算

沿用上一节新增的正式pp_equivalence.py逐层重计算选项，双卡PP2、
native FP32、小Qwen3六层、Interleaved1F1B、selective AC、seq32、
batch8、4microbatches、4步，DP/TP/EP/CP均1。session70528日志
outputs/qwen3-30b-npu/logs/pp_selective_interleaved_recompute_reference_20261024.log
报告failed ranks=0、all checks passed；loss最大AE4.768e-7、
全部梯度最大AE1.267e-7、更新参数最大AE1.192e-7，原tol1e-5。
每rank三个本地decoder每步原前向与重计算均非零；step2–4均为
[4,4]，首步有额外调用，不据此估算重计算开销。保留CPU/AiCPU回退。

完整复现沿用下方full/1F1B命令，将AC改为selective、位置参数改为
Interleaved1F1B，日志改为上述独立路径，其他环境及断言不变。
这是指定小模型独立训练参考及实际重算证据，不证明恢复或30B，
不关闭原TP/EP/CP精确恢复失败。全量质量门禁仍待本次修改后复验。

## 2026-10-24：PP独立参考的逐层实际重计算补证

正式pp_equivalence.py新增可选LLMTUNER_TEST_REQUIRE_RECOMPUTE=1，
默认关闭；full/selective模式逐步对每个本地decoder的wrapped module
注册pre-hook，按in_checkpoint_recompute区分原前向与重算，finally
移除hook，并要求两类计数均非零。原独立非AC完整模型参考及tol1e-5
保持不变，未修改生产实现。沿用仓库dev-guide验证职责与边界。

node92 hyper-parallel-env启动前npu-smi八卡无运行进程；session8903
exit0。NPU2、PP2、1F1B、native、FP32、小Qwen3、seq32、batch8、
4microbatches、4步、full AC，DP/TP/EP/CP均1。两rank每层每步
均确认重算；step2–4计数均[4,4]，step1 rank0[5,4]、rank1[5,5]，
计数包含初始化阶段可能发生的额外调用，不把计数解释为精确重算开销。
loss最大AE4.768e-7、全部梯度最大AE3.912e-8、更新参数最大AE
1.192e-7，failed ranks=0、all checks passed。日志保留CPU断言
回退及AiCPU ArgSort；不宣称纯AiCore或性能通过。

容器仓库根目录完整命令：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 LLMTUNER_TEST_AC=full LLMTUNER_TEST_REQUIRE_RECOMPUTE=1 timeout 300 torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_equivalence.py 1F1B > outputs/qwen3-30b-npu/logs/pp_full_recompute_reference_20261024.log 2>&1
```

仅指定full/1F1B小模型独立数值参考与实际重算通过，不外推selective、
Interleaved、恢复、30B或原TP/EP/CP间歇恢复失败。本轮局部Ruff及
format通过，完整CPU/pre-commit尚未针对本次修改重跑，整体目标开放。

## 2026-10-24：vocab-parallel逐token反向补验

正式vocab_parallel_loss_equivalence.py原reduction=none仅比较前向值，
现新增linspace(-1.5,2.0,64)非均匀上游梯度，与独立完整词表
F.cross_entropy(reduction=none)反向比较，并断言IGNORE_INDEX行
梯度严格为0。沿用原rtol1e-4/atol1e-6，未修改生产实现或放宽标准。

node92 hyper-parallel-env，启动前npu-smi八卡无运行进程；NPU0/1、
FP32、64tokens、词表8/7/5，实际分片分别4+4、4+3、3+2。
session28944 exit0，failed ranks=0、all checks passed；非均匀逐token
梯度最大AE分别1.192e-7、2.384e-7、1.192e-7。既有标量loss/梯度、
逐token前向、logprobs/entropy、完整词表dispatch及空分片拒绝均复验。
日志保留_assert_async CPU回退；不宣称纯AiCore、模型lm_head分片
训练或30B验收。

容器仓库目录完整命令：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 timeout 240 torchrun --standalone --nproc_per_node=2 tests/integration_tests/vocab_parallel_loss_equivalence.py > outputs/qwen3-30b-npu/logs/vocab_weighted_backward_20261024.log 2>&1
```

局部Ruff及format检查通过；本次仅修改显式torchrun集成测试，
未重新运行全量CPU/pre-commit，不将此前门禁记录作为本次新运行。
整体P0–P4及原有失败保持开放。

## 2026-10-24：用户指定四组八卡拓扑的实际rank分组

node92 hyper-parallel-env，启动前npu-smi八卡无运行进程。辅助探针
outputs/audit_four_topologies_20261024.py调用生产ParallelTopology，
逐rank读取get_process_group_ranks并与对应DeviceMesh.mesh逐位比较，
all_gather_object保存八rank记录。session70238 exit0，四组均通过，
日志outputs/qwen3-30b-npu/logs/four_topologies_rank_groups_20261024.log。
这是独立拓扑检查，不是新增训练或参数分片验收；不与旧训练拼为同次运行。

| dense dp_shard / TP / EP | dense DP分片组 | TP组 | EP组 | 专家FSDP组 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 8 / 1 / 1 | [0,1,2,3,4,5,6,7] | 各rank单例 | 各rank单例 | EP关闭，未启用专家FSDP | 通过 |
| 4 / 2 / 1 | [0,2,4,6]、[1,3,5,7] | [0,1]、[2,3]、[4,5]、[6,7] | 各rank单例 | EP关闭，未启用专家FSDP | 通过 |
| 4 / 2 / 4 | [0,2,4,6]、[1,3,5,7] | [0,1]、[2,3]、[4,5]、[6,7] | [0,1,2,3]、[4,5,6,7] | [0,4]、[1,5]、[2,6]、[3,7] | 通过 |
| 8 / 1 / 8 | [0,1,2,3,4,5,6,7] | 各rank单例 | [0,1,2,3,4,5,6,7] | 各rank单例 | 通过 |

四组replicate轴均单例，loss组均为全部八rank，batch组分别与dense
DP分片组相同。探针include_singleton_axes=True读取EP1的潜在sparse
视图，其efsdp全八rank仅为拓扑视图，不证明模型启用该专家分片路径。
第三组实测专家FSDP度数2，第四组度数1；EP复用设备，不乘入world size。

容器仓库根目录完整复现命令：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 240 torchrun --standalone --nproc_per_node=8 outputs/audit_four_topologies_20261024.py > outputs/qwen3-30b-npu/logs/four_topologies_rank_groups_20261024.log 2>&1
```

本轮未修改生产或正式测试源码，新增辅助探针及本状态证据。整体
P0–P4、原数值失败及硬件/编译缺口仍开放。

## 2026-10-24：TP×EP三步独立更新证据索引复核

重新读取原始目标、README组合声明、compatibility.py及正式
ep_fsdp_without_dense_dp.py。当前脚本分别执行生产与独立HF的
SGD.step，再从HF实际更新权重按TP/EP坐标切片比较；不是手算更新。
重新核对outputs/qwen3-30b-npu/logs/中的
tp_ep_actual_optimizer_{none,shard,replicate}_20261023.log：每份均有
step1/2/3及三次all checks passed，每步检查12项dense及三类expert。
原执行记录各exit0，本次读取不产生新的训练退出码。

| 配置 | 参考及阈值 | 三步logits/expert梯度/dense梯度/更新权重最大AE | 状态 | 日志 |
| --- | --- | --- | --- | --- |
| TP4/EP2，无额外DP，world4 | 独立HF/SGD，atol1e-5/rtol1e-4 | 1.192e-7/1.863e-9/2.980e-8/7.451e-9 | 通过 | tp_ep_actual_optimizer_none_20261023.log |
| TP4/EP2/DP分片2，world8 | 同上，不同DP输入及参考梯度差异断言 | 1.192e-7/1.630e-9/2.235e-8/7.451e-9 | 通过 | tp_ep_actual_optimizer_shard_20261023.log |
| TP4/EP2/DP复制2，world8 | 同上，实际复制placement断言 | 1.788e-7/9.313e-10/2.235e-8/7.451e-9 | 通过 | tp_ep_actual_optimizer_replicate_20261023.log |

完整命令沿用本文件“后续参考覆盖修正”及三步执行节，日志选择上述
actual_optimizer版本；运行时模型为tiny Qwen3MoE、FP32、32tokens、
8experts/topk2、CP1、AC none、aux0、SGD lr0.05。此索引将独立
三步更新与原单步前反向证据分开，不外推Trainer/AdamW/EMA/恢复/30B。
本轮只核对证据并修改状态文件，不启动NPU，不重复已有通过运行。
P2完整声明映射及既有失败仍开放；上一轮torch_checkpointing恢复
与本项是独立证据，不拼接成同次训练通过。

## 2026-10-24：torch_checkpointing 独立进程恢复

pp_checkpoint_equivalence.py 新增 LLMTUNER_TEST_CHECKPOINT_BACKEND，
默认 dcp；full/resume 均断言实际 manager 类型，保存检查使用所选后端
metadata 名称。未修改生产实现或既有精确阈值。

node92 hyper-parallel-env，启动前 npu-smi 无运行进程。NPU2、native
FSDP 实际分片、FP32、小 Qwen3、full AC、dynamic EMA、RNG/loader，
连续4步在step2保存；另启torchrun恢复2步。session32837/full和
60852/resume均exit0。两rank pre-update state failures=0；
模型、optimizer、scheduler、EMA、trainer/token counter、RNG、loader
及运行时状态按脚本精确比较通过。各层每步forward/recompute均[1,1]。
loss为4.855580806732178、4.852251052856445、4.858562469482422、
4.877256393432617；恢复尾部最大AE=0，all checks passed。

工作目录 outputs/torchckpt-fsdp-fullac-cold-retry1-20261024；日志为
outputs/qwen3-30b-npu/logs/torchckpt_fsdp_fullac_cold_{full,resume}_retry1_20261024.log。
首次session35532 exit1因当前容器未安装torch_checkpointing，日志
torchckpt_fsdp_fullac_cold_full_20261024.log保留。随后仅离线安装
torch_checkpointing-0.1.0 wheel到outputs/torchckpt-runtime，--no-index
--no-deps，并加入PYTHONPATH。日志披露_assert_async CPU、ArgSort
AiCPU回退，pin_memory因无CUDA而忽略；未据此宣称纯AiCore。

容器仓库目录完整full命令（resume将phase full改为resume，日志full改为resume）：

```bash
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=.:outputs:outputs/torchckpt-runtime:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime PYTHONPYCACHEPREFIX=/tmp/llmtuner-torchckpt-cold-20261024 ASCEND_RT_VISIBLE_DEVICES=0,1 LLMTUNER_TEST_CHECKPOINT_BACKEND=torch_checkpointing LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_REQUIRE_SHARDED=1 LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_TEST_REQUIRE_RECOMPUTE=1 && timeout 300 torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py full outputs/torchckpt-fsdp-fullac-cold-retry1-20261024 > outputs/qwen3-30b-npu/logs/torchckpt_fsdp_fullac_cold_full_retry1_20261024.log 2>&1
```

本项通过仅限上述小模型两卡独立进程恢复，不外推30B、PP或其他布局。
全仓Ruff、format（458文件）、diff检查通过。全量CPU session91211：
4302 passed、13 skipped、303 warnings，182.23s，日志
outputs/qwen3-30b-npu/logs/torchckpt_cold_full_cpu_20261024.log。
隔离pre-commit首次指出此前CP测试的一处isinstance格式问题，已修正；
复查session85115 exit1，隔离副本/tmp/llmtuner-current-quality-qjec6q0l
仅余hf_inspection.py两处isinstance及karpathy SKILL末尾换行既有问题，
未回写无关文件，未豁免。整体P0–P4及既有失败仍开放。

## 2026-10-24：训练指标精确比较失败的基线语义复核

重新读取原始目标及基线numerical_alignment.md，并核对当前
torchtitan f5c97dc0e2d6cca72313004aa853a196ca02ab49 的
torchtitan/trainer.py:1025–1039：其accumulated_loss已按全局token数
归一化，再通过dist_sum求和。TorchLLMTuner._step_metrics亦先做
accumulated_loss/global_valid_tokens再SUM。因此不能仅凭独立参考
先SUM后除法的差异认定生产公式有误，也不为通过测试改动生产归约顺序。
torchtitan-npu版本为b358a8063c34aa7494397efa3a8f5ad24f1caa1f。

重新核对outputs/qwen3-8b-npu/logs/metrics_compile_npu_20261022.log，
原精确断言仍失败：loss实际0.7500000596046448，参考0.75；绝对误差
5.960464477539063e-8、相对误差7.947285970052083e-8，FP32相距1 ULP。
ULP由struct按IEEE754 float32位模式计算，仅说明该标量差异大小，
不是HCCL全部配置正确性证明，不自动将精确失败改为近似通过。
基线规范把不同归约顺序归为数值近似，但本项既有精确标准未获变更确认，
故保留失败及原脚本，不新增通过结论。此复核不运行NPU或修改生产代码；
后续需先明确该指标用例采用同归约顺序精确参考，还是另行确认近似标准。

## 2026-10-24：KV模式扩展后的TP双轴回归

node92 hyper-parallel-env，启动前npu-smi确认八卡空闲；正式
cp_ulysses_varlen_equivalence.py，session82091终止exit0，日志
outputs/qwen3-30b-npu/logs/tp2_dual_cp_regression_20261024.log
包含all checks passed。world8，TP2/KV2/Ulysses2，DP/PP/EP均1，
FP32、小Qwen3两层、packed seq256，文档长度90/1/165；独立完整
序列参考，原tol1e-5。logits最大AE2.384e-7、平均loss AE0、
全参数梯度AE2.049e-8；文档边界反例差异0.6232，独立varlen
通信输出与梯度误差均0，实际双轴内核断言生效。

容器仓库目录复现：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLMTUNER_TEST_DP_MODE=none LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_DUAL_CP=1 LLMTUNER_TEST_KV_ONLY=0 timeout 300 torchrun --standalone --nproc_per_node=8 tests/integration_tests/cp_ulysses_varlen_equivalence.py > outputs/qwen3-30b-npu/logs/tp2_dual_cp_regression_20261024.log 2>&1
```

本轮未改生产或测试源码；通过仅限指定packed前反向回归，不证明
多步optimizer、恢复或30B。默认、无DP双轴、DP两模式双轴及TP双轴
已有本次扩展后的独立回归记录；全量pre-commit既有失败及整体
P0–P4其他缺口保持开放。

## 2026-10-24：CP模式扩展后全量pre-commit隔离复查

宿主执行 PYTHONPATH=outputs/quality-tools python outputs/check_quality_snapshot.py，
session62030终止exit1，当前git可见文件隔离副本为
/tmp/llmtuner-current-quality-2fydsq75。全部hooks实际执行；仅Ruff和
end-of-file-fixer失败，其他hooks通过。检查完整diff确认仅修改
llmtuner/parallel/expert_parallel/hf_inspection.py两处tuple isinstance
为union，以及.agent/karpathy-guidelines/SKILL.md末尾换行。
均为既有无关问题，未回写原工作区，未自行豁免，不宣称全门禁通过。

此前容器尝试cp_modes_precommit_20261024.log因Git所有权检查退出；
仅对重试命令设置safe.directory后，cp_modes_precommit_retry_20261024.log
在初始化hook时因github.com DNS解析失败exit3，未执行hooks。
随后改用宿主已有缓存完成上述实际检查，未在容器下载依赖。
本轮未修改生产或测试源码；整体P0–P4及原失败保持开放。

## 2026-10-24：KV模式扩展后的DP分片双轴回归

node92 hyper-parallel-env，启动前npu-smi确认八卡空闲；正式
cp_ulysses_varlen_equivalence.py，session26055终止exit0，日志
outputs/qwen3-30b-npu/logs/dp_shard_dual_cp_regression_20261024.log
包含all checks passed。world8，replicate1/shard2/KV2/Ulysses2，
TP/PP/EP均1，native FSDP、FP32、小Qwen3两层、packed seq256，
文档长度90/1/165；独立完整序列与DP平均梯度参考，原tol1e-5。
logits最大AE0、平均loss AE0、全参数梯度AE1.118e-8；文档边界
反例差异0.6232，独立varlen通信输出及梯度误差均0。实际双轴内核、
DTensor局部分片、不同DP输入及参考梯度差异断言均生效。

复现沿用下一节完整命令，将LLMTUNER_TEST_DP_MODE改为shard，
日志改为上述独立路径，其他环境及八卡torchrun参数保持相同。
本轮未改生产或测试源码；通过仅限指定packed前反向回归，
不证明TP、多步optimizer、恢复或30B。两种DP双轴回归均有
本次独立通过证据，完整门禁及整体P0–P4仍未完成。

## 2026-10-24：KV模式扩展后的DP复制双轴回归

node92 hyper-parallel-env，启动前npu-smi确认八卡无作业；正式
cp_ulysses_varlen_equivalence.py，session89611已终止exit0，
日志 outputs/qwen3-30b-npu/logs/dp_replicate_dual_cp_regression_20261024.log
包含all checks passed。world8，replicate2/shard1/KV2/Ulysses2，
TP/PP/EP均1，native FSDP、FP32、小Qwen3两层、packed seq256，
文档长度90/1/165；独立完整序列与DP平均梯度参考，原tol1e-5。
logits最大AE0、平均loss AE4.768e-7、全参数梯度AE1.118e-8；
文档边界反例差异0.6232，独立varlen通信输出和梯度误差均0。
实际双轴内核、DTensor局部分片、Replicate placement、不同DP输入
及参考梯度差异断言均生效。

容器仓库目录复现：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLMTUNER_TEST_DP_MODE=replicate LLMTUNER_TEST_DUAL_CP=1 LLMTUNER_TEST_KV_ONLY=0 timeout 300 torchrun --standalone --nproc_per_node=8 tests/integration_tests/cp_ulysses_varlen_equivalence.py > outputs/qwen3-30b-npu/logs/dp_replicate_dual_cp_regression_20261024.log 2>&1
```

本轮未改生产或测试源码；通过范围仅为指定packed前反向回归，
不外推DP分片双轴、TP、恢复或30B。完整门禁及整体P0–P4仍开放。

## 2026-10-24：CP测试模式扩展后全套CPU与静态检查

当前工作区CPU全套 session48469 已终止exit0：4293 passed、13 skipped、
273 warnings、149.67s。日志为
outputs/qwen3-30b-npu/logs/cp_modes_cpu_suite_20261024.log。
该次包含此前新增HF分设备初始化与30B offload配置回归；CPU套件不执行
torchrun集成脚本，不能替代CP模式的硬件回归或证明全部组合通过。

node92 hyper-parallel-env 容器仓库目录复现：

```bash
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 outputs/qwen3-30b-test-venv/bin/python -m pytest -q --disable-warnings > outputs/qwen3-30b-npu/logs/cp_modes_cpu_suite_20261024.log 2>&1
```

宿主Ruff check llmtuner tests通过，format --check llmtuner tests通过
（458 files already formatted）；容器python -m compileall -q llmtuner
exit0；git diff --check通过。本轮未改生产或测试源码。全量pre-commit
本轮未重跑，既有无关hf_inspection.py及技能EOF失败未获豁免，仍不算
全部质量门禁通过。P2完整映射及P0–P4其余缺口保持开放。

## 2026-10-24：KV模式扩展后的默认与双轴NPU回归

正式 cp_ulysses_varlen_equivalence.py 当前工作区版本，node92容器；
每次启动前npu-smi确认八卡无作业。两次独立进程均终止exit0，
日志均包含all checks passed，保持原tol1e-5。FP32、小Qwen3两层、
packed seq256、文档长度90/1/165，TP/PP/EP均1、无DP。

| 配置 | logits最大AE | 平均loss AE | 全参数梯度最大AE | 状态 | session / 日志 |
| --- | --- | --- | --- | --- | --- |
| 默认Ulysses2，world2 | 2.570e-7 | 0 | 2.980e-8 | 通过 | 20758 / ulysses_default_regression_20261024.log |
| KV2×Ulysses2，world4 | 0 | 9.537e-7 | 1.676e-8 | 通过 | 74866 / dual_cp_regression_20261024.log |

日志目录 outputs/qwen3-30b-npu/logs/。两次文档边界反例差异
均0.6232，独立varlen通信输出和梯度误差均0；双轴实际内核断言生效。
仅证明指定前反向回归，不外推DP双轴、TP、恢复或30B。

容器仓库目录，两次均设置：

```bash
export PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime
export LLMTUNER_TEST_DP_MODE=none LLMTUNER_TEST_KV_ONLY=0
```

默认配置设置 ASCEND_RT_VISIBLE_DEVICES=0,1、LLMTUNER_TEST_DUAL_CP=0，
执行 timeout 300 torchrun --standalone --nproc_per_node=2
tests/integration_tests/cp_ulysses_varlen_equivalence.py，重定向至表中默认日志。
双轴配置设置 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3、LLMTUNER_TEST_DUAL_CP=1，
执行相同命令但 --nproc_per_node=4，重定向至表中双轴日志。
本轮未改生产或测试源码；完整门禁、其他受影响配置及整体P0–P4仍开放。

## 2026-10-24：DP分片 × 单轴 KV 独立参考通过

session54749 已终止exit0；正式 cp_ulysses_varlen_equivalence.py
使用 LLMTUNER_TEST_KV_ONLY=1、LLMTUNER_TEST_DP_MODE=shard。
node92容器启动前npu-smi确认八卡无作业，world4、replicate1/shard2、
KV2/Ulysses1、TP/PP/EP均1，小Qwen3两层、FP32、单文档seq256。
原tol1e-5：完整序列参考logits AE0、平均loss AE0、全部参数梯度
AE8.382e-9，all checks passed；实际KV内核、DTensor局部分片、
不同DP输入及独立参考DP梯度差异断言均生效。
日志：outputs/qwen3-30b-npu/logs/dp_shard_single_kv_20261024.log。

复现沿用下一节完整命令，仅将 LLMTUNER_TEST_DP_MODE 改为 shard，
日志改为上述独立路径，其他环境变量及四卡torchrun参数保持相同。
通过范围限指定因果序列前反向，不外推恢复、optimizer多步或30B。
独立varlen通信子测试仍属Ulysses，不作为KV模型证据。
本轮局部Ruff、格式修正与git diff --check通过；默认及双轴回归、
完整质量门禁仍未完成，单轴KV packed拒绝边界及既有失败保持开放。

## 2026-10-24：DP复制 × 单轴 KV 因果序列独立参考

正式 cp_ulysses_varlen_equivalence.py 增加 LLMTUNER_TEST_KV_ONLY=1，
使用单文档因果序列及实际 CPSdpaKernel，保留完整序列 logits/loss、
全参数梯度、实际DTensor分片和Replicate placement断言；不同DP输入
与参考梯度差异断言仍生效。默认Ulysses及双轴配置不变。

node92容器启动前npu-smi确认八卡空闲；session35246终止exit0，
日志 outputs/qwen3-30b-npu/logs/dp_replicate_single_kv_20261024.log。
world4，replicate2/shard1/KV2/Ulysses1，TP/PP/EP均1，小Qwen3两层、
FP32、seq256；原tol1e-5，logits AE0、平均loss AE4.768e-7、
全参数梯度AE8.382e-9，all checks passed。该次日志仍有旧
ulysses/packed与doc_lens标签，以显式kv_only=True及实际配置为准；
随后仅修正输出标签，未改变数值计算。独立varlen通信子测试仍验证
Ulysses，不作为KV内核证据；KV证据来自实际模型及全部参数梯度。

容器仓库目录复现：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_DP_MODE=replicate LLMTUNER_TEST_DUAL_CP=0 LLMTUNER_TEST_KV_ONLY=1 timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/cp_ulysses_varlen_equivalence.py > outputs/qwen3-30b-npu/logs/dp_replicate_single_kv_20261024.log 2>&1
```

通过范围限指定普通因果序列前反向，不证明多步更新、恢复或30B。
单轴KV packed拒绝边界不变；DP分片KV及受影响默认/双轴回归、
完整质量门禁仍需继续，整体目标未完成。

## 2026-10-24：单轴 KV 后续验收范围核对

核对生产 CPSdpaKernel 构造契约：packed=True 明确拒绝，既有拒绝
记录保持有效；双轴 packed 证据不能外推单轴 KV。没有运行新硬件用例，
不新增通过结论。后续 DP复制/分片×单轴KV 独立参考应使用普通因果序列，
对照完整序列 logits、loss 与全部参数梯度，并核验实际 mesh/placement；
packed 单轴KV仍是未实现边界，不因选择普通序列而关闭。

正式 cp_ulysses_varlen_equivalence.py 的运行说明同步当前装配：
DP2×单轴Ulysses使用四卡，DP2×双轴CP使用八卡。仅修正文档字符串，
不改变测试语义。此前两种DP模式的硬件证据仍限各自已记录配置。

## 2026-10-24：DP分片 × 单轴 Ulysses packed 独立参考

node92 hyper-parallel-env 启动前 npu-smi 确认八卡无作业；正式测试
cp_ulysses_varlen_equivalence.py，session9243 已终止exit0，日志
outputs/qwen3-30b-npu/logs/dp_shard_single_ulysses_20261024.log
包含 all checks passed。world4，replicate1/shard2/KV1/Ulysses2，
TP/PP/EP均1，native FSDP、FP32、小Qwen3两层、packed seq256，
文档长度90/1/165；独立完整序列及DP平均梯度参考，原tol1e-5。
logits最大AE2.570e-7、loss AE9.537e-7、全参数梯度AE1.490e-8；
文档边界反例差异0.6232，varlen内核输出与梯度误差均0。
实际DTensor局部分片、不同DP输入及参考梯度差异断言均生效。

容器仓库目录复现：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_DP_MODE=shard LLMTUNER_TEST_DUAL_CP=0 timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/cp_ulysses_varlen_equivalence.py > outputs/qwen3-30b-npu/logs/dp_shard_single_ulysses_20261024.log 2>&1
```

状态：通过，仅限指定packed前反向，不外推多步optimizer、恢复或30B。
本轮未改生产或测试代码；单轴Ulysses的DP复制和分片均已有独立证据，
单轴KV及完整声明映射仍需继续，既有失败不变，整体目标保持开放。

## 2026-10-24：DP复制 × 单轴 Ulysses packed 独立参考

正式 cp_ulysses_varlen_equivalence.py 扩展拓扑装配，允许非双轴模式的
DP2×Ulysses2；默认两卡及双轴模式保持原配置。保留实际DTensor分片、
Replicate placement、DP不同输入及参考梯度差异非空泛断言、全参数梯度
与独立完整序列参考比较，原阈值1e-5不变。未修改生产实现。

node92 hyper-parallel-env 启动前 npu-smi 确认八卡无作业；session78718
已终止exit0。world4，replicate2/shard1/KV1/Ulysses2，TP/PP/EP均1，
native FSDP、FP32、小Qwen3两层、packed seq256，文档长度90/1/165。
logits最大AE2.570e-7、loss AE0、全参数梯度AE1.490e-8；文档边界
反例差异0.6232，独立varlen内核输出与梯度误差均0，all checks passed。
日志：outputs/qwen3-30b-npu/logs/dp_replicate_single_ulysses_20261024.log。
仅证明指定前反向，不证明多步optimizer、恢复、30B或DP分片配置。

容器仓库目录复现：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_DP_MODE=replicate LLMTUNER_TEST_DUAL_CP=0 timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/cp_ulysses_varlen_equivalence.py > outputs/qwen3-30b-npu/logs/dp_replicate_single_ulysses_20261024.log 2>&1
```

局部Ruff与diff检查通过；完整门禁及双轴回归未在本轮重跑。
DP分片×单轴Ulysses与单轴KV的独立配置仍需核对或补验，整体目标开放。

## 2026-10-24：纯复制 DP × PP Interleaved1F1B 独立参考补验

session8718 已结束，前轮终态返回 exit0；本轮重新读取实际日志，
确认 failed ranks=0、all checks passed，并检查节点上无该测试进程。
未重复启动作业。正式 pp_equivalence.py 检查实际 Replicate placement、
流水阶段参数总量与归属，以及独立完整模型参考的逐步 loss、norm、
裁剪后梯度和更新参数。

| 测试项 | 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码 | 日志 | 范围限制 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DP复制×PP独立训练参考 | world4，PP2/replicate2/shard1，TP/EP/CP均1；native FP32，Interleaved1F1B，4步，4 microbatches，batch8，seq32，无AC | 无流水完整模型；原定tol1e-5 | loss最大AE4.768e-7，梯度9.872e-8，参数2.119e-7，failed ranks0 | 通过 | 0 | outputs/qwen3-30b-npu/logs/pp_replica2_interleaved_20261024.log | 小模型；不证明30B、恢复或其他训练特性 |

复现：node92 的 hyper-parallel-env 容器，仓库目录执行：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_DP_REPLICATE_SIZE=2 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native timeout 240 torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_equivalence.py Interleaved1F1B > outputs/qwen3-30b-npu/logs/pp_replica2_interleaved_20261024.log 2>&1
```

日志明确报告 _assert_async CPU、ArgSort AiCPU 回退，以及 AdamW
从 fused 转为 for-loop；不宣称纯 AiCore。结合下节已有独立1F1B记录，
纯复制DP×PP两调度均有指定配置参考证据，不借用含CP或HSDP的组合替代。
本轮仅补状态证据；完整P2声明映射及既有KCP、compile、custom FSDP、
fullAC间歇恢复失败和P4关键权重标准仍开放。

## 2026-10-24：P2 单轴 KVCP 与复制 DP/PP 证据映射补齐

重新读取下表三份实际日志，均包含 failed ranks=0 和 all checks
passed；核对正式 pp_equivalence.py / pp_cp_equivalence.py 的参考
比较范围。这里补齐历史覆盖索引遗漏，不新增运行，不将包含 PP
的组合证据外推为无 PP 路径或30B模型通过。日志目录统一为
outputs/qwen3-8b-npu/logs/，完整命令见本文件原执行节。

| 具体配置 | 独立参考与阈值 | 实际结果 | 状态 | 日志 | 范围限制 |
| --- | --- | --- | --- | --- | --- |
| world8，PP2/KVCP2/shard2/replicate1，1F1B，native FP32，seq256/batch8，4步 | 无流水完整模型；逐步loss/norm、梯度及更新，tol1e-5 | loss AE4.768e-7，梯度4.703e-8，参数3.191e-6；failed ranks0 | 通过 | pp2_cp2_dp_shard2_20261022.log | 无AC；不证明恢复或其他CP策略 |
| 同上，Interleaved1F1B | 同上 | loss AE4.768e-7，梯度3.539e-8，参数1.565e-6；failed ranks0 | 通过 | pp2_cp2_dp_shard2_interleaved_20261022.log | 同上；日志明确world8/shard2/replicate1/cp2 |
| world4，PP2/CP1/shard1/replicate2，1F1B，native FP32，seq32/batch8，4步 | 无流水完整模型；逐步loss/norm、梯度及更新，tol1e-5 | loss AE0，梯度3.725e-8，参数7.339e-7；failed ranks0 | 通过 | pp2_dp_replicate2_20261022.log | 不外推Interleaved；HSDP两调度有独立记录 |

三份日志均明确报告CPU/AiCPU回退；保留原执行节exit0记录，
不以本次日志读取产生新的运行退出码。完整DP各声明路径映射
仍需继续，KCP、compile和原fullAC冷恢复失败未关闭。本轮只
更新状态文档，按文档范围检查，不重复训练或CPU全套。

## 2026-10-24：30B offload 配置开关自动化回归

test_npu_example_config.py 增加三组配置回归，覆盖 offload关闭、
offload开启且pin_memory关闭/开启，检查 CPU offload 使用 for-loop
优化器、非offload保持fused配置，并保持完整训练checkpoint策略。
pin_memory=True 这里只验证配置，不代表该选项已通过30B硬件验收。

node92 容器既有CPU测试环境执行
outputs/qwen3-30b-test-venv/bin/python -m pytest -q
tests/unit_tests/cpu/test_npu_example_config.py --disable-warnings，
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime，
ASCEND_RT_VISIBLE_DEVICES=0；session1890终止exit0：13 passed、
14 warnings、11.12s。本轮仅修改测试，Ruff及diff检查通过。
此前完整4289项测试不包含本轮新增的4项回归，未宣称新总数
的全套已运行。原30B十步计时证据已存在于P0执行节，未重复训练。
整体P0–P4、声明矩阵映射和既有失败保持开放。

## 2026-10-24：HF offload 分设备初始化契约回归

test_hf_wrapper.py 新增分设备初始化回归：meta 构造 Linear，
parameter_device=CPU、buffer device=meta，验证参数实际分配到 CPU，
持久及非持久 buffer 使用指定设备，参数及两类 buffer 的别名均保持
对象身份。meta 仅作为不同 buffer 设备的替身，不验证 buffer 数值，
不替代已记录的30B NPU训练和恢复证据。未修改生产实现。

node92 容器仓库目录，PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime，
ASCEND_RT_VISIBLE_DEVICES=0，执行
outputs/qwen3-30b-test-venv/bin/python -m pytest -q
tests/unit_tests/cpu/models/test_hf_wrapper.py --disable-warnings，
session20713终止exit0：88 passed、16 warnings、23.23s。
修改文件Ruff、format及git diff --check通过。此前4289项完整CPU
通过记录不包含此新增测试；本轮未重跑完整套件。整体目标保持开放。

## 2026-10-24：30B HSDP CPU offload 保存与独立恢复通过

正式入口连续训练 session80548、独立恢复 session7200 均终止 exit0。
两次启动前 node92 npu-smi 确认八卡空闲；模型 Qwen3-30B-A3B、
LIMO、BF16/fullAC、replicate2/shard4、TP/EP/PP/CP均1、seq4096、
global batch8、seed42/deterministic、CPU offload开启、pin_memory=False、
AdamW for-loop、总步数2、checkpoint间隔1，未启用EMA。
连续两步 loss0.53537→0.42943、grad norm3.3720→2.3506；
恢复step2 loss0.42943、grad norm2.3506。step1完整保存31.04s。
这些显示值不作为精确一致的依据，不从日志时间差推导训练性能。

输出分别为 outputs/qwen3-30b-npu/hsdp_offload_checkpoint_20261024
和 outputs/qwen3-30b-npu/hsdp_offload_resume_20261024；恢复目录
checkpoint/step-1 是连续训练step-1的绝对符号链接，只新增step-2。
日志分别位于 outputs/qwen3-30b-npu/logs/ 下同名 .log 文件。
复现：容器仓库目录设置以下环境，执行
timeout 1800 bash examples/train_qwen3_30b_npu.sh 并重定向至独立日志：

```bash
export PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime
export LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE=2 LLMTUNER_DATA_PARALLEL_SHARD_SIZE=4
export LLMTUNER_TENSOR_PARALLEL_SIZE=1 LLMTUNER_EXPERT_PARALLEL_SIZE=1 LLMTUNER_PIPELINE_PARALLEL_SIZE=1 LLMTUNER_CONTEXT_PARALLEL_SIZE=1
export LLMTUNER_FSDP_CPU_OFFLOAD=1 LLMTUNER_FSDP_CPU_OFFLOAD_PIN_MEMORY=0
export LLMTUNER_STEPS=2 LLMTUNER_GLOBAL_BATCH_SIZE=8 LLMTUNER_MAX_SEQ_LEN=4096 LLMTUNER_DETERMINISTIC=1
export LLMTUNER_SAVE_CHECKPOINT=1 LLMTUNER_CHECKPOINT_INTERVAL=1 LLMTUNER_ENABLE_TENSORBOARD=0
```

连续运行使用 MASTER_PORT=29683、LLMTUNER_DUMP_FOLDER 为上述连续目录。
独立恢复使用 MASTER_PORT=29684、LLMTUNER_DUMP_FOLDER 为上述恢复目录，
另设 LLMTUNER_CHECKPOINT_LOAD_STEP=1；总步数和调度保持不变。

完整状态审计 session54569 终止 exit0，13325个chunk全部精确一致：
model2124、optimizer11151、rng_state8、dataloader39、lr_scheduler1、
train_state2，failures=[]；RNG解码后比较。审计命令：

```bash
python outputs/compare_30b_checkpoint_chunks.py outputs/qwen3-30b-npu/hsdp_offload_checkpoint_20261024/checkpoint/step-2 outputs/qwen3-30b-npu/hsdp_offload_resume_20261024/checkpoint/step-2 > outputs/qwen3-30b-npu/logs/hsdp_offload_resume_state_audit_20261024.log 2>&1
```

本轮未修改生产代码。通过范围限本配置两步保存及独立恢复，
不外推长程训练、跨框架/布局对齐或其他offload选项；日志中的
AiCPU/CPU回退及HCCL限制继续保留。原无offload HSDP OOM、
pre-commit无关失败和整体P0–P4缺口保持开放。

## 2026-10-24：HF offload 修复 pre-commit 隔离复查

执行 PYTHONPATH=outputs/quality-tools python outputs/check_quality_snapshot.py，
session92606 终止 exit1。隔离副本为
/tmp/llmtuner-current-quality-q6jy8h_q；hook 仅修改已有
llmtuner/parallel/expert_parallel/hf_inspection.py 两处 isinstance
类型元组写法及 .agent/karpathy-guidelines/SKILL.md 的末尾换行。
与此前记录相同，本次 HF 初始化修复及30B入口无 hook 修改。
其余 hook 通过。未将隔离副本修改写回无关文件，未豁免失败，
不宣称完整 pre-commit 门禁通过；整体目标保持开放。

## 2026-10-24：HF offload 初始化修复全量 CPU 验证

生产初始化修复后，node92 容器完整 CPU 测试 session19309
终止 exit0：4289 passed、13 skipped、273 warnings，152.45s。
日志 outputs/qwen3-30b-npu/logs/offload_materialization_cpu_suite_20261024.log。
容器仓库目录复现：

```bash
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 outputs/qwen3-30b-test-venv/bin/python -m pytest -q --disable-warnings > outputs/qwen3-30b-npu/logs/offload_materialization_cpu_suite_20261024.log 2>&1
```

全仓 ruff check llmtuner tests examples/train_qwen3_30b_npu.py 和
ruff format --check 同范围通过（459 files）。node92 原容器执行
python -m compileall -q llmtuner，exit0；git diff --check 通过。
pre-commit 尚未对本次生产修复重跑，不宣称所有门禁通过。
30B offload checkpoint 保存、恢复和完整数值验收继续开放。

## 2026-10-24：30B HSDP CPU offload 两步容量验证通过

修复 HF meta 初始化的设备契约：materialize_meta_model 增加
parameter_device，assembly 在 CPU offload 下将参数分配到 CPU，
运行时 buffer 仍在计算设备恢复，保留别名恢复。来源映射已更新。
受影响 test_hf_wrapper.py 与 test_rotary_initialization.py 在 node92
既有 CPU 测试环境通过98项、16 warnings、31.88s，session12590
exit0；Ruff、format 和 git diff --check 通过。尚无独立新增的
分设备初始化单测，不将已有CPU测试视为该路径的硬件证据。

随后正式30B入口复跑，session7366终止exit0。配置沿用下节首次
失败：Qwen3-30B-A3B/LIMO、8卡、replicate2/shard4、TP/EP/PP/CP
均1、BF16/fullAC、seq4096/global batch8、seed42/deterministic、
CPU offload=True、pin_memory=False、AdamW for-loop、2步、save0。
启动前 npu-smi 确认八卡无进程。仅替换 MASTER_PORT=29682 和
LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/hsdp_offload_materialization_fix_20261024，
其余命令与下节相同；独立日志为
outputs/qwen3-30b-npu/logs/hsdp_offload_materialization_fix_20261024.log。

两步loss 0.53537→0.42943，grad norm 3.3720→2.3506，日志报告
memory 42.18GiB。HF加载5.82s，完成两个真实AdamW更新并退出。
日志明确报告ArgSort AiCPU、_assert_async CPU及HCCL gather使用
allgather替代；HCCL不支持运行中修改timeout，保留启动timeout。
未从日志时间差推导step性能，不将两步结果视为性能分布验收。

状态：本配置两步训练容量通过；checkpoint保存和独立恢复未跑，
跨框架/跨布局数值对齐未验收，30B运行尚无逐参数驻留审计。
原无offload HSDP OOM及整体P0–P4缺口保持开放。生产修复后
全量CPU、compileall和pre-commit尚待重跑。

## 2026-10-24：30B HSDP CPU offload 正式入口首次运行失败

node92 启动前八卡无运行进程，主机 available 947GiB。正式入口
使用 Qwen3-30B-A3B/LIMO、replicate2/shard4、TP/EP/PP/CP均1、
BF16/fullAC、seq4096/global batch8、seed42/deterministic、2步，
CPU offload=True、pin_memory=False、AdamW for-loop，save0。
session85573 已终止 exit1，未完成训练步，不作为容量通过。
日志 outputs/qwen3-30b-npu/logs/hsdp_offload_capacity_20261024.log。

加载 HF 权重后，首次 forward 的 FSDP lazy_init 拒绝 embedding
参数位于 NPU：FSDP parameters should be materialized on CPU when
enabling CPU offloading。当前 assembly.py 的 HF 初始化分支统一
调用 materialize_meta_model(model_part, trainer.device)，没有区分
offload 参数存储与运行时 buffer 的计算设备。下一步须修复该分配
契约并验证 HF 加载后参数仍驻留 CPU，同时保持 RoPE、MoE 统计
等运行时 buffer 的设备和别名语义；不能仅把整个模型搬回 CPU。

复现环境 PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime；
设置 LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE=2、
LLMTUNER_DATA_PARALLEL_SHARD_SIZE=4、LLMTUNER_TENSOR_PARALLEL_SIZE=1、
LLMTUNER_EXPERT_PARALLEL_SIZE=1、LLMTUNER_PIPELINE_PARALLEL_SIZE=1、
LLMTUNER_CONTEXT_PARALLEL_SIZE=1、LLMTUNER_FSDP_CPU_OFFLOAD=1、
LLMTUNER_FSDP_CPU_OFFLOAD_PIN_MEMORY=0、LLMTUNER_STEPS=2、
LLMTUNER_GLOBAL_BATCH_SIZE=8、LLMTUNER_MAX_SEQ_LEN=4096、
LLMTUNER_DETERMINISTIC=1、LLMTUNER_SAVE_CHECKPOINT=0、
LLMTUNER_ENABLE_TENSORBOARD=0、MASTER_PORT=29681、
LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/hsdp_offload_capacity_20261024，
容器仓库目录执行 timeout 1800 bash examples/train_qwen3_30b_npu.sh，
stdout/stderr 重定向到上述日志。原无 offload OOM 与整体缺口保持开放。

## 2026-10-24：30B 正式入口接入显式 CPU offload 配置

examples/train_qwen3_30b_npu.py 新增 LLMTUNER_FSDP_CPU_OFFLOAD
（默认0）和 LLMTUNER_FSDP_CPU_OFFLOAD_PIN_MEMORY（默认1），
复用 FSDPConfig；启用 offload 时明确使用 AdamW for-loop。
默认训练行为保持原设置，不将此配置替代原无 offload HSDP 失败。

node92 容器、ASCEND_RT_VISIBLE_DEVICES=0、既有 tf511/runtime
PYTHONPATH 下，outputs/qwen3-30b-test-venv/bin/python -m pytest -q
tests/unit_tests/cpu/test_npu_example_config.py --disable-warnings
结束 exit0，10 passed、14 warnings、11.12s（session59626）。
现有测试不覆盖新增开关；另用正式配置构造函数，显式设置
LLMTUNER_FSDP_CPU_OFFLOAD=1、
LLMTUNER_FSDP_CPU_OFFLOAD_PIN_MEMORY=0，断言 offload=True、
pin_memory=False、optimizer implementation=for-loop，session95983
结束 exit0。此项仅为配置装配证据，未运行30B offload训练。

修改文件 Ruff check、format --check、git diff --check 通过。
30B offload 容量、BF16 训练及恢复仍未验收；整体目标继续开放。

## 2026-10-24：CPU offload 与 full AC 八卡正式回归通过

node92 启动前 npu-smi 确认八卡无运行进程。沿用正式
tp_fsdp_equivalence.py，TP2/shard2/replicate2、CPU offload、
pin_memory=False、AdamW for-loop、FP32 tiny Llama、4步、seq32、
global batch8、seed42、deterministic，启用 full AC。保留独立完整
HF 参考、1e-4 绝对误差阈值、mesh 和逐步 CPU 参数/梯度/两个
AdamW 矩状态驻留断言，无 monkeypatch。

session75295 终止 exit0，failed ranks=0，decoder_calls=16，
满足实际重计算断言。loss maxAE4.768e-7、grad norm maxAE5.960e-8、
最终参数 maxAE2.123e-6。日志明确报告 ArgSort AiCPU 与
_assert_async CPU 回退。容器仓库目录复现：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_FSDP_EQ_CPU_OFFLOAD=1 TP_FSDP_EQ_DP_REPLICATE=2 TP_FSDP_EQ_AC=full timeout 240 torchrun --standalone --nproc_per_node=8 tests/integration_tests/tp_fsdp_equivalence.py > outputs/qwen3-30b-npu/logs/offload_formal_tp2_hsdp_fullac_20261024.log 2>&1
```

本轮未改生产实现或测试断言，仅补硬件证据。该结果不覆盖
BF16、selective AC、pin_memory=True、checkpoint 恢复或30B容量，
原无 offload HSDP OOM 和整体 P0–P4 缺口保持开放。

## 2026-10-24：正式 TP/FSDP CPU offload Trainer 回归通过

tp_fsdp_equivalence.py 新增显式 TP_FSDP_EQ_CPU_OFFLOAD=1，使用
pin_memory=False、AdamW for-loop；默认配置保持原行为。每个训练步
检查实际参数、梯度、exp_avg、exp_avg_sq 均为 CPU 本地存储的
DTensor，并要求检查非空。仅最终数值比较将完整 DTensor 搬回
通信设备；训练使用正式 Trainer 和生产裁剪实现，无 monkeypatch。

node92 各次启动前 npu-smi 确认八卡无运行进程。均为 FP32 tiny
Llama、4 步、seq32、global batch8、seed42、deterministic、AC none，
与独立完整 HF 参考比较，保持原绝对误差阈值 1e-4 和 mesh 断言。

| 配置 | loss maxAE | norm maxAE | 参数 maxAE | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- | --- | --- |
| TP2/shard2/replicate1/offload | 4.768e-7 | 5.960e-8 | 1.478e-6 | 通过 | 0 | offload_formal_tp2_dp2_20261024.log |
| TP2/shard2/replicate2/offload | 4.768e-7 | 5.960e-8 | 2.123e-6 | 通过 | 0 | offload_formal_tp2_hsdp_20261024.log |

四卡 session5179、八卡 session97566 均终止 exit0，failed ranks=0。
日志位于 outputs/qwen3-30b-npu/logs/，明确报告 ArgSort AiCPU 和
_assert_async CPU 回退，不声称纯 AiCore。容器仓库目录复现：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TP_FSDP_EQ_CPU_OFFLOAD=1 timeout 240 torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_fsdp_equivalence.py > outputs/qwen3-30b-npu/logs/offload_formal_tp2_dp2_20261024.log 2>&1
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_FSDP_EQ_CPU_OFFLOAD=1 TP_FSDP_EQ_DP_REPLICATE=2 timeout 240 torchrun --standalone --nproc_per_node=8 tests/integration_tests/tp_fsdp_equivalence.py > outputs/qwen3-30b-npu/logs/offload_formal_tp2_hsdp_20261024.log 2>&1
```

本轮仅修改集成测试，Ruff、格式及 diff 检查通过；未重跑全量
CPU/pre-commit。不外推 BF16、AC、pin_memory=True、checkpoint
恢复或 30B 容量，不关闭原无 offload HSDP OOM 和整体 P0–P4 缺口。

## 2026-10-24：TP2×DP2 CPU offload真实Trainer四步通过

使用outputs/probe_offload_trainer.py复用tp_fsdp_equivalence.py的
真实Trainer及独立完整HF参考，保持原loss、grad norm、参数和
非空泛断言及1e-4阈值；显式启用CPU offload、pin_memory=False、
AdamW for-loop，仅最终比较时将CPU DTensor搬至NPU聚合。
未修改训练循环或数值断言，不将诊断入口视为正式集成覆盖。

node92启动前npu-smi确认八卡空闲，容器仓库目录执行：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 timeout 240 torchrun --standalone --nproc_per_node=4 outputs/probe_offload_trainer.py > outputs/qwen3-30b-npu/logs/offload_trainer_tp2_dp2_20261024.log 2>&1
```

session85592终止exit0，四步FP32 tiny Llama、TP2/dense shard2/
replicate1、seq32/globalbatch8、seed42/deterministic、AC none。
loss maxAE4.768e-7，grad norm maxAE5.960e-8，最终参数
maxAE1.478e-6，failed ranks=0。日志明确记录ArgSort AiCPU及
_assert_async CPU回退，不声称纯AiCore。探针Ruff及format通过。

本结果补充真实Trainer的裁剪、有限性检查和AdamW更新证据，
尚未覆盖HSDP复制轴、BF16、AC、pin_memory=True、checkpoint
恢复或30B容量，不关闭原无offload HSDP OOM及整体P0–P4缺口。

## 2026-10-24：offload范数辅助函数边界单测

在test_collectives.py补充10项CPU边界用例，覆盖一/二/无穷
范数下FP16局部范数提升FP32、空本地存储贡献零、Partial布局
在读取存储前拒绝，以及零/负数/NaN范数阶拒绝。这些使用局部
替身对象验证纯计算与保护，不作为真实分布式或NPU证据。

沿用前节node92容器CPU测试环境运行
python -m pytest -q tests/unit_tests/cpu/accelerator/test_collectives.py --disable-warnings，
session91497终止exit0，24 passed、14 warnings、5.39s。新增用例
之后仅改测试文件，未改已通过四卡回归的生产实现；此前4279项
完整CPU通过记录不包含这10项，不宣称新增总数的完整套件已跑。
修改文件Ruff及git diff --check通过。真实BF16/空分片、多mesh、
Trainer/AdamW和恢复验收仍待完成，原30B HSDP容量失败保持。

## 2026-10-24：offload裁剪修复全量CPU及质量门禁复查

生产修复后的完整CPU套件session86077终止exit0：4279 passed、
13 skipped、273 warnings，150.34s。日志位于
outputs/qwen3-30b-npu/logs/offload_clip_cpu_suite_20261024.log；
使用node92容器、outputs/qwen3-30b-test-venv/bin/python、
ASCEND_RT_VISIBLE_DEVICES=0及前节CPU测试PYTHONPATH，执行
python -m pytest -q --disable-warnings。全仓ruff check llmtuner tests
及ruff format --check llmtuner tests通过（458 files），修改文件
内存compile语法检查与git diff --check通过。

容器隔离副本pre-commit初次因Git所有权检查失败；仅为单次命令
设置safe.directory后，因GitHub DNS解析失败而无法下载hook，
session52481 exit3，日志offload_clip_quality_retry_20261024.log。
改用宿主机已有缓存，执行PYTHONPATH=outputs/quality-tools python
outputs/check_quality_snapshot.py，所有自动修复仅作用于临时副本。
首次发现本次新增isinstance类型写法问题，已在生产文件修正。
复查session8883 exit1，副本/tmp/llmtuner-current-quality-c8umoz2q
只修改已有hf_inspection.py两处类型写法及
.agent/karpathy-guidelines/SKILL.md末尾换行；本次裁剪实现及
新增集成脚本无hook修改。未覆盖无关文件，未豁免门禁失败，
不宣称全量pre-commit通过。随后node92原容器仓库目录执行
python -m compileall -q llmtuner，exit0，全量语法编译检查通过；
整体P0–P4及原30B HSDP容量缺口继续开放。

## 2026-10-24：CPU offload生产梯度裁剪修复及回归

collectives.py新增CPU offload梯度范数路径：CPU本地分片计算，
仅将标量送至通信设备，沿Shard轴SUM/MAX，Replicate轴不重复
计数；FP16/BF16局部范数提升FP32，空分片贡献零，未支持的
placement及非正norm_type明确拒绝。裁剪系数回CPU后原地缩放
局部梯度，未搬运完整梯度。混合设备范数合并及TP/EP/PP标量
通信设备选择同步调整；这些扩展组合尚未完成硬件回归。

新增tests/integration_tests/fsdp_cpu_offload_clipping.py，直接调用
生产函数，在四卡HSDP2×2、FP32、pin_memory=False下验证1/2/inf
范数及foreach=False/True。固定max_norm=0.01并要求实际触发
裁剪；全局范数及全部裁剪后梯度对独立完整模型参考使用
atol=rtol=1e-6。完整梯度搬运仅用于测试比较。

node92启动前npu-smi确认无占卡进程，容器仓库目录运行：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 timeout 180 torchrun --standalone --nproc_per_node=4 tests/integration_tests/fsdp_cpu_offload_clipping.py > outputs/qwen3-30b-npu/logs/hsdp_offload_clipping_regression_20261024.log 2>&1
```

session74012结束exit0，六组断言全部通过。原生产探针也已
exit0（session53928），日志hsdp_offload_production_clip_20261024.log。
受影响CPU测试test_collectives.py通过14项、14warnings、5.38s，
session3326结束exit0；使用outputs/qwen3-30b-test-venv/bin/python，
ASCEND_RT_VISIBLE_DEVICES=0及上述PYTHONPATH去除outputs项。
Ruff及格式检查通过。尚未完成全仓门禁、BF16/空分片/Partial
回归、混合并行、真实Trainer/AdamW/恢复或30B offload容量验证；
不关闭原无offload HSDP OOM，不宣称整体P0–P4完成。

## 2026-10-24：offload三种范数实际裁剪诊断通过

扩展outputs/probe_hsdp_offload_clip.py，对一范数、二范数及无穷
范数分别验证标量归约和裁剪后全部完整梯度，atol=rtol=1e-6。
先前max_norm=0.1时无穷范数0.07813458未触发缩放，不能作为
实际裁剪证据；因此新增独立配置max_norm=0.01并断言范数大于
该值，误差阈值保持不变，保留原日志。

node92启动前npu-smi确认设备空闲。容器仓库目录，PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime，
LLMTUNER_PROBE_SCALAR_NORM=1、ASCEND_RT_VISIBLE_DEVICES=0,1,2,3，
依次设置LLMTUNER_PROBE_NORM_ORDER=1、2、inf，各执行
timeout 180 torchrun --standalone --nproc_per_node=4
outputs/probe_hsdp_offload_clip.py，日志分别为
outputs/qwen3-30b-npu/logs/hsdp_offload_active_clip_p1_20261024.log、
hsdp_offload_active_clip_p2_20261024.log、hsdp_offload_active_clip_pinf_20261024.log。
三个命令以成功条件顺序连接，session98299最终exit0，全部完成。

四rank三种范数及裁剪后完整梯度断言通过，梯度在CPU本地缩放，
只有标量参与NPU归约；完整梯度搬运仅用于独立比较。仍限于
FP32小模型HSDP2×2、单步、pin_memory=False、Replicate/Shard
布局，不覆盖BF16、空分片、Partial、多mesh、AdamW或真实Trainer。
本轮未修改生产裁剪实现，原生产offload裁剪失败及30B HSDP容量
失败保持；下一步应正式实现并补回归，不能把诊断通过记为修复。


## 2026-10-24：offload一范数与裁剪后梯度辅助验证

扩展outputs/probe_hsdp_offload_clip.py显式标量诊断分支，增加
LLMTUNER_PROBE_NORM_ORDER，有限正阶使用局部范数的p次幂沿Shard
轴SUM，无穷范数使用MAX；不沿Replicate轴重复计数。裁剪阈值
固定0.1，范数及裁剪后完整梯度与独立完整模型参考比较，
atol=rtol=1e-6。只有比较阶段将梯度搬回NPU聚合。未改生产代码。

node92启动前npu-smi确认八卡无进程，容器仓库目录执行：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime LLMTUNER_PROBE_SCALAR_NORM=1 LLMTUNER_PROBE_NORM_ORDER=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 timeout 180 torchrun --standalone --nproc_per_node=4 outputs/probe_hsdp_offload_clip.py > outputs/qwen3-30b-npu/logs/hsdp_offload_local_clip_p1_20261024.log 2>&1
```

session67330已结束，exit0。四rank范数均14.337223052978516，
与完整参考相同，裁剪后所有完整梯度断言通过。配置仍为四卡
HSDP2×2、FP32小模型、单步、CPU offload且pin_memory=False。
新增二范数及无穷范数裁剪后梯度分支尚未运行，先前二范数证据
仅覆盖范数。BF16、空分片、多mesh、Partial布局、AdamW、真实
Trainer及恢复仍缺证据；生产CPU offload裁剪失败与30B HSDP
容量失败均未关闭。探针Ruff检查通过。

## 2026-10-24：offload本地分片标量范数诊断通过

继续核验CPU offload裁剪方案。首先尝试对CPU DTensor调用
vector_norm再搬运标量；该操作本身已触发CPU functional all_reduce，
仍报无CPU backend，session47951最终exit1。日志为
outputs/qwen3-30b-npu/logs/hsdp_offload_scalar_norm_20261024.log。
因此不能假设DTensor范数会延迟归约到后续full_tensor。

随后只修改outputs/probe_hsdp_offload_clip.py的显式诊断分支：
对gradient.to_local()求二范数平方，仅将标量搬到NPU；按实际
placement仅沿Shard轴SUM，不沿Replicate轴重复计数，再合并参数
贡献。诊断明确拒绝其他placement，尚未接入生产裁剪。

启动前node92八卡npu-smi确认无进程，复现命令：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime LLMTUNER_PROBE_SCALAR_NORM=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 timeout 180 torchrun --standalone --nproc_per_node=4 outputs/probe_hsdp_offload_clip.py > outputs/qwen3-30b-npu/logs/hsdp_offload_local_norm_20261024.log 2>&1
```

session47658终止exit0。四rank实际placement均为
(Replicate(), Shard(dim=0))；全局范数0.5761054754257202，
独立完整参考0.576105535030365，预先设定atol=rtol=1e-6断言
全部通过。梯度保留CPU，仅范数标量通信；此为FP32小模型单步
二范数诊断，尚未检验裁剪后梯度、AdamW更新、其他范数、BF16、
多mesh、空分片、Partial布局或真实Trainer及恢复。不关闭生产
裁剪缺口，也不替代原30B HSDP验收。下一步须将已验证的局部
计算原则形成受限且有明确布局校验的实现，并补齐数值回归。
探针Ruff检查通过；本轮未修改生产代码。

## 2026-10-24：CPU offload生产裁剪通信缺口复现

新增独立诊断outputs/probe_hsdp_offload_clip.py，四卡真实apply_fsdp、
replicate2/shard2、TinyTokenDecoder、FP32、CPU offload且pin_memory=False，
rank不同输入，backward后直接调用生产clip_grad_norm_(max_norm=1,
foreach=True)，不通过比较函数、不改生产实现。启动前npu-smi确认空闲。

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 timeout 180 torchrun --standalone --nproc_per_node=4 outputs/probe_hsdp_offload_clip.py > outputs/qwen3-30b-npu/logs/hsdp_offload_clip_20261024.log 2>&1
```

各rank报告梯度位于CPU；调用栈在collectives.py的_local_norm中
get_total_norm触发DTensor范数部分布局归约，执行CPU functional
all_reduce时报No backend type associated with device type cpu。
这次是生产裁剪路径的实际失败，区别于此前测试full_tensor比较失败。
四步SGD辅助对照仍只证明未调用裁剪的路径，不外推真实Trainer。
后续修复须保留全局范数及复制/分片计数语义，不能把全部梯度搬回
设备而抵消offload容量收益，也不能跳过裁剪。原30B无offload配置
失败保持；本诊断未完成优化器更新。探针Ruff检查通过。

## 2026-10-24：HSDP CPU offload四步辅助对照通过

上一诊断失败明确位于测试比较路径。仅修改outputs诊断入口的_full，
将待比较的CPU DTensor搬到NPU后调用原full_tensor；未改生产路径、
优化器或任何数值断言。设备经npu-smi确认空闲后，沿用下节完整
命令，将日志改为hsdp_cpu_offload_compare_20261024.log。
session66210终止，exit0。四卡replicate2/shard2、TinyTokenDecoder、
FP32、SGD lr0.5、四步、CPU offload及pin_memory=False。

逐步loss为3.646781、3.603068、3.538551、3.518903，与完整参考
在预定1e-4阈值内一致；首步完整梯度及最终权重断言通过，最终权重
maxAE5.960e-8；非空泛检查各rank局部/全局梯度差均超过原1e-3
阈值，failed ranks=0。此结果证明该小模型已有offload前反向及
SGD更新路径可运行，前次比较通信错误不能用于否定训练支持。

仍未验证真实Trainer全局裁剪、AdamW、BF16、AC、checkpoint恢复、
pin_memory=True或30B容量，不能关闭原无offload HSDP失败。下一步
应单独验证生产clip_grad_norm_对CPU DTensor的归约及设备语义，
再决定正式offload集成范围。诊断Ruff/format及git diff --check通过。

## 2026-10-24：HSDP CPU offload辅助诊断的比较边界

核对现有fsdp_cpu_offload配置及训练顺序：全局梯度裁剪先于Adam更新，
不能通过backward内提前更新绕过容量问题而保持原有裁剪语义。
现有offload仅找到CPU配置转发测试，未找到对应NPU集成通过证据。

node92经npu-smi确认设备空闲后，新增outputs/probe_hsdp_cpu_offload.py，
仅在既有fsdp_mesh_equivalence.py入口启用CPU offload及pin_memory=False。
容器仓库目录运行以下独立小模型诊断：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 timeout 180 torchrun --standalone --nproc_per_node=4 outputs/probe_hsdp_cpu_offload.py hsdp > outputs/qwen3-30b-npu/logs/hsdp_cpu_offload_probe_20261024.log 2>&1
```

session18535已结束，exit1。四rank首次backward后，测试脚本第205行
为比较梯度调用_full(p.grad)，CPU DTensor.full_tensor()执行
all_gather_into_tensor时报No backend type associated with device type cpu。
失败发生在测试比较路径、首个opt.step之前，不是已证实的训练更新
失败，也未执行真实Trainer梯度裁剪。不能据此声称offload训练通过或
不支持；下一步须区分比较张量搬运与生产裁剪中的CPU集合通信需求，
保留完整梯度及权重断言。该小模型诊断不替代原30B配置验收，原
无offload HSDP反向/Adam容量失败保持。探针Ruff检查通过，未改生产。

## 2026-10-24：HSDP优化器入口及失败存储实测

扩展outputs/probe_hsdp_backward_memory.py的显式诊断开关
LLMTUNER_HSDP_OPTIMIZER_MEMORY=1，在真实OptimizersContainer.step
前及异常后统计现有参数、梯度和optimizer.state；不调用state_dict、
不初始化状态、不搬运到CPU、不同步设备。storage按设备及地址去重，
同时保留逻辑字节数和dtype。未修改生产代码。

node92八卡经npu-smi确认无进程，沿用下节受控释放运行的全部配置，
增加上述开关，将dump改为outputs/qwen3-30b-npu/hsdp_optimizer_probe_20261024，
日志改为outputs/qwen3-30b-npu/logs/hsdp_optimizer_probe_20261024.log；
同样执行timeout 900 torchrun --standalone --nproc_per_node=8
outputs/probe_hsdp_backward_memory.py。退出码1，session18730已结束。
模型Qwen3-30B-A3B/LIMO、replicate2/shard4、BF16/fullAC、seq4096、
batch8、seed42、deterministic、两步及save_checkpoint=0均不变。

八rank各有before_step与step_exception两条optimizer-rank*.jsonl。
入口参数及梯度均为BF16，各15,266,061,312字节，storage与逻辑字节
相同；allocated为30,532,411,904字节，all-reduce残留为0。
异常时仍无all-reduce残留；偶数rank两个moment分别已分配
13,864,103,424与13,763,440,128字节，allocated58,160,437,248；
奇数rank分别14,075,000,448与13,873,673,856字节，
allocated58,481,577,984。moment实际均为BF16；step标量在CPU。
参数、梯度与已分配moment合计几乎解释全部allocated，未见大型
未归属驻留；这不涵盖PyTorch allocator之外的设备内存。

原始异常仍在Adam._init_group创建exp_avg_sq，申请98或194MiB
失败，首步未完成。完整参数、梯度与两个同dtype moment所需
61,064,245,248字节（约56.87GiB）尚未全部建立便耗尽可用容量。
因此清理反向完成缓冲不能独自解决此配置；不能把最终梯度或moment
误判为FP32，也不能归因于optimizer入口残留通信buffer。
后续须评估保持训练语义的状态/梯度生命周期或明确的内存优化路径，
避免原样重复运行或仅调allocator参数。原配置仍失败，诊断释放
尚未进入生产；整体P0–P4未完成。探针Ruff及format检查通过。

## 2026-10-23：30B HSDP受控释放越过反向，Adam初始化仍OOM

重读原始目标并确认node92八卡空闲，将已完成缓冲释放作为显式诊断
选项LLMTUNER_HSDP_RELEASE_COMPLETED=1接入outputs探针；默认模式
不释放。条件仍为两类事件均完成、所有最终梯度存在且不共享buffer
storage，保留post-reduce事件，不强制同步，不改安装版或生产代码。

容器仓库目录运行：

```bash
export PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime
export LLMTUNER_GLOBAL_BATCH_SIZE=8 LLMTUNER_MAX_SEQ_LEN=4096 LLMTUNER_STEPS=2
export LLMTUNER_SAVE_CHECKPOINT=0 LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE=2 LLMTUNER_DATA_PARALLEL_SHARD_SIZE=4
export LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/hsdp_release_probe_20261023
export LLMTUNER_HSDP_QUERY_EVENTS=1 LLMTUNER_HSDP_RELEASE_COMPLETED=1 LLMTUNER_DETERMINISTIC=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
timeout 900 torchrun --standalone --nproc_per_node=8 outputs/probe_hsdp_backward_memory.py > outputs/qwen3-30b-npu/logs/hsdp_release_probe_20261023.log 2>&1
```

实际exit1，进程组已结束。八rank各100条记录，均到embedding反向
after；累计移除45–48组引用、27,728,475,136至29,597,837,056字节，
最后采样allocated约30.47–32.21GiB。该采样早于finalize，不当作
optimizer开始时的精确驻留量。调用栈明确越过backward与梯度裁剪，
失败在optimizer.step的Adam._init_group创建exp_avg_sq，分配98或
194MiB失败；已分配约54.17–54.47GiB，尚未完成step1。

这将反向归约缓冲峰值与Adam状态容量区分开来：完成缓冲受控释放
在本次诊断中消除了原反向失败点，但不足以使原配置训练成功，亦未
证明30B梯度或更新数值等价。不能将monkeypatch诊断记作生产修复。
下一步应核对实际参数/梯度/moment dtype及状态初始化驻留量，避免
再次重复同一反向释放试跑或通过降低精度/缩序列覆盖原失败。
诊断Ruff及format通过；原HSDP失败及整体P0–P4未完成状态保持。

## 2026-10-23：HSDP完成缓冲受控释放BF16对照

沿用上一节诊断条件，在outputs/hsdp_release_equivalence.py构建两组
相同TinyTokenDecoder，经真实apply_fsdp使用replicate2/shard2四卡
HSDP、BF16参数及默认FP32归约，四步SGD、rank不同输入。参考组
保留原生命周期；候选仅在两类事件query均完成、全部最终梯度存在且
不共享缓冲storage时删除_all_reduce_state，保留_post_reduce_event。
原方法仍执行，不修改安装版源码，不强制同步；逐步比较loss、全部
本地梯度及更新后权重，atol=rtol=0，并逐项要求BF16梯度有限。
候选各rank必须实际释放至少一次，避免空泛通过。

node92启动前npu-smi八卡无进程，容器仓库目录运行：

```bash
PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 timeout 240 torchrun --standalone --nproc_per_node=4 outputs/hsdp_release_equivalence.py > outputs/qwen3-30b-npu/logs/hsdp_release_equivalence_20261023.log 2>&1
```

实际exit0，四rank全部四步精确；rank0/1/2/3释放次数8/12/10/9，
累计移除引用字节6528/8704/7616/7072。这是小模型诊断，CPU搬运
比较会同步每步，未模拟慢通信、梯度累积、AC或AdamW，亦未证明
30B容量及性能。弱引用集合可能包含两组已观察对象，生产迁移仍需
明确模型所有权和作用域；不将本结果推广为通用释放实现。
诊断Ruff及format通过；首次导入排序检查失败后仅修正诊断脚本。
本轮未修改生产实现，未关闭原30B HSDP OOM及其他P0–P4缺口。

## 2026-10-23：HSDP完成缓冲释放的诊断条件

重读原始目标、开发指南及安装版FSDP后处理顺序。扩展忽略目录
outputs/probe_hsdp_backward_memory.py，只读记录最终梯度dtype及其
storage是否与归约缓冲共享；尚未运行新增别名观测。node92设备检查
八卡空闲，本轮没有启动训练。

新增独立诊断outputs/hsdp_completed_buffer.py，不接入生产或训练：
仅当all-reduce和post-reduce两事件query均完成、组内全部最终梯度
存在且均不共享缓冲storage时，移除_all_reduce_state引用；保留
_post_reduce_event及原通信等待逻辑，不强制同步。六项替身条件检查
exit0，覆盖事件未完成、梯度缺失、storage别名、成功及重复调用。
这些只证明分支条件，不验证真实allocator、异步通信或数值正确性。

两份诊断脚本Ruff通过，探针format及内存compile语法检查通过。
下一步仍须真实BF16参数/FP32归约的对照验证，再决定是否接入原30B
诊断；未修改生产实现，原HSDP失败及整体P0–P4未完成状态保持。

## 2026-10-23：HSDP归约缓冲完成事件实测

重读原始目标，扩展忽略目录outputs/probe_hsdp_backward_memory.py的
可选LLMTUNER_HSDP_QUERY_EVENTS=1，只读查询每组all-reduce与
post-reduce事件；不释放引用、不同步设备、不替换通信。源码显示
all-reduce事件先于dtype转换及梯度累积，故不能只检查前者。
NPU Event.query接口检查通过。node92八卡经npu-smi确认空闲后运行：

```bash
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime
export LLMTUNER_QWEN3_30B_PATH=/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B
export LLMTUNER_DATASET_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl
export LLMTUNER_GLOBAL_BATCH_SIZE=8 LLMTUNER_MAX_SEQ_LEN=4096 LLMTUNER_STEPS=2
export LLMTUNER_SAVE_CHECKPOINT=0 LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE=2 LLMTUNER_DATA_PARALLEL_SHARD_SIZE=4
export LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/hsdp_event_probe_20261023
export LLMTUNER_HSDP_QUERY_EVENTS=1 LLMTUNER_DETERMINISTIC=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
timeout 900 torchrun --standalone --nproc_per_node=8 outputs/probe_hsdp_backward_memory.py > outputs/qwen3-30b-npu/logs/hsdp_event_probe_20261023.log 2>&1
```

任务最终exit1，首次backward重现OOM，rank0分配770MiB失败；不是
探针接口异常。八rank各78条记录，最后均layer10，39组保留缓冲
23,989,751,296字节；其中38组的两类事件均完成，共23,366,630,656
字节，layer10自身尚未完成。逐rankJSONL位于上述dump目录。
这证明本次失败前有大量已完成操作仍持有缓冲，提供评估受控释放的新
证据，但尚未验证allocator跨stream复用、后续消费者和梯度正确性；
未修改生产实现、未提前释放缓冲，不关闭原HSDP失败。
诊断脚本内存compile语法检查通过；py_compile首次因既有容器缓存目录
权限失败，未改权限。整体P0–P4未完成。

## 2026-10-23：HSDP历史内存捕获八rank增长审计

重读原始目标及开发指南，发现原HSDP探针及退出记录已经存在，因此
没有重复启动训练。node92当前八卡有python3.10任务占用，本轮仅用
CPU解析历史JSONL，不干扰该任务。首次进程查询因shell引号错误未取得
进程列表，但npu-smi明确报告八个占卡PID，不据此误称设备空闲。

八rank各78条采样、39个after事件，全部最后记录到layer10，保留39组
all-reduce缓冲23,989,751,296字节，allocated55,193,243,648字节。
从首个after至最后after，allocated增长36,128,161,792字节，其中
retained缓冲增长23,678,584,320字节，差额12,449,577,472字节；
八rank这些统计完全相同，缓冲字节数均单调非减。差额只是记账，
不将其直接认定为梯度或泄漏，也不证明保留缓冲可以提前安全释放。

报告outputs/qwen3-30b-npu/logs/hsdp_retention_growth_audit_20261023.json。
首次宿主机写报告因容器目录权限失败，未改权限；通过node92原容器
重新解析保存成功，exit0。原探针run.log保留首次反向OOM及非零退出。
本轮无生产改动、无新增硬件通过结论；下一诊断应核验事件完成后释放
协议，不能再次重复相同峰值采样。原HSDP失败及整体P0–P4缺口保持。

## 2026-10-23：30B HSDP反向OOM的内存几何核对

重读原始目标，核对P4后续正式化记录，确认显式四项配置的100步指标
已有证据，未重复早期融合消融。本轮转向尚未关闭的30B HSDP2×4 OOM，
读取原始rank0/3完整调用栈：仅定位至autograd backward，不能确定具体
分配算子。当前FSDP默认无显式预取窗口，零值仍保留torch隐式预取。

在node92容器只读提取当前torch FSDP源码并保存至outputs日志目录的
hsdp_collectives_source_20261023.txt及hsdp_param_group_source_20261023.txt。
实际foreach_reduce在清空unsharded_grads之前分配reduce_dtype输入，
HSDP另保留all-reduce输入引用直至backward结束；存在明确事件等待，
没有证据证明引用泄漏或可安全提前释放，不修改第三方实现。

读取目标config.json独立计算：单层单个专家投影201326592元素，
BF16为384MiB、FP32为768MiB；48层三个投影在shard4下仅专家BF16
参数13.5GiB、对应FP32梯度27GiB。结果保存为
outputs/qwen3-30b-npu/logs/hsdp_memory_geometry_20261023.json，提取
进程exit0。报告torch版本2.12.0+cpu是安装版本字符串，不据此推断
torch-npu不可用。估算排除dense、padding、激活、optimizer和临时缓冲，
384/768MiB与失败382/770MiB相近不能作为分配来源证明。

本轮未启动训练、未修改生产实现或改变精度/序列/offload条件，原HSDP
失败保持。下一有效诊断应在原配置反向期间记录各FSDP组及通信缓冲的
实际驻留量和最后成功边界，不能用内存算术或allocator开关替代根因证据。
整体P0–P4未完成。

## 2026-10-23：EP HF初始加载声明纠偏

重读原始目标并核对README、设计文档与当前assembly.py的实际保护条件。
发现两份文档仍笼统声明EP拒绝HF初始加载，与现有条件放行实现不一致。
已修正为：Qwen3MoE须启用checkpoint.sharded_expert_parallel，TP>1
还须启用checkpoint.sharded_tensor_parallel；Kimi Linear使用独立
布局适配，其他EP模型仍拒绝。依据为trainer/runtime/assembly.py
的load_hf_weights保护及HFTransformerStateDictAdapter构造参数，
既有四方案目标训练和恢复证据保持各自适用范围，不新增运行通过结论。

本轮仅修正文档声明，未改生产代码、未启动硬件任务。git diff --check
通过，不重跑训练或CPU全套。完整P2声明映射、原冷恢复失败、compile、
KCP及P4等缺口仍开放，整体目标未完成。

## 2026-10-23：TP×EP TorchAO与AC的DP复制补验

连续三轮历史失败局部审计未确定首次分歧，本轮转向独立缺证据配置，
未修改生产代码或正式脚本。重读原始目标及开发指南，node92容器
npu-smi确认八卡无进程后，复用tp_fsdp_equivalence.py，固定world4、
TP2/EP2/dense shard1/replicate2、TorchAO、tiny Qwen3MoE两层hidden64、
FP32、seq32/globalbatch8、seed42/deterministic、四步真实Trainer/AdamW。
独立完整HF参考不使用AC或生产dispatcher，原绝对误差阈值1e-4不变。

node92容器仓库目录复现命令：

```bash
export PYTHONPATH=.:outputs/torchao-runtime:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
export TP_FSDP_EQ_EP=2 TP_FSDP_EQ_DISPATCHER=torchao TP_FSDP_EQ_DP_SHARD=1 TP_FSDP_EQ_DP_REPLICATE=2
for mode in full selective; do
  export TP_FSDP_EQ_AC="$mode"
  timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_fsdp_equivalence.py > "outputs/qwen3-30b-npu/logs/tp_ep_torchao_replicate_ac_${mode}_20261023.log" 2>&1
  result=$?; echo "$mode exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
```

full/selective均exit0，failed ranks0/all checks passed，decoder_calls16，
实际dispatcher类型、mesh与专家分片断言全部通过。两组逐步loss、grad norm、
最终全部参数最大AE均分别4.768e-7、2.384e-7、8.057e-6。保留日志中
CPU/AiCPU回退及格式警告，不声称纯AiCore或性能通过。两组任务已结束。
只新增有限组合证据和状态文档，不重复源码门禁；计数hook可能扰动时序，
不以本次连续训练通过关闭原精确冷恢复失败，不外推至30B或P4。
整体P0–P4目标仍未完成。

## 2026-10-23：历史失败捕获输出链连接审计

重读原始目标及恢复测试，上一轮输出投影重放属于新增局部证据，本轮
继续检查同一次失败捕获的模块间连接，未启动训练或NPU计算。读取
outputs/tp-ep-cp-ac-metrics-only-3-20261023中的
forward-failed_detail-resume-rank{0..3}.pt，weights_only=True、
map_location=cpu，仅选择step3原前向output事件。每rank44个事件。
最终model.model.norm输出reshape为512×64后与model.lm_head输入
逐位相同；model.lm_head输出与model最终返回值均为512×128且逐位相同。
四rank两项断言全部通过，审计进程exit0，报告保存至
outputs/qwen3-30b-npu/logs/failed_output_chain_audit_20261023.json。

该证据只覆盖此详细捕获内的两处连接，不证明连续参考输入一致，也不
证明另一次retained-logits失败运行的张量生命周期。没有拼接两次运行
作因果结论；原fullAC恢复失败保持失败，整体P0–P4未完成。


## 2026-10-23：历史恢复失败输出投影重放

重读原始目标及当前证据，node92容器npu-smi确认八卡无运行进程后，
新增忽略目录outputs/replay_failed_ac_head.py。仅使用同一历史失败目录
outputs/tp-ep-cp-ac-metrics-only-3-20261023的step3原前向lm_head
捕获及split-state权重；先验证rank0/1、rank2/3权重分片各自相同，再
拼接0/2重建128×64完整权重。四rank的512×64输入分别以存储偏移
0/1/7/31执行独立torch.mm，每种重复五次，与512×128捕获输出逐位比较。

首次运行exit1，因仅暴露NPU0而checkpoint包含其他设备编号，加载阶段
失败，尚未执行数值比较；保留failed_head_replay_20261023.log。修正
诊断脚本为map_location=cpu后运行，weights_only=True和精确断言不变：

```bash
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime ASCEND_RT_VISIBLE_DEVICES=0 timeout 120 python outputs/replay_failed_ac_head.py > outputs/qwen3-30b-npu/logs/failed_head_replay_cpu_load_20261023.log 2>&1
```

实际exit0，checked80/failures为空，任务已结束；保留DeviceMesh及分配
padding警告。仅证明保存输入和重建权重的局部输出投影可精确重放，
不能证明连续训练与恢复输入一致或异步生命周期正确，不关闭原恢复失败。
未修改生产代码，整体P0–P4仍未完成。

## 2026-10-23：历史恢复失败全部归一化重放

重读原始目标及开发指南，保留当前工作区改动；上一轮仅汇总四组既有
运行结果，无新增验证。本轮node92容器npu-smi确认八卡无运行进程后，
扩展忽略目录outputs/replay_failed_ac_norm.py，读取同一历史失败目录
outputs/tp-ep-cp-ac-metrics-only-3-20261023的split-state与
forward-failed_detail-resume-rank*.pt，仅使用step3原前向输出事件。
范围由input_layernorm扩大至两层input/post_attention layernorm、两层
Q/K norm及最终model.norm，共九个模块、四rank、四个存储偏移0/1/7/31。
先断言rank0/1及rank2/3权重分片分别一致，再拼接0/2重建完整权重；
独立RMSNorm公式逐位比较捕获输出，未调整容差或修改生产代码。

node92容器仓库目录运行：

```bash
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime ASCEND_RT_VISIBLE_DEVICES=0 timeout 120 python outputs/replay_failed_ac_norm.py > outputs/qwen3-30b-npu/logs/failed_all_norm_replay_20261023.log 2>&1
```

实际exit0，checked144/failures为空，任务已结束。保留DeviceMesh反序列化
及NPU分配padding警告。该证据仅表明保存输入在这些存储偏移下的局部
归一化可精确重放，不证明原始连续运行输入相同、异步生命周期正确或恢复
失败已修复；没有连续参考的同次详细捕获，仍不能确定首次分歧。原恢复
失败、compile、KCP和P4等缺口保持开放。

## 2026-10-23：TP×EP TorchAO与AC独立训练参考补验

本轮重读原始目标及当前状态，不改源码，复用正式tp_fsdp_equivalence.py，
在node92容器运行前用npu-smi确认八卡无进程。配置固定TP2/EP2、dense
shard2/replicate1、world4、tiny Qwen3MoE两层hidden64、FP32、seq32/batch8、
seed42/deterministic、四步AdamW及TorchAO，分别full/selective AC。
独立完整HF参考不启用AC或生产dispatcher，原绝对误差1e-4不变。

容器仓库目录复现：

```bash
export PYTHONPATH=.:outputs/torchao-runtime:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
export TP_FSDP_EQ_EP=2 TP_FSDP_EQ_DISPATCHER=torchao TP_FSDP_EQ_DP_SHARD=2 TP_FSDP_EQ_DP_REPLICATE=1
for mode in full selective; do
  export TP_FSDP_EQ_AC="$mode"
  timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_fsdp_equivalence.py > "outputs/qwen3-30b-npu/logs/tp_ep_torchao_ac_${mode}_20261023.log" 2>&1
  result=$?; echo "$mode exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
```

两组均实际exit0、failed ranks0/all checks passed；dispatcher=torchao、
decoder_calls16，实际类型、wrapper及专家分片断言通过。两组loss、grad norm、
最终全部参数最大AE分别4.768e-7、2.384e-7、8.057e-6。保留日志中的CPU/AiCPU
回退及格式警告，不声称纯AiCore或性能通过。所有作业已终止。
本轮仅补既定路径硬件证据及状态文档，git diff --check通过，不重复源码门禁。
计数hook可能扰动时序，不以连续训练通过关闭原fullAC冷恢复失败；不外推
DP复制×TorchAO×AC、30B或P4，完整目标仍未完成。

## 2026-10-23：TP×EP×DP的TorchAO真实Trainer参考

正式`tp_fsdp_equivalence.py`增加`TP_FSDP_EQ_DISPATCHER`（默认alltoall），
通过ExpertParallelConfig选择dispatcher，TorchAO模式逐MoE断言实际
TorchAOTokenDispatcher类型。独立参考仍完整HF，不使用生产dispatcher。
保持原四步AdamW、TP2/EP2、FP32、seq32/batch8、两层tiny MoE、AC none，
分别DP shard2/replicate2四rank；逐步loss及grad norm和全部最终权重仍用
原绝对误差1e-4，有限值、实际mesh/专家数量和TP不同切片断言全部保留。

node92起跑前npu-smi八卡无进程，使用既有离线TorchAO依赖，未安装新包：

```bash
export PYTHONPATH=.:outputs/torchao-runtime:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
export TP_FSDP_EQ_EP=2 TP_FSDP_EQ_DISPATCHER=torchao
for mode in shard replicate; do
  export TP_FSDP_EQ_DP_SHARD=2 TP_FSDP_EQ_DP_REPLICATE=1
  if [ "$mode" = replicate ]; then export TP_FSDP_EQ_DP_SHARD=1 TP_FSDP_EQ_DP_REPLICATE=2; fi
  timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_fsdp_equivalence.py > "outputs/qwen3-30b-npu/logs/tp_ep_torchao_trainer_${mode}_20261023.log" 2>&1
  result=$?; echo "$mode exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
python -m compileall -q llmtuner tests/integration_tests/tp_fsdp_equivalence.py
```

两组均exit0、failed ranks0/all checks passed；loss/grad norm/最终权重
最大AE均分别4.768e-7/2.384e-7/8.057e-6。全部进程结束，容器compileall
exit0，正式脚本Ruff/format、全部适用pre-commit与diff check通过。
仅集成脚本改变，不重复CPU全套；已有全量pre-commit无关失败不改变。
不以AllToAll证据替代TorchAO，亦不将本次AC none外推到AC/恢复、FP8或30B。
原fullAC冷恢复、Inductor、KCP及P4权重标准等缺口仍开放，整体目标未完成。

## 2026-10-23：checkpoint局部缓存重排审计

重读原始要求后检查ChunkedCheckpointTensor.refresh_from的别名防护，未发现
可据以修改生产实现的反例。忽略目录脚本
`outputs/audit_chunk_cache_permutations.py`枚举同一CPU存储的四个非连续chunk
全部24种排列；每次刷新及再次刷新均检查期望chunk内容精确一致，并检查
原始模型存储逐位不变。node92容器运行exit0，24项全部通过。

```bash
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime python outputs/audit_chunk_cache_permutations.py > outputs/qwen3-30b-npu/logs/chunk_cache_permutations_20261023.json
```

该证据仅覆盖CPU局部缓存的存储重排，不证明NPU异步生命周期或原fullAC
恢复失败已修复。没有因未发现问题而修改生产代码，也不将其重复计为新的
checkpoint端到端通过。未启动NPU训练；诊断脚本Ruff/format与diff check通过。
后续恢复调查仍需同次失败中的首次分歧证据，整体目标保持未完成。

## 2026-10-23：TP2×EP2真实Trainer/AdamW独立参考

后续补足实际EP非空泛断言：装配后检查EP mesh size2、MoE层数与配置一致，
每层inner_experts.num_experts=4，三个专家投影均为DTensor且全局首维4。
此处DTensor全局shape表示本EP坐标持有的专家，不误用其FSDP局部首维。
保留原TP不同切片、dense FSDP局部存储以及全部数值断言，不仅依赖EP配置。

node92起跑前npu-smi确认八卡无进程。沿用本节完整PYTHONPATH、TP2/EP2/
shard2四rank和四步AdamW配置，依次设置TP_FSDP_EQ_AC=none/full/selective，
执行同一正式脚本；日志为`outputs/qwen3-30b-npu/logs/`下
`tp_ep_topology_assert_{none,full,selective}_20261023.log`，三组实际exit0。
每组loss/grad norm/最终参数最大AE分别4.768e-7/2.384e-7/8.057e-6，
failed ranks0，full/selective的decoder_calls均16。容器compileall通过，
脚本Ruff/format、适用pre-commit及diff check通过，所有进程已结束。
本轮只加强正式集成脚本，不新增生产实现、阈值或支持范围，不重复CPU全套。
原精确冷恢复失败与整体P0–P4缺口保持开放。

进一步加入`TP_FSDP_EQ_AC=full/selective`，默认none不变；真实Trainer
应用生产AC，独立HF参考不包装AC。逐层断言CheckpointWrapper，底层decoder
仅注册整数计数pre-hook、不clone张量；四步×两层至少16次调用才通过，
防止仅配置AC却没有实际重算。该hook仍可能扰动时序，成功不覆盖冷恢复失败。
原loss/grad norm/全部最终权重标准1e-4及finite保护不变。

node92启动前npu-smi无设备进程，四rank，TP2/EP2/dense shard2、FP32、
seq32/batch8、四步AdamW、无EMA或恢复。使用下方完整PYTHONPATH，执行：

```bash
export TP_FSDP_EQ_EP=2
for mode in full selective; do
  export TP_FSDP_EQ_AC="$mode"
  timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_fsdp_equivalence.py > "outputs/qwen3-30b-npu/logs/tp_ep_trainer_ac_final_${mode}_20261023.log" 2>&1
  result=$?; echo "$mode exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
python -m compileall -q llmtuner tests/integration_tests/tp_fsdp_equivalence.py
```

两组均exit0、decoder_calls16、failed ranks0；独立非AC参考与生产AC的
loss/grad norm/参数最大AE均分别4.768e-7/2.384e-7/8.057e-6。
初轮full训练完成后在测试参考名字查找抛KeyError，日志
`tp_ep_trainer_ac_full_20261023.log`保留；原因是测试未去AC wrapper前缀，
正式修复使用仓库canonical_fqn，不修改训练参数名或断言。它不算数值失败。
最终full/selective分别独立复跑以上日志，未拼接首轮局部观测。
容器compileall、脚本Ruff/format、适用pre-commit hooks及diff check通过。
当前源码隔离快照全量pre-commit仍exit1，仅原hf_inspection两处Ruff与
karpathy技能EOF，日志`outputs/tp_ep_trainer_ac_precommit_20261023.log`，
无无关自动修正回写。仅集成脚本变更，不重复不收集该脚本的CPU全套。
本次不替代无hook的冷恢复，也未引入全局同步绕过原失败；fullAC间歇恢复、
compile、KCP及P4权重缺口保持，整体目标未完成。

后续补齐grad norm指标：独立HF参考记录clip_grad_norm_返回的裁剪前范数，
与Trainer.train_step实际metrics[grad_norm]逐步比较，保留逐步loss及最终
全部参数断言。范数仍采用本测试原绝对误差1e-4，显式拒绝非有限值；没有
擅用P4跨框架标准替换本仓库并行等价标准。实际Trainer范数路径同时包含
EP专家去重与TP分片/复制参数归约，本次检查不是从loss或最终权重推断。

node92起跑前npu-smi八卡无进程，完整复现沿用下方最终命令，只将日志名
`tp_ep_trainer_adamw_final_${mode}_20261023.log`替换为
`tp_ep_trainer_norm_${mode}_20261023.log`。三组已全部终止exit0：

| 配置 | 四步loss最大AE | 四步grad norm最大AE | 最终参数最大AE |
| --- | --- | --- | --- |
| TP2/EP2/shard2 | 4.768e-7 | 2.384e-7 | 8.057e-6 |
| TP2/EP2/replicate2 | 4.768e-7 | 2.384e-7 | 8.057e-6 |
| 原dense TP2/shard2 | 4.768e-7 | 5.960e-8 | 1.478e-6 |

每份日志failed ranks0/all checks passed，容器compileall exit0；受影响
脚本Ruff/format、全部适用pre-commit hooks及diff check通过。当前源码
隔离快照全量pre-commit exit1，仍仅既有hf_inspection与karpathy技能修正，
日志`outputs/tp_ep_trainer_norm_precommit_20261023.log`；未回写无关更改。
本轮仅正式集成脚本改变，CPU全套不收集它，保留最近4279passed/13skipped
证据而不重复；没有宣称原fullAC恢复失败、compile、KCP或P4关键权重通过。

正式`tp_fsdp_equivalence.py`新增可选`TP_FSDP_EQ_EP=2`，默认仍EP1/Llama。
MoE模式为Qwen3MoE两层hidden64、8experts/topk2、expert F48、aux0，TP2、
FP32、seq32/globalbatch8、seed42/deterministic、AdamW lr3e-4/weight_decay0、
4steps，无AC。独立完整HF参考按DP各自flatten的序列边界分别forward，再按
全局有效token归约，独立裁剪及AdamW更新；不是恢复自洽或手写生产替代循环。
生产训练使用实际Trainer.train_step及其归约/裁剪/optimizer。

最终参数按实际TP投影类型选择行/列切片，router映射到HF gate；每层三个
专家投影独立映射HF gate_up/down并按EP rank切片，断言专家项数量为层数×3。
保留不同TP权重、实际FSDP局部分片及复制placement非空泛检查。
loss逐步绝对误差与全部最终权重误差仍为原1e-4，新增显式finite检查，
避免原`NaN > TOL`为False造成漏判；参考loss统计detach消除测试自身告警。
仅最终权重比较，不宣称逐步全部梯度或optimizer moments比较。

node92起跑前npu-smi确认八卡无进程，使用四卡分别运行dense shard2和
replicate2；EP共享设备，不额外乘world。初始MoE分片运行也exit0，日志
`tp_ep_trainer_adamw_shard_20261023.log`，随后finite/detach修正后独立重验。
最终命令在容器仓库根目录执行：

```bash
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
for mode in shard replicate dense; do
  export TP_FSDP_EQ_EP=2 TP_FSDP_EQ_DP_SHARD=2 TP_FSDP_EQ_DP_REPLICATE=1
  if [ "$mode" = replicate ]; then export TP_FSDP_EQ_DP_SHARD=1 TP_FSDP_EQ_DP_REPLICATE=2; fi
  if [ "$mode" = dense ]; then export TP_FSDP_EQ_EP=1; fi
  timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_fsdp_equivalence.py > "outputs/qwen3-30b-npu/logs/tp_ep_trainer_adamw_final_${mode}_20261023.log" 2>&1
  result=$?; echo "$mode exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
python -m compileall -q llmtuner tests/integration_tests/tp_fsdp_equivalence.py
```

| 配置 | loss最大AE | 最终全部参数最大AE | 状态/退出码 |
| --- | --- | --- | --- |
| TP2/EP2/shard2，world4 | 4.768e-7 | 8.057e-6 | 通过/0 |
| TP2/EP2/replicate2，world4 | 4.768e-7 | 8.057e-6 | 通过/0 |
| 原dense TP2/shard2回归 | 4.768e-7 | 1.478e-6 | 通过/0 |

最终三份日志均failed ranks0/all checks passed，全部进程终止；容器compileall
exit0。全仓Ruff和457文件format、受影响脚本pre-commit及git diff --check
通过。当前git可见源码隔离快照全量pre-commit exit1，仍仅hf_inspection两处
Ruff及karpathy技能EOF，日志`outputs/tp_ep_trainer_adamw_precommit_20261023.log`，
未回写无关修改。此前完整CPU4279passed/13skipped仍有效；本轮仅集成脚本
改变且不由pytest收集，未重复CPU全套。保留CPU/AiCPU显式回退警告，不宣称
纯AiCore或性能通过。不外推30B、AC、EMA、checkpoint或P4；完整目标未完成。

## 2026-10-23：TP4×EP2共享mesh的三步SGD独立参考

进一步补全全局loss断言：HF完整序列loss按独立DP组平均；生产侧TP token
切片的summed CE除32×DP后，对world求和。此fixture world=TP4×DP、EP复用
rank空间，不额外重复EP归约。逐步比较标量仍用原atol1e-5/rtol1e-4，
同时保持logits、全部梯度与HF实际更新权重断言。权重统计前detach，避免
测试额外建立autograd图和requires_grad标量转换告警，未屏蔽NPU回退告警。

node92起跑前npu-smi八卡无进程。完整环境与下节三步命令一致，各模式日志
改为`tp_ep_global_loss_{none,shard,replicate}_20261023.log`，路径仍为
`outputs/qwen3-30b-npu/logs/`。三组已exit0，每组三个loss比较与全部原断言
通过；逐份检查标量转换告警已消失。none/shard/replicate的loss最大AE
分别9.537e-7/9.537e-7/4.768e-7；
none/shard/replicate更新权重最大AE分别1.192e-7/7.451e-9/7.451e-9，
为本次运行观测，不与旧运行拼成同次结果。容器compileall exit0，正式脚本
Ruff/format、全部适用pre-commit hooks及git diff --check通过。
CPU上轮4279passed结果保留，本轮仅修改不由pytest收集的集成脚本，未重复
全套CPU。此次验证仍是小模型SGD，不代替真实Trainer、30B或精确恢复验收；
整体P0–P4目标未完成。

后续参考覆盖修正：更新权重不再用`HF权重-lr*参考梯度`的手算公式，
改为双方optimizer.step后直接读取独立HF模型的实际权重，再按EP/TP坐标
切片比较。这样也覆盖参考optimizer实际执行，而不只验证SGD代数公式。
dense参数数量断言改为排除专家模块实际parameter身份后的计数，不假设HF
融合专家固定只有两个参数；本轮仅实测融合布局，不声明4.x分立布局已通过。

node92起跑前npu-smi八卡无进程；沿用下节完整PYTHONPATH、三步与拓扑，
各模式分别执行，日志改为`tp_ep_actual_optimizer_{none,shard,replicate}_20261023.log`，
均位于`outputs/qwen3-30b-npu/logs/`，各exit0且三步均all checks passed。
三步最大AE（打印精度）如下，阈值仍为原atol1e-5/rtol1e-4：

| DP模式 | logits | expert梯度 | dense梯度 | HF实际更新权重 |
| --- | --- | --- | --- | --- |
| none | 1.192e-7 | 1.863e-9 | 2.980e-8 | 7.451e-9 |
| shard2 | 1.192e-7 | 1.630e-9 | 2.235e-8 | 7.451e-9 |
| replicate2 | 1.788e-7 | 9.313e-10 | 2.235e-8 | 7.451e-9 |

这些是新运行结果，不覆盖或拼接旧运行。最终计数断言在首组运行加载后
调整，因此另跑最终默认一步回归，exit0，日志
`tp_ep_actual_optimizer_default_final_20261023.log`；容器compileall也exit0。
受影响脚本全部pre-commit
hooks、Ruff和format通过；宿主compileall因容器生成pycache的写权限退出1，
不修改目录权限，改在容器重验。仅正式集成脚本改变，未改生产训练实现；
上一轮完整CPU4279passed证据仍有效，本轮脚本不由pytest收集，不重复CPU全套。

扩展正式`tests/integration_tests/ep_fsdp_without_dense_dp.py`，新增
`LLMTUNER_TEST_OPTIMIZER_STEPS`默认1；每步清梯度，seed7+dp_rank+101*step
生成不同输入，参考与生产模型各自SGD(lr0.05)更新。原logits、三类专家及全部
dense梯度atol1e-5/rtol1e-4保持，新增逐参数更新权重比较，参考来自HF自身
权重减去独立参考梯度，不从生产权重构造期望值。HF融合/分立expert权重均有
读取分支，实际本轮使用融合布局。每步检查12项dense及三类expert。

固定tiny Qwen3MoE单层hidden64/intermediate128、FP32、32tokens、8experts/
topk2、TP4/EP2、CP1、无AC、aux0，先无DP world4，再DP分片2和复制2 world8。
实际分片存储、复制placement及不同DP参考梯度差>1e-3断言保留。仅是模型/
SGD循环，不替代真实Trainer、AdamW、EMA、恢复或30B验收。node92起跑前
npu-smi确认八卡无进程；各NPU进程组均已终止，三种模式实际exit0。

容器仓库根目录，完整共用环境：

```bash
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
LLMTUNER_TEST_OPTIMIZER_STEPS=3 timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/ep_fsdp_without_dense_dp.py > outputs/qwen3-30b-npu/logs/tp_ep_sgd_three_steps_none_runtime_20261023.log 2>&1
LLMTUNER_TEST_DP_MODE=shard LLMTUNER_TEST_OPTIMIZER_STEPS=3 timeout 300 torchrun --standalone --nproc_per_node=8 tests/integration_tests/ep_fsdp_without_dense_dp.py > outputs/qwen3-30b-npu/logs/tp_ep_sgd_three_steps_shard_20261023.log 2>&1
LLMTUNER_TEST_DP_MODE=replicate LLMTUNER_TEST_OPTIMIZER_STEPS=3 timeout 300 torchrun --standalone --nproc_per_node=8 tests/integration_tests/ep_fsdp_without_dense_dp.py > outputs/qwen3-30b-npu/logs/tp_ep_sgd_three_steps_replicate_20261023.log 2>&1
```

下表为三步打印精度下最大AE，不将其改为验收阈值：

| DP模式 | logits | expert梯度 | dense梯度 | 更新权重 | 状态/退出码 |
| --- | --- | --- | --- | --- | --- |
| none | 1.192e-7 | 1.630e-9 | 4.098e-8 | 7.451e-9 | 通过/0 |
| shard2 | 1.192e-7 | 9.313e-10 | 2.235e-8 | 1.490e-8 | 通过/0 |
| replicate2 | 1.192e-7 | 1.164e-9 | 2.235e-8 | 7.451e-9 | 通过/0 |

默认一步配置回归也exit0，日志`tp_ep_sgd_default_regression_20261023.log`，
命令沿用共用PYTHONPATH，不设置测试环境变量，torchrun4；容器compileall
通过。首轮漏带隔离依赖，在colorama导入阶段exit1，保留
`tp_ep_sgd_three_steps_none_20261023.log`，不视为数值失败或删除。

全仓Ruff/format457files、diff check及受影响正式脚本pre-commit均通过。
`outputs/check_quality_snapshot.py`复制当前git可见文件到临时独立git仓库，
执行全量pre-commit exit1，仅修改hf_inspection两处Ruff和karpathy技能EOF；
日志`outputs/tp_ep_multistep_precommit_20261023.log`，没有回写无关自动修正。
CPU首轮隐藏设备的完整pytest exit1：4069passed/210failed/13skipped，
256.68s，日志`tp_ep_multistep_cpu_20261023.log`。首批traceback表明
torch-npu optimizer补丁即使CPU参数也查询current_device，空的
ASCEND_RT_VISIBLE_DEVICES导致aclInit107001。此为实际失败，不冒充门禁通过；
已在空闲NPU0可见环境重验CPU全套，exit0：4279passed/13skipped/
273warnings，149.56s，日志`tp_ep_multistep_cpu_device_visible_20261023.log`。
复现为容器仓库目录执行
`ASCEND_RT_VISIBLE_DEVICES=0 outputs/qwen3-30b-test-venv/bin/python -m pytest -q --disable-warnings`。
受影响正式测试与状态文档pre-commit适用hooks全部通过，辅助快照脚本
Ruff/format通过。上述所有运行均已终止；全量pre-commit的原无关失败仍保留。
整体P0–P4未完成，原fullAC精确恢复失败、compile和KCP缺口保持。

### 归约布局接口进一步核对

契约检查已保存为outputs/audit_installed_reduction_axis_order.py，
在node92容器仓库目录执行python outputs/audit_installed_reduction_axis_order.py，
exit0；报告outputs/qwen3-30b-npu/logs/installed_reduction_axis_order_20261023.json。
脚本AST提取实际安装方法，三项构造检查分别确认一个冲突及两个一致对照；
Ruff和格式检查通过。这是依赖接口诊断，不是整模型运行时轴序捕获。

进一步通过AST提取安装版ReductionAnalysis.dense_size_list原方法，构造
三项独立接口检查：tiling=[x1,x0,r2]、golden=[r2,x1,x0]、reduced_dim=2
时生成[X1,X0,R2]而mask顺序为[X0,X1,R2]；另两项顺序一致的对照保持一致，
命令退出0。此检查证明方法在这些输入上的契约冲突，但构造输入不是整模型
运行时捕获，不证明所有kernel都满足该条件。应先在隔离副本统一单归约轴
的shape来源，再验证真实生成代码；多归约轴路径须单独保留并验证。

本轮真实NPU最小复现后，只读检查安装版依赖：
codegen/triton.py::IterationRangesEntryNPUIndex.get_axis_direction按
reversed(golden_var_list)确定mask轴；非归约dense_size_list亦按此顺序。
codegen/kernel_analysis.py::ReductionAnalysis.dense_size_list则使用
tiling_axis并将最后的归约轴插入reduced_dim；reduced_dim本身又从
reversed(golden_var_list)取得。因此数据形状和mask采用不同轴序来源，
不能直接通过交换mask或强制方形block认定通用修复。下一步应在隔离
依赖副本验证统一轴序的生成代码及非对称归约前反向，再复验整模型。
未修改共享安装依赖，未将最小kernel成功计为整模型compile通过。

## 2026-10-23：归约mask真实NPU最小复现

本轮重读原始目标和开发指南，上一轮详细恢复探针未捕获失败，本轮转向
独立compile缺口。node92的npu-smi确认八卡无运行进程后，仅使用NPU0。
运行既有outputs/reproduce_npu_reduction_mask.py，未设置TRITON_INTERPRET，
真实Triton编译执行；session84838结束，exit0。

命令：node92容器仓库目录执行
`PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 timeout 120 python outputs/reproduce_npu_reduction_mask.py`。
日志outputs/qwen3-30b-npu/logs/npu_reduction_mask_real_20261023.log；
机器可读报告outputs/qwen3-8b-npu/logs/npu_reduction_mask_repro_20261023.json。
torch-npu2.12.0，保留设备分配padding警告。

| block / 有效范围 | 修正mask | 原mask |
| --- | --- | --- |
| 1×4×8 / 1×3 | 逐位精确 | reshape编译错误 |
| 4×1×8 / 3×1 | 逐位精确 | reshape编译错误 |
| 4×4×8 / 2×3 | 逐位精确 | 可执行但数值错误 |

只读核对当前安装torch_npu/_inductor/codegen/triton.py：reduction约1221行
按dense_size_str重排value，约1259行仍直接连接原mask。新证据证明修正
mask的最小kernel能在真实设备执行，且强制方形block不能解决语义错误；
尚未修复依赖的通用轴映射，也没有验证整模型编译前反向或恢复。
下一步须从ReductionAnalysis取得数据轴映射，保持mask与数据布局一致，
再检验真实生成kernel；不直接修改归档源码作为正式修复。
本轮未改生产源码或依赖，原整模型compile及冷恢复失败保持，整体目标未完成。

## 2026-10-23：CP1联合分片EMA详细前向捕获

重读原始目标和开发指南，保留既有工作区。上一轮CP1无探针恢复失败
提供了新配置证据，本轮对该配置运行一组独立详细捕获。node92启动前
npu-smi确认八卡无运行进程，未修改生产计算或原精确断言。
配置沿用上一节联合EMA的NPU4/DP-shard2/TP2/EP2/CP1、FP32、
seq256/globalbatch8/fullAC/dynamic EMA/RNG/loader及两种分片选项。
新增环境LLMTUNER_PROBE_DETAILED=1、LLMTUNER_PROBE_TAG=jointdetail，
入口为outputs/probe_ac_forward_boundary.py，独立目录
outputs/tp-ep-joint-ema-detail-20261023，依次执行full和resume。
两阶段均由timeout 300 torchrun --standalone --nproc_per_node=4启动，
PYTHONPATH及其他环境完整沿用下节，未预热或改变AC。

full session33858 exit0，四步loss与原无探针连续参考相同；resume
session20366 exit0，原模型/optimizer/EMA/RNG/loader及loss精确断言通过。
使用python outputs/compare_ac_forward_boundaries.py
outputs/tp-ep-joint-ema-detail-20261023 jointdetail:full jointdetail:resume --all
比较，session71794 exit0，四rank各292个事件、step3/4指标全部精确，
changed_events=0、metrics_changes=[]。日志均位于
outputs/qwen3-30b-npu/logs/，分别为
tp_ep_joint_ema_detail_full_20261023.log、
tp_ep_joint_ema_detail_resume_20261023.log及
tp_ep_joint_ema_detail_comparison_20261023.log。

详细钩子及张量克隆会扰动执行，成功不能覆盖上一轮无探针失败，也未
定位首次分歧。原CP1联合EMA及CP2恢复失败保持，不宣称修复；不重复
累积同方式成功样本。下一有效调查需要不同观察量或明确执行状态差异。
本轮训练与审计进程均结束，仅更新状态记录，无生产源码修改。

## 2026-10-23：联合分片EMA真实NPU冷恢复失败

上一轮实现EMA联合分片并通过Gloo及完整单测；本轮重读原始目标，
扩展真实Trainer恢复脚本的chunk审计：比较全局shape/dtype、chunk数量、
偏移及各局部内容，保持精确标准；Adam moments非零检查改为逐局部块；
weights_only加载明确许可本仓库ChunkedCheckpointTensor类型。
增加LLMTUNER_TEST_SHARDED_TENSOR_CHECKPOINT和
LLMTUNER_TEST_SHARDED_EXPERT_CHECKPOINT两个配置入口，默认均0。

node92启动前npu-smi确认无运行进程。NPU4、PP1/DP-shard2/TP2/EP2/CP1、
native FSDP、FP32、seq256、globalbatch8、seed42/deterministic、fullAC、
动态EMA/RNG/loader，两个checkpoint分片选项均1。连续4步与独立2+2比较。
目录outputs/tp-ep-joint-ema-npu-20261023。连续阶段session53461 exit0，
loss=[4.867846965789795,4.859529495239258,4.865467071533203,4.866306304931641]。
安全加载修正后的恢复session82929 exit1：四rank pre-update state failures=0，
随后step3 loss=4.865463733673096，与参考差3.338e-6；最终模型、optimizer、
EMA均有精确比较失败。原断言保留，不改容差，不宣称NPU联合EMA通过。

日志位于outputs/qwen3-30b-npu/logs/：
tp_ep_joint_ema_npu_full_20261023.log和
tp_ep_joint_ema_npu_resume_safe_20261023.log。首个恢复日志
tp_ep_joint_ema_npu_resume_20261023.log因测试快照安全加载许可缺失exit1，
并非DCP加载失败；此前远程参数展开失败日志tp_ep_joint_ema_npu__20261023.log
保留，其外层exit0不代表训练通过。其后一次shell引号错误未启动训练。

复现：node92容器仓库目录，PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime；
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3，LLMTUNER_TEST_WORLD_SIZE=4、
LLMTUNER_TEST_PP_SIZE=1、LLMTUNER_TEST_TP_SIZE=2、LLMTUNER_TEST_EP_SIZE=2、
LLMTUNER_TEST_CP_SIZE=1、LLMTUNER_TEST_SEQ_LEN=256、LLMTUNER_TEST_MOE=1、
LLMTUNER_TEST_FSDP_IMPLEMENTATION=native、LLMTUNER_TEST_AC=full、
LLMTUNER_TEST_EMA=dynamic、LLMTUNER_TEST_RNG_STATE=1、
LLMTUNER_TEST_LOADER_STATE=1、上述两个分片选项均1；分别执行
timeout 300 torchrun --standalone --nproc_per_node=4
tests/integration_tests/pp_checkpoint_equivalence.py full（或resume）
outputs/tp-ep-joint-ema-npu-20261023。CPU断言、ArgSort AiCPU及优化器
for-loop回退明确记录。两个实际训练进程组均已结束。

此失败在CP1出现，不能将恢复问题限定为KVCP路径；恢复前状态一致只证明
保存状态覆盖项正确，不排除运行时执行差异。本轮未修复数值偏差，
整体P0–P4仍未完成；下一步应定位同次连续/恢复的首个前向分歧。

## 2026-10-23：EMA联合TP/EP分片恢复实现与回归

重读原始目标及开发指南，保留当前工作区和索引。上一轮局部norm重放
取得证据但未修复原恢复失败，本轮转向独立的EMA联合分片实现缺口。
EMA继承OptimizersContainer的联合chunk保存/恢复路径，构造函数仍有旧的
无条件拒绝。本轮移除该拒绝，继续复用既有disjoint TP/EP所有权保护，
不允许同一参数同时注册两种分片所有权。来源说明更新至upstream map。

扩展tests/integration_tests/tp_ep_checkpoint_layout.py：四进程Gloo、
TP2/EP2独立dense/expert参数，动态EMA第一步保存、冷构造恢复，再执行
第二步更新，与连续更新的全部chunk偏移及内容按atol=rtol=0比较。
所有rank精确通过，exit0，session29329结束；日志
outputs/qwen3-30b-npu/logs/tp_ep_joint_ema_cold_20261023.log。
命令：node92容器仓库目录，PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
timeout 180 torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_ep_checkpoint_layout.py。

同一PYTHONPATH、ASCEND_RT_VISIBLE_DEVICES=0，以
outputs/qwen3-30b-test-venv/bin/python -m pytest -q --disable-warnings
执行：EMA定向88passed；完整4279passed/13skipped/273warnings，
150.46s，exit0，session86091结束。日志分别为
tp_ep_joint_ema_unit_20261023.log和tp_ep_joint_ema_full_suite_20261023.log，
均位于outputs/qwen3-30b-npu/logs。设备使用前npu-smi确认无运行进程。
全仓Ruff、format457files、容器compileall及git diff --check通过。

隔离副本刷新最初因相对路径SameFileError失败，后以绝对源路径刷新四份
本轮文件重跑全量hooks，session51062 exit0；该副本保留此前两个无关文件
的自动修正，故仅证明本轮文件兼容hooks，不能证明原工作区全量门禁通过。
原hf_inspection及karpathy技能EOF失败仍保留，未应用修正或豁免。

本次Gloo布局恢复及单测不代表真实NPU联合分片EMA训练验收；真实Trainer
恢复脚本目前还需扩展chunk状态审计，不能直接外推其原普通tensor比较。
目标30B四方案本次未重跑，原TP2/EP2/CP2/fullAC恢复失败未关闭，
整体P0–P4仍未完成。

## 2026-10-23：失败input RMSNorm输入的局部重放

本轮重读原始目标，沿用仓库开发指南与现有版本、工作区，不改生产实现。
上一轮唯一调用审计属于进展；本轮补查尚未重放的input RMSNorm前向。
node92的npu-smi确认八卡无运行进程后，仅使用NPU0。
新增忽略目录outputs/replay_failed_ac_norm.py，读取同一历史失败目录
outputs/tp-ep-cp-ac-metrics-only-3-20261023的split-state与
forward-failed_detail-resume-rank*.pt，仅比较step3原前向input_layernorm。
按既定CP组重建权重：rank0/1、rank2/3对应分片须精确相等，再拼接0和2；
执行原生FP32 RMSNorm的pow/mean/rsqrt及乘法顺序，epsilon=1e-6。
四rank×两层×offset0/1/7/31，共32次逐位精确，failures=[]，exit0，
session4662结束。日志outputs/qwen3-30b-npu/logs/ac_failed_norm_replay_cpu_load_20261023.log。

复现：node92容器仓库目录执行
`PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime ASCEND_RT_VISIBLE_DEVICES=0 timeout 120 python outputs/replay_failed_ac_norm.py`。
首次因colorama缺失在导入阶段exit1，日志ac_failed_norm_replay_20261023.log；
补齐既有运行依赖路径后，未映射存储位置导致单卡加载其他设备产物失败，
日志ac_failed_norm_replay_runtime_20261023.log；两份失败均保留。
最终显式map_location=cpu加载，再将计算输入搬到NPU，未改变数值断言。
保留无PG的DeviceMesh反序列化警告及NPU分配padding警告，本次无collective/compile。

仅证明给定失败输入与权重的input RMSNorm局部前向可重放，不覆盖其他norm、
反向、真实分布式时序或未捕获的连续参考输入；原恢复失败与整体缺口保持。
未启动整网训练，不以局部成功替代P0–P4验收。

## 2026-10-23：失败恢复的唯一调用重计算审计

重新读取原始目标及开发指南，核对当前仓库commit为
c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19，torchtitan-npu为
b358a8063c34aa7494397efa3a8f5ad24f1caa1f，torchtitan为
f5c97dc0e2d6cca72313004aa853a196ca02ab49；保留现有工作区及索引。
上一轮仅汇总已有四方案结果，本轮转向未关闭恢复失败的新证据。

原重计算审计在同名模块候选中选择差异最少者，不能证明唯一调用归属。
本轮读取同一历史失败产物的事件顺序，确认两个decoder层共享CP kernel，
原前向按层0→1、重计算按层1→0；按该顺序固定匹配，不按误差选候选。
其他模块要求唯一原前向候选；层wrapper的重复输入须先精确相等。
新增忽略目录诊断outputs/audit_ac_unique_recompute.py，仅CPU读取产物，
未启动NPU训练、未改变checkpoint或生产实现。

node92容器仓库目录执行：
`python outputs/audit_ac_unique_recompute.py outputs/tp-ep-cp-ac-metrics-only-3-20261023 failed_detail`。
exit0，session16545已结束。四rank各132个重计算事件按固定归属比较，
unique_matching_failures=[]；日志
outputs/qwen3-30b-npu/logs/ac_unique_recompute_audit_20261023.log。
该结果补强同次失败恢复中原前向与重计算一致的证据，不证明连续参考与
恢复前向相等，不覆盖未捕获事件或反向collective，也不关闭原精确恢复失败。
下一调查应针对连续/冷启动前向差异或反向执行状态，不重复候选匹配审计。
仅诊断及状态文档变更，不重复完整pytest和训练；既有门禁失败保持。

## 2026-10-23：四方案后全量pre-commit复验

完整pytest可见设备复跑已确认exit0，4279 passed、13 skipped。
使用缓存wheel离线安装pre-commit4.7至/tmp/torchllmtuner-precommit-runtime，
复制当前tracked及未忽略新增文件到/tmp/llmtuner-matrix-gates-0rjz9uh3，
仅在副本初始化临时Git索引，未切换或修改原仓库索引。执行
PYTHONPATH=/tmp/torchllmtuner-precommit-runtime python -m pre_commit run --all-files。
session94111结束，exit1：Ruff0.9自动修正hf_inspection.py两处tuple
isinstance为union；end-of-file-fixer补.agent/karpathy-guidelines/SKILL.md
末尾换行。其余ruff-format、trailing-whitespace、yaml、merge-conflict、
mixed-line-ending hooks均通过。副本diff已核实，仅上述两文件变化。
原工作区未应用无关修正，未豁免失败；全量pre-commit门禁仍失败，
不能以完整pytest通过宣称整体P0–P4完成。

## 2026-10-23：四方案修复后完整pytest环境复验

node92容器、仓库目录执行完整pytest。首轮隐藏NPU运行exit1：
4069 passed、210 failed、13 skipped、219 warnings，257.34s；日志
outputs/qwen3-30b-npu/logs/post_matrix_full_cpu_20261023.log。
首个失败堆栈确认CPU AdamW默认实现选择进入torch_npu/utils/_optim.py，
查询npu.current_device导致aclInit107001；另有Inductor及非阻塞拷贝失败。
不据此声称每个失败具有同一根因，不修改测试或生产代码绕过。

确认node92八卡空闲后，保留设备0可见，以相同解释器、依赖和全部断言
重跑完整套件：4279 passed、13 skipped、273 warnings，175.14s。
日志outputs/qwen3-30b-npu/logs/post_matrix_visible_device_suite_20261023.log。
命令：ASCEND_RT_VISIBLE_DEVICES=0
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
outputs/qwen3-30b-test-venv/bin/python -m pytest -q --disable-warnings。
这是安装torch-npu且设备可见环境下的完整单测证据，不声称纯CPU环境
通过或替代分布式NPU验收。原隐藏设备失败日志保留；pre-commit与其他
未关闭矩阵项仍需完成，整体P0–P4未完成。

## 2026-10-23：DP8/EP8目标独立恢复完整状态通过

node92八卡无运行进程后，独立进程从dp8_ep8_auxfix_20261023的
step-1恢复到step-2，总步数和调度均保持2步。新输出目录
outputs/qwen3-30b-npu/dp8_ep8_resume_20261023，checkpoint/step-1
为原连续训练目录的绝对符号链接，仅新增step-2。恢复作业exit0，
session50414终止；step2 loss0.42917、grad_norm2.3705。
日志outputs/qwen3-30b-npu/logs/dp8_ep8_resume_20261023.log。

复现：node92容器仓库目录，PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime；
LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE=1、LLMTUNER_PIPELINE_PARALLEL_SIZE=1、
LLMTUNER_CONTEXT_PARALLEL_SIZE=1、LLMTUNER_DATA_PARALLEL_SHARD_SIZE=8、
LLMTUNER_TENSOR_PARALLEL_SIZE=1、LLMTUNER_EXPERT_PARALLEL_SIZE=8、
LLMTUNER_STEPS=2、LLMTUNER_GLOBAL_BATCH_SIZE=8、LLMTUNER_MAX_SEQ_LEN=4096、
LLMTUNER_DETERMINISTIC=1、LLMTUNER_SAVE_CHECKPOINT=1、
LLMTUNER_CHECKPOINT_INTERVAL=1、LLMTUNER_ENABLE_TENSORBOARD=1、
LLMTUNER_SHARDED_EXPERT_CHECKPOINT=1、LLMTUNER_SHARDED_TENSOR_CHECKPOINT=0、
LLMTUNER_CHECKPOINT_LOAD_STEP=1、MASTER_PORT=29670、LLMTUNER_DUMP_FOLDER
为上述新目录，执行bash examples/train_qwen3_30b_npu.sh。
模型、数据、精度、AC、seed沿用原30B入口，未启用EMA。

完整审计执行python outputs/compare_30b_checkpoint_chunks.py
outputs/qwen3-30b-npu/dp8_ep8_auxfix_20261023/checkpoint/step-2
outputs/qwen3-30b-npu/dp8_ep8_resume_20261023/checkpoint/step-2，exit0，
session43524终止；日志
outputs/qwen3-30b-npu/logs/dp8_ep8_resume_state_audit_20261023.log。
21473个chunk全部精确一致：model4632、optimizer16791、rng_state8、
dataloader39、lr_scheduler1、train_state2，failures=[]。
该结果仅覆盖本次两步恢复，不外推跨布局数值等价、EMA或整体P0–P4。

## 2026-10-23：DP4/TP2目标独立恢复完整状态通过

node92八卡空闲后，沿用dp4_tp2_chunkfix_20261023全部训练配置，独立
进程从step-1恢复到step-2，总步数及调度保持2步。新输出目录为
outputs/qwen3-30b-npu/dp4_tp2_resume_20261023；checkpoint/step-1
使用原连续训练目录的绝对符号链接，原始产物未覆盖。
恢复exit0，session85196终止；step2 loss0.38214、grad_norm2.8902。
日志outputs/qwen3-30b-npu/logs/dp4_tp2_resume_20261023.log。

复现沿用四方案DP4/TP2环境（shard4/TP2/EP1、专家分片0、TP分片1、
两步/seq4096/batch8/fullAC/BF16/seed42/deterministic/checkpoint间隔1），
增加LLMTUNER_CHECKPOINT_LOAD_STEP=1，MASTER_PORT=29669，
LLMTUNER_DUMP_FOLDER为上述新目录，执行bash examples/train_qwen3_30b_npu.sh。
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime；
replicate/PP/CP均1，保存checkpoint及TensorBoard均1。

完整审计执行python outputs/compare_30b_checkpoint_chunks.py
outputs/qwen3-30b-npu/dp4_tp2_chunkfix_20261023/checkpoint/step-2
outputs/qwen3-30b-npu/dp4_tp2_resume_20261023/checkpoint/step-2，
exit0，session76033终止；日志
outputs/qwen3-30b-npu/logs/dp4_tp2_resume_state_audit_20261023.log。
17917个chunk全部精确一致：model3660、optimizer14223、rng_state8、
dataloader23、lr_scheduler1、train_state2，failures=[]，未启用EMA。

本轮全仓ruff check及format --check通过（457 files），git diff --check
通过。宿主机Python3.13 compileall因已有__pycache__目录权限失败；
node92训练容器内python -m compileall -q llmtuner重跑exit0。
尚未重跑完整pytest/pre-commit，不宣称整体质量门禁通过。
该恢复证据仅限本配置；DP8/EP8独立恢复及整体P0–P4仍待完成。

## 2026-10-23：TP2/EP4目标独立恢复完整状态通过

沿用tp2_ep4_jointfix_20261023连续两步的全部配置，独立进程从step-1
恢复到step-2，总步数仍为2、调度不变。独立输出目录
outputs/qwen3-30b-npu/tp2_ep4_resume_20261023，其checkpoint/step-1
为原连续训练step-1目录的绝对符号链接，仅写新的step-2。启动前node92
npu-smi确认设备无进程。恢复作业exit0，session97152已终止；日志
outputs/qwen3-30b-npu/logs/tp2_ep4_resume_20261023.log。

复现沿用下节目标训练全部环境，仅修改MASTER_PORT=29668、
LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/tp2_ep4_resume_20261023，
增加LLMTUNER_CHECKPOINT_LOAD_STEP=1；执行同一训练shell入口。
恢复step2 loss0.38290、grad_norm2.8798。

完整审计在node92容器仓库目录执行：python
outputs/compare_30b_checkpoint_chunks.py
outputs/qwen3-30b-npu/tp2_ep4_jointfix_20261023/checkpoint/step-2
outputs/qwen3-30b-npu/tp2_ep4_resume_20261023/checkpoint/step-2。
退出码0，session61626终止；日志
outputs/qwen3-30b-npu/logs/tp2_ep4_resume_state_audit_20261023.log。
18541个chunk全部精确一致：model3660、optimizer14847、rng_state8、
dataloader23、lr_scheduler1、train_state2，failures=[]。RNG解码后比较，
不以pickle字节编码差异判断随机状态。该配置未启用EMA，恢复证据仅限
本次两步目标配置，不扩展至其他布局或整体P0–P4。

## 2026-10-23：四方案两步目标训练全部完成

TP2/EP4联合chunk修复后目标复跑exit0，loss0.63443→0.38290，
grad_norm4.2626→2.8798，最终完整训练checkpoint保存15.83s。
输出outputs/qwen3-30b-npu/tp2_ep4_jointfix_20261023，日志
outputs/qwen3-30b-npu/logs/tp2_ep4_jointfix_20261023.log。session72064已终止。
启动前npu-smi确认八卡无运行进程。日志明确报告_assert_async CPU及
ArgSort AiCPU回退，不声称纯AiCore。

复现：node92容器仓库目录，PYTHONPATH沿用下节；设置
LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE=1、LLMTUNER_PIPELINE_PARALLEL_SIZE=1、
LLMTUNER_CONTEXT_PARALLEL_SIZE=1、LLMTUNER_DATA_PARALLEL_SHARD_SIZE=4、
LLMTUNER_TENSOR_PARALLEL_SIZE=2、LLMTUNER_EXPERT_PARALLEL_SIZE=4、
LLMTUNER_STEPS=2、LLMTUNER_GLOBAL_BATCH_SIZE=8、LLMTUNER_MAX_SEQ_LEN=4096、
LLMTUNER_DETERMINISTIC=1、LLMTUNER_SAVE_CHECKPOINT=1、
LLMTUNER_CHECKPOINT_INTERVAL=1、LLMTUNER_ENABLE_TENSORBOARD=1、
LLMTUNER_SHARDED_EXPERT_CHECKPOINT=1、LLMTUNER_SHARDED_TENSOR_CHECKPOINT=1、
MASTER_PORT=29667、LLMTUNER_DUMP_FOLDER为上述输出目录，执行
bash examples/train_qwen3_30b_npu.sh，输出到上述独立日志。

配置允许DCP联合分片，运行时拒绝TP/EP参数所有权重叠；Qwen3 MoE
HF初始化在TP>1时要求同时启用两个分片选项。EMA联合模式仍拒绝。
配置回归test_joint_sharded_checkpoint_requires_dcp通过（1 passed）。
四组DP8、DP4/TP2、TP2/EP4、DP8/EP8均完成本次两步训练及保存；
旧失败日志保留。尚不证明目标配置独立进程恢复、跨布局数值等价或整体
P0–P4通过。

## 2026-10-23：联合chunk显式恢复验证

新增restore_disjoint_tp_ep_state：先生成独立TP/EP目标chunk并校验所有
chunk布局及dtype，再复制到调用方本地存储，返回本地参数形状。
tp_ep_checkpoint_layout.py增加独立clone源、NaN破坏、显式恢复的精确断言。
node92隐藏NPU运行Gloo4，退出码0；日志
outputs/qwen3-30b-npu/logs/tp_ep_joint_restore_20261023.log。
复现：容器仓库目录下设置ASCEND_RT_VISIBLE_DEVICES为空及
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime，
执行timeout 120 torchrun --standalone --nproc_per_node=4
tests/integration_tests/tp_ep_checkpoint_layout.py。局部Ruff检查、格式及diff检查通过。

追加错误专家chunk布局负例，断言拒绝后所有本地参数保持原值；Gloo4复跑
exit0，日志outputs/qwen3-30b-npu/logs/tp_ep_joint_restore_guard_20261023.log。

ModelCheckpointState已接入独立TP/EP联合chunk保存和恢复；配置层及optimizer
联合模式保护仍保留。四进程脚本追加wrapper DCP保存、NaN破坏及恢复精确
断言，exit0，日志outputs/qwen3-30b-npu/logs/tp_ep_joint_model_wrapper_20261023.log。

OptimizersContainer已接入联合chunk保存和恢复。四进程脚本追加模型与
optimizer共同DCP保存、恢复及下一步更新，参数和全部optimizer tensor
精确一致，exit0；日志outputs/qwen3-30b-npu/logs/tp_ep_joint_optimizer_20261023.log。
追加全新optimizer恢复：加载前断言inner state为空，恢复后下一步参数和
全部optimizer tensor与连续运行精确一致，Gloo4 exit0；日志
outputs/qwen3-30b-npu/logs/tp_ep_joint_cold_optimizer_20261023.log。
配置层与EMA联合保护仍保留；尚无独立进程恢复证据。
追加缺失专家源chunk负例，拒绝后所有本地参数保持原值；完整Gloo4脚本
复跑exit0，日志outputs/qwen3-30b-npu/logs/tp_ep_joint_missing_source_20261023.log。

新增tp_ep_dp_checkpoint_layout.py，八进程Gloo构建dense(4,2)和
sparse(2,4) mesh，验证DP/TP与efsdp/EP的DTensor偏移、模型wrapper
DCP恢复及完整权重重建，全部精确一致，exit0。日志
outputs/qwen3-30b-npu/logs/tp2_ep4_dp_layout_20261023.log。
复现沿用上述环境，torchrun --standalone --nproc_per_node=8
tests/integration_tests/tp_ep_dp_checkpoint_layout.py，timeout 120。
该脚本手工注册参数布局，未覆盖真实模型并行变换或NPU执行。

八进程脚本追加真实HFTransformerModel Qwen3 MoE，经apply_tp和apply_ep
转换后的联合wrapper DCP恢复精确通过，日志
outputs/qwen3-30b-npu/logs/tp2_ep4_actual_model_checkpoint_20261023.log，exit0。
追加由原始HF权重构造的逐专家safetensors源，将全部参数NaN破坏后通过
生产HF adapter/reader加载，全部dense/router/expert参数精确恢复，exit0；
日志outputs/qwen3-30b-npu/logs/tp2_ep4_actual_hf_load_20261023.log。
两项均CPU/Gloo8，小模型真实TP/EP变换，未应用FSDP或执行目标NPU训练。

尚未验证真实模型八rank optimizer恢复。组合配置保护仍保留，
TP2/EP4目标训练仍未通过，四组训练及整体P0–P4验收状态沿用下文。

## 2026-10-23：独立TP/EP参数的联合chunk验证

新增validate_disjoint_tp_ep_checkpoint_ownership及shard_disjoint_tp_ep_state，
仅允许dense TP和专家EP分别持有参数，同一参数两种所有权重叠明确拒绝。
正式Gloo4脚本tp_ep_checkpoint_layout.py增加交集负例、全局shape与偏移、
DCP保存及完整权重重建、全部本地参数NaN破坏后原地加载精确断言。
各轮exit0，最终日志outputs/qwen3-30b-npu/logs/tp_ep_disjoint_restore_20261023.log。
此前分阶段日志tp_ep_ownership_guard、tp_ep_disjoint_chunks、
tp_ep_disjoint_dcp_20261023.log保留。node92隐藏NPU，沿用runtime PYTHONPATH，
命令timeout 120 torchrun --standalone --nproc_per_node=4
tests/integration_tests/tp_ep_checkpoint_layout.py。Ruff及diff检查通过。

这仅证明普通tensor的独立TP/EP所有权表示，不证明实际8rank mesh、
HF联合加载、DTensor、optimizer或生产wrapper接入。现有组合配置和
HF初始化保护尚未解除；四方案中TP2/EP4仍未完成训练，其余三组两步通过
范围沿用前节，整体P0–P4仍未完成。

## 2026-10-23：DP4/TP2分片checkpoint修复后目标复跑

CheckpointConfig新增sharded_tensor_parallel显式开关，传入manager模型wrapper、
optimizer及EMA；限制DCP后端，与专家分片选项同时开启明确拒绝。30B入口
通过LLMTUNER_SHARDED_TENSOR_CHECKPOINT=1启用，默认关闭。
现有TP2×DP2/Gloo4普通及packed布局、冷optimizer恢复和下一步精确对齐
测试exit0，日志tp_config_wiring_integration_20261023.log。

目标DP4/TP2/EP1复跑exit0，完成两步及完整训练checkpoint；loss
0.63489→0.38214，grad_norm4.2312→2.8902，最终checkpoint耗时22.65s。
日志outputs/qwen3-30b-npu/logs/dp4_tp2_chunkfix_20261023.log，输出目录
outputs/qwen3-30b-npu/dp4_tp2_chunkfix_20261023。命令沿用下节DP4/TP2，
增加LLMTUNER_SHARDED_TENSOR_CHECKPOINT=1，端口29666并替换输出目录。
启动前npu-smi确认设备全部空闲；session27182已终止。ArgSort日志明确
报告AiCPU回退，不声称纯AiCore。Ruff/format及diff检查通过。

原dp4_tp2_matrix的OOM失败保留，本次仅证明两步训练与保存，不证明独立
进程恢复或跨布局数值对齐。当前四方案三组完成两步；TP2/EP4仍被HF
初始化保护拒绝，联合EP/TP checkpoint布局尚需验证，P0–P4未完成。

## 2026-10-23：用户指定四方案实际运行结果

四组均使用node92容器、Qwen3-30B-A3B/LIMO、8rank、BF16/fullAC、seed42、
deterministic、2步、seq4096、global batch8及完整训练checkpoint。

| 配置 | 实际结果 | 状态 | 退出码 | 日志（outputs/qwen3-30b-npu/logs/） |
| --- | --- | --- | --- | --- |
| dp_shard8/TP1/EP1 | loss0.53537→0.43012，grad_norm3.3731→2.3510，完整checkpoint保存完成 | 通过 | 0 | dp8_ep1_pair_20261023.log |
| dp_shard4/TP2/EP1 | checkpoint wrapper构造中gather_tensor_parallel_state/full_tensor聚合OOM，未训练；rank0已有54GiB分配，再申请386MiB失败 | 失败 | 1 | dp4_tp2_matrix_20261023.log |
| TP2/EP4，dense dp_shard4、sparse efsdp2 | 实际mesh确认4/2/4；EP/HF初始化保护拒绝，未加载权重或训练 | 失败 | 1 | tp2_ep4_matrix_20261023.log |
| dp_shard8/TP1/EP8，专家分片checkpoint | AuxLoss物化修复后loss0.53600→0.42917，grad_norm3.4291→2.3705，完整最终checkpoint保存15.67s | 通过 | 0 | dp8_ep8_auxfix_20261023.log |

所有对应tool session均已返回终止退出码，不再存活。通过仅限本次两步配置，
未做两布局数值等价验收，不以loss接近证明对齐或独立进程恢复通过。
第二、三组启动前均用npu-smi确认无设备进程，按顺序运行，没有并发抢占。

第二、三组复现沿用下节两组的PYTHONPATH及训练环境，明确设置
LLMTUNER_SHARDED_EXPERT_CHECKPOINT=0，TP=2、dp_shard=4；第二组EP=1、
MASTER_PORT=29664、DUMP_FOLDER=outputs/qwen3-30b-npu/dp4_tp2_matrix_20261023；
第三组EP=4、MASTER_PORT=29665、DUMP_FOLDER=outputs/qwen3-30b-npu/tp2_ep4_matrix_20261023。
分别执行bash examples/train_qwen3_30b_npu.sh并输出到表中独立日志。
第四组沿用dp8_ep8_pair命令，端口29663、DUMP_FOLDER改为
outputs/qwen3-30b-npu/dp8_ep8_auxfix_20261023，保留SHARDED_EXPERT_CHECKPOINT=1。
入口配置回归10 passed/14 warnings、exit0，日志ep8_entry_config_20261023.log。
下一步修复TP分片checkpoint生产接入及TP/EP联合HF加载；不删除保护、
不关闭checkpoint或随机初始化来绕过失败。P0–P4整体目标仍未完成。

## 2026-10-23：修复AuxLoss meta累计buffer重建

materialize_meta_model识别AuxLoss.instance_acc，按其构造契约重建为目标
设备FP32零标量；仍为non-persistent，不初始化HF参数，不移除未知buffer保护。
新增MicrobatchWiseLoadBalanceLoss meta测试，验证shape/dtype/零值及checkpoint
排除，并与unknown buffer拒绝测试一起2 passed/85 deselected、exit0，日志
`outputs/qwen3-30b-npu/logs/ep_aux_materialization_verified_20261023.log`。
首轮测试NameError及后续疑似共享文件pyc时间戳缓存旧代码失败日志保留为
ep_aux_materialization_unit/fixed_20261023.log；修改测试系数刷新文件后通过。
较宽materialization集合另有原RoPE整网前向测试在隐藏NPU环境inductor触发
aclInit107001，未宣称该项通过。Ruff/format及diff检查通过。
尚未复跑目标EP8；下一步确认设备空闲，以独立目录复跑原2步配置。

## 2026-10-23：两组本次运行结果

| 配置 | 结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- |
| dp_shard8/EP1，2步/seq4096/batch8 | step1 loss0.53537/grad_norm3.3731；step2 loss0.43012/grad_norm2.3510；完整训练checkpoint保存完成 | 通过 | 0 | outputs/qwen3-30b-npu/logs/dp8_ep1_pair_20261023.log |
| dp_shard8/EP8，同训练配置，开启专家分片checkpoint | 已通过原EP/HF保护，meta物化因MicrobatchWiseLoadBalanceLoss.instance_acc非checkpoint buffer缺少重建契约退出，未训练 | 失败 | 1 | outputs/qwen3-30b-npu/logs/dp8_ep8_pair_20261023.log |

两次作业均终止，session19064/7939已返回退出码。第二组不再是原guard失败，
下一步检查负载均衡统计buffer的初始化和物化语义，不删除未知buffer保护或
随机初始化HF权重。两组命令见下节。此结果不构成P0–P4整体通过。

## 2026-10-23：两组运行的第一组完成、第二组启动

dp8_ep1_pair_20261023实际exit0，完成2步及完整最终checkpoint，step2 loss
0.43012、grad_norm2.3510，最终checkpoint耗时18.05s。仅本次两步配置通过，
不外推P0–P4。设备重新确认全部无进程后启动第二组dp8_ep8_pair_20261023，
session7939仍需轮询，尚无退出码。

assembly仅对Qwen3 MoE、TP1且checkpoint.sharded_expert_parallel开启的HF
初始化使用原生专家adapter并放行，其他EP保护保留；30B入口增加环境开关
LLMTUNER_SHARDED_EXPERT_CHECKPOINT（默认0）。Ruff/format及diff检查通过，
新装配路径的目标NPU结果待本次运行验证。

第二组命令同下一节第一组，明确替换以下参数，其余全部相同：

```bash
export LLMTUNER_EXPERT_PARALLEL_SIZE=8 LLMTUNER_SHARDED_EXPERT_CHECKPOINT=1
export LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/dp8_ep8_pair_20261023 MASTER_PORT=29662
bash examples/train_qwen3_30b_npu.sh > outputs/qwen3-30b-npu/logs/dp8_ep8_pair_20261023.log 2>&1
```

## 2026-10-23：用户要求分别运行dp_shard8与dp_shard8/EP8

用户再次明确两组独立运行：dp_shard8/EP1、dp_shard8/EP8。node92启动前
npu-smi无运行进程；已启动第一组，torchrun PID64955及8个worker实际存活，
tool session19064可继续轮询。尚无退出码，不记通过。输出目录
`outputs/qwen3-30b-npu/dp8_ep1_pair_20261023`，外层日志
`outputs/qwen3-30b-npu/logs/dp8_ep1_pair_20261023.log`。

容器仓库根目录、既有runtime PYTHONPATH，完整参数：

```bash
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
export LLMTUNER_DATA_PARALLEL_SHARD_SIZE=8 LLMTUNER_EXPERT_PARALLEL_SIZE=1 LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE=1 LLMTUNER_TENSOR_PARALLEL_SIZE=1 LLMTUNER_PIPELINE_PARALLEL_SIZE=1 LLMTUNER_CONTEXT_PARALLEL_SIZE=1
export LLMTUNER_STEPS=2 LLMTUNER_GLOBAL_BATCH_SIZE=8 LLMTUNER_MAX_SEQ_LEN=4096 LLMTUNER_DETERMINISTIC=1 LLMTUNER_SAVE_CHECKPOINT=1 LLMTUNER_CHECKPOINT_INTERVAL=1 LLMTUNER_ENABLE_TENSORBOARD=1 LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/dp8_ep1_pair_20261023 MASTER_PORT=29661
bash examples/train_qwen3_30b_npu.sh > outputs/qwen3-30b-npu/logs/dp8_ep1_pair_20261023.log 2>&1
```

第二组待第一组结束并接通HF初始化后独立运行，不抢占第一组设备，不把此前
EP/HF guard退出算作训练通过。两组检查配置保持2步/seq4096/batch8、
BF16/fullAC/seed42与逐步checkpoint；不替代原P0–P4其余验收。

## 2026-10-23：实际DCP manager的EP分片保存恢复

Gloo4 EP2×DP2脚本新增实际DistributedCheckpointManager，注册model、optimizer、
EMA、LambdaLR并开启sharded_expert_parallel，保存step2、破坏模型为NaN、
加载step2精确恢复。三个DP分片维度均通过。首轮后续optimizer字段比较失败，
原因是仅源optimizer构建LambdaLR新增initial_lr；给冷路径同样构建scheduler
后完整脚本通过，未放宽比较或改生产实现。失败日志
`outputs/qwen3-30b-npu/logs/ep_dcp_manager_roundtrip_20261023.log`保留；复测
`outputs/qwen3-30b-npu/logs/ep_dcp_manager_fixed_20261023.log`，exit0。
环境与命令同既有Gloo4脚本并替换日志。Ruff及diff检查通过。
此轮manager只破坏并断言模型恢复，不声称所有注册状态均做破坏恢复检验。
HF初始化保护仍保留，30B dp_shard8/EP8尚未复跑。

## 2026-10-23：DCP分片配置接入manager与训练服务

CheckpointConfig新增sharded_expert_parallel=False显式开关，限制backend=dcp；
base manager据此创建EP分片ModelCheckpointState，assembly同步传给optimizer
及EMA，默认关闭。尚未解除EP/HF初始化保护，也未在30B入口开启该开关。
新增配置负例验证torch_checkpointing明确拒绝，1 passed/293 deselected、
exit0，日志`outputs/qwen3-30b-npu/logs/ep_chunk_backend_guard_20261023.log`。

配置和trainer CPU回归445 passed/11 failed/40 warnings、exit1，日志
`outputs/qwen3-30b-npu/logs/ep_chunk_config_assembly_20261023.log`；失败错误
均为torch-npu查询隐藏NPU的aclInit107001，未更改失败用例或放宽断言。
node92相同隐藏NPU/runtime环境，命令test-venv/bin/python -m pytest -q
tests/unit_tests/cpu/test_config.py tests/unit_tests/cpu/test_trainer.py。
Ruff及git diff --check通过。仍需manager端到端验证、初始化装配和实际NPU
验证；dp_shard8/EP8尚未复跑，不宣称生产训练已通过。

## 2026-10-23：本地EP chunk直接标准HF导出

Qwen3 to_hf_export新增ChunkedCheckpointTensor分支：仅拆本地持有专家，
逐专家二维chunk保留其他维偏移，不调用全局unbind、不聚合完整专家堆叠。
未持有专家的rank不输出该键，由DCP writer合并各rank所有权。
EP2/Gloo正式脚本通过实际save_hf_checkpoint发布标准HF目录，逐项读取
index指向的safetensors，核对全部专家及router键和值与原源精确相同，
再用HF reader加载本地wrapper通过。两轮exit0，日志分别为
`outputs/qwen3-30b-npu/logs/ep_chunk_hf_export_20261023.log`及
`outputs/qwen3-30b-npu/logs/ep_chunk_hf_export_values_20261023.log`，环境命令
沿用既有EP2脚本，仅替换日志。Ruff/format及diff检查通过。
此证据仅plain EP2标准HF writer路径，不证明torch_checkpointing后端支持
chunk tensor，也未完成生产manager选择与训练装配。30B dp_shard8/EP8待复跑。

## 2026-10-23：EMA可选EP分片及冷恢复后更新

EMA构造增加sharded_expert_parallel=False显式选项，复用optimizer容器
的分片序列化与恢复机制。EP2×DP2/Gloo4脚本联合保存model/optimizer/EMA，
冷对象加载后继续第二步及动态EMA更新，三个DP分片维度的全部EMA字段和
chunk偏移逐rank精确一致；原model/optimizer断言继续通过，exit0。日志
`outputs/qwen3-30b-npu/logs/ep_ema_cold_next_step_20261023.log`，运行命令环境
同既有四rank脚本，仅替换日志名。该证据仍为同进程冷对象。

EMA CPU回归88 passed、15 warnings、exit0，日志
`outputs/qwen3-30b-npu/logs/ep_ema_option_unit_20261023.log`。
Ruff/format及git diff --check通过。manager与训练装配尚未启用新路径，
30B dp_shard8/EP8未复跑，整体P0–P4目标未完成。

## 2026-10-23：EP冷optimizer及恢复下一步精确对齐

EP2×DP2/Gloo4脚本在AdamW第一步后联合保存model/optimizer，另建已有冷模型
对应的空状态optimizer，初始lr故意设0.9（源0.01）。先断言无moments，再
DCP加载联合checkpoint，连续与恢复两路执行相同0.25固定梯度下一步。
三个DP分片维度全部rank的本地参数和所有flat optimizer字段（chunk偏移及
moment、step、参数组）精确相同，exit0。日志
`outputs/qwen3-30b-npu/logs/ep_cold_optimizer_next_step_20261023.log`；环境命令
同前节Gloo4脚本，替换日志名。该证据为同进程冷对象，不替代独立进程训练恢复。

optimizer_container及EMA回归134 passed、1 failed、15 warnings，exit1，日志
`outputs/qwen3-30b-npu/logs/ep_optimizer_option_regression_20261023.log`。
失败仍为test_patterns_can_target_different_local_model_parts的参考AdamW默认
路径查询隐藏NPU报aclInit107001，与此前已记录失败一致；不宣称全套通过。
Ruff/format及git diff --check通过。manager/EMA新分片接入、NPU目标训练仍待
完成，30B dp_shard8/EP8未复跑。

## 2026-10-23：optimizer可选EP分片moments

OptimizersContainer新增sharded_expert_parallel=False选项及类级默认，保留
EMA未调用父构造的兼容。开启后专家moments通过本地chunk保存，加载前构建
本地状态并启用world元数据错误协调，再交给既有unflatten/load流程。
EP/TP两分片选项同时开启明确拒绝，默认路径未切换。

EP2×DP2/Gloo4正式脚本对三个DP分片维度执行AdamW一步、DCP保存、exp_avg
填NaN后加载，逐rank精确比较本地moment通过；其后模型恢复及原错误协调
断言继续通过，exit0。日志`outputs/qwen3-30b-npu/logs/ep_optimizer_chunks_20261023.log`，
环境及命令沿用四rank脚本前节并替换日志。for-loop实际日志报告fused=False、
foreach=False。Ruff/format及diff检查通过。
尚未验证冷optimizer、全部optimizer字段及下一步对齐，未接入manager，
EMA和目标NPU训练仍未验证；30B dp_shard8/EP8尚未复跑。

## 2026-10-23：EP恢复元数据错误的world协调

restore_sharded_expert_state新增显式coordinate_errors=False参数；开启时
所有world rank参与本地准备阶段协调，全部元数据检查结束前不复制目标。
默认保留直接调用的rank-local契约。仅协调本地准备，不承诺copy失败回滚，
也尚未改变wrapper默认行为。

EP2×DP2/Gloo4正式脚本在三个DP分片维度各注入仅rank0源shape错误，断言
rank0保留ValueError，其余rank得到peer失败，并逐rank精确核对目标未改写。
连同原保存、冷恢复、完整重建断言通过，exit0，日志
`outputs/qwen3-30b-npu/logs/ep_restore_peer_error_20261023.log`。node92隐藏
NPU，运行命令同EP2×DP2前节并替换日志名。Ruff/format及diff检查通过。
尚缺optimizer/EMA、manager装配和目标NPU训练；dp_shard8/EP8未重跑。

## 2026-10-23：EP2×DP2 DTensor三维分片及冷模型恢复

新增正式Gloo4脚本`tests/integration_tests/ep_dp_checkpoint_layout.py`。
使用ep2/dp2 mesh、完整shape(4,6,8)，逐项对EP局部tensor的dim0/1/2做
DTensor Shard，检查专家轴与DP偏移叠加、chunk和本地参数共享存储。
可选EP分片ModelCheckpointState完成DCP保存、NaN破坏后恢复、完整tensor
重建和完整源恢复，全部精确。第二轮另建独立NaN冷模型及wrapper加载，
三个维度全部精确，exit0。两轮日志分别为
`outputs/qwen3-30b-npu/logs/ep_dp_checkpoint_layout_20261023.log`和
`outputs/qwen3-30b-npu/logs/ep_dp_checkpoint_cold_20261023.log`。

```bash
export ASCEND_RT_VISIBLE_DEVICES=""
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
timeout 120 torchrun --standalone --nproc_per_node=4 tests/integration_tests/ep_dp_checkpoint_layout.py > outputs/qwen3-30b-npu/logs/ep_dp_checkpoint_cold_20261023.log 2>&1
```

node92容器运行；Ruff/format及git diff --check通过。该证据仅覆盖均匀
DTensor布局和同进程冷对象，不证明NPU FSDP训练、非均匀/空分片、optimizer
或独立进程恢复。尚未启用manager默认路径；30B dp_shard8/EP8仍待复跑。

## 2026-10-23：model wrapper可选EP分片状态

ModelCheckpointState新增sharded_expert_parallel=False选项。显式开启时通过
本地专家chunk构造和刷新缓存，通过restore_sharded_expert_state恢复；默认
完整专家路径保留。与sharded_tensor_parallel同时开启明确拒绝，组合未验证。

EP2/Gloo正式脚本新增实际Stateful wrapper DCP保存、全部参数NaN破坏及加载，
以及完整普通tensor源恢复，逐键精确通过。node92隐藏NPU运行exit0，日志
`outputs/qwen3-30b-npu/logs/ep_model_wrapper_shards_20261023.log`，环境命令
沿用既有EP2脚本，仅替换日志。test_model_state.py回归22 passed/16 warnings、
exit0，日志`outputs/qwen3-30b-npu/logs/ep_model_wrapper_unit_20261023.log`；
命令为相同环境下test-venv/bin/python -m pytest -q对应文件。
Ruff/format及git diff --check通过。尚未接入manager/optimizer/EMA默认路径，
未验证DTensor或跨rank错误协调，未重跑30B dp_shard8/EP8。

## 2026-10-23：EP本地chunk保存及完整源恢复

EP2/Gloo正式脚本新增DCP保存本地专家chunk，全部模型参数填NaN后原地
加载，逐键精确比较；另用普通完整tensor目的地读取同一checkpoint，核对
所有4个专家，证明不同rank专家没有被当作复制参数丢失。首轮exit0，日志
`outputs/qwen3-30b-npu/logs/ep_local_chunk_roundtrip_20261023.log`。

新增restore_sharded_expert_state，先验证全局shape/dtype及chunk所有权，
再复制本地chunk或按全局源的本地偏移切片。首轮普通tensor源使用chunk.copy_
报TypeError，exit1，保留`ep_chunk_restore_20261023.log`；修正为普通源切片
复制后，相同完整源恢复本地专家并逐键精确核对通过，exit0，日志
`outputs/qwen3-30b-npu/logs/ep_chunk_restore_fixed_20261023.log`。
运行命令环境沿用EP2脚本记录并替换日志名。Ruff/format及diff检查通过。

该函数尚未接入model/optimizer wrapper，尚缺DTensor和跨rank错误协调验证，
不代表生产恢复已支持；用户30B dp_shard8/EP8仍未重新运行。

## 2026-10-23：HF直接读取本地EP专家chunk

新增shard_expert_state，将本地专家权重表示为全局专家轴上的DCP chunk，
不执行专家all-gather、不分配全局权重。DTensor分支保留内部FSDP偏移，
显式拒绝Partial；该分支尚未实测。此函数尚未接入model/optimizer wrapper。

EP2/Gloo正式脚本断言全局shape、rank专家偏移和与模型参数相同data_ptr，
再通过实际HF adapter与reader直接加载这些本地chunk，精确比较三个投影。
其后原有全量wrapper导出与重新加载回归仍通过，exit0；日志
`outputs/qwen3-30b-npu/logs/ep_hf_local_chunks_20261023.log`。运行命令、环境
沿用EP2脚本前节记录，仅替换日志名。Ruff/format及git diff --check通过。
这只验证plain EP2本地读取，未证明FSDP、模型/optimizer恢复或目标NPU训练；
下一步补分片保存和恢复后接入wrapper，30B dp_shard8/EP8原失败仍保持。

## 2026-10-23：标准HF导出调用、索引及重新加载

两个checkpoint后端的HF导出分支改用adapter.to_hf_export（第三方adapter
缺方法时兼容原to_hf）；加载模板调用保持原语义。Qwen3 EP独立保存原HF
逐专家文件索引，其他模型保留原规范化后的导出索引，避免影响已有fused路径。
EP2/Gloo正式脚本使用save_hf_checkpoint发布标准HF目录及index，再经原生
reader重新加载并精确核对router和各rank专家权重，exit0。日志
`outputs/qwen3-30b-npu/logs/ep_hf_export_reload_20261023.log`，运行环境及命令
沿用下一节，替换日志名。

受影响CPU回归test_checkpointer.py及test_hf_expert_storage.py实际212 passed、
1 skipped、8 failed，exit1；日志
`outputs/qwen3-30b-npu/logs/ep_export_checkpointer_regression_20261023.log`。
其中5个init_optim_state用例因torch-npu默认AdamW查询隐藏NPU报aclInit107001；
3个interrupted_stager_close用例在DefaultStager构造报non-blocking copy
requires accelerator，未进入被测关闭逻辑。失败保留，不宣称门禁通过。
未为本轮导出变更修改这些路径。受影响文件Ruff及git diff --check通过。
尚未启用EP/HF训练装配，30B dp_shard8/EP8仍未重新运行。

## 2026-10-23：Qwen3原生专家的标准HF导出转换

HF adapter新增显式`to_hf_export`，复用原覆盖检查后，将完整原生w1/w3/w2
专家堆叠拆为标准HF逐专家gate/up/down二维权重，router沿用HF gate命名。
先协调元数据校验，拒绝非完整专家轴和键冲突，再执行可能通信的unbind。
原`to_hf`保留虚拟reader加载模板语义，避免影响已验证的切片读取。

正式EP2/Gloo测试在恢复本地专家后，通过wrapper重新聚合并导出，逐键逐值
精确比较原始HF权重，rank0实际写入safetensors。node92隐藏NPU运行exit0，
日志`outputs/qwen3-30b-npu/logs/ep_hf_export_20261023.log`；命令与下一节
相同，仅日志路径替换。Ruff/format、git diff --check通过。

尚未接入两个checkpoint后端的导出调用及标准HF文件索引映射，也未接入
EP训练装配；不宣称端到端HF导出或dp_shard8/EP8 NPU训练通过。下一步同步
导出键与文件索引，再处理checkpoint完整专家聚合内存。

## 2026-10-23：HF加载恢复EP分片的Gloo验证

新增正式集成脚本`tests/integration_tests/ep_hf_checkpoint_load.py`，两个Gloo
rank各持有4个专家中的2个，参数及router初始化为NaN。通过实际
ModelCheckpointState、HF adapter、DCP reader及wrapper.load_state_dict，
精确核对各rank的gate/up/down专家权重及完整router。node92容器隐藏NPU，
运行exit0，输出`PASS: HF router and all expert projections restore EP2 ownership`。

```bash
export ASCEND_RT_VISIBLE_DEVICES=""
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
timeout 120 torchrun --standalone --nproc_per_node=2 tests/integration_tests/ep_hf_checkpoint_load.py > outputs/qwen3-30b-npu/logs/ep_hf_checkpoint_load_20261023.log 2>&1
```

脚本Ruff/format及git diff --check通过。此测试仅验证CPU EP2加载布局，
不证明NPU训练或30B内存可行；运行装配仍保留EP/HF保护。标准HF导出转换、
checkpoint全量聚合内存及用户dp_shard8/EP8目标训练尚未完成，原失败状态保持。

## 2026-10-23：原生EP reader拒绝不兼容fused输入

native_ep_layout模式新增metadata保护：出现HF fused gate_up_proj/down_proj
时明确拒绝，当前仅支持独立专家投影，避免将fused源当成已转换native堆叠。
混合布局负例同时覆盖原fused模式与native EP模式。相关CPU测试12 passed、
19 warnings、exit0，日志outputs/qwen3-30b-npu/logs/hf_ep_layout_guards_20261023.log。
Ruff/format和git diff --check通过。此拒绝只证明保护，不代表fused转换支持；
adapter装配与分布式EP加载尚未接通，用户dp8/ep8目标失败状态保持。

## 2026-10-23：EP router gate双向映射实测

Qwen3 EP adapter增加原生`.mlp.router.gate.`与HF`.mlp.gate.`双向键映射，
不修改权重值。原生EP reader测试加入真实router tensor并更新HF index，经
adapter.to_hf、正式DCP reader、adapter.from_hf后，router及w1/w2/w3全部精确。
CPU相关文件11 passed/19 warnings/exit0，日志
outputs/qwen3-30b-npu/logs/hf_ep_router_adapter_20261023.log。
Ruff/format和git diff --check通过。运行装配仍未启用新adapter，HF export仍
缺标准专家键转换，EP/HF保护保留；用户dp_shard8/EP8配置尚未重新启动。

## 2026-10-23：HF adapter原生EP专家键覆盖

提取native_ep_expert_key供reader及adapter共用，避免两处专家投影命名漂移。
HFTransformerStateDictAdapter新增显式qwen3_expert_parallel开关，限制qwen3_moe，
开启后专家键覆盖和分片文件映射使用原生EP堆叠FQN，reader自动选native模式。
正式3专家测试通过adapter完成to/from wrapper前缀及覆盖检查，再实际DCP加载
全部w1/w2/w3精确比较。相关CPU文件11 passed/19 warnings/exit0，日志
outputs/qwen3-30b-npu/logs/hf_native_ep_adapter_20261023.log。
Ruff/format、git diff --check通过。该开关尚未在assembly启用，router gate映射
及标准HF export语义未完成，原EP/HF保护保留；不代表用户dp8/ep8目标已可训练。

## 2026-10-23：HF独立专家到原生EP堆叠reader

针对用户dp_shard8/EP8运行的实际拒绝，新增GroupedExpertHuggingFaceStorageReader
显式native_ep_layout模式：HF gate/up/down单专家二维tensor分别映射全局
routed_experts.inner_experts.w1_EFD/w3_EFD/w2_EDF的专家轴chunk，不做transpose
或gate/up拼接，沿用原虚拟storage slice读取。默认fused模式不变。
新增3专家全投影实际DCP读取精确断言，相关CPU文件11 passed/19 warnings/exit0，
日志outputs/qwen3-30b-npu/logs/hf_native_ep_reader_20261023.log。
Ruff/format、git diff --check通过。尚未接入HF adapter键覆盖、router名称转换
及EP运行装配，不移除原保护；此次reader证据不代表dp8/ep8目标训练通过。

## 2026-10-23：用户指定dp_shard8/EP8目标运行

用户明确要求dp_shard=8、EP=8。核实topology：8rank dense shard8，sparse
efsdp1/ep8，非64rank乘积；TP/PP/CP/replicate均1。node92启动前npu-smi报告
八设备无运行进程。30B入口新增LLMTUNER_DATA_PARALLEL_SHARD_SIZE（默认-1）
及LLMTUNER_EXPERT_PARALLEL_SIZE（默认1），配置单测10 passed/14 warnings。
Ruff/format、git diff --check通过，未切换此前分片checkpoint默认路径。

启动原30B-A3B/LIMO入口，BF16/fullAC/seed42/deterministic、seq4096/batch8、
2步，checkpoint启用且interval1，TensorBoard启用。运行目录
`outputs/qwen3-30b-npu/dp8_ep8_node92_20261023`。node92容器仓库根目录，
PYTHONPATH沿用tf511/runtime/baseline/tb-runtime，命令：

```bash
LLMTUNER_DATA_PARALLEL_SHARD_SIZE=8 LLMTUNER_EXPERT_PARALLEL_SIZE=8 LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE=1 LLMTUNER_TENSOR_PARALLEL_SIZE=1 LLMTUNER_PIPELINE_PARALLEL_SIZE=1 LLMTUNER_CONTEXT_PARALLEL_SIZE=1 LLMTUNER_STEPS=2 LLMTUNER_GLOBAL_BATCH_SIZE=8 LLMTUNER_MAX_SEQ_LEN=4096 LLMTUNER_DETERMINISTIC=1 LLMTUNER_SAVE_CHECKPOINT=1 LLMTUNER_CHECKPOINT_INTERVAL=1 LLMTUNER_ENABLE_TENSORBOARD=1 LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/dp8_ep8_node92_20261023 MASTER_PORT=29652 bash examples/train_qwen3_30b_npu.sh
```

实际作业exit1，日志确认dp_shard8/ep8 mesh构建成功，但运行校验拒绝
`expert_parallel_size=8 with checkpoint.initial_load_in_hf`：现有EP交换后的
stacked专家布局未由HF adapter完成转换。尚未加载权重或执行训练step，未生成
训练checkpoint。日志保留于上述目录logs/。该项记失败，不以随机初始化或删除
保护绕过；下一步检查Qwen3 MoE EP与HF专家布局转换契约。该配置不替代此前
TP2×FSDP4或HSDP失败项。

## 2026-10-23：冷模型及冷optimizer对象恢复

正式TP2×DP2/Gloo4测试新增独立model、wrapper及未分配状态的AdamW对象；加载
第一步联合checkpoint前断言所有inner.state为空，并故意将新optimizer lr设
0.9而原为0.01。DCP加载后执行相同第二步，四rank普通/packed参数、全部flat
optimizer字段均与连续路径精确一致，exit0，日志
`outputs/qwen3-30b-npu/logs/tp_cold_optimizer_resume_20261023.log`，端口29651。
这覆盖同进程冷对象、lazy moment分配及参数组恢复，不是独立进程或真实训练。

审查EMA继承关系发现其复用optimizer序列化但不调用父构造，为新选项增加类级
False默认值，保留其原默认行为。EMA CPU回归88 passed/15 warnings/exit0，
日志chunk_option_ema_regression_20261023.log。node92隐藏NPU/test-venv或既有
PYTHONPATH；Ruff/format、git diff --check通过。EMA分片选项、manager默认切换
及真实30B目标仍未完成，原OOM未关闭。

## 2026-10-23：模型与真实AdamW分片恢复后一步对齐

正式TP2×DP2/Gloo4脚本新增模型+optimizer联合DCP保存，连续第二步使用固定
0.25梯度；保存第二步参数local及optimizer全部flat字段快照，再恢复第一步
checkpoint并执行相同第二步。普通/packed下四rank参数、exp_avg、exp_avg_sq、
step及参数组字段全部atol=rtol=0或精确标量相等；完整脚本exit0，日志
`outputs/qwen3-30b-npu/logs/tp_optimizer_resume_step_20261023.log`。
node92隐藏NPU、既有PYTHONPATH，端口29650/nproc4。Ruff/format和diff检查通过。
这是同进程同optimizer对象的保存恢复，不含新进程/冷optimizer初始化、实际
前反向、EMA/scheduler/RNG/loader，不能替代P0或P4验收。manager默认未切换，
真实30B OOM保持失败；下一步需补冷optimizer对象恢复并接入管理器。

## 2026-10-23：真实optimizer分片状态接入及往返

OptimizersContainer增加显式sharded_tensor_parallel选项（默认False）。开启后
state_dict使用chunk转换，load按已初始化optimizer的实际本地moment布局恢复，
再走既有EP/unflatten加载。未切换manager或trainer默认配置。
正式Gloo4/TP2×DP2脚本增加实际for-loop AdamW：一步产生状态，DCP Stateful
保存optimizer，污染exp_avg为-47，再加载并逐chunk精确比较原moment。
普通/packed均通过，完整脚本exit0，日志
`outputs/qwen3-30b-npu/logs/tp_real_optimizer_chunked_20261023.log`，端口29649。
node92隐藏NPU/既有PYTHONPATH。该证据不是多步续训或全optimizer字段等价。

原optimizer container CPU回归46 passed/1 failed，日志
optimizer_chunk_option_unit_20261023.log；失败在测试参考AdamW默认内核选择，
torch-npu查询隐藏设备触发107001，不在新选项路径。未豁免或改动测试/依赖。
Ruff/format和git diff --check通过；完整质量门禁未通过。EMA、manager接入及
真实30B NPU运行仍待完成，原OOM未关闭。

## 2026-10-23：不均匀DP分片映射及模型加载

正式TP/DP脚本增加LLMTUNER_TEST_UNEVEN_DP=1，全局shape改为(5,8,7)，
保留默认(4,8,6)；局部元素数与实际参数local numel比较，不再假定均分1/4。
独立DP Shard(0)、Shard(2)两组Gloo4任务均exit0，普通/packed、三类键、全局
chunk值及DCP、wrapper Stateful恢复、HF组合加载、非法源协调全断言通过。
这覆盖DP首维3/2与末维4/3的不均匀局部块，未覆盖空所有权。
node92隐藏NPU、既有PYTHONPATH，分别以端口29647和29648/nproc4执行
tp_dp_checkpoint_layout.py，后一组加LLMTUNER_TEST_DP_SHARD_DIM=2。
日志为outputs/qwen3-30b-npu/logs/tp_uneven_dp_chunks_20261023.log与
tp_uneven_last_dp_chunks_20261023.log。Ruff/format及git diff --check通过。
不外推至EP或真实NPU，manager/optimizer默认路径仍未切换；30B OOM保持失败。

## 2026-10-23：恢复同视图时消除冗余局部副本

DCP先原地写入chunk目标后，ModelCheckpointState恢复会再次遇到相同活视图。
此前通用alias保护会为这些已相同的局部chunk克隆整套源，增加恢复峰值。
copy_新增严格同视图快捷返回：全局/局部布局匹配后，dtype/device一致，全部
对应chunk共享storage且stride/storage_offset相同才不复制；交叉交换仍保留
局部源快照保护。未放宽非同视图的复制语义。

新增TorchDispatchMode测试，遇到任何普通局部clone/copy_即失败，证明相同
视图路径没有隐藏的局部复制。相关两CPU文件39 passed/16 warnings/exit0，
日志 `outputs/qwen3-30b-npu/logs/chunked_identical_view_copy_20261023.log`。
node92隐藏NPU/test-venv；Ruff import修正、format及git diff --check通过。
未测真实NPU峰值，manager/optimizer仍未切换，30B OOM未关闭。

## 2026-10-23：分片dtype预检与恢复规则一致

模型状态预检新增chunk源/目标dtype匹配，与实际restore拒绝规则一致，避免
全局shape/offset正确但dtype错误直到恢复阶段才发现。普通完整tensor源原有
兼容语义不变。预检回归参数化offset/dtype错误，均断言活参数不变。
node92隐藏NPU/test-venv，两相关文件38 passed/16 warnings/exit0，日志
`outputs/qwen3-30b-npu/logs/tp_chunk_dtype_preflight_20261023.log`。
Ruff/format、git diff --check通过。未切换manager/optimizer，未重跑NPU目标；
此预检修正不关闭30B OOM或P0–P4剩余缺口。

## 2026-10-23：HF reader与分片模型包装器组合加载

正式tp_dp_checkpoint_layout.py补组合接口：rank0写完整safetensors权重，活参数
填-43，通过wrapper.state_dict提供chunk目标，正式IndexedHuggingFaceStorageReader
加载，再调用wrapper.load_state_dict恢复。四rank普通/packed均精确，恢复后
checkpoint chunks仍alias活DP局部storage；未创建完整模型checkpoint缓存。
原Stateful DCP、完整源兼容、逐键moment/EMA形状及拒绝协调回归也通过。
任务exit0，日志 `outputs/qwen3-30b-npu/logs/tp_model_hf_chunked_load_20261023.log`。
node92隐藏NPU/既有PYTHONPATH，torchrun端口29646/nproc4执行正式脚本。
Ruff/format、git diff --check通过。
此测试没有HF FQN adapter及完整manager，也不是30B/GroupedExpert生产加载；
manager默认调用与真实optimizer尚未切换，30B装配OOM保持失败。

## 2026-10-23：完整源兼容恢复与chunk预检

ModelCheckpointState.validate_state_dict在源/目标均为chunk表示时，增加局部
offset/shape预检；不只凭全局shape相同接受错误chunk。新增单测验证拒绝时活
参数不变。两相关CPU单测文件37 passed/16 warnings/exit0，日志
`outputs/qwen3-30b-npu/logs/tp_chunk_preflight_unit_20261023.log`。

正式Gloo4/TP2×DP2脚本补分片wrapper读取完整普通tensor源：活参数填-41，
load_state_dict恢复后local值精确、后续state_dict仍为全局shape/局部1/4
chunk表示。普通/packed、原Stateful DCP和键级回归全部完成，日志
`outputs/qwen3-30b-npu/logs/tp_model_legacy_restore_20261023.log`，端口29645。
node92隐藏NPU、既有runtime PYTHONPATH及test-venv。Ruff和git diff --check
通过。完整源兼容路径仍使用旧slice/distribute语义，并非低内存HF加载证据；
manager默认调用未切换，30B OOM及整体未完成状态保持。

## 2026-10-23：ModelCheckpointState分片保存恢复接入

模型状态包装器增加显式sharded_tensor_parallel选项（默认False），开启后保存
使用局部chunk表示，恢复以模型当前DP布局生成目标并复制chunk，再按既有EP
加载约定处理；未更改manager默认调用或optimizer表示。
正式TP2×DP2/Gloo4脚本增加真实Stateful DCP往返：检查缓存wrapper稳定、保存、
将活参数填-37、DCP加载并调用包装器恢复，普通/packed均精确回到原参数。
四rank完整脚本exit0，日志
`outputs/qwen3-30b-npu/logs/tp_model_chunked_roundtrip_20261023.log`，端口29644。
原逐键moment/EMA形状及拒绝协调回归保持通过；不是实际optimizer恢复证据。
同环境两单测文件36 passed/16 warnings/exit0，日志tp_model_chunked_unit_20261023.log。
node92隐藏NPU，PYTHONPATH及test-venv沿用前节。Ruff/format通过。
当前仅DP dim0/TP dim1、无EP的小模型Stateful接口证据，不代表完整manager、
30B HF加载或保存恢复通过，目标装配OOM保持失败。

## 2026-10-23：跨chunk交换复制的别名保护

审查发现直接copy_若源与目标以交换顺序共享storage，会先覆盖后续待读chunk。
正式ChunkedCheckpointTensor.copy_在任何source/destination storage alias时，
先保存全部局部源chunk副本，再执行写入；无alias的正常staging复制不增副本。
这只分配局部chunk，未聚合全局权重，不声称alias分支零额外内存。

补跨chunk storage swap复制及缓存refresh不修改活模型两项回归。node92隐藏
NPU/test-venv运行test_model_state.py和test_chunked_tensor.py，实际36 passed、
16 warnings、exit0，日志 `outputs/qwen3-30b-npu/logs/chunked_cross_alias_20261023.log`。
Ruff/format和git diff --check通过。默认TP路径尚未切换，30B装配OOM及全目标
未完成状态不变；该局部正确性修复不替代NPU完整保存恢复验收。

## 2026-10-23：分片缓存刷新接入模型状态包装器

ChunkedCheckpointTensor新增refresh_from，先核对全局及局部布局，再审查跨chunk
storage alias；相同活视图保留wrapper，新分配hook值复制到稳定缓存，变化的活
别名视图先detach/clone，避免缓存刷新写坏模型。ModelCheckpointState.state_dict
增加该类型专用刷新分支。默认TP gather尚未切换，故真实TP模型仍走旧表示。

新增三个chunk刷新单测，覆盖稳定复用、新值复制、转置别名不修改活模型；新增
经过真实ModelCheckpointState和state_dict post-hook的缓存生命周期单测。
node92隐藏NPU/test-venv执行两文件，33 passed/16 warnings/exit0，日志
`outputs/qwen3-30b-npu/logs/chunked_model_cache_full_20261023.log`。
原模型状态19项先独立回归也通过，日志chunked_model_cache_regression_20261023.log。
受影响模块及单测Ruff/format、git diff --check通过。未跑全CPU或全量precommit，
未验证完整TP load/HF真实训练、交叉chunk复杂别名或NPU staging；30B OOM未关闭。

## 2026-10-23：DP与TP同轴及末轴chunk覆盖

正式tp_dp_checkpoint_layout.py增加LLMTUNER_TEST_DP_SHARD_DIM=0/1/2选择，
默认0不变；仍固定TP dim1、global(4,8,6)、TP2×DP2。新增两个独立CPU/Gloo4
任务均exit0：DP dim1（与TP同轴）和dim2，普通/packed各三类键全部精确。
同轴packed实际使DP rank仅持有gate或up半区，验证区间交集非空部分的全局
offset、与活local storage alias、局部1/4元素数、DCP完整值及局部恢复布局。
原非法源协调拒绝和目标保持、旧gather/load回归也保留通过。

node92容器隐藏NPU、既有PYTHONPATH，分别执行：

```bash
ASCEND_RT_VISIBLE_DEVICES="" LLMTUNER_TEST_DP_SHARD_DIM=1 torchrun --master-port=29642 --nproc-per-node=4 tests/integration_tests/tp_dp_checkpoint_layout.py
ASCEND_RT_VISIBLE_DEVICES="" LLMTUNER_TEST_DP_SHARD_DIM=2 torchrun --master-port=29643 --nproc-per-node=4 tests/integration_tests/tp_dp_checkpoint_layout.py
```

日志为outputs/qwen3-30b-npu/logs/下tp_dp_same_axis_chunks_20261023.log及
tp_dp_last_axis_chunks_20261023.log。Ruff/format及git diff --check通过。
证据不含不均匀/空分片、Partial、EP及目标模型；默认manager仍未切换，30B
OOM未关闭。下一步应接入稳定缓存并复验实际模型加载，不继续重复均匀轴测试。

## 2026-10-23：chunk恢复错误的TP组协调

新增本地chunk阶段错误协调：捕获BaseException，依次完成所有已注册TP组的
状态交换，再传播原始错误或peer失败。chunk构造及恢复校验均使用此边界；
构造边界内部仅局部shape/placement计算和视图创建，不包含tensor collective。
恢复所有rank完成校验后才执行copy，避免单rank拒绝而peer已经修改目标。

正式TP2×DP2/Gloo4脚本新增每TP组仅rank0收到非chunk源的用例，断言两rank
均拒绝且目标保持-17；普通/packed和weight/moment/EMA三类键全部通过，
原健康往返、全rank非法源及DCP回归也通过，exit0，日志
`outputs/qwen3-30b-npu/logs/tp_chunk_restore_coordination_20261023.log`。
node92隐藏NPU、既有PYTHONPATH，torchrun端口29641/nproc4。
Ruff及git diff --check通过；仅证明所测TP组恢复协调，不证明跨DP/PP的manager
全局错误原子性、构造Partial/空分片单rank错误或interrupt复验。生产管理器
默认路径尚未切换，30B OOM保持失败。

## 2026-10-23：chunk恢复到调用方DP布局

新增 `restore_sharded_tensor_parallel_state`：调用方提供参数形状的本地目标，
复用shard转换生成目标chunk坐标，复制任何值前验证全部TP条目的源类型、全局
shape、chunk offset/shape和dtype；返回原目标对象，保留DTensor placement及
mesh。此函数尚未接入ModelCheckpointState或optimizer默认load。

正式TP2×DP2/Gloo4脚本增加普通/packed、weight/moment/EMA三类键恢复，逐项
验证目标identity、placement、mesh、local值精确一致。exit0，日志
`outputs/qwen3-30b-npu/logs/tp_dp_chunk_restore_20261023.log`，端口29639。
再补非chunk源拒绝及目标值保持-17断言，完整重验exit0，日志
`outputs/qwen3-30b-npu/logs/tp_dp_chunk_restore_rejection_20261023.log`，端口29640。
node92隐藏NPU、既有PYTHONPATH、nproc4，执行同一正式tp_dp_checkpoint_layout.py。
Ruff/format及git diff --check通过。新恢复检查仍为本地检查，跨rank新增错误
协调及其他placement/EP未验证；需补齐后才切换默认管理器。30B OOM未关闭。

## 2026-10-23：TP×DP局部视图到全局chunk索引转换

新增 `shard_tensor_parallel_state`，复用原TP准备校验和manifest协调，按DTensor
实际DP placement计算local offset，再将TP物理维度映射到全局坐标；packed
gate/up分别求DP局部区间交集。转换仅创建wrapper/视图，不调用full_tensor或
tensor all_gather。原gather/load默认行为未切换，尚未接入模型缓存。

扩展正式 `tests/integration_tests/tp_dp_checkpoint_layout.py`：TP2×DP2，
普通/packed两种布局，weight/Adam moment/EMA形状三类键；逐chunk精确比较
全局切片、检查与活参数local storage alias、局部元素数1/4，再DCP写出并加载
到完整tensor精确比较。保留原gather/load及DTensor mesh/placement回归。
四CPU/Gloo rank全部通过exit0，日志
`outputs/qwen3-30b-npu/logs/tp_dp_chunk_mapping_20261023.log`。
node92容器隐藏NPU、既有PYTHONPATH，torchrun端口29638/nproc4。
Ruff/format及git diff --check通过。

证据仅DP Shard(0)/TP dim1、均匀非空分片；Partial及空所有权目前明确拒绝，
其新增错误尚需加入跨rank协调测试。恢复chunk到活DP布局、缓存稳定性、EP及
其他DP placement仍待补充；未改变30B TP2×FSDP4失败状态或宣称整体完成。

## 2026-10-23：DefaultStager兼容性修复及CPU回归

正式ChunkedCheckpointTensor首次进入DefaultStager在new_empty明确拒绝，四rank
exit1，日志 `outputs/qwen3-30b-npu/logs/chunked_default_stager_20261023.log`。
源码核实StateDictStager先new_empty创建wrapper，再递归offload其属性中的局部
tensor；与BlockingAsyncStager的zeros_like/copy_机制不同。

正式类型增加受约束new_empty：仅接受相同逻辑全局shape，按已有chunk局部shape
分配，不允许任意shape操作。新增对应正反例，CPU单测11 passed/14 warnings，
exit0，日志 `outputs/qwen3-30b-npu/logs/chunked_tensor_unit_stager_20261023.log`。
复现用既有隐藏NPU的test-venv运行test_chunked_tensor.py。

四CPU/Gloo rank重验DefaultStager（pinned/shared/async/nonblocking均False），
HF与expert reader、快照独立性、活权重修改后保存/恢复均精确通过；日志
`outputs/qwen3-30b-npu/logs/chunked_default_stager_retry_20261023.log`。
沿用前节PYTHONPATH和PROBE开关，增加LLMTUNER_PROBE_DEFAULT_STAGER=1，
torchrun端口29637/nproc4。Ruff/format及git diff --check通过。
该实现new_empty分配的局部临时chunks会在属性递归offload后被替换，存在局部
CPU额外分配，未测峰值；不声称零复制。pinned/shared、真实异步NPU搬运及
完整TP接入仍待验证，30B装配OOM未关闭。

## 2026-10-23：正式分片表示模块及契约回归

新增 `components/checkpointer/chunked_tensor.py::ChunkedCheckpointTensor`，
全局shape仅为wrapper metadata，存储为调用方提供的多个非连续局部视图。
构造校验维数、bounds、dtype/device、materialized strided storage及局部重叠；
支持DCP三个协议、clone/detach/device-copy、zeros_like及相同布局copy_，未知
计算明确拒绝。尚未接入TP gather/load及ModelCheckpointState缓存。

新增CPU单测10 passed/14 warnings/exit0，日志
`outputs/qwen3-30b-npu/logs/chunked_tensor_unit_20261023.log`；node92隐藏NPU，
`outputs/qwen3-30b-test-venv/bin/python -m pytest -q tests/unit_tests/cpu/components/checkpointer/test_chunked_tensor.py`。

既有四CPU/Gloo探针改用该正式类型（`LLMTUNER_PROBE_PRODUCTION_TYPE=1`），
原packed协议/alias/HF/expert-HF/stage开关开启，端口29635/nproc4。
四rank同步DCP双向重分片、普通及专家HF加载、BlockingAsyncStager独立快照、
活权重修改后快照保存/恢复全部精确通过，exit0，日志
`outputs/qwen3-30b-npu/logs/chunked_tensor_dcp_contract_20261023.log`。
PYTHONPATH沿用tf511/runtime/baseline。模块及单测Ruff/format、git diff --check
通过；来源仅记录于upstream map。未跑全套门禁，未声称pinned/shared/NPU或
30B路径通过；目标TP2×FSDP4 OOM仍失败，下一步为布局转换及稳定缓存接入。

## 2026-10-23：staging后活权重修改隔离验证

对前节BlockingAsyncStager快照补实际文件写入验收：staging结束后将原始local
全部改为-12345，DCP保存快照，再分别加载到全局tensor及packed活视图。
四rank均精确恢复修改前reference；保存及全局加载期间原local保持-12345，
局部快照元素数仍为全局1/4。进程exit0，日志
`outputs/qwen3-30b-npu/logs/packed_dcp_staged_mutation_20261023.log`。
复现沿用前节环境及五个PROBE开关，torchrun端口29633/nproc4。Ruff/format通过。
这证明当前CPU同步staging快照可供DCP写入且隔离后续修改，不代表真实异步
NPU搬运或pinned/shared模式通过；生产路径尚未接入，30B OOM状态不变。

同探针新增`LLMTUNER_PROBE_EXPERT_HF=1`，构造4个独立HF expert gate/up/down
文件张量，用正式GroupedExpertHuggingFaceStorageReader加载packed gate/up
视图。node92/CPU/Gloo4、端口29634任务exit0，四rank gate/up原始local值均
atol=rtol=0，后续独立快照及修改隔离断言也通过。日志
`outputs/qwen3-30b-npu/logs/packed_dcp_expert_hf_20261023.log`。
本配置仅检验4专家gate/up目标；未检验down目标分片、多文件index、实际30B
或生产checkpoint包装，不外推为这些路径通过。

## 2026-10-23：HF reader及局部staging快照契约

扩展既有Tensor协议探针，rank0生成单文件safetensors，使用正式
IndexedHuggingFaceStorageReader加载两块packed非连续活视图。四rank原始local
值均atol=rtol=0；随后BlockingAsyncStager触发`aten.zeros_like.default`明确
拒绝，整组exit1。日志 `outputs/qwen3-30b-npu/logs/packed_dcp_hf_staging_20261023.log`。
源码核实普通Tensor子类进入普通tensor staging分支，不能仅靠DCP chunk协议
自动保留局部分片，不能直接宣称兼容异步checkpoint。

探针增加受开关保护的zeros_like/copy_分片语义：仅为本地chunk分配快照，复制前
核对全局shape及chunk metadata，未知算子继续拒绝。复验四rank均通过原同步
双向重分片、HF reader及BlockingAsyncStager独立快照值/非alias断言，exit0。
日志 `outputs/qwen3-30b-npu/logs/packed_dcp_hf_chunk_staging_20261023.log`。
node92容器仓库根目录，PYTHONPATH沿用tf511/runtime/baseline，复现：

```bash
ASCEND_RT_VISIBLE_DEVICES="" LLMTUNER_PROBE_ALIAS_VIEWS=1 LLMTUNER_PROBE_PROTOCOL_VIEWS=1 LLMTUNER_PROBE_HF=1 LLMTUNER_PROBE_STAGE=1 LLMTUNER_PROBE_STAGE_OPS=1 torchrun --master-port=29632 --nproc-per-node=4 outputs/probe_packed_dcp_shards.py
```

Ruff/format通过。证据不含GroupedExpert reader、HF export、DefaultStager
pinned/shared/nonblocking模式、PROCESS executor或NPU D2H快照；也未验证缓存
重复刷新及恢复后的完整训练。生产代码未修改，30B OOM和整体未完成状态不变。

## 2026-10-23：Tensor分片协议直接读写非连续视图通过

先实现普通Python对象的三个DCP分片协议。planner建图接受，但文件写入阶段
`find_state_dict_object`不为非Tensor对象按offset查找chunk，四rank协调失败，
exit1，日志 `outputs/qwen3-30b-npu/logs/packed_dcp_protocol_views_20261023.log`。
这是接口类型约束，不通过修改DCP检查规避。

随后在同一忽略目录探针实现Tensor wrapper subclass，逻辑shape为全局shape，
仅引用两块原始packed非连续视图；三个协议分别提供write metadata、chunk列表
和offset对应的实际视图。未实现通用计算，未知算子明确抛NotImplementedError。
不构造ShardedTensor，不执行TP/DP tensor gather，不对视图做contiguous复制。
四CPU/Gloo rank均通过：分片保存/恢复、分片源到全局目标、全局源到分片目标，
所有值atol=rtol=0；首次恢复还检查原始local tensor值及存储alias。
进程exit0，日志 `outputs/qwen3-30b-npu/logs/packed_dcp_tensor_protocol_20261023.log`。

node92容器仓库根目录复现：

```bash
ASCEND_RT_VISIBLE_DEVICES="" LLMTUNER_PROBE_ALIAS_VIEWS=1 LLMTUNER_PROBE_PROTOCOL_VIEWS=1 torchrun --master-port=29630 --nproc-per-node=4 outputs/probe_packed_dcp_shards.py
```

探针Ruff/format通过。证据仅支持当前torch2.12同步DCP文件路径及此固定布局；
不证明文件writer无临时copy、异步staging、HF reader/export、通用DP placement、
稳定缓存或生产30B路径通过。后续正式实现须补这些契约及错误协调回归。
本轮得到可直接引用活视图的候选表示，仍未修改生产路径或关闭30B OOM。

## 2026-10-23：packed零复制视图限制实测

沿用前节CPU/Gloo4双向DCP探针，取消gate/up局部chunk的contiguous副本，
直接以活packed tensor的非连续视图构造ShardedTensor。四rank均在构造期
明确拒绝：`Only torch.contiguous_format memory_format is currently supported`，
exit1，日志 `outputs/qwen3-30b-npu/logs/packed_dcp_alias_views_20261023.log`。
复现为node92容器仓库根目录、隐藏NPU，torchrun端口29628/nproc4；探针现以
`LLMTUNER_PROBE_ALIAS_VIEWS=1`选择该失败模式，默认保留原连续分片模式。

前节1/4是局部checkpoint chunk元素数，并非零额外分配或整个进程峰值结论。
该实验排除了直接使用ShardedTensor包装非连续活视图的方案，不推翻连续副本
模式的双向重分片结果。当前DCP源码同时提供`__create_write_items__`、
`__create_chunk_list__`和`__get_tensor_shard__`协议入口，后续应检验这些入口
能否表达非连续视图及稳定刷新，而不放宽保护或聚合全量权重。
本轮未改生产路径；30B TP2×FSDP4仍失败，整体目标未完成。

## 2026-10-23：packed分片DCP双向重分片诊断

原四CPU/Gloo探针进程已确认exit0：TP2×DP2全局形状(4,8,6)，每rank两块
不连续gate/up chunk，DCP保存后填NaN再加载，全部局部值精确恢复，本地tensor
元素数为全量1/4。日志 `outputs/qwen3-30b-npu/logs/packed_dcp_shards_probe_20261023.log`。

扩展忽略目录 `outputs/probe_packed_dcp_shards.py`，补两个方向：分片checkpoint
加载为全局普通tensor，以及全局普通tensor checkpoint加载进packed分片。
两方向均用atol=rtol=0断言；四rank全部通过，exit0，日志
`outputs/qwen3-30b-npu/logs/packed_dcp_global_interop_20261023.log`。
node92容器仓库根目录命令：

```bash
ASCEND_RT_VISIBLE_DEVICES="" torchrun --master-port=29627 --nproc-per-node=4 outputs/probe_packed_dcp_shards.py
```

这仅证明当前torch2.12 DCP能够处理该非连续布局和双向重分片，不证明HF reader、
正式ModelCheckpointState稳定缓存、optimizer/EMA、EP或目标30B路径已通过。
ShardedTensor运行明确报告弃用警告；正式设计还需评估其版本兼容性及DTensor
替代表示。未修改生产checkpoint实现，30B TP2×FSDP4装配OOM保持失败。
下一步为实现全局索引分片表示并接入保存/加载、缓存及HF路径后，逐项回归。

## 2026-10-23：TP低内存重构前的布局与错误协调基线

复核恢复契约：packed gate/up必须分别切分后重拼，DP DTensor placement及
专家轴语义需保留；普通连续Shard不能直接表达现有packed物理顺序。
本轮未以禁用checkpoint、构造期跳过或CPU回退规避30B OOM。

双CPU/Gloo rank执行 `tp_checkpoint_coordination.py` exit0：shape、缺参数、
packed不整除、manifest、sparse、meta错误在加载/保存两方向均协调拒绝；
KeyboardInterrupt/SystemExit协调及healthy retry通过。隐藏NPU，不占设备。
日志 `outputs/qwen3-30b-npu/logs/tp_checkpoint_coordination_20261023.log`。

新增四CPU/Gloo rank基线 `tests/integration_tests/tp_dp_checkpoint_layout.py`，
覆盖TP2×DP2下普通/packed权重、Adam moment、EMA形状状态的精确全局映射与
恢复，断言DP placement、mesh及局部值。任务已exit0，四rank普通/packed均
精确一致，日志 `outputs/qwen3-30b-npu/logs/tp_dp_checkpoint_layout_20261023.log`。
它验证旧路径语义，
不是低内存实现或30B已修复的证据。后续实现须同时满足这一契约和不聚合
整模型的内存条件，仍需真实目标模型重新验收。

两命令均在node92容器仓库根目录设置
`ASCEND_RT_VISIBLE_DEVICES=''` 及既有tf511/runtime/baseline PYTHONPATH，
分别 `torchrun --master-port=29624 --nproc-per-node=2 tests/integration_tests/tp_checkpoint_coordination.py`
与 `torchrun --master-port=29625 --nproc-per-node=4 tests/integration_tests/tp_dp_checkpoint_layout.py`。
整体任务仍未完成。

## 2026-10-23：30B TP2×FSDP4 checkpoint装配OOM定位

上一轮目标模型作业已exit1，日志
`outputs/qwen3-30b-npu/p2_tp2_fsdp4_node92_20261023/logs/`。真实mesh为
TP2、DP shard4，但没有完成HF加载或任何训练step。失败栈位于Trainer构造的
checkpointer装配：ModelCheckpointState.__init__ → _get_state_dict →
gather_tensor_parallel_state，value.full_tensor()/empty_like分配时OOM，
不是训练反向，也不能直接套用HSDP归约缓冲诊断。rank0请求386MiB时已分配
54GiB，多个rank失败。原始日志完整保留。

代码审计确认TP checkpoint路径对每个TP参数先聚合DP维，再all_gather TP维，
将完整tensor保存在result及cached_state_dict中；30B完整BF16权重约56.87GiB，
与已分片的活模型同时常驻，足以构成容量瓶颈。该缓存还在state_dict刷新及
load_state_dict契约中使用，不能仅跳过构造缓存、禁用checkpoint或静默CPU
回退充当根因修复。需要可分片的全局checkpoint表示或显式低内存存储协议，
并验证packed gate/up、TP×DP及恢复，不得只让初始化通过而转移到保存时OOM。
当前P2该配置记失败，未用小模型通过关闭；整体任务未完成。

## 2026-10-23：有限值断言复验通过与30B TP2×FSDP4验证

上一轮串行三组任务最终exit0。补强finite断言后，ddp2/shard2/hsdp2×2
四步FP32参考测试均failed ranks0，最终参数maxAE分别1.490e-8、0、5.960e-8。
日志 `outputs/qwen3-30b-npu/logs/p1_fsdp_finite_{ddp,shard,hsdp}_20261023.log`。
该证据仅限小模型，30B HSDP2×4 OOM及其他未测矩阵状态不变。

继续独立组合项：在同一node92启动30B-A3B/LIMO、8rank、TP2×FSDP4、
replicate1、BF16、fullAC、seq4096、batch8、seed42、deterministic、2步，
不保存checkpoint，启用TensorBoard。预期仅验证实际加载及有限loss/grad_norm
训练路径，尚未有跨拓扑输出/梯度等价性断言，不能单凭exit0关闭P2。
目录 `outputs/qwen3-30b-npu/p2_tp2_fsdp4_node92_20261023`，当前进行中。
复现使用既有30B入口命令，显式设置TENSOR_PARALLEL_SIZE=2、
DATA_PARALLEL_REPLICATE_SIZE=1、STEPS=2、SAVE_CHECKPOINT=0及上述dump目录；
其余seq4096/batch8/deterministic/TensorBoard/PYTHONPATH保持原配置。
未将此配置替代失败HSDP，亦未降低精度、序列长度或模型规模。

## 2026-10-23：FSDP参考测试有限值验收补强

本轮node92启动前八卡无进程。先运行native复制2与分片2参考用例，均exit0，
分片2最终参数maxAE0；日志为 `p1_fsdp_{ddp,shard}_node92_20261023.log`，
均位于 `outputs/qwen3-30b-npu/logs/`。仅代表TinyTokenDecoder，不代表30B。

审查脚本发现误差仅用 `> TOL` 比较，NaN会绕过该条件；最大参数误差跨参数
聚合也可能掩盖非有限项。已在 `fsdp_mesh_equivalence.py` 对各参数梯度误差、
双方逐步loss、所有最终参数误差和无通信反例gap增加math.isfinite检查。
未放宽tol或更改参考模型。修改后按ddp2、shard2、hsdp2×2顺序重跑，
日志为 `p1_fsdp_finite_{ddp,shard,hsdp}_20261023.log`，目前任务进行中，
旧版本通过不当作新增断言已获验证。完整命令如下：

```bash
ssh -o BatchMode=yes 10.1.0.92 "docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime
set -o pipefail
for mode in ddp shard hsdp; do
world=2; if [ \"\$mode\" = hsdp ]; then world=4; fi
torchrun --master-port=29623 --nproc-per-node=\"\$world\" tests/integration_tests/fsdp_mesh_equivalence.py \"\$mode\" 2>&1 | tee \"outputs/qwen3-30b-npu/logs/p1_fsdp_finite_\${mode}_20261023.log\" || exit \$?
done
'"
```

P4关键权重名称、时点与阈值仍未确定，已向用户询问；仅阻塞相关P4验收，
不阻止独立并行测试。HSDP30B OOM未关闭；本轮未修改依赖缓冲释放协议。

## 2026-10-23：原HSDP配置反向内存探针

node92启动前npu-smi八卡无运行进程。准备
`outputs/probe_hsdp_backward_memory.py`，包装安装版FSDPParamGroup.post_backward，
在原方法前后记录allocated/reserved、已观察组的all-reduce缓冲字节数及组数。
通过弱引用遍历，不主动释放缓冲、不同步设备、不替换通信；额外Python遍历和
文件IO仍会扰动调度，因此仅作诊断，不能作为正式性能或数值通过证据。
使用原30B/HSDP2×4/seq4096/batch8/BF16/fullAC/seed42/deterministic/2步配置，
不保存checkpoint，启动8rank，master-port29621。入口为探针脚本，其他环境
与前次失败相同。日志及逐rank JSONL在
`outputs/qwen3-30b-npu/hsdp_memory_probe_20261023/`。脚本Ruff/format通过。
探针作业最终exit1，同原配置在首次反向重现OOM，rank0仍尝试770MiB分配，
已分配53.57GiB。逐rank数据实际记录到decoder低层，保留归约缓冲随反向
逐层累积；例如rank0在layer10完成后，39组缓冲共23,989,751,296字节
（约22.34GiB），allocated55,193,243,648字节（约51.40GiB）。这是同次失败
中的真实引用存活量，不再只是静态上界；但不证明这些缓冲可安全提前释放，
也不能证明释放后Adam状态常驻容量足够。原失败状态保持不变。
本轮所有诊断rank已退出。后续应针对事件完成后的受控释放协议做独立
正确性评估，不能删除引用保护或强制同步后直接宣称性能/训练验收通过。

## 2026-10-23：HSDP2×4 内存容量与依赖生命周期审计

读取真实30B checkpoint safetensors头部，不加载参数内容，得到总参数
30,532,122,624、48层。诊断脚本 `outputs/audit_30b_hsdp_memory.py` exit0，
输出 `outputs/qwen3-30b-npu/logs/hsdp_header_memory_audit_20261023.json`。
shard4下BF16参数14.217628GiB、BF16梯度14.217628GiB，两份BF16 Adam
moment共28.435255GiB，总计56.870510GiB；尚不含激活、通信、临时张量、
allocator及设备预占。该估计假设moment与参数同dtype，不代表实测峰值。
单个最大decoder层FP32 reduce-scatter输入2.321305GiB，BF16完整参数
1.160653GiB。因此不能只靠释放一个临时张量就断言本配置能稳定完成更新。

只读检查node92安装版torch2.12的 `_fsdp_collectives.foreach_reduce`：
HSDP reduce_output按reduce_dtype分配，进行副本轴all-reduce后，再转换为
orig_dtype形成梯度；`_fsdp_param_group` 每组保留 `_all_reduce_state` 至
finalize_backward等待事件后释放，源码明确为防止跨stream复用造成梯度别名。
若全部组FP32归约缓冲同时存活，其理论量为28.435255GiB；这只是静态上界
线索，不是本次OOM时实际存活量或泄漏证据。当前没有捕获逐层存活张量，
不得移除该保护、绕过事件等待或宣称依赖缺陷已证实。

真实HSDP2×4/seq4096失败状态不变。下一步需要在原配置采集逐层已分配内存
及归约缓冲存活量，区分必要常驻状态与额外峰值；若容量不足需明确提出
offload/不同拓扑的独立配置及验收，不以精度下降、缩序列或小模型替代。
本轮未改生产代码或启动NPU训练；仅离线头部和已安装源码审计，diff检查通过。
整体P0–P4目标仍未完成。

## 2026-10-23：Q norm编译kernel活动地址覆盖审计

第三项不同证据来源为无转置对照。诊断脚本新增
`LLMTUNER_REPLAY_NO_TRANSPOSE=1`，将原激活的load/store地址直接表示为
`x3 + 128*x2 + 512*y0`，对齐grad的[x2,y0,x3]逻辑坐标，移除两次
permute，保留运算顺序、循环、mask、SUB和独立输出设置。生成文件
`outputs/qnorm_replay_extracted_kernel_separate_aligned.py`确认没有permute。
node92运行前npu-smi再次确认无设备进程，命令在容器仓库根目录执行：

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 LLMTUNER_REPLAY_SYNC_INPUT=1 LLMTUNER_REPLAY_MATCH_OPTIONS=1 LLMTUNER_REPLAY_SEPARATE_OUTPUT=1 LLMTUNER_REPLAY_NO_TRANSPOSE=1 timeout 240 python outputs/replay_qnorm_captured.py > outputs/qwen3-30b-npu/logs/qnorm_replay_no_transpose_20261023.log 2>&1
```

运行已exit0，仅表示完成收集。SUB3/4各24次输出全有限，最大AE分别
0.3772181273/0.0004107178，分别8/4份捕获的三次maxAE变化；无精确案例。
JSON为日志目录
`qnorm_isolated_replay_sync_input_matched_options_separate_output_no_transpose_20261023.json`。
因此去掉转置仍不能消除该重放异常，不支持将转置作为唯一根因。
本分支三轮已分别核验源级地址覆盖、分离alias和去除transpose，尚未解决；
遵照原目标的三轮规则，先转向独立验收项，有新的失败边界或后端证据再续。
不将诊断kernel纳入生产或改变正式compile失败状态。脚本Ruff/format、
git diff --check通过；P0–P4整体未完成。

后续独立输出对照：`outputs/replay_qnorm_captured.py`新增仅诊断的
`LLMTUNER_REPLAY_SEPARATE_OUTPUT=1`，通过AST只将唯一store的目标指针
改为独立out_ptr0，原inout读取及两次permute保留；独立输出预填NaN。
生成单独`outputs/qnorm_replay_extracted_kernel_separate.py`，不改归档或
生产实现。运行前node92的npu-smi确认八卡无运行进程。

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 LLMTUNER_REPLAY_SYNC_INPUT=1 LLMTUNER_REPLAY_MATCH_OPTIONS=1 LLMTUNER_REPLAY_SEPARATE_OUTPUT=1 timeout 240 python outputs/replay_qnorm_captured.py > outputs/qwen3-30b-npu/logs/qnorm_replay_separate_output_20261023.log 2>&1
```

命令在node92容器仓库根目录运行，进程已exit0。八份捕获×两种SUB×三次，
保持num_warps1/num_stages1和输入同步；SUB3/4各24次全有限，均无精确
案例。最大AE分别0.3772181273和0.0008571007，分别8/4份捕获的三次
maxAE不一致。JSON为日志目录
`qnorm_isolated_replay_sync_input_matched_options_separate_output_20261023.json`。
这些是独立运行结果，不与原in-place运行拼接为同次因果链；至少说明分离
输出未消除该诊断中的不稳定，不能以此作为正式修复。后续需检查转置lowering
及masked lane，而非继续盲目切换alias选项。Ruff/format和git diff --check
通过；整体compile仍失败，P0–P4目标未完成。

重读原始目标后，继续既有Inductor失败诊断，不重复已完成的四组30B两步训练。
上一轮补全goal.md四组配置属于任务规格进展，不新增硬件验收证据。
本轮新增忽略目录脚本`outputs/audit_qnorm_addresses.py`，按归档Q norm
kernel的grid16、Y0BLOCK4、X2SUB4、X3SUB128枚举活动lane，分别检查
YSUB3/4。显式纳入两次permute的坐标对应和每个program的尾部mask。

node92容器运行命令：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/audit_qnorm_addresses.py'
```

退出0，两组各32768个输出地址，全部且仅写一次；inout读写使用同一地址，
跨program无重叠；grad的32768个地址也各读取一次，无活动越界。
报告`outputs/qwen3-30b-npu/logs/qnorm_address_audit_20261023.json`。
宿主机初次执行因容器日志目录权限无法保存，未改变目录权限；随后在容器执行
并保存成功。该枚举只验证源级活动地址，不证明backend lowering正确，
也不排除masked lane、转置或in-place相关编译问题。未启动NPU训练。
下一步可用相同捕获输入比较独立输出与in-place版本，保留原始输入同步及
launch选项，分离alias与转置影响；不得把诊断成功当作正式compile通过。

## 2026-10-23：Inductor reduction轴序诊断超时记录

节点10.1.0.92、hyper-parallel-env，独立诊断运行
`pp2-inductor-axis-probe-20261023`，PP2/shard1/CP1/TP1/EP1、
1F1B四个microbatches、小型Qwen fixture。该运行不是30B目标验收。
`outputs/npu_inductor_axis_probe.py`仅在进程内调整单reduction轴的
ReductionAnalysis.dense_size_list，按reversed(golden_var_list)解析轴序，
并校验轴集合和reduced_dim；多reduction轴保留原方法。它同时继承split
探针的首配置autotune选择及CantSplit兼容补丁，均为诊断，未修改已安装依赖，
不能替代正式修复。

实际命令（在上述节点容器的仓库根目录执行）：

```bash
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_NPU_BACKEND=default LLMTUNER_TEST_COMPILE_BACKEND=inductor TORCH_COMPILE_DEBUG=1 timeout 300 torchrun --standalone --nproc_per_node=2 outputs/npu_inductor_axis_probe.py full outputs/pp2-inductor-axis-probe-20261023
```

状态：失败；退出码124。UTC12:16:52启动，12:21:52超时终止。
日志：`outputs/qwen3-30b-npu/logs/pp2_inductor_axis_probe_runtime_20261023.log`。
12:19:35生成rank0的model__1_forward_4调试源码；12:19:48仍观察到
Triton相关g++/cc1plus编译活动。没有训练步、反向或checkpoint完成证据，
不能据此宣称原reduction错误已修复，也不能仅凭超时断定死锁。
日志明确记录_assert_async CPU fallback和ArgSort AiCPU执行。

本轮复核未发现对应worker（102628/102629）或g++/cc1plus存活；
仅本次两个worker的调试目录归档至
`outputs/pp2-inductor-axis-debug-20261023/`，不修改其他运行或生成源码。
下一步须检查实际生成kernel的轴/掩码布局，或基于编译仍活跃的证据制定
更长且有限的编译预算；不得盲目重复同一300秒命令。即使诊断成功，也须将
根因修复正式化，验证真实前反向、训练及独立恢复后才能更新compile验收状态。
本轮仅补证据记录及归档，不重跑CPU全套；既有失败和质量门禁缺口保持。

### 同日延长预算诊断：发现双reduction代码生成错误

运行前npu-smi确认节点92无NPU运行进程。由于前次超时前仍有编译活动，
本次保持全部诊断补丁、配置和断言，仅将timeout改为900、输出目录改为
`outputs/pp2-inductor-axis-long-probe-20261023`，其余环境及参数同上。
日志独立保存为
`outputs/qwen3-30b-npu/logs/pp2_inductor_axis_long_probe_20261023.log`。
UTC12:27:27完成Trainer装配，12:30:01退出1，未触及900秒预算，状态失败。

两个rank均生成前向调试图；rank1于12:29:56生成反向调试图。实际反向源码中
原tmp49/tmp58已变为[X0BLOCK_SUB,X1BLOCK_SUB,R2BLOCK_SUB]，
tmp52/tmp61对应reshape为[X0BLOCK_SUB,X1BLOCK_SUB,1]，与掩码顺序一致。
这证明该次生成源码的轴序发生预期改变，不证明训练数值正确或原问题正式关闭。

首个异常是rank1的NoTritonConfigsError：双reduction kernel
`triton_unk_fused_add_matmul_mul_neg_slice_slice_backward_sum_transpose_unsqueeze_view_12`
在没有定义loop_x0的情况下生成
`x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,None,:]`，
触发NameError。单reduction轴序补丁明确不改此多reduction分支。rank0随后
由torchrun终止；没有完成训练步或checkpoint，不记compile通过。
仅本次112957/112958两个worker的调试目录归档至
`outputs/pp2-inductor-axis-long-debug-20261023/`。下一步检查多reduction
分支的空间循环生成与reduction输出形状，保留原始生成代码，不手改生成代码
或将loop_x0强设0绕过。正式修复及独立恢复验收仍未完成。

进一步只读核对已安装依赖：triton.py::codegen_body第959–962行仅在
numof_reduction_axis()<=1时为非末尾tiling轴生成循环，与本次空间轴
loop_x0未定义吻合。kernel_analysis.py::ReductionAnalysis第236–252行
对多reduction轴强制persistent、reduced_dim=0，并将全部tiling轴尺寸
相乘展平。该次实际kernel的tmp6把X0BLOCK_SUB、R2BLOCK_SUB和
R1BLOCK_SUB一起展平，随后tl.sum(tmp6,0)，输出却仍按x0寻址。
因此仅补空间循环不能证明正确：还须验证空间轴逐元素保留，以及两个
reduction轴的独立归约与尾块mask。以上是源码和生成图证据，尚未运行
对应数值复现，不将疑似错误表述为已验证的数值失败。未修改依赖或生成代码。

对归档output_code.py进行AST静态审计，精确提取上述kernel，确认其读取但
未定义的循环变量集合为[loop_x0]；审计退出0，仅证明源码检查完成。结果为
`outputs/qwen3-30b-npu/logs/dual_reduction_static_audit_20261023.json`，
同时保存展平和求和语句。同级torchtitan-npu的Python源码中未找到
dual_reduction/numof_reduction_axis/dense_size_list/CantSplit/
NPUIndexTritonKernel对应修复，不据此推断其他依赖版本也无修复。

### 同日双轴归约真实NPU数值复现

诊断脚本`outputs/reproduce_npu_dual_reduction.py`构造真实Triton NPU
kernel，对比保留空间轴的双归约与全部轴展平归约。节点92设备空闲后运行，
torch-npu=2.12.0，退出0。日志为
`outputs/qwen3-30b-npu/logs/npu_dual_reduction_real_20261023.log`，
结果为`outputs/qwen3-30b-npu/logs/npu_dual_reduction_repro_20261023.json`。

容器仓库根目录执行命令：

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 timeout 180 python outputs/reproduce_npu_dual_reduction.py > outputs/qwen3-30b-npu/logs/npu_dual_reduction_real_20261023.log 2>&1
```

三个案例的正确实现均逐元素精确匹配CPU参考。完整块[A=4,B=8,X=4]的
正确结果为[2016,2048,2080,2112]，展平结果为[8256,8256,8256,8256]；
尾块有效范围[3,5,3]的正确结果为[615,630,645]，展平结果为
[1890,1890,1890]。X=1时两者相同，该退化案例不能验证空间轴语义。
该隔离复现确认展平空间轴的数值风险，不代表原生成kernel已执行，也不
代表整模型修复。脚本Ruff检查通过；未修改已安装依赖、生成代码或生产代码。

同日进一步调用已安装TileGenerator，按实际失败kernel元数据
size_hints=[128,64,4]、axis_names=[x0,r2,r1]、tiling_axis=[0,1,2]、
split_axis=[0]、low_dims={0,2}、persistent/dual_reduction=True生成
21个候选，退出0。首个X0BLOCK=128/X0BLOCK_SUB=64，后续还包含
128/32、64/32等配置，因此既不能假定空间块为1，也不能省略空间循环。
候选中也存在SUB=6/5/3，修复验证须检查实际Triton支持条件，不据此宣称
所有候选可编译。结果保存为
`outputs/qwen3-30b-npu/logs/dual_reduction_candidates_20261023.json`。
首次读取遗漏离线Triton路径而ImportError退出1；补齐
`PYTHONPATH=.:outputs/baseline-runtime`后完成上述候选生成，未执行kernel。
此证据将后续修复约束明确为：空间循环覆盖整个BLOCK、逐空间元素保留、
只归约r轴、正确应用所有有效范围mask，不能仅消除未定义变量异常。

后续新增隔离诊断`outputs/reproduce_npu_dual_reduction_tiled.py`，尝试
BLOCK=128/SUB=64的实际首候选、跨program尾块及BLOCK不能整除SUB
的空间循环。节点92运行前npu-smi确认无运行进程。首案例即编译失败，
退出1：BiShengHIR报告UB需要2097152 bits、可用1572864 bits，未执行
数值断言，后两个案例未运行，不记通过。失败日志保留为
`outputs/qwen3-30b-npu/logs/dual_reduction_tiled_real_20261023.log`。

复现命令（上述节点容器仓库根目录）：

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 timeout 180 python outputs/reproduce_npu_dual_reduction_tiled.py > outputs/qwen3-30b-npu/logs/dual_reduction_tiled_real_20261023.log 2>&1
```

该手写kernel与原融合kernel不同，不能将其UB需求直接外推为原融合kernel
的测量值。但它证明正确的分轴归约实现仍需资源验证；首配置固定选择不能
替代候选编译筛选。诊断脚本Ruff/format通过，未修改生产代码或安装依赖。

逐候选隔离诊断随后退出0：SUB=64的UB编译失败单独保留为
compile_failed，不计为通过；N=259、有效归约范围[3,61]、BLOCK=128、
SUB=32的跨program尾块，与N=131、BLOCK=6、SUB=4的非整除空间循环
均逐元素精确匹配CPU参考（rtol=atol=0）。target初始为NaN，断言覆盖
全部有效输出。诊断仅捕获明确的UB溢出，其余编译异常及数值失败仍抛出。
结果为`outputs/qwen3-30b-npu/logs/dual_reduction_tiled_20261023.json`；
日志为`outputs/qwen3-30b-npu/logs/dual_reduction_tiled_candidates_real_20261023.log`。
运行命令同上，仅日志名替换为该候选日志；运行前再次确认NPU空闲。
这验证了手写空间循环及分轴归约的有限配置，不证明已安装codegen修复，
不将诊断exit0汇总为所有候选通过。后续正式实现仍须处理循环作用域、
轴布局、mask与资源筛选，并运行整模型前反向及独立恢复。

### 同日多轴codegen诊断：已执行至首步finite检查，失败

新增`outputs/npu_inductor_dual_codegen_probe.py`，进程内从已安装方法读取
并严格匹配唯一循环条件，为多reduction的空间tiling轴生成循环；多轴sum
按reversed(golden_var_list)确定布局，应用有效范围mask，按降序维度逐轴
tl.sum，仅将r轴置1并保留空间维。非sum/tuple多轴归约明确拒绝。单轴
及既有CantSplit/确定性诊断沿用先前探针。未修改安装依赖或生产代码。

运行前节点92 npu-smi确认空闲，实际命令为容器仓库根目录：

```bash
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_NPU_BACKEND=default LLMTUNER_TEST_COMPILE_BACKEND=inductor TORCH_COMPILE_DEBUG=1 timeout 900 torchrun --standalone --nproc_per_node=2 outputs/npu_inductor_dual_codegen_probe.py full outputs/pp2-inductor-dual-codegen-20261023 > outputs/qwen3-30b-npu/logs/pp2_inductor_dual_codegen_20261023.log 2>&1
```

PP2/shard1/TP1/EP1/CP1、小Qwen fixture、1F1B四microbatches，原断言
及checkpoint保持。UTC13:07:48启动，13:13:27退出1，状态失败，非超时。
rank1于13:09:37、rank0于13:13:23生成反向代码。rank1实际源码包含
空间for loop_x0和带r1/r2/x0 mask的两次tl.sum，结果reshape为
[1,1,X0BLOCK_SUB]，未再遇到先前未定义loop_x0异常。

随后两rank均在Trainer._clip_and_check_finite/_check_finite报
Loss or gradient norm is not finite at step 1，optimizer更新前停止。
这证明该次编译图已推进到前反向后的finite检查，不证明数值正确；首步
未成功，无checkpoint/独立恢复通过证据。日志未区分loss与gradient norm
哪个首先非有限，不能凭此认定具体算子错误。下一有效诊断须在同次失败中
记录loss及各参数梯度的finite状态，保留原断言并定位首个非有限边界。
仅本次122032/122033的debug目录归档至
`outputs/pp2-inductor-dual-codegen-debug-20261023/`。诊断Ruff通过；
该进程内修复仍未正式化，P3整模型compile继续失败，整体目标未完成。

### 同日裁剪前finite诊断：NaN限于rank0 Q分支及上游梯度

新增忽略目录诊断`outputs/npu_inductor_finite_probe.py`，包装
Trainer._clip_and_check_finite，仅在调用原方法前记录loss_is_finite及各
参数本地梯度的NaN/Inf数量、shape和max_abs。CPU读取会同步并扰动时序，
属诊断；原clip、world有限性归约和断言完整保留，不更新非有限optimizer。
运行前npu-smi确认节点92空闲，诊断脚本Ruff通过。

```bash
PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_NPU_BACKEND=default LLMTUNER_TEST_COMPILE_BACKEND=inductor timeout 900 torchrun --standalone --nproc_per_node=2 outputs/npu_inductor_finite_probe.py full outputs/pp2-inductor-finite-20261023 > outputs/qwen3-30b-npu/logs/pp2_inductor_finite_20261023.log 2>&1
```

UTC13:17:43退出1，状态失败；同次两个rank的loss_is_finite=True。
rank1全部13个参数梯度有限；rank0共12个参数，仅以下3个梯度非有限：
embed_tokens.weight（128×64，NaN256）、layers.0.self_attn.q_proj.weight
（512×64，NaN9600）、layers.0.input_layernorm.weight（64，NaN64）。
上述均无Inf；其余rank0梯度有限。原断言在optimizer更新前停止。
两份独立rank报告保存在
`outputs/qwen3-30b-npu/logs/inductor_preclip_rank{0,1}_20261023.json`。
该同次观测区分了前向loss与裁剪前梯度异常，将候选范围缩小至第一stage
Q分支及其上游反向路径，但不证明q_proj是首个错误算子。后续须检查其
q_norm/RoPE/attention反向边界，不能凭参数列表修改模型或关闭断言。
该报告max_abs为NaN时使用Python JSON的NaN表示，读取需相同解析语义。
未修改生产代码/安装依赖，未关闭compile验收缺口。

### 同日逐kernel有限性及Q norm输入捕获

诊断`outputs/npu_inductor_kernel_finite_probe.py`包装
NPUCachingAutotuner.run，调用前检查in_ptr/in_out_ptr，调用后检查
out_ptr/in_out_ptr，逐调用写独立rank JSON；原run及Trainer断言保留。
CPU读取同步会扰动时序，不属于正式验收。节点92运行前设备空闲。
使用前节完整环境、timeout900、nproc2，将脚本替换为此脚本，输出目录
`outputs/pp2-inductor-kernel-finite-20261023`，日志
`outputs/qwen3-30b-npu/logs/pp2_inductor_kernel_finite_20261023.log`。
UTC13:21:47退出1，原finite断言失败。rank0记录157个kernel调用，首个
非有限写出在零基event100的Q norm融合反向kernel（名称后缀
slice_slice_backward_t_transpose_unsqueeze_view_10）；五个输入均无
NaN/Inf，输出in_out_ptr0出现1个NaN。rank1记录195个调用，无非有限
写出。这只定位已观测Triton调用边界，不覆盖所有外部ATen kernel内部。
报告：`inductor_kernel_finite_rank{0,1}_20261023.json`，位于上述logs目录。

随后设置LLMTUNER_CAPTURE_QNORM=1，保持原配置与断言，在该特定
kernel前克隆CPU输入，保存调用后输出及实际launcher配置。独立运行目录
`outputs/pp2-inductor-qnorm-capture-20261023`，日志
`pp2_inductor_qnorm_capture_20261023.log`，UTC13:24:27退出1。
捕获文件`outputs/inductor-qnorm-capture-20261023/rank0_{0..7}.pt`。
此运行独立finite报告带_capture后缀，不覆盖前次观测。加入克隆后首个
非有限边界变为event103的embedding归一化反向kernel（后缀
add_div_embedding_dense_backward_expand_mul_pow_sum_12），七个输入
均有限，输出出现192个Inf。不能把两个运行拼接成同次根因链。

对八份Q norm捕获按生成源码公式独立CPU计算：所有参考结果有限，
输出0/2/4精确，1/3/6/7存在误差，5含Inf；部分输出偏差达2.085e35。
注意输出保存发生于after有限性扫描之后，其观测时点不同，不能声称这
两次读取完全一致。审计文件
`outputs/qwen3-30b-npu/logs/qnorm_capture_formula_audit_20261023.json`。
实际launcher配置Y0BLOCK=4/Y0BLOCK_SUB=3/X2BLOCK_SUB=4/
X3BLOCK_SUB=128，说明相同kernel/配置在调用间表现不同。下一步应独立
重放已捕获输入并核对输出完成/内存写入边界，不能把有限性扫描当成数值
正确证据，也不能仅凭单次边界认定Q norm为唯一根因。生产代码与已安装
依赖未修改，compile状态仍失败。

### 同日原Q norm kernel独立重放

诊断`outputs/replay_qnorm_captured.py`从归档rank0反向output_code.py
精确提取唯一后缀为slice_slice_backward_t_transpose_unsqueeze_view_10
的kernel函数体，原始归档不改；仅移除autotuner装饰器，独立生成
`outputs/qnorm_replay_extracted_kernel.py`并通过triton.jit直接运行。
八份捕获各用SUB=3（实际配置）及SUB=4（诊断对照）重放三次，
Y0BLOCK=4、X2BLOCK_SUB=4、X3BLOCK_SUB=128、grid16；每次重建
NPU输入，并在执行后显式同步再读取。CPU参考沿用该kernel的实际
运算公式。该验证绕开训练/PP与autotuner，不能替代其验收。

节点92运行前npu-smi确认空闲。命令为容器仓库根目录：

```bash
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0 timeout 240 python outputs/replay_qnorm_captured.py > outputs/qwen3-30b-npu/logs/qnorm_isolated_replay_20261023.log 2>&1
```

诊断退出0仅表示收集完成，不代表数值通过。SUB3的24次中23次有限、
1次非有限；SUB4全部24次有限，但两组均无逐位精确案例。部分相同输入
重复执行的maxAE不同，SUB4最大AE=0.00128235。报告为同目录同名JSON。
该处精确比较只是诊断，不擅自将其设为目标验收阈值；非有限值及重复
执行不稳定本身需处理，不能由仅有限的对照结果关闭问题。

另一次在NPU输入搬运后、kernel执行前增加显式同步，设置
LLMTUNER_REPLAY_SYNC_INPUT=1，其余参数相同，日志与JSON名均增加
_sync_input后缀。退出0，SUB3/4共48次均有限，但仍无精确案例；
最大AE分别0.00168037/0.000595957，分别6/4份捕获在三次重放中maxAE
变化。同步未消除数值不稳定，不纳入生产路径作为修复。两次均已终止。
Ruff及git diff --check通过；下一步须核实直接launch参数/编译选项、
实际写出覆盖和backend局部buffer复用，不能仅将空间块改为4宣称修复。

### 同日重放launch选项核对及multibuffer对照

复核发现此前直接triton.jit重放未显式传入原启发式的num_warps=1和
num_stages=1，不能将其表述为与autotuner完全相同的launch选项。本轮
新增LLMTUNER_REPLAY_MATCH_OPTIONS=1传入两项，保持输入同步，独立
日志`qnorm_replay_matched_options_20261023.log`，JSON为
`qnorm_isolated_replay_sync_input_matched_options_20261023.json`，均位于
outputs/qwen3-30b-npu/logs。退出0仅表示收集完成：两种SUB各24次均有限，
均无精确案例；SUB3最大AE=0.000721585、4份输入重复结果变化，SUB4
最大AE=0.000638768、3份输入重复变化。未解决不稳定，未宣称数值通过。

核对rank0_0捕获五个输入shape/stride：原激活[1,64,512]/[32768,512,1]，
grad[1,4,64,128]/[32768,8192,128,1]，weight[128]/[1]，rsqrt
[1,64,4,1]/[256,4,1,1]，reduced[1,64,4,1]/[256,4,1,256]；
storage_offset均0，匹配kernel对应非单例轴的固定地址公式。这是该份
捕获的布局审计，不外推全部图的布局正确性。

后端compiler.py将multibuffer编译选项映射至enable-auto-multi-buffer。
在上述环境增加LLMTUNER_REPLAY_NO_MULTIBUFFER=1，仅额外传入
multibuffer=False，日志`qnorm_replay_no_multibuffer_20261023.log`，JSON
名在matched_options后增加_no_multibuffer。退出0，两种SUB各24次均
有限但仍无精确案例；SUB3最大AE=7.51317、8份输入重复变化，SUB4
最大AE=0.00124206、4份输入重复变化。关闭多缓冲未消除问题，不能
作为生产修复。本轮开始时节点92无NPU进程，两次运行均终止。

复现为前节完整命令增加LLMTUNER_REPLAY_SYNC_INPUT=1及
LLMTUNER_REPLAY_MATCH_OPTIONS=1；第二组再增加
LLMTUNER_REPLAY_NO_MULTIBUFFER=1，分别写入上述独立日志。
后续须直接验证每个program的有效写入覆盖和后端局部buffer复用；不
重复同一同步/块大小/多缓冲开关试跑。生产实现及原失败状态保持。

## 2026-10-23：P1 HSDP 独立参考及30B入口覆盖

### 目标模型结果：HSDP2×4 首次反向 OOM

30B实际运行exit1，未完成step1，日志
`outputs/qwen3-30b-npu/p1_hsdp2x4_node92_20261023/logs/`。实际mesh明确为
dp_replicate2、dp_shard4，48层full AC及HSDP均装配，HF加载5.83s成功。
失败位于首次backward：rank0尝试分配770MiB，已分配53.57GiB、设备总容量
61.27GiB、报告free796.56MiB；多个rank同时OOM，不能仅归因于碎片。
此配置记失败，不是数值通过，也不以四卡TinyTokenDecoder结果关闭缺口。
后续须定位反向峰值及FSDP参数/梯度生命周期，再复验相同配置；缩短序列、
改变精度或offload只能作为另行标注的诊断/配置，不能覆盖该失败。
本轮所有NPU作业均已退出，未遗留训练进程。整体目标仍未完成。

node92启动前八卡无进程。四卡native HSDP2×2独立参考脚本
`tests/integration_tests/fsdp_mesh_equivalence.py hsdp` exit0：4步FP32，
预设tol1e-4，首步梯度、逐步loss、最终参数和无通信反例均检查；最终参数
maxAE5.960e-8，failed ranks0。这是TinyTokenDecoder证据，不外推30B。
日志 `outputs/qwen3-30b-npu/logs/p1_hsdp_reference_node92_20261023.log`。

```bash
ssh -o BatchMode=yes 10.1.0.92 "docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime
torchrun --master-port=29620 --nproc-per-node=4 tests/integration_tests/fsdp_mesh_equivalence.py hsdp
'"
```

30B入口增加 `LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE`（默认1），保留自动推导
shard degree。配置回归9 passed、14 warnings。接着启动目标30B-A3B/LIMO八卡
HSDP2×4，BF16/fullAC/seed42/deterministic/seq4096/batch8，2步、不保存，
输出 `outputs/qwen3-30b-npu/p1_hsdp2x4_node92_20261023`。该用例只检查目标
路径加载与训练，不替代HSDP checkpoint或完整数值等价验收；作业尚未结束。
复现沿用30B入口P0命令，设置replicate=2、steps=2、save_checkpoint=0及
上述独立dump目录，PYTHONPATH包含同一TensorBoard runtime。

## 2026-10-23：30B 独立恢复指标审计

统一解码后的完整审计已exit0：同一对连续/恢复step10 checkpoint 的19697个
chunk全部精确一致，分类计数为model4248、optimizer15399、dataloader39、
RNG8、scheduler1、train_state2，failures=[]。产物为
`outputs/qwen3-30b-npu/logs/p0_node92_decoded_state_comparison_20261023.log`。
因此本配置的10步训练、计时、完整训练状态保存及step5→10独立恢复已有通过
证据；未启用EMA，不声称覆盖EMA。该结论仅限30B-A3B/8卡native FSDP/
BF16/fullAC/LIMO/seq4096/batch8/seed42，不扩展至其他并行族及P1–P4。
质量门禁仍有历史环境失败，整体目标未完成。

完整状态审计复现（CPU分块解码，不启动NPU）：

```bash
ssh -o BatchMode=yes 10.1.0.92 "docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
python outputs/compare_30b_checkpoint_chunks.py outputs/qwen3-30b-npu/p0_node92_20261023/checkpoint/step-10 outputs/qwen3-30b-npu/p0_resume_state_node92_20261023/checkpoint/step-10
'"
```

第二次独立恢复（保存完整最终状态）exit0，日志
`p0_resume_state_node92_20261023/logs/train_20261023_060117.log`。step10保存
18.51s。第一次完整chunk审计共19697项，模型4248、optimizer15399、
dataloader39、scheduler1、train_state2均精确一致；仅8个RNG opaque pickle
字节不同，审计如实exit1，保留
`outputs/qwen3-30b-npu/logs/p0_node92_state_comparison_20261023.log`。
独立解码该8项后，全部rank的version/world_size/device_type/parallel_degrees、
Python/NumPy/CPU/NPU随机状态均精确一致，exit0；日志
`outputs/qwen3-30b-npu/logs/p0_rng_payload_audit_20261023.log`。pickle字节编码
不是随机状态语义，不能仅据字节差异认定随机流不同。已修改完整审计脚本在
同一次遍历中递归解码RNG，重新审计全部状态；不比较不同训练运行的拼接结果。

node92 第一次独立恢复进程 exit0，日志
`outputs/qwen3-30b-npu/p0_node92_20261023/logs/train_20261023_055725.log`。
加载step5耗时22.76s，实际继续step6–10。用
`outputs/compare_30b_resume_metrics.py` 读取两份独立TensorBoard event，逐步
检查 global_avg_loss、grad_norm、lr/AdamW、n_tokens_seen，共20项精确相等，
audit exit0，产物 `outputs/qwen3-30b-npu/logs/p0_node92_resume_metrics_20261023.json`。
这是TensorBoard float32标量证据，不是所有模型/optimizer/RNG/游标值的证明。
实际警告还包括 aten::_assert_async.msg CPU fallback 与整数ArgSort AiCPU，
保留回退记录，不宣称纯AiCore。

为补完整状态证据，启动第二次独立恢复，在新目录
`outputs/qwen3-30b-npu/p0_resume_state_node92_20261023` 中使用step5的只读输入
硬链接副本，启用最终step10完整保存；不覆盖原始连续训练step10产物。
配置与前次恢复一致，仅save_checkpoint=1及独立dump目录不同。准备
`outputs/compare_30b_checkpoint_chunks.py` 按DCP物理chunk逐个CPU解码并递归
精确比较，避免整模型及optimizer一次聚合内存；要求双方键及chunk布局相同，
不同则明确失败，不忽略状态分类。尚未运行该完整状态审计，P0恢复未关闭。

## 2026-10-23：专家局部读取回归与 node92 正式训练复验

本轮复查 node18 八卡仍被原 PID484066–484073 占用。依次尝试目标列出的
候选 node69/node71，八卡也均有运行进程，未启动训练或干扰作业；最终所用
node92（10.1.0.92，hostname `kcs-pcnlcbi-s-vw7pv`）八卡无运行进程。
在 node92 核对运行时后，按已固定 P0 配置启动10步训练，唯一变更为节点及
独立输出目录 `outputs/qwen3-30b-npu/p0_node92_20261023`。旧 node18 失败保留，
不覆盖、不用换节点结果掩盖历史资源冲突。

HF expert reader 增加重复数字索引拒绝：`experts.1` 与 `experts.01` 不得在
转换为整数后静默覆盖。新增 gate/up/down 三种投影的局部二维切片读取断言，
用实际 safetensors 和非零源 offsets 检查读取内容及 planner commit，属于
CPU 读取契约，不冒充 NPU TP/EP 数值证据。首轮重复索引测试 fixture 因共享
storage 被 safetensors 拒绝，改为独立 clone 后，受影响三文件41 passed、
29 warnings，退出0。未为此放宽任何生产断言。

node92 复现：沿用下方10步预定命令，将 ssh 地址改为 `10.1.0.92`，
`LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/p0_node92_20261023`。
训练阶段已 exit0，日志 `p0_node92_20261023/logs/train_20261023_055310.log`
（实际文件名以目录为准）。10步均完成，loss 有非单调波动；step5与step10
均保存成功。step5 metadata 实际8546项，包含 model、optimizer、lr_scheduler、
dataloader、rng_state、train_state；仅键存在不等于恢复值已验证。

通过 `outputs/audit_30b_p0_metrics.py` 从唯一训练 TensorBoard event 读取
10个step计时，排除step1–3后：min7.862334s、median7.950103s、mean10.643033s、
p95（线性插值）21.207358s、max26.875656s。step6窗口包含step5 checkpoint
保存耗时，不删该异常点；最后step10保存发生在最后一次metrics之后，不在
上述窗口分布中。另记录step5保存18.38s、末次保存18.63s。不是无checkpoint
纯计算吞吐，也不是从训练启动到最终退出的总耗时。

已启动新进程从step5恢复，总步数仍10，保持调度/精度/数据配置，
`LLMTUNER_CHECKPOINT_LOAD_STEP=5`、`LLMTUNER_SAVE_CHECKPOINT=0`，不覆盖
原checkpoint；继续使用同一dump目录，由时间戳生成独立日志/event。
尚需核对step6–10指标及模型/optimizer/RNG/数据游标的完整状态断言，不能
仅凭训练阶段或恢复进程退出0关闭P0恢复项。

## 2026-10-23：30B 专家布局适配与真实单步训练

### 本轮最新状态：正式训练资源冲突

10 步正式作业未完成，exit1，失败发生在首次 HCCL 初始化，错误 EI0020：
NPU 网络端口 16666 已绑定；日志
`outputs/qwen3-30b-npu/p0_train_20261023/logs/`。失败后再次检查 npu-smi，
八卡已被 PID484066–484073 占用，每卡约16GB；宿主 ps 确认其为其他用户
`luy01` 的 `code/train_novel.py`。未修改端口强行并跑、未终止其他任务。
当前仅正式 NPU 多步验收因资源 blocked，单步通过不扩大为 P0 通过；
解锁条件为所用节点八卡空闲后重新运行并保留本次失败记录。

全量 CPU 首轮为 4245 passed、13 skipped、2 failed、272 warnings：两项子进程
测试显式重写 PYTHONPATH，仅保留仓库根目录，导致隔离 runtime 中 colorama
不可见（multimodal 轻量导入、create_seed_checkpoint）。未修改测试绕过依赖。
另建独立 `outputs/qwen3-30b-test-venv`，通过该 venv 的 .pth 固定已安装
runtime 路径，重跑全套并设置 `ASCEND_RT_VISIBLE_DEVICES=''`，确认设备数0，
避免占用他人 NPU。日志为
`outputs/qwen3-30b-npu/logs/cpu_suite_expert_reader_venv_20261023.log`。
全仓 Ruff check、Ruff format --check（451 files）、git diff HEAD --check 通过；
pre-commit 尚未复验，不将局部门禁等同于全部门禁。

隔离 venv 全套最终 exit1：4037 passed、210 failed、13 skipped、218 warnings。
独立 `components -x` 首个失败确认：torch-npu 的 optimizer 自动选择补丁在 CPU
AdamW 更新时调用 `npu.current_device()`，无可见设备导致 aclInit107001；
不能将隐藏设备后的全套运行当作可用 CPU 环境，也未证明210项同一根因。
不更改 optimizer 或测试以绕过该环境问题。另在同一隐藏设备环境运行本次
reader、既有 adapter、示例配置三个受影响测试文件，共37 passed、29 warnings；
覆盖跨3个源 safetensors 的多线程读取、single-file、缺投影及混合布局拒绝。
`python -m compileall -q llmtuner` exit0。完整门禁仍未通过。

```bash
ssh -o BatchMode=yes 10.1.0.18 "docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
ASCEND_RT_VISIBLE_DEVICES=\"\" outputs/qwen3-30b-test-venv/bin/python -m pytest -q --disable-warnings tests/unit_tests/cpu/components/checkpointer/test_hf_expert_storage.py tests/unit_tests/cpu/models/test_state_dict_adapter.py tests/unit_tests/cpu/test_npu_example_config.py
'"
```

字段修正后的 seq256 诊断 exit1：LIMO 完整 SFT 样本超长，部分 DP 分片全部被
既有长度过滤拒绝；未截断 response、未绕过非空保护。该日志为
`outputs/qwen3-30b-npu/diagnostic_startup_fields_fixed_20261023/logs/train_20261023_052800.log`。
随后 seq4096 通过数据初始化，但 HF 初始权重覆盖检查 exit1：源为逐专家
`experts.N.{gate,up,down}_proj.weight`，Transformers 5.11 模型为融合
`experts.gate_up_proj/down_proj`；日志在
`outputs/qwen3-30b-npu/diagnostic_seq4096_20261023/logs/`。这是实际加载失败，
不是环境 blocked；未关闭 strict 检查或改为随机初始化。

新增 grouped expert HF reader，以每个二维专家投影作为融合三维张量的 DCP
chunk，仅读取当前 rank 需要的切片，不在各 rank 聚合完整权重。显式检查专家
数量、投影完整性、dtype、形状与混合布局。HF adapter 的键覆盖使用同一逻辑
布局；indexed 与 single-file CPU 精确读取及既有 adapter 回归共 28 passed、
29 warnings。相关 Ruff check/format 通过，全量 CPU suite 另行记录。

修复后真实诊断 `diagnostic_expert_reader_20261023`：8 卡 FSDP、BF16、
full AC、seq4096、batch8、seed42、deterministic、1 step、不存 checkpoint，
exit0。实际加载耗时 10.42 秒，step1 loss=0.53537、grad_norm=3.3731，日志
峰值约 32.53 GiB。optimizer 明确记录 fused 不可用，实际使用 AdamW for-loop；
未据此宣称纯 AiCore 或所有算子无回退。日志位于
`outputs/qwen3-30b-npu/diagnostic_expert_reader_20261023/logs/`。

TensorBoard 所缺 grpcio 等 wheel 已由宿主下载并在容器离线安装至独立
`outputs/qwen3-30b-tb-runtime`，SummaryWriter 导入成功，未覆盖训练 runtime
或容器全局包。上述单步已启用持久化指标；不以单步结果代替 P0。

### 预定 P0 训练与计时配置

运行前固定：30B-A3B/LIMO，8 卡 native FSDP，BF16，full AC，seed42，
deterministic，global batch8，seq4096，10 steps；AdamW lr1e-5、weight_decay0.1，
其余配置保持入口默认。step1–3 为预热，统计 step4–10 持久化的
`time_metrics/end_to_end(s)` 的 min/median/mean/p95/max；计时包括该窗口内
实际开销，不从控制台秒级时间戳推算。checkpoint interval5，保存完整训练
状态及最终 step10；恢复后另以相同总步数、调度和数据顺序比较 step6–10，
预定逐位一致，不因失败调整阈值。当前仅启动训练阶段，不将恢复检查记通过。

```bash
ssh -o BatchMode=yes 10.1.0.18 "docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime:outputs/qwen3-30b-tb-runtime
export LLMTUNER_STEPS=10 LLMTUNER_MAX_SEQ_LEN=4096 LLMTUNER_GLOBAL_BATCH_SIZE=8
export LLMTUNER_SAVE_CHECKPOINT=1 LLMTUNER_CHECKPOINT_INTERVAL=5
export LLMTUNER_DETERMINISTIC=1 LLMTUNER_ENABLE_TENSORBOARD=1
export LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/p0_train_20261023
bash examples/train_qwen3_30b_npu.sh
'"
```

单步诊断复现使用同一命令，改 steps=1、save_checkpoint=0、dump_folder 为
`outputs/qwen3-30b-npu/diagnostic_expert_reader_20261023`。历史两轮失败命令
不含 TensorBoard runtime，seq256/4096 分别对应上述目录。整体仍未完成。

## 2026-10-23：新目标切换为 Qwen3-30B-A3B

当前目标来源为附件 `f5f57b7a-4c55-450a-be6a-50b525fdba7a/pasted-text-1.txt`。
模型改为 Qwen3-30B-A3B，数据为 GAIR/LIMO，入口为
`examples/train_qwen3_30b_npu.sh`。下方历史 Qwen3-8B 和小模型证据保留原
适用范围，不视作 30B 目标通过；整体 P0–P4 仍未完成。

本轮仅检查所用节点 10.1.0.18（hostname `kcs-pcnlcbi-s-gwf82`）：
`hyper-parallel-env` 存在，8×Ascend910 均无运行进程，torch 2.12.0+cpu、
torch-npu 2.12.0、Python 3.12.13。模型 config.json 与 LIMO 数据可读；
模型配置为 qwen3_moe、48 层、128 专家、每 token 8 专家。
TorchLLMTuner HEAD 为 `c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19`，
torchtitan-npu 为 `b358a8063c34aa7494397efa3a8f5ad24f1caa1f`，torchtitan 为
`f5c97dc0e2d6cca72313004aa853a196ca02ab49`；工作区已有大量未提交修改，保留。

修正已有未跟踪 30B Python 副本中的 8B 模型路径、函数名、输出目录与错误
提示，改用 LIMO，并新增对应 shell 入口。新增配置回归测试，容器中 8 项通过，
14 warnings；相关文件 Ruff check/format 与 shell 语法检查通过。
这仅证明配置，不证明实际加载、训练、恢复或数值等价。

该节点原始容器缺少 colorama、pytest 等依赖，原始 transformers 为 4.56.2。
从宿主 wheels 离线安装到 `outputs/qwen3-30b-runtime`，不覆盖容器全局包；
运行时优先使用已有 `wheels/tf511` 中 transformers 5.11.0/hub 1.21.0。
pip 提示的全局依赖冲突尚未作为完整环境兼容验收关闭。tensorboard 安装因
离线缺 grpcio 未完成，真实端到端计时验收仍未执行。

```bash
ssh -o BatchMode=yes 10.1.0.18 "docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime
python -m pytest -q --disable-warnings tests/unit_tests/cpu/test_npu_example_config.py
'"
```

目标模型的启动诊断预定为 8 卡 FSDP、BF16、full AC、seed42、确定性开启、
batch8、seq256、1 step、不保存 checkpoint。它不是正式 P0，不能替代 seq4096
多步计时及独立进程恢复；诊断日志单独保存于
`outputs/qwen3-30b-npu/diagnostic_startup_20261023/logs/`。

| 测试项 | 目标配置 | 状态 | 尚需证据 |
| --- | --- | --- | --- |
| P0 | 30B、8 卡 FSDP、LIMO | 未跑 | 正式多步训练、端到端耗时、独立恢复 |
| P1 | FSDP/HSDP、TP、PP、CP、EP | 未跑 | 冻结目标矩阵并逐项验证，历史结果不外推 |
| P2 | 目标规定的并行组合 | 未跑 | 目标模型适用范围、实际分组和数值断言 |
| P3 | 目标规定的训练特性 | 未跑 | 正式路径与恢复证据；历史缺口仍保留 |
| P4 | 与 torchtitan-npu 真实入口对齐 | 未跑 | 30B 初态/输入对齐、逐步指标及关键权重标准 |

### 30B 启动诊断第一轮：LIMO 字段修正

第一轮 exit1，日志为上述目录 `train_20261023_052413.log`。48 层 full AC 与
FSDP 装配已执行，数据初始化报 `local_jsonl_sft row lacks configured field
'problem'`，尚未完成模型 checkpoint 加载及训练 step。逐行读取原始 LIMO，
817 行字段均为 `question/solution/answer`，值类型均为字符串；30B 入口默认
prompt 改为 `question`，response 保留完整 `solution`，不改用最终短答案。
相应回归断言同步修正，尚需重新运行目标模型启动诊断。

第一轮完整复现命令（修正后的源码不再预期复现同一字段错误）：

```bash
ssh -o BatchMode=yes 10.1.0.18 "docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime
export LLMTUNER_STEPS=1 LLMTUNER_MAX_SEQ_LEN=256 LLMTUNER_GLOBAL_BATCH_SIZE=8
export LLMTUNER_SAVE_CHECKPOINT=0 LLMTUNER_DETERMINISTIC=1
export LLMTUNER_DUMP_FOLDER=outputs/qwen3-30b-npu/diagnostic_startup_20261023
bash examples/train_qwen3_30b_npu.sh
'"
```

## 2026-10-23：Triton归约mask最小复现（设备占用下的interpreter证据）

本轮准备了可直接运行的最小脚本
`outputs/reproduce_npu_reduction_mask.py`，覆盖修正mask、旧mask及非对称/方形
block。启动前再次核实八卡仍被其他用户的`train_novel.py`占用，因此没有执行
真实NPU kernel，也没有终止或干扰其他任务。脚本通过Ruff、离线导入及Triton
JIT定义检查；这些检查不计NPU编译通过。

在`TRITON_INTERPRET=1`下运行同一kernel完成三组语义复现，输出
`outputs/qwen3-8b-npu/logs/triton_mask_interpreter_20261023.json`，退出0：

| block | 修正mask | 旧mask | 结论 |
| --- | --- | --- | --- |
| 1×4×8，有效1×3 | 逐位正确 | `reshape() cannot change total number of elements` | 旧路径编译/形状失败 |
| 4×1×8，有效3×1 | 逐位正确 | `reshape() cannot change total number of elements` | 旧路径编译/形状失败 |
| 4×4×8，有效2×3 | 逐位正确 | 可运行但与参考不一致 | 方形block不能掩盖mask语义错误 |

这与归档真实Inductor失败的`tmp51`/`tmp52`错误一致，并增加了独立的
“方形block也错误”反例；仍未证明torch_npu依赖中的轴映射已修复，也未替代
真实NPU非对称block和整模型反向。下一次设备空闲时先运行该最小脚本，再决定
是否需要依赖侧最小补丁；不得直接修改归档生成代码或强制两个空间block相等。
本轮未改生产源码或正式测试，诊断Ruff/format及git diff --check通过；整体
compile验收仍失败，P0–P4目标仍未完成。

## 2026-10-23：归约mask语义反例与设备占用检查

本轮原计划真实NPU最小编译复验，启动前npu-smi显示8卡均有其他用户作业，
PID134830–134837，各约18GB；宿主机ps确认均为其他用户环境的train_novel.py。
未启动NPU任务、未终止或修改其他作业。容器内pgrep torchrun为空不等于宿主
设备空闲，本次以npu-smi及宿主PID核对为准。该资源条件只阻塞NPU复验，
不将整体任务标为blocked，仍推进独立离线工作。

新增忽略目录`outputs/audit_inductor_mask_semantics.py`，用显式双循环
构造正确参考，验证把mask轴顺序与数据[X1,X0,R]一致时的归约语义：
block(1,4,8)/有效(1,3)、block(4,1,8)/有效(3,1)、block(4,4,8)/有效(2,3)
三项修正mask均逐位匹配参考，旧mask均错误。尤其方形block虽不触发reshape
异常，非对称有效边界仍错误，证明“强制block相等”不能修复mask语义。
这只是独立CPU语义证明，不是依赖代码已修复或NPU编译通过。

复现：容器内仓库根目录`python outputs/audit_inductor_mask_semantics.py`，
退出0，产物`outputs/qwen3-8b-npu/logs/inductor_mask_semantics_20261023.json`。
下一步须在空闲设备上执行真实Triton非对称block最小复现，再验证依赖侧
轴映射及整模型前反向；不直接修改归档生成代码充当正式修复。
本轮无正式代码变更，诊断Ruff/format及git diff --check通过，离线任务已结束。
整体目标仍未完成，compile及历史恢复失败保持原状态。

## 2026-10-23：整模型Inductor反向失败的轴顺序独立审计

本轮转向既有compile缺口，不重复恢复成功探针。读取原始目标、开发指南及
归档反向生成代码，定位到真实失败kernel的mask与数据维度顺序冲突：
`x0_mask`为[X0,1,1]，`x1_mask`为[1,X1,1]，而tmp49被reshape为[X1,X0,128]；
随后`tl.where(x0_mask & x1_mask,tmp49,0)`再沿axis2归约，最后要求reshape为
[X1,X0,1]。生成源码为既有归档`outputs/pp2-inductor-split-debug-20261023/`
下`model__1_backward_6.1/output_code.py`，原失败位于tmp52表达式。

新增忽略目录`outputs/audit_inductor_reduction_shape.py`先断言归档实际包含
上述表达式，再用CPU meta tensor独立验证广播/元素数，不启动NPU编译：

| X1/X0 | where实际形状 | 归约元素数/请求元素数 | 结论 |
| --- | --- | --- | --- |
| 1/4 | 4×4×128 | 16/4 | reshape非法 |
| 4/1 | 4×4×128 | 16/4 | reshape非法 |
| 4/4 | 4×4×128 | 16/16 | 形状合法，不证明语义正确 |
| 64/4 | 广播失败 | 无 | 两空间轴不兼容 |

容器内`python outputs/audit_inductor_reduction_shape.py`退出0，结果
`outputs/qwen3-8b-npu/logs/inductor_reduction_shape_audit_20261023.json`。
这些是形状契约测试，不是实际失败autotune block值的捕获，不证明所有候选
都属于该情形。与原Triton reshape错误一致，提供可独立复现的具体缺陷线索。

同时只读核对当前安装的torch_npu/_inductor/codegen/triton.py reduction实现：
约1220行先按dense_size_str reshape数据，约1260行仍直接连接原axis masks，
没有随数据轴顺序重排；不能仅修最后reshape形状掩盖错误mask语义。
直接导入该模块未设置隔离Triton路径时报缺triton，这是本轮只读探针环境
错误，不是新的模型训练失败；后续通过读取实际安装文件完成核对。

未修改依赖、生产源码或正式测试，也未重跑已失败三轮的整模型编译包装。
下一有效动作应从依赖侧证明数据与mask一致的轴映射，并用非对称block验证
真实编译前反向；仅把两个空间block强制相等不算通用修复。原整模型compile
项仍失败，不以meta审计或collective编译通过替代。诊断Ruff/format及
git diff --check通过，所有本轮离线任务已结束，整体目标仍未完成。

## 2026-10-23：forward当时与延后落盘前的logits摘要对照

重新读原始目标及开发指南后，针对旧保留引用产物的生命周期疑点，新增
忽略目录诊断`outputs/probe_logits_lifetime.py`。仅在step3原始模型forward
输出计算设备端(sum、square sum、max abs)，保留摘要和tensor引用及_version，
在正式恢复比较结束后重算摘要再转CPU。不在forward中host同步或复制全部
logits，但新增归约kernel仍会扰动调度。摘要有碰撞，不是逐元素不可变证明。

配置沿用原NPU4 TP2/EP2/KVCP2、native FP32、fullAC、dynamic EMA及
RNG/loader/RoPE精确检查。复现为原正式full/resume命令的入口换成
`outputs/probe_logits_lifetime.py`，工作目录参数改为
`outputs/tp-ep-cp-logits-lifetime-20261023`，完整参数/环境沿用本文件
“正式恢复测试补齐RoPE运行态”节，无其他变更。
日志`outputs/qwen3-8b-npu/logs/logits_lifetime_{full,resume}_20261023.log`。

full/resume均exit0，本次未复现历史失败。四rank两阶段各一个step3输出，
8组摘要初末精确一致、版本均0→0，对应full/resume摘要也相同。原始JSON为
产物目录`lifetime-{full,resume}-rankN.json`。这仅证明本次被采集的摘要及
版本未变，不能追溯证明旧失败产物完整无覆盖，也不能用成功运行排除失败
时的存储/调度问题。正式间歇性恢复项保持失败，不把诊断exit0计入关闭。

从引用边界、逐行离线分布到消费前后摘要的有限探针已完成；继续同类成功
探针收益不足，下一轮转向独立未完成项或新的算子/通信证据，不重复此组。
本轮无生产或正式测试修改，两个NPU作业与JSON审计已结束，诊断Ruff、format
及git diff --check通过；未重跑CPU套件。整体P0–P4目标仍未完成。

## 2026-10-23：已保存失败logits的逐token差异审计

本轮读取原始目标/指南，离线分析原
`outputs/tp-ep-cp-retained-logits-20261023`失败对，不新跑训练或重复成功探针。
脚本`outputs/audit_failed_logits_rows.py`读取四rank各512×128输出，按每行
maxAE排序，并记录绝对差均值、阈值以上行数及argmax变化；阈值只用于描述
差异分布，不是新增验收标准。复现：容器内仓库根目录运行
`python outputs/audit_failed_logits_rows.py`，退出0，产物
`outputs/qwen3-8b-npu/logs/failed_logits_row_audit_20261023.json`。

| rank | meanAE | maxAE | 行maxAE>1e-3 | 行maxAE>1e-2 | argmax改变行数 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.833864e-4 | 0.03602237 | 72 | 2 | 0 |
| 1 | 3.617112e-5 | 0.00058490 | 0 | 0 | 0 |
| 2 | 2.108346e-5 | 0.00037612 | 0 | 0 | 1 |
| 3 | 1.492593e-5 | 0.00032982 | 0 | 0 | 0 |

rank0最大差异局部行14，其次15/69/24；不擅自把局部行映射到全局样本，
因为原探针未同步保存输入与布局。不是单个异常元素或仅标量归约差异；但
仍只是延后落盘tensor的分布，不能单凭该产物证明forward当时值，旧存储
生命周期限制继续保留。源码核对模型返回lm_head结果，普通CE路径没有显式
原地修改logits；这只是静态线索，不能排除依赖内部或异步底层写入。

本结果给出后续采样优先位置（rank0前部token、rank2 argmax改变），不支持
直接归因router临界top-k或通信。下一有效实验应同时记录同次失败输入及
版本/存储元数据，或捕获消费logits时的校验值，不能将其他成功运行拼接到
本失败对。CPU离线审计已结束；诊断Ruff/format及git diff --check通过。
本轮无正式源码或测试改动，整体目标仍未完成。

## 2026-10-23：保留边界引用及版本计数的有限探针

本轮重新读原目标及指南。源码检索未发现训练路径显式原地修改logits，
但这不证明底层存储永不被改。扩展忽略目录探针
`outputs/probe_failed_logits_reference.py`，仅保留step3两层attention/MoE及
模型输出引用，同时记录tensor._version，检查结束后才转CPU落盘。
无clone/同步插入forward；仍可能因保留引用影响分配和调度，不是无扰动证明。

沿用原NPU4 TP2/EP2/KVCP2/native/FP32/fullAC/EMA/RNG/loader/RoPE精确
配置及上节入口，产物改为`outputs/tp-ep-cp-retained-boundaries-20261023`，
日志`outputs/qwen3-8b-npu/logs/retained_boundaries_{full,resume}_20261023.log`。
full/resume均exit0，本次未复现历史失败，不能修改正式失败状态。

实际每rank5事件（两层self_attn、两层mlp、model），四rank共20边界逐位一致，
maxAE0；两阶段所有记录的版本计数从hook到最终均未变化。tok_embeddings
不是注册名实际后缀，因此未捕获，不能宣称embedding边界已检查。首个离线
比较错误假定6事件而assert失败，纠正为实际非空配对后审计exit0，比较结果
`outputs/qwen3-8b-npu/logs/retained_boundaries_comparison_20261023.json`。
版本不变仅排除该次被版本计数跟踪的修改，不证明所有底层异步写入不存在；
本次成功也不能证明上次失败logits保留值未受覆盖。旧logits-only失败产物及
正式fullAC失败完整保留，本轮不与它们拼接推断首次分歧位置。

下一步不重复这组成功引用探针；应针对已保存失败的算子输入及布局做独立
重放，或使用不同证据来源验证缓冲生命周期。所有NPU和审计任务已终止。
本轮仅改忽略目录诊断及状态文档，诊断Ruff/format、git diff --check通过；
未改生产实现，不冒充CPU套件重跑。整体目标仍未完成。

## 2026-10-23：保留logits引用的同次失败前向证据

检查确定性设置已在Trainer进程组创建前执行，现有MoE combine也使用
deterministic_scatter_add；没有据此断言HCCL或算子确定性已获证明。
本轮改用忽略目录探针`outputs/probe_failed_logits_reference.py`：仅step3
原始forward保存模型输出detach引用，不在hook内clone、转CPU或同步，最终
检查结束后才落盘。保留引用仍可能影响内存分配，不是完全无扰动探针；
未替换任何生产算子/通信或禁用正式AC。未改生产源码和正式测试。

沿用上一节原NPU4/TP2/EP2/KVCP2/native/FP32/fullAC/EMA/RNG/loader/RoPE
精确配置，full/resume使用独立进程；将上一节正式命令入口换成
`outputs/probe_failed_logits_reference.py`，产物目录改为
`outputs/tp-ep-cp-retained-logits-20261023`，日志为
`outputs/qwen3-8b-npu/logs/retained_logits_{full,resume}_20261023.log`。
探针调用的仍是正式`pp_checkpoint_equivalence.py`全部断言，未放宽容差。

full退出0、resume退出1，四rank恢复前（含RoPE）failures0；step3连续loss
4.8652191162109375、恢复4.865217685699463，AE1.431e-6，失败实际保留。
每rank实际只有一个step3原forward输出shape512×128，非重计算输出。
两阶段对应保留logits均有限，但四rank均非精确一致：

| rank | logits最大AE | 不同元素数/65536 |
| --- | --- | --- |
| 0 | 0.0360223725438118 | 63739 |
| 1 | 0.0005849003791809082 | 65525 |
| 2 | 0.0003761202096939087 | 65516 |
| 3 | 0.0003298242809250951 | 65516 |

比较产物`outputs/qwen3-8b-npu/logs/retained_logits_comparison_20261023.json`，
原始tensor为产物目录`retained-logits-{full,resume}-rankN.pt`，使用
torch.equal、绝对差最大值及count_nonzero逐rank检查。比较任务退出0仅表示
审计完成，正式恢复仍失败。此同次运行证据将排查重点前移至step3前向；
不能把小loss AE当所有logits近似一致，也不能拼接其他成功探针认定输入或
内部边界相同。由于引用延后落盘，进一步定位须检查相关算子是否会修改
输出底层存储，避免不加限定地把落盘值当hook时刻的完整快照。

下一动作：保留同样低扰动方式缩小embedding/attention/MoE/logits边界，
最多有限尝试，取得同次失败边界再做局部重放；不重复旧的全模块clone探针。
本轮两组NPU及离线比较均已退出。诊断脚本Ruff/format、git diff --check通过；
无正式源码变更，不冒充重跑CPU全套。整体目标仍未完成。

## 2026-10-23：新增正式失败checkpoint的noAC诊断对照

本轮读取原始目标及开发指南，继续上节同一已保存失败checkpoint，不重新
生成连续参考，也不覆盖原始日志/状态。仅在独立resume进程设置AC=none，
其余原TP2/EP2/KVCP2、FP32、native、EMA/RNG/loader/RoPE状态断言保持。
这不是正式配置通过证据，更不是禁用AC的生产修复。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1
export LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_CP_SIZE=2
export LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_MOE=1
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=none
export LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1
export LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_TEST_ROTARY_STATE=1
timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_checkpoint_equivalence.py resume outputs/tp-ep-cp-formal-rotary-state-20261023 > outputs/qwen3-8b-npu/logs/formal_rotary_failed_noac_resume_20261023.log 2>&1
'
```

诊断退出1，四rank恢复前failures0；step3/4 loss分别为
4.865217685699463/4.867334365844727，与同checkpoint的fullAC恢复打印值完全
相同，均偏离连续参考4.8652191162109375/4.867324352264404。
因此该案例失败不以AC重计算为必要条件；仅loss标量相等不能证明两次恢复的
全部中间梯度/更新相等。不能由此排除连续训练阶段的AC或其他运行时影响，
也不能外推所有历史失败共享同一根因。RoPE状态同次检查继续精确。

本轮仅一组新NPU诊断、无源码/测试变更，作业已退出；git diff --check通过。
原fullAC正式验收仍失败，无阈值变更或静默回退。下一动作应检查不依赖AC
重放的首次前向差异（含布局与通信），不重复本checkpoint禁用AC对照。
整体目标仍未完成，其他独立声明映射及硬件/compile/P4缺口保持。

## 2026-10-23：正式恢复测试补齐RoPE运行态，并在同次失败中排除该差异

原_runtime_state只取model.state_dict，不能证明nonpersistent RoPE一致。
正式`pp_checkpoint_equivalence.py`新增可选`LLMTUNER_TEST_ROTARY_STATE=1`，
逐模块保存inv_freq、original_inv_freq及attention_scaling（tensor复制到CPU），
断言非空；split恢复前和final继续训练后均进入原零容差递归比较。不是保存
或恢复新生产状态，不修改训练算法，不比较设计上需清空的aux统计缓冲。

本次原配置NPU4、TP2/EP2/KVCP2、native FP32、非packed256小MoE、fullAC、
dynamic EMA/RNG/loader，连续4步vs独立进程2+2，原精确断言全部保留：

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1
export LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_CP_SIZE=2
export LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_MOE=1
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=full
export LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1
export LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_TEST_ROTARY_STATE=1
for phase in full resume; do
  timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" outputs/tp-ep-cp-formal-rotary-state-20261023 > outputs/qwen3-8b-npu/logs/formal_rotary_state_${phase}_20261023.log 2>&1
  result=$?; echo "$phase exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
'
```

| 测试项 | 基线/阈值 | 实际结果 | 状态 | 退出码/日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| 原混合并行fullAC精确冷恢复，含RoPE | 同次连续/恢复状态零容差 | full0/resume1；四rank恢复前failures0，包含RoPE；最终RoPE无差异，模型/optimizer/EMA及loss仍分歧 | 失败 | 0/1；`formal_rotary_state_{full,resume}_20261023.log` |

同一次失败step3 loss恢复4.865217685699463、连续4.8652191162109375，AE1.431e-6；
step4恢复4.867334365844727、连续4.867324352264404，AE1.001e-5。四rank保存的
split/final RoPE均实际有inv_freq/original_inv_freq各shape8及scale1.0。
因此这次失败不能归因于被检查的RoPE频率/缩放值差异；不排除其他运行态、
布局或kernel调度，不把本次状态检查外推为所有内部状态相同。
此证据比历史仅成功探针更强：新增覆盖与失败在同一次正式运行共存。
下一步须定位同次失败首次前反向分歧，不再重复成功buffer探针或放宽阈值。

本轮CPU全套4241passed/13skipped/273warnings（149.32秒）、exit0，日志
`formal_rotary_state_cpu_20261023.log`。Ruff全仓、449文件format、容器compileall
和diff check通过。刷新快照全量pre-commit仍exit1，仅已有hf_inspection两处
Ruff及karpathy SKILL EOF失败，日志`formal_rotary_state_precommit_20261023.log`，
未回写其自动修改。所有作业已终止，整体目标仍未完成。

## 2026-10-23：HSDP×PP×EP×full/selective AC独立参考

本轮仅运行既有正式脚本，固定world8、PP2/EP2、dense replicate2/shard2、
global batch16、4microbatches/DP、seq32、FP32、tiny Qwen3MoE两层hidden64、
seed42、aux.001、AdamW、4steps；有限矩阵为两dispatcher×两AC模式。
不继续增加任意交叉轴。独立非流水、非AC参考重建实际初始stage权重及专家，
逐步比较loss、全部梯度和更新参数，保留原tol1e-4。复制placement与实际分片
断言、TorchAO类型检查、每层CheckpointWrapper及重计算调用计数均实际执行。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=.:outputs/torchao-runtime:outputs/baseline-runtime
export LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_GLOBAL_BATCH=16
export LLMTUNER_TEST_DP_REPLICATE_SIZE=2
for dispatcher in alltoall torchao; do
  for mode in full selective; do
    timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_ep_equivalence.py --dispatcher "$dispatcher" --activation-checkpoint "$mode" > outputs/qwen3-8b-npu/logs/hsdp_pp_ep_ac_${dispatcher}_${mode}_20261023.log 2>&1
    result=$?; echo "$dispatcher $mode exit=$result"
    if [ "$result" != 0 ]; then exit "$result"; fi
  done
done
'
```

已完成的AllToAll full/selective与TorchAO full均exit0，decoder_calls=33，
超过预定下限32，所有独立参考断言通过。loss打印均为
4.850498/4.866161/4.868088/4.865897，对应参考
4.850498/4.866162/4.868088/4.865896；不从6位打印估算精确误差汇总。
保留显式CPU/AiCPU回退及格式警告，不声称纯AiCore。

| 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码/日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| HSDP×PP×EP / AllToAll full | 独立4步loss及逐参数梯度/更新，tol1e-4 | 全部断言通过，decoder_calls33 | 通过 | 0；`hsdp_pp_ep_ac_alltoall_full_20261023.log` |
| 同拓扑 / AllToAll selective | 同上 | 全部断言通过，decoder_calls33 | 通过 | 0；`hsdp_pp_ep_ac_alltoall_selective_20261023.log` |
| 同拓扑 / TorchAO full | 同上及dispatcher类型 | 全部断言通过，decoder_calls33 | 通过 | 0；`hsdp_pp_ep_ac_torchao_full_20261023.log` |
| 同拓扑 / TorchAO selective | 同上 | 全部断言通过，decoder_calls33 | 通过 | 0；`hsdp_pp_ep_ac_torchao_selective_20261023.log` |

最后TorchAO selective也正常终止，四组全部完成；运行前npu-smi无其他作业。
本轮无代码/测试变更，仅更新状态文档，git diff --check通过，不冒充重跑
CPU全套或门禁。不把本次前反向/更新通过当作冷恢复通过，原TP/EP/KVCP的
间歇性恢复失败仍开放。整体目标未完成，后续须收敛完整声明覆盖索引和失败
根因，不因这些成功子项宣称P2/P3/P4整体验收通过。

## 2026-10-23：DP4×PP2×EP2的TorchAO独立训练参考

本轮复用正式测试，不改源码/测试，补另一已声明dispatcher，不以AllToAll结果
替代TorchAO。world8、PP2/EP2、global batch16、FP32、tiny Qwen3MoE两层、
seq32、4steps、4microbatches/DP、aux.001、AC none、原tol1e-4；独立非流水
参考逐步比较loss、全部梯度及AdamW后参数，复制/分片和实际dispatcher类型
断言全部保留。版本HEAD仍为c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19。

首轮命令只设置PYTHONPATH=.，未包含既有隔离依赖，装配时因torchao缺失
退出1，未训练；日志`outputs/qwen3-8b-npu/logs/dp_pp_ep_torchao_replica1_20261023.log`。
核对历史环境后使用已离线安装的`outputs/torchao-runtime`（torchao0.18.0）及
`outputs/baseline-runtime`，并实际导入permute_and_pad，未联网安装或mock。
其`torchao/prototype/moe_training/ep/permute.py` SHA256为
`0112cc1b7742ac7c729af6f643bbe199aae872013fb4fe90447f5f2bfbfdfb42`。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=.:outputs/torchao-runtime:outputs/baseline-runtime
export LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_GLOBAL_BATCH=16
for replica in 1 2; do
  export LLMTUNER_TEST_DP_REPLICATE_SIZE=$replica
  timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_ep_equivalence.py --dispatcher torchao > outputs/qwen3-8b-npu/logs/dp_pp_ep_torchao_runtime_replica${replica}_20261023.log 2>&1
  result=$?; echo "replica=$replica exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
'
```

shard4/replicate1已exit0，全部独立参考断言通过；实际loss打印
4.850498/4.866161/4.868088/4.865897，参考
4.850498/4.866162/4.868088/4.865896。日志没有误差汇总，不从打印精度
推导精确maxAE。不外推FP8/MXFP8 grouped GEMM、AC、恢复或大模型。
保留CPU/AiCPU及内部格式警告，不声称纯AiCore。

| 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码/日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| world8 PP2/EP2/shard4/replicate1 TorchAO | 独立4步loss/全部梯度/参数，tol1e-4 | 全部断言通过 | 通过 | 0；`dp_pp_ep_torchao_runtime_replica1_20261023.log` |
| world8 PP2/EP2/shard2/replicate2 TorchAO | 同上及实际复制placement | 全部断言通过，loss打印同上 | 通过 | 0；`dp_pp_ep_torchao_runtime_replica2_20261023.log` |

两组运行已终止，未改生产代码或测试，不冒充重跑CPU套件或全量门禁；本轮
git diff --check通过。整体目标仍未完成，AC组合、完整声明映射、历史恢复
失败及硬件/compile/P4缺口保持开放。

## 2026-10-23：DP4×PP2×EP2独立Trainer训练参考

扩展正式`pp_ep_equivalence.py`支持world8、可配置global batch及DP复制度。
原world4/batch8默认不变。参考按world/PP计算DP微批次边界，收集实际PP/EP
坐标选择代表rank，并精确比较重复初始state；每步对重复梯度/参数按原tol
比较，再按EP顺序合并专家，避免八卡时把重复专家错误拼成16个。
复制轴逐参数断言DTensor mesh大小及Replicate placement；分片轴检查局部
参数小于全局存储。生产Trainer与独立非流水参考均4步、AdamW、tiny Qwen3MoE
两层hidden64、FP32、seq32、seed42、global batch16、4microbatches/DP、
aux coefficient .001、AllToAll、AC none，原tol1e-4不变。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_GLOBAL_BATCH=16
for replica in 1 2; do
  export LLMTUNER_TEST_DP_REPLICATE_SIZE=$replica
  timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_ep_equivalence.py > outputs/qwen3-8b-npu/logs/dp_pp_ep_reference_replica${replica}_20261023.log 2>&1
  result=$?; echo "replica=$replica exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
'
```

| 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码/日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| PP2/EP2/dense shard4/replicate1，world8 | 独立4步loss及每步全部梯度/参数，tol1e-4 | 所有rank断言通过；loss打印4.850498/4.866161/4.868088/4.865897，参考4.850498/4.866162/4.868088/4.865896 | 通过 | 0；`dp_pp_ep_reference_replica1_20261023.log` |
| PP2/EP2/dense shard2/replicate2，world8 | 同上，实际复制与分片断言 | 全部断言通过；loss打印与上行相同 | 通过 | 0；`dp_pp_ep_reference_replica2_20261023.log` |

日志未输出误差汇总，不从6位loss打印推导精确maxAE；通过依据为逐项assert_close
和全rank verdict。此处EP共享dense DP空间，不额外乘卡数；不外推TorchAO、
AC、恢复或大模型。未改生产源码，未放宽阈值，保留CPU/AiCPU回退警告。
全仓Ruff、449文件format及git diff --check通过。刷新隔离快照全量pre-commit
退出1，仍仅已有hf_inspection两处Ruff及karpathy SKILL EOF失败，未回写；
日志`dp_pp_ep_reference_precommit_20261023.log`。
默认world4/batch8回归exit0，所有原断言通过，日志
`dp_pp_ep_reference_default_20261023.log`；容器内命令为
`PYTHONPATH=. timeout 300 torchrun --standalone --nproc_per_node=4
tests/integration_tests/pp_ep_equivalence.py`，不设置其余测试环境变量。
CPU全套4241passed、13skipped、273warnings（151.52秒），exit0，日志
`dp_pp_ep_reference_cpu_20261023.log`；容器compileall通过，所有作业已退出。
整体目标仍未完成，完整声明映射、历史恢复失败及硬件/compile/P4缺口保留。

## 2026-10-23：DP2×TP4×EP2全部参数前反向参考

扩展正式`ep_fsdp_without_dense_dp.py`的`LLMTUNER_TEST_DP_MODE`，默认none
保持四rank；replicate/shard两模式使用八rank，TP4、EP2共享rank空间、CP1。
独立完整HF每DP用seed7+dp_rank生成不同32token输入/标签，先计算参考梯度，
断言DP间梯度差>1e-3，再取DP均值。生产loss除32×DP，真实FSDP及TP复制梯度
归约保持，不手工归约生产梯度。原atol1e-5/rtol1e-4不变；12项dense与三类
expert梯度逐项比较。复制模式检查每参数实际Replicate轴size2，expert mesh
size由2增至4；分片模式配置及生产装配日志为dp_shard2，但尚未新增dense
局部存储缩小断言，故不单凭本次结果宣称分片非空泛证明完整。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=.
for mode in replicate shard none; do
  export LLMTUNER_TEST_DP_MODE=$mode
  world=8; if [ "$mode" = none ]; then world=4; fi
  timeout 300 torchrun --standalone --nproc_per_node=$world tests/integration_tests/ep_fsdp_without_dense_dp.py > outputs/qwen3-8b-npu/logs/dp_tp_ep_gradients_final_${mode}_20261023.log 2>&1
  result=$?; echo "$mode exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
'
```

| 配置 | 基线/阈值 | logits/expert梯度/dense梯度最大AE | 状态 | 退出码/日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| DP复制2×TP4×EP2，world8 | 独立HF及DP平均梯度，atol1e-5/rtol1e-4 | 1.192e-7 / 4.657e-10 / 2.235e-8 | 通过（前反向） | 0；`dp_tp_ep_gradients_final_replicate_20261023.log` |
| DP分片2×TP4×EP2，world8 | 同上 | 1.192e-7 / 5.821e-10 / 2.235e-8 | 通过（数值子项） | 0；`dp_tp_ep_gradients_final_shard_20261023.log` |
| 原无DP回归，world4 | 原HF参考 | 1.192e-7 / 9.313e-10 / 2.980e-8 | 通过 | 0；`dp_tp_ep_gradients_final_none_20261023.log` |

首次补丁因上下文不匹配未应用，随后误启动旧脚本world8，world断言退出1；
日志`dp_tp_ep_gradients_replicate_20261023.log`保留，不是生产功能失败。
最终脚本三组均完成；不证明多步Trainer、optimizer或恢复。运行前NPU无其他
作业；保留格式/回退警告，不宣称纯AiCore。全仓Ruff、449文件format、
容器compileall与git diff --check通过。整体目标未完成，分片存储断言、
历史恢复失败及其余声明/硬件/compile/P4缺口继续保留。

随后补上分片模式embedding DTensor局部存储小于全局的正式断言，并独立复验
该模式exit0：logits1.192e-7、expert梯度5.821e-10、dense梯度1.490e-8，
日志`dp_tp_ep_gradients_shard_asserted_20261023.log`。复现为上方命令只运行
mode=shard，输出改为该日志；因此本节此前“尚缺分片存储断言”已关闭。
CPU全套4241passed/13skipped/273warnings，147.08秒，exit0，日志
`dp_tp_ep_gradients_cpu_20261023.log`；最后新增断言由该NPU复验覆盖。
全量pre-commit退出1，仅已有hf_inspection两处Ruff及karpathy SKILL EOF，
日志`dp_tp_ep_gradients_precommit_20261023.log`；该门禁在最后新增分片断言前
执行，不冒充最终文件全量hook复验。未回写其无关自动修改，所有作业均已退出。

## 2026-10-23：TP4×EP2专家FSDP的全部参数梯度参考

正式扩展`tests/integration_tests/ep_fsdp_without_dense_dp.py`，保留原三类
专家梯度检查，再加入所有非专家参数：router名称显式映射到HF gate，Q/K/V
按TP行切片，O按TP列切片，其余比较完整梯度。使用生产
`Trainer._allreduce_replicated_tp_grads`完成复制参数归约，未手写替代训练归约。
断言dense数量等于HF全部参数项减去两项融合expert参数，本fixture为12项；
原atol1e-5/rtol1e-4不变，专家梯度仍非零，assert_close拒绝NaN/Inf。

配置world4、TP4/EP2/efsdp2、dense DP/CP均1、tiny Qwen3MoE单层hidden64、
FP32、32tokens、同初始权重及输入，单次前反向。当前HEAD
`c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19`及现有改动工作区；不切分支。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
PYTHONPATH=. timeout 240 torchrun --standalone --nproc_per_node=4 tests/integration_tests/ep_fsdp_without_dense_dp.py > outputs/qwen3-8b-npu/logs/tp_ep_all_gradients_20261023.log 2>&1
'
```

| 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码/日志 |
| --- | --- | --- | --- | --- |
| TP4×EP2×expert-FSDP2全部参数前反向 | 独立完整HF，atol1e-5/rtol1e-4 | logits AE8.941e-8、三类expert梯度9.313e-10、12项dense梯度2.235e-8；所有rank断言通过 | 通过 | 0；`outputs/qwen3-8b-npu/logs/tp_ep_all_gradients_20261023.log` |

此证据关闭上一节“全部dense梯度未比较”的明确缺项，不证明DP复制/分片
组合、optimizer多步、完整Trainer或checkpoint恢复。保留NPU内部格式及回退
警告，不宣称纯AiCore。全仓Ruff check、449文件format及git diff --check通过。
刷新隔离快照全量pre-commit退出1，仍仅已有hf_inspection两处Ruff及karpathy
SKILL EOF失败，未回写；日志
`outputs/qwen3-8b-npu/logs/tp_ep_all_gradients_precommit_20261023.log`。
本轮CPU全套4241passed、13skipped、273warnings，151.29秒、退出0，日志
`outputs/qwen3-8b-npu/logs/tp_ep_all_gradients_cpu_20261023.log`；容器compileall
通过。所有本轮进程均已退出。
整体目标仍未完成，历史恢复失败及其他声明/硬件/compile/P4缺口保持。

## 2026-10-23：TP×EP声明范围与当前工作区复验

重新读取原始目标后核对TP×EP，不将TP×EP×CP恢复成功替代单独TP×EP正确性。
现有正式脚本`ep_fsdp_without_dense_dp.py`覆盖world4、TP4/EP2/efsdp2、
dense DP=CP=1；EP与TP共享rank空间而非额外乘法轴。真实生产装配、HF独立
参考前反向，断言专家参数为size2 mesh的DTensor，三类专家权重梯度均非零
且与完整HF参考切片满足atol1e-5/rtol1e-4。它不比较所有dense参数梯度，
不验证多步optimizer或完整Trainer，不能扩大为这些范围的通过。

本轮当前工作区实际复验（启动前本机NPU无作业）：

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
PYTHONPATH=. timeout 240 torchrun --standalone --nproc_per_node=4 tests/integration_tests/ep_fsdp_without_dense_dp.py > outputs/qwen3-8b-npu/logs/tp_ep_scope_recheck_20261023.log 2>&1
'
```

| 测试项 | 基线/阈值 | 实际结果 | 状态 | 退出码/证据 |
| --- | --- | --- | --- | --- |
| TP4×EP2×expert-FSDP2前反向 | 独立HF logits及三类expert梯度，atol1e-5/rtol1e-4 | logits最大AE8.941e-8；expert梯度9.313e-10；所有rank断言通过 | 通过（指定子项） | 0；`outputs/qwen3-8b-npu/logs/tp_ep_scope_recheck_20261023.log` |

另重新读取既有`tp_ep_cp_ac_spmd_fixed_{full,selective}_20261023.log`，均包含
四rank第三步27项gradient/update/momentum通过及EXIT_CODE=0；其范围是
TP2/EP2/KVCP2、aux.01、padding、uneven累积2、模型/SGD循环，不是完整Trainer
独立训练等价。`tp_ep_cp_trainer_noac_cold_{full,resume}_20261023.log`均
EXIT_CODE=0、resume maxAE0，只证明无AC配置真实Trainer精确恢复。
fullAC历史间歇恢复失败仍失败；不拼接这三个来源作同一次完整训练通过。

TP×EP的上述必测前反向子项已有当前证据；dense DP复制/分片与该组合的具体
有限配置、全部dense梯度及多步独立Trainer参考不能由这条日志推断，继续列为
覆盖待补。下一步应扩展既有独立参考检查范围，而非重新反复跑同一成功用例。
本轮未改源码/测试，作业已退出，git diff --check通过；保留显式CPU/AiCPU
fallback及内部格式警告，不宣称纯AiCore。整体目标仍未完成。

## 2026-10-23：HSDP2×2×PP2独立训练轨迹

本轮原先检查TP×EP覆盖，现场发现更直接的既定缺项：HSDP×PP此前仅有
validation参考，不足以证明独立训练轨迹。优先完成该缺项，不将validation
成功外推为训练正确。修改正式`pp_equivalence.py`：复制DP逐参数检查实际
DTensor、mesh轴大小及Replicate placement；新增可选
`LLMTUNER_TEST_GLOBAL_BATCH`默认8，原梯度/参数容差1e-5保持。

初始world8/PP2/replicate2/shard2配置因global batch8使每DP仅2行、无法切成
4microbatches，在装配保护退出1，未开始训练。日志
`pp_hsdp_training_1F1B_20261023.log`及
`pp_hsdp_training_final_1F1B_20261023.log`保留；两次均是同一batch配置拒绝，
不是生产训练失败。新增断言曾使用不存在的配置属性，运行未到达该断言；
静态核对后已改为`data_parallel_replicate_size`。不把此过程误报为已观察的
AttributeError。随后显式预定batch16满足保护，不绕过校验。

最终配置：world8、PP2/replicate2/shard2、CP/TP/EP1、native、FP32、seq32、
4steps、4microbatches、global batch16、seed42、deterministic=true、AC none。
两种调度分别2层/6层，独立非流水参考重建实际stage初始权重，使用相同DP
microbatch边界及AdamW，逐步比较loss、grad norm、全stage梯度及更新参数。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8
export LLMTUNER_TEST_DP_REPLICATE_SIZE=2 LLMTUNER_TEST_GLOBAL_BATCH=16
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native
for schedule in 1F1B Interleaved1F1B; do
  timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_equivalence.py "$schedule" > outputs/qwen3-8b-npu/logs/pp_hsdp_training_batch16_${schedule}_20261023.log 2>&1
  result=$?; echo "$schedule exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
'
```

| 配置 | 基线/阈值 | loss/梯度/参数最大AE | 状态 | 退出码/日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| HSDP2×2×PP2 / 1F1B | 独立4步全模型，tol1e-5 | 4.768e-7 / 1.336e-7 / 4.057e-6 | 通过 | 0；`pp_hsdp_training_batch16_1F1B_20261023.log` |
| 同拓扑 / Interleaved1F1B | 同上 | 4.768e-7 / 5.960e-8 / 7.483e-7 | 通过 | 0；`pp_hsdp_training_batch16_Interleaved1F1B_20261023.log` |

默认双卡1F1B回归也exit0，loss/梯度/参数AE分别4.768e-7/3.912e-8/1.192e-7，
日志`pp_hsdp_training_default_regression_20261023.log`。命令为容器内
`PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=2 timeout 240 torchrun --standalone
--nproc_per_node=2 tests/integration_tests/pp_equivalence.py 1F1B`，其余测试
环境变量不设置。所有本轮作业均已退出。
两组failed ranks均0，实际复制placement与分片存储断言通过。不外推AC、
checkpoint恢复、BF16或大模型。CPU全套4241passed/13skipped/273warnings，
146.11秒，exit0，日志`pp_hsdp_training_cpu_20261023.log`；该套件不导入本集成
脚本，其后新增batch环境读取由上述NPU用例验证。全仓Ruff、449文件format、
容器compileall和diff检查通过。刷新最终快照全量pre-commit仍exit1，仅已有
hf_inspection两处Ruff及karpathy SKILL EOF；日志
`pp_hsdp_training_batch16_precommit_20261023.log`，未回写无关自动修改。
整体目标未完成，TP×EP声明映射仍待继续，既有恢复失败和其他缺口不变。

## 2026-10-23：DP复制/分片×双轴CP独立packed前反向参考

补齐上一节覆盖索引明确列出的独立参考缺项，而非再次验证恢复自洽。
正式测试`cp_ulysses_varlen_equivalence.py`新增`LLMTUNER_TEST_DP_MODE`
（none默认/replicate/shard）；DP模式只允许dualCP且TP1，实际world8。
沿用生产apply_cp后apply_fsdp，未替换内核或训练源码。所有参数须为DTensor且
局部存储小于全局；复制模式另检查dp_replicate轴size2及Replicate placement。

预定tiny Qwen3两层/hidden64、FP32、seq256、三文档长度90/1/165、KV2×Ulysses2，
DP2使用不同token/label（第二DP输入平移17取模128），相同初始权重。
独立完整序列参考逐DP计算dense文档mask，再对参考梯度取DP均值；生产损失
按SEQ×DP缩放，FSDP实际归约CP及DP梯度，不手工重复归约生产梯度。
原tol=1e-5不变，所有参数梯度均比较，先断言不同DP参考梯度最大差>1e-3，
防止缺DP归约却因相同输入通过。logits/loss比较各DP自己的完整序列参考；
跨文档attention反例doc_gap=0.6232，通信/varlen反向断言也实际执行。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=. LLMTUNER_TEST_DUAL_CP=1
for mode in replicate shard none; do
  export LLMTUNER_TEST_DP_MODE=$mode
  world=8; if [ "$mode" = none ]; then world=4; fi
  timeout 300 torchrun --standalone --nproc_per_node=$world tests/integration_tests/cp_ulysses_varlen_equivalence.py > outputs/qwen3-8b-npu/logs/dp_dual_cp_reference_final_${mode}_20261023.log 2>&1
  result=$?; echo "$mode exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
'
```

| 配置 | 基线/阈值 | 实际logits/loss/全参数梯度最大AE | 状态 | 退出码/日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| replicate2/shard1/KV2/Ulysses2，world8 | 独立全序列及DP平均梯度，tol1e-5 | 0 / 0 / 1.118e-8 | 通过 | 0；`dp_dual_cp_reference_final_replicate_20261023.log` |
| replicate1/shard2/KV2/Ulysses2，world8 | 同上 | 0 / 4.768e-7 / 1.118e-8 | 通过 | 0；`dp_dual_cp_reference_final_shard_20261023.log` |
| 无DP双轴CP回归，world4 | 原完整序列/梯度参考，tol1e-5 | 0 / 4.768e-7 / 1.676e-8 | 通过 | 0；`dp_dual_cp_reference_final_none_20261023.log` |

加非空泛梯度断言前首轮两组也exit0，日志`dp_dual_cp_reference_{replicate,shard}_20261023.log`
保留，但最终结论以上表为准。本证据验证指定packed前反向，不证明optimizer
多步轨迹、BF16、AC或Qwen3-8B；对应冷恢复证据仍单列，不能拼为同一次运行。
CPU/AiCPU回退警告继续保留，不声称纯AiCore或性能通过。

CPU全套4241passed、13skipped、273warnings，153.44秒、exit0，日志
`outputs/qwen3-8b-npu/logs/dp_dual_cp_reference_cpu_20261023.log`。全仓Ruff、
449文件format、容器compileall及diff check通过。刷新隔离快照全量pre-commit
exit1，仍仅已有`hf_inspection.py`两处Ruff及karpathy SKILL EOF失败，未回写；
日志`outputs/qwen3-8b-npu/logs/dp_dual_cp_reference_precommit_20261023.log`。
本轮所有作业已结束，整体目标仍未完成；完整声明映射、历史间歇恢复失败、
KCP/compile/custom FSDP及P4关键权重标准保持开放。

## 2026-10-23：P2声明到证据的有限覆盖索引（首批核实）

2026-10-24 TP布局断言后双轴CP回归：node92启动前npu-smi确认八卡
无运行进程，session9991终止exit0，all checks passed。TP2/KV2/
Ulysses2、world8、DP/PP/EP1，小Qwen3两层hidden64、FP32 packed256，
文档90/1/165；新增投影类型、本地参数量及按TP rank零容差权重
切片断言全部执行。完整序列logits/loss/全参数梯度最大AE分别
2.384e-7/0/2.049e-8，原tol1e-5，文档边界反例gap0.6232。
日志outputs/qwen3-30b-npu/logs/tp2_dual_cp_packed_layout_20261024.log。
复现沿用本节单轴TP布局命令，devices改为0,1,2,3,4,5,6,7、
nproc_per_node=8、DUAL_CP=1、KV_ONLY=0、KV_PACKED=0，其余环境
不变。与session98616、88368分别构成KV、Ulysses及双轴三条路径
的新布局断言验证，不外推多步optimizer、恢复、BF16或30B。
本轮只修改正式集成测试及本状态文件；完整门禁尚待最终复验。

2026-10-24 TP布局断言后Ulysses复验：node92运行前npu-smi确认八卡
无运行进程，四卡TP2/Ulysses2、KV1、DP/PP/EP1，FP32小Qwen3
两层hidden64、packed256、文档90/1/165。session88368终止exit0，
日志outputs/qwen3-30b-npu/logs/tp2_single_ulysses_packed_layout_20261024.log
包含all checks passed；新增投影类型、本地参数量、按TP rank精确
权重切片断言全部执行，完整序列logits/loss/全参数梯度最大AE分别
2.682e-7/0/2.235e-8，原tol1e-5，文档边界反例gap0.6232。
复现沿用本节单轴Ulysses×TP完整命令，仅将pycache改为
/tmp/llmtuner-tp-layout-proof-20261024。与下方session98616共同
证明两条单轴路径的指定布局及前反向；不外推双轴、多步、恢复或30B。
新增布局断言后的双轴TP回归及完整质量门禁仍待执行。

2026-10-24 TP实际权重切片补证：packed集成脚本在apply_tp后逐一
检查ColumnParallelLinear/RowParallelLinear的tp_size、本地参数量，
并与独立参考模型按对应轴及TP rank切出的权重做零容差比较；
要求至少每层四个投影实际分片。此断言补充原完整序列及梯度比较，
未修改生产实现或数值阈值。node92启动前八卡无运行进程，
session98616四卡TP2/KV2 packed验证终止exit0，all checks passed；
FP32小Qwen3两层hidden64、seq256、文档90/1/165、DP/PP/EP1，
logits/loss/全参数梯度最大AE为2.384e-7/4.768e-7/2.049e-8，
原tol1e-5，文档边界反例gap0.6232。日志为
outputs/qwen3-30b-npu/logs/tp2_single_kv_packed_layout_20261024.log。
复现沿用下方单轴KV×TP完整命令，仅将pycache改为
/tmp/llmtuner-tp-layout-proof-20261024。局部Ruff、format、diff检查
通过；新增布局断言后的Ulysses及双轴TP硬件复验尚未执行，
完整门禁尚未复验，不外推多步训练、恢复或30B。

2026-10-24 单轴CP×TP packed独立参考补验：正式
cp_ulysses_varlen_equivalence.py扩展TP2入口至单轴KV/Ulysses，
均使用实际ParallelTopology构造CP/TP网格，保留原完整序列logits、
loss、全部参数梯度及文档边界反例断言，未修改生产实现或阈值。
node92容器四卡0,1,2,3，每次启动前npu-smi确认无运行进程；
小Qwen3两层hidden64、FP32 packed256，文档长度90/1/165，
DP/PP/EP1、TP2、CP2，原tol1e-5。KV session9158与Ulysses
session29213均终止exit0，all checks passed。logits/loss/全参数
梯度最大AE分别为2.384e-7/4.768e-7/2.014e-8及
2.682e-7/4.768e-7/2.142e-8，文档边界反例gap均0.6232。
附带varlen通信子测仍为Ulysses，不将其当作KV通信独立证明。
日志为outputs/qwen3-30b-npu/logs/tp2_single_kv_packed_20261024.log
及同目录tp2_single_ulysses_packed_20261024.log。仅证明指定前反向，
不外推多步optimizer、恢复、BF16或30B。容器仓库目录复现：

```bash
export PYTHONPYCACHEPREFIX=/tmp/llmtuner-tp-single-cp-20261024 PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_DP_MODE=none LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_DUAL_CP=0
LLMTUNER_TEST_KV_ONLY=1 LLMTUNER_TEST_KV_PACKED=1 timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/cp_ulysses_varlen_equivalence.py
LLMTUNER_TEST_KV_ONLY=0 LLMTUNER_TEST_KV_PACKED=0 timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/cp_ulysses_varlen_equivalence.py
```

本轮测试入口修改后局部Ruff、格式及diff检查通过；完整CPU与
pre-commit尚未对此修改复验，既有失败及整体目标保持开放。

2026-10-24 重计算测试选项修改后完整CPU复验：session63759终止exit0，
4302 passed、13 skipped、298 warnings，171.65秒。node92容器仓库目录
使用PYTHONPATH=.:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime、
ASCEND_RT_VISIBLE_DEVICES=0，执行outputs/qwen3-30b-test-venv/bin/python
-X pycache_prefix=/tmp/llmtuner-recompute-quality-20261024 -m pytest -q
--disable-warnings；日志outputs/qwen3-30b-npu/logs/recompute_full_cpu_20261024.log。
容器内python -m compileall -q llmtuner退出0。CPU套件不执行torchrun
重计算选项，其执行证据仍以此前full/selective NPU日志为准。完整
pre-commit仍为下方session74739的exit1，不记通过；整体目标未完成。

2026-10-24 重计算测试选项修改后静态门禁复验：全仓Ruff check、
format检查458文件和git diff --check通过。当前文件隔离快照
/tmp/llmtuner-current-quality-9g8d47ol执行完整pre-commit，
session74739终止exit1；已检查完整git diff，只有hf_inspection.py
两处tuple isinstance改union，以及.agent/karpathy-guidelines/SKILL.md
末尾换行。未回写无关修改，未记为通过或用户豁免。完整CPU套件尚未
针对新增重计算选项复跑；full/selective实际NPU恢复证据见下方。

2026-10-24 full AC实际重计算及冷恢复补证：session58657中full、resume
均exit0；node92设备0,1运行前npu-smi确认无运行进程。小Qwen3、
FP32 packed256、KV2/Ulysses1、PP/TP/EP1、native FSDP，full AC、
dynamic EMA/RNG/loader；连续4步对比step2独立进程恢复2步。启用
LLMTUNER_TEST_REQUIRE_RECOMPUTE=1，每rank每步两个decoder的
forward/recompute计数均为[[1,1],[1,1]]；恢复前状态failures=0，
最终状态断言全部通过，续训loss最大AE=0。日志保留_assert_async CPU
回退。该证据不证明逐算子策略、性能、30B或原TP/EP/CP间歇失败已解决。
日志位于outputs/qwen3-30b-npu/logs/，文件为
single_kv_packed_full_recompute_{full,resume}_20261024.log。
复现沿用下方selective实际重计算命令，将LLMTUNER_TEST_AC改为full，
PYTHONPYCACHEPREFIX改为/tmp/llmtuner-kv-full-recompute-20261024，
输出目录改为outputs/single-kv-packed-full-recompute-cold-20261024；
其余环境及精确断言不变。本轮未修改生产或测试代码；完整CPU及
pre-commit尚未针对上一轮新增测试选项复验，整体目标保持未完成。

2026-10-24 selective AC实际重计算补证：正式恢复脚本新增显式选项
LLMTUNER_TEST_REQUIRE_RECOMPUTE=1，每步在本地decoder的被包装模块
注册临时forward pre-hook，按in_checkpoint_recompute上下文分别计数；
finally移除hook，要求每个decoder前向及重计算均非零。默认关闭，
不改变历史未启用该选项的诊断方式；不以此证明逐算子缓存策略或性能。
node92两卡单轴KV2/Ulysses1、小Qwen3 FP32 packed256、selective AC、
native FSDP、dynamic EMA/RNG/loader，session28693连续4步及独立
step2恢复2步均exit0，日志均显示all checks passed；每rank每层每步计数
[[1,1],[1,1]]，恢复前状态failures=0，续训loss最大AE=0。
日志为outputs/qwen3-30b-npu/logs/下
single_kv_packed_selective_recompute_{full,resume}_20261024.log。
复现沿用本节selective冷恢复完整命令，额外设置
LLMTUNER_TEST_REQUIRE_RECOMPUTE=1，pycache改为
/tmp/llmtuner-kv-recompute-20261024，输出目录改为
outputs/single-kv-packed-selective-recompute-cold-20261024。
运行前npu-smi确认设备空闲；保留_assert_async CPU回退。
局部Ruff及格式通过；完整质量门禁尚未针对本次测试脚本修改复跑。
此结果不关闭原TP/EP/CP间歇恢复失败，不外推full AC或目标30B。

2026-10-24 DP×单轴KV packed冷恢复：node92四卡0,1,2,3，
replicate2/shard1/KV2与replicate1/shard2/KV2分别完成连续4步、
step2保存、独立进程恢复2步。session30123及86160均full/resume exit0；
各组四rank pre-update state failures=0，最终状态全部既定断言通过，
恢复loss最大AE=0。连续loss均为4.991749286651611、4.528769493103027、
4.273682594299316、4.132388591766357。实际复制placement或真实FSDP
DTensor局部分片断言执行，单轴KV生产内核断言执行。
配置为小Qwen3、FP32、seq256、AC none、PP/TP/EP1、Ulysses1、native
FSDP、dynamic EMA/RNG/loader；不外推AC组合、BF16、30B或跨框架对齐。
每组运行前npu-smi确认无其他NPU进程，保留_assert_async CPU回退记录。
代码版本为c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19加当前工作区改动；
本轮未修改生产或测试代码。日志位于outputs/qwen3-30b-npu/logs/，
分别为dp_replicate_single_kv_packed_{full,resume}_20261024.log及
dp_shard_single_kv_packed_{full,resume}_20261024.log。
复现：node92容器仓库根目录执行以下环境设置，再分别执行两种布局；
每个布局的PHASE依次为full、resume，各自独立启动torchrun。

```bash
export PYTHONPYCACHEPREFIX=/tmp/llmtuner-dp-kv-packed-cold-20261024 PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=1 LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=1 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=none LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
LLMTUNER_TEST_DP_REPLICATE_SIZE=2 LLMTUNER_TEST_REQUIRE_SHARDED=0 timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_checkpoint_equivalence.py PHASE outputs/dp-replicate-single-kv-packed-cold-20261024 --packed
LLMTUNER_TEST_DP_REPLICATE_SIZE=1 LLMTUNER_TEST_REQUIRE_SHARDED=1 timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_checkpoint_equivalence.py PHASE outputs/dp-shard-single-kv-packed-cold-20261024 --packed
```

2026-10-24 单轴KV packed selective AC冷恢复：node92设备0,1运行前
npu-smi确认无运行进程；session91609中full和独立resume均exit0。
小Qwen3、FP32、seq256、KV2/Ulysses1、PP/TP/EP1、native FSDP，
dynamic EMA/RNG/loader，连续4步对比从step2独立恢复2步。两rank
pre-update state failures=0，最终状态及loss全部既定断言通过，loss最大AE=0；
连续loss为4.980159759521484、4.524144172668457、4.270559310913086、
4.143191337585449。实际CheckpointWrapper及CPSdpaKernel断言执行。
此证据仅证明指定小模型恢复一致，未计量重计算次数，不外推至30B或独立
非CP数值参考；日志明确报告_assert_async CPU回退，未声称纯AiCore。
输出目录outputs/single-kv-packed-selective-cold-20261024；日志为
outputs/qwen3-30b-npu/logs/single_kv_packed_selective_{full,resume}_20261024.log。
复现：node92容器仓库目录，依次在独立torchrun进程执行以下命令，
PHASE分别取full、resume（两次均exit0）：

```bash
PYTHONPYCACHEPREFIX=/tmp/llmtuner-kv-packed-selective-20261024 PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 LLMTUNER_TEST_WORLD_SIZE=2 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=1 LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=1 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=selective LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 timeout 300 torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py PHASE outputs/single-kv-packed-selective-cold-20261024 --packed
```

2026-10-24 fullAC恢复测试修改后质量复验：全仓Ruff check通过，
format检查458文件通过。完整pre-commit隔离检查session49166退出1，
副本/tmp/llmtuner-current-quality-c0i6p1qe；已读取完整hooks差异，仍仅
hf_inspection.py两处tuple isinstance转union和karpathy技能EOF换行。
未回写这些无关修改，不记门禁通过或豁免。最新完整CPU套件仍为此前
4302 passed/13 skipped；本轮新增断言属于torchrun集成脚本，已在上述
fullAC连续/恢复两次实际NPU运行执行，未声称重新运行完整CPU套件。

2026-10-24单轴KV packed与full AC恢复通过：正式恢复脚本新增每个本地
decoder均为CheckpointWrapper的断言，以及NPU单轴KV实际使用
CPSdpaKernel、CP_SDPA_BACKEND和正确CP度数的断言；不以配置代替路径检查。
node92两卡、PP/TP/EP1、KV2/Ulysses1、native FSDP、FP32、seq256、
full AC、动态EMA/RNG/loader，连续4步在step2保存，再独立恢复2步。
full session58605与resume session8908均退出0；两rank加载后
pre-update state failures=0，最终模型/optimizer/scheduler/trainer/EMA/
RNG/loader等原精确状态断言通过，续训loss最大AE=0。连续loss为
4.980159759521484、4.524144172668457、4.270559310913086、4.143191337585449。
日志为outputs/qwen3-30b-npu/logs/下
single_kv_packed_fullac_full_20261024.log和single_kv_packed_fullac_resume_20261024.log。
复现沿用下方AC none完整命令，将LLMTUNER_TEST_AC改为full、
PYTHONPYCACHEPREFIX改为/tmp/llmtuner-kv-packed-fullac-20261024，
输出目录改为outputs/single-kv-packed-fullac-cold-20261024；full和resume
分别独立启动，两次前设备均确认空闲。保留原精确阈值及CPU回退日志。
新增断言证明包装及内核归属，未统计实际重计算次数；不将其解释为
性能或重计算覆盖证明。仅指定小模型恢复通过，不关闭原TP/EP/CP fullAC
间歇失败，也不外推selective AC、DP恢复、BF16或30B。修改仅正式测试
与状态文档；局部Ruff和format通过，完整门禁需在最终修改后另行复验。

2026-10-24单轴KV packed独立进程恢复通过：node92容器两卡，
PP/TP/EP均1、KV2/Ulysses1、native FSDP、FP32、seq256、AC none，
动态EMA及RNG/loader状态启用。正式pp_checkpoint_equivalence.py连续
4步在step2保存，再由独立torchrun恢复2步；full session32739及
resume session36434均退出0。两rank pre-update state failures=0，
最终模型、optimizer、scheduler、trainer、EMA、RNG及dataloader等
脚本状态断言通过。连续loss为4.980159759521484、4.524144172668457、
4.270559310913086、4.143191337585449；恢复后两步精确一致，maxAE=0。
日志为outputs/qwen3-30b-npu/logs/single_kv_packed_cold_full_20261024.log
及同目录single_kv_packed_cold_resume_20261024.log。两阶段前设备空闲。
复现在node92的hyper-parallel-env容器仓库目录执行；PHASE分别取full
和resume，两次独立启动，先确认full成功：

```bash
PYTHONPYCACHEPREFIX=/tmp/llmtuner-kv-packed-resume-20261024 PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 LLMTUNER_TEST_WORLD_SIZE=2 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=1 LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=1 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=none LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 timeout 300 torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py "$PHASE" outputs/single-kv-packed-cold-20261024 --packed
```

使用脚本生成的JSONL/tokenizer及生产NPU packed路径，无CPU attention
monkeypatch。保留_assert_async CPU回退、AdamW for-loop和HCCL gather
转allgather日志。仅证明指定小模型4步与2+2步恢复，不外推AC、DP恢复、
BF16、目标30B或跨框架对齐。本轮无源码修改，整体目标仍未完成。

2026-10-24补验DP×单轴KV packed：四卡replicate2/shard1/KV2与
replicate1/shard2/KV2分别运行，session31537、session28037均退出0，
all checks passed。两组logits/loss/全部参数梯度最大AE均为
0/4.768e-7/1.118e-8，原tol1e-5；文档长度90/1/165，跨文档反例
gap0.6232。实际Replicate placement或DTensor局部分片、不同DP输入及
独立完整模型DP平均梯度断言均保留，不以相同输入制造空泛通过。
日志分别为`outputs/qwen3-30b-npu/logs/dp_replicate_kv_packed_20261024.log`
及`outputs/qwen3-30b-npu/logs/dp_shard_kv_packed_20261024.log`。
两次启动前node92容器npu-smi均确认八卡无运行进程。
复现均在node92的hyper-parallel-env容器仓库根目录执行：

```bash
PYTHONPYCACHEPREFIX=/tmp/llmtuner-kv-packed-npu-20261024 PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_KV_ONLY=1 LLMTUNER_TEST_KV_PACKED=1 LLMTUNER_TEST_DUAL_CP=0 LLMTUNER_TEST_DP_MODE=replicate timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/cp_ulysses_varlen_equivalence.py
PYTHONPYCACHEPREFIX=/tmp/llmtuner-kv-packed-npu-20261024 PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_KV_ONLY=1 LLMTUNER_TEST_KV_PACKED=1 LLMTUNER_TEST_DUAL_CP=0 LLMTUNER_TEST_DP_MODE=shard timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/cp_ulysses_varlen_equivalence.py
```

范围仅tiny Qwen3两层hidden64、FP32、seq256的前反向；附带varlen通信
子测仍为Ulysses，KV结论依据实际KV内核的完整模型输出和全部参数梯度。
不外推优化器多步、恢复、负载均衡、BF16或目标30B。本轮无源码修改，
不重复上一轮已通过的完整CPU门禁；整体P0–P4仍未完成。

2026-10-24本轮最终回归：普通因果KV2（session15273）、Ulysses2 packed
（session21540）、KV2×Ulysses2 packed（session96723）均退出0、
all checks passed。三组logits/loss/全部参数梯度最大AE分别为
0/4.768e-7/1.676e-8、2.570e-7/0/2.980e-8、0/0/1.653e-8，
原tol1e-5未变，packed文档边界反例gap均0.6232。日志分别为
`single_kv_causal_packedfix_regression_20261024.log`、
`ulysses_packed_kvfix_regression_20261024.log`、
`dual_packed_kvfix_regression_20261024.log`，均在30B日志目录下。
复现沿用下述首次KV packed命令及独立缓存：普通KV设置KV_ONLY=1、
KV_PACKED=0、DUAL_CP=0、nproc2/devices0,1；Ulysses设置三开关均0、
nproc2/devices0,1；dual设置KV_ONLY=0、KV_PACKED=0、DUAL_CP=1、
nproc4/devices0,1,2,3；DP_MODE均none。各作业前设备检查无运行进程。
完整CPU套件session81796退出0：4302 passed、13 skipped、273 warnings、
158.69s，日志`outputs/qwen3-30b-npu/logs/kv_packed_full_cpu_20261024.log`；
使用测试venv、标准runtime路径、visible device0和独立pycache目录
`/tmp/llmtuner-kv-packed-current-20261024`。全仓Ruff、format458文件、
容器compileall及diff检查通过。完整pre-commit隔离副本
`/tmp/llmtuner-current-quality-51gt8noy`，session47616退出1；完整diff
确认仍仅hf_inspection.py两处isinstance转换及karpathy SKILL EOF换行，
未回写或豁免。单轴KV packed的负载均衡、DP组合、恢复及目标30B仍未验收，
整体P0–P4目标保持未完成。

2026-10-24单轴KV packed首次NPU参考通过：连续分片装配已启用，
packed与非空load_balancer仍在修改backend之前明确拒绝。node92容器
两卡、KV2/TP1/DP1/PP1/EP1，tiny Qwen3两层hidden64，FP32/seq256，
文档长度90/1/165，原tol1e-5。session81712退出0，logits最大AE=0、
loss AE=4.768e-7、全部参数梯度最大AE=1.653e-8，跨文档反例gap=0.6232，
all checks passed。日志`outputs/qwen3-30b-npu/logs/single_kv_packed_20261024.log`。
独立完整模型参考证明本次前反向；脚本附带varlen通信子测仍为Ulysses，
不将其单独作为KV通信证据。不外推DP组合、多步优化器、恢复、BF16或30B。
复现：node92容器仓库目录，设置
`PYTHONPYCACHEPREFIX=/tmp/llmtuner-kv-packed-npu-20261024`、
`PYTHONPATH=.:outputs:/home/jianzhnie/llmtuner/wheels/tf511:outputs/qwen3-30b-runtime:outputs/baseline-runtime`、
`ASCEND_RT_VISIBLE_DEVICES=0,1 LLMTUNER_TEST_KV_ONLY=1 LLMTUNER_TEST_KV_PACKED=1 LLMTUNER_TEST_DP_MODE=none`，
执行`timeout 300 torchrun --standalone --nproc_per_node=2 tests/integration_tests/cp_ulysses_varlen_equivalence.py`。
启动前npu-smi八卡无进程；保留NPU分配padding及依赖警告。独立缓存CPU
回归session25701退出0：25 passed、39 warnings、20.76s；此前旧测试名与
新源码行号混杂的失败不作为当前源码结论。完整CPU套件及完整质量门禁、
普通因果KV和dual/Ulysses硬件回归仍待执行。下文装配未启用描述为历史。

2026-10-24单轴KV packed实现推进：内核已接受完整四维bool文档mask，
在collective前检查缺失、dtype、维数、序列长度和设备，并按本rank查询
offset切片；非零attention dropout明确拒绝。模型包装层已在分片前生成
完整文档mask并传入KV forward。正式varlen集成脚本新增
`LLMTUNER_TEST_KV_PACKED=1`（要求`LLMTUNER_TEST_KV_ONLY=1`），保留
原普通因果模式和packed文档边界反例。
当前apply_cp仍明确拒绝单轴KV packed装配：负载均衡后的mask重排与NPU
完整模型/梯度验证尚未完成，不把内核或CPU测试通过当作功能验收。
CPU验证：node92容器，标准tf511/30b/baseline PYTHONPATH及
ASCEND_RT_VISIBLE_DEVICES=0，测试venv执行pytest -q --disable-warnings，
范围为test_hf_wrapper.py、test_cp_sdpa_kernel.py、test_dual_cp_sdpa.py；
session59165退出0，115 passed、16 warnings、23.28s。新增用例覆盖缺失/
非法mask、两rank查询切片、单token文档、跨文档反例和装配拒绝。
受影响文件Ruff check通过；尚未执行本轮NPU集成、完整CPU套件及完整
pre-commit。此前远程环境缺失判断源于shell引用错误，已用正确引用命令
核实原容器Python/torch正常，无需重建环境。下文构造拒绝描述是修改前历史。

2026-10-24单轴KV packed调用链核实：当前尚未修改生产实现。
`CPSdpaKernel.__init__`在packed=True时拒绝；仅移除该检查不足以支持
packed，因为`HFTransformerModel._shard_for_cp`只为Ulysses或dual SDPA
保留完整dense文档mask，单轴KV分支最终将mask设为None；
`_apply_attention`也仅为dual或Ulysses传递packed mask。
后续实现必须同时覆盖这三个环节，按KV查询偏移切片完整文档mask，
并在collective前拒绝缺失、错误dtype/维数/长度的mask。还须核对
load-balancer产生的token顺序，不能将连续分片的offset规则用于重排输入。
独立参考需包含跨CP边界文档、单token文档及跨文档attention反例，
比较输出、loss和全部参数梯度；既有普通因果KV与dual/Ulysses须回归。
node92本轮npu-smi检查八卡无运行进程；未启动硬件测试，不新增通过结论。

2026-10-24声明复核：README此前将两种CP策略统一描述为支持varlen/packed，
但`parallel/context_parallel/sdpa_attention.py::CPSdpaKernel.__init__`
仍明确拒绝`packed=True`。现将README区分为Ulysses packed、单轴KV普通
因果序列和双轴packed的独立证据范围。此修订纠正文档，不缩减必测范围，
不将拒绝保护计为功能通过；单轴KV packed仍是声明与实现的未关闭缺口，
需实现文档边界mask及独立完整序列/梯度参考后才能验收。

本节核对README“组合”声明与磁盘实际日志，整理覆盖边界，不新造运行结果。
日志前缀统一为`outputs/qwen3-8b-npu/logs/`。历史命令、环境、阈值与退出码
仍以对应执行节为准；本节已重新读取下列日志的实际断言结果。
“通过”仅限表中子项，不证明当前工作区所有历史配置重新复验通过。

| 声明/配置 | 已核实证据及范围 | 状态 | 尚缺证据/下一动作 |
| --- | --- | --- | --- |
| PP2×KV2×Ulysses2×AC，world8 | `pp_dual_cp_ac_{full,selective}_{1F1B,Interleaved1F1B}_20261023.log`四份重新核对，failed ranks=0、all checks passed；FP32 seq256，独立非AC完整模型4步loss/norm/全stage梯度与更新，原tol1e-5 | 通过 | 两AC与两调度分别成立；仅指定非packed小模型，包装断言不证明逐算子缓存策略，不外推30B或恢复 |
| PP2×KV2×Ulysses2 validation，world8 | `pp_dual_cp_validation_{1F1B,Interleaved1F1B}_20261023.log`重新核对均all checks passed；完整CPU参考loss最大AE分别0/4.787e-7，原abs tol1e-5；验证前后optimizer/scheduler/token计数/已有梯度精确保持 | 通过 | 仅指定FP32 seq256、非packed/非AC配置；validation不替代独立训练或恢复 |
| DP分片2×PP2×Ulysses2 validation，world8 | `pp_ulysses_shard_validation_{1F1B,Interleaved1F1B}_20261023.log`重新核对均all checks passed；CPU参考loss最大AE分别4.787e-7/0，原abs tol1e-5，实际Shard及验证状态保持断言 | 通过 | 仅指定FP32 seq256配置；复制DP及HSDP证据另列，不互相外推 |
| DP复制2×单轴KV2 packed，world4 | `outputs/qwen3-30b-npu/logs/dp_replicate_kv_packed_20261024.log`；FP32 packed256，独立完整序列及DP平均梯度，logits/loss/全参数梯度AE为0/4.768e-7/1.118e-8，tol1e-5；另有 `dp_replicate_single_kv_packed_{full,resume}_20261024.log` 的4vs2+2精确恢复 | 通过 | 两类独立证据分别成立；恢复为AC none、dynamic EMA/RNG/loader，不外推BF16、30B或AC组合 |
| DP分片2×单轴KV2 packed，world4 | `outputs/qwen3-30b-npu/logs/dp_shard_kv_packed_20261024.log`；同独立参考与阈值，AE为0/4.768e-7/1.118e-8；另有 `dp_shard_single_kv_packed_{full,resume}_20261024.log` 的4vs2+2精确恢复及真实FSDP分片断言 | 通过 | 本次重新读取参考和恢复日志，均all checks passed；退出码沿用各运行记录，不以读取日志产生新退出码 |
| DP复制2×单轴KV2，world4 | `outputs/qwen3-30b-npu/logs/dp_replicate_single_kv_20261024.log`；FP32因果seq256，独立完整序列/DP平均梯度，logits/loss/梯度AE为0/4.768e-7/8.382e-9，tol1e-5，exit0 | 通过 | 仅前反向；日志旧packed标签已在执行节澄清，不证明packed或恢复 |
| DP分片2×单轴KV2，world4 | `outputs/qwen3-30b-npu/logs/dp_shard_single_kv_20261024.log`；同参考与阈值，logits/loss/梯度AE为0/0/8.382e-9，exit0 | 通过 | 仅前反向；实际DTensor分片，不外推30B或恢复 |
| DP复制2×单轴Ulysses2，world4 | `outputs/qwen3-30b-npu/logs/dp_replicate_single_ulysses_20261024.log`；FP32 packed256，独立完整序列/DP平均梯度，logits/loss/梯度AE为2.570e-7/0/1.490e-8，tol1e-5，exit0 | 通过 | 仅前反向；实际Replicate placement，不外推恢复 |
| DP分片2×单轴Ulysses2，world4 | `outputs/qwen3-30b-npu/logs/dp_shard_single_ulysses_20261024.log`；同参考与阈值，logits/loss/梯度AE为2.570e-7/9.537e-7/1.490e-8，exit0 | 通过 | 仅前反向；文档边界反例与不同DP输入断言生效 |
| DP复制×TP2，world4 | `tp2_dp_replicate2_20261022.log`；4步、loss AE4.768e-7、参数1.478e-6、tol1e-4、failed ranks0 | 通过 | 仅指定native FP32配置，不外推compile/恢复 |
| DP分片×TP2，world4 | `tp2_fsdp2_layout_regression_20261022.log`；同阈值、参数1.478e-6、failed ranks0 | 通过 | 同上 |
| HSDP2×2×TP2，world8 | `tp2_hsdp2x2_20261022.log`；参数2.123e-6、tol1e-4、failed ranks0 | 通过 | 同上 |
| DP×EP2，world4/8 | `ep_hsdp_four_finite_20261022.log` / `ep_hsdp_eight_finite_20261022.log`；dense shard/replicate=2/2及4/2、efsdp1/2，前反向tol1e-4，failed ranks0 | 通过 | EP共享dense DP布局，不按TP式乘积解释；该证据非多步Trainer |
| DP分片4×EP2，world4，TorchAO | `torchao_full_moe_fsdp4_capacity_fixed_20261022.log`；实际dispatcher=torchao、efsdp2，FP32完整MoE前反向，loss实际/参考311.874298/311.874298，tol1e-4，failed ranks0 | 通过 | 独立HF参考及router/专家梯度；非量化grouped GEMM、多步Trainer或恢复 |
| HSDP shard2/replicate2×EP2，world4，TorchAO | `torchao_full_moe_hsdp4_capacity_fixed_20261022.log`；实际dispatcher=torchao、efsdp1，loss实际/参考311.874268/311.874290，tol1e-4，failed ranks0 | 通过 | 实际复制placement；仅指定FP32完整MoE前反向，不外推其他布局或30B |
| DP分片×PP2，world4 | `pp2_native_fsdp2_{1F1B,Interleaved1F1B}_asserted_20261022.log`；4步梯度/更新tol1e-5，failed ranks0 | 通过 | custom仍因torch2.14 API缺失blocked，不能由native替代 |
| DP复制2×PP2，world4，1F1B | `pp2_dp_replicate2_20261022.log`；native FP32、seq32/batch8、4步，独立完整模型参考，loss/梯度/参数AE为0/3.725e-8/7.339e-7，tol1e-5，failed ranks0 | 通过 | 纯复制DP、CP1；仅指定训练参考，不外推AC、恢复或30B |
| DP复制2×PP2，world4，Interleaved1F1B | `outputs/qwen3-30b-npu/logs/pp_replica2_interleaved_20261024.log`；同精度、序列、batch、步数及阈值，loss/梯度/参数AE为4.768e-7/9.872e-8/2.119e-7，failed ranks0 | 通过 | 实际Replicate placement与阶段归属断言；不借用CP/HSDP证据替代，不外推恢复 |
| DP复制×PP2×KVCP2，world8 | `pp2_cp2_dp_replicate2_group_reuse{,_interleaved}_20261022.log`；4步独立训练参考，参数AE3.215e-6/1.570e-6，tol1e-5 | 通过 | 仅已测布局；历史通信组构造失败由正式修复闭环，不删失败日志 |
| DP复制×PP2×Ulysses2×AC，world8 | `dp_replica_pp_ulysses_ac_full_1F1B_20261023.log`；4步独立参考、实际mesh/wrapper，参数AE8.792e-7、tol1e-5、failed ranks0；其余模式见执行节 | 通过 | 不推断packed或恢复 |
| HSDP2×2×PP2独立训练，world8 | `pp_hsdp_training_batch16_{1F1B,Interleaved1F1B}_20261023.log`；4步、batch16，loss AE均4.768e-7，梯度AE1.336e-7/5.960e-8，参数AE4.057e-6/7.483e-7，tol1e-5、failed ranks0 | 通过 | 仅native FP32、seq32、AC none，不外推恢复或30B |
| HSDP2×2×PP2 validation，world8 | `pp_hsdp_validation_{1F1B,Interleaved1F1B}_20261023.log`；CPU loss AE4.922e-7，状态精确保持，实际双DP轴 | 通过 | 独立训练参考另见上一行，不把validation作为其替代 |
| DP复制/分片×双轴CP，world8 | 已有4vs2+2精确冷恢复；后续本文件顶部独立packed参考补验两配置全参数梯度AE1.118e-8、tol1e-5 | 通过（指定前反向/恢复子项） | 两类证据独立，不证明所有训练特性组合 |
| TP4×EP2×expert-FSDP2，world4 | `tp_ep_all_gradients_20261023.log`；logits AE8.941e-8，expert梯度9.313e-10，12项dense梯度2.235e-8；atol1e-5/rtol1e-4 | 通过 | 指定独立HF前反向，不代表多步Trainer或恢复 |
| TP4×EP2三步独立更新，world4 | `tp_ep_actual_optimizer_none_20261023.log`；独立HF实际SGD更新，三步logits/全部梯度/权重，atol1e-5/rtol1e-4；三次all checks passed | 通过 | FP32 tiny MoE，非Trainer/AdamW/恢复；完整配置及误差见顶部复核节 |
| DP分片2×TP4×EP2三步更新，world8 | `tp_ep_actual_optimizer_shard_20261023.log`；同独立HF/SGD参考及阈值，三次all checks passed | 通过 | 不同DP输入及非空泛参考断言；不外推30B或恢复 |
| DP复制2×TP4×EP2三步更新，world8 | `tp_ep_actual_optimizer_replicate_20261023.log`；同独立HF/SGD参考及阈值，三次all checks passed | 通过 | 实际复制placement；不外推30B或恢复 |
| DP复制2×TP4×EP2，world8 | `dp_tp_ep_gradients_final_replicate_20261023.log`；logits/expert/dense梯度AE1.192e-7/4.657e-10/2.235e-8，12项dense，实际复制placement | 通过 | 同上；不同DP输入及参考梯度差异断言防空泛 |
| DP分片2×TP4×EP2，world8 | `dp_tp_ep_gradients_shard_asserted_20261023.log`；logits/expert/dense梯度AE1.192e-7/5.821e-10/1.490e-8，实际局部分片断言 | 通过 | 同上；使用补充分片断言后的独立复验，非较早final_shard日志 |
| TP2×KV2×Ulysses2，world8 | `tp2_dual_cp_sdpa_packed_model_20261023.log`；FP32 packed256，logits/loss/全参数梯度AE2.384e-7/9.537e-7/2.049e-8，tol1e-5，文档边界反例及通信反向 | 通过 | 指定前反向；模型/SGD与真实Trainer恢复证据各自单列 |
| HSDP2×2×PP2×EP2×AC，world8 | `hsdp_pp_ep_ac_{alltoall,torchao}_{full,selective}_20261023.log`四份；独立4步loss/全部梯度/更新，tol1e-4，实际dispatcher、复制/分片和重计算断言，decoder_calls33 | 通过 | 两dispatcher×两AC全部映射；不外推恢复、BF16或30B |
| TP2×EP2×KVCP2 full AC精确恢复 | 已保存失败对；详见运行时buffer与恢复诊断节 | 失败 | 原阈值不变；局部算子重放或单次成功不关闭该失败 |

这是覆盖索引，不是缩减后的验收矩阵。上述已映射CP×TP、TP×EP共享mesh及
PP×EP的有限子项；PP×AC、PP×validation、单轴/双轴CP以及DP各路径仍须全部映射到各自配置、
正确性和恢复断言；P3特性及P4独立保留。完整组合矩阵总行维持未跑，避免
把“有日志”或“冷恢复一致”当作所有正确性层面的通过。

本次追加八份实际日志核对：TP×EP三份、TP×双轴CP一份、HSDP×PP×EP四份，
均重新读取运行摘要与all checks passed，并核对对应正式脚本的断言范围。
历史执行节保留完整命令、退出码及环境，本轮不新造运行或把单步前反向扩大为
多步训练。原索引中的AllToAll/selective“仍须映射”已由四份AC证据补齐；
不关闭TP/EP/KVCP fullAC间歇恢复失败或整模型compile失败。只修改状态文档，
git diff --check通过，无新增NPU进程。

后续复核：原索引把HSDP×PP训练仍列为validation之外的缺项，实际已由
本文件“HSDP2×2×PP2独立训练轨迹”节补验。本轮重新读取两份batch16日志，
逐一确认4steps、tol1e-5、上述误差、failed ranks0及all checks passed，
并核对正式脚本仍检查实际Replicate placement和FSDP局部分片。故补入独立
训练行，不重复启动NPU作业；旧batch8装配拒绝日志保留，不当作训练失败覆盖
新结果。设备检查node92八卡无运行进程；本轮仅修正证据索引，未新增数值结论。

当时明确缺项DP复制/分片×双轴CP的独立参考已由后续补验关闭，见本文件顶部；
仍需完成其余声明映射，不继续扩大任意交叉积。硬件KCP、compile、custom依赖
和关键权重标准不因本索引豁免。
本轮仅更新唯一状态文件，git diff --check通过；未启动NPU作业或重跑门禁。

## 2026-10-23：HSDP×PP的validation独立参考补验

本轮无源码/正式测试修改，使用上一轮最终脚本验证复制与分片轴同时存在的
独立配置。TorchLLMTuner HEAD `c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19`，
基线HEAD `b358a8063c34aa7494397efa3a8f5ad24f1caa1f`，当前有改动工作区。
运行前npu-smi无作业，未干扰其他节点。

预定world8、PP2×dp_replicate2×dp_shard2、CP/TP/EP均1，FP32、seq32、
seed42、deterministic=true。1F1B为2层/2microbatches/global batch16；
Interleaved1F1B为6层/4microbatches/global batch32；均训练2步，每步验证2批。
CPU完整模型使用当前重建权重和相同样本、microbatch边界，原loss绝对容差1e-5。
实际mesh日志确认pp2/replicate2/shard2；所有参数复制轴Replicate检查、真实
FSDPModule/Shard及局部存储小于全局检查均执行通过，不从并行度乘积推断。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8
export LLMTUNER_TEST_DP_REPLICATE_SIZE=2 LLMTUNER_TEST_CP_SIZE=1
export LLMTUNER_TEST_ULYSSES_SIZE=1 LLMTUNER_TEST_SEQ_LEN=32
for schedule in 1F1B Interleaved1F1B; do
  export LLMTUNER_TEST_PP_SCHEDULE=$schedule
  timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_validation_equivalence.py > outputs/qwen3-8b-npu/logs/pp_hsdp_validation_${schedule}_20261023.log 2>&1
  result=$?; echo "$schedule exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
'
```

| 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码/日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| HSDP2×2×PP2 / 1F1B | CPU loss abs≤1e-5；validation前后状态精确一致 | maxAE4.922e-7；所有断言通过 | 通过 | 0；`pp_hsdp_validation_1F1B_20261023.log` |
| 同拓扑 / Interleaved1F1B | 同上 | maxAE4.922e-7；所有断言通过 | 通过 | 0；`pp_hsdp_validation_Interleaved1F1B_20261023.log` |

optimizer、scheduler、训练token计数和已有梯度均精确保持，模型恢复train模式；
重建CPU参考时也对重复stage权重做零容差比较。日志保留NPU算子CPU fallback
警告，不声称纯AiCore。此证据仅支持指定validation配置，不替代独立训练轨迹、
checkpoint恢复或大模型验收。两组作业均已退出，本轮仅更新本状态文档，
git diff --check通过；不冒充重新执行上一轮CPU全套或全量门禁。
整体目标仍未完成，继续保留完整矩阵、间歇性恢复失败、P4权重标准和
KCP/compile/custom FSDP缺口。

## 2026-10-23：复制DP×PP×Ulysses的validation独立参考补验

本轮推进P2/P3缺失的复制DP验证路径，不将此前分片DP结果外推到复制DP。
`pp_validation_equivalence.py`新增`LLMTUNER_TEST_DP_REPLICATE_SIZE`（默认1），
显式检查DP整除，并分别断言实际dp_shard/dp_replicate大小。复制配置逐参数
检查DTensor、dp_replicate mesh轴大小及Replicate placement；分片断言仅在
实际dp_shard>1执行，保持原有真实Shard及局部存储小于全局存储检查。

预定NPU8、PP2×replicate2×Ulysses2、shard1/KV1、FP32、seq256、seed42、
deterministic=true。两种调度各训练2步，每步validation2批，完整CPU参考
使用当前权重、相同样本与microbatch边界；保留原绝对误差1e-5，不改阈值。
optimizer、scheduler、训练token计数及已有梯度在validation前后精确一致，
且模型恢复train模式。此项不验证独立训练轨迹、checkpoint恢复或完整矩阵。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8
export LLMTUNER_TEST_DP_REPLICATE_SIZE=2 LLMTUNER_TEST_CP_SIZE=1
export LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256
for schedule in 1F1B Interleaved1F1B; do
  export LLMTUNER_TEST_PP_SCHEDULE=$schedule
  timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_validation_equivalence.py > outputs/qwen3-8b-npu/logs/pp_ulysses_replica_validation_${schedule}_20261023.log 2>&1
  result=$?; echo "$schedule exit=$result"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
'
```

| 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码/日志（均在outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| replicate2×PP2×Ulysses2 / 1F1B | CPU完整序列loss，abs≤1e-5；状态精确不变 | maxAE4.787e-7，所有断言通过 | 通过 | 0；`pp_ulysses_replica_validation_1F1B_20261023.log` |
| 同拓扑 / Interleaved1F1B | 同上 | maxAE0，所有断言通过 | 通过 | 0；`pp_ulysses_replica_validation_Interleaved1F1B_20261023.log` |
| 默认PP2回归 | 默认参数，原CPU参考和状态断言 | maxAE0 | 通过 | 0；`pp_replica_validation_regression_2_20261023.log` |
| PP2×shard2回归 | world4、replicate1，其余默认 | maxAE4.922e-7，实际Shard断言通过 | 通过 | 0；`pp_replica_validation_regression_4_20261023.log` |

回归命令：容器内`PYTHONPATH=. LLMTUNER_TEST_DP_REPLICATE_SIZE=1
LLMTUNER_TEST_WORLD_SIZE=N timeout 240 torchrun --standalone --nproc_per_node=N
tests/integration_tests/pp_validation_equivalence.py`，N分别为2/4，其余测试环境
变量不设置。运行前npu-smi无运行进程；未改生产训练源码或放宽支持范围。

本轮全CPU套件4241passed、13skipped、273warnings（153.11秒），退出0，
日志`outputs/qwen3-8b-npu/logs/pp_replica_validation_cpu_20261023.log`。
全仓Ruff check、449文件format check、compileall和git diff --check通过。
刷新隔离快照后全量pre-commit退出1，仅既有`hf_inspection.py`两处Ruff修复
和`.agent/karpathy-guidelines/SKILL.md` EOF失败，未回写；日志
`outputs/qwen3-8b-npu/logs/pp_replica_validation_precommit_20261023.log`。
整体目标未完成，既有恢复失败、完整矩阵、P4权重标准及硬件/compile缺口保持。

## 2026-10-22 执行记录

本轮重新建立状态矩阵，不恢复此前删除文件的内容，不继承历史通过结论。

- TorchLLMTuner：`main`，`3cac0266ff297f7162ba9974841c63af43ab6497`。
- torchtitan-npu：`b358a8063c34aa7494397efa3a8f5ad24f1caa1f`。
- 初始工作区：`goal.md`、`.agent/karpathy-guidelines/` 未跟踪；状态文件缺失。
- 本机：`10.1.0.14`；其他所列节点尚未验证，不宣称多节点可用。
- 容器：`hyper-parallel-env`；Python 3.12.13、torch 2.12.0+cpu、
  torch-npu 2.12.0；8 个 NPU 可见且 npu-smi 健康状态 OK。
  Ascend 环境变量指向 `/usr/local/Ascend/cann-9.1.0`。
- 模型、LIMO 数据路径存在。LIMO 字段为 `question` / `solution`。

## 状态矩阵

| 项目 | 配置 / 基线及断言 | 状态 | 命令与证据 |
| --- | --- | --- | --- |
| P0 初次训练 | Qwen3-8B，8 卡 FSDP，10 steps，seq=4096；exit 0、有限 loss、checkpoint | 失败 | `outputs/qwen3-8b-npu/logs/p0_20261022_launcher.log`；exit 1，入口固定读取不存在的 `problem` |
| P0 字段修复后训练 | 同上，显式 `LLMTUNER_PROMPT_FIELD=question` | 通过 | exit 0，完成 10 步及完整 checkpoint；日志 `outputs/qwen3-8b-npu/logs/p0_20261022_retry1.log`；此行仅训练子项通过，resume 单列 |
| P0 resume | deterministic=1，interval=5，10 步连续训练与第 5 步恢复 | 通过 | 双方 exit 0；6434 个 DCP 条目零容差一致；`p0_checkpoint_compare_20261022.log`；step 6–10 loss / grad norm 一致 |
| P1 FSDP2 shard | native，2 卡，4 steps，FP32 tol=1e-4 | 通过 | `p1_20261022.log`：exit 0，最终参数 max AE=0，非空泛梯度差异断言通过 |
| P1 FSDP2 replicate | native，2 卡，4 steps，FP32 tol=1e-4 | 通过 | 同日志：exit 0，最终参数 max AE=0，非空泛断言通过 |
| P1 HSDP | replicate=2 × shard=2，4 卡 NPU，tol=1e-4 | 通过 | `p1_p3_retries_20261022.log`，exit 0，参数 max AE=5.960e-8，非空泛梯度断言通过 |
| P1 FSDP checkpoint | native，shard/replicate 互转恢复；dense atol=1e-5、恢复 atol=0 | 通过 | `p1_20261022.log`，exit 0；custom 引擎子项因版本条件跳过，不计通过 |
| P1 TP | 切分、grad norm | 通过 | `p1_20261022.log`，两脚本 exit 0；logits max AE=1.490e-8，梯度=1.192e-7 |
| P1 PP 1F1B | 前反向、参数更新、真实 packing 语料及有限 validation | 通过 | `p1_20261022.log`，两脚本 exit 0；参数 max AE=1.192e-7 |
| P1 PP Interleaved1F1B | 虚拟 stage 调度，2 卡 NPU | 通过 | `p1_p3_retries_20261022.log`，exit 0 |
| P1 CP 通信 | KV all-gather / Ulysses round-trip，2 卡 NPU/HCCL | 通过 | `p1_20261022.log`：最大误差均为 0；模型路径与 KCP 另验 |
| P1 CP SDPA | NPU 前向及 Q/K/V 梯度，FP32 tol=1e-5 | 通过 | `cp_sdpa_npu_20261022.log`：exit 0；前向/Q 误差 0，K=4.768e-7，V=2.384e-7 |
| P1 CP Ulysses 模型 | MHA/GQA、通信反向、拒绝项与 headtail 反例 | 通过 | `cp_ulysses_retry2_20261022.log`，exit 0；logits AE=0；loss AE≤1.431e-6 |
| P1 CP KCP | KDA 状态并行 | blocked | 已取得上游源码并离线加载运行时；当前内核明确拒绝本机 Ascend910_9392 的 CP 路径，要求 arch3.5；见 2026-10-23 KCP 复核节，拒绝不计功能通过 |
| P1 EP AllToAll / FSDP | 专家分片、路由与梯度 | 通过 | `p1_20261022.log`，两脚本 exit 0；采用脚本既定容差 |
| P1 EP TorchAO padded dispatcher | EP=1/2，NPU FP32，top-k=2，pad=16；双卡随机及偏斜路由、空专家 | 通过 | 真实输出、输入及分数梯度与解析参考一致，atol/rtol=1e-5；`torchao_npu_dispatch_retry_20261022.log`、`torchao_dispatch_ep2_retry1_20261022.log` exit 0；不替代量化 grouped GEMM 端到端验证 |
| P2 完整组合矩阵 | DP 各路径及全部声明组合 | 未跑 | 表示完整覆盖尚未证明，不表示以下子项从未运行；逐配置范围、依赖及证据仍须补齐 |
| P2 TP4×EP2×DP三步更新 | none/world4、shard2/world8、replicate2/world8，FP32 tiny MoE，独立HF/SGD，逐步logits/全部梯度/权重，原atol1e-5/rtol1e-4 | 通过 | `tp_ep_sgd_three_steps_{none_runtime,shard,replicate}_20261023.log`各exit0；顶节完整环境及结果；非Trainer/AdamW/恢复/30B |
| P2 TP×双轴CP | NPU8，TP2×KV2×Ulysses2，FP32 packed256；完整序列参考logits/loss/全参数梯度1e-5，full/selective AC真实Trainer冷恢复精确 | 通过 | `tp2_dual_cp_sdpa_packed_model_20261023.log`及`tp2_dual_cp_sdpa_{selective,fullac}_{full,resume}_20261023.log`，均exit0；范围及完整命令见末尾2026-10-23节 |
| P2 DP复制×双轴CP冷恢复 | NPU8，replicate2/KV2/Ulysses2，native FP32 packed256，full/selective AC、EMA/RNG/loader；实际Replicate与Shard断言，4vs2+2状态及loss精确 | 通过 | `dp_replicate_dual_cp_{selective,fullac}_{full,resume}_20261023.log`与`dp_replicate_dual_cp_selective_resume_shards_20261023.log`均exit0；仅指定恢复配置，不替代全序列参考梯度等价 |
| P2 DP分片×双轴CP冷恢复 | NPU8，shard2/KV2/Ulysses2，native FP32 packed256，full/selective AC、EMA/RNG/loader；实际Shard断言，4vs2+2状态及loss精确 | 通过 | `dp_shard_dual_cp_{selective,full}_{full,resume}_20261023.log`四份exit0；仅指定恢复配置，不替代独立梯度参考 |
| P2 PP×双轴CP独立训练参考 | NPU8，PP2/KV2/Ulysses2、native FP32 seq256，1F1B/Interleaved1F1B、累积2、4步；逐步loss/norm/全stage梯度与更新tol1e-5 | 通过 | `pp_dual_cp_independent_{1F1B,Interleaved1F1B}_20261023.log`均exit0，failed ranks=0；不含packing/AC，不替代P4 |
| P2 PP×双轴CP×AC独立参考 | 同上，full/selective AC各配两种调度；实际wrapper断言，独立非AC整网参考tol1e-5 | 通过 | `pp_dual_cp_ac_{full,selective}_{1F1B,Interleaved1F1B}_20261023.log`四份exit0，failed ranks=0；仅指定非packed模型 |
| P2 DP复制×PP×Ulysses×AC参考 | NPU8，replicate2/PP2/Ulysses2，native FP32 seq256；两调度×full/selective，累积2、4步，逐步及逐参数tol1e-5 | 通过 | `dp_replica_pp_ulysses_ac_{full,selective}_{1F1B,Interleaved1F1B}_20261023.log`四份exit0，failed ranks=0；非packed小模型 |
| P2/P3 PP×双轴CP validation | NPU8，PP2/KV2/Ulysses2、native FP32 seq256，两调度；独立完整CPU参考abs tol1e-5，验证前后训练状态精确保持 | 通过 | `pp_dual_cp_validation_{1F1B,Interleaved1F1B}_20261023.log`均exit0，loss maxAE0/4.787e-7；无packing/AC |
| P2/P3 PP×Ulysses×DP分片 validation | NPU8，PP2/Ulysses2/shard2、native FP32 seq256，两调度；真实分片断言，独立CPU参考abs tol1e-5及训练状态保持 | 通过 | `pp_ulysses_shard_validation_{1F1B,Interleaved1F1B}_20261023.log`均exit0，loss maxAE4.787e-7/0；无packing/AC |
| P2 首批组合子项 | TP×FSDP、CP×TP、TP×EP×CP、PP×EP；各脚本既定配置/断言 | 通过 | `p2_p3_20261022.log` 对应四脚本 PASS；整个批次 8 passed / 3 failed，不作整批通过声明；数值及命令见首批结果节 |
| P2 TP×EP×KVCP 多步 AC/aux | NPU4，TP2/EP2/CP2，native FP32，full/selective AC，padding及不等长累积2 | 通过 | `tp_ep_cp_ac_spmd_fixed_full_20261023.log`、`tp_ep_cp_ac_spmd_fixed_selective_20261023.log`，exit0；3步27项梯度/更新/momentum等满足1e-6；仅指定配置，custom仍blocked |
| P2 TP×EP×KVCP Trainer冷恢复 | NPU4，TP2/EP2/CP2，native FP32，dynamic EMA/RNG/loader，连续4步vs冷启动2+2 | 通过 | 无AC配置，`tp_ep_cp_trainer_noac_cold_full/resume_20261023.log`均exit0，所有启用状态及loss精确；full AC冷恢复另见失败记录 |
| P2 PP×FSDP×validation | NPU4，PP2×native FSDP2，1F1B / Interleaved1F1B；loss tol=1e-5，训练状态保持 | 通过 | `pp_fsdp_validation_grad_restore_20261023.log`，EXIT_CODE=0；loss max AE=4.922e-7；仅指定配置 |
| P3 完整特性矩阵 | AC、compile、EMA、validation、loss、确定性、metrics/profiler、双 checkpoint、packing、多模态 | 未跑 | 完整覆盖尚未证明；已有子项不因此失效，也不得推广到所有模型/组合 |
| P3 metrics精确归约 | NPU4，CP/TP组合、不等token及空DP；独立先SUM后除法参考，原精确比较 | 失败 | metrics_compile_npu_20261022.log明确FAIL/exit1，实际0.7500000596046448对参考0.75；当前归约顺序与torchtitan一致，标准待确认，未改生产公式或阈值 |
| P3 NPU profiler | 实际Trainer连续4步，开启/关闭profiler更新精确一致；Ascend Hardware进程下实际KERNEL事件 | 通过 | profiler_multimodal_final_20261022.log：723个ph=X、dur>0、Task Type=KERNEL_*事件；原1494 flow事件不作为kernel数量，不证明性能 |
| P3 CC12M图像管线 | 本地PNG/JSONL、真实processor/packing、完整loader恢复及NPU往返精确，patch/label非空泛断言 | 通过 | npu_multimodal_cc12m_final_20261023.log，exit0；128-token batch，不证明网络source、视觉模型训练或视频 |
| P3 OBELICS图像管线 | 本地双图交错、真实processor/packing、完整loader恢复及NPU往返精确 | 通过 | npu_multimodal_obelics_final_20261023.log，exit0；256-token batch，双图/grid/patch/IGNORE_INDEX断言；不外推视频或模型训练 |
| P3 vocab-parallel逐token反向 | NPU2 FP32，词表8/7/5，非均匀上游梯度，独立完整词表CE参考，rtol1e-4/atol1e-6 | 通过 | vocab_weighted_backward_20261024.log，exit0；最大梯度AE2.384e-7、忽略行梯度严格0；算子验证不代表lm_head分片模型训练 |
| P3 torch_checkpointing独立进程恢复 | NPU2 native FSDP，full AC/dynamic EMA/RNG/loader，连续4步与step2独立进程恢复2步精确 | 通过 | torchckpt_fsdp_fullac_cold_{full,resume}_retry1_20261024.log，均exit0，加载前差异0、恢复loss AE0；仅指定小模型配置 |
| P3 TP collective Inductor | NPU2，FP32，fullgraph，default/Triton；解析梯度零容差 | 通过 | `tp_inductor_default_backend_20261023.log`；不代表整模型编译通过 |
| P3 PP2 整模型 Inductor | 确定性训练及恢复目标 | 失败 | `pp2_inductor_split_probe_20261023.log`，EXIT_CODE=1；诊断修复后仍有 backward codegen 错误，生产路径未解锁 |
| P3 torch_checkpointing×AC×EMA | NPU2 native FSDP2；full AC/dynamic EMA、selective AC/fixed EMA；4 vs 2+2，七类状态精确 | 通过 | `checkpointing_fullac_dynamic_20261023.log`、`checkpointing_selectiveac_fixed_20261023.log`，均 EXIT_CODE=0；不替代 AC 独立等价验收 |
| P3 chunked×原生FSDP 真实Trainer | NPU2，FP32，默认decoder布局，绑定/非绑定权重；全logits逐步梯度/更新参考及DCP冷恢复 | 通过 | chunked_root_trainer_gradient_equivalence_tied0/1_20261023.log，loss maxAE4.768e-7、梯度≤9.779e-9、参数7.451e-9；chunked_root_cold_tied0/1_full/resume_20261023.log四份exit0；自定义布局/offload/compile/混合模型并行仍拒绝，不作所有配置支持声明 |
| P2/P3 packed PP×Ulysses×FSDP 数值恢复 | NPU8，PP2×Ulysses2×FSDP2，full/selective AC，dynamic EMA，DCP 4 vs 2+2 | 通过 | 本轮 packed_dense_mask_1F1B / Interleaved1F1B 的 full/resume 四日志均 EXIT_CODE=0；各rank恢复前/后状态精确，loss AE=0；断言CPU回退明确报告，非纯AiCore支持声明 |
| 附加诊断：packed纯AiCore执行 | 相同配置，额外要求无CPU/AiCPU算子 | 失败 | 原ArgSort回退已通过消除冗余BlockMask构造避免；aten::_assert_async.msg仍明确回退CPU。纯AiCore不是原始目标的“无静默回退”，不擅自将此附加诊断变成整体完成条件 |
| P4 原始默认配置多步对齐 | torchtitan-npu，双卡100步，原阈值 | 失败 | 第61步 grad_norm AE=0.3347711563、RE=0.02443412476，超标；详见 P4 诊断节。非融合/替代attention/no-view/裁剪顺序诊断组合100步精确，不能替代默认配置验收 |

## P2/P3 首批结果

首批 8 passed / 3 failed，总 exit 1；日志 `p2_p3_20261022.log`。

| 项目 | 状态 | 证据 / 范围 |
| --- | --- | --- |
| TP×FSDP | 通过 | 4 卡 NPU；loss max AE=4.768e-7，参数=1.478e-6，tol=1e-4 |
| CP×TP | 通过 | 4 卡 NPU，FP32 tol=1e-5 |
| TP×EP×CP | 通过 | 4 卡 NPU，FP32 tol=1e-4；EP 共享布局，不凭维度乘积猜卡数 |
| PP×EP | 通过 | 4 卡 NPU，4 steps，tol=1e-4；FSDPIdentity forward 警告待审查 |
| PP×validation | 通过 | 2 卡，两个 validation 结果与独立参考完全一致 |
| AC | 通过 | 2 卡，4 steps，tol=1e-4；不替代 selective 全覆盖 |
| vocab-parallel loss | 通过 | 2 卡，覆盖不整除词表边界 |
| torch_checkpointing 冷启动 | 通过 | 实际 backend 的 EMA reseed 失败协调与重试；不替代完整恢复覆盖 |
| torch_checkpointing NPU双卡恢复 | 通过（指定配置） | 0.1.0实例级metadata设备适配；真实FSDP2、连续4步vs2+2，两个rank六类状态精确一致；2026-10-23修复节及sharded日志 |
| chunked×FSDP | 失败 | compatibility 明确拒绝并给出解锁条件；没有放宽拒绝或静默切换。TP 路径另测 |
| EMA layout | 通过 | 修正测试设备后定位并修复 Partial DTensor clone 布局变化；第二轮复验 exit 0，覆盖空/不均 shard、replicate、partial、buffer 及 DCP 空 rank 恢复 |
| PP×CP | 通过 | 原测试只接受 CPU substitute；新增真实 NPU SDPA/native-FSDP 路径后完整 4 卡训练等价通过，exit 0；原失败日志仍保留 |

本机 4–7 号设备复验已启动，日志 `p1_p3_retries_20261022.log`：EMA、PP×CP、
Interleaved1F1B、新增 HSDP（replicate=2 × shard=2）、chunked×TP；
各命令独立记录退出码，不因前项失败跳过后项。
CPU 全套复验 exit 0：4072 passed、13 skipped（`cpu_after_cp_20261022.log`）。

7 个远端节点均能 SSH 且容器存在（`node_probe_20261022.log`），不代表训练可用。
10.1.0.12 实测 torch/npu 2.12、8 卡健康空闲，但缺 colorama，测试入口被阻塞
（`node12_retries_20261022.log`）；未改远端环境，已转回本机完整环境复验。

P4 日志：`p4_baseline_import_20261022.log` 显示本地 torchtitan 缺旧 logging API；
用 wheels 的 torchtitan 0.3 隔离安装后，`p4_baseline_pinned_import_20261022.log`
显示缺 triton。未将导入失败当成数值不对齐，也未改 baseline 源码。

chunked×TP 独立测试 exit 0，4 steps loss max AE=4.768e-7；
不替代 chunked×FSDP 的缺口。命令设置 `CHUNKED_LOSS_EQ_TP=2`，
日志 `p1_p3_retries_20261022.log`。AC 首批实际覆盖 full 和 selective 两种模式。

EMA 复验定位到真实根因：torch 2.12 的 Partial DTensor clone 将副本布局变成
Replicate。初始化改为克隆本地 shard 并按原 mesh/placements/shape/stride
重建 DTensor，不触发全局通信。`ema_diagnostic_20261022.log` 保留失败证据；
第二轮复验 `ema_retry2_20261022.log` exit 0。全套 CPU 门禁已再次启动，
日志 `cpu_after_ema_20261022.log`。

## P0 训练命令

在仓库根目录运行；首轮未设置 `LLMTUNER_PROMPT_FIELD`，其余配置相同。

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export LLMTUNER_DATASET_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl
export LLMTUNER_PROMPT_FIELD=question
export LLMTUNER_STEPS=10 LLMTUNER_MAX_SEQ_LEN=4096
export LLMTUNER_DUMP_FOLDER=/home/jianzhnie/llmtuner/llm/TorchLLMTuner/outputs/qwen3-8b-npu/p0_20261022
bash examples/train_qwen3_8b_npu.sh
'
```

修复只新增字段环境变量覆盖，保留原数据集默认字段；回归测试使用临时文件，
不依赖本地模型或数据。另新增 checkpoint interval/load_step 的环境变量覆盖，
供相同总步数及学习率计划下的中间断点恢复验证。

## 已执行门禁与观察

### 本轮追加验证

- EMA 修复后的 CPU 全套 exit 0：4072 passed、13 skipped，
  `cpu_after_ema_20261022.log`。Profiler 后续修改后的全套仍需重跑。
- PP×full-AC×动态 EMA 的 DCP 完整/恢复双进程测试 exit 0；
  PP×selective-AC×动态 EMA 亦 exit 0。比较参数、optimizer、scheduler、
  trainer counters、EMA 及恢复 loss，使用脚本原来的零容差断言。
  日志 `p3_resume_compile_20261022.log`、`pp_selective_resume_20261022.log`。
- PP×aot_eager compile×动态 EMA 的完整/恢复测试 exit 0，检查实际
  OptimizedModule 与非零 graph 数；不代表 Inductor 或所有 compile 组合通过。
- Profiler 发现并修复真实能力缺口：旧实现 NPU 仅采集 CPU activity。
  新实现调用 torch_npu.profiler，采集 CPU+NPU，原始文件放在配置的 traces/raw，
  导出原生 JSON 而非伪 gzip。新增 CPU 参数测试与真实 NPU Trainer 对照测试。
  `profiler_validation_retry_20261022.log` 中 profiler exit 0：4 步更新零容差
  一致，trace 有 1494 个 async_npu 设备事件。最初断言误以为 Chrome trace
  顶层必为 dict 而失败，后兼容标准 list 格式；未去掉设备事件断言。
- NPU 两 rank 默认 RNG 恢复及无效状态隔离通过，
  `npu_profiler_retry1_20261022.log`；命令显式传 `device_rng_equivalence.py npu`。
- validation RNG 故障协调原用 10 秒 Gloo 初始化导致失败；改为共享 harness
  选择 NPU/HCCL 后，真实 Trainer 验证入口/退出故障注入通过，exit 0，
  `profiler_validation_retry_20261022.log`。
- `rng_checkpoint_equivalence.py` 是 CPU-only 测试，native/DDP 子项通过但
  custom 子项缺 torch 2.14 API 导致总 exit 1；不将其当作 NPU 覆盖或全通过。
- metric reduction 原先是 CPU tensor/Gloo，尽管部分底层组会搬运到设备，
  尚不足以宣称 NPU 原生标量覆盖。已改为 harness + NPU FP32，重新运行，
  同批验证 aot_eager TP collective 梯度；日志 `metrics_compile_npu_20261022.log`。
- profiler/EMA 定向 CPU 验证 163 passed；raw 采集产生的仓库根目录文件夹
  已移动到 outputs，未提交生成物。

### 最新复验（本轮最终代码）

- Profiler 全套修改后的 CPU 单测 exit 0：4073 passed、13 skipped，
  `cpu_after_profiler_20261022.log`；compileall 通过。
- 真实 NPU profiler 最终断言检查 **Ascend Hardware** 进程下 `ph=X`、
  `dur>0`、`Task Type=KERNEL_*` 的实际执行事件，不把 async_npu flow 箭头
  当成 kernel。最新 trace 包含 723 个实际 kernel，训练参数仍零容差一致。
  前述 1494 是 flow 事件计数，不能作为 kernel 数量；以此处新证据为准。
  日志 `profiler_multimodal_final_20261022.log`，整条命令 exit 0。
- 多模态管线 exit 0：生成真实 PNG、本地 JSONL/tokenizer，生产 processor、
  FirstFit packing、loader checkpoint/恢复、图片 patches 和 grid_thw 搬运到 NPU，
  batch 与恢复参考、搬运回 CPU 后的 tensor 均零容差一致，浮点值有限。
  使用 fixture source 仅做 base64 解码，不覆盖远端存储、视频或视觉模型训练。
  命令 `PYTHONPATH=. python tests/integration_tests/npu_multimodal_pipeline.py`，
  同上述最终日志。不把管线验证扩展为多模态模型端到端已支持。
- TP aot_eager compiled gather/scatter 梯度与解析值零容差一致（NPU/HCCL），
  `metrics_compile_npu_20261022.log`，该脚本 exit 0。
- NPU metric reduction 的严格相等测试失败：参考 loss=0.75，实际
  0.7500000596046448。保留失败，不事后放宽阈值；需继续确认 FP32/HCCL
  的归约舍入及验收依据。该批整体 exit 1，不记全部通过。
- 定位到 `_step_metrics` 在全局归约前执行各 rank 的 FP32 除法；独立参考
  先求和后除法，两者舍入顺序不同。当前没有证据说明训练或归约语义错误，
  因此不为满足严格测试而擅自修改生产计算顺序，也不将失败隐去。
- 修改文件的 pre-commit 全部通过（exit 0），日志
  `precommit_changed_20261022.log`。全仓门禁仍有已记录的基线格式/旧 Ruff
  自动修复冲突，不将定向通过描述为全仓通过。

新增 P3 命令均在容器、仓库根目录执行；NPU 单卡测试使用
`ASCEND_RT_VISIBLE_DEVICES=6`，双卡 RNG/validation 使用 `6,7`：

```bash
LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic python tests/integration_tests/run_all.py --master-port=29531 pp_checkpoint_equivalence
LLMTUNER_TEST_AC=selective LLMTUNER_TEST_EMA=dynamic python tests/integration_tests/run_all.py --master-port=29532 pp_checkpoint_equivalence
LLMTUNER_TEST_COMPILE_BACKEND=aot_eager LLMTUNER_TEST_EMA=dynamic python tests/integration_tests/run_all.py --master-port=29531 pp_checkpoint_equivalence
PYTHONPATH=. python tests/integration_tests/npu_profiler_equivalence.py --output outputs/npu-profiler-equivalence-20261022-retry3
PYTHONPATH=. python tests/integration_tests/npu_multimodal_pipeline.py
PYTHONPATH=. torchrun --master-port=29533 --nproc_per_node=2 tests/integration_tests/device_rng_equivalence.py npu
PYTHONPATH=. torchrun --master-port=29533 --nproc_per_node=2 tests/integration_tests/validation_rng_coordination.py
```

- 容器 CPU 单测：exit 0，4067 passed、13 skipped；
  日志 `outputs/qwen3-8b-npu/logs/cpu_20261022.log`。运行中新增的 checkpoint
  配置用例另行验证，目标文件共 4 passed。
- `ruff check llmtuner tests`、修改文件格式检查、compileall、diff 检查通过。
- 全量格式检查失败：现存 `tp_ep_cp_training_equivalence.py` 有格式差异，未改动。
- 宿主机下载 pre-commit 到 wheels 并离线装入 `outputs/quality-tools`，容器未改动。
  全量 pre-commit 执行失败：Ruff 0.9 自动修改无关 `hf_inspection.py` 两处写法，
  已撤销这些自动修改；其余 hooks 通过。日志 `precommit_retry1_20261022.log`。
- 首轮成功训练 rank 0 loss 从 0.51693 到 0.42940，均有限；记录的 step
  时间戳在预热后间隔约 2 秒，step 6 的 tps 从约 2140 降到 1649 后恢复。
  该采样只有秒级分辨率，不据此声称无抖动。
- NPU 输出显式 `_assert_async` CPU fallback 和 ArgSort AiCPU 警告；
  不是静默回退，但仍需确认调用位置、对目标训练路径与数值的影响。
  不将本轮结果扩展为所有算子均原生 NPU 执行的结论。

连续性验证保留完整参考于 `p0_continuity_20261022/reference-step-10`，
恢复结果写回同目录 `checkpoint/step-10`。比较工具为
`outputs/qwen3-8b-npu/compare_checkpoints.py`，逐批读取全部 DCP 条目，
以零容差检查权重、优化器、scheduler、计数、数据游标及随机状态。
连续训练与恢复均 exit 0，step 6–10 打印的 loss / grad_norm 全部一致；
完整 checkpoint 比较 exit 0，全部 6434 条目一致，包括解码后的 RNG payload。

## CP 修复与验证

- SDPA 测试原来硬编码 CPU/Gloo/FP64，现复用 harness 选设备。CPU tol=1e-12，
  NPU FP32 预设 tol=1e-5，拒绝非有限误差。GQA NPU 与 CPU/MHA NPU 回归均
  exit 0；日志 `cp_sdpa_npu_20261022.log`、`cp_sdpa_regression_20261022.log`。
- Ulysses 首次失败：torch 私有 BlockMask 分片强制 Inductor，NPU 编译失败。
  现在 NPU 显式复用已有 eager builder 重建 Q-sharded mask，保持全局 Q/KV
  索引、headtail 重排和 128 block 整除约束，不改 CPU/CUDA 路径。
  新增 CPU 两 rank × 有/无重排回归，4 passed。
- 第一轮复验发现测试反例强制使用 NPU 不支持的 FlexAttention；改用真实 NPU
  SDPA Ulysses kernel，保留错误 headtail 配置反例及所有拒绝检查。
  第二轮完整复验 exit 0；与错误重排语料差距为 0，与正确语料 gap=0.4002。
  失败日志 `p1_20261022.log`、`cp_ulysses_retry1_20261022.log` 均保留。

P2/P3 批量已启动，尚待逐项审查最终输出与覆盖范围：

```bash
python tests/integration_tests/run_all.py --master-port=29511 tp_fsdp_equivalence cp_tp_equivalence tp_ep_cp_equivalence pp_ep_equivalence pp_cp_equivalence pp_validation_equivalence ac_equivalence chunked_loss_equivalence vocab_parallel_loss_equivalence ema_layout_equivalence torch_checkpointing_cold_start
```

日志 `outputs/qwen3-8b-npu/logs/p2_p3_20261022.log`；不以批量 exit 0
代替设备、配置和 skip 范围检查。CP 修复后的全套 CPU 门禁已重新启动，
日志 `outputs/qwen3-8b-npu/logs/cpu_after_cp_20261022.log`。

P1 启动命令（日志均位于 `outputs/qwen3-8b-npu/logs/`）：

```bash
# 在容器中、仓库根目录下，PYTHONPATH=.
torchrun --master-port=29511 --nproc_per_node=2 tests/integration_tests/fsdp_mesh_equivalence.py shard
torchrun --master-port=29511 --nproc_per_node=2 tests/integration_tests/fsdp_mesh_equivalence.py ddp
python tests/integration_tests/run_all.py --master-port=29511 fsdp_checkpoint_equivalence tp_equivalence tp_grad_norm_equivalence pp_equivalence pp_real_corpus_equivalence cp_equivalence cp_ulysses_equivalence ep_equivalence ep_fsdp_equivalence
ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --master-port=29512 --nproc_per_node=2 tests/integration_tests/cp_sdpa_equivalence.py
```

## 2026-10-22 补充：真实依赖与数值诊断

- 基线 import：离线隔离安装 torchtitan 0.3 / triton-ascend 3.2.1 到
  `outputs/baseline-runtime`，未替换系统 torch/npu。`p4_triton_import_20261022.log`
  exit 0；仍有缺少 `torch_npu._inductor.ascendc` patch 的警告，不能据此声称
  基线要求的 torch/npu 2.14 与当前 2.12 全面兼容。
- 官方 RMSNorm 示例：`p4_rmsnorm_20261022.log` exit 0，FP32 shape=2×17×64，
  基线自身 RMSNorm/AscRMSNorm 前向及输入、权重梯度通过；不是跨框架验收。
- Qwen 诊断：真实基线 Qwen3 与 HFTransformerModel，2 层、hidden=64、FF=128、
  heads=4、kv=2、head_dim=16、vocab=128，FP32 单卡单文档 seq=32。
  基线 state-dict adapter 严格映射全部 25 个参数；初始 RoPE theta 不一致
  （1e6 vs 1e4）导致 logits max AE=0.01247，统一到 1e6 后降为 1.192e-7。
  这是探针配置修正，不是生产模型修复。
- 四步 AdamW（lr=1e-3、foreach=False），CE mean、grad clip=1，逐步复用官方
  metrics 并在 CPU 显式检查有限值。logits mean AE≤1.917e-8、mean RE≤2.283e-7、
  cos≥0.999999999999998；loss AE≤1.431e-6，grad norm AE≤1.193e-7，均通过官方阈值。
  每步更新后全部 25 个参数通过预设 atol=1e-5、rtol=1e-4 的诊断检查；
  该权重容差仅为 FP32 探针检查，不是已确认的正式 P4 权重验收依据。
  `p4_qwen_four_steps_20261022.log` exit 0，复现脚本
  `outputs/qwen3-8b-npu/p4_qwen_probe.py`。该脚本显式用基线 SDPA 替代 packed
  attention 配置，仅证明单文档诊断，不代表 packed / 完整 Trainer / 分布式对齐。
- TorchAO 0.18 隔离安装到 `outputs/torchao-runtime`；第一次探针 combine 缺少
  inputs 参数，exit 1，不是内核失败。修正调用后真实 padded dispatch/combine
  及输入梯度检查 exit 0，`torchao_npu_dispatch_retry_20261022.log`。
  仍有 ArgSort AiCPU 警告；未证明多 rank padded dispatcher 正确性。
- 额外组合：`remaining_combinations_20261022.log` exit 0，
  ep_checkpoint_ac_equivalence、pp_tp_trainer、shared_expert_tp_equivalence
  共 3 passed；PP×TP 的有限正 loss 断言不替代完整 dense 数值等价。

探针复现（容器内、仓库根目录，运行前确认卡空闲）：

```bash
PYTHONPATH=/home/jianzhnie/llmtuner/llm/torchtitan-npu:$PWD/outputs/baseline-runtime:. ASCEND_RT_VISIBLE_DEVICES=6 python outputs/qwen3-8b-npu/p4_qwen_probe.py
```

下一步仍需把诊断整理成正式可复现集成测试，确认权重容差依据，补齐真实 Trainer
及分布式基线对齐。KCP、完整 torch_checkpointing 恢复、Inductor 及组合覆盖缺口
保持未完成，不因本节的小模型结果缩小整体目标。

### TorchAO 双卡 dispatcher 补验

新增 `tests/integration_tests/torchao_dispatch_equivalence.py`：2 卡 HCCL，
8 experts、top-k=2、pad=16，各 rank 分别 13/19 tokens。专家运算采用全局专家
编号加一的乘法，独立解析参考验证路由、输出、输入梯度及路由分数梯度；
运行随机路由和仅命中专家 0/7 的偏斜路由，覆盖空专家与不均匀 token 数。
预设 FP32 atol=rtol=1e-5，双 rank 两种路由全部通过。

首轮测试 exit 1：探针错误要求 padded counts 总和等于分配行数。检查真实
torchao 源码后确认其按最坏情况预分配额外尾部 padding；探针为这些尾行设置
零因子，不改变实际有效专家分段及参考值。复验 exit 0，日志
`torchao_dispatch_ep2_retry1_20261022.log`，首轮失败证据保留于
`torchao_dispatch_ep2_20261022.log`。生产代码未修改，也未放宽数值容差。

```bash
PYTHONPATH=outputs/torchao-runtime:outputs/baseline-runtime:. ASCEND_RT_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master-port=29539 tests/integration_tests/torchao_dispatch_equivalence.py
```

新测试 Ruff check / format check 和 `git diff --check` 通过。此结果覆盖真实
多 rank padded dispatch 的前反向，不证明 FP8/MXFP8 grouped GEMM 或 MoE
完整训练；ArgSort AiCPU 警告仍保留。

### 正式保存 Qwen 单文档诊断

新增 `tests/integration_tests/npu_qwen_alignment.py`，将 outputs 中的临时探针
整理为仓库集成脚本。固定初始参数并零容差验证复制，保存实际初始 state dict、
四批 tokens/targets、HF 配置、优化器 defaults、随机种子、两侧 commit、官方
metrics 路径，以及逐步 logits/loss/grad norm/全部参数的指标。
权重仍只使用先前诊断容差，不冒充正式 P4 验收标准。

首轮 exit 1：baseline 放在 PYTHONPATH 最前时，其同名 tests 包遮蔽本仓库
`tests.caps`。将本仓库放在最前后复验 exit 0，四步全部通过；没有数值修复或
容差变更。日志为 `p4_qwen_integration_20261022.log` 和
`p4_qwen_integration_retry1_20261022.log`。完整产物在
`outputs/qwen-alignment-formal-20261022/`，包括 manifest.json、inputs.pt 和
step-0.json 至 step-3.json。

```bash
PYTHONPATH=.:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu ASCEND_RT_VISIBLE_DEVICES=6 python tests/integration_tests/npu_qwen_alignment.py --output outputs/qwen-alignment-formal-20261022
```

复现须选择不存在的输出目录。全量 Ruff check、新增文件 format check、diff
check 通过。宿主 compileall 因容器创建的 __pycache__ 所有权失败，改在容器内
执行 compileall（llmtuner 和新增脚本）通过；未改缓存权限。
正式脚本仍只覆盖 FP32 小模型单文档 SDPA，完整 Trainer、packed attention、
分布式及 Qwen3-8B 数值对齐保持未完成。

### 100 步确定性对齐与基线 Trainer 实测

正式脚本新增 `--steps`（默认 100），开启确定性算法，并在裁剪前记录和检查
全部 25 个参数梯度。100 步全部通过，产物为
`outputs/qwen-alignment-100steps-deterministic-20261022/`，日志
`p4_qwen_100steps_deterministic_20261022.log` exit 0。最大 logits mean AE
4.398e-8、mean RE 8.398e-7；loss AE 4.769e-7；grad norm AE 1.193e-7。
100 个 step JSON 均含 25 组梯度指标；仍不是完整 Trainer 验收。

```bash
PYTHONPATH=.:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu ASCEND_RT_VISIBLE_DEVICES=7 python tests/integration_tests/npu_qwen_alignment.py --steps 100 --output outputs/qwen-alignment-100steps-deterministic-20261022
```

基线 Trainer CLI 首次缺少 typeguard，宿主下载 4.4.4 后离线装到隔离 runtime；
带 `--module torchtitan.models.qwen3 --config qwen3_debugmodel --help` 成功。
未改系统环境或基线源码。随后真实单卡 debugmodel（约 32M 参数）、本地 c4_test、
seq32、batch1、1 step、无 AC/compile/checkpoint 实测：

- 默认 chunked loss：exit 1，首步 loss.backward 报非空 tensor storage 未分配；
  `p4_baseline_trainer_smoke_20261022.log`。与候选仓库 chunked×FSDP 限制相关，
  但尚不能直接断定同一根因。
- 显式非 chunked CrossEntropyLoss：通过前反向后，调用当前 torch 2.12 不存在的
  `torch.distributed.set_timeout` 失败，exit 1；
  `p4_baseline_trainer_nonchunked_20261022.log`。探针配置位于
  `outputs/qwen3-8b-npu/baseline_trainer_config.py`，仅用于定位，未伪装成原配置。

真实 Trainer 在当前环境存在明确 API 兼容缺口；不能以小模型直接调用结果替代。
下一步需检查 timeout 的必要语义及支持实现，或准备基线声明的 torch/npu 2.14
隔离环境，保持当前已通过训练环境不变。全量 Ruff、新文件格式检查、容器
compileall 及 diff check 通过。

### 基线 Trainer timeout 兼容探针

检查当前 torch 2.12 的 `_set_pg_timeout` 源码：仅枚举 NCCL/Gloo，不处理 NPU，
不能简单将它当作 HCCL 的等价替代。当前 ProcessGroupHCCL 实际暴露
`_set_default_timeout(timedelta)`。在 outputs 下的隔离配置中显式注册缺失的
`torch.distributed.set_timeout`，直接调用真实 HCCL backend setter 并打印警告，
保留基线原有 barrier / synchronize；未用 no-op 绕过、未修改系统或基线源码。

使用此前明确的非 chunked CrossEntropyLoss，真实 baseline Trainer 完成 3 步，
exit 0，正常 teardown。loss=7.68950/7.70972/7.00572，grad norm 均有限。
日志 `p4_baseline_trainer_timeout_probe_20261022.log`；兼容代码在
`outputs/qwen3-8b-npu/baseline_trainer_config.py`。setter 返回无警告，但尚未
通过真实超时故障注入证明 watchdog 行为；不得据此修改候选生产 timeout 策略。

```bash
export PYTHONPATH=.:outputs/qwen3-8b-npu:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
torchrun --nproc_per_node=1 --master-port=29540 -m torchtitan_npu.train --module baseline_trainer_config --config baseline_probe --hf-assets-path /home/jianzhnie/llmtuner/llm/torchtitan/tests/assets/tokenizer --dataloader.dataset-path /home/jianzhnie/llmtuner/llm/torchtitan/tests/assets/c4_test --training.steps 3 --training.local-batch-size 1 --training.seq-len 32 --training.disable-cuda-graphs --checkpoint.no-enable --compile.no-enable --dump-folder outputs/baseline-trainer-timeout-probe-20261022 activation-checkpoint:none
```

这是约 32M 参数 debugmodel 的真实 Trainer 可运行性证据，不是初始权重、数据
已经匹配的双侧比较，也不是原版 chunked 路径通过。下一步可以基于该入口保存
实际输入和权重，建立两侧相同配置的 Trainer 对齐，不能沿用此前无 seed 约束
的 smoke loss 作为数值基线。

### 真实 torch_checkpointing NPU 训练恢复

新增 `tests/integration_tests/npu_checkpoint_backend_equivalence.py`，直接调用
真实已安装的 torch_checkpointing 后端及 Trainer，不替换 reader/writer。
单卡微型 Qwen3、FP32、seq32、batch1、确定性，连续 4 步与训练 2 步保存后
重新构造 Trainer 恢复到 4 步比较；两侧 scheduler total_steps 固定为 4，
最后保存完整状态。模型、optimizer、scheduler、train_state、dataloader 和
解码后的 RNG（Python/NumPy/CPU/NPU 及元信息）全部零容差一致，exit 0。

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=6 python tests/integration_tests/npu_checkpoint_backend_equivalence.py --output outputs/torch-checkpointing-npu-resume-retry3-20261022
```

日志 `torch_checkpointing_npu_resume_retry3_20261022.log`；两侧原始最终状态
另存为上述输出目录的 reference.pt / resumed.pt，真实 checkpoint 同目录保留。
前几轮测试修正证据保留：首次 Trainer.train 自动 close 后访问 checkpointer=None；
retry1 错比较 RNG pickle 原始字节；retry2 解码后比较器不支持字符串元信息。
最终使用递归精确比较，NumPy 数组转换 tensor，不过滤随机状态或放宽容差。
这些是新增测试的生命周期与比较方式修正，未改生产实现。

全量 Ruff、新文件格式和 diff 检查通过。此结果填补单卡真实 I/O、恢复与继续
训练覆盖；多卡 FSDP/拓扑变更下该可选后端尚未验证，不能外推通过。

### torch_checkpointing 双卡 FSDP 实测失败

扩展正式脚本支持 torchrun：global batch 随 world size 设置，每 rank 单独保存
最终状态，断言实际 FSDPModule，DTensor 比较 placements、global shape 及本地
shard（全部 rank 都检查）。没有使用 gather 后的窄参考替代实际分片恢复。

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master-port=29542 tests/integration_tests/npu_checkpoint_backend_equivalence.py --output outputs/torch-checkpointing-npu-fsdp2-20261022
```

结果：失败，exit 1，日志 `torch_checkpointing_npu_fsdp2_20261022.log`。
实际训练到首次 checkpoint，安装的 torch_checkpointing 0.1.0 在
`DefaultMetadataManager._all_gather_pickled_data` 使用
`torch.tensor(..., device="cuda")` 和 `.cuda()` 聚合 metadata，NPU-only torch
报 `Torch not compiled with CUDA enabled`。另一个 rank 被 torchrun 终止。
检查其 CheckpointManager.Config，没有 metadata manager / 通信设备注入选项；
不能用改 backend 参数解决，也不能删除 resharder（否则改变 checkpoint 语义）。

解锁条件：提供设备无关或明确支持 NPU 的该依赖版本，或实现并验证有明确生命周期
和版本边界的 metadata 适配。不得全局将 CUDA API 偷换为 NPU，也不将单卡通过
扩展为双卡通过。当前保留失败测试与证据，生产代码未绕过该错误；其他独立项
仍可继续，整体任务并非因此全部 blocked。

扩展脚本后单卡回归 exit 0，六类完整状态再次精确一致；日志
`torch_checkpointing_npu_single_regression_20261022.log`，输出
`outputs/torch-checkpointing-npu-single-regression-20261022/`。
全量 Ruff、新文件格式、容器 compileall 和 diff check 通过。

### TP×EP×expert-FSDP 与真实 Inductor 补验

- `ep_fsdp_without_dense_dp`：四卡 NPU，TP4/EP2/efsdp2，dense DP=CP=1，
  真实 FSDP 专家 shard、HF 独立参考和反向。exit 0，logits max AE=8.941e-8，
  专家梯度 max AE=1.164e-9，采用既有 atol=1e-5/rtol=1e-4；
  `ep_fsdp_no_dense_dp_npu_20261022.log`。不替代其他 DP 布局。
- `tp_compile_collectives` 新增 `LLMTUNER_TEST_COMPILE_BACKEND`，默认仍为
  aot_eager，日志明确 backend。实际 inductor fullgraph 编译未启用错误抑制，
  双 rank exit 1：torch_npu codegen/triton.py 的 initialize_range_tree 访问
  `self.numels["r"]` 触发 KeyError。尚未执行到编译后的数值断言，不能算通过。
  `tp_inductor_npu_20261022.log`；debug 文件移到
  `outputs/tp-inductor-debug-20261022/`，不留在源码目录。
  尽管设置 TORCHINDUCTOR_NPU_BACKEND=ascendc，trace 实际进入 Triton codegen，
  不能据环境变量声称执行了 AscendC backend。

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tests/integration_tests/run_all.py --master-port=29545 ep_fsdp_without_dense_dp
PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=6,7 LLMTUNER_TEST_COMPILE_BACKEND=inductor TORCHINDUCTOR_NPU_BACKEND=ascendc torchrun --nproc_per_node=2 --master-port=29546 tests/integration_tests/tp_compile_collectives.py
```

Inductor 解锁需匹配 torch/torch_npu codegen API 或修复依赖；不在模型代码里
跳过编译、不把 aot_eager 通过扩大为 Inductor 通过。当前仅变更测试后端选择。

默认 aot_eager 回归 exit 0，双 rank gather/scatter 梯度均精确一致，日志
`tp_aot_regression_20261022.log`。全量 Ruff、修改文件格式和 diff 检查通过。

### 真实基线 Trainer 对齐输入采集

新增 `tests/integration_tests/npu_baseline_training_capture.py`，调用真实 baseline
ConfigManager、Trainer.build/train，不替代 dataloader/loss/optimizer/训练循环。
前向 hooks 记录实际输入、positions、attention mask 和 logits；透明的
forward_backward_step observer 保存原始 input_dict、已对齐 labels 和全局有效
token 数。初始和最终 FSDP 权重通过真实 adapter 转换为 HF 参数后落盘。

采用此前明确的非 chunked/timeout 兼容配置，单卡 debugmodel 3 步 exit 0，
日志 `p4_baseline_capture_retry1_20261022.log`。产物在
`outputs/baseline-trainer-capture-retry1-20261022/alignment-capture/`：
initial-hf.pt、final-hf.pt、config.json、forward-batches.pt、consumed-batches.pt。
实际加载复核：90 个 HF 权重张量，3 批输入及 labels，3 组 logits，
logits shape=[1,32,2048]、labels=[1,32]。

首轮失败是 BlockMask 内部局部 mask callable 无法 pickle，日志
`p4_baseline_capture_20261022.log`。修正为保存其实际 token 级布尔 mask，
不是 pickle callable，也没有改模型使用的原始 BlockMask。
该 observer 增加 CPU 同步，不将本次耗时作为性能验收证据。

```bash
export PYTHONPATH=.:outputs/qwen3-8b-npu:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
torchrun --nproc_per_node=1 --master-port=29547 tests/integration_tests/npu_baseline_training_capture.py --module baseline_trainer_config --config baseline_probe --hf-assets-path /home/jianzhnie/llmtuner/llm/torchtitan/tests/assets/tokenizer --dataloader.dataset-path /home/jianzhnie/llmtuner/llm/torchtitan/tests/assets/c4_test --training.steps 3 --training.local-batch-size 1 --training.seq-len 32 --training.disable-cuda-graphs --checkpoint.no-enable --compile.no-enable --dump-folder outputs/baseline-trainer-capture-retry1-20261022 activation-checkpoint:none
```

目前只是完整 Trainer 的比较基线产物，未运行候选回放，不宣称两侧通过。
下一步从实际 config 和权重建立相同 HF 架构、精度、优化器及 scheduler，
用捕获的 labels（不再次 shift）和 mask 驱动候选真实 Trainer。
全量 Ruff、新增脚本格式和 diff 检查通过。

### 完整候选 Trainer 三步回放与官方指标

新增 `tests/integration_tests/npu_trainer_alignment_replay.py`。另采集真实 baseline
debugmodel 的 FP32 参数计算基线（`--training.mixed-precision-param float32`），
输出 `outputs/baseline-trainer-capture-fp32-20261022/alignment-capture/`，日志
`p4_baseline_capture_fp32_20261022.log` exit 0。原 BF16 产物保留，不混用。

候选按真实 debugmodel 构造 8 层、hidden256、FF3072、heads16、kv8、head_dim128、
vocab2048、RoPE theta1e6、共享 embedding/head，严格加载初始权重并逐 tensor
零容差验证。基线 adapter 对共享 lm_head 省略重复键；首次加载因此失败，
修正为先断言参数共享，再使用同一 embedding tensor 补齐该键，不新增独立权重。
原失败日志 `p4_candidate_trainer_replay_20261022.log` 保留。

通过完整 `Trainer.train()`（不是手写 optimizer loop）回放捕获的三批输入和
已对齐 labels，不重复 shift。实际 positions 和捕获 mask 必须精确证明是
单文档 causal 才允许运行；不把当前脚本用于任意 packed mask。
AdamW lr8e-4、betas=(.9,.95)、wd=.1、eps1e-8，warmup2、decay_ratio=.8、
总步数3、clip1；候选实际 for-loop，日志明确记录 fused 请求回退。
基线 fused 与候选 for-loop 的实现差异尚未统一，作为诊断边界保留。

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=6 python tests/integration_tests/npu_trainer_alignment_replay.py --capture outputs/baseline-trainer-capture-fp32-20261022/alignment-capture --output outputs/candidate-trainer-replay-metrics-20261022
```

exit 0，日志 `p4_candidate_trainer_replay_metrics_20261022.log`；官方 metrics
比较三步 logits 与 loss 全部通过：logits mean AE≤2.016e-4、mean RE≤0.004577、
cos≥0.99999967；loss AE≤3.291e-5、max/mean RE≤4.567e-6。loss 参考从实际
baseline logits 和 labels 按 FP32 CE mean 计算，不使用终端四舍五入数值。
输出目录保存 candidate logits、最终参数和 diagnostic.json（含逐步指标及 LR）。

尚缺 baseline 原始 grad norm（只有格式化日志）、关键权重正式容差、精度选项
与 optimizer 实现完全一致的复验、100 步及分布式覆盖，不能将上述两个指标
通过标记为完整 P4 通过。全量 Ruff、新脚本格式、容器 compileall 和 diff 检查通过。

### 双侧完整 Trainer 100 步指标对齐

基线采集新增透明 metrics.log observer，保存未格式化的原始 loss、grad norm、
LR 和 token 计数，不再依赖终端日志精度。新采集使用 FP32 参数/归约、HF32
显式关闭、debug.seed=42、deterministic、AdamW for-loop、seq32、batch1，
运行真实 Trainer 100 步。候选复用该初始权重和实际 100 批数据，HF32 开关
按基线配置设置，使用相同 optimizer 实现和超参；完整 Trainer.train() 运行。

候选 mask 验证扩展为根据真实 positions 的文档重置构造布尔文档 causal mask，
必须与捕获的 token mask 零容差一致，否则拒绝回放；没有直接忽略 packed mask。
仅支持该文档 causal 语义，不声称支持其他 attention mask 家族。

基线 `p4_baseline_capture_100_20261022.log` exit 0；候选最终复验
`p4_candidate_trainer_replay_100_final_20261022.log` exit 0。
100 步 logits、原始 loss、原始 grad norm 均满足官方既定阈值：

| 对象 | 最大 mean AE | 最大 mean RE / max RE（标量） | 最小 cos |
| --- | --- | --- | --- |
| logits | 3.985e-6 | mean RE=1.004e-5，max RE 仅诊断 | 0.99999999999456 |
| loss | 5.246e-6 | 1.484e-6 | — |
| grad norm | 6.438e-5 | 1.875e-5 | — |

逐步 LR 和累计 token 数精确一致。最终全部 90 个非重复 HF 权重保存指标：
最大 mean AE=1.193e-7、最大 mean RE=2.123e-5、最小 cos=0.99999999996746。
权重仅作诊断，仍未替代明确的正式权重验收依据。

产物：`outputs/baseline-trainer-capture-100-20261022/alignment-capture/`
与 `outputs/candidate-trainer-replay-100-final-20261022/`。后者包含 candidate
logits、final 权重、diagnostic.json 的 100 步比较和 weight-diagnostics.json。

```bash
export PYTHONPATH=.:outputs/qwen3-8b-npu:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
torchrun --nproc_per_node=1 --master-port=29547 tests/integration_tests/npu_baseline_training_capture.py --module baseline_trainer_config --config baseline_probe --hf-assets-path /home/jianzhnie/llmtuner/llm/torchtitan/tests/assets/tokenizer --dataloader.dataset-path /home/jianzhnie/llmtuner/llm/torchtitan/tests/assets/c4_test --training.steps 100 --training.local-batch-size 1 --training.seq-len 32 --training.mixed-precision-param float32 --training.extension.no-allow-hf32 --debug.seed 42 --debug.deterministic --optimizer.implementation for-loop --training.disable-cuda-graphs --checkpoint.no-enable --compile.no-enable --dump-folder outputs/baseline-trainer-capture-100-20261022 activation-checkpoint:none
PYTHONPATH=. python tests/integration_tests/npu_trainer_alignment_replay.py --capture outputs/baseline-trainer-capture-100-20261022/alignment-capture --output outputs/candidate-trainer-replay-100-final-20261022
```

结果只证明当前单卡 debugmodel 完整 Trainer 的上述 100 步指标，不覆盖分布式、
Qwen3-8B 或原版 chunked baseline。完整 P4 仍未完成；独立的 KCP、可选后端
多卡 CUDA 硬编码、Inductor codegen 和其余组合缺口保持原状态。
本轮全量 Ruff、修改脚本格式、容器 compileall、diff 检查通过。

### 双卡完整 Trainer 100 步对齐：明确失败

采集与回放脚本扩展为每 rank 保存实际批次及指标，候选检查真实 FSDPModule
和 dp_shard=world_size；全局 loss 参考由各 rank logits 的 CE sum 经 HCCL
归约并除以捕获的 global_valid_tokens 得到，不用局部 loss 比较全局均值。
逐 rank 验证初始全局参数，按实际 mesh/placements distribute_tensor 后严格加载。

基线默认 full SPMD 路径构造失败：dp_mesh_dims 要求 full DTensor，实际参数
仍为 plain Tensor（torch 2.12 / baseline 0.3 组合），日志
`p4_baseline_fsdp2_100_20261022.log`。显式改为官方 partial_dtensor 路径后，
真实 dp_shard=2 的 baseline 完成 100 步，exit 0，日志
`p4_baseline_fsdp2_partial_100_20261022.log`；未删除 FSDP 或改成复制模型。
产物 `outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture/`。
加载复核：rank0/rank1 各 100 批，首批输入不同、global valid tokens=64，
90 个初始全局权重全部精确一致。

候选也完成真实 100 步，但官方指标比较失败，**不能记为对齐通过**：
step61 两 rank 的 baseline grad norm=13.700967788696289，candidate=
14.035738945007324；AE=0.3347711563（阈值0.02），RE=0.02443412476
（阈值0.005）。前 60 步通过逐步 logits/loss/grad norm 比较，step61 的
grad norm 首先触发断言。未放宽容差，也未只报告前60步来缩小验收范围。

首次候选启动端口冲突（`p4_candidate_fsdp2_100_20261022.log`）；改 standalone
后首次加载因 DCP helper 返回 plain Tensor 与 DTensor 混合而失败
（retry1 日志）。改为显式 distribute_tensor 后完成训练并暴露上述真实差异
（retry2 日志）；进一步保留原始逐步指标并加失败 step/metric 上下文后复验，
同一 step61 同样失败，日志 `p4_candidate_fsdp2_100_audit_20261022.log` exit 1。
原始 candidate logits、final DTensor、每 rank 100 步 metrics 和精确失败摘要
位于 `outputs/candidate-trainer-fsdp2-100-audit-20261022/`。

复现使用上一节基线命令，将 nproc_per_node 改为2、可见卡设为6,7，追加
`--parallelism.spmd-backend partial_dtensor` 并使用上述双卡 dump-folder；候选：

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 tests/integration_tests/npu_trainer_alignment_replay.py --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture --output outputs/candidate-trainer-fsdp2-100-audit-20261022
```

下一步定位 step61 前的参数/梯度偏差积累与 fused-QKV/分离-QKV 布局的归约
差异，不能仅凭数值接近推断根因。单卡已有结果保持其原范围；双卡 P4 状态
为失败。全量 Ruff、两脚本格式、容器 compileall 和 diff 检查通过。

### 双卡差异关键步快照定位

采集和回放脚本新增可选 `LLMTUNER_ALIGNMENT_TRACE_STEPS`，仅在指定步完成
真实 train_step 后保存全局参数与裁剪后的全局梯度。基线 fused QKV 梯度用其
原生 state-dict split 规则转换后交给真实 HF adapter；所有映射梯度非空。
不修改 forward/backward/optimizer；observer 的同步可能影响调度，因此不作为
性能证据。没有 trace 的失败结果独立保留。

设置 `LLMTUNER_ALIGNMENT_TRACE_STEPS=1,10,30,60,61`，按上一节同配置重跑：
基线 `p4_baseline_fsdp2_trace_20261022.log` exit 0；候选
`p4_candidate_fsdp2_trace_20261022.log` exit 1，仍在 step61、相同 grad norm
数值失败，表明 observer 未消除或改变该失败点。
快照目录为 `outputs/baseline-trainer-fsdp2-trace-20261022/alignment-capture/rank*/`
和 `outputs/candidate-trainer-fsdp2-trace-20261022/rank*/`。

独立比较工具 `outputs/qwen3-8b-npu/compare_training_traces.py` exit 0，
日志 `p4_trace_compare_20261022.log`；完整 top8/step/category 摘要存于候选
目录下 trace-comparison.json。rank0 关键观测：

| 步数 | 更新后参数最大 AE | 裁剪后梯度最大 AE |
| --- | --- | --- |
| 1 | 9.288e-6（layer0 V） | 2.981e-8（embedding） |
| 10 | 9.277e-6（layer0 V） | 1.158e-7（embedding） |
| 30 | 9.215e-6（layer0 V） | 1.749e-7（embedding） |
| 60 | 8.893e-5（layer2 Q） | 5.593e-5（embedding） |
| 61 | 4.012e-4（layer2 Q） | 5.670e-3（embedding） |

首步 Q/K/V 更新已有稀疏偏差，尽管梯度最大误差很小（V 3.260e-9）。
因此 step61 不是差异首次出现的位置，而是放大点。尚不能将差异归因于
FSDP、fused QKV、近零梯度的 Adam epsilon 敏感性或算子公式中的任何单一项；
下一步需检查首步 top-difference 坐标的原始/裁剪梯度和 Adam 状态，并测试
语义匹配的非 fused-QKV 基线诊断，不能调大 eps 或阈值来消除失败。

本轮 Ruff、格式和 diff 检查通过；没有生产实现修复，双卡 P4 仍失败。

### 首步 Adam 坐标解析与非 fused QKV 对照

`outputs/qwen3-8b-npu/inspect_first_update.py` 对首步 top 差异坐标逐元素计算
Adam 首步公式（lr=.0004、eps=1e-8）。layer0 V 的 flat index240638：
参考裁剪梯度=-5.397663205e-9，候选=-4.866144820e-9；公式预测更新差
-9.288062030e-6，实际=-9.288080037e-6。其他 top 坐标同样吻合。
日志 `p4_first_update_coordinates_20261022.log` exit0，完整9个坐标保存在
`outputs/candidate-trainer-fsdp2-trace-20261022/first-update-coordinates.json`。
这解释了近零梯度差异为何被 Adam 放大，不证明原始梯度差异的根因。

基线采集新增可选 `LLMTUNER_ALIGNMENT_INITIAL_HF`，通过真实 adapter 和
目标 DTensor placements 加载指定全局初始状态，并逐参数零容差验证。
显式选用官方 debugmodel_non_fused_qkv 架构，其他配置保持双卡100步基线一致。
隔离配置入口 `baseline_nonfused_probe`，生产及基线源码均未修改。
首次探针由于 adapter.from_hf 会原地补充共享 lm_head 键而导致验证 KeyError；
改为给 adapter 传字典浅拷贝，不跳过共享权重验证。失败日志保留。

非 fused 基线100步 exit0：`p4_baseline_fsdp2_nonfused_retry1_20261022.log`。
实际复核 rank0 的全部初始权重、100批输入及标签与原 fused 基线精确一致。
候选完整回放仍在 step61 失败，AE=1.3799533844、RE=0.1090373551；
`p4_candidate_fsdp2_nonfused_20261022.log` exit1。不是修复或通过。
产物分别在 `outputs/baseline-trainer-fsdp2-nonfused-retry1-20261022/`
和 `outputs/candidate-trainer-fsdp2-nonfused-20261022/`。

因此“仅拆分基线 fused QKV 即消除偏差”的假设被本对照否定；仍须检查具体
梯度算子、FSDP 归约和训练后期放大机制。不得调大 Adam eps 或既定容差。
复现沿用上一节命令，基线改 `--config baseline_nonfused_probe`，设置
`LLMTUNER_ALIGNMENT_INITIAL_HF` 为原基线 rank0/initial-hf.pt，采用上述新目录。
全量 Ruff 与 diff 检查通过；整体及双卡 P4 保持未完成/失败。

### 最新质量门禁与重复性核查

- 当前工作区完整 CPU suite：4073 passed、13 skipped、273 warnings，exit0，
  `cpu_full_latest_20261022.log`；不替代 NPU 硬件测试。
- 全量 Ruff check、compileall（llmtuner 及全部 integration scripts）和 diff
  check 通过。全量 format check 仍只有未修改的
  `tp_ep_cp_training_equivalence.py` 存在历史格式差异，不自动改无关文件。
- 独立加载无 trace 和带 trace 的两次真实 baseline 双卡100步产物比较：
  初始权重、全部原始逐步 metrics、最终参数全部精确一致；
  `p4_baseline_repeatability_20261022.log` exit0。因此当前证据不支持将失败
  简单归因于 baseline 自身随机不确定性。
- 回放增加显式 `LLMTUNER_ALIGNMENT_NORM_BACKEND` 诊断选项，默认 native
  不变。尝试 aten 的双卡运行失败，生产后端明确拒绝 NPU：
  `model.norm_backend='aten' currently requires CPU tensors`，日志
  `p4_candidate_fsdp2_aten_norm_20261022.log` exit1。这是未满足后端设备前提，
  不构成 norm 合并方式的数值证据；没有删除该 guard 或回退到 CPU。

双卡跨框架差异仍待定位；质量门禁和重复性证据不能替代其数值通过。

### 残差 norm view 合并边界诊断

回放脚本新增诊断变量 `LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS=1`：仅按函数
identity 删除 `_isolate_norm_input` prehook 及对应 kwargs 注册，断言共16个。
生产实现与默认回放不变，不删除其他 hook，不改变阈值或优化器。

使用原 fused baseline 双卡100步采集，设备6、7，完整 Trainer 回放后验收
exit1：step61 grad_norm mean_ae=0.08670711517333984，
mean_re/max_re=0.006328539451415675，仍超出0.02/0.005。
相较保留 view 的 AE=0.3347711563，本诊断偏差较小，但没有通过，
不能据此删除生产 hook 或认定其为唯一根因。

复现（容器内仓库根目录）：

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=6,7 \
LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS=1 \
torchrun --standalone --nproc_per_node=2 \
  tests/integration_tests/npu_trainer_alignment_replay.py \
  --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture \
  --output outputs/candidate-trainer-fsdp2-no-norm-views-20261022
```

日志：`outputs/qwen3-8b-npu/logs/p4_candidate_fsdp2_no_norm_views_20261022.log`。
目标脚本 Ruff check/format check 与 git diff check 通过。下一步需对照无 view
首步梯度与更新坐标，区分累加顺序影响和其余算子差异；双卡 P4 仍失败。

#### 无 view 快照复验

同一命令增加 `LLMTUNER_ALIGNMENT_TRACE_STEPS=1,10,30,60,61`，输出改为
`outputs/candidate-trainer-fsdp2-no-norm-views-trace-20261022`。完整100步后
仍以完全相同的 step61 grad_norm 指标失败（exit1），证明快照未改变该失败。
日志 `p4_candidate_fsdp2_no_norm_views_trace_20261022.log`。

对比 rank0 全局快照（比较脚本 exit0）：

| step | 最大参数 AE | 最大裁剪梯度 AE |
| --- | --- | --- |
| 1 | 2.1873042e-5（layer0 V） | 3.7252903e-8（embedding） |
| 30 | 2.1808315e-5（layer0 V） | 1.2782402e-7（embedding） |
| 60 | 3.4845434e-5（layer2 Q） | 1.8958002e-5（embedding） |
| 61 | 3.6514178e-4（layer2 Q） | 1.9508451e-3（embedding） |

无 view 的首步最大参数差异反而高于保留 view 的9.288e-6，但后期梯度偏差
较小，因此不能用首步最大参数差异单一指标预测后期对齐。
layer0 V flat index218136：参考梯度1.0813654416e-9，候选1.7961665488e-9；
Adam公式预测更新差-2.1873135075e-5，实际-2.1873041987e-5。
仍符合近零梯度被 Adam epsilon 放大的机制，不证明原始梯度差异根因。

比较脚本已支持 `--candidate`，分别运行
`outputs/qwen3-8b-npu/compare_training_traces.py` 和
`outputs/qwen3-8b-npu/inspect_first_update.py`，参数为上述输出的 `rank0`。
日志 `p4_no_norm_views_trace_compare_20261022.log`、
`p4_no_norm_views_first_update_20261022.log`；JSON结果保存在该输出根目录。
本轮仅诊断，无生产修复；保留双卡 P4 失败，后续应转向其他独立缺口，避免
反复以 norm view 单因素试错替代整体适配进展。

### PP2 × DP-shard2：独立四卡组合补验

`pp_equivalence.py` 增加非空洞断言：当 dp_shard>1 时，每个 rank 至少有一个
DTensor 参数的本地元素数小于全局元素数，防止仅构造 mesh 而未实际分片。
保持原先 tol=1e-5、4 steps、global batch8、seq32、4 microbatches，比较真实
Trainer 的逐步 loss、grad norm、全部裁剪梯度与更新后参数，对照相同 DP
microbatch 边界的未流水参考。

| 配置 | 状态 | exit | loss最大AE | 梯度最大AE | 参数最大AE |
| --- | --- | --- | --- | --- | --- |
| PP2×native FSDP2，1F1B | 通过 | 0 | 0 | 3.725e-8 | 7.339e-7 |
| PP2×native FSDP2，Interleaved1F1B | 通过 | 0 | 4.768e-7 | 9.872e-8 | 2.119e-7 |
| PP2×custom FSDP2，1F1B | blocked | 1 | 未训练 | 未训练 | 未训练 |

custom 在装配阶段明确拒绝：需要 torch2.14 multiple parameter groups 和
iteration reset API，当前2.12缺失。未绕过 guard，native 结果不替代 custom。
两项 native 仍有 FSDPIdentity 未执行 forward 的已有警告，数值断言通过但
警告未修复，不推断其他布局同样安全。运行存在 `_assert_async.msg` CPU
fallback 和 ArgSort AiCPU 警告；不宣称完全无主机回退或性能验收通过。

容器内复现：

```bash
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/pp_equivalence.py 1F1B
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/pp_equivalence.py Interleaved1F1B
```

custom 复现仅将实现变量改为 custom。日志目录统一为
`outputs/qwen3-8b-npu/logs/`，文件为
`pp2_native_fsdp2_1F1B_asserted_20261022.log`、
`pp2_native_fsdp2_Interleaved1F1B_asserted_20261022.log`、
`pp2_custom_fsdp2_20261022.log`。目标脚本 Ruff check/format check、diff check
通过。本轮未改变 chunked×FSDP 拒绝或生产实现；整体适配仍未完成。

### PP 边界 Identity FSDP 单元修复

根因：PP 将不属于本阶段的 embedding/norm/head 替换为 `nn.Identity`，FSDP
边界规划仅检查非 None，导致把未执行 forward 的占位对象加入联合分片单元。
`apply_fsdp_to_decoder` 现排除精确 `nn.Identity`，按剩余模块构造边界单元，
保留其他模块和自定义子类，不改变共享参数归属及梯度归约策略。

新增4个 CPU 回归（tied/untied × 首/尾阶段）；目标文件 suite 为115 passed、
7 skipped，exit0，日志 `fsdp_identity_unit_20261022.log`。
真实四卡 PP2×native FSDP2 重跑1F1B和Interleaved1F1B，均exit0，新增分片断言
成立，日志中不再有 `did not run forward before backward` 警告。loss/grad norm/
全参数梯度及更新检查保持原1e-5阈值，报告最大误差与修复前一致：
1F1B 参数7.339e-7，Interleaved参数2.119e-7。

复现使用上一节两条命令；修复后日志：
`pp2_fsdp2_identity_fixed_1F1B_20261022.log` 和
`pp2_fsdp2_identity_fixed_Interleaved1F1B_20261022.log`。
仅消除本项警告，不宣称消除所有NPU fallback或其他布局已通过。
全量 Ruff check、compileall、diff check通过；修改文件格式通过。全量格式
仍被未修改的 `tp_ep_cp_training_equivalence.py` 历史格式差异阻挡，未改该文件。
修复后完整 CPU suite exit0：4077 passed、13 skipped、273 warnings，141.55s，
日志 `outputs/qwen3-8b-npu/logs/cpu_fsdp_identity_full_20261022.log`。

### 八卡 PP2×CP2×DP-shard2 组合

沿用真实 NPU SDPA CP 路径，native FSDP、FP32、seq256、global batch8、
4 steps、4 microbatches，原tol=1e-5。两种调度全部逐步loss/grad norm、
裁剪梯度及参数更新与独立无流水参考比较通过，exit0。实际FSDP分片断言成立。

| 调度 | loss最大AE | 梯度最大AE | 参数最大AE | 日志 |
| --- | --- | --- | --- | --- |
| 1F1B | 4.768e-7 | 4.703e-8 | 3.191e-6 | `pp2_cp2_dp_shard2_20261022.log` |
| Interleaved1F1B | 4.768e-7 | 3.539e-8 | 1.565e-6 | `pp2_cp2_dp_shard2_interleaved_20261022.log` |

复现（容器内仓库根目录，日志均在约定目录）：

```bash
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native
torchrun --standalone --nproc_per_node=8 \
  tests/integration_tests/pp_cp_equivalence.py 1F1B
torchrun --standalone --nproc_per_node=8 \
  tests/integration_tests/pp_cp_equivalence.py Interleaved1F1B
```

共享测试入口现打印实际world/dp_shard/dp_replicate/cp及FSDP实现；第二项明确
输出 `world=8 dp_shard=2 dp_replicate=1 cp=2 fsdp=native`。新增测试环境变量
`LLMTUNER_TEST_DP_REPLICATE_SIZE`（默认1）可独立选择复制DP，不改变生产配置。
仍存在 `_assert_async.msg` CPU fallback/ArgSort AiCPU 警告；上述结果仅支持
该组合正确性，不支持完全无回退、吞吐达标或其他DP布局。脚本Ruff/格式与diff
检查通过。

### 八卡 PP2×CP2×DP-replicate2：新发现的通信失败

在上节1F1B命令上增加 `LLMTUNER_TEST_DP_REPLICATE_SIZE=2`，实际拓扑为
world8/pp2/cp2/dp_replicate2/dp_shard1，日志确认采用native HSDP。exit1，
在 `pp_equivalence.py` 收集初始权重的 DTensor.full_tensor→cpu 阶段失败，
尚未开始训练，不计为训练或数值通过。

HCCL报 `EI0005`：同一 HcomAllGather group_name_121 的 count 在rank0为4096、
对端为16384；rank5同时报告相反count。指向collective调用或组成员不一致，
尚未证明是测试遍历顺序、重建HSDP mesh还是生产装配问题，不能直接归因为环境
或调整超时掩盖。日志：
`outputs/qwen3-8b-npu/logs/pp2_cp2_dp_replicate2_20261022.log`。

下一步先独立打印每rank的PP/CP/replicate坐标、实际FSDP mesh/group成员和
state_dict遍历名称/形状，定位首个collective不一致点，再决定最小修复。
该配置明确记为失败，与已通过的两项DP-shard配置分开保留。

### HSDP×CP 通信组重建修复

诊断复验 `pp2_cp2_dp_replicate2_mesh_trace_20261022.log` exit1，再次触发初始
权重AllGather失败。逐rank输出表明PP0参数mesh为[[0,1],[2,3]]，PP1为
[[4,5],[6,7]]，局部组成员和权重形状各自看似正确；错误发生在不同阶段的
参数间（4096与16384），不支持仅通过排列state_dict规避问题。

定位到 `resolve_fsdp_mesh` 在replicate×shard×cp分支按每个PP子网的局部
rank表重新构造DeviceMesh。当前torch的构造会进入全局进程组创建流程，不同
PP子网传入不同rank表会造成通信组创建顺序/标识冲突。现用已有replicate组和
扁平化shard×cp组调用 `DeviceMesh.from_group`，不重建独立局部组；布局语义不变。
CPU回归明确断言复用两个组、扁平rank矩阵和轴名；目标suite116 passed、
7 skipped，exit0，`hsdp_cp_group_reuse_unit_20261022.log`。

同一八卡失败配置修复后：

| 调度 | 状态/exit | loss最大AE | 梯度最大AE | 参数最大AE |
| --- | --- | --- | --- | --- |
| 1F1B，诊断输出启用 | 通过/0 | 4.768e-7 | 4.284e-8 | 3.215e-6 |
| Interleaved1F1B，诊断输出关闭 | 通过/0 | 4.768e-7 | 2.980e-8 | 1.570e-6 |

真实Trainer4steps、seq256、global batch8、4microbatches及tol=1e-5均未改变。
日志 `pp2_cp2_dp_replicate2_group_reuse_20261022.log`、
`pp2_cp2_dp_replicate2_group_reuse_interleaved_20261022.log`。
复现沿用上一节复制DP命令，调度选择上述两项；诊断输出只由
`LLMTUNER_TEST_TRACE_STATE_MESH=1`开启，不改变collective调用。
保留原失败证据；通过仅覆盖该布局，不替代全部DP组合。

受影响路径回归：无CP的四卡native HSDP4steps exit0，参数最大AE=5.960e-8，
各rank的local-vs-global梯度差均>0.03，非空洞断言通过；日志
`hsdp_no_cp_group_reuse_regression_20261022.log`。完整CPU suite exit0：
4078 passed、13 skipped、273 warnings（139.54s），日志
`cpu_hsdp_cp_group_reuse_full_20261022.log`。全量Ruff check、修改文件format
check、compileall和diff check通过。全量format/pre-commit历史缺口未在本轮
宣称修复；整体适配仍未完成。

### 复制 DP 与 TP 布局拆分验证

四卡 PP2×DP-replicate2（CP1，native，1F1B）真实Trainer4steps通过，exit0，
tol=1e-5，loss最大AE=0、梯度3.725e-8、参数7.339e-7。复现：

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_DP_REPLICATE_SIZE=2 \
LLMTUNER_TEST_FSDP_IMPLEMENTATION=native \
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/pp_equivalence.py 1F1B
```

日志 `pp2_dp_replicate2_20261022.log`，实际拓扑输出cp1/dp_shard1/dp_replicate2。

`tp_fsdp_equivalence.py` 增加测试专用变量 `TP_FSDP_EQ_DP_SHARD`（默认2）、
`TP_FSDP_EQ_DP_REPLICATE`（默认1），参考路径按两者乘积划分独立DP序列。
复制布局断言实际DTensor replicate轴和placement；分片布局保留小本地shard
断言，纯复制禁止出现小本地shard，所有布局保留TP切片差异非空洞断言。

四卡 TP2×DP-replicate2 真实Trainer4steps通过，exit0，原tol=1e-4，
loss最大AE=4.768e-7、参数1.478e-6。复现：

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
TP_FSDP_EQ_DP_SHARD=1 TP_FSDP_EQ_DP_REPLICATE=2 \
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/tp_fsdp_equivalence.py
```

日志 `tp2_dp_replicate2_20261022.log`，均在约定日志目录。
上述仅为对应配置的正确性证据，不代表DP×所有路径完成。

八卡TP2×HSDP（dp_shard2/dp_replicate2）同脚本4steps exit0：loss最大AE=
4.768e-7、参数2.123e-6，tol=1e-4。实际replicate placement和参数分片断言
同时成立；日志 `tp2_hsdp2x2_20261022.log`。复现将上述变量设为2/2，设备0–7，
`--nproc_per_node=8`，其余不变。
默认四卡TP2×FSDP-shard2回归exit0：loss最大AE=4.768e-7、参数1.478e-6，
日志 `tp2_fsdp2_layout_regression_20261022.log`；变量2/1、设备4–7、进程4。
Ruff全量check、目标format、compileall和diff检查通过。仅修改集成测试入口，
本轮未重复CPU全套（上轮完整结果4078 passed/13 skipped仍保留，但不冒充本轮
重新执行）。运行仍有NPU assert CPU fallback，不宣称纯设备路径性能验收。

### EP2 × 复制DP/HSDP 配置补验

`ep_fsdp_equivalence.py` 新增 `--dp-replicate {1,2}`（默认1），支持4/8rank，
按实际world构造独立输入和dense参考，检查expert mesh大小为world/EP、dense
mesh大小为world，保留精确mesh成员断言。复制配置同时断言两类参数的
dp_replicate轴大小及Replicate placement，不放宽tol=1e-4。
新增loss、router/shared-Q/expert梯度误差有限性断言，避免NaN被`diff > TOL`
漏判。覆盖真实HF参考前反向、EP AllToAll与native FSDP，非完整Trainer多步。

首次四卡探针在最终打印使用不存在的cfg.dp_shard，exit1，日志
`ep2_dp_replicate2_shard2_20261022.log`保留，不计通过。改用实际parallel_dims
字段后，四卡复制、八卡HSDP、默认四卡三项均exit0，随后对新增有限性断言
执行最终复验，最终证据单列，不能把此前运行冒充最终脚本结果。

最终脚本三项均exit0，loss/梯度有限，router及三类expert权重梯度均满足原
1e-4容差，所有rank失败计数0：

| world | dense shard/replicate | sparse efsdp | loss实际/参考（日志6位） | 日志 |
| --- | --- | --- | --- | --- |
| 4 | 2/2 | 1 | 311.874298 / 311.874283 | `ep_hsdp_four_finite_20261022.log` |
| 8 | 4/2 | 2 | 621.338318 / 621.338287 | `ep_hsdp_eight_finite_20261022.log` |
| 4 | 4/1 | 2 | 311.874268 / 311.874290 | `ep_hsdp_default_finite_20261022.log` |

复现（容器内仓库根目录）：

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/ep_fsdp_equivalence.py --dp-replicate 2
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  tests/integration_tests/ep_fsdp_equivalence.py --dp-replicate 2
```

默认回归用第一条命令改复制度为1。日志均在约定目录。全量Ruff、目标格式、
compileall与diff检查通过。本轮仅修改集成测试，未重跑完整CPU suite；之前
4078 passed/13 skipped不是本轮结果。仍有NPU assert CPU fallback，不宣称
完全无回退。EP多步优化器/恢复、TorchAO完整MoE及其他未验配置仍不由此覆盖。

### TorchAO完整MoE：容量尾部裁剪修复

真实四卡EP2×FSDP4探针首先因PYTHONPATH遗漏既有triton隔离环境而exit1，
`torchao_full_moe_fsdp4_20261022.log`；加回outputs/baseline-runtime后触发
真实生产断言 `Expert token counts must sum to the number of routed tokens`，
`torchao_full_moe_fsdp4_retry1_20261022.log` exit1，不能将此前dispatcher的
独立解析测试通过上推为完整专家计算已通过。

根因：真实TorchAO0.18 `permute_and_pad`按最坏情况分配buffer，padded counts
只覆盖前面的实际专家分段，尾部为额外sentinel容量。原适配将整个buffer传给
GroupedExperts，违反其counts精确覆盖输入约束。现按padded counts之和裁剪
buffer和indices，保留输入shape与sentinel逆置换；不删除专家侧断言、不补造
专家计数、不更换算子。取sum.item引入主机同步，明确不宣称捕获/编译无同步。

CPU回归模拟额外8行未使用容量，检查覆盖、combine和输入梯度，目标suite
33 passed，exit0（`torchao_capacity_unit_20261022.log`）。真实NPU复验：

| 配置 | 状态/exit | loss实际/参考（6位） | 日志 |
| --- | --- | --- | --- |
| 四卡EP2×FSDP4，TorchAO | 通过/0 | 311.874298 / 311.874298 | `torchao_full_moe_fsdp4_capacity_fixed_20261022.log` |
| 四卡EP2×HSDP（shard2/replicate2），TorchAO | 通过/0 | 311.874268 / 311.874290 | `torchao_full_moe_hsdp4_capacity_fixed_20261022.log` |
| 双卡dispatcher随机/偏斜/空专家 | 通过/0 | 输出/输入梯度/score梯度与解析参考一致 | `torchao_dispatch_capacity_fixed_20261022.log` |

两项完整MoE使用FP32专家F.linear路径，真实TorchAO padded dispatch、NPU专家
计算、HF参考、router和三类expert权重梯度，原tol=1e-4且数值有限，断言真实
dispatcher类型。它们不是FP8/MXFP8量化grouped GEMM，也不是多步Trainer验收。
双卡dispatcher保持1e-5容差，新增返回行数必须等于counts之和的断言。

容器内复现：

```bash
export PYTHONPATH=.:outputs/torchao-runtime:outputs/baseline-runtime
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/ep_fsdp_equivalence.py --dispatcher torchao
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/ep_fsdp_equivalence.py --dispatcher torchao --dp-replicate 2
ASCEND_RT_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
  tests/integration_tests/torchao_dispatch_equivalence.py
```

全量CPU exit0：4079 passed、13 skipped、273 warnings，149.83s，日志
`cpu_torchao_capacity_full_20261022.log`。全量Ruff、目标格式、compileall和diff
检查通过；既有全量format/pre-commit缺口未宣称解决。所有日志位于约定目录。
原失败证据保留，整体适配仍未完成。

### TorchAO padded dispatch 的真实多步 Trainer 验证

`pp_ep_equivalence.py` 增加显式 `--dispatcher torchao`，默认alltoall不变。
真实Trainer执行PP2/EP2/native FSDP装配、synthetic数据、4个microbatches、
4次AdamW更新；各step比较loss、全部裁剪梯度和更新参数，独立参考仍采用
EP1非padded dispatcher，初始权重从实际阶段状态重建。断言真实TorchAO类型，
loss有限，保留原atol/rtol=1e-4，不改变优化器或精度（FP32）。

无AC路径exit0：实际loss为4.868320/4.865578/4.868018/4.877336，参考
4.868320/4.865579/4.868017/4.877336（日志6位）。
`pp_ep_torchao_trainer_20261022.log`。默认alltoall同脚本回归exit0，
`pp_ep_default_dispatch_regression_20261022.log`。

Full/selective首轮均exit0后，增加CheckpointWrapper断言和decoder内部前向
计数；计数必须至少为steps×microbatches×本地层数×2，不能仅凭配置或多1次
shape推导证明重计算。两种模式诊断均报告每rank33次（正常forward基数16，
另含shape推导），保持同一loss/梯度/参数断言。最终强化计数脚本复验结果单列。

容器内复现：

```bash
export PYTHONPATH=.:outputs/torchao-runtime:outputs/baseline-runtime
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/pp_ep_equivalence.py --dispatcher torchao
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/pp_ep_equivalence.py --dispatcher torchao \
  --activation-checkpoint full
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/pp_ep_equivalence.py --dispatcher torchao \
  --activation-checkpoint selective
```

仅扩展小型真实Trainer配置，不替代checkpoint恢复、Qwen3-8B MoE或FP8/MXFP8
量化grouped GEMM。运行仍有NPU assert CPU fallback/ArgSort AiCPU警告。
最终full/selective强化计数复验均exit0、decoder_calls=33、所有参数与梯度断言
通过，日志 `pp_ep_torchao_full_final_20261022.log` 和
`pp_ep_torchao_selective_final_20261022.log`，均在约定目录。全量Ruff check、
目标格式、compileall和diff check通过。本轮仅改集成测试，未重跑CPU全套；
此前4079 passed/13 skipped不作为本轮重新执行结果。整体目标仍未完成。

### TorchAO真实Trainer DCP恢复

复用 `pp_checkpoint_equivalence.py` 既有dispatcher配置，不替换训练循环。
四卡NPU、tiny Qwen3 MoE、FP32、动态EMA，连续4steps在step2保存完整DCP；
新torchrun进程恢复step2并执行后2steps，逐rank精确比较模型、AdamW状态、
scheduler、trainer step/token counters、EMA及最后两步loss。

| 配置 | full/resume exit | loss最大AE | 日志前缀 |
| --- | --- | --- | --- |
| PP2×EP2/native FSDP，full AC | 0/0 | 0 | `torchao_pp_ep_checkpoint_` |
| PP1×EP2/native FSDP4，selective AC | 0/0 | 0 | `torchao_ep_fsdp_checkpoint_` |

日志后缀分别为 `full_20261022.log` / `resume_20261022.log`，位于约定目录。
产物 `outputs/torchao-pp-ep-resume-20261022`、
`outputs/torchao-ep-fsdp-resume-20261022`。两项均严格检查dense/sparse mesh
归属，未把仅保存成功视为恢复通过。前两项未开启RNG/loader快照显式比较，
不能由loss精确一致推断那些状态已单独验收；增强状态运行另列。

容器内复现：

```bash
export PYTHONPATH=.:outputs/torchao-runtime:outputs/baseline-runtime
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 LLMTUNER_TEST_WORLD_SIZE=4
export LLMTUNER_TEST_PP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_MOE=1
export LLMTUNER_TEST_DISPATCHER=torchao LLMTUNER_TEST_FSDP_IMPLEMENTATION=native
export LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic
for phase in full resume; do
  torchrun --standalone --nproc_per_node=4 \
    tests/integration_tests/pp_checkpoint_equivalence.py "$phase" \
    outputs/torchao-pp-ep-resume-20261022
done
```

无流水配置改PP_SIZE=1、AC=selective，并使用上述另一独立产物目录。
本轮未修改生产或测试实现，未重跑CPU全套；整体适配仍未完成。

增强状态复验：无流水EP2×FSDP4/selective AC/动态EMA配置另设
`LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1`，产物目录
`outputs/torchao-ep-fsdp-resume-state-20261022`。full/resume均exit0，
`torchao_ep_fsdp_state_full_20261022.log` 与
`torchao_ep_fsdp_state_resume_20261022.log`。末两步loss精确一致，所有rank模型、
optimizer、scheduler、EMA、trainer、显式rng及dataloader状态比较全部通过。
额外读取4个final-state确认各rank均有非空rng（含python/numpy/cpu/device）及
loader（含steps/batch_layout）字段，未因选项遗漏变成空洞检查。该额外读取
未初始化PG，产生DeviceMesh反序列化警告，不能用于compile验收；真正恢复检查
在torchrun初始化后的Trainer内完成。diff check通过。

### 双轴CP KV2×Ulysses2 NPU前反向补验

`dual_cp_attention_equivalence.py` 从硬编码CPU/Gloo改用既有设备harness。
CPU保持FP64/atol=rtol=1e-10；NPU执行前固定FP32/atol=rtol=1e-5，与已有
CP SDPA目标设备策略一致，非失败后放宽。各rank使用相同CPU随机输入再搬到
本地设备。真实SeqToHead、KV all-gather、HeadToSeq及其反向保持生产实现；
仅将run_flex显式替换为目标NPU SDPA，并用独立全序列SDPA作参考。

四卡NPU九场景（noncausal/causal/packed × Q/KV头数4/4、4/2、6/2）全部
输出与Q/K/V梯度比较通过，exit0。梯度额外断言非空、有限、非零，GQA比例3
包含Ulysses后奇数query-head场景。日志
`outputs/qwen3-8b-npu/logs/dual_cp_sdpa_npu_20261022.log`。

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/dual_cp_attention_equivalence.py
```

此证据仅证明双轴通信与显式SDPA算术组合，不证明生产flex mask/内核或整网
Trainer双轴训练通过。本轮未重跑CPU/Gloo分布式场景，也未重跑CPU单测全套。
全量Ruff、目标格式、compileall、diff检查通过，无生产修改；整体仍未完成。

### PP×Ulysses 真实Trainer与双轴拒绝边界

共享PP测试新增 `LLMTUNER_TEST_ULYSSES_SIZE` 与 `LLMTUNER_TEST_CP_STRATEGY`，
默认不变，并打印cp_kv/cp_ulysses实际拓扑。四卡PP2×Ulysses2、KV1、native
FSDP、FP32、seq256/global batch8/4microbatches/4steps，通过真实NPU SDPA
生产入口运行，不用attention替代或mock。每步loss、grad norm、全部裁剪梯度
和AdamW参数与独立无流水参考比较，保持原atol/rtol=1e-5：

| 调度 | exit | loss最大AE | 梯度最大AE | 参数最大AE | 日志 |
| --- | --- | --- | --- | --- | --- |
| 1F1B | 0 | 4.768e-7 | 6.566e-8 | 4.723e-6 | `pp2_ulysses2_trainer_20261022.log` |
| Interleaved1F1B | 0 | 4.768e-7 | 6.706e-8 | 7.384e-6 | `pp2_ulysses2_interleaved_20261022.log` |

复现（容器内根目录）：

```bash
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_CP_SIZE=1
export LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_CP_STRATEGY=ulysses
export LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/pp_equivalence.py 1F1B
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/pp_equivalence.py Interleaved1F1B
```

另将world/进程设8、设备0–7、CP_SIZE=2，保留ULYSSES_SIZE=2，实测NPU SDPA
整网双轴在装配阶段明确抛错（exit1）：`only with a single CP axis`，无静默
降级。日志 `pp2_dual_cp_sdpa_rejection_20261022.log`；这是支持边界证据，
不是双轴训练通过。解锁需要实现并验证双轴mask语义的NPU生产kernel入口。
上一节双轴通信替代验证不能解锁此项。各日志均在约定目录；仍有assert CPU
fallback/ArgSort AiCPU提示。全量Ruff、目标format、compileall、diff通过，
本轮仅改测试配置入口，未重复CPU全套。整体适配未完成。

### packed PP×CP恢复探针：新增失败证据

四卡PP2×KV-CP2、真实JSONL packing、full AC、动态EMA、显式RNG/loader状态
比较探针在装配阶段exit1：`CPSdpaKernel`明确拒绝packed document mask。
日志 `packed_pp2_cp2_state_full_20261022.log`，产物目录
`outputs/packed-pp2-cp2-resume-state-20261022`。未训练、未保存、不启动resume，
不因其他CP/packing各自通过就视为组合通过。

恢复测试入口新增与训练等价入口相同的ULYSSES_SIZE/CP_STRATEGY变量（默认
不变）。切换受支持的Ulysses mask路径后，四卡PP2×Ulysses2真实packed训练
仍exit1：SDPA报告512与256的维度不匹配，发生在训练阶段、checkpoint前。
日志 `packed_pp2_ulysses2_state_full_20261022.log`，产物目录
`outputs/packed-pp2-ulysses2-resume-state-20261022`，未执行resume。

复现（容器根目录）：

```bash
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=2
export LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_ULYSSES_SIZE=2
export LLMTUNER_TEST_CP_STRATEGY=ulysses LLMTUNER_TEST_SEQ_LEN=256
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=full
export LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1
export LLMTUNER_TEST_LOADER_STATE=1
torchrun --standalone --nproc_per_node=4 \
  tests/integration_tests/pp_checkpoint_equivalence.py full \
  outputs/packed-pp2-ulysses2-resume-state-20261022 --packed
```

KV拒绝复现改CP_SIZE=2、ULYSSES_SIZE=1、CP_STRATEGY=kv_allgather及对应产物目录。
初步代码证据：`HFTransformerModel._shard_for_cp`对非flex丢弃预构建mask，
`_apply_attention`随后从局部positions重建packed Ulysses mask，而head交换后
attention长度为全局长度。须继续核对PP微批次mask传递并加独立document-boundary
断言，不能仅调整维度/关闭packed guard使测试跑通。本轮无生产修复，目标脚本
Ruff/format与diff检查通过，未重跑CPU全套。整体未完成。

### packed PP×Ulysses 全局 mask 修复与恢复复验（2026-10-22）

本轮继续适配目标，上一轮仅优化 goal.md，不构成硬件适配进展。本轮候选
HEAD 为 `c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19`，基线仍为
`b358a8063c34aa7494397efa3a8f5ad24f1caa1f`；保留工作区原有改动。

根因确认：Ulysses head swap 后 attention 消费全序列，非 flex 路径丢弃
全局文档 mask 后从局部 positions 重建，导致维度及文档语义错误。
仅保留旧 BlockMask 的初次修复不成立：CPU 单测1失败81通过，硬件full退出1，
日志 `packed_ulysses_mask_unit_20261022.log`、
`packed_ulysses_mask_fixed_full_20261022.log`。现于分片前构造全长dense mask，
forward 原样传递；不移除 KV packed 或双轴拒绝。

| 测试项 | 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- | --- | --- |
| 全局mask CPU回归 | HF wrapper，跨文档/未来位置屏蔽、对象传递 | 形状8×8及精确断言 | 82 passed | 通过 | 0 | `packed_ulysses_mask_unit_retry1_20261022.log` |
| packed Ulysses生产SDPA | 2卡FP32、seq256、doc lengths90/1/165、2层Qwen | 独立dense参考，atol/rtol1e-5 | logits maxAE2.570e-7，mean loss AE0，参数梯度maxAE2.980e-8，去文档mask差0.6232 | 通过 | 0 | `packed_ulysses_dense_equivalence_20261022.log` |
| packed DCP连续训练 | 4卡PP2×Ulysses2、KV1、native、full AC、dynamic EMA、RNG/loader开启 | 真实Trainer4步，保存step2 | loss4.885441→4.530151→4.375731→4.284118 | 通过 | 0 | `packed_ulysses_dense_retry1_full_20261022.log` |
| packed DCP恢复 | 同上、2+2步 | 与连续训练精确一致 | 恢复前状态检查各rank0失败；最终参数/optimizer/scheduler/EMA/trainer/RNG/loader及loss精确一致 | 通过 | 0 | `packed_ulysses_dense_retry1_resume_20261022.log` |
| CPU完整单测 | 当前工作区 | 原断言 | 4080 passed,13 skipped,273 warnings | 通过 | 0 | `cpu_packed_ulysses_dense_full_20261022.log` |

所有日志位于 `outputs/qwen3-8b-npu/logs/`；恢复产物为
`outputs/packed-pp2-ulysses2-dense-retry1-20261022`。复现（容器根目录）：

```bash
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=2
export LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_ULYSSES_SIZE=2
export LLMTUNER_TEST_CP_STRATEGY=ulysses LLMTUNER_TEST_SEQ_LEN=256
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=full
export LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1
export LLMTUNER_TEST_LOADER_STATE=1
for phase in full resume; do
  torchrun --standalone --nproc_per_node=4 \
    tests/integration_tests/pp_checkpoint_equivalence.py "$phase" \
    outputs/packed-pp2-ulysses2-dense-retry1-20261022 --packed || break
done
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  tests/integration_tests/cp_ulysses_varlen_equivalence.py
pytest -q --disable-warnings
```

独立等价脚本新增设备harness，CPU保持FP64/1e-9，NPU预设FP32/1e-5；NPU不
mock mask或内核，新增所有模型参数梯度比较。Q/K/V通信梯度检查仍为显式SDPA
算术检查，与整网生产验证分开。CPU/Gloo首次回归退出1：旧测试mask包装器未适配
当前PyTorch无`separate_full_blocks`参数；兼容检查按真实函数签名处理，未改阈值。
失败日志 `packed_ulysses_cpu_gloo_regression_20261022.log`，复验另存。

补充复验均exit0：
- 最终NPU等价日志 `packed_ulysses_dense_equivalence_final_20261022.log`，logits
  maxAE2.570e-7、mean loss AE4.768e-7、参数梯度maxAE2.980e-8；Q/K/V梯度AE0。
- CPU/Gloo重跑 `packed_ulysses_cpu_gloo_retry1_20261022.log`，FP64/1e-9原标准
  不变，logits/loss/QKV梯度AE均0；CPU不运行整网参数反向，不把打印的默认0
  作为该项证据，现已调整输出仅在实际执行的NPU分支报告参数梯度。
- 非packed四卡PP2×Ulysses2、4步真实Trainer，1F1B/Interleaved1F1B均保持
  atol/rtol1e-5通过。loss maxAE均4.768e-7，梯度maxAE分别6.566e-8/6.706e-8，
  参数maxAE分别4.723e-6/7.384e-6。日志
  `ulysses_dense_regression_1F1B_20261022.log`、
  `ulysses_dense_regression_Interleaved1F1B_20261022.log`；沿用上方环境变量，运行
  `torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_equivalence.py SCHEDULE`。
- CPU/Gloo复现：`PYTHONPATH=. LLMTUNER_TEST_DEVICE=cpu torchrun --standalone --nproc_per_node=2 tests/integration_tests/cp_ulysses_varlen_equivalence.py`。

全量Ruff check、目标format、compileall、diff检查通过；全量format仍因未修改的
`tests/integration_tests/tp_ep_cp_training_equivalence.py`失败，不修无关文件。
本轮未重跑pre-commit，先前历史失败仍未关闭。NPU日志仍有assert CPU fallback
及ArgSort AiCPU提示；不宣称纯AiCore执行。P4双卡100步对齐等缺口仍在，整体未完成。

### 八卡 packed PP2×Ulysses2×DP-shard2 恢复（2026-10-22）

上一轮修复与硬件复验属于进展。本轮扩展到真实分片DP：八卡、PP2、Ulysses2、
KV1、DP-shard2、native FSDP、FP32、seq256、global batch8、4 microbatches、
真实JSONL packing、dynamic EMA、显式RNG和loader状态比较。环境重新核实为
torch2.12.0+cpu/torch-npu2.12.0、8 NPU；未切分支、未修改依赖或无关代码。

测试增加 `LLMTUNER_TEST_REQUIRE_SHARDED=1` 可选断言：每个本地model part均为
FSDPModule、dp_shard至少2、所有参数均为带Shard placement的DTensor，且本地
numel严格小于全局numel。每个optimizer step后和恢复后都检查，不把CP通信或
FSDP配置字段本身当作实际参数分片证据。默认行为不变。

| 测试项 | 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- | --- | --- |
| packed完整训练 | 8卡PP2/Ulysses2/DP-shard2，1F1B/full AC，2层Qwen | Trainer4步、真实分片断言、step2 DCP | losses4.885098/4.532365/4.374949/4.286737 | 通过 | 0 | `packed_pp2_ulysses2_fsdp2_fullac_full_20261022.log` |
| packed恢复 | 同上，2+2步 | 全rank恢复前/后状态及loss精确一致 | pre-update失败数0，loss最大AE0，参数/optimizer/scheduler/EMA/trainer/RNG/loader均精确一致 | 通过 | 0 | `packed_pp2_ulysses2_fsdp2_fullac_resume_20261022.log` |
| packed完整训练 | 8卡PP2/Ulysses2/DP-shard2，Interleaved1F1B/selective AC，6层Qwen | Trainer4步、真实分片断言、step2 DCP | losses4.814473/4.375145/4.219149/4.152067 | 通过 | 0 | `packed_pp2_ulysses2_fsdp2_interleaved_selective_full_20261022.log` |
| packed恢复 | 同上，2+2步 | 全rank恢复前/后状态及loss精确一致 | pre-update失败数0，loss最大AE0，全部已列状态精确一致 | 通过 | 0 | `packed_pp2_ulysses2_fsdp2_interleaved_selective_resume_20261022.log` |

全部日志位于约定的 `outputs/qwen3-8b-npu/logs/`；命令在容器根目录执行：

```bash
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_PP_SIZE=2
export LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_ULYSSES_SIZE=2
export LLMTUNER_TEST_CP_STRATEGY=ulysses LLMTUNER_TEST_SEQ_LEN=256
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_REQUIRE_SHARDED=1
export LLMTUNER_TEST_AC=full LLMTUNER_TEST_PP_SCHEDULE=1F1B
export LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1
export LLMTUNER_TEST_LOADER_STATE=1
for phase in full resume; do
  torchrun --standalone --nproc_per_node=8 \
    tests/integration_tests/pp_checkpoint_equivalence.py "$phase" \
    outputs/packed-pp2-ulysses2-fsdp2-fullac-20261022 --packed || break
done
```

第二项将AC改selective、调度改Interleaved1F1B，产物目录改
`outputs/packed-pp2-ulysses2-fsdp2-interleaved-selective-20261022`，其余变量相同。
两组总步数均为4，scheduler保持同一总步数；不把不同层数/调度的loss互作数值基线。
AC装配日志证明分别应用full/selective wrapper，本轮未另设replay计数器，不用该
结果替代先前独立AC重计算等价测试。此次仅扩展DCP恢复组合证据，不声称八卡
packed训练与torchtitan-npu跨框架对齐、KV packed、双轴CP或所有组合已通过。

额外复核可选后端：当前torch_checkpointing的metadata聚合仍硬编码CUDA，
CheckpointManager.Config仍无metadata manager注入；多模态HF wrapper仍明确
拒绝media输入。仅检查源码，无新增通过结论，也未全局替换CUDA API或删除guard。
后续仍须解决这些真实缺口与P4双卡100步失败。全量Ruff、目标format、compileall、
diff检查通过；本轮无生产修改。CPU完整复验exit0：4080 passed、13 skipped、
273 warnings、142.50s，日志 `cpu_packed_pp_ulysses_fsdp_full_20261022.log`。
pre-commit本轮未重跑，既有历史门禁失败不记通过；整体未完成。

### P4 RMSNorm 算子对照：排除单独公式替换（2026-10-23）

上一轮八卡恢复验证属于进展，本轮返回双卡100步对齐失败。候选main/HEAD仍为
`c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19`，基线HEAD仍为
`b358a8063c34aa7494397efa3a8f5ad24f1caa1f`，工作区已有改动保留。

局部源码核对：基线配置使用torch RMSNorm，HF Qwen3使用显式FP32平方/均值/
rsqrt公式。回放新增诊断开关 `LLMTUNER_ALIGNMENT_TORCH_RMS_NORM=1`，仅在
测试实例上将33个Qwen3RMSNorm的forward改为`torch.nn.functional.rms_norm`，
参数身份、epsilon、数据、优化器及全部验收阈值不变。生产模型未修改。

初次两项运行均在训练前exit1：测试使用精确type匹配漏掉FSDP动态子类的最终
norm，仅计32个，触发33个的断言。改为isinstance后仍严格断言33个，不删除计数
约束。初始失败日志 `p4_candidate_fsdp2_torch_rms_20261022.log`、
`p4_candidate_fsdp2_torch_rms_no_views_20261022.log`，产物目录保留，重跑另存。

| 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- | --- |
| 2卡真实Trainer100步，33个torch RMSNorm，保留16个residual view hook | 原partial-DTensor基线，grad norm AE≤0.02且RE≤0.005 | 完整训练后step61仍失败，AE0.3347711563、RE0.02443412476 | 失败 | 1 | `p4_candidate_fsdp2_torch_rms_retry1_20261022.log` |
| 同上，显式删除16个view hook的诊断 | 同一基线/阈值 | step61仍失败，AE0.08670711517、RE0.006328539451 | 失败 | 1 | `p4_candidate_fsdp2_torch_rms_no_views_retry1_20261022.log` |
| 两组结果分别与先前对应HF公式运行对比 | 两rank全部100步logits/原始metrics及5个参数/梯度快照精确一致 | 全部精确一致 | 通过（诊断） | 0 | `p4_torch_rms_exact_audit_20261023.log` |

第三项不等于P4通过：它否定“仅替换HF norm公式为torch RMSNorm即可消除当前
偏差”的假设。两rank分别检查step1/10/30/60/61全部参数及梯度快照，没有以
相同失败点代替完整结果比较。原生产路径与移除view的诊断仍是不同配置，不合并
它们的验收结论；不因此移除生产hook或放宽阈值。

复现（容器根目录，日志均在约定目录；运行开始于10月22日UTC，记录于本地10月23日）：

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=6,7 \
LLMTUNER_ALIGNMENT_TORCH_RMS_NORM=1 \
LLMTUNER_ALIGNMENT_TRACE_STEPS=1,10,30,60,61 \
torchrun --standalone --nproc_per_node=2 \
  tests/integration_tests/npu_trainer_alignment_replay.py \
  --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture \
  --output outputs/candidate-trainer-fsdp2-torch-rms-retry1-20261022
```

无view对照追加`LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS=1`，设备4,5，输出改为
`outputs/candidate-trainer-fsdp2-torch-rms-no-views-retry1-20261022`。
精确比较的旧结果目录分别为`outputs/candidate-trainer-fsdp2-trace-20261022`
与`outputs/candidate-trainer-fsdp2-no-norm-views-trace-20261022`，各rank比较
`candidate-step-metrics.json`、`candidate-logits.pt`及上述5个trace-step文件。
各次原始状态及日志保留，后续应转向残差梯度累加/attention布局等尚未排除因素，
不能反复以已被否定的norm公式假设占用硬件。整体目标未完成。

补充离线100步全指标审计：不在首个断言失败后停止，对每rank全部100步logits
复用官方`compare_tensors`/`failed_logits_metrics`，原始loss/grad norm逐步执行
AE≤0.02及mean/max RE≤0.005，各步LR/token计数精确比较。两组各rank均仅发现
step61的grad norm失败，无其他logits/loss/grad norm失败；不代表整体通过。
日志`p4_torch_rms_allstep_audit_20261023.log`，详细失败数据分别保存于两组
产物根目录的`all-step-metric-audit.json`。审计命令exit0表示审计完成，不表示
数值验收通过。全量Ruff、目标format、compileall、diff检查通过；本轮仅增加
诊断脚本开关，未重跑CPU全套/pre-commit，上一轮4080项CPU结果不替代本轮门禁。

### P4 首步模块观测：偏差首先出现于 attention（2026-10-23）

两个真实Trainer observer新增可选`LLMTUNER_ALIGNMENT_MODULE_TRACE=1`，仅记录
第一步各层embedding、input norm、attention、post norm、MLP、block及最终norm
的输出和输出梯度；用tensor.register_hook采集，不使用会插入额外view的module
backward hook。首步完成即移除forward observer，默认训练不变。张量保存为本地
CPU副本，不在hook中执行collective；不把观测耗时作为性能数据。

沿用原双卡FP32/partial-DTensor/seed42/AdamW for-loop/100步配置重跑基线，
exit0，日志`p4_baseline_module_trace_20261023.log`，目录
`outputs/baseline-trainer-module-trace-20261023/alignment-capture/rank*/`。
候选以旧基线真实100批回放、启用同一observer，完整训练后原step61 grad norm
仍失败，AE0.3347711563/RE0.02443412476，exit1；日志
`p4_candidate_module_trace_20261023.log`，目录
`outputs/candidate-trainer-module-trace-20261023/rank*/`。两侧新增
`module-trace-step-1.pt`，比较摘要保存于候选根目录`module-comparison.json`。

独立复核两rank：新旧基线全部初始/最终权重、100批输入/labels/global token
计数、100步logits及原始metrics精确一致；候选新旧100步logits/metrics也精确
一致，日志`p4_module_trace_repeatability_20261023.log`，exit0。由此本轮hook未
改变已观察训练轨迹；不据此证明其他后端或编译模式无影响。

模块比较`p4_module_trace_comparison_20261023.log` exit0（诊断完成，非P4通过）：
rank0 embedding与layer0 input norm输出AE均0，首个非零前向偏差在layer0
attention输出，maxAE2.235174179e-7、meanAE3.126163506e-8；后续post norm
maxAE1.668930054e-6。反向layer0 attention输出梯度maxAE1.490116119e-8，
embedding梯度maxAE3.129243851e-7；rank1 embedding梯度maxAE2.663582563e-7。
这些局部误差仅定位首个观测边界，不证明attention内核为唯一根因，也不以局部
误差小来豁免step61失败。下一步应在同一初始状态继续分解QKV、Q/K norm、RoPE
与inner attention边界，而非继续试验已排除的norm公式。

复现：基线沿用“双侧完整Trainer100步”命令，nproc改2、设备6,7，追加
`--parallelism.spmd-backend partial_dtensor`，dump-folder改上述基线目录，并设置
`LLMTUNER_ALIGNMENT_MODULE_TRACE=1`。候选用同一变量、设备4,5运行：

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4,5 LLMTUNER_ALIGNMENT_MODULE_TRACE=1 \
torchrun --standalone --nproc_per_node=2 \
  tests/integration_tests/npu_trainer_alignment_replay.py \
  --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture \
  --output outputs/candidate-trainer-module-trace-20261023
```

日志均在`outputs/qwen3-8b-npu/logs/`。全量Ruff、两个脚本format、compileall、
diff检查通过；仅修改诊断测试，未重跑CPU全套/pre-commit，整体仍未完成。

### P4 attention 细分观测：Q/K norm 一致（2026-10-23）

沿用上一节真实双卡100步配置，observer增加各模块位置参数首tensor输入，及
各层Q norm、K norm、输出投影的输入/输出/输出梯度。不改算子或优化器，不改变
参数身份或阈值。基线exit0，候选完成100步后仍因同一step61 grad norm
AE0.3347711563/RE0.02443412476退出1。

| 观测 | rank0 maxAE | rank1 maxAE |
| --- | --- | --- |
| layer0 Q norm输入/输出 | 0 / 0 | 0 / 0 |
| layer0 K norm输入/输出 | 0 / 0 | 0 / 0 |
| layer0输出投影输入 | 1.788139344e-7 | 2.011656761e-7 |
| layer0输出投影输出 | 2.235174179e-7 | 2.980232239e-7 |

这将首个已观测前向偏差夹在Q/K norm输出与输出投影输入之间，尚未排除RoPE、
V投影、内层attention和布局；不能直接归因于SDPA或声明修复。反向Q/K norm
输出梯度也有约1e-9偏差，属于诊断而非新增验收阈值。

原始产物：`outputs/baseline-trainer-attention-trace-20261023/alignment-capture/rank*/`
及`outputs/candidate-trainer-attention-trace-20261023/rank*/`中的
`module-trace-step-1.pt`。日志位于约定目录：
`p4_baseline_attention_trace_20261023.log`、
`p4_candidate_attention_trace_20261023.log`。
复现完全沿用上一节基线/候选命令与MODULE_TRACE开关，仅将两个产物目录的
`module-trace`替换为`attention-trace`，仍为100步且候选回放原始基线采集。

独立比较首次exit1：基线attention输入为位置参数、HF为关键字参数，keys不相同；
retry1明确只允许缺少8层attention.input（其他keys必须相同），随后在整数
embedding token输入求mean时报错。retry2将诊断差值转换FP64再求统计，exit0；
未改模型数值或验收阈值。保留日志`p4_attention_trace_comparison_20261023.log`、
`p4_attention_trace_comparison_retry1_20261023.log`与
`p4_attention_trace_comparison_retry2_20261023.log`，完整比较保存在候选根目录
`attention-comparison.json`。缺失的关键字输入未作为已比较项。

retry2同时逐rank核实新旧基线初始/最终权重、全部100步logits/原始metrics精确
一致，候选新旧100步logits/metrics也精确一致。仍仅说明observer未改变本次
轨迹；不能以诊断exit0代替P4通过。全量Ruff/diff检查及两个脚本格式通过，
本轮未重跑CPU全套/pre-commit；无生产修复，整体目标保持未完成。

### P4 RoPE 后 Q/K 偏差已捕获（2026-10-23）

原始目标已重读；本轮继续定位上节未排除的区间。基线observer增加rope与
inner_attention模块及各tensor位置输入；候选增加显式INNER_TRACE开关，包装
HF原sdpa函数采集RoPE后的Q/K/V（统一为BLNH布局），调用原函数且返回原结果。
生产模型及阈值不变，仍为真实双卡100步。

初次诊断使用新attention注册名，改变HF按backend名称选择mask的行为，候选
完整运行后logits验收失败（meanAE1.087651、cos0.819314），日志
`p4_candidate_inner_trace_20261023.log`，产物
`outputs/candidate-trainer-inner-trace-20261023`。该运行不能作为原sdpa路径证据。
重试保持config的sdpa名称，仅包装进程内原sdpa注册项，不修改HF mask分派；
日志`p4_candidate_inner_trace_retry1_20261023.log`，产物
`outputs/candidate-trainer-inner-trace-retry1-20261023`。仍在step61原grad norm
阈值失败，exit1，不记为修复。

基线重跑exit0，日志`p4_baseline_inner_trace_20261023.log`，产物
`outputs/baseline-trainer-inner-trace-20261023/alignment-capture/rank*/`。
首步inner attention输入独立比较（`p4_inner_inputs_compare_20261023.log`，exit0）：

| RoPE后的输入 | rank0 maxAE | rank1 maxAE |
| --- | --- | --- |
| Q | 5.483627319e-6 | 5.841255188e-6 |
| K | 4.529953003e-6 | 4.630535841e-6 |
| V | 0 | 0 |

结合上节两rank Q/K norm输出精确一致，现在证据指向RoPE阶段已引入前向偏差，
不是仅在inner attention内部首次出现。仍未区分频率构造设备、缓存精度、旋转
公式与布局影响，也未证明这可单独解释step61放大；下一步应对比真实cos/sin
缓存与RoPE输出，不修改Adam epsilon或放宽阈值。

复现基线沿用上节命令，目录attention-trace改inner-trace；候选沿用上节命令，
输出改`outputs/candidate-trainer-inner-trace-retry1-20261023`，追加
`LLMTUNER_ALIGNMENT_INNER_TRACE=1`并保持`LLMTUNER_ALIGNMENT_MODULE_TRACE=1`。
比较分别读取两侧rank0/rank1的`module-trace-step-1.pt`，比较
`layer0.inner_attention.input/input1/input2`，候选采集时已从BNLH转为BLNH。
全部日志位于`outputs/qwen3-8b-npu/logs/`。没有删除原失败产物或更改支持声明。

额外核验日志`p4_inner_boundary_audit_20261023.log` exit0：重试候选两个rank
100步logits及原始metrics与旧运行精确一致；基线首步Q/K norm输出及输出投影
输入与上一轮observer精确一致。诊断观测未改变这些已核验结果。全量Ruff、
两个脚本format、compile检查及diff检查通过；未重跑CPU全套/pre-commit。
整体目标仍未完成，后续不可将上述局部定位视为跨框架数值验收通过。

### P4 RoPE频率构造设备差异确认（2026-10-23）

observer保存基线真实rope.cache及HF rotary模块返回的cos/sin；首步两rank
cos与sin maxAE均1.907348633e-6。基线exit0日志
`p4_baseline_rope_cache_20261023.log`，目录
`outputs/baseline-trainer-rope-cache-20261023/alignment-capture/rank*/`；候选
仍以原step61失败exit1，日志`p4_candidate_rope_cache_20261023.log`，目录
`outputs/candidate-trainer-rope-cache-20261023/rank*/`。比较日志
`p4_rope_cache_compare_20261023.log` exit0（仅诊断）。

独立NPU探针按实际debugmodel dim128/theta1e6/seq32比较构造方式：NPU上
pow/arange生成inverse frequency，再outer或matmul、NPU trig，可与真实基线
cache精确一致；CPU上构造inverse frequency后搬到NPU matmul/trig，可与真实
HF cos/sin精确一致。CPU trig会另有约1.5e-7误差，但不能解释原1.907e-6主差异。
日志`p4_rope_cache_device_probe_20261023.log` exit0。该对照仅覆盖本debug配置。

回放新增诊断开关`LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1`：仅重建此debug模型
的inverse frequency，原生产模型及默认回放不变，不改优化器、epsilon或阈值。
完整双卡100步后仍失败：step61 grad norm AE0.05457496643（阈值0.02），
RE0.003983292806（满足0.005但不能豁免AE失败）。较原AE0.334771降低不是通过。
日志`p4_candidate_npu_rope_freq_20261023.log` exit1，目录
`outputs/candidate-trainer-npu-rope-freq-20261023/rank*/`。

关键因果边界：该诊断使两rank首步实际cos/sin、RoPE后Q、K、V输入与基线
全部精确一致，日志`p4_npu_rope_freq_boundary_compare_20261023.log` exit0。
因此频率构造设备确实解释了此前首步RoPE前向差异，但不是当前已证明的唯一
训练对齐根因；剩余误差仍需定位，不能直接把诊断代码搬进生产或删除norm view。

复现沿用上节100步命令：基线目录改rope-cache，候选目录改rope-cache；
NPU频率诊断在候选命令追加`LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1`，输出改为
`outputs/candidate-trainer-npu-rope-freq-20261023`，仍开启MODULE_TRACE与
INNER_TRACE并使用原始100步baseline capture。日志位于约定目录。全量Ruff、
两个脚本format、compileall及diff检查通过；CPU全套/pre-commit未重跑。
整体目标未完成，保留全部失败记录与原验收标准。

### P4 频率对齐后 attention 算术与反向边界（2026-10-23）

原始目标已重读；上一轮频率定位为进展。本轮用已有首步输入做独立算术探针，
并继续运行完整100步诊断，不改生产实现及任何阈值。

1. NPU频率+移除16个residual norm view，完整双卡100步仍exit1：step61
   grad norm AE0.1120462418、RE0.008177980088。日志
   `p4_candidate_npu_rope_no_views_20261023.log`，目录
   `outputs/candidate-trainer-npu-rope-no-views-20261023`。相较频率对齐但保留
   view的AE0.054575更差，不能据此删除生产hook。
2. 频率对齐、保留view时，首步inner Q/K/V已全等，但输出投影输入maxAE仍为
   rank0 1.192092896e-7、rank1 8.940696716e-8；完整局部比较日志
   `p4_npu_rope_remaining_boundaries_20261023.log` exit0，仅为诊断证据。
3. 同一捕获Q/K/V、mask、scale在NPU重新计算，SDPA默认和MATH、原GQA与
   显式repeat K/V均与基线输出存在maxAE1.192092896e-7；日志
   `p4_sdpa_same_input_probe_20261023.log` exit0。因此仅切换GQA重复方式或
   SDPA backend未消除该前向差异。
4. 显式QK乘积后乘scale、masked softmax、乘V，可精确复现基线首层输出；
   预缩放Q或同时缩放Q/K则maxAE2.384185791e-7。日志
   `p4_attention_arithmetic_probe_20261023.log` exit0。这只证明捕获输入上的
   算术边界，不替代完整训练验收。

回放新增`LLMTUNER_ALIGNMENT_EAGER_ATTENTION=1`诊断选项，显式选用HF自身
eager attention（非替换mask或手写Trainer），同时开启NPU频率。完整100步
仍exit1，step61 grad norm AE0.4531240463、RE0.03307241162；日志
`p4_candidate_npu_rope_eager_20261023.log`，目录
`outputs/candidate-trainer-npu-rope-eager-20261023`。
首步两rank输出投影输入、attention输出和layer0输出均与基线精确一致，但
embedding输出梯度仍有maxAE2.682209015e-7/2.905726433e-7；日志
`p4_npu_rope_eager_boundaries_20261023.log` exit0。前向局部精确一致未带来
长期梯度对齐，后续应优先检查反向图/梯度累加与后续层，不能继续把局部前向
通过视为对齐修复，也不能按哪个诊断AE较小选择性报告通过。

复现均沿用上一节候选100步命令及原baseline capture：
- no-views：NPU_ROPE_FREQ=1、REMOVE_NORM_VIEWS=1、MODULE_TRACE=1、
  INNER_TRACE=1，输出改上述no-views目录。
- eager：NPU_ROPE_FREQ=1、EAGER_ATTENTION=1、MODULE_TRACE=1，不设INNER_TRACE
  （该开关专用于sdpa），输出改上述eager目录。保留原view。

上述变量均以前缀`LLMTUNER_ALIGNMENT_`使用；两项均另设TRACE_STEPS为
`1,10,30,60,61`，产物按rank保存。日志均在约定目录。全量Ruff、目标format、
compileall和diff检查通过；本轮未重跑CPU全套/pre-commit。整体目标仍未完成。

### P4 全层前向一致与末层反向定位（2026-10-23）

重读目标并核对无其他训练进程后，优先复用已有快照。原NPU频率+HF eager
诊断的两个rank首步八层input norm、attention、post norm、MLP、block输出
全部精确一致，最终norm输入/输出/输出梯度也精确一致；不是仅首层前向一致。
日志`p4_eager_all_layer_boundaries_20261023.log` exit0，详细统计保存在
`outputs/candidate-trainer-npu-rope-eager-20261023/all-layer-boundaries.json`。
源为该诊断与`outputs/baseline-trainer-rope-cache-20261023/alignment-capture/`
各rank的`module-trace-step-1.pt`，比较全部8层已列边界，不外推至后续步数。

末层反向进一步比较：block7输出梯度、MLP输出梯度、post norm输出梯度均AE0；
attention输出梯度在两rank已有maxAE1.164153218e-10，input norm输出梯度
maxAE分别5.820766091e-10/4.365574569e-10。日志
`p4_eager_last_layer_gradients_20261023.log` exit0，仅表示比较完成。

在相同频率+eager配置追加REMOVE_NORM_VIEWS=1，完整双卡100步后仍exit1，
step61 grad norm AE1.08784198761、RE0.07939891578，不能记通过。日志
`p4_candidate_npu_rope_eager_no_views_20261023.log`，产物
`outputs/candidate-trainer-npu-rope-eager-no-views-20261023/rank*/`。
复现沿用上一节eager命令，额外设置
`LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS=1`，输出改上述目录；原capture、100步、
优化器及阈值不变，仍开启MODULE_TRACE及TRACE_STEPS=1,10,30,60,61。

新快照显示两rank首步八层block输出仍精确一致，末层attention输出梯度现为
AE0（原保留view路径约1e-10），但末层input norm输出梯度仍有
maxAE3.783497959e-10。日志`p4_eager_no_views_last_layer_20261023.log` exit0。
因此view可改变一个残差梯度累加边界，但移除view并未消除attention反向剩余
差异，更未解决长期训练；无需再用单独view开关盲试。后续应在已相同的末层
attention输出梯度条件下，对比attention反向、QKV fused/split梯度累加与RoPE
反向的实际布局。当前尚不能把任一个局部因素认定为唯一根因。

本轮无代码修改，仅增加真实硬件/离线证据；全量Ruff和diff检查通过。
CPU全套/pre-commit未重跑。所有日志在约定目录，保留失败状态，整体未完成。

### P4 相同上游梯度 attention VJP 重放（2026-10-23）

原始目标已重读；上一轮全层观测为进展。本轮新增独立诊断脚本
`tests/integration_tests/npu_attention_vjp.py`，只消费已保存的debugmodel首步
末层输入、初始权重、真实rope cache、document mask和attention输出梯度。
两个rank的输入与上游梯度先与候选做零容差相等断言；不重训、不替换生产Trainer，
不以此脚本代替100步验收。HF32关闭、FP32/NPU，显式eager算术与既有诊断一致。

两条重放使用同一权重：三个独立投影，或按基线源码KV头交错布局打包为单个
fused投影（8KV头×[2Q,1K,1V]×128）。所有前向输出与真实baseline均精确一致。
以同一上游梯度计算VJP后：

| rank | 投影 | 与基线input norm输出梯度maxAE | 与候选同位置梯度maxAE |
| --- | --- | --- | --- |
| 0 | split | 3.783497959e-10 | 0 |
| 0 | fused | 6.402842700e-10 | 4.656612873e-10 |
| 1 | split | 3.783497959e-10 | 0 |
| 1 | fused | 5.238689482e-10 | 6.402842700e-10 |

进一步采集VJP内部Q/K norm输出梯度：split和fused重放均与候选精确一致，但
对基线Q梯度maxAE为rank0 7.082321535e-11/rank1 6.020092758e-11，K梯度为
5.820766091e-11/7.275957614e-11，且不随投影fused/split改变。这说明剩余
差异已经存在于QKV输入梯度归并之前，仅切换fused投影不足以解释或消除差异。
尚未区分eager flex的attention backward、RoPE backward或内存布局，不能
将本地非分布式重放外推为FSDP无差异的证明。

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0 \
python tests/integration_tests/npu_attention_vjp.py \
  --baseline outputs/baseline-trainer-rope-cache-20261023/alignment-capture \
  --candidate outputs/candidate-trainer-npu-rope-eager-no-views-20261023 \
  --output outputs/candidate-trainer-npu-rope-eager-no-views-20261023/attention-vjp-detailed-final.json
```

命令exit0（诊断执行完成，非验收通过），日志
`p4_attention_vjp_detailed_final_20261023.log`。初版只比较输入梯度，日志
`p4_attention_vjp_20261023.log`及`p4_attention_vjp_detailed_20261023.log`保留；
最终版本增加Q/K中间梯度，数值与初版一致。所有日志在约定目录。
全量Ruff、目标format、compileall、diff检查通过；未重跑CPU全套/pre-commit。
无生产修复、未改阈值，P4及整体仍未完成。

### P4 FlexAttention backward 的局部差异确认（2026-10-23）

目标已重读；继续复用相同末层输入、初始权重、真实cache/mask及相同上游梯度。
VJP脚本增加`--attention eager|flex`，默认eager不变；flex显式导入真实基线
`torchtitan_npu`适配后调用PyTorch FlexAttention，而非自行替换设备校验或伪造
backward。仍只作局部诊断，不替代真实双卡100步验收。

首次flex运行exit1：未导入基线适配时原PyTorch明确拒绝NPU设备。日志
`p4_attention_vjp_flex_20261023.log`保留；不绕过guard。核实基线自身
`extensions/torchtitan/eager_flex.py`确有NPU eager适配后，诊断分支显式导入
该基线依赖，独立进程重跑exit0，日志
`p4_attention_vjp_flex_retry1_20261023.log`。这是基线依赖的实际路径，不表示
本仓库默认生产FlexAttention已支持NPU；基线import的全局修改仅存在于该独立
诊断进程，未加入生产模块。

结果：两个rank、fused/split两种投影，前向输出均与基线精确一致；改用flex后
Q/K norm输出梯度全部与真实基线精确一致。对照上一轮eager时这些梯度则与
候选精确一致、与基线有约6e-11～7e-11偏差。因此同一输入与上游梯度下，
attention backward实现确实解释了这段Q/K梯度差异，不是仅依据前向误差推断。

但flex下attention对input norm输出的VJP仍不等于基线：split rank0/1最大AE
4.365574569e-10/3.783497959e-10，fused为5.529727787e-10/4.947651178e-10。
说明Q/K norm前后的反向或剩余布局/累加因素仍未解决；不能因一个中间梯度
相等就宣布P4修复，也不能删除生产guard来替换整网attention。

```bash
PYTHONPATH=.:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu \
ASCEND_RT_VISIBLE_DEVICES=0 python tests/integration_tests/npu_attention_vjp.py \
  --baseline outputs/baseline-trainer-rope-cache-20261023/alignment-capture \
  --candidate outputs/candidate-trainer-npu-rope-eager-no-views-20261023 \
  --attention flex \
  --output outputs/candidate-trainer-npu-rope-eager-no-views-20261023/attention-vjp-flex-retry1.json
```

日志均在约定目录。局部VJP是本轮有效进展，原双卡100步失败状态保持不变，
完整适配目标未完成。后续需捕获Q/K norm输入梯度及V分支梯度，定位已匹配的
norm输出梯度之后的差异，不继续把fused开关当作唯一修复手段。

eager默认路径另进程回归exit0，所有已有数值保持不变，日志
`p4_attention_vjp_eager_regression_20261023.log`，产物
`attention-vjp-eager-regression.json`保存在同一候选目录。全量Ruff、目标格式、
compileall及diff检查通过；未重跑CPU全套/pre-commit。本轮没有生产变更。

### P4 Q/K norm 输入与 V 梯度精确匹配（2026-10-23）

重读目标后，observer增加首步tensor输入梯度hook，不插入module backward
view；新采集仍跑真实基线双卡100步，exit0。日志
`p4_baseline_input_gradients_20261023.log`，产物
`outputs/baseline-trainer-input-gradients-20261023/alignment-capture/rank*/`。
新增hook不改变输出梯度值；独立检查两个rank初始/最终全部权重、100步logits
及原始metrics与旧采集精确一致，日志
`p4_input_gradient_observer_repeatability_20261023.log` exit0。

VJP脚本增加`--require-input-gradients`，严格要求baseline中存在末层Q/K norm
输入梯度和inner attention V输入梯度，不把旧快照缺字段当成验证通过。候选旧
产物缺少新增字段时仍只比较其实际已有边界，不报告这些缺失项已匹配。

相同输入与上游梯度下，flex重放的Q/K norm输入梯度、Q/K norm输出梯度及
V分支梯度，两个rank、split/fused投影均与新基线精确一致。eager对照继续与
候选已存梯度精确一致，但与基线的Q/K norm输入梯度仍有约1.7e-10～2.6e-10、
V梯度约1.7e-10～2.9e-10差异。这进一步确认局部attention反向差异；在flex
重放下，已匹配的分支梯度仍未合成精确匹配的attention输入梯度，下一步应追踪
实际投影反向的张量布局、权重布局或局部映射，而非再次试改RoPE/norm公式。

复现沿用上一节VJP命令，baseline改为本节目录，分别设`--attention eager`
和`--attention flex`，两者都追加`--require-input-gradients`。输出位于同一候选
目录的`vjp-input-gradients-eager.json`/`vjp-input-gradients-flex.json`。
日志`p4_vjp_input_gradients_eager_20261023.log`与
`p4_vjp_input_gradients_flex_20261023.log`均exit0，仅表示局部诊断完成，不是
完整Trainer数值验收通过。仍无生产修复，P4失败状态不变。

额外直接以捕获Q/K/V分支梯度及初始权重，按真实KV交错布局拼接后做matmul，
未消除attention输入梯度误差。首次探针未显式冻结HF32，日志
`p4_projection_captured_gradient_probe_20261023.log`仅作探索，不作验收依据；
另以HF32显式关闭独立重跑，日志
`p4_projection_captured_gradient_hf32off_20261023.log`。不据此简单归因于矩阵
乘法实现，也不能排除捕获布局、DTensor/局部映射对原始backward的影响。

基线复现沿用100步MODULE_TRACE命令，仅dump-folder改为本节input-gradients
目录；其他配置及阈值均不变。全部日志位于约定目录。整体目标仍未完成。

HF32关闭的直接投影探针exit0，rank0/1最大AE分别5.529727787e-10/
4.947651178e-10，未精确匹配。全量Ruff、目标格式、compileall、diff检查通过；
CPU全套/pre-commit本轮未重跑。

### P4 末层 attention VJP 精确复现：权重布局因素（2026-10-23）

重读原始目标后复用真实末层分支梯度，HF32显式关闭。将KV头交错打包梯度乘
相同数值的权重，3D/2D matmul仍有约5e-10误差；将权重做
`weight.T.contiguous().T`后，两个rank的attention输入梯度与真实基线精确一致。
权重数值不变，但stride改变。日志`p4_projection_layout_probe_20261023.log`
exit0；这证明布局影响该局部舍入结果，不证明原基线权重stride已被直接采集。

VJP脚本新增`--transposed-weight`，仅对fused重放的打包权重应用该布局，split
分支作为不变对照。使用真实基线FlexAttention适配、真实cache及相同上游梯度，
两个rank的fused attention前向输出、Q/K norm输入与输出梯度、V分支梯度、
attention输入梯度现在全部与基线精确一致；split对照保持此前非零输入梯度
差异。日志`p4_vjp_transposed_weight_20261023.log` exit0，产物
`outputs/candidate-trainer-npu-rope-eager-no-views-20261023/vjp-transposed-weight.json`。

复现沿用上一节flex VJP命令，追加`--transposed-weight`并使用上述输出路径。
该组合解释了当前末层局部VJP的剩余差异：仅fused而不匹配布局不够，仅匹配
前向而不匹配attention backward也不够。它仍不是完整训练修复，不能把独立
算子精确一致视为100步或其他层/模型/拓扑通过。下一步须直接核实基线实际
参数stride及FSDP materialization，再判断是否需要可配置、可验证的布局适配；
不得为追求精确结果任意改变生产参数身份或checkpoint布局。

本轮仅扩展诊断脚本和证据，未修改生产代码及阈值；P4与整体保持未完成。

### P4 真实forward权重布局核实：并非转置存储（2026-10-23）

重读目标后，为避免将局部精确匹配误当根因，直接在基线forward hook中采集
当时已materialize的权重shape/stride/contiguous/storage_offset及NPU格式。
MODULE_TRACE分支新增QKV投影观测，兼容fused的wqkv及non-fused的wq/wk/wv；
默认生产路径不变。真实双卡100步基线exit0，日志
`p4_baseline_weight_layout_20261023.log`，产物
`outputs/baseline-trainer-weight-layout-20261023/alignment-capture/rank*/`。

两rank末层QKV权重实际shape=(4096,256)、stride=(256,1)、contiguous=True、
offset=0、NPU format ND(2)；输出投影为(256,2048)/stride(2048,1)，同为连续ND。
这**不支持“真实forward权重本身为转置连续布局”的解释**。上一节
`weight.T.contiguous().T`精确匹配只证明替代布局可改变局部matmul舍入，不证明
它复刻了真实forward参数stride；仍需检查实际backward所保存的转置视图、
gradient stride及算子dispatch，不能据此将转置布局改动直接移入生产。

元数据首次保存的npu_format为第三方enum，weights_only读取被拒绝，日志
`p4_actual_weight_layout_20261023.log` exit1。对本机可信采集产物显式用
weights_only=False读取得到上述证据，日志
`p4_actual_weight_layout_retry1_20261023.log` exit0；同时核实两个rank全部
100步原始metrics与上一轮采集精确一致。observer后续将格式转为int保存，
避免新产物引入enum依赖；原日志与产物未覆盖。

复现沿用本节之前的baseline100步MODULE_TRACE命令，仅dump-folder改为
上述weight-layout目录。读取该目录rank*/module-trace-step-1.pt中的
`layer7.fused_projection.weight_layout`（该次运行使用此键；后续代码统一为
`layer7.projection.wqkv.weight_layout`）。日志位于约定目录。
本轮没有生产修复或阈值修改，整体未完成。

全量Ruff、目标格式（格式化后复验）、compileall及diff检查通过；CPU全套与
pre-commit本轮未重跑。后续元数据int化与投影命名的改动尚未重新硬件采集，
本节硬件结论严格对应已保存产物及当次hook版本。

### P4 原始投影梯度布局采集及格式负对照（2026-10-23）

上一轮仅修改Prompt，未推进适配验收；本轮重新读取原始目标后继续P4诊断。
基线observer新增`_local_tensor_layout`，在CPU快照前保存输入/输出梯度的
shape、stride、contiguous、storage_offset和整数NPU格式；不修改训练计算。

双卡真实基线Trainer仍使用FP32、seq32、global batch2、seed42、100步、
HF32关闭、for-loop AdamW及原有基线配置。沿用上一节完整命令，仅dump-folder
改为`outputs/baseline-trainer-gradient-layout-20261023`。日志
`outputs/qwen3-8b-npu/logs/p4_baseline_gradient_layout_20261023.log`，exit0。
两个rank均以weights_only=True成功读取新产物，验证整数格式序列化与新投影键名。
与`baseline-trainer-input-gradients-20261023`逐项比较：每rank415个共有
首步张量、全部100步原始metrics、初始和最终HF权重均精确一致。审计日志
`p4_gradient_layout_audit_20261023.log`，exit0。

两个rank末层真实wqkv权重均为(4096,256)、stride(256,1)、格式2；
输出梯度为(1,32,4096)、stride(131072,4096,1)、格式30；投影输入梯度
为(1,32,256)、stride(8192,256,1)、格式2。逻辑连续不等于设备存储格式相同。

负对照使用该次采集的真实wqkv上游梯度、输入、初始权重与参考输入梯度，
HF32显式关闭，分别将上游梯度转为格式2/30，再运行F.linear的真实autograd
与直接matmul。四种组合均未精确复现：rank0最大AE=5.529727786779404e-10，
rank1=4.94765117764473e-10，格式转换不改变结果。日志
`p4_projection_gradient_format_20261023.log`，exit0表示探针执行成功，
不表示数值验收通过。该证据排除“仅恢复上游梯度格式就足够”的解释；
仍须检查真实FSDP/DTensor线性backward dispatch及保存张量，而非直接修改参数存储。

本轮仅改诊断observer，未改生产代码或阈值。目标脚本Ruff/格式通过；
CPU全套和pre-commit未重跑。P4双卡100步候选训练失败及整体未完成状态不变。

### P4 真实投影保存操作数与autograd路径（2026-10-23）

observer在MODULE_TRACE下记录投影输出向前最多6层的autograd节点及保存张量
布局，不改变计算。真实双卡100步基线两次均exit0，日志
`p4_baseline_saved_layout_20261023.log`和`p4_baseline_saved_other_20261023.log`；
dump-folder分别为`outputs/baseline-trainer-saved-layout-20261023`和
`outputs/baseline-trainer-saved-other-20261023`。其他命令参数沿用上一节100步配置。
第一次未列入MatmulBackward的`_saved_other`，第二次补齐后重新硬件采集，未覆盖产物。

两rank末层投影真实节点均为MatmulBackward0，保存输入shape(1,32,256)、
stride(8192,256,1)、格式2；保存右操作数shape(256,4096)、stride(1,256)、
非连续、格式2。这直接证明真实反向保存的是权重转置视图，而非转置连续参数。

局部F.linear重放在weight.requires_grad=False/True下均为MatmulBackward0，
误差不变，排除先前重放未求权重梯度这一因素。日志
`p4_linear_autograd_dispatch_20261023.log`，exit0。该重放保存的右操作数stride
相同但格式为0，随后显式将权重转为格式2后重放，保存右操作数格式确为2，
仍不能精确复现：rank0最大AE=5.529727786779404e-10，rank1=
4.94765117764473e-10。格式0负对照结果相同；日志
`p4_saved_operand_format_20261023.log`，exit0。探针成功不代表验收通过。

因此已证实实际保存视图布局，且排除仅权重梯度开关、仅保存权重格式两个解释；
尚未证明完整FSDP/DTensor执行环境中的实际算子及精度上下文已复刻。
本轮未改生产路径或阈值，P4与整体仍未完成。全量Ruff、目标格式、compileall、
diff检查通过；CPU全套/pre-commit未重跑。两次新基线尚未审计全部metrics/最终
权重与上一节逐项相等，不能仅凭exit0声称observer对完整轨迹无影响。

### P4 保存张量observer完整轨迹审计与精度上下文负对照（2026-10-23）

本轮先补齐上节未完成的轨迹审计，而非再次启动相同基线。读取
`baseline-trainer-saved-layout-20261023`和`baseline-trainer-saved-other-20261023`
的两个rank，与`baseline-trainer-gradient-layout-20261023`递归比较
step-metrics、initial-hf、final-hf、consumed-batches全部内容，并比较每rank
447个共有首步张量，四组全部精确一致。审计日志
`outputs/qwen3-8b-npu/logs/p4_saved_layout_repeatability_20261023.log`，exit0。
因此上节observer对这两个已执行配置的数值轨迹无影响已有直接证据，
不推广为其他配置或所有观测工具无影响。

核对真实基线`extensions/torchtitan/trainer.py::set_allow_hf32`后，补充
局部投影探针：同时关闭matmul、conv、aclnn三个HF32开关，开启
torch.use_deterministic_algorithms(True)，使用真实捕获输入、上游梯度、初始权重，
权重格式2且requires_grad=True。两个rank输入梯度仍不精确一致，最大AE分别
5.529727786779404e-10与4.94765117764473e-10。日志
`p4_projection_all_hf32_controls_20261023.log`，exit0仅表示负对照执行成功。
注意之前两轮临时投影探针只显式关闭matmul/conv，不能声称已完整复制基线
HF32设置；本轮补齐后结果不变。正式npu_attention_vjp与Trainer replay脚本
原本已同时设置三个开关，不存在这项遗漏。

这些证据排除了保存布局观测改变轨迹，以及仅HF32/确定性开关造成当前局部误差
的解释；尚未证明真实反向算子分派已复刻。未改生产代码或验收阈值，
P4候选双卡100步仍失败，整体未完成；本轮仅证据审计，未重跑CPU/pre-commit。

### P4 真实首步投影dispatch采集（2026-10-23）

新增显式开关`LLMTUNER_ALIGNMENT_DISPATCH_TRACE=1`，只包围真实Trainer首步，
采集带4096维的矩阵操作的算子名、实际操作数数值与布局、输出及确定性状态。
默认关闭，不改变生产训练路径。首次模式未声明支持HigherOrderOperator，
FlexAttention拒绝该observer，日志`p4_baseline_dispatch_20261023.log` exit1。
给模式开启supports_higher_order_operators并原样转发操作后，以独立目录重跑。

成功运行沿用双卡FP32/seq32/100步基线命令，新增上述开关，dump-folder为
`outputs/baseline-trainer-dispatch-retry1-20261023`；日志
`outputs/qwen3-8b-npu/logs/p4_baseline_dispatch_retry1_20261023.log` exit0。
两rank均捕获16项：8个aten.matmul.default前向及8个
aten.matmul_backward.default反向。首个反向对应末层wqkv；其输出输入梯度
与module hook精确一致，捕获权重与初始HF权重按KV交错打包后精确一致，
捕获上游梯度与module hook精确一致，排除重放权重数值/布局映射错误。

直接恢复捕获操作数的stride/格式并调用相同aten.matmul_backward，仍有
rank0最大AE=5.529727786779404e-10、rank1=4.94765117764473e-10。
日志`p4_actual_dispatch_replay_20261023.log` exit0。该次恢复有内部格式
禁用警告，因此不能凭该次声称恢复物理格式成功。另显式设置
torch.npu.config.allow_internal_format=True后重跑，确认三个操作数格式实际为
[30,2,2]，误差仍相同；日志`p4_dispatch_internal_format_20261023.log` exit0。
该config为仅写接口，读取属性报AttributeError不能作为设置失败的依据。

成功采集另行完成完整轨迹审计：两个rank全部100步metrics、initial/final
HF权重、consumed batches与gradient-layout基线均精确一致；每rank16项
dispatch记录。日志`p4_dispatch_repeatability_20261023.log` exit0。
因此实际高层ATen算子及操作数已核实，但独立进程仍未复刻底层计算结果，
不能将局部替代布局精确一致解释为生产修复。下一步应检查后端算子选择/
运行时上下文，而非继续盲试单个stride。未修改阈值，P4及整体仍未完成。

### P4 局部反向差异定位：遗漏CLOSE_MATMUL_K_SHIFT（2026-10-23）

核对基线`patches/torch_npu/determinism.py`发现，基线不仅启用PyTorch确定性，
还设置CLOSE_MATMUL_K_SHIFT=1及HCCL_DETERMINISTIC=true。之前临时局部
投影探针调用torch.use_deterministic_algorithms(True)但未设置前者，因此
不能声称已完整复刻确定性环境。候选生产`llmtuner/utils/seeding.py`原本已
设置两个变量，不能将此次诊断遗漏认定为生产训练失败根因。

独立进程启动前补齐两个环境变量，恢复真实dispatch全部操作数的stride/格式，
调用原aten.matmul_backward：两个rank全部8层QKV投影输入梯度均精确匹配，
16项最大AE全部为0，逐项assert通过。日志
`outputs/qwen3-8b-npu/logs/p4_dispatch_k_shift_20261023.log`，exit0。
因此此前局部反向剩余误差不需要修改权重存储布局；替代转置布局的零误差
不能再作为建议生产布局变更的依据。此次环境对照同时设置了两个变量，
严格说尚未单独隔离每个变量的因果贡献。

正式局部VJP脚本现在复用本仓库set_deterministic_algorithms，避免遗漏环境。
以真实FlexAttention、原始连续打包权重重跑，不加--transposed-weight：
两个rank的fused末层前向、attention输入梯度、Q/K norm输入/输出梯度和V梯度
全部精确匹配真实基线。split投影负对照仍有约4e-10的attention输入梯度差异。
日志`p4_vjp_deterministic_environment_20261023.log`，exit0；产物
`outputs/candidate-trainer-npu-rope-eager-no-views-20261023/vjp-deterministic-environment.json`。

新增--require-exact-fused可选断言，用于明确局部精确复现，不替代Trainer验收。
复现：沿用之前flex VJP完整命令，不加--transposed-weight，追加
--require-input-gradients --require-exact-fused，输出vjp-deterministic-exact.json；
日志`p4_vjp_deterministic_exact_20261023.log`，exit0，两个rank的显式精确断言通过。
全量Ruff、两个诊断脚本格式通过；CPU全套/pre-commit未重跑。生产路径和阈值不变。
下一步应围绕已证明的fused/split及真实attention反向差异验证完整Trainer，
不再将布局探针的误差当成未解释的底层dispatch问题。P4整体仍未通过。

### P4 真实Trainer组合诊断：nonfused基线×FlexAttention×NPU RoPE（2026-10-23）

已有证据否定仅切换非融合基线，本轮不重复该单因素实验，而在候选真实Trainer
新增默认关闭的LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1。它导入真实基线
FlexAttention NPU适配，仍保留HF的sdpa注册名称以保持原mask分派语义，
每层每步检查实际HF mask与对应基线dense document mask零容差一致，
再构造BlockMask并执行真实flex_attention；断言seq32及dropout=0。
该路径仅诊断，不是生产后端支持声明，也不能替代原目标配置验收。

运行命令：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu ASCEND_RT_VISIBLE_DEVICES=6,7 LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1 LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS=1 LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1 LLMTUNER_ALIGNMENT_MODULE_TRACE=1 torchrun --standalone --nproc_per_node=2 tests/integration_tests/npu_trainer_alignment_replay.py --capture outputs/baseline-trainer-fsdp2-nonfused-retry1-20261022/alignment-capture --output outputs/candidate-trainer-nonfused-flex-rope-20261023'
```

候选实际完成100步并保存两个rank的全部logits、step metrics、最终权重及首步
模块观测，随后原阈值断言在step61 grad_norm失败，exit1。日志
`outputs/qwen3-8b-npu/logs/p4_candidate_nonfused_flex_rope_20261023.log`。
AE=0.40553760528564453，RE=0.03204365334285617，仍超AE≤0.02/RE≤0.005。
这不是修复，也不能用较此前误差降低来判定通过。

另外使用原metrics.py对两个rank全部100步logits、loss、grad_norm审计，
每rank仅step61 grad_norm不满足标准，其他已列指标均满足；日志
`p4_nonfused_flex_full_audit_20261023.log` exit0仅表示审计成功。结果分别保存于
候选rank*/full-failure-audit.json。最终权重仅已保存，未设未授权的权重阈值。
因此即使匹配非融合投影、真实attention路径、RoPE频率及移除norm view hook，
完整Trainer仍有剩余差异，不能将已解决的局部VJP问题推广为多步通过。

本轮仅修改集成诊断脚本；全量Ruff、目标格式、compileall和diff检查通过。
CPU全套/pre-commit未重跑。生产代码与验收阈值不变，P4/整体仍未完成。

### P1 KCP独立缺口复核：依赖可加载，实际CP受架构限制（2026-10-23）

P4多轮组合诊断仍失败，本轮依任务约束转向独立KCP项目。重新读取目标并核实
工作树main/c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19，保留现有改动。
默认容器仍无fla/triton_ascend_kernels，未覆盖或降级系统依赖。

在宿主机取得官方源码，保存在既有离线依赖目录下：

- `/home/jianzhnie/llmtuner/wheels/kcp_probe/triton-ascend-kernels`，
  GitCode Ascend/triton-ascend-kernels，commit
  `7acb9c40a6ca827e885d902b7eaee5d45457b230`。
- `/home/jianzhnie/llmtuner/wheels/kcp_probe/flash-linear-attention`，
  GitHub fla-org/flash-linear-attention，commit
  `12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e`。

访问和下载日志为kcp_upstream_access_20261023.log、kcp_upstream_clone_20261023.log、
kcp_fla_clone_20261023.log，均exit0。仅通过容器PYTHONPATH读取宿主机源码，
容器没有联网安装。设置：

```bash
export PYTHONPATH=.:outputs/baseline-runtime:/home/jianzhnie/llmtuner/wheels/kcp_probe/triton-ascend-kernels/src:/home/jianzhnie/llmtuner/wheels/kcp_probe/flash-linear-attention
```

容器运行`from llmtuner.parallel.context_parallel.kcp import _kcp_runtime;
print(_kcp_runtime())`成功加载真实causal_conv1d、chunk_kda、build_cp_context，
日志`outputs/qwen3-8b-npu/logs/kcp_offline_runtime_import_20261023.log` exit0。
这仅解除本次隔离探针的导入阻塞，不是算子或KCP训练验收。

进一步实测设备Ascend910_9392，上游is_arch35()返回False；真实chunk_kda
在给定cp_context后抛出ValueError：
`cp_context (context parallel) is only supported on A5 (arch3.5) hardware.`
日志`kcp_architecture_preflight_20261023.log` exit0仅表示预期拒绝被捕获，
不代表CP通过。复现片段（容器中沿用上述PYTHONPATH，设备0）：

```python
import torch
import torch_npu
from llmtuner.parallel.context_parallel.kcp import _kcp_runtime
from triton_ascend_kernels.attention.fla.kda.cp.context import FLACPContext

_, chunk_kda, _ = _kcp_runtime()
context = FLACPContext(cu_seqlens=torch.tensor([0, 64], device="npu"))
query = torch.randn(1, 64, 2, 64, device="npu", dtype=torch.bfloat16)
chunk_kda(
    q=query, k=query, v=query, g=-torch.rand_like(query),
    beta=torch.rand(1, 64, 2, device="npu"), cp_context=context,
)
```

该复现将原样抛错；日志探针则显式捕获并断言错误文本。未绕过上游架构检查，
未用mock或naive替代实际内核。上游tests/conftest.py将arch3.5定义为
Ascend910_95/Ascend950，CP KDA测试也按此门控。本机不是该架构，KCP保持blocked；
解锁需兼容本机的真实CP内核，或可用arch3.5设备及匹配运行时后再执行KDA模型
端到端前反向/梯度验收。此前未检查其他候选节点；后续只读核验如下。

### KCP其余候选节点的实际设备核验

重读原始目标后，并行查询其余三个明确列出的候选节点，没有启动算子或训练，
没有联网安装依赖。三条SSH及docker查询均实际exit0：

| 节点 | torch / torch-npu | 实际设备数及名称 | KCP架构解锁 |
| --- | --- | --- | --- |
| 10.1.0.18 | 2.12.0+cpu / 2.12.0 | 8，全部Ascend910_9392 | 未解除 |
| 10.1.0.69 | 同上 | 8，全部Ascend910_9392 | 未解除 |
| 10.1.0.71 | 同上 | 8，全部Ascend910_9392 | 未解除 |

每节点执行以下只读命令，将HOST分别替换为上述地址：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 HOST "docker exec hyper-parallel-env python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.npu.device_count()); print([torch.npu.get_device_name(i) for i in range(torch.npu.device_count())])'"
```

现场同时读取上游`triton_ascend_kernels/utils.py:42`，其is_arch35仅接受
设备名称含Ascend910_95或Ascend950。故明确候选节点的查询结果均不满足
该运行时的架构条件；不再仅凭镜像名推断，也不启动注定被该保护拒绝的训练。
此结论不代表所有集群节点不可用，且未对其余节点执行KCP功能验收。
当前解锁仍需要兼容设备或支持9392的真实CP内核；整个目标不因单项环境限制
标为blocked。P4关键权重对象、时点、阈值已向用户请求澄清，等待期间继续
独立工作，不套用logits阈值或按已有误差事后设线。

另有版本风险：Ascend内核源码元数据固定torch/torch_npu 2.7.1与triton-ascend
3.2.0；本次复用隔离Triton，导入显示3.2.0且FLA警告推荐≥3.3；FLA当前NPU
extra另指定torch2.10/triton-ascend3.6。不得据导入成功宣称这些源码组合已兼容
当前torch2.12/CANN9.1，需在目标硬件上重新验证，不自动更换系统软件。

本轮仅准备隔离依赖与更新证据，无生产代码修改；CPU/门禁未重跑。
原始KCP缺口从“缺包”收敛为已证实的内核架构限制及待验证版本兼容性。
P4失败和整体未完成状态不变。

### P3 torch_checkpointing双卡NPU metadata修复与恢复验收（2026-10-23）

针对既有torch_checkpointing 0.1.0 metadata聚合硬编码CUDA的实测失败，新增
`llmtuner/components/checkpointer/checkpointing_metadata.py`。仅在已初始化的
NPU分布式环境、精确依赖版本0.1.0、原DefaultMetadataManager类型下，将通信
方法绑定到当前metadata实例；不修改依赖类或CUDA API，不改变resharder、
数据格式、缓存、staging、writer及资源所有权。按实际process_group通过
get_comm_device选择通信设备，保留原8字节长度/pickle/padding聚合协议。
原始语义来源及边界已记录于docs/llmtuner_upstream_map.md。

首次真实双卡复验仍exit1：常驻manager已适配，但最后一步创建的临时manager
仍使用原CUDA通信；日志`torch_checkpointing_npu_metadata_20261023.log`保留。
随后将临时manager的适配放入已有try/cleanup生命周期内。修复后真实双卡
连续4步vs中断2步再恢复至4步，两个rank模型、optimizer、lr_scheduler、
train_state、dataloader、rng_state六类状态均精确一致，exit0；日志
`torch_checkpointing_npu_metadata_retry1_20261023.log`。

进一步补强正式集成脚本，断言dp_shard等于world size，且至少一个DTensor参数
local.numel小于global.numel，避免仅FSDP包装而实际未分片的空泛验收。
使用新目录完整重跑仍exit0，两个rank六类状态全部零容差一致：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 tests/integration_tests/npu_checkpoint_backend_equivalence.py --output outputs/torch-checkpointing-npu-fsdp2-metadata-sharded-20261023'
```

日志`outputs/qwen3-8b-npu/logs/torch_checkpointing_npu_metadata_sharded_20261023.log`。
原始参考及恢复状态分别保存于对应输出目录reference-rank{0,1}.pt及
resumed-rank{0,1}.pt，比较覆盖DTensor placements、global shape和各rank本地shard。
独立单卡回归也exit0、六类状态精确一致，日志
`torch_checkpointing_npu_single_metadata_regression_20261023.log`；命令同脚本
以ASCEND_RT_VISIBLE_DEVICES=0直接python运行，输出目录
`outputs/torch-checkpointing-npu-single-metadata-regression-20261023`。

质量门禁：

- checkpoint CPU模块322 passed/1 skipped；新增覆盖变长payload通信协议、
  非NPU/未初始化环境无导入副作用、版本边界及单实例隔离。
  日志checkpointing_metadata_cpu_20261023.log，exit0。
- 全量`pytest -q --disable-warnings`：4084 passed/13 skipped，exit0；
  日志cpu_checkpointing_metadata_full_20261023.log。
- 全量Ruff、修改文件format、compileall、git diff --check均通过。
- 全量format仍exit1，仅未修改的tp_ep_cp_training_equivalence.py不符合当前
  formatter；日志checkpointing_metadata_format_20261023.log，不修无关文件。
- 为防pre-commit自动修改无关文件，以当前tracked及新增任务文件的内容副本运行
  `pre-commit run --all-files`，没有切分支或修改工作树。exit1：Ruff0.9仅在副本
  修改未变更的parallel/expert_parallel/hf_inspection.py两处；其余hooks通过。
  日志checkpointing_metadata_precommit_snapshot_20261023.log。副本路径记录在
  checkpointing_metadata_gate_snapshot_path_20261023.txt。实际工作树未应用该修改。
  同副本显式指定本次四个代码/测试文件执行pre-commit，所有适用hooks通过，
  日志checkpointing_metadata_precommit_targeted_20261023.log，exit0。

日志均在outputs/qwen3-8b-npu/logs/。本次关闭指定torch_checkpointing版本和
双卡FSDP2配置的恢复缺口；不推广至其他依赖版本、拓扑变更、EMA或性能验收。
KCP仍blocked，P4仍失败；完整目标尚未完成。

### P3 EMA恢复扩展及确定性初始化时序修复（2026-10-23）

集成脚本新增--ema none/fixed/dynamic，默认none保持原配置。固定decay=0.5，
动态half_life_fraction=0.5；同时断言EMA状态非空且实际训练后至少一个平均
张量改变，避免空泛保存。仍使用离线tiny Qwen3（vocab128/hidden64/2层/
4Q头2KV头/seq32）、seed42、global batch=world size、总调度4步，
连续4步对比中断2步再恢复至4步；不替代Qwen3-8B验收。

首次双卡动态EMA exit0，两个rank七类状态（模型、optimizer、scheduler、EMA、
trainer、loader、RNG）全部精确一致，日志torch_checkpointing_npu_dynamic_ema_20261023.log。
首次四卡固定EMA则exit1，最终模型有约1e-8差异，未放宽阈值；日志
torch_checkpointing_npu_fixed_ema_fsdp4_20261023.log，失败产物全部保留。

对失败运行在恢复前的两个step2 checkpoint逐rank比较：每rank16个模型张量
及16个EMA张量已不同，模型max AE最高1.043081283569336e-7，EMA最高
7.450580596923828e-8。日志checkpointing_fixed_ema_split_audit_20261023.log
exit0。这证明差异在恢复前已经出现，不能仅凭最终差异归因于checkpoint加载。
在进程启动前显式设置HCCL_DETERMINISTIC=true/CLOSE_MATMUL_K_SHIFT=1后，
相同四卡固定EMA完整恢复七类状态全部精确一致，日志
torch_checkpointing_npu_fixed_ema_early_deterministic_20261023.log，exit0。

核查生产调用顺序发现：原assembly先初始化进程组，之后seed_everything才
启用确定性策略；测试harness还会更早创建外部组。生产修复在
assemble_trainer_runtime入口、init_dist_pytorch之前复用已有
set_deterministic_algorithms；随机seed及PP偏移仍在拓扑解析后设置。
保留原构造失败rollback边界。外部预建进程组无法事后修复其初始化环境，
调用者须在创建外部组前设置策略。新增两个CPU用例验证deterministic True/False
时策略先于进程组初始化，且False不主动启用策略。

正式集成测试移除主动harness初始化，让真实Trainer自行创建进程组。
不依赖额外确定性环境变量，使用最终生产代码重新验收：

| 配置 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- |
| FSDP2×dynamic EMA，4步vs2+2 | 通过 | 0 | torch_checkpointing_npu_dynamic_ema_bootstrap_20261023.log |
| FSDP4×fixed EMA，4步vs2+2 | 通过 | 0 | torch_checkpointing_npu_fixed_ema_bootstrap_20261023.log |

每rank七类完整状态全部零容差一致，包含真实分片/placements/global shape断言。
两次运行命令均为：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 tests/integration_tests/npu_checkpoint_backend_equivalence.py --ema fixed --output outputs/torch-checkpointing-npu-fsdp4-fixed-ema-bootstrap-20261023'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 tests/integration_tests/npu_checkpoint_backend_equivalence.py --ema dynamic --output outputs/torch-checkpointing-npu-fsdp2-dynamic-ema-bootstrap-20261023'
```

最终生产代码全量CPU：4086 passed/13 skipped，exit0，日志
cpu_early_deterministic_full_20261023.log。全量Ruff、修改文件format、compileall、
diff检查通过。当前内容副本重跑全量pre-commit仍因未修改hf_inspection.py的
Ruff0.9两项自动修复而exit1，其他hooks通过；修改文件定向hooks全部通过。
日志early_deterministic_precommit_all_snapshot_20261023.log及
early_deterministic_precommit_targeted_20261023.log。全量format仍仅原有
tp_ep_cp_training_equivalence.py失败，日志early_deterministic_format_all_20261023.log。
未将副本中的无关自动修改应用到工作树。

所有日志位于outputs/qwen3-8b-npu/logs/，来源及边界同步记录于upstream map。
本节证明指定2/4卡EMA恢复及生产bootstrap时序修正，不证明所有HCCL拓扑
确定性或跨框架数值对齐。KCP仍blocked，P4仍失败；整体目标未完成。

### P4 初始化修复回归及首个分歧时点定位（2026-10-23）

在生产确定性时序修复后，alignment replay也在外部harness创建进程组之前
复用set_deterministic_algorithms，遵守外部预建组的初始化契约。
无其他诊断开关，原始双卡100步候选重新回放原fused基线：仍在step61
grad_norm失败，AE=0.33477115631103516、RE=0.024434124762137712，exit1。
日志p4_candidate_early_deterministic_20261023.log。两个rank全部100步logits、
loss、grad_norm、lr和token计数与原candidate-trainer-fsdp2-100-audit-20261022
精确一致，日志p4_early_deterministic_trajectory_audit_20261023.log exit0。
因此本次初始化修复不是原双卡P4失败的根因，不能据四卡恢复通过推断P4通过。

针对先前nonfused×真实FlexAttention×NPU RoPE×移除norm views组合，新增
真实非融合基线MODULE_TRACE采集，仍100步、相同原始初始权重和数据；
产物outputs/baseline-trainer-nonfused-module-trace-20261023/alignment-capture。
日志p4_baseline_nonfused_module_trace_20261023.log exit0。
与既有candidate-trainer-nonfused-flex-rope-20261023首步观测比较，两个rank
各247个共有输入、输出及激活梯度全部精确一致，覆盖8层、embedding、末层norm。
日志p4_nonfused_module_boundary_audit_20261023.log exit0，逐项结果保存于
候选rank*/nonfused-module-boundary-audit.json。没有用不存在的字段填充零误差。

随后通过既有TRACE_STEPS=1采集双方真实Trainer的首步参数状态。
基线产物outputs/baseline-trainer-nonfused-update-trace-20261023/alignment-capture，
日志p4_baseline_nonfused_update_trace_20261023.log exit0。候选产物
outputs/candidate-trainer-nonfused-flex-update-trace-20261023，日志
p4_candidate_nonfused_flex_update_trace_20261023.log exit1，仍step61验收失败。
两个rank各90项clipped_gradients及90项weights_after_update全部精确一致，
日志p4_nonfused_update_boundary_audit_20261023.log exit0。另直接核实基线首步
90项权重全部相对initial-hf改变，故此处不是零学习率造成的空泛相等；
仍不将单步更新相等推广为后续全部更新相等。

继续审计100步logits：该组合前10步全部精确一致，首次非零差异均在step11，
rank0 maxAE=8.344650268554688e-7、rank1=9.5367431640625e-7；日志
p4_nonfused_first_divergence_20261023.log exit0。这不是新的验收阈值，
而是将后续定位收敛到第10次更新/第11步执行附近，不再重复首步算子猜测。

完整重复性审计：两个新基线的initial/final HF权重、全部100步metrics及
consumed batches均与既有nonfused基线精确一致；新候选全部100步logits及
loss/gradnorm/lr/tokens与旧组合候选精确一致，两个rank均检查。
日志p4_nonfused_trace_repeatability_20261023.log exit0。

复现基线沿用已有100步命令，--config baseline_nonfused_probe，
LLMTUNER_ALIGNMENT_INITIAL_HF指定baseline-trainer-fsdp2-trace-20261022/
alignment-capture/rank0/initial-hf.pt；模块采集设置MODULE_TRACE=1，更新采集
改为TRACE_STEPS=1，分别使用上述独立dump-folder。候选沿用本文件nonfused
Flex组合完整命令，追加TRACE_STEPS=1并改用上述新output。原始P4复验则不设
任何attention/RoPE/view诊断开关，capture仍原fsdp2-partial-100目录。

本轮仅在诊断脚本提前设置确定性策略，生产代码和阈值未再修改。全量Ruff、
目标format、compile及diff检查通过；当前脚本副本定向pre-commit全部通过，
日志p4_early_deterministic_precommit_targeted_20261023.log。CPU全套本轮未重跑，
最近生产代码门禁仍为4086 passed/13 skipped。日志均在约定目录。
P4和整体目标仍未完成。

### P4 第10次裁剪归约顺序定位：指定诊断组合100步精确对齐（2026-10-23）

沿用相同非融合基线和候选Flex/RoPE/no-view组合，将双方TRACE_STEPS改为
9,10,11，仍完整运行100步。基线目录
outputs/baseline-trainer-nonfused-step10-trace-20261023/alignment-capture，
日志p4_baseline_nonfused_step10_trace_20261023.log exit0；候选目录
outputs/candidate-trainer-nonfused-step10-trace-20261023，日志
p4_candidate_nonfused_step10_trace_20261023.log exit1，仍step61 grad_norm失败。

逐参数对照（两个rank均相同）：step9全部90项裁剪梯度和更新后权重精确一致；
step10全部90项裁剪梯度出现微差，57项更新后权重不同；step11仍90项梯度、
57项权重不同。详见p4_nonfused_step10_boundary_audit_20261023.log及候选
rank*/step10-boundary-audit.json。step10基线grad_norm=3.4686098098754883，
候选=3.4686100482940674，差2.384185791015625e-7（一个FP32 ULP）。

核实参数实际遍历顺序：基线MLP为gate/down/up，HF为gate/up/down。
新增默认关闭的LLMTUNER_ALIGNMENT_BASELINE_NORM_ORDER=1诊断开关，
仅用于debugmodel_non_fused_qkv，按照基线映射后参数顺序排列传入原
clip_grad_norm_的同一组参数；断言参数身份集合完全相同，不删除参数、
不改optimizer顺序、范数公式、clip系数公式或阈值；finally恢复原函数。
顺序输出为rank*/norm-parameter-order.json。该过程只发生在诊断脚本中，
不是生产模型重排或新公共API，不适用于原fused基线的参数一一对应。

在原组合上仅追加该归约顺序开关：真实双卡候选Trainer完成100步，
原metrics.py全部逐步断言通过，exit0。日志
p4_candidate_nonfused_norm_order_20261023.log；产物
outputs/candidate-trainer-nonfused-norm-order-20261023/rank*/。
独立审核两个rank全部100步logits、loss、grad_norm、lr及token计数均与
非融合基线精确一致，且各90项最终权重的实际本地shard精确一致；
读取时检查真实DTensor placements并按rank对照对应全局切片，没有仅检查
汇总JSON。日志p4_norm_order_full_exact_audit_20261023.log exit0。
step9/10/11各90项裁剪梯度及更新后权重也全部精确一致，日志
p4_norm_order_step10_audit_20261023.log exit0。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu ASCEND_RT_VISIBLE_DEVICES=2,3 LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1 LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS=1 LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1 LLMTUNER_ALIGNMENT_BASELINE_NORM_ORDER=1 LLMTUNER_ALIGNMENT_TRACE_STEPS=9,10,11 torchrun --standalone --nproc_per_node=2 tests/integration_tests/npu_trainer_alignment_replay.py --capture outputs/baseline-trainer-fsdp2-nonfused-retry1-20261022/alignment-capture --output outputs/candidate-trainer-nonfused-norm-order-20261023'
```

两个step10采集运行另行核实观测重复性：基线initial/final权重、全部metrics/
consumed batches与原非融合基线精确一致；未重排归约的候选100步logits及
loss/grad_norm/lr/tokens与原失败组合精确一致。日志
p4_step10_trace_repeatability_20261023.log exit0。因此不能将新结果归于
trace采集意外改变了旧失败轨迹。

这些证据将该组合的剩余分歧收敛到参数范数归约顺序引起的FP32舍入及其
后续影响；尚未独立保存第10步裁剪前全部参数梯度，不据此声称所有底层梯度
算子的任意输入均位级相等。成功配置同时依赖非融合基线、显式RoPE/flex、
移除norm view hooks和norm顺序诊断；它不替代原fused/default配置验收，
不据此删除生产保护检查或把hook直接移入生产。原默认P4仍失败，完整目标未完成。

全量Ruff、脚本format、compileall及diff检查通过；定向副本pre-commit通过，
日志p4_norm_order_precommit_targeted_20261023.log。本轮无生产代码修改，
未重跑CPU全套，最近生产门禁为4086 passed/13 skipped。日志均在约定目录。

#### 同设备无trace复验与独立权重审计（2026-10-23）

已核实复验训练进程终止，两个rank均输出100步完成标记，未见traceback；
日志p4_candidate_nonfused_norm_order_repeat_20261023.log。该次使用设备0,1
（与未重排归约的失败对照相同），保留四个诊断开关，不设置TRACE_STEPS；
输出位于outputs/candidate-trainer-nonfused-norm-order-repeat-20261023。
它验证加入非融合基线限制后的脚本及无额外step权重采集的轨迹，不能代替默认配置。

独立CPU审计exit0：两个rank的100步logits、loss、grad_norm、LR及token计数
与首次成功运行精确一致；按实际DTensor Shard/Replicate布局核对，90项基线
最终权重的本地分片及额外共享lm_head别名均精确一致。日志
p4_norm_order_repeat_exact_audit_v2_20261023.log。首版审计因直接要求state-dict
键集合一致而失败，原日志p4_norm_order_repeat_exact_audit_20261023.log保留；
查明候选多出共享lm_head.weight，脚本已有参数身份断言，修正审计以embedding
基线值检查该别名，未改训练输出、参数或阈值。本次仅更新证据，无生产代码修改。
原fused/default P4失败及其他未完成项保持原状态，完整目标仍未完成。

#### 保留生产norm view边界的消融（2026-10-23）

上一轮完成同设备复验及独立审计，属于证据进展。本轮保持设备0,1、非融合
基线、NPU RoPE、真实FlexAttention及基线裁剪归约顺序，仅不设置
LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS，保留生产残差norm view hooks。
读取model_setup.py确认它用于合并norm反向贡献后再累加残差，不能因对齐失败
直接删除。候选真实Trainer完成100步后验收失败，torchrun退出码1；
原始日志p4_norm_order_keep_views_20261023.log，输出目录
outputs/candidate-trainer-nonfused-norm-order-keep-views-20261023。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu ASCEND_RT_VISIBLE_DEVICES=0,1 LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1 LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1 LLMTUNER_ALIGNMENT_BASELINE_NORM_ORDER=1 torchrun --standalone --nproc_per_node=2 tests/integration_tests/npu_trainer_alignment_replay.py --capture outputs/baseline-trainer-fsdp2-nonfused-retry1-20261022/alignment-capture --output outputs/candidate-trainer-nonfused-norm-order-keep-views-20261023'
```

双rank均在step61 grad_norm断言失败：基线12.65578556060791，候选
12.005270004272461，AE0.6505155563354492、RE0.05140064622777966。
独立审计已保存的全部100步产物，exit0（审计完成，不代表训练通过），日志
p4_norm_order_keep_views_audit_20261023.log：相对精确成功对照，logits及
grad_norm首次非精确为step2，loss为step4，LR/token全程精确；step81
grad_norm也超相对阈值，AE0.012816667556762695、RE约0.00728667298。
说明在本组合里仅对齐裁剪归约顺序不足以保留生产view边界并通过，移除views
不是已证明可省略的诊断开关；不能推广为默认配置修复。未改生产代码或阈值。
宿主机追加退出码到容器root拥有日志时tee权限失败，未重跑训练；退出码来自
已结束exec会话及日志中torchrun子进程exitcode1，原失败日志保持完整。

### P3 真实Inductor TP collective解锁：纠正后端环境变量（2026-10-23）

上一轮norm边界消融提供了新失败证据，本轮推进独立compile项。重新检查容器
torch/torch_npu均2.12.0及已安装源码：torch_npu/_inductor/__init__.py的
_BACKEND_LOADERS仅注册default和mlir，未知值默认加载Triton；但codegen/
__init__.py的适配补丁仅在mod==default时安装。旧命令指定ascendc，既没有
选择AscendC，又跳过了这些补丁。旧日志的KeyError r不能单独证明当前版本
默认Inductor必然不可用；此前将解锁条件只归于版本/API兼容的判断不充分。
本轮不修改依赖、不monkeypatch、不抑制编译异常，以已注册default配置重跑
同一tp_compile_collectives.py原有fullgraph/精确梯度断言。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_NPU_BACKEND=default LLMTUNER_TEST_COMPILE_BACKEND=inductor torchrun --standalone --nproc_per_node=2 tests/integration_tests/tp_compile_collectives.py'
```

| 测试项 | 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- | --- | --- |
| TP collective真实Inductor | NPU2，FP32，fullgraph，default/Triton | 原解析梯度，atol=rtol=0 | 两rank gather_loss/scatter_loss梯度全精确 | 通过 | 0 | tp_inductor_default_backend_20261023.log |

日志记录两rank各自前向/反向编译图；生成代码包含async_compile.triton、
实际kernel.run以及_c10d_functional.all_gather_into_tensor/reduce_scatter_tensor。
结合实际运行后的零容差梯度断言，证明不是仅导入、aot_eager或静默eager回退。
生成调试文件已移至outputs/tp-inductor-default-debug-20261023，不留源码根目录。
旧ascendc失败记录保留。该通过限于这两个collective的前反向，不代表整个
模型Inductor、compile×FSDP/PP或全部编译组合通过。未修改生产/测试代码，
本轮不重跑CPU全套；git diff --check通过，P4默认路径及其余缺口仍未完成。

### P3 PP2整模型Inductor×确定性×DCP恢复补验（2026-10-23）

沿用刚验证的default后端，运行现有pp_checkpoint_equivalence.py的真实
Trainer四步训练及2+2断点恢复；保持脚本默认模型/数据/精度/零容差断言，
不以collective测试代替模型训练。配置为双NPU、PP2/1F1B、FP32、
deterministic=True、whole-model Inductor、DCP，未启用EMA或AC。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_NPU_BACKEND=default LLMTUNER_TEST_COMPILE_BACKEND=inductor TORCH_COMPILE_DEBUG=1 && { torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py full outputs/pp2-inductor-default-resume-20261023 && torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py resume outputs/pp2-inductor-default-resume-20261023; }'
```

失败，exit1，日志pp2_inductor_default_resume_20261023.log（含退出码）；
full阶段首次编译前向执行失败，resume因&&未执行，不能报告恢复通过。
失败栈已进入实际生成代码的
triton_unk_fused_arange_cat_slice_sub_unsqueeze_1.run，再经torch_npu的
autotune_to_one_config→bench→benchmarker.benchmark_gpu，触发torch的
may_ban_benchmarking：确定性模式不允许未经vetted标记的benchmark。
与旧ascendc配置KeyError不同，不能据此撤销已通过的TP collective结果。
未关闭确定性、放宽恢复断言、抑制异常或修改依赖；当前该组合仍失败。
安装源码显示NPU对reduction实现跳过autotune，但pointwise仍进入上述bench；
后续需验证符合确定性契约的autotune兼容适配，不能直接全局豁免benchmark。
日志另有ArgSort int32/int64在AiCpu执行警告，保留实际设备回退证据，
不宣称所有算子均在AiCore执行。进程已确认终止，无需重复启动；
未生成仓库根torch_compile_debug目录，故没有可移动的该目录。
本轮无生产代码修改，仅补验并记录新的可定位失败，完整目标保持未完成。

#### 确定性首配置策略诊断：暴露后续codegen缺口（2026-10-23）

读取安装源码确认NPUCachingAutotuner.run已有确定性reduction分支：编译全部
候选后选第一个成功launcher，不执行benchmark。为避免关闭确定性或全局
豁免benchmark，创建仅位于outputs的诊断入口
outputs/npu_deterministic_autotune_probe_20261023.py，仅在torch_npu2.12.0
进程中将该策略扩展到metadata声明确定性的内核；非确定性情况仍调用原方法。
没有改依赖文件、生产代码、模型、断言或全局benchmark保护；此进程局部类
方法替换是诊断，不是正式验收或可发布适配。

沿用上一节双卡PP2/default/Inductor环境，依次运行
`torchrun --standalone --nproc_per_node=2 outputs/npu_deterministic_autotune_probe_20261023.py full outputs/pp2-inductor-first-config-probe-20261023`
及相同参数的resume（以&&连接）。日志
pp2_inductor_first_config_probe_20261023.log含EXIT_CODE=1，进程已结束。
首次benchmarking拒绝不再出现，但rank1后续codegen在
torch_npu/_inductor/codegen/triton.py::_split_iteration_ranges的
add_multiple_range中执行raise CantSplit()，触发
TypeError: CantSplit.__init__() missing 2 required positional arguments:
expr and remaining。原有分割失败的异常又被构造签名不兼容遮蔽，不能只补
异常参数就声称kernel布局正确。full未通过，resume未执行。
调试目录已移至outputs/pp2-inductor-first-config-debug-20261023。
本证据说明benchmark适配不是整模型编译的充分修复，下一步需检查实际
range-split输入与torch2.12接口契约；不将诊断策略移入生产。
仅更新状态文档，git diff --check通过，整体目标仍未完成。

#### CantSplit契约与真实range输入诊断（2026-10-23）

已复核torch_npu is_compatible的契约：调用_split_iteration_ranges成功返回
True，捕获CantSplit返回False。因此这里抛CantSplit是拒绝不兼容融合候选的
正常控制流，不必然表示模型本身不能编译；当前异常构造签名错误破坏了拒绝路径。
独立SizeVarAllocator/V.graph探针exit0，日志
inductor_split_contract_cpu_20261023.log：groups=[6,4]、lengths=[[6,4],[]]
返回True；同numel但不可整除的[[8,3],[]]触发CantSplit构造TypeError，
而不是预期False。该探针只验证依赖契约，不计NPU模型通过。

新增outputs/npu_inductor_split_probe_20261023.py诊断入口：保留上一轮
确定性首配置策略，包装原_split_iteration_ranges，仅将精确匹配的
CantSplit构造TypeError转换为带expr/remaining的原CantSplit，让既有
is_compatible自行返回False；不将不兼容候选改为True，不跳过编译。
真实输入已捕获groups=[4,64,128,1]对lengths=((128,64,4),[1])或
((64,4,128),[1])的拒绝，见pp2_inductor_split_probe_20261023.log。
同双卡PP2/default/Inductor环境下运行该入口full，后以&&运行resume，
共享outputs/pp2-inductor-split-probe-20261023；原脚本断言保持不变。
启动后已核实进程运行中（exec session27961），尚无最终结果，不计通过；
后续必须轮询同一进程，不能因观察超时重复启动。所有包装仍仅作诊断。

上述同一session27961现已终止，exit1；非重启运行。is_compatible正常拒绝
候选后，rank1反向生成代码进入预编译，却因NoTritonConfigsError失败：
tl.sum(tmp51,2).reshape(X1BLOCK_SUB,X0BLOCK_SUB,1)报告
reshape() cannot change total number of elements in tensor。完整日志
pp2_inductor_split_probe_20261023.log保留；实际生成的反向output_code.py
中相同表达式位于1061行（下一项归约1066行），已归档到
outputs/pp2-inductor-split-debug-20261023/run_2026_10_22_19_43_30_709235-pid_384194/torchinductor/model__1_backward_6.1/。
该路径已先后经benchmark、CantSplit和本次反向reshape三轮诊断复验失败；
按连续三轮规则暂转独立验收项，不继续堆叠未经验证的依赖包装。
full失败、resume未执行；生产配置仍失败，未将任何诊断补丁移入生产。
下一次解锁需要匹配此torch/torch_npu/Triton组合的合法归约代码生成，
不能通过关闭断言、屏蔽编译错误或声称模型只需前向来通过。

### P2 PP×validation：Interleaved1F1B补覆盖（2026-10-23）

转向不依赖整模型Inductor的独立验收项。原pp_validation_equivalence.py
仅固定1F1B，新增LLMTUNER_TEST_PP_SCHEDULE选项（默认不变，仅允许
1F1B/Interleaved1F1B）。后者使用6 decoder层、每rank两个本地stage、
4 microbatches、global batch8，并断言实际model_parts数为2，避免空泛覆盖。
原有独立完整模型参考、loss容差1e-5、验证前后optimizer/scheduler/token/
gradient零容差及train模式恢复断言均保持。该测试未启用整模型compile；
原生attention内部的编译区域不等于整模型编译，不能混淆两类证据。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 LLMTUNER_TEST_PP_SCHEDULE=Interleaved1F1B torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_validation_equivalence.py'
```

| 配置 | 基线/阈值 | 实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- | --- |
| NPU2，PP2 Interleaved1F1B，FP32，两步训练且每步validation2 batches | 独立非PP同权重/同microbatch参考，loss AE≤1e-5，状态精确不变 | validation loss 4.8117075274/4.7618034117；maxAE9.84438e-7；状态断言全满足 | 通过 | 0 | pp_interleaved_validation_20261023.log |
| 默认1F1B回归，NPU2（设备2,3），其余原脚本默认 | 同上 | loss 4.7859743180/4.7065513365，与参考精确一致 | 通过 | 0 | pp_1f1b_validation_regression_20261023.log |

两组进程均已结束。调试文件移至outputs/pp-validation-debug-20261023。
全量Ruff、目标format、compileall及git diff --check通过；副本定向
pre-commit全部适用hook通过，日志pp_interleaved_validation_precommit_20261023.log。
本轮仅改集成测试，未改生产代码，未重跑CPU全套（最近4086 passed/13 skipped）；
全量pre-commit既有无关失败不记为通过。该结果仅覆盖明确双卡PP布局，
不扩展为所有PP×validation×FSDP/CP组合支持。P4与整模型Inductor等缺口仍在。

### P2 PP2×native FSDP2×validation：新增非空泛四卡失败用例（2026-10-23）

扩展pp_validation_equivalence.py以LLMTUNER_TEST_WORLD_SIZE=4运行，保持
默认双卡行为。四卡时global batch随DP度扩展，参考按DP×microbatch分块；
重建独立参考前将真实DTensor.full_tensor收齐再转CPU。新增dp_shard==2、
各part为FSDPModule、具有Shard placement及local_numel<global_numel断言。
不通过删除梯度/状态断言来适配新布局。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SCHEDULE=Interleaved1F1B torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_validation_equivalence.py'
```

失败，exit1，pp_interleaved_fsdp_validation_20261023.log：四rank在首次
validation之后的梯度状态断言失败，已有DTensor梯度变None。该组合不能计
通过，不将原双卡成功扩大为四卡。随后仅诊断入口
outputs/pp_validation_reshard_probe_20261023.py在原validate成功后调用每个
native FSDPModule的公开reshard()，同四卡环境运行，日志
pp_interleaved_fsdp_validation_reshard_probe_20261023.log，exit1且同样
None mismatch；排除“仅是未分片参数导致暂时看不到梯度、reshard即可恢复”
这一充分修复假设，未改生产代码。

源码新线索：torch2.12 pipelining/schedules.py::_initialize_pp_stages在
metadata初始化后调用PipelineStage._post_metadata_inference_cleanup；
stage.py:2042该方法会对FSDP模块reshard后显式param.grad=None。
这是可能破坏validation前已有梯度的清理路径，但尚未通过调用观测定位到
确切清理时点，不将源码线索写成已验证修复。下一步应捕获该调用前后梯度，
在生产拥有的pipeline/validation边界保存状态，不能全局禁用元数据清理。

四卡扩展后的双卡1F1B及Interleaved1F1B均回归exit0，日志
pp_validation_dp_extension_regression_20261023.log：两组loss最大误差分别0及
9.84438e-7，原状态断言均通过。CPU全套exit0：4086 passed/13 skipped、
273 warnings、172.76s，cpu_pp_validation_matrix_20261023.log。
全量Ruff、定向format/pre-commit、compileall和diff通过；定向hook日志
pp_fsdp_validation_precommit_20261023.log。debug归档
outputs/pp-fsdp-validation-debug-20261023。无生产修改，完整目标仍未完成。

### PP×FSDP validation梯度保持根因修复（2026-10-23）

实际调用观测已完成，pp_fsdp_validation_cleanup_trace_20261023.log exit1
保留原失败：validation进入时各part梯度12/12、22/22或13/13完整；模式
切换后的PipelineStage metadata cleanup在no_grad下运行，退出validation
时各part梯度全None。源码中cleanup执行reshard并将分片param.grad=None，
单独追加reshard无法恢复。原梯度引用保持诊断exit0，日志
pp_fsdp_validation_preserve_grad_probe_20261023.log，两步validation和训练
状态全部原断言通过，loss maxAE4.92219e-7，未关闭上游清理或修改阈值。

生产修复位于trainer/runtime/validation.py：进入验证前保存各参数grad引用，
在已有finally中恢复（包括None），不克隆梯度、不改参数身份、不全局patch。
保留metadata清理及其临时梯度回收。CPU新增成功/异常退出两种测试，覆盖
多个model parts、已有梯度与原None梯度及引用身份；定向83 passed。
日志validation_grad_restore_cpu_20261023.log，exit0。

生产路径无诊断入口四卡复验：
```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_WORLD_SIZE=4 && { LLMTUNER_TEST_PP_SCHEDULE=Interleaved1F1B torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_validation_equivalence.py && torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_validation_equivalence.py; }'
```
两组均通过，总exit0，pp_fsdp_validation_grad_restore_20261023.log。
实际PP2×native FSDP2 shard断言、optimizer/scheduler/token/gradient零容差
保持、训练模式恢复全部满足；两种调度validation loss对独立参考maxAE均
4.92219e-7（阈值1e-5），不扩展为所有并行组合通过。
来源映射已更新。全量Ruff、定向format/pre-commit、compileall、diff通过，
hook日志validation_grad_restore_precommit_20261023.log。
修改后的CPU全套已启动，日志cpu_validation_grad_restore_full_20261023.log，
exec session81410尚待最终结果，不能沿用修改前4086结果作为本轮通过。
P4/default及整模型Inductor等其他缺口保持原状态，完整目标仍未完成。

#### 梯度恢复修复最终回归门禁（2026-10-23）

同一CPU全套session81410已结束，exit0：4088 passed、13 skipped、
273 warnings、154.42s，cpu_validation_grad_restore_full_20261023.log。
新增两种梯度恢复单测使总通过数较修改前增加2；不复用修改前结果。
真实NPU2（设备4,5）validation_rng_coordination.py复验exit0，覆盖各rank
setup/cleanup的OSError、KeyboardInterrupt及SystemExit故障协调；日志
validation_grad_restore_rng_coordination_20261023.log。该测试验证异常协调，
梯度状态由新增CPU异常测试及四卡正常路径原断言分别证明。

全量format仍失败于未改的tp_ep_cp_training_equivalence.py三个lambda格式，
438文件已格式化，validation_grad_restore_format_full_20261023.log。
全量pre-commit首次在既有副本上返回0，但随后cmp发现副本hf_inspection.py
已被旧hook自动修过，与当前工作区不一致，因此该次绿色不计本工作区通过。
重新同步所有git跟踪文件到隔离副本再运行，exit1，日志
validation_grad_restore_precommit_synced_full_20261023.log：旧Ruff0.9
自动改hf_inspection.py两处isinstance tuple为union，其余hook通过。
未将这些无关修改复制回工作区。定向三文件hook已通过，不能替代全量门禁。
compileall、全量Ruff及git diff --check通过。调试文件归档
outputs/validation-grad-restore-final-debug-20261023，生产根目录不留生成物。
梯度保持缺口在上述明确四卡配置下已修复；其他缺口及整体未完成结论不变。

### Chunked×FSDP边界所有权独立诊断（2026-10-23）

重新读取原任务及现有guard，未解除Trainer的chunked_loss_fsdp保护。
生产native FSDP在非权重绑定模型中将final norm/lm_head组成同一单元，
但chunked训练先调用decoder，再在root forward外逐块调用head；需验证
这一边界，不能假设等待尚未发布的unshard context是唯一解锁方式。

独立outputs/chunked_fsdp_head_probe_20261023.py使用双NPU、FP32、
Linear16→32、13 tokens分3个不等长chunk、真实fully_shard及生产
chunked_lm_head_cross_entropy，与独立全logits参考对比loss/input-grad/
full weight-grad（atol=rtol=1e-5）。单独head作为FSDP root全部通过，
exit0，chunked_fsdp_head_probe_20261023.log。输入各rank相同，故本项不
证明不同DP样本归约；也不代表整模型、tied weights或checkpoint通过。

增加CHUNKED_GROUP_NORM=1，模拟norm/head共享FSDP单元：首轮缺父root，
被FSDP单root校验拒绝，chunked_fsdp_grouped_norm_probe_20261023.log
exit1，属诊断装配错误，不作功能缺口依据。补上单一fully_shard父root，
root.forward只调用norm、head由chunked函数在外调用，真实运行exit1，
chunked_fsdp_grouped_norm_root_probe_20261023.log：首次head linear
报mixed torch.Tensor and DTensor（尚未到第二chunk）。因此失败不仅可能
来自第二次backward释放，还涉及共有FSDP单元的前向调用与reshard边界；
独立head通过不能推广为生产guard可以删除。

复现：PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 outputs/chunked_fsdp_head_probe_20261023.py；
共单元变体另设CHUNKED_GROUP_NORM=1。脚本现包含父root修正，早期单独head
行为不变，失败日志各自保留。下一步需验证保持参数共享/identity及检查点键的
合法边界拆分或完整root生命周期，而非直接绕过guard。此轮仅outputs诊断
及状态文档变更，无生产修改；git diff --check通过，整体任务仍未完成。

#### 相同父root下独立norm/head单元及不同DP输入对照（2026-10-23）

在同一outputs/chunked_fsdp_head_probe_20261023.py新增
CHUNKED_SEPARATE_HEAD=1，仅改变norm/head各自fully_shard而非列表共单元，
父root及root外chunked head调用保持不变。双NPU相同输入对照exit0，
chunked_fsdp_separate_norm_root_probe_20261023.log：三不等长chunk的loss、
input-grad、head及norm全局参数梯度满足atol=rtol=1e-5。故共单元失败
不能推广为所有分片head多backward必然失败；候选解法是独立合法边界。

继续增强同一探针：head/norm初始化仍相同，输入以100+rank独立seed生成，
参考各rank对自己的全logits backward，然后显式all_reduce/除world得到
FSDP默认平均语义的head/norm梯度；输入梯度与各自局部参考比较，不错误平均。
此增强探针exit0，chunked_fsdp_separate_norm_distinct_dp_probe_20261023.log，
排除只因相同DP数据而隐藏归约问题。命令：
```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 CHUNKED_GROUP_NORM=1 CHUNKED_SEPARATE_HEAD=1 torchrun --standalone --nproc_per_node=2 outputs/chunked_fsdp_head_probe_20261023.py'
```
以上仍是局部诊断，不是Trainer验收。生产使用sum-grad语义、完整decoder、
可能绑定embedding/head或其他参数别名，尚须逐项验证身份/分片所有权、
不同DP输入的整模型更新及checkpoint；不能据此删除chunked_loss_fsdp或
fsdp_unshard_context保护。未改生产代码或阈值，未将诊断作为正式通过。
同一探针文件已增强为不同输入参考，旧相同输入日志保留；所有运行已结束。
本轮git diff --check通过，完整目标未完成。

#### 完整未绑定decoder四步与生产sum-grad语义诊断（2026-10-23）

新增outputs/chunked_fsdp_decoder_probe_20261023.py：真实HF llama decoder
（vocab128/hidden64/intermediate128/2层/4heads），明确未绑定权重，
embedding、各layer、norm、head分别fully_shard，最后shard父root。
每rank不同输入，63 tokens分4个不等长chunk，生产chunked CE函数，4步
AdamW，与同权重独立全logits模型比较每步loss、所有参数full梯度及更新。
梯度atol1e-5/rtol1e-4，参数atol=rtol=1e-4，loss atol1e-4/rtol1e-5。
此为诊断手写循环，不是双方真实Trainer验收，也未改Trainer guard。

默认FSDP平均语义下全部通过，exit0，
chunked_fsdp_decoder_separate_probe_20261023.log，4步最终参数maxAE约
5.40167e-8。随后CHUNKED_SUM_GRAD=1使用本仓库force_sum_grad_reduction
及disable_fsdp_gradient_division，并以全局126 tokens归一化，各rank独立
参考梯度显式SUM；复验exit0，chunked_fsdp_decoder_sum_grad_retry_20261023.log，
同样全部原断言通过。第一次sum探针漏调disable_fsdp_gradient_division，
导致梯度缩放不匹配exit1，chunked_fsdp_decoder_sum_grad_probe_20261023.log
保留；该错误属于诊断装配，未降低断言来修正。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 CHUNKED_SUM_GRAD=1 torchrun --standalone --nproc_per_node=2 outputs/chunked_fsdp_decoder_probe_20261023.py'
```
以上支持进一步实现未绑定模型的合法独立head边界，但尚缺真实Trainer
装配/恢复、tied参数及别名所有权测试，不能将“手写循环通过”替代目标验收。
后续生产方案须在保留保护语义前提下明确支持条件，拒绝未验证布局；当前
未修改生产代码、capability或guard。所有探针进程已结束，diff检查通过。

#### 权重绑定完整decoder边界反例（2026-10-23）

同一decoder诊断入口新增CHUNKED_TIED_WEIGHTS=1，以HF
tie_word_embeddings=True构造并断言enable_weight_tying；embedding/head
共用fully_shard单元，分片后明确断言weight仍同一对象，不通过复制参数来
伪造支持。norm和layers保持独立单元、父root存在、其他四步对照逻辑不变。
双NPU sum-grad路径exit1，chunked_fsdp_decoder_tied_probe_20261023.log，
首个head调用报Tensor/DTensor混用。该首轮共享单元用了默认前向reshard，
不能直接归因于生产默认策略。

进一步CHUNKED_KEEP_BOUNDARY=1将共享边界reshard_after_forward=False，
父root也显式False，保留参数身份；复验仍exit1、相同mixed Tensor/DTensor，
chunked_fsdp_decoder_tied_keep_boundary_probe_20261023.log。未走到第二
chunk，不能把失败写成已证明第二backward释放。全部进程已终止。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 CHUNKED_SUM_GRAD=1 CHUNKED_TIED_WEIGHTS=1 CHUNKED_KEEP_BOUNDARY=1 torchrun --standalone --nproc_per_node=2 outputs/chunked_fsdp_decoder_probe_20261023.py'
```

这明确阻止将未绑定模型的独立head成功泛化到绑定布局。后续设计需保持
共享参数唯一所有者并使root外head forward具备合法FSDP生命周期，或者将
chunked计算纳入完整root调用边界；不可分开shard同一物理参数、解除tie或
静默复制权重。探针仍手写循环诊断，不替代真实Trainer验收。
本轮仅outputs脚本及状态文档变化，未改生产guard/能力声明/数值阈值，
git diff --check通过；无生产改动故未重跑CPU，最近生产全套4088/13。
完整目标保持未完成。

#### 绑定head公开unshard/延迟reshard仍有梯度错误（2026-10-23）

在上一节绑定参数身份及不前向reshard配置之上，新增仅诊断开关
CHUNKED_HOLD_HEAD=1：decoder forward后调用head公开
set_reshard_after_backward(False,recurse=False)及unshard()，chunked计算
结束后恢复True并reshard()。未调用私有state，也未改共享weight引用。
同一四步参考梯度/权重断言及SUM/全局token语义保持，不关闭校验。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 CHUNKED_SUM_GRAD=1 CHUNKED_TIED_WEIGHTS=1 CHUNKED_KEEP_BOUNDARY=1 CHUNKED_HOLD_HEAD=1 torchrun --standalone --nproc_per_node=2 outputs/chunked_fsdp_decoder_probe_20261023.py'
```

进程已结束，exit1，chunked_fsdp_tied_public_hold_probe_20261023.log。
此策略不再触发首次head Tensor/DTensor混用，loss比较通过且chunked函数
完成，但两个rank均在step0的model.model.embed_tokens.weight全局梯度
对照断言失败；未进入optimizer更新，更不能宣称四步通过。日志的自定义
assert消息只指明参数，未保存该梯度的具体误差，因此不推测错误倍数或
断言来自embedding/head哪部分贡献。成功执行不等于累加/归约正确。

绑定布局已连续经历默认共享边界、保持前向边界、公开hold三轮复验失败，
本轮取得从调用错误推进到数值失败的新证据，但按规则暂转独立项目，不
继续堆叠未验证包装。进一步解锁须分离追踪共享weight的embedding与head
梯度贡献、每次chunk归约及最终decoder backward，保证唯一参数所有者。
此策略仍只在outputs，不提升为生产实现；生产guard、支持声明及阈值不变。
本轮git diff --check通过，无生产修改，整体目标仍未完成。

### P3 torch_checkpointing×FSDP2×AC×EMA恢复补验（2026-10-23）

绑定chunked三轮失败后推进独立验收项。扩展生产集成测试
npu_checkpoint_backend_equivalence.py，新增--ac none/full/selective（默认
none不变），传入真实activation_checkpoint_mode，并在非none时断言
CheckpointWrapper实际存在。保留FSDP真实Shard、EMA确实更新及连续4步
与2+2恢复所有状态零容差断言，未替换checkpoint后端或放宽恢复标准。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tests/integration_tests/npu_checkpoint_backend_equivalence.py --ema dynamic --ac full --output outputs/torch-checkpointing-fsdp2-fullac-dynamic-20261023'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 tests/integration_tests/npu_checkpoint_backend_equivalence.py --ema fixed --ac selective --output outputs/torch-checkpointing-fsdp2-selectiveac-fixed-20261023'
```

| 配置 | 基线/断言 | 实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- | --- |
| NPU2，native FSDP2，full AC，动态EMA，torch_checkpointing | 同总scheduler步数，连续4步vs2+2，七类状态精确一致 | 两rank的model/optimizer/lr_scheduler/ema/train_state/dataloader/rng_state全精确 | 通过 | 0 | checkpointing_fullac_dynamic_20261023.log |
| NPU2，native FSDP2，selective AC，固定EMA，torch_checkpointing | 同上 | 同上 | 通过 | 0 | checkpointing_selectiveac_fixed_20261023.log |

全部进程已结束，日志包含EXIT_CODE=0；实际AC装配覆盖两层且非空wrapper
断言成立。该结果证明指定AC配置下可选后端恢复，不证明AC相对无AC的
独立数值等价或所有并行/模型组合；此前AC独立等价证据仍需按各自范围解释。
本轮只改集成测试，未改生产代码。全量Ruff、目标format、compileall及
diff通过，定向副本pre-commit通过，checkpointing_ac_precommit_20261023.log。
未重跑CPU全套，最近生产修改门禁仍4088 passed/13 skipped；全量格式与
pre-commit既有无关失败不记为本轮通过。整体任务仍未完成。

### Chunked未绑定完整decoder原型的DCP恢复诊断（2026-10-23）

继续原完整decoder/独立norm-head/SUM-grad诊断，增加真实DCP在step2保存、
完成连续step4后加载step2再执行step3/4。参考模型和数据generator在诊断
进程内保存step2状态并恢复；最终要求实际本地参数分片及AdamW全状态与
连续step4零容差一致，不用对参考的近似阈值替代恢复精确性。

原生get_state_dict/set_state_dict首次恢复失败，exit1，日志
chunked_fsdp_decoder_dcp_probe_20261023.log：Missing optimizer state for
parameter tok_embeddings.weight，涉及HF wrapper别名/FQN不一致；前4步
独立数值比较通过不代表DCP恢复通过。未使用strict=False跳过缺失状态。
改用model.state_dict的HF键及本仓库get_flat_optim_state_dict/
load_flat_optim_state_dict，为优化器显式设置真实named_parameters的
param_names，加载model strict=True。该方案仍使用真实dcp.save/load。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 CHUNKED_SUM_GRAD=1 CHUNKED_DCP_OUTPUT=outputs/chunked-decoder-dcp-fqn-probe-20261023 torchrun --standalone --nproc_per_node=2 outputs/chunked_fsdp_decoder_probe_20261023.py'
```

复验exit0，chunked_fsdp_decoder_dcp_fqn_probe_20261023.log，两rank均
报告DCP 2+2 final parameter shards and AdamW state精确一致，恢复后两步
亦保留所有梯度及全logits参考参数更新断言。所有进程已终止。
范围限制：同进程恢复且数据generator/reference状态保存在内存，不是冷启动、
真实Trainer、scheduler/EMA/loader/RNG完整恢复，也不解决绑定权重失败。
因此仍只计原型诊断证据，生产chunked×FSDP保护保持不变；后续必须走
真实Trainer的FQN状态与冷恢复验收，不能据此缩减最终目标。本轮仅outputs
和本状态文件变更，git diff --check通过，无新增生产通过声明。

### Chunked保护的解锁说明纠正（2026-10-23）

复核生产装配接口后，本轮未直接接入独立head方案：真实Trainer仍缺安全的
绑定/参数别名所有权裁决及冷启动恢复证据。现有compatibility错误提示将
未来FSDPModule.unshard context API描述为解锁条件，而绑定原型已证明
持有存储仍可产生错误梯度；该提示会误导用户仅升级API便期待正确支持。

修改parallel/compatibility.py::chunked_loss_fsdp的docstring及错误提示，
明确共享单元、root外head调用、参数身份/梯度归约/恢复的完整要求，保留
chunked_loss_num_chunks=1的可执行替代配置。accelerator/capabilities.py
同步明确fsdp_unshard_context仅探测API形状，不证明chunked训练正确。
未改变probe返回逻辑、guard条件、异常类型或支持范围，不声称功能已解锁。

定向能力/组合测试exit0：17 passed，
chunked_guard_capability_guidance_cpu_20261023.log；全量Ruff、两文件format
及定向pre-commit通过，chunked_guidance_precommit_20261023.log。
git diff --check通过。本轮改动仅提示文字/docstrings，未新增NPU训练结论，
未重跑CPU全套；最近行为修复全套仍4088 passed/13 skipped。
后续仍须真实Trainer支持实现与验证，整体目标未完成。

### 当前总表证据复核（2026-10-23）

本轮重新读取原始目标，确认上一轮 Prompt 编辑仅改变任务说明，不算适配进展。
当前 TorchLLMTuner HEAD 为 c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19，
torchtitan-npu HEAD 为 b358a8063c34aa7494397efa3a8f5ad24f1caa1f；
工作区已有修改保留。容器 pgrep 未发现活跃 torchrun/指定诊断任务，未重启任务。

直接复读下列日志而非仅引用历史摘要：p2_p3_20261022.log、
pp_fsdp_validation_grad_restore_20261023.log、tp_inductor_default_backend_20261023.log、
checkpointing_fullac_dynamic_20261023.log、checkpointing_selectiveac_fixed_20261023.log、
p4_norm_order_repeat_exact_audit_v2_20261023.log、pp2_inductor_split_probe_20261023.log、
chunked_fsdp_tied_public_hold_probe_20261023.log。日志均位于本文件约定的
outputs/qwen3-8b-npu/logs/。补充总表有限子项，消除 P2/P3 总行“未跑”被误解为
全部未执行的问题；总行仅表示完整覆盖未证明，不升级整体通过结论。

复读 packed_pp2_ulysses2_fsdp2_fullac_resume_20261022.log 和
packed_pp2_ulysses2_fsdp2_interleaved_selective_resume_20261022.log 时发现，
两者虽数值/状态断言通过，均明确报告 aten::_assert_async.msg 回退 CPU，
以及 int32/int64 ArgSort 在 AiCPU 执行。原始目标同时要求目标路径生效及无静默
回退；现有证据不能支持无回退验收。新增总表失败行，保留历史数值证据，不删除
失败记录、不放宽标准。下一步需定位触发算子的调用栈，区分框架检查/排序与模型
计算；在修复或用户明确确认允许边界前，不把该组合计为完整目标通过。

P4 默认路径失败数值另直接核对 p4_candidate_early_deterministic_20261023.log
第61步 AssertionError，非融合诊断精确结果不替代此项。本轮仅状态文件修改，
git diff --check通过，未重跑训练或CPU测试；所有历史子项仍限于原运行版本和配置，
后续受影响生产修改必须重验。完整目标保持未完成。

### packed 回退调用栈与后端注册实测（2026-10-23）

上一轮为进展：发现回退并纠正总表。这轮运行真实 packed PP2×Ulysses2 四卡
Trainer/full AC/dynamic EMA，四步保存step2；使用 outputs 下 TorchDispatchMode
只记录首次 assert/sort/unique 调用栈，所有算子原样转发，无替代实现或删除断言。
运行前 npu-smi 确认无训练进程，设备健康；运行正常结束，EXIT_CODE=0。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=2 LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_CP_STRATEGY=ulysses LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_AC=full LLMTUNER_TEST_PP_SCHEDULE=1F1B LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1; torchrun --standalone --nproc_per_node=4 outputs/trace_packed_fallback_20261023.py full outputs/packed-fallback-trace-20261023 --packed'
```

证据日志 packed_fallback_trace_20261023.log（位于约定日志目录）：

- 首次 aten::_assert_async.msg 来自 trainer.py:635 的 global_valid_tokens>0
  校验，不是 attention 或 loss 主计算；必须保留“全mask batch不更新参数”的保护。
- aten.sort.stable 来自模型 preprocess → get_attention_masks →
  create_attention_mask → PyTorch create_block_mask → _dense_to_ordered。
  该上游函数先转 int32 再执行 stable argsort，触发 ArgSort AiCPU；不是 MoE routing。
- 四步 loss 为4.8854408264、4.5301513672、4.3757314682、4.2841176987，
  仅用于复现定位；本次没有 resume，也不替代八卡FSDP恢复验收。

额外单卡实际调度探针 npu_assert_sort_dispatch_probe_20261023.log，EXIT_CODE=0：
_dispatch_has_kernel_for_dispatch_key 显示 aten::_assert_async 及其 msg overload
均无 PrivateUse1 kernel，而 aten::sort.stable 有。固定二值输入0/1的 int32及
FP32稳定降序结果均与预定索引[1,3,4,0,2,5]零容差相同；int32出现AiCPU警告。
该单样例不证明通用浮点排序、反向mask转置或完整BlockMask可安全替换；不能把
一次无警告当作AiCore profiler证据。未向生产注入私有算子替换或全局patch。

定位结果将后续修复约束收敛为：NPU原生安全断言/明确同步策略，以及保持稀疏
mask构造和转置语义的排序实现；仅删断言、隐式转CPU或修改全局torch helper均
不满足目标。当前版本缺少原生断言kernel，严格无回退验收仍未通过；不能把已明确
报告的回退称为“静默回退”，也不能未经确认将控制面回退排除出验收范围。
本轮未改生产代码，不需重跑CPU全套；git diff --check通过，完整目标保持未完成。

### packed Ulysses消除未消费的BlockMask构造（2026-10-23）

本轮修改models/hf/model.py::_shard_for_cp：packed且Ulysses非flex时直接调用
原来最终使用的build_dense_attention_mask；原路径先get_attention_masks构造
BlockMask，随后无条件丢弃。保持全局positions、block_causal=True、dtype和
mask创建在token分片前的顺序；不更换attention内核，不修改torch helper，
flex Ulysses和KV-allgather分支保持原语义。来源映射已补充B类适配记录。
CPU测试要求dense路径绝不调用稀疏mask构造/分片，覆盖FP32/BF16及跨文档和
未来位置屏蔽；另测flex保留原全局mask对象，不误转dense。

| 配置 | 基线/阈值 | 结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- | --- |
| CPU HF wrapper | 原断言及新增dense/flex路径回归 | 84 passed | 通过 | 0 | packed_dense_mask_cpu_20261023.log |
| NPU2 packed Ulysses，FP32，doc lengths90/1/165 | 独立dense参考，atol/rtol1e-5；去文档mask须产生差异 | logits maxAE2.570e-7，平均loss AE4.768e-7，参数梯度maxAE2.980e-8；doc gap0.6232；QKV路径输出/梯度AE0 | 通过 | 0 | packed_dense_mask_npu_equivalence_20261023.log |
| CPU2 Gloo，原flex路径，FP64 | 原1e-9阈值；不宣称CPU整网参数反向 | logits/loss/QKV输出及梯度AE0 | 通过 | 0 | packed_dense_mask_cpu_gloo_20261023.log |
| NPU8 PP2×Ulysses2×FSDP2，1F1B/full AC，dynamic EMA | 连续4步vs冷启动2+2，各rank真实分片，全部启用状态精确 | 8rank pre-update failures=0，最终状态及loss精确，loss最大AE0 | 通过 | 0/0 | packed_dense_mask_1F1B_full_20261023.log / packed_dense_mask_1F1B_resume_20261023.log |
| NPU8 同上，Interleaved1F1B/selective AC | 同上，6层/4stages/4microbatches | 同上 | 通过 | 0/0 | packed_dense_mask_Interleaved1F1B_full_20261023.log / packed_dense_mask_Interleaved1F1B_resume_20261023.log |
| CPU全套 | 原断言 | 4090 passed,13 skipped,273 warnings，143.49s | 通过 | 0 | packed_dense_mask_cpu_full_20261023.log |

复现命令（均在容器内仓库根目录；日志均在约定目录）：

```bash
PYTHONPATH=. pytest -q --disable-warnings tests/unit_tests/cpu/models/test_hf_wrapper.py
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tests/integration_tests/cp_ulysses_varlen_equivalence.py
PYTHONPATH=. LLMTUNER_TEST_DEVICE=cpu torchrun --standalone --nproc_per_node=2 tests/integration_tests/cp_ulysses_varlen_equivalence.py
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_PP_SIZE=2
export LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_CP_STRATEGY=ulysses
export LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_REQUIRE_SHARDED=1
export LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
for schedule in 1F1B Interleaved1F1B; do
  export LLMTUNER_TEST_PP_SCHEDULE=$schedule
  if [ "$schedule" = 1F1B ]; then export LLMTUNER_TEST_AC=full; else export LLMTUNER_TEST_AC=selective; fi
  for phase in full resume; do
    torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" "outputs/packed-dense-mask-${schedule}-20261023" --packed || exit $?
  done
done
PYTHONPATH=. pytest -q --disable-warnings
```

四份八卡日志及双卡NPU日志均无ArgSort/AiCPU警告；这结合拒绝构造稀疏mask的
CPU测试证明本次删除冗余路径，不单凭无警告推断所有算子都在AiCore。
四份八卡日志仍明确报告aten::_assert_async.msg回退CPU；fused optimizer
选项也明确记录使用NPU for-loop实现，不冒充fused kernel。

纠正此前审计的范围解释：原始目标要求“目标路径实际生效且无静默回退”，
并未要求所有控制面算子纯AiCore。此前将已公开警告的assert/sort回退直接作为
“无回退验收失败”扩大了要求。本轮分开列出数值恢复通过与纯AiCore附加诊断失败，
不删除历史警告，不豁免目标attention/分片路径生效要求，也不靠此解释解决其他缺口。

全量Ruff、目标format、compileall、git diff --check通过。全量format仍仅
未改动tp_ep_cp_training_equivalence.py三处lambda失败，
packed_dense_mask_format_full_20261023.log。实际已跟踪文件复制到独立快照后
执行完整pre-commit，exit1：Ruff0.9仍自动改动未修改hf_inspection.py两处
isinstance；其他hooks通过。日志packed_dense_mask_precommit_full_20261023.log；
定向hooks通过，packed_dense_mask_precommit_targeted_20261023.log。
快照路径见packed_dense_mask_gate_snapshot_path_20261023.txt，不将hook自动改动
写回工作区。全部本轮测试session均已终止。P4默认对齐、KCP架构、整模型编译、
chunked×FSDP、实际多模态训练及完整组合覆盖等缺口仍在，整体目标未完成。

### Chunked绑定权重：贡献拆分与root所有权新证据（2026-10-23）

上一轮是生产优化及复验进展。本轮重新读取原始目标，转回chunked绑定权重缺口，
没有重复通过的packed子集，也没有移除Trainer保护。运行前容器无torchrun任务。

新增outputs/chunked_fsdp_contribution_probe_20261023.py诊断，沿用双卡真实
FSDP、两层Llama、vocab128/hidden64、每rank不同63tokens、四个不等长chunks。
参考模型将embedding/head解绑定但保持同初值，分别计算两部分梯度并DP归约，
以其和校验候选共享参数；参考更新只对总梯度更新一次，再复制到参考head，
避免对共享参数执行两次AdamW。另逐元素检查拼接hidden梯度和全部参数更新，
不以一个grad norm代替逐参数校验。

| 配置/实验 | 断言及实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- |
| 旧embedding/head分组，公开unshard/持有存储，追加set_is_last_backward(False) | step0 shared grad与分解参考之和maxAE0.0574421398；chunk1–3的已归约grad norm不再变化；最后backward仍报告同组embedding未forward | 失败 | 1 | chunked_tied_contribution_defer_last_20261023.log |
| 新root所有权：embedding/head不单独fully_shard，由root共同持有，root reshard_after_forward=False；layers/norm仍独立FSDP | 四步所有hidden/参数梯度及AdamW更新均满足原atol1e-5/rtol1e-4梯度及atol/rtol1e-4权重标准；共享grad与分解参考之和各步maxAE≤1.49011612e-8，最终权重maxAE1.11758709e-8 | 通过 | 0 | chunked_tied_root_owner_contribution_20261023.log |
| 同一root所有权，使用未修改生产chunked_lm_head_cross_entropy，SUM-grad及global126tokens，真实DCP | 连续四步全部梯度/更新对独立全logits参考通过；同进程加载step2后重放step3/4，最终本地参数分片及AdamW状态零容差一致，max weight AE7.45058060e-9 | 通过 | 0 | chunked_tied_root_owner_production_loss_dcp_20261023.log |

本节“通过”仅指对应诊断断言，不是Trainer支持验收。所有日志位于
outputs/qwen3-8b-npu/logs/，完整命令：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 CHUNKED_TIED_WEIGHTS=1 CHUNKED_HOLD_HEAD=1 torchrun --standalone --nproc_per_node=2 outputs/chunked_fsdp_contribution_probe_20261023.py'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 CHUNKED_TIED_WEIGHTS=1 CHUNKED_ROOT_OWNER=1 torchrun --standalone --nproc_per_node=2 outputs/chunked_fsdp_contribution_probe_20261023.py'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=.:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1 CHUNKED_TIED_WEIGHTS=1 CHUNKED_ROOT_OWNER=1 CHUNKED_SUM_GRAD=1 CHUNKED_DCP_OUTPUT=outputs/chunked-tied-root-owner-dcp-20261023 torchrun --standalone --nproc_per_node=2 outputs/chunked_fsdp_root_owner_probe_20261023.py'
```

新root方案不需要给head显式unshard、持有head storage、修改参数/梯度或替换
生产loss，也不使用已失败的“API存在即可解锁”假设。head是普通子模块，其chunk
梯度在root unsharded参数上累积，最后decoder backward才完成root归约；上述
分解参考对照支持这一解释。旧分组中观测到的梯度不再变化只定位生命周期问题，
未据此断言所有错误恰好等于某一个缺失分量。

当前限制仍明确保留：这是outputs原型及同进程DCP，非真实Trainer冷启动，
未覆盖scheduler/EMA/loader/RNG恢复、混合精度、AC、TP/CP/EP、CPU offload、
用户FSDP patterns或跨层alias。生产apply_fsdp_to_decoder仍按既有boundary组
装配，Trainer的chunked×FSDP guard未变。因此不将原型通过标成生产支持。
下一步可基于这项新证据实现root boundary所有权策略，保持非chunked默认不变，
实际参数alias必须纳入所有权合并，再用真实Trainer完成冷启动恢复和独立等价测试；
不能仅用配置tie_word_embeddings布尔值跳过保护。

本轮三个诊断session均已正常终止或以记录的断言失败退出，无活跃测试任务。
仅outputs及状态文件变更，git diff --check通过，未重跑CPU全套；最近生产修改
门禁仍4090 passed/13 skipped。整体目标保持未完成。

### Chunked原生FSDP正式装配与真实Trainer冷恢复（2026-10-23）

上一轮root所有权/梯度贡献原型属于进展；本轮在该新证据基础上正式实现并验证，
不把原型手写循环或同进程DCP替代目标验收。原始目标已重新读取。

生产变更：

- parallelize_hf_transformers/apply_fsdp显式接收training chunk数；普通与
  references重绑定两条调用路径均传递，默认1不改变既有非chunked装配。
- apply_fsdp_to_decoder增加root_owned_boundaries：将实际embedding/norm/head
  参数加入既有root所有权分析。与边界共享参数的层按原identity合并算法传递到root；
  合并含root时只对root自身fully_shard，避免把共享子模块再作为独立hook入口。
  root既有reshard_after_forward=False使外部head chunks保留同一参数，最终decoder
  backward完成归约。不换参数，不手工累加梯度，不改生产chunked loss公式。
- chunked_loss_fsdp不再依赖fsdp_unshard_context API探针。仅放行native默认decoder
  布局且无CPU offload/compile/TP/CP/EP/PP的路径；patterns/ignored/custom等仍明确
  拒绝。低层root策略也拒绝custom/offload/PP/EP。既有PP及validation保护不变。
- CPU测试覆盖绑定/非绑定、配置tie标记与实际身份冲突、跨层norm alias归入root、
  不支持配置拒绝及装配传参。README/design和来源映射已同步，披露root保留完整
  边界权重的驻留内存代价，不将此结论推广到大模型容量或未测精度。

正式集成验证均使用当前生产装配，不绕过guard或修改原生torch。chunked loss
等价脚本新增每步裁剪后参数梯度对独立全logits参考，阈值atol1e-5/rtol1e-4，
保持原loss/参数1e-4阈值；同时检查实际DTensor分片及embedding/head身份。
checkpoint脚本新增显式chunks/tied环境选项，并检查实际root状态持有每个boundary
参数；跨层alias用例检查整层参数实际归root，不仅检查Parameter相同。

| 配置 | 基线/断言 | 实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- | --- |
| NPU2 Llama2层，FP32，chunks3，untied | 真Trainer4步vs独立全logits参考；每步loss/梯度、最终参数；真实分片 | loss maxAE4.768e-7，grad maxAE7.451e-9，params maxAE7.451e-9 | 通过 | 0 | chunked_root_trainer_gradient_equivalence_tied0_20261023.log |
| 同上tied | 同上，验证实际共享身份 | loss maxAE4.768e-7，grad maxAE9.779e-9，params maxAE7.451e-9 | 通过 | 0 | chunked_root_trainer_gradient_equivalence_tied1_20261023.log |
| NPU2 Qwen2层，native FSDP2，chunks3，dynamic EMA，RNG/loader启用，untied | 连续4步vs独立torchrun冷启动2+2；恢复前/后启用状态精确 | 全rank pre-update failures=0，最终model/optimizer/scheduler/EMA/trainer/RNG/loader及loss精确 | 通过 | 0/0 | chunked_root_cold_tied0_full_20261023.log / chunked_root_cold_tied0_resume_20261023.log |
| 同上tied | 同上 | 同上 | 通过 | 0/0 | chunked_root_cold_tied1_full_20261023.log / chunked_root_cold_tied1_resume_20261023.log |
| 同上tied/full AC，layer0 norm与final norm实际alias，检查root所有权 | 同上，实际共享层归root，非重复独立FSDP单元 | 全rank恢复前/后状态及loss精确 | 通过 | 0/0 | chunked_root_alias_fullac_retry_full_20261023.log / chunked_root_alias_fullac_retry_resume_20261023.log |
| 同上untied/selective AC/accumulation2 | 同上，检查真实root boundary所有权 | 全rank恢复前/后状态及loss精确 | 通过 | 0/0 | chunked_root_selective_accum_full_20261023.log / chunked_root_selective_accum_resume_20261023.log |
| CPU定向FSDP/组合/装配/AC | 原断言及新增root/alias/拒绝项 | 237 passed,7 skipped | 通过 | 0 | chunked_root_ownership_cpu_retry_20261023.log |
| CPU全套 | 原断言 | 4100 passed,13 skipped,273 warnings，144.62s | 通过 | 0 | chunked_root_cpu_full_20261023.log |

所有日志在outputs/qwen3-8b-npu/logs/。复现（容器内仓库根目录）：

```bash
for tied in 0 1; do
  PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1 CHUNKED_LOSS_EQ_TIED=$tied torchrun --standalone --nproc_per_node=2 tests/integration_tests/chunked_loss_equivalence.py || exit $?
done
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=2,3
export LLMTUNER_TEST_WORLD_SIZE=2 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native
export LLMTUNER_TEST_REQUIRE_SHARDED=1 LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_TEST_CHUNKS=3
for tied in 0 1; do
  export LLMTUNER_TEST_TIED_WEIGHTS=$tied
  for phase in full resume; do
    torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" "outputs/chunked-root-cold-tied${tied}-20261023" || exit $?
  done
done
export LLMTUNER_TEST_TIED_WEIGHTS=1 LLMTUNER_TEST_AC=full LLMTUNER_TEST_SHARED_BOUNDARY=1
for phase in full resume; do
  torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" outputs/chunked-root-alias-fullac-retry-20261023 || exit $?
done
unset LLMTUNER_TEST_SHARED_BOUNDARY
export LLMTUNER_TEST_TIED_WEIGHTS=0 LLMTUNER_TEST_AC=selective LLMTUNER_TEST_ACCUMULATION=2
for phase in full resume; do
  torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" outputs/chunked-root-selective-accum-20261023 || exit $?
done
unset LLMTUNER_TEST_WORLD_SIZE LLMTUNER_TEST_PP_SIZE LLMTUNER_TEST_FSDP_IMPLEMENTATION
unset LLMTUNER_TEST_REQUIRE_SHARDED LLMTUNER_TEST_EMA LLMTUNER_TEST_RNG_STATE LLMTUNER_TEST_LOADER_STATE
unset LLMTUNER_TEST_CHUNKS LLMTUNER_TEST_TIED_WEIGHTS LLMTUNER_TEST_AC LLMTUNER_TEST_ACCUMULATION
PYTHONPATH=. pytest -q --disable-warnings
```

初次CPU定向失败2项是AC测试旧mock签名未接收新增chunk参数，已补齐显式签名，
未减少断言；保留chunked_root_ownership_cpu_20261023.log。初次alias/full AC
运行在第1步后失败，是集成脚本仍要求layer/norm各有相同独立FSDP state；新策略
正确归root后子层不再是FSDPModule。修正为检查root实际fsdp_params的sharded_param
身份，严格要求整个共享层归root，非删除所有权断言；保留
chunked_root_alias_fullac_full_20261023.log。新目录冷启动复验通过如表。
先前不含逐步梯度的两份chunked_root_trainer_equivalence_tied0/1日志亦exit0，
最终以新增梯度检查日志为更强证据。

全量Ruff、九个改动Python文件format、compileall及git diff --check通过。
全量format仍只在未改动tp_ep_cp_training_equivalence.py的三处lambda失败，
chunked_root_format_full_20261023.log。已跟踪当前文件复制到新独立快照执行完整
pre-commit，exit1仍为Ruff0.9改动未修改hf_inspection.py两处isinstance；其他hooks
通过。定向12文件hooks全部通过；日志chunked_root_precommit_full_20261023.log、
chunked_root_precommit_targeted_20261023.log；快照路径见
chunked_root_gate_snapshot_path_20261023.txt，自动改动未写回工作区。

所有本轮session均已终止。只声明上述NPU FP32、小模型、native默认布局配置；
不能以AC下恢复精确替代独立AC数值等价，不能以本轮通过推广到混合模型并行、
自定义FSDP/offload/compile、BF16或8B chunked训练。P4默认对齐、KCP硬件限制、
整模型Inductor、多模态模型训练及完整声明组合覆盖等仍有缺口，整体目标未完成。

### TP×EP×KVCP 多步训练实际 NPU 路径（2026-10-23）

原始目标重新核对：P3要求多模态管线，并未独立要求多模态模型训练；此前将后者
列为必需缺口的表述过宽。已有PNG管线证据只覆盖对应图像路径，不能推广全部模态。
上一轮Prompt编辑不构成适配进展。本轮补充真实NPU组合证据，目标仍未完成。

工作仓库main/c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19，基线
b358a8063c34aa7494397efa3a8f5ad24f1caa1f；保留已有工作区改动。
`tp_ep_cp_training_equivalence.py`现在通过harness选择真实设备，CPU保留显式
attention替代，NPU使用生产CP SDPA并断言实际backend及每层CPSdpaKernel。
指标对比先detach搬到CPU再用FP64：避免NPU不支持double造成断言dtype错误；
训练、参考模型及attention没有搬到CPU。原atol=rtol=1e-6未改，检查3步每rank
27个参数的梯度、更新及SGD momentum，并保留loss/logits、aux及AC执行次数断言。

以下均为4卡、TP2/KVCP2/EP2、native FSDP、FP32、小模型集成，不是P4真实Trainer验收。
日志目录统一为`outputs/qwen3-8b-npu/logs/`，表内日志均加`_20261023.log`。

| 配置（aux/padding/AC/累积/uneven） | 实际结果 | 状态 | exit | 日志前缀 |
| --- | --- | --- | --- | --- |
| 0/否/none/1/否 | 3步全部断言通过 | 通过 | 0 | tp_ep_cp_npu_multistep_native |
| .01/否/none/1/否 | 同上 | 通过 | 0 | tp_ep_cp_npu_aux_only_ablation |
| 0/是/none/2/是 | 同上 | 通过 | 0 | tp_ep_cp_npu_noaux_padding_ablation |
| 0/是/full/2/是 | 同上，含重计算次数 | 通过 | 0 | tp_ep_cp_npu_noaux_fullac_ablation |
| .01/是/none/1/否 | 3步全部断言通过 | 通过 | 0 | tp_ep_cp_npu_aux_padding_ablation |
| .01/否/none/2/否 | 同上 | 通过 | 0 | tp_ep_cp_npu_aux_accum_ablation |
| .01/是/none/2/是 | 同上 | 通过 | 0 | tp_ep_cp_npu_aux_padding_accum_noac |
| .01/是/full/2/是 | step0 embedding梯度maxAE=5.1378796342760324e-5 | 失败 | 1 | tp_ep_cp_npu_native_aux_fullac_uneven_retry |
| .01/是/selective/2/是 | 相同梯度误差 | 失败 | 1 | tp_ep_cp_npu_native_aux_selective_uneven |
| .01/是/full/2/否 | step0 embedding梯度maxAE=4.578445805236697e-5 | 失败 | 1 | tp_ep_cp_npu_aux_padding_fullac_uniform |
| custom/.01/是/selective/2/是 | torch2.12缺multiple parameter groups及iteration reset API | blocked | 1 | tp_ep_cp_npu_custom_aux_selective_uneven |

复现命令（将末尾参数替换为表中配置；无需aux时用`noaux`）：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 tests/integration_tests/tp_ep_cp_training_equivalence.py native aux padding none 2 uneven'
```

当前消融排除了“不等长累积本身导致失败”：相同输入无AC通过，等长full AC仍失败。
尚未定位根因，不能推断为舍入误差或放宽容差；下一步检查aux注入在AC重放的梯度
语义及独立单微批配置。custom解锁需要实际兼容torch API，不绕过能力保护。
初次full AC运行的dtype失败日志`tp_ep_cp_npu_native_aux_fullac_uneven_20261023.log`
保留，不计功能通过。此前全量CPU日志`tp_ep_cp_native_multistep_cpu_full_20261023.log`
实查为4100 passed/13 skipped/273 warnings，exit0；不替代以上NPU失败。

### AC辅助损失梯度偏差根因及修复（2026-10-23）

后续单微批消融`native aux padding full 1`也失败，embedding maxAE=
9.054597467184067e-5；`native aux nopadding full 1`同样exit1，因此问题并不
要求padding或累积。对应日志`tp_ep_cp_npu_aux_padding_full_1_20261023.log`、
`tp_ep_cp_npu_aux_nopadding_full_1_20261023.log`保留。

诊断日志`tp_ep_cp_ac_spmd_probe_20261023.log`中，原forward的current/sparse mesh
正常，aux归约axes为(cp,tp)；NPU反向重放切换线程后两种mesh均为None，axes变为
(cp,)，且实际group查询为空。该日志exit1，仅插入观测打印，不是验收通过证据。
根因为Python thread-local SPMD状态不跨autograd线程传播，而非浮点容差或
不等长token分母。生产修复捕获原forward的dense/sparse/current mesh，并在
`checkpoint_contexts`每次重放时显式进入该状态，退出时恢复重放线程原状态。
未修改aux公式、梯度归约、原断言或阈值；未把diagnostic monkeypatch用于验收。

| 配置 | 基线/阈值 | 实际结果 | 状态 | exit | 日志 |
| --- | --- | --- | --- | --- | --- |
| NPU4 TP2/EP2/KVCP2 native FP32 aux=.01 padding full AC accum2 uneven | 独立HF，atol=rtol=1e-6 | 3步各rank27项梯度/更新/momentum及所有前向、aux、重放断言通过 | 通过 | 0 | tp_ep_cp_ac_spmd_fixed_full_20261023.log |
| 同上selective AC | 同上 | 同上 | 通过 | 0 | tp_ep_cp_ac_spmd_fixed_selective_20261023.log |
| 同上full AC，等长累积1 | 同上 | 3步全部断言通过 | 通过 | 0 | tp_ep_cp_ac_spmd_fixed_uniform1_20261023.log |
| 同上full AC，等长累积2 | 同上 | 3步全部断言通过 | 通过 | 0 | tp_ep_cp_ac_spmd_fixed_uniform2_20261023.log |
| 同上full AC，无padding，累积1 | 同上 | 3步全部断言通过 | 通过 | 0 | tp_ep_cp_ac_spmd_fixed_nopadding_20261023.log |
| CPU上下文及AC定向 | 包含新线程回归、重复重放及异常清理 | 99 passed/14 warnings | 通过 | 0 | ac_spmd_final_cpu_targeted_20261023.log |

命令沿用上一节4卡torchrun，参数分别为`native aux padding full 2 uneven`和
`native aux padding selective 2 uneven`，本轮使用原脚本而非probe。新增CPU
回归分别检查独立线程恢复全部mesh、重复进入及异常退出，以及实际AC重放context
在独立线程中恢复状态。上述通过只覆盖该小模型FP32组合，不推广BF16/8B或P4。
全量Ruff、format（439文件）、compileall及git diff --check通过。
等长及无padding复验参数分别为`native aux padding full 1`、
`native aux padding full 2`、`native aux nopadding full 1`。
历史失败证据保留，以上生产路径复验关闭本节已复现的AC/aux梯度缺口。
全量pre-commit在独立同步快照exit1：仍仅自动修改未涉及的hf_inspection.py
两处isinstance tuple→union；已核对快照diff，不回写无关改动。其余hooks通过，
本轮6个相关已跟踪文件定向hooks全部通过，日志
`ac_spmd_precommit_full_20261023.log`、`ac_spmd_precommit_targeted_20261023.log`；
快照路径见`ac_spmd_gate_snapshot_path_20261023.txt`。该全量门禁不记通过。
最终代码完整CPU套件（已纳入所有3个新回归用例）4103 passed、13 skipped、
273 warnings，153.39s，exit0；日志`ac_spmd_final_full_cpu_20261023.log`，命令
`docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. pytest -q --disable-warnings'`。
所有本轮训练及测试进程已终止。P4默认100步对齐、KCP硬件限制、整模型Inductor、
custom FSDP依赖及完整声明矩阵等仍未完成；本轮修复不构成整体验收通过。

### TP×EP×KVCP 真实Trainer冷恢复及AC确定性缺口（2026-10-23）

本轮核对原始目标后，将上一节模型/SGD证据推进至真实Trainer、DCP冷恢复。
仓库仍为main/c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19，基线仍为
b358a8063c34aa7494397efa3a8f5ad24f1caa1f，未切分支或覆盖已有改动。
设备检查8卡健康空闲，运行占用0–3；实际日志确认Ascend910_9392/HCCL、
native FSDP、KV all-gather CP2、TP2/EP2，dense DP1、sparse efsdp2。
小模型Qwen3MoE两层/hidden64/vocab128、FP32、seq256、确定性模式、AdamW、
dynamic EMA、RNG及loader状态全部启用；4步连续训练对比独立torchrun的2+2恢复。
本轮未修改生产实现或集成测试；下列严格恢复断言沿用脚本，未放宽容差。

| 配置 | 基线/断言 | 实际结果 | 状态 | full/resume exit | 日志前缀（均加_20261023.log） |
| --- | --- | --- | --- | --- | --- |
| 无AC，真实Trainer | 连续4步vs冷启动2+2，启用状态及loss精确 | 4个rank恢复前state failures=0，最终全部检查通过，loss maxAE=0 | 通过 | 0/0 | tp_ep_cp_trainer_noac_cold_full/resume |
| full AC，真实Trainer | 同上 | 加载时4个rank state failures=0，但续训参数/optimizer/EMA等不精确；step3 loss AE=1.430511474609375e-6，rank0 embedding最终maxAE=8.46153125e-6 | 失败 | 0/1 | tp_ep_cp_trainer_ac_cold_full/resume |

复现（设AC为full或none，目录分别使用`tp-ep-cp-trainer-ac-resume-20261023`或
`tp-ep-cp-trainer-noac-resume-20261023`；各phase独立保存日志并记录exit）：

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1
export LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_CP_SIZE=2
export LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native
export LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic
export LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
for phase in full resume; do
  torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" outputs/tp-ep-cp-trainer-ac-resume-20261023 || exit $?
done'
```

诊断结果另列，不替代上表full AC失败：

- `outputs/probe_cp_gather_order.py`仅打印K/V gather backward次序；full/resume
  均exit0，最终启用状态及loss精确，日志`tp_ep_cp_gather_order_probe_full/resume_20261023.log`。
  插入打印改变时序即通过，尚未取得K/V collective实际错序证据，不据此修改归约实现。
- `outputs/probe_ac_singlethread.py`仅用torch公开autograd上下文关闭多线程反向；
  full/resume均exit0，状态及loss精确，日志`tp_ep_cp_ac_singlethread_probe_full/resume_20261023.log`。
  单次诊断不足以证明所有配置稳定，也未将禁用并行反向作为生产补丁。
- `outputs/compare_tp_ep_cp_split.py`比较原full AC、无AC和带打印诊断的step2
  模型保存点：全部4个rank的模型本地张量精确一致，maxAE=0；日志
  `tp_ep_cp_split_comparison_20261023.log`，exit0。比较不初始化PG，不进行训练或
  全局张量重构；unpickle DeviceMesh提示仅涉及离线诊断。

诊断复现：沿用上面的环境和phase参数，将脚本分别替换为两个probe，并使用独立
`outputs/tp-ep-cp-gather-order-probe-20261023`或
`outputs/tp-ep-cp-ac-singlethread-probe-20261023`目录。离线比较命令为
`docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/compare_tp_ep_cp_split.py'`。
日志明确报告NPU无fused AdamW实现而选for-loop，以及`aten::_assert_async.msg`
CPU fallback；无静默降级，也不宣称纯AiCore执行。

当前证据将缺口限定到此配置full AC冷恢复的精确确定性，未证明是checkpoint
序列化错误或K/V通信错序；下一步应对比恢复后第一步实际前向、原始梯度及图分支
累加顺序，验证多线程autograd调度假设。不得把关闭AC、打印同步或单线程诊断
通过改记为full AC默认路径通过。上一节1e-6等价性通过与本节精确恢复失败属于
不同验收条件，不能互相替代。整体P0–P4目标仍未完成。

### AC冷恢复：逐步快照与未插桩复跑（2026-10-23）

本轮继续原始目标，没有将诊断条件改为验收标准，也未修改生产代码或测试断言。
使用上一节相同4卡环境，给真实Trainer加只读快照：forward后clone输入及logits，
裁剪前clone本地原始梯度，在optimizer step结束后才搬CPU保存。不修改原梯度，
不替换模型、loss、优化器或collective；额外分配/拷贝仍可能改变调度时序。

| 实验 | full/resume exit | 结果及证据 | 用途 |
| --- | --- | --- | --- |
| 仅输入/logits/原始梯度快照 | 0/1 | 恢复前各rank state failures=0，最终loss maxAE=2.861e-6；`tp_ep_cp_ac_step_probe_full/resume_20261023.log` | 捕获到失败的诊断，非验收 |
| 增加positions、裁剪后梯度及更新后参数快照 | 0/0 | 所有恢复断言通过；`tp_ep_cp_ac_update_probe_full/resume_20261023.log` | 插桩诊断，不替代默认路径 |
| 同上并记录梯度layout的第二对运行 | 0/0 | 所有恢复断言通过；`tp_ep_cp_ac_update_probe2_full/resume_20261023.log` | 插桩诊断 |
| 原脚本、无任何诊断patch的独立复跑 | 0/0 | 各rank加载状态及最终所有启用状态精确、loss maxAE=0；`tp_ep_cp_ac_cold_repeat2_full/resume_20261023.log` | 指定运行通过；既有默认路径失败未关闭 |

失败诊断的逐步比较（`tp_ep_cp_ac_step_comparison_20261023.log`，exit0）：

- step3：全部4个rank的输入、logits和全部27项裁剪前本地梯度精确一致。
- step4：全部rank输入仍精确；rank0/1 logits精确，rank2/3 logits maxAE分别为
  0.017953403294086456及0.00025711581110954285；全部rank的27项原始梯度非精确。
  rank0/1最大梯度误差来自lm_head，为5.7217199355363846e-5。
- 该版本没有记录step3的裁剪后梯度、参数更新或position kwargs，不能据此排除
  裁剪/更新/位置输入，也不能将首个因果分歧直接归于step4 attention。

扩大快照的两对运行中，step3/4的输入、positions、logits、裁剪前/后梯度、
更新后参数和训练metrics都精确；第二对梯度shape/stride/placements/mesh也精确。
离线比较日志`tp_ep_cp_ac_update_comparison_20261023.log`和
`tp_ep_cp_ac_update_comparison2_20261023.log`，均exit0。扩大插桩后未捕获失败，
不能将其精确结果外推为原失败时各阶段也精确。

原默认AC训练与未插桩复跑的step2保存点，本地27个模型张量在全部rank均非精确，
maxAE依次为0.00045231450349092484、0.00040895119309425354、
0.0003270599991083145、0.00012663891538977623；日志
`tp_ep_cp_split_repeat_comparison_20261023.log`，exit0。两次训练未保存step0
权重快照，因此仅记录此现象，不能用“seed相同”替代初始权重一致的证明。
未插桩复跑精确恢复只证明该次运行通过；结合既有冷恢复失败，当前仍保留该配置
精确确定性缺口，不宣称已修复或checkpoint序列化有误。

复现方法：环境/phase参数沿用上一节，依次将脚本换成
`outputs/probe_cp_ac_step_state_v1.py`（保留最初失败快照版本），或
`outputs/probe_cp_ac_step_state.py`（扩大快照版本）；目录分别使用
`outputs/tp-ep-cp-ac-step-probe-20261023`、
`outputs/tp-ep-cp-ac-update-probe-20261023`及
`outputs/tp-ep-cp-ac-update-probe2-20261023`。第二版在第二对运行前增加layout
元数据字段，未改变计算张量；这些diagnostic monkeypatch仅用于定位。
原脚本复跑目录为`outputs/tp-ep-cp-ac-cold-repeat2-20261023`。

离线比较：容器中在仓库根运行
`python outputs/compare_cp_ac_steps.py outputs/tp-ep-cp-ac-step-probe-20261023`；
将末尾目录改为上述两项update目录可比较扩大快照。
`python outputs/compare_tp_ep_cp_split.py`比较包括未插桩复跑的各step2保存点。
所有原失败日志和产物保留。下一步需要在不强制同步的情况下捕获失败运行的
step0权重、step3裁剪/更新及位置状态；目前没有足以支持生产调度或归约补丁的证据。

### 延迟落盘诊断及dense TP×KVCP冷恢复（2026-10-23）

沿用上一节环境与原始目标；本轮未修改生产实现或验收断言。此前逐步CPU保存可能
改变时序，本轮`outputs/probe_cp_ac_delayed.py`把输入、positions、logits、
裁剪前/后梯度及更新后参数的clone留在NPU，在Trainer.close后统一保存；另保存
构造完成时step0参数。clone、训练metrics和原脚本checkpoint仍可能同步，故不称
无观测开销，也不把该诊断当默认路径验收。

| 配置 | 基线/断言 | 实际结果 | 状态 | full/resume exit | 日志前缀（加_20261023.log） |
| --- | --- | --- | --- | --- | --- |
| NPU4 dense Qwen3，TP2/KVCP2，native FSDP、full AC、dynamic EMA/RNG/loader、FP32 seq256 | 原Trainer连续4步vs冷启动2+2，所有启用状态及loss精确 | 全rank加载前state failures=0，最终all checks passed，loss maxAE=0 | 通过 | 0/0 | tp_cp_dense_fullac_cold_full/resume |
| MoE TP2/EP2/KVCP2，full AC，延迟落盘诊断 | 同样严格恢复断言，额外离线逐步比较 | 全部通过，step3/4输入/位置/logits/原始及裁剪后梯度/更新参数/metrics精确 | 诊断，不计验收 | 0/0 | tp_ep_cp_ac_delayed_probe_full/resume |
| 同上第二次独立延迟诊断 | 同上 | 同上 | 诊断，不计验收 | 0/0 | tp_ep_cp_ac_delayed_probe2_full/resume |

两次延迟诊断构造后的step0参数本地张量在全部4个rank精确一致（nonexact=0、
maxAE=0），`tp_ep_cp_ac_initial_comparison_20261023.log` exit0。这仅证明这两次
通过的诊断初值一致，不能追溯证明历史失败/复跑的step0相同。逐步比较日志为
`tp_ep_cp_ac_delayed_comparison_20261023.log`及
`tp_ep_cp_ac_delayed_comparison2_20261023.log`，均exit0。

dense正式复现：沿用上一节原torchrun脚本与phase循环，改为设备4,5,6,7、
`LLMTUNER_TEST_MOE=0 LLMTUNER_TEST_EP_SIZE=1`，保留TP2/CP2/full AC及全部状态
开关，目录`outputs/tp-cp-dense-fullac-cold-20261023`。诊断复现使用设备0–3、
MoE=1/EP2，将入口换为`outputs/probe_cp_ac_delayed.py`，两对目录分别为
`outputs/tp-ep-cp-ac-delayed-probe-20261023`与
`outputs/tp-ep-cp-ac-delayed-probe2-20261023`。
离线命令`python outputs/compare_cp_ac_steps.py DIRECTORY`（指定上述各目录）及
`python outputs/compare_cp_ac_initial.py`均在容器仓库根运行。

本轮未再捕获失败，不能断言延迟落盘修复了问题或将EP特有问题定为根因；dense
通过只能说明该指定非MoE配置，不豁免TP×EP×CP的间歇性精确恢复失败。已取得
新的独立正式验收证据，整体目标仍未完成；后续不应无限重复通过的插桩配置。

### TP×EP×KVCP selective AC正式冷恢复及多模态范围复核（2026-10-23）

重新读取原始目标后，选择尚缺真实Trainer冷恢复证据的selective AC配置；
未继续重复full AC插桩诊断。仓库和基线commit未变，未修改生产代码或断言。
实际NPU4、Qwen3MoE两层/hidden64/vocab128、FP32 seq256、TP2/EP2/KVCP2、
native FSDP、dynamic EMA/RNG/loader、确定性模式、DCP，连续4步对比冷启动2+2。

| 测试项 | 基线/阈值 | 实际结果 | 状态 | full/resume exit | 日志 |
| --- | --- | --- | --- | --- | --- |
| TP×EP×KVCP＋selective AC真实Trainer冷恢复 | 原脚本启用状态及逐步loss精确一致 | 各rank pre-update state failures=0，最终所有检查通过；loss maxAE=0 | 通过 | 0/0 | tp_ep_cp_selective_cold_full_20261023.log、tp_ep_cp_selective_cold_resume_20261023.log |

复现（日志在既定`outputs/qwen3-8b-npu/logs/`，两个phase独立保存exit）：

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1
export LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_CP_SIZE=2
export LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native
export LLMTUNER_TEST_AC=selective LLMTUNER_TEST_EMA=dynamic
export LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
for phase in full resume; do
  torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" outputs/tp-ep-cp-selective-cold-20261023 || exit $?
done'
```

该指定selective运行通过不替代full AC历史间歇性失败，也不证明所有精度或模型。
已明确报告的optimizer for-loop选择及`_assert_async` CPU回退范围沿用前节；
无静默回退不等于纯AiCore。此轮只执行目标NPU用例及文档diff检查，没有代码变更，
不将此前CPU全套冒充本轮重跑。

多模态静态范围复核：`datasets/multimodal/recipes.py`目前仅注册obelics、cc12m、
cc12m-test；其处理器分别进入`process_obelics_sample`/`process_cc12m_sample`，
最终公共处理器接受文本及图像bytes，并调用process_image。`load_video`和
`process_video`目前只定义在video.py，没有生产调用方；video token存在不证明
已有视频配方或视频端到端路径。此前PNG证据仍只证明相应本地图像管线，不证明
网络streaming的所有source、OBELICS交错语义或视频训练。原始目标未明确要求
新增视频配方，故本轮不擅自扩展实现，也不以静态审查宣称那些路径已通过。
检索命令`rg -n 'load_video|process_video' llmtuner --glob '*.py'`及上述配方/处理器
源码是该范围判断的依据；整体P0–P4仍未完成。

### OBELICS双图交错的真实处理、packing及NPU搬运（2026-10-23）

本轮扩展既有`tests/integration_tests/npu_multimodal_pipeline.py`的`--obelics`
模式，不修改生产实现。测试源只负责从本地JSONL还原PNG bytes并组成交错结构
`text/image/text/image/text`，处理使用真实`process_obelics_sample`和
MultimodalProcessor、packing/collator及loader恢复。不是网络OBELICS下载验证，
也不涉及视觉模型训练或视频。原CC12M模式和断言保留。

新增非空洞断言：交错样本产出的grid至少两项且为偶数；image占位符数量等于
grid patch数除以spatial merge面积4；对应视觉labels全部IGNORE_INDEX；
pixel_values行数精确等于grid patch总数。仍对恢复后的完整batch精确比较，
并将所有tensor搬至NPU后逐项零容差搬回对比及浮点有限检查。

| 配置 | 基线/阈值 | 实际结果 | 状态 | exit | 日志 |
| --- | --- | --- | --- | --- | --- |
| 本地CC12M单图配对，128-token batch | loader完整batch及NPU往返精确；新增patch/label断言 | 全部通过 | 通过 | 0 | npu_multimodal_cc12m_final_20261023.log |
| 本地OBELICS双图交错，256-token batch | 同上，额外双图非空洞断言 | 全部通过 | 通过 | 0 | npu_multimodal_obelics_final_20261023.log |
| CPU多模态data/build定向 | 原断言 | 192 passed/14 warnings | 通过 | 0 | npu_interleaved_cpu_targeted_20261023.log |

目标运行命令：容器仓库根`PYTHONPATH=. timeout 120 python
tests/integration_tests/npu_multimodal_pipeline.py`，交错模式加`--obelics`。
两个最终运行均使用上述新增全部断言，没有monkeypatch生产处理器。
初次交错配置沿用128-token预算，默认每图最小65536 pixels，双图加文本超过预算
而被全部过滤；确认进程仍活跃后只终止该测试PID，exit143，保留
`npu_obelics_interleaved_pipeline_20261023.log`。这是新测试预算错误，不计生产
管线通过；提高该双图测试token预算到256后exit0，保留retry日志，并以上表最终
增加断言的日志为验收证据。没有放宽数值阈值或移除过滤保护。

全量Ruff/format及diff检查通过；定向pre-commit通过。全量pre-commit同步快照
仍因未修改hf_inspection.py的两处isinstance写法自动修复而exit1，其他hooks通过；
不回写无关修改，不记全量通过。日志`npu_interleaved_precommit_full_20261023.log`、
`npu_interleaved_precommit_targeted_20261023.log`，快照路径见
`npu_interleaved_gate_snapshot_path_20261023.txt`。本轮只扩展集成测试，未增加来源
移植或改动生产公式，因此无需新增来源分类。整体P4等缺口不因此关闭。
最终完整CPU套件4103 passed、13 skipped、273 warnings，144.08s，exit0；
`npu_interleaved_full_cpu_20261023.log`，命令为容器仓库根
`PYTHONPATH=. pytest -q --disable-warnings`；其后compileall通过。
本轮所有测试进程已终止，失败日志保留。

### P4 norm view边界：裁剪前梯度直接证据（2026-10-23）

本轮返回P4而非继续增加已通过子项；重读原始目标，生产实现、阈值及默认支持
策略均未修改。新诊断`outputs/p4_raw_gradient_capture.py`包装真实Trainer的
`_clip_and_check_finite`，只在step1/2、调用原裁剪前保存每个参数的实际本地梯度。
未修改原梯度、范数参数集合、公式或优化器；CPU快照引入同步，另行核验观测影响。

控制配置仍为既有非融合基线＋候选实际FlexAttention/NPU RoPE/基线参数范数顺序，
仅比较保留生产16个norm view与移除它们的诊断。两边都由回放入口严格加载并逐
参数零容差核对同一初始权重，消费同一100步真实基线batch。不是原fused默认P4
验收，也不能把移除view诊断通过当作生产修复。

| 配置 | 实际结果（原阈值） | 状态 | exit | 日志 |
| --- | --- | --- | --- | --- |
| 保留norm view，采集step1/2原始梯度，真实Trainer100步 | step61 grad_norm AE=0.6505155563354492、RE=0.05140064622777966，仍失败 | 失败 | 1 | p4_raw_grad_keep_20261023.log |
| 移除norm view，其余相同 | 100步logits/loss/grad_norm对既有非融合基线精确 | 诊断，不计默认验收 | 0 | p4_raw_grad_remove_20261023.log |

两种候选路径裁剪前梯度直接比较（不把它写成已采集baseline原始梯度）：

| rank | step | 非精确参数数/全部参数 | 最大本地梯度AE |
| --- | --- | --- | --- |
| 0 | 1 | 85/90 | 2.2351741790771484e-7 |
| 1 | 1 | 85/90 | 4.470348358154297e-7 |
| 0 | 2 | 90/90 | 1.6316771507263184e-6 |
| 1 | 2 | 90/90 | 5.387701094150543e-7 |

step1首次更新前已有差异，直接否定“这两个路径的分歧仅来自后续范数归约”解释；
支持进一步分析view对autograd分支相加顺序的影响。此处比较的是实际FSDP本地
shard，不借本地maxAE宣称全局梯度位级相等。完整逐参数误差/非等元素数见
`outputs/p4-raw-gradient-comparison-20261023.json`，比较脚本
`outputs/p4_compare_raw_gradients.py`，日志`p4_raw_gradient_comparison_20261023.log`
exit0（诊断完成，不是P4通过）。

观测重复性：新keep/remove运行分别与此前对应keep-views及精确no-view运行的
全部100步logits和完整step metrics逐项精确一致，两rank均通过；日志
`p4_raw_capture_repeatability_20261023.log`，exit0。故这两次新增CPU采集没有
改变所比较的既有训练轨迹；该证据不推广任意插桩都无影响。

复现（容器仓库根；每次输出目录必须新建，保留失败产物）：

```bash
export PYTHONPATH=.:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu
export ASCEND_RT_VISIBLE_DEVICES=0,1 LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1
export LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1 LLMTUNER_ALIGNMENT_BASELINE_NORM_ORDER=1
for mode in keep remove; do
  if [ "$mode" = remove ]; then export LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS=1; else unset LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS; fi
  export P4_RAW_GRADIENT_DIR=outputs/p4-raw-grad-${mode}-20261023
  torchrun --standalone --nproc_per_node=2 outputs/p4_raw_gradient_capture.py --capture outputs/baseline-trainer-fsdp2-nonfused-retry1-20261022/alignment-capture --output outputs/p4-raw-grad-${mode}-trainer-20261023
done
python outputs/p4_compare_raw_gradients.py
python outputs/p4_raw_capture_repeatability.py
```

日志明确出现基线Flex路径ArgSort的AiCPU执行警告；这是显式诊断后端，不隐去或
当作生产纯AiCore证明。本轮只改outputs诊断及状态文件，git diff --check通过，
未重跑CPU门禁（最近完整4103/13结果见前节）。所有本轮训练进程已终止。
原fused默认对齐仍失败；后续应针对真实分支累加契约设计能同时保留eager/AOT
等价与上游语义的修复，不能直接删除norm view保护来获得某诊断路径通过。

### P4 norm＋残差局部VJP：三分支相加顺序实证（2026-10-23）

读取原始目标后，继续上节裁剪前分歧定位，未修改生产代码或验收标准。
`outputs/p4_norm_residual_vjp.py`使用既有真实step1 trace中rank0/1、layer0/7
post_attention_layernorm的输入及norm输出上游梯度，并取该decoder输出梯度
作为残差分支上游梯度。每个case为真实NPU FP32局部norm＋残差VJP，不是整网
或P4替代验收。weight从对应基线initial-hf读取，并严格断言初始全1，未用任意
随机激活代替；epsilon为该Qwen3配置的1e-6。

将显式RMSNorm反向拆为D（numerator直接分支）、I（variance间接分支），R为
残差贡献；分支分别通过autograd求导，枚举全部六种三项加法顺序。对同一数据
构造有/无view的实际autograd VJP，结果4个case全部满足以下逐位精确断言：

- 有view：`(D + I) + R`精确匹配。
- 无view：`(D + R) + I`精确匹配。
- 两种VJP必须存在非精确元素，防止空泛地同时匹配。

| trace rank | layer | 有/无view输入梯度maxAE | 非精确元素数（共8192） |
| --- | --- | --- | --- |
| 0 | 0 | 9.313225746154785e-10 | 2667 |
| 0 | 7 | 1.1641532182693481e-10 | 2326 |
| 1 | 0 | 1.862645149230957e-9 | 2665 |
| 1 | 7 | 1.1641532182693481e-10 | 2286 |

正式加断言复验exit0，日志`p4_norm_residual_vjp_verified_20261023.log`；完整六种
匹配结果`outputs/p4-norm-residual-vjp-20261023.json`。最初仅观测版本日志
`p4_norm_residual_vjp_20261023.log`保留，最终结论以带断言及实际初始weight核对
的版本为准。复现：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. python outputs/p4_norm_residual_vjp.py'
```

使用的输入文件为`outputs/candidate-trainer-nonfused-flex-rope-20261023/rank*/module-trace-step-1.pt`
及`outputs/baseline-trainer-fsdp2-nonfused-retry1-20261022/alignment-capture/rank*/initial-hf.pt`。
局部图只保留norm与残差两个输出的VJP，不复刻完整attention/MLP/FSDP执行图；
因此这是相加顺序改变舍入的直接反例，不声称重现了整网85个参数的全部误差。
结合上节整网裁剪前证据，下一步修复必须显式考虑norm分支与残差的联合反向
边界，而非仅改范数参数顺序或更换RMSNorm公式。不能由本例推断删除view能
保持eager/AOT一致，也不能把局部微差当作自动豁免100步P4失败的理由。
本轮只有outputs诊断及本状态文档变化，git diff --check通过，无需冒充重跑
生产CPU套件；最近完整结果仍4103 passed/13 skipped。整体目标未完成。

### P4联合norm/残差原型：保留view的100步诊断精确（2026-10-23）

本轮沿上节三分支VJP证据实现隔离原型，未修改生产文件、配置默认值或验收阈值。
`outputs/p4_joint_norm_prototype.py`注册诊断命名空间p4probe的双输出custom op：
forward返回norm和独立残差clone；联合backward显式按`(D + R) + I`合并梯度。
反向也用独立custom op边界避免AOT重新融合该相加顺序，参数梯度保持原公式。
该原型仅覆盖FP32一阶梯度，不是任意dtype、高阶梯度或任意模型支持声明。

实际NPU局部验证使用前节4个真实trace case：native无view参考、原型eager、
`torch.compile(backend="aot_eager", fullgraph=True)`原型的forward、输入梯度及
weight梯度均逐位精确；断言实际unique_graphs>0。日志
`p4_joint_norm_prototype_20261023.log` exit0，结果
`outputs/p4-joint-norm-prototype-20261023.json`。它证明局部AOT，不代表整网
compile/Inductor通过。

随后`outputs/p4_joint_norm_training.py`仅在诊断进程给Qwen3 decoder的两次norm
调用接入联合输出；仍调用原norm模块以保留其hooks及weight身份，attention/MLP、
FSDP、optimizer及训练循环不替换。最终版本在Trainer构造完成时逐层断言全部
16个residual norm仍有生产`_isolate_norm_input` hook，且全部使用paired_norm。
这不是直接删除view，但联合输出确实改变decoder反向边界；不能把hook存在等同
证明原生产所有边界不变量均已满足。额外残差clone的内存/性能代价尚未验收。

| 配置 | 实际结果 | exit | 日志 |
| --- | --- | --- | --- |
| NPU2 FP32 debugmodel、非融合基线、Flex/NPU RoPE/基线norm参数顺序，联合norm原型 | 真实Trainer100步，原metrics全部通过，logits精确 | 0 | p4_joint_norm_training_20261023.log |
| 同上独立复跑，显式16个生产view hook非空洞断言 | 同上，两rank hook断言通过 | 0 | p4_joint_norm_training_verified_20261023.log |
| 独立产物审计 | 两次运行、两rank全部100步logits/完整metrics及全部91个state条目的实际本地shard精确等于既有精确no-view对照；placements/mesh也相同 | 0 | p4_joint_training_audit_final_20261023.log |

91个state条目包含embedding/head绑定的两个键，对应90个唯一参数。审计最初
错误要求state_dict也只有90键，exit1，保留`p4_joint_training_audit_20261023.log`；
核实实际键集合及绑定键后要求91键，并检查双方embedding/head本地值一致，仍
逐一对比所有键，不删除参数或放宽数值断言。未因审计脚本错误重跑训练。
独立审计的精确对照是已与非融合基线100步及最终权重核验过的
`outputs/candidate-trainer-nonfused-norm-order-20261023/rank*`。

复现：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. python outputs/p4_joint_norm_prototype.py'
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export PYTHONPATH=.:outputs:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu
export ASCEND_RT_VISIBLE_DEVICES=2,3 LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1
export LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1 LLMTUNER_ALIGNMENT_BASELINE_NORM_ORDER=1
torchrun --standalone --nproc_per_node=2 outputs/p4_joint_norm_training.py --capture outputs/baseline-trainer-fsdp2-nonfused-retry1-20261022/alignment-capture --output outputs/p4-joint-norm-training-verified-20261023'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/p4_joint_training_audit.py'
```

初次训练使用设备0,1及`outputs/p4-joint-norm-training-20261023`目录，未含随后
新增hook计数断言；最终复跑使用上面最终脚本。所有产物保留，进程已终止。
本轮取得可继续推进的联合边界方案证据，不将诊断monkeypatch或非融合基线
上推为默认P4通过。正式装配、完整eager/AOT、FSDP/TP/PP/AC生命周期、保存恢复、
BF16及原fused基线均未由本原型证明；尤其不能直接全局patch HF类作为生产实现。
本轮只修改outputs脚本及状态文档，diff检查通过；未声称重跑生产CPU门禁。
整体目标仍未完成。

### 联合norm原型回到原fused基线：仍未通过P4（2026-10-23）

为避免把上节非融合诊断成功当作默认修复，本轮重新读取原始目标，用同一联合
norm原型对原fused基线完整运行100步；不改生产实现或阈值。两个配置都保留
16个生产view hook并严格断言，仍由实际Trainer运行，初始权重/输入严格加载
原capture。原fused权重映射、HF拆分Q/K/V参数布局和候选默认范数计算未更改。

| 配置 | 实际结果 | 状态 | exit | 日志 |
| --- | --- | --- | --- | --- |
| 原fused capture＋联合norm，候选默认attention/RoPE/范数顺序 | step61 grad_norm AE=0.08670711517333984、RE=0.006328539451415675，超原AE .02及RE .005 | 失败 | 1 | p4_joint_norm_original_fused_20261023.log |
| 同上，只增加诊断FlexAttention及NPU RoPE频率 | step61 grad_norm AE=0.6396446228027344、RE=0.04668609054978295 | 失败 | 1 | p4_joint_norm_fused_flex_rope_20261023.log |

独立审计两个rank全部100步（不是只看第一个异常）：两配置logits全部满足原
meanAE/meanRE/cos标准，loss全部满足原AE/RE，LR及token计数精确；grad_norm
只有step61未满足原标准。原默认attention/RoPE＋联合norm的100步全部logits及
完整metrics，与历史原fused/no-view诊断逐位一致。说明联合边界在此配置也复刻
了既有no-view轨迹，但该轨迹本身未通过，不能把改善后误差当作通过。
审计`outputs/p4_joint_fused_audit.py`复用基线metrics.py，结果
`outputs/p4-joint-fused-audit-20261023.json`，完整日志
`p4_joint_fused_audit_final_20261023.log` exit0（审计完成，非训练通过）；较早仅
标量审计日志`p4_joint_fused_audit_20261023.log`保留。

复现（容器仓库根）：

```bash
export PYTHONPATH=.:outputs:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu
export ASCEND_RT_VISIBLE_DEVICES=0,1
unset LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ LLMTUNER_ALIGNMENT_FLEX_ATTENTION
unset LLMTUNER_ALIGNMENT_BASELINE_NORM_ORDER LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS
torchrun --standalone --nproc_per_node=2 outputs/p4_joint_norm_training.py --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture --output outputs/p4-joint-norm-original-fused-20261023
export ASCEND_RT_VISIBLE_DEVICES=2,3 LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1 LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1
torchrun --standalone --nproc_per_node=2 outputs/p4_joint_norm_training.py --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture --output outputs/p4-joint-norm-fused-flex-rope-20261023
PYTHONPATH=.:outputs/baseline-runtime python outputs/p4_joint_fused_audit.py
```

以上两个torchrun退出码独立保存，未以shell最后一项exit0掩盖训练失败。完整
100步产物、失败日志均保留，所有进程已终止；git diff --check通过。本轮只有
诊断及状态文档修改，不声称重跑生产CPU门禁。不能据此推广联合边界至默认实现。
下一步应隔离fused QKV计算与梯度范数分组语义；局部VJP相加顺序证据并未消除
这些差异，也不再重复“只换norm即可修复原P4”的假设。整体目标保持未完成。

### 原fused QKV局部反向与范数分组隔离（2026-10-23）

重新读取原始目标并核对当前工作区。工作仓库仍为main/c1367535，基线为
b358a806；已有修改保留。前一轮仅优化goal.md，未推进适配验收；本轮转向上节
明确的QKV差异，不重复norm-only的100步试跑。设备检查确认8卡空闲，无训练
进程；未停止其他任务。

新增诊断`outputs/p4_fused_qkv_isolation.py`，使用已有真实fused基线第一步
module trace及完整初始权重，覆盖两个rank全部8层，共16组。严格按基线
`FusedQKVLinear`实现，将Q/K/V以8个KV组、每组2个Q头＋K＋V、head_dim128
打包，权重打包/解包零容差核验；读取原allow_hf32配置，先设置确定性策略。
不改生产代码、权重身份、训练阈值或基线。

| 测试项 | 配置/基线 | 实际结果 | 状态 | exit | 日志 |
| --- | --- | --- | --- | --- | --- |
| fused局部复现 | 原trace输入/上游梯度/权重，16组 | 前向及输入梯度均与原capture逐位一致 | 通过（局部诊断） | 0 | p4_fused_qkv_isolation_verified_20261023.log |
| split与fused局部对照 | 同输入/上游梯度/权重，QKV布局严格互逆 | 前向及权重梯度全部逐位一致；16组输入梯度均不一致，最大AE=3.958120942115784e-9 | 通过（差异隔离，非P4验收） | 0 | 同上 |
| 输入梯度求和排列 | 三个独立projection VJP，穷举6种左结合排列 | 16组均有排列复现split；没有任何排列逐位复现fused输入梯度 | 通过（局部诊断） | 0 | 同上 |
| 相同梯度值的范数重分组 | packed梯度单一norm，对比分成Q/K/V后的norm-of-norms | 最大AE=2.384185791015625e-7；仅局部FP32舍入差异，不推断全局训练影响 | 通过（局部诊断） | 0 | 同上 |

这排除了“只重排三个独立projection输入VJP的求和，就能逐位复现原fused”这一
局部假设；证据指向单次大GEMM与三次小GEMM的归约边界差异。不能据此认定
第61步超标的唯一根因，也不能用局部误差小证明100步合格。split参数仍各自做
optimizer更新；全模型范数的参数分组、顺序及FSDP归约仍需独立核验。

复现（容器仓库根，使用空闲物理设备0）：

```bash
PYTHONPATH=. python outputs/p4_fused_qkv_isolation.py
```

首次隔离日志`p4_fused_qkv_isolation_20261023.log` exit0；随后增加全部排列及
前向/权重梯度严格断言，最终复跑日志见表，exit0。所有日志位于
`outputs/qwen3-8b-npu/logs/`，最终16组结果为
`outputs/p4-fused-qkv-isolation-20261023.json`。两个进程均已正常结束。
本轮只新增outputs诊断并更新状态，未重跑生产CPU门禁；整体目标未完成。

### 融合QKV＋联合norm：原fused基线100步诊断首次过阈值（2026-10-23）

沿上一节局部证据，新增`outputs/p4_fused_qkv_training.py`，在诊断进程中将
HF三个projection的权重按真实KV分组拼接，单次linear后拆回Q/K/V。保留三个
注册模块及原参数，不创建替代Parameter；每个模块仍被原HF attention调用。
临时cache仅在单次attention forward内交接K/V，finally清空，**不是可直接推广
的生产装配**。与已有联合norm原型组合，保留16个生产norm view hook，使用实际
Trainer、原fused capture及原阈值；没有改成nonfused基线或更改范数参数顺序。

| 配置 | 100步实际结果 | 状态 | exit | 日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| fused QKV＋联合norm＋诊断Flex/NPU RoPE，设备0,1 | 两rank全部100步logits/loss/grad_norm满足原阈值 | 通过（诊断，非默认P4） | 0 | p4_fused_qkv_joint_norm_flex_rope_20261023.log |
| 同配置独立复跑，设备2,3；新增装配后参数身份断言 | 同上；与初跑100步完整metrics、全部logits及91项最终state逐位一致 | 通过（诊断，非默认P4） | 0 | p4_fused_qkv_joint_norm_flex_rope_verified_20261023.log |
| fused QKV＋生产原生norm/view＋Flex/NPU RoPE，设备0,1 | step61 grad_norm AE=0.18609237670898438、RE=0.013582425678171157 | 失败 | 1 | p4_fused_qkv_native_norm_flex_rope_20261023.log |
| fused QKV＋联合norm＋默认attention/RoPE，设备2,3 | step61 grad_norm AE=0.21531105041503906、RE=0.015715024933689512 | 失败 | 1 | p4_fused_qkv_joint_norm_default_20261023.log |

两次成功配置均完整训练100步后验收；复跑核对90个注册参数、24个QKV参数的
装配后身份在训练结束时未变。各层Q/K/V模块调用均为100次，cache均为空。
记录中的构造期meta参数id与装配后DTensor id不同是正常materialization，
不将两者相等当作身份断言。所有配置产物独立保存，失败未覆盖。

独立审计`outputs/p4_fused_qkv_audit.py`重算两个rank所有100步，读取基线
metrics.py，不依赖训练脚本首次失败位置。成功配置最坏指标如下：

| 指标 | rank0 | rank1 | 原标准 |
| --- | --- | --- | --- |
| logits最大meanAE | 0.0017778124275977802 | 0.001320256447952417 | ≤0.02 |
| logits最大meanRE | 0.00039447808508248487 | 0.0003134742935927845 | ≤0.02 |
| logits最小cos | 0.9999997860725938 | 0.9999998924073037 | ≥0.9999 |
| loss最大AE / RE | 0.00019025802612304688 / 0.00005761981636463437 | 同左 | ≤0.02 / ≤0.005 |
| grad_norm最大AE / RE（step61） | 0.010026931762695312 / 0.0007318411310307462 | 同左 | ≤0.02 / ≤0.005 |

最终权重按真实mesh坐标及Shard placement从完整基线切出对应local shard，
所有91项（90个独立参数＋tied head重复键）均有限，shape/key一致、tied值一致；
**与基线没有逐位相同项**，最大AE为rank0=0.0007038903422653675、
rank1=0.000631401315331459。权重无预先确定容差，故只报告诊断，不按logits
阈值判权重通过。两次成功运行之间全部最终state逐位一致由
`outputs/p4_fused_qkv_repeatability.py`独立核验。

复现（容器仓库根）：

```bash
export PYTHONPATH=.:outputs:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu
export ASCEND_RT_VISIBLE_DEVICES=2,3 LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1 LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1
export P4_QKV_JOINT_NORM=1 P4_QKV_AUDIT_DIR=outputs/p4-fused-qkv-joint-norm-flex-rope-verified-20261023-audit
unset LLMTUNER_ALIGNMENT_BASELINE_NORM_ORDER LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS
torchrun --standalone --nproc_per_node=2 outputs/p4_fused_qkv_training.py --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture --output outputs/p4-fused-qkv-joint-norm-flex-rope-verified-20261023
PYTHONPATH=.:outputs/baseline-runtime python outputs/p4_fused_qkv_audit.py outputs/p4-fused-qkv-joint-norm-flex-rope-verified-20261023
PYTHONPATH=.:outputs/baseline-runtime python outputs/p4_fused_qkv_repeatability.py
```

复现须用新output目录，原目录保留。初跑将设备换0,1并去掉目录名中的verified。
原生norm消融设置`P4_QKV_JOINT_NORM=0`；默认attention/RoPE消融保留joint=1
并unset两个ALIGNMENT Flex/RoPE变量，分别使用表中同名output和-audit目录。

成功配置独立审计日志`p4_fused_qkv_joint_norm_audit_final_20261023.log`及
`p4_fused_qkv_joint_norm_verified_audit_20261023.log`均exit0；首次审计忘记
对candidate DTensor设置map_location=cpu，CPU/NPU比较失败，原日志
`p4_fused_qkv_joint_norm_audit_20261023.log`保留，修正审计后未重跑训练。
重复性日志`p4_fused_qkv_repeatability_20261023.log` exit0，两个rank三项exact
均true；审计JSON位于各run目录同级的`-audit-result.json`。

消融显示融合QKV不是独立充分修复；当前成功依赖联合norm及Flex/RoPE组合，尚未
区分Flex与RoPE各自作用。下一步须设计无全局monkeypatch、无跨projection缓存
的正式装配并验证eager/AOT、参数/optimizer生命周期、混合精度及保存恢复，再
在生产路径重跑原基线；不能把本轮诊断过阈值写成默认P4通过。整体目标未完成。

两个消融也完成全100步独立审计：所有logits/loss通过，两个rank均仅step61
grad_norm失败。日志`p4_fused_qkv_native_norm_audit_20261023.log`及
`p4_fused_qkv_joint_norm_default_audit_20261023.log`均exit0，表示审计完成而非
训练通过；训练退出码仍为1。全部训练与审计进程均已终止。本轮仅修改outputs
诊断和本状态文件，语法检查及git diff --check通过，未重跑生产CPU门禁。

### Flex/RoPE分离消融与无缓存实例装配（2026-10-23）

重读原始目标，核对main/c1367535及基线b358a806未变，保留已有改动；上一轮
获得原fused基线100步过阈值及复跑证据，属于进展，不把诊断成功当作生产完成。
开始前设备无训练进程。本轮先拆分Flex与RoPE变量，再验证消除跨projection
cache及全局HF类改写是否改变已成功的轨迹。

| 配置 | 实际结果 | 状态 | exit | 日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| fused QKV＋联合norm，仅NPU RoPE，设备0,1 | step61 grad_norm AE=0.32843494415283203、RE=0.023971660193508427；其余逐步指标通过 | 失败 | 1 | p4_fused_qkv_joint_norm_rope_only_20261023.log |
| fused QKV＋联合norm，仅Flex，设备2,3 | step61 grad_norm AE=2.4046897888183594、RE=0.17551240364219386；另有后续范数及logits失败，详见下文 | 失败 | 1 | p4_fused_qkv_joint_norm_flex_only_20261023.log |
| 无cache、实例级attention/decoder/norm适配＋Flex/NPU RoPE，设备0,1 | 两rank100步logits/loss/grad_norm均满足原阈值；与上一节成功方案全部metrics/logits及91项最终state逐位一致 | 通过（诊断，非生产P4） | 0 | p4_stateless_qkv_joint_norm_flex_rope_20261023.log |

两个单变量消融均由实际Trainer完成100步后比较，不因首个断言失败截断数据。
独立全100步审计：仅RoPE配置两rank均只step61范数失败、logits/loss全部通过；
仅Flex配置两rank范数失败步为61/75/76/77/78/85，logits失败步为71–78，loss
全部通过。日志`p4_fused_qkv_rope_only_audit_20261023.log`及
`p4_fused_qkv_flex_only_audit_20261023.log`均exit0（完成审计，训练仍失败）。
这只证明当前有限配置下，去掉Flex或NPU RoPE任一项都未通过；不推广为所有
模型或精度都必须使用这套组合，也不再重复单变量配置碰运气。

新增`outputs/p4_stateless_qkv_training.py`：在Trainer装配后仅绑定目标实例的
forward，HF Qwen3类本身不改。一次attention调用内完成权重拼接、fused linear、
Q/K/V拆分、原Q/K norm、RoPE、原注册attention interface及输出projection；
不在Q/K/V模块间传递临时cache。norm/decoder也只做实例绑定，保留16个生产
norm view hook、90个注册参数和24个QKV参数的装配后身份。每层attention调用
严格为100次。与旧方案不同，融合投影不再调用三个独立Linear.forward，因此
先显式检查其forward pre/post hook为空，避免本诊断静默跳过用户forward hook；
这仍不能证明TP/FSDP其他粒度hook或未知HF版本的兼容性。

独立审计`outputs/p4_fused_qkv_audit.py`支持分别核对旧方案的Q/K/V调用计数和
新方案的fused attention调用计数，没有把未发生的Linear调用记作100次。
`p4_stateless_qkv_audit_20261023.log` exit0：两rank全部100步满足原阈值；
权重仍与基线不逐位相同，只报告上节相同的诊断误差，不作未约定容差的通过判定。
`outputs/p4_fused_qkv_repeatability.py`增加左右run及输出路径参数，默认行为
不变。对照日志`p4_stateless_qkv_comparison_20261023.log` exit0，两个rank
complete_metrics_exact/all_100_logits_exact/all_91_state_entries_exact均true，
结果`outputs/p4-stateless-qkv-comparison-20261023.json`。

复现（容器仓库根，output须换新目录，原产物不覆盖）：

```bash
export PYTHONPATH=.:outputs:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu
export ASCEND_RT_VISIBLE_DEVICES=0,1 LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1 LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1
export P4_QKV_AUDIT_DIR=outputs/p4-stateless-qkv-joint-norm-flex-rope-20261023-audit
unset LLMTUNER_ALIGNMENT_BASELINE_NORM_ORDER LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS
torchrun --standalone --nproc_per_node=2 outputs/p4_stateless_qkv_training.py --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture --output outputs/p4-stateless-qkv-joint-norm-flex-rope-20261023
PYTHONPATH=.:outputs/baseline-runtime python outputs/p4_fused_qkv_audit.py outputs/p4-stateless-qkv-joint-norm-flex-rope-20261023
PYTHONPATH=.:outputs/baseline-runtime python outputs/p4_fused_qkv_repeatability.py --right p4-stateless-qkv-joint-norm-flex-rope-20261023 --output outputs/p4-stateless-qkv-comparison-20261023.json
```

两个消融复用上节`p4_fused_qkv_training.py`，joint=1；rope-only unset Flex，
flex-only unset RoPE，output及-audit目录按表中run名独立设置；Flex-only用设备2,3。
对应完整run名为`p4-fused-qkv-joint-norm-{rope-only,flex-only}-20261023`。

接口核查发现正式迁移不能简单复制成功脚本：当前生产HF模型的NPU默认为SDPA，
baseline Flex诊断还依赖导入torchtitan_npu的全局eager Flex补丁；生产RoPE需遵守
已有meta materialization、别名及动态RoPE的原始频率状态；TP适配还识别独立
q_proj/k_proj/v_proj模块。不能为过一个配置直接全局替换HF类、默认选Flex、
删除norm view或绕过TP模块生命周期。新实例方案仍在outputs，通过Trainer
构造器诊断wrapper安装，尚不是正式模型配置/装配，也未证明BF16、compile、
PP/TP/CP、checkpoint冷恢复。下一步按这些契约实现正式入口及回归，不宣称
本轮完成生产适配。全部进程已结束，语法及diff检查通过；未重跑生产CPU门禁。

### QKV正式配置入口、CPU契约及NPU冷恢复（2026-10-23）

本轮将QKV单项从outputs诊断迁入生产模型构造，不把联合norm、Flex或RoPE的
诊断依赖一并冒充已迁移。原目标和两个仓库commit已核对，保留既有工作区修改；
默认行为及原阈值不变，无提交或分支切换。

生产变更：

- `ModelConfig.qkv_projection`显式接受separate/fused，默认separate，经根配置
  平坦视图、HF工厂及复合text config传递。
- `models/hf/qkv_projection.py`在构造时仅绑定dense Qwen3实例，按真实KV-group
  拼接已有权重/一致bias，一次linear后拆分；无新Parameter、键名变化或跨投影
  cache，无全局HF类改写，生产模块不导入torchtitan-npu。
- 保留Q/K norm、RoPE、attention interface、cache更新、输出projection及外围
  hook。融合不调用原Q/K/V Linear.forward，因此拒绝非plain Linear、模块前后向
  hook、实例forward覆写和不一致bias，不静默绕过。构造期先全量预检再绑定。
- 配置及TP装配均拒绝fused×TP，原因是独立projection通信边界尚未适配；默认
  separate的既有支持不变。不是删除原TP声明以使矩阵通过。
- 设计及来源分别更新原有design/upstream_map文档；无新增来源代码注释。

新增`tests/unit_tests/cpu/models/test_hf_qkv_projection.py`共34例，覆盖GQA/
MHA、bias、非连续输入、FP32/BF16/FP16、冻结权重、参数身份、optimizer状态、
幂等和原子预检、HF原生配置保存/加载、meta materialization、局部fullgraph
aot_eager前反向，以及hook/TP/其他模型拒绝。注意低精度单次GEMM的输入梯度
不同于三个已舍入梯度相加；使用独立FP64数学参考，未放宽任何P4阈值。

| 验证配置 | 实际结果 | 状态 | exit | 日志（outputs/qwen3-8b-npu/logs/） |
| --- | --- | --- | --- | --- |
| 定向CPU模型/config/compatibility，新增首批测试 | 425 passed | 通过 | 0 | fused_qkv_targeted_cpu_verified_20261023.log |
| 最终QKV契约测试 | 34 passed | 通过 | 0 | fused_qkv_contract_cpu_final_20261023.log |
| 最终全量CPU | 4137 passed / 13 skipped / 273 warnings，149.81s | 通过 | 0 | fused_qkv_full_cpu_final_20261023.log |
| 原fused基线100步，配置选择正式QKV，联合norm/Flex/RoPE仍为诊断 | 两rank100步logits/loss/grad_norm满足原阈值；90参数身份保持、各层100次；与先前成功方案完整metrics/logits及91项state逐位一致 | 通过（混合诊断，非默认P4） | 0 | p4_configured_qkv_joint_norm_flex_rope_verified_20261023.log |
| NPU2 nativeFSDP2＋正式QKV＋原生norm/SDPA＋full AC＋dynamic EMA/RNG/loader；FP32，4步连续 vs 独立进程2＋2步 | 加载前状态及最终模型/optimizer/scheduler/EMA/RNG/游标严格比较通过，续训loss最大AE=0 | 通过（有限配置） | 0/0 | fused_qkv_fullac_cold_full_20261023.log / fused_qkv_fullac_cold_resume_20261023.log |
| 同上，BF16构造参数、FP32 FSDP归约，独立进程full/resume | 同样严格恢复通过；独立检查两个rank的split/final各25项model state均为BF16 | 通过（有限配置） | 0/0 | fused_qkv_bf16_fullac_cold_full_20261023.log / fused_qkv_bf16_fullac_cold_resume_20261023.log |

实际冷恢复扩展已有`pp_checkpoint_equivalence.py`，通过
`LLMTUNER_TEST_QKV_PROJECTION=fused`选择正式配置，并在训练/恢复时核实实际
forward绑定。没有给QKV或checkpoint安装临时生产猴补丁；FP32直接运行脚本，
BF16用`outputs/fused_qkv_bf16_checkpoint.py`只设置PyTorch默认构造dtype后运行
同一脚本。BF16证据是BF16参数存储训练，不扩张为所有AMP/master-weight方案。
`fused_qkv_checkpoint_dtype_audit_20261023.log` exit0，实际核对split/final状态。
本配置仍出现已报告的BlockMask int32 AiCPU argsort，未静默隐藏或称纯AiCore。

100步审计日志`p4_configured_qkv_audit_20261023.log` exit0；与此前成功方案
的对照`p4_configured_qkv_comparison_20261023.log` exit0，两个rank三项exact
均true。最终权重与基线仍非逐位相同、容差未约定，只报告此前相同的诊断误差。
QKV正式化并未解决联合norm/Flex/RoPE生产迁移、TP融合通信或完整P2/P3覆盖。

关键复现（容器仓库根；已有产物保留，重跑须换output）：

```bash
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=2,3
export LLMTUNER_TEST_WORLD_SIZE=2 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=1
export LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_MOE=0
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_QKV_PROJECTION=fused
export LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py full outputs/fused-qkv-fullac-cold-20261023
torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py resume outputs/fused-qkv-fullac-cold-20261023
```

BF16用设备4,5、入口`outputs/fused_qkv_bf16_checkpoint.py`及目录
`outputs/fused-qkv-bf16-fullac-cold-20261023`，其他环境相同。
100步使用上节同一baseline capture、Flex/RoPE环境及设备2,3，额外设置
`LLMTUNER_ALIGNMENT_QKV_PROJECTION=fused`，入口
`outputs/p4_configured_qkv_training.py`，output为
`outputs/p4-configured-qkv-joint-norm-flex-rope-verified-20261023`、
`P4_QKV_AUDIT_DIR`为该目录加`-audit`。包装器仅核对正式QKV绑定、计数及参数
身份，联合norm仍由已有诊断脚本安装；不能把整个运行称无诊断干预。

质量门禁：全量Ruff、format（441 files）、compileall及diff检查均通过。
在同步快照`/tmp/llmtuner-fused-qkv-gates-c9H23q`执行pre-commit --all-files，
仍仅因未改动的`parallel/expert_parallel/hf_inspection.py`两处tuple-isinstance
被Ruff0.9自动改成union而exit1；逐文件cmp确认快照仅此文件与工作区不同，未把
修改带回实际工作区。日志`fused_qkv_precommit_20261023.log`。两个新增文件在
同一快照单独执行全部hook，exit0，日志`fused_qkv_precommit_new_files_20261023.log`。
不能把快照修正后的状态计为实际全门禁通过，既有无关失败仍需用户确认豁免。

失败历史保留：最早定向测试12个低精度输入梯度把split舍入轨迹误作数学真值，
另漏加matrix案例，合计13失败；修正参考对象及补齐案例后通过。首轮全CPU
1失败/4136通过：测试把仓库运行配置中的experts_implementation='native'哨兵
原样序列化交给HF，被HF误作具体kernel名称；改用原生HF配置单独检验新增QKV
字段、meta及状态恢复，未修复无关experts配置问题，也不宣称该运行配置直接
HF序列化重载已支持。最早100步wrapper误读Trainer.config（实际为cfg），训练
前exit1；原日志/目录保留，修正wrapper后换新目录复跑，无生产修复或阈值变更。
所有本轮进程已终止。整体目标仍未完成，下一步继续联合norm/Flex/RoPE正式化。

### 2026-10-23：联合残差归一化正式配置与验证闭环

本轮恢复原适配目标，先检查已有任务：容器内相关 torchrun/pytest 已无存活
进程，日志显示 full 阶段及100步训练完成，而非把观察超时当作失败重启。
仓库仍为 main/c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19，基线仍为
b358a8063c34aa7494397efa3a8f5ad24f1caa1f；保留已有工作区改动，不提交。

正式入口 `model.norm_backend=joint` 由配置/工厂进入
`models/hf/residual_normalization.py`，适配 dense Qwen3 的两处 residual norm
及 decoder 实例 forward。参数身份和键名不变；保留拥有的输入 view hook，
不修改全局 HF 类，不依赖诊断联合norm脚本。FP32 显式固定
`(direct + residual) + indirect`；低精度维持 norm cast 后加残差。
支持一阶梯度，残差 clone 有额外内存成本，不承诺性能或所有并行组合。
用户 norm hook、自定义 norm/block forward 及非目标家族明确拒绝。
设计说明与来源分类分别更新于设计文档及 upstream map。

日志均位于 `outputs/qwen3-8b-npu/logs/`：

| 验收配置 | 实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- |
| joint/QKV/config 定向CPU，含混合dtype、非连续/空输入、hook拒绝、完整decoder AOT | 211 passed，16 warnings，20.60s | 通过 | 0 | joint_norm_targeted_cpu_verified_20261023.log |
| 全量CPU | 4192 passed，13 skipped，273 warnings，153.69s | 通过 | 日志终结通过，原句柄未保留退出码 | joint_norm_full_cpu_20261023.log |
| 正式joint＋fused QKV，Flex/RoPE仍为诊断，原fused基线100步 | 两rank每步标量及logits全部满足原阈值 | 通过（混合诊断，非完整生产P4） | 独立审计0 | p4_configured_joint_flex_rope_20261023.log / p4_configured_joint_audit_20261023.log |
| 与此前成功诊断配方逐项对照 | 两rank全部metrics、100 logits、91项最终state逐位相同 | 通过（重复性） | 0 | p4_configured_joint_comparison_20261023.log |
| NPU2 nativeFSDP2＋joint/fusedQKV＋nativeSDPA/RoPE＋fullAC＋dynamicEMA/RNG/loader，FP32 | full4步 vs 独立进程2＋2步，加载及最终状态全部断言通过，续训loss AE=0 | 通过（有限配置） | full日志完成；resume 0 | joint_norm_qkv_fullac_cold_full_20261023.log / joint_norm_qkv_fullac_cold_resume_20261023.log |
| 同上，BF16参数、FP32 FSDP归约 | 严格恢复通过，续训loss AE=0 | 通过（有限配置） | full日志完成；resume 0 | joint_norm_qkv_bf16_fullac_cold_full_20261023.log / joint_norm_qkv_bf16_fullac_cold_resume_20261023.log |
| 两种精度split/final、两个rank的实际checkpoint dtype | 每份25项model state均为预期dtype | 通过 | 0 | joint_norm_checkpoint_dtype_audit_20261023.log |

100步审计最终权重与原基线仍为0/91项逐位相同，rank0/1最大AE分别为
0.0007038903422653675 / 0.000631401315331459。权重容差尚未确定，不能套用
logits阈值，更不能据此判定完整P4通过。观察包装器只检查正式绑定、90参数身份、
8层各100次调用及全局HF类不变；Flex注册与NPU RoPE重建仍是诊断依赖。

关键复现（容器仓库根，输出已有，重跑须换目录）：

```bash
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=2,3
export LLMTUNER_TEST_WORLD_SIZE=2 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=1
export LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_MOE=0
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_QKV_PROJECTION=fused
export LLMTUNER_TEST_NORM_BACKEND=joint LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic
export LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py full outputs/joint-norm-qkv-fullac-cold-20261023
torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py resume outputs/joint-norm-qkv-fullac-cold-20261023
```

BF16改设备4,5、入口 `outputs/fused_qkv_bf16_checkpoint.py` 及目录
`outputs/joint-norm-qkv-bf16-fullac-cold-20261023`，其余相同；包装器仅设置
默认构造dtype后runpy，不安装生产猴补丁。100步使用设备0,1及以下命令：

```bash
PYTHONPATH=.:outputs:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu ASCEND_RT_VISIBLE_DEVICES=0,1 LLMTUNER_ALIGNMENT_QKV_PROJECTION=fused LLMTUNER_ALIGNMENT_NORM_BACKEND=joint LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ=1 LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1 P4_QKV_AUDIT_DIR=outputs/p4-configured-joint-flex-rope-20261023-audit torchrun --standalone --nproc_per_node=2 outputs/p4_configured_joint_training.py --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture --output outputs/p4-configured-joint-flex-rope-20261023
PYTHONPATH=.:outputs/baseline-runtime python outputs/p4_fused_qkv_audit.py outputs/p4-configured-joint-flex-rope-20261023
PYTHONPATH=.:outputs/baseline-runtime python outputs/p4_fused_qkv_repeatability.py --right p4-configured-joint-flex-rope-20261023 --output outputs/p4-configured-joint-comparison-20261023.json
```

质量门禁：Ruff、format（443 files）、容器compileall、diff检查均exit0。
同步快照 `/tmp/llmtuner-joint-norm-gates-IUCEjx` 的pre-commit --all-files
仍exit1，仅既有未改动 `parallel/expert_parallel/hf_inspection.py` 两处tuple
isinstance被Ruff自动修改；cmp核实快照仅该文件不同，未回写工作区。
日志 `joint_norm_precommit_20261023.log`；两个新增joint文件独立hooks全部
通过，exit0，日志 `joint_norm_precommit_new_files_20261023.log`。
不能把修正快照计为真实门禁通过，仍需披露此无关失败。

初轮定向CPU出现1失败（207通过）：独立参考式误写为 `w*x*scale` 而非
`w*(x*scale)`，修正参考的运算关联后通过，未放宽容差或更改生产算子。
失败日志 `joint_norm_targeted_cpu_20261023.log` 保留。
整体仍未完成：Flex/RoPE正式化、权重验收依据、完整P0–P3矩阵及既有
fullAC TP/EP/CP间歇冷恢复、模型compile/硬件限制等缺口继续保留。

### 2026-10-23：目标设备 RoPE 正式配置，移除对齐中的频率猴补丁依赖

上一轮属实质进展（joint正式配置、CPU/NPU恢复及百步审计）。本轮按原目标
继续RoPE正式化，未缩小P0–P4范围。新增
`model.rotary_init_device=constructor|target`，默认constructor不变。
配置经root属性、HF工厂及text_config传递；目标策略仅接受dense Qwen3默认
RoPE。Trainer普通构造在搬运后、并行变换前执行；HF初始加载路径则在meta
materialization后执行。通过安装版HF原生构造器在实际device重建两个频率buffer、
attention scaling及缓存，不硬编码theta/head_dim，不依赖torchtitan-npu。
参数和state-dict键不变，保持重建入口已有的buffer alias；自定义rotary子类、
其他家族和scaled RoPE明确拒绝。直接HF wrapper用户须显式调用leaf helper，
单独`.to()`不执行该Trainer策略。设计与B类来源映射已更新。

日志统一为 `outputs/qwen3-8b-npu/logs/`：

| 验收配置 | 实际结果/基线 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- |
| 目标RoPE及既有HF wrapper/factory CPU定向 | 101 passed，16 warnings，18.63s；含默认不改、配置传播、meta、参数身份/键名、alias、拒绝路径、FP32频率及缓存 | 通过 | 0 | rotary_init_targeted_cpu_20261023.log |
| 全量CPU | 4203 passed，13 skipped，273 warnings，148.58s | 通过 | 0 | rotary_init_full_cpu_20261023.log |
| NPU普通构造和meta materialize的两个频率及共享引用 | 原生重建结果与此前NPU算式逐位一致，两个频率均FP32/NPU，入口共享引用保持 | 通过（局部） | 0 | rotary_init_npu_materialization_verified_20261023.log |
| 正式joint/fusedQKV/targetRoPE＋诊断Flex，原fused基线100步 | 两rank每步loss/grad_norm及logits均满足原阈值，scalar/logits failures均空 | 通过（仍非完整生产P4） | 训练0/审计0 | p4_configured_joint_target_rope_flex_20261023.log / p4_configured_joint_target_rope_audit_20261023.log |
| 与此前成功混合诊断配方对照 | 两rank全部metrics、100 logits、91项最终state逐位相同 | 通过（重复性） | 0 | p4_configured_joint_target_rope_comparison_20261023.log |
| NPU2 nativeFSDP2、joint/fusedQKV/targetRoPE、nativeSDPA、fullAC、dynamicEMA/RNG/loader，FP32 | 连续4步vs独立进程2＋2步；加载及最终状态全部断言通过，续训loss AE=0；实际两个频率逐位等于设备重建值 | 通过（有限配置） | full0/resume0 | target_rope_joint_qkv_fullac_cold_full_20261023.log / target_rope_joint_qkv_fullac_cold_resume_20261023.log |
| 同上BF16参数/FP32 FSDP归约 | 同样严格恢复通过，续训loss AE=0；两频率仍FP32而非BF16 | 通过（有限配置） | full0/resume0 | target_rope_joint_qkv_bf16_fullac_cold_full_20261023.log / target_rope_joint_qkv_bf16_fullac_cold_resume_20261023.log |

100步运行明确unset `LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ`，仅正式配置选择目标
初始化。训练使用原Trainer，observer只核对joint/QKV绑定与参数身份；Flex仍
由诊断路径注册，尚不能将整个配方称为无诊断干预。最终权重与原基线仍无
逐位相同项，rank0/1最大AE为0.0007038903422653675 / 0.000631401315331459，
容差仍未约定，不宣称权重验收通过。

复现（容器仓库根，已有输出不得覆盖；BF16替换项见命令后）：

```bash
unset LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ
export PYTHONPATH=.:outputs:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu ASCEND_RT_VISIBLE_DEVICES=0,1
export LLMTUNER_ALIGNMENT_QKV_PROJECTION=fused LLMTUNER_ALIGNMENT_NORM_BACKEND=joint LLMTUNER_ALIGNMENT_FLEX_ATTENTION=1
export LLMTUNER_ALIGNMENT_ROTARY_INIT_DEVICE=target
export P4_QKV_AUDIT_DIR=outputs/p4-configured-joint-target-rope-flex-20261023-audit
torchrun --standalone --nproc_per_node=2 outputs/p4_configured_joint_training.py --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture --output outputs/p4-configured-joint-target-rope-flex-20261023
PYTHONPATH=.:outputs/baseline-runtime python outputs/p4_fused_qkv_audit.py outputs/p4-configured-joint-target-rope-flex-20261023
PYTHONPATH=.:outputs/baseline-runtime python outputs/p4_fused_qkv_repeatability.py --right p4-configured-joint-target-rope-flex-20261023 --output outputs/p4-configured-joint-target-rope-comparison-20261023.json

export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=2,3
export LLMTUNER_TEST_WORLD_SIZE=2 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=1
export LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_MOE=0
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_QKV_PROJECTION=fused
export LLMTUNER_TEST_NORM_BACKEND=joint LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic
export LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
export LLMTUNER_TEST_ROTARY_INIT_DEVICE=target
torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py full outputs/target-rope-joint-qkv-fullac-cold-20261023
torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py resume outputs/target-rope-joint-qkv-fullac-cold-20261023
```

BF16改为设备4,5、入口 `outputs/fused_qkv_bf16_checkpoint.py`，目录
`outputs/target-rope-joint-qkv-bf16-fullac-cold-20261023`；其余环境相同。

质量门禁：Ruff、format（444 files）、容器compileall及diff检查exit0。
全量pre-commit在同步快照 `/tmp/llmtuner-rotary-init-gates-DNWYd4` exit1，
仍仅未改动 `parallel/expert_parallel/hf_inspection.py` 的两处tuple-isinstance
自动修改，cmp确认未涉及其他tracked文件，未回写实际工作区。日志
`rotary_init_precommit_20261023.log`；新增CPU测试单独hooks exit0，日志
`rotary_init_precommit_new_files_20261023.log`。既有无关失败未获豁免。

失败/范围说明：首次局部NPU探针把alias注册在普通`.to()`之前，搬运本身已
将两条buffer路径分别分配，helper入口不存在共享对象，alias断言失败exit1，
日志 `rotary_init_npu_materialization_20261023.log` 保留。修正探针以验证helper
入口已有的alias，另独立验证meta路径搬运前alias恢复，复跑exit0；未改生产
代码、阈值或将普通`.to()`对任意alias的行为包装成已修复。CPU配置传播及
meta测试也通过。首次查看HF源码时误查询ROPE_INIT_FUNCTIONS['default']产生
KeyError，确认default属于Qwen3类静态方法后使用原生构造器，无生产异常绕过。

本轮验证进程均已结束。下一步为Flex正式接入及剩余矩阵；KCP硬件、完整模型
compile、自定义FSDP版本、fullAC混合并行间歇恢复、权重验收依据等既有缺口
未被本轮关闭，整体仍未完成。

### 2026-10-23：独立 eager Flex leaf 的真实 NPU 等价证据（尚未接入 Trainer）

上一轮为实质进展（正式RoPE及全量CPU/真实恢复）；本轮重读原目标，继续
Flex正式化。核对基线实际实现为eager Flex数学路径，而非把它误称NPU融合核。
基线导入通过全局device检查覆写与debug开关允许调用，不能直接照搬成生产
依赖。验证直接调用同一PyTorch Flex HOP可保持前后向且无需导入基线包。

新增 `models/common/attention/eager_flex.py`，复用PyTorch的HOP/autograd和
kernel options，不重写数学梯度，不修改PyTorch/HF全局类或函数。显式校验
CPU/NPU、FP32/BF16/FP16、四维QKV、同设备/精度/batch及GQA整除、BlockMask
长度/设备及反向索引；compile明确拒绝，private API缺失给出环境解锁提示。
当前尚无HF配置、mask接口和Trainer生命周期接入，不将此leaf计为完整生产
Flex支持，也不宣称已经移除P4的诊断注册。

证据（日志位于 `outputs/qwen3-8b-npu/logs/`）：

| 项目/配置 | 基线/断言及实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- |
| NPU直接HOP，FP32小型GQA | forward/dQ/dK/dV有限，基线包不在sys.modules | 通过（诊断） | 0 | flex_direct_hop_probe_20261023.log |
| NPU12配置：3精度×length31/128×causal/packed，batch2、Qheads4/KVheads2 | 导入基线前运行全部候选；之后与基线eager Flex逐项比较，全部forward/dQ/dK/dV逐位相同 | 通过（诊断） | 0 | flex_direct_hop_equivalence_20261023.log |
| 同上12配置，候选改为正式leaf | 同样全部逐位相同；候选阶段未导入torchtitan_npu，数值有限 | 通过（leaf有限范围） | 0 | eager_flex_production_leaf_equivalence_20261023.log |
| CPU独立dense数学参考及输入拒绝 | 7 passed，14 warnings，4.40s；forward逐位相同，梯度预设容差FP32 2e-6/低精度.02 | 通过（CPU局部） | 0 | eager_flex_targeted_cpu_20261023.log |

复现（容器仓库根）：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/cpu/models/test_eager_flex.py
ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=.:outputs/baseline-runtime:/home/jianzhnie/llmtuner/llm/torchtitan-npu python outputs/flex_direct_hop_equivalence.py
```

脚本先执行所有正式leaf候选、断言基线未导入，再导入基线对照。原诊断结果
保留单独日志；当前脚本使用正式leaf。实际BlockMask仍出现int32/int64 argsort
AiCPU警告，明确保留，不宣称纯AiCore。来源适配记录在upstream map。
Ruff、format（446files）、diff检查exit0；仅本轮局部测试，尚未执行新增leaf
后的全量CPU/pre-commit，上一轮全门禁证据不能当作新文件的通过证据。

探索中两个源码查询命令分别因Python导入解析到HOP对象（无__file__）及访问
不存在的flex_attention_dense符号而exit1；改用importlib与实际函数名查看源码，
非生产训练失败，不绕过运行时guard。三个真实验证进程均终止且exit0。
下一步须接入显式HF后端、保留causal/packing掩码语义、明确并行/compile限制，
再以无基线导入的真实Trainer百步及恢复验证。原P0–P4目标仍未完成。

### 2026-10-23：显式 HF eager Flex 接入与 AC HOP 重放修复

上一轮已产生独立leaf的真实等价证据；本轮继续正式接入，未重新定义P0–P4。
新增 `model.attention_backend=auto|eager_flex`，默认auto不变，工厂/text_config
传播后选择独立HF接口 `llmtuner_eager_flex`，不覆盖sdpa注册。HF mask接口
复用SDPA的布尔规则，adapter将因果、packed文档及屏蔽mask转为BlockMask后
调用既有正式leaf。不导入torchtitan-npu，不覆写PyTorch设备guard/debug开关。
eager数学路径仍物化scores/mask、存在AiCPU argsort，不宣称融合核或纯AiCore。
dropout、attention bias和不等长缓存明确拒绝。TP/CP及compile未适配时在配置
或装配期拒绝，原auto路径支持不变；拒绝不计为原并行功能验收通过。

首次正式fullAC冷恢复在两种精度均exit1：本仓库RecomputeContext子类没有
继承到PyTorch为原始dispatch mode类型注册的HOP规则，Flex在replay报
NotImplementedError。修复该子类的 `supports_higher_order_operators=True`，
仍调用基类policy/storage逻辑，不绕过保存/重算策略。新增CPU完整Flex HOP
重放梯度逐位一致回归；不是把标志当作所有HOP均已验证。设计与B类来源更新。

日志目录统一 `outputs/qwen3-8b-npu/logs/`：

| 配置 | 基线/断言与结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- |
| eager Flex HF因果/packed前后向，与原sdpa模型比较；配置/adapter拒绝及兼容表 | 25 passed，15 warnings，9.51s；输出及参数梯度容差2e-6 | 通过（CPU有限范围） | 0 | eager_flex_backend_guards_verified_20261023.log |
| 上述HF测试＋完整AC CPU回归 | 98 passed，15 warnings，14.65s；Flex fullAC输出及dQ/dK/dV逐位一致 | 通过 | 0 | eager_flex_ac_regression_20261023.log |
| 正式QKV/joint/targetRoPE/eagerFlex，原fused基线100步，两个诊断开关均unset | 两rank逐步loss/grad_norm及logits满足原阈值；与此前配方完整metrics/100logits/91state逐位相同 | 通过（有限配置指标） | 训练0/审计0/对照0 | p4_production_flex_joint_rope_20261023.log / p4_production_flex_audit_20261023.log / p4_production_flex_comparison_20261023.log |
| NPU2 nativeFSDP2＋四项正式配置＋fullAC/dynamicEMA/RNG/loader，FP32 | full4步vs独立进程2＋2步，加载及最终状态全部断言通过，续训loss AE=0 | 通过（有限配置） | full0/resume0 | production_flex_verified_fullac_cold_full_20261023.log / production_flex_verified_fullac_cold_resume_20261023.log |
| 同上BF16参数、FP32归约 | 同样严格恢复通过，续训loss AE=0 | 通过（有限配置） | full0/resume0 | production_flex_verified_bf16_fullac_cold_full_20261023.log / production_flex_verified_bf16_fullac_cold_resume_20261023.log |

100步最终权重对原基线仍为0/91逐位一致，最大AE rank0/1为
0.0007038903422653675 / 0.000631401315331459，权重标准仍未确定，不宣称
完整P4通过。首次100步未使用追加的sys.modules断言；虽未给候选PYTHONPATH
加入基线包且诊断开关关闭，仍追加独立100步verified运行来直接证明前后均无
torchtitan_npu导入，并断言实际后端绑定，不能用包装器新代码追认旧证据。

失败历史：首批HF CPU fixture误传二维tokens给要求一维的wrapper（2失败/7
通过）；修正fixture形状后9通过。扩展guard测试误用只读属性tp_size/cp_size
作构造参数（3失败/22通过），改用真实字段后25通过。首轮全CPU收集了旧
fixture，且CP装配guard新增modules()遍历破坏既有SimpleNamespace测试契约，
合计7失败/4209通过/13跳过，日志eager_flex_backend_full_cpu_20261023.log；
CP改为读取原有根HF config后启动独立全量复验，不覆盖失败日志。
首次NPU fullAC失败日志production_flex[_bf16]_fullac_cold_full_20261023.log
及目录保留；修复后换verified目录full/resume，未在原目录伪造成功。

复现（容器仓库根，已有output须换新目录）：

```bash
export PYTHONPATH=.:outputs:outputs/baseline-runtime ASCEND_RT_VISIBLE_DEVICES=0,1
export LLMTUNER_ALIGNMENT_QKV_PROJECTION=fused LLMTUNER_ALIGNMENT_NORM_BACKEND=joint
export LLMTUNER_ALIGNMENT_ROTARY_INIT_DEVICE=target LLMTUNER_ALIGNMENT_ATTENTION_BACKEND=eager_flex
unset LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ LLMTUNER_ALIGNMENT_FLEX_ATTENTION
export P4_QKV_AUDIT_DIR=outputs/p4-production-flex-joint-rope-verified-20261023-audit
torchrun --standalone --nproc_per_node=2 outputs/p4_configured_joint_training.py --capture outputs/baseline-trainer-fsdp2-partial-100-20261022/alignment-capture --output outputs/p4-production-flex-joint-rope-verified-20261023

export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=2,3
export LLMTUNER_TEST_WORLD_SIZE=2 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=1
export LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_MOE=0
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_QKV_PROJECTION=fused
export LLMTUNER_TEST_NORM_BACKEND=joint LLMTUNER_TEST_ROTARY_INIT_DEVICE=target
export LLMTUNER_TEST_ATTENTION_BACKEND=eager_flex LLMTUNER_TEST_AC=full
export LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py full outputs/production-flex-fullac-cold-verified-20261023
torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py resume outputs/production-flex-fullac-cold-verified-20261023
```

BF16改设备4,5、入口outputs/fused_qkv_bf16_checkpoint.py及目录
outputs/production-flex-bf16-fullac-cold-verified-20261023，其他环境一致。
最终复验补记：全量CPU为4217 passed / 13 skipped / 273 warnings，150.37s，
exit0，日志 `eager_flex_backend_full_cpu_verified_20261023.log`。
带实际后端及训练前后无torchtitan_npu导入断言的100步verified运行exit0，日志
`p4_production_flex_joint_rope_verified_20261023.log`；独立审计与重复性对照均
exit0，日志 `p4_production_flex_verified_audit_20261023.log`、
`p4_production_flex_verified_comparison_20261023.log`。两rank原阈值全部通过、
与此前成功配方三项exact均true，最终权重对原基线的未解决误差保持相同。
该结果证明有限配方的Flex/RoPE/joint/QKV计算已由正式配置选择，无诊断计算
补丁；观察包装器仍存在，不能据此扩张至其他配置或全部P4验收。

两个rank的split/final共8份checkpoint独立dtype审计exit0，每份25项model
state均为预期FP32/BF16，日志 `production_flex_checkpoint_dtype_audit_20261023.log`。
真实NPU日志还报告 `aten::_assert_async.msg` CPU fallback及int32/int64 argsort
AiCPU执行，均保留并披露；未以关闭检查或修改dtype隐藏，不称纯AiCore。

最终质量门禁：Ruff、format（447files）、容器compileall和diff检查exit0。
快照 `/tmp/llmtuner-eager-flex-final-gates-UsGiQU` 执行全量pre-commit仍exit1，
仅既有未改动 `parallel/expert_parallel/hf_inspection.py` 两处tuple-isinstance
被Ruff改写；逐文件cmp确认快照只此文件不同，未回写工作区。日志
`eager_flex_backend_final_precommit_20261023.log`；三个新增leaf/backend测试文件
独立hooks全部通过，exit0，日志
`eager_flex_backend_final_precommit_new_files_20261023.log`。
不能把修正快照计为实际全门禁通过，既有失败仍未获用户豁免。
所有本节验证进程均已结束。原始P0–P4目标仍未完成；完整并行/特性矩阵、
权重验收依据及既有模型compile、KCP硬件、自定义FSDP版本和fullAC混合并行
间歇冷恢复等缺口继续保留，不能用新增配方的成功替代。

### 2026-10-23：selective Flex、PP packing 与历史混合并行复验

上一轮为实质进展（正式HF Flex及AC HOP修复）；本轮重读原目标后同时推进
P2/P3生命周期覆盖及历史失败复验。仓库main/HEAD及基线commit与上一节不变，
保留已有工作区，不提交。仅新增selective CPU回归及集成测试外部语料参数，
未改生产计算、梯度比较容差或checkpoint断言。

CPU回归将原fullAC HOP前后向逐位一致用例扩展至selective，验证保存/重放
路径；定向99 passed /15 warnings /14.88s，exit0，日志
`eager_flex_selective_ac_cpu_20261023.log`。全量CPU4218 passed /13 skipped /
273 warnings /142.48s，exit0，日志 `eager_flex_selective_full_cpu_20261023.log`。

以下NPU命令均为原集成脚本两个独立torchrun的full/resume，连续4步对比2＋2；
启用dynamic EMA、RNG与loader状态，原严格加载及最终状态比较、续训loss AE=0。
日志均位于 `outputs/qwen3-8b-npu/logs/`：

| 配置 | 结果 | 状态 | full/resume exit | 日志前缀（后接_full/resume_20261023.log） |
| --- | --- | --- | --- | --- |
| 历史原路径TP2/EP2/KVCP2、Qwen3MoE、FP32 seq256、nativeFSDP、fullAC，四卡0–3 | 本次全部原断言通过 | 通过（仅本次运行；历史间歇失败未关闭） | 0/0 | tp_ep_cp_fullac_post_hop |
| dense Qwen3、四项正式配置、nativeFSDP2、selectiveAC，双卡4,5 | 全部严格状态及loss通过 | 通过（有限配置） | 0/0 | production_flex_selective_cold |
| 同正式配置、PP2 1F1B、selectiveAC、生成JSONL经过实际tokenizer/packing/loader，双卡6,7 | 严格恢复通过；last-stage loss 4.894713→4.289792 | 通过（生成语料管线，不是真实LIMO） | 0/0 | production_flex_pp_selective_packed_cold |
| 同上Interleaved1F1B、六层/多个本地stage，双卡4,5 | 严格恢复通过；last-stage loss 4.815337→4.146901 | 通过（生成语料管线） | 0/0 | production_flex_interleaved_selective_packed_cold |

历史TP/EP/CP本轮并没有定位到间歇失败根因；fullAC HOP适配并非SDPA路径
确定性修复，因此不以本次成功注销原失败。三个新Flex配置仅FP32、指定小模型
及数据管线，未外推到所有精度/模型/并行度。

为避免把生成JSONL误称真实语料，集成脚本增加成对的
`LLMTUNER_TEST_CORPUS_PATH/TOKENIZER_PATH`，不生成或覆盖外部文件；参数不成对
或路径不存在时拒绝。另支持VOCAB_SIZE、DATASET、PROMPT_FIELD/RESPONSE_FIELD
以使用实际Qwen tokenizer与LIMO question/solution，不修改默认测试配置。
真实LIMO首次seq128配置未产出训练batch：日志明确全部过滤，主动只终止自建
torchrun（PID521544），没有因观察超时重启或干扰他人任务。原日志
`production_flex_pp_selective_limo_cold_full_20261023.log`保留，不计通过。

独立tokenizer长度审计确认817行，最短1189/最长35324；seq128及1024均0条可用，
2048为34条、4096为245条，tokenizer大小151669，日志
`limo_token_length_audit_verified_20261023.log` exit0。采用seq2048及模型padded
vocab151936预设新用例，并换独立目录；不截断响应、不关闭过滤。
首次审计误将Transformers返回BatchEncoding的键数当token数（min/max=1），
无效结果保留于 `limo_token_length_audit_20261023.log`，修正为显式取input_ids、
断言非空整数列表后复核。无效审计未用于宣称训练通过。

复现已完成子项（容器仓库根，已有目录须换新名）：

```bash
export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4,5
export LLMTUNER_TEST_WORLD_SIZE=2 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=1
export LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_MOE=0
export LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_QKV_PROJECTION=fused
export LLMTUNER_TEST_NORM_BACKEND=joint LLMTUNER_TEST_ROTARY_INIT_DEVICE=target
export LLMTUNER_TEST_ATTENTION_BACKEND=eager_flex LLMTUNER_TEST_AC=selective
export LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
for phase in full resume; do
  torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" outputs/production-flex-selective-cold-20261023 || exit $?
done
```

PP1F1B改设备6,7、PP_SIZE=2，目录
`outputs/production-flex-pp-selective-packed-cold-20261023`，两个命令均加--packed；
Interleaved改设备4,5、PP_SIZE=2、PP_SCHEDULE=Interleaved1F1B，目录
`outputs/production-flex-interleaved-selective-packed-cold-20261023`，同样--packed。
历史混合并行复验按本文件原TP/EP/CP完整命令，目录改
`outputs/tp-ep-cp-fullac-post-hop-20261023`，AC仍full，未启用四项新配置。

真实LIMO seq2048用例补记：PP2 1F1B、两层hidden64、vocab151936、FP32、正式
QKV/joint/targetRoPE/eagerFlex、selectiveAC、dynamicEMA/RNG/loader，实际Qwen3-8B
tokenizer及817行LIMO question/solution SFT数据，full/resume均exit0；日志
`production_flex_pp_selective_limo2048_cold_full_20261023.log`、
`production_flex_pp_selective_limo2048_cold_resume_20261023.log`。加载状态及最终
所有启用状态严格一致，loss为11.937691/11.846499/11.787813/11.745055，恢复
后两步maxAE=0。该模型不是8B预训练模型，不替代P0大模型验收；长响应仍按
既有SFT规则过滤，未截断。上文34/245为HF完整chat template的token长度审计
计数，不伪称训练processor已接受所有817行。seq128原用例因主动终止记录exit1，
并保留全过滤无batch证据，不计通过。

独立artifact审计exit0：rank0实际embedding、rank1实际head均为151936×64，
两rank均包含version/batch_layout/dp_rank_0/dp_world_size loader状态，日志
`production_flex_limo_artifact_audit_20261023.log`；不是继续使用小词表随机输入。
真实语料复现：在上述PP2 1F1B完整环境中再设置以下变量，使用两个独立phase：

```bash
export ASCEND_RT_VISIBLE_DEVICES=2,3 LLMTUNER_TEST_PP_SIZE=2
export LLMTUNER_TEST_SEQ_LEN=2048 LLMTUNER_TEST_VOCAB_SIZE=151936
export LLMTUNER_TEST_DATASET=local_jsonl_sft
export LLMTUNER_TEST_PROMPT_FIELD=question LLMTUNER_TEST_RESPONSE_FIELD=solution
export LLMTUNER_TEST_CORPUS_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl
export LLMTUNER_TEST_TOKENIZER_PATH=/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B
unset LLMTUNER_TEST_PP_SCHEDULE
for phase in full resume; do
  torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" outputs/production-flex-pp-selective-limo-2048-cold-20261023 --packed || exit $?
done
```

质量门禁：Ruff、format（447files）、容器compileall及diff检查exit0。
全量pre-commit在同步快照 `/tmp/llmtuner-flex-selective-gates-obK5Au` 仍exit1，
cmp确认仅既有未改动 `parallel/expert_parallel/hf_inspection.py` 两处tuple
isinstance被Ruff改写，未回写实际工作区。日志
`eager_flex_selective_precommit_20261023.log`；新增CPU用例文件单独hooks exit0，
日志 `eager_flex_selective_precommit_test_20261023.log`。外部语料参数补充后，
将实际集成脚本同步该快照并对两个受影响测试文件执行全部hooks，exit0，日志
`eager_flex_selective_external_corpus_precommit_20261023.log`。不能将这些局部
通过计为全量门禁通过，既有无关失败仍未豁免。

本节所有NPU/CPU验证进程已终止。TP/EP/CP fullAC间歇性失败仍无根因证明，
本次原路径通过不关闭历史缺口；完整声明矩阵、P4权重标准及硬件/compile/
custom FSDP缺口继续保留。全过滤配置无法产出batch的行为本轮只记录、未修复。
整体目标仍未完成。

### 2026-10-23：重复 map 首分片全过滤的构造期拒绝

本节承接真实LIMO seq128全过滤无batch的已复现问题。上一轮为实质进展；本轮
重读原目标后定位到Grain map过滤保留None索引，直接repeat再转iterator会
无限重访空槽位。未采用固定“1000条/过滤比例”阈值，因为合法稀疏源也可能
连续过滤超过1000条。原shuffle→DP切片→repeat及实际样本接受规则不变。

`datasets/composition.py::apply_iteration_policy` 在repeat之前索引探测初始本地
分片，首个非None即停止，整片均None则抛ConfigError，包含DP坐标、行数和
长度/过滤排查提示。用既有preserve_rng_state保护Python/NumPy/CPU Torch RNG，
不推进loader游标、不缓存整份语料、不截断响应。代价是增加构造期读取处理，
最坏扫描一整个本地分片。有限repeat=False保留空图语义；streaming、后续
shuffle/stochastic过滤epoch不在此检查证明范围。mix中各重复map子源分别
检查，不因静默丢弃空源而改变权重。自定义处理器须按Grain契约可按索引重入。
设计与C类来源说明分别更新于设计文档和upstream map。

日志统一位于 `outputs/qwen3-8b-npu/logs/`：

| 配置/测试项 | 断言及实际结果 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- |
| 新CPU回归：全过滤、shuffle、局部DP空片、有限空图、2048个索引中仅最后有效、三类RNG及短路读取 | 7 passed，14 warnings，5.20s；不误拒绝连续2047条过滤的稀疏源 | 通过 | 0 | empty_map_shard_isolated_verified_20261023.log |
| 全量CPU | 4225 passed，13 skipped，273 warnings，150.85s | 通过 | 0 | empty_map_shard_full_cpu_20261023.log |
| NPU2 PP2、真实LIMO/Qwen tokenizer、seq128 | 两rank构造期明确报告DP0/1的817行均不可用，训练exit1，不再无限等待batch；独立grep命中原错误 | 通过（拒绝测试，不是seq128训练通过） | torchrun1/错误匹配0 | limo_emptyguard_npu_20261023.log |
| NPU2 DP2/nativeFSDP，生成两行SFT：rank0长响应过滤、rank1短响应有效 | rank0 ConfigError明确DP0/2、1行；rank1收到“Dataloader construction failed on another rank”，两进程结束，无单rank卡住 | 通过（协调拒绝） | torchrun1/错误匹配0 | asymmetric_map_emptyguard_npu_20261023.log |
| NPU2 PP2 1F1B、LIMO seq2048、hidden64/vocab151936、正式QKV/joint/targetRoPE/eagerFlex、selectiveAC、dynamicEMA/RNG/loader | full4步vs独立进程2＋2步，原加载及最终状态断言全部通过，续训loss AE=0 | 通过（有限配置） | full0/resume0 | limo_nonemptyguard_full_20261023.log / limo_nonemptyguard_resume_20261023.log |
| 与修复前相同LIMO配置独立artifact对照 | 两rank split/final全部模型项及4步loss轨迹逐位相同（rank0 12项，rank1 13项） | 通过（指定配置不扰动） | 0 | emptyguard_cross_run_audit_20261023.log |

复现沿上一节真实LIMO完整环境：SEQ_LEN=128、设备2,3，调用full phase、目录
`outputs/production-flex-pp-limo-emptyguard-20261023`，期望非零并检查错误内容；
不能只把非零退出视为拒绝测试成功。有效seq2048改设备4,5、目录
`outputs/production-flex-pp-limo-nonemptyguard-20261023`，运行full与resume。
原失败配置未修改为通过；两者有独立日志与输出目录。

```bash
# 使用上一节真实LIMO完整配置；仅替换下面明确列出的设备/长度/output
export ASCEND_RT_VISIBLE_DEVICES=2,3 LLMTUNER_TEST_SEQ_LEN=128
torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py full outputs/production-flex-pp-limo-emptyguard-20261023 --packed
# 预期exit1，核对stderr含no usable samples及817 rows
export ASCEND_RT_VISIBLE_DEVICES=4,5 LLMTUNER_TEST_SEQ_LEN=2048
for phase in full resume; do
  torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" outputs/production-flex-pp-limo-nonemptyguard-20261023 --packed || exit $?
done
PYTHONPATH=. python outputs/emptyguard_cross_run_audit.py
```

非对称DP拒绝用例数据由 `outputs/create_emptyguard_asymmetric_fixture.py` 生成，
两行分别是重复1024次lorem ipsum的长响应和Two.短响应，仍用实际Qwen tokenizer；
使用上述SFT/词表配置，PP_SIZE=1、设备6,7、SEQ_LEN=128、CORPUS_PATH改为
`outputs/emptyguard-asymmetric-sft.jsonl`绝对路径，未启用四项新模型配置或AC。
调用同脚本full phase，目录 `outputs/asymmetric-map-emptyguard-npu-20261023`。
此fixture不是真实LIMO，不混入真实语料通过结论。

质量门禁Ruff、format（448files）、容器compileall及diff检查exit0。
快照 `/tmp/llmtuner-empty-map-gates-PQZUfJ` 全量pre-commit仍exit1，仅既有
未改动 `parallel/expert_parallel/hf_inspection.py` 两处tuple-isinstance自动
改写，cmp确认仅该tracked文件与工作区不同，未回写。日志
`empty_map_shard_precommit_20261023.log`；本轮实现/新增测试单独全部hooks
exit0，日志 `empty_map_shard_target_precommit_20261023.log`。

失败历史保留：初轮数据CPU为4失败/629通过，因新fixture误用make_policy的
repeat=False默认值而未触发guard；另一个独立-x运行同样1失败。修正fixture
为显式repeat=True，未改生产guard或放宽断言。日志分别
`empty_map_shard_targeted_cpu_20261023.log`、`empty_map_shard_isolated_20261023.log`。
容器拒绝验证wrapper首次使用未安装的rg，错误匹配状态127；实际torchrun确为
exit1。随后用容器已有grep独立核对错误内容，exit0，未为验证工具缺失重跑训练。
所有本节进程终止。该修复不关闭历史fullAC混合并行间歇失败、P4权重标准、
完整矩阵或硬件/compile/custom FSDP缺口，整体仍未完成。

## 2026-10-23：双轴 SDPA CP 独立内核验证，尚未解除装配保护

上一目标轮次仅修改Prompt，未推进原适配目标；本轮重新读取原始附件并核对工作区。
当前main与基线commit仍分别为c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19、
b358a8063c34aa7494397efa3a8f5ad24f1caa1f。容器实测torch2.12.0+cpu、
torch-npu2.12.0、8个NPU，启动前未发现其他torchrun训练任务。

新增 `DualCPSdpaKernel`：先Ulysses SeqToHead，再沿KV轴聚合K/V，使用KV轴
rank确定因果查询偏移，最后HeadToSeq恢复本地token。packed必须传完整四维
boolean文档mask，按KV查询范围裁切；拒绝缺失/错误类型mask和非零dropout。
复用既有_GatherSequence的梯度跨rank求和契约，不使用临时attention替换。
`apply_cp` 的双轴SDPA拒绝保护未动；尚未实现wrapper完整mask装配，不能作为
模型/Trainer或checkpoint恢复通过证据，也不关闭双轴生产路径缺口。

| 测试项 | 配置与基线/阈值 | 状态 | 退出码 | 日志 |
| --- | --- | --- | --- | --- |
| 双轴SDPA独立内核 | NPU4、KV2×Ulysses2、FP32；dense完整序列SDPA参考；输出与Q/K/V梯度atol=rtol=1e-5 | 通过 | 0 | `dual_cp_native_sdpa_kernel_20261023.log` |
| 原双轴Flex通信回归 | 相同9配置；仅此旧诊断分支替换attention，不能证明Flex内核 | 通过 | 0 | `dual_cp_flex_collective_regression_20261023.log` |

两组均覆盖noncausal/causal/packed × Q/KV heads=(4,4)/(4,2)/(6,2)，各9项，
梯度同时断言存在、有限且非零。packed文档长度5/6/5跨rank边界。
日志均在 `outputs/qwen3-8b-npu/logs/`。完整命令（仓库根目录）：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. LLMTUNER_TEST_DUAL_CP_BACKEND=sdpa timeout 240 torchrun --standalone --nproc_per_node=4 tests/integration_tests/dual_cp_attention_equivalence.py'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. LLMTUNER_TEST_DUAL_CP_BACKEND=flex timeout 240 torchrun --standalone --nproc_per_node=4 tests/integration_tests/dual_cp_attention_equivalence.py'
```

全仓Ruff、format（449files）、容器compileall及diff检查通过。新增5项CPU保护
测试覆盖collective前拒绝。容器仓库根目录执行 `pytest -q --disable-warnings`，
4230passed/13skipped/273warnings，143.11s，exit0；日志
`dual_cp_sdpa_full_cpu_20261023.log`。pre-commit在快照
`/tmp/llmtuner-dual-sdpa-gates-vBtw6P` 执行，全量exit1，仅既有
`parallel/expert_parallel/hf_inspection.py` 两处tuple-isinstance自动改写，
未回写工作区；本轮3个代码/测试文件的hooks全部通过，exit0。日志分别为
`dual_cp_sdpa_precommit_final_20261023.log`、
`dual_cp_sdpa_target_precommit_20261023.log`。首次使用不存在的pre-commit路径
exit127，随后未设quality-tools PYTHONPATH导致ModuleNotFoundError，均保留日志；
最终命令设置 `PYTHONPATH=$PWD/outputs/quality-tools` 并使用其bin/pre-commit。
这些工具启动失败不计作质量门禁通过。完整矩阵、历史fullAC间歇失败及P4权重
标准仍未关闭，整体目标未完成。

## 2026-10-23：双轴 SDPA 接入模型与 Trainer，packed 冷恢复

上一目标轮次新增内核及4卡数值证据，属于实质进展。本轮读取原始目标、开发
指南并复核main/双方commit及环境，与上一节一致。将 `DualCPSdpaKernel` 接入
`apply_cp`，以独立KV/Ulysses mesh装配，保留mesh尺寸、头整除、load-balancer
拒绝检查；不再无条件拒绝有实现的双轴SDPA。HF wrapper在全局positions仍完整
时建立boolean文档mask，先保留完整mask再切token；attention必须接收该全局
mask，缺失时明确拒绝，不从局部positions重建。单轴Ulysses原additive mask
契约不变。此变更不是绕过旧保护：先完成真实内核前反向证明，再补齐模型mask
及实际Trainer测试。O(T²)完整文档mask成本保留，不宣称稀疏/融合性能。

| 测试项 | 配置/阈值 | 状态 | 退出码 | 日志（统一日志目录） |
| --- | --- | --- | --- | --- |
| 双轴packed模型 | NPU4、KV2×Ulysses2、FP32、两层hidden64；dense全序列参考，atol/rtol=1e-5 | 通过 | 0 | `dual_cp_sdpa_packed_model_verified_20261023.log` |
| 单轴Ulysses回归 | NPU2、同模型/精度/阈值 | 通过 | 0 | `dual_cp_wiring_single_ulysses_regression_20261023.log` |
| 双轴Trainer冷恢复 | NPU4、PP1、native FSDP、AC关闭、packed256、动态EMA/RNG/loader，4步对2+2，状态精确一致 | 通过 | 0/0 | `dual_cp_sdpa_cold_verified_{full,resume}_20261023.log` |
| 双轴fullAC冷恢复 | 同上，AC=full | 通过 | 0/0 | `dual_cp_sdpa_fullac_{full,resume}_20261023.log` |
| PP×双轴×selectiveAC冷恢复 | NPU8、PP2/1F1B、KV2×Ulysses2、其余同上 | 通过 | 0/0 | `dual_cp_sdpa_pp_selective_{full,resume}_20261023.log` |

双轴模型logits maxAE=0，token平均loss AE=4.768e-7，全参数梯度maxAE=
1.676e-8；丢弃文档mask的参考logits差为0.6232，非空洞验证。单轴对应
logits=2.570e-7、loss=4.768e-7、参数梯度=2.980e-8。无attention替换，
模型与Trainer均断言实际DualCPSdpaKernel/CP_SDPA_BACKEND生效。
Trainer检查恢复前及最终model/optimizer/scheduler/trainer/EMA/RNG/loader
适用状态，尾部loss逐步bitwise相同，maxAE=0；检查标准未放宽。
PP1与fullAC的完整loss均为
`[4.980159759521484,4.524144172668457,4.270559310913086,4.143191337585449]`；
PP2/selective为
`[4.928006172180176,4.527803897857666,4.349355220794678,4.267665863037109]`。
此次fullAC有限配置成功不能关闭历史TP/EP/CP间歇冷恢复失败。
日志有 `aten::_assert_async.msg` 的显式CPU fallback警告，未隐瞒；不要求纯AiCore。

完整命令（仓库根目录；日志在 `outputs/qwen3-8b-npu/logs/`，每次独立保存）：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. LLMTUNER_TEST_DUAL_CP=1 timeout 240 torchrun --standalone --nproc_per_node=4 tests/integration_tests/cp_ulysses_varlen_equivalence.py'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. timeout 240 torchrun --standalone --nproc_per_node=2 tests/integration_tests/cp_ulysses_varlen_equivalence.py'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=1 LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1; for phase in full resume; do timeout 300 torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" outputs/dual-cp-sdpa-packed-cold-verified-20261023 --packed || exit $?; done'
```

fullAC命令在上述4卡命令export中增加 `LLMTUNER_TEST_AC=full`，独立目录为
`outputs/dual-cp-sdpa-fullac-packed-cold-20261023`。PP2/selective命令将
WORLD_SIZE/--nproc_per_node改为8、PP_SIZE改为2，增加
`LLMTUNER_TEST_AC=selective`，目录为
`outputs/dual-cp-sdpa-pp-selective-packed-cold-20261023`，每phase timeout360。

失败历史保留：模型首轮fixture误用build_device_mesh返回的world轴，装配按预期
拒绝无cp轴，exit1，日志 `dual_cp_sdpa_packed_model_20261023.log`；fixture
改为显式get_mesh("cp")，未修改生产mesh校验。首次Trainer full已完成一步，
新增路径断言引用局部作用域外device_type导致NameError，exit1，日志
`dual_cp_sdpa_cold_full_20261023.log`，修正为harness_device().type后使用新目录。
新增load-balancer单测首轮在配置构造时已被更早保护拒绝，目标测试1失败/134通过，
日志 `dual_cp_sdpa_wiring_final_target_cpu_20261023.log`；fixture改为先构造合法
配置再模拟运行期字段突变，以单独覆盖装配保护，未移除配置保护。

真实语料补验：NPU8、PP2/1F1B×KV2×Ulysses2、selectiveAC、动态EMA/RNG/loader、
FP32、seq2048、vocab151936，使用原始LIMO question/solution和Qwen3-8B tokenizer；
模型仍是两层hidden64的随机初始化小Qwen3，不是预训练8B模型。目录
`outputs/dual-cp-sdpa-pp-selective-limo-cold-20261023`，full/resume均exit0，
所有恢复断言通过，尾部loss精确一致，maxAE=0。完整loss为
`[11.937690734863281,11.846500396728516,11.787812232971191,11.74505615234375]`。
日志 `dual_cp_sdpa_pp_limo_{full,resume}_20261023.log`。保留95.50%过滤警告，
未禁用长度过滤、未截断或修改原始语料；该警告不等于源全空。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_PP_SIZE=2 LLMTUNER_TEST_TP_SIZE=1 LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_SEQ_LEN=2048 LLMTUNER_TEST_VOCAB_SIZE=151936 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=selective LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_TEST_CORPUS_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl LLMTUNER_TEST_TOKENIZER_PATH=/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B LLMTUNER_TEST_DATASET=local_jsonl_sft LLMTUNER_TEST_PROMPT_FIELD=question LLMTUNER_TEST_RESPONSE_FIELD=solution; for phase in full resume; do timeout 420 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" outputs/dual-cp-sdpa-pp-selective-limo-cold-20261023 --packed || exit $?; done'
```

独立artifact审计 `outputs/dual_cp_ac_cross_audit.py` 对照4卡noAC/fullAC各rank的
split/final模型、优化器、EMA、scheduler、trainer及token计数，全部精确相同，
完整loss轨迹也精确相同。DTensor先核对global shape、placements及mesh rank
布局，再逐rank比较本地分片，不进行无PG的DTensor全局equal；覆盖全部4rank。
容器仓库根目录运行 `PYTHONPATH=. python outputs/dual_cp_ac_cross_audit.py`，
exit0，日志 `dual_cp_sdpa_noac_fullac_cross_audit_verified_20261023.log`。
初次临时审计直接torch.equal(DTensor)触发无默认PG错误，exit1，保留
`dual_cp_sdpa_noac_fullac_cross_audit_20261023.log`；修正的是离线审计布局处理，
不是训练或恢复断言，未重新训练以选择结果。

最终容器根目录 `pytest -q --disable-warnings`：4238passed、13skipped、
273warnings、156.20s、exit0，日志
`dual_cp_sdpa_wiring_final_full_cpu_20261023.log`。此前版本全量4236passed，
不含后来补充的两项装配保护用例，不能代替最终门禁。Ruff、format449files、
容器compileall、git diff --check均exit0。
完整工作区tracked/untracked快照
`/tmp/llmtuner-dual-wiring-final-gates-6OuEL5` 全量pre-commit exit1，
仅既有 `hf_inspection.py` 两处tuple-isinstance及用户未跟踪
`.agent/karpathy-guidelines/SKILL.md` 缺EOF换行，未回写；本轮11个相关文件
全部hooks通过，exit0。日志为
`dual_cp_sdpa_wiring_final_precommit_20261023.log`、
`dual_cp_sdpa_wiring_target_precommit_20261023.log`。首个快照另发现本轮删除
docstring遗留的函数首空行及上一节状态文档末空行，已修正本轮文件并复验，
不混为无关失败。全量门禁未获用户豁免，不能标为通过。

本轮进程均终止。上述有限配置关闭双轴SDPA无实现/无模型验证缺口，但不能
证明所有精度、模型、TP/EP混合或性能范围；完整P2/P3矩阵、历史fullAC间歇
失败、KCP硬件、完整Inductor/custom FSDP及P4权重验收依据仍有未关闭项，
整体目标未完成。

## 2026-10-23：TP2×双轴CP独立参考与真实Trainer冷恢复

上一目标轮次已接入双轴SDPA并取得模型/PP/真实语料恢复证据，属于实质进展。
本轮重新读取原始附件及开发指南，核对当前main、TorchLLMTuner commit
c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19、基线commit
b358a8063c34aa7494397efa3a8f5ad24f1caa1f，保留已有工作区改动。容器仍为
torch2.12.0+cpu/torch-npu2.12.0，8个NPU；启动前没有其他torchrun作业。

仅扩展 `cp_ulysses_varlen_equivalence.py` 的验证能力，不修改生产实现：
`LLMTUNER_TEST_DUAL_CP=1 LLMTUNER_TEST_TP_SIZE=2` 选择8卡TP2×KV2×Ulysses2。
按生产顺序先apply_tp、再apply_cp，由真实preprocess负责CP后TP token切分。
独立完整序列HF SDPA参考不调用生产分片逻辑；聚合TP再CP还原logits，loss
按两轴求和。梯度按CP求和并使用Trainer的复制TP梯度归约；参考梯度仅按明确的
物理投影规则切片（Q/K/V/gate/up按dim0，O/down按dim1，其余复制），不复用
生产TP参数标记作为参考布局。每个参数名逐一匹配并检查有限梯度与误差，所有
rank共同裁决，不以rank0本地通过代替其他rank的断言。原TP1变体保留。

| 测试项 | 配置与基线/阈值 | 状态 | 退出码 | 日志（约定日志目录） |
| --- | --- | --- | --- | --- |
| TP×双轴packed模型 | NPU8，TP2/KV2/Ulysses2，两层hidden64/Q4/KV4、FP32，seq256，文档90/1/165；参考完整序列，atol/rtol=1e-5 | 通过 | 0 | `tp2_dual_cp_sdpa_packed_model_20261023.log` |
| 双轴TP1回归 | NPU4，KV2/Ulysses2，其他模型及阈值同上 | 通过 | 0 | `tp2_dual_cp_dual_only_regression_20261023.log` |
| 单轴Ulysses回归 | NPU2，Ulysses2，其他模型及阈值同上 | 通过 | 0 | `tp2_dual_cp_single_ulysses_regression_20261023.log` |
| TP×双轴selectiveAC冷恢复 | NPU8，TP2/KV2/Ulysses2，PP1、native FSDP、FP32、小Qwen3/packed256、dynamic EMA/RNG/loader，连续4步vs独立2+2，适用状态及逐步loss精确 | 通过 | 0/0 | `tp2_dual_cp_sdpa_selective_{full,resume}_20261023.log` |
| TP×双轴fullAC冷恢复 | 同上一行，AC=full | 通过 | 0/0 | `tp2_dual_cp_sdpa_fullac_{full,resume}_20261023.log` |

模型测试rank0报告logits maxAE=2.384e-7、token平均loss AE=9.537e-7、
全参数本地梯度maxAE=2.049e-8；所有rank各自阈值断言通过。丢弃文档mask的
参考logits差为0.6232，证明packed边界非空洞。TP1双轴对应logits/loss AE=0，
梯度=1.653e-8；单轴对应2.570e-7/4.768e-7/2.980e-8。
附带纯Ulysses QKV梯度诊断仍是单独通信检查，不将它冒充双轴或TP模型证据。

两组Trainer恢复前8rank state failures=0，最终model/optimizer/scheduler/
trainer与启用EMA/RNG/loader检查全部通过，loss尾部精确一致、maxAE=0。
两组完整loss均为
`[4.980159759521484,4.524144172668457,4.270559310913086,4.143191337585449]`。
实际DualCPSdpaKernel/CP_SDPA_BACKEND断言生效；日志显式报告
`aten::_assert_async.msg` CPU fallback，未删除保护或宣称纯AiCore。
本节fullAC仅证明该有限配置本次运行，不关闭历史TP2/EP2/KVCP2的间歇失败。

完整复现命令（仓库根目录，日志位于 `outputs/qwen3-8b-npu/logs/`）：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. LLMTUNER_TEST_DUAL_CP=1 LLMTUNER_TEST_TP_SIZE=2 timeout 300 torchrun --standalone --nproc_per_node=8 tests/integration_tests/cp_ulysses_varlen_equivalence.py'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. LLMTUNER_TEST_DUAL_CP=1 timeout 240 torchrun --standalone --nproc_per_node=4 tests/integration_tests/cp_ulysses_varlen_equivalence.py'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. timeout 240 torchrun --standalone --nproc_per_node=2 tests/integration_tests/cp_ulysses_varlen_equivalence.py'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1; for mode in selective full; do export LLMTUNER_TEST_AC="$mode"; if [ "$mode" = full ]; then label=fullac; else label=selective; fi; for phase in full resume; do timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" "outputs/tp2-dual-cp-sdpa-${label}-packed-cold-20261023" --packed || exit $?; done; done'
```

容器根目录 `pytest -q --disable-warnings`：4238passed/13skipped/273warnings，
152.62s，exit0，日志 `tp2_dual_cp_full_cpu_20261023.log`。全仓Ruff、
format449files、容器compileall及git diff --check均通过。完整工作区快照
`/tmp/llmtuner-tp-dual-gates-2i0ajI` 的全量pre-commit exit1，仍仅既有
`hf_inspection.py` 两处tuple-isinstance及未跟踪karpathy技能缺EOF换行，
自动改动仅在快照，未回写；日志 `tp2_dual_cp_full_precommit_20261023.log`。
本轮脚本及状态文档的全部hooks通过，exit0，日志
`tp2_dual_cp_target_precommit_20261023.log`。
未取得无关门禁豁免，不将全量门禁标为通过。
本节NPU/CPU运行均已终止，未提交或切换分支。新增有限组合证据不等于全部
DP/TP/EP/CP/PP声明矩阵通过，P4权重标准、硬件/compile/custom FSDP等仍有
未关闭项，整体目标未完成。

## 2026-10-23：原TP2/EP2/KVCP2 fullAC恢复失败的边界复现

本节均为诊断，不新增生产支持结论、不修改原始精确断言。工作仓库仍为main
`c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19`，基线仍为
`b358a8063c34aa7494397efa3a8f5ad24f1caa1f`；保留已有工作区改动。
运行环境沿用上述容器，NPU4、FP32、小Qwen3MoE两层hidden64/vocab128、
seq256非packed、native FSDP、TP2/EP2/KVCP2、fullAC、dynamic EMA、
RNG/loader恢复、确定性模式、AdamW lr=0.0003/max_norm=1。
连续4步对比独立进程2+2，入口仍是实际Trainer集成测试。

| 测试项 | 结果与状态 | 退出码 | 日志（约定日志目录） |
| --- | --- | --- | --- |
| step3更新后快照 | 本次model/裁剪梯度/Adam状态及step3/4指标精确；通过，仅有限诊断 | 0/0 | `ac_postupdate_boundary_{full,resume}_20261023.log` |
| 加入裁剪前布局元数据 | 本次布局/分组/格式/对齐及更新后状态精确；通过，仅有限诊断 | 0/0 | `ac_postupdate_layout_{full,resume}_20261023.log` |
| 仅保留已有CPU指标，三组成对运行 | 前两组通过；第3组恢复失败，四rank加载前检查均为0 | 0/0、0/0、0/1 | `ac_metrics_only_{1,2,3}_{full,resume}_20261023.log` |
| 第3组checkpoint冷重放两次 | 均复现step3 loss偏差；失败 | 1/1 | `ac_forward_same_checkpoint_{1,2}_20261023.log` |
| 同checkpoint加载前预热2步 | 仍复现相同偏差；失败，不能作为修复方案 | 1 | `ac_forward_same_checkpoint_warm2_20261023.log` |
| 成对前向边界快照 | pair1通过；pair2恢复失败，四rank加载前检查均为0 | 0/0、0/1 | `ac_forward_pair_{1,2}_{full,resume}_20261023.log` |

低扰动metrics-only第3组：step3 loss从4.865218162536621变为
4.8652167320251465，AE=1.430511474609375e-6；grad_norm从
0.2761554718017578变为0.2761615216732025。已在更新前的前向loss出现
差异，不能将所有恢复失败统一归因于裁剪或Adam更新。同一checkpoint的两次
冷重放及预热重放，9个step3前向事件的输入/kwargs/输出均精确一致，包括RoPE
position_embeddings；这只比较恢复运行，不证明其与未捕获的连续运行一致。

独立pair2：四rank的9个step3前向事件全部精确，step3 loss=
4.865217685699463、grad_norm=0.27616217732429504也精确；step4 loss
却从4.867334365844727变为4.867337226867676，AE=
2.86102294921875e-6。首次已观测差异在step3前向之后至step4之间，尚不能
区分step3反向/更新与step4前向。不能把另一历史失败的原始梯度精确观测拼接到
本pair2。逐rank审计见 `ac_forward_pair_{1,2}_comparison_verified_20261023.log`。
比较工具exit0只表示审计执行完成，不能覆盖训练exit1。

历史失败step3原始梯度审计：每rank27项full/resume逐位一致，全部rank本地
分片L2的保守重复计数合成值为0.3774660968301162，小于max_norm=1。
这不支持“轻微范数舍入跨裁剪阈值”的假设，但不还原当时实际分布式范数。
日志 `ac_historical_step3_norm_bound_verified_20261023.log`。
将历史rank0梯度在NPU上重建为stride1/2、offset0/1/7/31/128，foreach
True/False各5次，100次本地norm均为0.19168666005134583；stride2的格式
回退警告保留，未据此宣称分布式裁剪已排除。日志 `ac_norm_layout_probe_20261023.log`。

诊断脚本均位于忽略目录outputs：`probe_ac_postupdate_boundary.py`、
`compare_ac_postupdate_boundary.py`、`audit_failed_ac_step3_norm.py`、
`probe_npu_norm_layout.py`、`probe_ac_forward_boundary.py`、
`compare_ac_forward_boundaries.py`。布局首次审计因Format安全加载失败，前向首次
审计因全程4步/恢复2步列表不等长失败，均保留原日志并修正审计工具后重跑，
不计为训练失败或训练通过。相应verified日志是最终审计结果。

复现环境及命令（仓库根目录；每次使用独立目录，保留失败日志）：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_PROBE_POSTUPDATE_STEP=0; for phase in full resume; do timeout 300 torchrun --standalone --nproc_per_node=4 outputs/probe_ac_postupdate_boundary.py "$phase" outputs/tp-ep-cp-ac-metrics-only-3-20261023 || exit $?; done'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/compare_ac_postupdate_boundary.py outputs/tp-ep-cp-ac-metrics-only-3-20261023 --metrics-only'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/compare_ac_forward_boundaries.py outputs/tp-ep-cp-ac-forward-pair-2-20261023 pair2:full pair2:resume'
```

前向捕获沿用相同环境，将入口替换为`outputs/probe_ac_forward_boundary.py`、
设置`LLMTUNER_PROBE_TAG=pair2`，目录换为对应forward-pair目录；冷重放仅运行
resume，预热诊断另设置`LLMTUNER_PROBE_WARMUP_STEPS=2`。所有monkeypatch
和设备快照都会扰动时序，成功诊断不能排除其他失败运行存在差异。

已核实先前CPU全套实际结束：4238passed/13skipped/273warnings，141.91s，
日志 `ac_boundary_full_cpu_20261023.log`；当前无存活pytest。该结果不关闭
上述NPU精确恢复失败。继续采集同次运行的原始/裁剪梯度和更新后模型/Adam状态，
不降低精确标准，不添加全局同步或预热等未经根因证明的生产绕行。

### 同次运行设备侧更新边界：再次捕获裁剪前分歧

新增忽略目录诊断 `outputs/probe_ac_device_update_boundary.py`：step3/4裁剪前后
在设备侧clone梯度，train_step返回后clone模型及Adam状态；仅在测试退出后的
finally搬到CPU并保存。没有在裁剪前插入CPU值拷贝，但clone仍会扰动时序。
原始实际Trainer、fullAC、精确恢复断言不变；未修改生产源码。

第一组成对运行即复现：full exit0/resume exit1，目录
`outputs/tp-ep-cp-ac-device-update-1-20261023`；四rank加载前state failures=0。
step3 loss从4.8652167320251465变为4.865216255187988，AE=
4.76837158203125e-7。四rank的27项原始梯度均有差异；rank0 embedding梯度
maxAE=1.1433381587266922e-5，更新后embedding权重maxAE=
1.085922122001648e-6，Adam的一阶/二阶矩亦不同。
同次运行每个rank、两种phase、step3/4共16项检查均显示原始梯度与裁剪后梯度
逐位一致，证明本次裁剪未引入变化。不能据此排除所有历史失败的裁剪问题，亦不能
把本次梯度差异归因于AC重算而忽略已经存在的step3前向loss差异。

日志 `ac_device_update_1_{full,resume}_20261023.log`；审计
`ac_device_update_1_comparison_verified_20261023.log` exit0仅表示比较完成，
不代表恢复通过。预定最多三组的循环已在第一组失败后停止，第二/三组未跑。
复现沿用本节完整环境，入口替换为
`outputs/probe_ac_device_update_boundary.py`，目录替换为上述device-update目录；
审计命令：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/compare_ac_device_update_boundary.py outputs/tp-ep-cp-ac-device-update-1-20261023'
```

下一边界是同次失败运行的首次前向差异及AC重算；目前未取得根因修复，原配置
保持失败，整体目标保持未完成。

本轮全仓Ruff及format451files、容器compileall、git diff --check通过。
在既有隔离快照内对新增两份诊断脚本及本状态文档运行pre-commit，首次仅修正
诊断脚本三处tuple-isinstance写法，已同步相同修正；复验全部适用hooks通过，
exit0，日志 `ac_device_update_target_precommit_verified_20261023.log`。
全量pre-commit本轮未重跑，既有无关失败仍未豁免；CPU结果沿用本节已核实全套，
未在仅新增诊断/记录后重复运行。所有本轮NPU运行和审计均已结束。

## 2026-10-23：fullAC首次前向与重算的逐模块边界诊断

沿用上一节原TP2/EP2/KVCP2非packed小MoE配置、版本和精确恢复要求。
`outputs/probe_ac_forward_boundary.py`增加可选`LLMTUNER_PROBE_DETAILED=1`：
捕获step3/4所有已注册模块的前向pre/post边界，包含原始前向和AC重算标记，
设备clone保留至测试退出后才搬CPU。early_stop可能使重算只有pre事件，未改变
early_stop设置。未知非张量对象只记录限定类型名，不序列化动态模型类；这种摘要
不证明对象内部状态相等，张量值仍独立比较。默认非详细模式仍保留原采样范围。
审计工具可选`--all`输出全部差异，仍不将比较工具exit0当作训练通过。

首次详细捕获full进程训练完成，但finally保存时因kwargs里的动态TP attention
类无法pickle而exit1：`ac_detailed_1_full_20261023.log`。这是诊断工具失败，
不能计为恢复失败或通过；原日志和目录保留，修正摘要序列化后用新目录重跑。
前一轮取得新的失败边界及本轮修正探针均属进展，未修改生产实现或放宽断言。

复现命令（仓库根目录，每轮独立目录）：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_PROBE_DETAILED=1; for trial in 1 2 3; do export LLMTUNER_PROBE_TAG="detail${trial}"; for phase in full resume; do timeout 300 torchrun --standalone --nproc_per_node=4 outputs/probe_ac_forward_boundary.py "$phase" "outputs/tp-ep-cp-ac-detailed-verified-${trial}-20261023" > "outputs/qwen3-8b-npu/logs/ac_detailed_verified_${trial}_${phase}_20261023.log" 2>&1; result=$?; echo "trial=$trial phase=$phase exit=$result"; if [ "$result" != 0 ]; then exit "$result"; fi; done; done'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/compare_ac_forward_boundaries.py outputs/tp-ep-cp-ac-detailed-verified-1-20261023 detail1:full detail1:resume --all'
```

三组full/resume训练全部exit0，均通过原始精确冷恢复断言；每rank每组308个
step3/4边界事件和相应指标精确一致。日志
`ac_detailed_verified_{1,2,3}_{full,resume}_20261023.log`，逐rank审计
`ac_detailed_verified_{1,2,3}_comparison_20261023.log`均exit0。
这三组是成功样本，未捕获新的失败边界；不得覆盖上一节已复现失败。

新增离线 `outputs/audit_ac_recompute_boundaries.py` 审计第1组：每phase每rank
132个重算边界均找到同step/模块/边界类型的逐位相等原始前向候选。
首次按调用序号配对得到8项差异，但核对生产装配发现两层共享同一CP kernel
实例，named_modules只注册一个名字，而重算按层逆序发生；序号配对跨了层。
修正为候选精确匹配后全部为0，日志
`ac_detailed_verified_1_recompute_matching_20261023.log`；原误配审计日志
`ac_detailed_verified_1_recompute_audit_20261023.log`保留，不作为模型错误证据。
候选匹配只能证明存在相等值，不能独立证明共享模块的调用归属或唯一匹配。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/audit_ac_recompute_boundaries.py outputs/tp-ep-cp-ac-detailed-verified-1-20261023 detail1'
```

详细探针可能改变时序；未由三次成功推断原恢复问题已修复，也未将共享模块本身
判为根因。下一轮避免继续盲目增加相同成功样本，优先复用已稳定失败的checkpoint，
或转向独立未覆盖矩阵项。本轮未改生产源码，所有NPU作业及离线审计均已终止。

全仓Ruff、format451files、容器compileall及git diff --check通过；新增离线
审计脚本另行Ruff/format通过。隔离快照内本节三份诊断脚本及状态文档的
pre-commit全部适用hooks通过，exit0，日志
`ac_detailed_target_precommit_20261023.log`。本轮仅诊断/记录变更，未重跑CPU
全套或全量pre-commit；上一节4238passed及既有全量门禁失败范围保持不变。
P0–P4整体仍未完成，不新增正式矩阵通过声明。

## 2026-10-23：稳定失败checkpoint的详细重放

本轮先读原始目标，复核main及双方commit仍与前两节相同，保留既有改动，
开始时无存活torchrun/pytest。上一轮完成诊断工具和实际NPU证据，属于进展。
复用 `outputs/tp-ep-cp-ac-metrics-only-3-20261023` 的原checkpoint，不重写
full参考或历史失败日志。配置与上一节相同，仅tag改为`failed_detail`，phase
仅resume，设置`LLMTUNER_PROBE_DETAILED=1`。

详细重放exit1，四rank加载前state failures=0；step3 loss仍为
4.8652167320251465，相比原连续参考4.865218162536621差
1.430511474609375e-6。日志 `ac_failed_checkpoint_detailed_20261023.log`。
详细探针没有消除该已保存案例的失败；没有将失败改记blocked或放宽断言。

扩展比较工具`--coarse-step3`只保留step3原前向post事件及共同模块名，去掉
详细模式额外元数据，与历史cold1的9个事件逐一比较：四rank均精确，step3/4
指标也精确；日志 `ac_failed_checkpoint_coarse_detailed_comparison_20261023.log`。
扩展重算审计支持指定phase，仅检查该resume：每rank132个重算事件均找到同
step/模块/边界的相等原前向候选，日志
`ac_failed_checkpoint_recompute_audit_20261023.log`。两份审计exit0只表示
完成比较，不覆盖实际训练exit1；候选匹配不证明共享模块的唯一调用归属。
因此本案例不支持“重算数值与自身原前向不一致”作为已证根因；连续参考的逐模块
详细记录从未捕获，尚不能确定与连续参考的首次分歧位置。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_PROBE_DETAILED=1 LLMTUNER_PROBE_TAG=failed_detail; timeout 300 torchrun --standalone --nproc_per_node=4 outputs/probe_ac_forward_boundary.py resume outputs/tp-ep-cp-ac-metrics-only-3-20261023'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/compare_ac_forward_boundaries.py outputs/tp-ep-cp-ac-metrics-only-3-20261023 cold1 failed_detail --coarse-step3 --all'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/audit_ac_recompute_boundaries.py outputs/tp-ep-cp-ac-metrics-only-3-20261023 failed_detail resume'
```

接着对同一checkpoint运行AC=none、tag=`failed_noac_detail`的诊断恢复。
这是刻意变更AC的对照，不替代原fullAC验收；其他环境与上述命令相同，原full
参考不变，日志 `ac_failed_checkpoint_noac_diagnostic_20261023.log`。

无AC对照exit1，四rank加载前检查仍为0，step3 loss仍为
4.8652167320251465，与fullAC恢复相同；最终恢复断言仍失败。不能因为去掉AC
没有改善而证明所有历史fullAC问题均与AC无关，但本案例的偏差并不要求AC重算。
对照改变了验收配置，仅作诊断，不是fullAC支持修复。

首次直接全事件比较因重算事件数不同而exit1；仅保留step3原前向后仍因checkpoint
wrapper多两次同名层输出而exit1。两次审计失败日志均保留，不作模型差异证据。
新增显式`--common-events`选项：按共同模块名/出现次数比较，并报告排除项；
本次仅排除fullAC的两层wrapper各一个重复输出，保留42个原前向post事件。
四rank这42项输入/kwargs/输出及step3/4指标精确，日志
`ac_failed_checkpoint_fullac_noac_common_comparison_20261023.log`，审计exit0。
共同子集比较不是全事件等价证明，不用于替代原始测试断言；排除项有明确日志。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/compare_ac_forward_boundaries.py outputs/tp-ep-cp-ac-metrics-only-3-20261023 failed_detail failed_noac_detail --coarse-step3 --common-events --all'
```

本轮新增证据将该稳定失败的后续调查优先级转向连续/冷启动前向执行状态，而不是
直接修改重算语义。未插入生产同步、预热、全局关闭AC或放宽恢复精确标准。
正式fullAC矩阵项仍失败，P0–P4目标仍未完成。

本轮全仓Ruff、format451files、容器compileall及git diff --check通过；
隔离快照中两份受影响诊断脚本及状态文档的pre-commit适用hooks全部通过，
exit0，日志 `ac_failed_checkpoint_target_precommit_20261023.log`。本轮未改
生产代码，CPU全套及全量pre-commit未重复运行，既有无关全量门禁失败不变。
两次NPU恢复及三个比较任务均已终止，无作业留待后台。

## 2026-10-23：DP复制×双轴CP的真实Trainer冷恢复补验

先复核原始目标、开发指南及当前分支/版本，main和双方commit仍与前节一致，
保留所有既有改动；开始时无存活训练或pytest。上一轮已取得稳定失败checkpoint
的无AC对照证据，本轮转向独立P2缺项，不继续增加相同诊断成功样本。

集成脚本 `tests/integration_tests/pp_checkpoint_equivalence.py` 新增
`LLMTUNER_TEST_DP_REPLICATE_SIZE`（默认1）配置；replica>1时断言实际参数
DTensor mesh带指定大小的dp_replicate轴，且对应placement为Replicate；与CP
组合另断言仍存在真正Shard且local numel小于全局，避免只验证配置或纯复制。
保留NPU双轴实际DualCPSdpaKernel/backend断言及原有精确恢复断言。

本次有限配置：NPU8，dp_replicate2/dp_shard1/KV2/Ulysses2/TP1/PP1，
两层小Qwen3、hidden64/vocab128、FP32、packed256、native FSDP、dynamic
EMA/RNG/loader，连续4步vs冷启动2+2。原脚本实际Trainer/AdamW/DCP入口，
不是临时monkeypatch；模型/optimizer/scheduler/trainer及启用状态逐项精确。

selectiveAC full/resume均exit0，八rank恢复前state failures=0，尾部loss
maxAE=0；完整loss为
`[4.9917497634887695,4.528769493103027,4.273682594299316,4.132388114929199]`。
日志 `dp_replicate_dual_cp_selective_{full,resume}_20261023.log`。
增加真实Shard断言后另重放selective resume，并运行fullAC成对恢复，结果后补。
日志显式报告`aten::_assert_async.msg` CPU fallback，未声称纯AiCore。

完整命令（仓库根目录，日志目录沿用约定）：

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_DP_REPLICATE_SIZE=2 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1; for mode in selective full; do export LLMTUNER_TEST_AC="$mode"; if [ "$mode" = full ]; then label=fullac; else label=selective; fi; for phase in full resume; do timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" "outputs/dp-replicate-dual-cp-${label}-cold-20261023" --packed || exit $?; done; done'
```

补验selective分片断言使用同环境AC=selective/phase=resume及原selective目录，
日志 `dp_replicate_dual_cp_selective_resume_shards_20261023.log`。

最终结果：selective分片断言重放exit0，fullAC full/resume均exit0；后者八rank
恢复前state failures=0，尾部loss maxAE=0，完整loss与selective上述四步相同。
实际replica placement及真实参数分片断言均执行通过。fullAC日志
`dp_replicate_dual_cp_fullac_{full,resume}_20261023.log`。这增加一项正式有限
恢复配置，不等于DP×所有支持路径覆盖，也不替代该新拓扑的独立全序列梯度参考。
原TP2/EP2/KVCP2间歇恢复失败、P4权重标准及硬件/compile缺口仍未关闭。

质量门禁：CPU全套4238passed/13skipped/273warnings，146.34s，exit0，日志
`dp_replicate_dual_cp_cpu_20261023.log`；全仓Ruff、format449files、容器
compileall及git diff --check通过。受影响集成脚本pre-commit通过，exit0，
日志 `dp_replicate_dual_cp_script_precommit_20261023.log`。

全量pre-commit最初复用了已自动修正的旧快照而exit0，不能代表当前工作区；
随后重新复制全部git可见源码到隔离快照并运行，全量exit1，仍仅原有
`hf_inspection.py`两处tuple-isinstance和karpathy技能缺EOF换行。权威日志
`dp_replicate_dual_cp_full_precommit_current_snapshot_20261023.log`，自动
改动仅留在快照，未回写无关文件。旧exit0日志保留，不将全量门禁标为通过。
全部本轮NPU及CPU任务已结束；整体目标仍未完成，未提交或切分支。

## 2026-10-23：DP分片×双轴CP冷恢复补验

已重读原始目标及开发指南，main/双方commit与前节相同，保留既有工作区改动；
开始时无存活torchrun/pytest。上一轮补充了DP复制×双轴CP的正式有限恢复证据，
本轮验证独立DP分片轴，不将复制路径的结果外推至分片。

预定配置：NPU8，dp_replicate1/dp_shard2/KV2/Ulysses2/TP1/PP1、native
FSDP、FP32、两层hidden64/vocab128小Qwen3、packed256、dynamic EMA、
RNG/loader，AC=selective/full分别连续4步vs独立进程2+2。直接使用实际Trainer
集成脚本，无诊断monkeypatch；设置`LLMTUNER_TEST_REQUIRE_SHARDED=1`，
要求实际dp_shard>=2、FSDPModule和每个参数的Shard placement及非平凡本地
分片；双轴SDPA/backend仍有实际断言。预定状态和逐步loss均须精确一致。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_DP_REPLICATE_SIZE=1 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_REQUIRE_SHARDED=1 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1; for mode in selective full; do export LLMTUNER_TEST_AC="$mode"; for phase in full resume; do timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_checkpoint_equivalence.py "$phase" "outputs/dp-shard-dual-cp-${mode}-cold-20261023" --packed > "outputs/qwen3-8b-npu/logs/dp_shard_dual_cp_${mode}_${phase}_20261023.log" 2>&1; result=$?; echo "mode=$mode phase=$phase exit=$result"; if [ "$result" != 0 ]; then exit "$result"; fi; done; done'
```

本轮不修改生产或测试源码。此恢复用例不含独立完整序列参考，不能单凭恢复精确
证明新拓扑梯度归约与单卡参考等价，也不代表完整DP组合矩阵通过。

| 验收配置 | 基线/阈值与实际结果 | 状态 | 退出码 | 日志（约定日志目录） |
| --- | --- | --- | --- | --- |
| selectiveAC | 连续4步vs冷2+2，模型/optimizer/scheduler/trainer/EMA/RNG/loader及loss精确；八rank加载前检查0，loss maxAE=0 | 通过 | 0/0 | `dp_shard_dual_cp_selective_{full,resume}_20261023.log` |
| fullAC | 同上，实际模型/梯度布局及目标kernel断言生效；八rank加载前检查0，loss maxAE=0 | 通过 | 0/0 | `dp_shard_dual_cp_full_{full,resume}_20261023.log` |

两组完整loss均为
`[4.9917497634887695,4.528769493103027,4.273683071136475,4.132388114929199]`。
日志实际mesh为pp1/replicate1/shard2/cp4/tp1/ep1；与前节replicate2配置
数值不要求跨布局逐位一致，双方各自恢复精确才是本节标准。
`aten::_assert_async.msg` CPU fallback已明确报告，不声称纯AiCore。
所有四次NPU运行已结束。本轮只有状态文档变更，不重复CPU/源码质量门禁；
上轮CPU4238passed及全量pre-commit无关失败仍保留，整体目标未完成。

## 2026-10-23：PP×双轴CP的独立多步参考补验

已重读原始目标和开发指南，双方commit仍为前节记录版本，保留工作区改动；
开始时无存活torchrun/pytest。上一轮已取得DP分片恢复证据，本轮补验独立数值
参考，不用恢复精确代替训练等价。本轮不改生产或测试源码。

预定NPU8、PP2/KV2/Ulysses2、native FSDP、FP32、随机小Qwen3、seq256、
global_batch8、4microbatches、累积2、4步，分别1F1B和Interleaved1F1B。
实际Trainer路径无CPU attention替换；参考在同NPU上运行不切分的整网，从各
stage实际初始权重重建，按相同microbatch边界和有效token分母进行前反向及
AdamW更新。不调用生产CP分片/通信或PP schedule。逐步比较loss、grad norm、
各stage每个参数梯度和更新，原有atol/rtol=1e-5不变；stage划分及全rank失败
汇总断言保留。该脚本是P2独立参考，不是P4双方真实训练入口的替代品。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native; for schedule in 1F1B Interleaved1F1B; do timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_cp_equivalence.py "$schedule" 2 > "outputs/qwen3-8b-npu/logs/pp_dual_cp_independent_${schedule}_20261023.log" 2>&1; result=$?; echo "$schedule exit=$result"; if [ "$result" != 0 ]; then exit "$result"; fi; done'
```

1F1B已结束exit0：实际报告world8/dp_shard1/dp_replicate1/cp4/cp_kv2/
cp_ulysses2/native；loss maxAE=9.537e-7、梯度maxAE=2.072e-8、参数
maxAE=2.701e-6，failed ranks=0。日志
`pp_dual_cp_independent_1F1B_20261023.log`。日志显式报告CPU
`aten::_assert_async.msg`及AiCPU ArgSort，不将该结果描述为纯AiCore执行。

Interleaved1F1B同样exit0、failed ranks=0：loss maxAE=4.768e-7、梯度
maxAE=3.725e-8、参数maxAE=3.528e-6；日志
`pp_dual_cp_independent_Interleaved1F1B_20261023.log`。其六层模型每rank
两个虚拟stage；1F1B为两层、每rank一个stage。两组均hidden64/intermediate128/
vocab128/Q4/KV4，未启用packing/AC，不能外推至这些组合或Qwen3-8B。

| 调度 | 基线与阈值 | 实际结果 | 状态 | 退出码 |
| --- | --- | --- | --- | --- |
| 1F1B | 同设备完整模型、同初始权重/数据/microbatch；逐步及逐参数atol/rtol1e-5 | loss9.537e-7/grad2.072e-8/weight2.701e-6，全rank断言通过 | 通过 | 0 |
| Interleaved1F1B | 同上，六层与虚拟stage布局 | loss4.768e-7/grad3.725e-8/weight3.528e-6，全rank断言通过 | 通过 | 0 |

上述权重误差是该P2参考脚本预定容差下的结果，不是未确认的P4跨框架权重标准。
两次NPU作业已结束。本轮仅状态文档修改，未重跑CPU/源码门禁；上轮CPU全套及
既有全量pre-commit无关失败仍保留，原TP2/EP2/KVCP2恢复失败未关闭。
整体P0–P4目标仍未完成。

### P0 真实端到端计时补验（2026-10-23）

本次补齐历史日志时间戳不能证明的 step 耗时分布，不替代已有独立进程恢复
证据，也不关闭 TP2×EP2×KVCP2 的间歇性恢复失败。仓库 HEAD 为
`c1367535ab2ee2ff6f7d2ce4b88b050dd4f2dd19`，基线 HEAD 为
`b358a8063c34aa7494397efa3a8f5ad24f1caa1f`；使用当前有改动工作区。

示例新增默认关闭的 `LLMTUNER_ENABLE_TENSORBOARD=1` 开关，持久化现有
MetricsProcessor 的 perf_counter 端到端计时，无训练算法变更。运行前固定
预热为 step 1–3，统计 step 4–10；8 卡原规模 Qwen3-8B、seq4096、global
batch8、BF16、full AC、seed42、deterministic=false、最终 checkpoint 开启。
启动前 npu-smi 无运行进程，可用磁盘143T。完整执行命令：

```bash
docker exec hyper-parallel-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner
export LLMTUNER_DATASET_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl
export LLMTUNER_PROMPT_FIELD=question LLMTUNER_RESPONSE_FIELD=solution
export LLMTUNER_STEPS=10 LLMTUNER_MAX_SEQ_LEN=4096
export LLMTUNER_ENABLE_TENSORBOARD=1 LLMTUNER_SAVE_CHECKPOINT=1
export LLMTUNER_DUMP_FOLDER=/home/jianzhnie/llmtuner/llm/TorchLLMTuner/outputs/qwen3-8b-npu/p0_timing_20261023
timeout 900 bash examples/train_qwen3_8b_npu.sh > outputs/qwen3-8b-npu/logs/p0_timing_20261023.log 2>&1
'
```

| 测试项 | 基线/断言 | 实际结果 | 状态 | 退出码/证据 |
| --- | --- | --- | --- | --- |
| P0 10步与最终保存 | 实际8卡入口，10步指标有限，最终保存正常退出 | exit0，loss 0.516928→0.429602，期间范围0.393847–0.686207；不同batch有波动，不作收敛证明 | 通过（训练/计时子项） | 0；`outputs/qwen3-8b-npu/logs/p0_timing_20261023.log` |
| 真实端到端step耗时 | TensorBoard逐步记录1–10，排除预定预热1–3 | min1.888137 / median1.924933 / mean1.991996 / P95 2.308912 / max2.469963秒 | 通过（分布采集） | 0；`outputs/qwen3-8b-npu/logs/p0_timing_audit_20261023.json` |

P95为7个样本的线性插值，不代表长期性能保证。最慢step6的数据加载为
0.043703秒（1.769%），仅解释其中一部分，剩余波动未定位；其余统计步数据
加载约0.0008–0.0010秒。计时为metrics记录边界间的端到端时间，不是纯kernel
耗时，也不包含训练后的最终checkpoint落盘；不以此宣称无性能异常。
原始events保存在本次独立输出目录的`tb/`，审计脚本为
`outputs/audit_p0_timing.py`，逐步loss、grad norm和全部标量均检查有限。
显式保留CPU/AiCPU及其他后端警告，不作纯AiCore声明。

本轮验证：示例配置测试7passed；全CPU套件4241passed、13skipped、273warnings
（154.14秒），日志`outputs/qwen3-8b-npu/logs/p0_timing_cpu_20261023.log`。
Ruff check、449文件format check、容器compileall及git diff --check通过。
刷新隔离快照后执行全量pre-commit，仍仅已有无关文件
`llmtuner/parallel/expert_parallel/hf_inspection.py`的两处Ruff自动修复及
`.agent/karpathy-guidelines/SKILL.md`的EOF失败；未回写其自动修改。
日志`outputs/qwen3-8b-npu/logs/p0_timing_precommit_20261023.log`，退出码1。
整体目标仍未完成，P4关键权重标准、compile/KCP及组合矩阵缺口保持开放。

## 2026-10-23：无张量克隆的前向布局审计

已读原始目标及开发指南，main/双方commit仍与前节相同，保留既有改动；
开始时无存活训练/pytest。上一轮输出克隆探针属于新诊断证据，但未捕获失败，
本轮改用不同观察量，不继续相同探针。
新增忽略目录`outputs/probe_ac_forward_layout.py`：仅在step3/4原前向记录
各模块输入/kwargs/输出的shape、stride、storage_offset、指针模512对齐、
dtype/device，以及DTensor mesh/placements；不clone、不搬CPU、不读取张量
数值、不启动额外NPU数学操作，已有CPU指标原样保留，结束后JSON写盘。
Python hooks仍可能影响时序，且不包含NPU内部格式、完整地址或算法选择证据。
未知对象仅记类型，不能据此证明其内部状态一致。

配置沿用原NPU4 TP2/EP2/KVCP2/fullAC、native FP32、非packed256小MoE、
dynamic EMA/RNG/loader、确定性、4vs冷2+2。原实际Trainer及精确断言不变，
最多三组成对运行，首个训练失败即停；仅诊断，不修改生产源码。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1; for trial in 1 2 3; do for phase in full resume; do timeout 300 torchrun --standalone --nproc_per_node=4 outputs/probe_ac_forward_layout.py "$phase" "outputs/tp-ep-cp-ac-forward-layout-${trial}-20261023" > "outputs/qwen3-8b-npu/logs/ac_forward_layout_${trial}_${phase}_20261023.log" 2>&1; result=$?; echo "$trial $phase exit=$result"; if [ "$result" != 0 ]; then exit "$result"; fi; done; done'
python outputs/compare_ac_forward_layout.py outputs/tp-ep-cp-ac-forward-layout-1-20261023
```

比较脚本`outputs/compare_ac_forward_layout.py`对全部rank逐事件审计；exit0
只代表完成比较，不代表训练通过。第一组full/resume已exit0，每rank88个
事件布局及step3/4指标精确，日志`ac_forward_layout_1_comparison_20261023.log`。
成功诊断不能排除其他失败运行的布局变化，也不关闭历史恢复失败。

最终三组full/resume均exit0，三个逐rank比较exit0；每组每rank88个事件
所记录布局及step3/4指标精确。日志
`ac_forward_layout_{1,2,3}_{full,resume}_20261023.log`与
`ac_forward_layout_{1,2,3}_comparison_20261023.log`。未捕获失败，本轮三组
上限已到，不将相同成功诊断继续累积或用来替代原恢复要求。
六个NPU进程组及审计均已结束。本轮只新增忽略目录诊断及记录，未改生产逻辑。
两脚本Ruff/format、容器compileall、git diff --check通过；不重复CPU全套及
全量门禁，既有无关失败不变。下一调查须基于已保存失败输入的局部算子或明确的
执行状态差异，而不是以全局同步、关闭AC或容差变更制造通过。

### 2026-10-23：已保存失败路由logits的独立局部重放

本轮已重读原始目标及开发指南，双方commit保持前节版本，保留工作区改动，
开始时无存活torchrun/pytest。改用历史稳定失败checkpoint的真实原前向张量，
不重复整网成功探针。新增忽略目录`outputs/replay_failed_ac_router.py`，读取
`outputs/tp-ep-cp-ac-metrics-only-3-20261023/forward-cold1-resume-rank{rank}.pt`。
该产物对应先前实际失败resume，不从其他成功运行替换输入。

对四rank×两层已保存gate logits，独立执行softmax→topk2→gather→归一化→
scatter routing map，比较同次保存的router输出。单个NPU0，确定性开启、
FP32，存储offset0/1/7/31各5次，共160次。全部scores逐位精确、maxAE=0，
expert IDs及routing map也逐位精确。该测试不调用生产router实现、不使用阈值
放宽；日志 `ac_failed_router_replay_20261023.log`，exit0。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0 timeout 120 python outputs/replay_failed_ac_router.py'
```

仅证明给定已保存logits的局部路由结果可稳定重放，不包含gate GEMM、aux梯度、
EP通信或整网恢复；连续参考未捕获对应张量，不能据此排除首次差异发生在gate
之前或连续参考路由不同。分配padding警告已保留，不声称纯AiCore。
另一次探索读取split-state时在未初始化进程组下访问DeviceMesh.mesh失败，未
启动训练、未改变产物，不当作模型失败或gate重放证据；下一次需要先明确分片
重建规则，而不能直接把local权重当全量。未改生产源码，原恢复失败保持。

### 2026-10-23：已保存失败gate GEMM的权重重建与重放

已重读目标及开发指南，工作区及版本保持不变；上一轮实际router重放属于进展。
本轮核实split-state的gate权重为全局(8,64)、本地(4,64)、Shard(dim0)。
既定row-major world布局是pp/replicate/shard/cp/tp，TP2/CP2下CP分组为
[0,2]、[1,3]；dense FSDP沿dp_shard+cp分片。按此预定顺序拼接rank0和2
的行分片，不由输出误差选择排列；同时断言rank0/1、rank2/3对应分片逐位相等。
无需访问未初始化进程组的DeviceMesh.mesh或调用full_tensor collective。

新增忽略目录`outputs/replay_failed_ac_gate.py`，使用同一历史失败产物的
step2 split-state及cold1恢复step3 gate输入，独立FP32 torch.mm(input,weight.T)
比较同次保存的gate输出。四rank×两层、offset0/1/7/31各5次，共160次，
全部逐位精确，maxAE=0，exit0，日志`ac_failed_gate_replay_20261023.log`。
确定性开启、单NPU0；不调用生产gate自定义autograd或TP/CP通信。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0 timeout 120 python outputs/replay_failed_ac_gate.py'
```

结论仅限给定已保存输入/权重的前向GEMM：未复现布局偏移导致的数值变化。
不覆盖反向、真实分布式all-gather时序、连续参考未捕获的输入或RMSNorm/attention，
不关闭原恢复失败。加载DTensor时的无PG警告及NPU分配padding警告保留；没有
编译/collective调用，不把警告当模型失败。实际NPU重放已结束，未改生产源码。

### 2026-10-23：恢复前运行时RoPE buffer覆盖审计

已重读原始目标及开发指南，main及双方commit不变，保留已有改动；开始时
无存活torchrun/pytest。审查确认集成脚本`_runtime_state`模型快照仅包含
state_dict，未覆盖non-persistent buffers，因此历史pre-update failures=0
不证明所有前向运行时buffer一致。这是证据覆盖限制，不是已发现恢复根因。

新增忽略目录`outputs/probe_ac_rotary_buffers.py`，在step3开始、实际前向之前
捕获所有RotaryEmbedding的named_buffers、original_inv_freq及attention_scaling，
连续与恢复各自独立保存。CPU复制会同步并扰动时序，仅作诊断，不修改生产
buffer或断言；不将MoE统计scratch等有不同生命周期的buffer强行当持久状态。
原NPU4 TP2/EP2/KVCP2/fullAC、FP32、非packed256小MoE、native FSDP、
dynamic EMA/RNG/loader、4vs2+2配置保持不变。

full/resume均exit0。四rank各一个`model.model.rotary_emb`，上述全部buffer
与标量按atol=rtol=0比较精确。日志`ac_rotary_buffers_{full,resume}_20261023.log`
及`ac_rotary_buffers_comparison_20261023.log`；比较exit0。本次没有失败，
不能据此排除历史失败的buffer变化或关闭原恢复项，不继续重复相同成功探针。

复现沿用本节前述原TP2/EP2/KVCP2完整环境，入口替换为
`outputs/probe_ac_rotary_buffers.py`，依次phase=full/resume，目录
`outputs/tp-ep-cp-ac-rotary-buffers-20261023`。独立保存的
`rotary-{full,resume}-rank{0,1,2,3}.pt`均使用weights_only=True加载、
`torch.testing.assert_close(left,right,atol=0,rtol=0)`比较。
仅本次完整成对运行证明这些状态相等，不将另一历史checkpoint的观测拼接进来。

两次NPU运行和比较任务均已结束。诊断脚本Ruff/format及git diff --check通过，
未改生产源码、未降低精确标准，整体P0–P4目标仍未完成。

### 2026-10-23：按原始要求复核剩余验收与P0计时证据

本轮重新读取原始附件及基线numerical_alignment.md，不把既有成功子项当作
整体验收。前轮新增RoPE覆盖证据属于进展，但没有修复原恢复失败。本轮无新增
NPU任务，不修改实现或放宽标准，以下是缺口审计而非完成声明。

独立解析原P0日志仅带[Rank 0]的10条metrics记录，要求step严格为1至10，
避免将多rank重复日志当时间样本。原始日志日期为2026-10-22；排除step1–3，
step4–10的7个相邻日志间隔均为2秒，min/median/max/mean均2秒。但源时间戳
仅精确到秒，故这只是日志间隔，不是精细step耗时分布。step6 tps=1649，而
相邻step5/7为2139/2159，日志间隔仍同为2秒，实际证明秒级间隔掩盖吞吐变化。
不声称抖动消失，也不把由四舍五入吞吐反推的数值当原始计时。
机器可读审计产物：`outputs/qwen3-8b-npu/logs/p0_timestamp_distribution_audit_20261023.json`。

| 原始要求 | 当前权威证据及限制 | 尚需完成 |
| --- | --- | --- |
| P0预热后step耗时分布及异常解释 | 后续30B运行`p0_node92_20261023`已有唯一TensorBoard event的10步结构化计时；排除step1–3，min7.862334s、median7.950103s、mean10.643033s、p95（线性插值）21.207358s、max26.875656s | 此项已有证据，无需为补计时重跑；step6窗口包含step5保存18.38s，末次保存18.63s在计时窗口外。仅代表metrics端到端窗口，不代表纯计算或进程总耗时 |
| P1 KCP | 已实测当前910_9392内核明确拒绝，要求arch3.5 | 获得兼容设备并运行实际前反向；拒绝保护不算功能通过 |
| P2声明组合覆盖 | 有限子项、各具体日志仍有效；完整矩阵行仍为未跑 | 将DP复制/分片×每个声明路径逐项映射已有证据，缺项补正式参考，不无穷扩展交叉积 |
| 原TP2/EP2/KVCP2 fullAC精确冷恢复 | 已保存多个exit1案例；局部重放/带钩子成功未提供修复 | 捕获同次失败的连续/恢复首次分歧并根因修复，保留精确断言 |
| P3整模型compile | 实际Inductor整模型失败，collective局部成功不替代 | 修复真实编译训练或满足明确环境解锁条件后重验 |
| P4 logits/loss/grad_norm | 正式fusedQKV/joint/targetRoPE/eagerFlex双rank100步审计原阈值通过；默认配置旧失败保留 | 只对该显式配置承认指标子项，不宣称默认或其他布局通过 |
| P4关键权重 | 正式100步审计rank0/1最终state maxAE分别0.0007038903422653675/0.000631401315331459，91项均非逐位一致 | 用户确认权重对象、时点及依据；原规范没有权重近似阈值，不借logits标准或已有误差事后设线 |
| 质量门禁 | 后续CPU全套session48469退出0：4293 passed、13 skipped、273 warnings、149.67s；完整pre-commit隔离检查session62030退出1，仅`hf_inspection.py`及`.agent/karpathy-guidelines/SKILL.md`被hooks修改 | 保留pre-commit失败；隔离副本`/tmp/llmtuner-current-quality-2fydsq75`未回写，未经授权不修无关文件或声称豁免 |

本表的P0计时与质量门禁行已按2026-10-24可用的后续证据更新；上文秒级
日志间隔分析保留为历史诊断。30B计时复现脚本为
`outputs/audit_30b_p0_metrics.py`，详见30B P0记录；CPU日志为
`outputs/qwen3-30b-npu/logs/cp_modes_cpu_suite_20261024.log`。

P4最终state包含buffer等条目，maxAE不能直接当“关键权重”汇总指标。
对应正式日志为`p4_production_flex_audit_20261023.log`，两rank均明确
scalar_failures/logits_failures为空；不重新解释为权重通过。基线规范明确先选
对象和标准再比较，不能根据本表观测反推验收阈值。
此表不替代各历史节完整命令或扩大支持范围。整体目标仍未完成；目前仍有可独立
推进的证据盘点/修复工作，因此不将整个目标标为blocked。

### 2026-10-23：失败attention输入的独立KV拼接与SDPA重放

已重读原始目标及开发指南，保留当前工作区及既定版本；上一轮gate GEMM重放
属于进展。本轮使用同一稳定失败目录的`forward-failed_detail-resume-rank*.pt`，
过滤step3原前向kernel输出事件，四rank各两次调用；共享kernel名称不代表同层，
此处仅按原前向调用顺序标记layer_call，不混入逆序重算。

新增忽略目录`outputs/replay_failed_ac_attention.py`，按既定CP组[0,2]/[1,3]
独立拼接K/V序列，query保持本rank，独立按CP局部rank×local_len生成因果mask，
scale0.25、dropout0，直接SDPA并转回HF输出布局。没有调用生产CP kernel或
all-gather，比较同次失败原前向保存的输出；每个Q/K/V为FP32，局部seq1024、
head2、dim16。确定性开启，单NPU0，offset0/1/7/31各3次，四rank×两次
调用共96次，全部逐位精确、maxAE=0，exit0。
日志 `ac_failed_attention_replay_20261023.log`。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0 timeout 180 python outputs/replay_failed_ac_attention.py'
```

仅证明给定已保存Q/K/V及预定CP拼接顺序的局部SDPA可重放，不覆盖collective
时序、上游QKV/RoPE/归一化、反向或未捕获的连续参考输入。没有据此关闭原恢复
失败，也没有修改生产attention。NPU分配padding警告保留；本轮任务已结束。

## 2026-10-23：PP×双轴CP×AC独立训练参考

已重读原始目标和开发指南，保留当前main工作区及双方既定commit；上一轮已
取得无AC独立参考证据，本轮补上AC组合，不用冷恢复相等代替梯度参考。
`tests/integration_tests/pp_equivalence.py`新增`LLMTUNER_TEST_AC`，默认none；
full/selective要求每个本地decoder layer实际为CheckpointWrapper。梯度/参数
快照以canonical_fqn规范化wrapper名字，参考模型仍不应用AC、不调用CP/PP
通信。初始权重、数据及逐步loss/norm/参数梯度和更新的1e-5原阈值不变。
日志补充实际AC模式，不改生产实现或训练数值。

有限矩阵：NPU8，PP2/KV2/Ulysses2、native FSDP、FP32、seq256、global
batch8、microbatches4、累积2、4步；full/selective各配1F1B两层和
Interleaved1F1B六层。hidden64/intermediate128/vocab128/Q4/KV4，非packed。
全部走实际Trainer，NPU无attention替换；仅CPU参考适配按原脚本保留。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native; for mode in full selective; do export LLMTUNER_TEST_AC="$mode"; for schedule in 1F1B Interleaved1F1B; do timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_cp_equivalence.py "$schedule" 2 > "outputs/qwen3-8b-npu/logs/pp_dual_cp_ac_${mode}_${schedule}_20261023.log" 2>&1; result=$?; echo "$mode $schedule exit=$result"; if [ "$result" != 0 ]; then exit "$result"; fi; done; done'
```

本节权重容差是P2脚本已有的更新参考标准，不作为P4待确认的权重标准。
不覆盖packing/Qwen3-8B，不关闭原TP2/EP2/KVCP2间歇恢复失败。

| AC与调度 | 最大loss/梯度/参数AE | 状态 | 退出码 |
| --- | --- | --- | --- |
| full / 1F1B | 9.537e-7 / 2.072e-8 / 2.701e-6 | 通过 | 0 |
| full / Interleaved1F1B | 4.768e-7 / 3.725e-8 / 3.528e-6 | 通过 | 0 |
| selective / 1F1B | 9.537e-7 / 2.072e-8 / 2.701e-6 | 通过 | 0 |
| selective / Interleaved1F1B | 4.768e-7 / 3.725e-8 / 3.528e-6 | 通过 | 0 |

四组逐步loss/norm、逐参数梯度/更新断言均通过，failed ranks=0；首次full/1F1B
运行时报告尚未添加ac打印字段，但实际CheckpointWrapper断言已生效，其余组
同时打印实际ac模式。日志仍明确报告CPU断言回退和AiCPU ArgSort，不声称纯AiCore。
本轮四个NPU任务及CPU全套均已结束。

CPU全套4238passed/13skipped/273warnings，153.48s，exit0，日志
`pp_dual_cp_ac_cpu_20261023.log`；全仓Ruff、format449files、容器compileall、
git diff --check通过。当前源码全量隔离快照pre-commit exit1，仍仅既有
`hf_inspection.py`两处tuple-isinstance及karpathy技能缺EOF换行，日志
`pp_dual_cp_ac_full_precommit_20261023.log`；未回写无关自动修正，未豁免失败。
本轮证据不证明完整P2/P3矩阵或P4通过，整体目标仍未完成。

## 2026-10-23：DP复制×PP×Ulysses×AC独立训练参考

已重读原始目标及开发指南，main/双方commit与前节一致，保留工作区改动；
开始时无存活torchrun/pytest。上一轮新增AC集成入口和实际NPU参考证据属于进展。
本轮不改源码，补验DP复制轴与PP/Ulysses/AC的独立数值组合，不复用DP分片
或冷恢复通过来替代该参考。

预定NPU8、DP-replicate2/DP-shard1/PP2/KV1/Ulysses2、native FSDP、
FP32、seq256、global_batch8、4microbatches、累积2、4步。full/selective
各配1F1B两层或Interleaved1F1B六层；hidden64/intermediate128/vocab128/
Q4/KV4、非packed。保留实际AC wrapper、stage分割、每个参数梯度及更新、
逐步loss/norm与所有rank裁决。参考无PP/CP/AC，按dp_size展开每DP-rank的
microbatch边界，使用实际初始权重和全局有效token分母，原atol/rtol=1e-5。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_DP_REPLICATE_SIZE=2 LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_CP_STRATEGY=ulysses LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native; for mode in full selective; do export LLMTUNER_TEST_AC="$mode"; for schedule in 1F1B Interleaved1F1B; do timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_cp_equivalence.py "$schedule" 2 > "outputs/qwen3-8b-npu/logs/dp_replica_pp_ulysses_ac_${mode}_${schedule}_20261023.log" 2>&1; result=$?; echo "$mode $schedule exit=$result"; if [ "$result" != 0 ]; then exit "$result"; fi; done; done'
```

首组full/1F1B exit0，failed ranks=0；实际mesh报告replicate2/shard1/
PP2/KV1/Ulysses2，loss maxAE=4.768e-7、梯度maxAE=8.004e-9、参数
maxAE=8.792e-7。日志 `dp_replica_pp_ulysses_ac_full_1F1B_20261023.log`。
CPU断言回退及AiCPU ArgSort已明确报告，不声称纯AiCore。

| 配置 | 最大loss/梯度/参数AE | 状态 | 退出码 |
| --- | --- | --- | --- |
| full / 1F1B | 4.768e-7 / 8.004e-9 / 8.792e-7 | 通过 | 0 |
| full / Interleaved1F1B | 9.537e-7 / 2.416e-8 / 1.838e-6 | 通过 | 0 |
| selective / 1F1B | 4.768e-7 / 8.004e-9 / 8.792e-7 | 通过 | 0 |
| selective / Interleaved1F1B | 9.537e-7 / 2.416e-8 / 1.838e-6 | 通过 | 0 |

四组全部原始断言通过，failed ranks=0，日志名按上述完整命令展开；四个NPU
任务已结束。该结果不能外推到packing/双轴CP/Qwen3-8B，不替代P4双方真实
训练入口或其待确认权重标准。原TP2/EP2/KVCP2恢复失败仍保留。
本轮仅记录变更，不重跑CPU和源码门禁；最近CPU4238passed/13skipped及
全量pre-commit无关失败仍有效。整体P0–P4目标未完成，未切分支或提交。

## 2026-10-23：PP×双轴CP validation独立参考与声明路径审计

已读原始目标及开发指南，main及双方commit仍与前节相同，保留既有改动；
开始时无存活torchrun/pytest。上一轮完成DP复制训练参考，本轮补validation
而非再增加相同训练组合。README声明的`parallel/matrix.py`等路径实际不存在，
已改为当前compatibility/model_compile/topology及memory/trainer实际目录；
仅修正路径和组织说明，没有删除或缩减支持声明来制造通过。

`pp_validation_equivalence.py`新增KV/Ulysses度数和序列长度环境参数，默认
保持原配置；world允许8，DP按world/(PP×KV×Ulysses)计算并检查整除。
实际dual NPU必须断言CP_SDPA_BACKEND、_cp_sdpa_dual及DualCPSdpaKernel，
不单凭配置或无报错认定生效。实际Trainer validation与完整CPU模型参考比较，
参考从实际stage当前权重重建，保持相同microbatch边界及有效token分母。
原绝对误差阈值1e-5不变；验证不改变optimizer/scheduler/既有梯度/token计数，
结束恢复train模式等精确断言保留，所有rank共同裁决。

有限配置：NPU8、PP2/KV2/Ulysses2、DP-shard1、native FSDP、FP32、seq256、
hidden64/intermediate128/vocab128/Q4/KV4，训练2步，各步后验证2批；1F1B
两层/2microbatches/globalbatch4，Interleaved六层/4microbatches/globalbatch8。
非packed，无AC，不外推至其他特性或P4。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256; for schedule in 1F1B Interleaved1F1B; do export LLMTUNER_TEST_PP_SCHEDULE="$schedule"; timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_validation_equivalence.py > "outputs/qwen3-8b-npu/logs/pp_dual_cp_validation_${schedule}_20261023.log" 2>&1; result=$?; echo "$schedule exit=$result"; if [ "$result" != 0 ]; then exit "$result"; fi; done'
```

1F1B exit0，报告及参考validation losses均为
`[4.839190793504902,4.817880667892157]`，maxAE=0，全部原断言通过。
日志 `pp_dual_cp_validation_1F1B_20261023.log`；CPU断言fallback明确保留，
不声称纯AiCore。完整质量门禁及另一调度结果后补。

Interleaved1F1B exit0，报告losses=
`[4.849020565257353,4.837702971813726]`，参考=
`[4.849020565257353,4.837703450520833]`，maxAE=4.787e-7<1e-5。
两组优化器/scheduler/梯度/token计数及train模式恢复等原断言均通过，日志
`pp_dual_cp_validation_Interleaved1F1B_20261023.log`。两次双轴NPU任务均结束。

CPU全套4238passed/13skipped/273warnings，145.84s，exit0，日志
`pp_dual_cp_validation_cpu_20261023.log`。全仓Ruff/format449files、容器
compileall及git diff --check通过。当前源码隔离快照全量pre-commit exit1，
仍仅既有hf_inspection两处tuple-isinstance及karpathy技能缺EOF换行；日志
`pp_dual_cp_validation_full_precommit_20261023.log`。未回写无关修正或豁免。

默认NPU2/PP2/noCP/1F1B回归exit0，日志
`pp_dual_cp_validation_default_regression_20261023.log`；运行命令为容器根目录
`PYTHONPATH=. timeout 240 torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_validation_equivalence.py`。
README、状态文档及受影响脚本的pre-commit全部适用hooks通过，exit0，日志
`pp_dual_cp_validation_target_precommit_20261023.log`。所有本轮NPU/CPU进程
已结束，未提交或切分支。新增有限validation证据不关闭原恢复失败、硬件/compile
或P4缺口，整体目标未完成。

## 2026-10-23：PP×Ulysses×DP分片 validation补验

已读原始目标和开发指南，当前main/双方commit仍与前节相同，未覆盖已有改动；
开始时无存活训练/pytest。上一轮扩展脚本并验证双轴validation属于进展，但其
DP-shard=1不能证明DP分片路径，本轮补独立shard2配置，不改源码。

预定NPU8，PP2/KV1/Ulysses2/DP-shard2、native FSDP、FP32、seq256、
hidden64/intermediate128/vocab128/Q4/KV4，训练2步，各次validation2批。
1F1B两层/2microbatches/globalbatch8，Interleaved六层/4microbatches/
globalbatch16。无packing/AC。脚本要求实际DP-shard=2、FSDPModule及真正
Shard placement/较小local tensor；不是只查看配置。不切分完整CPU模型从
stage实际当前权重重建，按DP microbatch边界和全局有效token分母独立求loss，
绝对误差阈值1e-5；optimizer/scheduler/既有梯度/token计数及train模式保持
原精确断言，所有rank共同裁决。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. LLMTUNER_TEST_WORLD_SIZE=8 LLMTUNER_TEST_CP_SIZE=1 LLMTUNER_TEST_ULYSSES_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256; for schedule in 1F1B Interleaved1F1B; do export LLMTUNER_TEST_PP_SCHEDULE="$schedule"; timeout 360 torchrun --standalone --nproc_per_node=8 tests/integration_tests/pp_validation_equivalence.py > "outputs/qwen3-8b-npu/logs/pp_ulysses_shard_validation_${schedule}_20261023.log" 2>&1; result=$?; echo "$schedule exit=$result"; if [ "$result" != 0 ]; then exit "$result"; fi; done'
```

本轮仅补有限validation配置，不将其外推至AC/packing或P4，不关闭原恢复失败。

| 调度 | 报告loss / 参考loss | 实际maxAE | 状态 | 退出码 |
| --- | --- | --- | --- | --- |
| 1F1B | [4.8463244868259805,4.829609949448529] / [4.8463244868259805,4.829610428155637] | 4.787e-7 | 通过 | 0 |
| Interleaved1F1B | [4.848645258884804,4.839231962316177] / 相同 | 0 | 通过 | 0 |

实际mesh报告pp2/replicate1/shard2/cp2/tp1/ep1，所有原数值和状态保持断言
通过。CPU断言回退已明确报告，不声称纯AiCore。两次NPU任务均已结束。
本轮只更新状态文档，不重跑CPU/源码门禁；最近CPU4238passed/13skipped和
全量pre-commit既有无关失败保持有效。整体P0–P4目标仍未完成。

## 2026-10-23：恢复失败的逐模块输出低扰动捕获

已读原始目标及开发指南，main/双方commit仍为前节版本，保留工作区改动；
开始时无存活训练/pytest。上一轮取得DP分片validation证据，本轮回到未关闭
TP2/EP2/KVCP2/fullAC恢复失败，不把独立组合成功当作该失败修复。

忽略目录`outputs/probe_ac_forward_boundary.py`新增可选
`LLMTUNER_PROBE_OUTPUT_ONLY=1`：step3/4仅捕获各模块原前向post输出，不
clone输入/kwargs，不注册pre钩子，不采集重算；结束后才搬CPU。相比详细探针
减少克隆，但仍扰动时序，不属于正式生产路径。默认原模式不变。
原实际Trainer及精确断言不改，最多三组成对运行，首次训练失败即停止。
配置沿用原NPU4、TP2/EP2/KVCP2、FP32、非packed256小MoE、native FSDP、
dynamic EMA/RNG/loader、确定性模式、连续4步vs独立2+2。

```bash
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && export PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=native LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_PROBE_OUTPUT_ONLY=1; for trial in 1 2 3; do export LLMTUNER_PROBE_TAG="lean${trial}"; for phase in full resume; do timeout 300 torchrun --standalone --nproc_per_node=4 outputs/probe_ac_forward_boundary.py "$phase" "outputs/tp-ep-cp-ac-lean-${trial}-20261023" > "outputs/qwen3-8b-npu/logs/ac_lean_${trial}_${phase}_20261023.log" 2>&1; result=$?; echo "$trial $phase exit=$result"; if [ "$result" != 0 ]; then exit "$result"; fi; done; done'
docker exec hyper-parallel-env bash -lc 'cd /home/jianzhnie/llmtuner/llm/TorchLLMTuner && python outputs/compare_ac_forward_boundaries.py outputs/tp-ep-cp-ac-lean-1-20261023 lean1:full lean1:resume --all'
```

每组比较全部rank的事件及已有指标，比较工具exit0仅表示审计完成，不覆盖实际
训练结果。共享CP kernel仍按同次full/resume相同调用序列比较，不将逆序重算
事件混入。成功捕获不排除其他未捕获的失败，禁止据此自动关闭原失败项。

实际三组full/resume均exit0，三份比较审计也exit0；每rank88个step3/4输出
事件、已有step3/4指标均精确，未捕获新的失败边界。日志
`ac_lean_{1,2,3}_{full,resume}_20261023.log`及
`ac_lean_{1,2,3}_comparison_20261023.log`。原始精确断言保留，但这些是
带钩子的诊断成功，不新增正式支持结论，也不关闭既有metrics-only/forward-pair
失败。已达到本轮三组上限，不继续盲目重复同一探针。

下一有效动作须有不同证据来源，例如对已保存失败输入做局部算子重放或捕获不
引入tensor clone的布局元数据；不因未复现而全局同步、预热或禁用AC绕过。
本轮六个NPU进程组及比较任务均已结束；未改生产源码或正式测试。
诊断脚本Ruff/format、容器compileall及git diff --check通过。仅忽略目录诊断
与本状态文档变更，不重复CPU全套/全量门禁，既有无关门禁失败保持。
整体P0–P4目标仍未完成。
## 2026-10-24：P4 权重验收对象与捕获时点核查

本轮读取真实 Trainer 捕获与 replay 脚本及已有 JSON，未重跑训练，
未新增通过结论。跨框架最终诊断文件
`outputs/p4-production-flex-joint-rope-verified-20261023/rank0/weight-diagnostics.json`
实际含 90 个条目；候选之间的 comparison JSON 所称 91 个 state 条目
属于不同的比较对象，不能作为跨框架关键权重验收证据。

建议待确认的对象为 `model.embed_tokens.weight`、`model.norm.weight`，
以及每个 decoder 层的 `self_attn.{q,k,v,o}_proj.weight`、
`self_attn.{q,k}_norm.weight`、`mlp.{gate,up,down}_proj.weight`、
`input_layernorm.weight` 和 `post_attention_layernorm.weight`。
以上名称来自现有最终诊断；正式清单仍须与双方 named_parameters、
适配后的名称及共享参数身份核对，不以 state_dict 条目数证明参数数目。
共享 lm_head 与 embedding 应核对别名关系并只计一次独立权重。
本清单针对已有 dense debugmodel，不能外推为 Qwen3-30B MoE 专家清单。

建议比较时点为加载初态后、首次 optimizer 更新后及第 100 次更新后。
初态和最终状态已有 initial-hf.pt / final-hf.pt 与候选最终状态捕获路径；
中间状态仅在双方 trace_steps 启用并实际生成 trace-step-N.pt 时存在，
不能从脚本具有捕获分支推断本次运行已保存。两个脚本均在原 train_step
返回后保存 weights_after_update，便于按相同更新边界配对。

当前 replay 的 weight-diagnostics.json 仅计算基线 compare_tensors 指标，
没有权重阈值断言。例如上述 rank0 最终第 0 层 q_proj 的 mean_ae 为
2.5101452957382596e-5、mean_re 为 0.00616788476314266，
embedding 的 mean_ae 为 2.0703183333220087e-7。
这些是诊断观测，不用来事后设线，也不套用 logits 标准。
权重对象、时点及阈值未经确认，P4 权重验收继续 blocked；
仍需逐 rank 核对布局、形状、有限性及参数身份后执行正式断言。

本轮续查实际文件：正式 production-flex-joint-rope-verified 运行及
baseline-trainer-fsdp2-partial-100 的两个 rank 均无 trace-step-1.pt，
因此不能直接完成该次运行首次更新后的权重比较。两个 rank 的初态、
最终基线文件存在且各自 SHA-256 一致；候选最终分片文件均存在，
不同 rank 的文件校验值不同，不据此推断全局参数不同。
两个 weight-diagnostics.json 的 SHA-256 相同，名称集合一致，
均为 8 层、90 个以 .weight 结尾的条目，所有已记录数值指标有限；
这仅核实 JSON 内容，不替代重新检查张量或 named_parameters。
文件存在性、大小、SHA-256、逐 rank 诊断已保存至
outputs/p4-weight-evidence-inventory-20261024.json；本地 Python 审计 exit0。
初态精确检查位于 replay 加载后的 assert_close(atol=0, rtol=0)，
未单独保存候选初态文件。本轮未启动 NPU 任务或修改生产实现。

随后在 node92 的 hyper-parallel-env 中用 torch 读取双方 rank 的基线
initial-hf.pt / final-hf.pt（map_location=cpu、weights_only=True）。
两个 rank 均确认初态、最终状态与候选 weight-diagnostics.json 名称集合
完全一致：90 个张量、31,987,968 个元素；逐项初末形状和 dtype 相同，
初末数值均有限，90 项初末值均发生变化。审计 exit0，原始形状、dtype、
元素数和变化标记保存于
outputs/p4-baseline-weight-tensor-inventory-20261024.json。
这是基线原始张量与诊断覆盖范围的直接核对，尚未验证候选最终分片的
重建值、共享参数身份或权重容差，不改变 P4 权重 blocked 状态。

继续完成候选完整权重重建：node92 容器内 CPU 加载两个 rank 的
candidate-final.pt，逐项要求 DTensor mesh 的 rank 顺序为 [0,1]、
单轴 Shard(0)，再沿第 0 维拼接局部分片；不调用设备 collective。
90 项重建权重与基线最终状态形状和 dtype 一致，数值有限；复用基线
compare_tensors 重算的四项指标逐项精确等于已保存的全局诊断。
两个 rank 的 lm_head 与 embedding 局部值也相等，但值相等不证明
运行时 Parameter 身份共享。审计 exit0，结果保存于
outputs/p4-reconstructed-global-weight-audit-20261024.json。
加载时 DeviceMesh 提示未初始化进程组；本审计只读取 mesh 元数据和
本地 CPU 张量，不执行 compile 或进程组查找。未启动训练。
由此补齐该次最终候选权重重建的直接证据；首次更新快照、参数身份
和权重验收标准仍未补齐，不宣称 P4 通过。

共享身份证据更正：进一步核对实际 replay 与 configured 包装器后，
不能将该次候选身份检查整体列为缺失。replay 在加载初态前明确断言
lm_head.weight is model.embed_tokens.weight；configured 包装器保存
90 个 named_parameters 的身份，并在 replay 成功返回后断言名称与
身份映射不变。该次 verified 日志包含两 rank 的 replay complete 及
最后的 Configured production joint invocation counts，两份 rank 审计
均为八层各 100 次调用，与此前记录训练 exit0 一致。因此已有该次运行
的加载前共享身份及训练前后参数身份稳定证据；加载后没有另行显式
重复共享别名断言，不将保存张量的值相等本身解释为身份相同。
本轮仅更正证据边界，不重跑训练、不新增权重容差通过结论。
