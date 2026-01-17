# CUDA + CPU 调度器影响分析

## 概述

分析 CPU 调度器对 GPU 工作负载的实际影响，**区分调度问题与应用问题**，量化优化价值。

---

## 快速开始

### 1. 采集追踪数据

```bash
cd tools
sudo ./cuda_sched_trace > trace.csv 2> trace.log
# 在另一个终端运行你的 GPU 程序
# Ctrl-C 停止追踪
```

### 2. 分析数据

```bash
cd scripts/sched
python3 analyze_gpu_scheduler_impact.py ../../tools/trace.csv -o report.md
```

---

## 核心分析方法

### 🎯 关键问题

**不是看"launch delay 有多大"，而是看"context switch 是否造成了额外延迟"**

### 正确的分析流程

#### 1. **对比有无 Context Switch 的 Launch Pairs**

将连续的 launch pairs 分为两组：
- **Group A**: Launch 之间无 context switch（正常流程）
- **Group B**: Launch 之间有 context switch（被打断）

对比两组的 interval 分布：
```
Group A: P50=2µs, P90=4µs    (连续提交)
Group B: P50=15ms, P90=15ms  (被抢占)

Preemption Penalty = 15ms - 2µs = 14.998ms
```

**如果 Group B 显著慢于 Group A → 调度器有影响**

#### 2. **Tail Latency 归因分析**

找出 P95-P99 的慢 launch pairs，检查其中有多少伴随 context switch：

```
P95+ 慢 pairs: 2580 个
其中有 switch 的: 62 个 (2.4%)

结论：97.6% 的 tail latency 是应用特性，只有 2.4% 是调度造成的
```

**如果 tail latency 中大部分无 switch → 主要是应用问题，非调度**

#### 3. **Burst 模式识别**

识别批量提交模式（多个 launch 在短时间内连续提交）：

```
Burst 定义: N 个 launch 间隔 <100µs

分析结果：
- 54 个 burst，每个约 950 个 launch
- Burst 内部被打断: 11%
- Burst 之间间隔: 数十 ms

结论：应用是批量提交模式，优化 burst 间隔更重要
```

#### 4. **量化优化价值**

```
调度器影响 = 被打断的 pairs 数量 × Preemption Penalty
           = 62 × 15ms = 0.93 秒

占总运行时间: 0.93 / 79.5 = 1.2%

优化上限: 绑核/提高优先级最多省 1.2%
```

---

## 能分析出什么信息？

### 1. **调度抢占比例** ⭐ 最重要

**指标**: Launch pairs 中被 context switch 打断的百分比

**典型值**：
- 优秀：< 1% (几乎不被打断)
- 一般：1-5%
- 差：> 10% (频繁抢占)

**qwen3 实例**：
- 总 pairs: 51463
- 被打断: 62 (0.1%)
- **结论**: 调度器工作良好

### 2. **Preemption Penalty** ⭐ 影响大小

**指标**: 有 switch vs 无 switch 的 interval 差值

**典型值**：
- 轻微：< 1ms
- 中等：1-10ms
- 严重：> 10ms

**qwen3 实例**：
- 无 switch: P50 = 2µs
- 有 switch: P50 = 15.3ms
- **Penalty = 15ms (严重！)**

**关键洞察**：
- OFF-CPU 本身只有 28µs (P50)
- 但被抢占后等待重新调度很长
- 说明系统有其他高优先级任务

### 3. **Tail Latency 根因**

**指标**: P95-P99 outliers 中有 switch 的比例

**判断标准**：
- > 80%: 主要是调度问题 → 绑核/优先级有效
- 20-80%: 混合问题
- < 20%: 主要是应用问题 → 优化应用更重要

**qwen3 实例**：
- P95+ outliers: 2580 个
- 其中有 switch: 62 个 (2.4%)
- **结论**: 97.6% 的慢 launch 是应用特性，非调度

### 4. **Burst 提交模式**

**能看到的信息**：
- Burst 数量和大小
- Burst 内部被打断的概率
- Burst 之间的间隔分布

**qwen3 实例**：
- 54 个 burst，每个 ~950 launches
- Burst 内被打断: 11%
- **结论**: 批量提交模式，主要瓶颈在 burst 间隔

---

## 误区与正确理解

### ❌ 常见误区

#### 误区 1: "Launch Delay 很大，所以调度有问题"
```
错误分析：
平均 launch delay = 4.8ms → 延迟很大！

问题：这包含了 CPU 计算数据的时间，不是调度延迟
```

#### 误区 2: "OFF-CPU 占比高，调度器影响大"
```
错误分析：
OFF-CPU 占比 60% → 调度器抢占太多！

问题：GPU workload 本来就应该低 CPU 使用率
```

#### 误区 3: "Context Switch 频率高就是问题"
```
错误分析：
592 次 switch，7.44 Hz → 频繁切换！

问题：要看这些 switch 是否在关键路径上（launch 之间）
```

### ✅ 正确理解

#### 正确分析 1: 对比分组
```
无 switch: P50 = 2µs
有 switch: P50 = 15ms

结论：调度器造成 15ms penalty，但只影响 0.1% 的 pairs
```

#### 正确分析 2: 归因 Tail Latency
```
P99+ 的 515 个慢 pairs:
- 有 switch: 62 个 (12%)
- 无 switch: 453 个 (88%)

结论：88% 的 tail latency 是应用本身，调度优化效果有限
```

#### 正确分析 3: 识别模式
```
发现 54 个 burst，每个 950 launches
Burst 间隔远大于 burst 内部

结论：应用是批量模式，优化 burst 间的 CPU 计算更重要
```

---

## 什么是真正可优化的？

### ✅ 可优化（调度器层面）

#### 场景 1: 高抢占比例 + 高 Penalty
```
被打断: 10% 的 pairs
Penalty: 10ms
影响: 10% × 10ms × N pairs = 显著

优化措施：
- CPU 绑核 (taskset -c 0-3 ./app)
- 提高优先级 (nice -n -10)
- CPU 隔离 (isolcpus)
```

#### 场景 2: Tail Latency 主要由 Switch 造成
```
P95+ outliers: 80% 有 context switch

优化措施：
- 实时优先级 (chrt -f 50)
- CFS 调优 (减小 sched_min_granularity_ns)
```

#### 场景 3: Burst 内部被频繁打断
```
Burst 内抢占: 50%

优化措施：
- CPU 绑核 + 隔离其他任务
- 减少同时运行的进程
```

### ❌ 不可优化（应用问题）

#### 场景 1: 无 Switch 但 Interval 大
```
Group A (无 switch): P50 = 5ms
Group B (有 switch): P50 = 5.5ms

结论：主要是 CPU 计算慢，非调度

应优化：
- CPU 侧的数据准备
- 内存分配策略
- 算法优化
```

#### 场景 2: Tail Latency 无 Switch
```
P99+ outliers: 90% 无 context switch

结论：应用偶尔有长计算

应优化：
- 找到慢的 launch 对应的 kernel
- 优化 CPU 预处理逻辑
```

#### 场景 3: Burst 间隔主导
```
Burst 内: 2µs
Burst 间: 20ms (占总时间 90%)

结论：优化 burst 之间的逻辑

应优化：
- Pipeline CPU/GPU 工作
- 减少 sync 操作
- 增加 batch size
```

### 🤔 无法判断（数据缺失）

#### 缺失 1: GPU 实际执行时间
```
无法判断：
- Launch delay 50µs + GPU 执行 100ms → 影响 0.05% (无所谓)
- Launch delay 50µs + GPU 执行 60µs → 影响 45% (严重！)

需要：
- CUDA event timing
- Nsight 追踪
```

#### 缺失 2: 抢占来源
```
无法判断：
- 被系统 daemon 抢占 → 可能无法避免
- 被同优先级用户进程抢占 → 可以调优

需要：
- 记录 sched_switch 的 next 进程
```

#### 缺失 3: 内存传输影响
```
无法判断：
- cudaMemcpy 是否也被抢占？
- PCIe 是否瓶颈？

需要：
- Hook cudaMemcpy 系列函数
```

---

## 实际案例：qwen3.cu 分析

### 程序特征
- Qwen3 0.6B LLM 推理
- 单次推理生成约 30 tokens
- 每个 token 多个 transformer layers

### 追踪数据
```bash
sudo ./cuda_sched_trace > qwen3_trace.csv 2> qwen3.log &
./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1
# Output: 56 tok/s
pkill -f cuda_sched_trace
```

### 分析结果

**基础指标**：
- 总运行时间: 79.5 秒
- Kernel launches: 51,464 次
- Context switches: 592 次 (7.44 Hz)
- OFF-CPU 时间: 7.88 ms (0.01%)

**Launch Pair 分析**：
```
总 pairs: 51,463
被打断 (有 switch): 62 (0.1%)
未打断 (无 switch): 51,401 (99.9%)

Interval 分布：
- 无 switch: P50=2µs, P90=4µs, P99=4µs, Max=17.7s
- 有 switch: P50=15.3ms, P90=15.5ms, P99=5.0s, Max=12.7s

Preemption Penalty: 15.3ms (中位数)
```

**Tail Latency 归因**：
```
P95+ 慢 pairs: 2,580 个
  其中有 switch: 62 个 (2.4%)

P99+ 慢 pairs: 515 个
  其中有 switch: 62 个 (12.0%)

结论：97.6% 的 P95 tail latency 是应用特性
```

**Burst 模式**：
```
Burst 数量: 54 个
Burst 大小: 948-958 launches/burst (平均 952)
Burst 内被打断: 6/54 (11.1%)

模式识别：
每个 burst ≈ 1 个 token 的所有 layers
Burst 间隔 ≈ CPU 准备下一个 token 的数据
```

### 结论

**调度器影响**：
- 影响范围: 0.1% 的 pairs
- Penalty: 15ms/次
- 总影响: 62 × 15ms ≈ 0.93 秒
- 占比: 0.93 / 79.5 = **1.2%**

**优化建议**：
1. ✅ **不值得优化调度器** (只能省 1.2%)
2. ✅ **优先优化应用层**:
   - Burst 间隔占总时间 >50%
   - 优化 CPU 侧的 token 准备逻辑
   - Pipeline CPU/GPU 工作
3. ⚠️ **如果追求极致**:
   - CPU 绑核可能省 0.9 秒
   - 提高优先级避免被抢占

---

## 追踪数据格式

CSV 字段：

| 字段 | 说明 |
|------|------|
| `timestamp_ns` | 相对时间戳（纳秒） |
| `event_type` | `cuLaunchKernel`, `cudaLaunchKernel`, `syncEnter`, `syncExit`, `schedSwitch` |
| `pid`, `tid` | 进程/线程 ID |
| `comm` | 进程名 |
| `cpu` | CPU 核心 |
| `grid_x/y/z` | CUDA grid 维度 (launch 事件) |
| `block_x/y/z` | CUDA block 维度 (launch 事件) |
| `shared_mem` | Shared memory 大小 (launch 事件) |
| `stream` | CUDA stream 指针 (launch 事件) |
| `last_offcpu_ns` | 上次 OFF-CPU 的时间戳（0 = 当前 ON-CPU） |
| `last_oncpu_ns` | 上次 ON-CPU 的时间戳（0 = 当前 OFF-CPU） |

---

## 局限性

1. **eBPF 开销**：追踪本身有 1-5% 开销
2. **仅追踪 CUDA**：不支持其他 GPU API
3. **缺少 GPU 侧数据**：不知道 kernel 实际执行时间
4. **缺少抢占来源**：不知道是谁抢占了 GPU 进程
5. **需要权限**：必须 sudo 运行

---

## 总结

### 🎯 核心原则

1. **对比分组，而非绝对值**
   - 有 switch vs 无 switch 的差异才是调度影响

2. **归因 tail latency**
   - 多少 outlier 是调度造成的？

3. **识别提交模式**
   - Burst 模式下，burst 间隔通常更重要

4. **量化优化价值**
   - 调度器优化通常只有 1-5% 收益
   - 应用层优化才是大头

### 📊 能回答的问题

- ✅ 调度器是否打断了 GPU 提交流程？
- ✅ 被打断的影响有多大（Penalty）？
- ✅ Tail latency 有多少是调度造成的？
- ✅ 应用是批量提交还是单个提交？
- ✅ 优化调度器的价值有多大？
- ❌ GPU 内部执行瓶颈（需要 Nsight）
- ❌ PCIe 传输瓶颈（需要 memcpy 追踪）

---

**工具位置**：
- 追踪工具：`tools/cuda_sched_trace`
- 旧分析脚本：`scripts/sched/analyze_gpu_scheduler_impact.py`（基础统计）
- 新分析脚本：`scripts/sched/analyze_preemption_impact.py`（深度分析）
- 文档：`scripts/sched/README.md`

**参考资料**：
- Meta LPC 2025: Accelerating AI Training with sched_ext
  - 本地文件：`scripts/sched/sched_ext_ai.pdf`
  - 在线链接：https://lpc.events/event/19/contributions/2039/
