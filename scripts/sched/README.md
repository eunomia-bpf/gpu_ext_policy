# CUDA + CPU 调度器影响分析

## 概述

分析 CPU 调度器行为对 GPU 工作负载性能的影响，识别调度瓶颈并量化性能损失。

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

## 能分析出什么信息？

### 1. **Kernel Launch 延迟** ⭐ 最重要

**含义**：进程从被调度回 CPU 到提交 GPU kernel 的时间

**为什么重要**：
- 反映了 CPU 调度器对 GPU 提交的**直接影响**
- 延迟越大，GPU 空闲时间越长
- 这是调度器影响 GPU 性能的**主要途径**

**典型值**：
- 优秀：< 50µs
- 一般：50-200µs
- 差：> 200µs

**示例场景**：
```
时间线：
T0: GPU kernel 完成
T1: 进程被调度出 CPU (OFF-CPU)
T2: 进程被调度回 CPU (ON-CPU)  <-- 调度延迟
T3: 提交下一个 kernel              <-- Launch 延迟 = T3-T2

影响：GPU 从 T0 到 T3 处于空闲状态
```

### 2. **Sync 期间的调度行为**

**能看到的信息**：
- `cudaDeviceSynchronize()` 调用期间进程的调度状态
- OFF-CPU 时间占比
- 上下文切换次数

**为什么重要**：
- Sync 期间进程被调度出去 → GPU 完成后无法立即响应
- 理想情况：进程应该 spin-wait 或高效 yield

**示例**：
```
T0: cudaDeviceSynchronize() 进入
T1: 进程 OFF-CPU (等待中被抢占)
T2: GPU 实际完成 (但进程仍在 OFF-CPU!)
T3: 进程 ON-CPU
T4: Sync 返回

损失：T2 到 T3 的不必要延迟
```

### 3. **上下文切换频率**

**含义**：GPU 进程每秒被切换的次数

**为什么重要**：
- 每次切换有固定开销 (~1-10µs)
- 高频切换会累积大量开销
- 表明进程与其他任务竞争激烈

**典型值**：
- 优秀：< 50 Hz
- 一般：50-200 Hz
- 差：> 200 Hz

### 4. **整体 OFF-CPU 占比**

**含义**：进程在整个运行期间被调度出 CPU 的时间百分比

**注意事项**：
- 如果程序有 sleep/IO，高 OFF-CPU 是正常的
- **关键看 Launch 延迟**，而不是单看 OFF-CPU 占比
- 对于纯计算 GPU 负载，>50% OFF-CPU 才是问题

### 5. **调度模式识别**

通过事件序列可以识别：

**模式 A - 频繁抢占**：
```
Launch → OFF-CPU (5ms) → ON-CPU → Launch → OFF-CPU (8ms) → ...
```
→ 说明 CPU 负载高，进程被频繁抢占

**模式 B - 批量提交**：
```
Launch → Launch → Launch → OFF-CPU → ...
```
→ 说明 GPU 任务提交效率高

**模式 C - Sync 阻塞**：
```
Launch → SyncEnter → OFF-CPU (多次) → SyncExit
```
→ 说明 Sync 期间被频繁调度

---

## 核心洞察

### 💡 Launch 延迟 > OFF-CPU 占比

**为什么？**
- GPU 性能取决于**何时**提交任务，而不是进程总共运行了多久
- Launch 延迟 10µs vs 200µs：GPU 空闲时间相差 190µs
- OFF-CPU 50% vs 80%：如果 Launch 延迟一样，GPU 性能可能相同

**实际案例**：
```
场景 A：OFF-CPU 20%, Launch 延迟 150µs → GPU 利用率 70%
场景 B：OFF-CPU 60%, Launch 延迟 15µs  → GPU 利用率 95%

结论：场景 B 虽然 OFF-CPU 更高，但 GPU 性能更好！
```

### 💡 Sync 行为反映调度策略

**观察到的模式**：

1. **Spin-wait**：Sync 期间持续 ON-CPU
   - 优点：响应快
   - 缺点：占用 CPU

2. **Yield + 被抢占**：Sync 期间频繁 OFF-CPU
   - 优点：释放 CPU
   - 缺点：GPU 完成后响应慢

3. **混合**：短时间 spin，然后 yield
   - CUDA 默认行为（取决于 `cudaDeviceSchedule` 设置）

### 💡 调度器影响 GPU 的三个层次

**层次 1 - 任务提交延迟**（最直接）
```
调度延迟 → Launch 延迟增大 → GPU 空闲增加
```

**层次 2 - Sync 响应延迟**（中等）
```
Sync 期间被抢占 → GPU 完成后无法立即响应 → 端到端延迟增加
```

**层次 3 - 上下文切换开销**（累积）
```
高频切换 → 开销累积 → 总体性能下降
```

---

## 指标解读

### 调度影响评分 (0-100)

**计算公式**：
```
score = min(100, off_cpu_ratio * 100 + switch_freq_hz * 2)
```

**分级**：
- 0-20: 🟢 低影响 - 调度器干扰最小
- 20-50: 🟡 中等影响 - 可接受，有优化空间
- 50-100: 🔴 高影响 - 显著性能损失

**注意**：需结合 Launch 延迟判断是否为真问题

---

## 优化策略

### 策略 1: CPU 绑核

```bash
# 将 GPU 进程绑定到特定核心
taskset -c 0-3 ./your_gpu_app
```
**效果**：减少缓存失效，降低迁移开销

### 策略 2: 提高优先级

```bash
# 提高调度优先级
nice -n -10 ./your_gpu_app  # 需要权限
```
**效果**：减少被抢占次数

### 策略 3: CUDA 级别优化

```cpp
// 使用异步流避免阻塞
cudaStreamCreate(&stream);
kernel<<<grid, block, 0, stream>>>(...);
// 不用 cudaDeviceSynchronize，而是用 event
```

### 策略 4: 批量提交

```cpp
// 差的方式
for (int i = 0; i < N; i++) {
    kernel<<<1, 256>>>(...);  // N 次 launch 开销
}

// 好的方式
kernel<<<N, 256>>>(...);      // 1 次 launch 开销
```

---

## 实际案例

### 案例：测试程序分析

**程序**：5 次 kernel launch，每次间隔 100ms sleep

**结果**：
```
Launch 延迟：平均 12µs (最大 13.81µs)
OFF-CPU 占比：99.9%
上下文切换：11 次 (21.96 Hz)
评分：100/100 [高]
```

**解读**：
- ✅ Launch 延迟优秀 (< 15µs) → **调度器不影响 GPU 提交**
- ⚠️ OFF-CPU 极高但**预期内**（程序主动 sleep）
- ✅ 切换频率低
- 结论：虽然评分高，但**不是调度问题**，是程序设计就是 sleep

**关键判断依据**：Launch 延迟小 → 调度器工作良好

### 案例：LLM 推理优化（假设）

**优化前**：
```
Launch 延迟: 125µs
上下文切换: 180 Hz
Sync OFF-CPU: 60%
评分: 87/100
```

**优化措施**：
1. CPU 绑核 → Launch 延迟降到 35µs
2. 批量 token 处理 → 切换频率降到 90 Hz
3. 用 CUDA event 代替 sync → OFF-CPU 降到 20%

**优化后**：
```
Launch 延迟: 35µs (降低 72%)
上下文切换: 90 Hz (降低 50%)
评分: 42/100
推理延迟: 120ms → 85ms (快 29%)
```

---

## 追踪数据格式

CSV 关键字段：

| 字段 | 说明 |
|------|------|
| `timestamp_ns` | 相对时间戳（纳秒） |
| `event_type` | 事件类型：`cuLaunchKernel`, `cudaLaunchKernel`, `syncEnter`, `syncExit`, `schedSwitch` |
| `last_offcpu_ns` | 上次 OFF-CPU 的时间戳（0=当前是 ON-CPU 事件） |
| `last_oncpu_ns` | 上次 ON-CPU 的时间戳（0=当前是 OFF-CPU 事件） |

---

## 局限性

1. **eBPF 开销**：追踪本身有 1-5% 开销
2. **仅追踪 CUDA**：不支持其他 GPU API (OpenCL, Vulkan)
3. **Driver/Runtime 重复**：可能看到同一个 launch 的两个事件
4. **需要权限**：必须 sudo 运行

---

## 总结

### 🎯 核心原则

1. **Launch 延迟是关键指标** - 直接影响 GPU 利用率
2. **OFF-CPU 占比需结合场景** - 有 sleep/IO 时高 OFF-CPU 是正常的
3. **Sync 行为反映调度策略** - 可以看到 spin-wait vs yield 的权衡

### 📊 能回答的问题

- ✅ CPU 调度器是否延迟了 GPU 任务提交？
- ✅ Sync 期间进程的调度状态如何？
- ✅ 上下文切换是否过于频繁？
- ✅ 绑核/提高优先级是否有效？
- ❌ GPU 内部执行瓶颈（需要 Nsight）
- ❌ PCIe 传输瓶颈（需要其他工具）

### 🔧 优化决策树

```
Launch 延迟 > 100µs?
├─ 是 → CPU 绑核 + 提高优先级
└─ 否 → 调度器影响小
    ├─ 上下文切换 > 100 Hz?
    │  └─ 是 → 批量操作，减少 syscall
    └─ Sync OFF-CPU > 50%?
       └─ 是 → 用异步 stream + event
```

---

## 深入思考：还缺什么信息？

### 当前能看到的 ✅
1. Launch 延迟（ON-CPU 到 launch 的时间）
2. Sync 期间的 OFF-CPU 占比
3. 上下文切换频率和时机

### 缺失的关键信息 ❌

#### 1. **GPU 实际执行时间**
**为什么重要**：无法判断 GPU 是否真的被 CPU 延迟阻塞了
```
Launch 延迟 50µs + GPU 执行 0.1ms → 影响 0.05%（几乎无影响）
Launch 延迟 50µs + GPU 执行 60µs → 影响 45%（严重影响）
```
**如何获取**：需要 CUDA event 或 Nsight 追踪

#### 2. **CPU 侧的具体工作**
**为什么重要**：ON-CPU 期间在做什么？准备数据？还是其他计算？
```
ON-CPU 20ms → 是在准备下一个 kernel 的数据（必要）
ON-CPU 20ms → 在处理无关的 CPU 计算（可优化）
```
**如何获取**：需要用户态函数追踪（uprobe 更多函数）

#### 3. **数据传输时间**
**为什么重要**：`cudaMemcpy` 也会受调度影响
```
Timeline:
memcpy 开始 → OFF-CPU → ON-CPU (延迟!) → memcpy 继续
```
**如何获取**：hook `cudaMemcpy`/`cuMemcpy` 系列函数

#### 4. **抢占来源**
**为什么重要**：知道是谁抢占了 GPU 进程才能针对性优化
```
被 System daemon 抢占 → 可能无法避免
被同优先级用户进程抢占 → 可以调整优先级
被其他 GPU 进程抢占 → 可能需要资源隔离
```
**如何获取**：记录 `sched_switch` 的 `next` 进程信息

#### 5. **CUDA Runtime 开销**
**为什么重要**：区分真实的调度延迟 vs CUDA 自身开销
```
总 Launch 延迟 100µs = 调度延迟 20µs + CUDA 开销 80µs
→ 优化调度器只能改善 20%
```
**如何获取**：需要更细粒度的 CUDA 内部追踪

---

## 深入思考：什么才是真正可优化的？

### 🎯 本质问题
**CPU 调度器影响 GPU 性能的唯一机制**：
```
延迟 GPU 任务的提交时机
```

其他一切（OFF-CPU 占比、切换频率）都是**现象**，不是根因。

### 三类"优化"

#### 类型 A：真正的调度器优化
**手段**：
- CPU 绑核 (`taskset`)
- 提高优先级 (`nice`, `chrt`)
- 隔离 CPU (`isolcpus`)

**效果**：减少 Launch 延迟
**局限**：只能优化几十到几百微秒

#### 类型 B：减少对调度器的依赖
**手段**：
- 批量提交（减少 launch 次数）
- 异步 stream（不等待 GPU）
- Pipeline（CPU/GPU 并行）

**效果**：让调度延迟变得不重要
**本质**：不是优化调度器，是绕过调度器

#### 类型 C：改变 CUDA 行为
**手段**：
```cpp
cudaSetDeviceFlags(cudaDeviceScheduleYield);     // 主动 yield
cudaSetDeviceFlags(cudaDeviceScheduleSpin);      // Busy-wait
cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync); // 阻塞等待
```

**效果**：改变 Sync 期间的调度行为
**权衡**：CPU 占用 vs 响应速度

### 💡 核心洞察

**90% 的"调度问题"其实是应用设计问题**：
```
❌ 错误思路：优化调度器让 OFF-CPU 从 50% 降到 30%
✅ 正确思路：重新设计应用让 GPU 不依赖 CPU 实时响应

示例：
Sequential:  Launch → Sync → [OFF-CPU] → Launch → Sync
Pipeline:    Launch → Launch → Launch → (后台 Sync)
             ↑ GPU 满载，不关心 CPU 何时调度
```

**真正值得优化调度器的场景**：
1. **实时性要求高**：低延迟推理（<10ms）
2. **GPU kernel 很短**：<100µs，调度延迟占比大
3. **无法改变应用**：第三方库，无法重构

其他场景：**优先考虑应用层优化**（类型 B），成本更低，效果更好。

---

**工具位置**：
- 追踪工具：`tools/cuda_sched_trace`
- 分析脚本：`scripts/sched/analyze_gpu_scheduler_impact.py`
- 文档：`scripts/sched/README.md`
