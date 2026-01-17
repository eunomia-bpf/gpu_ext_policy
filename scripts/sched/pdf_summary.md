# Meta sched_ext AI Training PDF 详细内容

## 1. 分布式 AI 训练的核心机制（页面 6-8）

### 关键路径概念
```
CPU → [K1 ... KN S] → Accelerator → [K1 ... KN S] → ...
      ↑                                ↑
   提交命令                        GPU 执行

问题：
- 最慢的 rank 拖慢所有人（同步点）
- Kernel 执行时间因硬件差异、thermal throttling 而不同
- 最快的 rank 在同步时等待
```

### DELAY 的影响（页面 7）
```
如果在关键路径上发生 DELAY（如被抢占）：
→ GPU 变空闲
→ 或等待 straggler 完成同步
```

### 真实 Trace 观察（页面 8）
展示了两种阶段：

**Not latency bound 阶段**：
- GPU 因大量工作而落后
- CPU → GPU 的流程正常

**Latency bound 阶段**：
- GPU 运行很快（可能是模型上层）
- GPU 大部分时间在等 CPU
- **如果 trainer_main 被抢占 → workflow 丢失 QPS**

### 性能影响图表（页面 9）
展示了实际的性能曲线：
- Benchmark free run: ~150,000 QPS
- Start noisy neighbor: 降到 ~120,000 QPS（-20%）
- Start scx_layered: 恢复到 ~145,000 QPS
- Stop scx_layered: 又降回 ~110,000 QPS
- Restart scx_layered: 恢复到 ~145,000 QPS

**结论**：调度器对性能有显著影响！

## 2. 线程池复杂性（页面 10）

Meta 的训练应用有极其复杂的线程池结构：

| 线程池类型 | 数量 | 延迟重要性 | 特点 |
|-----------|------|----------|------|
| PyTorch kernel launching | 最多 24 个 | ⭐⭐⭐⭐⭐ 最高 | Cache locality 优先 |
| Accelerator monitoring | 10s | ⭐⭐⭐⭐ | - |
| Comms monitoring | 10s | ⭐⭐⭐⭐ | - |
| OpenMP tasks | 10s | ⭐⭐⭐ | *可能在关键路径上 |
| Data transforms | 100s | ⭐⭐ | - |
| Data loading | 100s | ⭐⭐ | - |
| Model checkpointing | 10s | ⭐ | - |
| RPC | 1000+ | ⭐ | - |
| Routing | 1000+ | ⭐ | - |

**挑战**：
1. 如何在不饿死低优先级任务的情况下优先处理 latency critical 任务？
2. 如何限制不合理大的线程池？
3. 如何处理 NUMA balancing？

## 3. sched_ext 的目标（页面 11）

Meta 希望调度器能够：
- ✅ 使用更多 CPU 周期做有用的工作（减少远程数据处理）
- ✅ **真正优先处理** latency critical 任务
- ✅ 有一个按重要性排序任务类别的机制
- ✅ 保护 latency critical 任务不受 CPU 频率降低的影响
- ✅ 为 latency critical 任务提供优先的内存带宽访问
- ✅ 有短的开发和部署周期
- ✅ 足够健壮，能部署在有数百种 corner cases 的集群上

## 4. scx_layered 设计（页面 12）

将任务分成层级，每层有不同的属性和策略：

**Layer 1 - Latency critical**：
- 任务：latency critical + comms monitoring
- 策略：减少抢占、运行在高频核心、保持 cache locality

**Layer 2 - Data pipeline**：
- 任务：data pipeline tasks
- 策略：可变 CPU 数量、可以运行在低频核心、限制内存带宽使用

**Layer 3 - Supporting frameworks**：
- 任务：supporting frameworks tasks
- 策略：限制到固定数量的 CPU，但足够避免 stall

**Layer 4 - The rest**：
- 任务：其他一切
- 策略：使用剩余资源

## 5. 部署策略（页面 15-17）

### AI Training Job Attempt 概念（页面 15）
```
Host 0: [Job 0 Attempt 0] → [Job 2 Attempt 0] → [Job 0 Attempt 1] → [Job 0 Attempt 2 (w/ sched_ext)]
Host 1: [Job 0 Attempt 0] → [Job 2 Attempt 0] → [Job 0 Attempt 1] → [Job 0 Attempt 2 (w/ sched_ext)]
...

可以在预定义的 attempt 上启用 sched_ext，进行 apples-to-apples 对比
```

### 主机组件（页面 16）

**Data Processing hosts**：
- Data Processing workers

**Training hosts**：
- Training container (data loader, trainer framework, checkpoint agent)
- Accelerator telemetry daemon
- Scheduler manager (管理 scx_layered 生命周期)
- Linux kernel (bpf + sched_ext)

**AI Job Scheduler (control plane)**：
- 协调训练任务的生命周期

### 部署旅程（页面 17）

**阶段 1 - Experiments**：
- 验证 sched_ext 是否有意义
- 识别尽可能多的 corner cases
- 确定哪些训练应用受益最多

**阶段 2 - Build Fine Grained Deployment**：
- Opt-in / Opt-out 机制
- 快速启用/禁用
- 按 job、model、训练范式、产品组、团队等部署

**阶段 3 - Build Observability**：
- 监控失败和告警
- 监控性能
- 量化基础设施成本节省

**阶段 4 - Target Use Cases / Small Scale Deployments**：
- 高度异构的训练（中小型内容理解模型）
- On-box data pipelines，减少对远程 data pipelines 的需求

**阶段 5 - Fleet Level Slow Roll-Out**：
- 慢慢启用 sched_ext
- 观察行为
- 继续直到完全部署

## 6. 早期实验发现（页面 18）

### 基础问题
- ✅ 如何识别 latency critical 任务？
- ✅ 如何在保护重要任务的同时不拖慢 data pipeline？
- ✅ CPU 频率随 CPU 利用率下降
- ✅ 在更高负载下，LLC miss 因内存带宽压力而变得更昂贵
- ✅ NUMA balancing 很困难
- ✅ **没有工具可以帮助非专家分析调度器效率**

### Corner Cases
- 我们的 "latency critical" 任务过于频繁地 yield()
- 大量任务依赖（Python GIL、data transforms → data loaders）
- 线程池被复制到各个应用，使用默认设置创建数十万任务
- **任务被固定到系统中的特定核心**
- 用户为数据预处理、加载、模型动态编译等启动了太多任务
- 创建了太多调度计算 kernel 的任务
- **IRQ 抢占我们的重要任务**

## 7. 性能度量（页面 19）

### 代理指标
- 没有通用的应用性能指标
- **硬件计数器**（SM Utilization & Tensor Core Active %）与应用性能相关性好
- 构建 A/A vs. A/B 聚合帮助推理集群范围的性能

### 实际结果图表
展示了 A/A 和 A/B 的对比：
- 横轴：window_qps
- 纵轴：qflops
- 显示 layered_0、layered_1、EEVDF_0、EEVDF_1 的分布

右侧直方图显示：
- scx / non-scx count vs non-scx / non-scx count
- 峰值在 1.0 附近，说明大部分情况性能相当或更好

## 8. 生产效果（页面 20）

### 吞吐量改进 CDF
- X 轴：Ops Diff（性能差异）
- Y 轴：Percentile
- **显著更多的 jobs 显示吞吐量改进**
- **没有病理性退化的迹象**
- **节省了约 4% 的 GPU 容量**（数万 GPU 的集群）
- **目标节省 ~30% 的远程数据处理容量**

## 9. 识别关键路径任务（页面 22）

### 问题
- 哪些任务是 latency critical？
- 如果它们在作业开始后一段时间才创建怎么办？
- 如果它们在执行过程中改变怎么办？
- 我们可以用什么来识别它们？

### 解决方案（Nvidia 特定）
- **Kprobe** Nvidia drivers 中的 mmap、poll、open 以获取实时 PID 到 GPU 映射
- 查询 **NVML** 以获取 PID 到 GPU 映射
- 从 NVML 获取 GPU 到 NUMA 节点映射
- **启发式方法**（CommPrefix、Cgroup Name、Group Leader status 等）

### 潜在改进
- ✅ **标准化方法**：让 accelerator vendors 报告哪些任务在 accelerators 上调度工作
  - 系统可以使用这些信息优先处理这些任务（它们很可能是 latency critical）
- ✅ **更多 tracepoints**：在 accelerator stack 中，允许我们识别：
  - Context creation
  - Async copies
  - Kernel launches
  - 等等

## 10. 任务管理问题和解决方案（页面 23）

### 问题 1：频繁 yield()
- **问题**：Latency critical 线程频繁 yield()
- **解决**：忽略高优先级层任务的一定百分比的 yield()

### 问题 2：Cache Locality
- **问题**：Latency critical 任务维护关于 tensors 的持久元数据结构，需要 locality
- **解决**：实现 "prev over idle policy"，选择任务之前所在的 CPU

### 问题 3：NUMA Sensitivity
- **问题**：应用对跨 NUMA 节点的错误任务放置非常敏感
- **解决**：在 scx_layered 中实现平衡策略，查询 Nvidia 库获取当前 NUMA 放置

### 问题 4：自定义核心绑定
- **问题**：应用使用自定义核心绑定，如果其他任务使用"专用" CPU 会增加 stall 风险
- **解决**：除了移除自定义 affinities，没有太多办法 :(

## 11. 潜在改进（页面 24）

### 讨论 1：内存迁移
- 希望能够在将任务移动到其使用的 GPU 本地时**移动已分配的内存**
- 理想情况下从 bpf，至少不需要修改 workload 代码

### 讨论 2：内存限制
- 希望能够将任务分配的内存**限制到当前 GPU 本地的内存**
- 帮助进一步减少跨 NUMA 节点流量

## 总结：Meta 的关键经验

1. **4% GPU 容量节省**在数万 GPU 集群上是巨大的成就
2. **识别 latency critical 任务**是最大挑战（需要多种方法组合）
3. **Corner cases 很多**：yield()、NUMA、IRQ、线程池等
4. **需要工具帮助非专家分析调度效率**（我们的工具！）
5. **标准化接口很重要**：GPU vendor 应该提供标准方式报告关键任务
6. **内存管理**是下一个优化方向（BPF 支持内存迁移）

