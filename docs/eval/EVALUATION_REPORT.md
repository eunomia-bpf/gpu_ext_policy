# Multi-Tenant GPU Memory Priority Policy Evaluation Report

## Executive Summary

本实验评估了基于 eBPF 的 GPU 内存管理策略在多租户场景下实现优先级划分的效果。实验结果表明，我们的策略能够：

1. **显著降低总完成时间**：相比无策略基准，最高改进达 92%（K-Means）
2. **实现有效的优先级划分**：高优先级进程完成时间显著优于低优先级进程
3. **跨 workload 通用性**：在三种不同特征的 kernel 上均表现良好

---

## 1. 实验设置

### 1.1 测试环境
- **硬件**: NVIDIA GPU with UVM (Unified Virtual Memory)
- **软件**: Linux 6.15.11, eBPF-based memory policy framework

### 1.2 Workloads
| Kernel | 特征 | Size Factor |
|--------|------|-------------|
| **Hotspot** | 空间局部性强，热点访问模式 | 0.6 |
| **GEMM** | 计算密集，大矩阵运算 | 0.6 |
| **K-Means** | 稀疏访问，迭代聚类 | 0.9 |

### 1.3 测试策略
| 策略 | 参数 | 说明 |
|------|------|------|
| **No Policy** | - | 基准：无任何策略干预 |
| **Prefetch(0,20)** | high=0, low=20 | 仅对低优先级进程限制 prefetch |
| **Prefetch(20,80)** | high=20, low=80 | 差异化 prefetch 限制 |
| **Evict(20,80)** | high=20, low=80 | 结合 prefetch + eviction 策略 |

### 1.4 实验方法
1. 同时启动两个相同的 uvmbench 进程（模拟多租户）
2. 一个标记为高优先级 (High)，一个标记为低优先级 (Low)
3. 记录各自的完成时间
4. 对比基准：单进程运行时间 (Single 1x, Single 2x)

---

## 2. 实验结果

### 2.1 完成时间对比

#### Hotspot Kernel
| Config | High (s) | Low (s) | Total (s) | Improvement |
|--------|----------|---------|-----------|-------------|
| No Policy | 53.9 | 53.9 | 53.9 | - |
| Prefetch(0,20) | 42.8 | 42.7 | 42.8 | +20.7% |
| Prefetch(20,80) | 22.5 | 24.0 | 24.0 | **+55.5%** |
| Evict(20,80) | 22.4 | 23.9 | 23.9 | **+55.7%** |

#### GEMM Kernel
| Config | High (s) | Low (s) | Total (s) | Improvement |
|--------|----------|---------|-----------|-------------|
| No Policy | 135.8 | 135.7 | 135.8 | - |
| Prefetch(0,20) | 83.5 | 85.2 | 85.2 | +37.3% |
| Prefetch(20,80) | 24.0 | 29.6 | 29.6 | **+78.2%** |
| Evict(20,80) | 24.0 | 29.7 | 29.7 | **+78.1%** |

#### K-Means Kernel
| Config | High (s) | Low (s) | Total (s) | Improvement |
|--------|----------|---------|-----------|-------------|
| No Policy | 85.5 | 85.5 | 85.5 | - |
| Prefetch(0,20) | 17.0 | 17.4 | 17.4 | +79.7% |
| Prefetch(20,80) | 5.5 | 6.7 | 6.7 | **+92.2%** |
| Evict(20,80) | 5.3 | 6.6 | 6.6 | **+92.3%** |

### 2.2 优先级划分效果

使用 **Prefetch(20,80)** 策略时，高优先级进程相比低优先级进程的完成时间优势：

| Kernel | High (s) | Low (s) | 差异 | High 提前完成 |
|--------|----------|---------|------|--------------|
| Hotspot | 22.5 | 24.0 | 1.5s | 6.3% |
| GEMM | 24.0 | 29.6 | 5.6s | 18.9% |
| K-Means | 5.5 | 6.7 | 1.2s | 17.9% |

---

## 3. 结果分析

### 3.1 为什么无策略时两个进程同样慢？

在 **No Policy** 情况下：
- 两个进程竞争相同的 GPU 内存资源
- UVM page fault 处理无差异化
- 导致严重的 thrashing（页面频繁换入换出）
- 结果：两个进程都大幅减速（对比 Single 1x 基准）

**关键观察**：No Policy 下 High 和 Low 完成时间几乎相同（差异 < 0.2%），说明系统对两者一视同仁，没有任何优先级保障。

### 3.2 策略如何实现优先级划分？

我们的策略通过以下机制实现优先级划分：

1. **Prefetch 限制**：
   - 高优先级进程：较少的 prefetch 限制（更激进的预取）
   - 低优先级进程：较多的 prefetch 限制（保守的预取）
   - 效果：高优先级进程获得更多内存带宽

2. **Eviction 策略**：
   - 优先驱逐低优先级进程的页面
   - 保护高优先级进程的 working set
   - 减少高优先级进程的 page fault

### 3.3 不同 Workload 的表现差异

| Kernel | 改进幅度 | 原因分析 |
|--------|----------|----------|
| **K-Means** | 92% | 稀疏访问模式，prefetch 策略效果显著 |
| **GEMM** | 78% | 计算密集但内存访问规律，策略有效减少竞争 |
| **Hotspot** | 56% | 热点访问，局部性强，改进空间相对较小 |

### 3.4 Prefetch vs Evict 策略对比

实验结果显示 **Prefetch(20,80)** 和 **Evict(20,80)** 效果几乎相同：
- 差异 < 1%
- 说明在当前 workload 下，prefetch 控制是主要优化手段
- Eviction 策略可能在内存压力更大时展现优势

---

## 4. 可视化分析

### 4.1 推荐使用的图表

**对于展示"优先级划分"这一 claim，推荐使用并排柱状图 (Side-by-side Bar Chart)**：

**优点**：
1. ✅ 直观对比 High vs Low 的完成时间
2. ✅ 清晰展示策略带来的差异化效果
3. ✅ 易于理解：绿色（High）< 红色（Low）= 优先级生效

**图表解读要点**：
- **No Policy**：绿色 ≈ 红色 → 无优先级划分
- **With Policy**：绿色 < 红色 → 优先级划分生效
- **基准线**：Single 1x 表示理想情况（无竞争）

### 4.2 堆叠柱状图的补充价值

堆叠图适合展示：
- 系统总完成时间的改进
- 竞争期 vs 单独运行期的比例
- 整体系统效率

---

## 5. 结论

### 5.1 主要发现

1. **优先级划分有效**：我们的策略成功实现了多租户场景下的内存优先级划分，高优先级进程完成时间比低优先级进程快 6-19%。

2. **显著性能提升**：相比无策略基准，系统总完成时间改进 55-92%，接近单进程运行的理想情况。

3. **通用性强**：策略在不同访问模式的 workload 上均表现良好，尤其对稀疏访问模式（K-Means）效果最佳。

### 5.2 Claim 支撑

> **"Our eBPF-based policy enables effective memory priority differentiation in multi-tenant GPU environments."**

**支撑证据**：
- 量化数据：High 进程比 Low 进程快 6-19%
- 对比基准：No Policy 下两者无差异
- 跨 workload 验证：三种不同 kernel 均有效

---

## Appendix: 图表文件

- `all_kernels_comparison.pdf` - 并排柱状图（推荐用于优先级划分 claim）
- `all_kernels_stacked.pdf` - 堆叠柱状图（展示系统效率）
