# GPU Memory Eviction Policy Design Guide

## 理解系统架构

### 虚拟地址到物理Chunk的映射关系

在NVIDIA UVM (Unified Virtual Memory) 系统中，理解虚拟地址和物理chunk之间的关系对于设计有效的eviction policy至关重要。

#### 关键概念

1. **物理Chunk (Physical Chunk)**
   - 这是GPU内存管理的基本单位
   - `chunk_addr` 是物理内存块的地址
   - **Eviction policy操作的对象是物理chunk**
   - Chunk大小通常是固定的（如64KB）

2. **VA Block (Virtual Address Block)**
   - 代表一个虚拟地址范围
   - 由 `va_start` 和 `va_end` 定义
   - 一个VA block可能映射到多个物理chunks
   - 通过 `va_block` 指针关联

3. **VA Block Page Index**
   - `va_page_index` 表示chunk在VA block内的页索引
   - 帮助理解访问模式在虚拟地址空间中的位置

#### 映射关系

```
Virtual Address Space              Physical Memory
┌─────────────────────┐           ┌──────────────┐
│   VA Block 1        │           │  Chunk A     │
│   [va_start, va_end]│────┬─────→│  (64KB)      │
│                     │    │      └──────────────┘
└─────────────────────┘    │      ┌──────────────┐
                           └─────→│  Chunk B     │
┌─────────────────────┐           │  (64KB)      │
│   VA Block 2        │──────────→└──────────────┘
│                     │           ┌──────────────┐
└─────────────────────┘           │  Chunk C     │
                                  │  (64KB)      │
                                  └──────────────┘
```

**重要发现：**
- 一个物理chunk可能被多个VA blocks引用（共享）
- Evicting一个物理chunk会影响所有引用它的VA blocks
- Policy需要考虑这种多对多的关系

---

## 从Trace数据中观察到的指标

### 1. Hook事件类型

系统提供三种BPF hook点：

#### `uvm_pmm_chunk_activate` (ACTIVATE)
- **何时触发**: Chunk变为可以被evict的状态
- **含义**: Chunk已经被分配并populated，现在可以加入eviction候选列表
- **Policy可以做什么**:
  - 初始化chunk的元数据（访问时间、频率等）
  - 决定chunk在eviction列表中的初始位置

#### `uvm_pmm_chunk_used` (POPULATE/USED)
- **何时触发**: Chunk被访问/使用
- **含义**: 应用程序访问了这个chunk的数据
- **Policy可以做什么**:
  - 更新访问时间戳（LRU）
  - 增加访问计数器（LFU）
  - 调整chunk在列表中的位置
  - **这是最关键的hook，决定了policy的效果**

#### `uvm_pmm_eviction_prepare` (EVICTION_PREPARE)
- **何时触发**: 系统需要evict内存时
- **含义**: 内存压力触发，需要准备eviction
- **Policy可以做什么**:
  - 重新排序列表（如果需要）
  - 最后的调整机会
  - 通常LRU不需要额外操作，因为列表已经按访问顺序排列

---

## 如何设计Eviction Policy

### 步骤1: 运行Trace工具收集数据

```bash
# 运行chunk_trace工具收集数据
sudo ./chunk_trace -d 10 -o /tmp/chunk_trace.csv

# 分析数据
python3 scripts/analyze_chunk_trace.py /tmp/chunk_trace.csv
python3 scripts/visualize_eviction.py /tmp/chunk_trace.csv -o /tmp
```

### 步骤2: 分析关键指标

#### 2.1 访问模式分析

从 `analyze_chunk_trace.py` 输出查看：

```
HOOK TYPE DISTRIBUTION
------------------------------------------------------------
ACTIVATE              10,234 (15.2%)
POPULATE              52,891 (78.6%)
EVICTION_PREPARE       4,123 ( 6.1%)
```

**解读**:
- POPULATE频率高 → 访问密集，需要考虑频率
- EVICTION_PREPARE频率高 → 内存压力大，policy需要高效

#### 2.2 Chunk生命周期分析

```
Chunk lifetime statistics:
  Mean:    2,450 ms
  Median:  1,200 ms
  Min:     10 ms
  Max:     45,000 ms
```

**解读**:
- 短生命周期 (< median) → 这些chunks可能是一次性使用，适合FIFO
- 长生命周期 (> mean) → 这些chunks可能是热数据，LRU效果好

#### 2.3 虚拟地址分析 (VA Analysis)

从 `va_analysis.txt` 查看：

```
VIRTUAL TO PHYSICAL CHUNK MAPPING
------------------------------------------------------------
Physical chunks per VA block:
  Mean:    3.2

VA blocks per physical chunk:
  Mean:    1.1
  Chunks mapped to multiple VA blocks: 250 (12.3%)
```

**解读**:
- 如果 `VA blocks per chunk > 1`: 说明有共享，evict时需要考虑影响面
- 如果 `Chunks per VA block` 很大: 说明VA block被分片，可能需要按VA block整体考虑

---

## 常见Policy设计模式

### Pattern 1: LRU (Least Recently Used)

**适用场景**:
- 访问有时间局部性 (temporal locality)
- 最近访问的数据很可能再次被访问

**实现要点**:
```c
SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used, ...) {
    // 将chunk移到列表头部（最不可能被evict）
    bpf_list_move_head(&chunk->list, list);
    return 0;
}
```

**何时选择**:
- Trace显示 `Inter-access time` 短且集中
- `Chunk reuse distribution` 显示大部分chunks被多次访问

### Pattern 2: FIFO (First In First Out)

**适用场景**:
- 访问是流式的，数据只用一次
- Sequential scan pattern

**实现要点**:
```c
SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used, ...) {
    // 不移动chunk，保持插入顺序
    return 1; // BYPASS default behavior
}
```

**何时选择**:
- Trace显示大部分chunks只被访问1-2次
- `Access pattern timeline` 显示sequential pattern

### Pattern 3: LFU (Least Frequently Used)

**适用场景**:
- 有些数据频繁访问，有些很少访问
- Workload有明显的热点数据

**实现需要**:
- BPF map存储访问计数器
- 在eviction_prepare时根据频率重排序

**何时选择**:
- `Chunk reuse distribution` 呈现长尾分布
- 少数chunks占据大部分访问

### Pattern 4: VA-Aware Policy

**适用场景**:
- VA block层面的访问模式明显
- 想要保护同一VA block的chunks

**实现要点**:
```c
SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used, ...) {
    uvm_va_block_t *va_block = chunk->va_block;

    // 根据VA block特征做决策
    // 例如：同一VA block的chunks一起保护
    if (is_hot_va_block(va_block)) {
        // 移到安全位置
        bpf_list_move_head(&chunk->list, list);
    }
    return 0;
}
```

**何时选择**:
- VA analysis显示明显的VA block访问模式
- `Chunks per VA block` 较大，适合整体管理

---

## 实战示例：从Trace到Policy

### 场景1: Sequential Scan Workload

**观察到的指标**:
```
ACTIVATE per chunk:
  1 ACTIVATEs: 95% chunks

POPULATE per chunk:
  1 POPULATEs: 85% chunks
  2 POPULATEs: 12% chunks

Inter-access time: N/A (大部分只访问一次)
```

**分析**:
- 绝大多数chunks只被访问一次
- 明显的streaming pattern

**推荐Policy**: **FIFO**
- 因为数据只用一次，LRU没有意义
- FIFO能保持简单的顺序eviction

**验证方法**:
```bash
# 测试FIFO policy
sudo bpftool struct_ops register obj lru_fifo.bpf.o
# 运行workload并对比性能
```

### 场景2: Random Access with Hotspots

**观察到的指标**:
```
POPULATE per chunk:
  1-5 POPULATEs:   60% chunks
  6-20 POPULATEs:  30% chunks
  >20 POPULATEs:   10% chunks (热点数据)

Chunk reuse distribution: 长尾分布
VA blocks per chunk: Mean 1.2 (有一些共享)
```

**分析**:
- 有明显的热点数据（10% chunks占据大量访问）
- 有chunk共享现象
- 时间局部性存在

**推荐Policy**: **LRU with frequency boost**
- 基础使用LRU捕捉时间局部性
- 对高频访问的chunks给予额外保护
- 考虑共享chunks的影响

**实现思路**:
```c
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u64);    // chunk address
    __type(value, u32);  // access count
} chunk_freq SEC(".maps");

SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used, ...) {
    u64 key = (u64)chunk;
    u32 *count = bpf_map_lookup_elem(&chunk_freq, &key);

    if (count) {
        (*count)++;
        // 高频chunk: 移到更安全的位置
        if (*count > HOTSPOT_THRESHOLD) {
            bpf_list_move_head(&chunk->list, list);
        } else {
            // 普通chunk: LRU行为
            bpf_list_move(&chunk->list, list,
                         POSITION_BASED_ON_FREQ);
        }
    }
    return 0;
}
```

### 场景3: Tiled/Blocked Access Pattern

**观察到的指标**:
```
VA block access patterns:
  Accesses per VA block: Mean 25, Median 20
  VA block lifetime: Mean 5,000ms

Physical chunks per VA block: Mean 4.5
```

**分析**:
- VA block层面的访问模式明显
- 同一VA block的chunks经常一起被访问
- VA block有较长的生命周期

**推荐Policy**: **VA-Block-Aware LRU**
- 以VA block为单位管理
- 同一VA block的chunks一起提升/降低优先级

---

## Policy设计检查清单

### ✅ 数据收集阶段
- [ ] 运行chunk_trace收集至少10秒的数据
- [ ] 确认VA block信息被正确捕获（coverage > 80%）
- [ ] 运行分析脚本生成报告和可视化

### ✅ 分析阶段
- [ ] 识别主要访问模式（sequential/random/mixed）
- [ ] 分析chunk复用率（单次访问 vs 多次访问）
- [ ] 检查时间局部性（inter-access time分布）
- [ ] 分析VA-to-chunk映射关系
- [ ] 识别是否有热点数据

### ✅ 设计阶段
- [ ] 选择基础策略（LRU/FIFO/LFU/Hybrid）
- [ ] 确定需要哪些BPF hooks
- [ ] 设计需要的数据结构（BPF maps）
- [ ] 考虑edge cases（chunk共享、内存压力等）

### ✅ 实现阶段
- [ ] 实现BPF programs
- [ ] 添加调试信息（bpf_printk）
- [ ] 测试基本功能
- [ ] 性能测试对比

### ✅ 验证阶段
- [ ] 对比default policy的性能
- [ ] 检查eviction频率是否降低
- [ ] 验证page fault减少
- [ ] 测试不同workload的适应性

---

## 调试和优化技巧

### 1. 使用bpf_printk调试

```c
SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used, ...) {
    bpf_printk("BPF: chunk_used chunk=%px va_block=%px\n",
               chunk, chunk->va_block);
    // ...
}
```

查看输出：
```bash
sudo cat /sys/kernel/debug/tracing/trace_pipe
```

### 2. 统计性能指标

在userspace收集：
- Page fault次数: `nvidia-smi dmon -s u`
- Eviction频率: 从trace数据统计EVICTION_PREPARE次数
- 应用性能: 运行时间、吞吐量

### 3. A/B测试

```bash
# Baseline (default kernel policy)
run_workload > baseline.log

# Test policy
sudo bpftool struct_ops register obj new_policy.bpf.o
run_workload > test.log

# Compare
compare_results baseline.log test.log
```

---

## 常见陷阱和注意事项

### ⚠️ 陷阱1: 忽略内存压力

**问题**: Policy在低内存压力时表现好，高压力时崩溃

**解决**:
- 在EVICTION_PREPARE时检测压力
- 高压力时切换到更aggressive的策略

### ⚠️ 陷阱2: 过度复杂的逻辑

**问题**: BPF program太复杂，验证失败或性能差

**解决**:
- 保持BPF代码简单
- 复杂计算放在userspace
- 使用BPF map传递信息

### ⚠️ 陷阱3: 不考虑chunk共享

**问题**: Evict了被多个VA blocks共享的chunk

**解决**:
- 检查chunk->va_block的引用情况
- 共享chunk给予更高优先级

### ⚠️ 陷阱4: 只看物理chunk，忽略虚拟地址

**问题**: 应用层面的访问模式在VA block层面，但policy只看chunk

**解决**:
- 分析VA block的访问模式
- 考虑实现VA-aware policy

---

## 性能评估标准

一个好的eviction policy应该：

1. **减少Page Faults**
   - 通过 `nvidia-smi dmon -s u` 监控
   - 目标: 比baseline减少20-50%

2. **减少Eviction频率**
   - 从trace数据统计EVICTION_PREPARE次数
   - 目标: 减少不必要的eviction

3. **提升应用性能**
   - 运行时间缩短
   - 吞吐量提升
   - 目标: 5-30%性能提升（取决于workload）

4. **保持低开销**
   - BPF program执行时间 < 1us
   - 内存开销 < 1MB (BPF maps)

---

## 参考资料

- [BPF List Operations Guide](../docs/lru/BPF_LIST_OPERATIONS_GUIDE.md)
- [UVM Kernel Parameters](../../memory/UVM_KERNEL_PARAMETERS.md)
- Example policies:
  - `src/lru_fifo.bpf.c` - FIFO policy
  - `src/prefetch_adaptive_simple.bpf.c` - Adaptive policy

---

## 总结

设计有效的eviction policy是一个**迭代过程**：

```
1. 收集Trace数据 → 2. 分析模式 → 3. 设计Policy →
4. 实现测试 → 5. 评估效果 → 6. 优化改进 → 回到步骤1
```

关键要点：
- **理解虚拟到物理的映射关系** - 这是设计的基础
- **从数据出发** - 让trace数据指导你的设计决策
- **简单开始** - 先实现基础策略，再逐步优化
- **持续验证** - 用实际workload测试效果

Good luck with your policy design! 🚀
