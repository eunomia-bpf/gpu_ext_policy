# GPU Memory Management BPF Policies

这个目录包含用于NVIDIA UVM (Unified Virtual Memory) 的BPF struct_ops策略实现，用于优化GPU内存管理。


timeout 30 sudo /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src/chunk_trace > /tmp/test_trace.csv



## 📋 目录

- [系统架构](#系统架构)
- [Chunk状态转换](#chunk状态转换)
- [可用的Hook点](#可用的hook点)
- [Eviction策略](#eviction策略)
- [Prefetch策略](#prefetch策略)
- [如何使用](#如何使用)

---

## 系统架构

### 核心概念

#### 1. **物理Chunk (Physical Chunk)**
- GPU内存管理的基本单位（通常64KB）
- `chunk_addr` - 物理内存块地址
- **Eviction policy操作的对象**

#### 2. **VA Block (Virtual Address Block)**
- 代表虚拟地址范围 `[va_start, va_end)`
- 一个VA block通常映射到多个物理chunks
- `va_block_page_index` - chunk在VA block内的页索引

#### 3. **映射关系**
```
Virtual Address Space              Physical Memory (GPU VRAM)
┌─────────────────────┐           ┌──────────────┐
│   VA Block 1        │           │  Chunk A     │
│   [va_start, va_end]│────┬─────→│  (64KB)      │
│   2MB                │    │      │  [ACTIVE]    │
└─────────────────────┘    │      └──────────────┘
                           │      ┌──────────────┐
┌─────────────────────┐    └─────→│  Chunk B     │
│   VA Block 2        │           │  (64KB)      │
│   2MB                │──────────→│  [ACTIVE]    │
└─────────────────────┘           └──────────────┘
                                  ┌──────────────┐
     (unmapped)                   │  Chunk C     │
                                  │  (64KB)      │
                                  │  [UNUSED]    │
                                  └──────────────┘
```

**关键特性**（基于实际trace数据）：
- 平均每个VA block使用 **10.8个物理chunks**
- 平均每个chunk在生命周期内被 **16.5个不同VA blocks重用**
- **⚠️ 注意**：这不是"同时共享"，而是**时间上的重用**
  - 同一时刻，一个chunk只映射到**一个VA block**
  - Chunk被evict后，会被分配给新的VA block
  - "16.5个VA/chunk"是整个trace期间的**累积重用次数**

---

## Chunk状态转换

### 状态机图

```
                    ┌──────────────────────────────┐
                    │                              │
                    │     Unused Pool             │
                    │  (Free chunks available)    │
                    │                              │
                    └──────────┬───────────────────┘
                               │
                               │ assign (from kernel)
                               ↓
                    ┌──────────────────────────────┐
                    │                              │
                    │        Active               │
              ┌────→│  (Mapped to VA block)       │────┐
              │     │                              │    │
              │     └──────────────────────────────┘    │
              │                                          │
              │ reuse                            evict   │
              │ (reassign)                       ↓       │
              │                         ┌────────────────┤
              │                         │                │
              │                         │  In-Eviction  │
              │                         │                │
              │                         └────────┬───────┘
              │                                  │
              └──────────────────────────────────┘
                         回到 Unused Pool
```

### 详细说明

#### 状态1: **Unused（未使用）**
- Chunk在free pool中
- 没有映射到任何VA block
- 等待被分配

#### 状态2: **Active（活跃）**
- Chunk被分配给某个VA block
- 处于"evictable"状态，可以被访问或evict
- **这是BPF policy关注的主要状态**

子状态：
- **Activated** - 刚被分配，加入evictable list
  - Hook: `uvm_pmm_chunk_activate`
- **Being Used** - 正在被访问
  - Hook: `uvm_pmm_chunk_used`

#### 状态3: **In-Eviction（驱逐中）**
- 正在从当前VA block解除映射
- 即将回到unused pool
- Hook: `uvm_pmm_eviction_prepare`

### 完整生命周期流程

```
1. Chunk从unused pool分配给VA Block X
   ↓
2. [uvm_pmm_chunk_activate]
   Chunk激活，加入evictable list（可被evict的候选）
   ↓
3. [uvm_pmm_chunk_used - 可能多次]
   VA Block X访问chunk，policy更新chunk的LRU位置
   ↓
4. [uvm_pmm_eviction_prepare]
   内存压力触发，需要回收内存
   Policy选择victim chunks（最不值得保留的）
   ↓
5. Chunk从VA Block X解除映射
   Chunk → unused pool
   ↓
6. （稍后）Chunk被重新分配给VA Block Y
   回到步骤1
```

### 关键点

1. **Eviction ≠ 直接重新分配**
   ```
   错误理解：
     Chunk A: VA Block X → [evict] → VA Block Y

   正确理解：
     Chunk A: VA Block X → [evict] → unused pool → [assign] → VA Block Y
                                       ↑
                                    中间状态
   ```

2. **为什么需要unused pool？**
   - **解耦evict和assign**：Eviction policy只负责"谁该被踢出去"
   - **批量操作效率**：可以一次evict多个chunks
   - **内存压力缓冲**：Pool大小影响系统性能

3. **Policy的职责边界**
   - ✅ **Policy负责**：选择哪些chunks被evict（victim selection）
   - ❌ **Policy不负责**：Chunk分配给哪个VA block（由内核决定）

---

## ⚠️ 理解Trace数据的常见误区

### 误区1: "16.5个VA blocks/chunk = 同时共享"

**错误理解**:
```
           VA Block 1 ─┐
           VA Block 2 ─┤
           VA Block 3 ─┼─→ Chunk A (同时被17个VA blocks共享)
               ...     ─┤
           VA Block 17─┘
```

**正确理解**:
```
时刻 T1: VA Block 1  → Chunk A
时刻 T2: VA Block 1 evicted, Chunk A → unused pool
时刻 T3: VA Block 5  → Chunk A (被重新分配)
时刻 T4: VA Block 5 evicted, Chunk A → unused pool
时刻 T5: VA Block 12 → Chunk A (再次重新分配)
...
总计: Chunk A 被 17个不同的VA blocks重用过

结论：16.5是"累积重用次数"，不是"同时引用数"
```

### 误区2: "所有chunks都共享 = 需要保护高引用chunks"

**为什么这个逻辑不成立**:

对于**Sequential streaming workload** (如seq_stream):
- 所有chunks的访问频率都是1次
- 没有"热点"chunks
- 高重用次数只说明chunk被**循环利用**得好
- 保护高重用chunk没有意义，因为：
  - 被evict的chunk已经不会再被当前VA访问
  - 重用次数高 = 在内存中呆的时间久，**应该被evict**

对于**Random with hotspots** workload:
- 少数chunks被频繁访问（真正的热点）
- 这时候保护高频访问的chunks才有意义
- 但这要看**访问频率**，不是**重用次数**

### 如何正确分析Trace数据

#### 1. 看访问模式，不是统计数字

```python
# 错误：只看平均值
avg_reuse = total_va_accesses / unique_chunks  # 16.5

# 正确：看时间序列
for chunk in chunks:
    access_times = get_access_times(chunk)
    if len(access_times) == 1:
        print("One-time use - streaming pattern")
    elif has_temporal_locality(access_times):
        print("Reused - keep in cache")
```

#### 2. 区分"重用"和"共享"

- **重用** (Reuse): 时间维度，同一chunk被不同VA使用
  - `chunk → VA1 → evict → VA2 → evict → VA3`
  - 例子：Sequential streaming，chunk循环利用

- **共享** (Sharing): 空间维度，多个VA同时引用同一chunk
  - `chunk ← VA1, VA2, VA3同时引用`
  - 例子：Shared memory, read-only data

#### 3. 从图表看本质

**Sequential pattern的特征**:
```
VA访问热力图：
时间 →
  0ms    5ms    10ms
┌─────┬─────┬─────┐
│█████│     │     │ ← VA Range 1 (只在开始被访问)
├─────┼─────┼─────┤
│     │█████│     │ ← VA Range 2 (中间被访问)
├─────┼─────┼─────┤
│     │     │█████│ ← VA Range 3 (最后被访问)
└─────┴─────┴─────┘

特点：垂直条纹，无重复
```

**Random with hotspots的特征**:
```
VA访问热力图：
时间 →
  0ms    5ms    10ms
┌─────┬─────┬─────┐
│█████│█████│█████│ ← VA Range 1 (一直被访问 - 热点!)
├─────┼─────┼─────┤
│█    │     │  █  │ ← VA Range 2 (偶尔访问)
├─────┼─────┼─────┤
│  █  │█    │█    │ ← VA Range 3 (偶尔访问)
└─────┴─────┴─────┘

特点：某些VA一直热，有明显的热点行
```

---

## 可用的Hook点

NVIDIA UVM提供6个BPF struct_ops hook点，分为两类：

### 类别A: Eviction相关（内存回收）

#### 1. `uvm_pmm_chunk_activate`
**触发时机**: Chunk被分配给VA block后，进入evictable状态

**参数**:
```c
int uvm_pmm_chunk_activate(
    uvm_pmm_gpu_t *pmm,              // GPU内存管理器
    uvm_gpu_chunk_t *chunk,          // 被激活的chunk
    struct list_head *list           // Evictable list
);
```

**Policy可以做什么**:
- 初始化chunk的元数据（访问时间、频率等）
- 决定chunk在eviction list中的初始位置
- 返回0使用默认行为，返回1 bypass默认

**示例**: LRU默认行为将chunk加到list尾部

#### 2. `uvm_pmm_chunk_used`
**触发时机**: Chunk被访问/使用（最关键的hook）

**参数**:
```c
int uvm_pmm_chunk_used(
    uvm_pmm_gpu_t *pmm,
    uvm_gpu_chunk_t *chunk,          // 被访问的chunk
    struct list_head *list
);
```

**Policy可以做什么**:
- 更新访问时间戳（LRU）
- 增加访问计数器（LFU）
- **调整chunk在list中的位置**（决定eviction优先级）
- 考虑chunk的共享度（被多少VA blocks引用）

**重要性**: ⭐⭐⭐⭐⭐ 这是决定policy效果的关键hook

#### 3. `uvm_pmm_eviction_prepare`
**触发时机**: 内存压力触发，需要evict内存

**参数**:
```c
int uvm_pmm_eviction_prepare(
    uvm_pmm_gpu_t *pmm,
    struct list_head *va_block_used,   // Used VA blocks list
    struct list_head *va_block_unused  // Unused VA blocks list
);
```

**Policy可以做什么**:
- 最后调整列表顺序（如果需要）
- 检测内存压力程度
- 动态切换策略（aggressive vs conservative）

**注意**: LRU通常不需要额外操作，因为list已经按访问顺序排列

---

### 类别B: Prefetch相关（预取优化）

Prefetch机制用于在实际访问前将数据从CPU迁移到GPU，减少page fault延迟。

#### 4. `uvm_prefetch_before_compute`
**触发时机**: 在GPU kernel开始计算前，决定要prefetch哪些页面

**参数**:
```c
int uvm_prefetch_before_compute(
    uvm_page_index_t page_index,                    // 触发prefetch的页面索引
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,   // Prefetch候选页面树
    uvm_va_block_region_t *max_prefetch_region,     // 最大可prefetch区域
    uvm_va_block_region_t *result_region            // [OUT] 实际prefetch区域
);
```

**返回值**:
- `0` (DEFAULT) - 使用内核默认策略
- `1` (BYPASS) - 使用`result_region`，跳过默认逻辑
- `2` (ENTER_LOOP) - 进入迭代模式，逐个检查`bitmap_tree`

**策略示例**:
- **Always Max**: 直接prefetch整个`max_prefetch_region`
- **None**: 设置`result_region`为空，禁用prefetch
- **Adaptive**: 返回ENTER_LOOP，让`uvm_prefetch_on_tree_iter`决定

#### 5. `uvm_prefetch_on_tree_iter`
**触发时机**: 当`uvm_prefetch_before_compute`返回ENTER_LOOP时，对每个候选区域调用

**参数**:
```c
int uvm_prefetch_on_tree_iter(
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t *max_prefetch_region,
    uvm_va_block_region_t *current_region,    // 当前检查的区域
    unsigned int counter,                     // 区域内的访问计数
    uvm_va_block_region_t *prefetch_region    // [OUT] 如果选择此区域
);
```

**返回值**:
- `0` - 不选择此区域
- `1` - 选择此区域进行prefetch（设置`prefetch_region`）

**自适应阈值示例**:
```c
// 只prefetch "热"区域（访问率 > threshold%）
if (counter * 100 > subregion_pages * threshold) {
    bpf_uvm_set_va_block_region(prefetch_region, first, outer);
    return 1;  // 选择这个区域
}
return 0;  // 跳过这个区域
```

#### 6. `uvm_bpf_test_trigger_kfunc`
**用途**: 测试/调试用，通过proc文件触发

---

## Eviction策略

### 已实现的策略

#### 1. **LRU (Least Recently Used) - 默认**
**文件**: 内核默认实现

**工作原理**:
```
List结构（UVM从HEAD开始evict）:
┌────────────────────────────────────────┐
│  HEAD (evict from here)    TAIL (safe) │
│  ↓                         ↓            │
│  [老数据] → [中等] → ... → [新数据]     │
│  list_first_entry() ←─ EVICT           │
└────────────────────────────────────────┘
```
- `chunk_activate`: 将chunk加到list**尾部**（tail = 新数据，暂时安全）
- `chunk_used`: 将chunk移到list**尾部**（tail = 最近使用，保护）
- `eviction_prepare`: List**头部**（HEAD）的chunks优先被evict

**适用场景**:
- ✅ 有时间局部性（temporal locality）
- ✅ 最近访问的数据很可能再次被访问

#### 2. **FIFO (First In First Out)**
**文件**: `lru_fifo.bpf.c`

**工作原理**:
```c
SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used, ...) {
    // 不移动chunk，保持插入顺序
    return 1; // BYPASS
}
```

**适用场景**:
- ✅ **Sequential scan（顺序扫描）** - 如`seq_stream` kernel
- ✅ **数据只使用一次（streaming workload）**
- ✅ Working set > GPU memory（频繁eviction场景）
- ❌ **不适合有重复访问的workload**（会evict掉可能再次访问的数据）

**为什么对seq_stream最优**:
1. **匹配访问模式**: Sequential = 一次性访问，最早访问的最先不需要
2. **零维护开销**: `chunk_used`时直接bypass，不移动list
3. **正确的victim选择**: FIFO自然选择最不需要的chunks
4. **性能提升**: 相比LRU减少10-20%的list操作开销

---

## Prefetch策略

### 已实现的策略

#### 1. **Always Max** - 激进预取
**文件**: `prefetch_always_max.bpf.c`

**工作原理**:
```c
SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute, ...) {
    // 直接prefetch整个max_prefetch_region
    bpf_uvm_set_va_block_region(result_region, max_first, max_outer);
    return 1; // BYPASS
}
```

**优点**: 最大化GPU端数据可用性，减少page faults
**缺点**: 可能预取不需要的数据，浪费PCIe带宽和GPU内存

**适用场景**:
- PCIe带宽充足
- GPU内存充足
- Kernel访问模式不确定

#### 2. **None** - 禁用预取
**文件**: `prefetch_none.bpf.c`

**工作原理**:
```c
SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute, ...) {
    // 设置为空region
    bpf_uvm_set_va_block_region(result_region, 0, 0);
    return 1; // BYPASS
}
```

**适用场景**:
- 需要按需迁移（demand paging）
- PCIe带宽有限
- 访问模式稀疏（sparse access）

#### 3. **Adaptive Simple** - 基于阈值的自适应
**文件**: `prefetch_adaptive_simple.bpf.c`

**工作原理**:
1. Userspace监控PCIe吞吐量，更新`threshold_map`
2. BPF根据阈值决定是否prefetch某个区域

```c
SEC("struct_ops/uvm_prefetch_on_tree_iter")
int BPF_PROG(uvm_prefetch_on_tree_iter, ...) {
    unsigned int threshold = get_threshold(); // 从map读取

    // 计算访问密度
    unsigned int subregion_pages = outer - first;

    // 只prefetch热区域: counter/pages > threshold%
    if (counter * 100 > subregion_pages * threshold) {
        bpf_uvm_set_va_block_region(prefetch_region, first, outer);
        return 1; // 选择此区域
    }

    return 0; // 跳过此区域
}
```

**优点**:
- 根据系统负载动态调整
- 平衡prefetch收益和带宽成本

**阈值含义**:
- `threshold = 50%`: 如果区域内50%页面被访问，才prefetch
- Higher threshold → 更保守（less prefetch）
- Lower threshold → 更激进（more prefetch）

**Userspace组件** (需要实现):
```c
// 伪代码：监控PCIe并更新阈值
while (1) {
    float pcie_usage = get_pcie_throughput();

    if (pcie_usage > 0.8) {
        threshold = 70;  // 高负载：更保守
    } else if (pcie_usage > 0.5) {
        threshold = 50;  // 中等负载：平衡
    } else {
        threshold = 30;  // 低负载：更激进
    }

    update_threshold_map(threshold);
    sleep(1);
}
```

### Prefetch vs Eviction的关系

```
                    Prefetch                     Eviction
                       ↓                            ↑
         CPU Memory ←───→ GPU Memory (VRAM) ←─────→ Unused Pool
                    (主动迁移)                (被动回收)

Prefetch目标: 提前将CPU内存迁移到GPU，减少未来的page fault
Eviction目标: 在GPU内存不足时，回收最不重要的chunks
```

**协同优化**:
1. 好的**prefetch策略**减少page faults，降低eviction压力
2. 好的**eviction策略**保留重要数据，减少re-fetch需求
3. 两者配合可以显著提升整体性能

---

## 如何使用

### 1. 编译BPF程序

```bash
cd /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src
make
```

生成的文件:
- `lru_fifo.bpf.o` - FIFO eviction policy
- `prefetch_always_max.bpf.o` - Always max prefetch
- `prefetch_none.bpf.o` - Disable prefetch
- `prefetch_adaptive_simple.bpf.o` - Adaptive prefetch

### 2. 加载Policy

```bash
# 加载FIFO eviction policy
sudo bpftool struct_ops register obj lru_fifo.bpf.o

# 加载prefetch policy
sudo bpftool struct_ops register obj prefetch_always_max.bpf.o
```

### 3. 验证加载

```bash
# 查看已加载的struct_ops
sudo bpftool struct_ops show

# 查看内核日志
sudo dmesg | tail -20
```

### 4. 卸载Policy

```bash
# 找到struct_ops ID
sudo bpftool struct_ops show

# 卸载
sudo bpftool struct_ops unregister id <ID>
```

### 5. 性能测试

```bash
# Baseline: 使用默认内核策略
time ./your_workload

# Test: 加载自定义policy
sudo bpftool struct_ops register obj lru_fifo.bpf.o
time ./your_workload

# 对比page faults
nvidia-smi dmon -s u
```

### 6. 调试

**查看BPF日志**:
```bash
sudo cat /sys/kernel/debug/tracing/trace_pipe
```

**Trace chunk生命周期**:
```bash
# 运行trace工具
cd ../scripts
sudo ./chunk_trace -d 10 -o /tmp/trace.csv

# 分析
./analyze_chunk_trace.py /tmp/trace.csv
./visualize_eviction.py /tmp/trace.csv -o /tmp
```

---

## 设计新Policy的步骤

1. **收集数据**
   ```bash
   sudo ./chunk_trace -d 10 -o /tmp/workload_trace.csv
   ```

2. **分析访问模式**
   ```bash
   ./visualize_eviction.py /tmp/workload_trace.csv -o /tmp
   # 查看生成的图表和统计
   ```

3. **选择策略类型**
   - **Sequential streaming** (如seq_stream) → **FIFO** ⭐
   - **Random with hotspots** → LFU (基于访问频率)
   - **Temporal locality** (重复访问) → LRU (默认)
   - **Mixed pattern** → Adaptive (动态切换)

4. **实现BPF程序**
   - 参考 `lru_fifo.bpf.c` 作为模板
   - 实现关键hooks（至少`chunk_used`）
   - 添加必要的BPF maps（统计、配置等）

5. **测试验证**
   ```bash
   make
   sudo bpftool struct_ops register obj my_policy.bpf.o
   # 运行workload
   # 对比性能指标
   ```

6. **迭代优化**
   - 根据新的trace数据调整
   - A/B测试不同参数
   - 监控page faults和性能

---

## 参考资料

- [Policy Design Guide](../docs/POLICY_DESIGN_GUIDE.md) - 详细的策略设计指南
- [BPF List Operations](../../docs/lru/BPF_LIST_OPERATIONS_GUIDE.md) - BPF链表操作
- [UVM Kernel Parameters](../../memory/UVM_KERNEL_PARAMETERS.md) - UVM内核参数

---

## Troubleshooting

### 问题1: Failed to attach struct_ops

**原因**: 内核模块未加载或BTF信息不匹配

**解决**:
```bash
# 检查nvidia-uvm模块
lsmod | grep nvidia_uvm

# 重新加载模块
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

### 问题2: BPF verifier错误

**原因**: BPF程序违反了verifier规则

**解决**:
- 检查数组边界访问
- 确保所有指针都经过NULL检查
- 使用`BPF_CORE_READ`读取内核结构

### 问题3: Struct_ops已存在

**原因**: 之前的instance未正确清理

**解决**:
```bash
# 找到并杀死持有struct_ops的进程
sudo bpftool map show | grep struct_ops
sudo kill <PID>

# 或强制卸载
sudo bpftool struct_ops unregister id <ID>
```

---

## 贡献

欢迎提交新的策略实现！请确保：
1. 添加详细的注释说明策略逻辑
2. 提供性能测试数据
3. 更新本README

Happy optimizing! 🚀
