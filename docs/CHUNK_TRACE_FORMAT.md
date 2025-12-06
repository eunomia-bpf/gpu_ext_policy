# Chunk Trace 输出格式文档

## 概述

`chunk_trace` 是一个 BPF 工具，用于跟踪 NVIDIA UVM (Unified Virtual Memory) 驱动的 GPU 内存 chunk 操作。它通过 kprobe 钩住 UVM 驱动的 BPF hook wrapper 函数，捕获内存激活、使用和驱逐事件。

## CSV 输出格式

```
time_ms,hook_type,pid,owner_pid,va_space,cpu,chunk_addr,list_addr,va_block,va_start,va_end,va_page_index
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `time_ms` | u64 | 事件时间戳（毫秒，相对于第一个事件） |
| `hook_type` | string | Hook 类型：`ACTIVATE`, `POPULATE`, `EVICTION_PREPARE` |
| `pid` | u32 | 当前内核线程 PID（通常是 UVM 工作线程，不是用户进程） |
| `owner_pid` | u32 | **拥有此内存的用户进程 PID**（从 mm->owner->tgid 获取） |
| `va_space` | hex | 进程唯一的 va_space 指针（可用于验证 owner_pid） |
| `cpu` | u32 | 执行此操作的 CPU 核心 |
| `chunk_addr` | hex | GPU chunk 结构体地址 |
| `list_addr` | hex | chunk 所在链表的地址 |
| `va_block` | hex | VA block 结构体地址 |
| `va_start` | hex | VA block 起始虚拟地址 |
| `va_end` | hex | VA block 结束虚拟地址 |
| `va_page_index` | u32 | chunk 在 VA block 中的页索引 |

## Hook 类型说明

### ACTIVATE
Chunk 变为可驱逐状态。当 GPU 内存 chunk 被添加到驱逐候选列表时触发。

### POPULATE
Chunk 被访问/使用。当 GPU 页面 fault 导致内存被填充时触发。

### EVICTION_PREPARE
驱逐准备阶段。当 UVM 驱动准备选择要驱逐的 chunk 时触发。此类型的事件中：
- `chunk_addr` 存储 used_list 地址
- `list_addr` 存储 unused_list 地址
- 其他 VA 相关字段为空

## 进程区分方法

### 使用 owner_pid（推荐）

`owner_pid` 字段直接记录拥有该内存的用户进程 PID，通过以下路径获取：

```
va_space -> va_space_mm.mm -> owner (task_struct*) -> tgid
```

**测试结果：**
```
=== Owner PID Statistics (from va_block) ===
Unique owner PIDs: 2

Owner PID breakdown:
  Owner PID 158040: 3885360 events
  Owner PID 158041: 3455248 events

✓ SUCCESS: Owner PIDs exactly match expected uvmbench PIDs!
```

### 使用 va_space（备用）

`va_space` 是每个进程唯一的 UVM 虚拟地址空间指针，也可用来区分不同进程的内存操作。

## 数据结构关系

```
uvm_gpu_chunk_struct
    └── va_block (uvm_va_block_t*)
            ├── start, end        // VA 地址范围
            └── managed_range (uvm_va_range_managed_t*)
                    └── va_range.va_space (uvm_va_space_t*)
                            └── va_space_mm.mm (mm_struct*)
                                    └── owner (task_struct*)
                                            └── tgid  // 用户进程 PID
```

### 关键结构体

```c
struct uvm_gpu_chunk_struct {
    u64 address;                    // GPU 物理地址
    // ... bitfields ...
    u16 va_block_page_index : 10;   // 页索引
    struct list_head list;          // 链表节点
    uvm_va_block_t *va_block;       // 指向 VA block
};

struct uvm_va_block_struct {
    // ...
    uvm_va_range_managed_t *managed_range;
    u64 start;                      // VA 起始地址
    u64 end;                        // VA 结束地址
};

struct uvm_va_range_managed_struct {
    struct uvm_va_range_struct va_range;
    // ...
};

struct uvm_va_range_struct {
    uvm_va_space_t *va_space;       // 进程的 va_space
    // ...
};

struct uvm_va_space_struct {
    // ...
    uvm_va_space_mm_t va_space_mm;  // 包含 mm 指针
};

struct uvm_va_space_mm_struct {
    struct mm_struct *mm;           // 进程的内存描述符
    // ...
};

// Linux 内核结构
struct mm_struct {
    // ...
    struct task_struct *owner;      // 拥有此 mm 的进程
};

struct task_struct {
    // ...
    pid_t tgid;                     // 线程组 ID (进程 PID)
};
```

## 示例输出

```csv
time_ms,hook_type,pid,owner_pid,va_space,cpu,chunk_addr,list_addr,va_block,va_start,va_end,va_page_index
0,ACTIVATE,158056,158040,0xffffcf69ca739008,18,0xffffcf69cb566d38,0xffff8d571e68da58,0xffff8d5ecee8e7c8,0x733f13800000,0x733f139fffff,0
0,ACTIVATE,158056,158040,0xffffcf69ca739008,18,0xffffcf69cb566d88,0xffff8d571e68da58,0xffff8d56087b2e20,0x733f18400000,0x733f185fffff,0
1,POPULATE,158056,158041,0xffffcf69ca825008,8,0xffffcf69cb566e28,0xffff8d571e68da58,0xffff8d5fed837350,0x733f1be00000,0x733f1bffffff,0
```

**字段解释：**
- `pid=158056`: UVM 内核工作线程
- `owner_pid=158040/158041`: 实际的 uvmbench 用户进程
- `va_space`: 两个进程有不同的 va_space 指针

## 使用建议

1. **区分进程**: 使用 `owner_pid` 字段（推荐）或 `va_space` 字段
2. **分析内存区域**: 使用 `va_start` 和 `va_end` 确定操作涉及的虚拟地址范围
3. **追踪特定 chunk**: 使用 `chunk_addr` 追踪单个 chunk 的生命周期
4. **统计分析**: 按 `owner_pid` 分组统计各进程的内存操作模式

## 运行方式

```bash
# 启动跟踪
sudo ./chunk_trace > trace.csv

# 在另一个终端运行 UVM 程序
./uvmbench --size_factor=0.6 --mode=uvm

# Ctrl+C 停止跟踪
```

## 测试脚本

参见 `/home/yunwei37/workspace/gpu/co-processor-demo/memory/micro/test_chunk_trace.py`

## 注意事项

1. `pid` 字段是内核工作线程的 PID，不是用户进程的 PID
2. `owner_pid` 是真正的用户进程 PID，通过 `mm->owner->tgid` 获取
3. 每个进程的 `va_space` 指针唯一，可用于交叉验证
4. `EVICTION_PREPARE` 事件不包含 VA 相关信息
