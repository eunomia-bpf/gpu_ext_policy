# GPreempt 实现分析

本文档详细分析 GPreempt 的实现，并与 USENIX ATC'25 论文《GPreempt: GPU Preemptive Scheduling Made General and Efficient》中的声明进行对比。

## 目录

1. [概述](#概述)
2. [论文声明与实现对比](#论文声明与实现对比)
3. [架构总览](#架构总览)
4. [驱动修改分析](#驱动修改分析)
5. [核心抢占机制](#核心抢占机制)
6. [块级抢占 (BLP)](#块级抢占-blp)
7. [GDRCopy 集成](#gdrcopy-集成)
8. [与我们 eBPF 方案的对比](#与我们-ebpf-方案的对比)
9. [局限性与未来工作](#局限性与未来工作)

---

## 概述

GPreempt 是一个 GPU 抢占式调度系统，在 NVIDIA A100 GPU 上实现了 <40µs 的抢占延迟。其核心创新包括：

1. **基于时间片的让步机制**：将 BE（尽力而为）任务的时间片设置为 ~200µs 以快速让步
2. **基于提示的预抢占**：使用 GDRCopy 将抢占与数据准备阶段重叠
3. **块级抢占 (BLP)**：针对非幂等工作负载的软件级抢占

实现组件包括：
- 修改后的 NVIDIA 驱动程序 (driver.patch)
- 用户空间抢占库 (gpreempt.cpp)
- 支持 BLP 的执行框架 (executor.cpp)
- 客户端实现 (gpreemptclient.cpp, blpclient.cpp)

---

## 论文声明与实现对比

### 声明 1：<40µs 抢占延迟

**论文**："GPreempt 在 NVIDIA A100 上实现了小于 40µs 的抢占延迟"

**实现情况**：
- 基于时间片的让步机制在 `gpreempt.cpp:52-59` 中实现：
```cpp
NV_STATUS NvRmModifyTS(NvContext ctx, NvU64 timesliceUs) {
    NVA06C_CTRL_TIMESLICE_PARAMS timesliceParams0;
    timesliceParams0.timesliceUs = timesliceUs;
    return NvRmControl(ctx.hClient, ctx.hObject,
                       NVA06C_CTRL_CMD_SET_TIMESLICE,
                       (NvP64)&timesliceParams0, sizeof(timesliceParams0));
}
```
- 优先级 0（高优先级/LC）获得 1,000,000µs（1秒）的时间片
- 优先级 1（低优先级/BE）获得 1µs 的时间片（强制频繁让步）
- ~44MB 上下文切换开销在 1.1TB/s 带宽下 = ~40µs，这是硬件限制

**结论**：✓ 实现与论文声明一致

### 声明 2：基于时间片的让步机制

**论文**："我们将 BE 任务的时间片设置为约 200µs...当 BE 任务的 GPU 时间片到期时，GPU 会切换到 LC 任务"

**实现情况**：
- 在 `gpreempt.cpp:61-72` 中：
```cpp
int set_priority(NvContext ctx, int priority) {
    NV_STATUS status;
    if (priority == 0){
        status = NvRmModifyTS(ctx, 1000000);  // LC: 1秒
    } else {
        status = NvRmModifyTS(ctx, 1);        // BE: 1µs（不是200µs！）
    }
    // ...
}
```
- **差异**：实现使用 1µs 而不是论文中的 200µs 作为 BE 任务的时间片
- 这比论文描述的更激进

**结论**：⚠ 实现与论文存在差异（1µs vs 200µs）

### 声明 3：基于提示的预抢占

**论文**："GPreempt 使用 GDRCopy 在 GPU 内核即将完成时通知 CPU，将抢占与数据准备阶段重叠"

**实现情况**：
- GDRCopy 集成在 `gpreemptclient.cpp:83-114`：
```cpp
int get_gdr_map(GdrEntry *entry) {
    gdr_t g = gdr_open();
    ASSERT_CUDA_ERROR(GPUMemAlloc(&d_pool, GPU_PAGE_SIZE * 2));
    gdr_pin_buffer(g, (unsigned long)d_pool, GPU_PAGE_SIZE, 0, 0, &g_mh);
    gdr_map(g, g_mh, (void**)&h_pool, GPU_PAGE_SIZE);
    // ...
}
```
- 预抢占守护线程在 `gpreemptclient.cpp:267-291`：
```cpp
void fooDaemon(std::atomic<bool> &stopFlag, int task_cnt) {
    while (!stopFlag.load()) {
        // 在预定时间处理提示
        while(hints.size() && system_clock::now() > hints.begin()->t) {
            start_blocking_stream(hints.begin()->stream, hints.begin()->signal);
            hints.erase(hints.begin());
        }
    }
}
```
- `SWITCH_TIME` 常量为 100µs（第21行），在预期完成前 100µs 调度抢占

**结论**：✓ 实现与论文声明一致

### 声明 4：支持非幂等工作负载

**论文**："GPreempt 支持复杂的非幂等工作负载，如图计算和科学模拟"

**实现情况**：
- 块级抢占 (BLP) 在 `executor.cpp:200-342`：
```cpp
Status BLPExecutor::resume(GPUstream stream) {
    getStopPoint(stream);
    if(stopIndex == -1) return Status::Succ;
    if(type == "dnn") {
        for(int i = stopIndex; i < get_kernel_num(); i++) {
            RETURN_STATUS(launch_kernel(i, stream));
        }
    } else {
        SciComputeBlp::perform_timestep(stream, (int*)dpStop, ...);
    }
}
```
- 支持三种工作负载类型：DNN、图计算、科学计算
- 使用 `dpStopIndex` 跟踪内核进度并从检查点恢复

**结论**：✓ 实现与论文声明一致

---

## 架构总览

### 组件层次结构

```
┌─────────────────────────────────────────────────────────────────┐
│                          用户空间                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ DISB         │  │ FooClient    │  │ Executor           │     │
│  │ 基准测试框架  │──│ (LC/BE)      │──│ (Base/BLP)         │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
│           │                │                   │                 │
│           ▼                ▼                   ▼                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              gpreempt.cpp API                           │     │
│  │  NvRmControl, NvRmQuery, NvRmModifyTS, NvRmPreempt     │     │
│  └────────────────────────────────────────────────────────┘     │
│                            │                                     │
│                    ioctl(/dev/nvidiactl)                         │
├─────────────────────────────────────────────────────────────────┤
│                         内核空间                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐     │
│  │           修改后的 NVIDIA 驱动 (driver.patch)           │     │
│  │  - NV_ESC_RM_QUERY_GROUP (0x60) 用于 TSG 句柄查询      │     │
│  │  - 安全绕过 (Nv04Control 替代 SecInfo 版本)            │     │
│  │  - KernelChannelGroupApi 中的 threadId 跟踪            │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### 关键数据结构

```cpp
// gpreempt.h:111-115
struct NvContext {
    NvHandle hClient;     // RM 客户端句柄
    NvHandle hObject;     // TSG（时间片组）句柄
    NvChannels channels;  // 用于禁用/启用的通道列表
};

// executor.h:60-84
class BLPExecutor : public Executor {
    GPUdeviceptr dpStop;       // 共享停止信号（GDRCopy 映射）
    GPUdeviceptr dpStopIndex;  // 当前内核索引（用于恢复）
    GPUdeviceptr executed;     // 已执行内核位图
    int stopIndex;             // 主机端停止索引
};
```

---

## 驱动修改分析

驱动补丁 (`patch/driver.patch`) 进行了以下关键修改：

### 1. 新增 ioctl 命令：NV_ESC_RM_QUERY_GROUP (0x60)

**位置**：`src/nvidia/arch/nvalloc/unix/src/escape.c:54-101`

```c
case NV_ESC_RM_QUERY_GROUP:
{
    // 从存储的 OS 信息获取客户端句柄
    status = rmapiGetClientHandlesFromOSInfo(g_clientOSInfo,
                                              &pClientHandleList,
                                              &clientHandleListSize);

    // 遍历所有客户端
    for(int i = 0; i < clientHandleListSize; ++i) {
        // 查找 KernelChannelGroupApi (TSG) 对象
        it = clientRefIter(pClient, NULL,
                          classId(KernelChannelGroupApi),
                          RS_ITERATE_DESCENDANTS, NV_TRUE);

        while (clientRefIterNext(pClient, &it)) {
            KernelChannelGroupApi *pKernelChannelGroupApi = ...;

            // 通过 threadId 匹配
            if(pKernelChannelGroupApi->threadId != threadId)
                continue;

            // 返回 TSG 的 hClient 和 hObject
            pApi->hClient = pClientHandleList[i];
            pApi->hObject = it.pResourceRef->hResource;

            // 同时返回通道列表用于 NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS
            // ...
        }
    }
}
```

**目的**：允许一个进程查询属于另一个进程 GPU 上下文的 TSG 句柄，通过 threadId 标识。

### 2. 安全绕过

**位置**：`src/nvidia/arch/nvalloc/unix/src/escape.c:864-866`

```c
// 原来：
// Nv04ControlWithSecInfo(pApi, secInfo);
// 改为：
Nv04Control(pApi);
```

**目的**：绕过阻止跨进程控制 GPU 资源的安全检查。

### 3. ThreadId 跟踪

**位置**：`src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c:84-85`

```c
pKernelChannelGroupApi->threadId = portThreadGetCurrentThreadId();
```

**目的**：将每个 TSG 与创建它的线程关联，使得可以通过 `NV_ESC_RM_QUERY_GROUP` 查找。

### 4. 全局 clientOSInfo 存储

**位置**：`src/nvidia/arch/nvalloc/unix/src/escape.c:36,44-49`

```c
static void* g_clientOSInfo;

// 分配 AMPERE_CHANNEL_GPFIFO_A 时：
if((bAccessApi ? pApiAccess->hClass : pApi->hClass) == AMPERE_CHANNEL_GPFIFO_A) {
    g_clientOSInfo = secInfo.clientOSInfo;
}
```

**目的**：存储 clientOSInfo 以便后续在 `NV_ESC_RM_QUERY_GROUP` 中查找使用。

---

## 核心抢占机制

### 1. 时间片修改

```cpp
// gpreempt.cpp:52-59
NV_STATUS NvRmModifyTS(NvContext ctx, NvU64 timesliceUs) {
    NVA06C_CTRL_TIMESLICE_PARAMS timesliceParams0;
    timesliceParams0.timesliceUs = timesliceUs;
    return NvRmControl(ctx.hClient, ctx.hObject,
                       NVA06C_CTRL_CMD_SET_TIMESLICE,  // 0xa06c0103
                       (NvP64)&timesliceParams0,
                       sizeof(timesliceParams0));
}
```

### 2. 直接抢占

```cpp
// gpreempt.cpp:74-81
NV_STATUS NvRmPreempt(NvContext ctx) {
    NVA06C_CTRL_PREEMPT_PARAMS preemptParams;
    preemptParams.bWait = NV_FALSE;      // 异步抢占
    preemptParams.bManualTimeout = NV_FALSE;
    return NvRmControl(ctx.hClient, ctx.hObject,
                       NVA06C_CTRL_CMD_PREEMPT,  // 0xa06c0105
                       (NvP64)&preemptParams,
                       sizeof(preemptParams));
}
```

### 3. 通道禁用/启用

```cpp
// gpreempt.cpp:103-122
NV_STATUS NvRmDisableCh(std::vector<NvContext> ctxs, NvBool bDisable) {
    NvChannels params;
    params.bDisable = bDisable;
    params.bOnlyDisableScheduling = NV_FALSE;
    params.pRunlistPreemptEvent = nullptr;
    params.bRewindGpPut = NV_FALSE;

    // 从所有上下文收集所有通道
    for(auto ctx : ctxs) {
        for(int i = 0; i < ctx.channels.numChannels; i++) {
            params.hClientList[params.numChannels] = ctx.channels.hClientList[i];
            params.hChannelList[params.numChannels] = ctx.channels.hChannelList[i];
            params.numChannels++;
        }
    }

    return NvRmControl(ctxs[0].hClient, NV_HSUBDEVICE,
                       NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS,  // 0x2080110b
                       (NvP64)&params, sizeof(NvChannels));
}
```

### 4. TSG 句柄查询

```cpp
// gpreempt.cpp:33-50
NV_STATUS NvRmQuery(NvContext *pContext) {
    NVOS54_PARAMETERS queryArgs;
    queryArgs.hClient = pContext->hClient;  // 这里传入 threadId
    queryArgs.status = 0x0;
    queryArgs.params = (NvP64)&pContext->channels;

    ioctl(fd, OP_QUERY, &queryArgs);  // OP_QUERY = 0xc0204660

    pContext->hClient = queryArgs.hClient;  // 返回实际的 hClient
    pContext->hObject = queryArgs.hObject;  // 返回 TSG hObject
    return queryArgs.status;
}
```

---

## 块级抢占 (BLP)

BLP 支持在内核边界处进行抢占，适用于不能在内核执行中途安全中断的工作负载。

### 实现流程

```
┌────────────────────────────────────────────────────────────────┐
│                    BE 任务 (BLPExecutor)                        │
├────────────────────────────────────────────────────────────────┤
│  1. 每个内核执行前检查 dpStop 标志                              │
│  2. 如果 dpStop == 1，记录当前内核索引到 dpStopIndex           │
│  3. 让步给 LC 任务                                              │
│  4. 恢复时，从 stopIndex 继续执行                               │
└────────────────────────────────────────────────────────────────┘
         │                                    ▲
         │ GDRCopy 写入 (1µs)                │ 恢复
         ▼                                    │
┌────────────────────────────────────────────────────────────────┐
│                    LC 任务 (BaseExecutor)                       │
├────────────────────────────────────────────────────────────────┤
│  1. 通过 GDRCopy 设置 dpStop = 1                                │
│  2. 等待 BE 任务让步 (running_be == 0)                         │
│  3. 执行 LC 内核                                                │
│  4. 通过 GDRCopy 设置 dpStop = 0                                │
└────────────────────────────────────────────────────────────────┘
```

### 代码流程 (blpclient.cpp:163-196)

```cpp
virtual void infer() override {
    if(priority != 0) {  // BE 任务
        foo::BLPExecutor *blp_executor = ...;

        // 等待 LC 任务完成
        while(running_lc.load() > 0) {}
        running_be.fetch_add(1);

        blp_executor->running = true;
        blp_executor->execute(stream);

        while(1) {
            GPUStreamSynchronize(stream);
            if(running_lc.load() > 0) {
                running_be.fetch_sub(1);
            } else {
                running_be.fetch_sub(1);
                blp_executor->running = false;
                return;
            }
            while(running_lc.load() > 0) {}
            blp_executor->resume(stream);  // 从检查点恢复
            running_be.fetch_add(1);
        }
    } else {  // LC 任务
        int cnt = running_lc.fetch_add(1);
        if(cnt == 0) {
            *(int*)g_stop.cpu_map = 1;  // 通过 GDRCopy 通知 BE 停止
        }
        while(running_be.load() > 0) {}  // 等待 BE 让步

        executor->execute(stream);
        GPUStreamSynchronize(stream);

        cnt = running_lc.fetch_sub(1);
        if(cnt == 1) {
            *(int*)g_stop.cpu_map = 0;  // 清除停止信号
        }
    }
}
```

---

## GDRCopy 集成

GDRCopy 提供 ~1µs 的 CPU 到 GPU 通知延迟，这对于预抢占至关重要。

### 设置 (gpreemptclient.cpp:83-114)

```cpp
int get_gdr_map(GdrEntry *entry) {
    gdr_t g = gdr_open();

    // 分配 GPU 内存（页对齐）
    ASSERT_CUDA_ERROR(GPUMemAlloc(&d_pool, GPU_PAGE_SIZE * 2));
    d_pool = (d_pool + GPU_PAGE_SIZE) & ~(GPU_PAGE_SIZE - 1);

    // 钉住并映射到 CPU
    gdr_pin_buffer(g, (unsigned long)d_pool, GPU_PAGE_SIZE, 0, 0, &g_mh);
    gdr_map(g, g_mh, (void**)&h_pool, GPU_PAGE_SIZE);

    // 获取实际偏移量
    gdr_info_t info;
    gdr_get_info(g, g_mh, &info);
    int off = info.va - d_pool;
    h_pool = (int*)((char*)h_pool + off);

    // 设置信号条目
    entry->d = d_pool + pool_top * sizeof(int);
    entry->d_ptr = &entry->d;
    entry->cpu_map = h_pool + pool_top;
    pool_top++;
}
```

### 信号使用

```cpp
// 从 CPU 发送停止信号到 GPU：
*(int*)gdr_stop.cpu_map = 1;

// 清除信号：
*(int*)gdr_stop.cpu_map = 0;
```

### 阻塞内核

`block.cu` 内核等待信号：
```cpp
__global__ void gpu_block(int *signal) {
    while(*signal == 0) {
        // 自旋等待
    }
}
```

---

## 与我们 eBPF 方案的对比

| 方面 | GPreempt | 我们的 eBPF 方案 |
|------|----------|------------------|
| **句柄获取** | 驱动补丁添加 `NV_ESC_RM_QUERY_GROUP` ioctl | eBPF tracepoints 在 TSG 创建时捕获句柄 |
| **安全绕过** | 用 `Nv04Control` 替换 `Nv04ControlWithSecInfo` | 驱动补丁绕过 `_kgspRpcRmApiControl` 安全检查 |
| **跨进程控制** | 使用存储的 `g_clientOSInfo` + threadId 匹配 | 使用 tracepoints 捕获的 hClient/hTsg |
| **通知机制** | GDRCopy (~1µs) | eBPF maps + 用户空间轮询 |
| **驱动修改量** | ~500 行补丁 | ~100 行补丁 |
| **侵入性** | 高（修改多个驱动文件） | 低（仅修改 escape.c） |
| **生产安全性** | 不推荐用于生产 | 不推荐用于生产 |
| **灵活性** | 固定的 TSG 查询机制 | 可编程的 eBPF 策略 |

### 关键差异

1. **句柄发现**：
   - GPreempt：需要修改 `KernelChannelGroupApi` 存储 `threadId`，然后通过 threadId 查询
   - 我们的方案：在 TSG 创建时通过 eBPF tracepoints 被动捕获句柄

2. **架构**：
   - GPreempt：与 NVIDIA 驱动内部紧密集成
   - 我们的方案：可观测性优先，使用标准内核追踪基础设施

3. **使用场景**：
   - GPreempt：为已知工作负载的 LC/BE 任务共存而设计
   - 我们的方案：通用 GPU 调度研究和调试

---

## 局限性与未来工作

### GPreempt 局限性

1. **驱动版本依赖**：补丁专门针对 NVIDIA 驱动 550.120
2. **安全问题**：完全绕过 RM 安全检查
3. **全局状态**：`g_clientOSInfo` 是单一全局变量（存在竞态条件风险）
4. **固定匹配逻辑**：`cnt != 8` 检查假设特定的通道数量
5. **无法上游合并**：由于安全影响无法合并到上游

### 我们 eBPF 方案的局限性

1. **仍需驱动修改**：跨进程控制仍需要 escape.c 补丁
2. **延迟开销**：eBPF 处理相比直接 GDRCopy 有一定开销
3. **无预抢占**：当前实现缺少基于提示的调度

### 未来工作建议

1. **混合方案**：结合 eBPF 可观测性与 GDRCopy 通知
2. **Struct_ops 集成**：使用 BPF struct_ops 实现可插拔调度策略
3. **上游协作**：与 NVIDIA 合作添加正式的跨进程 API
4. **容器支持**：扩展支持 Kubernetes 环境中的 GPU 共享

---

## 附录：使用的 RM 控制命令

| 命令 | 代码 | 描述 |
|------|------|------|
| `NVA06C_CTRL_CMD_SET_TIMESLICE` | 0xa06c0103 | 设置 TSG 时间片 |
| `NVA06C_CTRL_CMD_PREEMPT` | 0xa06c0105 | 触发 TSG 抢占 |
| `NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL` | 0xa06c0107 | 设置交织优先级 |
| `NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS` | 0x2080110b | 启用/禁用通道 |
| `NV2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY` | 0x20801115 | 设置调度策略 |
| `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE` | 0xa06f0103 | GPFIFO 调度控制 |
| `NVA06F_CTRL_CMD_RESTART_RUNLIST` | 0xa06f0111 | 重启运行列表 |

---

## 参考资料

1. GPreempt 论文："GPreempt: GPU Preemptive Scheduling Made General and Efficient" (USENIX ATC'25)
2. NVIDIA Open GPU Kernel Modules: https://github.com/NVIDIA/open-gpu-kernel-modules
3. GDRCopy: https://github.com/NVIDIA/gdrcopy
4. DISB 基准测试框架: https://github.com/SJTU-IPADS/disb
