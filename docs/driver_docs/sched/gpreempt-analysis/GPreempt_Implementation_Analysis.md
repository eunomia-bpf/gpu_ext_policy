# GPreempt Implementation Analysis

This document provides a detailed analysis of the GPreempt implementation, comparing it with the claims made in the USENIX ATC'25 paper "GPreempt: GPU Preemptive Scheduling Made General and Efficient."

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Paper Claims vs Implementation](#paper-claims-vs-implementation)
3. [Architecture Overview](#architecture-overview)
4. [Driver Modifications](#driver-modifications)
5. [Core Preemption Mechanisms](#core-preemption-mechanisms)
6. [Block-Level Preemption (BLP)](#block-level-preemption-blp)
7. [GDRCopy Integration](#gdrcopy-integration)
8. [Comparison with Our eBPF-based Approach](#comparison-with-our-ebpf-based-approach)
9. [Limitations and Future Work](#limitations-and-future-work)

---

## Executive Summary

GPreempt is a GPU preemptive scheduling system that achieves <40µs preemption latency on NVIDIA A100 GPUs. The key innovations are:

1. **Timeslice-based Yield**: Setting BE (Best-Effort) task timeslice to ~200µs for quick yield
2. **Hint-based Pre-preemption**: Overlapping preemption with data preparation using GDRCopy
3. **Block-Level Preemption (BLP)**: Software-based preemption for non-idempotent workloads

The implementation consists of:
- Modified NVIDIA driver (driver.patch)
- User-space preemption library (gpreempt.cpp)
- Execution framework with BLP support (executor.cpp)
- Client implementations (gpreemptclient.cpp, blpclient.cpp)

---

## Paper Claims vs Implementation

### Claim 1: <40µs Preemption Latency

**Paper**: "GPreempt achieves less than 40µs preemption latency on NVIDIA A100"

**Implementation Reality**:
- The timeslice-based yield mechanism is implemented in `gpreempt.cpp:52-59`:
```cpp
NV_STATUS NvRmModifyTS(NvContext ctx, NvU64 timesliceUs) {
    NVA06C_CTRL_TIMESLICE_PARAMS timesliceParams0;
    timesliceParams0.timesliceUs = timesliceUs;
    return NvRmControl(ctx.hClient, ctx.hObject,
                       NVA06C_CTRL_CMD_SET_TIMESLICE,
                       (NvP64)&timesliceParams0, sizeof(timesliceParams0));
}
```
- Priority 0 (high/LC) gets timeslice 1,000,000µs (1 second)
- Priority 1 (low/BE) gets timeslice 1µs (forcing frequent yield)
- The ~44MB context switch overhead at 1.1TB/s = ~40µs is hardware-limited

**Verdict**: ✓ Implementation matches paper claim

### Claim 2: Timeslice-based Yield Mechanism

**Paper**: "We set the timeslice of BE tasks to approximately 200µs... When the GPU timeslice of the BE task expires, the GPU context-switches to the LC task"

**Implementation Reality**:
- In `gpreempt.cpp:61-72`:
```cpp
int set_priority(NvContext ctx, int priority) {
    NV_STATUS status;
    if (priority == 0){
        status = NvRmModifyTS(ctx, 1000000);  // LC: 1 second
    } else {
        status = NvRmModifyTS(ctx, 1);        // BE: 1µs (not 200µs!)
    }
    // ...
}
```
- **Discrepancy**: The implementation uses 1µs instead of 200µs for BE tasks
- This is likely more aggressive than the paper suggests

**Verdict**: ⚠ Implementation differs from paper (1µs vs 200µs)

### Claim 3: Hint-based Pre-preemption

**Paper**: "GPreempt uses GDRCopy to notify the CPU when the GPU kernel is about to finish, overlapping the preemption with data preparation"

**Implementation Reality**:
- GDRCopy integration in `gpreemptclient.cpp:83-114`:
```cpp
int get_gdr_map(GdrEntry *entry) {
    gdr_t g = gdr_open();
    ASSERT_CUDA_ERROR(GPUMemAlloc(&d_pool, GPU_PAGE_SIZE * 2));
    gdr_pin_buffer(g, (unsigned long)d_pool, GPU_PAGE_SIZE, 0, 0, &g_mh);
    gdr_map(g, g_mh, (void**)&h_pool, GPU_PAGE_SIZE);
    // ...
}
```
- Pre-preemption daemon in `gpreemptclient.cpp:267-291`:
```cpp
void fooDaemon(std::atomic<bool> &stopFlag, int task_cnt) {
    while (!stopFlag.load()) {
        // Process hints at scheduled times
        while(hints.size() && system_clock::now() > hints.begin()->t) {
            start_blocking_stream(hints.begin()->stream, hints.begin()->signal);
            hints.erase(hints.begin());
        }
    }
}
```
- `SWITCH_TIME` constant is 100µs (line 21), scheduling preemption 100µs before expected completion

**Verdict**: ✓ Implementation matches paper claim

### Claim 4: Support for Non-idempotent Workloads

**Paper**: "GPreempt supports complex, non-idempotent workloads like graph computing and scientific simulations"

**Implementation Reality**:
- Block-Level Preemption (BLP) in `executor.cpp:200-342`:
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
- Supports three workload types: DNN, Graph Computing, Scientific Computing
- Uses `dpStopIndex` to track kernel progress and resume from checkpoints

**Verdict**: ✓ Implementation matches paper claim

---

## Architecture Overview

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Space                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ DISB         │  │ FooClient    │  │ Executor           │     │
│  │ Benchmark    │──│ (LC/BE)      │──│ (Base/BLP)         │     │
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
│                       Kernel Space                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐     │
│  │           Modified NVIDIA Driver (driver.patch)         │     │
│  │  - NV_ESC_RM_QUERY_GROUP (0x60) for TSG handle lookup  │     │
│  │  - Security bypass (Nv04Control instead of SecInfo)    │     │
│  │  - threadId tracking in KernelChannelGroupApi          │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Data Structures

```cpp
// gpreempt.h:111-115
struct NvContext {
    NvHandle hClient;     // RM client handle
    NvHandle hObject;     // TSG (Time Slice Group) handle
    NvChannels channels;  // Channel list for disable/enable
};

// executor.h:60-84
class BLPExecutor : public Executor {
    GPUdeviceptr dpStop;       // Shared stop signal (GDRCopy mapped)
    GPUdeviceptr dpStopIndex;  // Current kernel index for resume
    GPUdeviceptr executed;     // Bitmap of executed kernels
    int stopIndex;             // Host-side stop index
};
```

---

## Driver Modifications

The driver patch (`patch/driver.patch`) makes several critical modifications:

### 1. New ioctl Command: NV_ESC_RM_QUERY_GROUP (0x60)

**Location**: `src/nvidia/arch/nvalloc/unix/src/escape.c:54-101`

```c
case NV_ESC_RM_QUERY_GROUP:
{
    // Get client handles from stored OS info
    status = rmapiGetClientHandlesFromOSInfo(g_clientOSInfo,
                                              &pClientHandleList,
                                              &clientHandleListSize);

    // Iterate through all clients
    for(int i = 0; i < clientHandleListSize; ++i) {
        // Find KernelChannelGroupApi (TSG) objects
        it = clientRefIter(pClient, NULL,
                          classId(KernelChannelGroupApi),
                          RS_ITERATE_DESCENDANTS, NV_TRUE);

        while (clientRefIterNext(pClient, &it)) {
            KernelChannelGroupApi *pKernelChannelGroupApi = ...;

            // Match by threadId
            if(pKernelChannelGroupApi->threadId != threadId)
                continue;

            // Return hClient and hObject for the TSG
            pApi->hClient = pClientHandleList[i];
            pApi->hObject = it.pResourceRef->hResource;

            // Also return channel list for NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS
            // ...
        }
    }
}
```

**Purpose**: Allows one process to query TSG handles belonging to another process's GPU context, identified by threadId.

### 2. Security Bypass

**Location**: `src/nvidia/arch/nvalloc/unix/src/escape.c:864-866`

```c
// Changed from:
// Nv04ControlWithSecInfo(pApi, secInfo);
// To:
Nv04Control(pApi);
```

**Purpose**: Bypasses security checks that would prevent cross-process control of GPU resources.

### 3. ThreadId Tracking

**Location**: `src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c:84-85`

```c
pKernelChannelGroupApi->threadId = portThreadGetCurrentThreadId();
```

**Purpose**: Associates each TSG with the thread that created it, enabling lookup via `NV_ESC_RM_QUERY_GROUP`.

### 4. Global clientOSInfo Storage

**Location**: `src/nvidia/arch/nvalloc/unix/src/escape.c:36,44-49`

```c
static void* g_clientOSInfo;

// When allocating AMPERE_CHANNEL_GPFIFO_A:
if((bAccessApi ? pApiAccess->hClass : pApi->hClass) == AMPERE_CHANNEL_GPFIFO_A) {
    g_clientOSInfo = secInfo.clientOSInfo;
}
```

**Purpose**: Stores the clientOSInfo for later lookup in `NV_ESC_RM_QUERY_GROUP`.

---

## Core Preemption Mechanisms

### 1. Timeslice Modification

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

### 2. Direct Preemption

```cpp
// gpreempt.cpp:74-81
NV_STATUS NvRmPreempt(NvContext ctx) {
    NVA06C_CTRL_PREEMPT_PARAMS preemptParams;
    preemptParams.bWait = NV_FALSE;      // Asynchronous preemption
    preemptParams.bManualTimeout = NV_FALSE;
    return NvRmControl(ctx.hClient, ctx.hObject,
                       NVA06C_CTRL_CMD_PREEMPT,  // 0xa06c0105
                       (NvP64)&preemptParams,
                       sizeof(preemptParams));
}
```

### 3. Channel Disable/Enable

```cpp
// gpreempt.cpp:103-122
NV_STATUS NvRmDisableCh(std::vector<NvContext> ctxs, NvBool bDisable) {
    NvChannels params;
    params.bDisable = bDisable;
    params.bOnlyDisableScheduling = NV_FALSE;
    params.pRunlistPreemptEvent = nullptr;
    params.bRewindGpPut = NV_FALSE;

    // Collect all channels from all contexts
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

### 4. TSG Handle Query

```cpp
// gpreempt.cpp:33-50
NV_STATUS NvRmQuery(NvContext *pContext) {
    NVOS54_PARAMETERS queryArgs;
    queryArgs.hClient = pContext->hClient;  // threadId passed here
    queryArgs.status = 0x0;
    queryArgs.params = (NvP64)&pContext->channels;

    ioctl(fd, OP_QUERY, &queryArgs);  // OP_QUERY = 0xc0204660

    pContext->hClient = queryArgs.hClient;  // Returns actual hClient
    pContext->hObject = queryArgs.hObject;  // Returns TSG hObject
    return queryArgs.status;
}
```

---

## Block-Level Preemption (BLP)

BLP enables preemption at kernel boundaries for workloads that cannot be safely interrupted mid-kernel.

### Implementation Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    BE Task (BLPExecutor)                        │
├────────────────────────────────────────────────────────────────┤
│  1. Check dpStop flag before each kernel                        │
│  2. If dpStop == 1, record current kernel index to dpStopIndex │
│  3. Yield to LC task                                            │
│  4. When resumed, continue from stopIndex                       │
└────────────────────────────────────────────────────────────────┘
         │                                    ▲
         │ GDRCopy write (1µs)               │ Resume
         ▼                                    │
┌────────────────────────────────────────────────────────────────┐
│                    LC Task (BaseExecutor)                       │
├────────────────────────────────────────────────────────────────┤
│  1. Set dpStop = 1 via GDRCopy                                  │
│  2. Wait for BE tasks to yield (running_be == 0)               │
│  3. Execute LC kernels                                          │
│  4. Set dpStop = 0 via GDRCopy                                  │
└────────────────────────────────────────────────────────────────┘
```

### Code Flow (blpclient.cpp:163-196)

```cpp
virtual void infer() override {
    if(priority != 0) {  // BE Task
        foo::BLPExecutor *blp_executor = ...;

        // Wait for LC tasks to finish
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
            blp_executor->resume(stream);  // Resume from checkpoint
            running_be.fetch_add(1);
        }
    } else {  // LC Task
        int cnt = running_lc.fetch_add(1);
        if(cnt == 0) {
            *(int*)g_stop.cpu_map = 1;  // Signal BE to stop via GDRCopy
        }
        while(running_be.load() > 0) {}  // Wait for BE to yield

        executor->execute(stream);
        GPUStreamSynchronize(stream);

        cnt = running_lc.fetch_sub(1);
        if(cnt == 1) {
            *(int*)g_stop.cpu_map = 0;  // Clear stop signal
        }
    }
}
```

---

## GDRCopy Integration

GDRCopy provides ~1µs CPU-to-GPU notification latency, critical for pre-preemption.

### Setup (gpreemptclient.cpp:83-114)

```cpp
int get_gdr_map(GdrEntry *entry) {
    gdr_t g = gdr_open();

    // Allocate GPU memory (page-aligned)
    ASSERT_CUDA_ERROR(GPUMemAlloc(&d_pool, GPU_PAGE_SIZE * 2));
    d_pool = (d_pool + GPU_PAGE_SIZE) & ~(GPU_PAGE_SIZE - 1);

    // Pin and map to CPU
    gdr_pin_buffer(g, (unsigned long)d_pool, GPU_PAGE_SIZE, 0, 0, &g_mh);
    gdr_map(g, g_mh, (void**)&h_pool, GPU_PAGE_SIZE);

    // Get actual offset
    gdr_info_t info;
    gdr_get_info(g, g_mh, &info);
    int off = info.va - d_pool;
    h_pool = (int*)((char*)h_pool + off);

    // Set up entry for signal
    entry->d = d_pool + pool_top * sizeof(int);
    entry->d_ptr = &entry->d;
    entry->cpu_map = h_pool + pool_top;
    pool_top++;
}
```

### Signal Usage

```cpp
// Signal GPU to stop (from CPU):
*(int*)gdr_stop.cpu_map = 1;

// Clear signal:
*(int*)gdr_stop.cpu_map = 0;
```

### Blocking Kernel

The `block.cu` kernel waits on the signal:
```cpp
__global__ void gpu_block(int *signal) {
    while(*signal == 0) {
        // Spin wait
    }
}
```

---

## Comparison with Our eBPF-based Approach

| Aspect | GPreempt | Our eBPF Approach |
|--------|----------|-------------------|
| **Handle Acquisition** | Driver patch adds `NV_ESC_RM_QUERY_GROUP` ioctl | eBPF tracepoints capture handles during TSG creation |
| **Security Bypass** | Replaces `Nv04ControlWithSecInfo` with `Nv04Control` | Driver patch bypasses `_kgspRpcRmApiControl` security |
| **Cross-process Control** | Uses stored `g_clientOSInfo` + threadId matching | Uses captured hClient/hTsg from tracepoints |
| **Notification Mechanism** | GDRCopy (~1µs) | eBPF maps + userspace polling |
| **Driver Modification** | ~500 lines patch | ~100 lines patch |
| **Intrusiveness** | High (multiple driver files) | Low (single escape.c modification) |
| **Production Safety** | Not recommended | Not recommended |
| **Flexibility** | Fixed TSG query mechanism | Programmable eBPF policies |

### Key Differences

1. **Handle Discovery**:
   - GPreempt: Requires modifying `KernelChannelGroupApi` to store `threadId`, then queries by threadId
   - Our approach: Passively captures handles via eBPF tracepoints when TSGs are created

2. **Architecture**:
   - GPreempt: Tightly integrated with NVIDIA driver internals
   - Our approach: Observability-first, uses standard kernel tracing infrastructure

3. **Use Cases**:
   - GPreempt: Designed for LC/BE task coexistence with known workloads
   - Our approach: General-purpose GPU scheduling research and debugging

---

## Limitations and Future Work

### GPreempt Limitations

1. **Driver Version Dependency**: Patch is specific to NVIDIA driver 550.120
2. **Security Concerns**: Bypasses RM security checks entirely
3. **Global State**: `g_clientOSInfo` is a single global variable (race condition risk)
4. **Fixed Matching Logic**: `cnt != 8` check assumes specific channel count
5. **No Upstream Path**: Cannot be upstreamed due to security implications

### Our eBPF Approach Limitations

1. **Requires Driver Modification**: Still needs escape.c patch for cross-process control
2. **Latency Overhead**: eBPF processing adds some overhead vs direct GDRCopy
3. **No Pre-preemption**: Current implementation lacks hint-based scheduling

### Future Work Suggestions

1. **Hybrid Approach**: Combine eBPF observability with GDRCopy notification
2. **Struct_ops Integration**: Use BPF struct_ops for pluggable scheduling policies
3. **Upstream Collaboration**: Work with NVIDIA to add proper cross-process APIs
4. **Container Support**: Extend for GPU sharing in Kubernetes environments

---

## Appendix: RM Control Commands Used

| Command | Code | Description |
|---------|------|-------------|
| `NVA06C_CTRL_CMD_SET_TIMESLICE` | 0xa06c0103 | Set TSG timeslice |
| `NVA06C_CTRL_CMD_PREEMPT` | 0xa06c0105 | Trigger TSG preemption |
| `NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL` | 0xa06c0107 | Set interleave priority |
| `NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS` | 0x2080110b | Enable/disable channels |
| `NV2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY` | 0x20801115 | Set scheduling policy |
| `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE` | 0xa06f0103 | GPFIFO scheduling control |
| `NVA06F_CTRL_CMD_RESTART_RUNLIST` | 0xa06f0111 | Restart runlist |

---

## References

1. GPreempt Paper: "GPreempt: GPU Preemptive Scheduling Made General and Efficient" (USENIX ATC'25)
2. NVIDIA Open GPU Kernel Modules: https://github.com/NVIDIA/open-gpu-kernel-modules
3. GDRCopy: https://github.com/NVIDIA/gdrcopy
4. DISB Benchmark: https://github.com/SJTU-IPADS/disb
