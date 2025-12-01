# GPU Preempt Control Tool Design Document

基于 eBPF tracepoint 的 GPU TSG 抢占控制工具设计文档。

## 概述

`gpu_preempt_ctrl` 是一个用户态工具，通过 eBPF tracepoint 监听 NVIDIA GPU TSG（Time Slice Group）生命周期事件，捕获 `hClient` 和 `hTsg` 句柄信息，并提供跨进程 GPU 调度控制功能。

## 当前状态: ✅ 完全工作

### 功能列表

1. **TSG 事件监控** - 通过 tracepoint 捕获 TSG 创建、调度、销毁事件 ✅
2. **句柄捕获** - 成功获取 hClient 和 hTsg 句柄 ✅
3. **进程追踪** - 按 PID 跟踪 TSG 归属 ✅
4. **跨进程 Preempt** - 从外部进程发送 preempt ioctl ✅
5. **跨进程 Timeslice** - 设置 TSG 时间片 ✅
6. **跨进程 Interleave** - 设置 TSG 交错级别 ✅

### 测试验证

```bash
$ sudo ./test_preempt_ioctl ./test_preempt_ctrl

=== Testing PREEMPT ===
  PREEMPT hClient=0xc1d0002c hTsg=0x5c000046 result=0 (0x0) duration=356 us

=== Testing SET_TIMESLICE ===
  SET_TIMESLICE hClient=0xc1d0002c hTsg=0x5c000046 timeslice=2000 us result=0

=== Testing SET_INTERLEAVE_LEVEL ===
  SET_INTERLEAVE hClient=0xc1d0002c hTsg=0x5c000046 level=HIGH result=0
```

## 实现方案

### 驱动修改

为了支持跨进程 TSG 控制，需要修改 NVIDIA 驱动 `escape.c` 中的 `NV_ESC_RM_CONTROL` 处理：

```c
// 在 escape.c 的 NV_ESC_RM_CONTROL case 中

// GPU Preempt Control: bypass security check for scheduling control commands
// Commands: NVA06C_CTRL_CMD_PREEMPT (0xa06c0105), SET_TIMESLICE (0xa06c0103),
//           SET_INTERLEAVE_LEVEL (0xa06c0107)
if ((pApi->cmd & 0xffff0000) == 0xa06c0000) {
    // TSG control commands - bypass security checks by using kernel privilege
    // and clearing clientOSInfo to skip process ownership validation
    API_SECURITY_INFO tsgSecInfo = secInfo;
    tsgSecInfo.privLevel = RS_PRIV_LEVEL_KERNEL;
    tsgSecInfo.clientOSInfo = NULL;
    Nv04ControlWithSecInfo(pApi, tsgSecInfo);
} else {
    Nv04ControlWithSecInfo(pApi, secInfo);
}
```

这个修改：
- 仅对 TSG 调度控制命令 (0xa06c0000 范围) 生效
- 将 `privLevel` 设置为 `RS_PRIV_LEVEL_KERNEL` 以绕过用户权限检查
- 清除 `clientOSInfo` 以绕过进程所有权验证

### ioctl 调用方式

用户态程序需要使用正确的 ioctl 编码：

```c
#include <linux/ioctl.h>

#define NV_IOCTL_MAGIC      'F'
#define NV_IOCTL_BASE       200
#define NV_ESC_IOCTL_XFER_CMD   (NV_IOCTL_BASE + 11)
#define NV_ESC_RM_CONTROL   0x2A

typedef struct {
    uint32_t cmd;
    uint32_t size;
    void    *ptr __attribute__((aligned(8)));
} nv_ioctl_xfer_t;

// 使用 _IOWR 宏正确编码 ioctl 命令
ioctl(fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_IOCTL_XFER_CMD, nv_ioctl_xfer_t), &xfer);
```

**关键点**: 必须使用 `_IOWR()` 宏将结构大小编码到 ioctl 命令中，否则驱动会返回 `-EINVAL`。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Space                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              gpu_preempt_ctrl                                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │ │
│  │  │ Ring Buffer  │  │ TSG Tracker  │  │ ioctl Commands   │   │ │
│  │  │   Reader     │  │   (in-mem)   │  │ [WORKING!]       │   │ │
│  │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │ │
│  └─────────┼─────────────────┼───────────────────┼─────────────┘ │
│            │                 │                   │               │
│            │ Works!          │ Works!            ✓ Works!        │
└────────────┼─────────────────┼───────────────────┼───────────────┘
             │                 │                   │
═════════════╪═════════════════╪═══════════════════╪═══════════════
             ▼                 ▼                   ▼ Kernel Space
┌─────────────────────────────────────────────────────────────────┐
│              gpu_preempt_ctrl.bpf.c (eBPF)                      │
│  tracepoint/nvidia/nvidia_gpu_tsg_create   ───────────┐         │
│  tracepoint/nvidia/nvidia_gpu_tsg_schedule ───────────┤ Works!  │
│  tracepoint/nvidia/nvidia_gpu_tsg_destroy  ───────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NVIDIA Driver (nvidia.ko - Modified)         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  NV_ESC_RM_CONTROL ioctl                                  │   │
│  │    → TSG commands (0xa06c0000): Use kernel privilege      │   │
│  │    → Other commands: Normal security check                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
src/
├── gpu_preempt_ctrl.bpf.c      # eBPF 程序 (tracepoint handlers)
├── gpu_preempt_ctrl.c          # 用户态程序 (监控+控制)
├── gpu_preempt_ctrl_event.h    # 共享数据结构定义

docs/driver_docs/sched/test/
├── test_preempt_ctrl.cu        # CUDA 测试程序
├── test_preempt_ioctl.c        # ioctl 测试程序
└── test_gpu_preempt.sh         # 测试脚本

# 驱动修改
open-gpu-kernel-modules/src/nvidia/arch/nvalloc/unix/src/escape.c
```

## 使用方法

### 1. 编译驱动 (带修改)

```bash
cd /path/to/open-gpu-kernel-modules
make -j8 modules
sudo rmmod nvidia-uvm nvidia-drm nvidia-modeset nvidia
sudo insmod kernel-open/nvidia.ko
sudo insmod kernel-open/nvidia-modeset.ko
sudo insmod kernel-open/nvidia-drm.ko
sudo insmod kernel-open/nvidia-uvm.ko
```

### 2. 编译工具

```bash
cd src/
make gpu_preempt_ctrl
```

### 3. 监控 GPU TSG 活动

```bash
# 启动监控（verbose 模式）
sudo ./gpu_preempt_ctrl -v

# 在另一个终端运行 CUDA 程序
./any_cuda_program

# 观察输出
17:05:43.123 [CPU01] TSG_CREATE   PID=12345 my_cuda_app  hClient=0x... hTsg=0x...
17:05:43.456 [CPU01] TSG_SCHEDULE PID=12345 my_cuda_app  channels=8 timeslice=1024
```

### 4. 交互命令

```
gpu> list
=== Tracked TSGs (3) ===
hClient    hTsg       tsg_id     engine   runlist  timeslice  level    process
--------------------------------------------------------------------------------
0xc1d0008b 0x5c000013 1          COPY     0        1024       LOW      my_app (pid=12345)

gpu> preempt 0xc1d0008b 0x5c000013
  PREEMPT hClient=0xc1d0008b hTsg=0x5c000013 result=0 duration=250 us

gpu> timeslice 0xc1d0008b 0x5c000013 2000
  SET_TIMESLICE hClient=0xc1d0008b hTsg=0x5c000013 timeslice=2000 us result=0

gpu> interleave 0xc1d0008b 0x5c000013 2
  SET_INTERLEAVE hClient=0xc1d0008b hTsg=0x5c000013 level=HIGH result=0

gpu> preempt-pid 12345
  Preempting all TSGs for PID 12345...
  PREEMPT hClient=0xc1d0008b hTsg=0x5c000013 result=0 duration=250 us

gpu> help
gpu> quit
```

## 安全考虑

当前实现绕过了 NVIDIA RM 的安全检查，仅允许 root 用户使用（需要 CAP_SYS_ADMIN 加载 eBPF）。
未来可以考虑：
1. 添加更细粒度的权限检查（如检查调用者是否为 root）
2. 限制可控制的 TSG 范围
3. 添加审计日志

## 参考

- [GPreempt.patch](./GPreempt.patch) - GPreempt 论文的驱动修改方案
- [test_preempt_cuda.c](./test_preempt_cuda.c) - 同进程 preempt 测试程序
- NVIDIA open-gpu-kernel-modules escape.c - RM ioctl 处理
