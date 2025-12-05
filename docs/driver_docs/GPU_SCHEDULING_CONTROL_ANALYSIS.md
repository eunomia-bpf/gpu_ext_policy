# GPU 调度控制分析：用户态 vs 内核/驱动层

本文档详细分析了在不同层次（用户态 runtime vs 内核/驱动）进行 GPU 调度控制的能力差异、论文定位以及技术路线选择。

---

## 目录

1. [能力差异：用户态 vs 驱动/内核](#1-能力差异用户态-vs-驱动内核)
2. [论文视角：各层次的 claim 上限](#2-论文视角各层次的-claim-上限)
3. [中间路线：机制在驱动，策略在用户态](#3-中间路线机制在驱动策略在用户态)
4. [NVIDIA/AMD 驱动层可控制的具体内容](#4-nvidiaamd-驱动层可控制的具体内容)
5. [CUDA 用户态 API 的调度能力（极其有限）](#5-cuda-用户态-api-的调度能力极其有限)
6. [技术路线建议](#6-技术路线建议)

---

## 1. 能力差异：用户态 vs 驱动/内核

### 1.1 只在用户态（CUDA runtime / shim / LD_PRELOAD）能做的事

假设不能碰内核，只能：
- 写自己的 runtime（替代一部分 libcudart/libcuda 调用）
- 用 MPS/MIG、环境变量、cgroup 之类"系统接口"
- 搞个 sidecar daemon 轮询 `/proc/driver/nvidia` / `nvidia-smi`

**能做到的本质是：**

#### 1.1.1 对"合作的应用"做逻辑上的提交控制

可以自己设计"逻辑 scheduler"：
- 拦截 `cudaLaunchKernel`，延迟/合并/重排 launch
- 把大 kernel 改成很多小 kernel，自己分片控制
- 控制每个 app 的并发 stream 数、batch size、prefetch 策略

**但注意：你只控制"往 CUDA runtime 提交"这一层，后面怎么映射到真实 channel/TSG/runlist 是黑盒。**

#### 1.1.2 利用已有的少量软接口

- **stream priority**：只能在一个进程内部 hint GPU
- **MPS session / MIG**：粗粒度地把几个应用放到同一个虚拟 GPU / 分区
- **cudaGraph** 之类：在应用内部重构依赖图

这些都是"局部、软"的：你给 hint，厂商驱动听不听是另一回事，跨进程根本没保证。

#### 1.1.3 做"协调器"，但不是"执法者"

- 可以在用户态搞一个 scheduler daemon，要求所有作业通过它提交（像 MPS server、集群的 job manager 一样）
- 对**愿意接入你 runtime 的 app**，确实能控制谁先发 kernel，谁被延迟、谁被限流
- **但凡有一个进程绕开你（直接用原生 CUDA，或者 root 启动个独立程序），你完全没办法阻止它抢 GPU**

#### 1.1.4 读不到真正的调度状态

- 看不到 runlist、看不到 TSG priority/timeslice、看不到 preemption 事件
- 最多读一些高层 counter（SM 利用率、mem BW），它已经是"被对方调度完"的结果，而不是调度输入

**用户态方案总结：**
> 用户态方案 = "对一批自愿参加的应用，做 logical scheduling + shaping + soft hints"
>
> 但你无法：
> - 强制控制所有进程
> - 也无法直接改变厂商 GPU scheduler 的行为，只能跟着它转

### 1.2 改 driver / 内核能多做什么？

如果肯动 driver（AMDGPU/KFD、NVIDIA open kernel modules、或者自己的 gBPF GPU driver），能力就不一样了：

#### 1.2.1 总控制权：能管所有 GPU context / TSG / queue

- 内核知道所有 GPU context / channel / queue 的生命周期
- 可以拒绝/推迟 context / queue 创建（admission control）
- 可以调整每个 context 的 timeslice / priority / runlist entry
- 可以直接把某个 context/TSG 从 runlist 里摘掉，让它"暂停"

**换句话说：不管应用是不是配合，你都能硬插手。**

#### 1.2.2 直接操控硬件调度入口

- **NVIDIA**：可以直接改 runlist、TSG 属性（timeslice、interleave）
- **AMD**：可以改 queue priority，甚至关掉 MES，用 NO_HWS 自己 load/unload HQD
- 可以基于 preemption 支持，在调度点触发 context switch

这就从"hint"升级成了"命令"：**你给硬件/firmware 下发新的调度配置**。

#### 1.2.3 对所有进程建立统一的"调度视图"

内核看得到：谁在 GPU 上排队、每个进程有几个 active context/queue、每个队列的 backlog 是多少。

可以基于 PID / cgroup / 容器 / 用户 / job ID 做 global policy：
- 权重、份额（类似 CFS / DRR）
- RT 保证（某类作业的 deadline）
- 优先级 class（foreground / background）

这类东西，用户态 runtime 只能在自己掌控的那一撮进程里勉强模拟，**做不到系统级**。

#### 1.2.4 "执法"而不是"劝告"

比如定义"任意用户最多占用 50% SM"：
- **用户态做法**：只能不停测 SM 利用率，祈祷大家听话
- **驱动做法**：看到某个用户超配时，直接降它的 TSG priority、缩小 timeslice，甚至把它的 queue unload 掉，过一会儿再放回来

想防止"恶意/buggy app 抢占 GPU"：
- 用户态根本没法拦
- 驱动可以直接切断它的队列、标记它为 best-effort

---

## 2. 论文视角：各层次的 claim 上限

### 2.1 只在用户态的论文，典型 story 和天花板

假设只动用户态，论文通常会落在类似下面几种框架：

#### 框架一："一个新 GPU runtime / framework"

**例子**：更好的 job packing、时延敏感服务的批处理策略、带推理/训练特殊优化的 runtime

**卖点**：不改内核、不改厂商驱动、改一点 app 或用 LD_PRELOAD 就能用

**论文语言**：
- "we build a user-level runtime that coordinates GPU usage across applications…"
- "no kernel modifications are required"

**硬约束**：
- 调度 guarantees 都是"best-effort / empirical"，而不是强形式：
  - 你没法说 "no tenant can ever exceed its fair share"
  - 没法说 "we can preempt any kernel within X microseconds，无论它是谁"
- 对"非合作 workload"没有办法：你只能做"我们在可控环境里的评估"

**审稿人会问的尖锐问题**：
- "现实环境中还有一堆 legacy/CUDA 程序，不会改链接，你的 scheduler 对这些进程没用吧？"
- "既然你控制不了内核调度，你的 fairness / SLO guarantee 实际上只能靠实验数据支持，形式上讲不出什么强结论。"
- "和现有的 Paella/Mosaic/某某 runtime 相比，你到底多了什么？是不是只是另一个 fancy runtime？"

#### 框架二："集群层/系统层的协调器"

**卖点**：跨节点/跨 GPU 做 admission control、placement、批处理

GPU 调度这块，本质上还是只能通过 MPS/MIG/priority hints/人为限制 concurrency 来间接影响。

这种 paper 的重心在"系统设计 + scheduling policy + cluster eval"，而不是"OS/driver-level 抽象"。

**总结**：
> 用户态-only 的方案，论文定位偏向 **library/runtime/cluster-manager**，
> 很难 convincingly claim "we redesigned the OS/hardware interface for GPU as a first-class schedulable resource"。

### 2.2 改 driver / 内核的论文，典型 story 和天花板

如果肯动 driver（尤其是基于 open-source AMDGPU 或 NVIDIA open kernel modules），故事就可以明显上一个档次：

#### Claim 1："把 GPU 变成真正的 OS 资源"

可以在 abstract 里写类似：
- "We extend the OS kernel to treat GPUs as first-class schedulable resources, with per-process priorities, shares, and preemptive scheduling policies, analogous to CPU scheduling."
- "Our design enforces global GPU scheduling policies across all applications, including unmodified binaries, with strong fairness/latency guarantees."

**这类句子，用户态-only 根本写不了，因为你没有 enforcement capability。**

#### Claim 2：可以给出"系统级、不容抵赖的 invariants"

比如：
- "No process can consume more than W% of GPU compute over any T-second window, regardless of its behavior"
- "A high-priority class process can always preempt a low-priority class process within X microseconds"
- "We enforce strict prioritization or latency SLOs even under adversarial co-runners"

原因很简单：
- 内核掌握所有 context/queue/runlist
- 可以给出具体机制：
  - 怎么测 per-process GPU usage
  - 怎么调整 TSG priority/timeslice
  - 怎么触发 preemption/context switch
- 这些机制对所有进程有效，过程可验证

#### Claim 3：更强的话语权——提出新的"GPU 调度抽象"

可以定义新的抽象层：
- "per-cgroup GPU weight/priority + time-slice + quota"
- "GPU scheduling classes"（RT / best-effort / batch）
- "GPU segments"（类似 GCAPS 那类，把程序中的 GPU 阶段抽象出来，让内核调度这些段）

然后说：
- "Our kernel exposes a new scheduling abstraction that allows OS-level policies (weights, priorities, reservations) to be enforced across all GPU workloads."

**这跟写一个 runtime 完全不在一个 level 上。**

#### Claim 4：可以把"可编程策略"引入 driver（对 gBPF 尤其重要）

比如你做的是：
- 内核/driver 提供一组 GPU scheduling hooks：
  - on-queue-create/context-create
  - on-preempt/on-context-switch
  - per-period "budget enforcement" 钩子
- 策略逻辑用 eBPF/gBPF 写：
  - eBPF 程序定义 per-tenant weights、SLO
  - 动态调整 queue priority / timeslice / admission 等

那么你可以堂而皇之做一个大 claim：
> "We design the first safe, programmable GPU scheduling substrate integrated into the kernel, enabling arbitrary policies (fair sharing, priority, SLO, admission control) to be deployed without changing the driver."

**这类"substrate/平台级"贡献，用户态 runtime 基本做不到。**

#### Claim 5：评估可以设计得更"硬"

- 可以搞 adversarial workload：一个 tenant 疯狂发 kernel，另一个要保持低 tail latency
- 展示：无论怎么恶意，系统保证高优先级 tenant 的 P99 延迟不超过某个 bound
- 做 stress 测试：百个进程乱搞，仍能维持 fairness 和 throughput

审稿人会觉得：OK，这确实是在设计"OS support for GPU"，而不是一个"smart library"。

**当然，内核改动的负面点也要诚实面对**：
- 移植性差：紧绑特定驱动版本和 GPU 代数
- 工程复杂：要处理很多 corner case（reset、Xid、page fault、suspend/resume…）
- 论文里要花空间解释 hardware/driver 细节

不过从 OSDI/SOSP 的视角，这种痛苦反而是"credibility"：
> 你真去撬了 kernel/driver，并在生产/真实硬件上跑起来，比"用官方 API 叠了一层逻辑"更有说服力。

### 2.3 直接区分两条路的论文语言

**用户态-only 的工作，本质上只能说**：
> "We build a user-level GPU runtime that coordinates participating applications by reshaping their kernel submissions and providing priority hints to the vendor driver.
> Our approach requires applications to link against our runtime (or be launched via our shim), and cannot enforce policies on arbitrary, unmodified processes."

**驱动/内核级的工作，可以说**：
> "We extend the GPU driver and OS kernel to take control of the hardware scheduling interface (channels/TSGs/queues).
> Our system enforces global scheduling policies for all GPU workloads—including unmodified binaries—by directly controlling context priorities, time slices, and preemption at the driver level."

**这两类句子，reviewer 一眼就看出档次和 scope 的区别。**

---

## 3. 中间路线：机制在驱动，策略在用户态

其实不需要把全部 scheduler 都写死在 driver。比较健康、也和 eBPF/gBPF 设想贴合的一条路线是：

### 3.1 driver 提供"最小完备的控制原语"（机制）

- 枚举/控制所有 context/TSG/queue：
  - 调整 priority/timeslice/runlist entry（NVIDIA）
  - 调整 queue priority / sched policy / HQD 映射（AMD）
- per-process GPU usage 计数/统计
- 可控的 preemption 点（基于硬件支持）
- 某种"segment"标注接口（应用可以可选地告诉内核：我现在进入一个重要的 GPU 段）

### 3.2 策略留在用户态 daemon / eBPF 中

- 用户态 scheduler 可以非常复杂（在线 RL、SLO control、trace-driven policy）
- 定期通过 ioctl / sysfs / eBPF map 更新内核里的调度参数
- 对于不合作的应用，仍然能被内核 enforce（你可以降它 priority 或抢占它）

### 3.3 论文 framing

> "We move GPU scheduling mechanisms into the OS kernel/driver, making GPUs a first-class schedulable resource.
> Policy is programmable in user space via a small control API/eBPF interface, allowing system operators to implement custom fairness/QoS/SLO policies without touching applications or driver code."

这种结构在论文里有几个好处：
- 既能 claim OS-level 资源管理的 contribution（不是库）
- 又能讲"policy-agnostic / programmable"的故事（和 eBPF 一脉相承）
- 同时 deployment story 相对好讲：
  - 权限在 kernel
  - 策略可以陆续迭代/热更新

---

## 4. NVIDIA/AMD 驱动层可控制的具体内容

### 4.1 GPU 调度管线拆解

以 NVIDIA 为例（AMD 结构类似）：

**用户态**：
- CUDA stream 里排 kernel，FIFO
- runtime 把命令丢进 pushbuffer（在共享内存里），不用 syscall

**内核态 + GPU**：

1. **Context / channel / TSG 初始化**（GPU kernel driver）
   - 为每个进程建 GPU context
   - 分配若干 channel（每个 channel 绑定一条 pushbuffer）
   - 把这些 channel 打包成一个 **TSG（Time-Slice Group）**，设置 timeslice、priority 等
   - 把 TSG 插入某个 engine 的 **runlist**

2. **Hardware scheduler**
   - 按 runlist 做 TSG 级 time-slice round-robin；每个 TSG 的 timeslice + runlist 中出现次数反映优先级
   - 在当前 TSG 内，把有活的 channel 扫一遍，选一个，有命令就由 PBDMA 拉进 GPU

**AMD 那边也是类似三层**：user-mode queue（AQL）、kernel 维护的 MQD / HQD，MES firmware 再做第一层 queue → HW queue 映射，硬件 queue manager 按 priority / quantum 决定哪个 HW queue 跑。

### 4.2 NVIDIA：在 channel / TSG 创建路径能改到什么？

#### 4.2.1 关键对象：context / channel / TSG / runlist

- driver 在初始化阶段为每个任务创建：
  - 一个或多个 channel（pushbuffer + bookkeeping）
  - 把这些 channel 绑定到一个 **TSG**
  - 再把 TSG 插到某个 runlist 里；runlist 由 HW round-robin 执行

- runlist 其实就是"TSG entries 列表"，每个 entry 有一个 timeslice，TSG 的优先级反映在 runlist 里它出现的次数（interleave frequency）

#### 4.2.2 在创建 channel / TSG 时可以做的事

**1. 映射 OS 级"优先级"到 GPU context/TSG priority**

Tegra 文档给了公开接口：
- `NVRM_GPU_CHANNEL_INTERLEAVE`：按 LOW/MEDIUM/HIGH 设置 TSG 在 runlist 上的频率
- `NVRM_GPU_CHANNEL_TIMESLICE`：设置 TSG 的 timeslice（μs）
- `NVRM_GPU_NVGPU_FORCE_GFX_PREEMPTION` 等开关 preemption 类型

在 open kernel modules / 内部 RM 里，这些最终对应到 TSG 结构中的 priority、timeslice 等字段。可以在"创建 TSG"时：
- 看当前进程的 `nice` / `sched_policy` / cgroup / 自定义 ioctl
- 决定给它一个大 timeslice + 多次 runlist entry（高优先级），还是小 timeslice + 少 entry（低优先级）
- 对低优先级 context 开启 aggressive preemption（方便被抢）

**这已经能实现跨进程的静态优先级调度**（比如 real-time 进程 GPU 优先级更高）。

**2. 控制每个进程的 channel 数，从而限制/放大并发度**

同一 TSG 里 channel 的数量直接决定了该 context 内部能同时挂多少条"命令流"；channel 不够会导致不同 stream 之间出现"假依赖"，让一个 stream 被另一个卡住。

在 channel 创建时，可以：
- 对"关键任务"给足 channel（保持它的 intra-process 并发能力）
- 对"background / best-effort"任务严格限制 channel 数，让它们在自己内部就先排队，减少对 runlist 总体竞争
- 极端情况下，直接把某些 context 的 channel disable 掉，相当于"挂起这个 GPU context"

**3. 选择该 context 跑在哪个 engine / runlist 上**

一个进程可能会同时用 gfx engine 和 compute engine，还有 copy engine 等。可以在 channel/TSG 创建逻辑里：
- 强制某类 workload 走专门 runlist（例如"实时 compute" vs "图形 / 大模型推理"分开 runlist，再靠 timeslice 控制）
- 为单独应用开独立 runlist（如果硬件/driver 支持多 runlist per engine），实现更强隔离

**4. 调 preemption 粒度 / 模式**

NVIDIA 文档提到 compute preemption 一般在 thread-block 粒度，graphics 在像素/primitive 粒度。在初始化 TSG / channel 时可以选择 preemption type（例如 GFXP/CILP）。

这不是"调度算法"，但会极大影响**上下文切换延迟**，从而影响能实现的调度策略的颗粒度（50–750 μs 的 context switch）。

**5. 基于 PID/cgroup 做自动 QoS**

在 channel/TSG 创建路径里，掌握"进程是谁"的信息，这里是挂钩做 QoS 最自然的地方：
- 把 `pid` / `cgroup` / 容器标签映射到 GPU priority class
- 可以配一个内核线程后台监控系统负载，动态调高/调低某些 PID 的 TSG timeslice / interleave；或者干脆拒绝/延迟 channel 创建（软限流）

**这类功能基本不动 user-space CUDA 代码，只改 driver 即可。**

#### 4.2.3 只改"创建 channel"做不到的事

**1. "每个 kernel launch"级别的精确调度**

在默认栈里，driver 根本不解析每个 kernel 命令，它只把 pushbuffer 当 opaque 命令流，交给 GPU host scheduler；kernel 的顺序只由用户态 enqueue 顺序决定。

要做到：
- "先一半 A 的 kernel，再插一个 B，再继续 A"
- "两个进程交替执行单个 kernel 粒度的 RR"

那就意味着要在 driver 里：
- 拦截/解析 GPFIFO / pushbuffer 里的命令，识别出每个 kernel 的边界
- 临时缓存命令，自己重排，再重新写进一个真正提交给 GPU 的队列

这已经不是"改 driver"了，是在重写半个 UMD + command processor，而且还要跟闭源 user-space driver、GSP firmware 搭界，现实上非常重。

**2. 完全替换 NVIDIA 的 runlist 调度算法**

- 从公开文献看，runlist 调度默认是**work-conserving 的、TSG 级 preemptive RR**，TSG priority 体现在 runlist entry 数量和 timeslice 长度上
- 很多 logic 已经固化在硬件/固件（GSP）里，driver 主要是：建好 TSG，往对应的 runlist 写 entry，调一些寄存器

可以在 driver 里**怎么构造 runlist**，比如：
- 给高优先级 TSG 多个 entry + 大 timeslice
- 动态删/加 TSG entry（见 GCAPS 做法）

但没办法重写"GPU 硬件如何在 runlist 上走步"和"TSG 内部 channel 轮询方式"。那些是在硬件 host scheduler 里的。

**3. 跨 kernel 的"任意时间点 preempt"**

compute preemption 受限于硬件：通常在 thread-block 甚至 wavefront 颗粒；在 driver 里能做的只是：
- tweak timeslice，让某个 TSG 每 X μs 被切出去
- 触发 context switch 或移除某个 TSG 的 runlist entry

真要"到任意指令点 preempt 一个 kernel"几乎是不可能的，除非换架构或者在应用层自己分块 kernel（这已经是编译/程序结构层面的事了）。

### 4.3 AMD：改 queue / user-mode queue 创建能做到什么？

#### 4.3.1 调度结构

AMD 文档说得还更公开一点：
- 每个 engine 支持多个 **user queues**，由 scheduling firmware 动态映射到有限数量的 **hardware queue slots（HQD）**
- 当 user queues 数量超过硬件 slot，就按"priority + time quanta"做动态 map/unmap
- 2024 的 Micro Engine Scheduler (MES) 规范明确：
  - 一级调度在 firmware：决定哪些 application queue 映射到哪个 HW queue
  - 二级调度在 **Queue Manager 硬件**：在"ready 的 HW queue"之间按优先级/quantum 仲裁

内核里 KFD 接口也明说了调度模式：
- `KFD_SCHED_POLICY_HWS`：用 firmware (CP) 调度 user-mode queues 和 HIQ/DIQ，允许 oversubscription
- `KFD_SCHED_POLICY_HWS_NO_OVERSUBSCRIPTION`：同上，但禁用 oversubscription
- `KFD_SCHED_POLICY_NO_HWS`：完全不用硬件调度，由 driver 直接设置 CP 寄存器手动搞 queue（官方说仅供调试）

#### 4.3.2 在 queue 创建路径能调的

**1. queue priority / quantum / sched policy**

在创建 user-mode queue（AQL queue）或 kernel queue 时，driver 填 MQD，里面带了 priority、timeslice 等字段，MES firmware 按这些做映射和仲裁。

可以在 kfd_device_queue_manager / queue 创建逻辑里：
- 根据进程/任务优先级决定 queue priority
- 对实时任务把调度模式切到 `HWS_NO_OVERSUBSCRIPTION`，保证不会把一堆队列挤在同一个 HW queue 上
- 对 best-effort 任务允许 oversub，甚至故意给它一个低 priority + 小 quantum

**这可以实现跨进程的静态 QoS / 优先级调度。**

**2. 到底用 user queue 还是 kernel queue**

某些 IP（比如 VCN/JPEG）可能不支持 user queue，只能走 kernel queue；GFX/compute 则可以二选一。userq 文档提到：kernel queues 永远 mapped，在 HQD 竞争里优先于 user queues。

可以在 queue 创建路径里决定：
- 关键队列走 kernel queue（抢占性更强）
- 其他走 user queue（由 MES 管）

**3. 动态限制某进程的 queue 数量、类型**

类似 NVIDIA 的 channel 数控制，可以在 KFD 层限制每个进程的 queue 数：
- 降低某类任务的并发度，把压力留给高优先级任务
- 极端一点直接拒绝 queue 创建，达到 "soft admission control"

**4. 必要时切到 `NO_HWS`，自己实现调度（硬核玩法）**

KFD 提供了一个 `KFD_SCHED_POLICY_NO_HWS` 模式：禁用 firmware 调度，让 driver 直接操作 CP 寄存器装/卸 queue。

这个模式 AMD 自己说"只用于 debug"，但从架构角度，要搞**完全自定义调度**，这是唯一正道：
- 自己维护一个队列集合
- 按你的算法（priority/EDF/TP 等）决定哪几个 queue 被 load 到 HQD
- 主动换出/换入 HQD 相当于 context switch

缺点很现实：
- 所有 MES 提供的现成逻辑都得自己重实现
- 容易搞挂机，官方也不会给你任何 support

#### 4.3.3 不能指望的

和 NVIDIA 一样，光改 queue 创建：
- 控制的是"**哪些 queue 存在 + 什么属性**"，而不是"MES/queue manager 如何在 ready 的 HW queues 之间走步骤"
- MES firmware 里面的算法你动不了，只能通过 priority/quantum 这种"hint"间接影响

想对单个 kernel 的提交顺序、精确抢占点做控制，同样需要：要么重写 user runtime，要么自己实现 `NO_HWS` 模式下的完整 scheduler。

---

## 5. CUDA 用户态 API 的调度能力（极其有限）

### 5.1 CUDA 用户态真正能碰到的"调度相关"东西

#### 5.1.1 流优先级（stream priority）

唯一比较接近的是：
```c
cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority);
// driver API 对应的 cuStreamCreateWithPriority
```

这些做的事情：**在一个进程内部，为不同 stream 标一个 priority**。

实际效果：
- 文档说高优先级 stream 的 work 会优先于低优先级 stream 执行，特别是在有多个 stream 竞争时
- 但优先级如何映射到硬件层（TSG 内部的 channel 调度 vs SM 上的 warp 调度）完全没保证
- 更重要：这是进程内的；对"多个进程之间的 GPU 调度"基本没有保障，更别说 TSG/runlist 层次

**换句话讲：stream priority 是一个很弱、很局部的 hint，远远达不到 OS 级 GPU scheduler 的粒度。**

#### 5.1.2 上下文创建时的一点调度 flag（其实是 CPU 侧）

driver API 有几个 context 创建的 flag，比如：
- `CU_CTX_SCHED_AUTO`
- `CU_CTX_SCHED_SPIN`
- `CU_CTX_SCHED_YIELD`
- `CU_CTX_SCHED_BLOCKING_SYNC`

这些控制的是：**调用 CUDA API 的 host 线程在等待 GPU 完成时如何调度（spin/yield/blocking）**，跟 GPU 上下文调度基本无关。

它不会影响 GPU channel/TSG 在 runlist 上怎么排，更不会改 timeslice 或 preemption。

**所以别指望通过改这些 flag 做 GPU 调度。**

#### 5.1.3 MPS / MIG 类的"粗粒度资源划分"

NVIDIA 给云那套提供了一些侧面工具：
- **MPS**：多个进程共享一个 server 进程的 GPU context，有环境变量/配置可以限制每个 client 的并发 kernel 数、active thread block 限额等
- **MIG**（多实例 GPU）：把整个 GPU 切成几个硬分区，每个看成一块独立 GPU 设备

这些东西：
- 在 API 层面基本只是"多了几个 device / 多个 context"，通过 `cudaSetDevice` / `cuDeviceGet` 选择
- 真正的资源划分/调度策略是由 driver + firmware 决定的，在 CUDA API 里改不了 timeslice、runlist，甚至看不到 TSG 的影子

如果把 "MIG/MPS 配置 + per-process device 绑定"也算"调度"，那 CUDA 能做一点，但它是**非常粗、静态的资源切分**，不是内核提交级 Schedule Control。

### 5.2 CUDA 用户态**没有**暴露的东西

下面这些都是想控制的点，在公开 CUDA API 里**完全没有**：

1. **TSG / channel / runlist 结构本身**
   - 没有任何 API 能创建/销毁/重排 TSG
   - 不能控制 TSG 在 runlist 里出现几次、timeslice 是多少
   - 不能手动 preempt 某个 TSG 或 context

2. **GPU context 级别的 timeslice / preemption 模式**
   - 没有 "setContextTimeslice(ctx, μs)" 这种东西
   - 没有 "setPreemptionMode(ctx, GFXP/CILP)" 的 API
   - 这些全在 RM / 内核/固件那一层决定

3. **跨进程的 GPU 优先级 / QoS**
   - 无法在 user-level 用 CUDA 说："这个进程里的所有 kernel 都比另一个进程高一个 priority class"
   - 也没有 per-process 的配额/权重 API（除了 MPS 那套特定部署配置，不是 CUDA 程序本身能 portable 调的）

4. **对单个 kernel submit 的调度控制**
   - 没有 API 可以说"这个 kernel 以 deadline D 运行"、"这个 kernel 可以被谁 preempt"、"这个 kernel 必须先于另一个进程的 kernel 执行"等
   - 看到的 `cudaDeviceSetLimit` / `cudaFuncSetAttribute` 等都是资源/属性 hint，不是调度

**整体上，CUDA 把 GPU 调度当成"实现细节"，不想给你任何稳定接口。对 NVIDIA 来说，这一层逻辑有一部分在闭源 user-mode driver，一部分在内核，一部分已经固在硬件/固件里。**

### 5.3 AMD/ROCm/HIP 的对比

- HIP 也有 `hipDeviceGetStreamPriorityRange` / `hipStreamCreateWithPriority` 这一类
- HSA/ROCm 有一些 queue priority / sched policy 的概念（特别是在 KFD / MES 文档里），但用户态 runtime API 封装得比较死，正常 HIP 程序也碰不到"driver 层看到的那堆 MQD/HQD/MES 调度策略"的接口
- 真要调度还是得往 KFD/ioctl 走

也就是说，在"user-level runtime API"这一层，两家都只给你一些**很弱的 priority hint**和**粗粒度的资源切分**，真正有意思的调度控制点都被藏在 driver+firmware 里。

### 5.4 用户态可以利用什么？

从"deployment + 研究故事"的角度来看，用户态这边最多可以：

1. **流级别的逻辑调度 + 用户态 runtime**
   - 在自己的 runtime 里把 kernel 分成段（小 kernel / micro-batch），基于 stream priority / 多 stream 做一层"伪调度"
   - 但是这是**一个进程内部**的，只能玩 intra-context 的策略

2. **用户态调度 + 内核调度协同**
   - 搞一个用户态 daemon（或 eBPF 告警），决定哪个任务应该激活
   - 通过 ioctl 把这个决策告诉 driver（enable/disable 某些 TSG / queue、改 priority / timeslice）
   - CUDA 程序本身只需要 link 一个小 shim，在 critical section 里发 ioctl（类似 GCAPS 那套 "GPU segment begin/end" 报告）

3. **完全自带一个 GPU runtime（极端方案）**
   - 不用官方 `libcuda` 做调度，自己实现中间层 runtime，把应用 kernel 调用映射成自己的 command stream
   - 然后在 driver 里给这套 runtime 配一个专门 TSG / runlist 策略
   - 但这已经是"写一套半自制 CUDA"的级别，很重

---

## 6. 技术路线建议

### 6.1 如果真要搞"可控调度"，一个现实的技术路线

结合上面这些约束，一个比较合理的工程路线（也是 GCAPS/TimeWall 之类论文采用的模式）是：

**1. 在 driver 的 channel/TSG/queue 创建路径贴上足够多的 metadata**
- 任务 ID、优先级、deadline（如果想玩 EDF / TP）
- 所属 cgroup / 进程类型
- 在 NVIDIA 上设置初始 TSG timeslice、runlist interleave、preemption mode
- 在 AMD 上设置 queue priority 和 sched policy

**2. 在 runlist / queue scheduling 逻辑里实现自定义策略**
- **NVIDIA**：仿 GCAPS，在内核维护"GPU-using tasks 表 + 对应 TSG 集合"，在默认的"时间片轮转"基础上，按优先级/任务状态动态删/加 TSG runlist entries，实质上实现 priority-based preemptive scheduling
- **AMD**：要么利用 MES 给的 priority/quantum，做相对温和的 QoS；要么硬核地切到 `NO_HWS` 自己 load/unload HQD

**3. 给用户态一个简洁的"GPU 段标注接口"**
- 像 GCAPS 那样，在 GPU 使用段开始/结束插宏，宏里发一个 ioctl 通知 driver
- driver 用自己的算法抢占/恢复 TSG/queue

**4. 把 channel/queue 创建改造为"调度系统的一部分"，而不是唯一入口**
- channel/queue 创建负责"登记任务 + 设置静态属性"
- 真正的调度控制在 runlist / HQD 的维护逻辑里完成
- 这两块协同，才有可能写出一个说得出口的 GPU scheduler（无论是 RT 的，还是 QoS 的）

### 6.2 能力总结表

| 想控制的调度属性 | CUDA API 能否直接做 | 在哪一层可以 hook |
|-----------------|-------------------|-----------------|
| 单个 kernel 的执行顺序 | ❌ 不能 | 用户态 runtime（自己拦截/重排） |
| 进程内 stream 优先级 | ⚠️ 很弱的 hint | CUDA stream priority API |
| 跨进程 GPU context 优先级 | ❌ 不能 | Driver: TSG priority / runlist interleave |
| GPU context timeslice | ❌ 不能 | Driver: TSG timeslice 设置 |
| Preemption 模式 | ❌ 不能 | Driver: preemption type (GFXP/CILP) |
| Per-process GPU 配额 | ❌ 不能 | Driver: admission control + channel 限制 |
| 系统级公平性 / SLO | ❌ 不能 | Driver: 动态调整 runlist + preemption |
| 强制挂起/恢复任意进程 | ❌ 不能 | Driver: 从 runlist 删除/添加 TSG |

### 6.3 结论

如果目标是像 OSDI/SOSP 那种：
- "GPU 作为 OS 资源"
- "GPU scheduling substrate / programmable control plane"
- "系统级公平性 / SLO / 多租户隔离"

那么：

**只在用户态做**：
- 工程上省心、部署故事好讲
- 但论文上上限就是"smart runtime / cluster manager"
- 很难说服大家你改变了 GPU 的 OS abstraction

**改 driver（哪怕量不大）+ 用户态 policy**：
- 可以 claim "first-class resource management"
- 给出强的 invariants 和 adversarial eval
- 再把 policy 做成 eBPF/gBPF，这正好接上 programmable GPU story

**一句话**：
> 从"论文能说的话"来看，driver 是拿到 **control & enforcement** 的唯一入口；
> 只在用户态做，顶多是"调戏厂商调度器"，永远在它的影子下活动。

---

## 参考文献

1. GCAPS: GPU Context-Aware Preemptive Priority-Based Scheduling for Real-Time Tasks (ECRTS 2024)
2. Hardware Compute Partitioning on NVIDIA GPUs for Composable Systems (ECRTS 2025)
3. Unleashing the Power of Preemptive Priority-based Scheduling for Real-Time GPU Tasks (arXiv 2024)
4. Tegra GPU Scheduling Improvements (NVIDIA Docs)
5. User Mode Queues — The Linux Kernel documentation (AMD)
6. kgd_kfd_interface.h (Linux kernel source)
