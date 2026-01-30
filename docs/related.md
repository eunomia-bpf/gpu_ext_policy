我又把之前那份“bug/verification 点”的清单重新做了一次 **去重 + 补全（重点扫开源项目）+ CPU vs GPU 归类**。下面是我建议你在 paper 里使用的一套 **canonical taxonomy**（英文列表），同时我会先把“重复/需要修正”的地方说清楚，再给出补上的遗漏项与每项的 gpu_ext 用法。

## 0. 去重与需要修正的点（你可以直接放到 rebuttal/related work 里）

1. **“deadlock” vs “barrier divergence”重复**

   * barrier divergence（在 `__syncthreads()`/barrier 处线程分歧）本质上是 GPU 上最常见、最典型的 deadlock/liveness failure 诱因之一；所以建议：
   * **单列 “Barrier Divergence / Invalid Barrier Usage”** 作为 GPU 特有、可直接导致 hang 的 liveness bug；
   * 再单列 “Other Deadlocks / Liveness Bugs（locks/spin/cross-scope sync）”。
     论文与工具里也常这样分：GPUVerify 把“barrier divergence”当成核心 defect；AuCS/Wu/Simulee 也把它当作主要 sync bug 类型之一。([Nathan Chong][1])

2. **“uncoalesced” vs “non-coalesceable memory access”重复（同义项）**

   * GPUDrano 叫 *uncoalesced memory accesses*；GPUCheck 叫 *non‑coalesceable memory accesses*。本质都是：warp 内地址模式导致 memory transaction 被拆成很多次。([GitHub][2])
   * 建议合并成一个条目：**Uncoalesced / Non‑coalesceable Memory Access Patterns**。

3. **“data race”条目需要拆成 3 个子类，否则会混**

   * (a) **shared memory（块内共享内存）race**：Compute Sanitizer 的 racecheck 主要就盯这个。([NVIDIA Docs][3])
   * (b) **global memory race**：ScoRD/iGUARD 强调很多 detector 反而忽略 global memory races，且 GPU 上还牵涉 *scope* 概念（block/device/system）。([CSA - IISc Bangalore][4])
   * (c) **host↔device async race**：CuSan 专门针对“异步 CUDA 调用与 host 之间的数据竞争”。([GitHub][5])
     这三类都叫 race，但对 gpu_ext 的 verifier/linter 规则完全不同，必须拆开。

4. **“同步 bug”条目重复：应当用 AuCS/Wu 的三分法作为主干**

   * AuCS 明确把 CUDA sync bugs 的“大头”总结成三类：**data race / barrier divergence / redundant barrier**。([Shinhwei][6])
   * 你 paper 里可以先用这个三分法做“审稿人可理解的主干”，然后再扩展到 scope/ITS/CG 等“新 CUDA 特性引入的新 bug”。iGUARD 就是专门补这些“新特性 bug”。([Aditya K Kamath][7])

---

## 1. Canonical bug list (dedup + filled from OSS scan)

下面是 **英文**列表。每一项我都给：**(i) what it is + why it matters**、**(ii) gpu_ext should use it how**。同时我在标题里标注：

* **CPU-shared**：CPU 上也有同类问题
* **GPU-specific**：GPU/SIMT 独特语义
* **GPU-amplified**：CPU 也有但在 GPU 上更容易放大为 device/tenant 级事故（hang/DoS 级 tail latency）

---

### 1) Barrier Divergence at Block Barriers (`__syncthreads`) — GPU-specific, GPU-amplified (liveness)

* **What it is / why it matters.**
  A block-wide barrier requires *all* threads in the block to reach it. If the barrier is placed under a condition that evaluates differently across threads, some threads wait forever → deadlock / kernel hang. This is treated as a first-class defect in GPU kernel verification (e.g., "barrier divergence" in GPUVerify), and is also one of the main CUDA synchronization bug types characterized/targeted by AuCS/Wu. Tools like Compute Sanitizer `synccheck` report "divergent thread(s) in block" for this pattern; Oclgrind can also detect barrier divergence (OpenCL).([Nathan Chong][1])

* **Bug example.**

```cuda
__global__ void k(float* a) {
  if (threadIdx.x < 16) __syncthreads(); // divergent barrier => UB / deadlock
  a[threadIdx.x] = 1.0f;
}
```

* **How gpu_ext should use it.**
  Make this a *hard* verifier rule: gpu_ext policy code must not contain any block-wide barrier primitive (or any helper that can implicitly behave like a block-wide barrier). If you ever allow barriers in policy code, require **warp-/block-uniform control flow** for any path reaching a barrier (uniform predicate analysis), otherwise reject.

---

### 2) Invalid Warp Synchronization (`__syncwarp` mask, warp-level barriers) — GPU-specific

* **What it is / why it matters.**
  Warp-level sync requires correct participation masks. A common failure is calling `__syncwarp(mask)` where not all lanes that reach the barrier are included in `mask`, or where divergence causes only a subset to arrive. `synccheck` explicitly reports “Invalid arguments” and “Divergent thread(s) in warp” classes for these hazards, and iGUARD discusses how newer CUDA features (e.g., independent thread scheduling + cooperative groups) create new race/sync hazards beyond the classic model.([NERSC Documentation][8])
* **How gpu_ext should use it.**
  If gpu_ext policies can ever emit warp-level sync or cooperative-groups barriers, require a *verifiable* mask discipline: e.g., only `__syncwarp(0xffffffff)` (full mask) or masks proven to equal the active mask at the callsite. Otherwise, simplest is: **ban warp sync primitives entirely** inside policies.

---

### 3) Shared-Memory Data Races (`__shared__`) — CPU-shared, GPU-amplified

* **What it is / why it matters.**
  Threads in a block access on-chip shared memory concurrently; missing/incorrect synchronization causes races. This is a classic CUDA bug class (AuCS/Wu), and vendor tooling targets it directly: Compute Sanitizer `racecheck` is a runtime shared-memory hazard detector. GPUVerify also aims to prove race-freedom (including barrier-related races).([Shinhwei][6])

* **Bug example.**

```cuda
__global__ void k(int* g) {
  __shared__ int s;
  int t = threadIdx.x;
  if (t == 0) s = 1;
  if (t == 1) s = 2;   // write-write race on s
  __syncthreads();
  g[t] = s;
}
```

* **How gpu_ext should use it.**
  If policies have any shared state, require **warp-uniform side effects** or **single-lane side effects** (e.g., lane0 updates) plus explicit atomics. A conservative verifier rule is: policy code cannot write shared memory except via restricted helpers that are race-safe (e.g., per-warp aggregation).

---

### 4) Global-Memory Data Races + Scoped Race Bugs + Warp-divergence Race — CPU-shared, GPU-specific semantics

* **What it is / why it matters.**
  Races on global memory exist, but GPU adds *scope* and memory-model subtleties: "scoped races" can occur when synchronization/atomics are done at an insufficient scope. ScoRD explicitly argues that many GPU race detectors focus on shared memory and ignore global-memory races, and introduces *scoped races* due to insufficient scope. iGUARD further targets races introduced by "scoped synchronization" and advanced CUDA features.([CSA - IISc Bangalore][4])

  A "warp-divergence race" (as described in GKLEE) is a GPU-specific phenomenon where **divergence changes which threads are effectively concurrent**, producing racy outcomes that don't map cleanly to CPU assumptions. This is one of the reasons "CPU-style race reasoning" doesn't port directly: SIMT execution order + reconvergence can create subtle concurrency patterns.([Lingming Zhang][18])

* **Bug example (scoped race).**

```cuda
// Scoped race: using block-scope atomic when device-scope is needed
__global__ void k(int* counter) {
  atomicAdd_block(counter, 1);  // only block-scope, may race across blocks
}
```

* **Bug example (warp-divergence race).**

```cuda
__global__ void k(int* A) {
  int lane = threadIdx.x & 31;
  if (lane < 16) A[0] = 1;      // first half writes
  else           A[0] = 2;      // second half writes
  // outcome depends on SIMT execution + reconvergence
}
```

* **How gpu_ext should use it.**
  Treat scope as part of the verifier contract: if policies do atomic/synchronizing operations, require the *strongest* allowed scope (or forbid nontrivial scope usage). Practically: ban cross-block shared global updates unless they're done through a small set of "safe" helpers (e.g., per-SM/per-warp buffers → host aggregation). For warp-divergence races, require that any helper with side effects is guarded by a **warp-uniform predicate** or executed only by a designated lane (e.g., lane0).

---

### 5) Host ↔ Device Asynchronous Data Races (API ordering bugs) — CPU-shared-ish, GPU-specific in practice

* **What it is / why it matters.**
  CUDA exposes async kernel launches/memcpy/events; host code can race with device work if synchronization is missing. CuSan is an open-source detector for “data races between (asynchronous) CUDA calls and the host,” using Clang/LLVM instrumentation plus ThreadSanitizer. This is a major real-world bug source in heterogeneous programs and is *not* covered by pure kernel-only verifiers.([GitHub][5])
* **How gpu_ext should use it.**
  If gpu_ext policies interact with host-visible buffers or involve asynchronous map copies, define a strict **lifetime & ordering contract** (e.g., “policy writes are only consumed after a guaranteed sync point”). For testing, integrate CuSan into CI for host-side integration tests of the runtime/loader.

---

### 6) Deadlocks Beyond Barrier Divergence (locks/spin + SIMT lockstep + named-barrier misuse) — CPU-shared, GPU-amplified (+ sometimes GPU-specific)

* **What it is / why it matters.**
  Besides barrier divergence, SIMT lockstep can create deadlocks in patterns that are unusual on CPUs. iGUARD notes that lockstep execution can deadlock if threads within a warp use distinct locks—something not possible in typical CPU threading models. GKLEE also reports finding deadlocks via symbolic exploration of GPU kernels. ESBMC-GPU models and checks deadlock too.([Aditya K Kamath][7])

  Warp-specialized kernels often use **named barriers** or structured synchronization patterns between warps/roles (producer/consumer). Bugs include: (a) deadlock, (b) unsafe barrier reuse ("recycling") across iterations, (c) races between producers/consumers. WEFT addresses exactly these properties and verifies deadlock freedom, safe barrier recycling, and race freedom for producer-consumer synchronization.([zhangyuqun.github.io][19])

* **Bug example (spin deadlock).**

```cuda
__global__ void k(int* flag, int* data) {
  // Block 0 expects Block 1 to set flag, but no global sync exists
  if (blockIdx.x == 0) while (atomicAdd(flag, 0) == 0) { }  // may spin forever
  if (blockIdx.x == 1) { data[0] = 42; /* forgot to set flag */ }
}
```

* **Bug example (named-barrier misuse, sketch).**

```cuda
// Producer writes buffer then signals barrier B
// Consumer waits on B then reads buffer
// Bug: consumer waits on wrong barrier instance / reused incorrectly in loop
```

* **How gpu_ext should use it.**
  Ban blocking primitives in policy code (locks, spin loops, waiting on global conditions). Add a verifier rule: **no unbounded loops / no "wait until" patterns**. If you absolutely need synchronization, force "single-lane, nonblocking" patterns and bounded retries. Policies must not interact with named barriers (no waits, no signals).

---

### 7) Kernel Non-Termination / Infinite Loops — CPU-shared, GPU-amplified

* **What it is / why it matters.**
  Infinite loops can hang GPU execution and are highlighted as GPU-specific bug concerns in GPU debugging work: CL-Vis explicitly calls out infinite loops (together with barrier divergence) as GPU-specific bug types to detect/handle. In practice, non-termination is especially dangerous because GPU preemption/recovery can be coarse.([Computing and Informatics][9])
* **How gpu_ext should use it.**
  This is where “bounded overhead = correctness” is easiest to justify: enforce a **strict instruction/iteration bound** for policy code (like eBPF on CPU). If policies may contain loops, require compile-time bounded loops only, with conservative upper bounds.

---

### 8) Memory Safety: Out-of-Bounds / Misaligned / Use-After-Free / Use-After-Scope — CPU-shared

* **What it is / why it matters.**
  Classic memory safety includes both **spatial** (OOB, misaligned) and **temporal** (UAF, UAS) violations. Compute Sanitizer `memcheck` precisely detects OOB/misaligned accesses (and can detect memory leaks), and Oclgrind reports invalid memory accesses in its simulator. ESBMC-GPU also checks pointer safety and array bounds as part of its model checking. GKLEE's evaluation includes out-of-bounds global memory accesses as error cases.([NVIDIA Docs][3])

  Temporal bugs exist on GPUs too: pointers can outlive allocations (host frees while kernel still uses, device-side stack frame returns, etc.). cuCatch explicitly targets temporal violations using tagging mechanisms and discusses use-after-free and use-after-scope detection.([d1qx31qr3h6wln.cloudfront.net][20])

* **Bug example (OOB).**

```cuda
__global__ void k(float* a, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  a[tid + 1024] = 0.0f;   // OOB write
}
```

* **Bug example (Use-After-Scope).**

```cuda
__device__ int* bad() {
  int local[8];
  return local;          // returns pointer to dead stack frame (UAS)
}
__global__ void k() {
  int* p = bad();
  int x = p[0];          // UAS read
}
```

* **How gpu_ext should use it.**
  This is the "classic verifier" portion: keep eBPF-like pointer tracking, bounds checks, and restricted helpers. Ideally: policies cannot allocate/free; all policy-visible objects are managed by gpu_ext runtime and remain valid across policy execution (no UAF/UAS by construction). Also add a testing story: run policy-enabled kernels under Compute Sanitizer memcheck in CI for regression.

---

### 9) Uninitialized Global Memory Reads — CPU-shared

* **What it is / why it matters.**
  Compute Sanitizer `initcheck` reports cases where device global memory is accessed without being initialized by device writes or CUDA memcpy/memset. This is a frequent source of heisenbugs because GPU concurrency amplifies nondeterminism.([NVIDIA Docs][3])
* **How gpu_ext should use it.**
  If gpu_ext policies read from maps/buffers, require explicit initialization semantics (e.g., map lookup returns “not found” unless initialized; forbid reading uninitialized slots). In testing, run initcheck on representative workloads.

---

### 10) Resource Management: Memory Leaks / Incorrect Allocation / Lifecycle Bugs — CPU-shared

* **What it is / why it matters.**
  Compute Sanitizer memcheck includes leak checking (e.g., `cudaMalloc` leaks, device heap leaks) and can also report incorrect use of `malloc/free()` in kernels. These are availability issues in long-running services.([NVIDIA Docs][3])

  A lot of CUDA failures are not in kernel math but in lifecycle management: incorrect device allocation, memory leaks, early device reset calls, etc. Wu et al. highlight these as a major root-cause category ("improper resource management") and relate them to crashes.([arXiv][21])

* **Bug example (early reset).**

```cpp
cudaMalloc(&p, N);
kernel<<<...>>>(p);
cudaDeviceReset();     // early reset => invalidates work / crashes
```

* **How gpu_ext should use it.**
  Forbid dynamic allocation in policy code; if helpers allocate, require bounded allocations + automatic cleanup. Keep policy loading/unloading tied to safe lifecycle transitions (e.g., disallow unloading policies while kernels that might execute them are in flight). Treat leaks as "availability correctness," because persistent GPU agents/daemons can degrade over time.

---

### 11) Arithmetic Errors (overflow, division by zero) — CPU-shared

* **What it is / why it matters.**
  ESBMC-GPU explicitly lists arithmetic overflow and division-by-zero among the properties it checks for CUDA programs (alongside races/deadlocks/bounds). These errors can corrupt keys/indices and cascade into memory safety/perf disasters.([GitHub][10])
* **How gpu_ext should use it.**
  Optional but reviewer-friendly: add lightweight verifier checks for div-by-zero and dangerous shifts, and constrain pointer arithmetic (already typical in eBPF verifiers). For “perf correctness,” overflow in index computations is a common hidden cause of random/uncoalesced patterns.

---

### 12) Uncoalesced / Non‑Coalesceable Global Memory Access Patterns — GPU-specific (perf → bounded interference)

* **What it is / why it matters.**
  Warp memory coalescing is a GPU-specific performance contract. GPUDrano provides static analysis to find uncoalesced accesses and explains coalescing in terms of warps and adjacent addresses; GPUCheck focuses on non-coalesceable accesses caused by thread-divergent expressions; GKLEE also flags memory coalescing issues as performance defects.([GitHub][2])

* **Bug example.**

```cuda
__global__ void k(float* a, int stride) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[tid * stride];   // stride>1 => likely uncoalesced
  a[tid * stride] = x + 1.0f;
}
```

* **How gpu_ext should use it.**
  If you want "performance as correctness," this is a flagship rule: restrict policy memory ops to patterns provably coalesced (e.g., affine, lane-linear indexing with small stride), and/or require warp-level aggregation so only one lane performs global updates. For papers: cite GPUDrano/GPUCheck to justify why static analysis is needed.

---

### 13) Control-Flow Divergence (warp branch divergence) — GPU-specific (perf, and interacts with liveness)

* **What it is / why it matters.**
  SIMT divergence serializes paths within a warp. GPUCheck explicitly targets "branch divergence" as a performance problem arising from thread-divergent expressions; GKLEE also treats warp divergence as a performance defect. Divergence is also the root cause of barrier divergence when barriers are in conditional code.([WebDocs][11])

* **Bug example.**

```cuda
__global__ void k(float* out, float* in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if ((tid & 1) == 0) out[tid] = in[tid] * 2;
  else                out[tid] = in[tid] * 3;  // divergence within warp
}
```

* **How gpu_ext should use it.**
  Enforce **warp-uniform control flow** for policies (or at least for any code path that triggers side effects / heavy helpers). If you can't prove uniformity, force "single-lane execution" of policy side effects (others become no-ops) to prevent warp amplification.

---

### 14) Shared-Memory Bank Conflicts — GPU-specific (perf)

* **What it is / why it matters.**
  Bank conflicts are a shared-memory–specific performance pathology: accesses serialize when multiple lanes hit the same bank. GKLEE explicitly lists shared-memory bank conflicts among GPU performance defects it checks.([Peng Li's Homepage][12])

* **Bug example.**

```cuda
__global__ void k(int* out) {
  __shared__ int s[32*32];
  int lane = threadIdx.x & 31;
  // stride hits same bank pattern (illustrative)
  int x = s[lane * 32];
  out[threadIdx.x] = x;
}
```

* **How gpu_ext should use it.**
  If policies use shared scratchpads (e.g., per-block staging), restrict addressing patterns (e.g., padding, structure-of-arrays) or simply ban shared-memory indexing by untrusted lane-dependent expressions.

---

### 15) Redundant Barriers (unnecessary `__syncthreads`) — CPU-shared-ish, GPU-specific impact (perf)

* **What it is / why it matters.**
  AuCS/Wu include "redundant barrier" as a major synchronization bug class; it doesn't break correctness but can severely degrade performance. GPURepair tooling also exists to insert/remove barriers to fix races and remove unnecessary ones.([Shinhwei][6])

* **Bug example.**

```cuda
__global__ void k(int* out) {
  __shared__ int s[256];
  int t = threadIdx.x;
  s[t] = t;             // no cross-thread dependence here
  __syncthreads();      // redundant
  out[t] = s[t];
}
```

* **How gpu_ext should use it.**
  For gpu_ext, this supports your "performance = safety" story: even "correct" policies can be unacceptable if they introduce barrier overhead. The simplest policy-level stance: **ban barriers**; if helpers include barriers internally, you need cost models or architectural restrictions.

---

### 16) Configuration Sensitivity / Portability: Block-Size Dependence + Toolchain/Platform Variations — GPU-specific (correctness & tuning safety)

* **What it is / why it matters.**
  GPUDrano explicitly includes "block-size independence" analysis: changing block size (while keeping total threads) should not break program functionality, and this is essential for safe block-size tuning. This is a GPU-unique portability/correctness hazard.([GitHub][2])

  CUDA code can also fail or misbehave when moved across platforms (compiler versions, driver versions, GPU architectures). Wu et al. explicitly call this out as a root cause ("poor portability"), including cases where code uses deprecated intrinsics or assumes old shared-memory bank width that breaks on newer GPUs.([arXiv][21])

* **Bug example (block-size dependence).**

```cuda
__global__ void reduce(float* out, float* in) {
  // assumes gridDim.x == 1, but caller launches >1 blocks => wrong result / race
  // ...
}
```

* **How gpu_ext should use it.**
  If policy code assumes a particular block/warp mapping (e.g., keys use `threadIdx.x` directly), you can end up with correctness or performance regressions when kernels run under different launch configs. Add verifier rules that forbid hard-coded assumptions about blockDim/warp layout unless explicitly declared. Keep your verifier semantics architecture-aware (or restrict to a portable subset); if your policy ISA is PTX-level, pin the PTX version / codegen assumptions and validate them against target GPUs at load time.

---

### 17) Atomic Contention & Atomic-Scope Pitfalls — CPU-shared, GPU-amplified (perf → DoS), plus GPU-specific scope details

* **What it is / why it matters.**
  Heavy atomic contention is a classic "performance bug that behaves like a DoS" under massive parallelism. There is an open-source benchmark suite accompanying a 2025 paper on atomic contention, explicitly measuring atomic performance under contention and across different **memory scopes** (block/device/system) and access patterns.([GitHub][13])
* **How gpu_ext should use it.**
  Treat "atomic frequency + contention risk" as a verifier-enforced budget: e.g., allow at most one global atomic per warp, or require warp-aggregated updates. If policies use scoped atomics, require the scope to be explicit and conservative. For evaluation, you can reuse the open benchmark suite to calibrate "safe budgets" per GPU generation.

---

### 18) "Forgot Volatile" / Memory Visibility Pitfalls — GPU-specific (correctness)

* **What it is / why it matters.**
  GPU code often relies on compiler and memory-model subtleties. GKLEE reports a real-world category: forgetting to mark a shared memory variable as `volatile`, producing stale reads/writes due to compiler optimization or caching behavior. This is a GPU-flavored instance of memory visibility/ordering bugs that can be hard to reproduce.([Lingming Zhang][18])

* **Bug example.**

```cuda
__shared__ int flag;          // should sometimes be volatile / properly fenced
if (tid == 0) flag = 1;
__syncthreads();
while (flag == 0) { }         // may spin if compiler hoists load / visibility issues
```

* **How gpu_ext should use it.**
  Avoid exposing raw shared/global memory communication to policies; instead provide **helpers with explicit semantics** (e.g., "atomic increment" or "write once" patterns), and verify policies don't implement ad-hoc synchronization loops. Forbid spin-waiting on shared memory in policy code.

---

### 19) Multi-Tenant GPU Sharing: Lack of Fault Isolation for OOB — GPU-specific (security/availability)

* **What it is / why it matters.**
  In spatial sharing (streams/MPS), kernels share a GPU address space. An OOB access by one application can crash other co-running applications (fault isolation issue). Guardian's motivation explicitly calls out this problem and designs PTX-level fencing + interception as a fix.([arXiv][22])

* **Bug example (conceptual).**

```cuda
// Tenant A kernel writes OOB and corrupts Tenant B memory in same context.
```

* **How gpu_ext should use it.**
  This directly supports your "availability is correctness" story: if gpu_ext policies run in privileged/shared contexts, you must prevent policy code from generating OOB accesses. Either: (a) only allow map helpers (no raw memory), or (b) instrument policy memory ops with bounds checks (Guardian-style PTX rewriting).

---

### 20) Cross-Kernel Interference Channels — GPU-specific (performance as security/predictability)

* **What it is / why it matters.**
  In concurrent GPU usage, contention for shared resources makes execution time unpredictable. "Making Powerful Enemies on NVIDIA GPUs" explicitly studies **interference channels** and how adversarial "enemy" kernels can amplify slowdowns to stress worst-case execution times. This is the strongest literature anchor for the argument that performance interference is a *system-level safety* property when GPUs are shared.

* **Bug example (conceptual).**

```cuda
// Kernel A is "victim"
// Kernel B is "enemy" stressing cache/DRAM/SM resources => tail latency explosion
```

* **How gpu_ext should use it.**
  Add a verifier contract like: "policy executes in O(1) helper calls, O(1) global memory ops, no blocking, warp-uniform side effects." Then you can argue (a) no hangs, and (b) bounded added contention footprint—consistent with your multi-tenant threat model.

---

## 2. 你问的"哪些 CPU 也有 vs GPU 独特/更严重"：一句话结论

* **CPU 也有（但 GPU 上更严重/更难 debug）**
  data races、deadlocks、OOB/invalid pointer、uninitialized reads、memory leaks、arithmetic overflow/div0、non-termination。ESBMC-GPU/Compute Sanitizer/（部分）GPUVerify/GKLEE 都覆盖这些方向。([GitHub][10])

* **GPU/SIMT 独特语义（paper reviewer 最吃的“为什么需要 GPU-specific verification”）**
  barrier divergence、warp-sync mask/warp divergence around barriers、memory coalescing/uncoalesced、shared-memory bank conflicts、block-size dependence、scoped synchronization（scope 不足导致 scoped races）、independent thread scheduling/cooperative groups 引入的新 sync/race hazard。([Nathan Chong][1])

* **GPU 上“性能问题≈正确性问题（bounded interference）”最有说服力的几类**
  uncoalesced/non-coalesceable、warp divergence、atomic contention、(redundant) barriers —— 因为这些会被 SIMT/海量线程数放大成几十倍甚至 DoS 级尾延迟（这正是你 gpu_ext 想强调的 contract）。对应工具/论文链条在 GPUDrano/GPUCheck/GKLEE + 2025 atomic contention benchmark + AuCS/Wu（redundant barrier）里比较齐。([GitHub][2])

---

## 3. 开源项目补全清单（之前最容易漏、但对 gpu_ext 很好用）

这些是我这次扫开源补上的（你可以作为 artifact/related-work 的“tooling landscape”段落）：

* **GPUVerify (OSS)**: static verification for race- and divergence-freedom of CUDA/OpenCL kernels.([GitHub][14])
* **GKLEE (has OSS forks)**: symbolic/concolic-style exploration; reports OOB, deadlock, plus performance defects like coalescing/bank conflicts/warp divergence.([Peng Li's Homepage][12])
* **GPUDrano (OSS)**: static analysis for uncoalesced accesses + block-size independence.([GitHub][2])
* **ESBMC-GPU (OSS)**: model checking for data race, deadlock, pointer safety, array bounds, overflow, div0, assertions.([GitHub][10])
* **CuSan (OSS)**: detects data races between async CUDA calls and host via Clang/LLVM + ThreadSanitizer.([GitHub][5])
* **CUDA-Data-Race-Detector (OSS)**: dynamic device-side data race detector (NVBit-based).([GitHub][15])
* **Oclgrind (OSS, OpenCL)**: simulator/debugger that detects memory access errors, data races, barrier divergence; useful做“GPU bug 类型示例 + 教学 artifact”。([GitHub][16])
* **GPUAtomicContention (OSS benchmarks)**: atomic contention & scopes microbench suite (PLOS’25).([GitHub][13])
* **GPURepair (OSS)**: automated repair focusing on races/barrier divergence (uses GPUVerify).([GitHub][17])

---

如果你下一步要把这些变成 **paper 的 verification story**，最强的结构通常是：

1. 先用 AuCS/Wu 的三大类（race/divergence/redundant barrier）建立“GPU sync bugs 的经典主干”，再把 Compute Sanitizer 的 synccheck/racecheck 当成“工业界 corroboration”；([Shinhwei][6])
2. 再用 ScoRD/iGUARD 把“新 CUDA 特性（scope、ITS、CG）→ 新 bug”讲清楚，强调 CPU eBPF verifier 完全没建模这些；([CSA - IISc Bangalore][4])
3. 最后用 GPUDrano/GPUCheck/GKLEE + atomic contention benchmarks 把“性能 = bounded interference correctness”落地成可引用的分析/检测证据链。([GitHub][2])

如果你愿意，我也可以把上面 17 条再进一步压缩成你 paper 里的一张 **“GPU-specific failure modes ↔ verifier rule mapping”**（每条一行、可直接贴进设计章节），并且把“CPU-shared vs GPU-specific”用更审稿友好的措辞（threat model/contract）写成 2–3 段英文。

[1]: https://nchong.github.io/papers/oopsla12.pdf "https://nchong.github.io/papers/oopsla12.pdf"
[2]: https://github.com/upenn-acg/gpudrano-static-analysis_v1.0 "https://github.com/upenn-acg/gpudrano-static-analysis_v1.0"
[3]: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html "https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html"
[4]: https://www.csa.iisc.ac.in/~arkapravab/papers/isca20_ScoRD.pdf "https://www.csa.iisc.ac.in/~arkapravab/papers/isca20_ScoRD.pdf"
[5]: https://github.com/tudasc/cusan "https://github.com/tudasc/cusan"
[6]: https://www.shinhwei.com/cuda-repair.pdf "https://www.shinhwei.com/cuda-repair.pdf"
[7]: https://akkamath.github.io/files/SOSP21_iGUARD.pdf "https://akkamath.github.io/files/SOSP21_iGUARD.pdf"
[8]: https://docs.nersc.gov/tools/debug/compute-sanitizer/ "https://docs.nersc.gov/tools/debug/compute-sanitizer/"
[9]: https://cai.type.sk/content/2019/1/cl-vis-visualization-platform-for-understanding-and-checking-the-opencl-programs/4318.pdf "https://cai.type.sk/content/2019/1/cl-vis-visualization-platform-for-understanding-and-checking-the-opencl-programs/4318.pdf"
[10]: https://github.com/ssvlab/esbmc-gpu "https://github.com/ssvlab/esbmc-gpu"
[11]: https://webdocs.cs.ualberta.ca/~amaral/thesis/TaylorLloydMSc.pdf "https://webdocs.cs.ualberta.ca/~amaral/thesis/TaylorLloydMSc.pdf"
[12]: https://lipeng28.github.io/papers/ppopp12-gklee.pdf "https://lipeng28.github.io/papers/ppopp12-gklee.pdf"
[13]: https://github.com/KIT-OSGroup/GPUAtomicContention "https://github.com/KIT-OSGroup/GPUAtomicContention"
[14]: https://github.com/mc-imperial/gpuverify "https://github.com/mc-imperial/gpuverify"
[15]: https://github.com/yinengy/CUDA-Data-Race-Detector "https://github.com/yinengy/CUDA-Data-Race-Detector"
[16]: https://github.com/jrprice/Oclgrind "https://github.com/jrprice/Oclgrind"
[17]: https://github.com/cs17resch01003/gpurepair "https://github.com/cs17resch01003/gpurepair"
[18]: https://lingming.cs.illinois.edu/publications/icse2020b.pdf "https://lingming.cs.illinois.edu/publications/icse2020b.pdf"
[19]: https://zhangyuqun.github.io/publications/ase2019.pdf "https://zhangyuqun.github.io/publications/ase2019.pdf"
[20]: https://d1qx31qr3h6wln.cloudfront.net/publications/PLDI_2023_cuCatch_2.pdf "https://d1qx31qr3h6wln.cloudfront.net/publications/PLDI_2023_cuCatch_2.pdf"
[21]: https://arxiv.org/pdf/1905.01833 "https://arxiv.org/pdf/1905.01833"
[22]: https://arxiv.org/pdf/2401.09290 "https://arxiv.org/pdf/2401.09290"
