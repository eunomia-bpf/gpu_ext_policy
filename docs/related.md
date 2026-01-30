

## 1. Canonical bug list (dedup + filled from OSS scan)


### 1) Barrier Divergence at Block Barriers (`__syncthreads`) — GPU-specific, GPU-amplified (liveness)

#### What it is / why it matters
  A block-wide barrier requires *all* threads in the block to reach it. If the barrier is placed under a condition that evaluates differently across threads, some threads wait forever → deadlock / kernel hang. This is treated as a first-class defect in GPU kernel verification (e.g., "barrier divergence" in GPUVerify), and is also one of the main CUDA synchronization bug types characterized/targeted by AuCS/Wu.

#### Bug example

```cuda
__global__ void k(float* a) {
  if (threadIdx.x < 16) __syncthreads(); // divergent barrier => UB / deadlock
  a[threadIdx.x] = 1.0f;
}
```

#### Seen in / checked by
  * GPUVerify: checking divergence is a core goal ("divergence freedom").([Nathan Chong][1])
  * Simulee detects **barrier divergence bugs** in real-world code.([zhangyuqun.github.io][19])
  * Wu et al.: explicitly defines barrier divergence and places it under improper synchronization.([arXiv][21])
  * Tools like Compute Sanitizer `synccheck` report "divergent thread(s) in block"; Oclgrind can also detect barrier divergence (OpenCL).

#### Checking approach
  * **Static check (GPUVerify-style):** prove that each barrier is reached by all threads in the relevant scope, often via uniformity reasoning.([Nathan Chong][1])
  * **Dynamic check:** synccheck-style runtime validation, and Simulee-style bug finding.([zhangyuqun.github.io][19])

#### How gpu_ext should use it
  Make this a *hard* verifier rule: gpu_ext policy code must not contain any block-wide barrier primitive (or any helper that can implicitly behave like a block-wide barrier). If you ever allow barriers in policy code, require **warp-/block-uniform control flow** for any path reaching a barrier (uniform predicate analysis), otherwise reject. Simplest and strongest: **forbid `__syncthreads()` inside policies** — this directly eliminates an entire class of GPU hangs.

---

### 2) Invalid Warp Synchronization (`__syncwarp` mask, warp-level barriers) — GPU-specific

#### What it is / why it matters
  Warp-level sync requires correct participation masks. A common failure is calling `__syncwarp(mask)` where not all lanes that reach the barrier are included in `mask`, or where divergence causes only a subset to arrive.

#### Bug example

```cuda
__global__ void k(int* out) {
  int lane = threadIdx.x & 31;
  if (lane < 16) {
    __syncwarp(0xffffffff);  // only 16 lanes arrive, but mask expects all 32
  }
  out[threadIdx.x] = lane;
}
```

#### Seen in / checked by
  * Compute Sanitizer `synccheck` explicitly reports "Invalid arguments" and "Divergent thread(s) in warp" classes for these hazards.([NERSC Documentation][8])
  * iGUARD discusses how newer CUDA features (e.g., independent thread scheduling + cooperative groups) create new race/sync hazards beyond the classic model.([Aditya K Kamath][7])

#### Checking approach
  * Runtime validation via `synccheck`.
  * Static analysis to verify mask correctness at each `__syncwarp` callsite.

#### How gpu_ext should use it
  If gpu_ext policies can ever emit warp-level sync or cooperative-groups barriers, require a *verifiable* mask discipline: e.g., only `__syncwarp(0xffffffff)` (full mask) or masks proven to equal the active mask at the callsite. Otherwise, simplest is: **ban warp sync primitives entirely** inside policies.

---

### 3) Scoped Synchronization Bugs + Warp-divergence Race — GPU-specific semantics

#### What it is / why it matters
  GPU adds *scope* and memory-model subtleties that don't exist on CPUs. **Scoped races** occur when synchronization/atomics are done at an insufficient scope (e.g., using `atomicAdd_block` when `atomicAdd` with device scope is needed). A **warp-divergence race** is a GPU-specific phenomenon where **divergence changes which threads are effectively concurrent**, producing racy outcomes that don't map cleanly to CPU assumptions. This is one reason "CPU-style race reasoning" doesn't port directly: SIMT execution order + reconvergence can create subtle concurrency patterns.

#### Bug example (scoped race)

```cuda
// Scoped race: using block-scope atomic when device-scope is needed
__global__ void k(int* counter) {
  atomicAdd_block(counter, 1);  // only block-scope, may race across blocks
}
```

#### Bug example (warp-divergence race)

```cuda
__global__ void k(int* A) {
  int lane = threadIdx.x & 31;
  if (lane < 16) A[0] = 1;      // first half writes
  else           A[0] = 2;      // second half writes
  // outcome depends on SIMT execution + reconvergence
}
```

#### Seen in / checked by
  * ScoRD introduces *scoped races* due to insufficient scope and argues this is a distinct bug class.([CSA - IISc Bangalore][4])
  * iGUARD further targets races introduced by "scoped synchronization" and advanced CUDA features (independent thread scheduling, cooperative groups).([Aditya K Kamath][7])
  * GKLEE explicitly lists "warp-divergence race" among discovered bug classes.([Lingming Zhang][18])
  * Simulee stresses CUDA-aware race definitions and discusses GPU-specific race interpretation constraints (e.g., avoiding false positives due to warp lockstep).([zhangyuqun.github.io][19])

#### Checking approach
  * **Scope verification:** ensure atomics/sync use sufficient scope for the access pattern.
  * **Verifier rule:** treat "lane-divergent side effects" as forbidden unless proven safe.
  * Require that any helper with side effects is guarded by a **warp-uniform predicate** or executed only by a designated lane (e.g., lane0). Then the verifier only needs to prove **uniformity** (or single-lane execution), not full SIMT interleavings.

#### How gpu_ext should use it
  Treat scope as part of the verifier contract: if policies do atomic/synchronizing operations, require the *strongest* allowed scope (or forbid nontrivial scope usage). Practically: ban cross-block shared global updates unless they're done through a small set of "safe" helpers (e.g., per-SM/per-warp buffers → host aggregation). If policies use scoped atomics, require the scope to be explicit and conservative.

---

### 4) Uncoalesced / Non‑Coalesceable Global Memory Access Patterns — GPU-specific (perf → bounded interference)

#### What it is / why it matters
  Warp memory coalescing is a GPU-specific performance contract. "Uncoalesced" accesses can cause large slowdowns (memory transactions split into many).

#### Bug example

```cuda
__global__ void k(float* a, int stride) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[tid * stride];   // stride>1 => likely uncoalesced
  a[tid * stride] = x + 1.0f;
}
```

#### Seen in / checked by
  * GPUDrano: "detects uncoalesced global memory accesses" and treats them as performance bugs.([GitHub][2], [CAV17][23])
  * GKLEE: reports "non-coalesced memory accesses" as performance bugs it finds.([Lingming Zhang][18])
  * GPUCheck: detects "non-coalesceable memory accesses."([WebDocs][11])

#### Checking approach
  * **Static analysis (GPUDrano/GPUCheck-style):** analyze address expressions in terms of lane-to-address stride; flag when stride exceeds coalescing thresholds.([CAV17][23])

#### How gpu_ext should use it
  If you want "performance as correctness," this is a flagship rule: restrict policy memory ops to patterns provably coalesced (e.g., affine, lane-linear indexing with small stride), and/or require warp-level aggregation so only one lane performs global updates. Require map operations to use **warp-uniform keys** or **contiguous per-lane indices** (e.g., `base + lane_id`), not random hashes. If policies must do random accesses, restrict them to **lane0 only**, amortizing the uncoalesced behavior to 1 lane/warp.

---

### 5) Control-Flow Divergence (warp branch divergence) — GPU-specific (perf, and interacts with liveness)

#### What it is / why it matters
  SIMT divergence serializes paths within a warp, lowering "branch efficiency" and increasing worst-case overhead. Divergence is also the root cause of barrier divergence when barriers are in conditional code.

#### Bug example

```cuda
__global__ void k(float* out, float* in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if ((tid & 1) == 0) out[tid] = in[tid] * 2;
  else                out[tid] = in[tid] * 3;  // divergence within warp
}
```

#### Seen in / checked by
  * GPUCheck explicitly targets "branch divergence" as a performance problem arising from thread-divergent expressions.([WebDocs][11])
  * GKLEE: "divergent warps" as performance bugs.([Lingming Zhang][18])
  * Wu et al.: "non-optimal implementation" includes performance loss causes like branch divergence.([arXiv][21])

#### Checking approach
  * **Static taint + symbolic reasoning (GPUCheck-style):** identify conditions dependent on thread/lane id, and prove whether divergence is possible.([WebDocs][11])

#### How gpu_ext should use it
  Divergence is the *core reason* you can treat performance as correctness. Enforce **warp-uniform control flow** for policies (or at least for any code path that triggers side effects / heavy helpers). If you can't prove uniformity, force "single-lane execution" of policy side effects (others become no-ops) to prevent warp amplification. Put a hard cap on the number of helper calls on any path, to bound the "divergence amplification factor."

---

### 6) Shared-Memory Bank Conflicts — GPU-specific (perf)

#### What it is / why it matters
  Bank conflicts are a shared-memory–specific performance pathology: accesses serialize when multiple lanes hit the same bank.

#### Bug example

```cuda
__global__ void k(int* out) {
  __shared__ int s[32*32];
  int lane = threadIdx.x & 31;
  // stride hits same bank pattern (illustrative)
  int x = s[lane * 32];
  out[threadIdx.x] = x;
}
```

#### Seen in / checked by
  * GKLEE explicitly lists "memory bank conflicts" among detected performance bugs.([Peng Li's Homepage][12])

#### Checking approach
  * **Static heuristic:** classify shared-memory index expressions by lane stride and bank mapping; warn if likely conflict.

#### How gpu_ext should use it
  If policies use shared scratchpads (e.g., per-block staging), either forbid it or enforce a **conflict-free access pattern** (e.g., contiguous per-lane indexing). Most observability policies can avoid shared memory entirely, turning this into a rule: "no shared-memory accesses in policy." Or simply ban shared-memory indexing by untrusted lane-dependent expressions.

---

### 7) Configuration Sensitivity / Portability: Block-Size Dependence + Launch Config Assumptions + Toolchain Variations — GPU-specific (correctness & tuning safety)

#### What it is / why it matters
  Block-size independence is essential for safe block-size tuning. Many CUDA kernels assume certain launch configurations. CUDA code can also fail or misbehave when moved across platforms.

#### Bug example (launch config assumption)

```cuda
__global__ void reduce(float* out, float* in) {
  // assumes gridDim.x == 1, but caller launches >1 blocks => wrong result / race
  // ...
}
```

#### Seen in / checked by
  * GPUDrano explicitly includes "block-size independence" analysis.([GitHub][2])
  * Wu et al.'s discussion of detected bugs includes developer responses that kernels "should not be called with more than one block" and suggests adding assertions like `assert(gridDim.x == 1)`.([arXiv][21])
  * Wu et al. explicitly call out "poor portability" as a root cause, including deprecated intrinsics and architecture assumptions.([arXiv][21])

#### Checking approach
  * **Contract checking:** encode launch preconditions (gridDim, blockDim assumptions) and enforce them at runtime or statically.
  * **Static analysis (GPUDrano):** block-size independence analysis.([GitHub][2])

#### How gpu_ext should use it
  If policy code assumes a particular block/warp mapping (e.g., keys use `threadIdx.x` directly), you can end up with correctness or performance regressions when kernels run under different launch configs. Policies should not implicitly assume block/grid shapes unless gpu_ext can guarantee them. If a policy depends on warp- or block-level structure, require declaring it (metadata) and validate at attach time. Add verifier rules that forbid hard-coded assumptions about blockDim/warp layout unless explicitly declared. Keep your verifier semantics architecture-aware (or restrict to a portable subset); if your policy ISA is PTX-level, pin the PTX version / codegen assumptions and validate them against target GPUs at load time.

---

### 8) "Forgot Volatile" / Memory Visibility Pitfalls — GPU-specific (correctness)

#### What it is / why it matters
  GPU code often relies on compiler and memory-model subtleties. GKLEE reports a real-world category: forgetting to mark a shared memory variable as `volatile`, producing stale reads/writes due to compiler optimization or caching behavior. This is a GPU-flavored instance of memory visibility/ordering bugs that can be hard to reproduce.([Lingming Zhang][18])

#### Bug example

```cuda
__shared__ int flag;          // should sometimes be volatile / properly fenced
if (tid == 0) flag = 1;
__syncthreads();
while (flag == 0) { }         // may spin if compiler hoists load / visibility issues
```

#### Seen in / checked by
  * GKLEE explicitly lists "forgot volatile" as a discovered bug type.([Lingming Zhang][18])
  * Simulee and other tools' race detection can surface some of these issues when they manifest as data races.([zhangyuqun.github.io][19])

#### Checking approach
  * **Symbolic exploration (GKLEE-style):** explore memory access orderings and detect stale read scenarios.([Lingming Zhang][18])
  * **Pattern-based linting:** flag spin-wait loops on shared memory without volatile or fence.

#### How gpu_ext should use it
  Avoid exposing raw shared/global memory communication to policies; instead provide **helpers with explicit semantics** (e.g., "atomic increment" or "write once" patterns), and verify policies don't implement ad-hoc synchronization loops. Forbid spin-waiting on shared memory in policy code.

---

### 9) Cross-Kernel Interference Channels — GPU-specific (performance as security/predictability)

#### What it is / why it matters
  In concurrent GPU usage, contention for shared resources makes execution time unpredictable. "Making Powerful Enemies on NVIDIA GPUs" explicitly studies **interference channels** and how adversarial "enemy" kernels can amplify slowdowns to stress worst-case execution times. This is the strongest literature anchor for the argument that performance interference is a *system-level safety* property when GPUs are shared.

#### Bug example (conceptual)

```cuda
// Kernel A is "victim"
// Kernel B is "enemy" stressing cache/DRAM/SM resources => tail latency explosion
```

#### Seen in / checked by
  * "Making Powerful Enemies on NVIDIA GPUs" studies interference channels and adversarial slowdowns in shared GPU execution.
  * Guardian's threat model discusses interference isolation in multi-tenant GPU sharing.([arXiv][22])

#### Checking approach
  * **Resource-budget verification:** enforce bounded resource consumption per kernel/policy.
  * **Interference benchmarking:** characterize worst-case slowdowns under contention (as in "Powerful Enemies" methodology).

#### How gpu_ext should use it
  Add a verifier contract like: "policy executes in O(1) helper calls, O(1) global memory ops, no blocking, warp-uniform side effects." Then you can argue (a) no hangs, and (b) bounded added contention footprint—consistent with your multi-tenant threat model.

---

### 10) Shared-Memory Data Races (`__shared__`) — CPU-shared, GPU-amplified

#### What it is / why it matters
  Threads in a block access on-chip shared memory concurrently; missing/incorrect synchronization causes races. This is a classic CUDA bug class (AuCS/Wu).

#### Bug example

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

#### Seen in / checked by
  * GPUVerify explicitly targets **data-race freedom** and defines intra-group / inter-group races.([Nathan Chong][1])
  * GKLEE reports finding **races** (and related deadlocks) via symbolic exploration.([Lingming Zhang][18])
  * Simulee detects **data race bugs** in real projects and uses a CUDA-aware notion of race.([zhangyuqun.github.io][19])
  * Wu et al. classify **data race** under "improper synchronization" as a CUDA-specific root cause.([arXiv][21])
  * Compute Sanitizer `racecheck` is a runtime shared-memory hazard detector.([Shinhwei][6])

#### Checking approach
  * **Static verifier route (GPUVerify-style):** enforce "race-free under SIMT" by proving that any two potentially concurrent lanes/threads cannot perform conflicting accesses without proper synchronization.([Nathan Chong][1])
  * **Dynamic route (Simulee-style):** instrument / simulate memory accesses and flag conflicting pairs; good for bug-finding and regression tests.([zhangyuqun.github.io][19])

#### How gpu_ext should use it
  If policies have any shared state, require **warp-uniform side effects** or **single-lane side effects** (e.g., lane0 updates) plus explicit atomics. A conservative verifier rule is: policy code cannot write shared memory except via restricted helpers that are race-safe (e.g., per-warp aggregation).

  * **Option A – warp-/block-uniform single-writer rules** (e.g., "only lane 0 updates").
  * **Option B – atomic-only helpers** for shared objects.
  * **Option C – per-thread/per-warp sharding** (each lane updates its own slot).

---

### 11) Atomic Contention — GPU-amplified (perf → DoS)

#### What it is / why it matters
  Heavy atomic contention is a classic "performance bug that behaves like a DoS" under massive parallelism. Even when correctness is preserved, contention on a single address can cause extreme slowdowns (orders of magnitude). With millions of threads, a single hot atomic can serialize execution and cause tail latency explosion.

#### Bug example

```cuda
__global__ void k(int* counter) {
  // All threads atomically increment the same location => extreme contention
  atomicAdd(counter, 1);
}
// Called with <<<1000, 1024>>> => 1M threads contending on one address
```

#### Seen in / checked by
  * GPUAtomicContention: an open-source benchmark suite (2025) explicitly measuring atomic performance under contention and across different **memory scopes** (block/device/system) and access patterns.([GitHub][13])

#### Checking approach
  * **Budget-based verification:** limit atomic frequency per warp/block.
  * **Benchmarking:** use atomic contention benchmarks to calibrate safe budgets.
  * **Static analysis:** identify hot atomic targets and warn about contention risk.

#### How gpu_ext should use it
  Treat "atomic frequency + contention risk" as a verifier-enforced budget: e.g., allow at most one global atomic per warp, or require warp-aggregated updates. For evaluation, you can reuse the open benchmark suite to calibrate "safe budgets" per GPU generation. Consider requiring warp-level reduction before global atomics to reduce contention by 32x.

---

### 12) Deadlocks Beyond Barrier Divergence (locks/spin + SIMT lockstep + named-barrier misuse) — CPU-shared, GPU-amplified (+ sometimes GPU-specific)

#### What it is / why it matters
  Besides barrier divergence, SIMT lockstep can create deadlocks in patterns that are unusual on CPUs. Warp-specialized kernels often use **named barriers** or structured synchronization patterns between warps/roles (producer/consumer). Bugs include: (a) deadlock, (b) unsafe barrier reuse ("recycling") across iterations, (c) races between producers/consumers.

#### Bug example (spin deadlock)

```cuda
__global__ void k(int* flag, int* data) {
  // Block 0 expects Block 1 to set flag, but no global sync exists
  if (blockIdx.x == 0) while (atomicAdd(flag, 0) == 0) { }  // may spin forever
  if (blockIdx.x == 1) { data[0] = 42; /* forgot to set flag */ }
}
```

#### Bug example (named-barrier misuse, sketch)

```cuda
// Producer writes buffer then signals barrier B
// Consumer waits on B then reads buffer
// Bug: consumer waits on wrong barrier instance / reused incorrectly in loop
```

#### Seen in / checked by
  * iGUARD notes that lockstep execution can deadlock if threads within a warp use distinct locks.([Aditya K Kamath][7])
  * GKLEE reports finding deadlocks via symbolic exploration of GPU kernels.([Lingming Zhang][18])
  * ESBMC-GPU models and checks deadlock too.([GitHub][10])
  * WEFT verifies **deadlock freedom**, **safe barrier recycling**, and **race freedom** for producer-consumer synchronization (named barriers).([zhangyuqun.github.io][19])

#### Checking approach
  * **Protocol verification (WEFT-style):** for specific synchronization patterns, prove deadlock freedom + race freedom + safe reuse. Model barrier instances across loop iterations and prove safe reuse.([zhangyuqun.github.io][19])
  * **Symbolic exploration (GKLEE-style):** explore possible interleavings and detect deadlock states.([Lingming Zhang][18])

#### How gpu_ext should use it
  Ban blocking primitives in policy code (locks, spin loops, waiting on global conditions). Add a verifier rule: **no unbounded loops / no "wait until" patterns**. If you absolutely need synchronization, force "single-lane, nonblocking" patterns and bounded retries. Policies must not interact with named barriers (no waits, no signals). This aligns with the availability story: policies must not create device stalls.

---

### 13) Kernel Non-Termination / Infinite Loops — CPU-shared, GPU-amplified

#### What it is / why it matters
  Infinite loops can hang GPU execution. In practice, non-termination is especially dangerous because GPU preemption/recovery can be coarse.

#### Bug example

```cuda
__global__ void k(int* flag) {
  while (*flag == 0) { }  // infinite loop if flag never set
  // or: while (true) { /* missing break */ }
}
```

#### Seen in / checked by
  * CL-Vis explicitly calls out infinite loops (together with barrier divergence) as GPU-specific bug types to detect/handle.([Computing and Informatics][9])

#### Checking approach
  * **Static bounds analysis:** prove loop termination or enforce compile-time bounded loops.
  * **Runtime watchdog:** timeout-based detection (coarse but practical).

#### How gpu_ext should use it
  This is where "bounded overhead = correctness" is easiest to justify: enforce a **strict instruction/iteration bound** for policy code (like eBPF on CPU). If policies may contain loops, require compile-time bounded loops only, with conservative upper bounds.

---

### 14) Redundant Barriers (unnecessary `__syncthreads`) — CPU-shared-ish, GPU-specific impact (perf)

#### What it is / why it matters
  A redundant barrier is a performance-pathology class: removing the barrier **does not introduce a race**, so the barrier was unnecessary overhead.

#### Bug example

```cuda
__global__ void k(int* out) {
  __shared__ int s[256];
  int t = threadIdx.x;
  s[t] = t;             // no cross-thread dependence here
  __syncthreads();      // redundant
  out[t] = s[t];
}
```

#### Seen in / checked by
  * Wu et al.: defines "redundant barrier function."([arXiv][21])
  * Simulee: detects redundant barrier bugs and reports numbers across projects.([zhangyuqun.github.io][19])
  * AuCS: repairs synchronization bugs, including redundant barriers.([Shinhwei][6])
  * GPURepair tooling also exists to insert/remove barriers to fix races and remove unnecessary ones.([GitHub][17])

#### Checking approach
  * **Static/dynamic dependence analysis:** determine whether any read-after-write / write-after-read across threads is protected by the barrier; if not, barrier is removable (Simulee/AuCS angle).([zhangyuqun.github.io][19])

#### How gpu_ext should use it
  For gpu_ext, this supports your "performance = safety" story: even "correct" policies can be unacceptable if they introduce barrier overhead. Since policies should avoid barriers entirely, you can convert this into a simpler rule: **"no barriers in policy,"** and separately "policy overhead must be bounded," eliminating this issue by construction. If helpers include barriers internally, you need cost models or architectural restrictions.

---

### 15) Global-Memory Data Races — CPU-shared

#### What it is / why it matters
  Races on global memory are a fundamental correctness issue. Unlike shared memory (block-local), global memory is accessible by all threads across all blocks, making races harder to reason about. Many GPU race detectors historically focused on shared memory and ignored global-memory races.

#### Bug example

```cuda
__global__ void k(int* g, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Multiple threads may write to same location without sync
  if (tid < n) g[tid % 16] += 1;  // race if multiple threads hit same index
}
```

#### Seen in / checked by
  * ScoRD explicitly argues that many GPU race detectors focus on shared memory and ignore global-memory races.([CSA - IISc Bangalore][4])
  * iGUARD targets races in global memory introduced by advanced CUDA features.([Aditya K Kamath][7])
  * GKLEE reports global memory races via symbolic exploration.([Lingming Zhang][18])

#### Checking approach
  * **Static verification:** extend race-freedom proofs to global memory accesses.
  * **Dynamic detection:** instrument global memory accesses and track conflicting pairs.

#### How gpu_ext should use it
  If policies can write to global memory (maps, counters, logs), require either: (1) warp-uniform single-writer rules, (2) atomic-only helpers, or (3) per-thread/per-warp sharding. Ban unprotected global writes from policies.

---

### 16) Host ↔ Device Asynchronous Data Races (API ordering bugs) — CPU-shared-ish, GPU-specific in practice

#### What it is / why it matters
  CUDA exposes async kernel launches/memcpy/events; host code can race with device work if synchronization is missing. This is a major real-world bug source in heterogeneous programs and is *not* covered by pure kernel-only verifiers.

#### Bug example

```cpp
int* d_data;
cudaMalloc(&d_data, N * sizeof(int));
kernel<<<grid, block>>>(d_data);
// missing cudaDeviceSynchronize() here
int* h_data = (int*)malloc(N * sizeof(int));
cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);  // race with kernel
```

#### Seen in / checked by
  * CuSan is an open-source detector for "data races between (asynchronous) CUDA calls and the host," using Clang/LLVM instrumentation plus ThreadSanitizer.([GitHub][5])

#### Checking approach
  * **Dynamic detection (CuSan-style):** instrument host-side CUDA API calls and detect ordering violations at runtime.

#### How gpu_ext should use it
  If gpu_ext policies interact with host-visible buffers or involve asynchronous map copies, define a strict **lifetime & ordering contract** (e.g., "policy writes are only consumed after a guaranteed sync point"). For testing, integrate CuSan into CI for host-side integration tests of the runtime/loader.

---

### 17) Memory Safety: Out-of-Bounds / Misaligned / Use-After-Free / Use-After-Scope — CPU-shared, GPU-specific (security/availability in multi-tenant)

#### What it is / why it matters
  Classic memory safety includes both **spatial** (OOB, misaligned) and **temporal** (UAF, UAS) violations. Temporal bugs exist on GPUs too: pointers can outlive allocations (host frees while kernel still uses, device-side stack frame returns, etc.).

#### Bug example (OOB)

```cuda
__global__ void k(float* a, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  a[tid + 1024] = 0.0f;   // OOB write
}
```

#### Bug example (Use-After-Scope)

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

#### Seen in / checked by
  * Compute Sanitizer `memcheck` precisely detects OOB/misaligned accesses (and can detect memory leaks).([NVIDIA Docs][3])
  * Oclgrind reports invalid memory accesses in its simulator.([GitHub][16])
  * ESBMC-GPU checks pointer safety and array bounds as part of its model checking.([GitHub][10])
  * GKLEE's evaluation includes out-of-bounds global memory accesses as error cases.([Lingming Zhang][18])
  * Wu et al.: "unauthorized memory access" appears in root-cause characterization.([arXiv][21])
  * cuCatch explicitly targets temporal violations using tagging mechanisms and discusses UAF/UAS detection.([d1qx31qr3h6wln.cloudfront.net][20])
  * Guardian: PTX-level instrumentation + interception to fence illegal memory accesses under GPU sharing.([arXiv][22])

#### Checking approach
  * **Bounds-check instrumentation (Guardian/cuCatch-style):** insert base+bounds checks (or partition-fencing) around loads/stores.([arXiv][22])
  * **Temporal tagging + runtime checks (cuCatch-style):** tag allocations and validate before deref.([d1qx31qr3h6wln.cloudfront.net][20])
  * **Static verification (ESBMC-GPU):** model checking for pointer safety and array bounds.([GitHub][10])
  * **PTX-level instrumentation (Guardian-style):** insert bounds checks and interception to fence illegal accesses.([arXiv][22])
  * **Tagging mechanisms (cuCatch-style):** track allocation ownership and validate access rights.([d1qx31qr3h6wln.cloudfront.net][20])

#### How gpu_ext should use it
  This is the "classic verifier" portion: keep eBPF-like pointer tracking, bounds checks, and restricted helpers. Easiest for policies is to **ban arbitrary pointer dereferences** and force all memory access through safe helpers (maps/ringbuffers). Ideally: policies cannot allocate/free; all policy-visible objects are managed by gpu_ext runtime and remain valid across policy execution (no UAF/UAS by construction). Also add a testing story: run policy-enabled kernels under Compute Sanitizer memcheck in CI for regression.

#### Multi-tenant implications
  In spatial sharing (streams/MPS), kernels share a GPU address space. An OOB access by one application can crash other co-running applications (fault isolation issue). Guardian's motivation explicitly calls out this problem and designs PTX-level fencing + interception as a fix.([arXiv][22]) This directly supports the "availability is correctness" story: if gpu_ext policies run in privileged/shared contexts, you must prevent policy code from generating OOB accesses. Either: (a) only allow map helpers (no raw memory), or (b) instrument policy memory ops with bounds checks (Guardian-style PTX rewriting).

#### Bug example (multi-tenant OOB, conceptual)

  ```cuda
  // Tenant A kernel writes OOB and corrupts Tenant B memory in same context.
  ```

---

### 18) Uninitialized Global Memory Reads — CPU-shared

#### What it is / why it matters
  Accessing device global memory without initialization leads to nondeterministic behavior. This is a frequent source of heisenbugs because GPU concurrency amplifies nondeterminism.

#### Bug example

```cuda
__global__ void k(float* out, float* in, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // 'in' was cudaMalloc'd but never initialized or memset
  out[tid] = in[tid] * 2.0f;  // reading uninitialized memory
}
```

#### Seen in / checked by
  * Compute Sanitizer `initcheck` reports cases where device global memory is accessed without being initialized by device writes or CUDA memcpy/memset.([NVIDIA Docs][3])

#### Checking approach
  * **Runtime detection (initcheck):** track memory initialization state and flag uninitialized reads.

#### How gpu_ext should use it
  If gpu_ext policies read from maps/buffers, require explicit initialization semantics (e.g., map lookup returns "not found" unless initialized; forbid reading uninitialized slots). In testing, run initcheck on representative workloads.

---

### 19) Resource Management: Memory Leaks / Incorrect Allocation / Lifecycle Bugs — CPU-shared

#### What it is / why it matters
  A lot of CUDA failures are not in kernel math but in lifecycle management: incorrect device allocation, memory leaks, early device reset calls, etc. These are availability issues in long-running services.

#### Bug example (early reset)

```cpp
cudaMalloc(&p, N);
kernel<<<...>>>(p);
cudaDeviceReset();     // early reset => invalidates work / crashes
```

#### Seen in / checked by
  * Compute Sanitizer memcheck includes leak checking (e.g., `cudaMalloc` leaks, device heap leaks) and can also report incorrect use of `malloc/free()` in kernels.([NVIDIA Docs][3])
  * Wu et al. highlight these as a major root-cause category ("improper resource management"): "incorrect device resource allocation, memory leak, early device call reset, unauthorized memory access."([arXiv][21])

#### Checking approach
  * **Host-side linters / runtime guards:** track allocation/free and enforce API ordering (this is outside kernel verifiers).
  * **Runtime detection (memcheck):** detect leaks and invalid free operations.

#### How gpu_ext should use it
  Forbid dynamic allocation in policy code; if helpers allocate, require bounded allocations + automatic cleanup. Keep policy loading/unloading tied to safe lifecycle transitions (e.g., disallow unloading policies while kernels that might execute them are in flight). Make map lifetime explicit and externally managed. Treat leaks as "availability correctness," because persistent GPU agents/daemons can degrade over time.

---

### 20) Arithmetic Errors (overflow, division by zero) — CPU-shared

#### What it is / why it matters
  Arithmetic errors can corrupt keys/indices and cascade into memory safety/perf disasters.

#### Bug example

```cuda
__global__ void k(int* out, int* in, int divisor) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = in[tid] / divisor;  // div-by-zero if divisor == 0

  int idx = tid * 1000000;       // overflow for large tid
  out[idx] = 1;                  // corrupted index => OOB
}
```

#### Seen in / checked by
  * ESBMC-GPU explicitly lists arithmetic overflow and division-by-zero among the properties it checks for CUDA programs (alongside races/deadlocks/bounds).([GitHub][10])

#### Checking approach
  * **Model checking (ESBMC-GPU):** static verification of arithmetic properties.
  * **Lightweight runtime checks:** guard div/mod operations.

#### How gpu_ext should use it
  Optional but reviewer-friendly: add lightweight verifier checks for div-by-zero and dangerous shifts, and constrain pointer arithmetic (already typical in eBPF verifiers). For "perf correctness," overflow in index computations is a common hidden cause of random/uncoalesced patterns.

---

### Summary: Improper Synchronization as a Root-Cause Category (Wu et al.'s Three-Way Taxonomy)

Wu et al.'s empirical study explicitly groups CUDA-specific synchronization issues into three concrete bug types: **data race**, **barrier divergence**, and **redundant barrier functions**. They also highlight that these often manifest as inferior performance and flaky tests. Simulee is used to find these categories in real projects.([arXiv][21])

This is exactly the "verification story" hook for gpu_ext: your verifier can claim that policy code cannot introduce these synchronization root causes because:
* no barriers allowed,
* warp-uniform side effects enforced,
* bounded helper calls,
* and a restricted memory model for policies.

---


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
[23]: https://www.cis.upenn.edu/~alur/Cav17.pdf "https://www.cis.upenn.edu/~alur/Cav17.pdf"
