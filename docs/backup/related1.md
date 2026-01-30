Below is a consolidated **GPU / CUDA / SIMT bug taxonomy** extracted from (and checked against) the papers/tools you referenced earlier. I merged duplicates across papers and kept GPU-specific distinctions explicit. For each bug type I include:

* **Intro** (what it is + why it’s GPU/SIMT-special)
* **Bug example** (small CUDA-ish snippet)
* **Where it appears / is checked** (paper citations)
* **A plausible checking approach** + **how gpu_ext should use it** (verifier/linter/runtime policy)

---

## 1) Data races in global/shared memory

**Intro.**
In SIMT kernels, data races are a top-tier *correctness* and *nondeterminism* source. GPU kernels often use shared memory + barriers for intra-block coordination; if two threads can read/write the same address without an intervening barrier (or correct ordering), results can become nondeterministic (“flaky”) and sometimes “seem fine” until scale/scheduling changes. GPUVerify formalizes race freedom (intra-group + inter-group) as a core safety property. ([Nathan Chong][1])

**Bug example.**

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

**Seen in / checked by.**

* GPUVerify explicitly targets **data-race freedom** and defines intra-group / inter-group races. ([Nathan Chong][1])
* GKLEE reports finding **races** (and related deadlocks) via symbolic exploration. ([Lingming Zhang's Homepage][2])
* Simulee detects **data race bugs** in real projects and uses a CUDA-aware notion of race. ([zhangyuqun.github.io][3])
* Wu et al. classify **data race** under “improper synchronization” as a CUDA-specific root cause. ([arXiv][4])

**Checking approach + how gpu_ext should use it.**

* **Static verifier route (GPUVerify-style):** enforce “race-free under SIMT” by proving that any two potentially concurrent lanes/threads cannot perform conflicting accesses without proper synchronization. ([Nathan Chong][1])
* **Dynamic route (Simulee-style):** instrument / simulate memory accesses and flag conflicting pairs; good for bug-finding and regression tests. ([zhangyuqun.github.io][3])
* **gpu_ext implication:** if policies can mutate shared/global state (maps, counters, logs), you want either:

  1. **warp-/block-uniform single-writer rules** (e.g., “only lane 0 updates”), or
  2. **atomic-only helpers** for shared objects, or
  3. **per-thread/per-warp sharding** (each lane updates its own slot).
     Then the verifier can check a simple “no conflicting writes across lanes” policy contract.

---

## 2) Warp-divergence race (GPU-specific race semantics)

**Intro.**
A “warp-divergence race” (as described in GKLEE) is a GPU-specific phenomenon where **divergence changes which threads are effectively concurrent**, producing racy outcomes that don’t map cleanly to CPU assumptions. This is one of the reasons “CPU-style race reasoning” doesn’t port directly: SIMT execution order + reconvergence can create subtle concurrency patterns. ([Lingming Zhang's Homepage][2])

**Bug example.**

```cuda
__global__ void k(int* A) {
  int lane = threadIdx.x & 31;
  if (lane < 16) A[0] = 1;      // first half writes
  else           A[0] = 2;      // second half writes
  // outcome depends on SIMT execution + reconvergence
}
```

**Seen in / checked by.**

* GKLEE explicitly lists “warp-divergence race” among discovered bug classes. ([Lingming Zhang's Homepage][2])
* Simulee stresses CUDA-aware race definitions and discusses GPU-specific race interpretation constraints (e.g., avoiding false positives due to warp lockstep). ([zhangyuqun.github.io][3])

**Checking approach + how gpu_ext should use it.**

* **Verifier rule:** treat “lane-divergent side effects” as forbidden unless proven safe. This matches your gpu_ext direction of “warp-uniform side effects.”
* **gpu_ext use:** require that any helper with side effects (map update, counter increment, perf event emit) is guarded by a **warp-uniform predicate** or executed only by a designated lane (e.g., lane0). Then the verifier only needs to prove **uniformity** (or single-lane execution), not full SIMT interleavings.

---

## 3) Barrier divergence and undefined behavior at `__syncthreads`

**Intro.**
Barrier divergence is a GPU-unique liveness/correctness hazard: within a thread block, **all active threads must reach the same barrier** (conceptually). If some threads take a different control path and skip the barrier, behavior is undefined and may deadlock. GPUVerify explicitly checks “divergence freedom” (barrier divergence). Wu et al. categorize this as CUDA-specific improper synchronization. ([Nathan Chong][1])

**Bug example.**

```cuda
__global__ void k(float* a) {
  if (threadIdx.x < 16) __syncthreads(); // divergent barrier => UB / deadlock
  a[threadIdx.x] = 1.0f;
}
```

**Seen in / checked by.**

* GPUVerify: checking divergence is a core goal (“divergence freedom”). ([Nathan Chong][1])
* Simulee detects **barrier divergence bugs** in real-world code. ([zhangyuqun.github.io][3])
* Wu et al.: explicitly defines barrier divergence and places it under improper synchronization. ([arXiv][4])

**Checking approach + how gpu_ext should use it.**

* **Static check (GPUVerify-style):** prove that each barrier is reached by all threads in the relevant scope, often via uniformity reasoning. ([Nathan Chong][1])
* **Dynamic check:** synccheck-style runtime validation (common in tooling), and Simulee-style bug finding. ([zhangyuqun.github.io][3])
* **gpu_ext use:** simplest and strongest: **forbid `__syncthreads()` (and any block-wide barrier) inside policies**. This directly eliminates an entire class of GPU hangs. If you ever need barrier-like semantics for observability, implement it *outside* the policy in trusted runtime code.

---

## 4) Deadlocks due to synchronization misuse (beyond simple barrier divergence)

**Intro.**
Deadlocks on GPUs often arise from **mismatched synchronization protocols** (e.g., producers/consumers, named barriers, or divergent arrivals). Unlike CPUs, a deadlock may stall an SM/block and can have system-wide impact depending on runtime recovery. GKLEE reports deadlocks; WEFT focuses explicitly on proving deadlock freedom in warp-specialized/named-barrier synchronization. ([Lingming Zhang's Homepage][2])

**Bug example.**

```cuda
__global__ void k(int* flag, int* data) {
  // Block 0 expects Block 1 to set flag, but no global sync exists
  if (blockIdx.x == 0) while (atomicAdd(flag, 0) == 0) { }  // may spin forever
  if (blockIdx.x == 1) { data[0] = 42; /* forgot to set flag */ }
}
```

**Seen in / checked by.**

* GKLEE lists “deadlocks” among discovered bugs. ([Lingming Zhang's Homepage][2])
* WEFT targets **deadlock freedom** (and related properties) for GPU producer-consumer synchronization with named barriers. ([zhangyuqun.github.io][3])

**Checking approach + how gpu_ext should use it.**

* **Protocol verification (WEFT-style):** for specific synchronization patterns, prove deadlock freedom + race freedom + safe reuse. ([zhangyuqun.github.io][3])
* **gpu_ext use:** treat any **blocking/waiting** behavior in policy (spinning on atomics, waiting for flags) as forbidden. Verification rule: “policy must be non-blocking and bounded.” This aligns with your availability story: policies must not create device stalls.

---

## 5) Redundant barriers (unnecessary synchronization)

**Intro.**
A redundant barrier is a performance-pathology class: removing the barrier **does not introduce a race**, so the barrier was unnecessary overhead. Wu et al. treat this as a CUDA-specific synchronization issue; Simulee detects redundant barriers in the wild; AuCS repairs by removing/inserting barriers correctly. ([arXiv][4])

**Bug example.**

```cuda
__global__ void k(int* out) {
  __shared__ int s[256];
  int t = threadIdx.x;
  s[t] = t;             // no cross-thread dependence here
  __syncthreads();      // redundant
  out[t] = s[t];
}
```

**Seen in / checked by.**

* Wu et al.: defines “redundant barrier function.” ([arXiv][4])
* Simulee: detects redundant barrier bugs and reports numbers across projects. ([zhangyuqun.github.io][3])
* AuCS: repairs synchronization bugs, including redundant barriers. ([zhangyuqun.github.io][3])

**Checking approach + how gpu_ext should use it.**

* **Static/dynamic dependence analysis:** determine whether any read-after-write / write-after-read across threads is protected by the barrier; if not, barrier is removable (Simulee/AuCS angle). ([zhangyuqun.github.io][3])
* **gpu_ext use:** since policies should avoid barriers entirely, you can convert this into a simpler rule: “no barriers in policy,” and separately “policy overhead must be bounded,” eliminating this issue by construction.

---

## 6) Named-barrier misuse in warp-specialized producer/consumer kernels

**Intro.**
Warp-specialized kernels often use **named barriers** or structured synchronization patterns between warps/roles (producer/consumer). Bugs include: (a) deadlock, (b) unsafe barrier reuse (“recycling”) across iterations, (c) races between producers/consumers. WEFT addresses exactly these properties. ([zhangyuqun.github.io][3])

**Bug example (sketch).**

```cuda
// Producer writes buffer then signals barrier B
// Consumer waits on B then reads buffer
// Bug: consumer waits on wrong barrier instance / reused incorrectly in loop
```

**Seen in / checked by.**

* WEFT verifies **deadlock freedom**, **safe barrier recycling**, and **race freedom** for producer-consumer synchronization (named barriers). ([zhangyuqun.github.io][3])

**Checking approach + how gpu_ext should use it.**

* **Protocol-aware verification (WEFT):** model barrier instances across loop iterations and prove safe reuse. ([zhangyuqun.github.io][3])
* **gpu_ext use:** unless gpu_ext plans to support policies inside warp-specialized synchronization code, the practical contract is: **policies must not interact with named barriers** (no waits, no signals). Otherwise, you need a WEFT-like proof obligation for policies that touch those synchronization states—which is likely too heavy for dynamic eBPF-style loading.

---

## 7) Shared-memory bank conflicts (performance bug)

**Intro.**
Bank conflicts are GPU-specific: when threads in a warp access shared memory addresses mapping to the same bank, accesses serialize, reducing throughput. GKLEE explicitly flags “memory bank conflicts” as performance bugs it detects. ([Lingming Zhang's Homepage][2])

**Bug example.**

```cuda
__global__ void k(int* out) {
  __shared__ int s[32*32];
  int lane = threadIdx.x & 31;
  // stride hits same bank pattern (illustrative)
  int x = s[lane * 32];
  out[threadIdx.x] = x;
}
```

**Seen in / checked by.**

* GKLEE lists “memory bank conflicts” among detected performance bugs. ([Lingming Zhang's Homepage][2])

**Checking approach + how gpu_ext should use it.**

* **Static heuristic:** classify shared-memory index expressions by lane stride and bank mapping; warn if likely conflict.
* **gpu_ext use:** if policy code uses shared memory (e.g., staging per-block data), either forbid it or enforce a **conflict-free access pattern** (e.g., contiguous per-lane indexing). Most observability policies can avoid shared memory entirely, turning this into a rule: “no shared-memory accesses in policy.”

---

## 8) Non-coalesced global memory accesses (performance bug with massive amplification)

**Intro.**
Coalescing is SIMT-specific: global memory transactions are most efficient when a warp’s accesses fall into as few cache lines/transactions as possible. “Uncoalesced” accesses can cause large slowdowns. GPUDrano is a dedicated static analysis for uncoalesced global accesses; GKLEE and GPUCheck also target non-coalesced/non-coalescable accesses as performance issues. ([Computer and Information Science][5])

**Bug example.**

```cuda
__global__ void k(float* a, int stride) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[tid * stride];   // stride>1 => likely uncoalesced
  a[tid * stride] = x + 1.0f;
}
```

**Seen in / checked by.**

* GPUDrano: “detects uncoalesced global memory accesses” and treats them as performance bugs. ([Computer and Information Science][5])
* GKLEE: reports “non-coalesced memory accesses” as performance bugs it finds. ([Lingming Zhang's Homepage][2])
* GPUCheck: detects “non-coalesceable memory accesses.” ([Webdocs][6])

**Checking approach + how gpu_ext should use it.**

* **Static analysis (GPUDrano/GPUCheck-style):** analyze address expressions in terms of lane-to-address stride; flag when stride exceeds coalescing thresholds. ([Computer and Information Science][5])
* **gpu_ext use:** this maps directly to your “bounded interference” contract:

  * Require map operations to use **warp-uniform keys** or **contiguous per-lane indices** (e.g., `base + lane_id`), not random hashes.
  * If policies must do random accesses, restrict them to **lane0 only**, amortizing the uncoalesced behavior to 1 lane/warp.

---

## 9) Branch / warp divergence (performance bug, also a liveness amplifier)

**Intro.**
In SIMT, divergence makes a warp execute both paths serially (conceptually), lowering “branch efficiency” and increasing worst-case overhead. GKLEE explicitly flags “divergent warps” as performance bugs; GPUCheck is dedicated to detecting branch divergence; Wu et al. cite branch divergence as a common reason for “non-optimal implementation” performance issues. ([Lingming Zhang's Homepage][2])

**Bug example.**

```cuda
__global__ void k(float* out, float* in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if ((tid & 1) == 0) out[tid] = in[tid] * 2;
  else                out[tid] = in[tid] * 3;  // divergence within warp
}
```

**Seen in / checked by.**

* GKLEE: “divergent warps” as performance bugs. ([Lingming Zhang's Homepage][2])
* GPUCheck: detects branch divergence caused by thread-dependent expressions. ([Webdocs][6])
* Wu et al.: “non-optimal implementation” includes performance loss causes like branch divergence. ([arXiv][4])

**Checking approach + how gpu_ext should use it.**

* **Static taint + symbolic reasoning (GPUCheck-style):** identify conditions dependent on thread/lane id, and prove whether divergence is possible. ([Webdocs][6])
* **gpu_ext use:** divergence is the *core reason* you can treat performance as correctness:

  * Enforce **warp-uniform control flow** for policy code (or run policy in a single lane).
  * Put a hard cap on the number of helper calls on any path, to bound the “divergence amplification factor.”

---

## 10) “Forgot volatile” / memory visibility pitfalls (GPU-specific correctness)

**Intro.**
GPU code often relies on compiler and memory-model subtleties. GKLEE reports a real-world category: forgetting to mark a shared memory variable as `volatile`, producing stale reads/writes due to compiler optimization or caching behavior. This is a GPU-flavored instance of memory visibility/ordering bugs that can be hard to reproduce. ([Lingming Zhang's Homepage][2])

**Bug example (illustrative).**

```cuda
__shared__ int flag;          // should sometimes be volatile / properly fenced
if (tid == 0) flag = 1;
__syncthreads();
while (flag == 0) { }         // may spin if compiler hoists load / visibility issues
```

**Seen in / checked by.**

* GKLEE lists “forgetting to mark a shared memory variable as volatile” as a bug it found. ([Lingming Zhang's Homepage][2])

**Checking approach + how gpu_ext should use it.**

* **Linter rule:** forbid spin-waiting on shared memory in policy code; require helper APIs with defined memory semantics for signaling.
* **gpu_ext use:** avoid exposing raw shared/global memory communication to policies; instead provide **helpers with explicit semantics** (e.g., “atomic increment” or “write once” patterns), and verify policies don’t implement ad-hoc synchronization loops.

---

## 11) Improper synchronization as a root-cause category (data race + barrier divergence + redundant barriers)

**Intro.**
Wu et al.’s empirical study is useful because it explicitly groups CUDA-specific synchronization issues into three concrete bug types: **data race**, **barrier divergence**, and **redundant barrier functions**. They also highlight that these often manifest as inferior performance and flaky tests. ([arXiv][4])

**Bug example.**

```cuda
// same patterns as items #1, #3, #5; the key point is the *taxonomy* and prevalence
```

**Seen in / checked by.**

* Wu et al. define these three and quantify them in detected bugs; Simulee is used to find these categories in projects (Table 3). ([arXiv][4])

**Checking approach + how gpu_ext should use it.**

* **gpu_ext use:** this is exactly the “verification story” hook: your verifier can claim that policy code cannot introduce these synchronization root causes because:

  * no barriers,
  * warp-uniform side effects,
  * bounded helper calls,
  * and a restricted memory model for policies.

---

## 12) Memory safety: out-of-bounds (OOB) / unauthorized accesses inside kernels

**Intro.**
Memory safety is harder on GPUs because kernels share a unified device address space and huge parallelism makes OOB effects catastrophic (crashes, silent corruption, or—in multi-tenant settings—cross-tenant impact). Wu et al. classify “unauthorized memory access” under improper resource management. cuCatch is a GPU memory-safety tool that improves detection coverage of OOB. Guardian targets OOB isolation for multi-tenant spatial sharing by PTX-level bounds checking. ([arXiv][4])

**Bug example.**

```cuda
__global__ void k(float* a, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  a[tid + 1024] = 0.0f;   // OOB write
}
```

**Seen in / checked by.**

* Wu et al.: “unauthorized memory access” appears in root-cause characterization. ([arXiv][4])
* cuCatch: designed to catch memory safety violations, including OOB patterns; discusses OOB coverage on global memory. ([d1qx31qr3h6wln.cloudfront.net][7])
* Guardian: PTX-level instrumentation + interception to fence illegal memory accesses under GPU sharing. ([arXiv][8])

**Checking approach + how gpu_ext should use it.**

* **Bounds-check instrumentation (Guardian/cuCatch-style):** insert base+bounds checks (or partition-fencing) around loads/stores. ([arXiv][8])
* **gpu_ext use:** easiest for policies is to **ban arbitrary pointer dereferences** in policy code and force all memory access through safe helpers (maps/ringbuffers) whose implementations are bounds-checked. If you must support raw pointers, adopt Guardian-like PTX rewriting (or cuCatch-like metadata) for policy-generated memory ops.

---

## 13) Temporal memory safety: use-after-free / use-after-scope (UAF/UAS)

**Intro.**
Temporal bugs exist on GPUs too: pointers can outlive allocations (host frees while kernel still uses, device-side stack frame returns, etc.). cuCatch explicitly targets temporal violations using tagging mechanisms and discusses use-after-free and use-after-scope. ([d1qx31qr3h6wln.cloudfront.net][7])

**Bug example (conceptual).**

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

**Seen in / checked by.**

* cuCatch: discusses temporal checks and explicitly mentions use-after-free and use-after-scope detection behavior. ([d1qx31qr3h6wln.cloudfront.net][7])

**Checking approach + how gpu_ext should use it.**

* **Temporal tagging + runtime checks (cuCatch-style):** tag allocations and validate before deref. ([d1qx31qr3h6wln.cloudfront.net][7])
* **gpu_ext use:** ideally: policies cannot allocate/free; all policy-visible objects are managed by gpu_ext runtime and remain valid across policy execution. Then the verifier can claim “no UAF/UAS” by construction. If policies can hold pointers, you need lifetime tracking or tagging.

---

## 14) Improper resource management (GPU/host lifecycle bugs)

**Intro.**
A lot of CUDA failures are not in kernel math but in lifecycle management: incorrect device allocation, memory leaks, early device reset calls, etc. Wu et al. highlight these as a major root-cause category (“improper resource management”) and relate them to crashes. ([arXiv][4])

**Bug example (host-side sketch).**

```cpp
cudaMalloc(&p, N);
kernel<<<...>>>(p);
cudaDeviceReset();     // early reset => invalidates work / crashes
```

**Seen in / checked by.**

* Wu et al.: explicitly lists “incorrect device resource allocation, memory leak, early device call reset, unauthorized memory access” as scenarios under improper resource management. ([arXiv][4])

**Checking approach + how gpu_ext should use it.**

* **Host-side linters / runtime guards:** track allocation/free and enforce API ordering (this is outside kernel verifiers).
* **gpu_ext use:** keep policy loading/unloading tied to safe lifecycle transitions. For example: disallow unloading policies while kernels that might execute them are in flight; make map lifetime explicit and externally managed.

---

## 15) Kernel launch configuration & “dimension assumptions” bugs

**Intro.**
Many CUDA kernels assume certain launch configurations (e.g., “only one block”, “blockDim is power-of-two”, etc.). Violating these assumptions can cause wrong results, races, or “it works in tests but not in production.” Wu et al.’s discussion of detected bugs includes developer responses that kernels “should not be called with more than one block” and suggests adding assertions on gridDim. ([arXiv][4])

**Bug example.**

```cuda
__global__ void reduce(float* out, float* in) {
  // assumes gridDim.x == 1, but caller launches >1 blocks => wrong result / race
  // ...
}
```

**Seen in / checked by.**

* Wu et al. show real bug discussions where developers recommend assertions like `assert(gridDim.x == 1)`. ([arXiv][4])

**Checking approach + how gpu_ext should use it.**

* **Contract checking:** encode launch preconditions and enforce them at runtime or statically.
* **gpu_ext use:** policies should not implicitly assume block/grid shapes unless gpu_ext can guarantee them. If a policy depends on warp- or block-level structure, require declaring it (metadata) and validate at attach time.

---

## 16) Poor portability across toolchains / platforms / GPU generations

**Intro.**
CUDA code can fail or misbehave when moved across platforms (compiler versions, driver versions, GPU architectures). Wu et al. explicitly call this out as a root cause (“poor portability”). ([arXiv][4])

**Bug example.**

```cuda
// uses deprecated intrinsic or assumes old shared-memory bank width;
// breaks or changes behavior/perf on newer GPUs.
```

**Seen in / checked by.**

* Wu et al. define “poor portability” as a root-cause category with examples tied to specific build/platform contexts. ([arXiv][4])

**Checking approach + how gpu_ext should use it.**

* **gpu_ext use:** keep your verifier semantics architecture-aware (or restrict to a portable subset). If your policy ISA is PTX-level, pin the PTX version / codegen assumptions and validate them against target GPUs at load time.

---

## 17) Multi-tenant GPU sharing: lack of fault isolation for OOB → co-runner crashes

**Intro.**
In spatial sharing (streams/MPS), kernels share a GPU address space. An OOB access by one application can crash other co-running applications (fault isolation issue). Guardian’s motivation explicitly calls out this problem and designs PTX-level fencing + interception as a fix. ([arXiv][8])

**Bug example (conceptual).**

```cuda
// Tenant A kernel writes OOB and corrupts Tenant B memory in same context.
```

**Seen in / checked by.**

* Guardian: frames GPU sharing as introducing memory safety concerns and notes that OOB under some sharing mechanisms can crash co-runners; proposes PTX-level bounds checking and fencing. ([arXiv][8])

**Checking approach + how gpu_ext should use it.**

* **Isolation layer (Guardian-style):** partition address space and fence accesses. ([arXiv][8])
* **gpu_ext use:** this directly supports your “availability is correctness” story: if gpu_ext policies run in privileged/shared contexts, you must prevent policy code from generating OOB accesses. Either: (a) only allow map helpers (no raw memory), or (b) instrument policy memory ops with bounds checks.

---

## 18) Cross-kernel interference channels (performance “bugs” as predictability failures)

**Intro.**
In concurrent GPU usage, contention for shared resources makes execution time unpredictable. “Making Powerful Enemies on NVIDIA GPUs” explicitly studies **interference channels** and how adversarial “enemy” kernels can amplify slowdowns to stress worst-case execution times. This is the strongest literature anchor for your argument that performance interference is a *system-level safety* property when GPUs are shared. 

**Bug example (conceptual).**

```cuda
// Kernel A is “victim”
// Kernel B is “enemy” stressing cache/DRAM/SM resources => tail latency explosion
```

**Seen in / checked by.**

* The paper defines interference channels (shared GPU resources) and builds “enemy programs” to intentionally contend for them, showing slowdowns in real workloads. 

**Checking approach + how gpu_ext should use it.**

* **Policy-level cost bounding:** treat “unbounded interference” as a correctness violation.
* **gpu_ext use:** add a verifier contract like: “policy executes in O(1) helper calls, O(1) global memory ops, no blocking, warp-uniform side effects.” Then you can argue (a) no hangs, and (b) bounded added contention footprint—consistent with your multi-tenant threat model.

---

# What’s uniquely “GPU/SIMT different from CPU” across these bugs?

If you need a tight paragraph for the paper, these are the recurring GPU-specific themes the literature reinforces:

* **Barrier semantics are collective**; divergence around barriers yields undefined behavior / deadlock (GPUVerify, Wu, Simulee). ([Nathan Chong][1])
* **Warp-level lockstep & reconvergence** changes race and scheduling intuition (GKLEE’s warp-divergence races; Simulee’s CUDA-aware race notion). ([Lingming Zhang's Homepage][2])
* **Performance hazards are structural and amplifiable**: divergence and memory (non-)coalescing are first-order performance “bugs” detectable statically (GPUCheck, GPUDrano, GKLEE). ([Webdocs][6])
* **Sharing/tenancy changes “performance” into “safety”** because interference channels make execution times unpredictable and can be adversarially maximized (Making Powerful Enemies; Guardian’s fault isolation framing). 

---

# How to use this list to strengthen gpu_ext’s verification story

A practical way to turn this into a “complete set” argument (reviewer-friendly) is:

1. **Claim coverage of known CUDA-specific synchronization bug classes** (race, barrier divergence, redundant barrier), explicitly matching Wu’s taxonomy and Simulee’s detected categories. ([arXiv][4])
2. **Add GPU-specific performance correctness classes** that are statically analyzable (divergence, uncoalesced access) and cite GPUCheck/GPUDrano/GKLEE. ([Webdocs][6])
3. **Position multi-tenant safety** as (a) memory isolation (Guardian) and (b) bounded interference (enemy programs / interference channels). ([arXiv][8])

If you want, I can also rewrite your paper section so that each gpu_ext verifier rule is explicitly mapped to one (or more) of the bug types above, using the same “threat → rule → prevented failure mode” pattern reviewers expect from systems verification papers.

[1]: https://nchong.github.io/papers/oopsla12.pdf "https://nchong.github.io/papers/oopsla12.pdf"
[2]: https://lingming.cs.illinois.edu/publications/icse2020b.pdf "https://lingming.cs.illinois.edu/publications/icse2020b.pdf"
[3]: https://zhangyuqun.github.io/publications/ase2019.pdf "https://zhangyuqun.github.io/publications/ase2019.pdf"
[4]: https://arxiv.org/pdf/1905.01833 "https://arxiv.org/pdf/1905.01833"
[5]: https://www.cis.upenn.edu/~alur/Cav17.pdf "https://www.cis.upenn.edu/~alur/Cav17.pdf"
[6]: https://webdocs.cs.ualberta.ca/~amaral/thesis/TaylorLloydMSc.pdf "https://webdocs.cs.ualberta.ca/~amaral/thesis/TaylorLloydMSc.pdf"
[7]: https://d1qx31qr3h6wln.cloudfront.net/publications/PLDI_2023_cuCatch_2.pdf "https://d1qx31qr3h6wln.cloudfront.net/publications/PLDI_2023_cuCatch_2.pdf"
[8]: https://arxiv.org/pdf/2401.09290 "https://arxiv.org/pdf/2401.09290"
