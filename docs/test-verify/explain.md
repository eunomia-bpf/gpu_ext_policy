---

when / why / how we want that

- more usecase
- example of bugs show safety issues
    - thread divergence
    - memory ops too freqence (roofline)
    - deadlocks

## 1. Current Problem with the Verification Story

The current gpu_ext paper introduces verification primarily as a feature that parallels classic eBPF practice: ensuring memory safety, loop boundedness, and controlled helper usage. While these are necessary checks, the paper does not fully clarify **why verification on GPUs demands GPU-specific extensions**. Specifically, it does not sufficiently emphasize how GPU hardware and execution models introduce new classes of failure modes that traditional eBPF verification is unable to prevent.

This leads reviewers to mistakenly view verification as:

- Merely a “nice-to-have” safety property,
- A straightforward porting of eBPF verification techniques from CPU to GPU contexts, and
- Primarily about memory safety and bounded loops.

In reality, verification for gpu_ext is fundamentally **more critical, nuanced, and system-oriented**.

---

## 2. The True Motivation for Verification in gpu_ext

Verification in gpu_ext is driven by a twofold fundamental requirement:

**(A) System Availability and Liveness (Safety)**:

“fail torence”

In modern GPU deployments, the device operates as a multi-tenant, shared, privileged resource. Policy code that interacts with GPU kernels and driver paths inherently executes in a sensitive context. Unlike CPUs—where OS-level isolation, preemption, and fault recovery mechanisms are robust and granular—GPUs have limited isolation.

Consequence:

- Even minor policy bugs (e.g., divergence-induced deadlocks or GPU-wide synchronization errors) may cause irrecoverable stalls, affecting all co-located tenants. The severity is not just degraded performance but potentially complete device hangs, forced resets, and job failures.

**(B) Predictable and Bounded Interference (Performance)**:

“resource”

In shared-GPU environments, unbounded or unpredictable overhead from a single tenant’s policy translates into catastrophic tail-latency inflation for co-located jobs. Thus, bounded interference is itself a safety property in multi-tenant GPUs—one that traditional CPU verifiers do not model explicitly.

---

## 3. Clarifying the Differences from Traditional eBPF Verification

| Aspect | Traditional eBPF (CPU) | gpu_ext Verification (GPU-specific) |
| --- | --- | --- |
| Safety focus | Memory safety, loop bounds, termination. | SIMT-aware liveness, divergence-induced deadlocks, device-level hangs. |
| Performance concern | Instruction count limit (basic bound). | Bounded overhead under SIMT, per-warp efficiency, predictable execution costs to prevent DoS-like interference. |
| Execution model | Single-threaded abstraction. | Warp-uniform control flow, side-effects, and synchronization; explicit modeling of SIMT execution semantics. |
| Consequences of failure | Kernel-level panic/crash (serious but CPU isolation usually allows controlled recovery). | Global GPU hang/reset, severe service disruption (shared GPU isolation weaker, fault tolerance granularity coarse). |

Thus, gpu_ext verification is not simply an eBPF port; it is an **availability and performance guarantee** for dynamic policies within a GPU’s unique execution and failure domain.

---

## 4. A Concrete Failure Example: Why SIMT Awareness Is Required

To explicitly demonstrate why GPU verification is fundamentally necessary, consider this simple scenario, which passes traditional CPU-style verification:

```c
// Example GPU policy (pseudocode):
if (thread_id %2 ==0) {
    map_update(shared_counter, key=thread_id, val=1);
    helper_log_event(...);
}

```

**CPU-style verification** would accept this easily:

- It is memory-safe.
- Loop execution is trivially bounded (no loops).
- Helper use is controlled.

**However, GPU semantics differ dramatically:**

- `thread_id % 2` introduces **lane-divergent control flow**, breaking warp-level execution assumptions.
- Lane-divergent map updates cause massive serialization and potential contention, significantly amplifying overhead.
- Divergence interacting with GPU-specific synchronization can produce deadlocks or catastrophic stalls.

**Consequences on GPU:**

- Warp stall/hang.
- Possible SM-level stall or full-device hang if synchronization assumptions are violated.
- Unpredictable tail latency inflation (DoS-level interference) for other tenants.

Thus, without explicit SIMT-aware constraints, a policy that is safe and performant on CPU is catastrophically unsafe and unpredictable on GPU.

---

## 5. A Clear Mapping from Verification Rules to Prevented GPU Failure Modes

This explicit mapping demonstrates precisely how the verifier rules correspond to GPU-specific hazards:

| gpu_ext Verification Rule | GPU-Specific Failure Prevented |
| --- | --- |
| Warp-uniform control flow (predicates, loop bounds). | Divergence-induced deadlock, lane-inconsistent state, unpredictable warp-level stalls. |
| Warp-uniform side effects (helpers, maps, keys). | Data-structure corruption, severe contention, lane-divergent serialization that inflates tail latency. |
| No device-wide barriers or global synchronization within policies. | Cross-SM deadlock, global GPU hang, irrecoverable stalls. |
| Restrict lane-divergent atomic operations. | Pathological contention, massive amplification of overhead, unpredictable latency spikes. |
| Bounded work per hook invocation. | Unbounded execution overhead, soft hangs, denial-of-service in multi-tenant contexts. |

---

## 6. Clarifying the Fundamental Contract (What gpu_ext Guarantees)

Clearly state in the paper as follows:

> gpu_ext verification guarantees that a dynamically loaded GPU policy:
> 
> 1. **Cannot induce GPU-level hangs or deadlocks** caused by SIMT-specific divergence or synchronization hazards.
> 2. **Has bounded per-invocation overhead**, eliminating performance-based denial-of-service threats in shared-GPU settings.

This contract is the foundational motivation for verification, not merely a secondary property or optimization.

---

## 7. Recommended Rewrite (Directly Suitable for OSDI Paper)

Insert clearly and explicitly (e.g., in Design / Threat Model):

> Why GPU Verification is Fundamentally Necessary.
> 
> 
> GPU policy code executes in privileged, non-fault-tolerant contexts on shared, multi-tenant GPUs. Unlike CPUs, GPUs have limited isolation, recovery, and fine-grained fault tolerance. GPU-specific SIMT execution semantics introduce new classes of failure modes that do not exist in traditional CPU environments, including warp-divergence-induced deadlocks, catastrophic stalls, and unpredictable overhead amplification. Without explicit SIMT-aware verification, seemingly benign policy bugs can easily escalate to global GPU-level failures or severe performance interference, impacting all co-located workloads. Thus, traditional CPU eBPF-style verification (memory safety and loop boundedness) alone is insufficient.
> 
> Our verification framework explicitly prevents GPU-specific hazards by enforcing warp-uniform control flow and side-effects, disallowing unsafe synchronization primitives, and bounding execution overhead. This ensures operationally critical guarantees: **(1)** dynamic policies cannot hang the GPU, and **(2)** policy execution overhead is predictably bounded, ensuring safe deployment in shared GPU environments.
> 
