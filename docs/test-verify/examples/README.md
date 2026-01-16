# GPU eBPF Verification Gap Examples

These examples demonstrate why traditional eBPF verification is **insufficient** for GPU execution. Each example shows code that would pass CPU-style eBPF verification but causes serious issues on GPU due to SIMT execution semantics.

## Architecture

When eBPF programs are attached to CUDA kernels via bpftime, they execute as device functions at kernel entry/exit:

```
┌───────────────────────────────────────────────────────────────────┐
│                         CUDA Kernel                                │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ __global__ void userKernel(...) {                           │  │
│  │                                                             │  │
│  │     // ═══════════════════════════════════════════════════  │  │
│  │     // eBPF HOOK (Entry) - Injected by bpftime              │  │
│  │     // Uses: bpf_get_thread_idx(), bpf_map_update_elem()    │  │
│  │     // NO: __syncthreads(), shared memory                   │  │
│  │     // ═══════════════════════════════════════════════════  │  │
│  │     ebpf_hook();                                            │  │
│  │                                                             │  │
│  │     // ─────────────────────────────────────────────────    │  │
│  │     // ORIGINAL CUDA CODE                                   │  │
│  │     // May use __syncthreads(), shared memory, etc.         │  │
│  │     // ─────────────────────────────────────────────────    │  │
│  │     ... user's kernel code ...                              │  │
│  │ }                                                           │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

## Examples

| # | Example | Issue | Impact |
|---|---------|-------|--------|
| 1 | Thread Divergence | `if (tid % N)` in eBPF hook | Warp serialization, 2-4x slowdown |
| 2 | Memory Access | Non-coalesced map access | Bandwidth waste, 4-32x slowdown |
| 3 | Atomic Contention | All threads update same counter | Massive serialization, 10-100x slowdown |
| 4 | Deadlock | Spinlock/producer-consumer patterns | GPU hang, requires reset |

### Example 1: Thread Divergence

eBPF hook with divergent control flow based on thread ID.

```cuda
// BAD: Different threads take different paths
if (tid % 4 == 0) {
    bpf_map_update_elem(map, tid, val_a);
} else if (tid % 4 == 1) {
    bpf_map_update_elem(map, tid, val_b);
} ...
```

**Traditional eBPF**: PASS (all paths bounded, memory safe)
**GPU Reality**: 4-way warp divergence, 75% cycles wasted

### Example 2: Memory Access

eBPF hook with non-coalesced memory access patterns.

```cuda
// BAD: Strided access - each thread hits different cache line
key = tid * 32;  // stride of 32
bpf_map_update_elem(map, key, val);
```

**Traditional eBPF**: PASS (key is bounded)
**GPU Reality**: 32 memory transactions instead of 1, 32x bandwidth waste

### Example 3: Atomic Contention

eBPF hook where all threads atomically update same location.

```cuda
// BAD: All threads contend on single counter
bpf_atomic_inc(&global_counter);
```

**Traditional eBPF**: PASS (single atomic operation)
**GPU Reality**: 1M threads serialize on one address, catastrophic slowdown

### Example 4: Deadlock

eBPF hook with synchronization patterns that deadlock in SIMT.

```cuda
// BAD: Spinlock within warp
while (atomicCAS(&lock, 0, 1) != 0) { }  // Acquire
// critical section
atomicExch(&lock, 0);  // Release
```

**Traditional eBPF**: PASS (loop has exit condition)
**GPU Reality**: Warp cannot converge, DEADLOCK

## Verification Comparison

| Check | Traditional eBPF | GPU-Aware eBPF |
|-------|------------------|----------------|
| Memory Safety | ✓ | ✓ |
| Bounded Loops | ✓ | ✓ |
| Valid Helpers | ✓ | ✓ |
| Warp-Uniform Control Flow | - | ✓ Required |
| Warp-Uniform Side Effects | - | ✓ Required |
| Bounded Atomic Contention | - | ✓ Required |
| Coalesced Memory Access | - | ✓ Required |
| No Intra-Warp Sync | - | ✓ Required |

## Building & Running

```bash
make all     # Compile all examples
make run     # Run all examples (deadlock patterns are explained, not executed)

# Run individual examples
make run1    # Thread divergence
make run2    # Memory access
make run3    # Atomic contention
make run4    # Deadlock (explanation only)
```

## References

- [explain.md](../explain.md) - Full motivation for GPU-aware verification
- [bpftime GPU support](https://github.com/eunomia-bpf/bpftime) - eBPF on GPU implementation
