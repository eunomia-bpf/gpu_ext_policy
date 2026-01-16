# Traditional CUDA Programming Issues: Thread Divergence, Deadlocks, and Memory Bottlenecks

This document explains common issues in traditional CUDA programming that can cause severe performance degradation or program hangs, without involving eBPF. These are fundamental GPU programming pitfalls that every CUDA developer should understand.

## 1. Thread Divergence

### What is Thread Divergence?

In NVIDIA GPUs, threads are executed in groups of 32 called **warps**. All threads in a warp execute the same instruction at the same time (SIMT - Single Instruction, Multiple Threads). When threads in a warp take different execution paths due to conditional statements, this is called **thread divergence**.

### When Divergence Occurs

```cuda
// DIVERGENT: Different threads take different branches
if (threadIdx.x % 2 == 0) {
    // Even threads execute this
    do_something_A();
} else {
    // Odd threads execute this
    do_something_B();
}
```

The GPU must execute **both branches sequentially**, with half the warp idle during each phase. This effectively halves the warp's throughput.

### Common Causes of Divergence

1. **Thread ID-based conditionals**: `if (threadIdx.x < N)`, `if (tid % K == 0)`
2. **Data-dependent branches**: `if (data[tid] > threshold)`
3. **Early exit conditions**: `if (tid >= array_size) return;`
4. **Switch statements**: Different threads hitting different cases
5. **Loop bounds varying by thread**: `for (int i = 0; i < data[tid]; i++)`

### Impact

| Scenario | Theoretical Throughput | Actual Throughput |
|----------|----------------------|-------------------|
| No divergence | 100% | ~100% |
| 50% divergence (2 paths) | 100% | ~50% |
| 75% divergence (4 paths) | 100% | ~25% |

## 2. Deadlocks in CUDA

### What Causes GPU Deadlocks?

Unlike CPUs, GPUs have very limited preemption capabilities. Once a kernel is running, it generally cannot be interrupted. This makes certain synchronization patterns deadly.

### Classic Deadlock Patterns

#### Pattern 1: `__syncthreads()` in Divergent Code

```cuda
// DEADLOCK: __syncthreads() in divergent branch
if (threadIdx.x < 16) {
    // Only first 16 threads reach this barrier
    __syncthreads();  // Deadlock! Other 16 threads never arrive
}
```

`__syncthreads()` requires **all threads in a block** to reach the barrier. If some threads skip it due to a conditional, the kernel hangs forever.

#### Pattern 2: Warp-Level Spinlocks

```cuda
__device__ int lock = 0;

__global__ void spinlock_kernel() {
    // Thread 0 tries to acquire lock
    if (threadIdx.x == 0) {
        while (atomicCAS(&lock, 0, 1) != 0);  // Acquire
        // Critical section
        atomicExch(&lock, 0);  // Release
    } else {
        // Other threads also try to acquire
        while (atomicCAS(&lock, 0, 1) != 0);  // DEADLOCK!
    }
}
```

In SIMT execution, if thread 0 acquires the lock, threads 1-31 spin waiting. But thread 0 cannot proceed to release the lock until the entire warp converges - which never happens because threads 1-31 are stuck spinning.

#### Pattern 3: Producer-Consumer Within Warp

```cuda
__device__ volatile int flag = 0;

__global__ void producer_consumer() {
    if (threadIdx.x == 0) {
        // Producer
        data = compute_value();
        flag = 1;  // Signal ready
    } else {
        // Consumer
        while (flag == 0);  // Wait for producer - DEADLOCK!
        use(data);
    }
}
```

The producer (thread 0) cannot set the flag because it's waiting for the warp to converge, but the consumers are spinning waiting for the flag.

### Why GPUs Are Vulnerable to Deadlocks

1. **No preemption**: Stuck warps cannot be interrupted
2. **SIMT execution**: Threads in a warp execute in lockstep
3. **Limited recovery**: GPU hangs often require device reset
4. **No OS intervention**: GPU scheduler has limited deadlock detection

## 3. Memory Bottlenecks (Roofline Issues)

### Understanding GPU Memory Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Memory Hierarchy                  │
├─────────────────────────────────────────────────────────┤
│  Registers      │ ~256 KB/SM  │ ~8 TB/s   │ 1 cycle    │
│  Shared Memory  │ ~100 KB/SM  │ ~4 TB/s   │ ~5 cycles  │
│  L1 Cache       │ ~128 KB/SM  │ ~2 TB/s   │ ~30 cycles │
│  L2 Cache       │ ~6 MB       │ ~1 TB/s   │ ~200 cycles│
│  Global Memory  │ ~80 GB      │ ~1 TB/s   │ ~400 cycles│
└─────────────────────────────────────────────────────────┘
```

### Memory Coalescing

GPUs achieve high memory bandwidth through **coalescing** - combining multiple thread memory accesses into fewer transactions.

#### Coalesced Access (GOOD)

```cuda
// Consecutive threads access consecutive addresses
// 32 threads → 1 memory transaction (128 bytes)
float val = data[threadIdx.x];
```

#### Non-Coalesced Access (BAD)

```cuda
// Strided access - each thread accesses different cache line
// 32 threads → 32 memory transactions!
float val = data[threadIdx.x * STRIDE];
```

### Common Memory Bottleneck Patterns

#### Pattern 1: Strided Access

```cuda
// BAD: Stride of 32 means each thread hits different cache line
for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
    sum += matrix[i * 32];  // 32x bandwidth waste
}
```

#### Pattern 2: Random Access

```cuda
// BAD: Unpredictable access pattern
int idx = hash(threadIdx.x) % N;
float val = data[idx];  // No coalescing possible
```

#### Pattern 3: Shared Memory Bank Conflicts

```cuda
__shared__ float smem[32][32];

// BAD: All threads access same bank (column 0)
float val = smem[threadIdx.x][0];  // 32-way bank conflict!

// GOOD: Each thread accesses different bank
float val = smem[threadIdx.x][threadIdx.x];  // No conflicts
```

#### Pattern 4: Excessive Global Memory Traffic

```cuda
// BAD: Reading same data multiple times from global memory
for (int i = 0; i < 100; i++) {
    sum += global_data[tid] * coefficients[i];  // global_data read 100x!
}

// GOOD: Cache in register
float local = global_data[tid];
for (int i = 0; i < 100; i++) {
    sum += local * coefficients[i];  // Register access
}
```

### Roofline Model

The Roofline model shows the relationship between computational throughput and memory bandwidth:

```
Performance (FLOPS)
        │
        │           ┌─────────── Peak Compute
        │          /│
        │         / │
        │        /  │
        │       /   │
        │      /    │
        │     /     │
        │    /      │
        │   /       │
        └──/────────┴─────────── Arithmetic Intensity (FLOPS/Byte)
          Memory     Compute
          Bound      Bound
```

- **Memory-bound**: Application limited by memory bandwidth
- **Compute-bound**: Application limited by compute throughput

Most CUDA applications are **memory-bound**, making memory access patterns critical.

## 4. Summary of Issues and Mitigations

| Issue | Cause | Symptom | Mitigation |
|-------|-------|---------|------------|
| Thread Divergence | Conditional branches | 2-32x slowdown | Warp-uniform code paths |
| `__syncthreads()` Deadlock | Barrier in divergent code | Kernel hang | Ensure all threads reach barrier |
| Spinlock Deadlock | Lock in SIMT context | Kernel hang | Use atomic operations differently |
| Non-Coalesced Access | Strided/random patterns | Low bandwidth | Restructure data layout |
| Bank Conflicts | Same shared memory bank | Serialization | Pad or permute access |
| Excessive Memory Traffic | Redundant loads | High latency | Cache in registers/shared |

## 5. Why This Matters for GPU eBPF Verification

These same issues apply when eBPF programs run on GPUs:

1. **eBPF hooks execute on every thread** - divergent eBPF code affects all threads
2. **eBPF map operations are like memory accesses** - non-uniform access patterns cause contention
3. **eBPF helper calls may require synchronization** - divergent helper calls serialize
4. **Traditional eBPF verification doesn't check for these** - GPU-specific verification needed

A GPU-aware eBPF verifier must ensure:
- Warp-uniform control flow
- Warp-uniform side effects (map operations, helper calls)
- Bounded memory contention
- No synchronization patterns that could deadlock

## References

1. NVIDIA CUDA Programming Guide - https://docs.nvidia.com/cuda/cuda-c-programming-guide/
2. Roofline Model - https://crd.lbl.gov/divisions/amcrd/computer-science-amcrd/par/research/roofline/
3. GPU Memory Coalescing - https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
