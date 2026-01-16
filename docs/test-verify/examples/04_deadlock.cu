/**
 * Example 4: Deadlock Patterns in eBPF Hook
 *
 * Structure:
 *   - eBPF Hook (device function): Simulates eBPF program attached to kernel
 *   - Original CUDA Kernel: User's application code
 *
 * The eBPF hook would PASS traditional eBPF verification:
 *   ✓ Memory safe
 *   ✓ Loop has termination condition (flag becomes 1)
 *   ✓ Valid helper usage
 *
 * But causes GPU-specific DEADLOCK:
 *   ✗ Spinlock pattern within warp causes circular wait
 *   ✗ Producer-consumer pattern deadlocks in SIMT
 *   ✗ Data-dependent waiting on other threads' results
 *
 * WARNING: The actual deadlock kernels are NOT executed to avoid hanging GPU.
 *          This example demonstrates the patterns and explains why they fail.
 */

#include <stdio.h>
#include <cuda_runtime.h>

//=============================================================================
// Simulated eBPF Infrastructure (provided by bpftime)
//=============================================================================

// Simulated BPF map for lock/flag
__device__ int bpf_lock = 0;
__device__ int bpf_ready_flag = 0;
__device__ unsigned long long bpf_shared_data = 0;
__device__ unsigned long long bpf_results[1024];

// eBPF Helper: Get thread index
__device__ void bpf_get_thread_idx(unsigned long long *x, unsigned long long *y, unsigned long long *z) {
    *x = threadIdx.x + blockIdx.x * blockDim.x;
    *y = threadIdx.y;
    *z = threadIdx.z;
}

// eBPF Helper: Get global timer
__device__ unsigned long long bpf_get_globaltimer() {
    unsigned long long timer;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
    return timer;
}

// eBPF Helper: Atomic CAS (used for spinlock)
__device__ int bpf_atomic_cas(int *addr, int expected, int desired) {
    return atomicCAS(addr, expected, desired);
}

// eBPF Helper: Atomic exchange
__device__ int bpf_atomic_xchg(int *addr, int val) {
    return atomicExch(addr, val);
}

// eBPF Helper: Read flag
__device__ int bpf_read_flag(volatile int *flag) {
    return *flag;
}

//=============================================================================
// eBPF HOOK - DEADLOCK: Spinlock Pattern
//=============================================================================

/**
 * DEADLOCK Pattern 1: Spinlock within warp
 *
 * Traditional eBPF verifier sees:
 *   - Loop terminates when lock becomes 0
 *   - This is a valid termination condition
 *   - Memory operations are safe
 *
 * GPU reality (SIMT execution):
 *   - Thread 0 acquires lock (CAS succeeds, lock=1)
 *   - Threads 1-31 spin waiting (CAS fails, lock≠0)
 *   - Thread 0 cannot proceed to release lock because:
 *     → SIMT requires warp convergence before continuing
 *     → Threads 1-31 are stuck in while loop
 *     → Thread 0 waits for them to exit loop
 *     → Threads 1-31 wait for Thread 0 to release lock
 *   - CIRCULAR DEPENDENCY → DEADLOCK
 */
__device__ void ebpf_hook_DEADLOCK_spinlock() {
    unsigned long long tid, ty, tz;
    bpf_get_thread_idx(&tid, &ty, &tz);

    // Try to acquire spinlock
    // DEADLOCK: Thread 0 gets lock, threads 1-31 spin forever
    while (bpf_atomic_cas(&bpf_lock, 0, 1) != 0) {
        // Spin wait - looks like it will eventually terminate
        // But due to SIMT, it NEVER will
    }

    // Critical section (never reached by threads 1-31)
    bpf_results[tid] = bpf_get_globaltimer();

    // Release lock (thread 0 never reaches here due to warp stall)
    bpf_atomic_xchg(&bpf_lock, 0);
}

//=============================================================================
// eBPF HOOK - DEADLOCK: Producer-Consumer Pattern
//=============================================================================

/**
 * DEADLOCK Pattern 2: Producer-Consumer within warp
 *
 * Traditional eBPF verifier sees:
 *   - Producer sets flag to 1 (bounded operation)
 *   - Consumer waits for flag==1 (will terminate when flag is set)
 *   - Looks like valid synchronization
 *
 * GPU reality:
 *   - Thread 0 (producer) enters if-branch
 *   - Threads 1-31 (consumers) enter else-branch
 *   - Consumers spin waiting for flag
 *   - Producer cannot set flag because warp cannot converge
 *   - DEADLOCK
 */
__device__ void ebpf_hook_DEADLOCK_producer_consumer() {
    unsigned long long tid, ty, tz;
    bpf_get_thread_idx(&tid, &ty, &tz);

    if (tid % 32 == 0) {
        // Producer (thread 0 of each warp)
        bpf_shared_data = bpf_get_globaltimer();
        __threadfence();  // Ensure data is visible

        // Signal ready - but warp can't converge to reach here!
        bpf_atomic_xchg(&bpf_ready_flag, 1);
    } else {
        // Consumers (threads 1-31)
        // Wait for producer to signal
        while (bpf_read_flag(&bpf_ready_flag) == 0) {
            // Spin wait - DEADLOCK because producer can't set flag
        }

        // Use shared data (never reached)
        bpf_results[tid] = bpf_shared_data + tid;
    }
}

//=============================================================================
// eBPF HOOK - DEADLOCK: Cross-Thread Data Dependency
//=============================================================================

/**
 * DEADLOCK Pattern 3: Waiting for another thread's result
 *
 * Traditional eBPF verifier sees:
 *   - Thread waits for another thread to write data
 *   - Loop terminates when data is non-zero
 *   - Bounded wait pattern
 *
 * GPU reality:
 *   - Thread i waits for thread i-1 to write result
 *   - Thread 0 writes immediately, thread 1 waits for thread 0
 *   - But they're in same warp - thread 1 can't proceed until thread 0's write
 *   - Thread 0's write won't be visible until warp converges
 *   - DEADLOCK (or at minimum, severe serialization)
 */
__device__ void ebpf_hook_DEADLOCK_data_dependency() {
    unsigned long long tid, ty, tz;
    bpf_get_thread_idx(&tid, &ty, &tz);

    unsigned long long lane = tid % 32;

    if (lane == 0) {
        // First thread writes immediately
        bpf_results[tid] = bpf_get_globaltimer();
    } else {
        // Other threads wait for previous thread
        // DEADLOCK: Previous thread's write won't complete until warp converges
        while (bpf_results[tid - 1] == 0) {
            // Spin wait for previous thread
        }
        bpf_results[tid] = bpf_results[tid - 1] + 1;
    }
}

//=============================================================================
// eBPF HOOK - SAFE: No cross-thread dependencies
//=============================================================================

/**
 * SAFE Pattern: Each thread works independently
 */
__device__ void ebpf_hook_SAFE() {
    unsigned long long tid, ty, tz;
    bpf_get_thread_idx(&tid, &ty, &tz);

    // Each thread works on its own data - no waiting for others
    bpf_results[tid % 1024] = bpf_get_globaltimer();
}

//=============================================================================
// Original CUDA Kernels
//=============================================================================

// Safe kernel (for timing comparison)
__global__ void kernel_safe(float *data, int n) {
    ebpf_hook_SAFE();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

// Deadlock kernels - DO NOT RUN (will hang GPU)
__global__ void kernel_deadlock_spinlock(float *data, int n) {
    ebpf_hook_DEADLOCK_spinlock();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

__global__ void kernel_deadlock_producer_consumer(float *data, int n) {
    ebpf_hook_DEADLOCK_producer_consumer();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

__global__ void kernel_deadlock_data_dependency(float *data, int n) {
    ebpf_hook_DEADLOCK_data_dependency();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

__global__ void kernel_no_hook(float *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

//=============================================================================
// Main
//=============================================================================

int main() {
    const int N = 1024 * 1024;
    const int SIZE = N * sizeof(float);
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int ITERATIONS = 100;

    float *h_data = (float*)malloc(SIZE);
    float *d_data;

    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    cudaMalloc(&d_data, SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Example 4: Deadlock Patterns in eBPF Hook                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    printf("WARNING: Actual deadlock kernels are NOT executed.\n");
    printf("         Running them would hang the GPU and require reset.\n\n");

    // Run only safe kernels
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    kernel_no_hook<<<BLOCKS, THREADS>>>(d_data, N);
    cudaDeviceSynchronize();

    // Time baseline
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        kernel_no_hook<<<BLOCKS, THREADS>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float baseline;
    cudaEventElapsedTime(&baseline, start, stop);

    // Time safe hook
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        kernel_safe<<<BLOCKS, THREADS>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float safe_time;
    cudaEventElapsedTime(&safe_time, start, stop);

    printf("Safe Execution Times (%d iterations):\n", ITERATIONS);
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  No hook (baseline): %8.2f ms\n", baseline);
    printf("  SAFE hook:          %8.2f ms  (%.2fx overhead)\n\n", safe_time, safe_time/baseline);

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("                    DEADLOCK PATTERN ANALYSIS                       \n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("Pattern 1: SPINLOCK DEADLOCK\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  Code:\n");
    printf("    while (atomicCAS(&lock, 0, 1) != 0) { }  // Acquire\n");
    printf("    // critical section\n");
    printf("    atomicExch(&lock, 0);                    // Release\n\n");
    printf("  Why it deadlocks:\n");
    printf("    1. Thread 0 acquires lock (CAS returns 0, sets lock=1)\n");
    printf("    2. Threads 1-31 spin (CAS returns 1, lock unchanged)\n");
    printf("    3. Thread 0 cannot release: waiting for warp to converge\n");
    printf("    4. Threads 1-31 cannot exit loop: waiting for lock release\n");
    printf("    → Circular dependency = DEADLOCK\n\n");
    printf("  Traditional eBPF Verifier: PASS (loop has exit condition)\n");
    printf("  GPU-aware Verifier:        REJECT (intra-warp spinlock)\n\n");

    printf("Pattern 2: PRODUCER-CONSUMER DEADLOCK\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  Code:\n");
    printf("    if (tid == 0) {\n");
    printf("        data = compute();\n");
    printf("        flag = 1;  // Signal consumers\n");
    printf("    } else {\n");
    printf("        while (flag == 0);  // Wait for producer\n");
    printf("        use(data);\n");
    printf("    }\n\n");
    printf("  Why it deadlocks:\n");
    printf("    1. Thread 0 (producer) enters if-branch\n");
    printf("    2. Threads 1-31 (consumers) enter else-branch\n");
    printf("    3. Consumers spin on flag==0\n");
    printf("    4. Producer cannot set flag: warp diverged, can't converge\n");
    printf("    → Producer waits for consumers, consumers wait for producer\n\n");
    printf("  Traditional eBPF Verifier: PASS (flag will become 1)\n");
    printf("  GPU-aware Verifier:        REJECT (divergent sync pattern)\n\n");

    printf("Pattern 3: DATA DEPENDENCY DEADLOCK\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  Code:\n");
    printf("    if (lane == 0) {\n");
    printf("        results[tid] = compute();\n");
    printf("    } else {\n");
    printf("        while (results[tid-1] == 0);  // Wait for prev thread\n");
    printf("        results[tid] = results[tid-1] + 1;\n");
    printf("    }\n\n");
    printf("  Why it deadlocks:\n");
    printf("    1. Thread 0 writes immediately\n");
    printf("    2. Thread 1 waits for thread 0's write to be visible\n");
    printf("    3. Thread 0's write won't complete until warp converges\n");
    printf("    4. Warp can't converge: thread 1 is stuck in while loop\n");
    printf("    → Chain of waiting = DEADLOCK\n\n");
    printf("  Traditional eBPF Verifier: PASS (data will be written)\n");
    printf("  GPU-aware Verifier:        REJECT (cross-lane dependency)\n\n");

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("                         KEY TAKEAWAY                               \n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    printf("These patterns look SAFE to traditional eBPF verifier because:\n");
    printf("  • Loops have termination conditions\n");
    printf("  • Memory operations are bounded\n");
    printf("  • No infinite loops apparent\n\n");
    printf("But they DEADLOCK on GPU because:\n");
    printf("  • SIMT execution requires warp convergence\n");
    printf("  • Threads in same warp cannot truly run in parallel\n");
    printf("  • Cross-thread synchronization within warp is impossible\n\n");
    printf("GPU-aware verification must REJECT:\n");
    printf("  • Spinlock patterns (while + atomicCAS)\n");
    printf("  • Divergent producer-consumer patterns\n");
    printf("  • Cross-lane data dependencies in loops\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
