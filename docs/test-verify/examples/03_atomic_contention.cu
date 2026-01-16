/**
 * Example 3: Atomic Contention in eBPF Hook
 *
 * Structure:
 *   - eBPF Hook (device function): Simulates eBPF program attached to kernel
 *   - Original CUDA Kernel: User's application code
 *
 * The eBPF hook would PASS traditional eBPF verification:
 *   ✓ Memory safe (atomic operations are valid)
 *   ✓ Bounded execution (single atomic op)
 *   ✓ Valid helper usage
 *
 * But causes GPU-specific issues:
 *   ✗ All threads atomically update same memory location
 *   ✗ Serializes across entire grid (millions of threads)
 *   ✗ Creates DoS-level performance interference
 */

#include <stdio.h>
#include <cuda_runtime.h>

//=============================================================================
// Simulated eBPF Infrastructure (provided by bpftime)
//=============================================================================

// Simulated BPF map - single counter (like BPF_MAP_TYPE_ARRAY with 1 entry)
__device__ unsigned long long bpf_counter = 0;

// Simulated per-CPU (per-thread) counters
__device__ unsigned long long bpf_percpu_counters[1024 * 1024];

// eBPF Helper: Get thread index
__device__ void bpf_get_thread_idx(unsigned long long *x, unsigned long long *y, unsigned long long *z) {
    *x = threadIdx.x + blockIdx.x * blockDim.x;
    *y = threadIdx.y;
    *z = threadIdx.z;
}

// eBPF Helper: Atomic increment on map value (common eBPF pattern)
__device__ void bpf_map_atomic_inc(unsigned long long *counter) {
    atomicAdd(counter, 1ULL);
}

// eBPF Helper: Per-thread counter update (no contention)
__device__ void bpf_percpu_map_inc(unsigned long long *counters, unsigned long long tid) {
    if (tid < 1024 * 1024) {
        counters[tid] += 1;
    }
}

//=============================================================================
// eBPF HOOK - BAD: All threads contend on single counter
//=============================================================================

/**
 * This eBPF program causes massive atomic contention.
 *
 * Traditional eBPF verifier sees:
 *   - Single atomic operation (bounded)
 *   - Valid memory target
 *   - No loops
 *
 * GPU reality:
 *   - All N threads (e.g., 1,048,576) atomically increment SAME counter
 *   - GPU must serialize ALL atomic operations
 *   - Time = O(N) instead of O(1)
 *   - In multi-tenant GPU, this is a denial-of-service attack vector
 */
__device__ void ebpf_hook_BAD() {
    // Every thread increments the SAME global counter
    // This is a common eBPF pattern for counting events
    bpf_map_atomic_inc(&bpf_counter);
}

//=============================================================================
// eBPF HOOK - MEDIUM: Reduced contention (per-warp counters)
//=============================================================================

#define NUM_WARP_COUNTERS 32768  // One counter per warp (for large grids)
__device__ unsigned long long warp_counters[NUM_WARP_COUNTERS];

/**
 * Slightly better: one counter per warp
 * Still has contention but 32x less than single counter
 */
__device__ void ebpf_hook_MEDIUM() {
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long warp_id = tid / 32;

    // Only lane 0 of each warp increments
    if (tid % 32 == 0) {
        atomicAdd(&warp_counters[warp_id % NUM_WARP_COUNTERS], 32ULL);
    }
}

//=============================================================================
// eBPF HOOK - GOOD: Per-thread counters (no contention)
//=============================================================================

/**
 * GPU-aware eBPF: Per-thread counters, reduce later
 * No atomic contention - each thread updates its own counter
 */
__device__ void ebpf_hook_GOOD() {
    unsigned long long tid, ty, tz;
    bpf_get_thread_idx(&tid, &ty, &tz);

    // Each thread updates its OWN counter - zero contention
    bpf_percpu_map_inc(bpf_percpu_counters, tid);
}

//=============================================================================
// Original CUDA Kernel (User's Application)
//=============================================================================

__global__ void compute_with_bad_hook(float *data, int n) {
    ebpf_hook_BAD();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

__global__ void compute_with_medium_hook(float *data, int n) {
    ebpf_hook_MEDIUM();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

__global__ void compute_with_good_hook(float *data, int n) {
    ebpf_hook_GOOD();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

__global__ void compute_no_hook(float *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

// Reset counters
__global__ void reset_counters() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) bpf_counter = 0;
    if (tid < NUM_WARP_COUNTERS) warp_counters[tid] = 0;
    if (tid < 1024 * 1024) bpf_percpu_counters[tid] = 0;
}

//=============================================================================
// Main
//=============================================================================

int main() {
    const int N = 1024 * 1024;  // 1M threads
    const int SIZE = N * sizeof(float);
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int ITERATIONS = 20;  // Fewer iterations because BAD case is very slow

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
    printf("║  Example 3: Atomic Contention in eBPF Hook                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    printf("Configuration: %d threads total\n\n", N);

    // Warmup & reset
    reset_counters<<<BLOCKS, THREADS>>>();
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    compute_no_hook<<<BLOCKS, THREADS>>>(d_data, N);
    cudaDeviceSynchronize();

    // Baseline
    reset_counters<<<BLOCKS, THREADS>>>();
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        compute_no_hook<<<BLOCKS, THREADS>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float baseline;
    cudaEventElapsedTime(&baseline, start, stop);

    // Good hook
    reset_counters<<<BLOCKS, THREADS>>>();
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        compute_with_good_hook<<<BLOCKS, THREADS>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float good_time;
    cudaEventElapsedTime(&good_time, start, stop);

    // Medium hook
    reset_counters<<<BLOCKS, THREADS>>>();
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        compute_with_medium_hook<<<BLOCKS, THREADS>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float medium_time;
    cudaEventElapsedTime(&medium_time, start, stop);

    // Bad hook
    reset_counters<<<BLOCKS, THREADS>>>();
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        compute_with_bad_hook<<<BLOCKS, THREADS>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float bad_time;
    cudaEventElapsedTime(&bad_time, start, stop);

    printf("Results (%d iterations):\n", ITERATIONS);
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  No hook (baseline):           %8.2f ms\n", baseline);
    printf("  GOOD hook (per-thread):       %8.2f ms  (%.2fx overhead)\n", good_time, good_time/baseline);
    printf("  MEDIUM hook (per-warp):       %8.2f ms  (%.2fx overhead)\n", medium_time, medium_time/baseline);
    printf("  BAD hook (single counter):    %8.2f ms  (%.2fx overhead)\n\n", bad_time, bad_time/baseline);

    printf("Performance Impact:\n");
    printf("  BAD vs GOOD:    %.2fx slower\n", bad_time / good_time);
    printf("  MEDIUM vs GOOD: %.2fx slower\n\n", medium_time / good_time);

    printf("Analysis:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("Atomic Contention Levels:\n");
    printf("  BAD:    %d threads → 1 counter = %d-way contention\n", N, N);
    printf("  MEDIUM: %d threads → %d counters = %dx less contention\n", N, N/32, 32);
    printf("  GOOD:   %d threads → %d counters = 0 contention\n\n", N, N);

    printf("This is a common eBPF pattern:\n");
    printf("  bpf_map_atomic_inc(&event_counter);\n\n");
    printf("On CPU: Fine - single thread increments\n");
    printf("On GPU: Catastrophic - %d threads ALL serialize on same counter\n\n", N);

    printf("Traditional eBPF Verifier:  PASS (single bounded atomic)\n");
    printf("GPU-aware Verifier:         REJECT (unbounded contention)\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
