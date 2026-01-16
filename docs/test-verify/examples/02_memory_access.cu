/**
 * Example 2: Non-Coalesced Memory Access in eBPF Hook
 *
 * Structure:
 *   - eBPF Hook (device function): Simulates eBPF program attached to kernel
 *   - Original CUDA Kernel: User's application code
 *
 * The eBPF hook would PASS traditional eBPF verification:
 *   ✓ Memory safe (all accesses through valid helpers)
 *   ✓ Bounded execution
 *   ✓ Valid helper usage
 *
 * But causes GPU-specific issues:
 *   ✗ Non-coalesced memory access patterns
 *   ✗ Each thread accesses different cache line
 *   ✗ Wastes memory bandwidth (up to 32x)
 */

#include <stdio.h>
#include <cuda_runtime.h>

//=============================================================================
// Simulated eBPF Infrastructure (provided by bpftime)
//=============================================================================

#define MAP_SIZE (1024 * 1024)

// Simulated BPF map in GPU memory (large enough for strided access)
__device__ unsigned long long bpf_map[MAP_SIZE];

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

// eBPF Helper: Map lookup (returns pointer to map element)
__device__ unsigned long long* bpf_map_lookup_elem(unsigned long long *map, unsigned long long key) {
    if (key < MAP_SIZE) return &map[key];
    return nullptr;
}

// eBPF Helper: Map update
__device__ void bpf_map_update_elem(unsigned long long *map, unsigned long long key, unsigned long long val) {
    if (key < MAP_SIZE) {
        map[key] = val;  // Direct write (not atomic for better memory pattern visibility)
    }
}

//=============================================================================
// eBPF HOOK - BAD: Non-coalesced (strided) memory access
//=============================================================================

#define STRIDE 32  // Each thread accesses different cache line

/**
 * This eBPF program has non-coalesced memory access.
 *
 * Traditional eBPF verifier sees:
 *   - Memory access through valid helpers
 *   - Keys are bounded (within map size)
 *   - No invalid memory operations
 *
 * GPU reality:
 *   - Thread 0 accesses map[0], thread 1 accesses map[32], thread 2 accesses map[64]...
 *   - 32 threads in a warp access 32 different cache lines!
 *   - Instead of 1 memory transaction, GPU issues 32 transactions
 *   - Wastes 32x memory bandwidth
 */
__device__ void ebpf_hook_BAD_strided() {
    unsigned long long tid, ty, tz;
    bpf_get_thread_idx(&tid, &ty, &tz);
    unsigned long long ts = bpf_get_globaltimer();

    // NON-COALESCED: Strided access pattern
    // Thread i accesses map[i * STRIDE] - each hits different cache line
    unsigned long long key = (tid * STRIDE) % MAP_SIZE;
    bpf_map_update_elem(bpf_map, key, ts);

    // Read back with same strided pattern
    unsigned long long *val = bpf_map_lookup_elem(bpf_map, key);
    if (val) {
        bpf_map_update_elem(bpf_map, (key + 1) % MAP_SIZE, *val + 1);
    }
}

//=============================================================================
// eBPF HOOK - BAD: Random memory access (even worse)
//=============================================================================

/**
 * Random access pattern - worst case scenario
 */
__device__ void ebpf_hook_BAD_random() {
    unsigned long long tid, ty, tz;
    bpf_get_thread_idx(&tid, &ty, &tz);
    unsigned long long ts = bpf_get_globaltimer();

    // RANDOM: Hash-based access - completely unpredictable
    // No two adjacent threads access adjacent memory
    unsigned long long hash = (tid * 2654435761ULL) % MAP_SIZE;
    bpf_map_update_elem(bpf_map, hash, ts);

    unsigned long long *val = bpf_map_lookup_elem(bpf_map, hash);
    if (val) {
        unsigned long long hash2 = (hash * 2654435761ULL) % MAP_SIZE;
        bpf_map_update_elem(bpf_map, hash2, *val + 1);
    }
}

//=============================================================================
// eBPF HOOK - GOOD: Coalesced memory access
//=============================================================================

/**
 * GPU-aware eBPF: Coalesced access pattern
 * Adjacent threads access adjacent memory locations
 */
__device__ void ebpf_hook_GOOD() {
    unsigned long long tid, ty, tz;
    bpf_get_thread_idx(&tid, &ty, &tz);
    unsigned long long ts = bpf_get_globaltimer();

    // COALESCED: Thread i accesses map[i]
    // All 32 threads in warp access consecutive elements
    // GPU combines into 1 memory transaction (128 bytes)
    bpf_map_update_elem(bpf_map, tid % MAP_SIZE, ts);

    unsigned long long *val = bpf_map_lookup_elem(bpf_map, tid % MAP_SIZE);
    if (val) {
        bpf_map_update_elem(bpf_map, (tid + MAP_SIZE/2) % MAP_SIZE, *val + 1);
    }
}

//=============================================================================
// Original CUDA Kernel (User's Application)
//=============================================================================

__global__ void compute_with_strided_hook(float *data, int n) {
    ebpf_hook_BAD_strided();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f + 1.0f;
    }
}

__global__ void compute_with_random_hook(float *data, int n) {
    ebpf_hook_BAD_random();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f + 1.0f;
    }
}

__global__ void compute_with_good_hook(float *data, int n) {
    ebpf_hook_GOOD();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f + 1.0f;
    }
}

__global__ void compute_no_hook(float *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f + 1.0f;
    }
}

//=============================================================================
// Main
//=============================================================================

int main() {
    const int N = 1024 * 1024 * 4;
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
    printf("║  Example 2: Non-Coalesced Memory Access in eBPF Hook          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    // Warmup
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    compute_no_hook<<<BLOCKS, THREADS>>>(d_data, N);
    cudaDeviceSynchronize();

    // Baseline
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        compute_no_hook<<<BLOCKS, THREADS>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float baseline;
    cudaEventElapsedTime(&baseline, start, stop);

    // Good hook (coalesced)
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        compute_with_good_hook<<<BLOCKS, THREADS>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float good_time;
    cudaEventElapsedTime(&good_time, start, stop);

    // Bad hook (strided)
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        compute_with_strided_hook<<<BLOCKS, THREADS>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float strided_time;
    cudaEventElapsedTime(&strided_time, start, stop);

    // Bad hook (random)
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        compute_with_random_hook<<<BLOCKS, THREADS>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float random_time;
    cudaEventElapsedTime(&random_time, start, stop);

    printf("Results (%d iterations):\n", ITERATIONS);
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  No hook (baseline):        %8.2f ms\n", baseline);
    printf("  GOOD hook (coalesced):     %8.2f ms  (%.2fx overhead)\n", good_time, good_time/baseline);
    printf("  BAD hook (strided):        %8.2f ms  (%.2fx overhead)\n", strided_time, strided_time/baseline);
    printf("  BAD hook (random):         %8.2f ms  (%.2fx overhead)\n\n", random_time, random_time/baseline);

    printf("Performance Impact:\n");
    printf("  Strided vs GOOD: %.2fx slower\n", strided_time / good_time);
    printf("  Random vs GOOD:  %.2fx slower\n\n", random_time / good_time);

    printf("Analysis:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("Memory Access Patterns:\n");
    printf("  GOOD:    Thread 0→map[0], Thread 1→map[1], ... (consecutive)\n");
    printf("           → 32 threads = 1 memory transaction\n\n");
    printf("  Strided: Thread 0→map[0], Thread 1→map[32], Thread 2→map[64]...\n");
    printf("           → 32 threads = 32 memory transactions (32x waste)\n\n");
    printf("  Random:  Thread i→map[hash(i)] (unpredictable)\n");
    printf("           → No coalescing possible, maximum bandwidth waste\n\n");

    printf("Traditional eBPF Verifier:  PASS (memory access is bounded)\n");
    printf("GPU-aware Verifier:         REJECT (non-coalesced access pattern)\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
