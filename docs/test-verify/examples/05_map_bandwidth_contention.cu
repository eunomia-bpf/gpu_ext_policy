/**
 * Example 4: eBPF Map Bandwidth Contention - Slowing Down GEMM
 *
 * This example demonstrates how excessive eBPF map operations can
 * turn a compute-bound GEMM kernel into a memory-bound one.
 *
 * Scenario:
 *   - Original kernel: Matrix multiplication (GEMM) - typically compute-bound
 *   - eBPF hook: Attached for monitoring/tracing purposes
 *   - Problem: Excessive map reads/writes saturate memory bandwidth
 *
 * The eBPF hook would PASS traditional eBPF verification:
 *   ✓ Memory safe (all map accesses bounded)
 *   ✓ Bounded execution (finite number of map operations)
 *   ✓ Valid helper usage
 *
 * But causes GPU-specific issues:
 *   ✗ Excessive memory traffic from frequent map reads/writes
 *   ✗ Saturates memory bandwidth, starving GEMM of data
 *   ✗ Converts compute-bound kernel to memory-bound
 *   ✗ Severe performance degradation (10-100x slowdown possible)
 */

#include <stdio.h>
#include <cuda_runtime.h>

//=============================================================================
// Simulated eBPF Infrastructure (provided by bpftime)
//=============================================================================

#define MAP_SIZE (1024 * 1024)
#define EVENT_LOG_SIZE (1024 * 1024 * 4)

// Simulated BPF maps for various monitoring purposes
__device__ unsigned long long bpf_event_log[EVENT_LOG_SIZE];      // Event timestamps
__device__ unsigned long long bpf_thread_state[MAP_SIZE];          // Per-thread state
__device__ unsigned long long bpf_iteration_counts[MAP_SIZE];      // Iteration counters
__device__ unsigned long long bpf_memory_access_log[MAP_SIZE];     // Memory access tracking
__device__ unsigned long long bpf_performance_metrics[MAP_SIZE];   // Performance data

// Global counters
__device__ unsigned long long bpf_total_invocations = 0;
__device__ unsigned long long bpf_total_iterations = 0;

// eBPF Helper: Get global timer
__device__ unsigned long long bpf_get_globaltimer() {
    unsigned long long timer;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
    return timer;
}

// eBPF Helper: Map update (write to map)
__device__ void bpf_map_update(unsigned long long *map, unsigned long long key, unsigned long long val) {
    if (key < MAP_SIZE) {
        map[key] = val;
    }
}

// eBPF Helper: Map lookup (read from map)
__device__ unsigned long long bpf_map_lookup(unsigned long long *map, unsigned long long key) {
    if (key < MAP_SIZE) {
        return map[key];
    }
    return 0;
}

// eBPF Helper: Log event to event ring buffer
__device__ void bpf_log_event(unsigned long long tid, unsigned long long event_type, unsigned long long data) {
    unsigned long long idx = (tid * 4 + event_type) % EVENT_LOG_SIZE;
    bpf_event_log[idx] = data;
}

//=============================================================================
// eBPF HOOK - BAD: Excessive map operations (memory bandwidth hog)
//=============================================================================

/**
 * This eBPF program simulates "over-monitoring" - a common anti-pattern
 * where developers log too much data, causing memory bandwidth saturation.
 *
 * On CPU eBPF: Each event is single-threaded, impact is minimal
 * On GPU eBPF: Millions of threads all doing map operations simultaneously
 *              This SATURATES memory bandwidth
 *
 * Memory operations per thread: ~20 reads + ~20 writes = ~40 memory ops
 * With 1M threads: 40M memory operations competing for bandwidth!
 */
__device__ void ebpf_hook_BAD_bandwidth_hog(int row, int col, int K) {
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long ts_start = bpf_get_globaltimer();

    // ─────────────────────────────────────────────────────────────────────
    // Event logging - EACH of these is a memory write!
    // ─────────────────────────────────────────────────────────────────────

    // Log kernel entry event
    bpf_log_event(tid, 0, ts_start);

    // Update thread state - READ then WRITE
    unsigned long long prev_state = bpf_map_lookup(bpf_thread_state, tid % MAP_SIZE);
    bpf_map_update(bpf_thread_state, tid % MAP_SIZE, prev_state + 1);

    // Log thread coordinates - multiple WRITEs
    bpf_map_update(bpf_memory_access_log, (tid * 3) % MAP_SIZE, row);
    bpf_map_update(bpf_memory_access_log, (tid * 3 + 1) % MAP_SIZE, col);
    bpf_map_update(bpf_memory_access_log, (tid * 3 + 2) % MAP_SIZE, K);

    // ─────────────────────────────────────────────────────────────────────
    // Simulate per-iteration monitoring (common in ML profiling)
    // ─────────────────────────────────────────────────────────────────────

    // Track iteration count - READ + WRITE per "iteration"
    for (int i = 0; i < 5; i++) {
        unsigned long long iter_count = bpf_map_lookup(bpf_iteration_counts, (tid + i) % MAP_SIZE);
        bpf_map_update(bpf_iteration_counts, (tid + i) % MAP_SIZE, iter_count + 1);

        // Log intermediate timestamps
        bpf_log_event(tid, 1, bpf_get_globaltimer());
    }

    // ─────────────────────────────────────────────────────────────────────
    // Performance metrics collection - more memory operations
    // ─────────────────────────────────────────────────────────────────────

    // Read previous metrics
    unsigned long long prev_metric1 = bpf_map_lookup(bpf_performance_metrics, (tid * 4) % MAP_SIZE);
    unsigned long long prev_metric2 = bpf_map_lookup(bpf_performance_metrics, (tid * 4 + 1) % MAP_SIZE);
    unsigned long long prev_metric3 = bpf_map_lookup(bpf_performance_metrics, (tid * 4 + 2) % MAP_SIZE);
    unsigned long long prev_metric4 = bpf_map_lookup(bpf_performance_metrics, (tid * 4 + 3) % MAP_SIZE);

    // Update metrics
    unsigned long long ts_end = bpf_get_globaltimer();
    bpf_map_update(bpf_performance_metrics, (tid * 4) % MAP_SIZE, ts_end - ts_start);
    bpf_map_update(bpf_performance_metrics, (tid * 4 + 1) % MAP_SIZE, prev_metric1 + prev_metric2);
    bpf_map_update(bpf_performance_metrics, (tid * 4 + 2) % MAP_SIZE, prev_metric3 + 1);
    bpf_map_update(bpf_performance_metrics, (tid * 4 + 3) % MAP_SIZE, prev_metric4 + (ts_end - ts_start));

    // Log exit event
    bpf_log_event(tid, 2, ts_end);

    // Global counter update
    atomicAdd(&bpf_total_invocations, 1ULL);
}

//=============================================================================
// eBPF HOOK - MEDIUM: Reduced map operations
//=============================================================================

/**
 * Moderate monitoring: Only essential metrics, fewer map operations
 */
__device__ void ebpf_hook_MEDIUM(int row, int col, int K) {
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long ts = bpf_get_globaltimer();

    // Only log entry timestamp - 1 WRITE
    bpf_log_event(tid, 0, ts);

    // Single state update - 1 READ + 1 WRITE
    unsigned long long prev = bpf_map_lookup(bpf_thread_state, tid % MAP_SIZE);
    bpf_map_update(bpf_thread_state, tid % MAP_SIZE, prev + 1);

    // Single performance metric - 1 WRITE
    bpf_map_update(bpf_performance_metrics, tid % MAP_SIZE, ts);
}

//=============================================================================
// eBPF HOOK - GOOD: Minimal map operations with local aggregation
//=============================================================================

/**
 * GPU-aware monitoring: Minimize memory traffic
 * - Use registers for intermediate values
 * - Aggregate at warp level before writing
 * - Only write essential data
 */
__device__ void ebpf_hook_GOOD(int row, int col, int K) {
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long warp_id = tid / 32;
    int lane_id = tid % 32;

    // Compute metrics locally in registers (no memory traffic)
    unsigned long long ts = bpf_get_globaltimer();
    unsigned long long local_metric = ts ^ (row * col);  // Some computation

    // Warp-level aggregation: only lane 0 writes to map
    // Reduces memory traffic by 32x
    unsigned mask = __activemask();

    // Use warp shuffle to aggregate (register-to-register, no memory)
    for (int offset = 16; offset > 0; offset /= 2) {
        local_metric += __shfl_down_sync(mask, local_metric, offset);
    }

    // Only lane 0 writes the aggregated result - 1 WRITE per warp
    if (lane_id == 0) {
        bpf_map_update(bpf_performance_metrics, warp_id % MAP_SIZE, local_metric);
    }
}

//=============================================================================
// eBPF HOOK - BEST: Sampling-based monitoring
//=============================================================================

/**
 * Sampling: Only 1 in N threads actually logs
 * Reduces memory traffic by sampling_rate x
 */
#define SAMPLING_RATE 1024

__device__ void ebpf_hook_BEST_sampling(int row, int col, int K) {
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Only sample every Nth thread
    if ((tid % SAMPLING_RATE) != 0) {
        return;  // Early exit - no memory traffic
    }

    // Sampled thread logs data
    unsigned long long ts = bpf_get_globaltimer();
    unsigned long long sample_idx = tid / SAMPLING_RATE;

    bpf_map_update(bpf_performance_metrics, sample_idx % MAP_SIZE, ts);
    bpf_map_update(bpf_thread_state, sample_idx % MAP_SIZE, row * 1000 + col);
}

//=============================================================================
// GEMM Kernel - Matrix Multiplication (Compute-Bound)
//=============================================================================

#define TILE_SIZE 16

/**
 * Simple tiled GEMM: C = A * B
 * This is normally COMPUTE-BOUND (limited by FLOPS, not memory)
 * But excessive eBPF map operations can make it MEMORY-BOUND
 */
__global__ void gemm_with_bad_hook(float *A, float *B, float *C, int M, int N, int K) {
    // eBPF hook at kernel entry - SATURATES MEMORY BANDWIDTH
    ebpf_hook_BAD_bandwidth_hog(blockIdx.y * TILE_SIZE + threadIdx.y,
                                 blockIdx.x * TILE_SIZE + threadIdx.x, K);

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void gemm_with_medium_hook(float *A, float *B, float *C, int M, int N, int K) {
    ebpf_hook_MEDIUM(blockIdx.y * TILE_SIZE + threadIdx.y,
                     blockIdx.x * TILE_SIZE + threadIdx.x, K);

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void gemm_with_good_hook(float *A, float *B, float *C, int M, int N, int K) {
    ebpf_hook_GOOD(blockIdx.y * TILE_SIZE + threadIdx.y,
                   blockIdx.x * TILE_SIZE + threadIdx.x, K);

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void gemm_with_sampling_hook(float *A, float *B, float *C, int M, int N, int K) {
    ebpf_hook_BEST_sampling(blockIdx.y * TILE_SIZE + threadIdx.y,
                            blockIdx.x * TILE_SIZE + threadIdx.x, K);

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void gemm_no_hook(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

//=============================================================================
// Main
//=============================================================================

int main() {
    // Matrix dimensions: M x K * K x N = M x N
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const int SIZE_A = M * K * sizeof(float);
    const int SIZE_B = K * N * sizeof(float);
    const int SIZE_C = M * N * sizeof(float);

    const int ITERATIONS = 20;

    // Allocate host memory
    float *h_A = (float*)malloc(SIZE_A);
    float *h_B = (float*)malloc(SIZE_B);
    float *h_C = (float*)malloc(SIZE_C);

    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(i % 100) / 100.0f;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, SIZE_A);
    cudaMalloc(&d_B, SIZE_B);
    cudaMalloc(&d_C, SIZE_C);

    cudaMemcpy(d_A, h_A, SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE_B, cudaMemcpyHostToDevice);

    // Grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    int total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Example 4: eBPF Map Bandwidth Contention on GEMM             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    printf("Configuration:\n");
    printf("  Matrix size: %d x %d x %d\n", M, K, N);
    printf("  Total threads: %d\n", total_threads);
    printf("  GEMM FLOPs per call: %.2f GFLOP\n\n", 2.0 * M * N * K / 1e9);

    // Warmup
    gemm_no_hook<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Baseline (no hook)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_no_hook<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float baseline;
    cudaEventElapsedTime(&baseline, start, stop);
    float baseline_gflops = (2.0 * M * N * K * ITERATIONS) / (baseline / 1000.0) / 1e9;

    // Sampling hook (best)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_with_sampling_hook<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sampling_time;
    cudaEventElapsedTime(&sampling_time, start, stop);
    float sampling_gflops = (2.0 * M * N * K * ITERATIONS) / (sampling_time / 1000.0) / 1e9;

    // Good hook (warp aggregation)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_with_good_hook<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float good_time;
    cudaEventElapsedTime(&good_time, start, stop);
    float good_gflops = (2.0 * M * N * K * ITERATIONS) / (good_time / 1000.0) / 1e9;

    // Medium hook
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_with_medium_hook<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float medium_time;
    cudaEventElapsedTime(&medium_time, start, stop);
    float medium_gflops = (2.0 * M * N * K * ITERATIONS) / (medium_time / 1000.0) / 1e9;

    // Bad hook (bandwidth hog)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_with_bad_hook<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float bad_time;
    cudaEventElapsedTime(&bad_time, start, stop);
    float bad_gflops = (2.0 * M * N * K * ITERATIONS) / (bad_time / 1000.0) / 1e9;

    printf("Results (%d iterations):\n", ITERATIONS);
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  No hook (baseline):            %8.2f ms  (%6.1f GFLOPS)\n", baseline, baseline_gflops);
    printf("  BEST hook (sampling):          %8.2f ms  (%6.1f GFLOPS)  %.2fx overhead\n",
           sampling_time, sampling_gflops, sampling_time/baseline);
    printf("  GOOD hook (warp-aggregate):    %8.2f ms  (%6.1f GFLOPS)  %.2fx overhead\n",
           good_time, good_gflops, good_time/baseline);
    printf("  MEDIUM hook (reduced ops):     %8.2f ms  (%6.1f GFLOPS)  %.2fx overhead\n",
           medium_time, medium_gflops, medium_time/baseline);
    printf("  BAD hook (bandwidth hog):      %8.2f ms  (%6.1f GFLOPS)  %.2fx overhead\n\n",
           bad_time, bad_gflops, bad_time/baseline);

    printf("Performance Comparison:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  BAD vs Baseline:     %.2fx slowdown\n", bad_time / baseline);
    printf("  BAD vs BEST:         %.2fx slower\n", bad_time / sampling_time);
    printf("  MEDIUM vs BEST:      %.2fx slower\n", medium_time / sampling_time);
    printf("  GOOD vs BEST:        %.2fx slower\n\n", good_time / sampling_time);

    printf("Memory Bandwidth Analysis:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("Map operations per thread:\n");
    printf("  BAD:      ~40 ops (20 reads + 20 writes)\n");
    printf("  MEDIUM:   ~4 ops  (2 reads + 2 writes)\n");
    printf("  GOOD:     ~0.03 ops (1 write per warp = 1/32 per thread)\n");
    printf("  BEST:     ~0.002 ops (1 write per %d threads)\n\n", SAMPLING_RATE);

    printf("Total map memory traffic per kernel call:\n");
    printf("  BAD:      ~%d MB (40 ops × %d threads × 8 bytes)\n",
           (int)(40ULL * total_threads * 8 / 1024 / 1024), total_threads);
    printf("  MEDIUM:   ~%d MB (4 ops × %d threads × 8 bytes)\n",
           (int)(4ULL * total_threads * 8 / 1024 / 1024), total_threads);
    printf("  GOOD:     ~%d KB (1 op × %d warps × 8 bytes)\n",
           (int)(1ULL * (total_threads/32) * 8 / 1024), total_threads/32);
    printf("  BEST:     ~%d KB (2 ops × %d samples × 8 bytes)\n\n",
           (int)(2ULL * (total_threads/SAMPLING_RATE) * 8 / 1024), total_threads/SAMPLING_RATE);

    printf("Root Cause Analysis:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("GEMM is normally COMPUTE-BOUND:\n");
    printf("  - High arithmetic intensity (O(N³) FLOPs vs O(N²) memory)\n");
    printf("  - GPU compute units are the bottleneck\n\n");
    printf("Excessive eBPF map operations make it MEMORY-BOUND:\n");
    printf("  - BAD hook adds ~40 memory ops per thread\n");
    printf("  - With %d threads, that's ~%d million extra memory ops\n",
           total_threads, 40 * total_threads / 1000000);
    printf("  - Memory bandwidth saturates, compute units starve\n");
    printf("  - GEMM performance collapses\n\n");

    printf("Traditional eBPF Verifier:  PASS\n");
    printf("  ✓ All map accesses are bounded\n");
    printf("  ✓ Execution terminates\n");
    printf("  ✓ Valid helper calls\n\n");

    printf("GPU-aware Verifier should:  REJECT or LIMIT\n");
    printf("  ✗ Excessive map operations per thread\n");
    printf("  ✗ Memory traffic exceeds threshold\n");
    printf("  ✗ Would convert compute-bound to memory-bound\n\n");

    printf("Recommendations for GPU eBPF:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  1. Limit map operations per thread (e.g., max 2-4 ops)\n");
    printf("  2. Use warp-level aggregation before map writes\n");
    printf("  3. Consider sampling instead of per-thread logging\n");
    printf("  4. Cache intermediate results in registers\n");
    printf("  5. Avoid read-modify-write patterns in hot paths\n");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
